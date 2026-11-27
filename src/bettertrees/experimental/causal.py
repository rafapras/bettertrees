"""CausalFIGS: sums of shallow Newton-optimal trees for CATE estimation (R-learner)."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from ..sums._common import as_float_matrix, as_weights, bin_threshold, binned, rebin
from ..sums._kernels import best_cut_1d, node_hist
from ..sums.explain import _fmt, _sql_ident, feature_names_of, leaf_rules
from ..sums.smalltrees import SmallTree, optimal_tree


def _tree_depths(tree):
    depths = {0: 0}
    for k in range(len(tree.feature)):
        if tree.left[k] != -1:
            depths[tree.left[k]] = depths[tree.right[k]] = depths[k] + 1
    return depths


class CausalFIGS(BaseEstimator):
    """Orthogonal R-learner (Nie & Wager, 2021) over an additive sum of shallow trees.

    Minimizes ``0.5 * sum_i w_i * ((y_i - m(x_i)) - (d_i - e(x_i)) * tau(x_i))^2``
    plus ``0.5 * lam * sum_{k, l} v_{k, l}^2`` under a budget of ``max_splits`` cuts.
    """

    def __init__(
        self,
        *,
        max_splits=8,
        max_trees=None,
        max_depth=2,
        lookahead=False,
        lam=1.0,
        min_weight=10.0,
        max_bins=32,
        learning_rate=1.0,
        backfit_sweeps=2,
        cv=3,
        random_state=0,
    ):
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.lookahead = lookahead
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.backfit_sweeps = backfit_sweeps
        self.cv = cv
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _validate_X_predict(self, X):
        check_is_fitted(self, "trees_")
        names = feature_names_of(X)
        fitted_names = getattr(self, "feature_names_in_", None)
        if fitted_names is not None and names is not None and list(names) != list(fitted_names):
            raise ValueError("Feature names at prediction time must match fit.")
        Xm = as_float_matrix(X)
        if Xm.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xm.shape[1]} features, expected {self.n_features_in_}.")
        return Xm

    def fit(
        self, X, y, treatment, *, y_hat=None, m_hat=None, e_hat=None, sample_weight=None
    ):
        names = feature_names_of(X)
        X = as_float_matrix(X)
        n, p = X.shape
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        d = np.ascontiguousarray(np.asarray(treatment, dtype=np.float64))
        if y.shape != (n,) or not np.isfinite(y).all():
            raise ValueError("y must be a finite 1D vector aligned with X.")
        if d.shape != (n,) or not np.isfinite(d).all():
            raise ValueError("treatment must be a finite 1D vector aligned with X.")
        if y_hat is not None and m_hat is not None:
            raise ValueError("Pass at most one of y_hat and m_hat.")
        y_hat = m_hat if y_hat is None else y_hat
        if y_hat is not None:
            y_hat = np.ascontiguousarray(np.asarray(y_hat, dtype=np.float64))
            if y_hat.shape != (n,) or not np.isfinite(y_hat).all():
                raise ValueError("m_hat / y_hat must be a finite 1D vector aligned with X.")
        if e_hat is not None:
            e_hat = np.ascontiguousarray(np.asarray(e_hat, dtype=np.float64))
            if e_hat.shape != (n,) or not np.isfinite(e_hat).all():
                raise ValueError("e_hat must be a finite 1D vector aligned with X.")
        w = as_weights(sample_weight, n)
        if w.sum() <= 0:
            raise ValueError("sample_weight must have a positive sum.")

        self.n_features_in_ = p
        if names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        else:
            self.__dict__.pop("feature_names_in_", None)
        self.nan_features_ = np.isnan(X).any(axis=0)

        if y_hat is None or e_hat is None:
            kf = KFold(n_splits=int(self.cv), shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X))
            if y_hat is None:
                y_hat = np.empty(n, dtype=np.float64)
                bin_y = set(np.unique(y)).issubset({0.0, 1.0}) and len(np.unique(y)) == 2
                for tr_idx, va_idx in splits:
                    if bin_y:
                        est = HistGradientBoostingClassifier(
                            max_iter=50, max_leaf_nodes=15, random_state=self.random_state
                        )
                        est.fit(X[tr_idx], y[tr_idx].astype(int), sample_weight=w[tr_idx])
                        y_hat[va_idx] = est.predict_proba(X[va_idx])[:, 1]
                    else:
                        est = HistGradientBoostingRegressor(
                            max_iter=50, max_leaf_nodes=15, random_state=self.random_state
                        )
                        est.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
                        y_hat[va_idx] = est.predict(X[va_idx])
            if e_hat is None:
                e_hat = np.empty(n, dtype=np.float64)
                bin_d = set(np.unique(d)).issubset({0.0, 1.0}) and len(np.unique(d)) == 2
                for tr_idx, va_idx in splits:
                    if bin_d:
                        est = HistGradientBoostingClassifier(
                            max_iter=50, max_leaf_nodes=15, random_state=self.random_state
                        )
                        est.fit(X[tr_idx], d[tr_idx].astype(int), sample_weight=w[tr_idx])
                        e_hat[va_idx] = np.clip(est.predict_proba(X[va_idx])[:, 1], 0.01, 0.99)
                    else:
                        est = HistGradientBoostingRegressor(
                            max_iter=50, max_leaf_nodes=15, random_state=self.random_state
                        )
                        est.fit(X[tr_idx], d[tr_idx], sample_weight=w[tr_idx])
                        e_hat[va_idx] = est.predict(X[va_idx])

        self.m_hat_ = np.array(y_hat, dtype=np.float64, copy=True)
        self.e_hat_ = np.array(e_hat, dtype=np.float64, copy=True)
        y_res = y - self.m_hat_
        d_res = d - self.e_hat_
        h = np.ascontiguousarray(w * (d_res ** 2), dtype=np.float64)
        wy = np.ascontiguousarray(w * d_res * y_res, dtype=np.float64)

        self.lam_ = float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        self.base_effect_ = float(wy.sum() / max(h.sum(), 1e-12))
        tau = np.full(n, self.base_effect_, dtype=np.float64)

        Xb, self.bin_edges_, nb = binned(X, self.max_bins)
        B = int(nb.max())
        trees, contribs, ids_of = [], [], []
        root_ids = np.zeros(n, dtype=np.int64)

        def update_tree(tree, ids, contrib, tau_vec):
            g = h * tau_vec - wy
            L = len(tree.feature)
            G = np.bincount(ids, weights=g, minlength=L)
            H = np.bincount(ids, weights=h, minlength=L)
            for leaf in tree.leaves:
                step = -(G[leaf] + self.lam_ * tree.value[leaf]) / max(H[leaf] + self.lam_, 1e-12)
                tree.value[leaf] += self.learning_rate_ * step
            new_c = tree.value[ids]
            return new_c, tau_vec - contrib + new_c

        spent = 0
        while spent < int(self.max_splits):
            can_add = self.max_trees is None or len(trees) < int(self.max_trees)
            best_score = 0.0
            best_action = None

            # 1. Standard FIGS 1-cut candidates (extend existing leaf or add new stump)
            candidates = list(range(len(trees))) + ([None] if can_add else [])
            for k in candidates:
                if k is None:
                    ids, contrib = root_ids, 0.0
                    leaves, n_nodes = [0], 1
                else:
                    ids, contrib = ids_of[k], contribs[k]
                    depths = _tree_depths(trees[k])
                    leaves = [
                        lf for lf in trees[k].leaves
                        if self.max_depth is None or depths[lf] < int(self.max_depth)
                    ]
                    n_nodes = len(trees[k].feature)
                if not leaves:
                    continue
                g_other = np.ascontiguousarray(h * (tau - contrib) - wy, dtype=np.float64)
                hist = node_hist(Xb, g_other, h, w, ids, n_nodes, B)
                for leaf in leaves:
                    gains, cuts = best_cut_1d(hist[leaf], nb, self.lam_, float(self.min_weight))
                    f = int(np.argmax(gains))
                    if gains[f] > best_score:
                        best_score = float(gains[f])
                        best_action = ("cut", k, leaf, f, int(cuts[f]))

            # 2. Optional depth-2 optimal lookahead for new tree
            rem = int(self.max_splits) - spent
            if self.lookahead and can_add and rem >= 2 and (
                self.max_depth is None or int(self.max_depth) >= 2
            ):
                g_cur = np.ascontiguousarray(h * tau - wy, dtype=np.float64)
                opt = optimal_tree(
                    Xb, g_cur, h, w, nb, 2, self.lam_, float(self.min_weight)
                )
                if opt is not None and 1 < opt.n_splits <= rem:
                    ids_opt = opt.leaf_ids(Xb)
                    G = np.bincount(ids_opt, weights=g_cur, minlength=len(opt.feature))
                    H = np.bincount(ids_opt, weights=h, minlength=len(opt.feature))
                    raw_gain = sum(
                        G[lf] ** 2 / (H[lf] + self.lam_) for lf in opt.leaves
                    ) - (g_cur.sum() ** 2) / (h.sum() + self.lam_)
                    per_cut = float(raw_gain) / opt.n_splits
                    if per_cut > best_score:
                        best_score = per_cut
                        best_action = ("opt", opt)

            if best_action is None:
                break

            if best_action[0] == "cut":
                _, k, leaf, f, cut = best_action
                if k is None:
                    trees.append(SmallTree())
                    contribs.append(np.zeros(n, dtype=np.float64))
                    ids_of.append(None)
                    k = len(trees) - 1
                trees[k].split(leaf, f, cut)
                ids_of[k] = trees[k].leaf_ids(Xb)
                spent += 1
            else:
                opt = best_action[1]
                trees.append(opt)
                contribs.append(np.zeros(n, dtype=np.float64))
                ids_of.append(opt.leaf_ids(Xb))
                k = len(trees) - 1
                spent += opt.n_splits

            contribs[k], tau = update_tree(trees[k], ids_of[k], contribs[k], tau)
            for _ in range(int(self.backfit_sweeps)):
                for j, tr in enumerate(trees):
                    contribs[j], tau = update_tree(tr, ids_of[j], contribs[j], tau)
                delta0 = -float((h * tau - wy).sum() / max(h.sum(), 1e-12))
                self.base_effect_ += delta0
                tau += delta0

        self.trees_ = trees
        self.ate_ = float(np.average(tau, weights=w))
        return self

    @property
    def n_splits_(self):
        check_is_fitted(self, "trees_")
        return sum(t.n_splits for t in self.trees_)

    def predict(self, X):
        Xb = rebin(self._validate_X_predict(X), self.bin_edges_)
        tau = np.full(len(Xb), self.base_effect_, dtype=np.float64)
        for tree in self.trees_:
            tau += tree.value[tree.leaf_ids(Xb)]
        return tau

    def predict_contributions(self, X):
        Xb = rebin(self._validate_X_predict(X), self.bin_edges_)
        out = np.zeros((len(Xb), len(self.trees_)), dtype=np.float64)
        for k, tree in enumerate(self.trees_):
            out[:, k] = tree.value[tree.leaf_ids(Xb)]
        return out

    def _names(self, feature_names):
        if feature_names is not None:
            return list(feature_names)
        names = getattr(self, "feature_names_in_", None)
        return None if names is None else list(names)

    def rules(self, feature_names=None, precision=6):
        check_is_fitted(self, "trees_")
        names = self._names(feature_names)
        return [
            (k, conds, val)
            for k, tree in enumerate(self.trees_)
            for conds, val in leaf_rules(
                tree, self.bin_edges_, names, self.nan_features_, precision
            )
        ]

    def to_dict(self, feature_names=None, precision=6):
        check_is_fitted(self, "trees_")
        trees = {}
        for k, conds, v in self.rules(feature_names, precision):
            trees.setdefault(k, []).append(dict(conditions=conds, value=v))
        return dict(
            link="identity",
            target="cate",
            base_effect=float(self.base_effect_),
            ate=float(self.ate_),
            trees=[dict(rules=trees[k]) for k in sorted(trees)],
        )

    def explain(self, feature_names=None, precision=4):
        d = self.to_dict(feature_names, precision)
        lines = [
            f"tau(x) = base {d['base_effect']:+.{precision}f} + sum of {len(d['trees'])} trees "
            f"({self.n_splits_} cuts); ATE = {d['ate']:+.{precision}f}"
        ]
        for k, tree in enumerate(d["trees"], 1):
            lines.append(f"tree {k}:")
            for r in tree["rules"]:
                cond = " and ".join(r["conditions"]) or "always"
                lines.append(f"  {r['value']:+.{precision}f}  if {cond}")
        return "\n".join(lines)

    def to_sql(
        self, table="input", feature_names=None, precision=6, keep_columns=True, score_col="cate"
    ):
        check_is_fitted(self, "trees_")
        names = self._names(feature_names)
        name = (lambda j: _sql_ident(names[j])) if names is not None else (lambda j: f"x{j}")
        num = lambda v: _fmt(v, precision)
        cut = lambda v: np.format_float_positional(
            np.float64(v), unique=True, trim="-"
        )

        terms = [num(float(self.base_effect_))]
        for tree in self.trees_:
            def build(node, tree=tree):
                if tree.left[node] == -1:
                    return num(float(tree.value[node]))
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                nm = name(f)
                if t == 0:
                    cond = f"{nm} IS NULL"
                else:
                    v = cut(bin_threshold(self.bin_edges_[f], t))
                    cond = f"({nm} <= {v} OR {nm} IS NULL)"
                left_s = build(tree.left[node])
                right_s = build(tree.right[node])
                return f"CASE WHEN {cond} THEN {left_s} ELSE {right_s} END"

            terms.append(f"({build(0)})")

        sc = _sql_ident(score_col)
        keep = f"{table}.*, " if keep_columns else ""
        return f"SELECT {keep}{' + '.join(terms)} AS {sc} FROM {table};"
