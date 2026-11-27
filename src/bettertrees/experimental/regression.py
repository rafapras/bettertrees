"""Interpretable sums of optimal trees for regression and count GLM (Spec R6).

Reuses the histogram search kernels (``best_cut_1d``, ``best_depth2``,
``best_depth3``, ``hist_1d``, ``small_tree_leaf_ids``) with objective-specific
gradients and Hessians:

- ``loss="squared_error"`` (identity link): ``g = w * (m - y)``, ``h = w``;
- ``loss="poisson"`` (log link, ``y >= 0``): ``mu = exp(clip(m, -20, 20))``,
  ``g = w * (mu - y)``, ``h = w * max(mu, 1e-16)``.

Exposes ``FIGSRegressor`` (with exact L2 backfitting and free intercept),
``SumOfOptimalTreesRegressor``, and ``BudgetRegressor``.
"""

from numbers import Integral

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ..bins import _count_balanced_cut_positions, fit_bin_edges, transform_bins_row_major
from ..sums._common import bin_threshold, predict_input, rebin, top_features
from ..sums._kernels import best_cut_1d, best_depth2, best_depth3, hist_1d
from ..sums.smalltrees import SmallTree, _d2_nested, _side, greedy_tree

LOSSES = ("squared_error", "poisson")
LOSS_SQ, LOSS_POISSON = 0, 1


@njit(cache=True)
def _reg_gh(loss_id, m, y, w):
    if loss_id == 0:
        return w * (m - y), w
    mc = min(max(m, -20.0), 20.0)
    mu = np.exp(mc)
    h_val = mu if mu > 1e-16 else 1e-16
    return w * (mu - y), w * h_val


@njit(cache=True)
def reg_grad_hess(loss_id, y, margin, w):
    n = len(y)
    g = np.empty(n)
    h = np.empty(n)
    for i in range(n):
        gi, hi = _reg_gh(loss_id, margin[i], y[i], w[i])
        g[i] = gi
        h[i] = hi
    return g, h


@njit(cache=True, parallel=True)
def reg_node_hist_margin(loss_id, Xb, y, w, margin, contrib, node_of_row, n_nodes, B):
    """Histogram ``(n_nodes, p, B, 3)`` of ``(g, h, w)`` at ``margin - contrib``."""
    n, p = Xb.shape
    out = np.zeros((n_nodes, p, B, 3))
    BLK = 16
    nblk = (p + BLK - 1) // BLK
    for blk in prange(nblk):
        j0 = blk * BLK
        j1 = min(p, j0 + BLK)
        for i in range(n):
            k = node_of_row[i]
            if k < 0:
                continue
            g, h = _reg_gh(loss_id, margin[i] - contrib[i], y[i], w[i])
            for j in range(j0, j1):
                b = Xb[i, j]
                out[k, j, b, 0] += g
                out[k, j, b, 1] += h
                out[k, j, b, 2] += w[i]
    return out


@njit(cache=True)
def reg_l2_step_inplace(
    loss_id, y, w, margin, contrib, ids, is_leaf, value, lam, lr, cap, has_cap
):
    """Exact L2 Newton step on active leaves of one regression tree."""
    n = len(y)
    L = len(value)
    G = np.zeros(L)
    H = np.zeros(L)
    for i in range(n):
        g, h = _reg_gh(loss_id, margin[i], y[i], w[i])
        k = ids[i]
        G[k] += g
        H[k] += h
    for k in range(L):
        if is_leaf[k]:
            denom = H[k] + lam
            s = -(G[k] + lam * value[k]) / denom if denom > 0.0 else 0.0
            if has_cap:
                s = min(max(s, -cap), cap)
            value[k] = value[k] + lr * s
    for i in range(n):
        new = value[ids[i]]
        margin[i] = margin[i] - contrib[i] + new
        contrib[i] = new


@njit(cache=True)
def reg_base_step_inplace(loss_id, y, w, margin, lr):
    """Unpenalized Newton intercept step on ``margin``; returns ``delta``."""
    n = len(y)
    G = 0.0
    H = 0.0
    for i in range(n):
        g, h = _reg_gh(loss_id, margin[i], y[i], w[i])
        G += g
        H += h
    denom = H if H > 1e-12 else 1e-12
    delta = lr * (-G / denom)
    for i in range(n):
        margin[i] += delta
    return delta


def _reg_base_margin(y, w, loss):
    mean_y = float(np.average(y, weights=w))
    if loss == "squared_error":
        return mean_y
    return float(np.log(max(mean_y, 1e-12)))


def _fit_reg_inputs(est, X, y, sample_weight):
    from sklearn.utils.validation import _check_sample_weight, validate_data

    X, y = validate_data(
        est,
        X,
        y,
        reset=True,
        dtype=np.float64,
        order="C",
        ensure_all_finite="allow-nan",
        accept_sparse=False,
        y_numeric=True,
    )
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("y must be a finite 1D numeric vector.")
    w = _check_sample_weight(sample_weight, X, dtype=np.float64, ensure_non_negative=True)
    if not w.sum() > 0:
        raise ValueError("sample_weight sums to zero.")
    if est.loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}.")
    if est.loss == "poisson":
        if (y < 0).any():
            raise ValueError("Poisson regression requires non-negative y >= 0.")
        if np.average(y, weights=w) <= 0:
            raise ValueError("Poisson regression requires positive weighted mean of y.")
    est.nan_features_ = np.isnan(X).any(axis=0)
    return X, y, np.ascontiguousarray(w)


def _weighted_bin_edges(X, w, max_bins):
    """Learn bin edges using positive-weight rows and sample-weight counts."""
    edges = []
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = (~np.isnan(col)) & (w > 0.0)
        finite = col[mask]
        wf = w[mask]
        if len(finite) == 0:
            edges.append(np.empty(0, dtype=np.float64))
            continue
        unique, inv = np.unique(finite, return_inverse=True)
        unique = unique.astype(np.float64)
        if len(unique) <= max_bins:
            cuts = unique[:-1] / 2.0 + unique[1:] / 2.0
        else:
            w_counts = np.bincount(inv, weights=wf, minlength=len(unique)).astype(np.float64)
            # Keep total scale >= len(unique) so _count_balanced_cut_positions target >= 1.0
            scale = max(float(len(finite)) / max(float(w_counts.sum()), 1e-12), 1.0)
            positions = _count_balanced_cut_positions(w_counts * scale, max_bins)
            cuts = unique[positions] / 2.0 + unique[positions + 1] / 2.0
        edges.append(np.ascontiguousarray(cuts, dtype=np.float64))
    return tuple(edges)


def _binned_weighted(X, w, max_bins):
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    if np.all(w == 1.0):
        edges = fit_bin_edges(X32, max_bins)
    else:
        edges = _weighted_bin_edges(X32, w, max_bins)
    Xb = transform_bins_row_major(X32, edges)
    nb = np.array([len(e) + 2 for e in edges], dtype=np.int64)
    return Xb, edges, nb


class _RegSumBase(RegressorMixin, BaseEstimator):
    """Shared binning, backfitting, prediction, and interpretation for regression tree sums."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _prepare_reg(self, X, y, sample_weight):
        X, y, w = _fit_reg_inputs(self, X, y, sample_weight)
        Xb, edges, nb = _binned_weighted(X, w, self.max_bins)
        self.bin_edges_ = edges
        self.loss_id_ = LOSS_SQ if self.loss == "squared_error" else LOSS_POISSON
        return X, Xb, nb, y, w

    def _newton_step(self, tree, Xb, y, margin, contrib, w, ids=None):
        cap = getattr(self, "max_delta_step", None)
        if ids is None:
            ids = tree.leaf_ids(Xb)
        tree.value = np.ascontiguousarray(tree.value, dtype=np.float64)
        is_leaf = np.asarray(tree.left, dtype=np.int64) == -1
        reg_l2_step_inplace(
            self.loss_id_,
            y,
            w,
            margin,
            contrib,
            ids,
            is_leaf,
            tree.value,
            self.lam_,
            self.learning_rate_,
            0.0 if cap is None else float(cap),
            cap is not None,
        )
        return contrib, margin

    def _backfit(self, trees, contribs, Xb, y, w, sweeps, ids=None):
        margin = self.base_margin_ + np.sum(contribs, axis=0)
        for _ in range(sweeps):
            for k, tree in enumerate(trees):
                contribs[k], margin = self._newton_step(
                    tree, Xb, y, margin, contribs[k], w, None if ids is None else ids[k]
                )
            self.base_margin_ += reg_base_step_inplace(
                self.loss_id_, y, w, margin, self.learning_rate_
            )
        return margin

    def _linear_predict(self, X):
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
        m = np.full(len(Xb), float(self.base_margin_), dtype=np.float64)
        for tree in self.trees_:
            m += tree.value[tree.leaf_ids(Xb)]
        return m

    def predict(self, X):
        """Predicted target (linear score for ``squared_error``, ``exp(score)`` for ``poisson``)."""
        m = self._linear_predict(X)
        if self.loss == "poisson":
            return np.exp(np.clip(m, -500.0, 500.0))
        return m

    def predict_contributions(self, X):
        """(n_samples, n_trees) matrix on the linear predictor scale."""
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
        out = np.zeros((len(Xb), len(self.trees_)), dtype=np.float64)
        for k, tree in enumerate(self.trees_):
            out[:, k] = tree.value[tree.leaf_ids(Xb)]
        return out

    @property
    def n_splits_(self):
        check_is_fitted(self, "trees_")
        return sum(t.n_splits for t in self.trees_)

    def _names(self, feature_names):
        if feature_names is not None:
            return list(feature_names)
        names = getattr(self, "feature_names_in_", None)
        return None if names is None else list(names)

    def rules(self, feature_names=None, precision=6):
        """[(tree, conditions, linear contribution)] for every leaf."""
        from ..sums.explain import leaf_rules

        check_is_fitted(self, "base_margin_")
        names = self._names(feature_names)
        nan = getattr(self, "nan_features_", None)
        return [
            (k, conds, v)
            for k, tree in enumerate(self.trees_)
            for conds, v in leaf_rules(tree, self.bin_edges_, names, nan, precision)
        ]

    def get_trees(self, feature_names=None):
        check_is_fitted(self, "base_margin_")
        names = self._names(feature_names)
        out = []
        for tree in self.trees_:
            feat = np.asarray(tree.feature, dtype=np.int64)
            thr = np.array([
                bin_threshold(self.bin_edges_[f], t) if f >= 0 else np.nan
                for f, t in zip(tree.feature, tree.threshold)
            ])
            out.append(dict(
                feature=feat,
                feature_name=[
                    None if f < 0 else (names[f] if names else f"x{f}") for f in feat
                ],
                threshold=thr,
                children_left=np.asarray(tree.left, dtype=np.int64),
                children_right=np.asarray(tree.right, dtype=np.int64),
                value=np.asarray(tree.value, dtype=np.float64).copy(),
            ))
        return out

    def export_text(self, feature_names=None, precision=4):
        link = "log E[y]" if self.loss == "poisson" else "E[y]"
        lines = [f"base ({link}): {float(self.base_margin_):+.{precision}f}"]
        for k, t in enumerate(self.get_trees(feature_names), 1):
            lines.append(f"tree {k}")

            def walk(node, depth, t=t):
                pad = "|   " * depth
                if t["children_left"][node] == -1:
                    lines.append(f"{pad}|--- value: {t['value'][node]:+.{precision}f}")
                    return
                name, thr = t["feature_name"][node], t["threshold"][node]
                if np.isneginf(thr):
                    left, right = f"{name} is missing", f"{name} is present"
                else:
                    v = f"{thr:.{precision}g}"
                    left, right = f"{name} <= {v}", f"{name} >  {v}"
                lines.append(f"{pad}|--- {left}")
                walk(t["children_left"][node], depth + 1)
                lines.append(f"{pad}|--- {right}")
                walk(t["children_right"][node], depth + 1)

            walk(0, 0)
        return "\n".join(lines)

    def to_dict(self, feature_names=None, precision=6):
        check_is_fitted(self, "base_margin_")
        trees = {}
        for k, conds, v in self.rules(feature_names, precision):
            trees.setdefault(k, []).append(dict(conditions=conds, value=v))
        return dict(
            link="log" if self.loss == "poisson" else "identity",
            loss=self.loss,
            base_margin=float(self.base_margin_),
            trees=[dict(rules=trees[k]) for k in sorted(trees)],
        )

    def explain(self, feature_names=None, precision=4):
        d = self.to_dict(feature_names, precision)
        target_str = "log E[y]" if self.loss == "poisson" else "E[y]"
        lines = [
            f"{target_str} = base {d['base_margin']:+.{precision}f}"
            f" + sum of {len(d['trees'])} trees ({self.n_splits_} cuts)"
        ]
        for k, tree in enumerate(d["trees"], 1):
            lines.append(f"tree {k}:")
            for r in tree["rules"]:
                cond = " and ".join(r["conditions"]) or "always"
                lines.append(f"  {r['value']:+.{precision}f}  if {cond}")
        return "\n".join(lines)

    def to_sql(
        self,
        table="input",
        feature_names=None,
        precision=6,
        keep_columns=True,
        null_safe=True,
        score_col="score",
        pred_col="prediction",
    ):
        """Readable SQL query (``EXP(score)`` for Poisson, ``score`` for squared error)."""
        from ..sums.explain import _sql_ident

        check_is_fitted(self, "base_margin_")
        names = self._names(feature_names)
        nan = getattr(self, "nan_features_", None)

        col_ref = (lambda j: _sql_ident(names[j])) if names is not None else (lambda j: f"x{j}")
        num = lambda v: f"{float(v):.{precision}g}"

        def cut(v):
            return np.format_float_positional(np.float64(v), unique=True, trim="-")

        cols, blocks = [], []
        for k, tree in enumerate(self.trees_, 1):
            feats = []
            for f, lft in zip(tree.feature, tree.left):
                if lft != -1 and int(f) not in feats:
                    feats.append(int(f))
            raw_n = (
                names[feats[0]]
                if (names is not None and feats)
                else (f"x{feats[0]}" if feats else "")
            )
            suffix = "".join(c if c.isalnum() else "_" for c in str(raw_n))
            col_base = f"t{k}_{suffix}" if feats else f"t{k}"
            col = _sql_ident(col_base[:60])
            leaves = []

            def walk(node, path, tree=tree, leaves=leaves):
                if tree.left[node] == -1:
                    leaves.append((path, float(tree.value[node])))
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                walk(tree.left[node], [*path, (f, t, True)])
                walk(tree.right[node], [*path, (f, t, False)])

            walk(0, [])
            if len(leaves) == 1:
                blocks.append(f"    {num(leaves[0][1])} AS {col}")
                cols.append(col)
                continue
            lines = [f"    CASE  -- tree {k}"]
            for i, (path, v) in enumerate(leaves):
                bounds = {}
                for f, t, went_left in path:
                    lo, hi = bounds.setdefault(f, [None, None])
                    if went_left:
                        bounds[f][1] = t if hi is None else min(hi, t)
                    else:
                        bounds[f][0] = t if lo is None else max(lo, t)
                conds = []
                for f, (lo, hi) in bounds.items():
                    nm = col_ref(f)
                    nan_ok = bool(null_safe) or (nan is not None and bool(nan[f]))
                    if lo is None:
                        if hi == 0:
                            conds.append(f"{nm} IS NULL")
                            continue
                        c = f"{nm} <= {cut(bin_threshold(self.bin_edges_[f], hi))}"
                        conds.append(f"({c} OR {nm} IS NULL)" if nan_ok else c)
                    elif hi is None:
                        conds.append(
                            f"{nm} IS NOT NULL"
                            if lo == 0
                            else f"{nm} > {cut(bin_threshold(self.bin_edges_[f], lo))}"
                        )
                    else:
                        if lo != 0:
                            conds.append(f"{nm} > {cut(bin_threshold(self.bin_edges_[f], lo))}")
                        else:
                            conds.append(f"{nm} IS NOT NULL")
                        conds.append(f"{nm} <= {cut(bin_threshold(self.bin_edges_[f], hi))}")
                if i == len(leaves) - 1:
                    lines.append(f"      ELSE {num(v)}")
                else:
                    lines.append(f"      WHEN {' AND '.join(conds)} THEN {num(v)}")
            lines.append(f"    END AS {col}")
            blocks.append("\n".join(lines))
            cols.append(col)

        s_col = _sql_ident(score_col)
        p_col = _sql_ident(pred_col)
        pred_expr = f"EXP({s_col})" if self.loss == "poisson" else s_col
        link_prefix = "log E[y]" if self.loss == "poisson" else "E[y]"
        link_txt = f"{link_prefix} = base + sum of {len(self.trees_)} tree columns"
        head = f"-- {link_txt} ({self.n_splits_} cuts); {p_col} = {pred_expr}"
        keep = "*, " if keep_columns else ""
        if not blocks:
            contrib_body = ["  SELECT *"] if keep_columns else ["  SELECT 0 AS _bt_dummy"]
            scored_head = f"  SELECT {keep}{num(float(self.base_margin_))}  -- base"
            if keep_columns:
                final_line = (
                    f"SELECT *, {pred_expr} AS {p_col}"
                    if p_col != s_col
                    else "SELECT *"
                )
            else:
                final_line = (
                    f"SELECT {s_col}, {pred_expr} AS {p_col}"
                    if p_col != s_col
                    else f"SELECT {s_col}"
                )
            return "\n".join([
                head,
                "WITH contributions AS (",
                *contrib_body,
                f"  FROM {table}",
                "), scored AS (",
                scored_head,
                f"    AS {s_col}",
                "  FROM contributions",
                ")",
                final_line,
                "FROM scored;",
            ])
        if keep_columns:
            final_line = f"SELECT *, {pred_expr} AS {p_col}" if p_col != s_col else "SELECT *"
        else:
            final_line = (
                f"SELECT {s_col}, {pred_expr} AS {p_col}"
                if p_col != s_col
                else f"SELECT {s_col}"
            )
        return "\n".join([
            head,
            "WITH contributions AS (",
            f"  SELECT {keep}".rstrip(),
            ",\n".join(blocks),
            f"  FROM {table}",
            "), scored AS (",
            f"  SELECT {keep}{num(float(self.base_margin_))}  -- base",
            *[f"    + {c}" for c in cols],
            f"    AS {s_col}",
            "  FROM contributions",
            ")",
            final_line,
            "FROM scored;",
        ])


class FIGSRegressor(_RegSumBase):
    """FIGS with exact L2-regularized Newton leaves for squared-error or Poisson regression."""

    def __init__(
        self,
        *,
        max_splits=16,
        max_trees=None,
        lam="auto",
        min_weight=20.0,
        max_bins=32,
        backfit_sweeps=1,
        learning_rate=1.0,
        max_delta_step=None,
        loss="squared_error",
    ):
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.backfit_sweeps = backfit_sweeps
        self.learning_rate = learning_rate
        self.max_delta_step = max_delta_step
        self.loss = loss

    def fit(self, X, y, sample_weight=None):
        X, Xb, nb, y_vec, w = self._prepare_reg(X, y, sample_weight)
        del X
        if self.lam == "auto":
            if self.loss == "poisson":
                mean_h0 = float(np.average(y_vec, weights=w))
                self.lam_ = 2.0 * self.max_splits * max(mean_h0, 1.0)
            else:
                self.lam_ = 2.0 * self.max_splits
        else:
            self.lam_ = float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        self.base_margin_ = _reg_base_margin(y_vec, w, self.loss)

        n, B = len(Xb), int(nb.max())
        trees, contribs, ids_of = [], [], []
        margin = np.full(n, self.base_margin_, dtype=np.float64)
        root_ids, zero = np.zeros(n, dtype=np.int64), np.zeros(n)
        for _ in range(self.max_splits):
            best = (0.0, None, None, None, None)
            can_add = self.max_trees is None or len(trees) < self.max_trees
            candidates = list(range(len(trees))) + ([None] if can_add else [])
            for k in candidates:
                if k is None:
                    ids, contrib = root_ids, zero
                    leaves, n_nodes = [0], 1
                else:
                    ids, contrib = ids_of[k], contribs[k]
                    leaves, n_nodes = trees[k].leaves, len(trees[k].feature)
                hist = reg_node_hist_margin(
                    self.loss_id_, Xb, y_vec, w, margin, contrib, ids, n_nodes, B
                )
                for leaf in leaves:
                    gains, cuts = best_cut_1d(hist[leaf], nb, self.lam_, self.min_weight)
                    f = int(np.argmax(gains))
                    if gains[f] > best[0]:
                        best = (float(gains[f]), k, leaf, f, int(cuts[f]))
            _, k, leaf, f, t = best
            if k is None and leaf is None:
                break
            if k is None:
                trees.append(SmallTree())
                contribs.append(np.zeros(n))
                ids_of.append(None)
                k = len(trees) - 1
            trees[k].split(leaf, f, t)
            ids_of[k] = trees[k].leaf_ids(Xb)
            contribs[k], margin = self._newton_step(
                trees[k], Xb, y_vec, margin, contribs[k], w, ids_of[k]
            )
            if self.backfit_sweeps:
                margin = self._backfit(
                    trees, contribs, Xb, y_vec, w, self.backfit_sweeps, ids_of
                )
            else:
                self.base_margin_ += reg_base_step_inplace(
                    self.loss_id_, y_vec, w, margin, self.learning_rate_
                )
        self.trees_ = trees
        return self


def _stable_argmax(gains):
    mx = float(np.max(gains))
    if mx <= 0.0:
        return int(np.argmax(gains))
    tol = 1e-12 * max(1.0, mx)
    return int(np.flatnonzero(gains >= mx - tol)[0])


def _optimal_tree_reg(Xb, g, h, w, nb, depth, lam, min_weight, features=None):
    """Optimal Newton tree with deterministic float-tie breaking across features."""
    feats = np.arange(Xb.shape[1]) if features is None else np.asarray(features, dtype=np.int64)
    Xs = np.ascontiguousarray(Xb[:, feats])
    nbs = np.ascontiguousarray(nb[feats])

    def remap(spec):
        if spec is None:
            return None
        return (int(feats[spec[0]]), spec[1], remap(spec[2]), remap(spec[3]))

    if depth == 1:
        gains, cuts = best_cut_1d(hist_1d(Xs, g, h, w, int(nbs.max())), nbs, lam, min_weight)
        f = _stable_argmax(gains)
        spec = (f, int(cuts[f]), None, None) if gains[f] > 0 else None
    elif depth == 2:
        gains, res = best_depth2(Xs, g, h, w, nbs, lam, min_weight)
        f = _stable_argmax(gains)
        spec = (
            None
            if gains[f] <= 0
            else (f, int(res[f, 0]), _side(res[f, 1], res[f, 2]), _side(res[f, 3], res[f, 4]))
        )
    elif depth == 3:
        gains, res = best_depth3(Xs, g, h, w, nbs, lam, min_weight)
        f = _stable_argmax(gains)
        spec = (
            None
            if gains[f] <= 0
            else (f, int(res[f, 0]), _d2_nested(res[f, 1:7]), _d2_nested(res[f, 7:13]))
        )
    else:
        raise ValueError("depth must be 1, 2 or 3.")
    return None if spec is None else SmallTree.from_nested(remap(spec))


class SumOfOptimalTreesRegressor(_RegSumBase):
    """Sum of optimal depth 1-3 Newton trees for squared-error or Poisson regression."""

    def __init__(
        self,
        *,
        n_trees=2,
        depth=2,
        extra_stumps=0,
        learning_rate="auto",
        lam=2.0,
        min_weight=20.0,
        max_bins=16,
        backfit_sweeps=2,
        max_features_d3=24,
        max_features_d2=128,
        search="optimal",
        max_delta_step=None,
        loss="squared_error",
    ):
        self.n_trees = n_trees
        self.depth = depth
        self.extra_stumps = extra_stumps
        self.learning_rate = learning_rate
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.backfit_sweeps = backfit_sweeps
        self.max_features_d3 = max_features_d3
        self.max_features_d2 = max_features_d2
        self.search = search
        self.max_delta_step = max_delta_step
        self.loss = loss

    def fit(self, X, y, sample_weight=None):
        if self.search not in ("optimal", "greedy"):
            raise ValueError("search must be 'optimal' or 'greedy'.")
        grow = _optimal_tree_reg if self.search == "optimal" else greedy_tree
        self.lam_ = float(self.lam)
        self.learning_rate_ = (
            1.0 / (1.0 + 0.1 * self.n_trees)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )
        _, Xb, nb, y_vec, w = self._prepare_reg(X, y, sample_weight)
        self.base_margin_ = _reg_base_margin(y_vec, w, self.loss)
        margin = np.full(len(Xb), self.base_margin_, dtype=np.float64)
        trees, contribs = [], []
        k = self.max_features_d3 if self.depth == 3 else self.max_features_d2
        for depth in [self.depth] * self.n_trees + [1] * self.extra_stumps:
            g, h = reg_grad_hess(self.loss_id_, y_vec, margin, w)
            feats = None if depth == 1 else top_features(
                Xb, g, h, w, nb, self.lam_, self.min_weight, k
            )
            tree = grow(Xb, g, h, w, nb, depth, self.lam_, self.min_weight, feats)
            if tree is None:
                break
            contrib, margin = self._newton_step(
                tree, Xb, y_vec, margin, np.zeros(len(Xb)), w
            )
            trees.append(tree)
            contribs.append(contrib)
        if trees and self.backfit_sweeps:
            self._backfit(trees, contribs, Xb, y_vec, w, self.backfit_sweeps)
        self.trees_ = trees
        return self


class BudgetRegressor(RegressorMixin, BaseEstimator):
    """Untuned fixed-budget regression rule using ``FIGSRegressor``."""

    def __init__(self, max_splits=16, loss="squared_error"):
        self.max_splits = max_splits
        self.loss = loss

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _make(self):
        b = self.max_splits
        if isinstance(b, bool) or not isinstance(b, Integral) or b < 1:
            raise ValueError("max_splits must be an integer >= 1.")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}.")
        b = int(b)
        lr = 1.0 if b <= 8 else 0.3
        return FIGSRegressor(max_splits=b, learning_rate=lr, loss=self.loss)

    def fit(self, X, y, sample_weight=None):
        self.model_ = self._make().fit(X, y, sample_weight=sample_weight)
        self.n_features_in_ = self.model_.n_features_in_
        if hasattr(self.model_, "feature_names_in_"):
            self.feature_names_in_ = self.model_.feature_names_in_
        else:
            self.__dict__.pop("feature_names_in_", None)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_contributions(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_contributions(X)

    def __getattr__(self, name):
        if name.startswith("__") or "model_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["model_"], name)
