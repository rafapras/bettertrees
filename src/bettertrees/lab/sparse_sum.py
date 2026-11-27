"""SparseTreeSumClassifier (full size-loss frontier in one fit) and ResidualRatioFIGSClassifier."""

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..sums._common import (
    base_margin,
    bin_threshold,
    fit_inputs,
    grad_hess,
    log_loss_margin as _log_loss,
    predict_input,
    rebin,
)
from ..sums._kernels import best_cut_1d, combination_scores, hist_1d, node_hist, quadrant_scores
from ..sums.additive import AdditiveTreeBooster
from ..sums.compact import _TermSet
from ..sums.edit import TreeEditMixin, _key
from ..sums.explain import InterpretableSumMixin, _fmt, _sql_ident
from ..sums.smalltrees import SmallTree, _AdditiveTrees


class SparseTreeSumClassifier(TreeEditMixin, InterpretableSumMixin, ClassifierMixin, BaseEstimator):
    """Cost-weighted non-negative L1 path over candidate shallow optimal trees."""

    def __init__(
        self,
        *,
        budgets=(4, 8, 16, 32),
        n_candidates=60,
        n_lambdas=40,
        lam=2.0,
        min_weight=20.0,
        max_bins=16,
        refit=True,
        random_state=0,
    ):
        self.budgets = budgets
        self.n_candidates = n_candidates
        self.n_lambdas = n_lambdas
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.refit = refit
        self.random_state = random_state

    def decision_function(self, X):
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
        m = np.full(len(Xb), self.base_margin_, dtype=np.float64)
        for tree in self.trees_:
            m += tree.value[tree.leaf_ids(Xb)]
        return m

    def fit(self, X, y, sample_weight=None, y_soft=None):
        X_raw = X
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        n = len(X)
        self.classes_ = classes
        self.lam_ = float(self.lam)
        self.learning_rate_ = 1.0

        booster = AdditiveTreeBooster(
            depth=2,
            max_rounds=int(self.n_candidates),
            learning_rate=0.2,
            lam=self.lam_,
            min_weight=float(self.min_weight),
            max_bins=int(self.max_bins),
            validation_fraction=0.0,
            feature_screen="fast",
            random_state=self.random_state,
        ).fit(X_raw, y, sample_weight=w, y_soft=y_soft)

        self.bin_edges_ = booster.bin_edges_
        Xb = rebin(X, self.bin_edges_)
        self.base_margin_ = base_margin(target, w)

        term_set = _TermSet()
        for tr in booster.trees_:
            if tr.n_splits > 0:
                term_set.add(_key(tr), tr)

        raw_terms = list(term_set.terms.values())
        terms, cols, costs, means, stds = [], [], [], [], []
        w_norm = w / w.sum()
        for tr in raw_terms:
            pred = tr.value[tr.leaf_ids(Xb)]
            mu = float(w_norm @ pred)
            sd = float(np.sqrt(w_norm @ ((pred - mu) ** 2)))
            if sd > 1e-12:
                terms.append(tr)
                cols.append((pred - mu) / sd)
                costs.append(float(tr.n_splits))
                means.append(mu)
                stds.append(sd)

        budgets = tuple(sorted({int(b) for b in self.budgets}))
        if not terms:
            self.trees_ = []
            self.frontier_ = {b: copy.deepcopy(self) for b in budgets}
            return self

        U = np.column_stack(cols)
        costs_arr = np.asarray(costs, dtype=np.float64)
        means_arr = np.asarray(means, dtype=np.float64)
        stds_arr = np.asarray(stds, dtype=np.float64)
        M = len(terms)
        W_sum = float(w.sum())

        g0, _ = grad_hess(target, np.full(n, self.base_margin_), w)
        lam_max = float(np.max(np.maximum(0.0, -(U.T @ g0) / W_sum) / costs_arr))
        lam_max = max(lam_max, 1e-4)
        lambdas = np.geomspace(lam_max * 1.05, lam_max * 1e-4, int(self.n_lambdas))

        alphas = np.zeros(M, dtype=np.float64)
        alpha0 = 0.0
        max_b = max(budgets)
        init_loss = float(_log_loss(target, np.full(n, self.base_margin_), w))
        snapshots = [(0, init_loss, 0.0, alphas.copy())]

        for lam_l1 in lambdas:
            margin = self.base_margin_ + alpha0 + U @ alphas
            for _ in range(2):
                g, h = grad_hess(target, margin, w)
                Q = (U.T @ (h[:, None] * U)) / W_sum
                grad = (U.T @ g) / W_sum
                for _ in range(15):
                    for k in range(M):
                        q_kk = Q[k, k] + 1e-9
                        old_a = alphas[k]
                        new_a = max(0.0, old_a - (grad[k] + lam_l1 * costs_arr[k]) / q_kk)
                        diff = new_a - old_a
                        if diff != 0.0:
                            alphas[k] = new_a
                            grad += Q[:, k] * diff
                    margin = self.base_margin_ + alpha0 + U @ alphas
                    gc, hc = grad_hess(target, margin, w)
                    d0 = -float(gc.sum() / max(hc.sum(), 1e-12))
                    alpha0 += d0
                    margin += d0

            active = alphas > 1e-8
            c_used = int(costs_arr[active].sum())
            loss = float(_log_loss(target, margin, w))
            snapshots.append((c_used, loss, float(alpha0), alphas.copy()))
            if c_used > max_b * 1.5:
                break

        self.frontier_ = {}
        for b in budgets:
            eligible = [s for s in snapshots if s[0] <= b]
            _, _, a0_best, a_best = min(eligible, key=lambda s: s[1])
            active_idx = np.flatnonzero(a_best > 1e-8)
            shift = 0.0
            sel_trees = []
            for k in active_idx:
                scale_k = float(a_best[k] / stds_arr[k])
                shift -= scale_k * float(means_arr[k])
                tr = terms[k]
                sel_trees.append(
                    SmallTree(
                        list(tr.feature),
                        list(tr.threshold),
                        list(tr.left),
                        list(tr.right),
                        np.asarray(tr.value * scale_k, dtype=np.float64),
                    )
                )
            sub = copy.copy(self)
            sub.frontier_ = {}
            sub.base_margin_ = float(self.base_margin_ + a0_best + shift)
            sub.trees_ = sel_trees
            if self.refit and sel_trees:
                sub._refit_binned(Xb, target, w, self.lam_, sweeps=2)
            self.frontier_[b] = sub

        top_model = self.frontier_[max_b]
        self.base_margin_ = top_model.base_margin_
        self.trees_ = [
            SmallTree(
                list(t.feature),
                list(t.threshold),
                list(t.left),
                list(t.right),
                t.value.copy(),
            )
            for t in top_model.trees_
        ]
        return self

    def model_for_budget(self, budget):
        check_is_fitted(self, "frontier_")
        b = int(budget)
        if b not in self.frontier_:
            raise KeyError(f"budget {b} not in fitted budgets {sorted(self.frontier_)}.")
        return self.frontier_[b]


class ResidualRatioFIGSClassifier(_AdditiveTrees):
    """FIGS augmented with residual ratio/difference cuts gated by an analytical BIC shape test."""

    def __init__(
        self,
        *,
        max_splits=8,
        min_main_splits=2,
        max_ratio_splits=2,
        top_k=12,
        min_valid=0.9,
        bic_scale=2.0,
        lam="auto",
        min_weight=20.0,
        max_bins=32,
        learning_rate=1.0,
        backfit_sweeps=1,
        max_delta_step=None,
    ):
        self.max_splits = max_splits
        self.min_main_splits = min_main_splits
        self.max_ratio_splits = max_ratio_splits
        self.top_k = top_k
        self.min_valid = min_valid
        self.bic_scale = bic_scale
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.backfit_sweeps = backfit_sweeps
        self.max_delta_step = max_delta_step

    @staticmethod
    def _eval_comb_column(X, kind, fi, fj, thr):
        a, b = X[:, fi], X[:, fj]
        if kind == 0:
            ok = (~np.isnan(a)) & (~np.isnan(b)) & (a > 0) & (b > 0)
            z = np.where(ok, np.log(np.where(ok, a, 1.0)) - np.log(np.where(ok, b, 1.0)), np.nan)
        else:
            ok = (~np.isnan(a)) & (~np.isnan(b))
            z = np.where(ok, a - b, np.nan)
        # Ineligible/NaN goes left (bin 1), finite <= thr left (1), > thr right (2)
        if np.isneginf(thr):
            left = ~ok
        else:
            left = (~ok) | (z <= thr)
        return np.where(left, 1, 2).astype(np.uint8)

    def _augmented(self, X):
        Xb = rebin(X, self.bin_edges_)
        if not self.ratio_splits_:
            return Xb
        cols = [Xb]
        for s in self.ratio_splits_:
            col = self._eval_comb_column(X, s["kind"], s["fi"], s["fj"], s["threshold"])
            cols.append(col[:, None])
        return np.ascontiguousarray(np.hstack(cols))

    def fit(self, X, y, sample_weight=None, y_soft=None):
        self.lam_ = 2.0 * max(1, int(self.max_splits)) if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        X, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        n, p = X.shape
        B = int(nb.max())
        self.base_margin_ = base_margin(target, w)
        margin = np.full(n, self.base_margin_, dtype=np.float64)

        trees, contribs, ids_of = [], [], []
        self.ratio_splits_ = []
        used_ratios = set()
        Xaug = Xb
        root_ids = np.zeros(n, dtype=np.int64)
        spent = 0

        while spent < int(self.max_splits):
            best = (0.0, "axis", None, None, None, None)
            for k in [*range(len(trees)), None]:
                if k is None:
                    ids, c_vec, leaves, n_nodes = root_ids, 0.0, [0], 1
                else:
                    ids, c_vec = ids_of[k], contribs[k]
                    leaves, n_nodes = trees[k].leaves, len(trees[k].feature)
                g_o, h_o = grad_hess(target, margin - c_vec, w)
                hist = node_hist(Xb, g_o, h_o, w, ids, n_nodes, B)
                for leaf in leaves:
                    gains, cuts = best_cut_1d(hist[leaf], nb, self.lam_, float(self.min_weight))
                    f = int(np.argmax(gains))
                    if gains[f] > best[0]:
                        best = (float(gains[f]), "axis", k, leaf, f, int(cuts[f]))

            if (
                p >= 2
                and spent >= int(self.min_main_splits)
                and len(self.ratio_splits_) < int(self.max_ratio_splits)
            ):
                g, h = grad_hess(target, margin, w)
                gains_u, _ = best_cut_1d(
                    hist_1d(Xb, g, h, w, B), nb, self.lam_, float(self.min_weight)
                )
                top_u = list(np.argsort(-gains_u, kind="stable")[: int(self.top_k)])
                pairs = np.array(
                    [
                        (min(a, b), max(a, b))
                        for idx_a, a in enumerate(top_u)
                        for b in top_u[idx_a + 1 :]
                    ],
                    dtype=np.int64,
                ).reshape(-1, 2)
                if len(pairs):
                    quad, _, _ = quadrant_scores(
                        Xb, g, h, w, nb, pairs, self.lam_, float(self.min_weight)
                    )
                    axis_pair = np.maximum(gains_u[pairs[:, 0]], gains_u[pairs[:, 1]])
                    baseline_pair = np.maximum(axis_pair, quad)
                    # On null logistic residuals G_b^2 / H_b ~ chi2_1 (unit scale)
                    bic_pen = float(self.bic_scale) * np.log(
                        max(2 * len(pairs), 2) * max(np.log(n), 2.0)
                    )
                    for kind in (0, 1):
                        g_c, t_c, v_c = combination_scores(
                            X,
                            g,
                            h,
                            w,
                            pairs,
                            kind,
                            int(self.max_bins),
                            self.lam_,
                            float(self.min_weight),
                        )
                        shape_excess = g_c - baseline_pair
                        for q in np.argsort(-g_c, kind="stable"):
                            if v_c[q] < float(self.min_valid) or np.isnan(t_c[q]):
                                continue
                            fi, fj = int(pairs[q, 0]), int(pairs[q, 1])
                            if (kind, fi, fj) in used_ratios:
                                continue
                            if shape_excess[q] > bic_pen and g_c[q] > best[0]:
                                best = (float(g_c[q]), "comb", kind, fi, fj, float(t_c[q]))
                            break

            if best[2] is None and best[1] == "axis":
                if best[3] is None:
                    break

            if best[1] == "axis":
                _, _, k, leaf, f, cut = best
                if leaf is None:
                    break
                if k is None:
                    trees.append(SmallTree())
                    contribs.append(np.zeros(n, dtype=np.float64))
                    ids_of.append(None)
                    k = len(trees) - 1
                trees[k].split(leaf, f, cut)
                ids_of[k] = trees[k].leaf_ids(Xaug)
                spent += 1
            else:
                _, _, kind, fi, fj, thr = best
                col = self._eval_comb_column(X, kind, fi, fj, thr)
                j_aug = len(self.ratio_splits_)
                Xaug = np.ascontiguousarray(np.column_stack([Xaug, col]))
                self.ratio_splits_.append(dict(kind=kind, fi=fi, fj=fj, threshold=thr))
                used_ratios.add((kind, fi, fj))
                tr = SmallTree()
                tr.split(0, p + j_aug, 1)
                trees.append(tr)
                contribs.append(np.zeros(n, dtype=np.float64))
                k = len(trees) - 1
                ids_of.append(tr.leaf_ids(Xaug))
                spent += 1

            contribs[k], margin = self._newton_step(
                trees[k], Xaug, target, margin, contribs[k], w, ids_of[k]
            )
            if self.backfit_sweeps:
                margin = self._backfit(
                    trees, contribs, Xaug, target, w, self.backfit_sweeps, ids_of
                )

        self.trees_ = trees
        self.n_ratio_splits_ = len(self.ratio_splits_)
        return self

    def decision_function(self, X):
        Xaug = self._augmented(predict_input(self, X, "trees_"))
        m = np.full(len(Xaug), self.base_margin_, dtype=np.float64)
        for tree in self.trees_:
            m += tree.value[tree.leaf_ids(Xaug)]
        return m

    def predict_contributions(self, X):
        Xaug = self._augmented(predict_input(self, X, "trees_"))
        out = np.zeros((len(Xaug), len(self.trees_)), dtype=np.float64)
        for k, tree in enumerate(self.trees_):
            out[:, k] = tree.value[tree.leaf_ids(Xaug)]
        return out

    def _cond_str(self, f, t, left, names, precision):
        name = (lambda j: str(names[j])) if names else (lambda j: f"x{j}")
        if f >= self.n_features_in_:
            s = self.ratio_splits_[f - self.n_features_in_]
            a, b = name(s["fi"]), name(s["fj"])
            if s["kind"] == 0:
                val = _fmt(np.exp(s["threshold"]), precision)
                return f"({a} / {b} <= {val} or ineligible)" if left else f"{a} / {b} > {val}"
            val = _fmt(s["threshold"], precision)
            return f"({a} - {b} <= {val} or missing)" if left else f"{a} - {b} > {val}"
        v = bin_threshold(self.bin_edges_[f], t)
        if np.isneginf(v):
            return f"{name(f)} is missing" if left else f"{name(f)} is present"
        op = "<=" if left else ">"
        return f"{name(f)} {op} {_fmt(v, precision)}"

    def rules(self, feature_names=None, precision=6):
        check_is_fitted(self, "trees_")
        names = self._names(feature_names)
        out = []
        for k, tree in enumerate(self.trees_):
            def walk(node, path, tree=tree, k=k):
                if tree.left[node] == -1:
                    out.append((k, list(path), float(tree.value[node])))
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                walk(tree.left[node], [*path, self._cond_str(f, t, True, names, precision)])
                walk(tree.right[node], [*path, self._cond_str(f, t, False, names, precision)])

            walk(0, [])
        return out
