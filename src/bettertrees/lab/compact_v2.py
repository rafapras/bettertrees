"""Standalone CompactTreeBoosterV2 without LightGBM, with stratified val and L2 refit (Spec R4).

Fixes three production limitations of ``CompactTreeBooster``:
1. Removes silent ``lightgbm`` dependency by defaulting ``feature_screen="fast_cached"``,
   which uses Numba ``top_features`` refreshed every ``screen_interval`` rounds.
2. Uses stratified validation splitting (on ``target > 0.5``) and learns ``bin_edges_``
   strictly on ``X[~val]`` to prevent validation threshold leakage.
3. When stopping is budget-bound (``best_round`` within the last 5 boosting rounds),
   refits leaf values and ``base_margin_`` on 100% of rows with calibrated exact L2
   regularization ``lam_eff = lam / learning_rate`` (``"shrunk_l2"``) or ``"full"``.
"""

from numbers import Integral

import numpy as np

from ..sums._common import (
    base_margin,
    binned,
    fit_inputs,
    grad_hess,
    log_loss_margin as _log_loss,
    rebin,
    top_features,
)
from ..sums._kernels import best_cut_1d, hist_1d, newton_leaf_values, quadrant_scores
from ..sums.compact import CompactTreeBooster, _TermSet
from ..sums.edit import _key, _reachable
from ..sums.smalltrees import SmallTree
from .l2_figs import base_l2_step_inplace, newton_l2_step_inplace

SCREEN_MODES = ("fast_cached", "fast", "lgbm")
REFIT_BUDGET_MODES = ("off", "shrunk_l2", "full")


def _coarse_all_pairs_top_features(
    Xt, g, h, w, nb, lam, min_weight, k, Xq, nb_q, sub_idx=None
):
    """FAST screening pairing 32-bin 1D gains with coarse 4-bin all-pairs interaction scores."""
    n, p = Xt.shape
    if k is None or p <= k:
        return None
    gains_1d, _ = best_cut_1d(hist_1d(Xt, g, h, w, int(nb.max())), nb, lam, min_weight)
    score = gains_1d.copy()
    if sub_idx is not None:
        gs = np.ascontiguousarray(g[sub_idx])
        hs = np.ascontiguousarray(h[sub_idx])
        ws = np.ascontiguousarray(w[sub_idx])
        scale = float(w.sum()) / max(float(ws.sum()), 1e-12)
        lam_s = lam / scale
        mw_s = min_weight / scale
    else:
        gs, hs, ws = g, h, w
        scale = 1.0
        lam_s, mw_s = lam, min_weight
    g1_q, _ = best_cut_1d(hist_1d(Xq, gs, hs, ws, int(nb_q.max())), nb_q, lam_s, mw_s)
    if p <= 160:
        pairs = np.array(
            [(i, j) for i in range(p) for j in range(i + 1, p)], dtype=np.int64
        )
    else:
        top = np.argsort(-gains_1d, kind="stable")[:32]
        pairs = np.array([(i, j) for i in top for j in range(p) if j != i], dtype=np.int64)
    if len(pairs):
        q, _, _ = quadrant_scores(Xq, gs, hs, ws, nb_q, pairs, lam_s, mw_s)
        for idx_p, (i, j) in enumerate(pairs):
            pure = max(0.0, float(q[idx_p]) - max(float(g1_q[i]), float(g1_q[j]))) * scale
            if pure > score[i]:
                score[i] = pure
            if pure > score[j]:
                score[j] = pure
    return np.sort(np.argsort(-score, kind="stable")[:k])


class CompactTreeBoosterV2(CompactTreeBooster):
    """Autonomous ``CompactTreeBooster`` with FAST-cached screening and post-budget L2 refit.

    Parameters
    ----------
    max_splits, depth, learning_rate, lam, max_rounds, min_weight, max_bins,
    validation_fraction, patience, refit, refit_lam, refit_sweeps,
    max_features_d2, max_features_d3, feature_screen, new_cut_penalty,
    grow_top, max_depth, max_leaves, random_state
        As in ``CompactTreeBooster`` (default ``feature_screen="fast_cached"``).
    screen_interval : int, default=25
        Number of rounds between cached ``top_features`` refreshes when
        ``feature_screen="fast_cached"``.
    stratified_val : bool, default=True
        If True, stratify the internal validation split by ``target > 0.5`` and
        fit ``bin_edges_`` strictly on the training fold ``X[~val]``.
    refit_when_budget_bound : {"off", "shrunk_l2", "full"}, default="shrunk_l2"
        When ``refit=False`` and training terminates at the cut budget (with
        ``best_round`` within the final 5 rounds), refit leaves and
        ``base_margin_`` on 100% of ``X`` using exact L2 backfitting with
        ``lam / learning_rate`` (``"shrunk_l2"``) or ``refit_lam`` (``"full"``).
    """

    def __init__(
        self,
        *,
        max_splits=64,
        depth=2,
        learning_rate=0.2,
        lam=1.0,
        max_rounds=2000,
        min_weight=20.0,
        max_bins=32,
        validation_fraction=0.15,
        patience=50,
        refit=False,
        refit_lam=None,
        refit_sweeps=4,
        max_features_d2=32,
        max_features_d3=12,
        feature_screen="fast_cached",
        new_cut_penalty=1.0,
        grow_top=0,
        max_depth=4,
        max_leaves=0,
        random_state=0,
        screen_interval=25,
        stratified_val=True,
        refit_when_budget_bound="shrunk_l2",
    ):
        super().__init__(
            max_splits=max_splits,
            depth=depth,
            learning_rate=learning_rate,
            lam=lam,
            max_rounds=max_rounds,
            min_weight=min_weight,
            max_bins=max_bins,
            validation_fraction=validation_fraction,
            patience=patience,
            refit=refit,
            refit_lam=refit_lam,
            refit_sweeps=refit_sweeps,
            max_features_d2=max_features_d2,
            max_features_d3=max_features_d3,
            feature_screen=feature_screen,
            new_cut_penalty=new_cut_penalty,
            grow_top=grow_top,
            max_depth=max_depth,
            max_leaves=max_leaves,
            random_state=random_state,
        )
        self.screen_interval = screen_interval
        self.stratified_val = stratified_val
        self.refit_when_budget_bound = refit_when_budget_bound

    def _validate_v2(self):
        if self.depth not in (1, 2, 3, "auto"):
            raise ValueError("depth must be 1, 2, 3 or 'auto'.")
        if self.feature_screen not in SCREEN_MODES:
            raise ValueError(f"feature_screen must be one of {SCREEN_MODES}.")
        if self.refit_when_budget_bound not in REFIT_BUDGET_MODES:
            raise ValueError(f"refit_when_budget_bound must be one of {REFIT_BUDGET_MODES}.")
        if (
            isinstance(self.screen_interval, bool)
            or not isinstance(self.screen_interval, Integral)
            or self.screen_interval < 1
        ):
            raise ValueError("screen_interval must be an integer >= 1.")
        if not isinstance(self.stratified_val, (bool, np.bool_)):
            raise ValueError("stratified_val must be a boolean.")

    def _features(self, depth, Xb, g, h, w, nb):
        if depth == 1:
            return None
        if self.feature_screen == "lgbm":
            return self._fixed[depth]
        k = self.max_features_d2 if depth == 2 else self.max_features_d3
        if self.feature_screen == "fast_cached":
            last = self._last_screen_round.get(depth, -self.screen_interval)
            if self._round_idx - last >= self.screen_interval or self._fixed[depth] is None:
                Xq, nb_q, sub_idx = self._screen_cache
                self._fixed[depth] = _coarse_all_pairs_top_features(
                    Xb, g, h, w, nb, self.lam, self.min_weight, k, Xq, nb_q, sub_idx
                )
                self._last_screen_round[depth] = self._round_idx
            return self._fixed[depth]
        return top_features(Xb, g, h, w, nb, self.lam, self.min_weight, k)

    def _refit_l2_all(self, Xb, target, w, lam_val, sweeps):
        """Exact L2 Newton backfitting of leaves and ``base_margin_`` across all rows."""
        ids = [t.leaf_ids(Xb) for t in self.trees_]
        contribs = [t.value[ids[k]].copy() for k, t in enumerate(self.trees_)]
        margin = np.full(len(Xb), float(self.base_margin_), dtype=np.float64)
        for c in contribs:
            margin += c
        target = np.ascontiguousarray(target, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        for _ in range(int(sweeps)):
            for k, tree in enumerate(self.trees_):
                tree.value = np.ascontiguousarray(tree.value, dtype=np.float64)
                is_leaf = np.asarray(tree.left, dtype=np.int64) == -1
                newton_l2_step_inplace(
                    target, w, margin, contribs[k], ids[k], is_leaf, tree.value,
                    float(lam_val), 1.0, 0.0, False,
                )
            self.base_margin_ += base_l2_step_inplace(target, w, margin, 1.0)
        return margin

    def _split_validation(self, target, rng):
        n = len(target)
        val = np.zeros(n, dtype=bool)
        if self.validation_fraction <= 0:
            return val
        n_val = int(round(self.validation_fraction * n))
        if n_val <= 0:
            return val
        if self.stratified_val:
            pos = np.flatnonzero(target > 0.5)
            neg = np.flatnonzero(target <= 0.5)
            if len(pos) > 0 and len(neg) > 0:
                n_pos = int(round(n_val * len(pos) / n))
                n_pos = min(max(n_pos, 0), min(n_val, len(pos)))
                n_neg = n_val - n_pos
                if n_neg > len(neg):
                    n_neg = len(neg)
                    n_pos = min(len(pos), n_val - n_neg)
                if n_pos > 0:
                    val[rng.permutation(pos)[:n_pos]] = True
                if n_neg > 0:
                    val[rng.permutation(neg)[:n_neg]] = True
                return val
        val[rng.permutation(n)[:n_val]] = True
        return val

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Boost, merge, and optionally apply calibrated full-sample L2 refit."""
        self._validate_v2()
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        self.lam_ = float(self.lam)
        self.learning_rate_ = 1.0
        rng = np.random.default_rng(self.random_state)
        val = self._split_validation(target, rng)
        self.val_mask_ = val
        tr = ~val
        if self.stratified_val:
            Xt, edges, nb = binned(X[tr], self.max_bins)
            Xv = rebin(X[val], edges) if val.any() else np.empty((0, X.shape[1]), dtype=np.uint8)
            Xb_all = None
        else:
            Xb_all, edges, nb = binned(X, self.max_bins)
            Xt, Xv = np.ascontiguousarray(Xb_all[tr]), np.ascontiguousarray(Xb_all[val])
        self.classes_, self.bin_edges_ = classes, edges
        yt, wt, yv, wv = target[tr], w[tr], target[val], w[val]

        self._fixed = {2: None, 3: None}
        self._last_screen_round = {}
        self._screen_cache = None
        if self.feature_screen == "fast_cached" and self.depth != 1:
            n_tr = len(Xt)
            max_screen_rows = 4096
            if n_tr > max_screen_rows:
                sub_idx = np.linspace(0, n_tr - 1, max_screen_rows, dtype=np.int64)
                X_sub = Xt[sub_idx]
            else:
                sub_idx = None
                X_sub = Xt
            Xq = X_sub.copy()
            nb_q = np.empty(Xt.shape[1], dtype=np.int64)
            for j in range(Xt.shape[1]):
                m_bins = int(nb[j]) - 1
                if m_bins <= 4:
                    nb_q[j] = nb[j]
                else:
                    pos = Xq[:, j] > 0
                    Xq[pos, j] = (
                        1 + ((Xq[pos, j].astype(np.int64) - 1) * 4) // m_bins
                    ).astype(np.uint8)
                    nb_q[j] = 5
            self._screen_cache = (np.ascontiguousarray(Xq), nb_q, sub_idx)
        elif self.feature_screen == "lgbm" and self.depth != 1:
            from ..sums._common import teacher_top_features

            labels = (yt > 0.5).astype(int)
            self._fixed[2] = teacher_top_features(
                X[tr], labels, self.max_features_d2, wt, self.random_state
            )
            if self.depth in (3, "auto"):
                self._fixed[3] = teacher_top_features(
                    X[tr], labels, self.max_features_d3, wt, self.random_state
                )

        base = base_margin(yt, wt)
        m_tr, m_val = np.full(len(yt), base), np.full(len(yv), base)
        budget = np.inf if self.max_splits is None else int(self.max_splits)
        terms, rounds, history = _TermSet(), [], []
        self._keys, self._templates, self._last_new = [], {}, np.inf
        self._idmat = np.zeros((64, len(yt)), dtype=np.int8)
        best, best_round = np.inf, 0
        stopped_by_budget = False
        for r_idx in range(int(self.max_rounds)):
            self._round_idx = r_idx
            g, h = grad_hess(yt, m_tr, wt)
            nxt = self._next(terms, Xt, g, h, wt, nb)
            if nxt is None:
                break
            key, tree = nxt
            if terms.cost(key, tree.n_splits) > budget - terms.n_cuts:
                stump = self._new_tree(1, Xt, g, h, wt, nb) if budget > terms.n_cuts else None
                if stump is None or terms.cost(_key(stump), 1) > budget - terms.n_cuts:
                    stopped_by_budget = True
                    if not val.any():
                        best_round = len(rounds)
                    break
                key, tree = _key(stump), stump
            ids = tree.leaf_ids(Xt)
            self._remember(key, tree, ids)
            values = self.learning_rate * newton_leaf_values(
                ids, g, h, len(tree.feature), self.lam
            )
            m_tr += values[ids]
            step = SmallTree(tree.feature, tree.threshold, tree.left, tree.right, values)
            terms.add(key, step)
            rounds.append((key, step))
            if val.any():
                m_val += values[tree.leaf_ids(Xv)]
                loss = _log_loss(yv, m_val, wv)
            else:
                loss = _log_loss(yt, m_tr, wt)
            history.append(loss)
            if loss < best - 1e-12:
                best, best_round = loss, len(rounds)
            elif len(rounds) - best_round >= self.patience:
                break
        del (
            self._keys,
            self._templates,
            self._idmat,
            self._last_new,
            self._fixed,
            self._last_screen_round,
            self._round_idx,
            self._screen_cache,
        )
        kept = _TermSet()
        for key, step in rounds[:best_round] if val.any() else rounds:
            kept.add(key, step)
        self.n_rounds_ = best_round if val.any() else len(rounds)
        self.history_ = np.array(history)
        self.base_margin_ = base
        self.trees_ = [_reachable(t) for t in kept.terms.values()]
        self.budget_refitted_ = False

        if self.refit and self.trees_:
            if Xb_all is None:
                Xb_all = rebin(X, self.bin_edges_)
            self._refit_all(Xb_all, target, w)
        elif (
            self.trees_
            and self.refit_when_budget_bound != "off"
            and (budget < np.inf)
            and (stopped_by_budget or terms.n_cuts >= budget)
            and ((not val.any()) or (len(rounds) - best_round) <= 5)
        ):
            if Xb_all is None:
                Xb_all = rebin(X, self.bin_edges_)
            if self.refit_when_budget_bound == "shrunk_l2":
                lam_eff = float(self.lam) / max(float(self.learning_rate), 1e-12)
            else:
                lam_eff = float(self.lam if self.refit_lam is None else self.refit_lam)
            self._refit_l2_all(Xb_all, target, w, lam_eff, self.refit_sweeps)
            self.budget_refitted_ = True
        return self
