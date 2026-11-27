"""Exact L2-regularized backfitting and free intercept for FIGS (Spec R1).

In production ``FIGSClassifier``, each Newton step adds ``-G / (H + lam)`` to the
current leaf values ``v``, which corresponds to a proximal penalty
``(lam / 2) * ||v - v_old||^2`` rather than ``(lam / 2) * ||v||^2``. Over S
sweeps, shrinkage on early trees decays exponentially. Here, the Newton step for
``log-loss + (lam / 2) * sum_leaves v^2`` is

    Delta v_l = -(G_l + lam * v_l) / (H_l + lam)

paired with an unpenalized intercept step on ``base_margin_`` at the end of each
sweep. With ``penalty="proximal"`` and ``update_base=False``, the estimator
delegates to ``FIGSClassifier`` and matches it bit for bit.
"""

import numpy as np
from numba import njit, prange

from ..sums._common import base_margin, grad_hess
from ..sums._kernels import newton_leaf_values
from ..sums.edit import _project_monotone
from ..sums.smalltrees import FIGSClassifier, SmallTree

PENALTIES = ("l2", "proximal")


@njit(cache=True)
def _logistic_gh_l2(m, y, w):
    """(g, h) of one row; matches ``_kernels._logistic_gh`` bit for bit."""
    if m >= 0:
        p = 1.0 / (1.0 + np.exp(-m))
    else:
        e = np.exp(m)
        p = e / (1.0 + e)
    return w * (p - y), w * max(p * (1.0 - p), 1e-16)


@njit(cache=True)
def _best_cut_l2_hist(hist, nb, lam, min_weight):
    """Best cut of a histogram ``(nb, 3)``; matches ``_kernels._best_cut_hist``."""
    G = 0.0
    H = 0.0
    W = 0.0
    for b in range(nb):
        G += hist[b, 0]
        H += hist[b, 1]
        W += hist[b, 2]
    parent = G * G / (H + lam)
    best_gain = 0.0
    best_t = -1
    GL = 0.0
    HL = 0.0
    WL = 0.0
    for t in range(nb - 1):
        GL += hist[t, 0]
        HL += hist[t, 1]
        WL += hist[t, 2]
        if min_weight > WL or min_weight > W - WL:
            continue
        gain = GL * GL / (HL + lam) + (G - GL) * (G - GL) / (H - HL + lam) - parent
        if gain > best_gain:
            best_gain = gain
            best_t = t
    return best_gain, best_t


@njit(cache=True, parallel=True)
def fused_best_split_l2(
    Xb, nb, y, w, margin, contribs, ids_mat, leaf_slots, n_slots, lam, min_weight
):
    """Find best split across all candidate leaves/root in a single parallel pass over ``Xb``."""
    n, p = Xb.shape
    C = contribs.shape[0]
    B = 0
    for j in range(p):
        if nb[j] > B:
            B = nb[j]
    G_cand = np.empty((C, n), dtype=np.float64)
    H_cand = np.empty((C, n), dtype=np.float64)
    row_slot = np.empty((C, n), dtype=np.int64)
    for c in prange(C):
        for i in range(n):
            g, h = _logistic_gh_l2(margin[i] - contribs[c, i], y[i], w[i])
            G_cand[c, i] = g
            H_cand[c, i] = h
            row_slot[c, i] = leaf_slots[c, ids_mat[c, i]]

    slot_gain = np.zeros((p, n_slots), dtype=np.float64)
    slot_cut = np.full((p, n_slots), -1, dtype=np.int64)
    for j in prange(p):
        hist = np.zeros((n_slots, B, 3), dtype=np.float64)
        for i in range(n):
            b = Xb[i, j]
            wi = w[i]
            for c in range(C):
                s = row_slot[c, i]
                hist[s, b, 0] += G_cand[c, i]
                hist[s, b, 1] += H_cand[c, i]
                hist[s, b, 2] += wi
        nbj = nb[j]
        for s in range(n_slots):
            g_best, t_best = _best_cut_l2_hist(hist[s], nbj, lam, min_weight)
            slot_gain[j, s] = g_best
            slot_cut[j, s] = t_best

    best_gain = 0.0
    best_s = -1
    best_f = -1
    best_t = -1
    for s in range(n_slots):
        f_best = 0
        g_f = slot_gain[0, s]
        for j in range(1, p):
            if slot_gain[j, s] > g_f:
                g_f = slot_gain[j, s]
                f_best = j
        if g_f > best_gain:
            best_gain = g_f
            best_s = s
            best_f = f_best
            best_t = slot_cut[f_best, s]
    return best_gain, best_s, best_f, best_t


@njit(cache=True)
def fused_step_one_l2(
    k, y, w, margin, contribs, ids_mat, is_leaf_mat, val_mat, n_nodes, lam, lr, cap, has_cap
):
    """Single-tree exact L2 Newton step inside preallocated matrices."""
    n = len(y)
    G = np.zeros(n_nodes)
    H = np.zeros(n_nodes)
    for i in range(n):
        g, h = _logistic_gh_l2(margin[i], y[i], w[i])
        nd = ids_mat[k, i]
        G[nd] += g
        H[nd] += h
    for nd in range(n_nodes):
        if is_leaf_mat[k, nd]:
            denom = H[nd] + lam
            s = -(G[nd] + lam * val_mat[k, nd]) / denom if denom > 0.0 else 0.0
            if has_cap:
                s = min(max(s, -cap), cap)
            val_mat[k, nd] = val_mat[k, nd] + lr * s
    for i in range(n):
        new = val_mat[k, ids_mat[k, i]]
        margin[i] = margin[i] - contribs[k, i] + new
        contribs[k, i] = new


@njit(cache=True)
def fused_backfit_l2(
    y, w, base_margin_val, margin, contribs, ids_mat, is_leaf_mat, val_mat,
    n_nodes_arr, K, sweeps, lam, lr, cap, has_cap, update_base,
):
    """Exact L2 Newton backfit across ``K`` trees and optional free base step."""
    n = len(y)
    for i in range(n):
        acc = base_margin_val
        for k in range(K):
            acc += contribs[k, i]
        margin[i] = acc
    base_delta = 0.0
    for _ in range(sweeps):
        for k in range(K):
            L = n_nodes_arr[k]
            G = np.zeros(L)
            H = np.zeros(L)
            for i in range(n):
                g, h = _logistic_gh_l2(margin[i], y[i], w[i])
                nd = ids_mat[k, i]
                G[nd] += g
                H[nd] += h
            for nd in range(L):
                if is_leaf_mat[k, nd]:
                    denom = H[nd] + lam
                    s = -(G[nd] + lam * val_mat[k, nd]) / denom if denom > 0.0 else 0.0
                    if has_cap:
                        s = min(max(s, -cap), cap)
                    val_mat[k, nd] = val_mat[k, nd] + lr * s
            for i in range(n):
                new = val_mat[k, ids_mat[k, i]]
                margin[i] = margin[i] - contribs[k, i] + new
                contribs[k, i] = new
        if update_base:
            Gb = 0.0
            Hb = 0.0
            for i in range(n):
                g, h = _logistic_gh_l2(margin[i], y[i], w[i])
                Gb += g
                Hb += h
            denom = Hb if Hb > 1e-12 else 1e-12
            delta = lr * (-Gb / denom)
            for i in range(n):
                margin[i] += delta
            base_margin_val += delta
            base_delta += delta
    return base_delta


@njit(cache=True)
def newton_l2_step_inplace(y, w, margin, contrib, ids, is_leaf, value, lam, lr, cap, has_cap):
    """Exact L2 Newton step on active leaves (``is_leaf[k] == True``) of one tree.

    Updates ``value``, ``contrib`` and ``margin`` in place using
    ``s = -(G[k] + lam * value[k]) / (H[k] + lam)``.
    """
    n = len(y)
    L = len(value)
    G = np.zeros(L)
    H = np.zeros(L)
    for i in range(n):
        g, h = _logistic_gh_l2(margin[i], y[i], w[i])
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
def base_l2_step_inplace(y, w, margin, lr):
    """Unpenalized Newton intercept step on ``margin``; returns ``delta``."""
    n = len(y)
    G = 0.0
    H = 0.0
    for i in range(n):
        g, h = _logistic_gh_l2(margin[i], y[i], w[i])
        G += g
        H += h
    denom = H if H > 1e-12 else 1e-12
    delta = lr * (-G / denom)
    for i in range(n):
        margin[i] += delta
    return delta


def refit_leaves_l2(
    est,
    X,
    y,
    sample_weight=None,
    lam=None,
    sweeps=5,
    trees=None,
    refit_base=True,
    learning_rate=1.0,
    max_delta_step=None,
    monotone=None,
):
    """Re-estimate leaves (and optionally ``base_margin_``) with exact L2 Newton backfitting.

    Works on any fitted tree-sum estimator providing ``trees_``, ``bin_edges_``,
    ``base_margin_``, ``classes_`` and ``_binned_data``. Modifies ``est`` in place
    and returns ``est``.
    """
    Xb, target, w = est._binned_data(X, y, sample_weight)
    lam_val = float(lam if lam is not None else (getattr(est, "lam_", None) or 1.0))
    if trees is not None:
        for k in trees:
            est._check_node(k, 0)
    cons = est._constraints(monotone) if monotone else None
    active = range(len(est.trees_)) if trees is None else sorted(set(trees))
    ids = [t.leaf_ids(Xb) for t in est.trees_]
    contribs = [t.value[ids[k]].copy() for k, t in enumerate(est.trees_)]
    margin = np.full(len(Xb), float(est.base_margin_), dtype=np.float64)
    for c in contribs:
        margin += c
    lr = float(learning_rate)
    has_cap = max_delta_step is not None
    cap = 0.0 if not has_cap else float(max_delta_step)
    target = np.ascontiguousarray(target, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    for _ in range(int(sweeps)):
        for k in active:
            tree = est.trees_[k]
            tree.value = np.ascontiguousarray(tree.value, dtype=np.float64)
            is_leaf = np.asarray(tree.left, dtype=np.int64) == -1
            newton_l2_step_inplace(
                target, w, margin, contribs[k], ids[k], is_leaf, tree.value,
                lam_val, lr, cap, has_cap,
            )
            if cons:
                _, h = grad_hess(target, margin, w)
                mass = np.bincount(ids[k], weights=h, minlength=len(tree.feature))
                old = contribs[k].copy()
                _project_monotone(tree, cons, mass)
                contribs[k] = tree.value[ids[k]]
                margin = margin - old + contribs[k]
        if refit_base:
            est.base_margin_ += base_l2_step_inplace(target, w, margin, lr)
    return est


def _grow_l2_figs(est, Xb, nb, target, w, max_splits, max_trees=None):
    """FIGS growth supporting exact L2 leaf steps and free intercept updates."""
    n, B = len(Xb), int(nb.max())
    target = np.ascontiguousarray(target, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    margin = np.full(n, est.base_margin_, dtype=np.float64)
    if est.penalty != "l2":
        from ..sums._kernels import best_cut_1d, node_hist_margin

        trees, contribs, ids_of = [], [], []
        root_ids, zero = np.zeros(n, dtype=np.int64), np.zeros(n)
        for _ in range(max_splits):
            best = (0.0, None, None, None, None)
            can_add = max_trees is None or len(trees) < max_trees
            candidates = list(range(len(trees))) + ([None] if can_add else [])
            for k in candidates:
                if k is None:
                    ids, contrib = root_ids, zero
                    leaves, n_nodes = [0], 1
                else:
                    ids, contrib = ids_of[k], contribs[k]
                    leaves, n_nodes = trees[k].leaves, len(trees[k].feature)
                hist = node_hist_margin(Xb, target, w, margin, contrib, ids, n_nodes, B)
                for leaf in leaves:
                    gains, cuts = best_cut_1d(hist[leaf], nb, est.lam_, est.min_weight)
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
            contribs[k], margin = est._newton_step(
                trees[k], Xb, target, margin, contribs[k], w, ids_of[k]
            )
            if est.backfit_sweeps:
                margin = est._backfit(
                    trees, contribs, Xb, target, w, est.backfit_sweeps, ids_of
                )
            elif est.update_base:
                est.base_margin_ += base_l2_step_inplace(
                    target, w, margin, est.learning_rate_
                )
        return trees

    if max_splits <= 0 or (max_trees is not None and max_trees <= 0):
        return []

    cap_t = min(max_splits + 1, 16)
    cap_nodes = max(16, min(2 * max_splits + 1, 31))
    trees = []
    contribs = np.zeros((cap_t, n), dtype=np.float64)
    ids_mat = np.zeros((cap_t, n), dtype=np.int64)
    is_leaf_mat = np.zeros((cap_t, cap_nodes), dtype=np.bool_)
    val_mat = np.zeros((cap_t, cap_nodes), dtype=np.float64)
    leaf_slots = np.full((cap_t, cap_nodes), -1, dtype=np.int64)
    n_nodes_arr = np.zeros(cap_t, dtype=np.int64)
    cap_arg = getattr(est, "max_delta_step", None)
    has_cap = cap_arg is not None
    cap_val = 0.0 if not has_cap else float(cap_arg)

    for _ in range(max_splits):
        n_trees = len(trees)
        can_add = max_trees is None or n_trees < max_trees
        n_cand = n_trees + (1 if can_add else 0)
        if n_cand == 0:
            break
        if n_cand > cap_t:
            new_cap_t = max(n_cand, 2 * cap_t)
            c_new = np.zeros((new_cap_t, n), dtype=np.float64)
            c_new[:cap_t] = contribs
            contribs = c_new
            i_new = np.zeros((new_cap_t, n), dtype=np.int64)
            i_new[:cap_t] = ids_mat
            ids_mat = i_new
            il_new = np.zeros((new_cap_t, cap_nodes), dtype=np.bool_)
            il_new[:cap_t] = is_leaf_mat
            is_leaf_mat = il_new
            v_new = np.zeros((new_cap_t, cap_nodes), dtype=np.float64)
            v_new[:cap_t] = val_mat
            val_mat = v_new
            ls_new = np.full((new_cap_t, cap_nodes), -1, dtype=np.int64)
            ls_new[:cap_t] = leaf_slots
            leaf_slots = ls_new
            nn_new = np.zeros(new_cap_t, dtype=np.int64)
            nn_new[:cap_t] = n_nodes_arr
            n_nodes_arr = nn_new
            cap_t = new_cap_t

        slot_to_kl = []
        for k in range(n_trees):
            leaf_slots[k, : n_nodes_arr[k]] = -1
            for lf in trees[k].leaves:
                leaf_slots[k, lf] = len(slot_to_kl)
                slot_to_kl.append((k, lf))
        if can_add:
            leaf_slots[n_trees, 0] = len(slot_to_kl)
            slot_to_kl.append((None, 0))
            contribs[n_trees].fill(0.0)
            ids_mat[n_trees].fill(0)
        if not slot_to_kl:
            break

        _, best_s, f, t = fused_best_split_l2(
            Xb, nb, target, w, margin, contribs[:n_cand], ids_mat[:n_cand],
            leaf_slots[:n_cand], len(slot_to_kl), est.lam_, est.min_weight,
        )
        if best_s < 0:
            break
        k, leaf = slot_to_kl[best_s]
        if k is None:
            trees.append(SmallTree())
            k = len(trees) - 1
            n_nodes_arr[k] = 1
            is_leaf_mat[k, 0] = True
            val_mat[k, 0] = 0.0
        old_n = int(n_nodes_arr[k])
        if old_n + 2 > cap_nodes:
            new_cap_n = max(old_n + 2, 2 * cap_nodes)
            il_new = np.zeros((cap_t, new_cap_n), dtype=np.bool_)
            il_new[:, :cap_nodes] = is_leaf_mat
            is_leaf_mat = il_new
            v_new = np.zeros((cap_t, new_cap_n), dtype=np.float64)
            v_new[:, :cap_nodes] = val_mat
            val_mat = v_new
            ls_new = np.full((cap_t, new_cap_n), -1, dtype=np.int64)
            ls_new[:, :cap_nodes] = leaf_slots
            leaf_slots = ls_new
            cap_nodes = new_cap_n

        l_ch, r_ch = trees[k].split(leaf, f, t)
        is_leaf_mat[k, leaf] = False
        is_leaf_mat[k, l_ch] = True
        is_leaf_mat[k, r_ch] = True
        val_mat[k, l_ch] = val_mat[k, leaf]
        val_mat[k, r_ch] = val_mat[k, leaf]
        n_nodes_arr[k] = old_n + 2
        ids_mat[k] = trees[k].leaf_ids(Xb)

        fused_step_one_l2(
            k, target, w, margin, contribs, ids_mat, is_leaf_mat, val_mat,
            int(n_nodes_arr[k]), est.lam_, est.learning_rate_, cap_val, has_cap,
        )
        K_now = len(trees)
        if est.backfit_sweeps:
            est.base_margin_ += fused_backfit_l2(
                target, w, est.base_margin_, margin, contribs, ids_mat, is_leaf_mat,
                val_mat, n_nodes_arr, K_now, est.backfit_sweeps, est.lam_,
                est.learning_rate_, cap_val, has_cap, bool(est.update_base),
            )
        elif est.update_base:
            est.base_margin_ += base_l2_step_inplace(
                target, w, margin, est.learning_rate_
            )

    for k, tree in enumerate(trees):
        tree.value = val_mat[k, : n_nodes_arr[k]].copy()
    return trees


class L2FIGSClassifier(FIGSClassifier):
    """Experimental FIGS with true L2 leaf regularization and free intercept.

    Parameters
    ----------
    max_splits, max_trees, lam, min_weight, max_bins, backfit_sweeps,
    learning_rate, max_delta_step
        As in ``FIGSClassifier``.
    penalty : {"l2", "proximal"}, default="l2"
        ``"l2"`` penalizes ``(lam / 2) * ||v||^2``; ``"proximal"`` keeps the
        incremental ``-G / (H + lam)`` step of ``FIGSClassifier``.
    update_base : bool, default=True
        Whether to take an unpenalized Newton step on ``base_margin_`` at the
        end of each backfitting sweep.
    """

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
        penalty="l2",
        update_base=True,
    ):
        super().__init__(
            max_splits=max_splits,
            max_trees=max_trees,
            lam=lam,
            min_weight=min_weight,
            max_bins=max_bins,
            backfit_sweeps=backfit_sweeps,
            learning_rate=learning_rate,
            max_delta_step=max_delta_step,
        )
        self.penalty = penalty
        self.update_base = update_base

    def _validate_l2(self):
        if self.penalty not in PENALTIES:
            raise ValueError(f"penalty must be one of {PENALTIES}.")
        if not isinstance(self.update_base, (bool, np.bool_)):
            raise ValueError("update_base must be a boolean.")

    def _newton_step(self, tree, Xb, target, margin, contrib, w, ids=None):
        if self.penalty == "proximal":
            return super()._newton_step(tree, Xb, target, margin, contrib, w, ids=ids)
        cap = getattr(self, "max_delta_step", None)
        if ids is None:
            ids = tree.leaf_ids(Xb)
        tree.value = np.ascontiguousarray(tree.value, dtype=np.float64)
        is_leaf = np.asarray(tree.left, dtype=np.int64) == -1
        newton_l2_step_inplace(
            target, w, margin, contrib, ids, is_leaf, tree.value,
            self.lam_, self.learning_rate_, 0.0 if cap is None else float(cap), cap is not None,
        )
        return contrib, margin

    def _backfit(self, trees, contribs, Xb, target, w, sweeps, ids=None):
        if not self.update_base:
            return super()._backfit(trees, contribs, Xb, target, w, sweeps, ids=ids)
        margin = self.base_margin_ + np.sum(contribs, axis=0)
        for _ in range(sweeps):
            for k, tree in enumerate(trees):
                contribs[k], margin = self._newton_step(
                    tree, Xb, target, margin, contribs[k], w,
                    None if ids is None else ids[k],
                )
            self.base_margin_ += base_l2_step_inplace(target, w, margin, self.learning_rate_)
        return margin

    def _refit_binned(self, Xb, target, w, lam, sweeps, trees=None, refit_base=True,
                      monotone=None):
        if self.penalty == "proximal":
            return super()._refit_binned(
                Xb, target, w, lam, sweeps, trees=trees, refit_base=refit_base,
                monotone=monotone,
            )
        lam_old, lr_old = getattr(self, "lam_", None), getattr(self, "learning_rate_", None)
        self.lam_, self.learning_rate_ = float(lam), 1.0
        active = range(len(self.trees_)) if trees is None else sorted(set(trees))
        ids = [t.leaf_ids(Xb) for t in self.trees_]
        contribs = [t.value[ids[k]].copy() for k, t in enumerate(self.trees_)]
        margin = np.full(len(Xb), float(self.base_margin_), dtype=np.float64)
        for c in contribs:
            margin += c
        target = np.ascontiguousarray(target, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        for _ in range(int(sweeps)):
            for k in active:
                tree = self.trees_[k]
                tree.value = np.ascontiguousarray(tree.value, dtype=np.float64)
                is_leaf = np.asarray(tree.left, dtype=np.int64) == -1
                if monotone:
                    g, h = grad_hess(target, margin, w)
                    old = contribs[k].copy()
                    step = newton_leaf_values(ids[k], g, h, len(tree.feature), self.lam_)
                    # Adjust proximal step into exact L2 step on active leaves.
                    h_leaf = np.bincount(ids[k], weights=h, minlength=len(tree.feature))
                    adj = np.where(
                        is_leaf & ((h_leaf + self.lam_) > 0),
                        -self.lam_ * tree.value / np.maximum(h_leaf + self.lam_, 1e-12),
                        0.0,
                    )
                    tree.value = tree.value + np.where(is_leaf, step, 0.0) + adj
                    mass = np.bincount(ids[k], weights=h, minlength=len(tree.feature))
                    _project_monotone(tree, monotone, mass)
                    contribs[k] = tree.value[ids[k]]
                    margin = margin - old + contribs[k]
                else:
                    newton_l2_step_inplace(
                        target, w, margin, contribs[k], ids[k], is_leaf, tree.value,
                        self.lam_, 1.0, 0.0, False,
                    )
            if refit_base:
                self.base_margin_ += base_l2_step_inplace(target, w, margin, 1.0)
        self.lam_, self.learning_rate_ = lam_old, lr_old
        return margin

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Fit the sum with exact L2 regularization and optional base updates."""
        self._validate_l2()
        if self.penalty == "proximal" and not self.update_base:
            return super().fit(X, y, sample_weight=sample_weight, y_soft=y_soft)
        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        _, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        self.trees_ = _grow_l2_figs(self, Xb, nb, target, w, self.max_splits, self.max_trees)
        return self
