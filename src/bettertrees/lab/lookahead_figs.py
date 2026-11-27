"""Lookahead FIGS (``best_depth2`` per cut) and rare-class collapse guards (Spec R2).

Extends ``FIGSClassifier`` with:

- ``lookahead``: when ``remaining_budget >= 2`` and a new tree is allowed,
  competes the optimal depth-2 tree (from ``optimal_tree(..., depth=2)``) scored
  as ``raw_gain / (c ** cost_exponent)`` (ties favor the 1D cut), resolving pure
  interactions (XOR, sign products) without forcing every tree to depth 2;
- ``max_depth``: caps tree depth so small budgets cannot collapse into one deep
  tree;
- ``min_leaf_coverage``: minimum fraction of total sample weight required to
  split an existing leaf at depth >= 1, preventing repeated refinement of a tiny
  high-prevalence pocket while the rest of the population stays constant.

With ``lookahead=False, max_depth=None, min_leaf_coverage=0.0``, delegates to
``FIGSClassifier`` and reproduces it bit for bit.
"""

from numbers import Integral, Real

import numpy as np

from ..sums._common import base_margin, grad_hess, top_features
from ..sums._kernels import best_leaf_cuts_T
from ..sums.edit import _reachable
from ..sums.smalltrees import FIGSClassifier, SmallTree, optimal_tree


def _node_depths(tree):
    """Depth of every node in ``tree`` (root = 0)."""
    depths = [0] * len(tree.feature)
    for k in range(len(tree.feature)):
        lft = tree.left[k]
        if lft != -1:
            d = depths[k] + 1
            depths[lft] = d
            depths[tree.right[k]] = d
    return depths


def _prune_node_inplace(tree, node):
    tree.left[node] = -1
    tree.right[node] = -1
    tree.feature[node] = -1
    tree.threshold[node] = -1


def _tree_gain(tree, Xb, g, h, lam):
    ids = tree.leaf_ids(Xb)
    n_nodes = len(tree.feature)
    G = np.bincount(ids, weights=g, minlength=n_nodes)
    H = np.bincount(ids, weights=h, minlength=n_nodes)
    Gt = float(g.sum())
    Ht = float(h.sum())
    return float(np.sum(G * G / (H + lam)) - Gt * Gt / (Ht + lam))


def _fit_d2_constraints(tree, Xb, g, h, w, lam, min_leaf_coverage, W_tot, max_budget):
    """Apply ``min_leaf_coverage`` and ``max_budget`` to a depth-2 candidate tree.

    Returns ``(tree, raw_gain)`` or ``None`` if no valid >= 2-split tree remains.
    """
    if min_leaf_coverage > 0.0:
        go_left = Xb[:, tree.feature[0]] <= tree.threshold[0]
        w_left = float(w[go_left].sum())
        w_right = W_tot - w_left
        thresh = min_leaf_coverage * W_tot
        lc, rc = tree.left[0], tree.right[0]
        changed = False
        if tree.left[lc] != -1 and w_left < thresh:
            _prune_node_inplace(tree, lc)
            changed = True
        if tree.left[rc] != -1 and w_right < thresh:
            _prune_node_inplace(tree, rc)
            changed = True
        if changed:
            tree = _reachable(tree)
    if tree.n_splits < 2:
        return None
    if tree.n_splits > max_budget:
        # Budget is 2 and candidate has 3 splits: keep the better depth-1 child split.
        best = None
        for child in (tree.left[0], tree.right[0]):
            if tree.left[child] == -1:
                continue
            cand = SmallTree(
                list(tree.feature), list(tree.threshold), list(tree.left),
                list(tree.right), tree.value.copy(),
            )
            _prune_node_inplace(cand, child)
            cand = _reachable(cand)
            gain = _tree_gain(cand, Xb, g, h, lam)
            if best is None or gain > best[0]:
                best = (gain, cand)
        if best is None or best[0] <= 0.0 or best[1].n_splits < 2:
            return None
        return best[1], best[0]
    gain = _tree_gain(tree, Xb, g, h, lam)
    if gain <= 0.0:
        return None
    return tree, gain


class LookaheadFIGSClassifier(FIGSClassifier):
    """FIGS with depth-2 lookahead splits and depth/coverage collapse guards.

    Parameters
    ----------
    max_splits, max_trees, lam, min_weight, max_bins, backfit_sweeps,
    learning_rate, max_delta_step
        As in ``FIGSClassifier``.
    lookahead : bool, default=True
        When True and at least 2 splits remain in budget (and ``max_trees`` /
        ``max_depth`` allow it), also evaluate an optimal depth-2 tree against
        the current margin, competing on ``raw_gain / (c ** cost_exponent)``.
    max_depth : int or None, default=None
        Maximum allowed tree depth (root = 0, stump = 1).
    min_leaf_coverage : float, default=0.0
        Minimum fraction of total sample weight a leaf at depth >= 1 must hold
        to be eligible for splitting.
    max_features_d2 : int or None, default=32
        FAST screening cap for the depth-2 lookahead search when p is large.
    cost_exponent : float, default=1.0
        Cut-cost exponent in ``raw_gain / (c ** cost_exponent)`` for depth-2
        candidates (``c`` is the number of cuts in the candidate tree).
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
        lookahead=True,
        max_depth=None,
        min_leaf_coverage=0.0,
        max_features_d2=32,
        cost_exponent=1.0,
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
        self.lookahead = lookahead
        self.max_depth = max_depth
        self.min_leaf_coverage = min_leaf_coverage
        self.max_features_d2 = max_features_d2
        self.cost_exponent = cost_exponent

    def _validate_lookahead(self):
        if not isinstance(self.lookahead, (bool, np.bool_)):
            raise ValueError("lookahead must be a boolean.")
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, Integral)
            or self.max_depth < 1
        ):
            raise ValueError("max_depth must be None or an integer >= 1.")
        if (
            not isinstance(self.min_leaf_coverage, Real)
            or isinstance(self.min_leaf_coverage, bool)
            or not 0.0 <= float(self.min_leaf_coverage) <= 1.0
        ):
            raise ValueError("min_leaf_coverage must be in [0.0, 1.0].")
        if self.max_features_d2 is not None and (
            isinstance(self.max_features_d2, bool)
            or not isinstance(self.max_features_d2, Integral)
            or self.max_features_d2 < 1
        ):
            raise ValueError("max_features_d2 must be None or an integer >= 1.")
        if (
            not isinstance(self.cost_exponent, Real)
            or isinstance(self.cost_exponent, bool)
            or not np.isfinite(self.cost_exponent)
            or float(self.cost_exponent) < 0.0
        ):
            raise ValueError("cost_exponent must be finite and >= 0.0.")

    def _grow_lookahead(self, Xb, nb, target, w):
        n, B = len(Xb), int(nb.max())
        XT = np.ascontiguousarray(Xb.T)
        trees, contribs, ids_of = [], [], []
        margin = np.full(n, self.base_margin_, dtype=np.float64)
        target = np.ascontiguousarray(target, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        W_tot = float(w.sum())
        root_ids, zero = np.zeros(n, dtype=np.int64), np.zeros(n)
        root_leaves = np.zeros(1, dtype=np.int64)
        spent = 0
        while spent < self.max_splits:
            remaining = self.max_splits - spent
            # (score, tree_idx, leaf_node, feat, threshold, d2_tree)
            best = (0.0, None, None, None, None, None)
            can_add = self.max_trees is None or len(trees) < self.max_trees
            candidates = list(range(len(trees))) + ([None] if can_add else [])
            for k in candidates:
                if k is None:
                    ids, contrib = root_ids, zero
                    elig_arr, n_nodes = root_leaves, 1
                else:
                    ids, contrib = ids_of[k], contribs[k]
                    n_nodes = len(trees[k].feature)
                    depths = _node_depths(trees[k])
                    elig = []
                    w_by_node = (
                        np.bincount(ids, weights=w, minlength=n_nodes)
                        if self.min_leaf_coverage > 0.0
                        else None
                    )
                    for leaf in trees[k].leaves:
                        d = depths[leaf]
                        if self.max_depth is not None and d >= self.max_depth:
                            continue
                        if (
                            d >= 1
                            and w_by_node is not None
                            and float(w_by_node[leaf]) < self.min_leaf_coverage * W_tot
                        ):
                            continue
                        elig.append(leaf)
                    if not elig:
                        continue
                    elig_arr = np.asarray(elig, dtype=np.int64)
                gains_mat, cuts_mat = best_leaf_cuts_T(
                    XT, nb, target, w, margin, contrib, ids, elig_arr, n_nodes, B,
                    self.lam_, self.min_weight,
                )
                for li in range(len(elig_arr)):
                    f = int(np.argmax(gains_mat[li]))
                    g_val = float(gains_mat[li, f])
                    if g_val > best[0]:
                        best = (g_val, k, int(elig_arr[li]), f, int(cuts_mat[li, f]), None)

            if (
                self.lookahead
                and remaining >= 2
                and can_add
                and (self.max_depth is None or self.max_depth >= 2)
            ):
                g, h = grad_hess(target, margin, w)
                feats = top_features(
                    Xb, g, h, w, nb, self.lam_, self.min_weight, self.max_features_d2
                )
                d2 = optimal_tree(Xb, g, h, w, nb, 2, self.lam_, self.min_weight, feats)
                if d2 is not None:
                    checked = _fit_d2_constraints(
                        d2, Xb, g, h, w, self.lam_, self.min_leaf_coverage, W_tot, remaining
                    )
                    if checked is not None:
                        d2_clean, raw_gain = checked
                        c = d2_clean.n_splits
                        score = raw_gain / (c ** float(self.cost_exponent))
                        if score > best[0]:  # strict > favors 1D unit cut on ties
                            best = (score, None, None, -1, -1, d2_clean)

            _, k, leaf, f, t, d2_tree = best
            if d2_tree is not None:
                trees.append(d2_tree)
                k = len(trees) - 1
                ids_of.append(d2_tree.leaf_ids(Xb))
                contribs.append(np.zeros(n))
                spent += d2_tree.n_splits
            else:
                if k is None and leaf is None:
                    break
                if k is None:
                    trees.append(SmallTree())
                    contribs.append(np.zeros(n))
                    ids_of.append(None)
                    k = len(trees) - 1
                trees[k].split(leaf, f, t)
                ids_of[k] = trees[k].leaf_ids(Xb)
                spent += 1
            contribs[k], margin = self._newton_step(
                trees[k], Xb, target, margin, contribs[k], w, ids_of[k]
            )
            if self.backfit_sweeps:
                margin = self._backfit(trees, contribs, Xb, target, w, self.backfit_sweeps, ids_of)
        return trees

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Fit the lookahead / depth- and coverage-guarded FIGS model."""
        self._validate_lookahead()
        if not self.lookahead and self.max_depth is None and self.min_leaf_coverage == 0.0:
            return super().fit(X, y, sample_weight=sample_weight, y_soft=y_soft)
        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        _, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        self.trees_ = self._grow_lookahead(Xb, nb, target, w)
        return self
