"""Compact boosting: shrunken boosting of optimal shallow trees, counted in
DISTINCT cuts.

Boosting with shrinkage keeps re-using the same cuts: after merging identical
trees (and absorbing a stump into a larger tree with the same root), a model of
a few hundred rounds may only use a few dozen distinct cuts. Every round this
estimator compares

- the optimal Newton tree of each allowed depth (1-3), a NEW structure, and
- re-boosting a tree already in the model (no new cut),

scoring ``gain / (1 + new_cut_penalty * new_cuts)``, and adds the winner with a
learning rate. It stops when the next structure would exceed ``max_splits``
distinct cuts (or when the validation loss stops improving). The result is a
plain sum of trees: step functions per feature plus a few interaction trees.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .._typing import ArrayLike, Seed, SelfT
from ._common import base_margin, binned, fit_inputs, grad_hess, teacher_top_features
from ._kernels import best_cut_1d, newton_leaf_values, node_hist, reuse_gains
from .additive import _log_loss
from .edit import _key, _reachable
from .smalltrees import SmallTree, _AdditiveTrees, optimal_tree

_SLOTS = 32  # leaf-id width of the reuse kernel: trees of at most 31 nodes (15 cuts)


def _subtree_leaves(tree, node):
    stack, out = [node], []
    while stack:
        k = stack.pop()
        if tree.left[k] == -1:
            out.append(k)
        else:
            stack += [tree.left[k], tree.right[k]]
    return out


def _refines(big, small):
    """True if the partition of structure ``big`` refines that of ``small``
    (``small`` is ``big`` with some subtrees pruned to leaves)."""
    if small is None:
        return True
    if big is None or big[:2] != small[:2]:
        return False
    return _refines(big[2], small[2]) and _refines(big[3], small[3])


def _n_cuts(key):
    return 0 if key is None else 1 + _n_cuts(key[2]) + _n_cuts(key[3])


def _absorb(host, node, small, snode):
    """Add the leaf values of ``small`` (a coarsening of ``host``) to ``host``'s leaves."""
    if small.left[snode] == -1:
        for leaf in _subtree_leaves(host, node):
            host.value[leaf] += small.value[snode]
        return
    _absorb(host, host.left[node], small, small.left[snode])
    _absorb(host, host.right[node], small, small.right[snode])


def _leafwise_tree(Xb, g, h, w, nb, max_leaves, lam, min_weight):
    """Greedy best-first (leaf-wise) Newton tree with at most ``max_leaves`` leaves,
    the tree shape LightGBM grows; None if no cut gains."""
    n, B = len(Xb), int(nb.max())
    tree = SmallTree()
    ids = np.zeros(n, dtype=np.int64)
    best = {}

    def score(nodes):
        local = np.full(n, -1, dtype=np.int64)
        for j, k in enumerate(nodes):
            local[ids == k] = j
        hist = node_hist(Xb, g, h, w, local, len(nodes), B)
        for j, k in enumerate(nodes):
            gains, cuts = best_cut_1d(hist[j], nb, lam, min_weight)
            f = int(np.argmax(gains))
            if gains[f] > 0:
                best[k] = (float(gains[f]), f, int(cuts[f]))

    score([0])
    while best and len(tree.leaves) < max_leaves:
        k = max(best, key=lambda x: best[x][0])
        _, f, t = best.pop(k)
        left, right = tree.split(k, f, t)
        rows = ids == k
        go_left = Xb[:, f] <= t
        ids[rows & go_left] = left
        ids[rows & ~go_left] = right
        score([left, right])
    return tree if tree.n_splits else None


class _TermSet:
    """Merged terms {structure key: SmallTree}. A tree whose partition is refined
    by another one (identical, a stump with the same root, or a tree with a
    leaf split further) is absorbed into it, so only distinct cuts are paid."""

    def __init__(self):
        self.terms = {}

    def _host(self, key):
        if key in self.terms:
            return key
        for k in self.terms:
            if _refines(k, key):
                return k
        return None

    def _absorbed(self, key):
        return [k for k in self.terms if k != key and _refines(key, k)]

    def cost(self, key, n_splits=None):
        """New distinct cuts if a tree with this structure is added."""
        if self._host(key) is not None:
            return 0
        return max(0, _n_cuts(key) - sum(_n_cuts(k) for k in self._absorbed(key)))

    def add(self, key, tree):
        """Add ``tree`` (structure ``key``, values on its nodes)."""
        host = self._host(key)
        if host == key:
            self.terms[key].value = self.terms[key].value + tree.value
            return
        if host is not None:
            _absorb(self.terms[host], 0, tree, 0)
            return
        new = SmallTree(list(tree.feature), list(tree.threshold), list(tree.left),
                        list(tree.right), np.array(tree.value, dtype=np.float64))
        for k in self._absorbed(key):
            _absorb(new, 0, self.terms.pop(k), 0)
        self.terms[key] = new

    @property
    def n_cuts(self):
        return sum(t.n_splits for t in self.terms.values())


class CompactTreeBooster(_AdditiveTrees):
    """Boosting of optimal shallow trees under a budget of distinct cuts.

    Parameters
    ----------
    max_splits : int or None, default=64
        Budget of distinct cuts of the merged model (None = early stopping only).
    depth : {1, 2, 3, "auto"}, default=2
        Depth of the new optimal trees; "auto" compares the optimal stump and
        depth-2 tree every round (gain per new cut). Depth 3 is exhaustive within
        the ``max_features_d3`` most important features (slow).
    learning_rate : float, default=0.2
    lam : float, default=1.0
        L2 regularization of the leaves.
    max_rounds : int, default=2000
    min_weight : float, default=20.0
        Minimum sample weight per leaf.
    max_bins : int, default=32
    validation_fraction : float, default=0.15
        Rows held out for early stopping (0 disables it).
    patience : int, default=50
    refit : bool, default=False
        Re-estimate all leaves jointly at the end (full Newton backfitting on
        all rows, penalty ``refit_lam``).
    refit_lam : float or None, default=None
        None = ``lam``.
    refit_sweeps : int, default=4
    max_features_d2 : int, default=32
        Depth-2 search restricted to the most important features (optimal within).
    max_features_d3 : int, default=12
    feature_screen : {"lgbm", "fast"}, default="lgbm"
    new_cut_penalty : float, default=1.0
        0 = plain boosting (always the best new tree); larger values spend the
        budget more slowly and boost the chosen structure longer.
    grow_top : int, default=0
        Also try to split one leaf of the ``grow_top`` existing trees with the
        largest re-boosting gain (the FIGS move; 1 new cut, the old tree is
        absorbed). This lets a term get deeper only where the data asks for it.
        0 disables it.
    max_depth : int, default=4
        Depth limit of grown trees.
    max_leaves : int, default=0
        If >= 3, also offer every round a greedy leaf-wise tree with up to this
        many leaves (LightGBM's shape), scored per new cut like the others. 0 = off.
    random_state : int, default=0

    Attributes
    ----------
    trees_ : list of SmallTree
        The merged terms (cuts in bins; ``get_trees()`` gives real thresholds).
    base_margin_, classes_, bin_edges_, nan_features_, ...
    n_rounds_ : int
        Boosting rounds kept (before merging).
    history_ : ndarray
        Validation log-loss per round.
    """

    def __init__(self, *, max_splits: int = 64, depth: Literal[1, 2, 3, "auto"] = 2,
                 learning_rate: float = 0.2, lam: float = 1.0, max_rounds: int = 2000,
                 min_weight: float = 20.0, max_bins: int = 32, validation_fraction: float = 0.15,
                 patience: int = 50, refit: bool = False, refit_lam: float | None = None,
                 refit_sweeps: int = 4, max_features_d2: int | None = 32,
                 max_features_d3: int | None = 12,
                 feature_screen: Literal["lgbm", "fast"] = "lgbm", new_cut_penalty: float = 1.0,
                 grow_top: int = 0, max_depth: int = 4, max_leaves: int = 0,
                 random_state: Seed = 0) -> None:
        self.max_splits = max_splits
        self.depth = depth
        self.learning_rate = learning_rate
        self.lam = lam
        self.max_rounds = max_rounds
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.refit = refit
        self.refit_lam = refit_lam
        self.refit_sweeps = refit_sweeps
        self.max_features_d2 = max_features_d2
        self.max_features_d3 = max_features_d3
        self.feature_screen = feature_screen
        self.new_cut_penalty = new_cut_penalty
        self.grow_top = grow_top
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.random_state = random_state

    # ---------------------------------------------------------------- search

    def _features(self, depth, Xb, g, h, w, nb):
        if depth == 1:
            return None
        if self.feature_screen == "lgbm":
            return self._fixed[depth]
        from ._common import top_features
        k = self.max_features_d2 if depth == 2 else self.max_features_d3
        return top_features(Xb, g, h, w, nb, self.lam, self.min_weight, k)

    def _new_tree(self, depth, Xb, g, h, w, nb):
        return optimal_tree(Xb, g, h, w, nb, depth, self.lam, self.min_weight,
                            self._features(depth, Xb, g, h, w, nb))

    def _gain(self, tree, Xb, g, h):
        ids = tree.leaf_ids(Xb)
        n = len(tree.feature)
        G = np.bincount(ids, weights=g, minlength=n)
        H = np.bincount(ids, weights=h, minlength=n)
        return float(np.sum(G * G / (H + self.lam)) - G.sum() ** 2 / (H.sum() + self.lam))

    def _reuse(self, g, h):
        if not self._keys:
            return np.zeros(0)
        return reuse_gains(self._idmat, len(self._keys), g, h, float(self.lam), _SLOTS)

    def _grow(self, key, base_gain, Xb, g, h, w, nb):
        """(raw gain, tree) of ``key``'s template with its best leaf split, or None."""
        tree = self._templates[key]
        if len(tree.feature) + 2 > _SLOTS:
            return None
        depth = {0: 0}
        for k in range(len(tree.feature)):
            if tree.left[k] != -1:
                depth[tree.left[k]] = depth[tree.right[k]] = depth[k] + 1
        leaves = [k for k in tree.leaves if depth[k] < self.max_depth]
        if not leaves:
            return None
        ids = self._idmat[self._keys.index(key)].astype(np.int64)
        hist = node_hist(Xb, g, h, w, ids, len(tree.feature), int(nb.max()))
        best = (0.0, None, None, None)
        for leaf in leaves:
            gains, cuts = best_cut_1d(hist[leaf], nb, self.lam, self.min_weight)
            f = int(np.argmax(gains))
            if gains[f] > best[0]:
                best = (float(gains[f]), leaf, f, int(cuts[f]))
        if best[1] is None:
            return None
        new = SmallTree(list(tree.feature), list(tree.threshold), list(tree.left),
                        list(tree.right), np.zeros(len(tree.feature)))
        new.split(best[1], best[2], best[3])
        return base_gain + best[0], new

    def _next(self, terms, Xb, g, h, w, nb):
        """(key, tree without values) of this round's term, or None."""
        depths = (1, 2) if self.depth == "auto" else (int(self.depth),)
        pen = float(self.new_cut_penalty)
        gains = self._reuse(g, h) if pen > 0 else np.zeros(0)
        key = self._keys[int(np.argmax(gains))] if len(gains) else None
        gain = float(gains.max()) if len(gains) else 0.0
        # lazy search: raw gains shrink as boosting proceeds, so when re-boosting
        # already beats the last new-structure gain, a new tree cannot win this round
        if key is not None and gain >= self._last_new / (1 + pen):
            return key, self._templates[key]
        scored = [(gain, key, self._templates[key])] if key is not None else []
        best_raw = 0.0
        for j in np.argsort(-gains)[:int(self.grow_top)] if pen > 0 else []:
            grown = self._grow(self._keys[j], float(gains[j]), Xb, g, h, w, nb)
            if grown is None:
                continue
            raw, tree = grown
            best_raw = max(best_raw, raw)
            k = _key(tree)
            scored.append((raw / (1 + pen * terms.cost(k)), k, tree))
        extra = []
        if int(self.max_leaves) >= 3:
            lw = _leafwise_tree(Xb, g, h, w, nb, min(int(self.max_leaves), _SLOTS // 2),
                                self.lam, self.min_weight)
            if lw is not None:
                extra.append(lw)
        for tree in [self._new_tree(d, Xb, g, h, w, nb) for d in depths] + extra:
            if tree is None:
                continue
            raw = self._gain(tree, Xb, g, h)
            best_raw = max(best_raw, raw)
            k = _key(tree)
            scored.append((raw / (1 + pen * terms.cost(k, tree.n_splits)), k, tree))
        self._last_new = best_raw
        scored = [s for s in scored if s[0] > 0]
        if not scored:
            return None
        _, k, tree = max(scored, key=lambda s: s[0])
        return k, tree

    def _remember(self, key, tree, ids):
        if key in self._templates:
            return
        if len(self._keys) == len(self._idmat):
            self._idmat = np.concatenate([self._idmat, np.zeros_like(self._idmat)])
        self._idmat[len(self._keys)] = ids
        self._keys.append(key)
        self._templates[key] = tree

    # ---------------------------------------------------------------- fit

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None,
            y_soft: ArrayLike | None = None) -> SelfT:
        """Boost, merge and (optionally) refit; see the class docstring."""
        if self.depth not in (1, 2, 3, "auto"):
            raise ValueError("depth must be 1, 2, 3 or 'auto'.")
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        self.lam_ = float(self.lam)
        self.learning_rate_ = 1.0  # used by the final full-step refit
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        val = np.zeros(n, dtype=bool)
        if self.validation_fraction > 0:
            val[rng.permutation(n)[:int(round(self.validation_fraction * n))]] = True
        tr = ~val
        Xb, edges, nb = binned(X, self.max_bins)
        self.classes_, self.bin_edges_ = classes, edges
        Xt, Xv = np.ascontiguousarray(Xb[tr]), np.ascontiguousarray(Xb[val])
        yt, wt, yv, wv = target[tr], w[tr], target[val], w[val]
        self._fixed = {2: None, 3: None}
        if self.feature_screen == "lgbm" and self.depth != 1:
            labels = (yt > 0.5).astype(int)
            self._fixed[2] = teacher_top_features(X[tr], labels, self.max_features_d2, wt,
                                                  self.random_state)
            if self.depth in (3, "auto"):
                self._fixed[3] = teacher_top_features(X[tr], labels, self.max_features_d3, wt,
                                                      self.random_state)
        base = base_margin(yt, wt)
        m_tr, m_val = np.full(len(yt), base), np.full(len(yv), base)
        budget = np.inf if self.max_splits is None else int(self.max_splits)
        terms, rounds, history = _TermSet(), [], []
        self._keys, self._templates, self._last_new = [], {}, np.inf
        self._idmat = np.zeros((64, len(yt)), dtype=np.int8)
        best, best_round = np.inf, 0
        for _ in range(int(self.max_rounds)):
            g, h = grad_hess(yt, m_tr, wt)
            nxt = self._next(terms, Xt, g, h, wt, nb)
            if nxt is None:
                break
            key, tree = nxt
            if terms.cost(key, tree.n_splits) > budget - terms.n_cuts:
                # budget nearly spent: a stump may still fit; otherwise the structure
                # is final (the optional joint refit replaces further shrunken steps)
                stump = self._new_tree(1, Xt, g, h, wt, nb) if budget > terms.n_cuts else None
                if stump is None or terms.cost(_key(stump), 1) > budget - terms.n_cuts:
                    if not val.any():
                        best_round = len(rounds)
                    break
                key, tree = _key(stump), stump
            ids = tree.leaf_ids(Xt)
            self._remember(key, tree, ids)
            values = self.learning_rate * newton_leaf_values(ids, g, h, len(tree.feature),
                                                             self.lam)
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
        del self._keys, self._templates, self._idmat, self._last_new, self._fixed
        kept = _TermSet()
        for key, step in rounds[:best_round] if val.any() else rounds:
            kept.add(key, step)
        self.n_rounds_ = best_round if val.any() else len(rounds)
        self.history_ = np.array(history)
        self.base_margin_ = base
        self.trees_ = [_reachable(t) for t in kept.terms.values()]
        if self.refit and self.trees_:
            self._refit_all(Xb, target, w)
        return self

    def _refit_all(self, Xb, target, w):
        """Fully corrective refit of every leaf and the base on all rows (warm start)."""
        lam = float(self.lam if self.refit_lam is None else self.refit_lam)
        self._refit_binned(Xb, target, w, lam, self.refit_sweeps)
