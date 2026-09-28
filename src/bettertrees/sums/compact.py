"""Compact boosting: shrunken boosting of optimal shallow trees, counted in
DISTINCT cuts.

Boosting with shrinkage keeps re-using the same cuts: after merging identical
terms (and absorbing a stump into a depth-2 tree with the same root), a model
of a few hundred rounds may only use a few dozen distinct cuts. This estimator
boosts optimal depth-1/2 Newton trees with a learning rate, merges as it goes,
stops when the merged model reaches ``max_splits`` distinct cuts (or when the
validation loss stops improving), and finally re-estimates every leaf jointly
(fully corrective backfitting on all rows). The result is a plain sum of trees:
one step function per feature plus a few pair interactions.
"""

import numpy as np

from ._common import base_margin, binned, fit_inputs, grad_hess, teacher_top_features
from ._kernels import depth2_leaf_ids, newton_leaf_values, reuse_gains
from .additive import AdditiveTreeBooster, _log_loss
from .smalltrees import SmallTree, _AdditiveTrees, _side


def _n_cuts(spec):
    return 1 + (spec[2] >= 0) + (spec[4] >= 0)


def _stump(spec):
    return spec[2] < 0 and spec[4] < 0


class _TermSet:
    """Merged terms: {spec: 4 leaf values}; a stump is absorbed into a depth-2
    term with the same root (its left value goes to leaves 0-1, right to 2-3)."""

    def __init__(self):
        self.terms = {}

    def host(self, spec):
        """Where ``spec`` would land: itself, or a depth-2 term that absorbs it."""
        if spec in self.terms:
            return spec
        if _stump(spec):
            for s in self.terms:
                if s[0] == spec[0] and s[1] == spec[1] and not _stump(s):
                    return s
        return None

    def cost(self, spec):
        """New distinct cuts if ``spec`` is added (0 when merged or absorbed)."""
        if self.host(spec) is not None:
            return 0
        freed = 0
        if not _stump(spec):  # an existing stump with this root gets absorbed
            root = (spec[0], spec[1], -1, -1, -1, -1)
            freed = 1 if root in self.terms else 0
        return _n_cuts(spec) - freed

    def add(self, spec, values):
        h = self.host(spec)
        if h is None:
            v = np.array(values, dtype=np.float64)
            if not _stump(spec):
                root = (spec[0], spec[1], -1, -1, -1, -1)
                if root in self.terms:
                    s = self.terms.pop(root)
                    v[:2] += s[0]
                    v[2:] += s[2]
            self.terms[spec] = v
        elif h == spec:
            self.terms[h] += values
        else:  # stump absorbed by a depth-2 term
            self.terms[h][:2] += values[0]
            self.terms[h][2:] += values[2]

    @property
    def n_cuts(self):
        return sum(_n_cuts(s) for s in self.terms)


class CompactTreeBooster(_AdditiveTrees):
    """Boosting of optimal shallow trees under a budget of distinct cuts.

    Parameters
    ----------
    max_splits : int or None, default=64
        Budget of distinct cuts of the merged model (None = early stopping only).
    depth : {1, 2, "auto"}, default=2
        Depth of each boosted tree (1 = an additive model of step functions;
        "auto" = per round, the optimal stump or depth-2 tree with the larger
        gain per new distinct cut).
    learning_rate : float, default=0.1
    lam : float, default=1.0
        L2 regularization of the leaves during boosting.
    max_rounds : int, default=2000
    min_weight : float, default=20.0
        Minimum sample weight per leaf.
    max_bins : int, default=32
    validation_fraction : float, default=0.15
        Rows held out for early stopping (0 disables it).
    patience : int, default=50
    refit : bool, default=True
        Re-estimate all leaves jointly at the end (full Newton backfitting on
        all rows, penalty ``refit_lam``).
    refit_lam : float or None, default=None
        None = ``lam``.
    refit_sweeps : int, default=4
    max_features_d2 : int, default=32
        Depth-2 search restricted to the features with the largest gain
        (optimal within that set); keeps a round in milliseconds.
    feature_screen : {"lgbm", "fast"}, default="lgbm"
    new_cut_penalty : float, default=0.0
        Each round compares the best new tree with re-boosting a term already
        in the model, scoring ``gain / (1 + new_cut_penalty * new_cuts)``
        (re-using a term adds no cut). 0 = plain boosting; larger values spend
        the budget more slowly and boost the chosen structure longer.
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

    def __init__(self, *, max_splits=64, depth=2, learning_rate=0.1, lam=1.0, max_rounds=2000,
                 min_weight=20.0, max_bins=32, validation_fraction=0.15, patience=50,
                 refit=True, refit_lam=None, refit_sweeps=4, max_features_d2=32,
                 feature_screen="lgbm", new_cut_penalty=0.0, random_state=0):
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
        self.feature_screen = feature_screen
        self.new_cut_penalty = new_cut_penalty
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Boost, merge and refit (see the class docstring)."""
        if self.depth not in (1, 2, "auto"):
            raise ValueError("depth must be 1, 2 or 'auto'.")
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
        self._fixed_feats = None
        if self.depth != 1 and self.feature_screen == "lgbm":
            self._fixed_feats = teacher_top_features(X[tr], (yt > 0.5).astype(int),
                                                     self.max_features_d2, wt, self.random_state)
        base = base_margin(yt, wt)
        m_tr, m_val = np.full(len(yt), base), np.full(len(yv), base)
        budget = np.inf if self.max_splits is None else int(self.max_splits)
        terms, rounds, history = _TermSet(), [], []
        self._specs, self._slot = [], {}
        self._idmat = np.zeros((64, len(yt)), dtype=np.int8)
        best, best_round = np.inf, 0
        for _ in range(int(self.max_rounds)):
            g, h = grad_hess(yt, m_tr, wt)
            spec = self._next_term(terms, Xt, g, h, wt, nb)
            if spec is None:
                break
            if terms.cost(spec) > budget - terms.n_cuts:
                # budget nearly spent: a stump may still fit, otherwise the structure is
                # final and the joint refit below replaces further shrunken steps
                spec = None
                if budget - terms.n_cuts >= 1 and self.depth != 1:
                    depth, self.depth = self.depth, 1
                    stump = AdditiveTreeBooster._term(self, Xt, g, h, wt, nb)
                    self.depth = depth
                    if stump is not None and terms.cost(stump) <= budget - terms.n_cuts:
                        spec = stump
                if spec is None:
                    if not val.any():
                        best_round = len(rounds)
                    break
            spec = tuple(int(v) for v in spec)
            ids = depth2_leaf_ids(Xt, *spec)
            if spec not in self._slot:
                if len(self._specs) == len(self._idmat):  # grow the leaf-id matrix
                    self._idmat = np.concatenate([self._idmat, np.zeros_like(self._idmat)])
                self._slot[spec] = len(self._specs)
                self._idmat[len(self._specs)] = ids
                self._specs.append(spec)
            values = self.learning_rate * newton_leaf_values(ids, g, h, 4, self.lam)
            m_tr += values[ids]
            terms.add(spec, values)
            rounds.append((spec, values))
            if val.any():
                m_val += values[depth2_leaf_ids(Xv, *spec)]
                loss = _log_loss(yv, m_val, wv)
            else:
                loss = _log_loss(yt, m_tr, wt)
            history.append(loss)
            if loss < best - 1e-12:
                best, best_round = loss, len(rounds)
            elif len(rounds) - best_round >= self.patience:
                break
        del self._specs, self._slot, self._idmat
        kept = _TermSet()
        for spec, values in rounds[:best_round] if val.any() else rounds:
            kept.add(spec, values)
        self.n_rounds_ = best_round if val.any() else len(rounds)
        self.history_ = np.array(history)
        self.base_margin_ = base
        self.trees_ = [self._to_tree(s, v) for s, v in kept.terms.items()]
        if self.refit and self.trees_:
            self._refit_all(Xb, target, w)
        return self

    def _term_at(self, depth, Xb, g, h, w, nb):
        saved, self.depth = self.depth, depth
        try:
            return AdditiveTreeBooster._term(self, Xb, g, h, w, nb)
        finally:
            self.depth = saved

    def _gain(self, spec, Xb, g, h):
        ids = depth2_leaf_ids(Xb, *spec)
        G = np.bincount(ids, weights=g, minlength=4)
        H = np.bincount(ids, weights=h, minlength=4)
        return float(np.sum(G * G / (H + self.lam)) - G.sum() ** 2 / (H.sum() + self.lam))

    def _reuse(self, g, h):
        """(gain, spec) of the best term already in the model."""
        if not self._specs:
            return 0.0, None
        gains = reuse_gains(self._idmat, len(self._specs), g, h, float(self.lam))
        j = int(np.argmax(gains))
        return float(gains[j]), self._specs[j]

    def _next_term(self, terms, Xb, g, h, w, nb):
        """The best of: the optimal tree of ``depth`` (both depths with "auto")
        and, with ``new_cut_penalty`` > 0, re-boosting an existing term; scored
        by gain / (1 + penalty * new cuts), with penalty 1 for "auto" alone."""
        depths = (1, 2) if self.depth == "auto" else (self.depth,)
        cands = [s for s in (self._term_at(d, Xb, g, h, w, nb) for d in depths) if s is not None]
        pen = self.new_cut_penalty or (1.0 if self.depth == "auto" else 0.0)
        if pen <= 0:
            return cands[0] if cands else None
        scored = [(self._gain(s, Xb, g, h) / (1 + pen * terms.cost(tuple(int(v) for v in s))), s)
                  for s in cands]
        gain, spec = self._reuse(g, h)
        if spec is not None:
            scored.append((gain, spec))
        scored = [x for x in scored if x[0] > 0]
        return max(scored, key=lambda x: x[0])[1] if scored else None

    @staticmethod
    def _to_tree(spec, values):
        f1, t1, fl, tl, fr, tr = spec
        tree = SmallTree.from_nested((f1, t1, _side(fl, tl), _side(fr, tr)))
        for child, leaves in ((tree.left[0], (0, 1)), (tree.right[0], (2, 3))):
            if tree.left[child] == -1:
                tree.value[child] = values[leaves[0]]
            else:
                tree.value[tree.left[child]] = values[leaves[0]]
                tree.value[tree.right[child]] = values[leaves[1]]
        return tree

    def _refit_all(self, Xb, target, w):
        """Fully corrective refit of every leaf on all rows (warm start)."""
        lam, self.lam_ = self.lam_, float(self.lam if self.refit_lam is None else self.refit_lam)
        contribs = [t.value[t.leaf_ids(Xb)] for t in self.trees_]
        margin = self.base_margin_ + np.sum(contribs, axis=0)
        for _ in range(int(self.refit_sweeps)):
            for k, tree in enumerate(self.trees_):
                contribs[k], margin = self._newton_step(tree, Xb, target, margin, contribs[k], w)
            g, h = grad_hess(target, margin, w)  # intercept: one Newton step
            self.base_margin_ -= float(np.sum(g)) / max(float(np.sum(h)), 1e-12)
            margin = self.base_margin_ + np.sum(contribs, axis=0)
        self.lam_ = lam
