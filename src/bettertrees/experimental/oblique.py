"""Experimental oblique-split FIGS (in the spirit of RO-FIGS, Matjasec et al., 2025).

The production FIGS path is untouched. With ``oblique=False`` the growth below
runs the production kernels in the production order and reproduces
``FIGSClassifier`` bit for bit.

Representation. Each tree is a ``SmallTree`` over an *augmented* binned matrix
``[Xb | Zb]``: columns ``0..p-1`` are the usual feature bins; column ``p + j``
holds the outcome of oblique split ``j`` (bin 1 when ``w_j . x~ <= t_j``, bin 2
otherwise), so an oblique node has ``feature = p + j`` and ``threshold = 1``.
Leaf ids, Newton steps and backfitting are then exactly the production ones.

Oblique candidates (deviations from RO-FIGS in brackets):

- ``x~`` = raw features standardized with the training (sample-weighted) mean
  and standard deviation; NaN becomes 0 after standardization, i.e. **mean
  imputation** inside oblique conditions. Axis cuts keep the package rule
  (NaN goes left). [RO-FIGS min-max scales and does not discuss NaN.]
- Supports ``S``: subsets of ``max_oblique_features`` features. ``"top"``
  (default): the first ``n_subsets`` subsets, in colex order, of the features
  ranked by their best axis gain in that leaf (with |S| = 2 and 6 subsets: the
  6 pairs of the top 4). ``"random"``: ``n_subsets`` uniform random subsets,
  redrawn for every leaf evaluation. [RO-FIGS draws one random subset of
  ``beam_size`` features per iteration, repeated up to r = 5 times.]
- Direction: the h-weighted ridge regression of the Newton working response
  ``-g/h`` on ``x~_S`` over the leaf's rows (with intercept, ridge
  ``oblique_ridge``, default lambda), normalized to unit L2 norm. [RO-FIGS
  learns ``w`` by gradient descent on a sigmoid-relaxed weighted-variance
  impurity with an L1/2 penalty (SPYCT); we use a closed-form Newton-consistent
  direction and no sparsity penalty.]
- Threshold: the leaf's projections are cut at their (unweighted) quantiles
  (``oblique_bins``, default ``max_bins``) and every cut is scored with the
  same Newton gain as an axis cut, ``G_L^2/(H_L+lam) + G_R^2/(H_R+lam) -
  G^2/(H+lam)`` at the margin of the other trees, with the same mass
  constraint (both children >= ``min_weight``). [RO-FIGS optimizes the
  threshold jointly with ``w``.]
- Axis and oblique candidates compete by gain in every leaf (ties go to the
  axis cut). [RO-FIGS only uses oblique splits.] An oblique split consumes
  ``oblique_cost`` units of ``max_splits`` and competes with ``gain /
  oblique_cost``.
"""

from itertools import combinations
from math import comb
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..sums._common import (
    base_margin,
    bin_threshold,
    grad_hess,
    predict_input,
    rebin,
)
from ..sums._kernels import best_cut_1d, node_hist_margin, small_tree_leaf_ids
from ..sums.smalltrees import FIGSClassifier, SmallTree
from ._oblique_kernels import projection_hist

STRATEGIES = ("top", "random")
LEFT_BIN, RIGHT_BIN = 1, 2  # bins of an oblique column (0 would be NaN; never used)


def project(columns, weights):
    """``sum_j weights[j] * columns[j]``, accumulated in a fixed order.

    The same elementwise arithmetic is used during the search (on a leaf's rows),
    when the split is materialized (all training rows) and at prediction, so a
    row's projection is bit-for-bit the same in the three places.
    """
    z = np.zeros(len(columns[0]))
    for c, v in zip(columns, weights):
        z = z + v * c
    return z


def quantile_edges(z, n_bins):
    """Candidate thresholds of a projection: unique (unweighted) quantiles below the max.

    ``z <= edge`` goes left. Same rule as ``combination_scores``: edge s is
    ``sort(z)[(s * n) // n_bins]``, s = 1..n_bins-1; with ``n_bins >= n`` this is
    every distinct value except the largest.
    """
    zs = np.sort(z)
    n = len(zs)
    edges = np.unique(zs[(np.arange(1, n_bins) * n) // n_bins])
    return edges[edges < zs[-1]]


def scan_projection(z, g, h, w, edges, lam, min_weight):
    """Best Newton cut ``z <= edges[t]`` (gain, t); (0.0, -1) when none gains."""
    ne = len(edges)
    if ne == 0:
        return 0.0, -1
    hist = projection_hist(z, g, h, w, edges)  # z <= edges[t]  <=>  bin <= t
    G, H, W = hist[:, 0], hist[:, 1], hist[:, 2]
    GL, HL, WL = np.cumsum(G)[:-1], np.cumsum(H)[:-1], np.cumsum(W)[:-1]
    Gt, Ht, Wt = G.sum(), H.sum(), W.sum()
    gain = GL ** 2 / (HL + lam) + (Gt - GL) ** 2 / (Ht - HL + lam) - Gt ** 2 / (Ht + lam)
    small = (min_weight > WL) | (min_weight > Wt - WL)
    gain[small] = -np.inf
    t = int(np.argmax(gain))
    if not gain[t] > 0:
        return 0.0, -1
    return float(gain[t]), t


class ObliqueFIGSClassifier(ClassifierMixin, BaseEstimator):
    """FIGS (Newton/logit leaves, backfitting) whose cuts may be oblique.

    Every step scores, in every leaf of every tree and at the root of a new tree,
    the best axis cut and the best oblique cut ``[w . x~ > t]`` (``w`` supported
    on at most ``max_oblique_features`` features), with the same Newton gain at
    the margin of the other trees; the best one is applied, its tree takes a
    Newton step and all trees are backfitted, as in ``FIGSClassifier``. See the
    module docstring for how ``x~``, the subsets, ``w`` and ``t`` are chosen.

    Parameters
    ----------
    max_splits, max_trees, lam, min_weight, max_bins, backfit_sweeps,
    learning_rate, max_delta_step
        As in ``FIGSClassifier`` (``lam="auto"`` = 2 * max_splits).
    oblique : bool, default=True
        False = axis cuts only: ``FIGSClassifier`` bit for bit.
    max_oblique_features : int, default=2
        |S|: number of features in an oblique condition (>= 2).
    n_subsets : int, default=6
        Candidate subsets per leaf.
    subset_strategy : {"top", "random"}, default="top"
        ``"top"``: subsets of the features with the best axis gain in the leaf
        (deterministic); ``"random"``: uniform random subsets (RO-FIGS-like).
    oblique_bins : int or None, default=None
        Quantile thresholds per leaf projection; None = ``max_bins``.
    oblique_ridge : float or "auto", default="auto"
        Ridge of the direction regression; "auto" = ``lam``.
    oblique_cost : int, default=1
        Units of ``max_splits`` consumed by one oblique split; it competes with
        ``gain / oblique_cost``.
    random_state : int, default=0
        Seed of the ``"random"`` strategy.

    Attributes
    ----------
    trees_ : list of SmallTree
        Trees over the augmented bins (feature ``p + j`` = oblique split j).
    oblique_splits_ : list of dict
        Per oblique split: ``features``, ``weights`` (standardized units, unit
        norm), ``threshold`` (standardized projection), ``coef`` and
        ``raw_threshold`` (the same condition in raw units:
        ``coef . x > raw_threshold``, NaN entering at ``center_``), ``gain``.
    center_, scale_ : ndarray of shape (n_features,)
        Standardization of the oblique conditions.
    split_history_ : list of dict
        Every accepted split in order (kind, tree, leaf, gain, ...).
    base_margin_, classes_, n_features_in_, bin_edges_, nan_features_
        As in ``FIGSClassifier``.

    Accounting: ``n_splits_`` = ``n_axis_splits_`` + ``n_oblique_splits_``;
    ``n_params_`` = leaves + nonzero oblique weights (the convention of the
    LightGBM linear-leaf diagnostic: leaf values plus nonzero coefficients,
    cut thresholds not counted); ``n_params_with_thresholds_`` adds one per
    split; ``n_distinct_splits_``; ``conditions_read(X)`` and
    ``variables_read(X)``.
    """

    def __init__(self, *, max_splits=16, max_trees=None, lam="auto", min_weight=20.0,
                 max_bins=32, backfit_sweeps=1, learning_rate=1.0, max_delta_step=None,
                 oblique=True, max_oblique_features=2, n_subsets=6, subset_strategy="top",
                 oblique_bins=None, oblique_ridge="auto", oblique_cost=1, random_state=0):
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.backfit_sweeps = backfit_sweeps
        self.learning_rate = learning_rate
        self.max_delta_step = max_delta_step
        self.oblique = oblique
        self.max_oblique_features = max_oblique_features
        self.n_subsets = n_subsets
        self.subset_strategy = subset_strategy
        self.oblique_bins = oblique_bins
        self.oblique_ridge = oblique_ridge
        self.oblique_cost = oblique_cost
        self.random_state = random_state

    # the production pieces, used verbatim
    _prepare = FIGSClassifier._prepare
    _newton_step = FIGSClassifier._newton_step
    _backfit = FIGSClassifier._backfit

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.allow_nan = True
        return tags

    # ------------------------------------------------------------ parameters

    def _validate(self):
        def integer(name, low, none_ok=False):
            v = getattr(self, name)
            if none_ok and v is None:
                return
            if isinstance(v, bool) or not isinstance(v, Integral) or v < low:
                raise ValueError(f"{name} must be an integer >= {low}"
                                 + (" or None." if none_ok else "."))

        integer("max_splits", 0)
        integer("backfit_sweeps", 0)
        integer("max_trees", 0, none_ok=True)
        integer("max_oblique_features", 2)
        integer("n_subsets", 1)
        integer("oblique_cost", 1)
        integer("oblique_bins", 2, none_ok=True)
        if self.subset_strategy not in STRATEGIES:
            raise ValueError(f"subset_strategy must be one of {STRATEGIES}.")
        if not isinstance(self.learning_rate, Real) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if self.max_delta_step is not None and not (
                np.isfinite(self.max_delta_step) and self.max_delta_step > 0):
            raise ValueError("max_delta_step must be None or finite and positive.")
        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        if not np.isfinite(self.lam_) or self.lam_ < 0:
            raise ValueError("lam must be 'auto' or finite and non-negative.")
        self.ridge_ = self.lam_ if self.oblique_ridge == "auto" else float(self.oblique_ridge)
        if not np.isfinite(self.ridge_) or self.ridge_ < 0:
            raise ValueError("oblique_ridge must be 'auto' or finite and non-negative.")
        self.learning_rate_ = float(self.learning_rate)

    # ------------------------------------------------------------ fit

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Fit the sum (arguments as in ``FIGSClassifier.fit``)."""
        self._validate()
        X, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        self._fit_standardization(X, w)
        self.oblique_splits_, self.split_history_ = [], []
        self.trees_ = self._grow(X, Xb, nb, target, w)
        return self

    def _fit_standardization(self, X, w):
        ok = ~np.isnan(X)
        mass = (w[:, None] * ok).sum(axis=0)
        Xz = np.where(ok, X, 0.0)
        safe = np.where(mass > 0, mass, 1.0)
        center = np.where(mass > 0, w @ Xz / safe, 0.0)
        var = np.where(mass > 0, (w @ (np.where(ok, X - center, 0.0) ** 2)) / safe, 0.0)
        scale = np.sqrt(var)
        self.center_ = center
        self.scale_ = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)

    def _standardize(self, X, features=None):
        """``(x - center) / scale`` with NaN -> 0 (mean imputation), float64."""
        f = slice(None) if features is None else np.asarray(features, dtype=np.int64)
        Xs = (X[:, f] - self.center_[f]) / self.scale_[f]
        Xs[np.isnan(Xs)] = 0.0
        return Xs

    def _subsets(self, axis_gains, s, rng):
        p = len(axis_gains)
        if self.subset_strategy == "top":
            order = np.argsort(-axis_gains, kind="stable")
            pool = s
            while pool < p and comb(pool, s) < self.n_subsets:
                pool += 1
            combos = sorted(combinations(range(pool), s), key=lambda c: (c[::-1]))
            return [tuple(sorted(int(order[i]) for i in c)) for c in combos[: self.n_subsets]]
        out = []
        for _ in range(self.n_subsets):
            S = tuple(sorted(int(v) for v in rng.choice(p, s, replace=False)))
            if S not in out:
                out.append(S)
        return out

    def _best_oblique(self, Xs, rows, g, h, w, axis_gains, rng, return_all=False):
        """Best oblique cut of the leaf made of ``rows`` (g, h: all rows).

        Returns a dict (gain, features, weights, threshold) or None; with
        ``return_all`` also the list of every scored candidate.
        """
        p = Xs.shape[1]
        s = min(self.max_oblique_features, p)
        gr, hr, wr = g[rows], h[rows], w[rows]
        found = []
        if s < 2 or wr.sum() < 2 * self.min_weight:
            return (None, found) if return_all else None
        subsets = self._subsets(axis_gains, s, rng)
        pool = sorted({f for S in subsets for f in S})
        where = {f: i for i, f in enumerate(pool)}
        Xp = Xs[np.ix_(rows, pool)]
        Ht = hr.sum()
        Xc = Xp - (hr @ Xp) / Ht
        A = Xc.T @ (Xc * hr[:, None])
        b = -(Xc.T @ gr)
        n_bins = self.max_bins if self.oblique_bins is None else self.oblique_bins
        best = None
        for S in subsets:
            idx = [where[f] for f in S]
            try:
                v = np.linalg.solve(A[np.ix_(idx, idx)] + self.ridge_ * np.eye(len(idx)), b[idx])
            except np.linalg.LinAlgError:
                continue
            norm = np.sqrt(v @ v)
            if not np.isfinite(norm) or norm <= 1e-12:
                continue
            v = v / norm
            keep = np.abs(v) > 1e-12
            if keep.sum() < 2:  # degenerate: an axis cut, already scored
                continue
            feats = tuple(f for f, k in zip(S, keep) if k)
            v = v[keep]
            z = project([Xp[:, where[f]] for f in feats], v)
            edges = quantile_edges(z, n_bins)
            gain, t = scan_projection(z, gr, hr, wr, edges, self.lam_, self.min_weight)
            if t < 0:
                continue
            cand = dict(gain=gain, features=feats, weights=v, threshold=float(edges[t]))
            found.append(cand)
            if best is None or gain > best["gain"]:
                best = cand
        return (best, found) if return_all else best

    def _grow(self, X, Xb, nb, target, w):
        n, p = Xb.shape
        B = int(nb.max())
        trees, contribs, ids_of = [], [], []
        margin = np.full(n, self.base_margin_)
        target = np.ascontiguousarray(target, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        root_ids, zero = np.zeros(n, dtype=np.int64), np.zeros(n)
        Xaug = Xb
        Xs = self._standardize(X) if self.oblique and p >= 2 else None
        rng = np.random.default_rng(self.random_state)
        cost, spent = int(self.oblique_cost), 0
        while spent < self.max_splits:
            # score, tree, node, feature, threshold, oblique spec
            best = (0.0, None, None, None, None, None)
            can_add = self.max_trees is None or len(trees) < self.max_trees
            candidates = list(range(len(trees))) + ([None] if can_add else [])
            use_oblique = Xs is not None and self.max_splits - spent >= cost
            for k in candidates:
                if k is None:
                    ids, contrib = root_ids, zero
                    leaves, n_nodes = [0], 1
                else:
                    ids, contrib = ids_of[k], contribs[k]
                    leaves, n_nodes = trees[k].leaves, len(trees[k].feature)
                hist = node_hist_margin(Xb, target, w, margin, contrib, ids, n_nodes, B)
                if use_oblique:
                    g, h = grad_hess(target, margin - contrib, w)
                    order = np.argsort(ids, kind="stable")
                    start = np.concatenate([[0], np.cumsum(np.bincount(ids, minlength=n_nodes))])
                for leaf in leaves:
                    gains, cuts = best_cut_1d(hist[leaf], nb, self.lam_, self.min_weight)
                    f = int(np.argmax(gains))
                    if gains[f] > best[0]:
                        best = (float(gains[f]), k, leaf, f, int(cuts[f]), None)
                    if use_oblique:
                        rows = order[start[leaf]: start[leaf + 1]]
                        spec = self._best_oblique(Xs, rows, g, h, w, gains, rng)
                        if spec is not None and spec["gain"] / cost > best[0]:  # ties: axis
                            best = (spec["gain"] / cost, k, leaf, -1, -1, spec)
            score, k, leaf, f, t, spec = best
            if k is None and leaf is None:
                break
            if k is None:
                trees.append(SmallTree())
                contribs.append(np.zeros(n))
                ids_of.append(None)
                k = len(trees) - 1
            if spec is None:
                trees[k].split(leaf, f, t)
                spent += 1
                self.split_history_.append(dict(kind="axis", tree=k, leaf=leaf, feature=f,
                                                threshold=t, gain=score))
            else:
                j = len(self.oblique_splits_)
                z = project([Xs[:, f] for f in spec["features"]], spec["weights"])
                col = np.where(z <= spec["threshold"], LEFT_BIN, RIGHT_BIN).astype(np.uint8)
                Xaug = np.ascontiguousarray(np.column_stack([Xaug, col]))
                self.oblique_splits_.append(self._raw_units(spec))
                trees[k].split(leaf, p + j, LEFT_BIN)
                spent += cost
                self.split_history_.append(dict(kind="oblique", tree=k, leaf=leaf, index=j,
                                                features=spec["features"], gain=spec["gain"]))
            ids_of[k] = trees[k].leaf_ids(Xaug)  # only the tree that received the cut changes
            contribs[k], margin = self._newton_step(trees[k], Xaug, target, margin, contribs[k], w,
                                                    ids_of[k])
            if self.backfit_sweeps:
                margin = self._backfit(trees, contribs, Xaug, target, w, self.backfit_sweeps,
                                       ids_of)
        return trees

    def _raw_units(self, spec):
        f = np.asarray(spec["features"], dtype=np.int64)
        v = np.asarray(spec["weights"], dtype=np.float64)
        coef = v / self.scale_[f]
        return dict(features=tuple(int(x) for x in f), weights=v, threshold=spec["threshold"],
                    coef=coef, raw_threshold=float(spec["threshold"] + coef @ self.center_[f]),
                    gain=float(spec["gain"]))

    # ------------------------------------------------------------ prediction

    def _augmented(self, X):
        """[Xb | Zb] of validated raw X."""
        Xb = rebin(X, self.bin_edges_)
        if not self.oblique_splits_:
            return Xb
        cols = [Xb]
        for s in self.oblique_splits_:
            Xs = self._standardize(X, s["features"])
            z = project([Xs[:, i] for i in range(Xs.shape[1])], s["weights"])
            side = np.where(z <= s["threshold"], LEFT_BIN, RIGHT_BIN).astype(np.uint8)
            cols.append(side[:, None])
        return np.ascontiguousarray(np.hstack(cols))

    def _leaf_ids(self, Xaug):
        return [small_tree_leaf_ids(Xaug, *t.arrays()) for t in self.trees_]

    def decision_function(self, X):
        """Logit of ``P(y = classes_[1])``: base plus the sum of the trees."""
        Xaug = self._augmented(predict_input(self, X, "trees_"))
        m = np.full(len(Xaug), self.base_margin_)
        for tree in self.trees_:
            m += tree.value[tree.leaf_ids(Xaug)]
        return m

    def predict_proba(self, X):
        p = 0.5 * (1.0 + np.tanh(0.5 * self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        margin = self.decision_function(X)  # checks the fit before reading classes_
        return self.classes_[(margin > 0).astype(int)]

    def predict_contributions(self, X):
        """(n_samples, n_trees) logit contributions; base + row sum = ``decision_function``."""
        Xaug = self._augmented(predict_input(self, X, "trees_"))
        return np.column_stack([t.value[i] for t, i in zip(self.trees_, self._leaf_ids(Xaug))]
                               ) if self.trees_ else np.zeros((len(Xaug), 0))

    # ------------------------------------------------------------ accounting

    def _internal(self):
        p = self.n_features_in_
        for tree in self.trees_:
            for f, t, lft in zip(tree.feature, tree.threshold, tree.left):
                if lft != -1:
                    yield int(f), int(t), f >= p

    @property
    def n_splits_(self):
        """Total number of cuts (internal nodes), axis and oblique."""
        check_is_fitted(self, "trees_")
        return sum(t.n_splits for t in self.trees_)

    @property
    def n_axis_splits_(self):
        return sum(1 for *_, obl in self._internal() if not obl)

    @property
    def n_oblique_splits_(self):
        return sum(1 for *_, obl in self._internal() if obl)

    @property
    def n_leaves_(self):
        check_is_fitted(self, "trees_")
        return sum(len(t.leaves) for t in self.trees_)

    @property
    def n_params_(self):
        """Leaf values + nonzero oblique weights (cut thresholds not counted).

        Mirrors the LightGBM linear-leaf count (leaves + nonzero coefficients).
        Degrees of freedom: an axis cut has 1 (its threshold, not counted); an
        oblique cut on |S| features has |S| (unit-norm direction: |S| - 1, plus the
        offset), and is charged |S|. Relative to an axis cut an oblique cut is thus
        charged |S| where its true extra is |S| - 1: the convention is
        conservative against oblique cuts by one per oblique cut.
        """
        p = self.n_features_in_
        nnz = sum(int(np.count_nonzero(self.oblique_splits_[f - p]["weights"]))
                  for f, _, obl in self._internal() if obl)
        return self.n_leaves_ + nnz

    @property
    def n_params_with_thresholds_(self):
        """``n_params_`` plus one threshold per cut (axis and oblique)."""
        return self.n_params_ + self.n_splits_

    @property
    def n_distinct_splits_(self):
        """Distinct conditions: (feature, bin threshold) for axis cuts, one per oblique split."""
        return len({(f, t) for f, t, _ in self._internal()})

    def _visits(self, X):
        """{condition key: bool[n] rows that read it}, condition key -> variables read."""
        Xaug = self._augmented(predict_input(self, X, "trees_"))
        p, n = self.n_features_in_, len(Xaug)
        visits, variables = {}, {}
        for tree in self.trees_:
            at = {0: np.ones(n, dtype=bool)}
            for k, (f, t, lft, rgt) in enumerate(zip(tree.feature, tree.threshold, tree.left,
                                                      tree.right)):
                if lft == -1:
                    continue
                here = at[k]
                key = (int(f), int(t))
                visits[key] = visits.get(key, np.zeros(n, dtype=bool)) | here
                variables[key] = (self.oblique_splits_[f - p]["features"] if f >= p else (int(f),))
                go_left = Xaug[:, f] <= t
                at[lft], at[rgt] = here & go_left, here & ~go_left
        return visits, variables, n

    def conditions_read(self, X):
        """Mean number of distinct conditions evaluated per row along all trees.

        A condition is an axis cut (feature, threshold) or an oblique split; one
        shared by two trees counts once; an oblique condition counts 1.
        """
        visits, _, n = self._visits(X)
        return float(sum(v.sum() for v in visits.values()) / max(n, 1))

    def variables_read(self, X, distinct=False):
        """Mean number of raw variables read per row.

        Each distinct condition read by the row adds the number of variables it
        reads (1 for an axis cut, |S| for an oblique one); with ``distinct=True``
        a variable read by several conditions counts once.
        """
        visits, variables, n = self._visits(X)
        if not distinct:
            return float(sum(v.sum() * len(variables[k]) for k, v in visits.items()) / max(n, 1))
        per_var = {}
        for key, v in visits.items():
            for f in variables[key]:
                per_var[f] = per_var.get(f, np.zeros(n, dtype=bool)) | v
        return float(sum(v.sum() for v in per_var.values()) / max(n, 1))

    # ------------------------------------------------------------ text

    def _names(self, feature_names):
        if feature_names is not None:
            return list(feature_names)
        names = getattr(self, "feature_names_in_", None)
        return None if names is None else list(names)

    def _condition(self, f, t, left, names, precision):
        name = (lambda j: str(names[j])) if names else (lambda j: f"x{j}")
        if f >= self.n_features_in_:
            s = self.oblique_splits_[f - self.n_features_in_]
            lhs = " ".join(f"{'+' if c >= 0 else '-'} {abs(c):.{precision}g}*{name(j)}"
                           for c, j in zip(s["coef"], s["features"]))
            lhs = lhs[2:] if lhs.startswith("+ ") else "-" + lhs[2:]
            return f"{lhs} {'<=' if left else '>'} {s['raw_threshold']:.{precision}g}"
        thr = bin_threshold(self.bin_edges_[f], t)
        if np.isneginf(thr):
            return f"{name(f)} is missing" if left else f"{name(f)} is present"
        return f"{name(f)} <= {thr:.{precision}g}" if left else f"{name(f)} > {thr:.{precision}g}"

    def export_text(self, feature_names=None, precision=4):
        """The trees as text; oblique conditions in raw units (NaN enters at the mean)."""
        check_is_fitted(self, "trees_")
        names = self._names(feature_names)
        lines = [f"base (logit): {float(self.base_margin_):+.{precision}f}"]
        if self.oblique_splits_:
            lines.append("(oblique conditions: a missing value enters at its training mean)")
        for k, tree in enumerate(self.trees_, 1):
            lines.append(f"tree {k}")

            def walk(node, depth, tree=tree):
                pad = "|   " * depth
                if tree.left[node] == -1:
                    lines.append(f"{pad}|--- value: {tree.value[node]:+.{precision}f}")
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                lines.append(f"{pad}|--- {self._condition(f, t, True, names, precision)}")
                walk(tree.left[node], depth + 1)
                lines.append(f"{pad}|--- {self._condition(f, t, False, names, precision)}")
                walk(tree.right[node], depth + 1)

            walk(0, 0)
        return "\n".join(lines)

    def rules(self, feature_names=None, precision=6):
        """[(tree, [condition strings from the root], logit value)] for every leaf."""
        check_is_fitted(self, "trees_")
        names, out = self._names(feature_names), []
        for k, tree in enumerate(self.trees_):
            def walk(node, path, tree=tree, k=k):
                if tree.left[node] == -1:
                    out.append((k, list(path), float(tree.value[node])))
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                walk(tree.left[node], [*path, self._condition(f, t, True, names, precision)])
                walk(tree.right[node], [*path, self._condition(f, t, False, names, precision)])

            walk(0, [])
        return out
