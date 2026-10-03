"""Logit sums of a few small trees: K optimal trees and FIGS.

Shared representation (``SmallTree``): per-node arrays (feature, threshold in
bins, left, right, value); a leaf has left = right = -1. ``x <= t`` (in bins)
goes left, NaN (bin 0) always goes left. The complexity cost is the number of
cuts (internal nodes), summed over trees.

- ``SumOfOptimalTrees``: K **optimal** Newton trees of depth d in {1, 2, 3}
  fitted in sequence on the residual, then ``backfit_sweeps`` passes of joint
  leaf re-estimation (each tree against the margin of the others).
- ``InterleavedTreeClassifier`` (the Interleaved Tree Model, ITM; ``FIGSClassifier`` is
  the same class): growth as in FIGS (Tan et al., 2022) with Newton/logit leaves and a
  full leaf re-fit after each cut (RGF, Johnson and Zhang, 2014): at each step, the
  best cut in any leaf of any tree, or the root of a new tree, against the margin of
  the other trees; stops at ``max_splits`` cuts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .._typing import ArrayLike, FloatArray, Seed, SelfT
from ._common import (
    base_margin,
    binned,
    fit_inputs,
    grad_hess,
    predict_input,
    rebin,
    sigmoid,
    teacher_top_features,
    top_features,
)
from ._kernels import (
    best_cut_1d,
    best_depth2,
    best_depth3,
    hist_1d,
    newton_leaf_values,
    newton_step_inplace,
    node_hist_margin,
    small_tree_leaf_ids,
)
from .edit import TreeEditMixin
from .explain import InterpretableSumMixin, leaf_rules


@dataclass
class SmallTree:
    """One tree of a sum, stored as parallel per-node lists (cuts in bins).

    ``feature[k]`` and ``threshold[k]`` (a bin index; rows with bin <= threshold go
    left) describe node ``k``; ``left``/``right`` are child indices, -1 at a leaf;
    ``value`` holds the logit contribution of each node (only leaves enter the sum).
    Fitted sums keep their trees in ``trees_``; ``get_trees()`` returns them with
    real thresholds.
    """

    feature: list = field(default_factory=lambda: [-1])
    threshold: list = field(default_factory=lambda: [-1])
    left: list = field(default_factory=lambda: [-1])
    right: list = field(default_factory=lambda: [-1])
    value: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def split(self, node, f, t):
        """Turn leaf ``node`` into a cut; the children inherit its value."""
        k = len(self.feature)
        self.feature += [-1, -1]
        self.threshold += [-1, -1]
        self.left += [-1, -1]
        self.right += [-1, -1]
        self.feature[node], self.threshold[node] = int(f), int(t)
        self.left[node], self.right[node] = k, k + 1
        self.value = np.append(self.value, [self.value[node], self.value[node]])
        return k, k + 1

    def arrays(self):
        return (np.array(self.feature, dtype=np.int64), np.array(self.threshold, dtype=np.int64),
                np.array(self.left, dtype=np.int64), np.array(self.right, dtype=np.int64))

    def leaf_ids(self, Xb):
        return small_tree_leaf_ids(Xb, *self.arrays())

    @property
    def n_splits(self):
        return sum(1 for v in self.left if v != -1)

    @property
    def leaves(self):
        return [k for k, v in enumerate(self.left) if v == -1]

    @classmethod
    def from_nested(cls, nested):
        """Build from nested (f, t, left, right) tuples; None = leaf."""
        tree = cls()

        def grow(node, spec):
            if spec is None:
                return
            left, right = tree.split(node, spec[0], spec[1])
            grow(left, spec[2])
            grow(right, spec[3])

        grow(0, nested)
        return tree

    def rules(self, edges, names=None, nan_features=None):
        """Leaves as (per-feature interval conditions, value); see ``explain``."""
        return leaf_rules(self, edges, names, nan_features)


def _side(f, t):
    return None if f < 0 else (int(f), int(t), None, None)


def _d2_nested(s):
    """Depth-2 side (f2, t2, fl, tl, fr, tr) -> nested tuple."""
    if s[0] < 0:
        return None
    return (int(s[0]), int(s[1]), _side(s[2], s[3]), _side(s[4], s[5]))


def optimal_tree(Xb, g, h, w, nb, depth, lam, min_weight, features=None):
    """Optimal Newton tree of depth 1-3 (without values); None if no cut gains.

    ``features`` restricts the search (needed in practice for depth 3 with large p).
    """
    feats = np.arange(Xb.shape[1]) if features is None else np.asarray(features, dtype=np.int64)
    Xs = np.ascontiguousarray(Xb[:, feats])
    nbs = np.ascontiguousarray(nb[feats])

    def remap(spec):
        if spec is None:
            return None
        return (int(feats[spec[0]]), spec[1], remap(spec[2]), remap(spec[3]))

    if depth == 1:
        gains, cuts = best_cut_1d(hist_1d(Xs, g, h, w, int(nbs.max())), nbs, lam, min_weight)
        f = int(np.argmax(gains))
        spec = (f, int(cuts[f]), None, None) if gains[f] > 0 else None
    elif depth == 2:
        gains, res = best_depth2(Xs, g, h, w, nbs, lam, min_weight)
        f = int(np.argmax(gains))
        spec = (None if gains[f] <= 0 else
                (f, int(res[f, 0]), _side(res[f, 1], res[f, 2]), _side(res[f, 3], res[f, 4])))
    elif depth == 3:
        gains, res = best_depth3(Xs, g, h, w, nbs, lam, min_weight)
        f = int(np.argmax(gains))
        spec = (None if gains[f] <= 0 else
                (f, int(res[f, 0]), _d2_nested(res[f, 1:7]), _d2_nested(res[f, 7:13])))
    else:
        raise ValueError("depth must be 1, 2 or 3.")
    return None if spec is None else SmallTree.from_nested(remap(spec))


def greedy_tree(Xb, g, h, w, nb, depth, lam, min_weight, features=None):
    """Greedy control for ``optimal_tree``: best single cut at the root, then in
    each child (same Newton gain, same bins); None if no cut gains."""
    feats = np.arange(Xb.shape[1]) if features is None else np.asarray(features, dtype=np.int64)
    Xs = np.ascontiguousarray(Xb[:, feats])
    nbs = np.ascontiguousarray(nb[feats])
    nmax = int(nbs.max())

    def grow(rows, d):
        if d == 0 or len(rows) == 0:
            return None
        Xr = np.ascontiguousarray(Xs[rows])
        gains, cuts = best_cut_1d(hist_1d(Xr, g[rows], h[rows], w[rows], nmax), nbs, lam,
                                  min_weight)
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            return None
        left = Xr[:, j] <= cuts[j]
        return (int(feats[j]), int(cuts[j]), grow(rows[left], d - 1), grow(rows[~left], d - 1))

    spec = grow(np.arange(len(Xs)), depth)
    return None if spec is None else SmallTree.from_nested(spec)


class _AdditiveTrees(TreeEditMixin, InterpretableSumMixin, ClassifierMixin, BaseEstimator):
    """Base class: binning, margin, backfitting and prediction for a sum of SmallTree."""

    def _prepare(self, X, y, sample_weight, y_soft):
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        Xb, edges, nb = binned(X, self.max_bins)
        self.classes_, self.bin_edges_ = classes, edges
        return X, Xb, nb, target, w

    def _newton_step(self, tree, Xb, target, margin, contrib, w, ids=None):
        """Incremental Newton step on the leaves of ``tree``, from the current values.

        ``margin`` includes the tree's current contribution; returns (new
        contribution, new margin). Starting from the current values (not from
        zero) keeps backfitting stable when trees are nearly collinear.

        With ``ids`` (the tree's cached leaf of each row) the step runs in one
        compiled pass and updates ``margin`` and ``contrib`` in place; the result is
        bit-for-bit the same as the NumPy path below.
        """
        cap = getattr(self, "max_delta_step", None)
        if ids is not None:
            tree.value = np.ascontiguousarray(tree.value, dtype=np.float64)
            newton_step_inplace(target, w, margin, contrib, ids, tree.value, self.lam_,
                                self.learning_rate_, 0.0 if cap is None else float(cap),
                                cap is not None)
            return contrib, margin
        g, h = grad_hess(target, margin, w)
        ids = tree.leaf_ids(Xb)
        step = newton_leaf_values(ids, g, h, len(tree.feature), self.lam_)
        cap = getattr(self, "max_delta_step", None)
        if cap is not None:
            # a single Newton step on a nearly pure leaf of rare-class data can be hundreds
            # of logits (tiny hessian); capping it, as XGBoost's max_delta_step, prevents the
            # divergence and the sign flips that backfitting then produces
            step = np.clip(step, -cap, cap)
        tree.value = tree.value + self.learning_rate_ * step
        new = tree.value[ids]
        return new, margin - contrib + new

    def _backfit(self, trees, contribs, Xb, target, w, sweeps, ids=None):
        margin = self.base_margin_ + np.sum(contribs, axis=0)
        for _ in range(sweeps):
            for k, tree in enumerate(trees):
                contribs[k], margin = self._newton_step(tree, Xb, target, margin, contribs[k], w,
                                                        None if ids is None else ids[k])
        return margin

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Logit of ``P(y = classes_[1])``: base plus the sum of the trees."""
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_,
                   dtype=getattr(self, "input_dtype_", np.float32))
        m = np.full(len(Xb), self.base_margin_)
        for tree in self.trees_:
            m += tree.value[tree.leaf_ids(Xb)]
        return m


class SumOfOptimalTrees(_AdditiveTrees):
    """Logit sum of a few optimal Newton trees: the fixed-budget model.

    ``logit(p) = base + Σ_k tree_k(x)``. Each tree of depth ``depth`` is the
    optimal one (exhaustive search over bins) on the residual of the previous
    trees; then ``backfit_sweeps`` passes re-estimate all leaves jointly.
    Budget: ``n_trees * (2**depth - 1) + extra_stumps`` cuts (e.g. 16 cuts =
    5 depth-2 trees + 1 stump).

    Parameters
    ----------
    n_trees : int, default=2
        Number of trees of depth ``depth``.
    depth : {1, 2, 3}, default=2
        Depth of the trees.
    learning_rate : float or "auto", default="auto"
        Shrinkage of the Newton steps on the leaves (1 = full step). "auto" =
        1 / (1 + 0.1 * n_trees), the median of the benchmark tuning (0.9 with
        one tree, 0.3 with 21).
    lam : float, default=2.0
        L2 regularization of the leaves (in the gain and the Newton step).
    min_weight : float, default=20.0
        Minimum hessian mass per leaf.
    max_bins : int, default=16
        Bins per feature.
    backfit_sweeps : int, default=2
        Passes of joint leaf re-estimation.
    max_features_d3, max_features_d2 : int, default=24, 128
        With more features, the search is restricted to the most important ones
        (optimal within that set).
    feature_screen : {"lgbm", "fast"}, default="lgbm"
        Screening: LightGBM gain importance (computed once; falls back to "fast"
        without LightGBM) or FAST scores recomputed every round.
    search : {"optimal", "greedy"}, default="optimal"
        ``"greedy"`` grows each tree by the best single cut (the control).
    extra_stumps : int, default=0
        Stumps (depth 1) added after the trees, to fill the budget.
    max_delta_step : float or None, default=None
        Cap on each leaf's Newton step, in logits (as XGBoost's ``max_delta_step``);
        prevents divergence on rare-class data. None = no cap.

    Attributes
    ----------
    trees_ : list of SmallTree
        The trees, with cuts in bins; ``get_trees()`` gives real thresholds.
    base_margin_ : float
        Initial constant logit.
    lam_, learning_rate_ : float
        Effective regularization and step.
    classes_, n_features_in_, feature_names_in_, bin_edges_, nan_features_
        scikit-learn and binning conventions.

    Interpretation methods: ``explain``, ``rules``, ``to_dict``, ``to_sql``,
    ``get_trees``, ``export_text``, ``predict_contributions``,
    ``plot_contributions``, ``plot_shapes``, ``to_shap_model``; and the editing
    methods of every sum.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bettertrees import SumOfOptimalTrees
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = SumOfOptimalTrees(n_trees=2, depth=2, feature_screen="fast").fit(X, y)
    >>> len(model.trees_), model.n_splits_
    (2, 6)
    """

    def __init__(self, *, n_trees: int = 2, depth: Literal[1, 2, 3] = 2,
                 learning_rate: float | Literal["auto"] = "auto", lam: float = 2.0,
                 min_weight: float = 20.0, max_bins: int = 16, backfit_sweeps: int = 2,
                 max_features_d3: int | None = 24, max_features_d2: int | None = 128,
                 feature_screen: Literal["lgbm", "fast"] = "lgbm",
                 search: Literal["optimal", "greedy"] = "optimal", extra_stumps: int = 0,
                 max_delta_step: float | None = None) -> None:
        self.n_trees = n_trees
        self.depth = depth
        self.learning_rate = learning_rate
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.backfit_sweeps = backfit_sweeps
        self.max_features_d3 = max_features_d3
        self.max_features_d2 = max_features_d2
        self.feature_screen = feature_screen
        self.search = search
        self.extra_stumps = extra_stumps
        self.max_delta_step = max_delta_step

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None,
            y_soft: ArrayLike | None = None) -> SelfT:
        """Fit the sum.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Numeric features; NaN is accepted (it always goes to the ``<=`` side).
            With a DataFrame, the column names become ``feature_names_in_``.
        y : array-like of shape (n_samples,)
            Labels of two classes.
        sample_weight : array-like of shape (n_samples,), optional
            Non-negative row weights.
        y_soft : array-like of shape (n_samples,), optional
            Soft target in [0, 1] (e.g. a teacher's probability) used instead
            of y; ``classes_`` still comes from y.

        Returns
        -------
        self
        """
        if self.search not in ("optimal", "greedy"):
            raise ValueError("search must be 'optimal' or 'greedy'.")
        grow = optimal_tree if self.search == "optimal" else greedy_tree
        self.lam_ = float(self.lam)
        self.learning_rate_ = (1.0 / (1.0 + 0.1 * self.n_trees) if self.learning_rate == "auto"
                               else float(self.learning_rate))
        X, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        margin = np.full(len(Xb), self.base_margin_)
        trees, contribs = [], []
        k = self.max_features_d3 if self.depth == 3 else self.max_features_d2
        fixed = (teacher_top_features(X, (target > 0.5).astype(int), k, w)
                 if self.feature_screen == "lgbm" and self.depth >= 2 else None)
        for depth in [self.depth] * self.n_trees + [1] * self.extra_stumps:
            g, h = grad_hess(target, margin, w)
            if depth == 1:  # stump: full search is cheap and optimal = greedy
                feats = None
            else:
                feats = fixed if self.feature_screen == "lgbm" else top_features(
                    Xb, g, h, w, nb, self.lam_, self.min_weight, k)
            tree = grow(Xb, g, h, w, nb, depth, self.lam_, self.min_weight, feats)
            if tree is None:
                break
            contrib, margin = self._newton_step(tree, Xb, target, margin, np.zeros(len(Xb)), w)
            trees.append(tree)
            contribs.append(contrib)
        if trees and self.backfit_sweeps:
            self._backfit(trees, contribs, Xb, target, w, self.backfit_sweeps)
        self.trees_ = trees
        return self


class InterleavedTreeClassifier(_AdditiveTrees):
    """Interleaved Tree Model (ITM): a logit sum of small trees grown together, binary
    classification.

    ``logit(p) = base + sum_k tree_k(x)``. The trees are small and share one budget of
    distinct cuts (``max_splits``). At each step the best cut by Newton gain
    ``G^2 / (H + lam)``, in any leaf of any tree or as the root of a new tree, against
    the margin of the other trees, enters the model, and then every leaf is
    re-estimated (one backfitting sweep by default). The number and shape of the trees
    come from the data.

    The growth rule comes from FIGS (Tan et al., 2022); re-fitting all the leaves after
    each cut comes from RGF (Johnson and Zhang, 2014). The budget is counted in
    distinct cuts.

    The constructor's defaults are a full Newton step (``learning_rate=1.0``) and
    ``lam="auto"`` (2 * max_splits), with no cap on the step. The rule the benchmark
    validated for each budget is in ``BudgetClassifier``: lam = 2b; up to 8 cuts a full
    step capped at 4 logits (``max_delta_step=4.0``); above 8 cuts ``learning_rate=0.3``
    (evaluated up to 64 cuts, applied unvalidated above). Use
    ``BudgetClassifier(b)`` to get that rule instead of the plain constructor.

    ``FIGSClassifier`` is an alias: the same class under its older name.

    Parameters
    ----------
    max_splits : int, default=16
        Total number of cuts (may stop earlier if no cut has a positive gain).
    max_trees : int or None, default=None
        Maximum number of trees.
    lam : float or "auto", default="auto"
        L2 regularization of the leaves (in the gain and the Newton step).
        "auto" = 2 * max_splits, the value of the benchmark rule: larger budgets
        need more regularization.
    min_weight : float, default=20.0
        Minimum hessian mass per leaf.
    max_bins : int, default=32
        Bins per feature.
    backfit_sweeps : int, default=1
        Passes of leaf re-estimation after each cut.
    learning_rate : float, default=1.0
        Shrinkage of every Newton step on the leaves (1 = the full Newton step).
        Below 1 it damps the first move of a new leaf; the later sweeps keep
        moving it toward its full value.
    max_delta_step : float or None, default=None
        Cap on each leaf's Newton step, in logits (as XGBoost's ``max_delta_step``).
        With a rare class, a full step on a nearly pure leaf can reach hundreds of
        logits and backfitting then flips signs; the cap prevents it (the benchmark
        rule uses 4.0 up to 8 cuts). None = no cap.

    Attributes
    ----------
    trees_ : list of SmallTree
        The trees, with cuts in bins; ``get_trees()`` gives real thresholds.
    base_margin_ : float
        Initial constant logit.
    lam_, learning_rate_ : float
        Effective regularization and step.
    classes_, n_features_in_, feature_names_in_, bin_edges_, nan_features_
        scikit-learn and binning conventions.

    Interpretation methods: ``explain``, ``rules``, ``to_dict``, ``to_sql``,
    ``get_trees``, ``export_text``, ``predict_contributions``,
    ``plot_contributions``, ``plot_shapes``, ``to_shap_model``; editing methods:
    ``prune``, ``set_cut``, ``split_leaf``, ``add_stump``, ``refit_leaves``, ...

    See Also
    --------
    BudgetClassifier : this model with the benchmark rule for each budget.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bettertrees import InterleavedTreeClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = InterleavedTreeClassifier(max_splits=6).fit(X, y)
    >>> model.n_splits_
    6
    >>> model.predict_proba(X[:2]).shape
    (2, 2)
    """

    def __init__(self, *, max_splits: int = 16, max_trees: int | None = None,
                 lam: float | Literal["auto"] = "auto", min_weight: float = 20.0,
                 max_bins: int = 32, backfit_sweeps: int = 1, learning_rate: float = 1.0,
                 max_delta_step: float | None = None) -> None:
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.backfit_sweeps = backfit_sweeps
        self.learning_rate = learning_rate
        self.max_delta_step = max_delta_step

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None,
            y_soft: ArrayLike | None = None) -> SelfT:
        """Fit the sum.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Numeric features; NaN is accepted (it always goes to the ``<=`` side).
            With a DataFrame, the column names become ``feature_names_in_``.
        y : array-like of shape (n_samples,)
            Labels of two classes.
        sample_weight : array-like of shape (n_samples,), optional
            Non-negative row weights.
        y_soft : array-like of shape (n_samples,), optional
            Soft target in [0, 1] (e.g. a teacher's probability) used instead
            of y; ``classes_`` still comes from y.

        Returns
        -------
        self
        """
        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        _, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        self.trees_ = grow_figs(self, Xb, nb, target, w, self.max_splits, self.max_trees)
        return self


# the name of the same model in the FIGS paper; an alias, not a subclass
FIGSClassifier = InterleavedTreeClassifier


def grow_figs(est, Xb, nb, target, w, max_splits, max_trees=None):
    """FIGS growth on binned data (shared by ``InterleavedTreeClassifier`` and its bagged
    and Rashomon variants). ``est`` provides ``base_margin_``, ``lam_``,
    ``learning_rate_``, ``min_weight`` and ``backfit_sweeps``; returns the trees."""
    n, B = len(Xb), int(nb.max())
    trees, contribs, ids_of = [], [], []  # ids_of[k]: leaf of each row in tree k (cached)
    margin = np.full(n, est.base_margin_)
    target = np.ascontiguousarray(target, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    root_ids, zero = np.zeros(n, dtype=np.int64), np.zeros(n)
    for _ in range(max_splits):
        best = (0.0, None, None, None, None)  # gain, tree, node, feature, threshold
        can_add = max_trees is None or len(trees) < max_trees
        candidates = list(range(len(trees))) + ([None] if can_add else [])
        for k in candidates:
            # g, h at the margin of the other trees, computed inside the histogram kernel
            # (bit-for-bit grad_hess + node_hist)
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
        ids_of[k] = trees[k].leaf_ids(Xb)  # only the tree that received the cut changes
        contribs[k], margin = est._newton_step(trees[k], Xb, target, margin, contribs[k], w,
                                               ids_of[k])
        if est.backfit_sweeps:
            margin = est._backfit(trees, contribs, Xb, target, w, est.backfit_sweeps, ids_of)
    return trees


class BoostedOptimalTrees(_AdditiveTrees):
    """Boosting of optimal Newton trees of depth 1-3, with early stopping.

    Internal: a benchmark arm, not part of the public API (use
    ``AdditiveTreeBooster`` for depth 1-2).

    Extends ``AdditiveTreeBooster`` (depth 1/2) to depth 3: every round fits the
    optimal tree of depth ``depth`` on the residual (at depth 3, restricted to
    the ``max_features_d3`` most important features), with leaves set by a
    Newton step times ``learning_rate``. Stops when the internal validation
    log-loss (``validation_fraction``) has not improved for ``patience`` rounds
    and keeps the trees up to the best round. Parameters as in
    ``AdditiveTreeBooster``.

    Attributes
    ----------
    trees_ : list of SmallTree
        The trees, with cuts in bins; ``get_trees()`` gives real thresholds.
    base_margin_ : float
        Initial constant logit.
    classes_, n_features_in_, feature_names_in_, bin_edges_, nan_features_
        scikit-learn and binning conventions.

    Interpretation methods: ``explain``, ``rules``, ``to_dict``,
    ``get_trees``, ``export_text``, ``predict_contributions``,
    ``plot_contributions``, ``plot_shapes``, ``to_shap_model``.
    """

    def __init__(self, *, depth: Literal[1, 2, 3] = 3, learning_rate: float = 0.1,
                 max_rounds: int = 300, patience: int = 30, lam: float = 1.0,
                 min_weight: float = 20.0, max_bins: int = 32, validation_fraction: float = 0.15,
                 max_features_d3: int | None = 32, max_features_d2: int | None = 128,
                 feature_screen: Literal["lgbm", "fast"] = "lgbm", random_state: Seed = 0) -> None:
        self.depth = depth
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.patience = patience
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.validation_fraction = validation_fraction
        self.max_features_d3 = max_features_d3
        self.max_features_d2 = max_features_d2
        self.feature_screen = feature_screen
        self.random_state = random_state

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None,
            y_soft: ArrayLike | None = None) -> SelfT:
        """Fit the sum.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Numeric features; NaN is accepted (it always goes to the ``<=`` side).
            With a DataFrame, the column names become ``feature_names_in_``.
        y : array-like of shape (n_samples,)
            Labels of two classes.
        sample_weight : array-like of shape (n_samples,), optional
            Non-negative row weights.
        y_soft : array-like of shape (n_samples,), optional
            Soft target in [0, 1] (e.g. a teacher's probability) used instead
            of y; ``classes_`` still comes from y.

        Returns
        -------
        self
        """
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        self.lam_, self.learning_rate_ = float(self.lam), float(self.learning_rate)
        rng = np.random.default_rng(self.random_state)
        val = np.zeros(len(X), dtype=bool)
        val[rng.permutation(len(X))[:int(round(self.validation_fraction * len(X)))]] = True
        Xb, edges, nb = binned(X[~val], self.max_bins)
        Xv = rebin(X[val], edges)
        yt, wt, yv, wv = target[~val], w[~val], target[val], w[val]
        self.classes_, self.bin_edges_ = classes, edges
        self.base_margin_ = base_margin(yt, wt)
        m_tr = np.full(len(yt), self.base_margin_)
        m_val = np.full(len(yv), self.base_margin_)
        trees, history, best, best_round = [], [], np.inf, 0
        k = self.max_features_d3 if self.depth == 3 else self.max_features_d2
        fixed = (teacher_top_features(X[~val], (yt > 0.5).astype(int), k, wt, self.random_state)
                 if self.feature_screen == "lgbm" and self.depth >= 2 else None)
        for _ in range(self.max_rounds):
            g, h = grad_hess(yt, m_tr, wt)
            feats = fixed if self.feature_screen == "lgbm" else top_features(
                Xb, g, h, wt, nb, self.lam_, self.min_weight, k)
            tree = optimal_tree(Xb, g, h, wt, nb, self.depth, self.lam_, self.min_weight, feats)
            if tree is None:
                break
            ids = tree.leaf_ids(Xb)
            tree.value = self.learning_rate * newton_leaf_values(ids, g, h, len(tree.feature),
                                                                 self.lam_)
            m_tr += tree.value[ids]
            trees.append(tree)
            if len(yv):
                m_val += tree.value[tree.leaf_ids(Xv)]
                p = np.clip(sigmoid(m_val), 1e-15, 1 - 1e-15)
                loss = float(-np.sum(wv * (yv * np.log(p) + (1 - yv) * np.log(1 - p))) / wv.sum())
            else:
                loss = -len(trees)
            history.append(loss)
            if loss < best - 1e-12:
                best, best_round = loss, len(trees)
            elif len(trees) - best_round >= self.patience:
                break
        self.trees_ = trees[:best_round] if len(yv) else trees
        self.history_ = np.array(history)
        return self
