"""Minimal Python API of the classification tree (scikit-learn compatible).

The exact engine follows CART (Breiman et al., 1984); the histogram engine
follows LightGBM (Ke et al., 2017) and scikit-learn's HistGradientBoosting;
leaf probabilities can use hierarchical shrinkage (Agarwal et al., ICML 2022).
"""

from __future__ import annotations

from numbers import Integral, Real
from time import perf_counter
from typing import Literal

import numpy as np
from numba import config as numba_config
from numba import get_num_threads, set_num_threads
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from ._data import NodeArrays, prepare_training_data, validate_X
from ._typing import ArrayLike, FeatureNames, FloatArray, IndexArray, LabelArray, Seed, SelfT
from .bins import fit_bin_edges, transform_bins_row_major
from .builder import grow_tree_exact, grow_tree_hist
from .kernels import apply_nodes
from .postprocess import (
    finite_leaf_regions,
    hierarchical_shrinkage_probabilities,
    predict_proba_nodes,
    project_monotonic_leaf_probabilities,
    prune_tree_cost_complexity,
)
from .splitters import resolve_splitter_spec


class FastDecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """Decision tree classifier (binary or multiclass) grown on weighted Gini.

    Parameters
    ----------
    splitter : {'hist', 'exact'}, default='hist'
        Split engine; training is never delegated to scikit-learn.
    max_feature_repeats : int >= 1 or None, default=None
        Maximum number of times one feature may appear on each root-to-leaf
        path. ``None`` grows without this limit; the counter is shared along the
        path and allows, for example, two occurrences to express an interval.
    monotonic_cst : array-like of {-1, 0, 1} or None, default=None
        Constraints on the positive-class probability in binary classification.
        ``+1`` requires the probability not to decrease as the feature grows,
        ``-1`` the opposite, and ``0`` leaves the feature free. The constraint
        applies to finite values; NaN follows the direction learned at each
        node and is outside the monotonic contract.
    max_depth : int >= 1 or None
        Maximum depth; the root has depth zero.
    min_samples_leaf : int >= 1
        Minimum number of positive-weight rows per child.
    max_leaf_nodes : int >= 2 or None
        Leaf budget; when set, the tree grows best-first.
    random_state : int or None
        Fixed feature permutation used to break ties, reproducible by seed.
    min_impurity_decrease : float >= 0
        Minimum gain to accept a split. For Gini it is the impurity decrease
        weighted by the node mass relative to the root; for precision it is
        the increase of the best child precision over the parent's.
    max_bins : int in [2, 255]
        Maximum number of finite bins per feature; NaN uses the separate bin 0.
    search_stopping : {'bound', 'off'}, default='bound'
        'bound': stop a scan only through an admissible upper bound; 'off': scan
        every candidate. This is not holdout-based stopping. The
        ``exact`` engine always searches exhaustively; when ``bound`` is
        requested, ``fit_stats_['search_stopping_effective']`` records ``'off'``.
    gain_tolerance : float >= 0, default=0.0
        Minimum local improvement still worth searching beyond the incumbent.
        Zero: exact bound. Positive: an approximation with a tolerance on the
        local Gini, with no guarantee on out-of-sample loss. Only active when
        the histogram bound is active.
    n_jobs : int >= 1, default=1
        Number of Numba threads for the parallel variant of the histogram
        engine. ``1`` keeps the serial baseline.
    reuse_parent_histograms : bool, default=False
        Experimental variant for depth-first growth with unit weights: builds
        the smaller child and gets the larger one by subtracting it from the
        parent histogram.
    leaf_smoothing : float >= 0, default=0.0
        Prior mass added to the leaves when predicting probabilities; the prior
        is the weighted class distribution at the root. Zero keeps the empirical
        frequencies exactly; it does not change splits or pruning.
    ccp_alpha : float >= 0, default=0.0
        Per-leaf penalty of minimal cost-complexity post-pruning with weighted
        Gini risk. Zero keeps the original tree. For now positive values are
        only accepted for ``objective='gini'``.
    leaf_shrinkage : float >= 0, default=0.0
        Hierarchical shrinkage (Agarwal et al., ICML 2022) of the probabilities:
        each leaf becomes a convex combination of its ancestors' frequencies,
        with weight 1 / (1 + lambda / parent mass) per level. It does not change
        the tree; it reduces overfitting of small leaves. Mutually exclusive
        with ``leaf_smoothing`` and, for now, with ``monotonic_cst``. To choose
        lambda and the number of leaves by internal validation, use
        ``FastDecisionTreeClassifierCV``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    n_classes_, n_features_in_ : int
        Number of classes and of features.
    feature_names_in_ : ndarray of str
        Column names, when ``X`` is a DataFrame.
    nodes_ : object
        The fitted tree as per-node arrays (``feature``, ``threshold``, ``left``,
        ``right``, ``missing_left``, ...).
    leaf_probabilities_ : ndarray or None
        Per-node class probabilities after hierarchical shrinkage when
        ``leaf_shrinkage > 0``; None otherwise.
    feature_importances_ : ndarray of shape (n_features,)
        Weighted Gini decrease per feature, normalized.
    fit_stats_ : dict
        Timings and search counters of the fit.

    Notes
    -----
    Cuts are scored by weighted Gini. An experimental precision objective lives in
    ``bettertrees.experimental.PrecisionTreeClassifier``.
    get_params/set_params come from BaseEstimator and ``clone`` works. There is
    no internal validation or automatic pruning selection here; smoothing and
    post-pruning are opt-in.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bettertrees import FastDecisionTreeClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> tree = FastDecisionTreeClassifier(max_leaf_nodes=8).fit(X, y)
    >>> tree.get_n_leaves()
    8
    >>> print(tree.export_text())  # doctest: +SKIP
    """

    classes_: LabelArray
    n_features_in_: int
    n_classes_: int
    feature_names_in_: LabelArray
    nodes_: NodeArrays
    bin_edges_: tuple[FloatArray, ...] | None

    # the cut objective and its options; only PrecisionTreeClassifier exposes them
    objective = "gini"
    positive_class = None
    min_precision = 0.9
    min_support = 1.0

    def __init__(self, *, splitter: Literal["hist", "exact"] = "hist",
                 max_depth: int | None = None, min_samples_leaf: int = 1,
                 max_leaf_nodes: int | None = None, random_state: Seed = None,
                 min_impurity_decrease: float = 0.0, max_bins: int = 255,
                 search_stopping: Literal["bound", "off"] = "bound",
                 gain_tolerance: float = 0.0, max_feature_repeats: int | None = None,
                 n_jobs: int = 1, reuse_parent_histograms: bool = False,
                 monotonic_cst: ArrayLike | None = None, leaf_smoothing: float = 0.0,
                 ccp_alpha: float = 0.0, leaf_shrinkage: float = 0.0) -> None:
        """Store the parameters without any training work (clone/set_params)."""
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.max_bins = max_bins
        self.search_stopping = search_stopping
        self.gain_tolerance = gain_tolerance
        self.max_feature_repeats = max_feature_repeats
        self.n_jobs = n_jobs
        self.reuse_parent_histograms = reuse_parent_histograms
        self.monotonic_cst = monotonic_cst
        self.leaf_smoothing = leaf_smoothing
        self.ccp_alpha = ccp_alpha
        self.leaf_shrinkage = leaf_shrinkage

    def _validate_parameters(self):
        """Reject invalid configurations before allocating data or compiling."""
        for name, minimum, allow_none in (("max_depth", 1, True),
                                          ("min_samples_leaf", 1, False),
                                          ("max_leaf_nodes", 2, True),
                                          ("max_bins", 2, False)):
            value = getattr(self, name)
            if value is None and allow_none:
                continue
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Integral)
                    or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.max_bins > 255:
            raise ValueError("max_bins must be <= 255.")
        if self.max_feature_repeats is not None and (
                isinstance(self.max_feature_repeats, (bool, np.bool_))
                or not isinstance(self.max_feature_repeats, Integral)
                or self.max_feature_repeats < 1):
            raise ValueError(
                "max_feature_repeats must be a positive integer or None.")
        if (isinstance(self.n_jobs, (bool, np.bool_))
                or not isinstance(self.n_jobs, Integral) or self.n_jobs < 1):
            raise ValueError("n_jobs must be an integer >= 1.")
        if not isinstance(self.reuse_parent_histograms, (bool, np.bool_)):
            raise ValueError("reuse_parent_histograms must be a boolean.")
        if self.reuse_parent_histograms and self.splitter != "hist":
            raise ValueError("Parent-child histogram reuse requires splitter='hist'.")
        if self.reuse_parent_histograms and self.max_leaf_nodes is not None:
            raise ValueError("Parent-child histogram reuse requires max_leaf_nodes=None.")
        if self.monotonic_cst is not None and np.isscalar(self.monotonic_cst):
            raise ValueError("monotonic_cst must be a vector or None.")
        resolve_splitter_spec(self.splitter, self.objective, self.search_stopping)
        for name in ("min_impurity_decrease", "gain_tolerance",
                     "leaf_smoothing", "ccp_alpha", "leaf_shrinkage"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.leaf_shrinkage > 0 and self.leaf_smoothing > 0:
            raise ValueError("Use leaf_shrinkage OR leaf_smoothing, not both.")
        if self.leaf_shrinkage > 0 and self.monotonic_cst is not None:
            raise ValueError("leaf_shrinkage cannot be combined with monotonic_cst yet.")
        if (self.random_state is not None and (isinstance(self.random_state, bool)
                or not isinstance(self.random_state, Integral))):
            raise ValueError("random_state must be an integer or None.")
        check_random_state(self.random_state)
        if self.objective == "precision":
            if self.positive_class is None:
                raise ValueError("positive_class is required for objective='precision'.")
            if self.gain_tolerance != 0:
                raise ValueError("gain_tolerance does not apply to objective='precision'.")
            if self.ccp_alpha > 0:
                raise ValueError("A positive ccp_alpha requires objective='gini'.")
        if (isinstance(self.min_precision, (bool, np.bool_))
                or not isinstance(self.min_precision, Real)
                or not np.isfinite(self.min_precision)
                or not 0 <= self.min_precision <= 1):
            raise ValueError("min_precision must be between 0 and 1.")
        if (isinstance(self.min_support, (bool, np.bool_))
                or not isinstance(self.min_support, Real)
                or not np.isfinite(self.min_support)
                or self.min_support <= 0):
            raise ValueError("min_support must be finite and positive.")

    @staticmethod
    def _feature_names(X):
        """Return text column names when X provides them.

        The core accepts any dense numeric matrix; the detection lives in the
        facade so that DataFrames stay optional and pandas never enters the
        training path of NumPy arrays.
        """
        columns = getattr(X, "columns", None)
        if columns is None:
            return None
        try:
            names = tuple(columns)
        except TypeError:
            return None
        if names and all(isinstance(name, str) for name in names):
            return names
        return None

    def _validate_feature_names(self, X):
        """Check DataFrame names without rejecting NumPy arrays at predict time."""
        fitted_names = getattr(self, "feature_names_in_", None)
        if fitted_names is None:
            return
        names = self._feature_names(X)
        if names is None:
            return
        if names != tuple(fitted_names):
            raise ValueError(
                "Feature names at prediction time must match the "
                "names and order seen during fit."
            )

    def _validate_predict_X(self, X):
        """Validate X for prediction, including the feature-name contract."""
        self._validate_feature_names(X)
        return validate_X(X, n_features=self.n_features_in_)

    def _validate_monotonic_cst(self, n_features, n_classes):
        """Validate the directions and return a stable integer vector for fit."""
        if self.monotonic_cst is None:
            return None
        if n_classes != 2:
            raise ValueError("monotonic_cst is only supported for binary classification.")
        try:
            values = np.asarray(self.monotonic_cst)
        except Exception as exc:  # pragma: no cover - contract message
            raise ValueError("monotonic_cst must be a vector of -1, 0 and 1.") from exc
        if values.ndim != 1 or len(values) != n_features:
            raise ValueError(
                "monotonic_cst must have one direction per feature.")
        if any(isinstance(value, (bool, np.bool_)) for value in values.tolist()):
            raise ValueError("monotonic_cst only accepts -1, 0 and 1.")
        try:
            numeric = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("monotonic_cst only accepts -1, 0 and 1.") from exc
        if (not np.isfinite(numeric).all()
                or not np.isin(numeric, [-1.0, 0.0, 1.0]).all()):
            raise ValueError("monotonic_cst only accepts -1, 0 and 1.")
        return numeric.astype(np.int8)

    @staticmethod
    def _finite_leaf_regions(nodes, n_features):
        """Finite boxes of the leaves; see ``postprocess.finite_leaf_regions``."""
        return finite_leaf_regions(nodes, n_features)

    def _project_monotonic_leaf_probabilities(self, nodes, directions,
                                              positive_class_index):
        """Delegate the monotonic projection to post-processing."""
        return project_monotonic_leaf_probabilities(
            nodes, directions, positive_class_index, self.leaf_smoothing)

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike,
            sample_weight: ArrayLike | None = None) -> SelfT:
        """Fit the tree (with a temporary scope for the Numba thread count).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric features; NaN is accepted.
        y : array-like of shape (n_samples,)
            Class labels.
        sample_weight : array-like of shape (n_samples,), optional
            Non-negative row weights.

        Returns
        -------
        self
        """
        self._validate_parameters()
        previous_threads = get_num_threads()
        # n_jobs above the machine's cores is clamped (Numba refuses more threads)
        set_num_threads(min(int(self.n_jobs), numba_config.NUMBA_NUM_THREADS))
        try:
            return self._fit_impl(X, y, sample_weight)
        finally:
            set_num_threads(previous_threads)

    def _fit_impl(self, X, y, sample_weight=None):
        """Validate the data, learn bins when needed and build the tree.

        All preparation, binning and growth belongs to the timed fit.
        Attributes ending in '_' are published only after the builder returns,
        so an incomplete fit never looks successful.
        """
        fit_start = perf_counter()
        feature_names = self._feature_names(X)
        splitter_spec = resolve_splitter_spec(
            self.splitter, self.objective, self.search_stopping)
        prepare_start = perf_counter()
        X, encoded, weights, classes = prepare_training_data(X, y, sample_weight)
        prepare_seconds = perf_counter() - prepare_start
        monotonic_directions = self._validate_monotonic_cst(
            X.shape[1], len(classes))
        positive_class_index = -1
        if self.objective == "precision":
            matches = np.flatnonzero(classes == self.positive_class)
            if len(matches) != 1:
                raise ValueError("positive_class must appear exactly once among the classes of y.")
            positive_class_index = int(matches[0])
        params = dict(n_classes=len(classes), max_depth=self.max_depth,
                      min_samples_leaf=self.min_samples_leaf,
                      max_leaf_nodes=self.max_leaf_nodes,
                      min_impurity_decrease=self.min_impurity_decrease,
                      feature_order=check_random_state(self.random_state).permutation(X.shape[1]),
                      stopping=self.search_stopping, gain_tolerance=self.gain_tolerance,
                      positive_class=positive_class_index,
                      min_precision=float(self.min_precision),
                      min_support=float(self.min_support),
                      max_feature_repeats=self.max_feature_repeats)
        edges = None
        stats = {}
        if self.splitter == "hist":
            effective_max_bins = self.max_bins
            bin_start = perf_counter()
            edges = fit_bin_edges(X, effective_max_bins, n_jobs=self.n_jobs)
            bin_learning_seconds = perf_counter() - bin_start
            transform_start = perf_counter()
            X_binned = transform_bins_row_major(X, edges, n_jobs=self.n_jobs)
            bin_transform_seconds = perf_counter() - transform_start
            builder_start = perf_counter()
            nodes = grow_tree_hist(X, X_binned, encoded, weights, edges,
                                    stats=stats, objective=splitter_spec.name,
                                    parallel=self.n_jobs > 1,
                                    reuse_parent_histograms=self.reuse_parent_histograms,
                                    **params)
        else:
            bin_learning_seconds = 0.0
            bin_transform_seconds = 0.0
            builder_start = perf_counter()
            nodes = grow_tree_exact(X, encoded, weights, stats=stats,
                                    objective=splitter_spec.name, **params)
        builder_seconds = perf_counter() - builder_start
        preprune_nodes = len(nodes.left)
        prune_start = perf_counter()
        if self.ccp_alpha > 0:
            nodes = prune_tree_cost_complexity(nodes, float(self.ccp_alpha))
        prune_seconds = perf_counter() - prune_start
        monotonic_positive_class_index = (
            positive_class_index if self.objective == "precision" else 1)
        if monotonic_directions is None:
            monotonic_leaf_probabilities = None
        else:
            monotonic_leaf_probabilities = (
                self._project_monotonic_leaf_probabilities(
                    nodes, monotonic_directions,
                    monotonic_positive_class_index))
        self.nodes_, self.classes_ = nodes, classes
        self.n_features_in_, self.n_classes_ = X.shape[1], len(classes)
        if feature_names is None:
            self.__dict__.pop("feature_names_in_", None)
        else:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.bin_edges_ = edges
        self.nan_features_ = np.isnan(X).any(axis=0)  # as in the sums: which columns had NaN
        self.monotonic_cst_ = monotonic_directions
        self.monotonic_positive_class_index_ = (
            monotonic_positive_class_index if monotonic_directions is not None
            else None)
        self.monotonic_leaf_probabilities_ = monotonic_leaf_probabilities
        stats.update({
            "splitter": self.splitter,
            "objective": splitter_spec.name,
            "objective_experimental": splitter_spec.experimental,
            "positive_class": self.positive_class if self.objective == "precision" else None,
            "min_precision": float(self.min_precision),
            "min_support": float(self.min_support),
            "leaf_smoothing": float(self.leaf_smoothing),
            "ccp_alpha": float(self.ccp_alpha),
            "max_feature_repeats": self.max_feature_repeats,
            "monotonic_cst": (None if monotonic_directions is None
                               else monotonic_directions.tolist()),
            "search_stopping": self.search_stopping,
            "search_stopping_effective": (
                "off" if self.splitter == "exact" else self.search_stopping),
            "gain_tolerance": float(self.gain_tolerance),
            "min_impurity_decrease": float(self.min_impurity_decrease),
            "n_jobs": int(self.n_jobs),
            "parallel": bool(self.n_jobs > 1 and self.splitter == "hist"),
            "reuse_parent_histograms": bool(self.reuse_parent_histograms),
            "max_bins_requested": int(self.max_bins),
            "max_bins_effective": int(effective_max_bins) if self.splitter == "hist" else None,
            "prepare_seconds": prepare_seconds,
            "bin_learning_seconds": bin_learning_seconds,
            "bin_transform_seconds": bin_transform_seconds,
            "builder_seconds": builder_seconds,
            "prune_seconds": prune_seconds,
            "preprune_nodes": preprune_nodes,
            "nodes_removed_by_pruning": preprune_nodes - len(nodes.left),
            "fit_complete_seconds": perf_counter() - fit_start,
            "n_nodes": len(nodes.left),
            "n_leaves": int(np.count_nonzero(nodes.left == -1)),
        })
        self.leaf_probabilities_ = (
            hierarchical_shrinkage_probabilities(nodes, float(self.leaf_shrinkage))
            if self.leaf_shrinkage > 0 else None)
        self.fit_stats_ = stats
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Class probabilities, columns ordered as ``classes_``."""
        check_is_fitted(self, "nodes_")
        return predict_proba_nodes(
            self._validate_predict_X(X), self.nodes_,
            self.monotonic_leaf_probabilities_,
            positive_class=(self.monotonic_positive_class_index_
                            if self.monotonic_positive_class_index_ is not None
                            else 1),
            leaf_smoothing=self.leaf_smoothing,
            leaf_probabilities=getattr(self, "leaf_probabilities_", None),
        )

    def predict(self, X: ArrayLike) -> LabelArray:
        """Original class labels; ties go to the lowest index in ``classes_``."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def predict_log_proba(self, X: ArrayLike) -> FloatArray:
        """Logarithm of the predicted probabilities."""
        probabilities = self.predict_proba(X)
        with np.errstate(divide="ignore"):
            return np.log(probabilities)

    def apply(self, X: ArrayLike) -> IndexArray:
        """Index of the leaf that receives each sample."""
        check_is_fitted(self, "nodes_")
        X_validated = self._validate_predict_X(X)
        return np.asarray(apply_nodes(
            X_validated, self.nodes_.left, self.nodes_.right,
            self.nodes_.feature, self.nodes_.threshold,
            self.nodes_.missing_left,
        ), dtype=np.intp)

    def export_text(self, feature_names: FeatureNames | None = None, precision: int = 4) -> str:
        """The tree drawn as text, like ``sklearn.tree.export_text``.

        Each leaf shows the probability ``predict_proba`` gives (with shrinkage,
        smoothing or monotone projection applied) and its training weight. A cut
        reads ``x <= v``; for a column that had NaN in training, the side that
        receives them is marked "or missing".
        """
        from .postprocess import node_probabilities
        check_is_fitted(self, "nodes_")
        nodes = self.nodes_
        names = (list(feature_names) if feature_names is not None
                 else list(getattr(self, "feature_names_in_", []))
                 or [f"x{j}" for j in range(self.n_features_in_)])
        probs = node_probabilities(
            nodes, np.arange(len(nodes.left)), self.monotonic_leaf_probabilities_,
            positive_class=(self.monotonic_positive_class_index_
                            if self.monotonic_positive_class_index_ is not None else 1),
            leaf_smoothing=self.leaf_smoothing,
            leaf_probabilities=getattr(self, "leaf_probabilities_", None))
        binary = len(self.classes_) == 2

        def leaf(node):
            p, w = probs[node], float(nodes.class_weight[node].sum())
            if binary:
                return f"P({self.classes_[1]}) = {p[1]:.{precision}f}  (weight {w:.6g})"
            k = int(np.argmax(p))
            return f"class {self.classes_[k]}, P = {p[k]:.{precision}f}  (weight {w:.6g})"

        lines = []

        def walk(node, depth):
            pad = "|   " * depth
            if nodes.left[node] == -1:
                lines.append(f"{pad}|--- {leaf(node)}")
                return
            name = names[int(nodes.feature[node])]
            thr = f"{float(nodes.threshold[node]):.{precision}g}"
            nan = getattr(self, "nan_features_", None)
            has_nan = nan is None or bool(nan[int(nodes.feature[node])])
            miss_left = bool(nodes.missing_left[node])
            left_tag = " or missing" if has_nan and miss_left else ""
            right_tag = " or missing" if has_nan and not miss_left else ""
            lines.append(f"{pad}|--- {name} <= {thr}{left_tag}")
            walk(int(nodes.left[node]), depth + 1)
            lines.append(f"{pad}|--- {name} >  {thr}{right_tag}")
            walk(int(nodes.right[node]), depth + 1)

        walk(0, 0)
        return "\n".join(lines)

    def get_depth(self) -> int:
        """Maximum depth of the tree; the root has depth zero."""
        check_is_fitted(self, "nodes_")
        max_depth = 0
        pending = [(0, 0)]
        while pending:
            node_id, depth = pending.pop()
            left = int(self.nodes_.left[node_id])
            if left == -1:
                continue
            max_depth = max(max_depth, depth + 1)
            pending.append((left, depth + 1))
            pending.append((int(self.nodes_.right[node_id]), depth + 1))
        return max_depth

    def get_n_leaves(self) -> int:
        """Number of leaves of the tree."""
        check_is_fitted(self, "nodes_")
        return int(np.count_nonzero(self.nodes_.left == -1))

    @property
    def feature_importances_(self) -> FloatArray:
        """Importance normalized by the weighted Gini decrease.

        The node mass is the sum of ``class_weight``, so with sample weights
        the decrease follows the weighted mass, not the raw row count. Trees
        without cuts return zeros, as in scikit-learn.
        """
        check_is_fitted(self, "nodes_")
        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for node_id, feature in enumerate(self.nodes_.feature):
            feature = int(feature)
            if feature < 0:
                continue
            left = int(self.nodes_.left[node_id])
            right = int(self.nodes_.right[node_id])
            parent_mass = self.nodes_.class_weight[node_id]
            left_mass = self.nodes_.class_weight[left]
            right_mass = self.nodes_.class_weight[right]
            parent_total = float(parent_mass.sum())
            left_total = float(left_mass.sum())
            right_total = float(right_mass.sum())
            if parent_total <= 0:
                continue
            parent_impurity = 1.0 - float(
                np.square(parent_mass / parent_total).sum())
            left_impurity = (0.0 if left_total <= 0 else 1.0 - float(
                np.square(left_mass / left_total).sum()))
            right_impurity = (0.0 if right_total <= 0 else 1.0 - float(
                np.square(right_mass / right_total).sum()))
            reduction = (parent_total * parent_impurity
                         - left_total * left_impurity
                         - right_total * right_impurity)
            importances[feature] += max(0.0, reduction)
        total = float(importances.sum())
        if total > 0.0:
            importances /= total
        return importances
