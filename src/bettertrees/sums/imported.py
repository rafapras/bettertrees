"""Import a LightGBM model as an editable sum of trees, and refit its leaves.

``from_lightgbm`` turns a fitted binary LightGBM model into a ``TreeSum``: the
same trees on bins whose edges are LightGBM's own thresholds, so predictions
match exactly. The sum then has the whole bettertrees toolkit: identical trees
are merged (``merge_duplicates``), leaves can be re-estimated jointly
(``refit_leaves``: full Newton backfitting with an L2 penalty instead of
boosting's shrunken, one-pass values), cuts can be pruned or moved, and rules,
contributions and the SHAP export follow every edit.

``LightGBMRefitClassifier`` does it end to end: fit LightGBM, import, merge and
refit the leaves with ``refit_lam``. With ``max_splits`` it is a LightGBM with
at most that many cuts (as ``n_estimators * (num_leaves - 1)``), refitted.

Missing values: bettertrees always sends NaN to the left branch. The estimator
replaces NaN by a value below each column's minimum before LightGBM sees the
data, so both agree; ``from_lightgbm`` refuses splits that route NaN right.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .._typing import ArrayLike, FloatArray, Seed, SelfT
from ._common import fit_inputs, predict_input
from .smalltrees import SmallTree, _AdditiveTrees


class TreeSum(_AdditiveTrees):
    """A fitted sum of trees built from another library's model (no ``fit``).

    Created by ``from_lightgbm``; exposes prediction, the interpretation API and
    the editing API of every bettertrees sum. ``refit_leaves(X, y)`` re-estimates its
    leaves jointly.

    Attributes
    ----------
    trees_ : list of SmallTree
        The imported trees (constant trees folded into ``base_margin_``).
    base_margin_ : float
        LightGBM's initial score.
    classes_, n_features_in_, feature_names_in_, bin_edges_
        scikit-learn conventions; the bins are LightGBM's thresholds.
    """

    def __init__(self) -> None:
        pass

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike,
            sample_weight: ArrayLike | None = None) -> SelfT:
        raise TypeError("TreeSum is built by from_lightgbm(); refit its leaves with "
                        "refit_leaves(X, y).")


def _collect(node, thresholds):
    if "split_index" not in node:
        return
    if node.get("decision_type", "<=") != "<=":
        raise ValueError("only numerical '<=' splits are supported (no categorical splits).")
    if node.get("missing_type") == "Zero":
        raise ValueError("zero_as_missing splits are not supported; train LightGBM with "
                         "zero_as_missing=False.")
    if node.get("missing_type") == "NaN" and not node.get("default_left", True):
        raise ValueError("a split sends NaN to the right; bettertrees always sends NaN left. "
                         "Fill NaN below the minimum before training LightGBM.")
    thresholds[node["split_feature"]].add(float(node["threshold"]))
    _collect(node["left_child"], thresholds)
    _collect(node["right_child"], thresholds)


def from_lightgbm(model: Any, X: ArrayLike, y: ArrayLike, n_trees: int | None = None) -> TreeSum:
    """``TreeSum`` equivalent to a fitted binary ``lightgbm.LGBMClassifier``.

    Parameters
    ----------
    model : lightgbm.LGBMClassifier or lightgbm.Booster
        A fitted binary model with numerical ``<=`` splits; a split that sends NaN to
        the right raises (bettertrees always sends NaN left).
    X, y : array-like
        The training data or a sample of it; they set the scikit-learn state
        (``classes_``, feature names, ``n_features_in_``).
    n_trees : int or None, default=None
        Keep only the first ``n_trees`` trees.

    Numerical inputs keep float64 precision. At most 254 distinct thresholds
    per feature fit the internal uint8 bins; larger models are rejected.

    Returns
    -------
    TreeSum
        Its ``decision_function`` equals LightGBM's raw score.

    Examples
    --------
    >>> import lightgbm as lgb  # doctest: +SKIP
    >>> from bettertrees import from_lightgbm
    >>> lgbm = lgb.LGBMClassifier(n_estimators=5, num_leaves=4).fit(X, y)  # doctest: +SKIP
    >>> model = from_lightgbm(lgbm, X, y)  # doctest: +SKIP
    >>> print(model.explain())  # doctest: +SKIP
    """
    booster = model.booster_ if hasattr(model, "booster_") else model
    info = booster.dump_model()
    if info.get("num_class", 1) != 1:
        raise ValueError("only binary models are supported.")
    trees_info = info["tree_info"][:n_trees]
    p = int(info["max_feature_idx"]) + 1
    thresholds = [set() for _ in range(p)]
    for t in trees_info:
        _collect(t["tree_structure"], thresholds)
    for f, cuts in enumerate(thresholds):
        if len(cuts) > 254:
            raise ValueError(f"feature {f} has {len(cuts)} distinct thresholds; "
                             "at most 254 are supported by the uint8 bins. "
                             "Import fewer trees or retrain with fewer bins.")
    out = TreeSum()
    Xv, classes, _, _ = fit_inputs(out, X, y)
    if Xv.shape[1] != p:
        raise ValueError(f"X has {Xv.shape[1]} columns, the model {p}.")
    out.classes_ = classes
    out.input_dtype_ = "float64"
    out.bin_edges_ = tuple(np.array(sorted(s), dtype=np.float64) for s in thresholds)
    trees = []
    for t in trees_info:
        tree = SmallTree()

        def grow(node, n, tree=tree):
            if "split_index" not in n:
                tree.value[node] = float(n["leaf_value"])
                return
            f = int(n["split_feature"])
            b = int(np.searchsorted(out.bin_edges_[f], float(n["threshold"]))) + 1
            left, right = tree.split(node, f, b)
            grow(left, n["left_child"])
            grow(right, n["right_child"])

        grow(0, t["tree_structure"])
        trees.append(tree)
    # constant trees (no split) carry LightGBM's initial score: fold them into the base
    out.base_margin_ = float(sum(t.value[0] for t in trees if t.n_splits == 0))
    out.trees_ = [t for t in trees if t.n_splits > 0]
    out.lam_, out.learning_rate_ = 1.0, 1.0
    return out


class LightGBMRefitClassifier(_AdditiveTrees):
    """LightGBM with its leaves re-estimated jointly by bettertrees.

    Fits ``lightgbm.LGBMClassifier`` (``n_estimators``, ``num_leaves``,
    ``learning_rate``, ``reg_lambda``, ``min_child_samples``, 32 bins), imports
    it with ``from_lightgbm``, merges identical trees and refits all leaves and
    the base with full Newton backfitting (``refit_sweeps`` passes, L2 penalty
    ``refit_lam``). ``refit_lam=None`` keeps LightGBM's values.

    Parameters
    ----------
    max_splits : int or None, default=None
        Positive cut budget. The leaf limit is reduced when the budget is
        smaller than ``num_leaves - 1``; ``n_estimators`` then fits that budget.
    n_estimators, num_leaves, learning_rate, reg_lambda, min_child_samples
        Passed to LightGBM.
    refit_lam : float or None, default=10.0
        L2 penalty of the joint refit; None keeps LightGBM's leaf values.
    refit_sweeps : int, default=3
        Passes of the joint refit.
    random_state : int, default=0
        Seed passed to LightGBM.

    Attributes
    ----------
    trees_, base_margin_, classes_, bin_edges_, ...
        As every bettertrees sum (editable, explainable).
    lgbm_ : lightgbm.LGBMClassifier
        The fitted LightGBM before the refit.

    Examples
    --------
    >>> from bettertrees import LightGBMRefitClassifier
    >>> model = LightGBMRefitClassifier(max_splits=16).fit(X, y)  # doctest: +SKIP
    >>> print(model.explain())  # doctest: +SKIP
    """

    def __init__(self, *, max_splits: int | None = None, n_estimators: int = 100,
                 num_leaves: int = 4, learning_rate: float = 0.1, reg_lambda: float = 1.0,
                 min_child_samples: int = 20, refit_lam: float | None = 10.0,
                 refit_sweeps: int = 3, random_state: Seed = 0) -> None:
        self.max_splits = max_splits
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.refit_lam = refit_lam
        self.refit_sweeps = refit_sweeps
        self.random_state = random_state

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike,
            sample_weight: ArrayLike | None = None) -> SelfT:
        """Fit LightGBM, import, merge and refit (see the class docstring)."""
        import lightgbm as lgb
        if self.max_splits is not None and (
                isinstance(self.max_splits, bool | np.bool_)
                or not isinstance(self.max_splits, int | np.integer)
                or self.max_splits < 1):
            raise ValueError("max_splits must be a positive integer or None.")
        X, classes, target, w = fit_inputs(self, X, y, sample_weight)
        low = np.nanmin(np.where(np.isnan(X), np.inf, X), axis=0)
        self.nan_fill_ = np.where(np.isfinite(low), low - 1.0, -1.0)
        Xf = np.where(np.isnan(X), self.nan_fill_, X)
        leaves = int(max(2, self.num_leaves))
        if self.max_splits is not None:
            leaves = min(leaves, int(self.max_splits) + 1)
        n_est = (int(self.max_splits) // (leaves - 1) if self.max_splits is not None
                 else int(self.n_estimators))
        self.lgbm_ = lgb.LGBMClassifier(
            n_estimators=n_est, num_leaves=leaves, max_depth=-1, learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda, min_child_samples=int(self.min_child_samples),
            min_child_weight=1e-3, max_bin=32, n_jobs=1, random_state=self.random_state,
            verbose=-1).fit(Xf, target.astype(int), sample_weight=w)
        imported = from_lightgbm(self.lgbm_, Xf, target.astype(int))
        self.classes_ = classes
        self.input_dtype_ = imported.input_dtype_
        self.bin_edges_ = imported.bin_edges_
        self.trees_ = imported.trees_
        self.base_margin_ = imported.base_margin_
        self.lam_, self.learning_rate_ = 1.0, 1.0
        self.merge_duplicates()
        if self.refit_lam is not None and self.trees_:
            from ._common import rebin
            self._refit_binned(rebin(Xf, self.bin_edges_, dtype=self.input_dtype_),
                               target, w, float(self.refit_lam),
                               int(self.refit_sweeps))
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Logit of ``P(y = classes_[1])`` (NaN filled as at fit time)."""
        from ._common import rebin
        check_is_fitted(self, "trees_")
        X = predict_input(self, X, "trees_")
        # validated once here: handing the filled array back to the parent's
        # decision_function would re-validate it and warn about missing names
        Xb = rebin(np.where(np.isnan(X), self.nan_fill_, X), self.bin_edges_,
                   dtype=getattr(self, "input_dtype_", np.float32))
        m = np.full(len(Xb), self.base_margin_)
        for tree in self.trees_:
            m += tree.value[tree.leaf_ids(Xb)]
        return m
