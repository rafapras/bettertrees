"""Choice of capacity and shrinkage by internal cross-validation.

A best-first tree with L leaves is the prefix of the first L-1 expansions of
the tree with L_max leaves: the expansion order does not depend on the budget.
So each inner fold needs ONE fit (with L_max); every capacity and every lambda
of hierarchical shrinkage is scored on it. The final choice is refitted on the
full training set with the winning parameters.

References
----------
Agarwal, Tan, Ronen, Singh, Yu. "Hierarchical Shrinkage: Improving the Accuracy
and Interpretability of Tree-Based Methods." ICML 2022 (the leaf shrinkage).
Friedman, Hastie, Tibshirani. "Additive logistic regression: a statistical view
of boosting." Annals of Statistics, 2000 (best-first growth by gain).
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from ._data import prepare_training_data
from ._typing import ArrayLike, FeatureNames, FloatArray, IndexArray, LabelArray, Seed, SelfT
from .estimator import FastDecisionTreeClassifier
from .postprocess import expansion_steps, hierarchical_shrinkage_probabilities, prefix_leaf_ids

# Wide grid: tuned sklearn + HS often picks > 256 leaves and lambda > 200;
# stopping at 256/200 cost 0.7% log-loss in the benchmark.
# Leaves beyond n/min_samples_leaf are never reached, so small n pays nothing.
DEFAULT_LEAVES = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_SHRINKAGE = (1.0, 5.0, 20.0, 50.0, 200.0, 500.0, 1000.0)


def _log_loss(y, proba, weights):
    picked = np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.average(np.log(picked), weights=weights))


class FastDecisionTreeClassifierCV(ClassifierMixin, BaseEstimator):
    """Best-first tree whose number of leaves and shrinkage are chosen by CV.

    Parameters
    ----------
    leaves_grid : sequence of int >= 2
        Candidate capacities (maximum number of leaves).
    shrinkage_grid : sequence of float > 0
        Candidate ``leaf_shrinkage`` values.
    cv : int >= 2
        Stratified inner folds; the criterion is the validation log-loss.
        Each class with positive sample weight needs at least ``cv`` rows.
        Zero-weight rows do not enter the inner folds; the final tree retains
        all classes observed in ``y``.
    min_samples_leaf, splitter, max_bins, n_jobs, random_state
        Passed to ``FastDecisionTreeClassifier`` (no pruning: truncating by
        prefix needs node ids in expansion order).

    Attributes
    ----------
    best_params_ : dict
        The chosen ``max_leaf_nodes`` and ``leaf_shrinkage``.
    best_estimator_ : FastDecisionTreeClassifier
        The tree refitted on all rows with ``best_params_``.
    cv_scores_ : ndarray of shape (len(leaves_grid), len(shrinkage_grid))
        Mean validation log-loss of each pair.
    classes_, n_features_in_, feature_names_in_
        As in ``best_estimator_``.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bettertrees import FastDecisionTreeClassifierCV
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> tree = FastDecisionTreeClassifierCV(leaves_grid=(4, 8), shrinkage_grid=(1.0, 10.0))
    >>> sorted(tree.fit(X, y).best_params_)
    ['leaf_shrinkage', 'max_leaf_nodes']
    """

    classes_: LabelArray
    n_features_in_: int
    feature_names_in_: LabelArray
    cv_scores_: FloatArray
    best_params_: dict[str, int | float]
    best_estimator_: FastDecisionTreeClassifier

    def __init__(self, *, leaves_grid: Sequence[int] = DEFAULT_LEAVES,
                 shrinkage_grid: Sequence[float] = DEFAULT_SHRINKAGE, cv: int = 3,
                 min_samples_leaf: int = 5, splitter: Literal["hist", "exact"] = "hist",
                 max_bins: int = 255, n_jobs: int = 1, random_state: Seed = 0) -> None:
        self.leaves_grid = leaves_grid
        self.shrinkage_grid = shrinkage_grid
        self.cv = cv
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.max_bins = max_bins
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _tree(self, **extra):
        return FastDecisionTreeClassifier(
            splitter=self.splitter, min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins, n_jobs=self.n_jobs,
            random_state=self.random_state, **extra)

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike,
            sample_weight: ArrayLike | None = None) -> SelfT:
        try:
            leaves = list(self.leaves_grid)
        except TypeError as exc:
            raise ValueError("leaves_grid must contain integers >= 2.") from exc
        if not leaves or any(isinstance(v, bool | np.bool_)
                             or not isinstance(v, Integral) or v < 2 for v in leaves):
            raise ValueError("leaves_grid must contain integers >= 2.")
        try:
            shrinkage = list(self.shrinkage_grid)
        except TypeError as exc:
            raise ValueError("shrinkage_grid must contain finite values > 0.") from exc
        if not shrinkage or any(isinstance(v, bool | np.bool_)
                                or not isinstance(v, Real)
                                or not np.isfinite(v) or v <= 0 for v in shrinkage):
            raise ValueError("shrinkage_grid must contain finite values > 0.")
        if (isinstance(self.cv, bool | np.bool_)
                or not isinstance(self.cv, Integral) or self.cv < 2):
            raise ValueError("cv must be an integer >= 2.")
        leaves = sorted({int(v) for v in leaves})
        shrinkage = sorted({float(v) for v in shrinkage})
        self._tree()._validate_parameters()
        X_input = X  # the final fit gets the original input, so it keeps the column names
        y_input = y
        X, encoded, weights, classes = prepare_training_data(X, y, sample_weight)
        counts = np.bincount(encoded, minlength=len(classes))
        if np.any(counts[counts > 0] < self.cv):
            raise ValueError(
                "Each class with positive sample weight must have at least cv rows.")
        y = classes[encoded]
        scores = np.zeros((len(leaves), len(shrinkage)))
        folds = StratifiedKFold(n_splits=int(self.cv), shuffle=True,
                                random_state=self.random_state)
        for fit_rows, val_rows in folds.split(X, encoded):
            model = self._tree(max_leaf_nodes=leaves[-1]).fit(
                X[fit_rows], y[fit_rows], sample_weight=weights[fit_rows])
            nodes = model.nodes_
            steps = expansion_steps(nodes)
            # Classes missing from the inner fold: column with probability ~0.
            columns = np.searchsorted(classes, model.classes_)
            X_val = model._validate_predict_X(X[val_rows])
            y_val, w_val = encoded[val_rows], weights[val_rows]
            node_probs = []
            for lam in shrinkage:
                inner = hierarchical_shrinkage_probabilities(nodes, lam)
                full = np.full((len(inner), len(classes)), 1e-15)
                full[:, columns] = inner
                node_probs.append(full)
            for i, n_leaves in enumerate(leaves):
                ids = prefix_leaf_ids(X_val, nodes, steps, n_leaves)
                for j, probs in enumerate(node_probs):
                    scores[i, j] += _log_loss(y_val, probs[ids], w_val) / int(self.cv)
        i, j = np.unravel_index(int(np.argmin(scores)), scores.shape)
        best_params = dict(max_leaf_nodes=leaves[i], leaf_shrinkage=shrinkage[j])
        best_estimator = self._tree(**best_params).fit(
            X_input, y_input, sample_weight=sample_weight)
        self.best_params_, self.cv_scores_ = best_params, scores
        self.best_estimator_ = best_estimator
        self.classes_ = self.best_estimator_.classes_
        self.n_features_in_ = self.best_estimator_.n_features_in_
        # the inner tree validates names at predict time; mirror them here (sklearn contract)
        if hasattr(self.best_estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.best_estimator_.feature_names_in_
        else:
            self.__dict__.pop("feature_names_in_", None)
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)

    def predict(self, X: ArrayLike) -> LabelArray:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def apply(self, X: ArrayLike) -> IndexArray:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.apply(X)

    def export_text(self, feature_names: FeatureNames | None = None, precision: int = 4) -> str:
        """The selected tree as text (see ``FastDecisionTreeClassifier.export_text``)."""
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.export_text(feature_names, precision)
