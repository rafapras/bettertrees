"""One estimator per cut budget, no tuning: the fixed rule of the benchmark."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .._typing import ArrayLike, FloatArray, LabelArray, SelfT
from .smalltrees import InterleavedTreeClassifier


class BudgetClassifier(ClassifierMixin, BaseEstimator):
    """The Interleaved Tree Model for a budget of ``max_splits`` cuts, without tuning.

    A fixed rule sets everything but the budget b (binary classification). It is the
    rule of the benchmark (56 datasets of 10 thousand to 2.2 million rows), where it was
    evaluated from 4 to 64 cuts:

    - up to 8 cuts: ``InterleavedTreeClassifier(max_splits=b, max_delta_step=4.0)``, full
      Newton steps capped at 4 logits (without the cap, a nearly pure leaf of
      rare-class data can diverge);
    - above 8 cuts: ``InterleavedTreeClassifier(max_splits=b, learning_rate=0.3)``.

    Both use ``lam = 2b`` (the ``InterleavedTreeClassifier`` default). Above 64 cuts the
    same rule is applied, without validation.

    The fitted model is ``model_``; its interpretation and editing methods
    (``explain``, ``rules``, ``to_sql``, ``predict_contributions``, ``prune``, ...) are
    available directly on this estimator.

    Parameters
    ----------
    max_splits : int >= 1, default=16
        The cut budget: the model has at most ``max_splits`` cuts (internal nodes, one per
        split operation). A (feature, threshold) pair may repeat, so the number of distinct
        pairs is at most the budget.

    Attributes
    ----------
    model_ : InterleavedTreeClassifier
        The fitted model the rule chose.
    classes_, n_features_in_, feature_names_in_
        As in the chosen model.

    See Also
    --------
    InterleavedTreeClassifier : the model, with every setting exposed.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bettertrees import BudgetClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = BudgetClassifier(max_splits=8).fit(X, y)
    >>> model.model_.max_delta_step, model.model_.learning_rate
    (4.0, 1.0)
    >>> BudgetClassifier(max_splits=16).fit(X, y).model_.learning_rate
    0.3
    """

    def __init__(self, max_splits: int = 16) -> None:
        self.max_splits = max_splits

    def _make(self):
        b = self.max_splits
        if isinstance(b, bool) or not isinstance(b, Integral) or b < 1:
            raise ValueError("max_splits must be an integer >= 1.")
        b = int(b)
        if b <= 8:
            return InterleavedTreeClassifier(max_splits=b, max_delta_step=4.0)
        return InterleavedTreeClassifier(max_splits=b, learning_rate=0.3)

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike,
            sample_weight: ArrayLike | None = None) -> SelfT:
        """Fit the model the rule picks for ``max_splits``."""
        self.model_ = self._make().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        if hasattr(self.model_, "feature_names_in_"):
            self.feature_names_in_ = self.model_.feature_names_in_
        else:
            self.__dict__.pop("feature_names_in_", None)
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)

    def predict(self, X: ArrayLike) -> LabelArray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def decision_function(self, X: ArrayLike) -> FloatArray:
        check_is_fitted(self, "model_")
        return self.model_.decision_function(X)

    def __getattr__(self, name: str) -> Any:
        # only reached for names not found on the wrapper: public methods and fitted
        # attributes of the chosen model (explain, rules, trees_, base_margin_, ...)
        if name.startswith("__") or "model_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["model_"], name)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.allow_nan = True
        return tags
