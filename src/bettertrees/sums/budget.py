"""One estimator per cut budget, no tuning: the fixed rule of the benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .._typing import ArrayLike, FloatArray, LabelArray, SelfT
from .categorical import CategoricalInterleavedTreeClassifier, _is_cat_dtype
from .smalltrees import InterleavedTreeClassifier

CategoricalSpec = Literal["auto"] | Sequence[int | str] | None


def _has_categorical_columns(X: Any, spec: CategoricalSpec) -> bool:
    """Return True when ``X`` has categorical columns under ``spec``."""
    if spec is None:
        return False
    if spec != "auto":
        if isinstance(spec, (str, bytes)) or not hasattr(spec, "__iter__"):
            raise ValueError(
                "categorical_features must be 'auto', None, or a sequence of ints/names."
            )
        return len(list(spec)) > 0
    cols = getattr(X, "columns", None)
    if cols is not None:
        return any(_is_cat_dtype(X.iloc[:, j].dtype) for j in range(len(cols)))
    arr = np.asarray(X)
    if arr.ndim == 2 and (arr.dtype == object or arr.dtype.kind in ("U", "S")):
        if arr.dtype.kind in ("U", "S"):
            return True
        return any(isinstance(v, str) for j in range(arr.shape[1]) for v in arr[:, j])
    return False


class BudgetClassifier(ClassifierMixin, BaseEstimator):
    """The Interleaved Tree Model for a budget of ``max_splits`` cuts, without tuning.

    A fixed rule sets everything but the budget b (binary classification). It is the
    rule of the benchmark (56 datasets of 10 thousand to 2.2 million rows), where it was
    evaluated from 4 to 64 cuts:

    - up to 8 cuts: ``InterleavedTreeClassifier(max_splits=b, max_delta_step=4.0)``, full
      Newton steps capped at 4 logits (without the cap, a nearly pure leaf of
      rare-class data can diverge);
    - above 8 cuts: ``InterleavedTreeClassifier(max_splits=b, learning_rate=0.3)``.

    Both use ``lam = 2b`` (the ``InterleavedTreeClassifier`` default). When passed a
    DataFrame with categorical/string columns and ``categorical_features="auto"``,
    dispatches to ``CategoricalInterleavedTreeClassifier`` (Fisher+MDL subset splits,
    separate missing vs unseen codes, and a near-unique/timestamp guardrail). Purely
    numeric inputs keep bit-for-bit parity with ``InterleavedTreeClassifier``.

    The fitted model is ``model_``; its interpretation and editing methods
    (``explain``, ``rules``, ``to_sql``, ``predict_contributions``, ``prune``, ...) are
    available directly on this estimator.

    Parameters
    ----------
    max_splits : int >= 1, default=16
        The cut budget: the model has at most ``max_splits`` cuts (internal nodes, one per
        split operation). A (feature, threshold) pair may repeat, so the number of distinct
        pairs is at most the budget.
    categorical_features : "auto", sequence of int/str, or None, default="auto"
        Native categorical split handling. ``"auto"`` uses
        ``CategoricalInterleavedTreeClassifier`` when categorical/string columns are
        present; pass ``None`` to disable categorical auto-detection.

    Attributes
    ----------
    model_ : InterleavedTreeClassifier
        The fitted model the rule chose.
    classes_, n_features_in_, feature_names_in_
        As in the chosen model.

    See Also
    --------
    InterleavedTreeClassifier : the model, with every setting exposed.
    CategoricalInterleavedTreeClassifier : the same model with native categorical splits.

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

    def __init__(
        self,
        max_splits: int = 16,
        *,
        categorical_features: CategoricalSpec = "auto",
    ) -> None:
        self.max_splits = max_splits
        self.categorical_features = categorical_features

    def _make(self, X: Any = None) -> InterleavedTreeClassifier:
        b = self.max_splits
        if isinstance(b, bool) or not isinstance(b, Integral) or b < 1:
            raise ValueError("max_splits must be an integer >= 1.")
        b = int(b)
        use_cat = X is not None and _has_categorical_columns(X, self.categorical_features)
        if use_cat:
            if b <= 8:
                return CategoricalInterleavedTreeClassifier(
                    max_splits=b,
                    max_delta_step=4.0,
                    categorical_features=self.categorical_features,
                )
            return CategoricalInterleavedTreeClassifier(
                max_splits=b,
                learning_rate=0.3,
                categorical_features=self.categorical_features,
            )
        if b <= 8:
            return InterleavedTreeClassifier(max_splits=b, max_delta_step=4.0)
        return InterleavedTreeClassifier(max_splits=b, learning_rate=0.3)

    def fit(self: SelfT, X: ArrayLike, y: ArrayLike,
            sample_weight: ArrayLike | None = None) -> SelfT:
        """Fit the model the rule picks for ``max_splits``."""
        self.model_ = self._make(X).fit(X, y, sample_weight=sample_weight)
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
