"""One estimator per cut budget, no tuning: the rule the benchmark validated."""

from numbers import Integral

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .smalltrees import InterleavedTreeClassifier


class BudgetClassifier(ClassifierMixin, BaseEstimator):
    """The best sum of trees for a budget of ``max_splits`` cuts, without tuning.

    The rule our benchmark validated (lam = 2b, the ``InterleavedTreeClassifier`` default,
    in both ITM regimes) (binary classification, 10k to over 1M rows,
    against LightGBM and XGBoost tuned under the same budget):

    - up to 8 cuts: ``InterleavedTreeClassifier(max_splits=b, max_delta_step=4.0)``, full
      Newton steps capped at 4 logits (without the cap, a nearly pure leaf of
      rare-class data can diverge);
    - above 8 cuts: ``InterleavedTreeClassifier(max_splits=b, learning_rate=0.3)``.

    The rule was evaluated from 4 to 64 cuts; above 64 the same rule is applied without
    validation in the benchmark. ``CompactTreeBooster`` is a public estimator of its own but
    the rule no longer chooses it.

    The fitted model is ``model_``; its interpretation and editing methods
    (``explain``, ``rules``, ``predict_contributions``, ``prune``, ...) are
    available directly on this estimator.

    Parameters
    ----------
    max_splits : int >= 1, default=16
        The cut budget.

    Attributes
    ----------
    model_ : InterleavedTreeClassifier
        The fitted model the rule chose.
    classes_, n_features_in_, feature_names_in_
        As in the chosen model.
    """

    def __init__(self, max_splits=16):
        self.max_splits = max_splits

    def _make(self):
        b = self.max_splits
        if isinstance(b, bool) or not isinstance(b, Integral) or b < 1:
            raise ValueError("max_splits must be an integer >= 1.")
        b = int(b)
        if b <= 8:
            return InterleavedTreeClassifier(max_splits=b, max_delta_step=4.0)
        return InterleavedTreeClassifier(max_splits=b, learning_rate=0.3)

    def fit(self, X, y, sample_weight=None):
        """Fit the model the rule picks for ``max_splits``."""
        self.model_ = self._make().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        if hasattr(self.model_, "feature_names_in_"):
            self.feature_names_in_ = self.model_.feature_names_in_
        else:
            self.__dict__.pop("feature_names_in_", None)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        return self.model_.decision_function(X)

    def __getattr__(self, name):
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
