"""A single tree whose cuts maximize the precision of one class (experimental)."""

from ..estimator import FastDecisionTreeClassifier


class PrecisionTreeClassifier(FastDecisionTreeClassifier):
    """``FastDecisionTreeClassifier`` with the experimental precision objective.

    Each cut is scored by the best precision among the children with minimum
    support, minus the parent's precision, without the Gini bound. It does not
    maximize true positives or coverage and tends toward extreme cuts that
    isolate the purest child. ``feature_importances_`` is still measured in Gini.

    Parameters
    ----------
    positive_class : scalar
        The class whose precision is maximized (required).
    min_precision : float in [0, 1], default=0.9
        Minimum precision of an eligible leaf.
    min_support : float > 0, default=1.0
        Minimum weighted mass of an eligible leaf.
    search_stopping : {'off'}, default='off'
        The precision objective has no admissible bound, so the scan is exhaustive.
    Other parameters
        As in ``FastDecisionTreeClassifier`` (``ccp_alpha`` must stay 0 and
        ``gain_tolerance`` does not apply).

    Attributes
    ----------
    As in ``FastDecisionTreeClassifier``.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bettertrees.experimental import PrecisionTreeClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> tree = PrecisionTreeClassifier(positive_class=1, max_depth=2).fit(X, y)
    >>> tree.get_depth() <= 2
    True
    """

    objective = "precision"

    def __init__(self, *, positive_class=None, min_precision=0.9, min_support=1.0,
                 splitter="hist", max_depth=None, min_samples_leaf=1, max_leaf_nodes=None,
                 random_state=None, min_impurity_decrease=0.0, max_bins=255,
                 search_stopping="off", gain_tolerance=0.0, max_feature_repeats=None,
                 n_jobs=1, reuse_parent_histograms=False, monotonic_cst=None,
                 leaf_smoothing=0.0, ccp_alpha=0.0, leaf_shrinkage=0.0):
        super().__init__(
            splitter=splitter, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes, random_state=random_state,
            min_impurity_decrease=min_impurity_decrease, max_bins=max_bins,
            search_stopping=search_stopping, gain_tolerance=gain_tolerance,
            max_feature_repeats=max_feature_repeats, n_jobs=n_jobs,
            reuse_parent_histograms=reuse_parent_histograms, monotonic_cst=monotonic_cst,
            leaf_smoothing=leaf_smoothing, ccp_alpha=ccp_alpha, leaf_shrinkage=leaf_shrinkage)
        self.positive_class = positive_class
        self.min_precision = min_precision
        self.min_support = min_support
