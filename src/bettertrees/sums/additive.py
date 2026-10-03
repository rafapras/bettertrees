"""Additive boosting of shallow optimal trees (depth 1 or 2) in logit space.

Each round picks the **optimal** Newton tree of depth ``depth`` over the bins
(exhaustive search, not greedy; with more than ``max_features_d2`` features,
optimal within the features with the largest single-cut gain in that round),
with leaf values from a Newton step and shrinkage ``learning_rate``. Early
stopping on a validation fraction. The final model is a sum of readable
terms: each term is a rule of at most two cuts that adds a value to the logit
(one scorecard line).

References
----------
Lou, Caruana, Gehrke, Hooker. "Accurate Intelligible Models with Pairwise
Interactions." KDD 2013 (GA2M), and Nori, Jenkins, Koch, Caruana.
"InterpretML: A Unified Framework for Machine Learning Interpretability." 2019
(EBM): the additive shape of the model.
Chen, Guestrin. "XGBoost." KDD 2016: second-order gain and leaf values.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .._typing import ArrayLike, FeatureNames, FloatArray, Seed, SelfT
from ._common import (
    base_margin,
    bin_threshold,
    binned,
    fit_inputs,
    grad_hess,
    log_loss_margin,
    predict_input,
    rebin,
    teacher_top_features,
    top_features,
)
from ._kernels import best_cut_1d, best_depth2, depth2_leaf_ids, hist_1d, newton_leaf_values
from .edit import TreeEditMixin
from .explain import InterpretableSumMixin
from .smalltrees import SmallTree, _side


class AdditiveTreeBooster(TreeEditMixin, InterpretableSumMixin, ClassifierMixin, BaseEstimator):
    """Long sum of shallow optimal Newton trees (depth 1/2), with early stopping.

    Free capacity with a simple structure: each term is a tree with at most two
    cuts (a main effect or a pair of features) added to the logit. Every round
    fits the optimal tree on the residual (exhaustive search over bins), with
    leaves set by a Newton step times ``learning_rate``; it stops when the
    internal validation log-loss has not improved for ``patience`` rounds.

    Parameters
    ----------
    depth : {1, 2}, default=2
        Depth of each term.
    learning_rate : float, default=0.3
        Shrinkage of the leaf values.
    max_rounds : int, default=300
        Maximum number of terms.
    lam : float, default=1.0
        L2 regularization of the leaves (in the gain and the Newton step).
    min_weight : float, default=20.0
        Minimum hessian mass per leaf.
    max_bins : int, default=32
        Bins per feature.
    validation_fraction : float, default=0.15
        Share of rows used for early stopping (0 disables it).
    patience : int, default=20
        Rounds without improvement before stopping.
    max_features_d2 : int, default=128
        With more features, the depth-2 search uses the most important ones.
    feature_screen : {"lgbm", "fast"}, default="lgbm"
        Screening: LightGBM gain importance or FAST scores every round.
    random_state : int, default=0
        Seed of the validation split.

    Attributes
    ----------
    trees_ : list of SmallTree
        The terms as trees (see ``get_trees`` for real thresholds). Predictions
        read them, so the editing API (``prune``, ``set_cut``, ``refit_leaves``, ...)
        works as in the other sums.
    terms_ : list
        Compact representation of the terms as fitted (bins); not updated by edits.
    lam_, learning_rate_ : float
        Effective regularization and shrinkage (the editing refit uses ``lam_``).
    base_margin_ : float
        Initial constant logit.
    classes_, n_features_in_, feature_names_in_, bin_edges_, nan_features_
        scikit-learn and binning conventions.
    history_ : ndarray
        Validation log-loss per round.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bettertrees import AdditiveTreeBooster
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = AdditiveTreeBooster(depth=1, max_rounds=50, feature_screen="fast").fit(X, y)
    >>> len(model.trees_) <= 50
    True
    """

    def __init__(self, *, depth: Literal[1, 2] = 2, learning_rate: float = 0.3,
                 max_rounds: int = 300, lam: float = 1.0, min_weight: float = 20.0,
                 max_bins: int = 32, validation_fraction: float = 0.15, patience: int = 20,
                 max_features_d2: int | None = 128,
                 feature_screen: Literal["lgbm", "fast"] = "lgbm", random_state: Seed = 0) -> None:
        self.depth = depth
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.max_features_d2 = max_features_d2
        self.feature_screen = feature_screen
        self.random_state = random_state

    def _term(self, Xb, g, h, w, nb):
        if self.depth == 1:
            gains, cuts = best_cut_1d(hist_1d(Xb, g, h, w, int(nb.max())), nb,
                                      self.lam, self.min_weight)
            f = int(np.argmax(gains))
            if gains[f] <= 0:
                return None
            return (f, int(cuts[f]), -1, -1, -1, -1)
        feats = (self._fixed_feats if self.feature_screen == "lgbm" else
                 top_features(Xb, g, h, w, nb, self.lam, self.min_weight, self.max_features_d2))
        if feats is not None:  # many features: optimal depth 2 within the top k
            Xs = np.ascontiguousarray(Xb[:, feats])
            gains, res = best_depth2(Xs, g, h, w, np.ascontiguousarray(nb[feats]), self.lam,
                                     self.min_weight)
            f1 = int(np.argmax(gains))
            if gains[f1] <= 0:
                return None
            r = res[f1]
            remap = lambda j: int(feats[j]) if j >= 0 else -1
            return (int(feats[f1]), int(r[0]), remap(r[1]), int(r[2]), remap(r[3]), int(r[4]))
        gains, res = best_depth2(Xb, g, h, w, nb, self.lam, self.min_weight)
        f1 = int(np.argmax(gains))
        if gains[f1] <= 0:
            return None
        return (f1, *[int(v) for v in res[f1]])

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
        if self.depth not in (1, 2):
            raise ValueError("depth must be 1 or 2.")
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        val = np.zeros(n, dtype=bool)
        if self.validation_fraction > 0:
            val[rng.permutation(n)[:int(round(self.validation_fraction * n))]] = True
        tr = ~val
        Xb_all, edges, nb = binned(X[tr], self.max_bins)
        self._fixed_feats = None
        if self.depth == 2 and self.feature_screen == "lgbm":
            self._fixed_feats = teacher_top_features(X[tr], (target[tr] > 0.5).astype(int),
                                                     self.max_features_d2, w[tr],
                                                     self.random_state)
        Xb_val = rebin(X[val], edges)
        y_tr, w_tr, y_val, w_val = target[tr], w[tr], target[val], w[val]
        base = base_margin(y_tr, w_tr)
        m_tr = np.full(tr.sum(), base)
        m_val = np.full(val.sum(), base)
        terms, history = [], []
        best, best_round = np.inf, 0
        for _ in range(self.max_rounds):
            g, h = grad_hess(y_tr, m_tr, w_tr)
            spec = self._term(Xb_all, g, h, w_tr, nb)
            if spec is None:
                break
            ids = depth2_leaf_ids(Xb_all, *spec)
            values = self.learning_rate * newton_leaf_values(ids, g, h, 4, self.lam)
            m_tr += values[ids]
            terms.append((spec, values))
            if val.any():
                m_val += values[depth2_leaf_ids(Xb_val, *spec)]
                loss = log_loss_margin(y_val, m_val, w_val)
            else:
                loss = log_loss_margin(y_tr, m_tr, w_tr)
            history.append(loss)
            if loss < best - 1e-12:
                best, best_round = loss, len(terms)
            elif len(terms) - best_round >= self.patience:
                break
        self.terms_ = terms[:best_round] if val.any() else terms
        self.base_margin_ = base
        self.bin_edges_ = edges
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.history_ = np.array(history)
        self.lam_, self.learning_rate_ = float(self.lam), float(self.learning_rate)
        self.trees_ = self._terms_to_trees()
        return self

    def _terms_to_trees(self):
        """Terms as SmallTree (leaves 0-1 on the left, 2-3 on the right)."""
        trees = []
        for (f1, t1, fl, tl, fr, tr), values in self.terms_:
            tree = SmallTree.from_nested((f1, t1, _side(fl, tl), _side(fr, tr)))
            for child, leaves in ((tree.left[0], (0, 1)), (tree.right[0], (2, 3))):
                if tree.left[child] == -1:
                    tree.value[child] = values[leaves[0]]
                else:
                    tree.value[tree.left[child]] = values[leaves[0]]
                    tree.value[tree.right[child]] = values[leaves[1]]
            trees.append(tree)
        return trees

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Logit of ``P(y = classes_[1])``: base plus the sum of the terms."""
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
        m = np.full(len(Xb), self.base_margin_)
        for tree in self.trees_:
            m += tree.value[tree.leaf_ids(Xb)]
        return m

    def scorecard(self, feature_names: FeatureNames | None = None) -> list[tuple[int, str, float]]:
        """Readable terms: a list of (rule, logit value) per leaf of each term, as fitted.

        Reads ``terms_``, so it ignores later edits; ``rules()`` always reflects them.

        Identical leaves on a side without a cut appear once. ``x <= v``
        includes NaN (NaN always goes left).
        """
        check_is_fitted(self, "terms_")
        name = (lambda j: feature_names[j]) if feature_names is not None else (lambda j: f"x{j}")
        e = self.bin_edges_

        def cond(f, t, left):
            v = bin_threshold(e[f], t)
            return f"{name(f)} {'<=' if left else '>'} {v:.6g}"

        rows = []
        for k, ((f1, t1, fl, tl, fr, tr), values) in enumerate(self.terms_):
            for side, (fc, tc, leaves) in enumerate(((fl, tl, (0, 1)), (fr, tr, (2, 3)))):
                root = cond(f1, t1, side == 0)
                if fc < 0:
                    rows.append((k, root, float(values[leaves[0]])))
                else:
                    rows.append((k, f"{root} & {cond(fc, tc, True)}", float(values[leaves[0]])))
                    rows.append((k, f"{root} & {cond(fc, tc, False)}", float(values[leaves[1]])))
        return rows
