"""Distillation of a LightGBM teacher into a single tree.

- ``crossfit_teacher``: out-of-fold probability (each row is predicted by a
  teacher that did not see it), with internal early stopping per fold;
- ``soft_label_expand``: each row becomes two (y=1 with weight w*p, y=0 with
  weight w*(1-p)); the engine's weighted Gini then optimizes against the soft
  target without changing the engine. ``min_samples_leaf`` counts expanded rows;
- ``restate_leaf_masses``: re-estimates all node masses with the real y (the
  structure comes from the teacher, the final word comes from the data);
- ``MixedDepthTree``: top of depth ``top_depth`` fitted on one target and the
  bottom on the other.

References
----------
Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." 2015.
Craven, Shavlik. "Extracting Tree-Structured Representations of Trained
Networks." NeurIPS 1995.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from ..estimator import FastDecisionTreeClassifier
from ..postprocess import hierarchical_shrinkage_probabilities
from ..sums._common import as_float_matrix, as_target, as_weights

TEACHER_DEFAULTS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                        min_child_samples=20, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, verbose=-1, deterministic=True,
                        force_row_wise=True, n_jobs=1)


def crossfit_teacher(X, y, *, n_splits=5, random_state=0, params=None,
                     early_stopping_fraction=0.15, return_models=False,
                     folds=None):
    """Out-of-fold probability of the positive class (largest label) from LightGBM.

    Inside each training fold, ``early_stopping_fraction`` becomes the early
    stopping validation set. ``folds`` (n,) fixes the partition (e.g. the same
    across the arms of an experiment); ``None`` uses StratifiedKFold. Returns a
    dict with ``p`` (n,), ``fold`` (n,), ``best_iterations`` and, if requested,
    ``models``.
    """
    import lightgbm as lgb

    X = as_float_matrix(X)
    classes, yy = np.unique(np.asarray(y), return_inverse=True)
    if len(classes) != 2:
        raise ValueError("crossfit_teacher supports binary targets only.")
    cfg = dict(TEACHER_DEFAULTS, random_state=random_state, **(params or {}))
    p = np.full(len(X), np.nan)
    fold = np.full(len(X), -1, dtype=np.int64)
    best, models = [], []
    rng = np.random.default_rng(random_state)
    if folds is None:
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X, yy))
    else:
        folds = np.asarray(folds)
        splits = [(np.flatnonzero(folds != k), np.flatnonzero(folds == k))
                  for k in np.unique(folds)]
    for k, (tr, te) in enumerate(splits):
        tr = rng.permutation(tr)
        n_val = int(round(early_stopping_fraction * len(tr)))
        fit_idx, val_idx = tr[n_val:], tr[:n_val]
        model = lgb.LGBMClassifier(**cfg)
        model.fit(X[fit_idx], yy[fit_idx], eval_set=[(X[val_idx], yy[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        p[te] = model.predict_proba(X[te])[:, 1]
        fold[te] = k
        best.append(int(model.best_iteration_ or cfg["n_estimators"]))
        if return_models:
            models.append(model)
    out = dict(p=p, fold=fold, best_iterations=best, classes=classes)
    if return_models:
        out["models"] = models
    return out


def soft_label_expand(X, p, sample_weight=None):
    """(X2, y2, w2): X duplicated, y2 = [1..., 0...], w2 = [w*p, w*(1-p)].

    Zero-weight rows are dropped by the engine; with p in {0, 1} the result is
    exactly the original data.
    """
    X = as_float_matrix(X)
    n = len(X)
    p = as_target(p, n)
    w = as_weights(sample_weight, n)
    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    w2 = np.concatenate([w * p, w * (1 - p)])
    return X2, y2, w2


def restate_leaf_masses(model, X, y, sample_weight=None):
    """Rewrite ``class_weight``/``n_samples`` of every node from the real (X, y).

    ``y`` uses labels compatible with ``model.classes_``. Recomputes
    ``leaf_probabilities_`` if the model uses hierarchical shrinkage. Nodes
    that receive no rows inherit the parent's mass (they never show up when
    predicting on the same data, but the tree stays valid).
    """
    check_is_fitted(model, "nodes_")
    nodes = model.nodes_
    X = as_float_matrix(X)
    w = as_weights(sample_weight, len(X))
    cls = np.searchsorted(model.classes_, np.asarray(y))
    if (cls >= len(model.classes_)).any() or (model.classes_[cls] != np.asarray(y)).any():
        raise ValueError("y contains classes not in model.classes_.")
    leaves = model.apply(X)
    cw = np.zeros_like(nodes.class_weight)
    ns = np.zeros_like(nodes.n_samples)
    np.add.at(cw, (leaves, cls), w)
    np.add.at(ns, leaves, 1)
    order = []  # iterative post-order
    stack = [0]
    while stack:
        node = stack.pop()
        order.append(node)
        if nodes.left[node] != -1:
            stack.extend((nodes.left[node], nodes.right[node]))
    for node in reversed(order):
        if nodes.left[node] != -1:
            cw[node] = cw[nodes.left[node]] + cw[nodes.right[node]]
            ns[node] = ns[nodes.left[node]] + ns[nodes.right[node]]
    for node in order:  # parents before children
        if cw[node].sum() <= 0:
            parent = np.flatnonzero((nodes.left == node) | (nodes.right == node))
            cw[node] = cw[parent[0]] if len(parent) else nodes.class_weight[node]
    model.nodes_ = nodes._replace(class_weight=cw, n_samples=ns)
    if getattr(model, "leaf_shrinkage", 0) > 0:
        model.leaf_probabilities_ = hierarchical_shrinkage_probabilities(
            model.nodes_, float(model.leaf_shrinkage))
    return model


def fit_tree_on_target(X, y, p, target, *, leaf_target="y", sample_weight=None,
                       classes=None, **tree_params):
    """Tree on the ``'y'`` (label) or ``'p'`` (soft) target; leaves on ``leaf_target``.

    ``classes``: the two global labels (needed when the subset of y has a
    single class and the target is soft).
    """
    X = as_float_matrix(X)
    if target == "y":
        model = FastDecisionTreeClassifier(**tree_params).fit(X, y, sample_weight)
        return model
    if target != "p":
        raise ValueError("target must be 'y' or 'p'.")
    classes = np.unique(np.asarray(y)) if classes is None else np.asarray(classes)
    if len(classes) != 2:
        raise ValueError("A soft target requires a binary y.")
    X2, y2, w2 = soft_label_expand(X, p, sample_weight)
    model = FastDecisionTreeClassifier(**tree_params).fit(X2, classes[y2], w2)
    if leaf_target == "y":
        restate_leaf_masses(model, X, y, sample_weight)
    elif leaf_target != "p":
        raise ValueError("leaf_target must be 'y' or 'p'.")
    return model


class MixedDepthTree(ClassifierMixin, BaseEstimator):
    """Top (``top_depth`` levels) fitted on one target, bottom on the other; binary.

    ``fit(X, y, p)``: ``p`` is the teacher's out-of-fold probability. The final
    leaves are estimated on ``leaf_target``. Each bottom subtree is a
    ``FastDecisionTreeClassifier`` of depth ``depth - top_depth`` on the rows of
    its top leaf (bins re-learned on those rows).
    """

    def __init__(self, *, depth=6, top_depth=3, top_target="p",
                 bottom_target="y", leaf_target="y", tree_params=None):
        self.depth = depth
        self.top_depth = top_depth
        self.top_target = top_target
        self.bottom_target = bottom_target
        self.leaf_target = leaf_target
        self.tree_params = tree_params

    def fit(self, X, y, p, sample_weight=None):
        X = as_float_matrix(X)
        y = np.asarray(y)
        w = as_weights(sample_weight, len(X))
        p = as_target(p, len(X))
        if not 0 < self.top_depth <= self.depth:
            raise ValueError("top_depth must satisfy 0 < top_depth <= depth.")
        params = dict(self.tree_params or {})
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("MixedDepthTree supports binary targets only.")
        self.top_ = fit_tree_on_target(X, y, p, self.top_target,
                                       leaf_target=self.leaf_target, sample_weight=w,
                                       classes=self.classes_,
                                       max_depth=self.top_depth, **params)
        self.bottom_ = {}
        rest = self.depth - self.top_depth
        if rest > 0:
            leaves = self.top_.apply(X)
            for leaf in np.unique(leaves):
                rows = leaves == leaf
                if len(np.unique(y[rows])) < 2 and self.bottom_target == "y":
                    continue  # pure leaf: nothing to cut
                sub = fit_tree_on_target(X[rows], y[rows], p[rows], self.bottom_target,
                                         leaf_target=self.leaf_target,
                                         sample_weight=w[rows], classes=self.classes_,
                                         max_depth=rest, **params)
                if list(sub.classes_) != list(self.classes_):
                    continue
                self.bottom_[int(leaf)] = sub
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "top_")
        X = as_float_matrix(X)
        out = self.top_.predict_proba(X)
        leaves = self.top_.apply(X)
        for leaf, sub in self.bottom_.items():
            rows = leaves == leaf
            if rows.any():
                out[rows] = sub.predict_proba(X[rows])
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def get_n_leaves(self):
        top = self.top_.get_n_leaves()
        return top - len(self.bottom_) + sum(s.get_n_leaves() for s in self.bottom_.values())
