"""Small public-API gates: catch a broken estimator without testing every kernel."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import log_loss

from bettertrees import FastDecisionTreeClassifier


@pytest.mark.parametrize("splitter", ["hist", "exact"])
def test_public_binary_workflow_with_missing_weights_and_postprocessing(splitter):
    rng = np.random.default_rng(20260922)
    X = rng.normal(size=(240, 3)).astype(np.float32)
    y_binary = (X[:, 0] + 0.6 * X[:, 1] > 0).astype(np.int32)
    y = np.where(y_binary == 1, "good", "bad")
    X[rng.random(X.shape) < 0.08] = np.nan
    weights = rng.uniform(0.5, 2.0, len(X))
    weights[0] = 0.0
    X_original = X.copy()
    train, test = np.arange(180), np.arange(180, 240)
    params = dict(splitter=splitter, max_depth=4, max_leaf_nodes=12,
                  min_samples_leaf=5, random_state=7,
                  leaf_smoothing=3.0, ccp_alpha=0.001)
    model = FastDecisionTreeClassifier(**params).fit(
        X[train], y[train], sample_weight=weights[train])
    assert model.classes_.tolist() == ["bad", "good"]
    assert model.n_features_in_ == X.shape[1]
    assert 1 <= model.get_n_leaves() <= 12
    assert model.fit_stats_["n_nodes"] == len(model.nodes_.left)
    assert model.fit_stats_["nodes_removed_by_pruning"] >= 0
    p = model.predict_proba(X[test])
    assert p.shape == (len(test), 2)
    assert np.isfinite(p).all() and (p > 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.predict_log_proba(X[test]), np.log(p))
    np.testing.assert_array_equal(model.predict(X[test]),
                                  model.classes_[p.argmax(axis=1)])
    assert len(model.apply(X[test])) == len(test)
    assert np.isfinite(log_loss(y[test], p, labels=model.classes_))
    assert np.array_equal(X, X_original, equal_nan=True)
    duplicate = clone(model).fit(X[train], y[train], sample_weight=weights[train])
    np.testing.assert_array_equal(p, duplicate.predict_proba(X[test]))


@pytest.mark.parametrize("splitter", ["hist", "exact"])
def test_public_precision_workflow_remains_usable(splitter):
    X = np.arange(24, dtype=np.float32).reshape(-1, 1)
    y = (X[:, 0] >= 12).astype(np.int32)
    model = FastDecisionTreeClassifier(
        splitter=splitter, objective="precision", positive_class=1,
        min_precision=0.9, min_support=5, search_stopping="off",
        max_depth=2, leaf_smoothing=1.0, random_state=0).fit(X, y)
    p = model.predict_proba([[5], [18]])
    assert p.shape == (2, 2)
    assert p[0, 1] < p[1, 1]
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
