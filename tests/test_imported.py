"""Importing LightGBM into an editable sum: exact predictions, refit, NaN, editing."""

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from bettertrees import LightGBMRefitClassifier, from_lightgbm  # noqa: E402


def _data(seed=0, n=3000, nan=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logit = X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * np.sign(X[:, 3])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    if nan:
        X[rng.random(n) < 0.1, 1] = np.nan
    return X, y


def _logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def test_import_reproduces_lightgbm_exactly():
    X, y = _data()
    m = lgb.LGBMClassifier(n_estimators=30, num_leaves=6, max_bin=32, verbose=-1).fit(X, y)
    s = from_lightgbm(m, X, y)
    np.testing.assert_allclose(s.decision_function(X), m.predict_proba(X, raw_score=True),
                               atol=1e-9)
    assert s.n_splits_ <= 30 * 5
    s.rules()
    np.testing.assert_allclose(s.base_margin_ + s.predict_contributions(X).sum(axis=1),
                               s.decision_function(X), atol=1e-9)


def test_import_refuses_nan_routed_right():
    X, y = _data(nan=True)
    m = lgb.LGBMClassifier(n_estimators=20, num_leaves=4, verbose=-1).fit(X, y)
    info = m.booster_.dump_model()

    def any_right(n):
        if "split_index" not in n:
            return False
        return ((n.get("missing_type") == "NaN" and not n.get("default_left", True))
                or any_right(n["left_child"]) or any_right(n["right_child"]))
    if any(any_right(t["tree_structure"]) for t in info["tree_info"]):
        with pytest.raises(ValueError, match="NaN"):
            from_lightgbm(m, X, y)


def test_refit_classifier_budget_nan_and_editing():
    X, y = _data(nan=True)
    m = LightGBMRefitClassifier(max_splits=24, num_leaves=5, learning_rate=0.3).fit(X, y)
    assert m.n_splits_ <= 24
    p = m.predict_proba(X)[:, 1]
    assert np.isfinite(p).all()
    raw = LightGBMRefitClassifier(max_splits=24, num_leaves=5, learning_rate=0.3,
                                  refit_lam=None).fit(X, y)
    assert _logloss(y, p) <= _logloss(y, raw.predict_proba(X)[:, 1]) + 1e-9  # refit on train
    t = next(i for i, tr in enumerate(m.trees_) if tr.n_splits)
    m.prune(t, 0)
    m.refit_leaves(X, y)
    assert np.isfinite(m.predict_proba(X)).all()


def test_refit_classifier_is_an_sklearn_estimator():
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    X, y = _data(n=1200)
    est = LightGBMRefitClassifier(max_splits=16, num_leaves=4)
    assert not hasattr(clone(est), "trees_")
    s = cross_val_score(est, X, y, cv=3, scoring="neg_log_loss")
    assert np.isfinite(s).all()


def test_import_preserves_float64_inputs_through_prediction_refit_and_prune():
    X = np.linspace(1., 1.00000005, 1000).reshape(-1, 1)
    y = (X[:, 0] > 1.000000025).astype(int)
    m = lgb.LGBMClassifier(n_estimators=1, num_leaves=2, min_child_samples=1,
                           n_jobs=1, verbose=-1).fit(X, y)
    original = m.booster_.predict(X, raw_score=True)
    s = from_lightgbm(m, X, y)
    np.testing.assert_allclose(s.decision_function(X), original, atol=1e-12)
    np.testing.assert_allclose(s.base_margin_ + s.predict_contributions(X).sum(axis=1),
                               original, atol=1e-12)
    expected = original.copy()
    for value in np.unique(original):
        rows = original == value
        p = 1. / (1. + np.exp(-value))
        expected[rows] -= np.sum(p - y[rows]) / (rows.sum() * p * (1. - p) + 1.)
    s.refit_leaves(X, y, lam=1., sweeps=1, refit_base=False)
    np.testing.assert_allclose(s.decision_function(X), expected, atol=1e-12)
    pruned = from_lightgbm(m, X, y).prune(0, 0, X=X)
    np.testing.assert_allclose(pruned.decision_function(X), original.mean(), atol=1e-12)


@pytest.mark.parametrize("n_thresholds", [254, 255])
def test_import_checks_uint8_capacity_at_the_boundary(n_thresholds):
    # A deterministic numerical LightGBM dump exercises the last valid bin,
    # without depending on how a training run distributes its chosen cuts.
    from types import SimpleNamespace
    trees = [dict(tree_structure=dict(
        split_index=0, split_feature=0, threshold=float(t), decision_type="<=",
        missing_type="None", default_left=True,
        left_child=dict(leaf_value=-1.), right_child=dict(leaf_value=1.)))
        for t in range(n_thresholds)]
    booster = SimpleNamespace(dump_model=lambda: dict(
        num_class=1, max_feature_idx=0, tree_info=trees))
    X = np.array([[-1.], [253.], [254.], [256.]])
    y = np.array([0, 0, 1, 1])
    if n_thresholds > 254:
        with pytest.raises(ValueError, match="at most 254"):
            from_lightgbm(booster, X, y)
    else:
        s = from_lightgbm(booster, X, y)
        cuts = np.arange(n_thresholds)
        expected = np.where(cuts >= X, -1., 1.).sum(axis=1)
        np.testing.assert_array_equal(s.decision_function(X), expected)


def test_import_refuses_zero_as_missing():
    X = np.repeat([-3., -2., -1., -.5, 0., 1., 2.], 100).reshape(-1, 1)
    y = np.repeat([0, 0, 1, 1, 0, 1, 1], 100)
    m = lgb.LGBMClassifier(n_estimators=1, num_leaves=2, min_child_samples=1,
                           zero_as_missing=True, n_jobs=1, verbose=-1).fit(X, y)
    with pytest.raises(ValueError, match="zero_as_missing"):
        from_lightgbm(m, X, y)


@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_refit_classifier_respects_budgets_smaller_than_requested_tree(budget):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1000, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = LightGBMRefitClassifier(max_splits=budget, num_leaves=8, refit_lam=None,
                                min_child_samples=1).fit(X, y)
    assert 0 < m.n_splits_ <= budget
    np.testing.assert_allclose(m.decision_function(X),
                               m.lgbm_.booster_.predict(X, raw_score=True), atol=1e-12)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_refit_classifier_rejects_invalid_cut_budget(budget):
    X, y = _data(n=100)
    with pytest.raises(ValueError, match="max_splits"):
        LightGBMRefitClassifier(max_splits=budget).fit(X, y)
