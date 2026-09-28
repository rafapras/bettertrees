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
