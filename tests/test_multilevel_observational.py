"""Offline contracts and opt-in checks for downloaded observational data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark_multilevel_observational import (DATASETS, EXPECTED,
                                                _finish, _indicators,
                                                _numeric_pair,
                                                load_observational)
from benchmark_multilevel_scale_real import choose_train, fit_real_engine


def test_categorical_vocabulary_and_imputation_come_only_from_train():
    train = pd.DataFrame({"c": ["A", "A", "B"], "v": [1.0, np.nan, 3.0]})
    test = pd.DataFrame({"c": ["NEW"], "v": [np.nan]})
    a, b, categories = _indicators(train, test, "c", 2)
    assert categories == ["A", "B"]
    np.testing.assert_array_equal(b, [[0, 0]])
    x, z = _numeric_pair(train, test, ["v"])
    assert x[1, 0] == z[0, 0] == 2.0
    assert a.shape == (3, 2)


def test_dataset_contract_rejects_missing_rows_and_nonfinite_features(monkeypatch):
    X = np.array([[0], [1]], dtype=np.float32)
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="incompleto"):
        _finish("skin", X, y, X, y, ["B"], "test")
    monkeypatch.setitem(EXPECTED, "skin", 4)
    with pytest.raises(ValueError, match="incompleto"):
        _finish("skin", np.array([[0], [np.nan]]), y, X, y,
                ["B"], "test")


@pytest.mark.parametrize("name", DATASETS)
def test_downloaded_real_data_integrity_and_smoke(name):
    root = Path("dados_originais/arvore_rapida/observacional")
    required = {
        "covtype": root.parent / "fresh" / "covertype",
        "skin": root / "skin.zip",
        "census": root / "census" / "census-income.data",
        "airlines": root / "airlines.arff",
        "creditcard": root / "creditcard.arff",
        "diabetes": root / "diabetes.zip",
    }[name]
    if not required.exists():
        pytest.skip("observational dataset was not downloaded")
    X_pool, y_pool, X_test, y_test, recipe = load_observational(name, root)
    assert len(X_pool) + len(X_test) == EXPECTED[name]
    assert X_pool.shape[1] == X_test.shape[1] == len(recipe["features"])
    assert set(np.unique(y_pool)) == set(np.unique(y_test)) == {0, 1}
    assert np.isfinite(X_pool).all() and np.isfinite(X_test).all()
    if name == "census":
        assert len(X_pool) == 199523 and len(X_test) == 99762
        assert not any("income" in feature.lower() for feature in recipe["features"])
    if name == "creditcard":
        assert "Time" not in recipe["features"]
        assert "Class" not in recipe["features"]
    if name == "diabetes":
        assert recipe["patient_overlap"] == 0
    X_small, y_small = choose_train(X_pool, y_pool, 1000, 0)
    for engine in ("greedy_exact", "block_hist_hist"):
        model = fit_real_engine(engine, X_small, y_small, depth=4,
                                max_bins=8, min_leaf=20, smoothing=2,
                                candidate_limit=256, seed=0)
        proba = model.predict_proba(X_test[:32])
        assert proba.shape == (32, 2)
        assert np.isfinite(proba).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1)
