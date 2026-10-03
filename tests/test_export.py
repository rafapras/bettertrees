"""Execute exported artifacts with an independent, dependency-free evaluator."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from bettertrees import AdditiveTreeBooster, BudgetClassifier, FIGSClassifier, SumOfOptimalTrees

EXECUTOR = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/predict_export.py"))


def _roundtrip(model, X, precision=6):
    exported = json.loads(json.dumps(model.to_dict(precision=precision), allow_nan=False))
    rows = [[None if np.isnan(v) else float(v) for v in row] for row in X]
    np.testing.assert_allclose(EXECUTOR["decision_function"](exported, rows),
                               model.decision_function(X), rtol=0, atol=1e-12)
    np.testing.assert_allclose(EXECUTOR["predict_proba"](exported, rows),
                               model.predict_proba(X), rtol=0, atol=1e-12)
    return exported


def test_export_preserves_full_precision_and_unseen_missing():
    X = np.array([[1.234565], [1.234565], [1.234569], [1.234569]])
    model = FIGSClassifier(max_splits=1, min_weight=1).fit(X, [0, 0, 1, 1])
    query = np.array([[1.234568], [np.nan]])
    assert model.decision_function(query)[0] > 0
    assert model.decision_function(query)[1] < 0
    exported = _roundtrip(model, query, precision=1)
    assert exported["bin_edges"][0] == model.bin_edges_[0].tolist()


@pytest.mark.parametrize("make", [
    lambda: FIGSClassifier(max_splits=6, min_weight=1),
    lambda: SumOfOptimalTrees(n_trees=2, min_weight=1),
    lambda: AdditiveTreeBooster(max_rounds=5, min_weight=1),
    lambda: BudgetClassifier(max_splits=4),
    lambda: BudgetClassifier(max_splits=65),
])
def test_export_matches_on_random_data_boundaries_and_after_edits(make):
    rng = np.random.default_rng(821)
    X = rng.normal(size=(240, 2))
    X[::11, 1] = np.nan
    y = (X[:, 0] + np.nan_to_num(X[:, 1], copy=True) > 0).astype(int)
    model = make().fit(X, y)
    boundary = []
    for j, edges in enumerate(model.bin_edges_):
        for edge in edges:
            for value in (edge, edge - 1e-9, edge + 1e-9):
                row = np.zeros(2)
                row[j] = value
                boundary.append(row)
    query = np.vstack([X, np.array([[np.nan, np.nan]]), np.asarray(boundary).reshape(-1, 2)])
    _roundtrip(model, query)
    model.add_stump(0, 0, left_value=-0.123456789, right_value=0.234567891)
    _roundtrip(model, query)


def test_shap_and_real_tree_export_cast_inputs_before_comparison():
    shap = pytest.importorskip("shap")
    X = np.array([[0.], [0.], [1.], [1.]])
    model = FIGSClassifier(max_splits=1, min_weight=1).fit(X, [0, 0, 1, 1])
    query = np.array([[0.5 + 1e-9], [np.nan]])
    tree = model.get_trees()[0]
    assert tree["input_dtype"] == "float32"
    explainer = shap.TreeExplainer(model.to_shap_model(), data=X,
                                   feature_perturbation="interventional")
    reconstructed = explainer.expected_value + explainer.shap_values(query).sum(axis=1)
    np.testing.assert_allclose(reconstructed, model.decision_function(query), atol=1e-12)


def test_export_imported_lightgbm_preserves_float64():
    lgb = pytest.importorskip("lightgbm")
    from bettertrees import from_lightgbm
    X = np.linspace(1., 1.00000005, 1000).reshape(-1, 1)
    y = (X[:, 0] > 1.000000025).astype(int)
    source = lgb.LGBMClassifier(n_estimators=1, num_leaves=2, min_child_samples=1,
                                verbose=-1, n_jobs=1).fit(X, y)
    model = from_lightgbm(source, X, y)
    exported = _roundtrip(model, np.vstack([X, [[np.nan]]]))
    assert exported["input_dtype"] == "float64"


def test_export_lightgbm_refit_preserves_missing_preprocessing():
    pytest.importorskip("lightgbm")
    from bettertrees import LightGBMRefitClassifier
    rng = np.random.default_rng(902)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X[::7, 0] = np.nan
    model = LightGBMRefitClassifier(max_splits=6, min_child_samples=1).fit(X, y)
    exported = _roundtrip(model, np.vstack([X, [[np.nan, np.nan]]]))
    assert exported["nan_fill"] == model.nan_fill_.tolist()
    model.add_stump(0, 0, left_value=-0.2, right_value=0.2)
    _roundtrip(model, X)


def test_executor_rejects_unknown_schema_and_overflow():
    model = FIGSClassifier(max_splits=1, min_weight=1).fit([[0], [1]], [0, 1]).to_dict()
    with pytest.raises(ValueError, match="float32"):
        EXECUTOR["decision_function"](model, [[1e100]])
    model["schema_version"] = 2
    with pytest.raises(ValueError, match="schema-1"):
        EXECUTOR["decision_function"](model, [[0]])
