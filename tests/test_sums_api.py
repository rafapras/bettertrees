"""Public API of the interpretable sums: sklearn, rules, contributions and plots."""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
from _sklearn_checks import run_estimator_check
from sklearn.utils.estimator_checks import parametrize_with_checks

from bettertrees.lab import BaggedFIGSClassifier, RashomonFIGSClassifier
from bettertrees.sums import (
    AdditiveTreeBooster,
    BudgetClassifier,
    CompactTreeBooster,
    FIGSClassifier,
    LightGBMRefitClassifier,
    SumOfOptimalTrees,
)
from bettertrees.sums.smalltrees import BoostedOptimalTrees

matplotlib.use("Agg")

ESTIMATORS = [SumOfOptimalTrees(), FIGSClassifier(), AdditiveTreeBooster(max_rounds=20),
              BoostedOptimalTrees(depth=2, max_rounds=20),
              BaggedFIGSClassifier(max_splits=8, n_bags=4, distill=True),
              RashomonFIGSClassifier(max_splits=8, n_mutations=5),
              CompactTreeBooster(max_splits=12),
              LightGBMRefitClassifier(max_splits=12, min_child_samples=1),
              BudgetClassifier(max_splits=4), BudgetClassifier(max_splits=96)]


def _expected_failures(est):
    if isinstance(est, LightGBMRefitClassifier):
        # LightGBM uses row counts for bins/min_child_samples: a weighted row is
        # not identical to several copies even when the objective mass matches.
        return {"check_sample_weight_equivalence_on_dense_data":
                "LightGBM row-count bins and leaf support are not invariant to repetition"}
    if isinstance(est, BaggedFIGSClassifier | RashomonFIGSClassifier):
        # bootstrap replicates and the validation split both draw ROWS
        return {"check_sample_weight_equivalence_on_dense_data":
                "row-based resampling is not invariant to repetition"}
    if isinstance(est, AdditiveTreeBooster | BoostedOptimalTrees | CompactTreeBooster):
        # early stopping draws 15% of the ROWS for validation: repeating a row
        # is not the same as weight 2 (the repeated row may land on both sides)
        return {"check_sample_weight_equivalence_on_dense_data":
                "row-based internal validation is not invariant to repetition"}
    return {}


@parametrize_with_checks(ESTIMATORS, expected_failed_checks=_expected_failures)
def test_sklearn_compatible(estimator, check, monkeypatch):
    run_estimator_check(estimator, check, monkeypatch)


def _data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["age", "income", "debt", "score"])
    X.loc[rng.random(n) < 0.1, "income"] = np.nan
    logit = X.age + np.nan_to_num(X.income.to_numpy(copy=True)) * X.debt + 0.5 * np.sign(X.score)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


MODELS = [
    lambda: SumOfOptimalTrees(n_trees=3, depth=2, extra_stumps=2, learning_rate=0.5),
    lambda: SumOfOptimalTrees(n_trees=3, depth=2, search="greedy"),
    lambda: FIGSClassifier(max_splits=8),
    lambda: FIGSClassifier(max_splits=8, learning_rate=0.3),
    lambda: BaggedFIGSClassifier(max_splits=8, n_bags=5),
    lambda: RashomonFIGSClassifier(max_splits=8, n_mutations=10),
    lambda: CompactTreeBooster(max_splits=16),
    lambda: CompactTreeBooster(max_splits=16, depth=1, refit=False),
    lambda: AdditiveTreeBooster(max_rounds=10),
    lambda: BoostedOptimalTrees(depth=3, max_rounds=10),
]


@pytest.mark.parametrize("make", MODELS)
def test_contributions_reproduce_logit_and_rules_are_consistent(make):
    X, y = _data()
    m = make().fit(X, y)
    c = m.predict_contributions(X)
    np.testing.assert_allclose(m.base_margin_ + c.sum(axis=1), m.decision_function(X),
                               atol=1e-12)
    assert list(m.feature_names_in_) == list(X.columns)
    rules = m.rules()
    # one row per leaf; the leaves of each tree add up that tree's contribution
    trees = m._explain_trees()
    assert len(rules) == sum(len(t.leaves) for t in trees)
    for k in range(c.shape[1]):
        assert set(np.round(c[:, k], 12)) <= {round(v, 12) for kk, _, v in rules if kk == k}
    d = json.loads(json.dumps(m.to_dict()))
    assert d["link"] == "logit" and len(d["trees"]) == c.shape[1]
    assert m.explain().startswith("logit P(y = 1)")


def test_rules_merge_path_into_intervals_and_flag_missing():
    X, y = _data()
    m = SumOfOptimalTrees(n_trees=4, depth=2, learning_rate=0.5).fit(X, y)
    text = [c for _, conds, _ in m.rules() for c in conds]
    # never two conditions on the same feature in the same path
    for _, conds, _ in m.rules():
        feats = [next(n for n in X.columns if n in c) for c in conds]
        assert len(feats) == len(set(feats))
    # only income had NaN in training: only it may show up as "or missing"
    assert all("income" in c for c in text if "missing" in c)
    assert not m.nan_features_[0] and m.nan_features_[1]


def test_shape_functions_and_plots():
    X, y = _data()
    m = SumOfOptimalTrees(n_trees=2, depth=2, extra_stumps=3, learning_rate=0.5).fit(X, y)
    shapes = m.shape_functions()
    assert shapes  # the stumps are main effects
    for f, (vals, count) in shapes.items():
        assert len(vals) == len(m.bin_edges_[f]) + 2 and count >= 1
    # every tree is a main effect (counted in the shape) or an interaction
    assert len(m.interaction_trees()) + sum(c for _, c in shapes.values()) == len(m.trees_)
    fig = m.plot_shapes()
    assert fig.axes
    ax = m.plot_contributions(X.iloc[0].to_numpy())
    assert "P =" in ax.get_xlabel()


@pytest.mark.parametrize("make", [MODELS[2], MODELS[0]])
def test_waterfall_labels_are_the_leaf_each_row_falls_in(make):
    # FIGS appends nodes, so node-id order differs from the order rules() lists the leaves;
    # the waterfall once paired each bar with the wrong leaf's conditions
    X, y = _data()
    m = make().fit(X, y)
    rules = {(k, " and ".join(conds) or "always", round(v, 10)) for k, conds, v in m.rules(precision=4)}
    for i in range(20):
        c = m.predict_contributions(X.iloc[[i]])[0]
        ax = m.plot_contributions(X.iloc[i].to_numpy(), max_terms=len(c))
        labels = [t.get_text() for t in ax.get_yticklabels()][1:]  # skip "base"
        order = np.argsort(-np.abs(c), kind="stable")
        for label, k in zip(labels, order):
            assert (int(k), label, round(float(c[k]), 10)) in rules
        matplotlib.pyplot.close("all")


def test_predict_rejects_wrong_width_and_unfitted():
    from sklearn.exceptions import NotFittedError
    X, y = _data()
    with pytest.raises(NotFittedError):
        FIGSClassifier().predict(X)
    m = FIGSClassifier(max_splits=4).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X.iloc[:, :3])


@pytest.mark.parametrize("make", [
    lambda: SumOfOptimalTrees(n_trees=3, depth=2, extra_stumps=1),
    lambda: FIGSClassifier(max_splits=8),
    lambda: AdditiveTreeBooster(),
])
def test_shap_tree_explainer_reproduces_logit(make):
    shap = pytest.importorskip("shap")
    X, y = _data()
    X = X.to_numpy()
    m = make().fit(X, y)
    ex = shap.TreeExplainer(m.to_shap_model(), data=X[:100],
                            feature_perturbation="interventional")
    sv = ex.shap_values(X[:200])
    np.testing.assert_allclose(ex.expected_value + sv.sum(axis=1),
                               m.decision_function(X[:200]), atol=1e-5)


@pytest.mark.parametrize("make", [
    *MODELS,
    lambda: pytest.importorskip("bettertrees").LightGBMRefitClassifier(n_estimators=10),
])
def test_dataframe_in_sklearn_tools_raises_no_feature_name_warning(make):
    # LightGBMRefitClassifier re-validated its own filled array and warned on every call
    import warnings
    pytest.importorskip("lightgbm")
    X, y = _data()
    with warnings.catch_warnings():  # direct calls: sklearn's CV helpers swallow errors
        warnings.filterwarnings("error", message=".*feature names.*")
        m = make().fit(X, y)
        m.predict_proba(X)
        m.predict(X)
        m.decision_function(X)


@pytest.mark.parametrize(("budget", "kind", "settings"), [
    (4, FIGSClassifier, dict(max_delta_step=4.0, learning_rate=1.0)),
    (8, FIGSClassifier, dict(max_delta_step=4.0)),
    (16, FIGSClassifier, dict(learning_rate=0.3, max_delta_step=None)),
    (64, FIGSClassifier, dict(learning_rate=0.3)),
    (65, FIGSClassifier, dict(learning_rate=0.3)),
    (128, FIGSClassifier, dict(learning_rate=0.3, max_delta_step=None)),
])
def test_budget_classifier_applies_the_benchmark_rule_and_delegates(budget, kind, settings):
    X, y = _data()
    m = BudgetClassifier(max_splits=budget).fit(X, y)
    assert type(m.model_) is kind and m.model_.max_splits == budget
    for name, value in settings.items():
        assert getattr(m.model_, name) == value
    np.testing.assert_array_equal(m.predict_proba(X), m.model_.predict_proba(X))
    assert list(m.feature_names_in_) == list(X.columns)
    assert m.explain() == m.model_.explain() and m.rules() == m.model_.rules()
    # an edit through the wrapper changes what the wrapper predicts
    before = m.predict_proba(X)
    m.set_leaf_value(0, m.trees_[0].leaves[0], 5.0)
    assert not np.allclose(before, m.predict_proba(X))
    with pytest.raises(ValueError):
        BudgetClassifier(max_splits=0).fit(X, y)
    with pytest.raises(AttributeError):
        BudgetClassifier().explain  # not fitted: nothing to delegate to
