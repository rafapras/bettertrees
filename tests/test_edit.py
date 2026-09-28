"""Editing a fitted sum and its local Rashomon view stay consistent with the API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import log_loss

from bettertrees import (
    BaggedFIGSClassifier,
    CompactTreeBooster,
    FIGSClassifier,
    RashomonFIGSClassifier,
    SumOfOptimalTrees,
)

MODELS = [
    lambda: FIGSClassifier(max_splits=10),
    lambda: SumOfOptimalTrees(n_trees=3, depth=2),
    lambda: CompactTreeBooster(max_splits=12),
    lambda: CompactTreeBooster(max_splits=12, depth=1),
    lambda: BaggedFIGSClassifier(max_splits=8, n_bags=3),
    lambda: RashomonFIGSClassifier(max_splits=8, n_mutations=5),
]


def _data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["age", "income", "debt", "score"])
    X.loc[rng.random(n) < 0.1, "income"] = np.nan
    logit = X.age + np.nan_to_num(X.income) * X.debt + 0.5 * np.sign(X.score)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def _consistent(m, X):
    """Every output reads the same trees: contributions, rules and arrays agree."""
    d = m.decision_function(X)
    c = m.predict_contributions(X)
    np.testing.assert_allclose(m.base_margin_ + c.sum(axis=1), d, atol=1e-9)
    assert m.n_splits_ == sum(int((np.asarray(t["children_left"]) != -1).sum())
                              for t in m.get_trees())
    for tree in m.trees_:  # no orphan nodes after an edit
        seen, stack = set(), [0]
        while stack:
            k = stack.pop()
            seen.add(k)
            if tree.left[k] != -1:
                stack += [tree.left[k], tree.right[k]]
        assert seen == set(range(len(tree.feature)))
    m.to_dict()
    m.rules()


def _first_cut(m):
    for i, t in enumerate(m.trees_):
        for k, left in enumerate(t.left):
            if left != -1:
                return i, k
    raise AssertionError("no cut")


@pytest.mark.parametrize("make", MODELS)
def test_prune_set_cut_refit_keep_outputs_consistent(make):
    X, y = _data()
    m = make().fit(X, y)
    before = m.n_splits_
    i, k = _first_cut(m)
    real = m.set_cut(i, k, "age", 0.3)
    assert abs(real - 0.3) < 0.5 and m.trees_[i].feature[k] == 0
    _consistent(m, X)
    m.prune(i, k)
    assert m.n_splits_ < before
    _consistent(m, X)
    loss_before = log_loss(y, m.predict_proba(X)[:, 1])
    m.refit_leaves(X, y)
    assert log_loss(y, m.predict_proba(X)[:, 1]) <= loss_before + 1e-9
    _consistent(m, X)


def test_merge_duplicates_keeps_predictions():
    X, y = _data()
    m = FIGSClassifier(max_splits=6).fit(X, y)
    before = m.decision_function(X)
    m.trees_.append(_clone_tree(m.trees_[0]))
    m.trees_[-1].value = m.trees_[-1].value * 0.5
    doubled = m.decision_function(X)
    assert m.merge_duplicates() == 1
    np.testing.assert_allclose(m.decision_function(X), doubled, atol=1e-12)
    assert not np.allclose(before, doubled)
    _consistent(m, X)


def _clone_tree(t):
    from bettertrees.sums import SmallTree
    return SmallTree(list(t.feature), list(t.threshold), list(t.left), list(t.right),
                     t.value.copy())


def test_drop_tree_and_leaf_override():
    X, y = _data()
    m = FIGSClassifier(max_splits=8).fit(X, y)
    n = len(m.trees_)
    m.drop_tree(n - 1)
    assert len(m.trees_) == n - 1
    t = m.trees_[0]
    leaf = t.leaves[0]
    m.set_leaf_value(0, leaf, 3.0)
    assert m.trees_[0].value[leaf] == 3.0
    _consistent(m, X)
    with pytest.raises(ValueError):
        m.set_leaf_value(0, 0, 1.0)  # the root is a cut
    with pytest.raises(ValueError):
        m.prune(0, leaf)
    with pytest.raises(IndexError):
        m.drop_tree(99)
    with pytest.raises(KeyError):
        m.set_cut(*_first_cut(m), "salary", 1.0)


def test_edits_do_not_touch_a_clone():
    X, y = _data()
    m = FIGSClassifier(max_splits=8).fit(X, y)
    c = clone(m)
    assert not hasattr(c, "trees_")
    m.prune(*_first_cut(m))
    c.fit(X, y)
    assert c.n_splits_ == 8


def test_cut_alternatives_is_a_rashomon_view():
    X, y = _data()
    m = FIGSClassifier(max_splits=6).fit(X, y)
    before = m.decision_function(X).copy()
    i, k = _first_cut(m)
    alts = m.cut_alternatives(X, y, i, k, epsilon=0.05, top=3)
    np.testing.assert_array_equal(m.decision_function(X), before)  # model untouched
    assert sum(r["current"] for r in alts) == 1
    losses = [r["loss"] for r in alts]
    assert losses == sorted(losses)
    for r in alts:
        assert r["current"] or r["delta"] <= 0.05
        assert r["feature_name"] in X.columns


def test_shap_follows_edits():
    shap = pytest.importorskip("shap")
    X, y = _data(n=800)
    m = CompactTreeBooster(max_splits=10).fit(X, y)
    m.prune(*_first_cut(m))
    ex = shap.TreeExplainer(m.to_shap_model(), data=X.iloc[:100].to_numpy(),
                            feature_perturbation="interventional")
    sv = ex.shap_values(X.iloc[:50].to_numpy())
    np.testing.assert_allclose(ex.expected_value + sv.sum(axis=1),
                               m.decision_function(X.iloc[:50]), atol=1e-6)
