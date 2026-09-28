"""Adding cuts by hand, partial refits, monotone constraints and the usable Rashomon set."""

import numpy as np
import pandas as pd
import pytest

from bettertrees import CompactTreeBooster, FIGSClassifier, RashomonFIGSClassifier


def _data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["age", "income", "debt", "score"])
    logit = X.age + X.income * X.debt + 0.5 * np.sign(X.score)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def _consistent(m, X):
    np.testing.assert_allclose(m.base_margin_ + m.predict_contributions(X).sum(axis=1),
                               m.decision_function(X), atol=1e-9)
    m.rules()
    m.to_dict()


def test_split_leaf_and_add_stump_then_refit():
    X, y = _data()
    m = FIGSClassifier(max_splits=6).fit(X, y)
    before = m.decision_function(X).copy()
    tree = next(i for i, t in enumerate(m.trees_) if t.n_splits)
    leaf = m.trees_[tree].leaves[0]
    left, right, real = m.split_leaf(tree, leaf, "score", 0.0)
    assert m.n_splits_ == 7 and abs(real) < 0.5
    np.testing.assert_allclose(m.decision_function(X), before)  # children inherit the value
    k, _ = m.add_stump("age", 1.0, left_value=-0.1, right_value=0.1)
    assert m.trees_[k].n_splits == 1 and m.n_splits_ == 8
    m.refit_leaves(X, y)
    _consistent(m, X)
    with pytest.raises(ValueError):
        m.split_leaf(tree, 0, "age", 0.0)  # the root is a cut


def test_partial_refit_freezes_the_other_trees():
    X, y = _data()
    m = FIGSClassifier(max_splits=10).fit(X, y)
    assert len(m.trees_) >= 2
    frozen = [t.value.copy() for t in m.trees_[1:]]
    base = m.base_margin_
    for t in m.trees_:  # perturb, then refit only tree 0 without the base
        t.value = t.value + 0.3
    frozen = [v + 0.3 for v in frozen]
    m.refit_leaves(X, y, trees=[0], refit_base=False)
    for t, v in zip(m.trees_[1:], frozen):
        np.testing.assert_array_equal(t.value, v)
    assert m.base_margin_ == base
    _consistent(m, X)


def _monotone_on_grid(m, X, col, increasing=True):
    """Empirical check: sweeping ``col`` never moves the logit the wrong way."""
    rows = X.iloc[:200].copy()
    grid = np.quantile(X[col], np.linspace(0, 1, 25))
    prev = None
    for v in grid:
        rows[col] = v
        d = m.decision_function(rows)
        if prev is not None:
            step = d - prev if increasing else prev - d
            assert step.min() >= -1e-9
        prev = d


@pytest.mark.parametrize("make", [lambda: FIGSClassifier(max_splits=16),
                                  lambda: CompactTreeBooster(max_splits=16, depth="auto")])
def test_monotone_refit_and_enforce(make):
    X, y = _data()
    y = y.copy()
    m = make().fit(X, y)
    m.refit_leaves(X, y, monotone={"age": +1, "debt": -1})
    assert not m.monotone_violations("age", increasing=True)
    assert not m.monotone_violations("debt", increasing=False)
    _monotone_on_grid(m, X, "age", True)
    _monotone_on_grid(m, X, "debt", False)
    _consistent(m, X)
    m2 = make().fit(X, y)
    m2.enforce_monotone({"income": -1})
    assert not m2.monotone_violations("income", increasing=False)
    _monotone_on_grid(m2, X, "income", False)


def test_monotone_violation_is_detected():
    X, y = _data()
    m = FIGSClassifier(max_splits=8).fit(X, y)
    t, leaf = next((i, t.leaves) for i, t in enumerate(m.trees_) if t.n_splits)
    tree = m.trees_[t]
    f = int(tree.feature[0])
    name = X.columns[f]
    a, b = tree.left[0], tree.right[0]
    if tree.left[a] == -1 and tree.left[b] == -1:  # a stump-like root: force a decrease
        tree.value[a], tree.value[b] = 1.0, -1.0
        assert m.monotone_violations(name, increasing=True)


def test_rashomon_set_is_usable():
    X, y = _data()
    m = RashomonFIGSClassifier(max_splits=8, n_mutations=40, epsilon=0.05).fit(X, y)
    models = m.rashomon_models()
    assert len(models) == len(m.rashomon_members_) >= 1
    losses = [mm.validation_loss_ for mm in models]
    assert losses == sorted(losses)
    cuts = [frozenset((int(f), int(t)) for tr in mm.trees_ for f, t, lc in
                      zip(tr.feature, tr.threshold, tr.left) if lc != -1) for mm in models]
    assert len(cuts) == len(set(cuts))  # distinct structures
    for mm in models[:3]:
        p = mm.predict_proba(X)[:, 1]
        assert np.isfinite(p).all()
        _consistent(mm, X)
    refit = m.rashomon_models(X, y)[0]
    assert np.isfinite(refit.decision_function(X)).all()
    imp = m.rashomon_importance(X)
    assert set(imp) == set(X.columns)
    for lo, mean, hi in imp.values():
        assert 0 <= lo <= mean <= hi <= 1
    assert abs(sum(v[1] for v in imp.values()) - 1) < 1e-9
