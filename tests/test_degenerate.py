"""Degenerate inputs: every sum must fit, stay within budget and predict finite
probabilities (or fail with a clear error), never crash deep inside a kernel."""

import numpy as np
import pytest

from bettertrees import (
    BaggedFIGSClassifier,
    CompactTreeBooster,
    FIGSClassifier,
    RashomonFIGSClassifier,
    SumOfOptimalTrees,
)

MAKERS = {
    "figs": lambda b: FIGSClassifier(max_splits=b),
    "figs_lr": lambda b: FIGSClassifier(max_splits=b, learning_rate=0.3),
    "sum_d2": lambda b: SumOfOptimalTrees(n_trees=max(1, b // 3), depth=2),
    "bag": lambda b: BaggedFIGSClassifier(max_splits=b, n_bags=3),
    "bag_distill": lambda b: BaggedFIGSClassifier(max_splits=b, n_bags=3, vocab_size=0,
                                                  distill=True),
    "rashomon": lambda b: RashomonFIGSClassifier(max_splits=b, n_mutations=5),
    "compact_d1": lambda b: CompactTreeBooster(max_splits=b, depth=1),
    "compact_d2": lambda b: CompactTreeBooster(max_splits=b, depth=2),
    "compact_auto": lambda b: CompactTreeBooster(max_splits=b, depth="auto"),
    "compact_reuse": lambda b: CompactTreeBooster(max_splits=b, new_cut_penalty=2.0),
}


def _base(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(size=n) > 0).astype(int)
    return X, y


def _check(m, X, budget):
    p = m.predict_proba(X)[:, 1]
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert m.n_splits_ <= budget
    np.testing.assert_allclose(m.base_margin_ + m.predict_contributions(X).sum(axis=1),
                               m.decision_function(X), atol=1e-9)


CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


@case
def constant_and_all_nan_columns():
    X, y = _base()
    X[:, 3] = 7.0
    X[:, 4] = np.nan
    return X, y


@case
def rare_class():
    X, y = _base(n=2000)
    y = np.zeros(len(y), dtype=int)
    y[np.argsort(-X[:, 0])[:12]] = 1  # 0.6% positives, all with large x0
    return X, y


@case
def tiny_n():
    X, y = _base(n=30)
    return X, y


@case
def separable():
    X, y = _base()
    return X, (X[:, 0] > 0).astype(int)


@case
def duplicated_rows():
    X, y = _base(n=100)
    return np.repeat(X, 6, axis=0), np.repeat(y, 6)


@case
def string_labels():
    X, y = _base()
    return X, np.where(y == 1, "bad", "good")


@pytest.mark.parametrize("name", list(MAKERS))
@pytest.mark.parametrize("case_name", list(CASES))
@pytest.mark.parametrize("budget", [1, 64])
def test_degenerate_inputs(name, case_name, budget):
    X, y = CASES[case_name]()
    m = MAKERS[name](budget).fit(X, y)
    _check(m, X, budget if name != "sum_d2" else 3 * max(1, budget // 3))
    assert set(m.predict(X)) <= set(np.unique(y))


@pytest.mark.parametrize("name", list(MAKERS))
def test_one_class_is_a_clear_error(name):
    X, _ = _base()
    with pytest.raises(ValueError, match=r"two classes|one class"):
        MAKERS[name](8).fit(X, np.zeros(len(X), dtype=int))


def test_constant_feature_is_never_cut():
    X, y = CASES["constant_and_all_nan_columns"]()
    for make in MAKERS.values():
        m = make(16).fit(X, y)
        used = {f for t in m.trees_ for f, left in zip(t.feature, t.left) if left != -1}
        assert 3 not in used  # the constant column
