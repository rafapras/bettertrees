"""Invariants over the whole flag matrix and metamorphic equivalences.

Correctness here means "the tree obeys its own contract", not "the tree is
the same as before". The metamorphic equivalences compare two runs that, by
the definition of the algorithm, must produce the same tree.
"""

import itertools

import numpy as np
import pytest
from tree_invariants import check_tree_invariants

from bettertrees import FastDecisionTreeClassifier


def _dataset(kind, seed=0, n=1200, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    if kind == "lowcard":
        # Few values and many ties: exercises tie-breaks, cuts with equal-size
        # children and bins that cover every value.
        X = np.round(X * 2.0)
    # Informative 0/1 indicator: its only cut is the last finite bin, the edge
    # candidate that continuous data almost never makes optimal.
    X[:, 5] = rng.random(n) < 0.3
    X[rng.random(X.shape) < 0.05] = np.nan
    signal = (np.nan_to_num(X[:, 0])
              + np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2])
              + 0.8 * np.nan_to_num(X[:, 5])
              + 0.6 * rng.normal(size=n))
    if kind == "multiclass":
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
        y = np.array(["a", "b", "c"])[y]
    else:
        y = (signal > 0.3).astype(np.int64)
    weights = rng.uniform(0.2, 2.0, size=n)
    weights[::13] = 0.0
    return X.astype(np.float32), y, weights


_PRECISION = dict(objective="precision", positive_class=1,
                  search_stopping="off", min_support=12.0, min_precision=0.8)


def _matrix():
    growths = [dict(max_depth=5), dict(max_leaf_nodes=16),
               dict(max_depth=4, max_leaf_nodes=9)]
    posts = [dict(), dict(ccp_alpha=2e-3), dict(leaf_smoothing=3.0),
             dict(monotonic_cst=[1, 0, 0, 0, -1, 0]),
             dict(leaf_smoothing=3.0, ccp_alpha=2e-3,
                  monotonic_cst=[1, 0, 0, 0, 0, 0])]
    cases = []
    for (kind, splitter, objective, growth, repeats, jobs, post) in itertools.product(
            ("binary", "lowcard", "multiclass"), ("hist", "exact"),
            ("gini", "precision"), growths, (None, 1, 2), (1, 3), posts):
        if splitter == "exact" and jobs > 1:
            continue
        if objective == "precision" and (kind == "multiclass" or "ccp_alpha" in post):
            continue
        if kind == "multiclass" and "monotonic_cst" in post:
            continue
        params = dict(splitter=splitter, min_samples_leaf=4, random_state=5,
                      n_jobs=jobs, max_feature_repeats=repeats, **growth, **post)
        if objective == "precision":
            params.update(_PRECISION)
        cases.append(pytest.param(kind, params, id=f"{kind}-{len(cases)}"))
    return cases


@pytest.mark.parametrize("kind, params", _matrix())
def test_every_flag_combination_respects_invariants(kind, params):
    X, y, weights = _dataset(kind)
    model = FastDecisionTreeClassifier(**params).fit(X, y, sample_weight=weights)
    check_tree_invariants(model, X, y, weights)


def _reuse_cases():
    # No depth limit: cuts with equal-size children only exercise the reuse
    # when the children still search for a cut.
    return [(kind, seed, extra)
            for kind in ("binary", "lowcard") for seed in range(4)
            for extra in (dict(), dict(max_feature_repeats=2), dict(n_jobs=3))]


@pytest.mark.parametrize("kind, seed, extra", _reuse_cases())
def test_parent_histogram_reuse_invariants_with_unit_weights(kind, seed, extra):
    X, y, _ = _dataset(kind, seed=seed)
    params = dict(max_depth=None, min_samples_leaf=3, random_state=2, **extra)
    reused = FastDecisionTreeClassifier(
        reuse_parent_histograms=True, **params).fit(X, y)
    check_tree_invariants(reused, X, y)
    baseline = FastDecisionTreeClassifier(**params).fit(X, y)
    _assert_same_tree(baseline, reused, X)


def _assert_same_tree(a, b, X):
    for left, right in zip(a.nodes_, b.nodes_):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def _canonical_partition(model, X):
    """Leaf labels renumbered by order of appearance over the rows."""
    leaves = model.apply(X)
    _, first, inverse = np.unique(leaves, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inverse]


# Metamorphic equivalences: pairs of runs the algorithm defines as equal.
# Each one covers an optimization or shortcut of the engine.

@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("growth", [dict(max_depth=7), dict(max_leaf_nodes=24)])
def test_bound_equals_exhaustive_search(seed, growth):
    X, y, w = _dataset("binary", seed)
    params = dict(min_samples_leaf=3, random_state=seed, **growth)
    off = FastDecisionTreeClassifier(search_stopping="off", **params).fit(X, y, w)
    bound = FastDecisionTreeClassifier(search_stopping="bound", **params).fit(X, y, w)
    _assert_same_tree(off, bound, X)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("extra", [dict(), dict(max_leaf_nodes=24),
                                   dict(max_feature_repeats=2), _PRECISION])
def test_parallel_equals_serial(seed, extra):
    X, y, w = _dataset("binary", seed)
    params = dict(max_depth=7, min_samples_leaf=3, random_state=seed, **extra)
    serial = FastDecisionTreeClassifier(n_jobs=1, **params).fit(X, y, w)
    parallel = FastDecisionTreeClassifier(n_jobs=3, **params).fit(X, y, w)
    _assert_same_tree(serial, parallel, X)


@pytest.mark.parametrize("splitter", ["hist", "exact"])
@pytest.mark.parametrize("extra", [dict(), _PRECISION])
def test_non_binding_repeat_limit_equals_no_limit(splitter, extra):
    X, y, w = _dataset("binary", 3)
    params = dict(splitter=splitter, max_depth=6, min_samples_leaf=3,
                  random_state=1, **extra)
    free = FastDecisionTreeClassifier(**params).fit(X, y, w)
    loose = FastDecisionTreeClassifier(max_feature_repeats=50, **params).fit(X, y, w)
    _assert_same_tree(free, loose, X)


@pytest.mark.parametrize("splitter", ["hist", "exact"])
def test_zero_weight_rows_equal_removed_rows(splitter):
    X, y, w = _dataset("binary", 4)
    keep = w > 0
    params = dict(splitter=splitter, max_depth=6, min_samples_leaf=3, random_state=0)
    weighted = FastDecisionTreeClassifier(**params).fit(X, y, w)
    dropped = FastDecisionTreeClassifier(**params).fit(X[keep], y[keep], w[keep])
    _assert_same_tree(weighted, dropped, X)


def test_integer_weight_equals_duplicated_rows_in_exact():
    # min_samples_leaf=1: the row count does not enter the decision, only the mass.
    X, y, _ = _dataset("lowcard", 5)
    counts = np.random.default_rng(5).integers(1, 4, size=len(X))
    params = dict(splitter="exact", max_depth=6, min_samples_leaf=1, random_state=0)
    weighted = FastDecisionTreeClassifier(**params).fit(X, y, counts.astype(float))
    duplicated = FastDecisionTreeClassifier(**params).fit(
        np.repeat(X, counts, axis=0), np.repeat(y, counts))
    np.testing.assert_array_equal(weighted.nodes_.feature, duplicated.nodes_.feature)
    np.testing.assert_array_equal(weighted.nodes_.threshold, duplicated.nodes_.threshold)
    np.testing.assert_allclose(weighted.predict_proba(X), duplicated.predict_proba(X),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_hist_equals_exact_partition_when_bins_cover_all_values(seed):
    # Thresholds differ by design (global edges vs node midpoint);
    # the partition of the training rows does not.
    X, y, w = _dataset("lowcard", seed)
    params = dict(max_depth=6, min_samples_leaf=3, random_state=seed,
                  search_stopping="off")
    hist = FastDecisionTreeClassifier(splitter="hist", max_bins=255, **params).fit(X, y, w)
    exact = FastDecisionTreeClassifier(splitter="exact", **params).fit(X, y, w)
    np.testing.assert_array_equal(hist.nodes_.feature, exact.nodes_.feature)
    np.testing.assert_array_equal(_canonical_partition(hist, X),
                                  _canonical_partition(exact, X))


def test_best_first_tie_goes_to_lower_node_id():
    # Mirrored halves: both root children have identical priority, so only
    # the heap's tie-break (smallest id) decides which one is expanded.
    rng = np.random.default_rng(0)
    half = rng.normal(size=(200, 1)).astype(np.float32)
    labels = (half[:, 0] > 0.3).astype(int)
    X = np.vstack([np.hstack([np.zeros((200, 1)), half]),
                   np.hstack([np.ones((200, 1)), half])]).astype(np.float32)
    y = np.concatenate([labels, 1 - labels])
    model = FastDecisionTreeClassifier(max_leaf_nodes=3, min_samples_leaf=5,
                                       random_state=0).fit(X, y)
    assert model.nodes_.feature[0] == 0
    assert model.nodes_.left[1] != -1 and model.nodes_.left[2] == -1
    check_tree_invariants(model, X, y)
