"""Contracts for experimental, locally Gini-optimal depth-2 blocks."""

from itertools import product

import numpy as np
import pytest
from tree_invariants import check_conservation, check_multilevel_blocks, check_structure

from bettertrees import FastDecisionTreeClassifier, fit_multilevel_tree


def _risk(y, weight):
    if not len(y):
        return 0.0
    mass = np.bincount(y, weights=weight, minlength=2)
    return float(mass.sum() - np.dot(mass, mass) / mass.sum())


def _partitions(X, rows, min_leaf):
    """Independent exhaustive exact candidates, including missing-only cuts."""
    for feature in range(X.shape[1]):
        values = X[rows, feature]
        missing = np.isnan(values)
        unique = np.unique(values[~missing])
        thresholds = [(float(a) + float(b)) / 2
                      for a, b in zip(unique[:-1], unique[1:])]
        if missing.any() and (~missing).any():
            thresholds.append(np.inf)
        for threshold in thresholds:
            for nan_left in ((False, True) if missing.any()
                             and np.isfinite(threshold) else (False,)):
                mask = (values <= threshold) & ~missing
                if nan_left:
                    mask |= missing
                if mask.sum() >= min_leaf and (~mask).sum() >= min_leaf:
                    yield rows[mask], rows[~mask]


def _one_level_risk(X, y, weight, rows, min_leaf):
    best = _risk(y[rows], weight[rows])
    for left, right in _partitions(X, rows, min_leaf):
        best = min(best, _risk(y[left], weight[left])
                   + _risk(y[right], weight[right]))
    return best


def _depth2_risk(X, y, weight, min_leaf):
    rows = np.arange(len(X))
    best = _risk(y, weight)
    for left, right in _partitions(X, rows, min_leaf):
        best = min(best, _one_level_risk(X, y, weight, left, min_leaf)
                   + _one_level_risk(X, y, weight, right, min_leaf))
    return best


def _model_risk(model, X, y, weight):
    leaves = model.apply(X)
    return sum(_risk(y[leaves == leaf], weight[leaves == leaf])
               for leaf in np.unique(leaves))


@pytest.mark.parametrize("root,child", list(product(("hist", "exact"), repeat=2)))
def test_all_engine_combinations_predict_and_obey_depth(root, child):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 3, dtype=float)
    y = np.array([0, 1, 1, 0] * 3)
    model = fit_multilevel_tree(
        X, y, max_depth=2, root_splitter=root, child_splitter=child,
        max_bins=2, min_samples_leaf=1, candidate_limit=10, random_state=3)
    assert model.get_depth() == 2
    assert model.get_n_leaves() == 4
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
    assert model.fit_stats_["root_candidates"] == 2


def test_exact_block_matches_independent_exhaustive_weighted_gini_with_nan():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                  [2, 0], [2, 1], [np.nan, 0], [np.nan, 1]], dtype=float)
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    weight = np.array([1, 2, 4, 1, 3, 2, 1, 5], dtype=float)
    model = fit_multilevel_tree(
        X, y, weight, max_depth=2, root_splitter="exact",
        child_splitter="exact", min_samples_leaf=1,
        candidate_limit=100, random_state=0)
    assert _model_risk(model, X, y, weight) == pytest.approx(
        _depth2_risk(X.astype(np.float32), y, weight, 1), abs=1e-10)
    assert model.get_depth() <= 2
    assert np.isfinite(model.predict_proba(X)).all()


@pytest.mark.parametrize("seed", range(5))
def test_exact_block_random_small_matches_independent_enumeration(seed):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 4, size=(11, 2)).astype(float)
    X[seed % 5, 0] = np.nan
    y = rng.randint(0, 2, size=len(X))
    weight = rng.randint(1, 5, size=len(X)).astype(float)
    model = fit_multilevel_tree(
        X, y, weight, max_depth=2, root_splitter="exact",
        child_splitter="exact", min_samples_leaf=2,
        candidate_limit=100, random_state=seed)
    assert _model_risk(model, X, y, weight) == pytest.approx(
        _depth2_risk(X.astype(np.float32), y, weight, 2), abs=1e-10)


def test_missing_only_root_in_mixed_mode():
    X = np.array([[1.], [1.], [1.], [np.nan], [np.nan], [np.nan]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = fit_multilevel_tree(
        X, y, max_depth=2, root_splitter="exact", child_splitter="hist",
        max_bins=2, min_samples_leaf=1)
    assert np.isinf(model.nodes_.threshold[0])
    np.testing.assert_array_equal(model.predict(X), y)


def test_multiclass_weights_and_leaf_smoothing():
    X = np.array([[0.], [0.], [1.], [1.], [2.], [2.]])
    y = np.array(["a", "a", "b", "b", "c", "c"])
    weight = np.array([1, 2, 1, 3, 2, 1], dtype=float)
    model = fit_multilevel_tree(
        X, y, weight, max_depth=2, root_splitter="hist",
        child_splitter="exact", max_bins=3, min_samples_leaf=1,
        leaf_smoothing=2.0, candidate_limit=10)
    proba = model.predict_proba(X)
    assert proba.shape == (6, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    np.testing.assert_array_equal(model.predict(X), y)


def test_depth4_composes_blocks_on_aligned_frontier():
    X = np.array(list(product((0, 1), repeat=4)) * 3, dtype=float)
    y = np.tile(np.random.RandomState(3).randint(0, 2, 16), 3)
    model = fit_multilevel_tree(
        X, y, max_depth=4, root_splitter="hist", child_splitter="exact",
        max_bins=2, min_samples_leaf=1, candidate_limit=20, random_state=0)
    assert model.get_depth() == 4
    assert model.fit_stats_["depth2_blocks"] == 5
    assert model.get_n_leaves() <= 16
    assert _model_risk(model, X, y, np.ones(len(y))) < _risk(y, np.ones(len(y)))


def test_candidate_limit_raises_instead_of_silently_approximating():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.arange(12) % 2
    with pytest.raises(ValueError, match="candidate_limit"):
        fit_multilevel_tree(X, y, max_depth=2, root_splitter="exact",
                            candidate_limit=1, min_samples_leaf=1)


def test_baseline_remains_separate_and_result_is_deterministic():
    rng = np.random.RandomState(17)
    X = rng.randn(80, 3)
    X[::11, 0] = np.nan
    y = (X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
    base = FastDecisionTreeClassifier(
        splitter="hist", max_depth=2, max_bins=8,
        min_samples_leaf=3, search_stopping="off", random_state=8).fit(X, y)
    model = fit_multilevel_tree(
        X, y, max_depth=2, root_splitter="hist", child_splitter="hist",
        max_bins=8, min_samples_leaf=3, candidate_limit=50, random_state=8)
    repeat = fit_multilevel_tree(
        X, y, max_depth=2, root_splitter="hist", child_splitter="hist",
        max_bins=8, min_samples_leaf=3, candidate_limit=50, random_state=8)
    np.testing.assert_array_equal(model.apply(X), repeat.apply(X))
    np.testing.assert_allclose(model.predict_proba(X), repeat.predict_proba(X))
    assert _model_risk(model, X, y, np.ones(len(y))) <= (
        _model_risk(base, X, y, np.ones(len(y))) + 1e-8)


@pytest.mark.parametrize("seed", range(3))
def test_child_histogram_reuse_preserves_tree_and_probabilities(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(240, 6)).astype(np.float32)
    X[:, 0] = rng.integers(0, 4, size=len(X))
    X[::17, 1] = np.nan
    X[:, 5] = 1.0
    y = (X[:, 0] + np.nan_to_num(X[:, 1]) * X[:, 2] > 1).astype(int)
    settings = dict(max_depth=4, max_bins=8, min_samples_leaf=5,
                    candidate_limit=64, leaf_smoothing=2.0,
                    random_state=seed)
    base = fit_multilevel_tree(X, y, **settings)
    reused = fit_multilevel_tree(X, y, reuse_child_histograms=True,
                                **settings)
    for field in ("left", "right", "feature", "threshold", "missing_left",
                  "class_weight", "n_samples"):
        np.testing.assert_array_equal(getattr(base.nodes_, field),
                                      getattr(reused.nodes_, field))
    np.testing.assert_array_equal(base.apply(X), reused.apply(X))
    np.testing.assert_array_equal(base.predict_proba(X),
                                  reused.predict_proba(X))


def test_child_histogram_reuse_rejects_nonunit_weights():
    X = np.array([[0], [1], [2], [3]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="unit sample weights"):
        fit_multilevel_tree(X, y, sample_weight=np.array([1, 2, 1, 1]),
                            max_depth=2, min_samples_leaf=1,
                            reuse_child_histograms=True)


@pytest.mark.parametrize("kwargs", [
    {"max_depth": 0}, {"max_depth": 2.0}, {"root_splitter": "foo"},
    {"child_splitter": "foo"}, {"max_bins": 256},
    {"min_samples_leaf": 0}, {"candidate_limit": 0},
    {"leaf_smoothing": -1},
])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_multilevel_tree(np.array([[0.], [1.]]), np.array([0, 1]), **kwargs)


def _block_dataset(kind, seed, n=240, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    if kind == "lowcard":
        X = np.round(X * 1.5)  # ties and equal-size children
    X[:, 3] = rng.random(n) < 0.3
    X[rng.random(X.shape) < 0.06] = np.nan
    signal = (np.nan_to_num(X[:, 0]) * np.nan_to_num(X[:, 1])
              + 0.8 * np.nan_to_num(X[:, 3]) + 0.5 * rng.normal(size=n))
    if kind == "multiclass":
        y = np.array(["a", "b", "c"])[
            np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))]
    else:
        y = (signal > 0.2).astype(np.int64)
    weights = rng.uniform(0.2, 2.0, size=n)
    weights[::17] = 0.0
    return X.astype(np.float32), y, weights


def _block_cases():
    cases = []
    for kind, (root, child), depth, weighted in product(
            ("binary", "lowcard", "multiclass"),
            product(("hist", "exact"), repeat=2), (2, 4), (False, True)):
        cases.append(pytest.param(kind, root, child, depth, weighted, False,
                                  id=f"{kind}-{root}-{child}-d{depth}"
                                     f"{'-w' if weighted else ''}"))
    for kind in ("binary", "lowcard"):
        cases.append(pytest.param(kind, "hist", "hist", 4, False, True,
                                  id=f"{kind}-hist-hist-d4-reuse"))
    return cases


@pytest.mark.parametrize("kind, root, child, depth, weighted, reuse",
                         _block_cases())
def test_every_block_is_the_depth2_optimum(kind, root, child, depth,
                                           weighted, reuse):
    X, y, weights = _block_dataset(kind, seed=depth)
    weights = weights if weighted else None
    model = fit_multilevel_tree(
        X, y, weights, max_depth=depth, root_splitter=root,
        child_splitter=child, max_bins=8, min_samples_leaf=4,
        candidate_limit=10_000, random_state=1,
        reuse_child_histograms=reuse)
    check_structure(model.nodes_, model.n_features_in_)
    check_conservation(model.nodes_)
    check_multilevel_blocks(model, X, y, weights)


@pytest.mark.parametrize("kind", ["binary", "lowcard", "multiclass"])
@pytest.mark.parametrize("depth, bins", [(2, 4), (2, 16), (4, 16), (4, 64)])
def test_block_kernel_matches_per_candidate_search(kind, depth, bins):
    # Unit weights: the joint-histogram kernel must reproduce the
    # per-candidate search (same tree and same number of candidates).
    X, y, _ = _block_dataset(kind, seed=depth + bins, n=500, p=5)
    kw = dict(max_depth=depth, max_bins=bins, min_samples_leaf=4,
              candidate_limit=10 ** 9, random_state=3)
    fast = fit_multilevel_tree(X, y, block_kernel=True, **kw)
    slow = fit_multilevel_tree(X, y, block_kernel=False, **kw)
    assert fast.fit_stats_.get("block_kernel_calls", 0) > 0
    assert "block_kernel_calls" not in slow.fit_stats_
    for field in ("left", "right", "feature", "threshold", "missing_left",
                  "class_weight", "n_samples"):
        np.testing.assert_array_equal(getattr(fast.nodes_, field),
                                      getattr(slow.nodes_, field))
    assert fast.fit_stats_["root_candidates"] == slow.fit_stats_["root_candidates"]
    check_multilevel_blocks(fast, X, y)


@pytest.mark.parametrize("depth", [3, 6])
def test_deeper_stacks_keep_every_block_optimal(depth):
    X, y, _ = _block_dataset("binary", seed=depth, n=400, p=4)
    model = fit_multilevel_tree(X, y, max_depth=depth, max_bins=8, min_samples_leaf=4,
                                candidate_limit=10 ** 9, random_state=0)
    assert model.get_depth() <= depth
    check_structure(model.nodes_, model.n_features_in_)
    check_conservation(model.nodes_)
    check_multilevel_blocks(model, X, y)
