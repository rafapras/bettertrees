"""Tests of contracts and primitives; they do not attest training quality or speed."""

from concurrent.futures import ThreadPoolExecutor

import _core as core
import numpy as np
import pandas as pd
import pytest
from _core import (
    _node_class_mass,
    allocate_nodes,
    build_feature_histogram,
    build_feature_histogram_into,
    cannot_improve,
    fit_bin_edges,
    gini,
    partition_samples,
    predict_proba_nodes,
    prepare_training_data,
    prune_tree_cost_complexity,
    remaining_gain_upper_bound,
    scan_histogram_feature,
    transform_bins,
    transform_bins_row_major,
    validate_X,
)
from numba import get_num_threads
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import estimator_checks

from bettertrees import FastDecisionTreeClassifier, builder, search
from bettertrees.splitters import resolve_objective, resolve_splitter_spec


def test_prepare_weights_labels_and_no_mutation():
    X = np.array([[0.0], [1.0], [np.nan], [3.0]])
    original = X.copy()
    values, labels, weights, classes = prepare_training_data(X, ["b", "a", "b", "z"], [1, 2, 3, 0])
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(classes, ["a", "b", "z"])
    np.testing.assert_array_equal(labels, [1, 0, 1])
    np.testing.assert_array_equal(weights, [1, 2, 3])
    assert values.dtype == np.float32 and values.flags.c_contiguous


def test_splitter_objective_contract_resolves_once_without_fallback():
    spec = resolve_splitter_spec("hist", "gini", "bound")
    assert spec is resolve_objective("gini")
    assert spec.supports_admissible_bound
    precision = resolve_splitter_spec("hist", "precision", "off")
    assert precision is resolve_objective("precision")
    assert not precision.supports_admissible_bound
    with pytest.raises(ValueError):
        resolve_splitter_spec("hist", "precision", "bound")
    with pytest.raises(ValueError):
        resolve_splitter_spec("hist", "unknown", "off")


@pytest.mark.parametrize("X", [[], [1, 2], [[np.inf]], [[1e100]], [[1j]], [["a"]], np.empty((2, 0))])
def test_invalid_X(X):
    with pytest.raises(ValueError):
        validate_X(X)


@pytest.mark.parametrize("weight", [[0, 0], [1, -1], [1, np.nan], [1], [[1], [2]]])
def test_invalid_weights(weight):
    with pytest.raises(ValueError):
        prepare_training_data([[0], [1]], [0, 1], weight)


def test_invalid_target_and_feature_count():
    with pytest.raises(ValueError):
        prepare_training_data([[0], [1]], [0.2, 0.9])
    with pytest.raises(ValueError):
        validate_X([[1, 2]], n_features=1)


def test_bins_missing_constant_boundary_and_holdout():
    X = validate_X([[0, 4, np.nan], [1, 4, np.nan], [0, 4, np.nan]])
    edges = fit_bin_edges(X, 2)
    assert edges[0].tolist() == [0.5]
    assert len(edges[1]) == len(edges[2]) == 0
    heldout = validate_X([[0.5, 10, np.nan], [0.5001, -10, 2], [np.nan, 4, 0]])
    result = transform_bins(heldout, edges)
    np.testing.assert_array_equal(result, [[1, 1, 0], [2, 1, 1], [0, 1, 1]])
    assert result.dtype == np.uint8 and result.flags.c_contiguous


def test_many_bins_respect_raw_thresholds():
    X = np.linspace(-10, 10, 1001, dtype=np.float32).reshape(-1, 1)
    edges = fit_bin_edges(X, 16)
    bins = transform_bins(X, edges)[:, 0]
    for b, threshold in enumerate(edges[0], start=1):
        np.testing.assert_array_equal(bins <= b, X[:, 0].astype(np.float64) <= threshold)


@pytest.mark.parametrize("size", [256, 1020])
def test_max_bins_never_overflows_uint8_on_small_high_cardinality_data(size):
    X = np.r_[np.arange(size, dtype=np.float32), np.nan].reshape(-1, 1)
    edges = fit_bin_edges(X, 255)
    bins = transform_bins_row_major(X, edges)[:, 0]
    assert len(edges[0]) <= 254
    assert bins[:-1].min() >= 1
    assert bins[:-1].max() <= 255
    assert bins[-1] == 0
    np.testing.assert_array_equal(transform_bins(X, edges)[:, 0], bins)
    model = FastDecisionTreeClassifier(
        max_depth=4, max_bins=255, min_samples_leaf=5,
        random_state=0,
    ).fit(X, (X[:, 0] >= 128).astype(np.int32))
    assert model.get_n_leaves() > 1


def test_shallow_tree_allocates_only_depth_budget(monkeypatch):
    sizes = []
    original = core.allocate_nodes

    def record_capacity(capacity, n_classes):
        sizes.append(capacity)
        return original(capacity, n_classes)

    monkeypatch.setattr(builder, "allocate_nodes", record_capacity)
    X = np.arange(2000, dtype=np.float32).reshape(-1, 1)
    y = (X[:, 0] > 1000).astype(np.int32)
    model = FastDecisionTreeClassifier(
        splitter="exact", search_stopping="off", max_depth=2,
    ).fit(X, y)
    assert sizes == [7]
    assert model.get_depth() <= 2


def test_exact_reports_exhaustive_search_when_bound_requested():
    X = np.arange(32, dtype=np.float32).reshape(-1, 1)
    y = (X[:, 0] > 12).astype(np.int32)
    models = [FastDecisionTreeClassifier(
        splitter="exact", search_stopping=mode, max_depth=3,
        random_state=3,
    ).fit(X, y) for mode in ("off", "bound")]
    assert models[1].fit_stats_["search_stopping"] == "bound"
    assert models[1].fit_stats_["search_stopping_effective"] == "off"
    assert all(np.array_equal(a, b, equal_nan=True)
               for a, b in zip(models[0].nodes_, models[1].nodes_))


@pytest.mark.parametrize("n_classes", [2, 4])
def test_exact_numba_scan_matches_python_reference(n_classes):
    rng = np.random.default_rng(729 + n_classes)
    for trial in range(20):
        X = rng.integers(-3, 5, size=(80, 5)).astype(np.float32)
        if trial % 2:
            X[:, 0] = rng.normal(size=len(X)).astype(np.float32)
        X[::(3 + trial % 5), trial % X.shape[1]] = np.nan
        y = rng.integers(0, n_classes, size=len(X), dtype=np.int32)
        weights = rng.uniform(0.1, 3.0, size=len(X))
        rows = rng.permutation(len(X)).astype(np.int64)
        order = rng.permutation(X.shape[1]).astype(np.int64)
        params = (X, y, weights, rows, 5, len(rows) - 4,
                  n_classes, 1 + trial % 6, order)
        actual_stats, expected_stats = {}, {}
        actual = core.find_best_split_exact(*params, stats=actual_stats)
        expected = core._find_best_split_exact_reference(
            *params, stats=expected_stats)
        assert actual.feature == expected.feature
        assert actual.missing_left == expected.missing_left
        assert actual.n_left == expected.n_left
        np.testing.assert_allclose(actual.threshold, expected.threshold,
                                   rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(actual.gain, expected.gain,
                                   rtol=0, atol=1e-12)
        assert actual_stats == expected_stats
    assert core._find_best_split_exact_gini_numba.nopython_signatures


@pytest.mark.parametrize("n_classes", [2, 4])
def test_exact_numba_full_tree_matches_python_reference(monkeypatch, n_classes):
    rng = np.random.default_rng(832 + n_classes)
    X = rng.normal(size=(256, 5)).astype(np.float32)
    X[::13, 0] = np.nan
    X[:, 2] = rng.integers(0, 4, size=len(X)).astype(np.float32)
    y = rng.integers(0, n_classes, size=len(X), dtype=np.int32)
    weights = rng.uniform(0.25, 2.0, size=len(X))
    params = dict(splitter="exact", search_stopping="off", max_depth=4,
                  max_leaf_nodes=12, min_samples_leaf=5, random_state=7)
    candidate = FastDecisionTreeClassifier(**params).fit(
        X, y, sample_weight=weights)
    with monkeypatch.context() as context:
        context.setattr(search, "find_best_split_exact",
                        core._find_best_split_exact_reference)
        baseline = FastDecisionTreeClassifier(**params).fit(
            X, y, sample_weight=weights)
    for candidate_array, baseline_array in zip(candidate.nodes_, baseline.nodes_):
        np.testing.assert_allclose(candidate_array, baseline_array,
                                   rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(candidate.predict_proba(X),
                                  baseline.predict_proba(X))


@pytest.mark.parametrize("n_classes", [2, 4])
def test_exact_precision_numba_scan_matches_python_reference(n_classes):
    rng = np.random.default_rng(931 + n_classes)
    for trial in range(20):
        X = rng.integers(-4, 6, size=(96, 4)).astype(np.float32)
        if trial % 2:
            X[:, 0] = rng.normal(size=len(X)).astype(np.float32)
        X[::(4 + trial % 5), trial % X.shape[1]] = np.nan
        y = rng.integers(0, n_classes, size=len(X), dtype=np.int32)
        weights = rng.uniform(0.2, 2.5, size=len(X))
        rows = rng.permutation(len(X)).astype(np.int64)
        order = rng.permutation(X.shape[1]).astype(np.int64)
        params = (X, y, weights, rows, 3, len(rows) - 3,
                  n_classes, 1 + trial % 5, order, 1, 0.9,
                  1.0 + trial % 12)
        actual_stats, expected_stats = {}, {}
        actual = core.find_best_split_exact_precision(*params, stats=actual_stats)
        expected = core._find_best_split_exact_precision_reference(
            *params, stats=expected_stats)
        assert actual.feature == expected.feature
        assert actual.missing_left == expected.missing_left
        assert actual.n_left == expected.n_left
        np.testing.assert_allclose(actual.threshold, expected.threshold,
                                   rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(actual.gain, expected.gain,
                                   rtol=0, atol=1e-12)
        assert actual_stats == expected_stats
    assert core._scan_exact_feature_precision_numba.nopython_signatures


def test_exact_precision_numba_full_tree_matches_reference(monkeypatch):
    rng = np.random.default_rng(932)
    X = rng.normal(size=(256, 5)).astype(np.float32)
    X[::17, 0] = np.nan
    X[:, 2] = rng.integers(0, 5, size=len(X)).astype(np.float32)
    y = (X[:, 1] + np.nan_to_num(X[:, 0]) > 0).astype(np.int32)
    weights = rng.uniform(0.25, 2.0, size=len(X))
    params = dict(splitter="exact", objective="precision", positive_class=1,
                  min_precision=0.95, min_support=4.0,
                  search_stopping="off", max_depth=4, max_leaf_nodes=12,
                  min_samples_leaf=5, random_state=7)
    candidate = FastDecisionTreeClassifier(**params).fit(
        X, y, sample_weight=weights)
    with monkeypatch.context() as context:
        context.setattr(search, "find_best_split_exact_precision",
                        core._find_best_split_exact_precision_reference)
        baseline = FastDecisionTreeClassifier(**params).fit(
            X, y, sample_weight=weights)
    for candidate_array, baseline_array in zip(candidate.nodes_, baseline.nodes_):
        np.testing.assert_allclose(candidate_array, baseline_array,
                                   rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(candidate.predict_proba(X),
                                  baseline.predict_proba(X))


def test_row_major_bins_match_reference_with_missing_and_boundaries():
    rng = np.random.default_rng(104)
    X = rng.normal(size=(2048, 4)).astype(np.float32)
    X[::7, 0] = np.nan
    X[:, 1] = (np.arange(len(X)) % 3).astype(np.float32)
    X[:, 2] = np.nan
    X[:, 3] = 4.0
    edges = fit_bin_edges(X, 32)
    holdout = X.copy()
    holdout[0, 0] = np.float32(edges[0][0])
    holdout[1, 0] = np.nextafter(np.float32(edges[0][0]), np.float32(np.inf))
    holdout[2, 0] = np.finfo(np.float32).min
    holdout[3, 0] = np.finfo(np.float32).max
    expected = transform_bins(holdout, edges)
    actual = transform_bins_row_major(holdout, edges)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous and actual.dtype == np.uint8


def test_partition_and_prediction_share_missing_and_equality_rules():
    X = validate_X([[999], [0], [0.5], [1], [np.nan], [-999]])
    indices = np.arange(len(X), dtype=np.int64)
    mid = partition_samples(X, indices, 1, 5, 0, 0.5, True)
    assert mid == 4 and set(indices[1:mid]) == {1, 2, 4}
    assert indices[0] == 0 and indices[-1] == 5
    nodes = allocate_nodes(3, 2)
    nodes.left[0], nodes.right[0], nodes.feature[0] = 1, 2, 0
    nodes.threshold[0], nodes.missing_left[0] = 0.5, True
    nodes.class_weight[1], nodes.class_weight[2] = [3, 1], [0, 2]
    proba = predict_proba_nodes(X, nodes)
    np.testing.assert_allclose(proba[[1, 2, 4]], [[0.75, 0.25]] * 3)
    np.testing.assert_allclose(proba[[0, 3]], [[0, 1]] * 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_constant_class_and_invalid_leaf():
    nodes = allocate_nodes(1, 1)
    with pytest.raises(ValueError):
        predict_proba_nodes(validate_X([[0]]), nodes)
    nodes.class_weight[0, 0] = 3
    np.testing.assert_array_equal(predict_proba_nodes(validate_X([[0], [np.nan]]), nodes), [[1], [1]])


@pytest.mark.parametrize("splitter", ["hist", "exact"])
def test_leaf_smoothing_preserves_tree_and_uses_weighted_root_prior(splitter):
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    weights = np.array([2.0, 1.0, 1.0, 1.0])
    params = dict(splitter=splitter, max_depth=1, min_samples_leaf=1,
                  random_state=0)
    raw = FastDecisionTreeClassifier(**params).fit(X, y, sample_weight=weights)
    smooth = FastDecisionTreeClassifier(**params, leaf_smoothing=2.0).fit(
        X, y, sample_weight=weights)
    np.testing.assert_array_equal(raw.apply(X), smooth.apply(X))
    np.testing.assert_array_equal(raw.nodes_.feature, smooth.nodes_.feature)
    np.testing.assert_allclose(raw.predict_proba(X), [[1, 0], [1, 0], [0, 1], [0, 1]])
    # Root: positive class 2/5. Leaves: (0+2*2/5)/(3+2),
    # (2+2*2/5)/(2+2).
    np.testing.assert_allclose(smooth.predict_proba(X)[:, 1],
                               [0.16, 0.16, 0.7, 0.7])
    assert smooth.get_params()["leaf_smoothing"] == 2.0
    assert smooth.fit_stats_["leaf_smoothing"] == 2.0


def test_leaf_smoothing_multiclass_and_monotonic_projection():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.float32)
    multiclass = FastDecisionTreeClassifier(
        splitter="exact", max_depth=2, leaf_smoothing=3.0,
        random_state=0).fit(X, [0, 0, 1, 1, 2, 2])
    p = multiclass.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()
    monotone = FastDecisionTreeClassifier(
        splitter="exact", max_depth=2, leaf_smoothing=2.0,
        monotonic_cst=[1], random_state=0).fit(X, [0, 0, 0, 1, 1, 1])
    monotone_p = monotone.predict_proba(X)[:, 1]
    assert (np.diff(monotone_p) >= 0).all()
    assert (monotone_p > 0).all() and (monotone_p < 1).all()


@pytest.mark.parametrize("bad", [-1, np.inf, np.nan, True])
def test_leaf_smoothing_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="leaf_smoothing"):
        FastDecisionTreeClassifier(leaf_smoothing=bad).fit([[0], [1]], [0, 1])


def test_cost_complexity_pruning_uses_weighted_gini_and_compacts_nodes():
    nodes = allocate_nodes(3, 2)
    nodes.left[0], nodes.right[0], nodes.feature[0] = 1, 2, 0
    nodes.threshold[0], nodes.missing_left[0] = 0.5, True
    nodes.class_weight[0] = [4, 1]
    nodes.class_weight[1], nodes.class_weight[2] = [4, 0], [0, 1]
    nodes.n_samples[:] = [5, 4, 1]
    kept = prune_tree_cost_complexity(nodes, 0.31)
    pruned = prune_tree_cost_complexity(nodes, 0.32)
    assert len(kept.left) == 3
    assert len(pruned.left) == 1
    assert pruned.feature[0] == -1
    np.testing.assert_array_equal(pruned.class_weight[0], [4, 1])
    np.testing.assert_allclose(
        predict_proba_nodes(validate_X([[0], [1], [np.nan]]), pruned),
        [[0.8, 0.2]] * 3)
    np.testing.assert_array_equal(nodes.left, [1, -1, -1])


def test_cost_complexity_pruning_keeps_missing_route_after_partial_prune():
    nodes = allocate_nodes(5, 2)
    nodes.left[0], nodes.right[0], nodes.feature[0] = 1, 2, 0
    nodes.left[1], nodes.right[1], nodes.feature[1] = 3, 4, 1
    nodes.threshold[:2] = 0.0
    nodes.missing_left[0] = True
    nodes.class_weight[0] = [5, 7]
    nodes.class_weight[1] = [5, 2]
    nodes.class_weight[2] = [0, 5]
    nodes.class_weight[3] = [3, 1]
    nodes.class_weight[4] = [2, 1]
    pruned = prune_tree_cost_complexity(nodes, 0.01)
    assert len(pruned.left) == 3
    assert pruned.left[0] == 1 and pruned.right[0] == 2
    assert pruned.missing_left[0]
    np.testing.assert_allclose(
        predict_proba_nodes(validate_X([[np.nan, 10], [-1, -1]]), pruned)[:, 1],
        [2 / 7, 2 / 7])
    np.testing.assert_allclose(
        predict_proba_nodes(validate_X([[1, -1]]), pruned)[:, 1], [1.0])


@pytest.mark.parametrize("splitter", ["hist", "exact"])
def test_ccp_alpha_preserves_missing_route_weights_and_default(splitter):
    X = np.array([[0], [0], [1], [1], [np.nan], [np.nan]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0, 0])
    weights = np.array([2, 1, 1, 1, 1, 1], dtype=np.float64)
    params = dict(splitter=splitter, max_depth=2, min_samples_leaf=1,
                  search_stopping="off", random_state=0)
    control = FastDecisionTreeClassifier(**params).fit(X, y, sample_weight=weights)
    same = FastDecisionTreeClassifier(**params, ccp_alpha=0.0).fit(
        X, y, sample_weight=weights)
    np.testing.assert_array_equal(control.predict_proba(X), same.predict_proba(X))
    assert control.nodes_.missing_left[0] == same.nodes_.missing_left[0]
    pruned = FastDecisionTreeClassifier(**params, ccp_alpha=1.0).fit(
        X, y, sample_weight=weights)
    assert pruned.get_n_leaves() == 1
    assert len(pruned.nodes_.left) == 1
    np.testing.assert_allclose(pruned.predict_proba(X), [[5 / 7, 2 / 7]] * len(X))
    assert pruned.fit_stats_["nodes_removed_by_pruning"] > 0


@pytest.mark.parametrize("bad", [-1, np.inf, np.nan, True])
def test_ccp_alpha_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="ccp_alpha"):
        FastDecisionTreeClassifier(ccp_alpha=bad).fit([[0], [1]], [0, 1])


def test_ccp_alpha_positive_requires_gini_objective():
    with pytest.raises(ValueError, match="ccp_alpha"):
        FastDecisionTreeClassifier(
            objective="precision", positive_class=1,
            search_stopping="off", ccp_alpha=0.1).fit([[0], [1]], [0, 1])


def test_ccp_alpha_projects_monotonicity_after_pruning():
    X = np.arange(20, dtype=np.float32).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
                  0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    model = FastDecisionTreeClassifier(
        splitter="exact", max_depth=4, monotonic_cst=[1],
        ccp_alpha=0.01, leaf_smoothing=2.0,
        search_stopping="off", random_state=0).fit(X, y)
    p = model.predict_proba(np.linspace(0, 19, 101).reshape(-1, 1))[:, 1]
    assert np.min(np.diff(p)) >= -1e-12


def test_histogram_conserves_mass_and_counts():
    mass, count = build_feature_histogram(np.array([1, 0, 2, 1], dtype=np.uint8),
        np.array([0, 1, 1, 0], dtype=np.int32), np.array([1, 4, 2, 3.0]),
        np.arange(4, dtype=np.int64), 0, 4, 3, 2)
    np.testing.assert_array_equal(count, [1, 2, 1])
    np.testing.assert_allclose(mass, [[0, 4], [4, 0], [0, 2]])
    mass_buffer = np.empty((4, 2), dtype=np.float64)
    count_buffer = np.empty(4, dtype=np.int64)
    reused_mass, reused_count = build_feature_histogram_into(
        np.array([1, 0, 2, 1], dtype=np.uint8),
        np.array([0, 1, 1, 0], dtype=np.int32),
        np.array([1, 4, 2, 3.0]), np.arange(4, dtype=np.int64),
        0, 4, 3, 2, mass_buffer, count_buffer)
    np.testing.assert_array_equal(reused_count, count)
    np.testing.assert_allclose(reused_mass, mass)


def test_node_class_mass_respects_indices_weights_and_slice():
    y = np.array([1, 0, 1, 2, 0], dtype=np.int32)
    weights = np.array([0.5, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    indices = np.array([4, 2, 0, 3, 1], dtype=np.int64)

    mass = _node_class_mass(y, weights, indices, 1, 4, 3)

    np.testing.assert_allclose(mass, [0.0, 3.5, 4.0])


@pytest.mark.parametrize("classes", [2, 7])
@pytest.mark.parametrize("missing_left", [False, True])
def test_bound_dominates_every_remaining_split(classes, missing_left):
    """Compare the bound with an exhaustive enumeration of the later cuts."""
    rng = np.random.default_rng(987)
    for _ in range(20):
        hist = rng.uniform(0.01, 20, (12, classes))
        missing, finite = hist[0], hist[1:]
        parent = hist.sum(axis=0)
        for prefix in range(1, len(finite)-1):
            left_fixed = finite[:prefix].sum(axis=0) + (missing if missing_left else 0)
            right_fixed = finite[-1] + (0 if missing_left else missing)
            upper = remaining_gain_upper_bound(parent, left_fixed, right_fixed)
            for split in range(prefix, len(finite)):
                left = finite[:split].sum(axis=0) + (missing if missing_left else 0)
                right = parent - left
                gain = gini(parent) - (left.sum()*gini(left)+right.sum()*gini(right))/parent.sum()
                assert gain <= upper + 1e-12


def test_stopping_requires_bound_and_distinguishes_tolerance():
    assert not cannot_improve(0.2, -np.inf, 2)
    assert not cannot_improve(0.2, 0.19, 2)
    assert not cannot_improve(0.2, 0.2, 2)
    assert cannot_improve(0.19, 0.2, 2)
    assert cannot_improve(0.2, 0.19, 2, 0.02)


@pytest.mark.parametrize("n_classes", [2, 7])
def test_bound_trigger_matches_exhaustive_scan_when_tolerance_is_zero(n_classes):
    """The trigger only discards cuts unable to beat the incumbent."""
    rng = np.random.default_rng(4100 + n_classes)
    for trial in range(30):
        n_finite_bins = 18 + trial % 7
        mass = rng.uniform(0.01, 20.0,
                           size=(n_finite_bins + 1, n_classes)).astype(np.float64)
        count = rng.integers(1, 20, size=n_finite_bins + 1, dtype=np.int64)
        if trial % 3 == 0:
            mass[0] = 0.0
            count[0] = 0
        parent = mass.sum(axis=0)
        exhaustive = scan_histogram_feature(
            mass, count, np.empty(0, dtype=np.float64), parent,
            min_samples_leaf=3, incumbent_gain=-np.inf,
            search_stopping="off")
        bounded = scan_histogram_feature(
            mass, count, np.empty(0, dtype=np.float64), parent,
            min_samples_leaf=3, incumbent_gain=-np.inf,
            search_stopping="bound")
        assert exhaustive[0] == bounded[0]
        assert exhaustive[1] == bounded[1]
        assert exhaustive[3] == bounded[3]
        np.testing.assert_allclose(exhaustive[2], bounded[2], rtol=0, atol=1e-12)
        assert bounded[5] >= 0


@pytest.mark.parametrize("n_classes", [2, 5])
def test_numba_histogram_scan_matches_python_reference(n_classes):
    """Compare the kernel with the reference on supports, NaN and empty bins."""
    rng = np.random.default_rng(6200 + n_classes)
    for trial in range(24):
        n_bins = (2, 3, 9, 18)[trial % 4]
        counts_by_class = rng.integers(0, 6, size=(n_bins, n_classes))
        if trial % 3 == 0:
            counts_by_class[0] = 0  # no NaN in training
        if trial % 4 == 0:
            counts_by_class[2::3] = 0  # bins internos vazios
        count = counts_by_class.sum(axis=1).astype(np.int64)
        mass = counts_by_class.astype(np.float64)
        parent_mass = mass.sum(axis=0)
        if parent_mass.sum() == 0:
            continue
        kwargs = dict(min_samples_leaf=1 + trial % 4,
                      incumbent_gain=-np.inf, search_stopping="off")
        actual = scan_histogram_feature(
            mass, count, np.empty(0), parent_mass, **kwargs)
        expected = core._scan_histogram_feature_reference(
            mass, count, np.empty(0), parent_mass, **kwargs)
        assert actual[:2] == expected[:2]
        np.testing.assert_allclose(actual[2], expected[2], rtol=0, atol=1e-12)
        assert actual[3:] == expected[3:]


def test_estimator_fit_predict_and_clone_work_for_both_engines():
    model = FastDecisionTreeClassifier(max_depth=3, random_state=42)
    assert clone(model).get_params() == model.get_params()
    model.set_params(max_bins=32)
    X = [[0.0], [1.0], [2.0], [3.0], [np.nan], [4.0]]
    y = [0, 0, 0, 1, 1, 1]
    for engine in ("hist", "exact"):
        model.set_params(splitter=engine)
        model.fit(X, y)
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(X), 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert model.predict(X).shape == (len(X),)
        assert model.fit_stats_["splitter"] == engine
        assert model.fit_stats_["objective"] == "gini"
        assert model.fit_stats_["n_nodes"] >= 1


def test_parallel_hist_matches_serial_tree_and_probabilities():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(5_000, 12)).astype(np.float32)
    y = ((X[:, 0] - 0.4 * X[:, 1] + 0.2 * X[:, 2]) > 0).astype(np.int32)
    params = dict(max_depth=5, max_leaf_nodes=32, min_samples_leaf=20,
                  max_bins=64, random_state=7, search_stopping="bound")
    serial = FastDecisionTreeClassifier(n_jobs=1, **params).fit(X, y)
    parallel = FastDecisionTreeClassifier(n_jobs=2, **params).fit(X, y)
    assert all(np.array_equal(a, b, equal_nan=True)
               for a, b in zip(serial.nodes_, parallel.nodes_))
    np.testing.assert_array_equal(serial.predict_proba(X), parallel.predict_proba(X))
    assert parallel.fit_stats_["parallel"]


def test_critical_numba_kernels_compile_in_nopython_mode():
    rng = np.random.default_rng(313)
    X = rng.normal(size=(512, 4)).astype(np.float32)
    X[::23, 0] = np.nan
    y = ((X[:, 1] + X[:, 2]) > 0).astype(np.int32)
    FastDecisionTreeClassifier(
        max_depth=3, min_samples_leaf=10, max_bins=16,
        random_state=4, n_jobs=1,
    ).fit(X, y).predict_proba(X)
    FastDecisionTreeClassifier(
        max_depth=3, min_samples_leaf=10, max_bins=16,
        random_state=4, n_jobs=2,
    ).fit(X, y)
    kernels = (
        core._transform_bins_row_major_kernel,
        core._transform_bins_row_major_parallel_kernel,
        core._build_all_histograms_row_major,
        core._build_all_histograms_feature_parallel,
        core._scan_histograms_row_major_numba,
        core._scan_histograms_row_major_parallel_numba,
        core.partition_samples,
        core.apply_nodes,
    )
    assert all(kernel.nopython_signatures for kernel in kernels)


def test_hist_hot_path_batches_candidates_without_python_per_cut(monkeypatch):
    """Scanning many cuts crosses Python -> Numba once per node."""
    rng = np.random.default_rng(318)
    X = rng.normal(size=(768, 12)).astype(np.float32)
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(np.int32)
    params = dict(splitter="hist", objective="gini", search_stopping="off",
                  max_depth=1, min_samples_leaf=5, max_bins=64,
                  random_state=9, n_jobs=1)
    # Warm the relevant specializations before observing Python entry points.
    FastDecisionTreeClassifier(**params).fit(X, y)
    batch_kernel = core._scan_histograms_row_major_numba
    gini_kernel = core.gini
    calls = {"batch": 0, "gini": 0}

    def observed_batch(*args):
        calls["batch"] += 1
        return batch_kernel(*args)

    def observed_gini(*args):
        calls["gini"] += 1
        return gini_kernel(*args)

    def unexpected_scalar_scan(*args, **kwargs):
        pytest.fail("The scalar scan went back to the per-feature Python path.")

    monkeypatch.setattr(search, "_scan_histograms_row_major_numba", observed_batch)
    monkeypatch.setattr(search, "scan_histogram_feature", unexpected_scalar_scan)
    monkeypatch.setattr(search, "_scan_histogram_feature_numba", unexpected_scalar_scan)
    monkeypatch.setattr(search, "gini", observed_gini)
    model = FastDecisionTreeClassifier(**params).fit(X, y)

    assert model.fit_stats_["hist_candidates_evaluated"] > 100
    assert model.fit_stats_["nodes_split"] == 1
    assert calls["batch"] == 1
    # Gini may be called by Python orchestration, never once per candidate.
    assert calls["gini"] < 10


@pytest.mark.parametrize("objective", ["gini", "precision"])
def test_exact_hot_path_calls_numba_without_python_per_candidate(monkeypatch, objective):
    rng = np.random.default_rng(1219)
    X = rng.normal(size=(512, 8)).astype(np.float32)
    X[::17, 0] = np.nan
    y = (np.nan_to_num(X[:, 0]) + X[:, 1] > 0).astype(np.int32)
    params = dict(splitter="exact", objective=objective, search_stopping="off",
                  max_depth=1, min_samples_leaf=5, random_state=3)
    if objective == "precision":
        params.update(positive_class=1, min_precision=0.95, min_support=5.0)
        kernel_name = "_scan_exact_feature_precision_numba"
    else:
        # Gini: a single kernel per node gathers, sorts and scans every feature.
        kernel_name = "_find_best_split_exact_gini_numba"
    FastDecisionTreeClassifier(**params).fit(X, y)
    kernel = getattr(core, kernel_name)
    calls = []

    def observed_kernel(*args):
        calls.append(len(args[0]))
        return kernel(*args)

    monkeypatch.setattr(search, kernel_name, observed_kernel)
    model = FastDecisionTreeClassifier(**params).fit(X, y)
    assert model.fit_stats_["exact_candidates_evaluated"] > 100
    assert model.fit_stats_["nodes_split"] == 1
    if objective == "gini":
        assert calls == [len(X)]  # one call, at the root node
    else:
        assert len(calls) == X.shape[1]
        assert all(count > 1 for count in calls)
    assert kernel.nopython_signatures


def test_concurrent_fits_restore_numba_thread_setting():
    rng = np.random.default_rng(314)
    X = rng.normal(size=(1024, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    params = dict(max_depth=3, min_samples_leaf=10, max_bins=32,
                  random_state=4)
    # Compile both paths before starting concurrent work.
    for jobs in (1, 2):
        FastDecisionTreeClassifier(n_jobs=jobs, **params).fit(X, y)

    def fit_with_thread_count(jobs):
        before = get_num_threads()
        model = FastDecisionTreeClassifier(n_jobs=jobs, **params).fit(X, y)
        return before, get_num_threads(), model.nodes_, model.predict_proba(X)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(fit_with_thread_count, jobs)
                   for jobs in (1, 2)]
        results = [future.result() for future in futures]
    for before, after, _, _ in results:
        assert after == before
    assert all(np.array_equal(a, b, equal_nan=True)
               for a, b in zip(results[0][2], results[1][2]))
    np.testing.assert_array_equal(results[0][3], results[1][3])


def test_parent_hist_reuse_matches_depth_first_tree_and_probabilities():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(6_000, 10)).astype(np.float32)
    X[::19, 3] = np.nan
    y = rng.integers(0, 4, size=len(X), dtype=np.int32)
    params = dict(max_depth=6, max_leaf_nodes=None, min_samples_leaf=20,
                  max_bins=64, random_state=11, search_stopping="bound",
                  n_jobs=2)
    baseline = FastDecisionTreeClassifier(**params).fit(X, y)
    reused = FastDecisionTreeClassifier(
        reuse_parent_histograms=True, **params).fit(X, y)
    assert all(np.array_equal(a, b, equal_nan=True)
               for a, b in zip(baseline.nodes_, reused.nodes_))
    np.testing.assert_array_equal(baseline.predict_proba(X),
                                  reused.predict_proba(X))
    assert (reused.fit_stats_["histogram_reuse_nodes"]
            + reused.fit_stats_["histogram_reuse_terminal_saves"]
            == reused.fit_stats_["nodes_split"])


def test_parent_hist_reuse_handles_equal_size_children():
    # Regression: with equal-size children, the right one received the left
    # child's histogram. This data produces at least one 34/34 cut.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 8))
    X[rng.random(X.shape) < 0.05] = np.nan
    signal = (np.nan_to_num(X[:, 0])
              + np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2])
              + 0.5 * rng.normal(size=len(X)))
    X = X.astype(np.float32)
    y = (signal > 0.3).astype(int)
    params = dict(max_depth=6, min_samples_leaf=5, random_state=0)
    baseline = FastDecisionTreeClassifier(**params).fit(X, y)
    reused = FastDecisionTreeClassifier(
        reuse_parent_histograms=True, **params).fit(X, y)
    nodes = baseline.nodes_
    split_ids = np.flatnonzero(nodes.left >= 0)
    assert np.any(nodes.n_samples[nodes.left[split_ids]]
                  == nodes.n_samples[nodes.right[split_ids]])
    assert all(np.array_equal(a, b, equal_nan=True)
               for a, b in zip(baseline.nodes_, reused.nodes_))
    np.testing.assert_array_equal(baseline.predict_proba(X),
                                  reused.predict_proba(X))


def test_hist_depth_one_honors_requested_max_bins():
    X = np.linspace(-10, 10, 2048, dtype=np.float32).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(np.int32)
    model = FastDecisionTreeClassifier(
        splitter="hist", max_depth=1, max_bins=255, random_state=42
    ).fit(X, y)

    assert model.fit_stats_["max_bins_requested"] == 255
    assert model.fit_stats_["max_bins_effective"] == 255
    assert len(model.bin_edges_[0]) > 32


def test_unfitted_estimator_still_rejects_prediction():
    with pytest.raises(NotFittedError):
        FastDecisionTreeClassifier().predict([[0]])


def test_sklearn_tree_facade_apply_depth_leaves_and_log_proba():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    model = FastDecisionTreeClassifier(
        splitter="exact", max_depth=2, search_stopping="off",
        random_state=0,
    ).fit(X, y)

    leaves = model.apply(X)
    assert leaves.shape == (len(X),)
    assert leaves.dtype == np.intp
    np.testing.assert_array_equal(
        leaves,
        model.apply(np.asarray(X, dtype=np.float64)),
    )
    assert model.get_depth() == 1
    assert model.get_n_leaves() == 2
    log_proba = model.predict_log_proba(X)
    with np.errstate(divide="ignore"):
        np.testing.assert_array_equal(log_proba, np.log(model.predict_proba(X)))
    assert np.isneginf(log_proba).any()


def test_feature_importances_match_weighted_gini_and_zero_for_leaf_tree():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    model = FastDecisionTreeClassifier(
        splitter="exact", max_depth=1, search_stopping="off",
        random_state=0,
    ).fit(X, y, sample_weight=weights)
    reference = DecisionTreeClassifier(
        criterion="gini", max_depth=1, random_state=0,
    ).fit(X, y, sample_weight=weights)

    np.testing.assert_allclose(
        model.feature_importances_, reference.feature_importances_,
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(model.feature_importances_.sum(), 1.0)

    leaf = FastDecisionTreeClassifier(splitter="exact").fit(X, [1, 1, 1, 1])
    np.testing.assert_array_equal(leaf.feature_importances_, [0.0, 0.0])
    with pytest.raises(NotFittedError):
        FastDecisionTreeClassifier().feature_importances_


@pytest.mark.parametrize("splitter", ["exact", "hist"])
def test_precision_objective_selects_eligible_positive_leaf(splitter):
    X = np.arange(6, dtype=np.float64).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = FastDecisionTreeClassifier(
        splitter=splitter, objective="precision", positive_class=1,
        min_precision=0.9, min_support=3, max_depth=1,
        max_bins=8, search_stopping="off", random_state=0,
    ).fit(X, y)

    assert model.fit_stats_["objective"] == "precision"
    assert model.get_n_leaves() == 2
    assert model.nodes_.threshold[0] == 2.5
    positive_prediction = model.predict(X) == 1
    assert positive_prediction.sum() == 3
    assert np.all(y[positive_prediction] == 1)
    assert model.predict_proba(X)[positive_prediction, 1].min() >= 0.9


def test_precision_objective_returns_leaf_when_no_child_reaches_support():
    X = np.arange(6, dtype=np.float64).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = FastDecisionTreeClassifier(
        splitter="exact", objective="precision", positive_class=1,
        min_precision=0.9, min_support=6, max_depth=1,
        search_stopping="off",
    ).fit(X, y)
    assert model.get_n_leaves() == 1
    np.testing.assert_allclose(model.predict_proba(X), [[0.5, 0.5]] * len(X))


def test_precision_support_uses_weighted_mass():
    X = np.arange(4, dtype=np.float64).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    weights = np.array([10.0, 1.0, 2.0, 2.0])
    model = FastDecisionTreeClassifier(
        splitter="exact", objective="precision", positive_class=1,
        min_precision=0.9, min_support=3.5, max_depth=1,
        search_stopping="off",
    ).fit(X, y, sample_weight=weights)
    assert model.get_n_leaves() == 2
    positive_prediction = model.predict(X) == 1
    np.testing.assert_array_equal(positive_prediction, [False, False, True, True])


def test_precision_reuses_min_impurity_decrease_as_minimum_objective_gain():
    X = np.arange(6, dtype=np.float64).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    accepted = FastDecisionTreeClassifier(
        splitter="exact", objective="precision", positive_class=1,
        min_precision=0.9, min_support=3, min_impurity_decrease=0.5,
        max_depth=1, search_stopping="off",
    ).fit(X, y)
    rejected = FastDecisionTreeClassifier(
        splitter="exact", objective="precision", positive_class=1,
        min_precision=0.9, min_support=3, min_impurity_decrease=0.51,
        max_depth=1, search_stopping="off",
    ).fit(X, y)
    assert accepted.get_n_leaves() == 2
    assert rejected.get_n_leaves() == 1


@pytest.mark.parametrize("splitter", ["exact", "hist"])
def test_precision_rejects_zero_gain_on_xor(splitter):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    model = FastDecisionTreeClassifier(
        splitter=splitter, objective="precision", positive_class=1,
        min_precision=0.9, min_support=2, min_impurity_decrease=0,
        search_stopping="off", random_state=0,
    ).fit(X, y)
    assert model.get_n_leaves() == 1


def test_precision_objective_requires_explicit_class_and_off_search():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="positive_class"):
        FastDecisionTreeClassifier(
            objective="precision", search_stopping="off",
        ).fit(X, y)
    with pytest.raises(ValueError, match="admissible bound"):
        FastDecisionTreeClassifier(
            objective="precision", positive_class=1,
        ).fit(X, y)


def test_feature_names_are_recorded_and_checked_but_arrays_remain_accepted():
    X = pd.DataFrame({"idade": [0.0, 1.0, 2.0, 3.0]})
    model = FastDecisionTreeClassifier(
        splitter="exact", max_depth=1, search_stopping="off",
    ).fit(X, [0, 0, 1, 1])

    np.testing.assert_array_equal(model.feature_names_in_, ["idade"])
    model.predict(np.asarray(X))
    with pytest.raises(ValueError, match="Feature names"):
        model.predict(pd.DataFrame({"renda": [0.0, 1.0]}))

    model.fit(np.asarray(X), [0, 0, 1, 1])
    assert not hasattr(model, "feature_names_in_")


def test_pipeline_clone_and_grid_search_smoke():
    X = np.arange(16, dtype=np.float64).reshape(-1, 1)
    y = (X[:, 0] >= 8).astype(np.int32)
    pipeline = Pipeline([
        ("tree", FastDecisionTreeClassifier(
            splitter="exact", search_stopping="off", random_state=0,
        )),
    ])
    assert clone(pipeline).get_params()["tree__max_depth"] is None
    search = GridSearchCV(
        pipeline, {"tree__max_depth": [1, 2]}, cv=2, scoring="accuracy",
    ).fit(X, y)
    assert search.best_estimator_.score(X, y) >= 0.5


@pytest.mark.parametrize("check_name", [
    "check_estimator_cloneable",
    "check_estimator_repr",
    "check_get_params_invariance",
    "check_set_params",
    "check_estimators_fit_returns_self",
    "check_estimators_unfitted",
    "check_estimators_overwrite_params",
])
def test_selected_sklearn_estimator_checks(check_name):
    estimator_checks.__dict__[check_name](
        "FastDecisionTreeClassifier",
        FastDecisionTreeClassifier(
            splitter="exact", search_stopping="off", random_state=0,
        ),
    )


@pytest.mark.parametrize("splitter", ["exact", "hist"])
def test_pure_nodes_stop_but_xor_zero_gain_nodes_can_continue(splitter):
    pure = FastDecisionTreeClassifier(splitter=splitter, max_depth=4,
                                      max_leaf_nodes=8, random_state=0).fit(
                                          [[0], [1], [2], [3]], [0, 0, 0, 0])
    assert len(pure.nodes_.left) == 1

    xor = FastDecisionTreeClassifier(splitter=splitter, max_depth=2,
                                     max_leaf_nodes=4, random_state=0,
                                     search_stopping="off").fit(
                                         [[0, 0], [0, 1], [1, 0], [1, 1]],
                                         [0, 1, 1, 0])
    np.testing.assert_array_equal(xor.predict([[0, 0], [0, 1], [1, 0], [1, 1]]),
                                  [0, 1, 1, 0])


def _max_feature_repetition_per_path(model):
    """Count the largest repetition of a feature on any path."""
    pending = [(0, {})]
    maximum = 0
    while pending:
        node_id, counts = pending.pop()
        feature = int(model.nodes_.feature[node_id])
        if feature < 0:
            maximum = max(maximum, max(counts.values(), default=0))
            continue
        child_counts = dict(counts)
        child_counts[feature] = child_counts.get(feature, 0) + 1
        pending.append((int(model.nodes_.left[node_id]), child_counts))
        pending.append((int(model.nodes_.right[node_id]), child_counts))
    return maximum


@pytest.mark.parametrize("splitter", ["exact", "hist"])
def test_max_feature_repeats_is_per_path_and_two_represents_interval(splitter):
    X = np.arange(16, dtype=np.float64).reshape(-1, 1)
    y = ((X[:, 0] >= 4) & (X[:, 0] < 12)).astype(np.int32)
    params = dict(splitter=splitter, max_depth=3, min_samples_leaf=1,
                  max_bins=16, search_stopping="off", random_state=0)

    unrestricted = FastDecisionTreeClassifier(**params).fit(X, y)
    once = FastDecisionTreeClassifier(**params, max_feature_repeats=1).fit(X, y)
    twice = FastDecisionTreeClassifier(**params, max_feature_repeats=2).fit(X, y)

    assert _max_feature_repetition_per_path(unrestricted) == 2
    assert _max_feature_repetition_per_path(once) == 1
    assert _max_feature_repetition_per_path(twice) == 2
    assert twice.get_n_leaves() == unrestricted.get_n_leaves()
    assert np.mean(twice.predict(X) == y) > np.mean(once.predict(X) == y)
    assert once.fit_stats_["features_skipped_by_path"] > 0
    if splitter == "hist":
        assert once.fit_stats_["histograms"] < unrestricted.fit_stats_["histograms"]


@pytest.mark.parametrize("direction", [1, -1])
def test_monotonic_cst_is_global_over_finite_leaf_regions(direction):
    X = np.arange(20, dtype=np.float64).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
                  0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    model = FastDecisionTreeClassifier(
        splitter="exact", search_stopping="off", max_depth=4,
        random_state=0, monotonic_cst=[direction],
    ).fit(X, y)
    grid = np.linspace(0, 19, 101).reshape(-1, 1)
    positive_probability = model.predict_proba(grid)[:, 1]
    changes = np.diff(positive_probability)
    if direction > 0:
        assert changes.min() >= -1e-12
    else:
        assert changes.max() <= 1e-12

    assert model.monotonic_cst_.tolist() == [direction]
    assert model.fit_stats_["monotonic_cst"] == [direction]


def test_monotonic_2d_touching_boxes_do_not_create_cycle():
    # Four quadrants: boxes on opposite sides touch on another feature,
    # but share no finite point because of > in the right child.
    nodes = allocate_nodes(7, 2)
    nodes.left[0], nodes.right[0], nodes.feature[0] = 1, 2, 0
    nodes.left[1], nodes.right[1], nodes.feature[1] = 3, 4, 1
    nodes.left[2], nodes.right[2], nodes.feature[2] = 5, 6, 1
    nodes.threshold[:3] = 0.0
    nodes.class_weight[0] = [8, 8]
    nodes.class_weight[3:] = [[3, 1], [2, 2], [2, 2], [1, 3]]
    estimator = FastDecisionTreeClassifier(monotonic_cst=[1, -1])
    projected = estimator._project_monotonic_leaf_probabilities(
        nodes, np.array([1, -1]), 1)
    for fixed_y in (-1.0, 0.0, 1.0):
        grid = validate_X([[-1, fixed_y], [0, fixed_y], [1, fixed_y]])
        p = predict_proba_nodes(grid, nodes, projected)[:, 1]
        assert np.min(np.diff(p)) >= -1e-12
    for fixed_x in (-1.0, 0.0, 1.0):
        grid = validate_X([[fixed_x, -1], [fixed_x, 0], [fixed_x, 1]])
        p = predict_proba_nodes(grid, nodes, projected)[:, 1]
        assert np.max(np.diff(p)) <= 1e-12


@pytest.mark.parametrize("splitter", ["hist", "exact"])
def test_monotonic_2d_random_fits_respect_finite_global_order(splitter):
    for seed in range(10):
        rng = np.random.default_rng(seed + 629)
        X = rng.normal(size=(72, 2)).astype(np.float32)
        y = rng.integers(0, 2, size=72)
        model = FastDecisionTreeClassifier(
            splitter=splitter, max_depth=4, max_leaf_nodes=16,
            min_samples_leaf=3, monotonic_cst=[1, -1],
            random_state=seed).fit(X, y)
        axis = np.linspace(-3, 3, 25)
        for fixed in (-2.0, 0.0, 2.0):
            increase = np.column_stack((axis, np.full(len(axis), fixed)))
            decrease = np.column_stack((np.full(len(axis), fixed), axis))
            assert np.min(np.diff(model.predict_proba(increase)[:, 1])) >= -1e-12
            assert np.max(np.diff(model.predict_proba(decrease)[:, 1])) <= 1e-12


def test_monotonic_cst_contract_rejects_multiclass_and_bad_directions():
    with pytest.raises(ValueError, match="binary classification"):
        FastDecisionTreeClassifier(
            splitter="exact", search_stopping="off", monotonic_cst=[1],
        ).fit([[0], [1], [2]], [0, 1, 2])
    with pytest.raises(ValueError, match="-1, 0 and 1"):
        FastDecisionTreeClassifier(
            splitter="exact", search_stopping="off", monotonic_cst=[2],
        ).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize("params", [dict(max_bins=256), dict(max_depth=0),
    dict(min_samples_leaf=0), dict(max_leaf_nodes=1), dict(splitter="best"),
    dict(min_impurity_decrease=-1), dict(gain_tolerance=-1),
    dict(objective="precision"), dict(random_state=True), dict(n_jobs=0),
    dict(max_feature_repeats=0), dict(max_feature_repeats=1.5)])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        FastDecisionTreeClassifier(**params).fit([[0], [1]], [0, 1])


def test_n_jobs_above_available_cores_is_clamped():
    from numba import config
    X = np.random.default_rng(0).normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(int)
    model = FastDecisionTreeClassifier(n_jobs=config.NUMBA_NUM_THREADS + 7, max_depth=3)
    reference = FastDecisionTreeClassifier(n_jobs=1, max_depth=3).fit(X, y)
    np.testing.assert_array_equal(model.fit(X, y).predict_proba(X), reference.predict_proba(X))
