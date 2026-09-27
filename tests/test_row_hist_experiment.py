"""Correção da acumulação por linha frente ao splitter preservado."""

import numpy as np
import pytest

import arvore_rapida.core as core
from arvore_rapida import FastDecisionTreeClassifier


@pytest.mark.parametrize("classes", [2, 7])
@pytest.mark.parametrize("weighted", [False, True])
def test_row_hist_matches_feature_hist_with_missing_and_weights(classes, weighted):
    rng = np.random.default_rng(733 + classes + int(weighted))
    X = rng.normal(size=(3072, 13)).astype(np.float32)
    X[rng.random(X.shape) < 0.08] = np.nan
    X[:, 12] = 1.0
    y = rng.integers(0, classes, size=len(X))
    weights = rng.uniform(0.1, 3.0, size=len(X)) if weighted else None
    params = dict(max_depth=5, min_samples_leaf=20, max_bins=64,
                  random_state=42, search_stopping="bound")
    original = core.find_best_split_hist
    try:
        core.find_best_split_hist = core._find_best_split_hist_feature_major
        baseline = FastDecisionTreeClassifier(**params).fit(X, y, sample_weight=weights)
        core.find_best_split_hist = original
        candidate = FastDecisionTreeClassifier(**params).fit(X, y, sample_weight=weights)
    finally:
        core.find_best_split_hist = original
    for left, right in zip(baseline.nodes_, candidate.nodes_):
        assert np.array_equal(left, right, equal_nan=True)
    assert np.array_equal(baseline.predict_proba(X), candidate.predict_proba(X))
    assert baseline.fit_stats_["candidate_evaluated"] == candidate.fit_stats_["candidate_evaluated"]
    assert baseline.fit_stats_["candidate_skipped"] == candidate.fit_stats_["candidate_skipped"]


def test_batched_hist_scan_matches_feature_major_best_split():
    rng = np.random.default_rng(811)
    X = rng.normal(size=(768, 4)).astype(np.float32)
    X[rng.random(X.shape) < 0.06] = np.nan
    y = rng.integers(0, 3, size=len(X), dtype=np.int32)
    weights = rng.uniform(0.2, 2.0, size=len(X)).astype(np.float64)
    edges = core.fit_bin_edges(X, max_bins=32)
    X_binned = core.transform_bins_row_major(X, edges)
    indices = np.arange(len(X), dtype=np.int64)
    parent_mass = np.bincount(y, weights=weights, minlength=3).astype(np.float64)
    feature_order = np.array([2, 0, 3, 1], dtype=np.int64)

    candidate = core.find_best_split_hist(
        X_binned, y, weights, indices, 0, len(X), edges, 3, 12,
        feature_order, search_stopping="bound", parent_mass=parent_mass)
    reference = core._find_best_split_hist_feature_major(
        X_binned, y, weights, indices, 0, len(X), edges, 3, 12,
        feature_order, search_stopping="bound", parent_mass=parent_mass)

    assert candidate == reference
