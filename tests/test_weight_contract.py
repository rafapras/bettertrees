"""Inactive rows and conversion overflow must not silently change a fit."""

import numpy as np
import pytest

from bettertrees import BudgetClassifier, FIGSClassifier, SumOfOptimalTrees


@pytest.mark.parametrize("make", [
    lambda: FIGSClassifier(max_splits=1, max_bins=2, min_weight=1, backfit_sweeps=0),
    lambda: SumOfOptimalTrees(n_trees=1, max_bins=2, min_weight=1),
    lambda: BudgetClassifier(max_splits=4),
])
def test_zero_weight_rows_do_not_enter_binning(make):
    X = np.arange(100.).reshape(-1, 1)
    y = (X[:, 0] >= 50).astype(int)
    baseline = make().fit(X, y)
    expanded = np.vstack([X, np.full((1000, 1), -1000.)])
    target = np.r_[y, np.zeros(1000, dtype=int)]
    weights = np.r_[np.ones(100), np.zeros(1000)]
    actual = make().fit(expanded, target, sample_weight=weights)
    np.testing.assert_array_equal(actual.bin_edges_[0], baseline.bin_edges_[0])
    np.testing.assert_array_equal(actual.predict_proba(X), baseline.predict_proba(X))


def test_weight_sum_overflow_is_rejected():
    X = np.arange(100.).reshape(-1, 1)
    with pytest.raises(ValueError, match=r"sum.*float64"):
        FIGSClassifier().fit(X, (X[:, 0] >= 50).astype(int),
                              sample_weight=np.full(100, 1e307))


def test_float32_conversion_overflow_is_rejected_on_fit_and_predict():
    X = np.arange(100.).reshape(-1, 1)
    y = (X[:, 0] >= 50).astype(int)
    with pytest.raises(ValueError, match="float32 range"):
        FIGSClassifier().fit(X * 1e38, y)
    model = FIGSClassifier().fit(X, y)
    with pytest.raises(ValueError, match="float32 range"):
        model.predict_proba([[1e100]])
