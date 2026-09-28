"""Equivalence of the 0/1 shortcut with the preserved quantiles."""

import numpy as np
from _core import _fit_bin_edges_binary, fit_bin_edges, transform_bins


def test_binary_edges_match_reference_across_prevalence_bins_and_missing():
    for n in (1100, 10001, 500000):
        for zeros in (0, 1, 10, n // 255, n // 2, n - 10, n - 1, n):
            for max_bins in (2, 32, 255):
                for missing in (0, 11):
                    values = np.r_[np.zeros(zeros, dtype=np.float32),
                                   np.ones(n - zeros, dtype=np.float32),
                                   np.full(missing, np.nan, dtype=np.float32)]
                    X = values.reshape(-1, 1)
                    reference = fit_bin_edges(X, max_bins)
                    candidate = _fit_bin_edges_binary(X, max_bins)
                    assert np.array_equal(reference[0], candidate[0])
                    assert np.array_equal(transform_bins(X, reference),
                                          transform_bins(X, candidate))
