"""Learning and applying bin edges (NaN -> bin 0).

Quantile histograms follow the approach popularized by LightGBM (Ke et al.,
NeurIPS 2017) and scikit-learn's HistGradientBoosting: at most 255 bins per
feature, missing values in a dedicated bin.

Numba kernels in this module do not call kernels from other modules: Numba's
cache is invalidated per file and does not track cross-module dependencies.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange


def _count_balanced_cut_positions(counts, max_bins):
    """Positions i (a cut between the i-th and the (i+1)-th distinct value).

    Pure quantiles waste the budget when a few values hold most of the mass:
    several quantiles fall on the same value, ``unique`` merges them, and the
    tail of distinct values gets few bins. Here, values whose count is at least
    the average mass per free bin get their own bins (iterating, since each
    one frees budget), and the rest is split by count among the runs of light
    values between them. Without ties (all counts 1) this reproduces
    equal-count bins, like quantiles.
    """
    n_values = len(counts)
    heavy = np.zeros(n_values, dtype=bool)
    while True:
        free_bins = max_bins - int(heavy.sum())
        rest = float(counts[~heavy].sum())
        if free_bins <= 0 or rest <= 0:
            break
        new = ~heavy & (counts >= rest / free_bins)
        if not new.any():
            break
        heavy |= new
    free_bins = max(1, max_bins - int(heavy.sum()))
    target = max(float(counts[~heavy].sum()) / free_bins, 1.0)
    positions = set()
    for i in np.flatnonzero(heavy):
        if i > 0:
            positions.add(int(i) - 1)
        if i < n_values - 1:
            positions.add(int(i))
    # Runs of light values between heavy values (or at the ends).
    start = 0
    while start < n_values:
        if heavy[start]:
            start += 1
            continue
        stop = start
        while stop < n_values and not heavy[stop]:
            stop += 1
        run = counts[start:stop]
        run_bins = int(round(float(run.sum()) / target))
        if run_bins > 1 and stop - start > 1:
            cumulative = np.cumsum(run)
            goals = cumulative[-1] * np.arange(1, run_bins) / run_bins
            inner = np.unique(np.searchsorted(cumulative, goals, side="left"))
            positions.update(int(start + i) for i in inner if start + i < stop - 1)
        start = stop
    positions = np.array(sorted(positions), dtype=np.int64)
    # Rounding may exceed the budget: merge the pair of neighbouring bins
    # with the smallest count until it fits.
    while len(positions) > max_bins - 1:
        bounds = np.r_[-1, positions, n_values - 1]
        cumulative = np.r_[0, np.cumsum(counts)]
        sizes = cumulative[bounds[1:] + 1] - cumulative[bounds[:-1] + 1]
        merged = sizes[:-1] + sizes[1:]
        positions = np.delete(positions, int(np.argmin(merged)))
    return positions


def _fit_bin_edges_column(col, max_bins, quantiles):
    """Learn the cuts of one column; a standalone function for parallelism."""
    finite = col[~np.isnan(col)]
    unique, counts = np.unique(finite, return_counts=True)
    unique = unique.astype(np.float64)
    # Bin 0 is reserved for NaN and uint8 holds at most ids 1..255.
    # So keep every interval only when they fit in the budget; high
    # cardinality goes through the split by count.
    if len(unique) <= max_bins:
        cuts = unique[:-1] / 2 + unique[1:] / 2
    else:
        positions = _count_balanced_cut_positions(counts, max_bins)
        cuts = unique[positions] / 2 + unique[positions + 1] / 2
    return np.ascontiguousarray(cuts, dtype=np.float64)


def fit_bin_edges(X, max_bins=255, n_jobs=1):
    """Learn the edges ONLY on the validated training data, ignoring NaN.

    Returns a tuple of increasing float64 vectors with at most max_bins-1 cuts.
    Uses midpoints when there are few unique values, which preserves indicator
    columns; otherwise unweighted quantiles without sampling. Constant and
    all-NaN columns get an empty vector. max_bins in [2, 255] fits uint8 with
    bin 0 reserved for NaN. Weights affect impurity, not the location of these
    quantiles.
    """
    if (isinstance(max_bins, (bool, np.bool_))
            or not isinstance(max_bins, (int, np.integer))
            or not 2 <= max_bins <= 255):
        raise ValueError("max_bins must be an integer between 2 and 255.")
    if (isinstance(n_jobs, (bool, np.bool_))
            or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1):
        raise ValueError("n_jobs must be a positive integer.")
    quantiles = np.arange(1, max_bins) / max_bins
    columns = tuple(X[:, j] for j in range(X.shape[1]))
    if n_jobs == 1:
        return tuple(_fit_bin_edges_column(col, max_bins, quantiles)
                     for col in columns)
    with ThreadPoolExecutor(max_workers=int(n_jobs)) as pool:
        return tuple(pool.map(
            lambda col: _fit_bin_edges_column(col, max_bins, quantiles),
            columns))


@njit(cache=True)
def _count_binary_values(values):
    """Count zeros in a 0/1 column; return -1 on any other value."""
    zeros = 0
    for value in values:
        if value == 0:
            zeros += 1
        elif value != 1:
            return -1
    return zeros


def _fit_bin_edges_binary(X, max_bins=255):
    """Learn edges with quantiles identical to the reference on 0/1 columns.

    The shortcut avoids partitioning large indicator columns. Other columns
    keep the original path; `fit_bin_edges` remains the production baseline.
    """
    if (isinstance(max_bins, (bool, np.bool_))
            or not isinstance(max_bins, (int, np.integer))
            or not 2 <= max_bins <= 255):
        raise ValueError("max_bins must be an integer between 2 and 255.")
    edges = []
    quantiles = np.arange(1, max_bins) / max_bins
    for col in X.T:
        finite = col[~np.isnan(col)]
        unique = np.unique(finite).astype(np.float64)
        if len(unique) <= max_bins:
            cuts = unique[:-1] / 2 + unique[1:] / 2
        else:
            zeros = _count_binary_values(finite)
            if zeros >= 0:
                if zeros == 0 or zeros == len(finite):
                    cuts = np.empty(0, dtype=np.float64)
                else:
                    positions = (len(finite) - 1) * quantiles
                    lower = np.floor(positions).astype(np.int64)
                    upper = np.ceil(positions).astype(np.int64)
                    cuts = np.unique(np.where(upper < zeros, 0.0,
                                              np.where(lower >= zeros, 1.0,
                                                       positions - lower)))
                    cuts = cuts[cuts < 1.0]
            else:
                finite_min = float(finite.min())
                finite_max = float(finite.max())
                cuts = np.unique(np.quantile(finite, quantiles))
                cuts = cuts[(cuts >= finite_min) & (cuts < finite_max)]
        edges.append(np.ascontiguousarray(cuts, dtype=np.float64))
    return tuple(edges)


def transform_bins(X, edges):
    """Apply frozen edges and return C-contiguous uint8 with the shape of X.

    NaN -> 0; finite values -> 1..B. A value equal to an edge goes to the lower
    bin (searchsorted side='left'), exactly like X <= threshold at prediction.
    New extremes fall into the outer bins; quantiles are never recomputed.
    Internal contract: X already validated and edges produced by fit_bin_edges.
    """
    if len(edges) != X.shape[1]:
        raise ValueError("One list of edges is required per column.")
    result = np.empty(X.shape, dtype=np.uint8)
    for j, cuts in enumerate(edges):
        result[:, j] = np.searchsorted(cuts, X[:, j], side="left") + 1
        result[np.isnan(X[:, j]), j] = 0
    return result


@njit(cache=True)
def _transform_bins_row_major_kernel(X, padded_edges, lengths):
    """Apply side='left' row by row, respecting the C layout of X and of the output."""
    result = np.empty(X.shape, dtype=np.uint8)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            value = X[i, j]
            if np.isnan(value):
                result[i, j] = 0
            else:
                lo, hi = 0, lengths[j]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if padded_edges[j, mid] < value:
                        lo = mid + 1
                    else:
                        hi = mid
                result[i, j] = lo + 1
    return result


@njit(cache=True, parallel=True)
def _transform_bins_row_major_parallel_kernel(X, padded_edges, lengths):
    """Row-major transform, parallel over rows, without changing the cuts."""
    result = np.empty(X.shape, dtype=np.uint8)
    for i in prange(X.shape[0]):
        for j in range(X.shape[1]):
            value = X[i, j]
            if np.isnan(value):
                result[i, j] = 0
            else:
                lo, hi = 0, lengths[j]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if padded_edges[j, mid] < value:
                        lo = mid + 1
                    else:
                        hi = mid
                result[i, j] = lo + 1
    return result


def transform_bins_row_major(X, edges, n_jobs=1):
    """Row-major variant; reproduces the NumPy reference bins exactly.

    Preparing the edges is part of the measured time. `transform_bins` stays
    available as a comparable baseline; both require an already validated X.
    """
    if len(edges) != X.shape[1]:
        raise ValueError("One list of edges is required per column.")
    if (isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer))
            or n_jobs < 1):
        raise ValueError("n_jobs must be a positive integer.")
    lengths = np.fromiter((len(cuts) for cuts in edges), count=len(edges), dtype=np.int64)
    padded = np.zeros((len(edges), int(lengths.max(initial=0))), dtype=np.float64)
    for j, cuts in enumerate(edges):
        padded[j, :len(cuts)] = cuts
    if n_jobs == 1:
        return _transform_bins_row_major_kernel(X, padded, lengths)
    return _transform_bins_row_major_parallel_kernel(X, padded, lengths)
