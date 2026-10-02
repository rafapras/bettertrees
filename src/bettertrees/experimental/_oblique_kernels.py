"""Numba kernels of the experimental oblique FIGS."""

import numpy as np
from numba import njit


@njit(cache=True)
def projection_hist(z, g, h, w, edges):
    """Histogram (len(edges) + 1, 3) of (g, h, w) over the bins of ``z``.

    Bin of a row = number of edges strictly below z (``np.searchsorted(edges, z,
    side="left")``), so ``z <= edges[t]`` iff bin <= t. ``edges`` sorted ascending.
    """
    ne = len(edges)
    out = np.zeros((ne + 1, 3))
    for i in range(len(z)):
        v = z[i]
        lo = 0
        hi = ne
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        out[lo, 0] += g[i]
        out[lo, 1] += h[i]
        out[lo, 2] += w[i]
    return out
