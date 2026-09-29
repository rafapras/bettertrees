"""Univariate screening: Newton gain of the best single cut and its stability."""

from __future__ import annotations

import numpy as np

from .._typing import ArrayLike, FloatArray, Seed
from ._common import (
    as_float_matrix,
    as_target,
    as_weights,
    base_margin,
    bin_threshold,
    binned,
    grad_hess,
)
from ._kernels import best_cut_1d, hist_1d


def screen_features(X: ArrayLike, y: ArrayLike, margin: ArrayLike | None = None,
                    sample_weight: ArrayLike | None = None, *, n_boot: int = 30,
                    max_bins: int = 32, lam: float = 1.0, min_weight: float = 20.0,
                    top_k: int = 5, random_state: Seed = 0) -> dict[str, FloatArray]:
    """Score each feature by its best single cut on the residual of ``margin``.

    ``margin`` is the margin (logit) of a base model; ``None`` uses the
    constant. The bootstrap uses Poisson(1) weights over the same bins.

    Returns a dict of per-feature arrays: ``gain`` and ``threshold`` on the full
    sample; ``rank_mean`` (mean bootstrap rank, 1 = best), ``top_k_freq``
    (share of replicates in the top k) and ``threshold_iqr`` (IQR of the
    threshold across replicates: an unstable threshold is a cut that does not
    hold up).
    """
    X = as_float_matrix(X)
    n, p = X.shape
    y = as_target(y, n)
    w = as_weights(sample_weight, n)
    m = np.full(n, base_margin(y, w)) if margin is None else np.asarray(margin, float)
    Xb, edges, nb = binned(X, max_bins)
    B = int(nb.max())

    def one(weights):
        g, h = grad_hess(y, m, weights)
        return best_cut_1d(hist_1d(Xb, g, h, weights, B), nb, lam, min_weight)

    gain, cut = one(w)
    rng = np.random.default_rng(random_state)
    ranks = np.empty((n_boot, p))
    thresholds = np.full((n_boot, p), np.nan)
    for r in range(n_boot):
        gb, cb = one(w * rng.poisson(1.0, n))
        ranks[r] = np.argsort(np.argsort(-gb, kind="stable"), kind="stable") + 1
        for j in range(p):
            thresholds[r, j] = bin_threshold(edges[j], cb[j])
    threshold = np.array([bin_threshold(edges[j], cut[j]) for j in range(p)])
    if n_boot == 0:  # no bootstrap: gain and threshold only
        nan = np.full(p, np.nan)
        return dict(gain=gain, threshold=threshold, rank_mean=nan,
                    top_k_freq=nan, threshold_iqr=nan)
    finite = np.where(np.isfinite(thresholds), thresholds, np.nan)
    iqr = np.full(p, np.nan)
    for j in range(p):
        col = finite[:, j][np.isfinite(finite[:, j])]
        if len(col):
            q75, q25 = np.percentile(col, [75, 25])
            iqr[j] = q75 - q25
    return dict(
        gain=gain,
        threshold=threshold,
        rank_mean=ranks.mean(axis=0),
        top_k_freq=(ranks <= top_k).mean(axis=0),
        threshold_iqr=iqr,
    )
