"""Shared vocabulary of combinations (ratios/differences) chosen by a shape test.

No node estimates weights: the combinations enter before the tree as new
columns, chosen by a per-pair shape test. For the pair (i, j):

- ``gain_axis``: best single cut on x_i or on x_j (2 leaves);
- ``gain_quad``: best quadrant on (x_i, x_j) (4 cells, an "AND" interaction);
- ``gain_comb``: best single cut on log(x_i/x_j) or x_i - x_j (2 leaves).

``shape = gain_comb - max(gain_axis, gain_quad)`` > 0 means the diagonal
boundary explains more than one axis and more than the 4-cell "AND": the pair
has a ratio shape. Choose the vocabulary on a fold separate from the one that
evaluates the tree (selection bias).
"""

import numpy as np

from ..sums._common import as_float_matrix, as_target, as_weights, base_margin, binned, grad_hess
from ..sums._kernels import best_cut_1d, combination_scores, hist_1d, quadrant_scores
from .interactions import all_pairs

_KINDS = {"ratio": 0, "diff": 1}


def top_pairs(matrix, k):
    """The k pairs (i < j) with the largest value in a symmetric matrix (e.g. co-occurrence)."""
    iu, ju = np.triu_indices(matrix.shape[0], 1)
    order = np.argsort(-matrix[iu, ju], kind="stable")[:k]
    return np.stack([iu[order], ju[order]], axis=1).astype(np.int64)


def pair_shape_scores(X, y, margin=None, sample_weight=None, *, pairs=None,
                      kind="ratio", max_bins=32, lam=1.0, min_weight=20.0):
    """Teste de forma por par; dict de arrays alinhados a ``pairs``."""
    if kind not in _KINDS:
        raise ValueError("kind must be 'ratio' or 'diff'.")
    X = as_float_matrix(X)
    n, p = X.shape
    y = as_target(y, n)
    w = as_weights(sample_weight, n)
    m = np.full(n, base_margin(y, w)) if margin is None else np.asarray(margin, float)
    pairs = all_pairs(p) if pairs is None else np.ascontiguousarray(pairs, dtype=np.int64)
    Xb, _, nb = binned(X, max_bins)
    g, h = grad_hess(y, m, w)
    axis, _ = best_cut_1d(hist_1d(Xb, g, h, w, int(nb.max())), nb, lam, min_weight)
    quad, _, _ = quadrant_scores(Xb, g, h, w, nb, pairs, lam, min_weight)
    comb, thr, valid = combination_scores(X, g, h, w, pairs, _KINDS[kind],
                                          max_bins, lam, min_weight)
    gain_axis = np.maximum(axis[pairs[:, 0]], axis[pairs[:, 1]])
    return dict(pairs=pairs, gain_axis=gain_axis, gain_quad=quad, gain_comb=comb,
                threshold=thr, valid_frac=valid,
                shape=comb - np.maximum(gain_axis, quad))


class RatioVocabulary:
    """Select up to ``max_terms`` ratio-shaped combinations and append them to X.

    Parameters
    ----------
    kind : {'ratio', 'diff'}
        Ratio x_i/x_j (both > 0) or difference x_i - x_j (same unit).
    candidates : 'all' or array (k, 2)
        Candidate pairs; use ``top_pairs(teacher_path_pairs(...), k)`` for the
        "teacher pairs" arm.
    min_valid : float
        Minimum share of eligible rows (ratio: both positive and finite).
    min_shape : float
        Minimum ``shape`` margin; 0 = the diagonal only has to win.
    """

    def __init__(self, *, max_terms=10, kind="ratio", candidates="all",
                 min_valid=0.9, min_shape=0.0, max_bins=32, lam=1.0,
                 min_weight=20.0):
        self.max_terms = max_terms
        self.kind = kind
        self.candidates = candidates
        self.min_valid = min_valid
        self.min_shape = min_shape
        self.max_bins = max_bins
        self.lam = lam
        self.min_weight = min_weight

    def fit(self, X, y, margin=None, sample_weight=None, feature_names=None):
        X = as_float_matrix(X)
        pairs = None if isinstance(self.candidates, str) else self.candidates
        s = pair_shape_scores(X, y, margin, sample_weight, pairs=pairs,
                              kind=self.kind, max_bins=self.max_bins,
                              lam=self.lam, min_weight=self.min_weight)
        keep = (s["valid_frac"] >= self.min_valid) & (s["shape"] > self.min_shape)
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-s["shape"][idx], kind="stable")][:self.max_terms]
        self.scores_ = s
        self.pairs_ = s["pairs"][idx]
        names = (list(feature_names) if feature_names is not None
                 else [f"x{j}" for j in range(X.shape[1])])
        op = "/" if self.kind == "ratio" else "-"
        self.term_names_ = [f"{names[i]}{op}{names[j]}" for i, j in self.pairs_]
        self.n_features_in_ = X.shape[1]
        return self

    def terms(self, X):
        """Only the new columns (n, n_terms); NaN where the row is ineligible."""
        X = as_float_matrix(X)
        a = X[:, self.pairs_[:, 0]]
        b = X[:, self.pairs_[:, 1]]
        if self.kind == "diff":
            return a - b
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
        out[~((a > 0) & (b > 0))] = np.nan
        return out

    def transform(self, X):
        """The original X followed by the vocabulary columns."""
        X = as_float_matrix(X)
        return np.hstack([X, self.terms(X)])

    def fit_transform(self, X, y, margin=None, sample_weight=None, feature_names=None):
        return self.fit(X, y, margin, sample_weight, feature_names).transform(X)
