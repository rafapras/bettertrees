"""Shared preparation for the sums: validation, binning and base margin."""

import numpy as np

from ..bins import fit_bin_edges, transform_bins_row_major
from ._kernels import logistic_grad_hess


def as_float_matrix(X):
    """C-contiguous float64 X (the combination kernels read raw values)."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X


def as_target(y, n):
    """Binary 0/1 or soft target in [0, 1], as float64."""
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if y.shape != (n,) or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
        raise ValueError("y must be a vector in [0, 1] aligned with X.")
    return y


def as_weights(sample_weight, n):
    if sample_weight is None:
        return np.ones(n)
    w = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float64))
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("sample_weight must be finite, non-negative and aligned with X.")
    return w


def binned(X, max_bins):
    """(Xb uint8, edges, nb int64) with bin 0 = NaN and nb[j] = len(edges[j]) + 2."""
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    edges = fit_bin_edges(X32, max_bins)
    Xb = transform_bins_row_major(X32, edges)
    nb = np.array([len(e) + 2 for e in edges], dtype=np.int64)
    return Xb, edges, nb


def rebin(X, edges):
    return transform_bins_row_major(np.ascontiguousarray(X, dtype=np.float32), edges)


def bin_threshold(edges_j, t):
    """Value v of the cut ``x <= v`` equivalent to bin t (NaN always goes left)."""
    if t < 0:
        return np.nan
    if t == 0:
        return -np.inf
    return float(edges_j[t - 1])


def base_margin(y, w):
    """Logit of the weighted mean: the constant initial margin."""
    m = float(np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6))
    return np.log(m / (1 - m))


def grad_hess(y, margin, w):
    return logistic_grad_hess(y, np.ascontiguousarray(margin, dtype=np.float64), w)


def sigmoid(m):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(m, dtype=np.float64)))


def top_features(Xb, g, h, w, nb, lam, min_weight, k, probe=8):
    """The k most promising features for the depth-2/3 search (increasing index order).

    Score = max(gain of the best single cut, best FAST quadrant the feature
    forms with the ``probe`` best univariate features). The univariate gain
    alone misses pure interactions (e.g. sign(x1*x2) without main effects),
    which is exactly what depth 2 looks for. The search result is optimal
    WITHIN this set. With p <= k it returns None (full search).
    """
    from ._kernels import best_cut_1d, hist_1d, quadrant_scores
    p = Xb.shape[1]
    if k is None or p <= k:
        return None
    gains, _ = best_cut_1d(hist_1d(Xb, g, h, w, int(nb.max())), nb, lam, min_weight)
    score = gains.copy()
    top = np.argsort(-gains, kind="stable")[:probe]
    pairs = np.array([(i, j) for i in top for j in range(p) if j != i], dtype=np.int64)
    q, _, _ = quadrant_scores(Xb, g, h, w, nb, pairs, lam, min_weight)
    for (i, j), v in zip(pairs, q):
        score[i] = max(score[i], v)
        score[j] = max(score[j], v)
    return np.sort(np.argsort(-score, kind="stable")[:k])


def teacher_top_features(X, y, k, w=None, random_state=0):
    """Top-k features by the GAIN importance of a LightGBM (the classic choice).

    Runs once per fit (100 trees, 31 leaves, learning rate 0.1). Gain
    importance also accumulates splits inside interactions, so it sees features
    that only act through interactions. With p <= k it returns None. Without
    LightGBM installed (optional dependency) it falls back to FAST screening
    (``top_features``) on the gradient of the constant margin.
    """
    if k is None or X.shape[1] <= k:
        return None
    try:
        import lightgbm as lgb
    except ImportError:
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64)
        yf = np.asarray(y, dtype=np.float64)
        Xb, _, nb = binned(X, 32)
        g, h = grad_hess(yf, np.full(len(X), base_margin(yf, w)), w)
        return top_features(Xb, g, h, w, nb, 1.0, 20.0, k)
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                               importance_type="gain", n_jobs=1, random_state=random_state,
                               verbose=-1)
    model.fit(X, y, sample_weight=w)
    return np.sort(np.argsort(-model.feature_importances_, kind="stable")[:k])


_VALIDATE = dict(dtype=np.float64, order="C", ensure_all_finite="allow-nan", accept_sparse=False)


def fit_inputs(est, X, y, sample_weight=None, y_soft=None):
    """``fit`` validation through the sklearn API (sets ``n_features_in_`` and
    ``feature_names_in_``). Returns (C float64 X, classes, target in [0, 1],
    weights) and stores ``nan_features_``. ``y_soft`` replaces the 0/1 target
    with probabilities, keeping the classes of ``y``."""
    from sklearn.utils.multiclass import check_classification_targets, type_of_target
    from sklearn.utils.validation import _check_sample_weight, validate_data
    X, y = validate_data(est, X, y, reset=True, **_VALIDATE)
    check_classification_targets(y)
    y_type = type_of_target(y, input_name="y", raise_unknown=True)
    if y_type != "binary":  # sklearn's standard message for binary-only estimators
        raise ValueError("Only binary classification is supported. The type of the target "
                         f"is {y_type}.")
    classes, encoded = np.unique(y, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("Binary classification needs two classes; y has only one class.")
    target = encoded.astype(np.float64) if y_soft is None else as_target(y_soft, len(X))
    w = _check_sample_weight(sample_weight, X, dtype=np.float64, ensure_non_negative=True)
    if not w.sum() > 0:
        raise ValueError("sample_weight sums to zero.")
    est.nan_features_ = np.isnan(X).any(axis=0)
    return X, classes, target, np.ascontiguousarray(w)


def predict_input(est, X, attribute):
    """``predict`` validation: requires a fitted model and the same number/names of columns."""
    from sklearn.utils.validation import check_is_fitted, validate_data
    check_is_fitted(est, attribute)
    return validate_data(est, X, reset=False, **_VALIDATE)


def log_loss_margin(y, margin, w):
    """Weighted log-loss of a 0/1 or soft target ``y`` at the logit ``margin``.

    Probabilities are clipped to [1e-15, 1 - 1e-15]. (``autotune._log_loss`` is another
    function: it scores a probability matrix by class index.)
    """
    p = np.clip(sigmoid(margin), 1e-15, 1 - 1e-15)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def subtree_leaves(tree, node):
    """Leaves under ``node`` of a ``SmallTree`` (depth-first, right child popped first)."""
    stack, out = [node], []
    while stack:
        k = stack.pop()
        if tree.left[k] == -1:
            out.append(k)
        else:
            stack += [tree.left[k], tree.right[k]]
    return out
