"""Node containers and input validation; no Numba here.

``NodeArrays`` and ``Split`` only group arrays/scalars: there is no object per
node. Validation converts X to C-contiguous float32 and drops zero-weight rows
before any binning, support count or split.
"""

from typing import NamedTuple

import numpy as np
from sklearn.utils.multiclass import check_classification_targets


class NodeArrays(NamedTuple):
    """Contiguous arrays; root 0, children -1 at leaves, feature -1 at leaves.

    Each position identifies a node. class_weight has shape (capacity, K) and
    holds per-class masses, not probabilities. No published leaf may have zero
    total mass. threshold is float64; X is float32. The builder returns the
    arrays trimmed to the nodes actually used.
    """

    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    missing_left: np.ndarray
    class_weight: np.ndarray
    n_samples: np.ndarray


class Split(NamedTuple):
    """Best cut: feature=-1 means there is no admissible cut.

    gain is the LOCAL Gini decrease; the builder weights it by the mass
    relative to the root when applying min_impurity_decrease. threshold is
    always on the original scale; bin_threshold=-1 in the exact engine. n_left
    counts active rows regardless of the magnitude of their positive weights.
    """

    feature: int
    threshold: float
    missing_left: bool
    gain: float
    n_left: int
    bin_threshold: int


def validate_X(X, *, n_features=None):
    """Convert a dense numeric matrix to C-contiguous float32, allowing NaN.

    Rejects empty, complex, infinite (including conversion overflow) and sparse
    inputs and a wrong number of columns. Does not modify the argument. Its
    cost is part of the fit/predict time of the public API.
    """
    raw = np.asarray(X)
    if raw.ndim != 2 or 0 in raw.shape or raw.dtype.kind not in "biuf":
        raise ValueError("X must be a non-empty dense numeric array.")
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.ascontiguousarray(raw, dtype=np.float32)
    if np.isinf(out).any():
        raise ValueError("X cannot contain infinity or values outside the float32 range.")
    if n_features is not None and out.shape[1] != n_features:
        raise ValueError("The number of columns differs from training.")
    return out


def prepare_training_data(X, y, sample_weight=None):
    """Validate training data and return (X32, y_int32, weights64, original_classes).

    Accepts a one-dimensional binary/multiclass target, including strings.
    classes follows np.unique; its index breaks ties at prediction. Weights
    must be finite, non-negative, aligned and have a positive sum. Zero-weight
    rows take no part in bins, minimum support or splits; classes_ keeps every
    class observed before that exclusion. The inputs are never modified and
    no holdout is used.
    """
    X = validate_X(X)
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != len(X):
        raise ValueError("y must have one label per row of X.")
    check_classification_targets(target)
    classes, encoded = np.unique(target, return_inverse=True)
    weights = (np.ones(len(X), dtype=np.float64) if sample_weight is None
               else np.asarray(sample_weight, dtype=np.float64))
    if weights.shape != (len(X),) or not np.isfinite(weights).all():
        raise ValueError("sample_weight deve ser um vetor finito alinhado a X.")
    if (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Sample weights must be non-negative with a positive sum.")
    if not np.isfinite(weights.sum()):
        raise ValueError("The sum of the weights exceeds the float64 range.")
    active = weights > 0
    if not active.all():
        X, encoded, weights = X[active], encoded[active], weights[active]
    return (np.ascontiguousarray(X), np.ascontiguousarray(encoded, dtype=np.int32),
            np.ascontiguousarray(weights), classes)


def allocate_nodes(capacity, n_classes):
    """Allocate node arrays for an UNTRAINED tree, all nodes initially leaves.

    The builder starts small, grows the arrays geometrically and fills the
    masses before publishing the tree; it avoids allocating 2**max_depth up front.
    """
    if capacity < 1 or n_classes < 1:
        raise ValueError("capacity e n_classes devem ser positivos.")
    return NodeArrays(
        np.full(capacity, -1, dtype=np.int32),
        np.full(capacity, -1, dtype=np.int32),
        np.full(capacity, -1, dtype=np.int32),
        np.full(capacity, np.nan, dtype=np.float64),
        np.zeros(capacity, dtype=np.bool_),
        np.zeros((capacity, n_classes), dtype=np.float64),
        np.zeros(capacity, dtype=np.int64),
    )
