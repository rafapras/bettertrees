"""Python reference implementations, used only by tests.

No production path imports this module.
"""

import numpy as np
from numba import njit, prange

from bettertrees._data import Split
from bettertrees.kernels import (
    cannot_improve,
    gini,
    precision_split_gain,
    remaining_gain_upper_bound,
)
from bettertrees.sums._kernels import _best_cut_hist, _score


def _scan_histogram_feature_reference(mass, count, edges, parent_mass, *, min_samples_leaf,
                                       incumbent_gain, search_stopping="bound", gain_tolerance=0.0,
                                       bound_interval=1, parent_gini=None):
    """Scan the bins of one feature and optionally use the admissible bound.

    Returns (bin_threshold, missing_left, gain, n_left, evaluated, skipped).
    1. Try the missing-only cut before excluding that possibility.
    2. Scan prefixes of masses/counts; check support and both NaN directions.
    3. Update the best admissible gain; build the fixed masses of the rest.
    4. Compute the maximum of the bounds for NaN-left and NaN-right;
       cannot_improve allows stopping the whole feature. If only one direction
       fails, discard only that direction. 'off' always finishes the scan.
    Records evaluated/skipped cuts and the cost of the bounds. On tied gains
    the earlier candidate is kept; never stops just because of a run without
    improvement. 'heuristic' is rejected until it has an explicit statistical
    specification.
    """
    if search_stopping not in ("off", "bound"):
        raise ValueError("search_stopping must be 'off' or 'bound'.")
    if (isinstance(bound_interval, (bool, np.bool_))
            or not isinstance(bound_interval, (int, np.integer))
            or bound_interval < 1):
        raise ValueError("bound_interval deve ser inteiro positivo.")
    mass = np.asarray(mass, dtype=np.float64)
    count = np.asarray(count, dtype=np.int64)
    parent_mass = np.asarray(parent_mass, dtype=np.float64)
    if mass.ndim != 2 or count.ndim != 1 or mass.shape[0] != len(count):
        raise ValueError("Histogram with an inconsistent shape.")
    if mass.shape[1] != len(parent_mass):
        raise ValueError("Histogram and parent weights do not match.")
    if len(mass) == 0:
        return -1, False, -np.inf, 0, 0, 0

    n_finite_bins = len(mass) - 1
    missing_mass = mass[0].copy()
    missing_count = int(count[0])
    finite_mass = mass[1:]
    finite_count = count[1:]
    parent_total = float(parent_mass.sum())
    if parent_total <= 0:
        return -1, False, -np.inf, 0, 0, 0
    parent_impurity = (float(gini(parent_mass)) if parent_gini is None
                       else float(parent_gini))

    best_bin = -1
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated = 0
    skipped = 0

    def consider(bin_threshold, missing_left, left_mass, left_count):
        nonlocal best_bin, best_missing_left, best_gain, best_n_left, evaluated
        right_mass = parent_mass - left_mass
        right_count = int(count.sum()) - int(left_count)
        if left_count < min_samples_leaf or right_count < min_samples_leaf:
            return
        if left_mass.sum() <= 0 or right_mass.sum() <= 0:
            return
        gain = float(parent_impurity - (
            left_mass.sum() * gini(left_mass)
            + right_mass.sum() * gini(right_mass)) / parent_total)
        evaluated += 1
        # A scan is in canonical order: lower bin first, NaN right first.
        # Keeping the previous candidate on exact ties implements that rule.
        tie_break = (
            gain == best_gain
            and (best_bin < 0
                 or bin_threshold < best_bin
                 or (bin_threshold == best_bin and best_missing_left and not missing_left))
        )
        if gain > best_gain or tie_break:
            best_bin = int(bin_threshold)
            best_missing_left = bool(missing_left)
            best_gain = gain
            best_n_left = int(left_count)

    # The finite-versus-missing cut is a real candidate and must be considered
    # before a bound can fix the last finite bin on the right.
    if missing_count > 0 and n_finite_bins > 0:
        consider(n_finite_bins, False, finite_mass.sum(axis=0), int(finite_count.sum()))

    prefix_mass = np.zeros_like(parent_mass)
    prefix_count = 0
    total_rows = int(count.sum())
    for b in range(1, n_finite_bins):
        prefix_mass += finite_mass[b - 1]
        prefix_count += int(finite_count[b - 1])
        directions = (False, True) if missing_count > 0 else (False,)
        if prefix_count == 0 or prefix_count == total_rows - missing_count:
            # Empty or complete finite prefix in the node repeats the NaN-only cut.
            directions = ()
        for missing_left in directions:
            left_mass = prefix_mass + (missing_mass if missing_left else 0.0)
            left_count = prefix_count + (missing_count if missing_left else 0)
            if missing_count == 0:
                # NaN observed only at prediction time goes to the larger
                # child; ties go right. It has no effect on training gain.
                missing_left = bool(left_count > total_rows - left_count)
            consider(b, missing_left, left_mass, left_count)

        check_bound = (b % int(bound_interval) == 0
                       or b == n_finite_bins - 2)
        if search_stopping == "bound" and check_bound and b < n_finite_bins - 1:
            # For every later finite boundary, the current prefix is fixed on
            # the left and the last finite bin is fixed on the right. Compute
            # one safe bound per still possible NaN direction.
            target = max(float(incumbent_gain), best_gain)
            if np.isfinite(target):
                bounds = []
                for missing_left in ((False, True) if missing_count > 0 else (False,)):
                    fixed_left = prefix_mass + (missing_mass if missing_left else 0.0)
                    fixed_right = finite_mass[-1] + (0.0 if missing_left else missing_mass)
                    bounds.append(float(remaining_gain_upper_bound(
                        parent_mass, fixed_left, fixed_right, parent_impurity)))
                if bounds and all(cannot_improve(bound, target, len(parent_mass), gain_tolerance)
                                  for bound in bounds):
                    remaining_boundaries = n_finite_bins - 1 - b
                    direction_count = 2 if missing_count > 0 else 1
                    skipped += remaining_boundaries * direction_count
                    break

    return (best_bin, best_missing_left, best_gain, best_n_left,
            int(evaluated), int(skipped))


def _find_best_split_exact_reference(X, y, weights, sample_indices, start, end,
                                     n_classes, min_samples_leaf, feature_order,
                                     search_stopping="off", gain_tolerance=0.0,
                                     stats=None, parent_mass=None):
    """Exhaustively find the best Gini cut of the node without changing indices.

    Sorts the finite values per feature and walks the boundaries between
    distinct values, updating per-class masses. Tries NaN on the left and on the
    right, and also the finite-versus-NaN cut (threshold=+inf,
    missing_left=False). Both children need min_samples_leaf active rows and
    positive mass. Thresholds are float64 midpoints of float32 X values. With
    no NaN in training, future NaN goes to the child with more rows; ties go
    right. Gain ties: first feature_order, then the smaller threshold, then NaN
    on the right. Without a candidate: Split(-1, nan, False, -inf, 0, -1).
    """
    if search_stopping not in ("off", "bound"):
        raise ValueError("search_stopping must be 'off' or 'bound'.")
    rows = np.asarray(sample_indices[start:end], dtype=np.int64)
    parent_mass = np.zeros(n_classes, dtype=np.float64)
    for row in rows:
        parent_mass[y[row]] += weights[row]
    total_rows = len(rows)
    total_mass = float(parent_mass.sum())
    if total_mass <= 0:
        return Split(-1, np.nan, False, -np.inf, 0, -1)

    best = Split(-1, np.nan, False, -np.inf, 0, -1)
    evaluated = 0
    missing_seen = 0

    def better(feature, threshold, missing_left, gain, n_left):
        nonlocal best
        if gain > best.gain:
            best = Split(int(feature), float(threshold), bool(missing_left),
                         float(gain), int(n_left), -1)

    for feature in np.asarray(feature_order, dtype=np.int64):
        values = X[rows, feature]
        finite_mask = ~np.isnan(values)
        finite_rows = rows[finite_mask]
        missing_rows = rows[~finite_mask]
        missing_count = len(missing_rows)
        missing_mass = np.zeros(n_classes, dtype=np.float64)
        for row in missing_rows:
            missing_mass[y[row]] += weights[row]
        missing_seen += missing_count
        order = np.argsort(X[finite_rows, feature], kind="mergesort")
        ordered_rows = finite_rows[order]
        finite_count = len(ordered_rows)
        prefix_mass = np.zeros(n_classes, dtype=np.float64)
        prefix_count = 0

        for pos in range(finite_count - 1):
            row = ordered_rows[pos]
            prefix_mass[y[row]] += weights[row]
            prefix_count += 1
            current = float(X[row, feature])
            next_value = float(X[ordered_rows[pos + 1], feature])
            if current == next_value:
                continue
            threshold = (current + next_value) / 2.0
            directions = (False, True) if missing_count > 0 else (False,)
            for missing_left in directions:
                left_mass = prefix_mass + (missing_mass if missing_left else 0.0)
                left_count = prefix_count + (missing_count if missing_left else 0)
                if missing_count == 0:
                    missing_left = bool(left_count > total_rows - left_count)
                right_mass = parent_mass - left_mass
                right_count = total_rows - left_count
                if (left_count < min_samples_leaf or right_count < min_samples_leaf
                        or left_mass.sum() <= 0 or right_mass.sum() <= 0):
                    continue
                gain = float(gini(parent_mass) - (
                    left_mass.sum() * gini(left_mass)
                    + right_mass.sum() * gini(right_mass)) / total_mass)
                evaluated += 1
                better(feature, threshold, missing_left, gain, left_count)

        # The only useful split when all finite values are on one side of NaN
        # is finite-left / missing-right. A threshold of +inf preserves the
        # same routing rule in partition_samples and apply_nodes.
        if missing_count > 0 and finite_count > 0:
            finite_mass = parent_mass - missing_mass
            left_count = finite_count
            right_count = missing_count
            if (left_count >= min_samples_leaf and right_count >= min_samples_leaf
                    and finite_mass.sum() > 0 and missing_mass.sum() > 0):
                gain = float(gini(parent_mass) - (
                    finite_mass.sum() * gini(finite_mass)
                    + missing_mass.sum() * gini(missing_mass)) / total_mass)
                evaluated += 1
                better(feature, np.inf, False, gain, left_count)

    if stats is not None:
        stats["exact_candidates_evaluated"] = stats.get("exact_candidates_evaluated", 0) + evaluated
        stats["missing_rows_scanned"] = stats.get("missing_rows_scanned", 0) + missing_seen
    return best


def _find_best_split_exact_precision_reference(
        X, y, weights, sample_indices, start, end, n_classes,
        min_samples_leaf, feature_order, positive_class, min_precision,
        min_support, stats=None, parent_mass=None):
    """Reference for the precision objective (best child precision - parent's).

    ``parent_mass`` is accepted to mirror the real search and ignored: the
    reference always recomputes the mass.
    """
    rows = np.asarray(sample_indices[start:end], dtype=np.int64)
    parent_mass = np.zeros(n_classes, dtype=np.float64)
    for row in rows:
        parent_mass[y[row]] += weights[row]
    best = Split(-1, np.nan, False, -np.inf, 0, -1)
    evaluated = 0

    for feature in np.asarray(feature_order, dtype=np.int64):
        values = X[rows, feature]
        finite_mask = ~np.isnan(values)
        finite_rows = rows[finite_mask]
        missing_rows = rows[~finite_mask]
        missing_count = len(missing_rows)
        missing_mass = np.zeros(n_classes, dtype=np.float64)
        for row in missing_rows:
            missing_mass[y[row]] += weights[row]
        order = np.argsort(X[finite_rows, feature], kind="mergesort")
        ordered_rows = finite_rows[order]
        finite_count = len(ordered_rows)
        prefix_mass = np.zeros(n_classes, dtype=np.float64)
        prefix_count = 0

        for pos in range(finite_count - 1):
            row = ordered_rows[pos]
            prefix_mass[y[row]] += weights[row]
            prefix_count += 1
            current = float(X[row, feature])
            next_value = float(X[ordered_rows[pos + 1], feature])
            if current == next_value:
                continue
            threshold = (current + next_value) / 2.0
            directions = (False, True) if missing_count > 0 else (False,)
            for missing_left in directions:
                left_mass = prefix_mass + (missing_mass if missing_left else 0.0)
                left_count = prefix_count + (missing_count if missing_left else 0)
                if missing_count == 0:
                    missing_left = bool(left_count > len(rows) - left_count)
                right_mass = parent_mass - left_mass
                right_count = len(rows) - left_count
                if (left_count < min_samples_leaf or right_count < min_samples_leaf
                        or left_mass.sum() <= 0 or right_mass.sum() <= 0):
                    continue
                gain = float(precision_split_gain(
                    parent_mass, left_mass, right_mass, positive_class,
                    min_support))
                evaluated += 1
                if gain > best.gain and gain > 0.0:
                    best = Split(int(feature), threshold, bool(missing_left),
                                 gain, int(left_count), -1)

        if missing_count > 0 and finite_count > 0:
            finite_mass = parent_mass - missing_mass
            left_count = finite_count
            right_count = missing_count
            if (left_count >= min_samples_leaf and right_count >= min_samples_leaf
                    and finite_mass.sum() > 0 and missing_mass.sum() > 0):
                gain = float(precision_split_gain(
                    parent_mass, finite_mass, missing_mass, positive_class,
                    min_support))
                evaluated += 1
                if gain > best.gain and gain > 0.0:
                    best = Split(int(feature), np.inf, False, gain,
                                 int(left_count), -1)

    if stats is not None:
        stats["exact_candidates_evaluated"] = (
            stats.get("exact_candidates_evaluated", 0) + evaluated)
    return best


@njit(cache=True, parallel=True)
def best_depth2_reference(Xb, g, h, w, nb, lam, min_weight):
    """Optimal depth-2 Newton tree by exhaustive search over the bins.

    For each root (f1, t1), each child takes its best single cut (or stays a
    leaf if none gains). Returns, per f1, (gain, t1, fL, tL, fR, tR); the caller
    picks the f1 with the largest gain (ties: smallest f1). O(n p^2) passes over
    the data and O(p * B^2 * p) search.
    """
    n, p = Xb.shape
    B = 0
    for j in range(p):
        if nb[j] > B:
            B = nb[j]
    res_gain = np.zeros(p)
    res = np.full((p, 5), -1, dtype=np.int64)
    Gt = 0.0
    Ht = 0.0
    for i in range(n):
        Gt += g[i]
        Ht += h[i]
    parent = _score(Gt, Ht, lam)
    for f1 in prange(p):
        n1 = nb[f1]
        hist = np.zeros((n1, p, B, 3))
        for i in range(n):
            b1 = Xb[i, f1]
            for f2 in range(p):
                b2 = Xb[i, f2]
                hist[b1, f2, b2, 0] += g[i]
                hist[b1, f2, b2, 1] += h[i]
                hist[b1, f2, b2, 2] += w[i]
        total = np.zeros((p, B, 3))
        for b1 in range(n1):
            total += hist[b1]
        left = np.zeros((p, B, 3))
        right = np.empty((p, B, 3))
        best = 0.0
        for t1 in range(n1 - 1):
            left += hist[t1]
            GL = 0.0
            HL = 0.0
            WL = 0.0
            for b2 in range(nb[0]):
                GL += left[0, b2, 0]
                HL += left[0, b2, 1]
                WL += left[0, b2, 2]
            GR = Gt - GL
            HR = Ht - HL
            WR = 0.0
            for b2 in range(nb[0]):
                WR += total[0, b2, 2] - left[0, b2, 2]
            if min_weight > WL or min_weight > WR:
                continue
            right[:] = total - left
            root_gain = _score(GL, HL, lam) + _score(GR, HR, lam) - parent
            bl = 0.0
            fl = -1
            tl = -1
            br = 0.0
            fr = -1
            tr = -1
            for f2 in range(p):
                gl, cl = _best_cut_hist(left[f2], nb[f2], lam, min_weight)
                if gl > bl:
                    bl = gl
                    fl = f2
                    tl = cl
                gr, cr = _best_cut_hist(right[f2], nb[f2], lam, min_weight)
                if gr > br:
                    br = gr
                    fr = f2
                    tr = cr
            gain = root_gain + bl + br
            if gain > best:
                best = gain
                res[f1, 0] = t1
                res[f1, 1] = fl
                res[f1, 2] = tl
                res[f1, 3] = fr
                res[f1, 4] = tr
        res_gain[f1] = best
    return res_gain, res
