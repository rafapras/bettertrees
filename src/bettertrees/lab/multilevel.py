"""Experimental local depth-2 lookahead, composed in successive tree blocks.

The optimum is exact only for one block, under the chosen finite candidate
space and weighted Gini. Composing blocks is not a global depth-4 optimum.
The existing greedy estimator is intentionally left untouched.
"""

import itertools
from time import perf_counter

import numpy as np
from sklearn.utils.validation import check_random_state

from .._data import NodeArrays, Split, prepare_training_data, validate_X
from ..bins import fit_bin_edges, transform_bins_row_major
from ..kernels import _solve_depth2_block_hist_numba, apply_nodes, gini
from ..postprocess import predict_proba_nodes
from ..search import (
    _build_histograms_for_node,
    find_best_split_exact,
    find_best_split_hist,
    hist_edge_layout,
)

# Above this the block's joint histogram (bins^2 x features x classes)
# gets too large; the per-candidate path is used instead.
_BLOCK_KERNEL_MAX_CELLS = 50_000_000


class MultiLevelTreeModel:
    """Fitted result of fit_multilevel_tree; experimental prediction facade."""

    def __init__(self, nodes, classes, n_features, leaf_smoothing, stats, edges):
        self.nodes_ = nodes
        self.classes_ = classes
        self.n_features_in_ = n_features
        self.n_classes_ = len(classes)
        self.leaf_smoothing = leaf_smoothing
        self.fit_stats_ = stats
        self.bin_edges_ = edges

    def apply(self, X):
        X = validate_X(X, n_features=self.n_features_in_)
        return apply_nodes(X, self.nodes_.left, self.nodes_.right,
                           self.nodes_.feature, self.nodes_.threshold,
                           self.nodes_.missing_left).astype(np.intp)

    def predict_proba(self, X):
        X = validate_X(X, n_features=self.n_features_in_)
        return predict_proba_nodes(X, self.nodes_,
                                   leaf_smoothing=self.leaf_smoothing)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def get_n_leaves(self):
        return int(np.count_nonzero(self.nodes_.left == -1))

    def get_depth(self):
        maximum = 0
        pending = [(0, 0)]
        while pending:
            node, depth = pending.pop()
            maximum = max(maximum, depth)
            if self.nodes_.left[node] != -1:
                pending.append((int(self.nodes_.left[node]), depth + 1))
                pending.append((int(self.nodes_.right[node]), depth + 1))
        return maximum


def _candidate_roots(X, y, weights, rows, feature_order, root_splitter,
                     edges, n_classes, min_samples_leaf):
    parent_mass = np.bincount(y[rows], weights=weights[rows],
                              minlength=n_classes)
    parent_total = float(parent_mass.sum())
    parent_gini = float(gini(parent_mass))
    seen = set()
    for feature in feature_order:
        feature = int(feature)
        values = X[rows, feature]
        missing = np.isnan(values)
        finite = values[~missing]
        if root_splitter == "hist":
            thresholds = [float(edge) for edge in edges[feature]]
        else:
            unique = np.unique(finite)
            thresholds = [(float(a) + float(b)) / 2.0
                          for a, b in itertools.pairwise(unique)]
        if missing.any() and len(finite):
            thresholds.append(np.inf)  # finite versus missing
        for threshold in thresholds:
            for initial_missing_left in ((False, True) if missing.any()
                                         and np.isfinite(threshold) else (False,)):
                mask = (values <= threshold) & ~missing
                if initial_missing_left:
                    mask |= missing
                left_count = int(np.count_nonzero(mask))
                right_count = len(rows) - left_count
                if (left_count < min_samples_leaf
                        or right_count < min_samples_leaf):
                    continue
                # Equal training partitions of different features still define
                # different prediction rules; only deduplicate within feature.
                signature = (feature, mask.tobytes())
                if signature in seen:
                    continue
                seen.add(signature)
                left_rows, right_rows = rows[mask], rows[~mask]
                left_mass = np.bincount(y[left_rows], weights=weights[left_rows],
                                        minlength=n_classes)
                right_mass = np.bincount(y[right_rows], weights=weights[right_rows],
                                         minlength=n_classes)
                left_total, right_total = left_mass.sum(), right_mass.sum()
                if left_total <= 0 or right_total <= 0:
                    continue
                missing_left = (initial_missing_left if missing.any()
                                else left_count > right_count)
                gain = parent_gini - (
                    left_total * gini(left_mass)
                    + right_total * gini(right_mass)) / parent_total
                split = Split(feature, threshold, bool(missing_left),
                              float(gain), left_count, -1)
                yield split, left_rows, right_rows, float(left_total), float(right_total)


def _best_one_level(X, X_binned, y, weights, rows, edges, n_classes,
                    min_samples_leaf, feature_order, splitter, stats,
                    prebuilt_histograms=None):
    if len(rows) < 2 * min_samples_leaf:
        return None
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if splitter == "hist":
        parent_mass = (None if prebuilt_histograms is None else
                       prebuilt_histograms[0][0].sum(axis=0))
        split = find_best_split_hist(
            X_binned, y, weights, rows, 0, len(rows), edges, n_classes,
            min_samples_leaf, feature_order, search_stopping="off",
            stats=stats, prebuilt_histograms=prebuilt_histograms,
            parent_mass=parent_mass)
    else:
        split = find_best_split_exact(
            X, y, weights, rows, 0, len(rows), n_classes,
            min_samples_leaf, feature_order, search_stopping="off",
            stats=stats)
    return split if split.feature >= 0 and split.gain > 0.0 else None


def _split_from_bin(edges, feature, bin_threshold, missing_left, gain, n_left):
    n_finite_bins = len(edges[feature]) + 1
    threshold = (np.inf if bin_threshold == n_finite_bins
                 else float(edges[feature][bin_threshold - 1]))
    return Split(int(feature), threshold, bool(missing_left), float(gain),
                 int(n_left), int(bin_threshold))


def _solve_block_kernel(X_binned, y, weights, rows, edges, layout, n_classes,
                        min_samples_leaf, feature_order, candidate_limit, stats):
    """Bloco depth 2 hist/hist numa chamada (ver ``_solve_depth2_block_hist_numba``)."""
    max_bins, lengths = layout
    stats["depth2_blocks"] += 1
    count, root, left, right = _solve_depth2_block_hist_numba(
        X_binned, y, weights, np.ascontiguousarray(rows, dtype=np.int64),
        lengths, feature_order, n_classes, max_bins, min_samples_leaf,
        candidate_limit)
    if count < 0:
        raise ValueError(
            f"The depth-2 block has more than {candidate_limit} root cuts; "
            "reduce max_bins/features or raise candidate_limit explicitly.")
    stats["root_candidates"] += int(count)
    stats["child_searches"] += 2 * int(count)
    stats["block_kernel_calls"] = stats.get("block_kernel_calls", 0) + 1
    feature, bin_id, only_missing, missing_left, gain, n_left = root
    if feature < 0:
        return None
    threshold = np.inf if only_missing else float(edges[feature][bin_id - 1])
    root_split = Split(int(feature), threshold, bool(missing_left), float(gain),
                       int(n_left), -1)
    children = [None if child[0] < 0 else _split_from_bin(edges, *child)
                for child in (left, right)]
    return root_split, children[0], children[1]


def _solve_block(X, X_binned, y, weights, rows, edges, n_classes,
                 min_samples_leaf, feature_order, root_splitter,
                 child_splitter, remaining_depth, candidate_limit, stats,
                 reuse_child_histograms=False, block_layout=None):
    if len(rows) < 2 * min_samples_leaf:
        return None
    if block_layout is not None and remaining_depth == 2:
        return _solve_block_kernel(X_binned, y, weights, rows, edges,
                                   block_layout, n_classes, min_samples_leaf,
                                   feature_order, candidate_limit, stats)
    if remaining_depth == 1:
        stats["one_level_searches"] += 1
        split = _best_one_level(
            X, X_binned, y, weights, rows, edges, n_classes,
            min_samples_leaf, feature_order, root_splitter, stats)
        return None if split is None else (split, None, None)

    stats["depth2_blocks"] += 1
    parent_histograms = None
    if reuse_child_histograms and child_splitter == "hist":
        parent_histograms = _build_histograms_for_node(
            X_binned, y, weights, rows, 0, len(rows), edges, n_classes)
        stats["histograms"] = stats.get("histograms", 0) + X_binned.shape[1]
    best = None
    best_improvement = 0.0
    best_splits = 0
    count = 0
    for root, left_rows, right_rows, left_total, right_total in _candidate_roots(
            X, y, weights, rows, feature_order, root_splitter,
            edges, n_classes, min_samples_leaf):
        count += 1
        if count > candidate_limit:
            raise ValueError(
                f"The depth-2 block has more than {candidate_limit} root cuts; "
                "reduce max_bins/features or raise candidate_limit explicitly.")
        left_histograms = right_histograms = None
        if parent_histograms is not None:
            smaller_left = len(left_rows) <= len(right_rows)
            smaller_rows = left_rows if smaller_left else right_rows
            smaller_histograms = _build_histograms_for_node(
                X_binned, y, weights, smaller_rows, 0, len(smaller_rows),
                edges, n_classes)
            larger_histograms = (
                parent_histograms[0] - smaller_histograms[0],
                parent_histograms[1] - smaller_histograms[1])
            if smaller_left:
                left_histograms, right_histograms = (
                    smaller_histograms, larger_histograms)
            else:
                right_histograms, left_histograms = (
                    smaller_histograms, larger_histograms)
            stats["histograms"] += X_binned.shape[1]
        left_split = _best_one_level(
            X, X_binned, y, weights, left_rows, edges, n_classes,
            min_samples_leaf, feature_order, child_splitter, stats,
            prebuilt_histograms=left_histograms)
        right_split = _best_one_level(
            X, X_binned, y, weights, right_rows, edges, n_classes,
            min_samples_leaf, feature_order, child_splitter, stats,
            prebuilt_histograms=right_histograms)
        stats["child_searches"] += 2
        improvement = (float(weights[rows].sum()) * root.gain
                       + (0.0 if left_split is None else left_total * left_split.gain)
                       + (0.0 if right_split is None else right_total * right_split.gain))
        n_splits = 1 + (left_split is not None) + (right_split is not None)
        if (improvement > best_improvement + 1e-12
                or (best is not None
                    and abs(improvement - best_improvement) <= 1e-12
                    and n_splits < best_splits)):
            best = (root, left_split, right_split)
            best_improvement = improvement
            best_splits = n_splits
    stats["root_candidates"] += count
    return best


def fit_multilevel_tree(X, y, sample_weight=None, *, max_depth=4,
                        root_splitter="hist", child_splitter=None,
                        max_bins=16, min_samples_leaf=5,
                        candidate_limit=128, leaf_smoothing=0.0,
                        random_state=None, reuse_child_histograms=False,
                        block_kernel=True):
    """Fit local Gini-optimal depth-2 blocks and stack them to ``max_depth``.

    Even depths are pure stacks of depth-2 blocks; an odd depth ends with a
    one-level greedy search.

    `root_splitter` selects root candidates; `child_splitter` searches each
    child with the existing backend. The optimum is within each block's
    candidate space, not across all depth-4 trees. No silent truncation:
    candidate_limit raises when the full block cannot be searched.
    ``block_kernel`` (default) solves hist/hist blocks with unit weights in
    one Numba call over pairwise histograms; same candidates and tie-breaks,
    so the tree is identical to ``block_kernel=False``.
    """
    if (isinstance(max_depth, (bool, np.bool_)) or not isinstance(max_depth, (int, np.integer))
            or max_depth < 1):
        raise ValueError("The multilevel max_depth must be an integer >= 1.")
    if child_splitter is None:
        child_splitter = root_splitter
    if root_splitter not in ("hist", "exact") or child_splitter not in ("hist", "exact"):
        raise ValueError("root_splitter and child_splitter must be 'hist' or 'exact'.")
    for name, value, minimum in (("max_bins", max_bins, 2),
                                 ("min_samples_leaf", min_samples_leaf, 1),
                                 ("candidate_limit", candidate_limit, 1)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if max_bins > 255:
        raise ValueError("max_bins must be <= 255.")
    if not np.isscalar(leaf_smoothing) or isinstance(leaf_smoothing, (bool, np.bool_)):
        raise ValueError("leaf_smoothing must be finite and non-negative.")
    try:
        valid_smoothing = float(leaf_smoothing)
    except (TypeError, ValueError) as exc:
        raise ValueError("leaf_smoothing must be finite and non-negative.") from exc
    if not np.isfinite(valid_smoothing) or valid_smoothing < 0:
        raise ValueError("leaf_smoothing must be finite and non-negative.")

    started = perf_counter()
    X, encoded, weights, classes = prepare_training_data(X, y, sample_weight)
    if reuse_child_histograms and not np.all(weights == 1.0):
        raise ValueError("reuse_child_histograms requires unit sample weights.")
    n_classes = len(classes)
    feature_order = np.ascontiguousarray(
        check_random_state(random_state).permutation(X.shape[1]), dtype=np.int64)
    use_hist = root_splitter == "hist" or child_splitter == "hist"
    if use_hist:
        edges = fit_bin_edges(X, max_bins, n_jobs=1)
        X_binned = transform_bins_row_major(X, edges, n_jobs=1)
    else:
        edges = None
        X_binned = None
    block_layout = None
    if (root_splitter == "hist" and child_splitter == "hist" and block_kernel
            and np.all(weights == 1.0)):
        layout = hist_edge_layout(edges)
        if layout[0] ** 2 * X.shape[1] * n_classes <= _BLOCK_KERNEL_MAX_CELLS:
            block_layout = layout
    stats = dict(depth2_blocks=0, one_level_searches=0,
                 root_candidates=0, child_searches=0,
                 root_splitter=root_splitter, child_splitter=child_splitter,
                 max_depth=max_depth, candidate_limit=candidate_limit,
                 min_samples_leaf=min_samples_leaf)
    left, right, feature, threshold, missing_left, class_weight, n_samples = (
        [], [], [], [], [], [], [])

    def add_node(rows):
        node = len(left)
        left.append(-1)
        right.append(-1)
        feature.append(-1)
        threshold.append(np.nan)
        missing_left.append(False)
        class_weight.append(np.bincount(
            encoded[rows], weights=weights[rows], minlength=n_classes))
        n_samples.append(len(rows))
        return node

    def partition(rows, split):
        values = X[rows, split.feature]
        missing = np.isnan(values)
        mask = ((values <= split.threshold) & ~missing)
        if split.missing_left:
            mask |= missing
        return rows[mask], rows[~mask]

    def set_split(node, split, left_node, right_node):
        left[node], right[node] = left_node, right_node
        feature[node], threshold[node] = split.feature, split.threshold
        missing_left[node] = split.missing_left

    def grow(rows, depth):
        node = add_node(rows)
        if depth >= max_depth:
            return node
        plan = _solve_block(
            X, X_binned, encoded, weights, rows, edges, n_classes,
            min_samples_leaf, feature_order, root_splitter, child_splitter,
            min(2, max_depth - depth), candidate_limit, stats,
            reuse_child_histograms=reuse_child_histograms,
            block_layout=block_layout)
        if plan is None:
            return node
        root, left_split, right_split = plan
        left_rows, right_rows = partition(rows, root)

        def attach_child(child_rows, child_split):
            if child_split is None:
                # Blocks meet only at depths 0, 2 and 4. A shorter arm of a
                # block is terminal; it must not start an overlapping block.
                return add_node(child_rows)
            child_node = add_node(child_rows)
            grand_left, grand_right = partition(child_rows, child_split)
            left_node = grow(grand_left, depth + 2)
            right_node = grow(grand_right, depth + 2)
            set_split(child_node, child_split, left_node, right_node)
            return child_node

        left_node = attach_child(left_rows, left_split)
        right_node = attach_child(right_rows, right_split)
        set_split(node, root, left_node, right_node)
        return node

    grow(np.arange(len(X), dtype=np.int64), 0)
    nodes = NodeArrays(np.asarray(left, dtype=np.int32),
                       np.asarray(right, dtype=np.int32),
                       np.asarray(feature, dtype=np.int32),
                       np.asarray(threshold, dtype=np.float64),
                       np.asarray(missing_left, dtype=np.bool_),
                       np.ascontiguousarray(np.stack(class_weight)),
                       np.asarray(n_samples, dtype=np.int64))
    stats["n_nodes"] = len(nodes.left)
    stats["n_leaves"] = int(np.count_nonzero(nodes.left == -1))
    stats["fit_complete_seconds"] = perf_counter() - started
    return MultiLevelTreeModel(nodes, classes, X.shape[1], valid_smoothing,
                               stats, edges)
