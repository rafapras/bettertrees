"""Transformations after growth: pruning, probabilities and monotonicity.

No function here reads the training X: per-node masses are enough.

References
----------
Breiman, Friedman, Olshen, Stone. "Classification and Regression Trees." 1984
(minimal cost-complexity pruning).
Agarwal, Tan, Ronen, Singh, Yu. "Hierarchical Shrinkage." ICML 2022.
"""

import numpy as np

from ._data import allocate_nodes
from .kernels import apply_nodes, gini


def prune_tree_cost_complexity(nodes, alpha):
    """Select the subtree by weighted Gini risk + alpha per leaf.

    The dynamic program is linear in the number of nodes and does not read X:
    class_weight already holds each node's training mass. The output is
    compact, so introspection and prediction only see reachable nodes.
    """
    if alpha <= 0.0:
        return nodes
    root_total = float(nodes.class_weight[0].sum())
    if root_total <= 0.0:
        raise ValueError("The tree has a root without valid weight.")
    order = []
    pending = [0]
    while pending:
        node = pending.pop()
        order.append(node)
        left = int(nodes.left[node])
        if left != -1:
            pending.append(int(nodes.right[node]))
            pending.append(left)

    cost = np.empty(len(nodes.left), dtype=np.float64)
    keep_split = np.zeros(len(nodes.left), dtype=np.bool_)
    for node in reversed(order):
        mass = nodes.class_weight[node]
        leaf_cost = float(mass.sum()) / root_total * gini(mass) + alpha
        left = int(nodes.left[node])
        if left == -1:
            cost[node] = leaf_cost
            continue
        children_cost = cost[left] + cost[int(nodes.right[node])]
        if leaf_cost <= children_cost:
            cost[node] = leaf_cost
        else:
            cost[node] = children_cost
            keep_split[node] = True

    reachable = []
    pending = [0]
    while pending:
        node = pending.pop()
        reachable.append(node)
        if keep_split[node]:
            pending.append(int(nodes.right[node]))
            pending.append(int(nodes.left[node]))
    mapping = {old: new for new, old in enumerate(reachable)}
    pruned = allocate_nodes(len(reachable), nodes.class_weight.shape[1])
    for new, old in enumerate(reachable):
        pruned.class_weight[new] = nodes.class_weight[old]
        pruned.n_samples[new] = nodes.n_samples[old]
        if keep_split[old]:
            pruned.left[new] = mapping[int(nodes.left[old])]
            pruned.right[new] = mapping[int(nodes.right[old])]
            pruned.feature[new] = nodes.feature[old]
            pruned.threshold[new] = nodes.threshold[old]
            pruned.missing_left[new] = nodes.missing_left[old]
    return pruned


def hierarchical_shrinkage_probabilities(nodes, shrinkage):
    """Probability of EVERY node with hierarchical shrinkage (Agarwal et al. 2022).

    p(node) = p(parent) + (freq(node) - freq(parent)) / (1 + shrinkage / mass(parent)),
    starting from the root frequency. Since the weight 1 / (1 + lambda / mass)
    falls with depth, p(leaf) is a convex combination of its ancestors'
    frequencies: always a valid distribution and, with lambda > 0, without
    zeros for classes present at the root. lambda = 0 returns the raw frequencies.
    """
    mass = np.asarray(nodes.class_weight, dtype=np.float64)
    totals = mass.sum(axis=1)
    freq = mass / np.where(totals > 0, totals, 1.0)[:, None]
    out = np.empty_like(freq)
    out[0] = freq[0]
    stack = [0]
    while stack:
        node = stack.pop()
        left = int(nodes.left[node])
        if left == -1:
            continue
        weight = 1.0 / (1.0 + shrinkage / totals[node])
        for child in (left, int(nodes.right[node])):
            out[child] = out[node] + (freq[child] - freq[node]) * weight
            stack.append(child)
    return out


def expansion_steps(nodes):
    """Step at which each node was expanded (inf = leaf).

    Without pruning, children ids follow the builder's expansion order, so the
    best-first tree with L leaves is the prefix with the first L-1 expansions
    (the order does not depend on the budget).
    """
    left = np.asarray(nodes.left)
    internal = np.flatnonzero(left != -1)
    order = internal[np.argsort(left[internal])]
    steps = np.full(len(left), np.inf)
    steps[order] = np.arange(len(order))
    return steps


def prefix_leaf_ids(X, nodes, steps, n_leaves):
    """Final node of each row in the prefix tree with ``n_leaves`` leaves."""
    budget = n_leaves - 1
    node = np.zeros(len(X), dtype=np.int64)
    active = np.ones(len(X), dtype=bool)
    while active.any():
        idx = np.flatnonzero(active)
        current = node[idx]
        grows = (nodes.left[current] != -1) & (steps[current] < budget)
        idx, current = idx[grows], current[grows]
        active[:] = False
        if not len(idx):
            break
        values = X[idx, nodes.feature[current]]
        missing = np.isnan(values)
        go_left = np.where(missing, nodes.missing_left[current],
                           values <= nodes.threshold[current])
        node[idx] = np.where(go_left, nodes.left[current], nodes.right[current])
        active[idx] = True
    return node


def predict_proba_nodes(X, nodes, positive_leaf_probabilities=None,
                        positive_class=1, leaf_smoothing=0.0,
                        leaf_probabilities=None):
    """Normalize leaf masses into float64 probabilities (n, K).

    With leaf_smoothing=0, the same zeros and frequencies as a CART leaf;
    otherwise the masses are shrunk toward the root distribution. A leaf with
    invalid mass is a builder error and raises ValueError instead of inventing
    a distribution. The public API validates X before this call.
    """
    ids = apply_nodes(X, nodes.left, nodes.right, nodes.feature,
                      nodes.threshold, nodes.missing_left)
    return node_probabilities(nodes, ids, positive_leaf_probabilities, positive_class,
                              leaf_smoothing, leaf_probabilities)


def node_probabilities(nodes, ids, positive_leaf_probabilities=None, positive_class=1,
                       leaf_smoothing=0.0, leaf_probabilities=None):
    """Probabilities (len(ids), K) of the nodes ``ids``: the rule ``predict_proba`` uses,
    shared with the text export so the two can never disagree."""
    ids = np.asarray(ids, dtype=np.intp)
    mass = nodes.class_weight[ids]
    totals = mass.sum(axis=1, keepdims=True)
    if (mass < 0).any() or not np.isfinite(mass).all() or (totals <= 0).any():
        raise ValueError("The tree has a leaf without valid weight.")
    if leaf_probabilities is not None:
        return np.asarray(leaf_probabilities, dtype=np.float64)[ids]
    if positive_leaf_probabilities is None:
        if leaf_smoothing == 0:
            return mass / totals
        root_mass = nodes.class_weight[0]
        prior = root_mass / root_mass.sum()
        return (mass + float(leaf_smoothing) * prior) / (totals + leaf_smoothing)
    if mass.shape[1] != 2:
        raise ValueError("The monotonic projection requires exactly two classes.")
    positive = np.asarray(positive_leaf_probabilities, dtype=np.float64)[ids]
    result = np.empty((len(ids), 2), dtype=np.float64)
    result[:, positive_class] = positive
    result[:, 1 - positive_class] = 1.0 - positive
    return result


def finite_leaf_regions(nodes, n_features):
    """Return the finite boxes of the leaves, propagating bounds through the nodes."""
    regions = []

    def visit(node_id, lower, upper):
        left = int(nodes.left[node_id])
        if left == -1:
            possible = True
            for lo, hi in zip(lower, upper):
                # The lower bound comes from a right child (> threshold),
                # so a box with lo == hi holds no finite point.
                if not lo < hi:
                    possible = False
                    break
            if possible:
                regions.append((int(node_id), lower, upper))
            return
        feature = int(nodes.feature[node_id])
        threshold = float(nodes.threshold[node_id])
        left_lower = lower.copy()
        left_upper = upper.copy()
        left_upper[feature] = min(left_upper[feature], threshold)
        right_lower = lower.copy()
        right_upper = upper.copy()
        right_lower[feature] = max(right_lower[feature], threshold)
        visit(left, left_lower, left_upper)
        visit(int(nodes.right[node_id]), right_lower, right_upper)

    visit(0, np.full(n_features, -np.inf),
          np.full(n_features, np.inf))
    return regions

def project_monotonic_leaf_probabilities(nodes, directions,
                                         positive_class_index,
                                         leaf_smoothing=0.0):
    """Project leaves onto a conservative, deterministic global order.

    Each edge compares two leaf boxes that overlap on the other features and
    are ordered on the constrained feature. Topological propagation of the
    largest predecessor probability keeps the empirical frequency whenever
    possible and only raises leaves that would violate the order.
    """
    leaf_probabilities = np.zeros(len(nodes.left), dtype=np.float64)
    leaf_ids = np.flatnonzero(nodes.left == -1)
    masses = nodes.class_weight[leaf_ids]
    totals = masses.sum(axis=1)
    prior_mass = nodes.class_weight[0]
    prior_positive = prior_mass[positive_class_index] / prior_mass.sum()
    leaf_probabilities[leaf_ids] = (
        (masses[:, positive_class_index]
         + float(leaf_smoothing) * prior_positive)
        / (totals + float(leaf_smoothing)))
    active_directions = np.flatnonzero(directions != 0)
    if len(active_directions) == 0:
        return leaf_probabilities
    regions = finite_leaf_regions(nodes, len(directions))
    region_by_leaf = {leaf_id: (lower, upper)
                      for leaf_id, lower, upper in regions}
    region_leaves = [leaf_id for leaf_id in leaf_ids
                     if int(leaf_id) in region_by_leaf]
    edges = set()
    for feature in active_directions:
        direction = int(directions[feature])
        for first_pos, first_leaf in enumerate(region_leaves):
            first_lower, first_upper = region_by_leaf[first_leaf]
            for second_leaf in region_leaves[first_pos + 1:]:
                second_lower, second_upper = region_by_leaf[second_leaf]
                overlaps_elsewhere = True
                for other in range(len(directions)):
                    if other == feature:
                        continue
                    # Boxes are (lower, upper]. Touching only at the
                    # bound is not an overlap: the point is outside
                    # one of them and cannot justify an edge.
                    if (max(first_lower[other], second_lower[other])
                            >= min(first_upper[other], second_upper[other])):
                        overlaps_elsewhere = False
                        break
                if not overlaps_elsewhere:
                    continue
                if first_upper[feature] <= second_lower[feature]:
                    low_leaf, high_leaf = first_leaf, second_leaf
                elif second_upper[feature] <= first_lower[feature]:
                    low_leaf, high_leaf = second_leaf, first_leaf
                else:
                    continue
                edge = ((low_leaf, high_leaf) if direction > 0
                        else (high_leaf, low_leaf))
                edges.add((int(edge[0]), int(edge[1])))
    if not edges:
        return leaf_probabilities
    active_leaves = sorted({node for edge in edges for node in edge})
    position = {node: i for i, node in enumerate(active_leaves)}
    adjacency = [[] for _ in active_leaves]
    indegree = np.zeros(len(active_leaves), dtype=np.int64)
    for source, target in sorted(edges):
        source_pos, target_pos = position[source], position[target]
        adjacency[source_pos].append(target_pos)
        indegree[target_pos] += 1
    queue = [i for i, value in enumerate(indegree) if value == 0]
    order = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(active_leaves):
        raise ValueError("The monotonic regions formed an unexpected cycle.")
    for source_pos in order:
        source = active_leaves[source_pos]
        for target_pos in adjacency[source_pos]:
            target = active_leaves[target_pos]
            leaf_probabilities[target] = max(
                leaf_probabilities[target], leaf_probabilities[source])
    return leaf_probabilities
