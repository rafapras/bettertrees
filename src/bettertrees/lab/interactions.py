"""Interaction detection: FAST on the residual and co-occurrence in a teacher.

References
----------
Lou, Caruana, Gehrke, Hooker. "Accurate Intelligible Models with Pairwise
Interactions." KDD 2013 (FAST).
"""

from itertools import combinations

import numpy as np

from ..sums._common import as_float_matrix, as_target, as_weights, base_margin, binned, grad_hess
from ..sums._kernels import quadrant_scores


def all_pairs(p):
    return np.array(list(combinations(range(p), 2)), dtype=np.int64).reshape(-1, 2)


def fast_pair_scores(X, y, margin=None, sample_weight=None, *, pairs=None,
                     max_bins=32, lam=1.0, min_weight=20.0):
    """FAST (Lou et al., KDD 2013): gain of the best quadrant for each pair.

    Measures the interaction left **after** the base model: pass the margin
    (logit) of an additive model in ``margin`` (e.g. ``AdditiveTreeBooster``
    with ``depth=1``). Without a margin the score mixes main effects and
    interaction. Returns a dict with ``pairs`` (k, 2), ``gain``, ``order``
    (decreasing).
    """
    X = as_float_matrix(X)
    n, p = X.shape
    y = as_target(y, n)
    w = as_weights(sample_weight, n)
    m = np.full(n, base_margin(y, w)) if margin is None else np.asarray(margin, float)
    pairs = all_pairs(p) if pairs is None else np.ascontiguousarray(pairs, dtype=np.int64)
    Xb, _, nb = binned(X, max_bins)
    g, h = grad_hess(y, m, w)
    gain, _, _ = quadrant_scores(Xb, g, h, w, nb, pairs, lam, min_weight)
    return dict(pairs=pairs, gain=gain, order=np.argsort(-gain, kind="stable"))


def teacher_path_pairs(booster, n_features, weight="count"):
    """Co-occurrence of features on the same root-to-leaf path of a teacher (FRINGE).

    ``booster``: a ``lightgbm.Booster`` (or ``LGBMClassifier.booster_``). Each
    leaf adds to the pair (i, j) of features on its path the number of training
    rows in the leaf (``weight='count'``) or 1 (``'leaf'``). Returns a symmetric
    (p, p) matrix normalized to sum 1 over the upper triangle.
    """
    M = np.zeros((n_features, n_features))
    for info in booster.dump_model()["tree_info"]:
        stack = [(info["tree_structure"], ())]
        while stack:
            node, path = stack.pop()
            if "split_feature" in node:
                nxt = (*path, node["split_feature"])
                stack.append((node["left_child"], nxt))
                stack.append((node["right_child"], nxt))
                continue
            feats = sorted(set(path))
            v = float(node.get("leaf_count", 1)) if weight == "count" else 1.0
            for a, b in combinations(feats, 2):
                M[a, b] += v
    total = M.sum()
    if total > 0:
        M /= total
    return M + M.T
