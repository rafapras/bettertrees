"""Compiled kernels for the ITM interaction terms. No fastmath.

All compiled helpers called here live here as well, for Numba cache invalidation.
Each candidate term is a product of two centered, scaled factors u = phi1 * phi2
and is scored as G^2 / (H + lam), G = sum g*u, H = sum h*u^2 (the split-score
convention, no factor 1/2). Indicator factors (a leaf region or a cut) are
centered and scaled with h weights: c = sum_h(d) / sum_h, s = sqrt(c (1 - c)).
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _gh(m, y, w):
    if m >= 0:
        p = 1.0 / (1.0 + np.exp(-m))
    else:
        e = np.exp(m)
        p = e / (1.0 + e)
    return w * (p - y), w * max(p * (1.0 - p), 1e-16)


@njit(cache=True)
def affine_grad_hess(y, w, margin, contrib, a):
    """Zero-tree search for tree k: eta = C + a*T_k, scored at T_k = 0."""
    g, h = np.empty(len(y)), np.empty(len(y))
    for i in range(len(y)):
        gi, hi = _gh(margin[i] - a[i] * contrib[i], y[i], w[i])
        g[i], h[i] = gi * a[i], hi * a[i] * a[i]
    return g, h


@njit(cache=True, parallel=True)
def pair_scores(z, g, h, pairs, lam):
    """M1: tree x tree, u = z_j * z_k."""
    gains = np.zeros(len(pairs))
    for e in prange(len(pairs)):
        j, k = pairs[e]
        G, H = 0.0, 0.0
        for i in range(len(g)):
            u = z[i, j] * z[i, k]
            G += g[i] * u
            H += h[i] * u * u
        if H > 0.0:
            gains[e] = G * G / (H + lam)
    return gains


@njit(cache=True, parallel=True)
def leaf_tree_sums(ids, g, h, w, z, n_nodes):
    """Per (tree j, node a, tree k): sum g*z_k and h*z_k^2 over the rows in leaf a of j.

    Also per (tree j, node a): sum g, h, w. ``ids`` (n, K) is the leaf of each row in
    each tree, ``z`` (n, K) the h-normalized tree outputs. One pass, O(n K^2).
    """
    n, K = ids.shape
    A = np.zeros((K, n_nodes, K, 2))
    L = np.zeros((K, n_nodes, 3))
    for j in prange(K):
        for i in range(n):
            a = ids[i, j]
            L[j, a, 0] += g[i]
            L[j, a, 1] += h[i]
            L[j, a, 2] += w[i]
            for k in range(K):
                q = z[i, k]
                A[j, a, k, 0] += g[i] * q
                A[j, a, k, 1] += h[i] * q * q
    return A, L


@njit(cache=True, parallel=True)
def leaf_pair_sums(ids, g, h, w, n_nodes):
    """Per (tree j < tree k, leaf a of j, leaf b of k): sum g, h, w over the cell."""
    n, K = ids.shape
    C = np.zeros((K, K, n_nodes, n_nodes, 3))
    for j in prange(K):
        for i in range(n):
            a = ids[i, j]
            for k in range(j + 1, K):
                b = ids[i, k]
                C[j, k, a, b, 0] += g[i]
                C[j, k, a, b, 1] += h[i]
                C[j, k, a, b, 2] += w[i]
    return C


@njit(cache=True, parallel=True)
def cut_pair_sums(D, g, h, w):
    """Per pair of indicator columns p <= q: sum g, h, w where both are 1 (p == q: singles)."""
    n, Q = D.shape
    S = np.zeros((Q, Q, 3))
    for p in prange(Q):
        for i in range(n):
            if D[i, p]:
                for q in range(p, Q):
                    if D[i, q]:
                        S[p, q, 0] += g[i]
                        S[p, q, 1] += h[i]
                        S[p, q, 2] += w[i]
    return S


@njit(cache=True, parallel=True)
def cut_tree_hist(Xb, g, h, w, z, B):
    """M4 histogram (p, m, B, 4) of (g*z_j, h*z_j^2, h, w) per feature, partner and bin."""
    n, p = Xb.shape
    m = z.shape[1]
    out = np.zeros((p, m, B, 4))
    for f in prange(p):
        for i in range(n):
            b = Xb[i, f]
            for j in range(m):
                q = z[i, j]
                out[f, j, b, 0] += g[i] * q
                out[f, j, b, 1] += h[i] * q * q
                out[f, j, b, 2] += h[i]
                out[f, j, b, 3] += w[i]
    return out


@njit(cache=True)
def best_cut_tree(hist, nb, lam, min_weight):
    """M4 scan: best new cut [x_f > t] gating partner j. Returns (gain[p, m], t[p, m]).

    u = (d - c) / s * z_j with d = [bin > t]; both sides need ``min_weight``.
    """
    p, m = hist.shape[0], hist.shape[1]
    gains = np.zeros((p, m))
    cuts = np.full((p, m), -1, dtype=np.int64)
    for f in range(p):
        for j in range(m):
            Tg, Th2, Th, Tw = 0.0, 0.0, 0.0, 0.0
            for b in range(nb[f]):
                Tg += hist[f, j, b, 0]
                Th2 += hist[f, j, b, 1]
                Th += hist[f, j, b, 2]
                Tw += hist[f, j, b, 3]
            Lg, Lh2, Lh, Lw = 0.0, 0.0, 0.0, 0.0
            for t in range(nb[f] - 1):
                Lg += hist[f, j, t, 0]
                Lh2 += hist[f, j, t, 1]
                Lh += hist[f, j, t, 2]
                Lw += hist[f, j, t, 3]
                if min_weight > Lw or min_weight > Tw - Lw or Th <= 0.0:
                    continue
                c = (Th - Lh) / Th
                s2 = c * (1.0 - c)
                if s2 <= 1e-12:
                    continue
                G = ((Tg - Lg) - c * Tg) / np.sqrt(s2)
                H = ((1.0 - 2.0 * c) * (Th2 - Lh2) + c * c * Th2) / s2
                gain = G * G / (H + lam)
                if gain > gains[f, j]:
                    gains[f, j] = gain
                    cuts[f, j] = t
    return gains, cuts


@njit(cache=True)
def tree_step(y, w, margin, values, ids, leaf_values, k, a, lam, lr, cap, has_cap):
    """Newton step on the leaves of tree k with multiplier a = d eta / d T_k.

    eta is affine in T_k, so the margin update a * (new - old) is exact.
    """
    G, H = np.zeros(len(leaf_values)), np.zeros(len(leaf_values))
    for i in range(len(y)):
        g, h = _gh(margin[i], y[i], w[i])
        node = ids[i]
        G[node] += g * a[i]
        H[node] += h * a[i] * a[i]
    for node in range(len(leaf_values)):
        denominator = H[node] + lam
        s = -G[node] / denominator if denominator > 0 else 0.0
        if has_cap:
            s = min(max(s, -cap), cap)
        leaf_values[node] += lr * s
    for i in range(len(y)):
        new = leaf_values[ids[i]]
        margin[i] += a[i] * (new - values[i, k])
        values[i, k] = new


@njit(cache=True)
def joint_system(U, g, h, lam):
    """Damped Newton system for all term coefficients: G = U'g, H = U'diag(h)U + lam I."""
    n, E = U.shape
    G, H = np.zeros(E), np.zeros((E, E))
    for i in range(n):
        for e in range(E):
            G[e] += g[i] * U[i, e]
            for f in range(e + 1):
                H[e, f] += h[i] * U[i, e] * U[i, f]
    for e in range(E):
        H[e, e] += lam
        for f in range(e):
            H[f, e] = H[e, f]
    return G, H
