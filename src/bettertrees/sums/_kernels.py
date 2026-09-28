"""Numba kernels of the tree sums (Newton steps on binned histograms).

Same rule as ``bettertrees.kernels``: every kernel that calls another kernel
lives in this file (``cache=True`` is invalidated per file), no fastmath.

Conventions:
- uint8 bins, NaN (or an ineligible row) in bin 0; ``nb[j]`` bins used by j;
- a cut ``t`` sends bins ``<= t`` left (so bin 0, NaN, always goes left;
  ``t = 0`` is "NaN vs the rest");
- histograms have three channels: G = sum of g, H = sum of h, W = sum of weights;
- leaf score: G^2 / (H + lambda); gain = children - parent (>= 0 if accepted);
- a child is valid only with W >= ``min_weight``.

The second-order gain and leaf value are those of XGBoost (Chen & Guestrin,
KDD 2016). The exhaustive depth-2 search over sorted bins is in the spirit of
the specialized depth-two subroutines of MurTree (Demirović et al., JMLR 2022)
and ConTree (Briţa, van der Linden, Demirović, AAAI 2025), adapted from
misclassification on thresholds to the Newton log-loss gain on histogram bins.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def logistic_grad_hess(y, margin, w):
    """Gradient and Hessian of the log-loss with target y in [0, 1] (soft targets allowed)."""
    n = len(y)
    g = np.empty(n)
    h = np.empty(n)
    for i in range(n):
        m = margin[i]
        if m >= 0:
            p = 1.0 / (1.0 + np.exp(-m))
        else:
            e = np.exp(m)
            p = e / (1.0 + e)
        g[i] = w[i] * (p - y[i])
        h[i] = w[i] * max(p * (1.0 - p), 1e-16)
    return g, h


@njit(cache=True)
def _score(G, H, lam):
    return G * G / (H + lam)


@njit(cache=True, parallel=True)
def hist_1d(Xb, g, h, w, B):
    """Histograma (p, B, 3) de (G, H, W) por feature e bin."""
    n, p = Xb.shape
    out = np.zeros((p, B, 3))
    for j in prange(p):
        for i in range(n):
            b = Xb[i, j]
            out[j, b, 0] += g[i]
            out[j, b, 1] += h[i]
            out[j, b, 2] += w[i]
    return out


@njit(cache=True)
def _best_cut_hist(hist, nb, lam, min_weight):
    """Best cut of a histogram (nb, 3); returns (gain, t) or (0, -1)."""
    G = 0.0
    H = 0.0
    W = 0.0
    for b in range(nb):
        G += hist[b, 0]
        H += hist[b, 1]
        W += hist[b, 2]
    parent = _score(G, H, lam)
    best_gain = 0.0
    best_t = -1
    GL = 0.0
    HL = 0.0
    WL = 0.0
    for t in range(nb - 1):
        GL += hist[t, 0]
        HL += hist[t, 1]
        WL += hist[t, 2]
        if min_weight > WL or min_weight > W - WL:
            continue
        gain = _score(GL, HL, lam) + _score(G - GL, H - HL, lam) - parent
        if gain > best_gain:
            best_gain = gain
            best_t = t
    return best_gain, best_t


@njit(cache=True)
def best_cut_1d(hist, nb, lam, min_weight):
    """Best cut per feature: (gain[p], t[p]); t = -1 when no cut is valid."""
    p = hist.shape[0]
    gains = np.zeros(p)
    cuts = np.full(p, -1, dtype=np.int64)
    for j in range(p):
        gains[j], cuts[j] = _best_cut_hist(hist[j], nb[j], lam, min_weight)
    return gains, cuts


@njit(cache=True)
def _best_quadrant_hist(h2, na, nb, lam, min_weight):
    """Best pair of cuts (ta, tb) with 4 cells; (gain, ta, tb)."""
    # 2D prefix: P[a, b] = sum of the cells with bin_i <= a and bin_j <= b
    P = np.zeros((na, nb, 3))
    for a in range(na):
        for b in range(nb):
            for c in range(3):
                v = h2[a, b, c]
                if a > 0:
                    v += P[a - 1, b, c]
                if b > 0:
                    v += P[a, b - 1, c]
                if a > 0 and b > 0:
                    v -= P[a - 1, b - 1, c]
                P[a, b, c] = v
    T = P[na - 1, nb - 1]
    parent = _score(T[0], T[1], lam)
    best_gain = 0.0
    best_a = -1
    best_b = -1
    for ta in range(na - 1):
        for tb in range(nb - 1):
            ll0 = P[ta, tb, 0]
            ll1 = P[ta, tb, 1]
            ll2 = P[ta, tb, 2]
            lr0 = P[ta, nb - 1, 0] - ll0
            lr1 = P[ta, nb - 1, 1] - ll1
            lr2 = P[ta, nb - 1, 2] - ll2
            rl0 = P[na - 1, tb, 0] - ll0
            rl1 = P[na - 1, tb, 1] - ll1
            rl2 = P[na - 1, tb, 2] - ll2
            rr0 = T[0] - ll0 - lr0 - rl0
            rr1 = T[1] - ll1 - lr1 - rl1
            rr2 = T[2] - ll2 - lr2 - rl2
            if (ll2 < min_weight or lr2 < min_weight or rl2 < min_weight
                    or rr2 < min_weight):
                continue
            gain = (_score(ll0, ll1, lam) + _score(lr0, lr1, lam)
                    + _score(rl0, rl1, lam) + _score(rr0, rr1, lam) - parent)
            if gain > best_gain:
                best_gain = gain
                best_a = ta
                best_b = tb
    return best_gain, best_a, best_b


@njit(cache=True, parallel=True)
def quadrant_scores(Xb, g, h, w, nb, pairs, lam, min_weight):
    """FAST: best quadrant for each pair (i, j); (gain, ta, tb) per pair."""
    n = Xb.shape[0]
    k = pairs.shape[0]
    gains = np.zeros(k)
    ta = np.full(k, -1, dtype=np.int64)
    tb = np.full(k, -1, dtype=np.int64)
    for q in prange(k):
        fi = pairs[q, 0]
        fj = pairs[q, 1]
        h2 = np.zeros((nb[fi], nb[fj], 3))
        for r in range(n):
            a = Xb[r, fi]
            b = Xb[r, fj]
            h2[a, b, 0] += g[r]
            h2[a, b, 1] += h[r]
            h2[a, b, 2] += w[r]
        gains[q], ta[q], tb[q] = _best_quadrant_hist(h2, nb[fi], nb[fj], lam,
                                                     min_weight)
    return gains, ta, tb


@njit(cache=True, parallel=True)
def combination_scores(X, g, h, w, pairs, kind, n_bins, lam, min_weight):
    """Best single cut on a combination of each pair, with its own bins.

    kind 0: log(x_i) - log(x_j) (ratio; requires both > 0);
    kind 1: x_i - x_j (difference; the caller is responsible for the units).
    Ineligible rows (NaN, non-positive for the ratio) go to bin 0. Returns
    (gain, threshold on z, eligible share) per pair; threshold NaN when there
    is no cut.
    """
    n = X.shape[0]
    k = pairs.shape[0]
    gains = np.zeros(k)
    thresholds = np.full(k, np.nan)
    valid_frac = np.zeros(k)
    for q in prange(k):
        fi = pairs[q, 0]
        fj = pairs[q, 1]
        z = np.empty(n)
        ok = np.zeros(n, dtype=np.bool_)
        nv = 0
        for r in range(n):
            a = X[r, fi]
            b = X[r, fj]
            if np.isnan(a) or np.isnan(b):
                continue
            if kind == 0:
                if a <= 0 or b <= 0:
                    continue
                z[r] = np.log(a) - np.log(b)
            else:
                z[r] = a - b
            ok[r] = True
            nv += 1
        valid_frac[q] = nv / n
        if nv < 2:
            continue
        zs = np.empty(nv)
        c = 0
        for r in range(n):
            if ok[r]:
                zs[c] = z[r]
                c += 1
        zs.sort()
        # quantile cuts, without repeats
        edges = np.empty(n_bins - 1)
        ne = 0
        for s in range(1, n_bins):
            v = zs[(s * nv) // n_bins]
            if (ne == 0 or v > edges[ne - 1]) and v < zs[nv - 1]:
                edges[ne] = v
                ne += 1
        nbq = ne + 2  # bin 0 ineligible, 1..ne+1 finite
        hq = np.zeros((nbq, 3))
        for r in range(n):
            if ok[r]:
                lo = 0
                hi = ne
                v = z[r]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if edges[mid] < v:
                        lo = mid + 1
                    else:
                        hi = mid
                b = lo + 1
            else:
                b = 0
            hq[b, 0] += g[r]
            hq[b, 1] += h[r]
            hq[b, 2] += w[r]
        gain, t = _best_cut_hist(hq, nbq, lam, min_weight)
        gains[q] = gain
        if t >= 1:
            thresholds[q] = edges[t - 1]
        elif t == 0:
            thresholds[q] = -np.inf  # only "ineligible vs the rest"
    return gains, thresholds, valid_frac


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


@njit(cache=True)
def depth2_leaf_ids(Xb, f1, t1, fl, tl, fr, tr):
    """Leaf (0..3) of each row; a child without a cut (f = -1) uses the first leaf of its side."""
    n = Xb.shape[0]
    ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        if Xb[i, f1] <= t1:
            ids[i] = 0 if (fl < 0 or Xb[i, fl] <= tl) else 1
        else:
            ids[i] = 2 if (fr < 0 or Xb[i, fr] <= tr) else 3
    return ids


@njit(cache=True)
def newton_leaf_values(ids, g, h, n_leaves, lam):
    """Newton value per leaf: -G / (H + lambda)."""
    G = np.zeros(n_leaves)
    H = np.zeros(n_leaves)
    for i in range(len(ids)):
        G[ids[i]] += g[i]
        H[ids[i]] += h[i]
    out = np.zeros(n_leaves)
    for k in range(n_leaves):
        out[k] = -G[k] / (H[k] + lam)
    return out


# ---------------------------------------------------------------- profundidade 3

@njit(cache=True)
def _pair_cell(L, T, mode, f2, b2, f3, b3, c):
    """Cell of the side's pair tensor: mode 0 = L, mode 1 = T - L."""
    if mode == 0:
        return L[f2, b2, f3, b3, c]
    return T[f2, b2, f3, b3, c] - L[f2, b2, f3, b3, c]


@njit(cache=True)
def _solve_d2_from_pairs(L, T, mode, nb, lam, min_weight):
    """Optimal depth-2 tree of one side given the pair tensor S[f2, b2, f3, b3, (G, H, W)].

    Returns (gain over the side as a leaf, f2, t2, fl, tl, fr, tr); f2 = -1 when
    no cut is worth it (the side stays a leaf).
    """
    p = L.shape[0]
    B = L.shape[1]
    # marginal of the side per feature (via f2 = 0: sum over b2)
    M = np.zeros((p, B, 3))
    for b2 in range(nb[0]):
        for f3 in range(p):
            for b3 in range(nb[f3]):
                for c in range(3):
                    M[f3, b3, c] += _pair_cell(L, T, mode, 0, b2, f3, b3, c)
    G = 0.0
    H = 0.0
    W = 0.0
    for b3 in range(nb[0]):
        G += M[0, b3, 0]
        H += M[0, b3, 1]
        W += M[0, b3, 2]
    parent = _score(G, H, lam)
    best = 0.0
    out = np.full(6, -1, dtype=np.int64)
    left = np.zeros((p, B, 3))
    right = np.zeros((p, B, 3))
    for f2 in range(p):
        left[:] = 0.0
        right[:] = M
        GL = 0.0
        HL = 0.0
        WL = 0.0
        for t2 in range(nb[f2] - 1):
            for f3 in range(p):
                for b3 in range(nb[f3]):
                    for c in range(3):
                        v = _pair_cell(L, T, mode, f2, t2, f3, b3, c)
                        left[f3, b3, c] += v
                        right[f3, b3, c] -= v
            GL = 0.0
            HL = 0.0
            WL = 0.0
            for b3 in range(nb[0]):
                GL += left[0, b3, 0]
                HL += left[0, b3, 1]
                WL += left[0, b3, 2]
            if min_weight > WL or min_weight > W - WL:
                continue
            gain = _score(GL, HL, lam) + _score(G - GL, H - HL, lam) - parent
            bl = 0.0
            fl = -1
            tl = -1
            br = 0.0
            fr = -1
            tr = -1
            for f3 in range(p):
                gl, cl = _best_cut_hist(left[f3], nb[f3], lam, min_weight)
                if gl > bl:
                    bl = gl
                    fl = f3
                    tl = cl
                gr, cr = _best_cut_hist(right[f3], nb[f3], lam, min_weight)
                if gr > br:
                    br = gr
                    fr = f3
                    tr = cr
            gain += bl + br
            if gain > best:
                best = gain
                out[0] = f2
                out[1] = t2
                out[2] = fl
                out[3] = tl
                out[4] = fr
                out[5] = tr
    return best, out


@njit(cache=True, parallel=True)
def best_depth3(Xb, g, h, w, nb, lam, min_weight):
    """Optimal depth-3 Newton tree (exhaustive search over the bins).

    For each root (f1, t1), each child gets the optimal depth-2 tree of its
    side (or stays a leaf). Pair tensor (p, B, p, B, 3) per thread: use with
    small p and B (the caller limits the features). Returns, per f1,
    (gain, t1, left[6], right[6]) in the format of ``_solve_d2_from_pairs``.
    """
    n, p = Xb.shape
    B = 0
    for j in range(p):
        if nb[j] > B:
            B = nb[j]
    T = np.zeros((p, B, p, B, 3))
    for f2 in prange(p):
        for i in range(n):
            b2 = Xb[i, f2]
            for f3 in range(p):
                b3 = Xb[i, f3]
                T[f2, b2, f3, b3, 0] += g[i]
                T[f2, b2, f3, b3, 1] += h[i]
                T[f2, b2, f3, b3, 2] += w[i]
    Gt = 0.0
    Ht = 0.0
    Wt = 0.0
    for i in range(n):
        Gt += g[i]
        Ht += h[i]
        Wt += w[i]
    parent = _score(Gt, Ht, lam)
    res_gain = np.zeros(p)
    res = np.full((p, 13), -1, dtype=np.int64)
    for f1 in prange(p):
        # rows grouped by the bin of f1 (counting sort)
        counts = np.zeros(nb[f1] + 1, dtype=np.int64)
        for i in range(n):
            counts[Xb[i, f1] + 1] += 1
        for b in range(nb[f1]):
            counts[b + 1] += counts[b]
        order = np.empty(n, dtype=np.int64)
        fill = counts.copy()
        for i in range(n):
            b = Xb[i, f1]
            order[fill[b]] = i
            fill[b] += 1
        L = np.zeros((p, B, p, B, 3))
        GL = 0.0
        HL = 0.0
        WL = 0.0
        best = 0.0
        for t1 in range(nb[f1] - 1):
            for k in range(counts[t1], counts[t1 + 1]):
                i = order[k]
                GL += g[i]
                HL += h[i]
                WL += w[i]
                for f2 in range(p):
                    b2 = Xb[i, f2]
                    for f3 in range(p):
                        b3 = Xb[i, f3]
                        L[f2, b2, f3, b3, 0] += g[i]
                        L[f2, b2, f3, b3, 1] += h[i]
                        L[f2, b2, f3, b3, 2] += w[i]
            if min_weight > WL or Wt - WL < min_weight:
                continue
            root = _score(GL, HL, lam) + _score(Gt - GL, Ht - HL, lam) - parent
            gl, sl = _solve_d2_from_pairs(L, T, 0, nb, lam, min_weight)
            gr, sr = _solve_d2_from_pairs(L, T, 1, nb, lam, min_weight)
            gain = root + gl + gr
            if gain > best:
                best = gain
                res[f1, 0] = t1
                for c in range(6):
                    res[f1, 1 + c] = sl[c]
                    res[f1, 7 + c] = sr[c]
        res_gain[f1] = best
    return res_gain, res


# ---------------------------------------------------------------- small trees

@njit(cache=True)
def small_tree_leaf_ids(Xb, feature, threshold, left, right):
    """Leaf node of each row in a small tree (bins; x <= t goes left)."""
    n = Xb.shape[0]
    ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        node = 0
        while left[node] != -1:
            if Xb[i, feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        ids[i] = node
    return ids


@njit(cache=True, parallel=True)
def node_hist(Xb, g, h, w, node_of_row, n_nodes, B):
    """Histogram (n_nodes, p, B, 3) of the rows grouped by node."""
    n, p = Xb.shape
    out = np.zeros((n_nodes, p, B, 3))
    for j in prange(p):
        for i in range(n):
            k = node_of_row[i]
            if k < 0:
                continue
            b = Xb[i, j]
            out[k, j, b, 0] += g[i]
            out[k, j, b, 1] += h[i]
            out[k, j, b, 2] += w[i]
    return out


@njit(cache=True, parallel=True)
def best_depth2(Xb, g, h, w, nb, lam, min_weight):
    """Optimal depth-2 tree; BIT-FOR-BIT the same result as ``best_depth2_reference``.

    Swaps the loop order: for each root f1, one pair (f1, f2) at a time with a
    (nb[f1], B, 3) histogram that fits in cache, over column-major X. Each cell
    sums the rows in the same order (increasing i) and the cumulative sums per
    t1 follow the same order, so gains and tie-breaking (increasing f2 and t1,
    strict ``>``) are identical.
    """
    n, p = Xb.shape
    B = 0
    for j in range(p):
        if nb[j] > B:
            B = nb[j]
    XT = np.empty((p, n), dtype=np.uint8)
    for i in range(n):
        for j in range(p):
            XT[j, i] = Xb[i, j]
    Gt = 0.0
    Ht = 0.0
    for i in range(n):
        Gt += g[i]
        Ht += h[i]
    parent = _score(Gt, Ht, lam)
    res_gain = np.zeros(p)
    res = np.full((p, 5), -1, dtype=np.int64)
    for f1 in prange(p):
        n1 = nb[f1]
        col1 = XT[f1]
        bl = np.zeros(n1)
        fl = np.full(n1, -1, dtype=np.int64)
        tl = np.full(n1, -1, dtype=np.int64)
        br = np.zeros(n1)
        fr = np.full(n1, -1, dtype=np.int64)
        tr = np.full(n1, -1, dtype=np.int64)
        root = np.zeros(n1)
        valid = np.zeros(n1, dtype=np.bool_)
        h2 = np.zeros((n1, B, 3))
        total = np.zeros((B, 3))
        left = np.zeros((B, 3))
        right = np.zeros((B, 3))
        for f2 in range(p):
            m2 = nb[f2]
            col2 = XT[f2]
            h2[:] = 0.0
            for i in range(n):
                a = col1[i]
                b = col2[i]
                h2[a, b, 0] += g[i]
                h2[a, b, 1] += h[i]
                h2[a, b, 2] += w[i]
            total[:] = 0.0
            for b1 in range(n1):
                for b2 in range(B):
                    for c in range(3):
                        total[b2, c] += h2[b1, b2, c]
            left[:] = 0.0
            for t1 in range(n1 - 1):
                for b2 in range(B):
                    for c in range(3):
                        left[b2, c] += h2[t1, b2, c]
                if f2 == 0:
                    # statistics of the root cut, as in the reference (via f2 = 0)
                    GL = 0.0
                    HL = 0.0
                    WL = 0.0
                    WR = 0.0
                    for b2 in range(nb[0]):
                        GL += left[b2, 0]
                        HL += left[b2, 1]
                        WL += left[b2, 2]
                        WR += total[b2, 2] - left[b2, 2]
                    if min_weight <= WL and min_weight <= WR:
                        valid[t1] = True
                        root[t1] = (_score(GL, HL, lam) + _score(Gt - GL, Ht - HL, lam)
                                    - parent)
                if not valid[t1]:
                    continue
                for b2 in range(B):
                    for c in range(3):
                        right[b2, c] = total[b2, c] - left[b2, c]
                gl, cl = _best_cut_hist(left, m2, lam, min_weight)
                if gl > bl[t1]:
                    bl[t1] = gl
                    fl[t1] = f2
                    tl[t1] = cl
                gr, cr = _best_cut_hist(right, m2, lam, min_weight)
                if gr > br[t1]:
                    br[t1] = gr
                    fr[t1] = f2
                    tr[t1] = cr
        best = 0.0
        for t1 in range(n1 - 1):
            if not valid[t1]:
                continue
            gain = root[t1] + bl[t1] + br[t1]
            if gain > best:
                best = gain
                res[f1, 0] = t1
                res[f1, 1] = fl[t1]
                res[f1, 2] = tl[t1]
                res[f1, 3] = fr[t1]
                res[f1, 4] = tr[t1]
        res_gain[f1] = best
    return res_gain, res


# ---------------------------------------------------------------- L1 logistic path

@njit(cache=True)
def _soft(u, t):
    if u > t:
        return u - t
    if u < -t:
        return u + t
    return 0.0


@njit(cache=True)
def _col_dot(indptr, indices, j, v):
    acc = 0.0
    for q in range(indptr[j], indptr[j + 1]):
        acc += v[indices[q]]
    return acc


@njit(cache=True)
def l1_logistic_path(indptr, indices, scale, y, lambdas, cost, max_cost, tol, max_outer,
                     max_sweeps, beta_init, b0_init):
    """L1 logistic regression path (glmnet style) over sparse binary columns.

    Column j equals ``scale[j]`` on rows ``indices[indptr[j]:indptr[j+1]]`` and
    0 elsewhere. Objective at each lambda: (1/n) * sum log-loss + lambda *
    sum |beta_j|, unpenalized intercept. For each lambda (decreasing, warm
    start): IRLS outside, coordinate descent inside, only over the strong set
    (strong rules of Tibshirani et al. 2012: |grad_j| >= 2 lambda_k -
    lambda_{k-1}, plus the active set). At convergence, KKT is checked on the
    discarded columns and violators are re-admitted, so the solution is that of
    the full problem. Stops when the active cost (sum of cost_j over beta_j !=
    0) exceeds ``max_cost``. ``beta_init``/``b0_init``: starting point (warm
    start of a sub-path); ``b0_init`` NaN = start from zero with the intercept
    of the mean rate. Returns (beta per lambda, intercept per lambda, number of
    lambdas solved).
    """
    m = len(indptr) - 1
    n = len(y)
    K = len(lambdas)
    betas = np.zeros((K, m))
    b0s = np.zeros(K)
    beta = np.zeros(m)
    ybar = 0.0
    for i in range(n):
        ybar += y[i]
    ybar /= n
    ybar = min(max(ybar, 1e-6), 1 - 1e-6)
    b0 = np.log(ybar / (1 - ybar))
    active = np.zeros(m, dtype=np.bool_)
    if not np.isnan(b0_init):
        b0 = b0_init
        for j in range(m):
            beta[j] = beta_init[j]
            active[j] = beta[j] != 0.0
    eta = np.full(n, b0)
    for j in range(m):
        if beta[j] != 0.0:
            for q in range(indptr[j], indptr[j + 1]):
                eta[indices[q]] += scale[j] * beta[j]
    r = np.empty(n)
    wt = np.empty(n)
    xw2 = np.zeros(m)
    strong = np.zeros(m, dtype=np.bool_)
    done = 0
    thr_prev = n * lambdas[0]
    for k in range(K):
        thr = n * lambdas[k]
        for outer in range(max_outer):
            wsum = 0.0
            for i in range(n):
                e = eta[i]
                pr = 1.0 / (1.0 + np.exp(-e)) if e >= 0 else np.exp(e) / (1.0 + np.exp(e))
                wi = max(pr * (1 - pr), 1e-5)
                wt[i] = wi
                r[i] = y[i] - pr  # = w * (z - eta) with z the working response
                wsum += wi
            if outer == 0:
                cut = 2 * thr - thr_prev
                for j in range(m):
                    strong[j] = active[j] or (
                        abs(scale[j] * _col_dot(indptr, indices, j, r)) >= cut)
            for j in range(m):
                if strong[j]:
                    xw2[j] = _col_dot(indptr, indices, j, wt) * scale[j] * scale[j]
            max_eta_change = 0.0
            while True:
                sweeps = 0
                only_active = False
                while sweeps < max_sweeps:
                    sweeps += 1
                    max_delta = 0.0
                    entered = False
                    for j in range(m):
                        if not strong[j] or (only_active and not active[j]) or xw2[j] <= 0:
                            continue
                        u = scale[j] * _col_dot(indptr, indices, j, r) + xw2[j] * beta[j]
                        new = _soft(u, thr) / xw2[j]
                        d = new - beta[j]
                        if d != 0.0:
                            if beta[j] == 0.0:
                                entered = True
                            beta[j] = new
                            active[j] = new != 0.0
                            step = scale[j] * d
                            for q in range(indptr[j], indptr[j + 1]):
                                i = indices[q]
                                r[i] -= wt[i] * step
                                eta[i] += step
                            if abs(step) > max_eta_change:
                                max_eta_change = abs(step)
                            change = d * d * xw2[j]
                            if change > max_delta:
                                max_delta = change
                    s = 0.0
                    for i in range(n):
                        s += r[i]
                    d0 = s / wsum
                    b0 += d0
                    for i in range(n):
                        r[i] -= wt[i] * d0
                        eta[i] += d0
                    if max_delta < tol:
                        if not only_active and not entered:
                            break
                        only_active = False  # ativos convergiram: varre o conjunto forte
                    else:
                        only_active = True
                # KKT on the columns outside the strong set
                violated = False
                for j in range(m):
                    if not strong[j] and abs(scale[j] * _col_dot(indptr, indices, j, r)) > thr:
                        strong[j] = True
                        xw2[j] = _col_dot(indptr, indices, j, wt) * scale[j] * scale[j]
                        violated = True
                if not violated:
                    break
            if max_eta_change < 1e-6:
                break
        thr_prev = thr
        betas[k] = beta
        b0s[k] = b0
        done = k + 1
        c = 0.0
        for j in range(m):
            if beta[j] != 0.0:
                c += cost[j]
        if c > max_cost:
            break
    return betas, b0s, done


@njit(cache=True, parallel=True)
def reuse_gains(ids, n_terms, g, h, lam):
    """Newton gain of re-boosting each existing term (``CompactTreeBooster``).

    ``ids`` is (capacity, n) int8 with the leaf (0..3) of every row in each of
    the first ``n_terms`` terms; returns gain[n_terms] = Σ_leaf G²/(H+λ) - G²/(H+λ).
    """
    n = ids.shape[1]
    out = np.zeros(n_terms)
    for s in prange(n_terms):
        G = np.zeros(4)
        H = np.zeros(4)
        for i in range(n):
            k = ids[s, i]
            G[k] += g[i]
            H[k] += h[i]
        gt = 0.0
        ht = 0.0
        acc = 0.0
        for k in range(4):
            acc += G[k] * G[k] / (H[k] + lam)
            gt += G[k]
            ht += H[k]
        out[s] = acc - gt * gt / (ht + lam)
    return out
