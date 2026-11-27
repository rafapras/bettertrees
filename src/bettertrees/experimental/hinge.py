"""HingeFIGSClassifier: sums of 1-knot continuous piecewise-linear hinges and trees."""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..sums._common import base_margin, binned, fit_inputs, grad_hess, predict_input, rebin
from ..sums._kernels import best_leaf_cuts_T
from ..sums.explain import InterpretableSumMixin, _fmt, _sql_ident
from ..sums.smalltrees import SmallTree


@njit(cache=True)
def _solve_hinge_3x3(H00, H01, H11, H02, H22, G0, G1, G2, lam):
    """Exact closed-form solve of the arrowhead 3x3 hinge Newton system with L2=lam."""
    reg = lam if lam > 1e-6 else 1e-6
    s1 = H11 + reg
    s2 = H22 + reg
    inv1 = 1.0 / s1
    inv2 = 1.0 / s2
    # By Cauchy-Schwarz, H00 - H01^2/s1 - H02^2/s2 >= 0, so d0 >= reg always.
    d0 = (H00 + reg) - (H01 * H01) * inv1 - (H02 * H02) * inv2
    if d0 < reg:
        d0 = reg
    b0 = (-G0 + G1 * H01 * inv1 + G2 * H02 * inv2) / d0
    b1 = (-G1 - H01 * b0) * inv1
    b2 = (-G2 - H02 * b0) * inv2
    return b0, b1, b2


@njit(cache=True, parallel=True)
def _best_hinge_kernel(XT, ZT, nb, taus_mat, used_hinges, g, h, w, lam, min_weight):
    """Column-major L1-resident parallel Numba scan for the best unused 1-knot hinge."""
    p, n = XT.shape
    Gt = 0.0
    Ht = 0.0
    for i in range(n):
        Gt += g[i]
        Ht += h[i]
    parent = (Gt * Gt) / (Ht + lam)

    feat_gain = np.zeros(p, dtype=np.float64)
    feat_t = np.full(p, -1, dtype=np.int64)
    feat_bnan = np.zeros(p, dtype=np.float64)
    feat_b0 = np.zeros(p, dtype=np.float64)
    feat_b1 = np.zeros(p, dtype=np.float64)
    feat_b2 = np.zeros(p, dtype=np.float64)

    for j in prange(p):
        nb_j = nb[j]
        m = nb_j - 2
        if m < 1:
            continue
        xcol = XT[j]
        zcol = ZT[j]
        S0 = np.zeros(nb_j, dtype=np.float64)
        S1 = np.zeros(nb_j, dtype=np.float64)
        S2 = np.zeros(nb_j, dtype=np.float64)
        R0 = np.zeros(nb_j, dtype=np.float64)
        R1 = np.zeros(nb_j, dtype=np.float64)
        Wb = np.zeros(nb_j, dtype=np.float64)
        for i in range(n):
            b = xcol[i]
            gi = g[i]
            hi = h[i]
            Wb[b] += w[i]
            R0[b] += gi
            S0[b] += hi
            if b > 0:
                zij = zcol[i]
                hz = hi * zij
                S1[b] += hz
                S2[b] += hz * zij
                R1[b] += gi * zij

        W_nan = Wb[0]
        G_nan = R0[0]
        H_nan = S0[0]
        W_fin = 0.0
        T_S0 = 0.0
        T_S1 = 0.0
        T_S2 = 0.0
        T_R0 = 0.0
        T_R1 = 0.0
        for b in range(1, nb_j):
            W_fin += Wb[b]
            T_S0 += S0[b]
            T_S1 += S1[b]
            T_S2 += S2[b]
            T_R0 += R0[b]
            T_R1 += R1[b]
        if W_fin < 2.0 * min_weight:
            continue

        if W_nan >= min_weight:
            b_nan = -G_nan / (H_nan + lam)
            nan_score = (G_nan * G_nan) / (H_nan + lam)
        else:
            b_nan = 0.0
            nan_score = 0.0

        best_g = 0.0
        best_t_j = -1
        best_b0_j = 0.0
        best_b1_j = 0.0
        best_b2_j = 0.0
        WL = 0.0
        L0 = 0.0
        L1 = 0.0
        L2 = 0.0
        RL0 = 0.0
        RL1 = 0.0
        for t in range(m):
            b = t + 1
            WL += Wb[b]
            L0 += S0[b]
            L1 += S1[b]
            L2 += S2[b]
            RL0 += R0[b]
            RL1 += R1[b]
            if used_hinges[j, t] or WL < min_weight or (W_fin - WL) < min_weight:
                continue
            U0 = T_S0 - L0
            U1 = T_S1 - L1
            U2 = T_S2 - L2
            RU0 = T_R0 - RL0
            RU1 = T_R1 - RL1

            tau = taus_mat[j, t]
            tau2 = tau * tau
            H00 = L0 + U0
            H01 = L1 - tau * L0
            H11 = L2 - 2.0 * tau * L1 + tau2 * L0
            H02 = U1 - tau * U0
            H22 = U2 - 2.0 * tau * U1 + tau2 * U0
            G0 = RL0 + RU0
            G1 = RL1 - tau * RL0
            G2 = RU1 - tau * RU0

            b0, b1, b2 = _solve_hinge_3x3(H00, H01, H11, H02, H22, G0, G1, G2, lam)
            gain = -(G0 * b0 + G1 * b1 + G2 * b2) + nan_score - parent
            if gain > best_g:
                best_g = gain
                best_t_j = t
                best_b0_j = b0
                best_b1_j = b1
                best_b2_j = b2

        feat_gain[j] = best_g
        feat_t[j] = best_t_j
        feat_bnan[j] = b_nan
        feat_b0[j] = best_b0_j
        feat_b1[j] = best_b1_j
        feat_b2[j] = best_b2_j

    best_gain = 0.0
    best_j = -1
    for j in range(p):
        if feat_gain[j] > best_gain:
            best_gain = feat_gain[j]
            best_j = j
    if best_j < 0:
        return 0.0, -1, -1, 0.0, 0.0, 0.0, 0.0
    return (
        best_gain,
        best_j,
        feat_t[best_j],
        feat_bnan[best_j],
        feat_b0[best_j],
        feat_b1[best_j],
        feat_b2[best_j],
    )


@njit(cache=True)
def _clip_step(v, cap):
    if cap <= 0.0:
        return v
    if v > cap:
        return cap
    if v < -cap:
        return -cap
    return v


@njit(cache=True)
def _step_hinge_kernel(
    xb_col, z_col, tau, target, margin, contrib, w, beta, lam, min_weight, lr, cap
):
    """Zero-allocation Newton step and in-place margin/contrib update for one hinge."""
    n = len(target)
    W_nan = 0.0
    G_nan = 0.0
    H_nan = 0.0
    H00 = 0.0
    H01 = 0.0
    H11 = 0.0
    H02 = 0.0
    H22 = 0.0
    G0 = 0.0
    G1 = 0.0
    G2 = 0.0

    for i in range(n):
        m = margin[i]
        if m >= 0.0:
            p = 1.0 / (1.0 + np.exp(-m))
        else:
            e = np.exp(m)
            p = e / (1.0 + e)
        wi = w[i]
        gi = wi * (p - target[i])
        hi = wi * (p * (1.0 - p))
        if hi < wi * 1e-16:
            hi = wi * 1e-16
        if xb_col[i] == 0:
            W_nan += wi
            G_nan += gi
            H_nan += hi
        else:
            diff = z_col[i] - tau
            H00 += hi
            G0 += gi
            if diff < 0.0:
                H01 += hi * diff
                H11 += hi * diff * diff
                G1 += gi * diff
            elif diff > 0.0:
                H02 += hi * diff
                H22 += hi * diff * diff
                G2 += gi * diff

    reg = lam if lam > 1e-6 else 1e-6
    if W_nan >= min_weight:
        d_nan = -(G_nan + reg * beta[0]) / max(H_nan + reg, 1e-12)
    else:
        d_nan = -beta[0]

    g0_r = G0 + reg * beta[1]
    g1_r = G1 + reg * beta[2]
    g2_r = G2 + reg * beta[3]
    d0, d1, d2 = _solve_hinge_3x3(H00, H01, H11, H02, H22, g0_r, g1_r, g2_r, reg)

    beta[0] += lr * _clip_step(d_nan, cap)
    beta[1] += lr * _clip_step(d0, cap)
    beta[2] += lr * _clip_step(d1, cap)
    beta[3] += lr * _clip_step(d2, cap)

    bnan = beta[0]
    b0 = beta[1]
    b1 = beta[2]
    b2 = beta[3]
    for i in range(n):
        if xb_col[i] == 0:
            nc = bnan
        else:
            diff = z_col[i] - tau
            if diff < 0.0:
                nc = b0 + b1 * diff
            else:
                nc = b0 + b2 * diff
        margin[i] += nc - contrib[i]
        contrib[i] = nc


@njit(cache=True)
def _eval_hinge_z_kernel(xb_col, z_col, tau, beta):
    n = len(xb_col)
    out = np.empty(n, dtype=np.float64)
    bnan, b0, b1, b2 = beta[0], beta[1], beta[2], beta[3]
    for i in range(n):
        if xb_col[i] == 0:
            out[i] = bnan
        else:
            diff = z_col[i] - tau
            out[i] = b0 + (b1 * diff if diff < 0.0 else b2 * diff)
    return out


@dataclass
class HingeTerm:
    feature: int
    bin_cut: int
    threshold_val: float
    mu: float
    sd: float
    beta: np.ndarray  # [b_nan, b_0, b_1, b_2]
    x_min: float = -np.inf
    x_max: float = np.inf
    clamp: bool = True

    def evaluate(self, X):
        col = X[:, self.feature]
        nan_mask = np.isnan(col)
        fin = np.where(nan_mask, 0.0, col)
        if self.clamp:
            fin = np.clip(fin, self.x_min, self.x_max)
        z = np.where(nan_mask, 0.0, (fin - self.mu) / self.sd)
        tau = (self.threshold_val - self.mu) / self.sd
        diff = z - tau
        fin_val = (
            self.beta[1]
            + self.beta[2] * np.minimum(diff, 0.0)
            + self.beta[3] * np.maximum(diff, 0.0)
        )
        return np.where(nan_mask, self.beta[0], fin_val)

    def raw_pieces(self):
        """Return (b_nan, a_left, slope_left, a_right, slope_right) in raw x units."""
        b_nan, b0, b1, b2 = (float(v) for v in self.beta)
        s_left = b1 / self.sd
        s_right = b2 / self.sd
        a_left = b0 - s_left * self.threshold_val
        a_right = b0 - s_right * self.threshold_val
        return b_nan, a_left, s_left, a_right, s_right


class HingeFIGSClassifier(InterpretableSumMixin, ClassifierMixin, BaseEstimator):
    """Sum of continuous 1-knot piecewise-linear hinges (and optional axis trees)."""

    def __init__(
        self,
        *,
        max_splits=8,
        lam="auto",
        min_weight=20.0,
        max_bins=32,
        learning_rate=0.5,
        backfit_sweeps=2,
        allow_tree_splits=True,
        hinge_cost=1,
        clamp_range=True,
        clip_quantile=0.01,
        max_delta_step=4.0,
    ):
        self.max_splits = max_splits
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.backfit_sweeps = backfit_sweeps
        self.allow_tree_splits = allow_tree_splits
        self.hinge_cost = hinge_cost
        self.clamp_range = clamp_range
        self.clip_quantile = clip_quantile
        self.max_delta_step = max_delta_step

    def _fit_standardization(self, X, w):
        n, p = X.shape
        x_min = np.zeros(p, dtype=np.float64)
        x_max = np.zeros(p, dtype=np.float64)
        mu = np.zeros(p, dtype=np.float64)
        sd = np.ones(p, dtype=np.float64)
        q = float(self.clip_quantile) if self.clamp_range else 0.0
        ZT = np.zeros((p, n), dtype=np.float64)
        for j in range(p):
            col = X[:, j]
            ok = ~np.isnan(col)
            if not ok.any():
                continue
            fin = col[ok]
            lo, hi = float(np.min(fin)), float(np.max(fin))
            if q > 0.0 and lo < hi:
                q_lo, q_hi = np.quantile(fin, [q, 1.0 - q])
                edges = self.bin_edges_[j]
                if len(edges):
                    q_lo = min(float(q_lo), float(edges[0]))
                    q_hi = max(float(q_hi), float(edges[-1]))
                if q_lo < q_hi:
                    lo, hi = float(q_lo), float(q_hi)
            x_min[j], x_max[j] = lo, hi
            vals = np.clip(fin, lo, hi) if self.clamp_range else fin
            wf = w[ok]
            mass = float(wf.sum())
            if mass > 0.0:
                mj = float(np.dot(wf, vals) / mass)
                vj = float(np.dot(wf, (vals - mj) ** 2) / mass)
                sj = float(np.sqrt(vj))
                mu[j] = mj
                sd[j] = sj if (np.isfinite(sj) and sj > 1e-12) else 1.0
                ZT[j, ok] = (vals - mj) / sd[j]
        self.x_min_ = x_min
        self.x_max_ = x_max
        self.mu_ = mu
        self.sd_ = sd
        return ZT

    def _build_taus_matrix(self, nb):
        p = len(nb)
        max_m = max(int(nb.max()) - 2, 1)
        taus = np.zeros((p, max_m), dtype=np.float64)
        for j, edges in enumerate(self.bin_edges_):
            m = len(edges)
            if m > 0:
                taus[j, :m] = (edges - self.mu_[j]) / self.sd_[j]
        return taus

    def _best_hinge(self, Xb, Z, nb, taus_mat, used_hinges, g, h, w):
        gain, j, t_pos, b_nan, b0, b1, b2 = _best_hinge_kernel(
            Xb, Z, nb, taus_mat, used_hinges, g, h, w, float(self.lam_), float(self.min_weight)
        )
        if j < 0 or t_pos < 0:
            return 0.0, None
        beta = np.array([b_nan, b0, b1, b2], dtype=np.float64)
        return float(gain), (
            int(j),
            int(t_pos) + 1,
            float(self.bin_edges_[int(j)][int(t_pos)]),
            float(self.mu_[int(j)]),
            float(self.sd_[int(j)]),
            beta,
        )

    def _step_tree(self, tree, ids, target, margin, contrib, w, cap):
        g, h = grad_hess(target, margin, w)
        L = len(tree.feature)
        G = np.bincount(ids, weights=g, minlength=L)
        H = np.bincount(ids, weights=h, minlength=L)
        for leaf in tree.leaves:
            step = -(G[leaf] + self.lam_ * tree.value[leaf]) / max(H[leaf] + self.lam_, 1e-12)
            if cap > 0.0:
                step = float(np.clip(step, -cap, cap))
            tree.value[leaf] += self.learning_rate_ * step
        new_c = tree.value[ids]
        margin += new_c - contrib
        return new_c, margin

    def fit(self, X, y, sample_weight=None, y_soft=None):
        if not isinstance(self.clamp_range, (bool, np.bool_)):
            raise ValueError("clamp_range must be a boolean.")
        q = float(self.clip_quantile)
        if not np.isfinite(q) or not 0.0 <= q < 0.5:
            raise ValueError("clip_quantile must be finite and in [0.0, 0.5).")
        cap = 0.0 if self.max_delta_step is None else float(self.max_delta_step)
        if not np.isfinite(cap) or cap < 0.0:
            raise ValueError("max_delta_step must be None or a finite number >= 0.")
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        n = len(X)
        self.classes_ = classes
        Xb, self.bin_edges_, nb = binned(X, self.max_bins)
        B = int(nb.max())
        ZT = self._fit_standardization(X, w)
        XT = np.ascontiguousarray(Xb.T)
        taus_mat = self._build_taus_matrix(nb)
        target = np.ascontiguousarray(target, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)

        self.lam_ = 2.0 * max(1, int(self.max_splits)) if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        self.base_margin_ = base_margin(target, w)
        margin = np.full(n, self.base_margin_, dtype=np.float64)

        trees, tree_contribs, tree_ids = [], [], []
        hinges, hinge_contribs = [], []
        used_hinges = np.zeros_like(taus_mat, dtype=np.bool_)
        root_ids = np.zeros(n, dtype=np.int64)
        root_leaves = np.zeros(1, dtype=np.int64)
        zero = np.zeros(n, dtype=np.float64)
        cost = int(self.hinge_cost)
        spent = 0

        while spent < int(self.max_splits):
            best_score = 0.0
            best_action = None

            if int(self.max_splits) - spent >= cost:
                g, h = grad_hess(target, margin, w)
                h_gain, h_spec = self._best_hinge(XT, ZT, nb, taus_mat, used_hinges, g, h, w)
                if h_spec is not None and h_gain / cost > best_score:
                    best_score = h_gain / cost
                    best_action = ("hinge", h_spec)

            if self.allow_tree_splits and spent + 1 <= int(self.max_splits):
                for k in [*range(len(trees)), None]:
                    if k is None:
                        ids, c_vec, leaves_arr, n_nodes = root_ids, zero, root_leaves, 1
                    else:
                        ids, c_vec = tree_ids[k], tree_contribs[k]
                        leaves_arr = np.asarray(trees[k].leaves, dtype=np.int64)
                        n_nodes = len(trees[k].feature)
                    gains_mat, cuts_mat = best_leaf_cuts_T(
                        XT, nb, target, w, margin, c_vec, ids, leaves_arr, n_nodes, B,
                        self.lam_, float(self.min_weight),
                    )
                    for li in range(len(leaves_arr)):
                        f = int(np.argmax(gains_mat[li]))
                        g_val = float(gains_mat[li, f])
                        if g_val > best_score:
                            best_score = g_val
                            best_action = ("tree", k, int(leaves_arr[li]), f, int(cuts_mat[li, f]))

            if best_action is None:
                break

            if best_action[0] == "hinge":
                j, b_cut, thr_val, mu, sd, beta = best_action[1]
                tau = (thr_val - mu) / sd
                used_hinges[j, b_cut - 1] = True
                init_beta = np.clip(beta, -cap, cap) if cap > 0.0 else beta
                term = HingeTerm(
                    j,
                    b_cut,
                    thr_val,
                    mu,
                    sd,
                    self.learning_rate_ * init_beta,
                    x_min=float(self.x_min_[j]),
                    x_max=float(self.x_max_[j]),
                    clamp=bool(self.clamp_range),
                )
                c_new = _eval_hinge_z_kernel(XT[j], ZT[j], tau, term.beta)
                hinges.append(term)
                hinge_contribs.append(c_new)
                margin += c_new
                spent += cost
            else:
                _, k, leaf, f, cut = best_action
                if k is None:
                    trees.append(SmallTree())
                    tree_contribs.append(np.zeros(n, dtype=np.float64))
                    tree_ids.append(None)
                    k = len(trees) - 1
                trees[k].split(leaf, f, cut)
                tree_ids[k] = trees[k].leaf_ids(Xb)
                tree_contribs[k], margin = self._step_tree(
                    trees[k], tree_ids[k], target, margin, tree_contribs[k], w, cap
                )
                spent += 1

            for _ in range(int(self.backfit_sweeps)):
                for k, tr in enumerate(trees):
                    tree_contribs[k], margin = self._step_tree(
                        tr, tree_ids[k], target, margin, tree_contribs[k], w, cap
                    )
                for m_idx, hg in enumerate(hinges):
                    fj = hg.feature
                    tau = (hg.threshold_val - hg.mu) / hg.sd
                    _step_hinge_kernel(
                        XT[fj],
                        ZT[fj],
                        tau,
                        target,
                        margin,
                        hinge_contribs[m_idx],
                        w,
                        hg.beta,
                        float(self.lam_),
                        float(self.min_weight),
                        self.learning_rate_,
                        cap,
                    )
                g_b, h_b = grad_hess(target, margin, w)
                d0 = -float(g_b.sum() / max(h_b.sum(), 1e-12))
                if cap > 0.0:
                    d0 = float(np.clip(d0, -cap, cap))
                self.base_margin_ += d0
                margin += d0

        self.trees_ = trees
        self.hinges_ = hinges
        return self

    @property
    def n_splits_(self):
        check_is_fitted(self, "trees_")
        return sum(t.n_splits for t in self.trees_) + int(self.hinge_cost) * len(self.hinges_)

    def predict_contributions(self, X):
        Xv = predict_input(self, X, "trees_")
        Xb = rebin(Xv, self.bin_edges_)
        cols = [tr.value[tr.leaf_ids(Xb)] for tr in self.trees_]
        cols.extend(hg.evaluate(Xv) for hg in self.hinges_)
        if not cols:
            return np.zeros((len(Xv), 0), dtype=np.float64)
        return np.column_stack(cols)

    def decision_function(self, X):
        c = self.predict_contributions(X)
        return self.base_margin_ + c.sum(axis=1)

    def rules(self, feature_names=None, precision=4):
        """Tree leaf rules plus piecewise-linear hinge formulas."""
        check_is_fitted(self, "trees_")
        out = super().rules(feature_names=feature_names, precision=precision)
        names = self._names(feature_names)
        for k, hg in enumerate(self.hinges_, 1):
            nm = names[hg.feature] if names is not None else f"x{hg.feature}"
            b_nan, a_l, s_l, a_r, s_r = hg.raw_pieces()
            thr = _fmt(hg.threshold_val, precision)
            tid = f"hinge_{k}"
            lhs = f"{_fmt(a_l, precision)} + ({_fmt(s_l, precision)})*{nm}"
            rhs = f"{_fmt(a_r, precision)} + ({_fmt(s_r, precision)})*{nm}"
            out.append((tid, [f"{nm} <= {thr}"], lhs))
            out.append((tid, [f"{nm} > {thr}"], rhs))
            if self.nan_features_[hg.feature]:
                out.append((tid, [f"{nm} is missing"], float(b_nan)))
        return out

    def shape_functions(self):
        """Main effects per feature, combining 1D trees and 1-knot hinges."""
        check_is_fitted(self, "trees_")
        out = super().shape_functions()
        for hg in self.hinges_:
            f = hg.feature
            edges = np.asarray(self.bin_edges_[f], dtype=np.float64)
            nb = len(edges) + 2
            vals, count = out.get(f, (np.zeros(nb, dtype=np.float64), 0))
            pts = np.empty(nb, dtype=np.float64)
            pts[0] = np.nan
            if len(edges) == 0:
                pts[1] = hg.threshold_val
            else:
                pts[1] = (
                    min(hg.x_min, float(edges[0]))
                    if np.isfinite(hg.x_min)
                    else float(edges[0])
                )
                if len(edges) > 1:
                    pts[2:-1] = 0.5 * (edges[:-1] + edges[1:])
                pts[-1] = (
                    max(hg.x_max, float(edges[-1]))
                    if np.isfinite(hg.x_max)
                    else float(edges[-1])
                )
            X_tmp = np.zeros((nb, self.n_features_in_), dtype=np.float64)
            X_tmp[:, f] = pts
            out[f] = (vals + hg.evaluate(X_tmp), count + 1)
        return out

    def plot_contributions(self, x, feature_names=None, max_terms=12, ax=None):
        """Waterfall of one prediction including both trees and hinges."""
        from ..sums.explain import _pyplot, leaf_order, leaf_rules

        plt = _pyplot()
        if hasattr(x, "to_frame"):
            x = x.to_frame().T.astype(float)
        elif not hasattr(x, "columns"):
            x = np.asarray(x, dtype=float).reshape(1, -1)
        contrib = self.predict_contributions(x)[0]
        xv = predict_input(self, x, "trees_")
        Xb = rebin(xv, self.bin_edges_)
        names = self._names(feature_names)
        nan = getattr(self, "nan_features_", None)
        labels = []
        for tree in self.trees_:
            leaf = int(tree.leaf_ids(Xb)[0])
            tr_rules = leaf_rules(tree, self.bin_edges_, names, nan, precision=4)
            labels.append(" and ".join(tr_rules[leaf_order(tree).index(leaf)][0]) or "always")
        for hg in self.hinges_:
            nm = names[hg.feature] if names is not None else f"x{hg.feature}"
            val = float(xv[0, hg.feature])
            if np.isnan(val):
                labels.append(f"hinge: {nm} is missing")
            else:
                op = "<=" if val <= hg.threshold_val else ">"
                labels.append(f"hinge: {nm}={_fmt(val, 4)} ({op} {_fmt(hg.threshold_val, 4)})")
        order = np.argsort(-np.abs(contrib), kind="stable")
        keep, rest = order[:max_terms], order[max_terms:]
        steps = [("base", float(self.base_margin_))]
        steps += [(labels[k], float(contrib[k])) for k in keep]
        if len(rest):
            steps.append((f"other {len(rest)} terms", float(contrib[rest].sum())))
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 0.45 * len(steps) + 1.2))
        pos = 0.0
        for i, (_, v) in enumerate(steps):
            start = 0.0 if i == 0 else pos
            ax.barh(i, v, left=start, color="C0" if i == 0 else ("C2" if v >= 0 else "C3"))
            pos = start + v
        ax.set_yticks(range(len(steps)), [s[0] for s in steps], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(pos, color="0.3", linestyle="--", linewidth=1)
        p = 1.0 / (1.0 + np.exp(-pos))
        ax.set_xlabel(f"cumulative logit (final {pos:+.3f}, P = {p:.3f})")
        return ax

    def to_dict(self, feature_names=None, precision=6):
        check_is_fitted(self, "trees_")
        trees = {}
        for k, conds, v in InterpretableSumMixin.rules(self, feature_names, precision):
            trees.setdefault(k, []).append(dict(conditions=conds, value=v))
        d = dict(
            link="logit",
            base_margin=float(self.base_margin_),
            classes=[c.item() if hasattr(c, "item") else c for c in self.classes_],
            trees=[dict(rules=trees[k]) for k in sorted(trees)],
        )
        names = self._names(feature_names)
        hinges_out = []
        for hg in self.hinges_:
            b_nan, a_l, s_l, a_r, s_r = hg.raw_pieces()
            fname = names[hg.feature] if names is not None else f"x{hg.feature}"
            hinges_out.append(
                dict(
                    feature=hg.feature,
                    feature_name=fname,
                    threshold=hg.threshold_val,
                    x_min=float(hg.x_min),
                    x_max=float(hg.x_max),
                    clamp=bool(hg.clamp),
                    nan_value=b_nan,
                    left_intercept=a_l,
                    left_slope=s_l,
                    right_intercept=a_r,
                    right_slope=s_r,
                )
            )
        d["hinges"] = hinges_out
        return d

    def explain(self, feature_names=None, precision=4):
        d = self.to_dict(feature_names, precision)
        n_trees, n_hinges = len(d["trees"]), len(self.hinges_)
        lines = [
            f"logit P(y = {d['classes'][1]}) = base {d['base_margin']:+.{precision}f}"
            f" + sum of {n_trees} trees and {n_hinges} hinges ({self.n_splits_} cuts)"
        ]
        for k, tree in enumerate(d["trees"], 1):
            lines.append(f"tree {k}:")
            for r in tree["rules"]:
                cond = " and ".join(r["conditions"]) or "always"
                lines.append(f"  {r['value']:+.{precision}f}  if {cond}")
        names = self._names(feature_names)
        for k, hg in enumerate(self.hinges_, 1):
            nm = names[hg.feature] if names is not None else f"x{hg.feature}"
            b_nan, a_l, s_l, a_r, s_r = hg.raw_pieces()
            thr = _fmt(hg.threshold_val, precision)
            clamp_note = (
                f", clamped [{_fmt(hg.x_min, precision)}, {_fmt(hg.x_max, precision)}]"
                if hg.clamp
                else ""
            )
            lines.append(f"hinge {k} ({nm}, knot <= {thr}{clamp_note}):")
            lines.append(
                f"  {_fmt(a_l, precision)} + ({_fmt(s_l, precision)}) * {nm}  if {nm} <= {thr}"
            )
            lines.append(
                f"  {_fmt(a_r, precision)} + ({_fmt(s_r, precision)}) * {nm}  if {nm} > {thr}"
            )
            if self.nan_features_[hg.feature]:
                lines.append(f"  {b_nan:+.{precision}f}  if {nm} is missing")
        return "\n".join(lines)

    def to_sql(
        self,
        table="input",
        feature_names=None,
        precision=6,
        keep_columns=True,
        null_safe=True,
        score_col="score",
        proba_col="p",
    ):
        sql = super().to_sql(
            table=table,
            feature_names=feature_names,
            precision=precision,
            keep_columns=keep_columns,
            null_safe=null_safe,
            score_col=score_col,
            proba_col=proba_col,
        )
        if not self.hinges_:
            return sql
        names = self._names(feature_names)
        name = (lambda j: _sql_ident(names[j])) if names is not None else (lambda j: f"x{j}")
        num = lambda v: format(float(v), f".{precision}g")  # noqa: E731
        cut = lambda v: np.format_float_positional(  # noqa: E731
            np.float64(v), unique=True, trim="-"
        )

        extra_cols = []
        extra_sums = []
        for idx, hg in enumerate(self.hinges_, 1):
            nm = name(hg.feature)
            b_nan, a_l, s_l, a_r, s_r = hg.raw_pieces()
            thr = cut(hg.threshold_val)
            if hg.clamp and np.isfinite(hg.x_min) and np.isfinite(hg.x_max):
                xmin = cut(hg.x_min)
                xmax = cut(hg.x_max)
                val_expr = (
                    f"(CASE WHEN {nm} < {xmin} THEN {xmin} "
                    f"WHEN {nm} > {xmax} THEN {xmax} ELSE {nm} END)"
                )
            else:
                val_expr = nm
            alias = f"h{idx}"
            extra_cols.append(
                f"    CASE WHEN {nm} IS NULL THEN {num(b_nan)} "
                f"WHEN {val_expr} <= {thr} THEN ({num(a_l)}) + ({num(s_l)}) * {val_expr} "
                f"ELSE ({num(a_r)}) + ({num(s_r)}) * {val_expr} END AS {alias}"
            )
            extra_sums.append(f"    + {alias}")

        lines = sql.splitlines()
        frm = lines.index(f"  FROM {table}")
        lines[frm - 1] += ","
        lines[frm:frm] = [",\n".join(extra_cols)]
        s_col = _sql_ident(score_col)
        stop = lines.index(f"    AS {s_col}")
        lines[stop:stop] = extra_sums
        return "\n".join(lines)
