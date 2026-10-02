"""Experimental ITM, step 3: sums of small trees plus one-parameter product terms.

The production FIGS path is untouched. Every cut and every term costs one unit.
With no terms, growth and backfitting use the production kernels verbatim.
Plan and decision rules: Econ Research/paper/notes/PLANO_DEGRAU3.md (v2).
"""

import copy
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..sums._common import (
    base_margin,
    bin_threshold,
    fit_inputs,
    grad_hess,
    predict_input,
    rebin,
)
from ..sums._kernels import best_cut_1d, node_hist, node_hist_margin
from ..sums.explain import InterpretableSumMixin
from ..sums.smalltrees import FIGSClassifier, SmallTree, grow_figs
from ._itm_kernels import (
    affine_grad_hess,
    best_cut_tree,
    cut_pair_sums,
    cut_tree_hist,
    joint_system,
    leaf_pair_sums,
    leaf_tree_sums,
    pair_scores,
    tree_step,
)

TREE, NODE, CUT = 0, 1, 2
FORMS = ("M1", "M2", "M3", "M4", "M5")
ORDERS = ("interleaved", "two_stage")


def _subtree_leaves(tree, node):
    out, stack = [], [node]
    while stack:
        k = stack.pop()
        if tree.left[k] == -1:
            out.append(k)
        else:
            stack += [tree.left[k], tree.right[k]]
    return np.asarray(out, dtype=np.int64)


def _path(tree, node):
    """[(feature, bin threshold, went_left)] from the root to ``node``."""
    parent = {}
    for k, (lft, rgt) in enumerate(zip(tree.left, tree.right)):
        if lft != -1:
            parent[lft], parent[rgt] = (k, True), (k, False)
    steps = []
    while node in parent:
        k, left = parent[node]
        steps.append((int(tree.feature[k]), int(tree.threshold[k]), left))
        node = k
    return steps[::-1]


def _key(f1, f2):
    return frozenset([tuple(f1), tuple(f2)])


class InteractingFIGSClassifier(InterpretableSumMixin, ClassifierMixin, BaseEstimator):
    """ITM with one-parameter product terms, binary classification.

    ``eta = base + sum_k T_k + sum_e gamma_e * phi_e1 * phi_e2``, where each factor is
    ``phi = (raw - center) / scale`` with center and scale (h-weighted) fixed when the
    term is born. Raw factors: a tree output T_k, the indicator of a node region of a
    tree (a leaf when the term is born; it keeps meaning the same rows when that leaf
    is later split), or the indicator ``[x_f > t]`` of a cut (NaN goes left, as in the
    trees). The forms:

    - M1 tree x tree; M2 node region of T_j x T_k; M3 region of T_j x region of T_k;
    - M4 a new cut x T_k (the cut is used only inside the term);
    - M5 two cuts already used by the trees, on different features.

    ``max_splits`` is the TOTAL unit budget (cuts + terms). A term competes with the
    best cut by Newton gain G^2/(H + lam) of its centered column; on a tie the cut
    wins. After every accepted operation, every tree's leaves take a Newton step (with
    the multiplier d eta / d T_k) and all term coefficients take one joint Newton
    step, with the same rate and cap as the leaves.

    ``order="interleaved"`` (O1) lets cuts and terms compete at every step;
    ``"two_stage"`` (O2) grows the additive model with ``b - m`` cuts, m = b // 4
    (at least 1), then adds up to m terms with the cuts frozen. ``forms=()`` or
    ``max_interactions=0`` delegates to FIGS, bit for bit. ``max_partners`` limits the
    M4 search to the trees with the largest |sum g z_k| / sqrt(sum h z_k^2 + lam).
    """

    def __init__(
        self,
        *,
        max_splits=16,
        max_trees=None,
        lam="auto",
        min_weight=20.0,
        max_bins=32,
        backfit_sweeps=1,
        learning_rate=1.0,
        max_delta_step=None,
        forms=FORMS,
        order="interleaved",
        max_interactions=None,
        interaction_lam="auto",
        max_partners=8,
        min_gain=0.0,
    ):
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.backfit_sweeps = backfit_sweeps
        self.learning_rate = learning_rate
        self.max_delta_step = max_delta_step
        self.forms = forms
        self.order = order
        self.max_interactions = max_interactions
        self.interaction_lam = interaction_lam
        self.max_partners = max_partners
        self.min_gain = min_gain

    # Used only on the exact no-term path of the production FIGS grower.
    _prepare = FIGSClassifier._prepare
    _newton_step = FIGSClassifier._newton_step
    _backfit = FIGSClassifier._backfit

    # ------------------------------------------------------------ parameters

    def _validate_params_itm(self):
        for name in ("max_splits", "backfit_sweeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("max_trees", "max_interactions"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            ):
                raise ValueError(f"{name} must be None or a non-negative integer.")
        if (isinstance(self.max_partners, bool) or not isinstance(self.max_partners, Integral)
                or self.max_partners < 1):
            raise ValueError("max_partners must be a positive integer.")
        forms = (self.forms,) if isinstance(self.forms, str) else tuple(self.forms)
        if any(f not in FORMS for f in forms):
            raise ValueError(f"forms must be a subset of {FORMS}.")
        self.forms_ = tuple(f for f in FORMS if f in forms)
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}.")
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if not np.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError("min_gain must be finite and non-negative.")
        if self.max_delta_step is not None and (
            not np.isfinite(self.max_delta_step) or self.max_delta_step <= 0
        ):
            raise ValueError("max_delta_step must be None or finite and positive.")
        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        if not np.isfinite(self.lam_) or self.lam_ < 0:
            raise ValueError("lam must be 'auto' or finite and non-negative.")
        self.interaction_lam_ = (
            max(self.lam_, 1e-12) if self.interaction_lam == "auto" else float(self.interaction_lam)
        )
        if not np.isfinite(self.interaction_lam_) or self.interaction_lam_ <= 0:
            raise ValueError("interaction_lam must be 'auto' or finite and positive.")
        self.learning_rate_ = float(self.learning_rate)

    def _reset_terms(self):
        self.terms_ = []
        self.interaction_coefficients_ = np.empty(0)
        self.operation_history_ = []

    # ------------------------------------------------------------ fit

    def fit(self, X, y, sample_weight=None, y_soft=None):
        self._validate_params_itm()
        _, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        self._reset_terms()
        units = self.max_splits
        if not self.forms_ or self.max_interactions == 0:
            self.trees_ = grow_figs(self, Xb, nb, target, w, units, self.max_trees)
        elif self.order == "two_stage":
            m = min(units, max(1, units // 4))
            self.trees_ = grow_figs(self, Xb, nb, target, w, units - m, self.max_trees)
            self._grow(Xb, nb, target, w, m, frozen=True)
        else:
            self.trees_ = []
            self._grow(Xb, nb, target, w, units)
        self.training_margin_ = self._margin_from(Xb)
        return self

    @classmethod
    def from_fitted(cls, model, X, y, *, max_interactions=4, forms=FORMS, sample_weight=None,
                    y_soft=None):
        """E1: copy a fitted FIGS, add up to ``max_interactions`` terms with its cuts frozen.

        Damping, bins, rate and cap come from the fitted baseline. The input model is
        never mutated. Each accepted term costs one additional unit.
        """
        if not isinstance(model, FIGSClassifier):
            raise TypeError("from_fitted requires a fitted FIGSClassifier.")
        check_is_fitted(model, "trees_")
        est = cls(max_splits=model.n_splits_ + max_interactions, max_trees=model.max_trees,
                  lam=model.lam_, min_weight=model.min_weight, max_bins=model.max_bins,
                  backfit_sweeps=model.backfit_sweeps, learning_rate=model.learning_rate_,
                  max_delta_step=model.max_delta_step, forms=forms,
                  max_interactions=max_interactions)
        est._validate_params_itm()
        Xb, nb, target, w = est._adopt(model, X, y, sample_weight, y_soft)
        est.trees_ = copy.deepcopy(model.trees_)
        est._reset_terms()
        est._grow(Xb, nb, target, w, max_interactions, frozen=True)
        est.training_margin_ = est._margin_from(Xb)
        return est

    @classmethod
    def from_structure(cls, reference, trees, terms, X, y, *, refits=0, sample_weight=None,
                       y_soft=None, **params):
        """A model with given trees and terms, on the bins of ``reference`` (a fitted model).

        ``terms``: dicts with ``f1``, ``f2`` (kind, a, b), ``center``, ``scale`` and
        ``gamma``. With ``refits > 0`` the leaf values and the coefficients take that many
        joint refit passes (the structure stays frozen). Used to generate data from a
        known model and for the oracle tests.
        """
        est = cls(**params)
        est._validate_params_itm()
        Xb, nb, target, w = est._adopt(reference, X, y, sample_weight, y_soft)
        est.trees_ = copy.deepcopy(trees)
        est._reset_terms()
        for term in terms:
            est.terms_.append(dict(form=term.get("form", "given"), f1=tuple(term["f1"]),
                                   f2=tuple(term["f2"]), center=tuple(term["center"]),
                                   scale=tuple(term["scale"])))
        est.interaction_coefficients_ = np.asarray([t["gamma"] for t in terms], dtype=float)
        if refits:
            values, ids_of = est._values(Xb)
            margin = est._margin(values, ids_of, Xb)
            for _ in range(refits):
                margin = est._refit_active(values, ids_of, Xb, target, w, margin)
        est.training_margin_ = est._margin_from(Xb)
        return est

    def _adopt(self, model, X, y, sample_weight, y_soft):
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        if not np.array_equal(classes, model.classes_):
            raise ValueError("y classes must match the fitted model.")
        if self.n_features_in_ != model.n_features_in_:
            raise ValueError("X must have the fitted model's features.")
        self.classes_ = classes
        self.bin_edges_ = copy.deepcopy(model.bin_edges_)
        self.nan_features_ |= model.nan_features_
        self.base_margin_ = float(model.base_margin_)
        Xb = rebin(X, self.bin_edges_)
        nb = np.asarray([len(e) + 2 for e in self.bin_edges_], dtype=np.int64)
        return Xb, nb, target, w

    # ------------------------------------------------------------ factors and margin

    def _values(self, Xb):
        ids_of = [t.leaf_ids(Xb) for t in self.trees_]
        values = np.empty((len(Xb), len(self.trees_)), order="F")
        for k, tree in enumerate(self.trees_):
            values[:, k] = tree.value[ids_of[k]]
        return values, ids_of

    def _raw(self, factor, values, ids_of, Xb):
        kind, a, b = factor
        if kind == TREE:
            return values[:, a]
        if kind == NODE:
            return np.isin(ids_of[a], _subtree_leaves(self.trees_[a], b)).astype(np.float64)
        return (Xb[:, a] > b).astype(np.float64)

    def _phis(self, values, ids_of, Xb):
        E = len(self.terms_)
        P1, P2 = np.empty((len(Xb), E)), np.empty((len(Xb), E))
        for e, term in enumerate(self.terms_):
            P1[:, e] = (self._raw(term["f1"], values, ids_of, Xb) - term["center"][0]) / term["scale"][0]
            P2[:, e] = (self._raw(term["f2"], values, ids_of, Xb) - term["center"][1]) / term["scale"][1]
        return P1, P2

    def _margin(self, values, ids_of, Xb):
        margin = self.base_margin_ + values.sum(axis=1)
        if self.terms_:
            P1, P2 = self._phis(values, ids_of, Xb)
            margin = margin + (P1 * P2) @ self.interaction_coefficients_
        return margin

    def _margin_from(self, Xb):
        values, ids_of = self._values(Xb)
        return self._margin(values, ids_of, Xb)

    def _multiplier(self, k, P1, P2):
        """d eta / d T_k at every row."""
        a = np.ones(P1.shape[0])
        for e, term in enumerate(self.terms_):
            gamma = self.interaction_coefficients_[e]
            if term["f1"][0] == TREE and term["f1"][1] == k:
                a += gamma * P2[:, e] / term["scale"][0]
            if term["f2"][0] == TREE and term["f2"][1] == k:
                a += gamma * P1[:, e] / term["scale"][1]
        return a

    def _is_factor_tree(self, k):
        return any((t["f1"][0] == TREE and t["f1"][1] == k) or (t["f2"][0] == TREE and t["f2"][1] == k)
                   for t in self.terms_)

    def _tree_step(self, k, values, ids_of, Xb, target, w, margin):
        P1, P2 = self._phis(values, ids_of, Xb)
        a = self._multiplier(k, P1, P2)
        cap = self.max_delta_step
        self.trees_[k].value = np.ascontiguousarray(self.trees_[k].value, dtype=np.float64)
        tree_step(target, w, margin, values, ids_of[k], self.trees_[k].value, k, a, self.lam_,
                  self.learning_rate_, 0.0 if cap is None else float(cap), cap is not None)

    def _refit_active(self, values, ids_of, Xb, target, w, margin):
        """One pass: every tree's leaves, then all coefficients jointly. Returns the margin."""
        for _ in range(max(1, self.backfit_sweeps)):
            for k in range(len(self.trees_)):
                self._tree_step(k, values, ids_of, Xb, target, w, margin)
        if not self.terms_:
            return margin
        P1, P2 = self._phis(values, ids_of, Xb)
        U = np.ascontiguousarray(P1 * P2)
        g, h = grad_hess(target, margin, w)
        G, H = joint_system(U, g, h, self.interaction_lam_)
        step = -np.linalg.solve(H, G)
        if self.max_delta_step is not None:
            step = np.clip(step, -self.max_delta_step, self.max_delta_step)
        step *= self.learning_rate_
        self.interaction_coefficients_ = self.interaction_coefficients_ + step
        return margin + U @ step

    # ------------------------------------------------------------ candidate terms

    def _tree_frame(self, values, h):
        mass = h.sum()
        mu = h @ values / mass
        sd = np.sqrt(h @ (values - mu) ** 2 / mass)
        live = sd > 1e-12
        z = np.zeros_like(values)
        z[:, live] = (values[:, live] - mu[live]) / sd[live]
        return np.ascontiguousarray(z), mu, sd, live

    def _best_term(self, values, ids_of, Xb, nb, g, h, w):
        """Best candidate term of the enabled forms: (gain, term dict) or (0, None)."""
        lam, mw = self.interaction_lam_, self.min_weight
        K = len(self.trees_)
        if K == 0:
            return 0.0, None
        used = {_key(t["f1"], t["f2"]) for t in self.terms_}
        z, mu, sd, live = self._tree_frame(values, h)
        Htot, Gtot, Wtot = h.sum(), g.sum(), w.sum()
        best = [0.0, None]

        def offer(gain, form, f1, c1, s1, f2, c2, s2):
            if gain > best[0] and _key(f1, f2) not in used:
                best[0], best[1] = float(gain), dict(form=form, f1=tuple(int(v) for v in f1),
                                                     f2=tuple(int(v) for v in f2),
                                                     center=(float(c1), float(c2)),
                                                     scale=(float(s1), float(s2)))

        leaves = [np.asarray(t.leaves, dtype=np.int64) for t in self.trees_]
        n_nodes = max(len(t.feature) for t in self.trees_)
        need_ids = any(f in self.forms_ for f in ("M2", "M3"))
        ids = np.ascontiguousarray(np.column_stack(ids_of)) if need_ids else None

        if "M1" in self.forms_ and K > 1:
            pairs = np.asarray([(j, k) for j in range(K) for k in range(j + 1, K)
                                if live[j] and live[k]], dtype=np.int64).reshape(-1, 2)
            if len(pairs):
                gains = pair_scores(z, g, h, pairs, lam)
                for e in np.argsort(-gains, kind="stable"):
                    j, k = pairs[e]
                    if _key((TREE, j, -1), (TREE, k, -1)) not in used:
                        offer(gains[e], "M1", (TREE, j, -1), mu[j], sd[j], (TREE, k, -1), mu[k], sd[k])
                        break

        if "M2" in self.forms_ and K > 1:
            A, L = leaf_tree_sums(ids, g, h, w, z, n_nodes)
            Gz, Hz = g @ z, h @ (z * z)
            for j in range(K):
                for a in leaves[j]:
                    c = L[j, a, 1] / Htot
                    s2 = c * (1 - c)
                    if s2 <= 1e-12:
                        continue
                    for k in range(K):
                        if k == j or not live[k]:
                            continue
                        G = (A[j, a, k, 0] - c * Gz[k]) / np.sqrt(s2)
                        H = ((1 - 2 * c) * A[j, a, k, 1] + c * c * Hz[k]) / s2
                        offer(G * G / (H + lam), "M2", (NODE, j, a), c, np.sqrt(s2),
                              (TREE, k, -1), mu[k], sd[k])

        if "M3" in self.forms_ and K > 1:
            if "M2" not in self.forms_:
                _, L = leaf_tree_sums(ids, g, h, w, z, n_nodes)
            C = leaf_pair_sums(ids, g, h, w, n_nodes)
            for j in range(K):
                for k in range(j + 1, K):
                    for a in leaves[j]:
                        for b in leaves[k]:
                            Gab, Hab, Wab = C[j, k, a, b]
                            if mw > Wab or mw > Wtot - Wab:
                                continue
                            gain, ca, sa, cb, sb = _dummy_pair_gain(
                                Gab, Hab, L[j, a, 0], L[j, a, 1], L[k, b, 0], L[k, b, 1],
                                Gtot, Htot, lam)
                            offer(gain, "M3", (NODE, j, a), ca, sa, (NODE, k, b), cb, sb)

        if "M5" in self.forms_:
            cuts = sorted({(int(f), int(t)) for tree in self.trees_
                           for f, t, lft in zip(tree.feature, tree.threshold, tree.left) if lft != -1})
            if len(cuts) > 1:
                D = np.ascontiguousarray(np.column_stack([Xb[:, f] > t for f, t in cuts]).astype(np.uint8))
                S = cut_pair_sums(D, g, h, w)
                for p in range(len(cuts)):
                    for q in range(p + 1, len(cuts)):
                        if cuts[p][0] == cuts[q][0]:
                            continue
                        Gab, Hab, Wab = S[p, q]
                        if mw > Wab or mw > Wtot - Wab:
                            continue
                        gain, ca, sa, cb, sb = _dummy_pair_gain(
                            Gab, Hab, S[p, p, 0], S[p, p, 1], S[q, q, 0], S[q, q, 1], Gtot, Htot, lam)
                        offer(gain, "M5", (CUT, *cuts[p]), ca, sa, (CUT, *cuts[q]), cb, sb)

        if "M4" in self.forms_:
            partners = np.flatnonzero(live)
            if len(partners) > self.max_partners:
                score = np.abs(g @ z[:, partners]) / np.sqrt(h @ (z[:, partners] ** 2) + lam)
                partners = np.sort(partners[np.argsort(-score, kind="stable")[: self.max_partners]])
            if len(partners):
                hist = cut_tree_hist(Xb, g, h, w, np.ascontiguousarray(z[:, partners]), int(nb.max()))
                gains, cuts = best_cut_tree(hist, nb, lam, mw)
                # the best (feature, partner) whose term is not in the model yet
                for flat in np.argsort(-gains, axis=None, kind="stable"):
                    f, jj = (int(v) for v in np.unravel_index(flat, gains.shape))
                    t, k = int(cuts[f, jj]), int(partners[jj])
                    if t < 0 or gains[f, jj] <= best[0]:
                        break
                    if _key((CUT, f, t), (TREE, k, -1)) in used:
                        continue
                    c = hist[f, jj, t + 1: nb[f], 2].sum() / hist[f, jj, : nb[f], 2].sum()
                    offer(gains[f, jj], "M4", (CUT, f, t), c, np.sqrt(c * (1 - c)),
                          (TREE, k, -1), mu[k], sd[k])
                    break
        return best[0], best[1]

    # ------------------------------------------------------------ growth

    def _grow(self, Xb, nb, target, w, units, frozen=False):
        n, B = len(Xb), int(nb.max())
        target = np.ascontiguousarray(target, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        values, ids_of = self._values(Xb)
        margin = self._margin(values, ids_of, Xb)
        root_ids, zero = np.zeros(n, dtype=np.int64), np.zeros(n)
        spent = 0
        while spent < units:
            active = len(self.terms_) > 0
            best = (self.min_gain, None, None, None, None)
            if not frozen:
                candidates = list(range(len(self.trees_)))
                if self.max_trees is None or len(self.trees_) < self.max_trees:
                    candidates.append(None)
                P = self._phis(values, ids_of, Xb) if active else None
                for k in candidates:
                    if k is None:
                        ids, contrib, leaves, count = root_ids, zero, [0], 1
                    else:
                        ids, contrib = ids_of[k], values[:, k]
                        leaves, count = self.trees_[k].leaves, len(self.trees_[k].feature)
                    if k is None or not self._is_factor_tree(k):
                        hist = node_hist_margin(Xb, target, w, margin, contrib, ids, count, B)
                    else:
                        a = self._multiplier(k, *P)
                        g_, h_ = affine_grad_hess(target, w, margin, np.ascontiguousarray(contrib), a)
                        hist = node_hist(Xb, g_, h_, w, ids, count, B)
                    for leaf in leaves:
                        gains, cuts = best_cut_1d(hist[leaf], nb, self.lam_, self.min_weight)
                        f = int(np.argmax(gains))
                        if gains[f] > best[0]:
                            best = (float(gains[f]), k, leaf, f, int(cuts[f]))
            term = None
            if self.max_interactions is None or len(self.terms_) < self.max_interactions:
                g, h = grad_hess(target, margin, w)
                tgain, tspec = self._best_term(values, ids_of, Xb, nb, g, h, w)
                if tspec is not None and tgain > best[0]:  # ties go to the cut
                    term = (tgain, tspec)
            if term is not None:
                tgain, tspec = term
                self.terms_.append(tspec)
                self.interaction_coefficients_ = np.append(self.interaction_coefficients_, 0.0)
                self.operation_history_.append(dict(kind=tspec["form"], f1=tspec["f1"],
                                                    f2=tspec["f2"], gain=tgain))
                spent += 1
                margin = self._refit_active(values, ids_of, Xb, target, w, margin)
                continue
            gain, k, leaf, f, cut = best
            if leaf is None:
                break
            if k is None:
                self.trees_.append(SmallTree())
                values = np.asfortranarray(np.column_stack([values, zero]))
                ids_of.append(root_ids)
                k = len(self.trees_) - 1
            self.trees_[k].split(leaf, f, cut)
            ids_of[k] = self.trees_[k].leaf_ids(Xb)
            self.operation_history_.append(dict(kind="split", tree=k, leaf=leaf, feature=f,
                                                threshold=cut, gain=gain))
            spent += 1
            if active:
                self._tree_step(k, values, ids_of, Xb, target, w, margin)
                margin = self._refit_active(values, ids_of, Xb, target, w, margin)
            else:
                _, margin = self._newton_step(self.trees_[k], Xb, target, margin, values[:, k], w,
                                              ids_of[k])
                if self.backfit_sweeps:
                    contribs = [values[:, j] for j in range(len(self.trees_))]
                    margin = self._backfit(self.trees_, contribs, Xb, target, w,
                                           self.backfit_sweeps, ids_of)
        self._incremental_margin_ = margin

    # ------------------------------------------------------------ counts

    @property
    def n_interactions_(self):
        check_is_fitted(self, "interaction_coefficients_")
        return len(self.interaction_coefficients_)

    @property
    def n_units_(self):
        return self.n_splits_ + self.n_interactions_

    def _distinct_cuts(self):
        cuts = {(int(f), int(t)) for tree in self.trees_
                for f, t, left in zip(tree.feature, tree.threshold, tree.left) if left != -1}
        for term in self.terms_:
            for kind, a, b in (term["f1"], term["f2"]):
                if kind == CUT:
                    cuts.add((a, b))
        return cuts

    @property
    def n_distinct_splits_(self):
        check_is_fitted(self, "trees_")
        return len(self._distinct_cuts())

    def conditions_read(self, X):
        """Mean number of distinct conditions (feature, threshold) read to score one row.

        Tree paths count once per row; a node-region factor reuses its tree's path; a
        cut factor adds its condition unless the row's paths already read it.
        """
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
        total = 0
        for i in range(len(Xb)):
            seen = set()
            for tree in self.trees_:
                node = 0
                while tree.left[node] != -1:
                    f, t = int(tree.feature[node]), int(tree.threshold[node])
                    seen.add((f, t))
                    node = tree.left[node] if Xb[i, f] <= t else tree.right[node]
            for term in self.terms_:
                for kind, a, b in (term["f1"], term["f2"]):
                    if kind == CUT:
                        seen.add((a, b))
            total += len(seen)
        return total / max(len(Xb), 1)

    # ------------------------------------------------------------ prediction and export

    def predict_contributions(self, X):
        """Columns: K tree terms, then E product terms, in order."""
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
        values, ids_of = self._values(Xb)
        out = np.empty((len(Xb), len(self.trees_) + self.n_interactions_))
        out[:, : len(self.trees_)] = values
        if self.terms_:
            P1, P2 = self._phis(values, ids_of, Xb)
            out[:, len(self.trees_):] = P1 * P2 * self.interaction_coefficients_
        return out

    def decision_function(self, X):
        if not self.n_interactions_:
            return FIGSClassifier.decision_function(self, X)
        return self.base_margin_ + self.predict_contributions(X).sum(axis=1)

    def _factor_text(self, factor, names, precision):
        kind, a, b = factor
        name = (lambda j: str(names[j])) if names else (lambda j: f"x{j}")
        num = lambda v: f"{v:.{precision}g}"  # noqa: E731
        if kind == TREE:
            return f"tree {a + 1}"

        def cond(f, t, left):
            if t == 0:
                return f"{name(f)} is missing" if left else f"{name(f)} is present"
            v = num(bin_threshold(self.bin_edges_[f], t))
            return f"{name(f)} <= {v}" if left else f"{name(f)} > {v}"

        if kind == NODE:
            return f"[tree {a + 1}: " + " and ".join(cond(*s) for s in _path(self.trees_[a], b)) + "]"
        return "[" + cond(a, b, False) + "]"

    def _term_text(self, names, precision):
        num = lambda v: f"{v:.{precision}g}"  # noqa: E731
        out = []
        for e, (term, gamma) in enumerate(zip(self.terms_, self.interaction_coefficients_)):
            f1 = self._factor_text(term["f1"], names, precision)
            f2 = self._factor_text(term["f2"], names, precision)
            (c1, c2), (s1, s2) = term["center"], term["scale"]
            out.append(f"term {e + 1} ({term['form']}): {gamma:+.{precision}g} * "
                       f"(({f1} - {num(c1)}) / {num(s1)}) * (({f2} - {num(c2)}) / {num(s2)})")
        return out

    def to_dict(self, feature_names=None, precision=6):
        result = super().to_dict(feature_names, precision)
        names = self._names(feature_names)
        result.update(
            model="InteractingFIGSClassifier",
            n_units=self.n_units_,
            n_splits=self.n_splits_,
            n_distinct_splits=self.n_distinct_splits_,
            terms=[dict(form=t["form"], factors=[self._factor_text(t["f1"], names, precision),
                                                 self._factor_text(t["f2"], names, precision)],
                        center=list(t["center"]), scale=list(t["scale"]), coefficient=float(gamma))
                   for t, gamma in zip(self.terms_, self.interaction_coefficients_)],
        )
        return result

    def explain(self, feature_names=None, precision=4):
        return "\n".join([super().explain(feature_names, precision),
                          *self._term_text(self._names(feature_names), precision)])

    def export_text(self, feature_names=None, precision=4):
        return "\n".join([super().export_text(feature_names, precision),
                          *self._term_text(self._names(feature_names), precision)])

    def to_sql(self, table="input", feature_names=None, precision=6, keep_columns=True):
        """Production CASE/NULL export of the trees, plus one line per product term.

        Indicator factors become 0/1 columns of the first CTE; a tree factor reads the
        tree's column. NaN goes left, as in the trees.
        """
        from ..sums.explain import _sql_ident
        sql = super().to_sql(table, feature_names, precision, keep_columns)
        if not self.n_interactions_:
            return sql
        names = self._names(feature_names)
        name = (lambda j: _sql_ident(names[j])) if names else (lambda j: f"x{j}")
        num = lambda v: format(float(v), f".{precision}g")  # noqa: E731
        cut = lambda v: np.format_float_positional(np.float32(v), unique=True, trim="-")  # noqa: E731
        lines = sql.splitlines()
        start = lines.index(next(s for s in lines if s.strip().endswith("-- base")))
        stop = lines.index("    AS score")
        tree_cols = [s.strip()[2:] for s in lines[start + 1: stop]]

        def cond(f, t, left):
            if t == 0:
                return f"{name(f)} IS NULL" if left else f"{name(f)} IS NOT NULL"
            v = cut(bin_threshold(self.bin_edges_[f], t))
            return f"({name(f)} <= {v} OR {name(f)} IS NULL)" if left else f"{name(f)} > {v}"

        extra, terms = [], []
        for e, (term, gamma) in enumerate(zip(self.terms_, self.interaction_coefficients_)):
            cols = []
            for side, factor in ((1, term["f1"]), (2, term["f2"])):
                kind, a, b = factor
                if kind == TREE:
                    cols.append(tree_cols[a])
                    continue
                col = f"i{e + 1}_{side}"
                conds = [cond(*s) for s in _path(self.trees_[a], b)] if kind == NODE else [cond(a, b, False)]
                extra.append(f"    CASE WHEN {' AND '.join(conds)} THEN 1.0 ELSE 0.0 END AS {col}")
                cols.append(col)
            (c1, c2), (s1, s2) = term["center"], term["scale"]
            terms.append(f"    + ({num(gamma)}) * (({cols[0]} - ({num(c1)})) / {num(s1)})"
                         f" * (({cols[1]} - ({num(c2)})) / {num(s2)})")
        if extra:  # indicator columns go in the first CTE, next to the tree columns
            frm = lines.index(f"  FROM {table}")
            lines[frm - 1] += ","
            lines[frm:frm] = [",\n".join(extra)]
        stop = lines.index("    AS score")
        lines[stop:stop] = terms
        return "\n".join(lines)

    def to_shap_model(self):
        if self.n_interactions_:
            raise NotImplementedError("Product terms require a product-tree SHAP export.")
        return super().to_shap_model()

    def shape_functions(self):
        if self.n_interactions_:
            raise NotImplementedError("Use predict_contributions for product terms.")
        return super().shape_functions()


def _dummy_pair_gain(Gab, Hab, Ga, Ha, Gb, Hb, G, H, lam):
    """Gain of u = (1a - ca)(1b - cb) / (sa sb), from cell and marginal sums of g and h."""
    ca, cb = Ha / H, Hb / H
    sa2, sb2 = ca * (1 - ca), cb * (1 - cb)
    if sa2 <= 1e-12 or sb2 <= 1e-12:
        return 0.0, ca, 1.0, cb, 1.0
    sa, sb = np.sqrt(sa2), np.sqrt(sb2)
    Gu = (Gab - cb * Ga - ca * Gb + ca * cb * G) / (sa * sb)
    Hu = ((1 - 2 * ca) * (1 - 2 * cb) * Hab + (1 - 2 * ca) * cb * cb * Ha
          + ca * ca * (1 - 2 * cb) * Hb + ca * ca * cb * cb * H) / (sa2 * sb2)
    return Gu * Gu / (Hu + lam), ca, sa, cb, sb
