"""Variance control for FIGS at medium budgets (32-64 cuts).

With many cuts and little data, the full-step FIGS picks unstable cuts. Two
structure selectors fight that variance while keeping one readable model of at
most ``max_splits`` cuts:

- ``BaggedFIGSClassifier``: FIGS on bootstrap replicates (sample weights, the
  bins are shared), a vote on the cuts (feature, bin) that recur across
  replicates, and one final FIGS on the full data restricted to that
  vocabulary. The bag can also act as a teacher (``distill=True``): its
  out-of-bag probability becomes the soft target of the final model.
- ``RashomonFIGSClassifier``: a FIGS on a training split, then a local search
  that mutates one cut at a time (threshold shift or a runner-up cut at that
  node), refits the leaves and keeps the structures whose validation loss is
  within ``epsilon`` of the best (the Rashomon set); the selected structure is
  refitted on all the data.

The bag and the search are used to choose the structure, not to predict: the
fitted model is a plain sum of trees with the same interpretation API.
"""

import numpy as np

from ._common import base_margin, grad_hess, sigmoid
from ._kernels import best_cut_1d, hist_1d
from .smalltrees import FIGSClassifier, SmallTree, grow_figs


def _log_loss(target, margin, w):
    p = np.clip(sigmoid(margin), 1e-15, 1 - 1e-15)
    return float(-np.sum(w * (target * np.log(p) + (1 - target) * np.log(1 - p))) / w.sum())


def _cuts(trees):
    return {(int(f), int(t)) for tree in trees for f, t, lc in
            zip(tree.feature, tree.threshold, tree.left) if lc != -1}


def _coarsen(Xb, nb, vocab):
    """Rebin so that only the vocabulary cuts exist.

    For feature j with sorted vocabulary thresholds T_j, coarse bin
    c(b) = 1 + #{t in T_j : t < b} for b >= 1. A coarse cut
    k >= 1 (``c <= k``) is the original cut ``b <= T_j[k - 1]``; NaN joins
    coarse bin 1 unless the NaN-only cut is in the vocabulary; features
    without vocabulary become constant. Returns (Xc, nbc, T)."""
    p = Xb.shape[1]
    T = [np.array(sorted(t for f, t in vocab if f == j), dtype=np.int64) for j in range(p)]
    Xc = np.empty_like(Xb)
    nbc = np.empty(p, dtype=np.int64)
    for j in range(p):
        if len(T[j]) == 0:
            Xc[:, j] = 1
            nbc[j] = 2
            continue
        lut = 1 + np.searchsorted(T[j], np.arange(int(nb[j])), side="left")
        # NaN (bin 0) goes left of every cut, so it joins coarse bin 1 unless the
        # NaN-only cut (t = 0) is itself in the vocabulary
        lut[0] = 0 if T[j][0] == 0 else 1
        Xc[:, j] = lut[Xb[:, j]]
        nbc[j] = len(T[j]) + 2
    return np.ascontiguousarray(Xc), nbc, T


def _uncoarsen(trees, T):
    for tree in trees:
        for k, (f, t, lc) in enumerate(zip(tree.feature, tree.threshold, tree.left)):
            if lc != -1:
                tree.threshold[k] = int(T[f][t - 1]) if t >= 1 else 0
    return trees


class BaggedFIGSClassifier(FIGSClassifier):
    """FIGS whose cuts are chosen by a bootstrap vote (and optionally distilled
    from the bag).

    1. Fit ``n_bags`` FIGS of ``max_splits`` cuts on bootstrap replicates
       (multinomial sample weights; the bins are computed once).
    2. Count in how many replicates each cut (feature, bin) appears and keep
       the ``vocab_size`` most frequent ones.
    3. Fit one FIGS of ``max_splits`` cuts on the full data using only those
       cuts; its leaves are estimated on the full data.

    With ``distill=True`` the final FIGS is fitted to
    ``distill_mix * y + (1 - distill_mix) * p_bag``, where ``p_bag`` is the
    out-of-bag average (in logit) of the replicates: a teacher that is itself
    a forest of FIGS, whose shape the student can represent.

    Parameters
    ----------
    max_splits, max_trees, lam, min_weight, max_bins, backfit_sweeps, learning_rate
        As in ``FIGSClassifier``; they apply to the replicates and to the final model.
    n_bags : int, default=20
        Bootstrap replicates.
    vocab_size : int or None, default=None
        Cuts kept by the vote; None = 2 * max_splits. ``0`` disables the
        vocabulary (the final FIGS may use any cut; useful with ``distill``).
    min_votes : float, default=0.0
        Also drop cuts seen in fewer than this fraction of the replicates.
    distill : bool, default=False
        Fit the final model to the bag's out-of-bag probability.
    distill_mix : float, default=0.5
        Weight of the observed label in the distillation target.
    random_state : int, default=0

    Attributes
    ----------
    trees_, base_margin_, classes_, bin_edges_, ...
        As in ``FIGSClassifier``.
    cut_votes_ : dict
        {(feature, bin): number of replicates using that cut}.
    vocabulary_ : list of (feature, bin)
        The cuts the final model could use.
    """

    def __init__(self, *, max_splits=32, max_trees=None, lam="auto", min_weight=20.0,
                 max_bins=32, backfit_sweeps=1, learning_rate=1.0, n_bags=20,
                 vocab_size=None, min_votes=0.0, distill=False, distill_mix=0.5,
                 random_state=0):
        super().__init__(max_splits=max_splits, max_trees=max_trees, lam=lam,
                         min_weight=min_weight, max_bins=max_bins,
                         backfit_sweeps=backfit_sweeps, learning_rate=learning_rate)
        self.n_bags = n_bags
        self.vocab_size = vocab_size
        self.min_votes = min_votes
        self.distill = distill
        self.distill_mix = distill_mix
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Fit the replicates, vote, and fit the final sum (see the class docstring)."""
        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        _, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        n = len(Xb)
        rng = np.random.default_rng(self.random_state)
        votes = {}
        oob_sum, oob_cnt = np.zeros(n), np.zeros(n)
        for _ in range(int(self.n_bags)):
            counts = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
            wb = w * counts
            self.base_margin_ = base_margin(target, wb)
            trees = grow_figs(self, Xb, nb, target, wb, self.max_splits, self.max_trees)
            for cut in _cuts(trees):
                votes[cut] = votes.get(cut, 0) + 1
            if self.distill:
                oob = counts == 0
                m = np.full(int(oob.sum()), self.base_margin_)
                for tree in trees:
                    m += tree.value[tree.leaf_ids(Xb[oob])]
                oob_sum[oob] += m
                oob_cnt[oob] += 1
        self.cut_votes_ = votes
        if self.distill:
            teacher = sigmoid(oob_sum / np.maximum(oob_cnt, 1))
            teacher[oob_cnt == 0] = target[oob_cnt == 0]
            target = self.distill_mix * target + (1 - self.distill_mix) * teacher
        self.base_margin_ = base_margin(target, w)
        size = 2 * self.max_splits if self.vocab_size is None else int(self.vocab_size)
        if size == 0:
            self.vocabulary_ = None
            self.trees_ = grow_figs(self, Xb, nb, target, w, self.max_splits, self.max_trees)
            return self
        floor = self.min_votes * self.n_bags
        ranked = sorted(((v, c) for c, v in votes.items() if v >= floor), reverse=True)
        self.vocabulary_ = [c for _, c in ranked[:size]]
        Xc, nbc, T = _coarsen(Xb, nb, self.vocabulary_)
        trees = grow_figs(self, Xc, nbc, target, w, self.max_splits, self.max_trees)
        self.trees_ = _uncoarsen(trees, T)
        return self


class RashomonFIGSClassifier(FIGSClassifier):
    """FIGS refined by a local search over near-optimal structures.

    1. Split off ``validation_fraction`` of the rows and fit a FIGS on the rest.
    2. ``n_mutations`` times: copy the current structure, change one cut —
       shift its threshold by up to ``max_shift`` bins, or replace it with one
       of the ``n_alternatives`` best cuts for the rows reaching that node —
       refit the leaves (backfitting on the training split) and score it on
       the validation split. A mutation that improves the validation loss
       becomes the current structure (hill climbing).
    3. Every scored structure within ``epsilon`` (relative) of the best
       validation loss is kept in ``rashomon_set_``. ``select="best"`` takes
       the best one; ``select="stable"`` the one whose cuts are most common in
       the set. Its leaves are then refitted on all the data.

    Parameters
    ----------
    max_splits, max_trees, lam, min_weight, max_bins, backfit_sweeps, learning_rate
        As in ``FIGSClassifier``.
    n_mutations : int, default=150
    validation_fraction : float, default=0.2
    epsilon : float, default=0.01
        Relative tolerance on the validation log-loss for the Rashomon set.
    max_shift : int, default=3
    n_alternatives : int, default=4
    select : {"best", "stable"}, default="best"
    refit_sweeps : int, default=3
        Backfitting passes after each mutation and in the final refit.
    random_state : int, default=0

    Attributes
    ----------
    trees_, base_margin_, classes_, bin_edges_, ...
        As in ``FIGSClassifier``.
    rashomon_set_ : list of (validation log-loss, set of cuts)
    rashomon_members_ : list of (validation log-loss, trees)
        Distinct near-optimal structures; use ``rashomon_models()`` to get them
        as estimators and ``rashomon_importance(X)`` for the range of each
        feature's importance across the set.
    history_ : ndarray
        Validation loss of the current structure after each mutation.
    """

    def __init__(self, *, max_splits=32, max_trees=None, lam="auto", min_weight=20.0,
                 max_bins=32, backfit_sweeps=1, learning_rate=1.0, n_mutations=150,
                 validation_fraction=0.2, epsilon=0.01, max_shift=3, n_alternatives=4,
                 select="best", refit_sweeps=3, random_state=0):
        super().__init__(max_splits=max_splits, max_trees=max_trees, lam=lam,
                         min_weight=min_weight, max_bins=max_bins,
                         backfit_sweeps=backfit_sweeps, learning_rate=learning_rate)
        self.n_mutations = n_mutations
        self.validation_fraction = validation_fraction
        self.epsilon = epsilon
        self.max_shift = max_shift
        self.n_alternatives = n_alternatives
        self.select = select
        self.refit_sweeps = refit_sweeps
        self.random_state = random_state

    # ---------------------------------------------------------------- helpers

    def _refit(self, trees, Xb, target, w, sweeps):
        """Leaves from zero by full-step backfitting; returns (contribs, margin)."""
        lr, self.learning_rate_ = self.learning_rate_, 1.0
        for tree in trees:
            tree.value = np.zeros(len(tree.feature))
        contribs = [np.zeros(len(Xb)) for _ in trees]
        margin = np.full(len(Xb), self.base_margin_)
        for _ in range(sweeps):
            for k, tree in enumerate(trees):
                contribs[k], margin = self._newton_step(tree, Xb, target, margin,
                                                        contribs[k], w)
        self.learning_rate_ = lr
        return contribs, margin

    def _margin(self, trees, Xb):
        m = np.full(len(Xb), self.base_margin_)
        for tree in trees:
            m += tree.value[tree.leaf_ids(Xb)]
        return m

    @staticmethod
    def _copy(trees):
        return [SmallTree(list(t.feature), list(t.threshold), list(t.left), list(t.right),
                          t.value.copy()) for t in trees]

    @staticmethod
    def _rows_at(tree, node, Xb):
        """Boolean mask of the rows that reach ``node``."""
        parent = {c: (k, side) for k, (lc, rc) in enumerate(zip(tree.left, tree.right))
                  if lc != -1 for c, side in ((lc, 0), (rc, 1))}
        mask = np.ones(len(Xb), dtype=bool)
        while node in parent:
            k, side = parent[node]
            go_left = Xb[:, tree.feature[k]] <= tree.threshold[k]
            mask &= go_left if side == 0 else ~go_left
            node = k
        return mask

    def _mutate(self, trees, Xb, nb, target, w, rng):
        internal = [(i, k) for i, t in enumerate(trees) for k, lc in enumerate(t.left) if lc != -1]
        i, k = internal[rng.integers(len(internal))]
        tree = trees[i]
        if rng.random() < 0.5:  # threshold shift
            f = tree.feature[k]
            shift = int(rng.integers(1, self.max_shift + 1)) * (1 if rng.random() < 0.5 else -1)
            tree.threshold[k] = int(np.clip(tree.threshold[k] + shift, 1, nb[f] - 2))
            return trees
        rows = self._rows_at(tree, k, Xb)
        if rows.sum() < 2 * self.min_weight:
            return trees
        margin = self._margin(trees, Xb[rows]) - tree.value[tree.leaf_ids(Xb[rows])]
        g, h = grad_hess(target[rows], margin, w[rows])
        gains, cuts = best_cut_1d(hist_1d(np.ascontiguousarray(Xb[rows]), g, h, w[rows],
                                          int(nb.max())), nb, self.lam_, self.min_weight)
        order = [int(j) for j in np.argsort(-gains)
                 if gains[j] > 0 and (int(j), int(cuts[j])) != (tree.feature[k], tree.threshold[k])]
        if order:
            j = order[rng.integers(min(len(order), self.n_alternatives))]
            tree.feature[k], tree.threshold[k] = j, int(cuts[j])
        return trees

    # ---------------------------------------------------------------- fit

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Fit the initial FIGS, search the structures and refit the selected one."""
        if self.select not in ("best", "stable"):
            raise ValueError("select must be 'best' or 'stable'.")
        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = float(self.learning_rate)
        _, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        rng = np.random.default_rng(self.random_state)
        val = np.zeros(len(Xb), dtype=bool)
        val[rng.permutation(len(Xb))[:int(round(self.validation_fraction * len(Xb)))]] = True
        tr = ~val
        Xt, yt, wt = np.ascontiguousarray(Xb[tr]), target[tr], w[tr]
        Xv, yv, wv = np.ascontiguousarray(Xb[val]), target[val], w[val]
        self.base_margin_ = base_margin(yt, wt)
        current = grow_figs(self, Xt, nb, yt, wt, self.max_splits, self.max_trees)
        if not current or not val.any():
            self.base_margin_ = base_margin(target, w)
            self.trees_ = current
            self._refit(self.trees_, Xb, target, w, self.refit_sweeps)
            self.rashomon_set_, self.history_ = [], np.array([])
            self.rashomon_members_ = [(np.nan, self._copy(self.trees_))]
            self.rashomon_base_ = float(self.base_margin_)
            return self
        self._refit(current, Xt, yt, wt, self.refit_sweeps)
        cur_loss = _log_loss(yv, self._margin(current, Xv), wv)
        scored = [(cur_loss, self._copy(current))]
        history = []
        for _ in range(int(self.n_mutations)):
            cand = self._mutate(self._copy(current), Xt, nb, yt, wt, rng)
            self._refit(cand, Xt, yt, wt, self.refit_sweeps)
            loss = _log_loss(yv, self._margin(cand, Xv), wv)
            scored.append((loss, cand))
            if loss < cur_loss:
                current, cur_loss = cand, loss
            history.append(cur_loss)
        best = min(s[0] for s in scored)
        near = [(loss, trees) for loss, trees in scored if loss <= best * (1 + self.epsilon)]
        self.rashomon_set_ = [(loss, _cuts(trees)) for loss, trees in near]
        if self.select == "stable":
            freq = {}
            for _, cuts in self.rashomon_set_:
                for c in cuts:
                    freq[c] = freq.get(c, 0) + 1
            chosen = max(near, key=lambda s: (sum(freq[c] for c in _cuts(s[1])), -s[0]))[1]
        else:
            chosen = min(near, key=lambda s: s[0])[1]
        self.history_ = np.array(history)
        # the usable Rashomon set: distinct structures, leaves fitted on the training split
        seen, members = set(), []
        for loss, trees in sorted(near, key=lambda s: s[0]):
            key = frozenset(_cuts(trees))
            if key not in seen:
                seen.add(key)
                members.append((loss, self._copy(trees)))
        self.rashomon_members_ = members
        self.rashomon_base_ = float(self.base_margin_)
        self.base_margin_ = base_margin(target, w)
        self.trees_ = chosen
        self._refit(self.trees_, Xb, target, w, self.refit_sweeps)
        return self

    # ---------------------------------------------------------------- Rashomon set

    def rashomon_models(self, X=None, y=None, sample_weight=None):
        """The Rashomon set as fitted estimators (best validation loss first).

        Each is a copy of this estimator holding one near-optimal structure
        (distinct cut sets only), with the leaves fitted on the training split;
        with ``X, y`` the leaves are refitted on them. Every copy has the full
        prediction, explanation and editing API."""
        import copy

        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "rashomon_members_")
        out = []
        for loss, trees in self.rashomon_members_:
            m = copy.copy(self)
            m.rashomon_members_ = []
            m.trees_ = self._copy(trees)
            m.base_margin_ = self.rashomon_base_
            m.validation_loss_ = float(loss)
            if X is not None:
                m.refit_leaves(X, y, sample_weight=sample_weight, lam=self.lam_,
                               sweeps=self.refit_sweeps)
            out.append(m)
        return out

    def rashomon_importance(self, X):
        """Range of each feature's importance over the Rashomon set.

        Importance of a feature in one model = mean |contribution| of the trees
        that use it (a tree's share is split equally among its features),
        normalized to sum 1. Returns {feature name: (min, mean, max)} over the
        set: a feature whose minimum is near 0 is not needed by some equally
        good model."""
        from ._common import predict_input, rebin
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
        names = (list(self.feature_names_in_) if hasattr(self, "feature_names_in_")
                 else [f"x{j}" for j in range(self.n_features_in_)])
        rows = []
        for _, trees in self.rashomon_members_:
            imp = np.zeros(len(names))
            for t in trees:
                feats = sorted({int(f) for f, lc in zip(t.feature, t.left) if lc != -1})
                if feats:
                    share = float(np.mean(np.abs(t.value[t.leaf_ids(Xb)]))) / len(feats)
                    imp[feats] += share
            rows.append(imp / imp.sum() if imp.sum() > 0 else imp)
        rows = np.array(rows)
        return {n: (float(rows[:, j].min()), float(rows[:, j].mean()), float(rows[:, j].max()))
                for j, n in enumerate(names)}
