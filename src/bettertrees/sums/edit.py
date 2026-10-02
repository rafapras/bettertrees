"""Mechanical editing of a fitted sum of trees, and its local Rashomon view.

Every sum whose prediction is ``base_margin_ + Σ_k tree_k`` (FIGS, the sums of
optimal trees, the bagged/Rashomon FIGS, ``CompactTreeBooster``,
``AdditiveTreeBooster``) gets these methods. Cuts live on the bins learned at fit
time: a new threshold snaps to the nearest bin edge (the method returns the value
actually used). After a
structural edit the leaves keep their old values until ``refit_leaves`` is
called with data, so an edit is never silently re-estimated.

Editing          ``prune``, ``set_cut``, ``split_leaf``, ``add_stump``, ``drop_tree``,
                 ``merge_duplicates``, ``set_leaf_value``, ``refit_leaves`` (all or some
                 trees, optionally under monotone constraints)
Constraints      ``monotone_violations``, ``enforce_monotone``
Rashomon view    ``cut_alternatives``: other cuts for one node whose refitted
                 model is within ``epsilon`` of the current loss.

Every edit keeps ``predict``, ``rules``, ``to_dict``, ``get_trees``,
``predict_contributions`` and ``to_shap_model`` consistent, because they all
read the same ``trees_``.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._common import as_weights, grad_hess, log_loss_margin, rebin, subtree_leaves
from ._kernels import best_cut_1d, hist_1d, newton_leaf_values


def _reachable(tree):
    """Rebuild ``tree`` with only the nodes reachable from the root (preorder)."""
    from .smalltrees import SmallTree
    new = SmallTree()
    new.value = np.array([tree.value[0]], dtype=np.float64)

    def copy(old, node):
        if tree.left[old] == -1:
            new.value[node] = tree.value[old]
            return
        left, right = new.split(node, tree.feature[old], tree.threshold[old])
        copy(tree.left[old], left)
        copy(tree.right[old], right)

    copy(0, 0)
    return new


def _key(tree):
    """Structure of a tree (cuts only), for merging identical trees."""
    def walk(node):
        if tree.left[node] == -1:
            return None
        return (int(tree.feature[node]), int(tree.threshold[node]),
                walk(tree.left[node]), walk(tree.right[node]))
    return walk(0)


def _leaf_boxes(tree):
    """{leaf: {feature: (lo, hi)}}: the leaf holds bins lo < b <= hi on each feature."""
    out = {}

    def walk(node, box):
        if tree.left[node] == -1:
            out[node] = box
            return
        f, t = int(tree.feature[node]), int(tree.threshold[node])
        lo, hi = box.get(f, (-1, 1 << 30))
        walk(tree.left[node], {**box, f: (lo, min(hi, t))})
        walk(tree.right[node], {**box, f: (max(lo, t), hi)})

    walk(0, {})
    return out


def _compatible(a, b, skip):
    """Boxes ``a`` and ``b`` overlap on every feature except ``skip``."""
    for f in set(a) | set(b):
        if f == skip:
            continue
        lo = max(a.get(f, (-1, 0))[0], b.get(f, (-1, 0))[0])
        hi = min(a.get(f, (0, 1 << 30))[1], b.get(f, (0, 1 << 30))[1])
        if lo >= hi:
            return False
    return True


def _monotone_pairs(tree, feature):
    """(low leaf, high leaf) pairs that must be ordered for monotonicity in
    ``feature``: for every cut on ``feature``, a leaf of its left subtree and a
    leaf of its right subtree reachable from each other by changing ``feature`` only."""
    boxes = _leaf_boxes(tree)
    pairs = []
    for k in range(len(tree.feature)):
        if tree.left[k] != -1 and int(tree.feature[k]) == feature:
            for a in subtree_leaves(tree, tree.left[k]):
                for b in subtree_leaves(tree, tree.right[k]):
                    if _compatible(boxes[a], boxes[b], feature):
                        pairs.append((a, b))
    return pairs


def _project_monotone(tree, constraints, weights=None, max_passes=500):
    """Pool violating leaf pairs (weighted mean) until ``tree`` is monotone in every
    constrained feature. ``constraints``: {feature: +1 increasing / -1 decreasing}."""
    wts = np.ones(len(tree.feature)) if weights is None else np.maximum(weights, 1e-12)
    wts = np.asarray(wts, dtype=np.float64).copy()
    pairs = [(a, b, sign) for f, sign in constraints.items()
             for a, b in _monotone_pairs(tree, f)]
    if not pairs:
        return tree
    v = np.asarray(tree.value, dtype=np.float64).copy()
    for _ in range(max_passes):
        worst, pick = 1e-12, None
        for a, b, sign in pairs:
            gap = sign * (v[a] - v[b])
            if gap > worst:
                worst, pick = gap, (a, b)
        if pick is None:
            break
        a, b = pick
        v[a] = v[b] = (wts[a] * v[a] + wts[b] * v[b]) / (wts[a] + wts[b])
        wts[a] = wts[b] = wts[a] + wts[b]
    tree.value = v
    return tree


class TreeEditMixin:
    """Editing and local Rashomon methods for sums stored in ``trees_``."""

    # ---------------------------------------------------------------- helpers

    def _check_node(self, tree, node, internal=None):
        check_is_fitted(self, "trees_")
        if not 0 <= tree < len(self.trees_):
            raise IndexError(f"tree {tree} out of range (0..{len(self.trees_) - 1}).")
        t = self.trees_[tree]
        if not 0 <= node < len(t.feature):
            raise IndexError(f"node {node} out of range for tree {tree}.")
        if internal is True and t.left[node] == -1:
            raise ValueError(f"node {node} of tree {tree} is a leaf, not a cut.")
        if internal is False and t.left[node] != -1:
            raise ValueError(f"node {node} of tree {tree} is a cut, not a leaf.")
        return t

    def _snap(self, feature, threshold):
        """(bin, real threshold) of the bin edge closest to ``threshold``."""
        if not 0 <= feature < len(self.bin_edges_):
            raise IndexError(f"feature {feature} out of range.")
        edges = np.asarray(self.bin_edges_[feature], dtype=np.float64)
        if len(edges) == 0:
            raise ValueError(f"feature {feature} is constant in the training data: no cut.")
        k = int(np.argmin(np.abs(edges - float(threshold))))
        return k + 1, float(edges[k])

    def _feature_index(self, feature):
        if isinstance(feature, str):
            names = getattr(self, "feature_names_in_", None)
            if names is None or feature not in list(names):
                raise KeyError(f"unknown feature {feature!r}.")
            return int(list(names).index(feature))
        return int(feature)

    def _binned_data(self, X, y, sample_weight):
        from ._common import predict_input
        Xv = predict_input(self, X, "trees_")
        yv = np.asarray(y)
        if yv.shape != (len(Xv),):
            raise ValueError("y must be a vector aligned with X.")
        target = (yv == self.classes_[1]).astype(np.float64)
        if not np.isin(yv, self.classes_).all():
            raise ValueError("y has labels not seen at fit time.")
        return rebin(Xv, self.bin_edges_), target, as_weights(sample_weight, len(Xv))

    def _refit_binned(self, Xb, target, w, lam, sweeps, trees=None, refit_base=True,
                      monotone=None):
        """Newton backfitting of the leaves of ``trees`` (indices; None = all) and,
        with ``refit_base``, of the base; the other trees stay frozen. With
        ``monotone`` ({feature index: +1/-1}) every step is projected back onto
        the monotone set (pooling of violating leaves, weighted by hessian mass)."""
        lam_old, lr_old = getattr(self, "lam_", None), getattr(self, "learning_rate_", None)
        self.lam_, self.learning_rate_ = float(lam), 1.0
        active = range(len(self.trees_)) if trees is None else sorted(set(trees))
        ids = [t.leaf_ids(Xb) for t in self.trees_]
        margin = np.full(len(Xb), float(self.base_margin_))
        for t, i in zip(self.trees_, ids):
            margin = margin + t.value[i]
        for _ in range(int(sweeps)):
            for k in active:
                tree = self.trees_[k]
                g, h = grad_hess(target, margin, w)
                old = tree.value[ids[k]]
                tree.value = tree.value + newton_leaf_values(ids[k], g, h, len(tree.feature),
                                                             self.lam_)
                if monotone:
                    mass = np.bincount(ids[k], weights=h, minlength=len(tree.feature))
                    _project_monotone(tree, monotone, mass)
                margin = margin + tree.value[ids[k]] - old
            if refit_base:
                g, h = grad_hess(target, margin, w)
                delta = -float(np.sum(g)) / max(float(np.sum(h)), 1e-12)
                self.base_margin_ += delta
                margin = margin + delta
        self.lam_, self.learning_rate_ = lam_old, lr_old
        return margin

    # ---------------------------------------------------------------- editing

    def prune(self, tree, node, X=None, sample_weight=None):
        """Turn the cut ``node`` of ``tree`` into a leaf (its subtree is removed).

        The new leaf takes the mean of the removed leaves, weighted by the rows of
        ``X`` that reach each one (pass the training data). Without ``X`` the mean
        is unweighted, because leaf sizes are not stored: a tiny leaf then counts as
        much as a large one. ``refit_leaves`` re-estimates properly. Returns self."""
        from ._common import predict_input
        t = self._check_node(tree, node, internal=True)
        stack, leaves = [node], []
        while stack:
            k = stack.pop()
            if t.left[k] == -1:
                leaves.append(k)
            else:
                stack += [t.left[k], t.right[k]]
        vals = np.array([t.value[k] for k in leaves], dtype=np.float64)
        value = float(vals.mean())
        if X is not None:
            Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
            w = as_weights(sample_weight, len(Xb))
            ids = t.leaf_ids(Xb)
            mass = np.array([w[ids == k].sum() for k in leaves])
            if mass.sum() > 0:
                value = float(vals @ mass / mass.sum())
        t.left[node] = t.right[node] = -1
        t.feature[node] = t.threshold[node] = -1
        t.value[node] = value
        self.trees_[tree] = _reachable(t)
        return self

    def set_cut(self, tree, node, feature, threshold):
        """Change the cut of ``node`` to ``feature <= threshold`` (children kept).

        ``feature`` is an index or a column name; the threshold snaps to the
        nearest bin edge. Returns the real threshold used."""
        t = self._check_node(tree, node, internal=True)
        f = self._feature_index(feature)
        b, real = self._snap(f, threshold)
        t.feature[node], t.threshold[node] = f, b
        return real

    def split_leaf(self, tree, leaf, feature, threshold):
        """Add the cut ``feature <= threshold`` at ``leaf`` (both children inherit its
        value; call ``refit_leaves`` to re-estimate). Returns (left, right, real threshold)."""
        t = self._check_node(tree, leaf, internal=False)
        f = self._feature_index(feature)
        b, real = self._snap(f, threshold)
        left, right = t.split(leaf, f, b)
        return left, right, real

    def add_stump(self, feature, threshold, left_value=0.0, right_value=0.0):
        """Append a one-cut tree (a new rule) to the sum. Returns (tree index, real threshold)."""
        from .smalltrees import SmallTree
        check_is_fitted(self, "trees_")
        f = self._feature_index(feature)
        b, real = self._snap(f, threshold)
        t = SmallTree()
        t.split(0, f, b)
        t.value = np.array([0.0, float(left_value), float(right_value)])
        self.trees_.append(t)
        return len(self.trees_) - 1, real

    def drop_tree(self, tree):
        """Remove a whole tree from the sum. Returns self."""
        self._check_node(tree, 0)
        del self.trees_[tree]
        return self

    def merge_duplicates(self):
        """Merge trees with identical cuts into one (values add up; predictions
        unchanged). Returns the number of trees removed."""
        check_is_fitted(self, "trees_")
        kept, index = [], {}
        for tree in self.trees_:
            k = _key(tree)
            if k is None:  # a constant tree: fold it into the base
                self.base_margin_ += float(tree.value[0])
                continue
            if k in index:
                kept[index[k]].value = kept[index[k]].value + tree.value
            else:
                index[k] = len(kept)
                kept.append(_reachable(tree))
        removed = len(self.trees_) - len(kept)
        self.trees_ = kept
        return removed

    def set_leaf_value(self, tree, node, value):
        """Overwrite the logit value of a leaf (e.g. a business override). Returns self."""
        t = self._check_node(tree, node, internal=False)
        t.value[node] = float(value)
        return self

    def refit_leaves(self, X, y, sample_weight=None, lam=None, sweeps=5, trees=None,
                     refit_base=True, monotone=None):
        """Re-estimate the leaves for the current structure (cuts are not changed).

        Full Newton backfitting on (X, y) with L2 penalty ``lam`` (default: the
        fitted ``lam_``, or 1). ``trees``: indices of the trees to refit (the
        others stay frozen; None = all). ``refit_base``: also re-estimate the base.
        ``monotone``: {feature (index or name): +1 increasing / -1 decreasing};
        every Newton step is projected, so the result is monotone. Returns self."""
        Xb, target, w = self._binned_data(X, y, sample_weight)
        lam = lam if lam is not None else (getattr(self, "lam_", None) or 1.0)
        if trees is not None:
            for k in trees:
                self._check_node(k, 0)
        cons = self._constraints(monotone)
        self._refit_binned(Xb, target, w, lam, sweeps, trees, refit_base, cons)
        return self

    # ---------------------------------------------------------------- constraints

    def _constraints(self, monotone):
        if not monotone:
            return None
        out = {}
        for f, sign in monotone.items():
            if sign not in (1, -1):
                raise ValueError("monotone directions must be +1 or -1.")
            out[self._feature_index(f)] = int(sign)
        return out

    def monotone_violations(self, feature, increasing=True, tol=1e-12):
        """Leaf pairs that break monotonicity in ``feature``, as a list of
        ``(tree, lower_leaf, upper_leaf, gap)`` (empty = monotone). Exact per
        tree: for every cut on ``feature``, each leaf of its left subtree is
        compared with each leaf of its right subtree reachable by changing
        ``feature`` only; a sum of monotone trees is monotone."""
        check_is_fitted(self, "trees_")
        f = self._feature_index(feature)
        sign = 1 if increasing else -1
        out = []
        for i, t in enumerate(self.trees_):
            for a, b in _monotone_pairs(t, f):
                gap = sign * (t.value[a] - t.value[b])
                if gap > tol:
                    out.append((i, a, b, float(gap)))
        return out

    def enforce_monotone(self, monotone):
        """Make the sum monotone ({feature: +1/-1}) by pooling violating leaves
        (equal weights; ``refit_leaves(..., monotone=...)`` weighs by data). Returns self."""
        check_is_fitted(self, "trees_")
        cons = self._constraints(monotone)
        for t in self.trees_:
            _project_monotone(t, cons)
        return self

    # ---------------------------------------------------------------- Rashomon

    def cut_alternatives(self, X, y, tree, node, *, sample_weight=None, top=5, epsilon=0.01,
                         lam=None, sweeps=3):
        """Other cuts for ``node`` that fit (X, y) about as well (local Rashomon set).

        For each feature, the best cut for the rows reaching ``node`` (Newton
        gain against the rest of the sum) is tried in place of the current one;
        every candidate model is refitted (``refit_leaves`` on a copy) and
        scored by log-loss on (X, y). Returns a list of dicts
        ``{feature, feature_name, threshold, loss, delta, current}`` sorted by
        loss, keeping the current cut and the candidates within ``epsilon``
        (relative) of the best, at most ``top`` + 1 rows. Pass held-out data
        to judge the alternatives out of sample. The model is not changed.
        """
        import copy
        t = self._check_node(tree, node, internal=True)
        Xb, target, w = self._binned_data(X, y, sample_weight)
        lam = lam if lam is not None else (getattr(self, "lam_", None) or 1.0)
        rows = np.ones(len(Xb), dtype=bool)
        parent = {c: (k, s) for k, (lc, rc) in enumerate(zip(t.left, t.right)) if lc != -1
                  for c, s in ((lc, 0), (rc, 1))}
        k = node
        while k in parent:
            p, side = parent[k]
            go = Xb[:, t.feature[p]] <= t.threshold[p]
            rows &= go if side == 0 else ~go
            k = p
        others = self.base_margin_ + sum(
            (u.value[u.leaf_ids(Xb)] for j, u in enumerate(self.trees_) if j != tree),
            np.zeros(len(Xb)))
        g, h = grad_hess(target[rows], others[rows], w[rows])
        nb = np.array([len(e) + 2 for e in self.bin_edges_], dtype=np.int64)
        gains, cuts = best_cut_1d(hist_1d(np.ascontiguousarray(Xb[rows]), g, h, w[rows],
                                          int(nb.max())), nb, lam, 0.0)
        current = (int(t.feature[node]), int(t.threshold[node]))
        cands = {current}
        for f in np.argsort(-gains):
            if gains[f] > 0 and len(cands) <= 3 * top:
                cands.add((int(f), int(cuts[f])))
        names = getattr(self, "feature_names_in_", None)
        out = []
        for f, b in cands:
            m = copy.deepcopy(self)
            m.trees_[tree].feature[node], m.trees_[tree].threshold[node] = f, b
            margin = m._refit_binned(Xb, target, w, lam, sweeps)
            real = float(self.bin_edges_[f][b - 1]) if b >= 1 else -np.inf
            out.append(dict(feature=f, feature_name=None if names is None else str(names[f]),
                            threshold=real, loss=log_loss_margin(target, margin, w),
                            current=(f, b) == current))
        best = min(r["loss"] for r in out)
        for r in out:
            r["delta"] = r["loss"] / best - 1
        out.sort(key=lambda r: r["loss"])
        near = [r for r in out if r["delta"] <= epsilon and not r["current"]][:top]
        return sorted(near + [r for r in out if r["current"]], key=lambda r: r["loss"])
