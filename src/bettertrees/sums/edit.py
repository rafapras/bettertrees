"""Mechanical editing of a fitted sum of trees, and its local Rashomon view.

Every sum whose prediction is ``base_margin_ + Σ_k tree_k`` (FIGS, the sums of
optimal trees, the bagged/Rashomon FIGS, ``CompactTreeBooster``) gets these
methods. Cuts live on the bins learned at fit time: a new threshold snaps to
the nearest bin edge (the method returns the value actually used). After a
structural edit the leaves keep their old values until ``refit_leaves`` is
called with data, so an edit is never silently re-estimated.

Editing          ``prune``, ``set_cut``, ``drop_tree``, ``merge_duplicates``,
                 ``set_leaf_value``, ``refit_leaves``
Rashomon view    ``cut_alternatives``: other cuts for one node whose refitted
                 model is within ``epsilon`` of the current loss.

Every edit keeps ``predict``, ``rules``, ``to_dict``, ``get_trees``,
``predict_contributions`` and ``to_shap_model`` consistent, because they all
read the same ``trees_``.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._common import as_weights, grad_hess, rebin, sigmoid
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


def _log_loss(y, margin, w):
    p = np.clip(sigmoid(margin), 1e-15, 1 - 1e-15)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum())


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

    def _refit_binned(self, Xb, target, w, lam, sweeps):
        lam_old, lr_old = getattr(self, "lam_", None), getattr(self, "learning_rate_", None)
        self.lam_, self.learning_rate_ = float(lam), 1.0
        contribs = [t.value[t.leaf_ids(Xb)] for t in self.trees_]
        margin = np.full(len(Xb), float(self.base_margin_))
        if contribs:
            margin = margin + np.sum(contribs, axis=0)
        for _ in range(int(sweeps)):
            for k, tree in enumerate(self.trees_):
                ids = tree.leaf_ids(Xb)
                g, h = grad_hess(target, margin, w)
                step = newton_leaf_values(ids, g, h, len(tree.feature), self.lam_)
                tree.value = tree.value + step
                contribs[k] = contribs[k] + step[ids]
                margin = margin + step[ids]
            g, h = grad_hess(target, margin, w)
            delta = -float(np.sum(g)) / max(float(np.sum(h)), 1e-12)
            self.base_margin_ += delta
            margin = margin + delta
        self.lam_, self.learning_rate_ = lam_old, lr_old
        return margin

    # ---------------------------------------------------------------- editing

    def prune(self, tree, node):
        """Turn the cut ``node`` of ``tree`` into a leaf (its subtree is removed).

        The new leaf takes the mean of the removed leaves; call ``refit_leaves``
        to re-estimate. Returns self."""
        t = self._check_node(tree, node, internal=True)
        stack, vals = [node], []
        while stack:
            k = stack.pop()
            if t.left[k] == -1:
                vals.append(t.value[k])
            else:
                stack += [t.left[k], t.right[k]]
        t.left[node] = t.right[node] = -1
        t.feature[node] = t.threshold[node] = -1
        t.value[node] = float(np.mean(vals))
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

    def refit_leaves(self, X, y, sample_weight=None, lam=None, sweeps=5):
        """Re-estimate every leaf and the base for the current structure.

        Full Newton backfitting on (X, y) with L2 penalty ``lam`` (default: the
        fitted ``lam_``, or 1). Cuts are not changed. Returns self."""
        Xb, target, w = self._binned_data(X, y, sample_weight)
        lam = lam if lam is not None else (getattr(self, "lam_", None) or 1.0)
        self._refit_binned(Xb, target, w, lam, sweeps)
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
                            threshold=real, loss=_log_loss(target, margin, w),
                            current=(f, b) == current))
        best = min(r["loss"] for r in out)
        for r in out:
            r["delta"] = r["loss"] / best - 1
        out.sort(key=lambda r: r["loss"])
        near = [r for r in out if r["delta"] <= epsilon and not r["current"]][:top]
        return sorted(near + [r for r in out if r["current"]], key=lambda r: r["loss"])
