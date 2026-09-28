"""Reduced-error pruning of a sum of trees down to a budget of cuts.

Grow first, prune second (the CART recipe, for sums): start from a model with
more cuts than wanted and remove, one at a time, the cut whose removal hurts a
held-out log-loss the least. A cut is removable when both its children are
leaves (the two leaves merge; the new value is their hessian-weighted mean on
the fitting rows) or when it is a whole stump (the tree is dropped and its mean
folded into the base). The leaves are re-estimated on the fitting rows every
``refit_every`` removals and at the end.
"""

import numpy as np

from ._common import grad_hess, sigmoid
from .edit import _reachable


def _ll_rows(y, m, w):
    p = np.clip(sigmoid(m), 1e-15, 1 - 1e-15)
    return -(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))


def _candidates(tree):
    """Internal nodes whose two children are leaves."""
    return [k for k in range(len(tree.feature)) if tree.left[k] != -1
            and tree.left[tree.left[k]] == -1 and tree.left[tree.right[k]] == -1]


def prune_to_budget(model, max_splits, Xf, yf, wf, Xv, yv, wv, lam=1.0, refit_every=8,
                    sweeps=3):
    """Prune ``model.trees_`` in place until ``model.n_splits_ <= max_splits``.

    ``Xf, yf, wf``: binned fitting rows (leaf values); ``Xv, yv, wv``: binned
    held-out rows (which cut to remove). Returns the number of cuts removed.
    """
    removed = 0
    ids_f = [t.leaf_ids(Xf) for t in model.trees_]
    ids_v = [t.leaf_ids(Xv) for t in model.trees_]
    m_f = model.base_margin_ + sum((t.value[i] for t, i in zip(model.trees_, ids_f)),
                                   np.zeros(len(Xf)))
    m_v = model.base_margin_ + sum((t.value[i] for t, i in zip(model.trees_, ids_v)),
                                   np.zeros(len(Xv)))
    _, h_f = grad_hess(yf, m_f, wf)
    while model.n_splits_ > max_splits and model.trees_:
        best = (np.inf, None, None)
        for j, tree in enumerate(model.trees_):
            Hf = np.bincount(ids_f[j], weights=h_f, minlength=len(tree.feature))
            for k in _candidates(tree):
                a, b = tree.left[k], tree.right[k]
                ha, hb = Hf[a], Hf[b]
                v = (ha * tree.value[a] + hb * tree.value[b]) / max(ha + hb, 1e-12)
                ra, rb = ids_v[j] == a, ids_v[j] == b
                rows = ra | rb
                if not rows.any():
                    best = min(best, (0.0, j, k), key=lambda x: x[0])
                    continue
                new = m_v[rows] + np.where(ra[rows], v - tree.value[a], v - tree.value[b])
                delta = float(np.sum(_ll_rows(yv[rows], new, wv[rows]))
                              - np.sum(_ll_rows(yv[rows], m_v[rows], wv[rows])))
                if delta < best[0]:
                    best = (delta, j, k)
        _, j, k = best
        if j is None:
            break
        tree = model.trees_[j]
        a, b = tree.left[k], tree.right[k]
        Hf = np.bincount(ids_f[j], weights=h_f, minlength=len(tree.feature))
        v = (Hf[a] * tree.value[a] + Hf[b] * tree.value[b]) / max(Hf[a] + Hf[b], 1e-12)
        old_f, old_v = tree.value[ids_f[j]], tree.value[ids_v[j]]
        tree.left[k] = tree.right[k] = tree.feature[k] = tree.threshold[k] = -1
        tree.value[k] = v
        tree = _reachable(tree)
        if len(tree.feature) == 1:  # a dropped stump: its constant goes to the base
            model.base_margin_ += float(tree.value[0])
            m_f += tree.value[0] - old_f
            m_v += tree.value[0] - old_v
            del model.trees_[j], ids_f[j], ids_v[j]
        else:
            model.trees_[j] = tree
            ids_f[j], ids_v[j] = tree.leaf_ids(Xf), tree.leaf_ids(Xv)
            m_f += tree.value[ids_f[j]] - old_f
            m_v += tree.value[ids_v[j]] - old_v
        removed += 1
        if removed % refit_every == 0 or model.n_splits_ <= max_splits:
            m_f = model._refit_binned(Xf, yf, wf, lam, sweeps)
            ids_v = [t.leaf_ids(Xv) for t in model.trees_]
            m_v = model.base_margin_ + sum((t.value[i] for t, i in zip(model.trees_, ids_v)),
                                           np.zeros(len(Xv)))
            ids_f = [t.leaf_ids(Xf) for t in model.trees_]
        _, h_f = grad_hess(yf, m_f, wf)
    return removed
