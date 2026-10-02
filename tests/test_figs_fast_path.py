"""The compiled FIGS path (``node_hist_margin``, ``newton_step_inplace``, the leaf cache in
``grow_figs``, ``_newton_step(..., ids=...)``) against the NumPy path it replaced, bit for bit."""

import numpy as np
import pytest

from bettertrees import FIGSClassifier
from bettertrees.sums._common import base_margin, grad_hess
from bettertrees.sums._kernels import best_cut_1d, newton_step_inplace, node_hist, node_hist_margin
from bettertrees.sums.smalltrees import SmallTree, grow_figs


def _data(seed=0, n=600, p=7, nan_frac=0.1, weights=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[rng.random((n, p)) < nan_frac] = np.nan
    z = np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * z))).astype(int)
    w = rng.uniform(0.2, 3.0, n) if weights else None
    return X, y, w


def _prepared(max_splits=8, lr=1.0, cap=None, seed=0, weights=True, **kw):
    X, y, w = _data(seed, weights=weights)
    est = FIGSClassifier(max_splits=max_splits, learning_rate=lr, max_delta_step=cap, **kw)
    est.lam_ = 2.0 * max_splits
    est.learning_rate_ = float(lr)
    _, Xb, nb, target, wt = est._prepare(X, y, w, None)
    est.base_margin_ = base_margin(target, wt)
    return est, Xb, nb, target, wt


def _grow_figs_numpy(est, Xb, nb, target, w, max_splits, max_trees=None):
    """``grow_figs`` as it was before the optimization (``grad_hess`` + ``node_hist``,
    ids recomputed, ``_newton_step`` without ``ids``)."""
    n, B = len(Xb), int(nb.max())
    trees, contribs = [], []
    margin = np.full(n, est.base_margin_)
    for _ in range(max_splits):
        best = (0.0, None, None, None, None)
        can_add = max_trees is None or len(trees) < max_trees
        candidates = list(range(len(trees))) + ([None] if can_add else [])
        for k in candidates:
            if k is None:
                g, h = grad_hess(target, margin, w)
                ids = np.zeros(n, dtype=np.int64)
                leaves, n_nodes = [0], 1
            else:
                g, h = grad_hess(target, margin - contribs[k], w)
                ids = trees[k].leaf_ids(Xb)
                leaves, n_nodes = trees[k].leaves, len(trees[k].feature)
            hist = node_hist(Xb, g, h, w, ids, n_nodes, B)
            for leaf in leaves:
                gains, cuts = best_cut_1d(hist[leaf], nb, est.lam_, est.min_weight)
                f = int(np.argmax(gains))
                if gains[f] > best[0]:
                    best = (float(gains[f]), k, leaf, f, int(cuts[f]))
        gain, k, leaf, f, t = best
        if gain <= 0.0:
            break
        if k is None:
            trees.append(SmallTree())
            contribs.append(np.zeros(n))
            k = len(trees) - 1
        trees[k].split(leaf, f, t)
        contribs[k], margin = est._newton_step(trees[k], Xb, target, margin, contribs[k], w)
        if est.backfit_sweeps:
            margin = est._backfit(trees, contribs, Xb, target, w, est.backfit_sweeps)
    return trees


@pytest.mark.parametrize("seed", [0, 1])
def test_node_hist_margin_matches_grad_hess_then_node_hist(seed):
    est, Xb, nb, target, w = _prepared(seed=seed)
    n, B = len(Xb), int(nb.max())
    rng = np.random.default_rng(seed + 10)
    margin = rng.normal(scale=3.0, size=n)
    contrib = rng.normal(size=n)
    ids = rng.integers(-1, 5, size=n).astype(np.int64)  # includes excluded rows (-1)
    g, h = grad_hess(target, margin - contrib, w)
    ref = node_hist(Xb, g, h, w, ids, 5, B)
    got = node_hist_margin(Xb, np.ascontiguousarray(target), np.ascontiguousarray(w),
                           margin, contrib, ids, 5, B)
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("lr", [0.3, 1.0])
@pytest.mark.parametrize("cap", [None, 0.7])
@pytest.mark.parametrize("weights", [True, False])
def test_newton_step_with_ids_is_bitwise_the_numpy_step(lr, cap, weights):
    est, Xb, nb, target, w = _prepared(lr=lr, cap=cap, weights=weights, max_splits=6)
    trees = _grow_figs_numpy(est, Xb, nb, target, w, 6)
    tree = trees[0]
    rng = np.random.default_rng(3)
    margin0 = est.base_margin_ + rng.normal(size=len(Xb))
    contrib0 = tree.value[tree.leaf_ids(Xb)]
    margin0 = margin0 + contrib0
    for _ in range(3):  # repeated steps from the current values
        t_ref = SmallTree(**{k: (v.copy() if hasattr(v, "copy") else list(v))
                             for k, v in vars(tree).items()})
        t_new = SmallTree(**{k: (v.copy() if hasattr(v, "copy") else list(v))
                             for k, v in vars(tree).items()})
        c_ref, m_ref = est._newton_step(t_ref, Xb, target, margin0.copy(), contrib0.copy(), w)
        ids = t_new.leaf_ids(Xb)
        m_new, c_new = margin0.copy(), contrib0.copy()
        est._newton_step(t_new, Xb, target, m_new, c_new, w, ids)
        assert np.array_equal(t_ref.value, t_new.value)
        assert np.array_equal(c_ref, c_new)
        assert np.array_equal(m_ref, m_new)
        tree, margin0, contrib0 = t_ref, m_ref, c_ref


def test_newton_step_inplace_kernel_cap_flag():
    est, Xb, nb, target, w = _prepared(max_splits=4)
    tree = _grow_figs_numpy(est, Xb, nb, target, w, 4)[0]
    ids = tree.leaf_ids(Xb)
    n = len(Xb)
    for cap, has in [(0.0, False), (0.25, True)]:
        v = tree.value.copy()
        m, c = np.full(n, 0.3), np.zeros(n)
        newton_step_inplace(target, w, m, c, ids, v, est.lam_, 1.0, cap, has)
        g, h = grad_hess(target, np.full(n, 0.3), w)
        G = np.bincount(ids, g, len(v))
        H = np.bincount(ids, h, len(v))
        step = -G / (H + est.lam_)
        if has:
            step = np.clip(step, -cap, cap)
        assert np.allclose(v, tree.value + step, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("lr", [0.3, 1.0])
@pytest.mark.parametrize("cap", [None, 1.5])
@pytest.mark.parametrize("sweeps", [1, 2])
@pytest.mark.parametrize("weights", [True, False])
def test_grow_figs_cache_is_bitwise_the_numpy_growth(lr, cap, sweeps, weights):
    est, Xb, nb, target, w = _prepared(max_splits=10, lr=lr, cap=cap, weights=weights,
                                       backfit_sweeps=sweeps)
    ref = _grow_figs_numpy(est, Xb, nb, target, w, 10)
    got = grow_figs(est, Xb, nb, target, w, 10)
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert a.feature == b.feature and a.threshold == b.threshold
        assert a.left == b.left and a.right == b.right
        assert np.array_equal(a.value, b.value)


def test_grow_figs_max_trees_and_estimator_end_to_end():
    est, Xb, nb, target, w = _prepared(max_splits=9)
    ref = _grow_figs_numpy(est, Xb, nb, target, w, 9, max_trees=2)
    got = grow_figs(est, Xb, nb, target, w, 9, max_trees=2)
    assert len(got) <= 2
    assert all(np.array_equal(a.value, b.value) and a.feature == b.feature
               for a, b in zip(ref, got))
    X, y, sw = _data(5)
    m = FIGSClassifier(max_splits=8, max_trees=3).fit(X, y, sample_weight=sw)
    assert np.isfinite(m.decision_function(X)).all()
