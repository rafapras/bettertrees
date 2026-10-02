"""Experimental oblique-split FIGS (``bettertrees.experimental.oblique``)."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.utils.estimator_checks import parametrize_with_checks

from bettertrees import FIGSClassifier
from bettertrees.experimental import ObliqueFIGSClassifier
from bettertrees.experimental.oblique import project
from bettertrees.sums._common import grad_hess


def rule(budget):
    return dict(max_splits=budget, learning_rate=1. if budget <= 8 else .3,
                max_delta_step=4. if budget <= 8 else None)


def mixed(seed=0, n=1500, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    eta = 1.5 * np.sign(X[:, 0]) + X[:, 1] - X[:, 2] * X[:, 3] + .5 * X[:, 4]
    soft = 1 / (1 + np.exp(-eta))
    return X, (rng.random(n) < soft).astype(int), soft


def diagonal(seed=0, n=6000, p=5, noise=.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = ((X[:, 0] + X[:, 1] + noise * rng.normal(size=n)) > 0).astype(int)
    return X, y


def rare(seed=3, n=3000, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    eta = -3.5 + 1.5 * (X[:, 0] > 1) + X[:, 1]
    return X, (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int), None


def assert_same(a, b, X):
    assert a.base_margin_ == b.base_margin_
    assert len(a.trees_) == len(b.trees_)
    for ta, tb in zip(a.trees_, b.trees_):
        for name in ("feature", "threshold", "left", "right", "value"):
            np.testing.assert_array_equal(getattr(ta, name), getattr(tb, name))
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


# ------------------------------------------------------------ (i) oblique=False = FIGS, bit for bit

@pytest.mark.parametrize("make", [mixed, rare])
@pytest.mark.parametrize("budget", [4, 16])
@pytest.mark.parametrize("rate", [1.0, 0.3])
@pytest.mark.parametrize("cap", [None, 4.0])
def test_axis_only_is_figs_bitwise(make, budget, rate, cap):
    X, y, soft = make()
    X = X.copy()
    X[::13, 0] = np.nan
    X[:, -1] = 1.0  # a constant column
    params = dict(max_splits=budget, learning_rate=rate, max_delta_step=cap)
    a = FIGSClassifier(**params).fit(X, y)
    b = ObliqueFIGSClassifier(**params, oblique=False).fit(X, y)
    assert b.n_oblique_splits_ == 0 and b.oblique_splits_ == []
    assert_same(a, b, X)


@pytest.mark.parametrize("extra", [dict(max_trees=1), dict(max_trees=2, backfit_sweeps=2),
                                   dict(lam=3.0, min_weight=5.0, max_bins=8)])
def test_axis_only_is_figs_bitwise_weights_soft_and_options(extra):
    X, y, soft = mixed(seed=4)
    w = np.arange(len(y)) % 4
    a = FIGSClassifier(**rule(8), **extra).fit(X, y, sample_weight=w, y_soft=soft)
    b = ObliqueFIGSClassifier(**rule(8), **extra, oblique=False).fit(X, y, sample_weight=w,
                                                                       y_soft=soft)
    assert_same(a, b, X)


def test_no_oblique_winner_keeps_figs_bitwise():
    # one feature: no oblique candidate exists, so the oblique path must equal FIGS
    X, y, _ = mixed()
    a = FIGSClassifier(**rule(8)).fit(X[:, :1], y)
    b = ObliqueFIGSClassifier(**rule(8)).fit(X[:, :1], y)
    assert_same(a, b, X[:, :1])


# ------------------------------------------------------------ (ii) diagonal boundary

def test_one_oblique_cut_beats_eight_axis_cuts_on_a_diagonal():
    X, y = diagonal()
    Xt, yt = diagonal(seed=1)
    one = ObliqueFIGSClassifier(**rule(1)).fit(X, y)
    eight = FIGSClassifier(**rule(8)).fit(X, y)
    assert one.n_oblique_splits_ == 1 and one.n_splits_ == 1
    ll_one = log_loss(yt, one.predict_proba(Xt)[:, 1])
    ll_eight = log_loss(yt, eight.predict_proba(Xt)[:, 1])
    assert ll_one < ll_eight
    s = one.oblique_splits_[0]
    assert s["features"] == (0, 1)
    cos = abs(s["weights"] @ np.array([1, 1]) / np.sqrt(2))
    assert cos > 0.99
    raw = s["coef"] / np.linalg.norm(s["coef"])  # raw units, same scale per feature here
    assert abs(raw @ np.array([1, 1]) / np.sqrt(2)) > 0.99
    assert abs(s["raw_threshold"] / np.linalg.norm(s["coef"])) < .1


# ------------------------------------------------------------ (iii) dense reference

def test_projection_hist_matches_searchsorted():
    from bettertrees.experimental._oblique_kernels import projection_hist
    from bettertrees.experimental.oblique import quantile_edges
    rng = np.random.default_rng(12)
    z = np.round(rng.normal(size=500), 1)  # ties on the edges
    g, h, w = rng.normal(size=500), rng.uniform(.1, .3, 500), rng.uniform(0, 2, 500)
    edges = quantile_edges(z, 16)
    b = np.searchsorted(edges, z, side="left")
    ref = np.column_stack([np.bincount(b, v, len(edges) + 1) for v in (g, h, w)])
    np.testing.assert_allclose(projection_hist(z, g, h, w, edges), ref, rtol=1e-12, atol=1e-12)
    assert ((z <= edges[3]) == (b <= 3)).all()


def dense_oblique(X, y, w, margin, lam, ridge, min_weight, subsets):
    """Exhaustive best oblique cut: dense ridge with intercept, every distinct threshold."""
    g, h = grad_hess(y.astype(float), margin, w)
    mu = (w @ X) / w.sum()
    sd = np.sqrt(w @ (X - mu) ** 2 / w.sum())
    Xs = (X - mu) / sd
    best = (0.0, None, None, None)
    for S in subsets:
        D = np.column_stack([np.ones(len(X)), Xs[:, S]])
        P = np.diag([0.0] + [ridge] * len(S))
        beta = np.linalg.solve(D.T @ (D * h[:, None]) + P, D.T @ (h * (-g / h)))
        v = beta[1:] / np.linalg.norm(beta[1:])
        z = Xs[:, S] @ v
        for t in np.unique(z)[:-1]:
            right = z > t
            if w[~right].sum() < min_weight or w[right].sum() < min_weight:
                continue
            gain = sum(g[m].sum() ** 2 / (h[m].sum() + lam) for m in (right, ~right)) \
                - g.sum() ** 2 / (h.sum() + lam)
            if gain > best[0]:
                best = (gain, tuple(S), v, right)
    return best


@pytest.mark.parametrize("weighted", [False, True])
def test_root_oblique_split_matches_dense_reference(weighted):
    X, y = diagonal(seed=5, n=300, p=4, noise=.5)
    w = (1.0 + np.arange(len(y)) % 3) if weighted else np.ones(len(y))
    est = ObliqueFIGSClassifier(max_splits=1, lam=2.0, min_weight=10.0, oblique_bins=10_000,
                                n_subsets=6).fit(X, y, sample_weight=w)
    first = est.split_history_[0]
    assert first["kind"] == "oblique"
    margin = np.full(len(y), est.base_margin_)
    pairs = [list(c) for c in combinations(range(4), 2)]  # top-4 pairs = every pair here
    gain, S, v, right = dense_oblique(X, y, w, margin, 2.0, 2.0, 10.0, pairs)
    assert first["gain"] == pytest.approx(gain, rel=1e-9)
    s = est.oblique_splits_[0]
    assert s["features"] == S
    np.testing.assert_allclose(s["weights"], v, atol=1e-8)
    Xs = est._standardize(X, s["features"])
    z = project([Xs[:, i] for i in range(len(S))], s["weights"])
    np.testing.assert_array_equal(z > s["threshold"], right)


def test_oblique_gain_of_every_leaf_candidate_matches_masks():
    # a fitted model: rescore the leaves of tree 0 at the margin of the other trees
    X, y, _ = mixed(seed=2, n=2000)
    est = ObliqueFIGSClassifier(**rule(8)).fit(X, y)
    w = np.ones(len(y))
    Xaug = est._augmented(X)
    ids = est._leaf_ids(Xaug)
    margin = est.decision_function(X) - est.trees_[0].value[ids[0]]
    g, h = grad_hess(y.astype(float), margin, w)
    Xs = est._standardize(X)
    rng = np.random.default_rng(0)
    checked = 0
    for leaf in est.trees_[0].leaves:
        rows = np.flatnonzero(ids[0] == leaf)
        best, found = est._best_oblique(Xs, rows, g, h, w, np.arange(X.shape[1], 0, -1.0), rng,
                                        return_all=True)
        for c in found:
            z = project([Xs[rows, f] for f in c["features"]], c["weights"])
            right = z > c["threshold"]
            # a dot product rounds differently, but only rows at the boundary can move
            alt = Xs[np.ix_(rows, c["features"])] @ c["weights"] > c["threshold"]
            assert (alt != right).sum() <= 2
            gr, hr = g[rows], h[rows]
            ref = sum(gr[m].sum() ** 2 / (hr[m].sum() + est.lam_) for m in (right, ~right)) \
                - gr.sum() ** 2 / (hr.sum() + est.lam_)
            assert c["gain"] == pytest.approx(ref, rel=1e-9, abs=1e-12)
            assert right.sum() >= est.min_weight and (~right).sum() >= est.min_weight
            checked += 1
        if found:
            assert best["gain"] == max(c["gain"] for c in found)
    assert checked > 0


# ------------------------------------------------------------ (iv) NaN and raw X

def test_nan_enters_oblique_conditions_at_the_mean():
    X, y = diagonal(seed=6)
    X = X.copy()
    X[::7, 0] = np.nan
    est = ObliqueFIGSClassifier(**rule(1)).fit(X, y)
    assert est.n_oblique_splits_ == 1
    rows = np.array([[np.nan, .3, 0, 0, 0], [est.center_[0], .3, 0, 0, 0],
                     [np.nan, -2.0, 0, 0, 0], [est.center_[0], -2.0, 0, 0, 0]])
    m = est.decision_function(rows)
    assert m[0] == m[1] and m[2] == m[3] and m[0] != m[2]
    assert np.isfinite(est.decision_function(X)).all()


def test_predict_on_raw_dataframe_and_training_margin():
    X, y, _ = mixed(seed=7, n=2500)
    X[::9, 1] = np.nan
    df = pd.DataFrame(X, columns=[f"c{j}" for j in range(X.shape[1])])
    est = ObliqueFIGSClassifier(**rule(16)).fit(df, y)
    assert est.n_oblique_splits_ > 0
    with pytest.warns(UserWarning, match="feature names"):
        np.testing.assert_array_equal(est.decision_function(df), est.decision_function(X))
    np.testing.assert_allclose(est.base_margin_ + est.predict_contributions(df).sum(1),
                               est.decision_function(df), atol=1e-12)
    text = est.export_text()
    assert "*c" in text and "missing value enters at its training mean" in text
    assert len(est.rules()) == est.n_leaves_
    # the raw-unit condition is the standardized one (up to rounding at the boundary)
    for s in est.oblique_splits_:
        Xs = est._standardize(X, s["features"])
        std = Xs @ s["weights"] > s["threshold"]
        raw_x = np.where(np.isnan(X[:, s["features"]]), est.center_[list(s["features"])],
                         X[:, s["features"]])
        raw = raw_x @ s["coef"] > s["raw_threshold"]
        assert (std == raw).mean() > .999


def test_stored_split_reproduces_training_partition():
    X, y, _ = mixed(seed=8, n=3000)
    X[::5, 2] = np.nan
    est = ObliqueFIGSClassifier(**rule(16), subset_strategy="random", n_subsets=8).fit(X, y)
    assert est.n_oblique_splits_ > 0
    # predicting the training rows again gives the margin built during growth
    from bettertrees.sums._common import rebin
    p = X.shape[1]
    Xaug = est._augmented(X)
    np.testing.assert_array_equal(Xaug[:, :p], rebin(X, est.bin_edges_))
    assert set(np.unique(Xaug[:, p:])) <= {1, 2}


# ------------------------------------------------------------ (v) accounting

def brute_counts(est, X):
    Xaug = est._augmented(X)
    p = est.n_features_in_
    cond = var = 0
    for i in range(len(Xaug)):
        seen = set()
        for tree in est.trees_:
            node = 0
            while tree.left[node] != -1:
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                seen.add((f, t))
                node = tree.left[node] if Xaug[i, f] <= t else tree.right[node]
        cond += len(seen)
        var += sum(len(est.oblique_splits_[f - p]["features"]) if f >= p else 1 for f, _ in seen)
    return cond / len(Xaug), var / len(Xaug)


@pytest.mark.parametrize("oblique", [True, False])
def test_accounting_is_consistent(oblique):
    X, y, _ = mixed(seed=9, n=2500)
    est = ObliqueFIGSClassifier(**rule(16), oblique=oblique).fit(X, y)
    assert est.n_splits_ == est.n_axis_splits_ + est.n_oblique_splits_ <= 16
    assert est.n_leaves_ == est.n_splits_ + len(est.trees_)
    assert est.n_params_ == est.n_leaves_ + sum(len(s["features"]) for s in est.oblique_splits_)
    assert est.n_params_with_thresholds_ == est.n_params_ + est.n_splits_
    assert est.n_distinct_splits_ <= est.n_splits_
    cond, var = est.conditions_read(X[:400]), est.variables_read(X[:400])
    assert 1 <= cond <= est.n_splits_
    assert (cond, var) == pytest.approx(brute_counts(est, X[:400]), rel=1e-12)
    assert est.variables_read(X[:400], distinct=True) <= var
    if oblique:
        assert est.n_oblique_splits_ > 0 and var > cond
    else:
        assert est.n_params_ == est.n_leaves_ and var == cond


def test_oblique_cost_spends_budget():
    X, y, _ = mixed(seed=10, n=2500)
    est = ObliqueFIGSClassifier(**rule(16), oblique_cost=2).fit(X, y)
    assert est.n_axis_splits_ + 2 * est.n_oblique_splits_ <= 16


def test_three_feature_subsets_and_random_strategy_are_deterministic():
    X, y, _ = mixed(seed=11, n=2500)
    a = ObliqueFIGSClassifier(**rule(8), max_oblique_features=3).fit(X, y)
    assert all(len(s["features"]) <= 3 for s in a.oblique_splits_)
    r1 = ObliqueFIGSClassifier(**rule(8), subset_strategy="random", random_state=3).fit(X, y)
    r2 = ObliqueFIGSClassifier(**rule(8), subset_strategy="random", random_state=3).fit(X, y)
    np.testing.assert_array_equal(r1.decision_function(X), r2.decision_function(X))


# ------------------------------------------------------------ (vi) sklearn

@parametrize_with_checks([ObliqueFIGSClassifier(max_splits=8),
                          ObliqueFIGSClassifier(max_splits=4, subset_strategy="random",
                                                min_weight=1.0)])
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_clone_get_params_and_refit():
    X, y, _ = mixed(n=1500)
    est = ObliqueFIGSClassifier(**rule(8), n_subsets=3)
    assert clone(est).get_params() == est.get_params()
    est.fit(X, y)
    est.set_params(oblique=False).fit(X, y)
    assert est.n_oblique_splits_ == 0 and est.oblique_splits_ == []


@pytest.mark.parametrize("params", [dict(max_splits=-1), dict(max_oblique_features=1),
                                    dict(n_subsets=0), dict(subset_strategy="all"),
                                    dict(oblique_cost=0), dict(oblique_bins=1),
                                    dict(oblique_ridge=-1.0), dict(learning_rate=0),
                                    dict(lam=float("nan")), dict(max_delta_step=-1)])
def test_invalid_parameters(params):
    X, y, _ = mixed(n=100)
    with pytest.raises(ValueError):
        ObliqueFIGSClassifier(**params).fit(X, y)
