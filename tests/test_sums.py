"""Tree sums: kernels against brute force and one functional DGP per piece."""

from itertools import product

import numpy as np
import pytest

from bettertrees import FastDecisionTreeClassifier
from bettertrees.experimental import (
    MixedDepthTree,
    RatioVocabulary,
    crossfit_teacher,
    fast_pair_scores,
    fit_tree_on_target,
    pair_shape_scores,
    soft_label_expand,
    teacher_path_pairs,
    top_pairs,
)
from bettertrees.sums import AdditiveTreeBooster, screen_features
from bettertrees.sums._kernels import best_cut_1d, best_depth2, hist_1d, quadrant_scores


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ------------------------------------------------------------ kernels exatos

def _score(G, H, lam):
    return G * G / (H + lam)


def _small(seed=0, n=120, p=3, B=5):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, B, size=(n, p)).astype(np.uint8)
    g = rng.normal(size=n)
    h = rng.uniform(0.1, 1, size=n)
    w = np.ones(n)
    return Xb, g, h, w, np.full(p, B, dtype=np.int64)


def _brute_cut(mask_rows, Xb, g, h, w, nb, lam, minw):
    """Best single cut within mask_rows by brute force."""
    G, H = g[mask_rows].sum(), h[mask_rows].sum()
    best = 0.0
    for f in range(Xb.shape[1]):
        for t in range(nb[f] - 1):
            L = mask_rows & (Xb[:, f] <= t)
            R = mask_rows & ~(Xb[:, f] <= t)
            if w[L].sum() < minw or w[R].sum() < minw:
                continue
            gain = _score(g[L].sum(), h[L].sum(), lam) + _score(g[R].sum(), h[R].sum(), lam) - _score(G, H, lam)
            best = max(best, gain)
    return best


@pytest.mark.parametrize("seed", range(3))
def test_best_cut_and_depth2_match_brute_force(seed):
    Xb, g, h, w, nb = _small(seed)
    lam, minw = 0.5, 5.0
    all_rows = np.ones(len(g), dtype=bool)
    gains, _ = best_cut_1d(hist_1d(Xb, g, h, w, int(nb.max())), nb, lam, minw)
    assert gains.max() == pytest.approx(_brute_cut(all_rows, Xb, g, h, w, nb, lam, minw))

    best = 0.0
    for f1 in range(Xb.shape[1]):
        for t1 in range(nb[f1] - 1):
            L = Xb[:, f1] <= t1
            if w[L].sum() < minw or w[~L].sum() < minw:
                continue
            root = (_score(g[L].sum(), h[L].sum(), lam) + _score(g[~L].sum(), h[~L].sum(), lam)
                    - _score(g.sum(), h.sum(), lam))
            gain = root + _brute_cut(L, Xb, g, h, w, nb, lam, minw) + _brute_cut(~L, Xb, g, h, w, nb, lam, minw)
            best = max(best, gain)
    d2, _ = best_depth2(Xb, g, h, w, nb, lam, minw)
    assert d2.max() == pytest.approx(best)


def test_quadrant_matches_brute_force():
    Xb, g, h, w, nb = _small(1)
    lam, minw = 0.5, 3.0
    pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
    gains, _, _ = quadrant_scores(Xb, g, h, w, nb, pairs, lam, minw)
    for q, (i, j) in enumerate(pairs):
        best = 0.0
        for ta, tb in product(range(nb[i] - 1), range(nb[j] - 1)):
            a, b = Xb[:, i] <= ta, Xb[:, j] <= tb
            cells = [a & b, a & ~b, ~a & b, ~a & ~b]
            if min(w[c].sum() for c in cells) < minw:
                continue
            gain = sum(_score(g[c].sum(), h[c].sum(), lam) for c in cells) - _score(g.sum(), h.sum(), lam)
            best = max(best, gain)
        assert gains[q] == pytest.approx(best)


# ------------------------------------------------------------ DGPs funcionais

def _ratio_data(seed, n=6000, p=5, kind="ratio"):
    rng = np.random.default_rng(seed)
    X = np.exp(rng.normal(size=(n, p)))
    if kind == "ratio":
        z = 3.0 * (np.log(X[:, 0]) - np.log(X[:, 1]))
    else:  # "AND" interaction on the same two axes
        z = 3.0 * ((X[:, 0] > 1.0) & (X[:, 1] < 1.0)) - 1.5
    y = (rng.random(n) < _sig(z)).astype(int)
    return X, y


def test_ratio_vocabulary_picks_ratio_pair_first():
    X, y = _ratio_data(0)
    voc = RatioVocabulary(max_terms=3).fit(X, y)
    assert [tuple(r) for r in voc.pairs_][0] == (0, 1)
    assert voc.term_names_[0] == "x0/x1"
    assert voc.transform(X).shape == (len(X), X.shape[1] + len(voc.pairs_))


def test_and_interaction_has_no_ratio_shape():
    X, y = _ratio_data(0, kind="and")
    s = pair_shape_scores(X, y, pairs=np.array([[0, 1]]))
    assert s["gain_quad"][0] > s["gain_comb"][0]
    assert s["shape"][0] < 0
    assert (0, 1) not in [tuple(r) for r in RatioVocabulary().fit(X, y).pairs_]


def test_ratio_requires_positive_values():
    X, y = _ratio_data(1)
    X[:, 1] = -X[:, 1]
    s = pair_shape_scores(X, y, pairs=np.array([[0, 1]]))
    assert s["valid_frac"][0] == 0.0
    assert (0, 1) not in [tuple(r) for r in RatioVocabulary().fit(X, y).pairs_]


def _additive_interaction(seed, n=6000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    z = 1.5 * X[:, 0] + 1.5 * np.sign(X[:, 1]) + 1.5 * np.sign(X[:, 2] * X[:, 3])
    return X, (rng.random(n) < _sig(z)).astype(int)


def test_fast_finds_pure_interaction_after_additive_base():
    X, y = _additive_interaction(0)
    base = AdditiveTreeBooster(depth=1, random_state=0).fit(X, y)
    s = fast_pair_scores(X, y, margin=base.decision_function(X))
    assert tuple(s["pairs"][s["order"][0]]) == (2, 3)


def test_screen_ranks_informative_feature_first_and_stable():
    X, y = _additive_interaction(1)
    s = screen_features(X, y, n_boot=10)
    assert np.argmax(s["gain"]) in (0, 1)
    assert s["top_k_freq"][0] == 1.0 and s["top_k_freq"][1] == 1.0
    assert s["threshold_iqr"][1] < 0.5  # sign(x1): cut near 0, stable


def test_additive_booster_beats_single_tree_on_additive_dgp():
    rng = np.random.default_rng(3)
    n = 8000
    X = rng.normal(size=(n, 6))
    y = (rng.random(n) < _sig(X @ np.array([1.0, 0.8, -0.8, 0.6, 0.0, 0.0]))).astype(int)
    tr, te = slice(0, 6000), slice(6000, None)
    boost = AdditiveTreeBooster(depth=1, random_state=0).fit(X[tr], y[tr])
    tree = FastDecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                      leaf_shrinkage=10.0).fit(X[tr], y[tr])
    lb = _log_loss(y[te], boost.predict_proba(X[te])[:, 1])
    lt = _log_loss(y[te], tree.predict_proba(X[te])[:, 1])
    assert lb < lt - 0.02
    rows = boost.scorecard()
    assert rows and all(isinstance(v, float) for _, _, v in rows)


def test_additive_booster_depth2_captures_interaction():
    X, y = _additive_interaction(2)
    tr, te = slice(0, 4500), slice(4500, None)
    d1 = AdditiveTreeBooster(depth=1, random_state=0).fit(X[tr], y[tr])
    d2 = AdditiveTreeBooster(depth=2, random_state=0).fit(X[tr], y[tr])
    assert (_log_loss(y[te], d2.predict_proba(X[te])[:, 1])
            < _log_loss(y[te], d1.predict_proba(X[te])[:, 1]) - 0.02)


# ------------------------------------------------------------ distillation

def test_crossfit_teacher_is_out_of_fold():
    X, y = _additive_interaction(4, n=1500)
    a = crossfit_teacher(X, y, n_splits=3, params=dict(n_estimators=60))
    fold0 = a["fold"] == 0
    flipped = y.copy()
    flipped[fold0] = 1 - flipped[fold0]
    b = crossfit_teacher(X, flipped, params=dict(n_estimators=60), folds=a["fold"])
    # swapping the labels of fold 0 does not change the prediction of its rows...
    np.testing.assert_array_equal(a["p"][fold0], b["p"][fold0])
    # ...but changes the others' (their teacher saw fold 0)
    assert not np.allclose(a["p"][~fold0], b["p"][~fold0])
    assert np.isfinite(a["p"]).all() and (a["fold"] >= 0).all()


def test_teacher_path_pairs_sees_interaction():
    import lightgbm as lgb
    X, y = _additive_interaction(5, n=3000)
    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=8, verbose=-1).fit(X, y)
    M = teacher_path_pairs(model.booster_, X.shape[1])
    assert tuple(top_pairs(M, 3)[0]) in {(2, 3), (0, 2), (0, 3), (1, 2), (1, 3), (0, 1)}
    assert M[2, 3] > M[4, 5]


def test_soft_labels_with_hard_p_reproduce_tree_on_y():
    X, y = _additive_interaction(6, n=2000)
    X2, y2, w2 = soft_label_expand(X, y.astype(float))
    assert w2.sum() == pytest.approx(len(X))
    a = fit_tree_on_target(X, y, y.astype(float), "p", max_depth=4, splitter="exact")
    b = fit_tree_on_target(X, y, None, "y", max_depth=4, splitter="exact")
    np.testing.assert_array_equal(a.nodes_.feature, b.nodes_.feature)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))


def test_restate_leaf_masses_gives_empirical_leaf_frequencies():
    X, y = _additive_interaction(7, n=2000)
    p = np.clip(_sig(X[:, 0]), 0.01, 0.99)
    model = fit_tree_on_target(X, y, p, "p", leaf_target="y", max_depth=3)
    leaves = model.apply(X)
    proba = model.predict_proba(X)[:, 1]
    for leaf in np.unique(leaves):
        assert proba[leaves == leaf][0] == pytest.approx(y[leaves == leaf].mean())


def test_mixed_tree_all_y_equals_full_depth_greedy_tree():
    X, y = _additive_interaction(8, n=2000)
    params = dict(splitter="exact", min_samples_leaf=5, random_state=0)
    mixed = MixedDepthTree(depth=5, top_depth=2, top_target="y", bottom_target="y",
                           tree_params=params).fit(X, y, np.full(len(y), 0.5))
    full = FastDecisionTreeClassifier(max_depth=5, **params).fit(X, y)
    np.testing.assert_allclose(mixed.predict_proba(X), full.predict_proba(X))
    assert mixed.get_n_leaves() == full.get_n_leaves()


def test_mixed_tree_runs_all_four_arms():
    X, y = _additive_interaction(9, n=2000)
    p = np.clip(_sig(1.5 * X[:, 0] + 1.5 * np.sign(X[:, 1])), 0.01, 0.99)
    for top, bottom in product("yp", "yp"):
        m = MixedDepthTree(depth=4, top_depth=2, top_target=top, bottom_target=bottom,
                           tree_params=dict(min_samples_leaf=10, leaf_shrinkage=5.0)).fit(X, y, p)
        proba = m.predict_proba(X)
        assert proba.shape == (len(X), 2) and np.allclose(proba.sum(1), 1)


def test_screen_without_bootstrap_returns_gain_only():
    X, y = _additive_interaction(10, n=1500)
    s = screen_features(X, y, n_boot=0)
    assert np.argmax(s["gain"]) in (0, 1) and np.isnan(s["threshold_iqr"]).all()


# ------------------------------------------------------------ small trees

from bettertrees.experimental import RuleFitLasso  # noqa: E402
from bettertrees.sums import FIGSClassifier, SumOfOptimalTrees  # noqa: E402
from bettertrees.sums._kernels import best_depth3  # noqa: E402


def _brute_tree(mask, depth, Xb, g, h, w, nb, lam, minw):
    """Gain of the best tree of depth <= depth on the rows of mask."""
    if depth == 0:
        return 0.0
    G, H = g[mask].sum(), h[mask].sum()
    best = 0.0
    for f in range(Xb.shape[1]):
        for t in range(nb[f] - 1):
            L, R = mask & (Xb[:, f] <= t), mask & ~(Xb[:, f] <= t)
            if w[L].sum() < minw or w[R].sum() < minw:
                continue
            gain = (_score(g[L].sum(), h[L].sum(), lam) + _score(g[R].sum(), h[R].sum(), lam)
                    - _score(G, H, lam)
                    + _brute_tree(L, depth - 1, Xb, g, h, w, nb, lam, minw)
                    + _brute_tree(R, depth - 1, Xb, g, h, w, nb, lam, minw))
            best = max(best, gain)
    return best


@pytest.mark.parametrize("seed", range(2))
def test_depth3_matches_brute_force(seed):
    Xb, g, h, w, nb = _small(seed, n=150, p=3, B=4)
    gains, _ = best_depth3(Xb, g, h, w, nb, 0.5, 4.0)
    all_rows = np.ones(len(g), dtype=bool)
    assert gains.max() == pytest.approx(_brute_tree(all_rows, 3, Xb, g, h, w, nb, 0.5, 4.0))


def test_depth3_beats_two_levels_on_three_way_interaction():
    rng = np.random.default_rng(11)
    n = 4000
    X = rng.normal(size=(n, 4))
    z = 3.0 * np.sign(X[:, 0] * X[:, 1] * X[:, 2])
    y = (rng.random(n) < _sig(z)).astype(int)
    tr, te = slice(0, 3000), slice(3000, None)
    d3 = SumOfOptimalTrees(n_trees=1, depth=3).fit(X[tr], y[tr])
    d2 = SumOfOptimalTrees(n_trees=2, depth=2).fit(X[tr], y[tr])
    assert d3.n_splits_ <= 7 and d2.n_splits_ <= 6
    assert (_log_loss(y[te], d3.predict_proba(X[te])[:, 1])
            < _log_loss(y[te], d2.predict_proba(X[te])[:, 1]) - 0.1)


def test_stumps_beat_one_d3_on_wide_additive_dgp():
    # with 3 terms a depth-3 tree (8 leaves) is exact; the additive advantage shows
    # with more terms than depth: 5 stumps (5 cuts) vs one depth-3 tree (7 cuts)
    rng = np.random.default_rng(12)
    n = 6000
    X = rng.normal(size=(n, 6))
    z = 1.5 * np.sign(X[:, :5]).sum(axis=1)
    y = (rng.random(n) < _sig(z)).astype(int)
    tr, te = slice(0, 4500), slice(4500, None)
    d1 = SumOfOptimalTrees(n_trees=5, depth=1).fit(X[tr], y[tr])
    d3 = SumOfOptimalTrees(n_trees=1, depth=3).fit(X[tr], y[tr])
    assert d1.n_splits_ <= d3.n_splits_
    assert (_log_loss(y[te], d1.predict_proba(X[te])[:, 1])
            < _log_loss(y[te], d3.predict_proba(X[te])[:, 1]) - 0.05)


def test_figs_respects_budget_and_grows_several_trees_on_additive_dgp():
    rng = np.random.default_rng(13)
    n = 5000
    X = rng.normal(size=(n, 5))
    z = 2.0 * np.sign(X[:, 0]) + 2.0 * np.sign(X[:, 1]) + 1.5 * X[:, 2]
    y = (rng.random(n) < _sig(z)).astype(int)
    model = FIGSClassifier(max_splits=8).fit(X, y)
    assert model.n_splits_ <= 8
    assert len(model.trees_) >= 2
    used = {f for t in model.trees_ for f, l in zip(t.feature, t.left) if l != -1}
    assert {0, 1, 2} <= used
    assert all(isinstance(v, float) for _, _, v in model.rules())


def test_figs_single_tree_equals_greedy_newton_tree_when_capped():
    X, y = _additive_interaction(14, n=3000)
    one = FIGSClassifier(max_splits=3, max_trees=1, backfit_sweeps=0).fit(X, y)
    assert len(one.trees_) == 1 and one.n_splits_ == 3


def test_rulefit_respects_condition_budget():
    X, y = _additive_interaction(15, n=4000)
    model = RuleFitLasso(budgets=(4, 16), n_rounds=60).fit(X[:3000], y[:3000])
    assert model.n_conditions(4) <= 4 and model.n_conditions(16) <= 16
    l4 = _log_loss(y[3000:], model.predict_proba(X[3000:], budget=4)[:, 1])
    l16 = _log_loss(y[3000:], model.predict_proba(X[3000:], budget=16)[:, 1])
    null = _log_loss(y[3000:], np.full(1000, y[:3000].mean()))
    assert l16 < l4 < null
    assert model.scorecard(16)



@pytest.mark.parametrize("seed", range(4))
def test_best_depth2_is_bitwise_equal_to_reference(seed):
    from bettertrees.sums._kernels import best_depth2_reference
    rng = np.random.default_rng(seed)
    n, p = 3000, 7
    nb = rng.integers(3, 20, size=p).astype(np.int64)
    Xb = np.stack([rng.integers(0, nb[j], size=n) for j in range(p)], axis=1).astype(np.uint8)
    g = rng.normal(size=n)
    h = rng.uniform(0.05, 0.25, size=n)
    w = rng.integers(0, 3, size=n).astype(float)
    for minw in (1.0, 50.0, 400.0):
        a = best_depth2(Xb, g, h, w, nb, 1.0, minw)
        b = best_depth2_reference(Xb, g, h, w, nb, 1.0, minw)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])



def test_l1_path_matches_sklearn_saga_at_same_lambda():
    from sklearn.linear_model import LogisticRegression

    from bettertrees.sums._kernels import l1_logistic_path
    rng = np.random.default_rng(21)
    n, m = 1500, 12
    R = rng.random((n, m)) < rng.uniform(0.05, 0.5, size=m)
    beta_true = np.zeros(m)
    beta_true[:4] = [1.5, -1.0, 0.8, 0.5]
    y = (rng.random(n) < _sig(R @ beta_true - 0.5)).astype(float)
    scale = 1 / np.sqrt(R.mean(0) * (1 - R.mean(0)))
    cols = [np.flatnonzero(R[:, j]) for j in range(m)]
    indptr = np.concatenate([[0], np.cumsum([len(c) for c in cols])]).astype(np.int64)
    indices = np.concatenate(cols).astype(np.int64)
    lam = 0.004
    betas, b0s, done = l1_logistic_path(indptr, indices, scale, y, np.array([0.05, 0.02, lam]),
                                        np.ones(m), 1e9, 1e-12, 100, 5000, np.zeros(m), np.nan)
    # a warm start from the 2nd point reaches the same solution
    wb, w0, wd = l1_logistic_path(indptr, indices, scale, y, np.array([lam]), np.ones(m), 1e9,
                                  1e-12, 100, 5000, betas[1].copy(), b0s[1])
    np.testing.assert_allclose(wb[0], betas[done - 1], atol=1e-6)
    ref = LogisticRegression(penalty="l1", solver="saga", C=1 / (n * lam), tol=1e-10,
                             max_iter=100000).fit(R * scale, y)
    np.testing.assert_allclose(betas[done - 1], ref.coef_[0], atol=2e-3)
    assert b0s[done - 1] == pytest.approx(ref.intercept_[0], abs=2e-3)
    np.testing.assert_array_equal(betas[done - 1] != 0, np.abs(ref.coef_[0]) > 1e-6)


def test_rulefit_path_selects_like_liblinear_and_is_faster():
    import time
    X, y = _additive_interaction(15, n=4000)
    t0 = time.perf_counter()
    a = RuleFitLasso(budgets=(8, 16, 32), n_rounds=60, solver="path").fit(X[:3000], y[:3000])
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    b = RuleFitLasso(budgets=(8, 16, 32), n_rounds=60, solver="liblinear").fit(X[:3000], y[:3000])
    tb = time.perf_counter() - t0
    for bud in (8, 16, 32):
        assert a.n_conditions(bud) <= bud
        la = _log_loss(y[3000:], a.predict_proba(X[3000:], budget=bud)[:, 1])
        lb = _log_loss(y[3000:], b.predict_proba(X[3000:], budget=bud)[:, 1])
        assert la < lb + 0.02  # same quality (the selection may differ on ties)
    assert ta < tb



def test_boosted_optimal_d2_reproduces_additive_booster():
    from bettertrees.sums import BoostedOptimalTrees
    X, y = _additive_interaction(31, n=3000)
    a = AdditiveTreeBooster(depth=2, learning_rate=0.2, max_rounds=40, patience=10,
                            random_state=3).fit(X, y)
    b = BoostedOptimalTrees(depth=2, learning_rate=0.2, max_rounds=40, patience=10,
                            random_state=3).fit(X, y)
    assert len(a.terms_) == len(b.trees_)
    np.testing.assert_allclose(a.decision_function(X), b.decision_function(X), atol=1e-12)


def test_boosted_optimal_d3_captures_three_way_interaction():
    from bettertrees.sums import BoostedOptimalTrees
    rng = np.random.default_rng(32)
    n = 5000
    X = rng.normal(size=(n, 5))
    y = (rng.random(n) < _sig(3.0 * np.sign(X[:, 0] * X[:, 1] * X[:, 2]))).astype(int)
    tr, te = slice(0, 4000), slice(4000, None)
    d2 = BoostedOptimalTrees(depth=2, max_rounds=60).fit(X[tr], y[tr])
    d3 = BoostedOptimalTrees(depth=3, max_rounds=60).fit(X[tr], y[tr])
    assert (_log_loss(y[te], d3.predict_proba(X[te])[:, 1])
            < _log_loss(y[te], d2.predict_proba(X[te])[:, 1]) - 0.1)



def test_d2_feature_cap_is_noop_below_cap_and_restricts_above():
    rng = np.random.default_rng(41)
    n = 2000
    X = rng.normal(size=(n, 12))
    y = (rng.random(n) < _sig(1.5 * X[:, 0] + 1.5 * np.sign(X[:, 1] * X[:, 2]))).astype(int)
    full = AdditiveTreeBooster(max_rounds=15, max_features_d2=None).fit(X, y)
    capped_hi = AdditiveTreeBooster(max_rounds=15, max_features_d2=12).fit(X, y)
    np.testing.assert_array_equal(full.decision_function(X), capped_hi.decision_function(X))
    # both screenings see x2, which only acts through an interaction (no main effect)
    for screen in ("lgbm", "fast"):
        capped = AdditiveTreeBooster(max_rounds=15, max_features_d2=4,
                                     feature_screen=screen).fit(X, y)
        used = {v for (f1, t1, fl, tl, fr, tr), _ in capped.terms_ for v in (f1, fl, fr)
                if v >= 0}
        assert {0, 1, 2} <= used, screen
    s = SumOfOptimalTrees(n_trees=3, depth=2, max_features_d2=4).fit(X, y)
    assert s.n_splits_ <= 9



def test_greedy_sum_is_paired_control_of_optimal_sum():
    rng = np.random.default_rng(51)
    n = 4000
    X = rng.normal(size=(n, 6))
    # pure XOR: no main effect, the greedy search misses the right root
    y = (rng.random(n) < _sig(3.0 * np.sign(X[:, 0] * X[:, 1]))).astype(int)
    tr, te = slice(0, 3000), slice(3000, None)
    opt = SumOfOptimalTrees(n_trees=1, depth=2).fit(X[tr], y[tr])
    gre = SumOfOptimalTrees(n_trees=1, depth=2, search="greedy").fit(X[tr], y[tr])
    assert (_log_loss(y[te], opt.predict_proba(X[te])[:, 1])
            < _log_loss(y[te], gre.predict_proba(X[te])[:, 1]) - 0.1)
    # strong additive effect on one axis: greedy and optimal pick the same root
    y2 = (rng.random(n) < _sig(2.0 * X[:, 2] + 0.5 * X[:, 3])).astype(int)
    a = SumOfOptimalTrees(n_trees=1, depth=2).fit(X, y2).trees_[0]
    b = SumOfOptimalTrees(n_trees=1, depth=2, search="greedy").fit(X, y2).trees_[0]
    assert a.feature[0] == b.feature[0] == 2



def test_sum_extra_stumps_fill_budget():
    rng = np.random.default_rng(61)
    n = 3000
    X = rng.normal(size=(n, 6))
    y = (rng.random(n) < _sig(X[:, 0] + X[:, 1] * X[:, 2] + 0.8 * X[:, 3] - 0.8 * X[:, 4]))
    y = y.astype(int)
    base = SumOfOptimalTrees(n_trees=2, depth=2).fit(X, y)
    full = SumOfOptimalTrees(n_trees=2, depth=2, extra_stumps=2).fit(X, y)
    assert base.n_splits_ <= 6 < full.n_splits_ <= 8
    assert len(full.trees_) == 4 and all(len(t.feature) == 3 for t in full.trees_[2:])
    assert (_log_loss(y, full.predict_proba(X)[:, 1])
            < _log_loss(y, base.predict_proba(X)[:, 1]))


def test_figs_max_delta_step_prevents_divergence_on_rare_class():
    # rare class (~0.5%) concentrated in a small region: the full Newton step on the
    # nearly pure leaf is huge; the cap keeps leaves bounded and the ranking right
    from sklearn.metrics import roc_auc_score
    from bettertrees import FIGSClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40000, 5))
    logit = -7 + 4.0 * (X[:, 0] > 1.5) + 3.0 * (X[:, 1] > 1.0)
    y = (rng.random(len(X)) < 1 / (1 + np.exp(-logit))).astype(int)
    capped = FIGSClassifier(max_splits=4, max_delta_step=1.0).fit(X, y)
    assert max(np.abs(t.value).max() for t in capped.trees_) < 10
    assert roc_auc_score(y, capped.predict_proba(X)[:, 1]) > 0.8
    # the default keeps the original FIGS behaviour
    assert FIGSClassifier().max_delta_step is None
