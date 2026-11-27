"""G0 correctness and invariant tests for experimental research tracks R1-R6."""

import json
import math
import sqlite3
import sys
from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss
from sklearn.utils.estimator_checks import parametrize_with_checks

from bettertrees import FIGSClassifier
from bettertrees.bins import fit_bin_edges
from bettertrees.experimental import (
    BudgetRegressor,
    CategoricalFIGSClassifier,
    CompactTreeBoosterV2,
    FIGSRegressor,
    L2FIGSClassifier,
    LookaheadFIGSClassifier,
    SumOfOptimalTreesRegressor,
    refit_leaves_l2,
)
from bettertrees.sums.categorical import _best_cat_hist
from bettertrees.sums._common import grad_hess, rebin
from bettertrees.sums.smalltrees import SmallTree


def _assert_same_classifier(a, b, X):
    assert a.base_margin_ == b.base_margin_
    assert len(a.trees_) == len(b.trees_)
    for ta, tb in zip(a.trees_, b.trees_):
        for attr in ("feature", "threshold", "left", "right", "value"):
            np.testing.assert_array_equal(getattr(ta, attr), getattr(tb, attr))
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def _binary_data(seed=0, n=1200, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    eta = 1.2 * np.sign(X[:, 0]) + 0.9 * X[:, 1] - 0.8 * X[:, 2] * X[:, 3]
    soft = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < soft).astype(int)
    return X, y, soft


# ------------------------------------------------------------ R1: L2FIGSClassifier


@pytest.mark.parametrize("budget", [4, 16])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("lr_cap", [(1.0, 4.0), (0.3, None)])
def test_r1_proximal_no_base_update_matches_figs_bitwise(budget, weighted, lr_cap):
    X, y, soft = _binary_data(seed=1)
    X = X.copy()
    X[::11, 0] = np.nan
    X[:, -1] = 1.0
    w = (np.arange(len(y)) % 4).astype(float) if weighted else None
    lr, cap = lr_cap
    params = dict(max_splits=budget, learning_rate=lr, max_delta_step=cap, backfit_sweeps=2)
    ref = FIGSClassifier(**params).fit(X, y, sample_weight=w, y_soft=soft if weighted else None)
    cand = L2FIGSClassifier(**params, penalty="proximal", update_base=False).fit(
        X, y, sample_weight=w, y_soft=soft if weighted else None
    )
    _assert_same_classifier(ref, cand, X)


def test_r1_l2_fixed_structure_converges_to_exact_penalized_gradient():
    # Single tree (disjoint leaves) with lam=10 converges below 1e-9 within 30 sweeps;
    # multi-tree sum also converges below 1e-9 with lam=10.0 as sweeps increase.
    X, y, _ = _binary_data(seed=2, n=150)
    est1 = L2FIGSClassifier(
        max_splits=4, max_trees=1, lam=10.0, min_weight=5.0, penalty="l2", update_base=True
    ).fit(X, y)
    refit_leaves_l2(
        est1, X, y, lam=10.0, sweeps=30, refit_base=True, learning_rate=1.0, max_delta_step=None
    )
    Xb1 = rebin(X, est1.bin_edges_)
    g1, _ = grad_hess(y.astype(float), est1.decision_function(X), np.ones(len(y)))
    assert abs(float(g1.sum())) < 1e-9
    max_g1 = max(
        abs(float(g1[t.leaf_ids(Xb1) == lf].sum()) + 10.0 * float(t.value[lf]))
        for t in est1.trees_
        for lf in t.leaves
    )
    assert max_g1 < 1e-9

    X_m, y_m, _ = _binary_data(seed=2, n=1500)
    est_m = L2FIGSClassifier(max_splits=8, lam=10.0, penalty="l2", update_base=True).fit(X_m, y_m)
    refit_leaves_l2(
        est_m, X_m, y_m, lam=10.0, sweeps=300, refit_base=True, learning_rate=1.0,
        max_delta_step=None,
    )
    Xb_m = rebin(X_m, est_m.bin_edges_)
    gm, _ = grad_hess(y_m.astype(float), est_m.decision_function(X_m), np.ones(len(y_m)))
    assert abs(float(gm.sum())) < 1e-9
    max_gm = max(
        abs(float(gm[t.leaf_ids(Xb_m) == lf].sum()) + 10.0 * float(t.value[lf]))
        for t in est_m.trees_
        for lf in t.leaves
    )
    assert max_gm < 1e-9


def test_r1_l2_identical_trees_converge_symmetrically_without_drift():
    X, y, _ = _binary_data(seed=3, n=1000)
    est = L2FIGSClassifier(max_splits=1, lam=10.0).fit(X, y)
    t0 = est.trees_[0]
    t1 = SmallTree(
        list(t0.feature), list(t0.threshold), list(t0.left), list(t0.right), t0.value.copy()
    )
    # Deliberately break symmetry strongly before refitting
    t0.value = np.array([0.0, 5.0, -4.0])
    t1.value = np.array([0.0, -3.0, 6.0])
    est.trees_ = [t0, t1]
    refit_leaves_l2(
        est, X, y, lam=10.0, sweeps=250, refit_base=True, learning_rate=1.0, max_delta_step=None
    )
    np.testing.assert_allclose(
        est.trees_[0].value[est.trees_[0].leaves],
        est.trees_[1].value[est.trees_[1].leaves],
        atol=1e-9,
    )
    assert np.all(np.isfinite(est.trees_[0].value))


def test_r1_fused_growth_matches_edge_options():
    X, y, _ = _binary_data(seed=5, n=600)
    # Zero budget / zero max_trees edge cases
    e0 = L2FIGSClassifier(max_splits=0).fit(X, y)
    assert e0.trees_ == []
    et0 = L2FIGSClassifier(max_splits=4, max_trees=0).fit(X, y)
    assert et0.trees_ == []

    # backfit_sweeps=0, update_base=False, max_delta_step, and max_trees=1 paths
    for kw in (
        dict(max_splits=12, backfit_sweeps=0, update_base=False, max_delta_step=2.0),
        dict(max_splits=12, backfit_sweeps=2, update_base=True, max_trees=1, learning_rate=0.5),
        dict(max_splits=6, penalty="proximal", update_base=True, max_delta_step=3.0),
    ):
        est = L2FIGSClassifier(**kw).fit(X, y)
        p = est.predict_proba(X)
        assert p.shape == (len(X), 2)
        assert np.all(np.isfinite(p))
        assert est.n_splits_ <= kw["max_splits"]


# ------------------------------------------------------------ R2: LookaheadFIGSClassifier


@pytest.mark.parametrize("budget", [4, 16])
@pytest.mark.parametrize("weighted", [False, True])
def test_r2_disabled_lookahead_and_guards_matches_figs_bitwise(budget, weighted):
    X, y, soft = _binary_data(seed=4)
    X[::13, 0] = np.nan
    w = (1 + np.arange(len(y)) % 3).astype(float) if weighted else None
    params = dict(max_splits=budget, learning_rate=0.3 if budget > 8 else 1.0)
    ref = FIGSClassifier(**params).fit(X, y, sample_weight=w, y_soft=soft if weighted else None)
    cand = LookaheadFIGSClassifier(
        **params, lookahead=False, max_depth=None, min_leaf_coverage=0.0
    ).fit(X, y, sample_weight=w, y_soft=soft if weighted else None)
    _assert_same_classifier(ref, cand, X)


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 8])
def test_r2_respects_budget_depth_coverage_and_api(budget):
    rng = np.random.default_rng(7)
    n = 3000
    X = rng.normal(size=(n, 5))
    # Pure XOR: encourages depth-2 lookahead even on small budgets
    eta = 3.0 * np.sign(X[:, 0] * X[:, 1]) + 0.5 * X[:, 2]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    X[::17, 0] = np.nan
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(5)])

    est = LookaheadFIGSClassifier(
        max_splits=budget, lookahead=True, max_depth=2, min_leaf_coverage=0.05
    ).fit(df, y)
    assert est.n_splits_ <= budget

    def max_depth_of(tree, node=0):
        if tree.left[node] == -1:
            return 0
        return 1 + max(max_depth_of(tree, tree.left[node]), max_depth_of(tree, tree.right[node]))

    for t in est.trees_:
        assert max_depth_of(t) <= 2

    if budget >= 3:
        assert est.n_splits_ >= 2
        assert len(est.rules()) == sum(len(t.leaves) for t in est.trees_)
        assert "tree 1:" in est.explain()
        np.testing.assert_allclose(
            est.base_margin_ + est.predict_contributions(df).sum(axis=1),
            est.decision_function(df),
            atol=1e-12,
        )
        # SQL check
        sql = est.to_sql("rows", precision=17)
        with sqlite3.connect(":memory:") as con:
            con.create_function("EXP", 1, math.exp)
            df.iloc[:200].to_sql("rows", con, index=False)
            out = pd.read_sql(sql, con)
        np.testing.assert_allclose(out["score"], est.decision_function(df.iloc[:200]), atol=1e-10)
        # TreeEditMixin check
        cut_node = next(k for k, lc in enumerate(est.trees_[0].left) if lc != -1)
        est.prune(0, cut_node)
        est.refit_leaves(df, y)
        assert est.n_splits_ < budget


def test_r2_max_depth_one_forces_stumps_only():
    X, y, _ = _binary_data(seed=8, n=1000)
    est = LookaheadFIGSClassifier(max_splits=6, lookahead=True, max_depth=1).fit(X, y)
    assert est.n_splits_ <= 6
    assert all(t.n_splits == 1 for t in est.trees_)


# ------------------------------------------------------------ R4: CompactTreeBoosterV2


def test_r4_no_lightgbm_import_and_stratified_val_no_leakage(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    rng = np.random.default_rng(11)
    n, p = 1500, 45
    X = rng.normal(size=(n, p))
    eta = -3.8 + 2.5 * (X[:, 0] > 0.5) + 1.5 * X[:, 1] * X[:, 2]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)

    est = CompactTreeBoosterV2(
        max_splits=16,
        depth="auto",
        validation_fraction=0.15,
        feature_screen="fast_cached",
        screen_interval=10,
        stratified_val=True,
        refit_when_budget_bound="shrunk_l2",
    ).fit(X, y)
    assert est.n_splits_ <= 16

    val = est.val_mask_
    n_val = int(val.sum())
    assert abs(y[val].mean() - y.mean()) <= 1.0 / n_val + 1e-12

    # Bin edges learned strictly from training fold X[~val]
    ref_edges = fit_bin_edges(np.ascontiguousarray(X[~val], dtype=np.float32), est.max_bins)
    for e_est, e_ref in zip(est.bin_edges_, ref_edges):
        np.testing.assert_array_equal(e_est, e_ref)

    # InterpretableSumMixin + TreeEditMixin checks
    np.testing.assert_allclose(
        est.base_margin_ + est.predict_contributions(X).sum(axis=1),
        est.decision_function(X),
        atol=1e-12,
    )
    assert "tree 1:" in est.explain()
    assert len(est.rules()) == sum(len(t.leaves) for t in est.trees_)
    d = json.loads(json.dumps(est.to_dict()))
    assert d["link"] == "logit"

    sql = est.to_sql("rows", precision=17)
    df = pd.DataFrame(X[:150], columns=[f"x{j}" for j in range(p)])
    with sqlite3.connect(":memory:") as con:
        con.create_function("EXP", 1, math.exp)
        df.to_sql("rows", con, index=False)
        out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out["score"], est.decision_function(X[:150]), atol=1e-10)

    # TreeEditMixin check
    first_t, first_k = next(
        (i, k) for i, t in enumerate(est.trees_) for k, lc in enumerate(t.left) if lc != -1
    )
    est.prune(first_t, first_k)
    est.refit_leaves(X, y)
    assert est.n_splits_ < 16


def test_r4_coarse_screen_catches_xor_and_triggers_budget_refit(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    rng = np.random.default_rng(17)
    n, p = 3500, 50
    X = rng.normal(size=(n, p))
    # Pure XOR pair (X[:, 5], X[:, 6]) plus multi-feature additive signal to bind small budget
    eta = (
        sum(0.6 * np.tanh(X[:, j]) for j in range(5))
        + 1.6 * np.sign(X[:, 5] * X[:, 6])
        + 0.5 * X[:, 7]
        - 0.5 * X[:, 8]
    )
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    est = CompactTreeBoosterV2(
        max_splits=16,
        patience=50,
        new_cut_penalty=0.5,
        feature_screen="fast_cached",
        refit_when_budget_bound="shrunk_l2",
    ).fit(X, y)
    assert est.n_splits_ <= 16
    assert est.budget_refitted_ is True
    used_feats = {int(f) for t in est.trees_ for f, lf in zip(t.feature, t.left) if lf != -1}
    assert 5 in used_feats and 6 in used_feats


# ------------------------------------------------------------ R5: CategoricalFIGSClassifier


def test_r5_fisher_exact_matches_exhaustive_bipartitions():
    rng = np.random.default_rng(19)
    C = 8
    for _ in range(15):
        g = rng.normal(size=C)
        h = rng.uniform(0.5, 5.0, size=C)
        w = rng.uniform(1.0, 10.0, size=C)
        hist = np.column_stack([g, h, w])
        lam = float(rng.choice([0.0, 2.0, 10.0]))
        gain_fisher, mask = _best_cat_hist(hist, C, lam, 0.0, 0.0)
        assert 1 <= int(mask.sum()) < C

        Gt, Ht = g.sum(), h.sum()
        parent = Gt * Gt / (Ht + lam)
        best_brute = 0.0
        for bits in product((0, 1), repeat=C - 1):
            m = np.array([1, *bits], dtype=bool)
            if m.all():
                continue
            gl, hl = g[m].sum(), h[m].sum()
            gr, hr = Gt - gl, Ht - hl
            gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent
            best_brute = max(best_brute, float(gain))
        assert gain_fisher == pytest.approx(best_brute, abs=1e-12, rel=1e-12)


def test_r5_no_categorical_columns_matches_figs_bitwise():
    X, y, soft = _binary_data(seed=23, n=1000)
    X[::9, 0] = np.nan
    df = pd.DataFrame(X, columns=[f"c{j}" for j in range(X.shape[1])])
    ref = FIGSClassifier(max_splits=8, learning_rate=0.3).fit(df, y, y_soft=soft)
    cand = CategoricalFIGSClassifier(
        max_splits=8, learning_rate=0.3, categorical_features="auto"
    ).fit(df, y, y_soft=soft)
    _assert_same_classifier(ref, cand, df)
    cand_none = CategoricalFIGSClassifier(
        max_splits=8, learning_rate=0.3, categorical_features=None
    ).fit(X, y)
    ref_np = FIGSClassifier(max_splits=8, learning_rate=0.3).fit(X, y)
    _assert_same_classifier(ref_np, cand_none, X)


def test_r5_mixed_dataframe_apostrophes_unseen_nans_and_sqlite():
    rng = np.random.default_rng(29)
    n = 2500
    cats = np.array(["O'Brien", "D'Angelo", "Smith", "Müller", "Li", "García", "Chen", "Sato"])
    c_idx = rng.integers(0, len(cats), size=n)
    num = rng.normal(size=n)
    x_cat = pd.Series(cats[c_idx], dtype="object")
    x_cat.iloc[::19] = None
    num[::17] = np.nan
    # Positive categories: O'Brien, D'Angelo, García
    good = np.isin(cats[c_idx], ["O'Brien", "D'Angelo", "García"]).astype(float)
    eta = -0.8 + 2.2 * good + 0.9 * np.nan_to_num(num)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    df = pd.DataFrame({"city name": x_cat, "num val": num})

    est = CategoricalFIGSClassifier(max_splits=8, cat_smooth=0.0).fit(df, y)
    assert est.n_splits_ <= 8
    assert est.is_categorical_.tolist() == [True, False]

    # Test with unseen categories, apostrophes, and NULLs
    df_test = df.iloc[:250].copy()
    df_test.iloc[0, 0] = "Unseen'City"
    df_test.iloc[1, 0] = None
    df_test.iloc[2, 1] = np.nan

    np.testing.assert_allclose(
        est.base_margin_ + est.predict_contributions(df_test).sum(axis=1),
        est.decision_function(df_test),
        atol=1e-12,
    )
    assert len(est.rules()) == sum(len(t.leaves) for t in est.trees_)
    assert "O'Brien" in est.export_text() or "tree 1" in est.explain()
    json.loads(json.dumps(est.to_dict()))

    sql = est.to_sql("rows", precision=17)
    with sqlite3.connect(":memory:") as con:
        con.create_function("EXP", 1, math.exp)
        df_test.to_sql("rows", con, index=False)
        out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out["score"], est.decision_function(df_test), atol=1e-10)
    np.testing.assert_allclose(out["p"], est.predict_proba(df_test)[:, 1], atol=1e-10)


def test_r5_unseen_separated_from_informative_nan_and_sqlite():
    rng = np.random.default_rng(77)
    n = 2000
    # 90% known low-risk cities, 10% missing (None) with extreme positive risk
    cats = np.array(["A", "B", "C", "D"])
    col = pd.Series(cats[rng.integers(0, 4, size=n)], dtype="object")
    is_miss = rng.random(n) < 0.10
    col[is_miss] = None
    eta = np.where(is_miss, 3.0, -2.0)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    df = pd.DataFrame({"city": col, "z": rng.normal(size=n)})

    est = CategoricalFIGSClassifier(max_splits=4).fit(df, y)
    probe = pd.DataFrame({"city": [None, "A", "BrandNewCity"], "z": [0.0, 0.0, 0.0]})
    probs = est.predict_proba(probe)[:, 1]
    # Missing has high probability; BrandNewCity must route to majority (low risk), NOT NaN!
    assert probs[0] > 0.70
    assert probs[2] < 0.30
    assert abs(probs[2] - probs[1]) < 0.10

    sql = est.to_sql("rows", precision=17)
    with sqlite3.connect(":memory:") as con:
        con.create_function("EXP", 1, math.exp)
        probe.to_sql("rows", con, index=False)
        out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out["score"], est.decision_function(probe), atol=1e-10)


def test_r5_timestamp_guardrail_and_max_subset_size():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(101)
    n = 1600
    # 1. Near-unique ISO-8601 timestamps (1600 unique strings > 253) with strong chronological signal
    days = rng.integers(0, 336, size=n)
    secs = rng.integers(0, 86400, size=n)
    ts = [
        f"2024-{1 + int(d // 28):02d}-{1 + int(d % 28):02d}T{int(s):05d}Z"
        for d, s in zip(days, secs)
    ]
    cities = np.array([f"c{k}" for k in range(10)])
    c_idx = rng.integers(0, 10, size=n)
    eta = 3.0 * (days > 168) + 1.5 * (c_idx < 3) - 2.0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    df = pd.DataFrame({
        "ts": pd.Series(ts, dtype="object"),
        "city": pd.Series(cities[c_idx], dtype="object"),
    })

    est = CategoricalFIGSClassifier(max_splits=6, max_subset_size=2).fit(df[:1200], y[:1200])
    # ts must be downgraded via guardrail (not collapsed to 'other'), while city stays categorical
    assert est.is_categorical_.tolist() == [False, True]
    assert 0 in est._ord_vocabs
    auc_guarded = roc_auc_score(y[1200:], est.predict_proba(df[1200:])[:, 1])
    est_unguarded = CategoricalFIGSClassifier(max_splits=6, min_cat_coverage=0.0).fit(
        df[:1200], y[:1200]
    )
    auc_unguarded = roc_auc_score(y[1200:], est_unguarded.predict_proba(df[1200:])[:, 1])
    assert auc_guarded > 0.80
    assert auc_guarded > auc_unguarded + 0.10
    # Every categorical split must respect max_subset_size=2
    for mask in est.cat_splits_:
        s_left = int(mask[2:].sum())
        s_right = len(mask) - 2 - s_left
        assert min(s_left, s_right) <= 2

    # SQLite parity on unseen timestamps + categorical column
    sql = est.to_sql("rows", precision=17)
    with sqlite3.connect(":memory:") as con:
        con.create_function("EXP", 1, math.exp)
        df[1200:1300].to_sql("rows", con, index=False)
        out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out["score"], est.decision_function(df[1200:1300]), atol=1e-10)


def test_r5_h2_rare_category_pooling_and_noise_hijack_defense():
    # 1. Rare-in-leaf categories (< min_cat_weight) are pooled into one group and share mask side.
    hist = np.array([
        [-10.0, 20.0, 40.0],
        [10.0, 20.0, 40.0],
        [3.0, 1.0, 2.0],
        [-3.0, 1.0, 2.0],
        [2.5, 1.0, 2.0],
    ], dtype=np.float64)
    gain_h, mask_h = _best_cat_hist(hist, 5, 2.0, 10.0, 1.0, 10.0, 2.0 * np.log(2.0))
    assert gain_h > 0.0
    # Rare categories 2, 3, 4 must move together in mask_h
    assert mask_h[2] == mask_h[3] == mask_h[4]

    # 2. Pure-noise high-cardinality ID columns cannot hijack splits under hardened defaults.
    rng = np.random.default_rng(77)
    n = 6000
    X_num = rng.normal(size=(n, 3))
    eta = 1.4 * (X_num[:, 0] > 0.0) - 1.1 * (X_num[:, 1] > 0.2) + 0.7 * X_num[:, 2]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    df = pd.DataFrame({
        "x0": X_num[:, 0],
        "x1": X_num[:, 1],
        "x2": X_num[:, 2],
        "noise_c100": pd.Series([f"id_{k}" for k in rng.integers(0, 100, size=n)], dtype="object"),
        "noise_c200": pd.Series([f"id_{k}" for k in rng.integers(0, 200, size=n)], dtype="object"),
    })
    df_tr, y_tr, df_te, y_te = df.iloc[:4000], y[:4000], df.iloc[4000:], y[4000:]

    unhardened = CategoricalFIGSClassifier(
        max_splits=8, min_cat_weight=0.0, cat_dof_penalty=0.0
    ).fit(df_tr, y_tr)
    hardened = CategoricalFIGSClassifier(max_splits=8).fit(df_tr, y_tr)

    def noise_frac(est):
        feats = [int(f) for t in est.trees_ for f, lf in zip(t.feature, t.left) if lf != -1]
        return sum(1 for f in feats if f >= 3) / max(len(feats), 1)

    assert noise_frac(unhardened) >= 0.50
    assert noise_frac(hardened) <= 0.05
    ll_un = log_loss(y_te, np.clip(unhardened.predict_proba(df_te)[:, 1], 1e-15, 1 - 1e-15))
    ll_ha = log_loss(y_te, np.clip(hardened.predict_proba(df_te)[:, 1], 1e-15, 1 - 1e-15))
    assert ll_ha < ll_un - 0.03

    # 3. Backward-compatible defaults when cat_smooth == 0.0 & validation checks.
    compat = CategoricalFIGSClassifier(max_splits=4, cat_smooth=0.0).fit(df_tr, y_tr)
    assert compat.cat_dof_penalty_ == 0.0
    assert compat.min_cat_weight_ == 0.0
    with pytest.raises(ValueError):
        CategoricalFIGSClassifier(min_cat_weight=-1.0).fit(df_tr, y_tr)
    with pytest.raises(ValueError):
        CategoricalFIGSClassifier(cat_dof_penalty="bad").fit(df_tr, y_tr)


# ------------------------------------------------------------ R6: Regression & Poisson


@parametrize_with_checks([
    FIGSRegressor(max_splits=6, min_weight=1.0),
    SumOfOptimalTreesRegressor(min_weight=1.0),
    BudgetRegressor(max_splits=6),
])
def test_r6_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.mark.parametrize("loss", ["squared_error", "poisson"])
@pytest.mark.parametrize("cls_fn", [
    lambda loss: FIGSRegressor(max_splits=8, loss=loss),
    lambda loss: SumOfOptimalTreesRegressor(n_trees=2, depth=2, extra_stumps=1, loss=loss),
    lambda loss: BudgetRegressor(max_splits=8, loss=loss),
])
def test_r6_contributions_and_sql_match_sqlite_with_nans(loss, cls_fn):
    rng = np.random.default_rng(37)
    n = 1500
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b c", "d", "e"])
    df.loc[rng.random(n) < 0.08, "a"] = np.nan
    eta = 0.4 + 0.7 * np.nan_to_num(df["a"]) - 0.5 * (df["b c"] > 0) + 0.4 * df["d"] * df["e"]
    if loss == "squared_error":
        y = 3.0 + eta + 0.3 * rng.normal(size=n)
    else:
        y = rng.poisson(np.exp(np.clip(eta, -3, 3))).astype(float)

    est = cls_fn(loss).fit(df, y)
    preds = est.predict(df)
    lin = est.base_margin_ + est.predict_contributions(df).sum(axis=1)
    if loss == "squared_error":
        np.testing.assert_allclose(lin, preds, atol=1e-12)
    else:
        np.testing.assert_allclose(np.exp(lin), preds, atol=1e-12)

    assert len(est.rules()) == sum(len(t.leaves) for t in est.trees_)
    assert "tree 1" in est.explain()
    assert "tree 1" in est.export_text()
    json.loads(json.dumps(est.to_dict()))

    for keep in (True, False):
        sql = est.to_sql("rows", precision=17, keep_columns=keep)
        with sqlite3.connect(":memory:") as con:
            con.create_function("EXP", 1, math.exp)
            df.iloc[:200].to_sql("rows", con, index=False)
            out = pd.read_sql(sql, con)
        np.testing.assert_allclose(out["prediction"], preds[:200], atol=1e-10)
        np.testing.assert_allclose(out["score"], lin[:200], atol=1e-10)


def test_r6_poisson_rejects_negative_y_and_optimal_beats_greedy_on_xor():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(2500, 5))
    y_xor = 2.0 * np.sign(X[:, 0] * X[:, 1]) + 0.2 * rng.normal(size=len(X))
    with pytest.raises(ValueError):
        FIGSRegressor(loss="poisson").fit(X, y_xor)

    tr, te = slice(0, 2000), slice(2000, None)
    opt = SumOfOptimalTreesRegressor(n_trees=1, depth=2, search="optimal").fit(X[tr], y_xor[tr])
    gre = SumOfOptimalTreesRegressor(n_trees=1, depth=2, search="greedy").fit(X[tr], y_xor[tr])
    mse_opt = float(np.mean((y_xor[te] - opt.predict(X[te])) ** 2))
    mse_gre = float(np.mean((y_xor[te] - gre.predict(X[te])) ** 2))
    assert mse_opt < mse_gre - 1.0

    # Poisson auto lam scales proportionally with mean Hessian (mean count >= 1)
    yp_lo = rng.poisson(0.5, size=200).astype(float)
    yp_hi = rng.poisson(20.0, size=200).astype(float)
    r_lo = FIGSRegressor(max_splits=8, lam="auto", loss="poisson").fit(X[:200], yp_lo)
    r_hi = FIGSRegressor(max_splits=8, lam="auto", loss="poisson").fit(X[:200], yp_hi)
    assert r_lo.lam_ == pytest.approx(16.0)
    assert r_hi.lam_ == pytest.approx(16.0 * float(np.mean(yp_hi)))
