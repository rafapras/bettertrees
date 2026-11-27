"""Unit and G0 gate tests for R7 (CausalFIGS), R8 (RashomonMCR), R9 (HingeFIGS) and R10."""

import math
import sqlite3

import numpy as np
import pandas as pd

from bettertrees import FIGSClassifier
from bettertrees.experimental.causal import CausalFIGS
from bettertrees.experimental.hinge import HingeFIGSClassifier
from bettertrees.experimental.mcr import feature_importance_shares, rashomon_mcr
from bettertrees.lab.sparse_sum import (
    ResidualRatioFIGSClassifier,
    SparseTreeSumClassifier,
)


def test_r7_causal_figs_cate_recovery_and_sql():
    rng = np.random.default_rng(0)
    n, p = 1200, 5
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=["age", "income", "x2", "x3", "x4"])
    X.iloc[::20, 0] = np.nan

    x0_clean = np.nan_to_num(X["age"].to_numpy(), nan=-1.0)
    x1 = X["income"].to_numpy()
    tau_true = 1.2 * (x0_clean > 0) - 1.0 * (x1 > 0)
    propensity = 1.0 / (1.0 + np.exp(-(0.8 * x0_clean + 0.6 * x1)))
    treatment = rng.binomial(1, propensity).astype(float)
    y = 2.0 * x0_clean + X["x2"].to_numpy() + treatment * tau_true + 0.2 * rng.normal(size=n)

    model = CausalFIGS(max_splits=4, max_depth=2, random_state=0).fit(X, y, treatment)
    pred = model.predict(X)
    contribs = model.predict_contributions(X)
    assert pred.shape == (n,)
    assert model.n_splits_ <= 4
    np.testing.assert_allclose(model.base_effect_ + contribs.sum(axis=1), pred, atol=1e-12)
    assert np.corrcoef(pred, tau_true)[0, 1] > 0.85

    zero = CausalFIGS(max_splits=0, random_state=0).fit(X, y, treatment)
    assert zero.n_splits_ == 0
    np.testing.assert_allclose(zero.predict(X), zero.base_effect_, atol=1e-12)

    sql = model.to_sql("rows", precision=17)
    with sqlite3.connect(":memory:") as con:
        X.iloc[:100].to_sql("rows", con, index=False)
        out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out["cate"].to_numpy(), model.predict(X.iloc[:100]), atol=1e-10)
    assert "tau(x)" in model.explain()


def test_r7_causal_figs_precomputed_nuisances_hook():
    rng = np.random.default_rng(123)
    n, p = 1000, 5
    X = rng.normal(size=(n, p))
    tau = 1.3 * (X[:, 0] > 0) - 0.9 * (X[:, 1] > 0)
    e_true = 1.0 / (1.0 + np.exp(-(0.7 * X[:, 0] - 0.5 * X[:, 1])))
    d = rng.binomial(1, e_true).astype(float)
    y = 1.5 * X[:, 0] + d * tau + 0.3 * rng.normal(size=n)

    m_hat = 1.5 * X[:, 0] + e_true * tau
    e_hat = e_true
    m1 = CausalFIGS(max_splits=4, random_state=0).fit(X, y, d, m_hat=m_hat, e_hat=e_hat)
    m2 = CausalFIGS(max_splits=4, random_state=0).fit(X, y, d, y_hat=m_hat, e_hat=e_hat)
    np.testing.assert_array_equal(m1.m_hat_, m_hat)
    np.testing.assert_array_equal(m1.e_hat_, e_hat)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
    assert np.corrcoef(m1.predict(X), tau)[0, 1] > 0.85

    import pytest

    with pytest.raises(ValueError):
        CausalFIGS(max_splits=4).fit(X, y, d, y_hat=m_hat, m_hat=m_hat, e_hat=e_hat)
    with pytest.raises(ValueError):
        CausalFIGS(max_splits=4).fit(X, y, d, m_hat=m_hat[:-1], e_hat=e_hat)


def test_r8_rashomon_mcr_distinguishes_collinear_from_indispensable():
    rng = np.random.default_rng(42)
    n = 1500
    x0 = rng.normal(size=n)
    x1 = x0 + 0.01 * rng.normal(size=n)  # near-exact substitute for x0
    x2 = rng.normal(size=n)              # unique orthogonal signal
    x3 = rng.normal(size=n)              # pure noise
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3})
    logit = 2.0 * (x0 > 0) - 2.0 * (x2 > 0)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    model = FIGSClassifier(max_splits=2, learning_rate=1.0).fit(X.iloc[:1000], y[:1000])
    shares = feature_importance_shares(model, X.iloc[1000:])
    np.testing.assert_allclose(shares.sum(), 1.0, atol=1e-12)

    res = rashomon_mcr(model, X.iloc[:1000], y[:1000], X.iloc[1000:], y[1000:], epsilon=0.03)
    feats = res["features"]
    for info in feats.values():
        assert -1e-12 <= info["mcr_min"] <= info["base_importance"] + 1e-12
        assert info["base_importance"] - 1e-12 <= info["mcr_max"] <= 1.0 + 1e-12

    # x2 has no substitute -> indispensable; whichever of (x0, x1) was picked has x1/x0 -> dispensable
    assert feats["x2"]["indispensable"] is True
    picked_dup = "x0" if feats["x0"]["active"] else "x1"
    assert feats[picked_dup]["indispensable"] is False
    assert feats[picked_dup]["mcr_min"] == 0.0
    alt = res["alternatives"][picked_dup]
    dup_idx = 0 if picked_dup == "x0" else 1
    assert all(
        int(f) != dup_idx for tr in alt.trees_ for f, lft in zip(tr.feature, tr.left) if lft != -1
    )


def test_r9_hinge_figs_knot_continuity_and_sql():
    rng = np.random.default_rng(3)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f0", "f1", "f2", "f3"])
    X.iloc[::15, 1] = np.nan
    x1 = np.nan_to_num(X["f1"].to_numpy(), nan=-0.5)
    logit = 1.2 * X["f0"].to_numpy() - 1.8 * np.maximum(x1 - 0.3, 0.0)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    model = HingeFIGSClassifier(max_splits=4, allow_tree_splits=True).fit(X, y)
    assert len(model.hinges_) >= 1
    assert model.n_splits_ <= 4
    dec = model.decision_function(X)
    contribs = model.predict_contributions(X)
    np.testing.assert_allclose(model.base_margin_ + contribs.sum(axis=1), dec, atol=1e-12)

    for hg in model.hinges_:
        _, a_l, s_l, a_r, s_r = hg.raw_pieces()
        np.testing.assert_allclose(a_l + s_l * hg.threshold_val, a_r + s_r * hg.threshold_val, atol=1e-12)

    sql = model.to_sql("rows", precision=17)
    with sqlite3.connect(":memory:") as con:
        con.create_function("EXP", 1, math.exp)
        X.iloc[:120].to_sql("rows", con, index=False)
        out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out["score"].to_numpy(), model.decision_function(X.iloc[:120]), atol=1e-10)
    np.testing.assert_allclose(out["p"].to_numpy(), model.predict_proba(X.iloc[:120])[:, 1], atol=1e-10)


def test_r9_h1_clamped_support_outlier_invariance_and_sql():
    rng = np.random.default_rng(42)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f0", "f1", "f2", "f3"])
    X.iloc[::20, 0] = np.nan
    x0 = np.nan_to_num(X["f0"].to_numpy(), nan=0.0)
    logit = 1.6 * x0 - 1.4 * np.maximum(X["f1"].to_numpy() - 0.2, 0.0)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    clamped = HingeFIGSClassifier(max_splits=4, allow_tree_splits=False, clamp_range=True).fit(X, y)
    unclamped = HingeFIGSClassifier(
        max_splits=4, allow_tree_splits=False, clamp_range=False
    ).fit(X, y)
    assert len(clamped.hinges_) >= 1

    # Verify flat tails outside [x_min, x_max] and extreme SQLite exactness under 100-sigma shocks
    X_extreme = X.iloc[:120].copy()
    X_extreme.iloc[:30, 0] = -1e6
    X_extreme.iloc[30:60, 0] = 1e6
    X_extreme.iloc[60:80, 1] = 1e6
    X_extreme.iloc[80:95, 1] = -1e6
    X_extreme.iloc[95:105, 0] = np.nan

    dec_c = clamped.decision_function(X_extreme)
    dec_u = unclamped.decision_function(X_extreme)
    assert np.all(np.isfinite(dec_c))
    assert np.max(np.abs(dec_c)) < 50.0
    assert np.max(np.abs(dec_u)) > 1e4

    # Exact flat boundary equality: X_extreme clipped to [x_min_, x_max_] gives identical scores
    X_at_bounds = X_extreme.copy()
    for j, col in enumerate(X_at_bounds.columns):
        X_at_bounds[col] = np.clip(X_at_bounds[col].to_numpy(), clamped.x_min_[j], clamped.x_max_[j])
    np.testing.assert_allclose(dec_c, clamped.decision_function(X_at_bounds), atol=1e-12)

    d_json = clamped.to_dict()
    assert all("x_min" in h and "x_max" in h and h["clamp"] is True for h in d_json["hinges"])
    assert "clamped [" in clamped.explain()

    sql = clamped.to_sql("rows", precision=17)
    with sqlite3.connect(":memory:") as con:
        con.create_function("EXP", 1, math.exp)
        X_extreme.to_sql("rows", con, index=False)
        out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out["score"].to_numpy(), dec_c, atol=1e-10)
    np.testing.assert_allclose(
        out["p"].to_numpy(), clamped.predict_proba(X_extreme)[:, 1], atol=1e-10
    )


def test_r10_sparse_sum_frontier_and_bic_ratio_gate():
    rng = np.random.default_rng(11)
    n = 1200
    X_add = rng.normal(size=(n, 5))
    y_add = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X_add[:, 0] - X_add[:, 1])))).astype(int)

    sparse = SparseTreeSumClassifier(budgets=(4, 8, 16), n_candidates=25, random_state=0).fit(X_add, y_add)
    for b in (4, 8, 16):
        mb = sparse.model_for_budget(b)
        assert mb.n_splits_ <= b
        np.testing.assert_allclose(
            mb.base_margin_ + mb.predict_contributions(X_add[:50]).sum(axis=1),
            mb.decision_function(X_add[:50]),
            atol=1e-12,
        )

    # Purely additive data -> BIC gate rejects spurious ratio cuts
    null_ratios = 0
    for seed in range(5):
        r_rng = np.random.default_rng(100 + seed)
        Xa = np.exp(0.5 * r_rng.normal(size=(800, 5)))
        ya = (r_rng.random(800) < 1.0 / (1.0 + np.exp(-(np.log(Xa[:, 0]) - 0.5)))).astype(int)
        rm = ResidualRatioFIGSClassifier(max_splits=4, min_main_splits=2, max_ratio_splits=2).fit(Xa, ya)
        if rm.n_ratio_splits_ == 0:
            null_ratios += 1
    assert null_ratios >= 4

    # Genuine accounting ratio signal x0 / x1 > 1.5 -> BIC gate admits the ratio cut
    Xr = np.exp(0.7 * rng.normal(size=(2000, 5)))
    logit_r = 3.0 * ((Xr[:, 0] / Xr[:, 1]) > 1.5) - 1.5 + 0.8 * (Xr[:, 2] > 1.0)
    yr = (rng.random(2000) < 1.0 / (1.0 + np.exp(-logit_r))).astype(int)
    rm_pos = ResidualRatioFIGSClassifier(max_splits=4, min_main_splits=1, max_ratio_splits=2).fit(Xr, yr)
    assert rm_pos.n_ratio_splits_ >= 1
    np.testing.assert_allclose(
        rm_pos.base_margin_ + rm_pos.predict_contributions(Xr[:50]).sum(axis=1),
        rm_pos.decision_function(Xr[:50]),
        atol=1e-12,
    )
