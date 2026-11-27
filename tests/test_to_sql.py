"""to_sql(): the query reproduces decision_function / predict_proba, NaN included."""

import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from bettertrees import FIGSClassifier


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=["age", "income", "debt ratio", "x3", "x4"])
    z = X.age + np.tanh(X.income) - 0.8 * (X["debt ratio"] > 0.5) + 0.5 * X.x3 * (X.x4 > 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    X.loc[rng.random(n) < 0.05, "age"] = np.nan
    X.loc[rng.random(n) < 0.05, "debt ratio"] = np.nan
    return X, y


@pytest.mark.parametrize("budget", [4, 16])
def test_sql_matches_model(budget):
    X, y = _data()
    kw = dict(max_delta_step=4.0) if budget <= 8 else dict(learning_rate=0.3)
    m = FIGSClassifier(max_splits=budget, **kw).fit(X, y)
    sql = m.to_sql("rows", precision=12)
    con = sqlite3.connect(":memory:")
    con.create_function("EXP", 1, math.exp)
    X.to_sql("rows", con, index=False)
    out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out.score, m.decision_function(X), atol=1e-8)
    np.testing.assert_allclose(out.p, m.predict_proba(X)[:, 1], atol=1e-8)
    assert '"debt ratio"' in sql  # names that are not plain words get quoted


def test_sql_max_splits_zero():
    X, y = _data(n=200)
    m = FIGSClassifier(max_splits=0).fit(X, y)
    con = sqlite3.connect(":memory:")
    con.create_function("EXP", 1, math.exp)
    X.to_sql("rows", con, index=False)
    for keep in [True, False]:
        sql = m.to_sql("rows", keep_columns=keep, precision=17)
        out = pd.read_sql(sql, con)
        if not keep:
            assert list(out.columns) == ["score", "p"]
        np.testing.assert_allclose(out.score, m.decision_function(X), atol=1e-12)
        np.testing.assert_allclose(out.p, m.predict_proba(X)[:, 1], atol=1e-12)


def test_sql_adjacent_float32():
    # u1 has odd float32 mantissa, u2 even: float32((u1+u2)/2) would round up to u2
    u1 = np.float32(1.0) + np.float32(np.finfo(np.float32).eps)
    u2 = np.nextafter(u1, np.float32(2.0))
    X = pd.DataFrame({"x0": np.repeat([float(u1), float(u2)], 30)})
    y = np.repeat([0, 1], 30)
    m = FIGSClassifier(max_splits=1, min_weight=1.0).fit(X, y)
    assert m.n_splits_ == 1
    sql = m.to_sql("rows", precision=17)
    con = sqlite3.connect(":memory:")
    con.create_function("EXP", 1, math.exp)
    X.to_sql("rows", con, index=False)
    out = pd.read_sql(sql, con)
    np.testing.assert_allclose(out.score, m.decision_function(X), atol=1e-12)


def test_sql_unseen_test_nulls_and_custom_columns():
    rng = np.random.default_rng(7)
    X_tr = pd.DataFrame({
        'a"b': rng.normal(size=400),
        "score": rng.normal(size=400),
        "p": rng.normal(size=400),
    })
    y_tr = (X_tr['a"b'] + 0.5 * X_tr["score"] > 0).astype(int).to_numpy()
    m = FIGSClassifier(max_splits=4, min_weight=5.0).fit(X_tr, y_tr)
    assert not m.nan_features_.any()  # zero NaNs seen at fit time
    X_te = X_tr.iloc[:60].copy()
    X_te.iloc[::3, 0] = np.nan
    X_te.iloc[::4, 1] = np.nan
    X_te.iloc[::5, 2] = np.nan
    sql = m.to_sql("rows", score_col="logit_score", proba_col="pred_p", precision=17)
    con = sqlite3.connect(":memory:")
    con.create_function("EXP", 1, math.exp)
    X_te.to_sql("rows", con, index=False)
    out = pd.read_sql(sql, con)
    assert '"a""b"' in sql
    np.testing.assert_allclose(out.logit_score, m.decision_function(X_te), atol=1e-12)
    np.testing.assert_allclose(out.pred_p, m.predict_proba(X_te)[:, 1], atol=1e-12)
