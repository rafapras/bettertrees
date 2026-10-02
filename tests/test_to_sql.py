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
