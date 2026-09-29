"""Simulated credit-default data with known effects, used by the example notebooks.

The true model is a logit sum of one-feature shapes plus one interaction, so the
notebooks can check what a fitted model recovers. No download needed.
"""

import numpy as np
import pandas as pd

FEATURES = ["age", "income", "debt_ratio", "late_payments", "credit_lines",
            "months_employed", "has_mortgage"]


def true_effects():
    """The one-feature pieces of the true logit, as functions of the raw value."""
    return {
        "age": lambda a: 0.0006 * (a - 45) ** 2 - 0.2,
        "income": lambda v: np.where(np.isnan(v), 0.4, -0.8 * (np.log(np.maximum(v, 1)) - 10.5)),
        "debt_ratio": lambda d: 3.0 * np.maximum(d - 0.4, 0),
        "late_payments": lambda k: 0.7 * np.minimum(k, 4),
        "credit_lines": lambda c: -0.1 * np.minimum(c, 8),
        "months_employed": lambda m: np.where(m < 12, 0.5, 0.0),
        "has_mortgage": lambda h: -0.3 * h,
    }


def make_credit_data(n=20_000, seed=0):
    """(X DataFrame, y Series): about 10% defaults; 5% of incomes are missing."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(18, 81, n).astype(float),
        "income": np.round(rng.lognormal(10.5, 0.5, n), -2),
        "debt_ratio": np.round(rng.beta(2, 5, n), 3),
        "late_payments": np.minimum(rng.poisson(0.4, n), 6).astype(float),
        "credit_lines": (rng.poisson(4, n) + 1).astype(float),
        "months_employed": np.round(rng.exponential(48, n)),
        "has_mortgage": rng.binomial(1, 0.35, n).astype(float),
    })
    X.loc[rng.random(n) < 0.05, "income"] = np.nan
    logit = -2.3 + sum(f(X[c].to_numpy()) for c, f in true_effects().items())
    # one interaction: high debt matters more for people who already paid late
    logit = logit + 0.6 * ((X["debt_ratio"] > 0.5) & (X["late_payments"] > 0)).to_numpy()
    y = pd.Series(rng.random(n) < 1 / (1 + np.exp(-logit)), name="default").astype(int)
    return X, y
