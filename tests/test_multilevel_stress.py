"""Reproducibility and fixed-gate checks for the synthetic stress suite."""

import numpy as np

from benchmark_multilevel_stress import (FAMILIES, generate_family,
                                         primary_summary)


def test_all_generators_are_reproducible_and_probabilistic():
    for family in FAMILIES:
        X, y, p = generate_family(family, 2000, 7)
        X2, y2, p2 = generate_family(family, 2000, 7)
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
        np.testing.assert_array_equal(p, p2)
        assert X.shape == (2000, 6)
        assert y.shape == (2000,)
        assert (p > 0).all() and (p < 1).all()
        assert not np.array_equal(X, generate_family(family, 2000, 200007)[0])


def test_null_has_no_structural_signal_and_rare_is_imbalanced():
    _, _, p_null = generate_family("null", 1000, 0)
    _, _, p_rare = generate_family("rare_recursive", 1000, 0)
    np.testing.assert_array_equal(p_null, .5)
    assert p_rare.max() <= .26 + 1e-12
    assert p_rare.min() >= .02 - 1e-12


def test_xor_has_no_one_feature_signal_but_a_depth_two_interaction():
    X, _, p = generate_family("xor", 20000, 3)
    assert abs(p[X[:, 0] > 0].mean() - p[X[:, 0] <= 0].mean()) < .02
    assert np.isclose(p[(X[:, 0] > 0) ^ (X[:, 1] > 0)].mean(), .85)
    assert np.isclose(p[~((X[:, 0] > 0) ^ (X[:, 1] > 0))].mean(), .15)


def test_primary_gate_requires_all_predeclared_seeds_and_cost():
    rows = []
    for seed in range(8):
        for engine, loss, accuracy, fit_seconds in (
                ("greedy_exact", .55, .72, .1),
                ("block_hist_exact", .53 + seed * .0001, .72, 1.5)):
            rows.append(dict(family="recursive_four", n_train=10000,
                             seed=seed, depth=4, engine=engine,
                             fit_seconds_median=fit_seconds,
                             metrics=dict(log_loss=loss, accuracy=accuracy)))
    outcome = primary_summary(rows)
    assert outcome["status"] == "complete"
    assert outcome["quality_gate_pass"]
    assert outcome["cost_gate_pass"]
    assert outcome["mechanism_gate_pass"]
    assert primary_summary(rows[:-1])["status"] == "partial"
    slower = [dict(item) for item in rows]
    for item in slower:
        if item["engine"] == "block_hist_exact":
            item["fit_seconds_median"] = 3.0
    assert not primary_summary(slower)["cost_gate_pass"]
