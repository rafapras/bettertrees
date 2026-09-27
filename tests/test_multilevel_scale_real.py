"""Smoke and split contracts for the public large-data benchmark."""

import numpy as np
from zipfile import ZipFile

from benchmark_multilevel_scale_real import (choose_train, fit_real_engine,
                                             load_public_data,
                                             locked_partition)
from benchmark_multilevel_stress import generate_family


def test_locked_test_does_not_change_with_training_size_or_seed():
    X = np.arange(4000, dtype=np.float32).reshape(1000, 4)
    y = np.tile(np.array([0, 1], dtype=np.int32), 500)
    X_pool, X_test, y_pool, y_test = locked_partition(X, y)
    assert len(X_test) == 200
    assert len(np.unique(X_test[:, 0])) == 200
    assert not np.isin(X_pool[:, 0], X_test[:, 0]).any()
    for seed in (0, 1):
        X_small, y_small = choose_train(X_pool, y_pool, 100, seed)
        assert len(X_small) == 100
        assert y_small.sum() == 50
    np.testing.assert_array_equal(choose_train(X_pool, y_pool, 800, 0)[0],
                                  X_pool)


def test_poker_loader_uses_only_features_and_declares_binary_target(tmp_path):
    archive_path = tmp_path / "poker.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("poker-hand-testing.data",
                         "1,2,1,3,1,4,1,5,1,6,0\n"
                         "2,2,2,3,2,4,2,5,2,6,1\n"
                         "3,2,3,3,3,4,3,5,3,6,9\n")
    X, y, recipe = load_public_data(
        "poker", data_home=tmp_path, poker_zip=archive_path)
    assert X.shape == (3, 10)
    np.testing.assert_array_equal(y, [0, 1, 1])
    assert recipe["target_transform"] == "pair_or_better_vs_nothing"


def test_posthoc_leaf_control_never_exceeds_matched_budget():
    X, y, _ = generate_family("recursive_four", 1000, 7)
    model = fit_real_engine("greedy_exact_leaf11", X, y, depth=4,
                            max_bins=8, min_leaf=5, smoothing=2,
                            candidate_limit=128, seed=0)
    assert model.get_n_leaves() <= 11
    assert model.predict_proba(X[:10]).shape == (10, 2)
