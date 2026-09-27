"""Benchmark adapter gates; PyConTree is optional for the package itself."""

import numpy as np
import pytest

from arvore_rapida import FastDecisionTreeClassifier
from benchmark_multilevel_suite import (CalibratedConTree, balanced_indices,
                                        contree_structure, evaluate, route_paths,
                                        weighted_gini_from_ids)


class FakeTree:
    def __init__(self, feature=None, threshold=None, left=None, right=None):
        self.feature, self.threshold = feature, threshold
        self.left, self.right = left, right

    def is_leaf_node(self):
        return self.feature is None

    def get_split_feature(self):
        return self.feature

    def get_split_threshold(self):
        return self.threshold

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right


class FakeTopology:
    def __init__(self, tree):
        self.tree = tree

    def get_tree(self):
        return self.tree


def test_balance_only_topology_and_recalibrate_on_original_prevalence():
    y = np.array([0] * 9 + [1] * 3)
    selected = balanced_indices(y, 17)
    np.testing.assert_array_equal(np.bincount(y[selected]), [3, 3])
    np.testing.assert_array_equal(selected, balanced_indices(y, 17))
    X = np.arange(len(y), dtype=float).reshape(-1, 1)
    root = FakeTree(0, 8.5, FakeTree(), FakeTree())
    model = CalibratedConTree(FakeTopology(root), X, y, smoothing=2,
                              timed_out=False, topology_n=len(selected),
                              topology_positive_rate=.5)
    p = model.predict_proba(X)
    expected_positive = (3 + 2 * .25) / (3 + 2)
    assert p[-1, 1] == pytest.approx(expected_positive)
    assert model.prior_[1] == pytest.approx(.25)
    assert model.mass_.sum() == len(y)


def test_contree_routes_threshold_equality_once():
    tree = FakeTree(0, .5, FakeTree(), FakeTree())
    paths = contree_structure(tree)
    np.testing.assert_array_equal(
        route_paths(paths, np.array([[0], [.5], [1]], dtype=float)),
        [0, 0, 1])


def test_evaluate_reports_train_and_test_loss_and_leaf_gini():
    X_train = np.arange(16, dtype=np.float32).reshape(8, 2)
    y_train = np.array([0, 0, 1, 1, 0, 1, 1, 1], dtype=np.int32)
    X_test = np.array([[1, 2], [5, 6], [9, 10], [13, 14]], dtype=np.float32)
    y_test = np.array([0, 1, 0, 1], dtype=np.int32)
    model = FastDecisionTreeClassifier(
        max_depth=2, max_bins=2, min_samples_leaf=1,
        random_state=0, leaf_smoothing=1.0).fit(X_train, y_train)

    metrics, curve = evaluate(model, X_train, y_train, X_test, y_test)

    assert metrics["train_log_loss"] >= 0
    assert metrics["log_loss"] >= 0
    assert metrics["train_gini"] == pytest.approx(
        weighted_gini_from_ids(model.apply(X_train), y_train))
    assert metrics["test_gini"] == pytest.approx(
        weighted_gini_from_ids(model.apply(X_test), y_test))
    assert curve


def test_actual_pycontree_adapter_if_installed():
    pytest.importorskip("pycontree")
    from pycontree import ConTree

    X = np.array([[0], [0], [1], [1], [2], [2], [3], [3]], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    solver = ConTree(max_depth=2, max_gap=0, time_limit=1).fit(X, y)
    model = CalibratedConTree(solver, X, y, smoothing=0, timed_out=False,
                              topology_n=len(y), topology_positive_rate=.5)
    np.testing.assert_array_equal(model.predict_proba(X), solver.predict_proba(X))
    assert model.get_depth() == solver.get_depth()
    assert model.get_n_leaves() == solver.get_num_leaf_nodes()
