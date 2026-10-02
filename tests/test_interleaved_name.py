"""The Interleaved Tree Model is the old FIGSClassifier under its new name."""

import numpy as np

import bettertrees
from bettertrees import BudgetClassifier, FIGSClassifier, InterleavedTreeClassifier
from bettertrees.lab import InteractingFIGSClassifier, InteractingTreeClassifier


def test_figs_is_the_same_class():
    assert FIGSClassifier is InterleavedTreeClassifier
    assert bettertrees.sums.FIGSClassifier is bettertrees.sums.InterleavedTreeClassifier
    assert InteractingFIGSClassifier is InteractingTreeClassifier
    assert "InterleavedTreeClassifier" in bettertrees.__all__ and "FIGSClassifier" in bettertrees.__all__


def test_params_and_defaults_unchanged():
    params = InterleavedTreeClassifier().get_params()
    assert params == dict(max_splits=16, max_trees=None, lam="auto", min_weight=20.0, max_bins=32,
                          backfit_sweeps=1, learning_rate=1.0, max_delta_step=None)


def test_budget_builds_the_itm():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + rng.normal(size=400) > 0).astype(int)
    for b in (4, 16):
        m = BudgetClassifier(b).fit(X, y)
        assert type(m.model_) is InterleavedTreeClassifier
    assert BudgetClassifier(4)._make().get_params()["max_delta_step"] == 4.0
    assert BudgetClassifier(16)._make().get_params()["learning_rate"] == 0.3
