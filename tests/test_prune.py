"""Reduced-error pruning of a fitted sum down to a cut budget."""

import numpy as np

from bettertrees import CompactTreeBooster, FIGSClassifier
from bettertrees.sums._common import rebin
from bettertrees.sums.prune import prune_to_budget


def _data(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logit = X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * np.sign(X[:, 3])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def _prune(m, X, y, budget):
    fi, va = np.arange(len(y)) % 5 != 0, np.arange(len(y)) % 5 == 0
    Xf, Xv = rebin(X[fi], m.bin_edges_), rebin(X[va], m.bin_edges_)
    return prune_to_budget(m, budget, Xf, y[fi].astype(float), np.ones(fi.sum()),
                           Xv, y[va].astype(float), np.ones(va.sum()), lam=1.0)


def test_prune_reaches_the_budget_and_keeps_outputs_consistent():
    X, y = _data()
    for m in (FIGSClassifier(max_splits=24).fit(X, y),
              CompactTreeBooster(max_splits=24, validation_fraction=0).fit(X, y)):
        before = m.n_splits_
        removed = _prune(m, X, y, 10)
        assert m.n_splits_ <= 10 and removed >= before - 10
        np.testing.assert_allclose(m.base_margin_ + m.predict_contributions(X).sum(axis=1),
                                   m.decision_function(X), atol=1e-9)
        p = m.predict_proba(X)[:, 1]
        assert np.isfinite(p).all()
        m.rules()


def test_prune_to_zero_cuts_leaves_a_constant_model():
    X, y = _data(n=1500)
    m = FIGSClassifier(max_splits=6).fit(X, y)
    _prune(m, X, y, 0)
    assert m.n_splits_ == 0
    d = m.decision_function(X)
    assert np.allclose(d, d[0])
