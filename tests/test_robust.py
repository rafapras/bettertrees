"""Structure selectors for FIGS: bootstrap vote, bag distillation, Rashomon search."""

import numpy as np
import pytest

from bettertrees.sums import BaggedFIGSClassifier, FIGSClassifier, RashomonFIGSClassifier
from bettertrees.sums.robust import _coarsen, _cuts, _uncoarsen
from bettertrees.sums.smalltrees import SmallTree


def _data(seed=0, n=4000, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[rng.random(n) < 0.05, 1] = np.nan
    logit = X[:, 0] + np.nan_to_num(X[:, 1]) * X[:, 2] + 0.5 * np.sign(X[:, 3])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_coarsen_maps_cuts_back_to_the_same_partition():
    rng = np.random.default_rng(1)
    Xb = rng.integers(0, 10, size=(500, 3)).astype(np.uint8)
    nb = np.full(3, 10, dtype=np.int64)
    vocab = [(0, 3), (0, 6), (2, 0), (2, 5)]
    Xc, nbc, T = _coarsen(Xb, nb, vocab)
    assert (Xc[:, 1] == 1).all()  # feature without vocabulary is constant
    for f, t in vocab:
        k = int(np.searchsorted(T[f], t)) + 1
        tree = _uncoarsen([SmallTree.from_nested((f, k, None, None))], T)[0]
        assert tree.threshold[0] == t
        np.testing.assert_array_equal(Xc[:, f] <= k, Xb[:, f] <= t)


def test_bagged_final_cuts_come_from_the_vocabulary():
    X, y = _data()
    m = BaggedFIGSClassifier(max_splits=12, n_bags=6, vocab_size=10).fit(X, y)
    assert len(m.vocabulary_) <= 10
    assert _cuts(m.trees_) <= set(m.vocabulary_)
    assert m.n_splits_ <= 12
    assert sum(m.cut_votes_.values()) >= m.n_splits_


def test_bagged_distillation_and_no_vocabulary():
    X, y = _data()
    m = BaggedFIGSClassifier(max_splits=10, n_bags=5, vocab_size=0, distill=True).fit(X, y)
    assert m.vocabulary_ is None and 0 < m.n_splits_ <= 10
    p = m.predict_proba(X)[:, 1]
    assert np.isfinite(p).all() and p.min() > 0 and p.max() < 1


def test_bagged_is_deterministic_given_random_state():
    X, y = _data()
    a = BaggedFIGSClassifier(max_splits=8, n_bags=4, random_state=3).fit(X, y)
    b = BaggedFIGSClassifier(max_splits=8, n_bags=4, random_state=3).fit(X, y)
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))


@pytest.mark.parametrize("select", ["best", "stable"])
def test_rashomon_set_and_selection(select):
    X, y = _data()
    m = RashomonFIGSClassifier(max_splits=10, n_mutations=30, select=select).fit(X, y)
    losses = [s[0] for s in m.rashomon_set_]
    assert losses and max(losses) <= min(losses) * (1 + m.epsilon) + 1e-12
    assert len(m.history_) == 30 and np.all(np.diff(m.history_) <= 1e-12)  # hill climbing
    assert 0 < m.n_splits_ <= 10


def test_figs_learning_rate_one_is_the_default_path():
    X, y = _data()
    a = FIGSClassifier(max_splits=8).fit(X, y)
    b = FIGSClassifier(max_splits=8, learning_rate=1.0).fit(X, y)
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))
    c = FIGSClassifier(max_splits=8, learning_rate=0.3).fit(X, y)
    assert c.learning_rate_ == 0.3
    assert not np.allclose(a.decision_function(X), c.decision_function(X))


def test_compact_counts_distinct_cuts_and_reuses_terms():
    from bettertrees import CompactTreeBooster
    X, y = _data()
    plain = CompactTreeBooster(max_splits=20, depth=2, validation_fraction=0).fit(X, y)
    reuse = CompactTreeBooster(max_splits=20, depth=2, new_cut_penalty=2.0,
                               validation_fraction=0).fit(X, y)
    for m in (plain, reuse):
        assert m.n_splits_ <= 20
        keys = [tuple(zip(t.feature, t.threshold)) for t in m.trees_]
        assert len(keys) == len(set(keys))  # identical trees were merged
    assert reuse.n_rounds_ > plain.n_rounds_  # re-boosting existing terms is free


def test_compact_stump_is_absorbed_by_a_tree_with_the_same_root():
    from bettertrees.sums.compact import _TermSet
    ts = _TermSet()
    ts.add((0, 5, -1, -1, -1, -1), np.array([1.0, 0, 2.0, 0]))
    assert ts.n_cuts == 1
    assert ts.cost((0, 5, 1, 3, 2, 4)) == 2  # 3 cuts minus the absorbed stump
    ts.add((0, 5, 1, 3, 2, 4), np.array([0.1, 0.2, 0.3, 0.4]))
    assert ts.n_cuts == 3 and len(ts.terms) == 1
    np.testing.assert_allclose(ts.terms[(0, 5, 1, 3, 2, 4)], [1.1, 1.2, 2.3, 2.4])
    ts.add((0, 5, -1, -1, -1, -1), np.array([1.0, 0, -1.0, 0]))  # absorbed again, free
    assert ts.n_cuts == 3
    np.testing.assert_allclose(ts.terms[(0, 5, 1, 3, 2, 4)], [2.1, 2.2, 1.3, 1.4])


def test_compact_auto_depth_mixes_stumps_and_pairs():
    from bettertrees import CompactTreeBooster
    X, y = _data(n=6000)
    m = CompactTreeBooster(max_splits=30, depth="auto").fit(X, y)
    sizes = {t.n_splits for t in m.trees_}
    assert m.n_splits_ <= 30 and sizes <= {1, 2, 3}
