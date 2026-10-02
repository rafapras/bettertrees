"""Structure selectors for FIGS: bootstrap vote, bag distillation, Rashomon search."""

import numpy as np
import pytest

from bettertrees.lab import BaggedFIGSClassifier, RashomonFIGSClassifier
from bettertrees.lab.robust import _coarsen, _cuts, _uncoarsen
from bettertrees.sums import FIGSClassifier
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


def _tree(nested, values):
    from bettertrees.sums.smalltrees import SmallTree
    t = SmallTree.from_nested(nested)
    t.value = np.asarray(values, dtype=float)
    return t


def test_compact_merges_refined_trees_and_counts_distinct_cuts():
    from bettertrees.sums.compact import _TermSet
    from bettertrees.sums.edit import _key
    stump = _tree((0, 5, None, None), [0, 1.0, 2.0])
    d2 = _tree((0, 5, (1, 3, None, None), (2, 4, None, None)), np.arange(7) / 10)
    ts = _TermSet()
    ts.add(_key(stump), stump)
    assert ts.n_cuts == 1 and ts.cost(_key(d2)) == 2  # 3 cuts minus the absorbed stump
    ts.add(_key(d2), d2)
    assert ts.n_cuts == 3 and len(ts.terms) == 1
    np.testing.assert_allclose(ts.terms[_key(d2)].value, [0, 0.1, 0.2, 1.3, 1.4, 2.5, 2.6])
    ts.add(_key(stump), stump)  # a coarser tree is absorbed for free
    assert ts.n_cuts == 3
    np.testing.assert_allclose(ts.terms[_key(d2)].value, [0, 0.1, 0.2, 2.3, 2.4, 4.5, 4.6])
    grown = _tree((0, 5, (1, 3, (3, 1, None, None), None), (2, 4, None, None)), np.zeros(9))
    assert ts.cost(_key(grown)) == 1  # growing a leaf pays only the new cut
    ts.add(_key(grown), grown)
    assert ts.n_cuts == 4 and len(ts.terms) == 1


def test_compact_merged_model_predicts_like_the_sum_of_rounds():
    """Merging and absorption must not change predictions: refit-free model =
    base + sum of every boosting step kept."""
    from bettertrees import CompactTreeBooster
    X, y = _data(n=3000)
    m = CompactTreeBooster(max_splits=24, depth="auto", validation_fraction=0).fit(X, y)
    trees = m.trees_
    keys = [tuple(zip(t.feature, t.threshold, t.left)) for t in trees]
    assert len(keys) == len(set(keys))
    assert m.n_splits_ <= 24
    np.testing.assert_allclose(m.base_margin_ + m.predict_contributions(X).sum(axis=1),
                               m.decision_function(X), atol=1e-9)


@pytest.mark.parametrize("depth", [1, 2, 3, "auto"])
def test_compact_depths_and_growth_respect_limits(depth):
    from bettertrees import CompactTreeBooster
    X, y = _data(n=5000)
    m = CompactTreeBooster(max_splits=30, depth=depth, max_depth=3).fit(X, y)
    assert 0 < m.n_splits_ <= 30
    for t in m.trees_:
        depth_of = {0: 0}
        for k in range(len(t.feature)):
            if t.left[k] != -1:
                depth_of[t.left[k]] = depth_of[t.right[k]] = depth_of[k] + 1
        assert max(depth_of.values()) <= 3
