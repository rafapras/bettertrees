"""Shrinkage hierárquico das folhas e escolha de capacidade por CV interna."""

import numpy as np
import pytest

from arvore_rapida import FastDecisionTreeClassifier, FastDecisionTreeClassifierCV
from arvore_rapida.postprocess import (expansion_steps,
                                       hierarchical_shrinkage_probabilities,
                                       prefix_leaf_ids)
from tree_invariants import check_tree_invariants


def _data(seed=0, n=1500, p=6, noise=1.0, classes=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    signal = np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2])
    signal = signal + noise * rng.normal(size=n)
    if classes == 2:
        return X, (signal > 0).astype(int)
    return X, np.digitize(signal, np.quantile(signal, np.linspace(0, 1, classes + 1)[1:-1]))


def _ancestors(nodes):
    parent = np.full(len(nodes.left), -1)
    for node in np.flatnonzero(nodes.left != -1):
        parent[nodes.left[node]] = parent[nodes.right[node]] = node
    return parent


@pytest.mark.parametrize("classes", [2, 3])
def test_shrinkage_is_convex_combination_of_ancestor_frequencies(classes):
    X, y = _data(classes=classes)
    model = FastDecisionTreeClassifier(max_leaf_nodes=40, min_samples_leaf=3,
                                       random_state=0).fit(X, y)
    nodes = model.nodes_
    freq = nodes.class_weight / nodes.class_weight.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(hierarchical_shrinkage_probabilities(nodes, 0.0), freq,
                               atol=1e-12)
    parent = _ancestors(nodes)
    for lam in (0.5, 10.0, 1e6):
        probs = hierarchical_shrinkage_probabilities(nodes, lam)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
        assert (probs > 0).all()
        for leaf in np.flatnonzero(nodes.left == -1):
            path, node = [], leaf
            while node >= 0:
                path.append(node)
                node = parent[node]
            lo, hi = freq[path].min(axis=0), freq[path].max(axis=0)
            assert np.all(probs[leaf] >= lo - 1e-12) and np.all(probs[leaf] <= hi + 1e-12)
    # λ enorme: tudo volta para a raiz
    np.testing.assert_allclose(hierarchical_shrinkage_probabilities(nodes, 1e12),
                               np.tile(freq[0], (len(freq), 1)), atol=1e-6)


def test_estimator_uses_shrunk_probabilities_and_keeps_invariants():
    X, y = _data()
    model = FastDecisionTreeClassifier(max_leaf_nodes=30, min_samples_leaf=3,
                                       leaf_shrinkage=20.0, random_state=0).fit(X, y)
    expected = hierarchical_shrinkage_probabilities(model.nodes_, 20.0)[model.apply(X)]
    np.testing.assert_allclose(model.predict_proba(X), expected, atol=0)
    check_tree_invariants(model, X, y)


def test_shrinkage_rejects_conflicting_options():
    X, y = _data(n=200)
    with pytest.raises(ValueError, match="leaf_shrinkage OU leaf_smoothing"):
        FastDecisionTreeClassifier(leaf_shrinkage=1.0, leaf_smoothing=1.0).fit(X, y)
    with pytest.raises(ValueError, match="monotonic_cst"):
        FastDecisionTreeClassifier(leaf_shrinkage=1.0,
                                   monotonic_cst=[1, 0, 0, 0, 0, 0]).fit(X, y)


@pytest.mark.parametrize("splitter", ["hist", "exact"])
def test_prefix_of_largest_tree_equals_direct_fit(splitter):
    X, y = _data(seed=3)
    params = dict(splitter=splitter, min_samples_leaf=4, random_state=2)
    big = FastDecisionTreeClassifier(max_leaf_nodes=64, **params).fit(X, y)
    steps = expansion_steps(big.nodes_)
    for leaves in (2, 5, 16, 37, 64):
        direct = FastDecisionTreeClassifier(max_leaf_nodes=leaves, **params).fit(X, y)
        truncated = prefix_leaf_ids(X, big.nodes_, steps, leaves)
        # mesma partição das linhas (ids diferem entre as duas árvores)
        pairs = set(zip(truncated.tolist(), direct.apply(X).tolist()))
        assert len(pairs) == len(set(truncated.tolist())) == len(set(direct.apply(X).tolist()))


def test_cv_picks_less_capacity_on_noise_than_on_signal():
    X_noise, y_noise = _data(seed=5, noise=50.0)
    X_sig, y_sig = _data(seed=5, noise=0.1)
    noise = FastDecisionTreeClassifierCV(random_state=0).fit(X_noise, y_noise)
    signal = FastDecisionTreeClassifierCV(random_state=0).fit(X_sig, y_sig)
    assert noise.best_params_["max_leaf_nodes"] < signal.best_params_["max_leaf_nodes"]
    for model, X in ((noise, X_noise), (signal, X_sig)):
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        assert model.best_params_["max_leaf_nodes"] in model.leaves_grid
        assert model.best_params_["leaf_shrinkage"] in model.shrinkage_grid
        assert model.best_estimator_.get_n_leaves() <= model.best_params_["max_leaf_nodes"]


def test_cv_handles_multiclass_and_string_labels():
    X, y = _data(seed=7, classes=3)
    labels = np.array(["baixo", "meio", "alto"])[y]
    model = FastDecisionTreeClassifierCV(leaves_grid=(4, 16), shrinkage_grid=(5.0,),
                                         random_state=1).fit(X, labels)
    assert set(model.predict(X)) <= set(labels)
    assert model.predict_proba(X).shape == (len(X), 3)
