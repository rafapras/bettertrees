"""Hierarchical leaf shrinkage and the choice of capacity by inner CV."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from tree_invariants import check_tree_invariants

from bettertrees import FastDecisionTreeClassifier, FastDecisionTreeClassifierCV
from bettertrees.postprocess import (
    expansion_steps,
    hierarchical_shrinkage_probabilities,
    prefix_leaf_ids,
)


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
    # huge lambda: everything goes back to the root
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
    with pytest.raises(ValueError, match="leaf_shrinkage OR leaf_smoothing"):
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
        # same partition of the rows (ids differ between the two trees)
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


def test_cv_zero_weight_rows_do_not_create_empty_validation_mass():
    X = np.arange(100, dtype=np.float32).reshape(-1, 1)
    y = (X[:, 0] >= 50).astype(int)
    active, _ = next(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    weights = np.zeros(len(y))
    weights[active] = 1.0
    params = dict(leaves_grid=(2, 4), shrinkage_grid=(1.0, 5.0), cv=3)
    weighted = FastDecisionTreeClassifierCV(**params).fit(X, y, sample_weight=weights)
    filtered = FastDecisionTreeClassifierCV(**params).fit(X[active], y[active])
    assert weighted.best_params_ == filtered.best_params_
    np.testing.assert_array_equal(weighted.cv_scores_, filtered.cv_scores_)
    np.testing.assert_array_equal(weighted.predict_proba(X), filtered.predict_proba(X))


def test_cv_retains_classes_present_only_in_zero_weight_rows():
    X = np.arange(24, dtype=np.float32).reshape(-1, 1)
    y = np.repeat(["a", "b", "inactive"], 8)
    weights = np.r_[np.ones(16), np.zeros(8)]
    model = FastDecisionTreeClassifierCV(leaves_grid=(2,), shrinkage_grid=(1.0,), cv=3)
    model.fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(model.classes_, ["a", "b", "inactive"])
    p = model.predict_proba(X)
    assert p.shape == (24, 3)
    np.testing.assert_array_equal(p[:, 2], 0.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_cv_rejects_insufficient_active_class_support():
    X = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = np.repeat([0, 1], 6)
    weights = np.r_[np.ones(6), np.ones(2), np.zeros(4)]
    with pytest.raises(ValueError, match=r"positive sample weight.*at least cv rows"):
        FastDecisionTreeClassifierCV(cv=3).fit(X, y, sample_weight=weights)


@pytest.mark.parametrize(("params", "message"), [
    ({"leaves_grid": (2.5,)}, "leaves_grid"),
    ({"leaves_grid": (True, 2)}, "leaves_grid"),
    ({"leaves_grid": ()}, "leaves_grid"),
    ({"shrinkage_grid": (np.nan, 1.0)}, "shrinkage_grid"),
    ({"shrinkage_grid": (np.inf, 1.0)}, "shrinkage_grid"),
    ({"shrinkage_grid": (0.0,)}, "shrinkage_grid"),
    ({"shrinkage_grid": (True,)}, "shrinkage_grid"),
    ({"cv": 2.5}, "cv must be an integer"),
    ({"cv": True}, "cv must be an integer"),
    ({"cv": 1}, "cv must be an integer"),
])
def test_cv_rejects_invalid_grids_and_fold_count(params, message):
    X = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = np.repeat([0, 1], 6)
    with pytest.raises(ValueError, match=message):
        FastDecisionTreeClassifierCV(**params).fit(X, y)


@pytest.mark.parametrize("weights", [np.ones(11), np.zeros(12), [-1] + [1] * 11,
                                      [np.nan] + [1] * 11])
def test_cv_validates_weights_before_forming_folds(weights):
    X = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = np.repeat([0, 1], 6)
    with pytest.raises(ValueError):
        FastDecisionTreeClassifierCV().fit(X, y, sample_weight=weights)


def test_cv_keeps_column_names_and_rejects_reordered_columns():
    # the CV once fitted the final tree on a NumPy copy: names were lost and a DataFrame
    # with the columns in another order was scored without any error
    pd = pytest.importorskip("pandas")
    X, y = _data()
    frame = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
    model = FastDecisionTreeClassifierCV(leaves_grid=(2, 8), cv=3).fit(frame, y)
    assert list(model.feature_names_in_) == list(frame.columns)
    with pytest.raises(ValueError):
        model.predict_proba(frame[frame.columns[::-1]])
    # a refit on an array drops the old names
    assert not hasattr(model.fit(X, y), "feature_names_in_")


@pytest.mark.parametrize("kwargs", [dict(), dict(leaf_shrinkage=10.0)])
def test_export_text_shows_the_probabilities_predict_proba_returns(kwargs):
    import re
    X, y = _data()
    tree = FastDecisionTreeClassifier(max_leaf_nodes=9, **kwargs).fit(X, y)
    text = tree.export_text(precision=10)
    shown = [float(v) for v in re.findall(r"P\(\S+\) = ([0-9.]+)", text)]
    assert len(shown) == tree.get_n_leaves()
    predicted = {round(v, 10) for v in tree.predict_proba(X)[:, 1]}
    assert predicted <= {round(v, 10) for v in shown}
    cv = FastDecisionTreeClassifierCV(leaves_grid=(2, 8), cv=3).fit(X, y)
    assert cv.export_text().count("P(") == cv.best_estimator_.get_n_leaves()
