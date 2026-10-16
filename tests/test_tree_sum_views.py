"""as_tree_sum (one entry point for tree models) and the scorecard views of every sum."""

import itertools

import matplotlib
import numpy as np
import pandas as pd
import pytest

from bettertrees import BudgetClassifier, InterleavedTreeClassifier, as_tree_sum

matplotlib.use("Agg")


def _data(n=3000, seed=0, nan=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "income": rng.lognormal(10, 0.6, n),
            "debt": rng.uniform(0, 1, n),
            "late": rng.poisson(0.6, n).astype(float),
            "age": rng.uniform(18, 80, n),
        }
    )
    logit = (
        -2
        + 1.5 * (X.debt > 0.6)
        + 0.8 * np.minimum(X.late, 3)
        - 0.5 * (X.income > 30000)
        + 0.6 * (X.late > 1) * (X.debt > 0.4)
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    if nan:
        X.loc[rng.uniform(size=n) < 0.05, "income"] = np.nan
    return X, y


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _same_score(model, X, atol=1e-9):
    s = as_tree_sum(model, X)
    np.testing.assert_allclose(
        s.decision_function(X), _logit(model.predict_proba(X)[:, 1]), atol=atol
    )
    return s


# ------------------------------------------------------------------ conversion


def test_bettertrees_sums_are_returned_as_they_are():
    X, y = _data()
    itm = InterleavedTreeClassifier(max_splits=6).fit(X, y)
    assert as_tree_sum(itm, X) is itm
    budget = BudgetClassifier(6).fit(X, y)
    assert as_tree_sum(budget, X) is budget.model_


def test_lightgbm():
    lgb = pytest.importorskip("lightgbm")
    X, y = _data()
    s = _same_score(lgb.LGBMClassifier(n_estimators=5, num_leaves=3, verbose=-1).fit(X, y), X)
    assert sum(t.n_splits for t in s.trees_) == 10


@pytest.mark.parametrize("nan", [False, True])
def test_xgboost_matches_in_float32(nan):
    xgb = pytest.importorskip("xgboost")
    X, y = _data(nan=nan)
    m = xgb.XGBClassifier(n_estimators=6, max_depth=2, learning_rate=0.5).fit(X, y)
    try:
        _same_score(m, X, atol=1e-5)  # XGBoost's own output is float32
    except ValueError as exc:  # a learnt default direction that sends NaN right is refused
        assert nan and "missing values" in str(exc)


@pytest.mark.parametrize("nan", [False, True])
def test_catboost_symmetric_trees(nan):
    cb = pytest.importorskip("catboost")
    X, y = _data(nan=nan)
    m = cb.CatBoostClassifier(iterations=8, depth=3, verbose=False, allow_writing_files=False).fit(
        X, y
    )
    s = _same_score(m, X)
    assert all(t.n_splits == 7 for t in s.trees_)  # depth 3: a full binary tree


@pytest.mark.parametrize("nan", [False, True])
def test_sklearn_hist_gradient_boosting(nan):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _data(nan=nan)
    _same_score(HistGradientBoostingClassifier(max_iter=6, max_leaf_nodes=4).fit(X, y), X)


def test_sklearn_gradient_boosting_and_tree():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    X, y = _data()
    _same_score(GradientBoostingClassifier(n_estimators=6, max_depth=2).fit(X, y), X)
    tree = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf=50).fit(X, y)
    s = _same_score(tree, X, atol=1e-6)
    assert len(s.trees_) == 1 and s.trees_[0].n_splits == 7


def test_unsupported_model_is_refused():
    from sklearn.linear_model import LogisticRegression

    X, y = _data()
    with pytest.raises(TypeError, match="not supported"):
        as_tree_sum(LogisticRegression(max_iter=2000).fit(X / X.max(), y), X)


# ------------------------------------------------------------------ views


@pytest.fixture(scope="module")
def fitted():
    X, y = _data(n=4000, nan=True)
    return BudgetClassifier(10).fit(X, y).model_, X


def test_segments_add_up_to_the_model(fitted):
    m, X = fitted
    seg = m.segments(X)
    pts = seg.filter(like="_points")
    np.testing.assert_allclose(seg["base"] + pts.sum(axis=1), seg["score"])
    np.testing.assert_allclose(seg["p"], m.predict_proba(X)[:, 1], rtol=1e-9)
    assert [c for c in seg.columns if c.startswith("tree_") and not c.endswith("_points")] == [
        f"tree_{k}" for k in range(1, len(m.trees_) + 1)
    ]
    logit = m.segments(X, scale="logit")
    np.testing.assert_allclose(logit["score"], m.decision_function(X))


def test_bad_class_flips_the_score(fitted):
    m, X = fitted
    a, b = m.segments(X), m.segments(X, bad=m.classes_[0])
    np.testing.assert_allclose(a["score"] - 600, -(b["score"] - 600))
    np.testing.assert_allclose(a["p"] + b["p"], 1.0)
    with pytest.raises(ValueError, match="bad must be"):
        m.segments(X, bad="nope")


def test_observed_profiles_are_real_rows(fitted):
    m, X = fitted
    seg, prof = m.segments(X), m.profiles(X)
    assert prof.loc["best", "score"] == pytest.approx(seg["score"].max())
    assert prof.loc["worst", "score"] == pytest.approx(seg["score"].min())
    assert seg.loc[prof.loc["worst", "row"], "score"] == pytest.approx(seg["score"].min())
    cols = [f"tree_{k}" for k in range(1, len(m.trees_) + 1)]
    share = (seg[cols] == prof.loc["modal", cols]).all(axis=1).mean()
    assert prof.loc["modal", "share"] == pytest.approx(share)
    assert share == pytest.approx(seg.groupby(cols).size().max() / len(seg))


def test_feasible_profiles_bound_the_observed_ones_and_are_exact(fitted):
    m, X = fitted
    obs, feas = m.profiles(X), m.profiles(X, kind="feasible")
    assert feas.loc["best", "score"] >= obs.loc["best", "score"] - 1e-9
    assert feas.loc["worst", "score"] <= obs.loc["worst", "score"] + 1e-9
    # brute force over every combination of leaves of a small model
    from bettertrees.sums.profiles import _leaf_boxes

    nb = [len(e) + 2 for e in m.bin_edges_]
    boxes = [_leaf_boxes(t, nb, m.nan_features_) for t in m.trees_]
    best = worst = None
    for combo in itertools.product(*[list(b.items()) for b in boxes]):
        merged, ok = {}, True
        for _, box in combo:
            for f, (lo, hi) in box.items():
                a, c = merged.get(f, (-1, nb[f] - 1))
                a, c = max(a, lo), min(c, hi)
                ok &= a < c
                merged[f] = (a, c)
        if ok:
            v = sum(float(t.value[leaf]) for t, (leaf, _) in zip(m.trees_, combo))
            best, worst = (
                min(v, best if best is not None else v),
                max(v, worst if worst is not None else v),
            )
    factor = 20 / np.log(2)
    base = 600 - factor * m.base_margin_
    assert feas.loc["best", "score"] == pytest.approx(base - factor * best)
    assert feas.loc["worst", "score"] == pytest.approx(base - factor * worst)


def test_plots_return_figures(fitted):
    m, X = fitted
    fig = m.plot_profiles(X)
    assert fig.axes
    fig = m.plot_trees(X.iloc[[0]], X=X)
    assert fig.axes
    lgb = pytest.importorskip("lightgbm")
    Xf = X.fillna(X.min() - 1)
    big = as_tree_sum(
        lgb.LGBMClassifier(n_estimators=30, num_leaves=4, verbose=-1).fit(Xf, (m.predict(X))), Xf
    )
    assert big.plot_profiles(Xf, max_trees=5).axes  # the rest summed in one line
    assert big.plot_trees(Xf.iloc[0], max_trees=4).axes
    matplotlib.pyplot.close("all")


def test_growth_order_interleaved_against_sequential():
    lgb = pytest.importorskip("lightgbm")
    X, y = _data(n=4000)
    itm = InterleavedTreeClassifier(max_splits=10).fit(X, y)
    growth, recorded = itm._growth()
    assert recorded and len(growth) == itm.n_splits_
    booster = as_tree_sum(lgb.LGBMClassifier(n_estimators=5, num_leaves=3, verbose=-1).fit(X, y), X)
    seq, rec = booster._growth()
    assert rec and [k for k, _ in seq] == sorted(k for k, _ in seq)  # one tree after another
    assert booster._model_name() == "LightGBM" and itm._model_name() == "ITM"
    assert itm.plot_growth().figure is not None
    if len(itm.trees_) > 1:  # an edit invalidates the record: falls back to tree by tree
        itm.drop_tree(len(itm.trees_) - 1)
        assert itm._growth()[1] is False
    matplotlib.pyplot.close("all")


def test_median_and_chosen_rows(fitted):
    m, X = fitted
    seg = m.segments(X)
    prof = m.profiles(X, which=("median",), rows=[X.index[5], X.index[7]])
    assert list(prof.index) == ["median", f"client {X.index[5]}", f"client {X.index[7]}"]
    assert prof.loc["median", "score"] == pytest.approx(
        np.sort(seg["score"].to_numpy())[(len(X) - 1) // 2]
    )
    assert prof.loc[f"client {X.index[5]}", "score"] == pytest.approx(seg.loc[X.index[5], "score"])
    with pytest.raises(ValueError, match="unknown profiles"):
        m.profiles(X, which=("typical",))
    fig = m.plot_profiles(X, which=("best", "median", "worst"), rows=[X.index[5]])
    assert fig.axes
    fig = m.plot_trees(X.loc[[X.index[5]]], X=X)  # the client's values on its path
    assert any(
        t.get_text().startswith("= ") or t.get_text() == "missing"
        for ax in fig.axes
        for t in ax.texts
    )
    matplotlib.pyplot.close("all")


def test_plots_say_what_they_add_up(fitted):
    m, X = fitted
    texts = lambda fig: [t.get_text() for t in fig.texts]
    seg = m.segments(X)
    fig = m.plot_profiles(X, bad_label="default")
    assert "base" in texts(fig)
    assert any("odds of default" in t for t in texts(fig))
    assert any("P(default)" in t for t in texts(fig))
    row = X.loc[[X.index[3]]]
    fig = m.plot_trees(row, X=X, bad_label="default")
    footer = next(t for t in texts(fig) if t.startswith("this client:"))
    terms, total = footer.removeprefix("this client:").split(" points")[0].split(" = ")
    assert sum(int(v) for v in terms.split()) == int(total)  # adds up as printed, in points
    assert abs(int(total) - seg.loc[X.index[3], "score"]) <= len(m.trees_) / 2 + 0.5
    assert any("#k = the order" in t and "% = share" in t for t in texts(fig))
    fig = m.plot_trees(row, scale="logit")
    footer = next(t for t in texts(fig) if t.startswith("this client:"))
    assert "points" not in footer
    ax = m.plot_growth()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "opens a new tree",
        "deepens a tree",
    ]
    matplotlib.pyplot.close("all")
