"""One entry point for every tree model that is a logit sum of trees: ``as_tree_sum``.

``as_tree_sum(model, X)`` returns a bettertrees sum with the same raw score as
``model``, so the interpretation API (``segments``, ``profiles``, ``plot_profiles``,
``plot_trees``, ``explain``, ``to_sql``, ...) reads any of them the same way:

- bettertrees sums (ITM, ``BudgetClassifier``, ...): returned as they are;
- LightGBM (``LGBMClassifier`` / ``Booster``): through ``from_lightgbm``;
- XGBoost (``XGBClassifier`` / ``Booster``, ``binary:logistic``);
- CatBoost (``CatBoostClassifier``, numerical features; its symmetric trees become
  full binary trees with one cut per level);
- scikit-learn: ``HistGradientBoostingClassifier``, ``GradientBoostingClassifier`` and
  ``DecisionTreeClassifier`` (a single tree; its leaves become the logit of their
  class frequency, clipped to [1e-6, 1 - 1e-6]).

Thresholds are taken in the precision each library compares in (float32 for
XGBoost and scikit-learn's ``tree`` module, float64 for the others), so the scores
match. bettertrees sends a missing value to the left branch: a cut that sends it
right is accepted only when the feature has no missing value in ``X`` (the direction
is then never used on that data), and refused otherwise. Categorical splits are
refused. ``RandomForestClassifier`` averages probabilities, not logits, and is not
a sum of trees in this sense.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import ArrayLike
from ._common import fit_inputs
from .imported import TreeSum
from .smalltrees import SmallTree

MAX_THRESHOLDS = 254  # uint8 bins: bin 0 = missing, one bin per threshold plus one


def _node(feature, threshold, left, right, nan_left=True, order=None):
    out = {
        "f": int(feature),
        "t": float(threshold),
        "l": left,
        "r": right,
        "nan_left": bool(nan_left),
    }
    if order is not None:
        out["order"] = int(order)  # the library's own order of the cuts inside a tree
    return out


def _leaf(value):
    return {"v": float(value)}


def _build(trees, base, X, y, classes, dtype, source):
    """``TreeSum`` from nested node dicts (``x <= t`` goes left)."""
    out = TreeSum()
    if y is None:
        y = np.resize(np.asarray(classes), max(len(X), 2))[: len(X)]
    Xv, cls, _, _ = fit_inputs(out, X, y)
    if classes is not None and not np.array_equal(np.asarray(cls), np.asarray(classes)):
        cls = np.asarray(classes)
    out.classes_ = cls
    nan_cols = np.isnan(Xv).any(axis=0)
    thresholds = [set() for _ in range(Xv.shape[1])]

    def collect(n):
        if "v" in n:
            return
        if n["f"] >= Xv.shape[1]:
            raise ValueError(
                f"the {source} model uses feature {n['f']}; X has {Xv.shape[1]} columns."
            )
        if not n["nan_left"] and nan_cols[n["f"]]:
            raise ValueError(
                f"a {source} split sends missing values of feature {n['f']} to the right; "
                "bettertrees always sends them left. Fill the missing values below the minimum "
                "before training."
            )
        thresholds[n["f"]].add(n["t"])
        collect(n["l"])
        collect(n["r"])

    for t in trees:
        collect(t)
    for f, cuts in enumerate(thresholds):
        if len(cuts) > MAX_THRESHOLDS:
            raise ValueError(
                f"feature {f} has {len(cuts)} distinct thresholds; at most {MAX_THRESHOLDS} "
                "fit the uint8 bins. Convert fewer trees or train with fewer bins."
            )
    out.input_dtype_ = dtype
    out.bin_edges_ = tuple(np.array(sorted(s), dtype=np.float64) for s in thresholds)
    small, orders = [], []
    for t in trees:
        tree, order = SmallTree(), []

        def grow(node, n, depth, tree=tree, order=order):
            if "v" in n:
                tree.value[node] = n["v"]
                return
            b = int(np.searchsorted(out.bin_edges_[n["f"]], n["t"])) + 1
            order.append((n.get("order", depth), len(order), node))
            left, right = tree.split(node, n["f"], b)
            grow(left, n["l"], depth + 1)
            grow(right, n["r"], depth + 1)

        grow(0, t, 0)
        small.append(tree)
        orders.append(order)
    out.base_margin_ = float(base) + float(sum(t.value[0] for t in small if t.n_splits == 0))
    kept = [k for k, t in enumerate(small) if t.n_splits > 0]
    out.trees_ = [small[k] for k in kept]
    # one tree after another; within a tree, the library's order if known, else level by level
    out.growth_ = [(j, node) for j, k in enumerate(kept) for *_, node in sorted(orders[k])]
    out.lam_, out.learning_rate_ = 1.0, 1.0
    out.source_ = source
    return out


# ------------------------------------------------------------------ XGBoost


def _from_xgboost(model, X, y):
    import json

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    cfg = json.loads(booster.save_config())
    obj = cfg["learner"]["objective"]["name"]
    if obj not in ("binary:logistic", "binary:logitraw"):
        raise ValueError(f"only binary logistic XGBoost models are supported (objective {obj}).")
    base_score = float(cfg["learner"]["learner_model_param"]["base_score"].strip("[]"))
    base = base_score if obj == "binary:logitraw" else float(np.log(base_score / (1 - base_score)))
    names = booster.feature_names
    index = {n: i for i, n in enumerate(names)} if names else None

    def feat(s):
        if index is not None and s in index:
            return index[s]
        return int(s.lstrip("f"))

    def conv(n):
        if "leaf" in n:
            return _leaf(n["leaf"])
        if "split_condition" not in n:
            raise ValueError("only numerical XGBoost splits are supported (no categorical splits).")
        kids = {c["nodeid"]: c for c in n["children"]}
        # x < c goes "yes"; in float32, x < c  <=>  x <= the float32 just below c
        thr = float(np.nextafter(np.float32(n["split_condition"]), np.float32(-np.inf)))
        return _node(
            feat(n["split"]),
            thr,
            conv(kids[n["yes"]]),
            conv(kids[n["no"]]),
            nan_left=n["missing"] == n["yes"],
        )

    dumps = booster.get_dump(dump_format="json")
    trees = [conv(json.loads(d)) for d in dumps]
    classes = getattr(model, "classes_", None)
    return _build(trees, base, X, y, classes, "float32", "XGBoost")


# ------------------------------------------------------------------ CatBoost


def _from_catboost(model, X, y):
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.json")
        model.save_model(path, format="json")
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
    feats = info.get("features_info", {})
    if (
        feats.get("categorical_features")
        or feats.get("text_features")
        or feats.get("embedding_features")
    ):
        raise ValueError("only CatBoost models with numerical features are supported.")
    float_feats = feats.get("float_features", [])
    col = {
        ff["feature_index"]: ff.get("flat_feature_index", ff["feature_index"]) for ff in float_feats
    }
    nan_mode = {ff["feature_index"]: ff.get("nan_value_treatment", "AsIs") for ff in float_feats}
    scale, bias = info.get("scale_and_bias", [1.0, [0.0]])
    bias = float(bias[0] if isinstance(bias, list) else bias)
    trees = []
    for t in info["oblivious_trees"]:
        splits = t.get("splits", [])
        if any(s.get("split_type", "FloatFeature") != "FloatFeature" for s in splits):
            raise ValueError("only CatBoost float-feature splits are supported.")
        vals = np.asarray(t["leaf_values"], dtype=float) * float(scale)
        depth = len(splits)

        def grow(level, index, splits=splits, vals=vals, depth=depth):
            if level < 0:
                return _leaf(vals[index])
            s = splits[level]
            fi = int(s["float_feature_index"])
            # bit ``level`` of the leaf index is 1 when x > border (missing as minimum: left)
            return _node(
                col.get(fi, fi),
                float(s["border"]),
                grow(level - 1, index),
                grow(level - 1, index | (1 << level)),
                nan_left=nan_mode.get(fi, "AsIs") != "AsTrue",
            )

        trees.append(grow(depth - 1, 0) if depth else _leaf(vals[0]))
    classes = getattr(model, "classes_", None)
    return _build(trees, bias, X, y, classes, "float32", "CatBoost")


# ------------------------------------------------------------------ scikit-learn


def _sk_tree(tree_, values):
    left, right = tree_.children_left, tree_.children_right
    miss = getattr(tree_, "missing_go_to_left", None)

    def conv(i):
        if left[i] == -1:
            return _leaf(values[i])
        return _node(
            tree_.feature[i],
            float(tree_.threshold[i]),
            conv(left[i]),
            conv(right[i]),
            nan_left=True if miss is None else bool(miss[i]),
            order=i,  # scikit-learn numbers the nodes as it creates them
        )

    return conv(0)


def _from_hist_gradient_boosting(model, X, y):
    if getattr(model, "n_trees_per_iteration_", 1) != 1:
        raise ValueError("only binary HistGradientBoostingClassifier models are supported.")
    trees = []
    for (pred,) in model._predictors:
        nodes = pred.nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits of HistGradientBoosting are not supported.")

        def conv(i, nodes=nodes):
            n = nodes[i]
            if n["is_leaf"]:
                return _leaf(n["value"])
            return _node(
                n["feature_idx"],
                float(n["num_threshold"]),
                conv(int(n["left"])),
                conv(int(n["right"])),
                nan_left=bool(n["missing_go_to_left"]),
            )

        trees.append(conv(0))
    base = float(np.ravel(model._baseline_prediction)[0])
    return _build(trees, base, X, y, model.classes_, "float64", "HistGradientBoosting")


def _from_gradient_boosting(model, X, y):
    loss = type(getattr(model, "_loss", None)).__name__
    if model.estimators_.shape[1] != 1 or loss not in ("HalfBinomialLoss", "NoneType"):
        raise ValueError("only binary log-loss GradientBoostingClassifier models are supported.")
    lr = float(model.learning_rate)
    trees = [_sk_tree(est.tree_, lr * est.tree_.value[:, 0, 0]) for est in model.estimators_[:, 0]]
    out = _build(trees, 0.0, X, y, model.classes_, "float32", "GradientBoosting")
    # the initial raw score (the prior's logit): the gap between the two raw scores
    Xv = X.iloc[:1] if hasattr(X, "iloc") else np.asarray(X, dtype=np.float64)[:1]
    out.base_margin_ = float(
        model.decision_function(Xv)[0] - (out.decision_function(Xv)[0] - out.base_margin_)
    )
    return out


def _from_decision_tree(model, X, y):
    if len(model.classes_) != 2:
        raise ValueError("only binary DecisionTreeClassifier models are supported.")
    v = model.tree_.value[:, 0, :]
    p = np.clip(v[:, 1] / v.sum(axis=1), 1e-6, 1 - 1e-6)
    return _build(
        [_sk_tree(model.tree_, np.log(p / (1 - p)))],
        0.0,
        X,
        y,
        model.classes_,
        "float32",
        "DecisionTree",
    )


# ------------------------------------------------------------------ dispatch


def as_tree_sum(model: Any, X: ArrayLike, y: ArrayLike | None = None) -> Any:
    """The model as a bettertrees sum with the same raw score (logit of ``classes_[1]``).

    Parameters
    ----------
    model : fitted binary classifier
        A bettertrees sum (returned unchanged), LightGBM, XGBoost, CatBoost,
        ``HistGradientBoostingClassifier``, ``GradientBoostingClassifier`` or
        ``DecisionTreeClassifier``.
    X : array-like
        Training data or a sample of it with the model's columns, in the same order:
        it sets feature names and checks missing values.
    y : array-like or None, default=None
        Labels of ``X``; only needed when the model does not expose ``classes_``.

    Returns
    -------
    A fitted bettertrees sum: ``decision_function(X)`` equals the model's raw score.

    Examples
    --------
    >>> import lightgbm as lgb  # doctest: +SKIP
    >>> from bettertrees import as_tree_sum
    >>> lgbm = lgb.LGBMClassifier(n_estimators=5, num_leaves=3).fit(X, y)  # doctest: +SKIP
    >>> as_tree_sum(lgbm, X).plot_profiles(X)  # doctest: +SKIP
    """
    if type(model).__name__ == "BudgetClassifier":
        return model.model_
    if hasattr(type(model), "_explain_trees"):
        return model
    mod, name = type(model).__module__, type(model).__name__
    if mod.startswith("lightgbm"):
        from .imported import from_lightgbm

        if y is None:
            y = (
                np.resize(np.asarray(model.classes_), len(X))
                if hasattr(model, "classes_")
                else None
            )
        if y is None:
            raise ValueError("a LightGBM Booster has no classes_: pass y.")
        return from_lightgbm(model, X, y)
    if mod.startswith("xgboost"):
        return _from_xgboost(model, X, y)
    if mod.startswith("catboost"):
        return _from_catboost(model, X, y)
    if name == "HistGradientBoostingClassifier":
        return _from_hist_gradient_boosting(model, X, y)
    if name == "GradientBoostingClassifier":
        return _from_gradient_boosting(model, X, y)
    if name == "DecisionTreeClassifier":
        return _from_decision_tree(model, X, y)
    raise TypeError(
        f"{mod}.{name} is not supported: as_tree_sum reads bettertrees sums, LightGBM, "
        "XGBoost, CatBoost, HistGradientBoostingClassifier, GradientBoostingClassifier and "
        "DecisionTreeClassifier."
    )
