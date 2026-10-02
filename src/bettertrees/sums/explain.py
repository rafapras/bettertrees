"""Interpretable form of logit sums of trees.

Every sum (``SumOfOptimalTrees``, ``FIGSClassifier``, ``BoostedOptimalTrees``,
``AdditiveTreeBooster``) is ``logit(p) = base + Σ_k tree_k(x)``, and each tree
is a list of leaves "conditions → value". The mixin here gives all of them the
same output:

- ``rules()``: (tree, conditions, logit value) per leaf, with the conditions of
  a path merged into per-feature intervals (``3 < x <= 7``);
- ``predict_contributions(X)``: each tree's contribution per row; added to the
  base, it reproduces ``decision_function`` exactly;
- ``explain()``: a text scorecard; ``to_dict()``: the same as JSON;
- ``get_trees()``: each tree as arrays (like sklearn's ``tree_``), with cuts in
  real values; ``export_text()``: the trees drawn as text.

Cut convention (bins): ``x <= v`` includes NaN (NaN always goes left);
``x > v`` excludes it. A cut at bin 0 only separates "missing" from "present".
The "or missing" note appears only for features that had NaN in training.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._common import bin_threshold, predict_input, rebin


def feature_names_of(X):
    """Column names (DataFrame with string column names) or None."""
    cols = getattr(X, "columns", None)
    if cols is None:
        return None
    names = np.asarray(cols, dtype=object)
    return names if all(isinstance(c, str) for c in names) else None


def _sql_ident(s):
    """A column name as a SQL identifier (quoted unless it is a plain word)."""
    s = str(s)
    if s.replace("_", "").isalnum() and not s[0].isdigit():
        return s
    return '"' + s.replace('"', '""') + '"'


def _fmt(v, precision):
    return f"{v:.{precision}g}"


def leaf_rules(tree, edges, names=None, nan_features=None, precision=6):
    """Leaves of ``tree`` as (interval conditions, value)."""
    name = (lambda j: str(names[j])) if names is not None else (lambda j: f"x{j}")
    out = []

    def conditions(path):
        bounds = {}  # feature -> [largest bin to the right (lo), smallest bin to the left (hi)]
        for f, t, went_left in path:
            lo, hi = bounds.setdefault(f, [None, None])
            if went_left:
                bounds[f][1] = t if hi is None else min(hi, t)
            else:
                bounds[f][0] = t if lo is None else max(lo, t)
        conds = []
        for f in sorted(bounds, key=lambda j: [p[0] for p in path].index(j)):
            lo, hi = bounds[f]
            nm = name(f)
            nan_ok = nan_features is not None and bool(nan_features[f])
            if lo is None:  # only left turns: NaN included
                if hi == 0:
                    conds.append(f"{nm} is missing")
                    continue
                txt = f"{nm} <= {_fmt(bin_threshold(edges[f], hi), precision)}"
                conds.append(f"({txt} or missing)" if nan_ok else txt)
            elif hi is None:
                conds.append(f"{nm} is present" if lo == 0 else
                             f"{nm} > {_fmt(bin_threshold(edges[f], lo), precision)}")
            else:
                low = "" if lo == 0 else f"{_fmt(bin_threshold(edges[f], lo), precision)} < "
                conds.append(f"{low}{nm} <= {_fmt(bin_threshold(edges[f], hi), precision)}")
        return conds

    def walk(node, path):
        if tree.left[node] == -1:
            out.append((conditions(path), float(tree.value[node])))
            return
        f, t = int(tree.feature[node]), int(tree.threshold[node])
        walk(tree.left[node], [*path, (f, t, True)])
        walk(tree.right[node], [*path, (f, t, False)])

    walk(0, [])
    return out


def leaf_order(tree):
    """Leaf node ids in the order ``leaf_rules`` lists them (depth first, left first).

    Not the node-id order: FIGS and the editing API append nodes, so a leaf created
    later can sit to the left of an older one."""
    out = []

    def walk(node):
        if tree.left[node] == -1:
            out.append(node)
            return
        walk(tree.left[node])
        walk(tree.right[node])

    walk(0)
    return out


class InterpretableSumMixin:
    """Shared interpretable output; the class defines ``_explain_trees()``."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False  # binary only (v1)
        tags.input_tags.allow_nan = True  # NaN has its own bin and always goes left
        return tags

    def _explain_trees(self):
        return self.trees_

    # ------------------------------------------------------------ prediction

    def predict_proba(self, X):
        """Probabilities of both classes.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Columns follow ``classes_``; ``[:, 1] = sigmoid(decision_function(X))``.
        """
        p = 0.5 * (1.0 + np.tanh(0.5 * self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Most likely class (``classes_[1]`` when the logit is positive)."""
        margin = self.decision_function(X)  # checks the fit before reading classes_
        return self.classes_[(margin > 0).astype(int)]

    @property
    def n_splits_(self):
        """Total number of cuts (internal nodes) over all trees."""
        return sum(t.n_splits for t in self._explain_trees())

    # ------------------------------------------------------------ trees

    def get_trees(self, feature_names=None):
        """The trees of the sum as arrays, in the format of sklearn's ``tree_``.

        Each tree is a dict of per-node arrays: ``feature`` (-1 at leaves),
        ``feature_name``, ``threshold`` (real value: ``x <= threshold`` goes
        left, NaN always goes left; ``-inf`` separates "missing" from
        "present"), ``children_left``/``children_right`` (-1 at leaves) and
        ``value`` (logit contribution; only leaves enter the sum).
        ``logit(p) = base_margin_ + Σ_k value_k[leaf of x in tree k]``.
        """
        check_is_fitted(self, "base_margin_")
        names = self._names(feature_names)
        out = []
        for tree in self._explain_trees():
            feat = np.asarray(tree.feature, dtype=np.int64)
            thr = np.array([bin_threshold(self.bin_edges_[f], t) if f >= 0 else np.nan
                            for f, t in zip(tree.feature, tree.threshold)])
            out.append(dict(
                feature=feat,
                feature_name=[None if f < 0 else (names[f] if names else f"x{f}")
                              for f in feat],
                threshold=thr,
                children_left=np.asarray(tree.left, dtype=np.int64),
                children_right=np.asarray(tree.right, dtype=np.int64),
                value=np.asarray(tree.value, dtype=np.float64).copy()))
        return out

    def to_shap_model(self):
        """The sum in SHAP's custom tree format, for exact TreeSHAP.

        ``shap.TreeExplainer(model.to_shap_model(), data=background,
        feature_perturbation="interventional")``; ``expected_value`` plus the row
        sum of the SHAP values reproduces ``decision_function`` (logit scale).
        Main-effect trees get the same attribution as ``predict_contributions``
        minus its background mean; SHAP only adds information for interaction
        trees. Node covers are not stored, so use the interventional mode.
        """
        trees = []
        for t in self.get_trees():
            leaf = t["children_left"] == -1
            trees.append(dict(
                children_left=t["children_left"], children_right=t["children_right"],
                children_default=t["children_left"].copy(),  # NaN always goes left
                features=np.where(leaf, -2, t["feature"]),
                thresholds=np.where(leaf, 0.0, t["threshold"]),
                values=np.where(leaf, t["value"], 0.0).reshape(-1, 1),
                node_sample_weight=np.ones(len(leaf))))
        return dict(trees=trees, base_offset=float(self.base_margin_))

    def export_text(self, feature_names=None, precision=4):
        """The trees drawn as text (in the style of ``sklearn.tree.export_text``)."""
        lines = [f"base (logit): {float(self.base_margin_):+.{precision}f}"]
        for k, t in enumerate(self.get_trees(feature_names), 1):
            lines.append(f"tree {k}")

            def walk(node, depth, t=t):
                pad = "|   " * depth
                if t["children_left"][node] == -1:
                    lines.append(f"{pad}|--- value: {t['value'][node]:+.{precision}f}")
                    return
                name, thr = t["feature_name"][node], t["threshold"][node]
                if np.isneginf(thr):
                    left, right = f"{name} is missing", f"{name} is present"
                else:
                    v = f"{thr:.{precision}g}"
                    left, right = f"{name} <= {v}", f"{name} >  {v}"
                lines.append(f"{pad}|--- {left}")
                walk(t["children_left"][node], depth + 1)
                lines.append(f"{pad}|--- {right}")
                walk(t["children_right"][node], depth + 1)

            walk(0, 0)
        return "\n".join(lines)

    def _names(self, feature_names):
        if feature_names is not None:
            return list(feature_names)
        names = getattr(self, "feature_names_in_", None)
        return None if names is None else list(names)

    def rules(self, feature_names=None, precision=6):
        """[(tree, conditions, logit value)] for every leaf."""
        check_is_fitted(self, "base_margin_")
        names, nan = self._names(feature_names), getattr(self, "nan_features_", None)
        return [(k, conds, v) for k, tree in enumerate(self._explain_trees())
                for conds, v in leaf_rules(tree, self.bin_edges_, names, nan, precision)]

    def predict_contributions(self, X):
        """(n_samples, n_trees) matrix of logit contributions; base + row sum
        = ``decision_function(X)``."""
        Xb = rebin(predict_input(self, X, "base_margin_"), self.bin_edges_)
        trees = self._explain_trees()
        out = np.zeros((len(Xb), len(trees)))
        for k, tree in enumerate(trees):
            out[:, k] = tree.value[tree.leaf_ids(Xb)]
        return out

    def to_dict(self, feature_names=None, precision=6):
        """The whole model as a JSON-ready structure (base + rule trees)."""
        check_is_fitted(self, "base_margin_")
        trees = {}
        for k, conds, v in self.rules(feature_names, precision):
            trees.setdefault(k, []).append(dict(conditions=conds, value=v))
        return dict(link="logit", base_margin=float(self.base_margin_),
                    classes=[c.item() if hasattr(c, "item") else c for c in self.classes_],
                    trees=[dict(rules=trees[k]) for k in sorted(trees)])

    def to_sql(self, table="input", feature_names=None, precision=6, keep_columns=True):
        """The model as one readable SQL query (additive ``CASE WHEN``, one column per tree).

        Each tree becomes a column ``t{k}_{features}`` holding its logit contribution;
        ``score = base + Σ columns`` and ``p = 1 / (1 + EXP(-score))``. Every ``WHEN``
        is a whole leaf (the path merged into per-feature intervals), so the branches
        read on their own, in any order. Missing values follow the model: ``x <= v``
        includes NULL for features that had NaN in training; other features are
        assumed non-null. Uses only ``CASE``, ``AND``, ``IS NULL`` and ``EXP``
        (PostgreSQL, DuckDB, BigQuery, Snowflake, SQL Server, SQLite >= 3.35).
        """
        check_is_fitted(self, "base_margin_")
        names = self._names(feature_names)
        nan = getattr(self, "nan_features_", None)
        name = (lambda j: _sql_ident(names[j])) if names is not None else (lambda j: f"x{j}")
        num = lambda v: _fmt(v, precision)  # noqa: E731
        # cuts: the shortest decimal that is the same float32 (the model bins in float32)
        cut = lambda v: np.format_float_positional(np.float32(v), unique=True, trim="-")  # noqa: E731
        trees = self._explain_trees()
        cols, blocks = [], []
        for k, tree in enumerate(trees, 1):
            feats = []
            for f, lft in zip(tree.feature, tree.left):
                if lft != -1 and int(f) not in feats:
                    feats.append(int(f))
            col = f"t{k}_" + "_".join(name(f).strip('"') for f in feats) if feats else f"t{k}"
            col = _sql_ident(col[:60])
            leaves = []

            def walk(node, path, tree=tree, leaves=leaves):
                if tree.left[node] == -1:
                    leaves.append((path, float(tree.value[node])))
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                walk(tree.left[node], [*path, (f, t, True)])
                walk(tree.right[node], [*path, (f, t, False)])

            walk(0, [])
            if len(leaves) == 1:
                blocks.append(f"    {num(leaves[0][1])} AS {col}")
                cols.append(col)
                continue
            lines = [f"    CASE  -- tree {k}"]
            for i, (path, v) in enumerate(leaves):
                bounds = {}
                for f, t, left in path:
                    lo, hi = bounds.setdefault(f, [None, None])
                    if left:
                        bounds[f][1] = t if hi is None else min(hi, t)
                    else:
                        bounds[f][0] = t if lo is None else max(lo, t)
                conds = []
                for f, (lo, hi) in bounds.items():
                    nm = name(f)
                    nan_ok = nan is not None and bool(nan[f])
                    if lo is None:
                        if hi == 0:
                            conds.append(f"{nm} IS NULL")
                            continue
                        c = f"{nm} <= {cut(bin_threshold(self.bin_edges_[f], hi))}"
                        conds.append(f"({c} OR {nm} IS NULL)" if nan_ok else c)
                    elif hi is None:
                        conds.append(f"{nm} IS NOT NULL" if lo == 0 else
                                     f"{nm} > {cut(bin_threshold(self.bin_edges_[f], lo))}")
                    else:
                        if lo != 0:
                            conds.append(f"{nm} > {cut(bin_threshold(self.bin_edges_[f], lo))}")
                        else:
                            conds.append(f"{nm} IS NOT NULL")
                        conds.append(f"{nm} <= {cut(bin_threshold(self.bin_edges_[f], hi))}")
                if i == len(leaves) - 1:
                    lines.append(f"      ELSE {num(v)}")
                else:
                    lines.append(f"      WHEN {' AND '.join(conds)} THEN {num(v)}")
            lines.append(f"    END AS {col}")
            blocks.append("\n".join(lines))
            cols.append(col)
        n_cuts = sum(t.n_splits for t in trees)
        head = (f"-- logit P(y = {self.classes_[1]}) = base + sum of {len(trees)} tree columns "
                f"({n_cuts} cuts); p = 1 / (1 + EXP(-score))")
        keep = "*, " if keep_columns else ""
        return "\n".join([
            head,
            "WITH contributions AS (",
            f"  SELECT {keep}".rstrip(),
            ",\n".join(blocks),
            f"  FROM {table}",
            "), scored AS (",
            f"  SELECT *, {num(float(self.base_margin_))}  -- base",
            *[f"    + {c}" for c in cols],
            "    AS score",
            "  FROM contributions",
            ")",
            "SELECT *, 1.0 / (1.0 + EXP(-score)) AS p",
            "FROM scored;",
        ])

    def explain(self, feature_names=None, precision=4):
        """Text scorecard: add the base and the leaf value of every tree."""
        d = self.to_dict(feature_names, precision)
        n_cuts = sum(t.n_splits for t in self._explain_trees())
        lines = [f"logit P(y = {d['classes'][1]}) = base {d['base_margin']:+.{precision}f}"
                 f" + sum of {len(d['trees'])} trees ({n_cuts} cuts)"]
        for k, tree in enumerate(d["trees"], 1):
            lines.append(f"tree {k}:")
            for r in tree["rules"]:
                cond = " and ".join(r["conditions"]) or "always"
                lines.append(f"  {r['value']:+.{precision}f}  if {cond}")
        return "\n".join(lines)

    # ------------------------------------------------------------ plots

    def shape_functions(self):
        """Main effects: {feature: (values per bin, number of trees)}.

        Sums, per feature, the trees that use only that feature (the shape
        function, as in a GAM/EBM). Bin 0 = missing; bin t >= 1 covers
        ``(edges[t-2], edges[t-1]]``. Trees with more than one feature are
        interactions and are left out (see ``interaction_trees``).
        """
        check_is_fitted(self, "base_margin_")
        out = {}
        for tree in self._explain_trees():
            feats = {int(f) for f, lft in zip(tree.feature, tree.left) if lft != -1}
            if len(feats) != 1:
                continue
            f = feats.pop()
            nb = len(self.bin_edges_[f]) + 2
            Xb = np.zeros((nb, len(self.bin_edges_)), dtype=np.uint8)
            Xb[:, f] = np.arange(nb)
            vals, count = out.get(f, (np.zeros(nb), 0))
            out[f] = (vals + tree.value[tree.leaf_ids(Xb)], count + 1)
        return out

    def interaction_trees(self):
        """Indices of the trees that use more than one feature."""
        return [k for k, t in enumerate(self._explain_trees())
                if len({int(f) for f, lft in zip(t.feature, t.left) if lft != -1}) > 1]

    def plot_shapes(self, feature_names=None, features=None, ncols=3, figsize=None):
        """Shape functions (main effects on the logit scale), one panel per feature."""
        plt = _pyplot()
        shapes = self.shape_functions()
        names = self._names(feature_names)
        feats = sorted(shapes) if features is None else [f for f in features if f in shapes]
        if not feats:
            raise ValueError("No single-feature tree to plot.")
        nrows = -(-len(feats) // ncols)
        fig, axes = plt.subplots(nrows, min(ncols, len(feats)), squeeze=False,
                                 figsize=figsize or (4 * min(ncols, len(feats)), 3 * nrows))
        for ax, f in zip(axes.flat, feats):
            vals, count = shapes[f]
            e = np.asarray(self.bin_edges_[f], dtype=float)
            span = (e[-1] - e[0]) if len(e) > 1 else 1.0
            lo, hi = (e[0] - 0.05 * span, e[-1] + 0.05 * span) if len(e) else (-1.0, 1.0)
            ax.stairs(vals[1:], np.concatenate([[lo], e, [hi]]), baseline=None, linewidth=2)
            if getattr(self, "nan_features_", None) is not None and self.nan_features_[f]:
                ax.axhline(vals[0], linestyle=":", linewidth=1)
                ax.annotate("missing", (hi, vals[0]), ha="right", va="bottom", fontsize=8)
            ax.axhline(0, color="0.6", linewidth=0.8)
            ax.set_title(f"{names[f] if names else f'x{f}'} ({count} trees)", fontsize=10)
            ax.set_ylabel("logit contribution")
        for ax in list(axes.flat)[len(feats):]:
            ax.set_visible(False)
        n_int = len(self.interaction_trees())
        if n_int:
            fig.suptitle(f"main effects; {n_int} interaction tree(s) not shown", fontsize=10)
        fig.tight_layout()
        return fig

    def plot_contributions(self, x, feature_names=None, max_terms=12, ax=None):
        """Waterfall of one prediction: the base plus the leaf ``x`` falls in, per tree."""
        plt = _pyplot()
        if hasattr(x, "to_frame"):  # a pandas row (Series): keep the column names
            x = x.to_frame().T.astype(float)
        elif not hasattr(x, "columns"):
            x = np.asarray(x, dtype=float).reshape(1, -1)
        contrib = self.predict_contributions(x)[0]
        x = predict_input(self, x, "base_margin_")
        trees = self._explain_trees()
        names, nan = self._names(feature_names), getattr(self, "nan_features_", None)
        Xb = rebin(x, self.bin_edges_)
        labels = []
        for tree in trees:
            leaf = int(tree.leaf_ids(Xb)[0])
            rules = leaf_rules(tree, self.bin_edges_, names, nan, precision=4)
            labels.append(" and ".join(rules[leaf_order(tree).index(leaf)][0]) or "always")
        order = np.argsort(-np.abs(contrib), kind="stable")
        keep, rest = order[:max_terms], order[max_terms:]
        steps = [("base", float(self.base_margin_))]
        steps += [(labels[k], float(contrib[k])) for k in keep]
        if len(rest):
            steps.append((f"other {len(rest)} trees", float(contrib[rest].sum())))
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 0.45 * len(steps) + 1.2))
        pos = 0.0
        for i, (_, v) in enumerate(steps):
            start = 0.0 if i == 0 else pos
            ax.barh(i, v, left=start, color="C0" if i == 0 else ("C2" if v >= 0 else "C3"))
            pos = start + v
        ax.set_yticks(range(len(steps)), [s[0] for s in steps], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(pos, color="0.3", linestyle="--", linewidth=1)
        p = 1.0 / (1.0 + np.exp(-pos))
        ax.set_xlabel(f"cumulative logit (final {pos:+.3f}, P = {p:.3f})")
        return ax


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Plotting requires matplotlib: pip install matplotlib") from exc
    return plt
