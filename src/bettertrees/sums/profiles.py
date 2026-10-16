"""Scorecard views of a logit sum of trees: the data marked tree by tree, and the best,
typical and worst client.

``logit P(y = classes_[1]) = base + Σ_k tree_k(x)``, so each tree is a segmentation of
the data and each leaf a number of points. In the credit convention used here
(``scale="points"``) a score is ``offset - factor * logit`` with ``factor = pdo / ln 2``:
``offset`` points at odds 1:1 and every ``pdo`` points halve the odds of the ``bad`` class
(``classes_[1]`` by default; higher score = lower probability of default). Each leaf is
worth ``-factor * value`` points and the base ``offset - factor * base`` (signs flip when
``bad`` is ``classes_[0]``).
``scale="logit"`` keeps the raw logit contributions.

- ``segments(X)``: one column per tree with the rule of the leaf each row falls in,
  one column with its points, the total score and the probability;
- ``profiles(X)``: the best, the modal (most frequent combination of leaves) and the
  worst client, observed in ``X`` or, with ``kind="feasible"``, the best and worst
  combination of leaves that the cuts allow;
- ``plot_profiles(X)`` and ``plot_trees()`` draw them.

These methods work for every bettertrees sum and, through ``as_tree_sum``, for
LightGBM, XGBoost, CatBoost and scikit-learn tree models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .._typing import ArrayLike, FeatureNames
from ._common import bin_threshold, predict_input, rebin


def _num(v, precision=4):
    """A threshold for people: 0 for LightGBM's 1e-35, thousands separators, no exponents."""
    v = float(v)
    if not np.isfinite(v):
        return str(v)
    if abs(v) < 1e-9:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{precision}g}"


def _bad_text(bad_class, bad_label):
    """How the plots name the ``bad`` class: ``bad_label`` if given, else the quoted class."""
    return str(bad_label) if bad_label else f"'{bad_class}'"


class _Scale:
    """Score units; ``sign`` = +1 when the score counts against ``classes_[1]``, -1 when
    against ``classes_[0]``."""

    def __init__(self, scale, offset, pdo, sign=1.0):
        if scale not in ("points", "logit"):
            raise ValueError("scale must be 'points' or 'logit'.")
        if scale == "points" and not pdo > 0:
            raise ValueError("pdo must be positive.")
        self.scale, self.offset, self.sign = scale, float(offset), float(sign)
        self.factor = float(pdo) / np.log(2.0) if scale == "points" else 1.0

    def leaf(self, v):
        v = self.sign * np.asarray(v, dtype=float)
        return -self.factor * v if self.scale == "points" else v

    def base(self, b):
        return (
            self.offset - self.factor * self.sign * b if self.scale == "points" else self.sign * b
        )

    def proba(self, score):
        """Probability of the ``bad`` class."""
        logit = (self.offset - score) / self.factor if self.scale == "points" else score
        return 1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=float)))


def _leaf_boxes(tree, n_bins, nan_features):
    """{leaf: {feature: (lo, hi)}}: the leaf holds bins b with lo < b <= hi."""
    out = {}

    def walk(node, box):
        if tree.left[node] == -1:
            out[node] = box
            return
        f, t = int(tree.feature[node]), int(tree.threshold[node])
        lo, hi = box.get(
            f, (0 if nan_features is not None and not nan_features[f] else -1, n_bins[f] - 1)
        )
        walk(tree.left[node], {**box, f: (lo, min(hi, t))})
        walk(tree.right[node], {**box, f: (max(lo, t), hi)})

    walk(0, {})
    return {k: b for k, b in out.items() if all(lo < hi for lo, hi in b.values())}


class ScorecardMixin:
    """``segments``, ``profiles``, ``plot_profiles`` and ``plot_trees`` (see the module)."""

    # ------------------------------------------------------------ helpers

    def _scale(self, scale, offset, pdo, bad):
        classes = list(self.classes_)
        if bad is None or bad == classes[1]:
            return _Scale(scale, offset, pdo, 1.0), classes[1]
        if bad == classes[0]:
            return _Scale(scale, offset, pdo, -1.0), classes[0]
        raise ValueError(f"bad must be one of the classes {classes}.")

    def _binned_input(self, X):
        Xv = predict_input(self, X, "base_margin_")
        return rebin(Xv, self.bin_edges_, dtype=getattr(self, "input_dtype_", np.float32))

    def _leaf_text(self, feature_names, precision):
        """[{leaf id: rule text}] per tree, with readable numbers (``_num``)."""
        names, nan = self._names(feature_names), getattr(self, "nan_features_", None)
        name = (lambda j: str(names[j])) if names is not None else (lambda j: f"x{j}")
        out = []
        for tree in self._explain_trees():
            texts = {}

            def walk(node, bounds, tree=tree, texts=texts):
                if tree.left[node] == -1:
                    conds = []
                    for f, (lo, hi) in bounds.items():
                        nm, nan_ok = name(f), nan is not None and bool(nan[f])
                        cut = lambda b, f=f: _num(bin_threshold(self.bin_edges_[f], b), precision)
                        if lo is None:
                            if hi == 0:
                                conds.append(f"{nm} is missing")
                            else:
                                conds.append(
                                    f"({nm} <= {cut(hi)} or missing)"
                                    if nan_ok
                                    else f"{nm} <= {cut(hi)}"
                                )
                        elif hi is None:
                            conds.append(f"{nm} is present" if lo == 0 else f"{nm} > {cut(lo)}")
                        else:
                            low = "" if lo == 0 else f"{cut(lo)} < "
                            conds.append(f"{low}{nm} <= {cut(hi)}")
                    texts[node] = " and ".join(conds) or "always"
                    return
                f, t = int(tree.feature[node]), int(tree.threshold[node])
                lo, hi = bounds.get(f, (None, None))
                walk(tree.left[node], {**bounds, f: (lo, t if hi is None else min(hi, t))})
                walk(tree.right[node], {**bounds, f: (t if lo is None else max(lo, t), hi)})

            walk(0, {})
            out.append(texts)
        return out

    def _model_name(self):
        """``source_`` of an imported model (LightGBM, XGBoost, ...), "ITM" for the ITM."""
        name = getattr(self, "source_", None) or type(self).__name__
        return "ITM" if name == "InterleavedTreeClassifier" else name

    def _growth(self):
        """(tree, node) of every cut in the order it was made, and whether that order was recorded.

        The ITM records it while growing (``growth_``); an imported booster grows one tree
        after another. After an edit (prune, drop, ...) the record no longer matches the
        trees and the order falls back to tree by tree, level by level."""
        trees = self._explain_trees()
        cuts = [(k, i) for k, t in enumerate(trees) for i, lft in enumerate(t.left) if lft != -1]
        rec = getattr(self, "growth_", None)
        if rec is not None and sorted(map(tuple, rec)) == sorted(cuts):
            return [tuple(map(int, c)) for c in rec], True
        order = []
        for k, t in enumerate(trees):
            level, nodes = [0], []
            while level:
                nodes += [i for i in level if t.left[i] != -1]
                level = [c for i in level if t.left[i] != -1 for c in (t.left[i], t.right[i])]
            order += [(k, i) for i in nodes]
        return order, False

    def _leaf_ids(self, X):
        Xb = self._binned_input(X)
        return (
            np.column_stack([t.leaf_ids(Xb) for t in self._explain_trees()])
            if self._explain_trees()
            else np.zeros((len(Xb), 0), dtype=np.int64)
        )

    # ------------------------------------------------------------ tables

    def segments(
        self,
        X: ArrayLike,
        feature_names: FeatureNames | None = None,
        *,
        scale: str = "points",
        offset: float = 600.0,
        pdo: float = 20.0,
        bad: Any = None,
        precision: int = 4,
    ) -> Any:
        """The data marked tree by tree (a pandas DataFrame, one row per row of ``X``).

        Columns: ``tree_k`` (the rule of the leaf the row falls in), ``tree_k_points``
        (its points, or ``tree_k_logit`` with ``scale="logit"``), ``base``, ``score`` and
        ``p`` (the probability of ``classes_[1]``). ``base + Σ_k points = score`` exactly.
        """
        import pandas as pd

        check_is_fitted(self, "base_margin_")
        sc, _ = self._scale(scale, offset, pdo, bad)
        ids = self._leaf_ids(X)
        text = self._leaf_text(feature_names, precision)
        suffix = "points" if scale == "points" else "logit"
        cols, base = {}, float(sc.base(float(self.base_margin_)))
        total = np.full(len(ids), base)
        for k, tree in enumerate(self._explain_trees()):
            pts = sc.leaf(tree.value[ids[:, k]])
            cols[f"tree_{k + 1}"] = [text[k][i] for i in ids[:, k]]
            cols[f"tree_{k + 1}_{suffix}"] = pts
            total = total + pts
        out = pd.DataFrame(cols, index=getattr(X, "index", None))
        out.insert(0, "base", base)
        out["score"] = total
        out["p"] = sc.proba(total)
        return out

    def profiles(
        self,
        X: ArrayLike,
        feature_names: FeatureNames | None = None,
        *,
        kind: str = "observed",
        which: Sequence[str] = ("best", "modal", "worst"),
        rows: Sequence[Any] | None = None,
        scale: str = "points",
        offset: float = 600.0,
        pdo: float = 20.0,
        bad: Any = None,
        precision: int = 4,
        max_nodes: int = 2_000_000,
    ) -> Any:
        """Typical and extreme clients (a pandas DataFrame, one row per profile).

        ``which`` picks among ``"best"``, ``"modal"``, ``"median"`` and ``"worst"``; ``rows``
        adds given clients (index labels of ``X``, or positions for an array), named
        ``"client <label>"``.

        - best / worst: with ``kind="observed"`` the rows of ``X`` with the best and the
          worst score (real clients; ``row`` gives their index). With ``kind="feasible"``
          the combinations of leaves with the best and the worst score that the cuts allow
          (no contradictory conditions; a missing value only where training had one), found
          by branch and bound; it may describe a client that ``X`` does not have.
        - modal: the most frequent combination of leaves in ``X`` (``share`` = its fraction
          of rows).
        - median: the row of ``X`` with the median score.

        Columns ``tree_k`` / ``tree_k_points`` as in ``segments``, then ``score`` and ``p``.
        """
        import pandas as pd

        check_is_fitted(self, "base_margin_")
        if kind not in ("observed", "feasible"):
            raise ValueError("kind must be 'observed' or 'feasible'.")
        unknown = set(which) - {"best", "modal", "median", "worst"}
        if unknown:
            raise ValueError(f"unknown profiles {sorted(unknown)}; use best, modal, median, worst.")
        sc, _ = self._scale(scale, offset, pdo, bad)
        trees = self._explain_trees()
        ids = self._leaf_ids(X)
        text = self._leaf_text(feature_names, precision)
        base = float(sc.base(float(self.base_margin_)))
        pts = (
            np.column_stack([sc.leaf(t.value[ids[:, k]]) for k, t in enumerate(trees)])
            if trees
            else np.zeros((len(ids), 0))
        )
        score = base + pts.sum(axis=1)
        order = np.argsort(score, kind="stable")  # ascending
        top_first = scale == "points"  # more points = better client; in logit, lower = better
        picks = {}
        if "modal" in which:
            _, inverse, counts = np.unique(ids, axis=0, return_inverse=True, return_counts=True)
            inverse = np.asarray(inverse).ravel()
            top = int(np.argmax(counts))
            r = int(np.flatnonzero(inverse == top)[0])
            picks["modal"] = (ids[r], r, counts[top] / len(ids))
        if "median" in which:
            r = int(order[(len(order) - 1) // 2])
            picks["median"] = (ids[r], r, None)
        for name, want_best in (("best", True), ("worst", False)):
            if name not in which:
                continue
            if kind == "observed":
                r = int(order[-1] if want_best == top_first else order[0])
                picks[name] = (ids[r], r, None)
            else:
                picks[name] = (self._extreme(sc, want_best, max_nodes), None, None)
        index = getattr(X, "index", None)
        names = [n for n in ("best", "modal", "median", "worst") if n in which]
        for label in rows or ():
            r = int(index.get_loc(label)) if index is not None else int(label)
            picks[f"client {label}"] = (ids[r], r, None)
            names.append(f"client {label}")
        suffix = "points" if scale == "points" else "logit"
        records = []
        for name in names:
            leaves, r, share = picks[name]
            rec = {
                "profile": name,
                "row": None if r is None else (index[r] if index is not None else r),
                "share": share,
                "base": base,
            }
            s = base
            for k, tree in enumerate(trees):
                v = float(sc.leaf(tree.value[int(leaves[k])]))
                rec[f"tree_{k + 1}"] = text[k][int(leaves[k])]
                rec[f"tree_{k + 1}_{suffix}"] = v
                s += v
            rec["score"], rec["p"] = s, float(sc.proba(s))
            records.append(rec)
        return pd.DataFrame(records).set_index("profile")

    def _extreme(self, sc, want_best, max_nodes):
        """Leaf per tree of the best (or worst) feasible combination (branch and bound)."""
        trees = self._explain_trees()
        nb = [len(e) + 2 for e in self.bin_edges_]
        nan = getattr(self, "nan_features_", None)
        # minimise "badness": for points, the best client has the highest score
        sign = -1.0 if (want_best == (sc.scale == "points")) else 1.0
        options = []
        for tree in trees:
            boxes = _leaf_boxes(tree, nb, nan)
            opts = sorted(
                (
                    (sign * float(sc.leaf(tree.value[leaf])), leaf, box)
                    for leaf, box in boxes.items()
                ),
                key=lambda o: o[0],
            )
            options.append(opts)
        order = sorted(range(len(trees)), key=lambda k: -(options[k][-1][0] - options[k][0][0]))
        rest = np.zeros(len(order) + 1)
        for j in range(len(order) - 1, -1, -1):
            rest[j] = rest[j + 1] + options[order[j]][0][0]
        best_val, best_pick, visited = np.inf, None, [0]

        def dfs(j, acc, box, pick):
            nonlocal best_val, best_pick
            visited[0] += 1
            if visited[0] > max_nodes:
                raise ValueError(
                    "feasible search too large; use kind='observed' or raise max_nodes."
                )
            if acc + rest[j] >= best_val:
                return
            if j == len(order):
                best_val, best_pick = acc, dict(pick)
                return
            k = order[j]
            for val, leaf, lbox in options[k]:
                if acc + val + rest[j + 1] >= best_val:
                    break
                merged, ok = dict(box), True
                for f, (lo, hi) in lbox.items():
                    a, b = merged.get(f, (-1, nb[f] - 1))
                    a, b = max(a, lo), min(b, hi)
                    if a >= b:
                        ok = False
                        break
                    merged[f] = (a, b)
                if ok:
                    pick[k] = leaf
                    dfs(j + 1, acc + val, merged, pick)
                    del pick[k]

        dfs(0, 0.0, {}, {})
        if best_pick is None:  # pragma: no cover - the leaves of a real row are always feasible
            raise ValueError("no feasible combination of leaves.")
        return np.array([best_pick[k] for k in range(len(trees))], dtype=np.int64)

    # ------------------------------------------------------------ plots

    def plot_profiles(
        self,
        X: ArrayLike,
        feature_names: FeatureNames | None = None,
        *,
        kind: str = "observed",
        which: Sequence[str] = ("best", "modal", "worst"),
        rows: Sequence[Any] | None = None,
        scale: str = "points",
        offset: float = 600.0,
        pdo: float = 20.0,
        bad: Any = None,
        bad_label: str | None = None,
        max_trees: int = 10,
        title: str | None = None,
        precision: int = 4,
    ) -> Any:
        """Score distribution of ``X`` and, tree by tree, the profiles of ``profiles``.

        One column per profile (``which``, then ``rows``), starting from the base and adding
        one leaf per tree. With more than ``max_trees`` trees, the ones whose leaves differ
        most among the profiles are shown and the rest are summed in one line. ``bad_label``
        names the ``bad`` class in the text (e.g. ``"default"``).
        """
        plt = _pyplot()
        seg = self.segments(
            X, feature_names, scale=scale, offset=offset, pdo=pdo, bad=bad, precision=precision
        )
        prof = self.profiles(
            X,
            feature_names,
            kind=kind,
            which=which,
            rows=rows,
            scale=scale,
            offset=offset,
            pdo=pdo,
            bad=bad,
            precision=precision,
        )
        bad_text = _bad_text(self._scale(scale, offset, pdo, bad)[1], bad_label)
        suffix = "points" if scale == "points" else "logit"
        trees = self._explain_trees()
        n_trees, n_cuts = len(trees), sum(t.n_splits for t in trees)
        names = list(prof.index)
        spread = [
            prof[f"tree_{k}_{suffix}"].max() - prof[f"tree_{k}_{suffix}"].min()
            for k in range(1, n_trees + 1)
        ]
        shown = (
            sorted(np.argsort(-np.asarray(spread), kind="stable")[:max_trees]) if n_trees else []
        )
        hidden = [k for k in range(n_trees) if k not in set(shown)]
        n_rows = 1 + len(shown) + (1 if hidden else 0)
        nd = 0 if scale == "points" else 2
        printed = {}  # the total of the rounded terms, so every column adds up as printed
        for name in names:
            terms = [prof.loc[name, "base"]] + [
                prof.loc[name, f"tree_{k + 1}_{suffix}"] for k in shown
            ]
            if hidden:
                terms.append(sum(prof.loc[name, f"tree_{k + 1}_{suffix}"] for k in hidden))
            printed[name] = sum(round(float(v), nd) for v in terms)
        width = max(19.5, 7.0 + 4.3 * len(names))
        fig = plt.figure(figsize=(width, 3.2 + 0.62 * max(n_rows, 6)))
        fig.patch.set_facecolor("white")
        h = fig.get_figheight()
        top = 1 - 1.15 / h
        fixed = {"best": "#1d4ed8", "modal": "#1f2430", "median": "#7c3aed", "worst": "#b91c1c"}
        colors = {n: fixed.get(n, "#0f766e") for n in names}
        fig.text(
            0.03,
            1 - 0.45 / h,
            title or f"{self._model_name()}: {n_trees} trees, {n_cuts} cuts",
            fontsize=16,
            fontweight="bold",
            color="#1f2430",
        )
        unit = (
            f"score: {offset:g} points at odds 1:1, every {pdo:g} points halve the odds of "
            f"{bad_text}"
            if scale == "points"
            else f"logit of P({bad_text})"
        )
        notes = []
        if {"best", "worst"} & set(names):
            where = "rows of the data" if kind == "observed" else "feasible combinations of leaves"
            notes.append(f"best and worst = {where}")
        if "modal" in names:
            notes.append(
                f"modal = the most frequent combination of leaves "
                f"({prof.loc['modal', 'share']:.1%} of rows)"
            )
        if "median" in names:
            notes.append("median = the row with the median score")
        fig.text(0.03, 1 - 0.8 / h, "; ".join(notes) + f". {unit}.", fontsize=10, color="#6b7280")
        hist_w = 0.21 * 19.5 / width
        ax = fig.add_axes([0.03, 0.09, hist_w, top - 0.17])
        ax.hist(seg["score"], bins=50, color="#cbd5e1")
        ymax = ax.get_ylim()[1]
        for j, name in enumerate(names):
            s = printed[name]
            ax.axvline(prof.loc[name, "score"], color=colors[name], lw=2)
            ax.text(
                s,
                ymax * (0.98 - 0.13 * (j % 5)),
                f" {name}\n {s:.0f}" if scale == "points" else f" {name}\n {s:+.2f}",
                color=colors[name],
                va="top",
                fontsize=9,
            )
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("score" if scale == "points" else "logit", fontsize=9)
        ax.set_title("score distribution", loc="left", fontsize=10, color="#6b7280")
        x0 = 0.06 + hist_w
        colw = (0.985 - x0 - 0.07) / max(len(names), 1)
        dy = (top - 0.13) / (n_rows + 1)
        y = top - 0.02
        fig.text(x0, y, "tree", fontsize=10, color="#6b7280", va="top")
        for j, name in enumerate(names):
            head = name if name.startswith("client ") else f"{name} client"
            fig.text(
                x0 + 0.07 + j * colw,
                y,
                head,
                fontsize=11,
                fontweight="bold",
                color=colors[name],
                va="top",
            )
        fmt = (lambda v: f"{v:+.0f}") if scale == "points" else (lambda v: f"{v:+.2f}")
        good = (lambda v: v > 0) if scale == "points" else (lambda v: v < 0)
        fnames = self._names(feature_names)
        y -= dy
        fig.text(x0, y, "base", fontsize=10, fontweight="bold", color="#1f2430", va="top")
        fig.text(x0, y - 0.022, "same for all", fontsize=7, color="#6b7280", va="top")
        for j, name in enumerate(names):
            b = prof.loc[name, "base"]
            fig.text(
                x0 + 0.07 + j * colw,
                y,
                f"{b:.0f}" if scale == "points" else f"{b:+.2f}",
                fontsize=11,
                fontweight="bold",
                color="#1f2430",
                va="top",
            )
        for k in shown:
            y -= dy
            tree = trees[k]
            feats = sorted({int(f) for f, lft in zip(tree.feature, tree.left) if lft != -1})
            label = "\n".join((str(fnames[f]) if fnames else f"x{f}")[:22] for f in feats[:3])
            label += "\n..." if len(feats) > 3 else ""
            fig.text(x0, y, f"{k + 1}", fontsize=10, fontweight="bold", color="#1f2430", va="top")
            fig.text(x0, y - 0.022, label, fontsize=7, color="#6b7280", va="top")
            for j, name in enumerate(names):
                v = prof.loc[name, f"tree_{k + 1}_{suffix}"]
                fig.text(
                    x0 + 0.07 + j * colw,
                    y,
                    fmt(v),
                    fontsize=11,
                    fontweight="bold",
                    color="#1d4ed8" if good(v) else "#b91c1c",
                    va="top",
                )
                rule = prof.loc[name, f"tree_{k + 1}"].replace(" and ", "\n")
                fig.text(x0 + 0.105 + j * colw, y, rule, fontsize=8.5, color="#1f2430", va="top")
        if hidden:
            y -= dy
            fig.text(x0, y, f"other {len(hidden)} trees", fontsize=9, color="#6b7280", va="top")
            for j, name in enumerate(names):
                v = sum(prof.loc[name, f"tree_{k + 1}_{suffix}"] for k in hidden)
                fig.text(
                    x0 + 0.07 + j * colw,
                    y,
                    fmt(v),
                    fontsize=11,
                    fontweight="bold",
                    color="#1d4ed8" if good(v) else "#b91c1c",
                    va="top",
                )
        y -= dy * 0.9
        for j, name in enumerate(names):
            s, p = printed[name], prof.loc[name, "p"]
            txt = f"= score {s:.0f}" if scale == "points" else f"= logit {s:+.2f}"
            fig.text(
                x0 + 0.07 + j * colw,
                y,
                f"{txt}  ·  P({bad_text}) = {p:.1%}",
                fontsize=11,
                fontweight="bold",
                color="#1f2430",
                va="top",
            )
        return fig

    def plot_trees(
        self,
        x: ArrayLike | None = None,
        feature_names: FeatureNames | None = None,
        *,
        X: ArrayLike | None = None,
        scale: str = "points",
        offset: float = 600.0,
        pdo: float = 20.0,
        bad: Any = None,
        bad_label: str | None = None,
        max_trees: int = 12,
        ncols: int = 4,
        precision: int = 4,
        title: str | None = None,
    ) -> Any:
        """The trees side by side (``+`` between them), leaves in the units of ``scale``.

        Leaves are blue when they lower the risk of ``bad`` and red when they raise it.
        ``X`` adds the share of rows in each leaf; ``x`` (one row) draws its path in every
        tree, writes its value under each cut it passes and the sum ``base + leaves`` at the
        bottom. With more than ``max_trees`` trees, the ones with the widest range of leaf
        values are drawn. A line under the title says what each mark means.
        """
        plt = _pyplot()
        from matplotlib import colors as mcolors

        check_is_fitted(self, "base_margin_")
        sc, bad_class = self._scale(scale, offset, pdo, bad)
        bad_text = _bad_text(bad_class, bad_label)
        points = scale == "points"
        fmt = (lambda v: f"{v:+.0f}") if points else (lambda v: f"{v:+.2f}")
        growth, recorded = self._growth()
        step = {cut: j + 1 for j, cut in enumerate(growth)}
        sign = sc.sign
        trees = self._explain_trees()
        names = self._names(feature_names)
        rng = [
            float(np.ptp(t.value[[i for i, lft in enumerate(t.left) if lft == -1]])) for t in trees
        ]
        shown = sorted(np.argsort(-np.asarray(rng), kind="stable")[:max_trees]) if trees else []
        share = None
        if X is not None:
            ids = self._leaf_ids(X)
            share = [
                np.bincount(ids[:, k], minlength=len(t.left)) / len(ids)
                for k, t in enumerate(trees)
            ]
        path_nodes, contrib, xraw = None, None, None
        if x is not None:
            if hasattr(x, "to_frame"):
                x = x.to_frame().T
            elif not hasattr(x, "columns"):
                x = np.asarray(x, dtype=float).reshape(1, -1)
            Xb = self._binned_input(x)
            xraw = predict_input(self, x, "base_margin_")[0]
            path_nodes = []
            for t in trees:
                node, nodes = 0, {0}
                while t.left[node] != -1:
                    node = (
                        t.left[node]
                        if Xb[0, t.feature[node]] <= t.threshold[node]
                        else t.right[node]
                    )
                    nodes.add(node)
                path_nodes.append(nodes)
            contrib = sc.leaf(self.predict_contributions(x)[0])
        leaf_vals = (
            sign
            * np.concatenate(
                [t.value[[i for i, lft in enumerate(t.left) if lft == -1]] for t in trees]
            )
            if trees
            else np.zeros(1)
        )
        vmax = max(float(np.abs(leaf_vals).max()), 1e-9)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r")
        nrow = max(1, -(-len(shown) // ncols))
        fig, axes = plt.subplots(
            nrow, ncols, figsize=(4.6 * ncols, 3.4 * nrow + 1.2), squeeze=False
        )
        fig.patch.set_facecolor("white")
        ink, edge, muted = "#1f2430", "#c4c8cf", "#6b7280"
        for slot, k in enumerate(shown):
            t, ax = trees[k], axes[slot // ncols, slot % ncols]
            pos, nxt = {}, [0]

            def place(i, d, t=t, pos=pos, nxt=nxt):
                if t.left[i] == -1:
                    pos[i] = (nxt[0], d)
                    nxt[0] += 1
                else:
                    place(t.left[i], d + 1)
                    place(t.right[i], d + 1)
                    pos[i] = ((pos[t.left[i]][0] + pos[t.right[i]][0]) / 2, d)

            place(0, 0)
            depth = max(d for _, d in pos.values())
            ax.set_xlim(-0.8, max(nxt[0], 3) - 0.2)
            ax.set_ylim(-depth - 0.7, 0.6)
            ax.axis("off")
            on = path_nodes[k] if path_nodes is not None else set()
            for i, (px, d) in pos.items():
                if t.left[i] == -1:
                    continue
                for c, lab in ((t.left[i], "yes"), (t.right[i], "no")):
                    cx, cd = pos[c]
                    hot = i in on and c in on
                    ax.plot(
                        [px, cx],
                        [-d, -cd],
                        color=ink if hot else edge,
                        lw=2.4 if hot else 1.3,
                        zorder=1,
                    )
                    ax.text(
                        (px + cx) / 2 + (-0.12 if cx < px else 0.12),
                        -(d + cd) / 2,
                        lab,
                        fontsize=7,
                        color=muted,
                        ha="center",
                        va="center",
                    )
            for i, (px, d) in pos.items():
                if t.left[i] != -1:
                    f = int(t.feature[i])
                    nm = str(names[f]) if names else f"x{f}"
                    thr = int(t.threshold[i])
                    txt = (
                        f"{nm} is missing"
                        if thr == 0
                        else f"{nm} ≤ {_num(bin_threshold(self.bin_edges_[f], thr), precision)}"
                    )
                    ax.text(
                        px,
                        -d,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=ink,
                        zorder=3,
                        bbox=dict(
                            boxstyle="round,pad=0.35",
                            fc="white",
                            ec=ink if i in on else edge,
                            lw=1.6 if i in on else 1,
                        ),
                    )
                    if xraw is not None and i in on:  # the client's value at this cut
                        v = xraw[f]
                        ax.text(
                            px,
                            -d - 0.3,
                            "missing" if np.isnan(v) else f"= {_num(v, precision)}",
                            ha="center",
                            va="top",
                            fontsize=7.5,
                            color="#0f766e",
                            fontweight="bold",
                            zorder=4,
                        )
                    if recorded:  # the cut's place in the order of growth
                        ax.text(
                            px,
                            -d + 0.27,
                            f"#{step[(k, i)]}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="#b45309",
                            fontweight="bold",
                            zorder=4,
                        )
                else:
                    v = sign * float(t.value[i])  # risk direction, for the colour
                    dark = abs(norm(v) - 0.5) > 0.3
                    ax.text(
                        px,
                        -d,
                        fmt(float(sc.leaf(t.value[i]))),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        zorder=3,
                        color="white" if dark else ink,
                        fontweight="bold" if i in on else None,
                        bbox=dict(
                            boxstyle="round,pad=0.35",
                            fc=cmap(norm(v)),
                            ec=ink if i in on else "none",
                            lw=2 if i in on else 0,
                        ),
                    )
                    if share is not None:
                        ax.text(
                            px,
                            -d - 0.33,
                            f"{share[k][i]:.0%}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=muted,
                        )
            ax.set_title(
                f"tree {k + 1} · {t.n_splits} cut{'s' if t.n_splits != 1 else ''}",
                fontsize=9,
                color=muted,
                loc="left",
            )
            if slot % ncols:
                ax.text(-0.85, 0, "+", fontsize=20, color=muted, ha="center", va="center")
        for slot in range(len(shown), nrow * ncols):
            axes[slot // ncols, slot % ncols].axis("off")
        b0 = float(sc.base(float(self.base_margin_)))
        n_cuts = sum(t.n_splits for t in trees)
        more = (
            f" ({len(trees) - len(shown)} more trees not drawn)" if len(shown) < len(trees) else ""
        )
        H = fig.get_figheight()
        fig.text(
            0.01,
            1 - 0.12 / H,
            title or f"{self._model_name()}: {len(trees)} trees, {n_cuts} cuts{more}",
            fontsize=12,
            fontweight="bold",
            color=ink,
            va="top",
        )
        how = (
            f"score = {b0:.0f} (base) + one leaf per tree, in points; every {pdo:g} points "
            f"halve the odds of {bad_text}"
            if points
            else f"logit P({bad_text}) = {b0:+.2f} (base) + one leaf per tree"
        )
        marks = [how, "blue leaves lower the risk, red raise it"]
        if share is not None:
            marks.append("% = share of rows in the leaf")
        if recorded:
            marks.append("#k = the order in which the cut was made")
        if contrib is not None:
            marks.append("bold = this client's path, = v its value at the cut")
        fig.text(0.01, 1 - 0.42 / H, "  ·  ".join(marks), fontsize=9, color=muted, va="top")
        if contrib is not None:
            s = b0 + float(contrib.sum())
            parts = "".join(f" {fmt(v)}" for v in contrib[:20]) + (
                " ..." if len(contrib) > 20 else ""
            )
            nd = 0 if points else 2  # the total of the rounded terms, so the line adds up
            shown_s = round(b0, nd) + sum(round(float(v), nd) for v in contrib)
            unit = f"{shown_s:.0f} points" if points else f"{shown_s:+.2f}"
            base = f"{b0:.0f}" if points else f"{b0:+.2f}"
            fig.text(
                0.01,
                0.01,
                f"this client: {base}{parts} = {unit}  ->  P({bad_text}) = {sc.proba(s):.1%}",
                fontsize=10.5,
                color=ink,
                fontweight="bold",
            )
        fig.tight_layout(rect=(0, 0.035, 1, 1 - 0.7 / H))
        return fig

    def plot_growth(
        self,
        feature_names: FeatureNames | None = None,
        *,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """The order in which the cuts were made: cut number (x) against the tree it went to (y).

        Boosting grows one tree after another (a staircase); the ITM chooses, at every cut,
        any tree or a new one (the line jumps between trees). A hollow marker opens a new
        tree, a filled one deepens an existing tree. Models without a recorded order (after
        an edit) are drawn tree by tree, level by level, and say so in the title.
        """
        plt = _pyplot()
        check_is_fitted(self, "base_margin_")
        growth, recorded = self._growth()
        trees, names = self._explain_trees(), self._names(feature_names)
        if ax is None:
            _, ax = plt.subplots(
                figsize=(max(6.0, 0.42 * len(growth) + 2), 0.55 * len(trees) + 1.8)
            )
        xs = np.arange(1, len(growth) + 1)
        ys = np.array([k + 1 for k, _ in growth])
        ax.plot(xs, ys, color="#cbd5e1", lw=1.5, zorder=1)
        seen = set()
        for x, (k, node) in zip(xs, growth):
            new = k not in seen
            seen.add(k)
            ax.scatter(
                [x],
                [k + 1],
                s=90,
                zorder=3,
                facecolors="white" if new else "#1f2430",
                edgecolors="#1f2430",
                linewidths=1.6,
            )
            f = int(trees[k].feature[node])
            ax.annotate(
                str(names[f]) if names else f"x{f}",
                (x, k + 1),
                xytext=(0, 9),
                textcoords="offset points",
                ha="left",
                va="bottom",
                rotation=35,
                fontsize=7,
                color="#6b7280",
            )
        ax.set_xlim(0.3, len(growth) + 1.2)
        ax.set_ylim(len(trees) + 0.8, 0.4)
        ax.set_yticks(range(1, len(trees) + 1), [f"tree {k}" for k in range(1, len(trees) + 1)])
        ax.set_xticks(xs)
        ax.set_xlabel("cut number (order of growth)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        n_switch = int(np.sum(ys[1:] != ys[:-1]))
        if not recorded:
            how = "order not recorded: drawn tree by tree"
        elif n_switch <= len(trees) - 1:
            how = "one tree at a time"
        else:
            how = f"cuts interleaved across trees ({n_switch} switches)"
        ax.set_title(
            title or f"{self._model_name()}: {len(growth)} cuts in {len(trees)} trees, {how}",
            loc="left",
            fontsize=10,
        )
        style = dict(marker="o", ls="none", ms=8, mec="#1f2430", mew=1.6)
        ax.legend(
            handles=[
                ax.plot([], [], mfc="white", label="opens a new tree", **style)[0],
                ax.plot([], [], mfc="#1f2430", label="deepens a tree", **style)[0],
            ],
            loc="lower left",
            fontsize=8,
            frameon=False,
        )
        return ax


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Plotting requires matplotlib: pip install matplotlib") from exc
    return plt
