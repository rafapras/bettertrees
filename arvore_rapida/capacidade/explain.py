"""Forma interpretável das somas de árvores em logit.

Toda soma (``SumOfOptimalTrees``, ``FIGSClassifier``, ``BoostedOptimalTrees``,
``AdditiveTreeBooster``) é ``logit(p) = base + Σ_k árvore_k(x)``, e cada
árvore é uma lista de folhas "condições → valor". O mixin daqui dá a todas a
mesma saída:

- ``rules()``: (árvore, condições, valor no logit) por folha, com as condições
  de um caminho fundidas em intervalos por feature (``3 < x <= 7``);
- ``predict_contributions(X)``: contribuição de cada árvore por linha; somada à
  base, reproduz ``decision_function`` exatamente;
- ``explain()``: scorecard em texto; ``to_dict()``: o mesmo em JSON;
- ``get_trees()``: cada árvore em arrays (como ``tree_`` do sklearn), com os
  cortes em valores reais; ``export_text()``: as árvores desenhadas em texto.

Convenção dos cortes (bins): ``x <= v`` inclui NaN (NaN vai sempre à
esquerda); ``x > v`` exclui. Um corte no bin 0 separa só "ausente × presente".
A anotação "ou ausente" aparece só nas features que tinham NaN no treino.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._common import bin_threshold, predict_input, rebin


def feature_names_of(X):
    """Nomes das colunas (DataFrame com nomes em texto) ou None."""
    cols = getattr(X, "columns", None)
    if cols is None:
        return None
    names = np.asarray(cols, dtype=object)
    return names if all(isinstance(c, str) for c in names) else None


def _fmt(v, precision):
    return f"{v:.{precision}g}"


def leaf_rules(tree, edges, names=None, nan_features=None, precision=6):
    """Folhas de ``tree`` como (condições em intervalos, valor)."""
    name = (lambda j: str(names[j])) if names is not None else (lambda j: f"x{j}")
    out = []

    def conditions(path):
        bounds = {}  # feature -> [maior bin à direita (lo), menor bin à esquerda (hi)]
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
            if lo is None:  # só cortes à esquerda: NaN incluído
                if hi == 0:
                    conds.append(f"{nm} ausente")
                    continue
                txt = f"{nm} <= {_fmt(bin_threshold(edges[f], hi), precision)}"
                conds.append(f"({txt} ou ausente)" if nan_ok else txt)
            elif hi is None:
                conds.append(f"{nm} presente" if lo == 0 else
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


class InterpretableSumMixin:
    """Saída interpretável comum; a classe define ``_explain_trees()``."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False  # binário (v1)
        tags.input_tags.allow_nan = True  # NaN tem bin próprio e vai sempre à esquerda
        return tags

    def _explain_trees(self):
        return self.trees_

    # ------------------------------------------------------------ previsão

    def predict_proba(self, X):
        """Probabilidades das duas classes.

        Returns
        -------
        ndarray, shape (n, 2)
            Colunas na ordem de ``classes_``; ``[:, 1] = sigmoid(decision_function(X))``.
        """
        p = 0.5 * (1.0 + np.tanh(0.5 * self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Classe mais provável (``classes_[1]`` quando o logit é positivo)."""
        margin = self.decision_function(X)  # valida o ajuste antes de ler classes_
        return self.classes_[(margin > 0).astype(int)]

    @property
    def n_splits_(self):
        """Total de cortes (nós internos) somado entre as árvores."""
        return sum(t.n_splits for t in self._explain_trees())

    # ------------------------------------------------------------ árvores

    def get_trees(self, feature_names=None):
        """As árvores da soma em arrays, no formato do ``tree_`` do sklearn.

        Cada árvore é um dict com arrays por nó: ``feature`` (-1 nas folhas),
        ``feature_name``, ``threshold`` (valor real: ``x <= threshold`` vai à
        esquerda, NaN sempre à esquerda; ``-inf`` separa "ausente" de
        "presente"), ``children_left``/``children_right`` (-1 nas folhas) e
        ``value`` (contribuição no logit; só as folhas entram na soma).
        ``logit(p) = base_margin_ + Σ_k value_k[folha de x na árvore k]``.
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

    def export_text(self, feature_names=None, precision=4):
        """As árvores desenhadas em texto (no estilo de ``sklearn.tree.export_text``)."""
        lines = [f"base (logit): {float(self.base_margin_):+.{precision}f}"]
        for k, t in enumerate(self.get_trees(feature_names), 1):
            lines.append(f"árvore {k}")

            def walk(node, depth, t=t):
                pad = "|   " * depth
                if t["children_left"][node] == -1:
                    lines.append(f"{pad}|--- valor: {t['value'][node]:+.{precision}f}")
                    return
                name, thr = t["feature_name"][node], t["threshold"][node]
                if np.isneginf(thr):
                    left, right = f"{name} ausente", f"{name} presente"
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
        """[(árvore, condições, valor no logit)] de todas as folhas."""
        check_is_fitted(self, "base_margin_")
        names, nan = self._names(feature_names), getattr(self, "nan_features_", None)
        return [(k, conds, v) for k, tree in enumerate(self._explain_trees())
                for conds, v in leaf_rules(tree, self.bin_edges_, names, nan, precision)]

    def predict_contributions(self, X):
        """Matriz (n, n_árvores) de contribuições no logit; base + soma das
        colunas = ``decision_function(X)``."""
        Xb = rebin(predict_input(self, X, "base_margin_"), self.bin_edges_)
        trees = self._explain_trees()
        out = np.zeros((len(Xb), len(trees)))
        for k, tree in enumerate(trees):
            out[:, k] = tree.value[tree.leaf_ids(Xb)]
        return out

    def to_dict(self, feature_names=None, precision=6):
        """Modelo inteiro em estrutura JSON (base + árvores de regras)."""
        check_is_fitted(self, "base_margin_")
        trees = {}
        for k, conds, v in self.rules(feature_names, precision):
            trees.setdefault(k, []).append(dict(conditions=conds, value=v))
        return dict(link="logit", base_margin=float(self.base_margin_),
                    classes=[c.item() if hasattr(c, "item") else c for c in self.classes_],
                    trees=[dict(rules=trees[k]) for k in sorted(trees)])

    def explain(self, feature_names=None, precision=4):
        """Scorecard em texto: some a base e o valor da folha de cada árvore."""
        d = self.to_dict(feature_names, precision)
        n_cuts = sum(t.n_splits for t in self._explain_trees())
        lines = [f"logit P(y = {d['classes'][1]}) = base {d['base_margin']:+.{precision}f}"
                 f" + soma de {len(d['trees'])} árvores ({n_cuts} cortes)"]
        for k, tree in enumerate(d["trees"], 1):
            lines.append(f"árvore {k}:")
            for r in tree["rules"]:
                cond = " e ".join(r["conditions"]) or "sempre"
                lines.append(f"  {r['value']:+.{precision}f}  se {cond}")
        return "\n".join(lines)

    # ------------------------------------------------------------ gráficos

    def shape_functions(self):
        """Efeitos principais: {feature: (valores por bin, contagem de árvores)}.

        Soma, por feature, as árvores que só usam essa feature (a função de
        forma, como num GAM/EBM). Bin 0 = ausente; bin t >= 1 cobre
        ``(edges[t-2], edges[t-1]]``. Árvores com mais de uma feature são
        interações e ficam de fora (ver ``interaction_trees``).
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
        """Índices das árvores que usam mais de uma feature."""
        return [k for k, t in enumerate(self._explain_trees())
                if len({int(f) for f, lft in zip(t.feature, t.left) if lft != -1}) > 1]

    def plot_shapes(self, feature_names=None, features=None, ncols=3, figsize=None):
        """Funções de forma (efeitos principais no logit), uma por feature."""
        plt = _pyplot()
        shapes = self.shape_functions()
        names = self._names(feature_names)
        feats = sorted(shapes) if features is None else [f for f in features if f in shapes]
        if not feats:
            raise ValueError("nenhuma árvore de uma feature só para desenhar.")
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
                ax.annotate("ausente", (hi, vals[0]), ha="right", va="bottom", fontsize=8)
            ax.axhline(0, color="0.6", linewidth=0.8)
            ax.set_title(f"{names[f] if names else f'x{f}'} ({count} árv.)", fontsize=10)
            ax.set_ylabel("contribuição no logit")
        for ax in list(axes.flat)[len(feats):]:
            ax.set_visible(False)
        n_int = len(self.interaction_trees())
        if n_int:
            fig.suptitle(f"efeitos principais; {n_int} árvore(s) de interação fora do gráfico",
                         fontsize=10)
        fig.tight_layout()
        return fig

    def plot_contributions(self, x, feature_names=None, max_terms=12, ax=None):
        """Waterfall de uma previsão: base + a folha de cada árvore em que ``x`` cai."""
        plt = _pyplot()
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
            leaves = [k for k, lft in enumerate(tree.left) if lft == -1]
            labels.append(" e ".join(rules[leaves.index(leaf)][0]) or "sempre")
        order = np.argsort(-np.abs(contrib), kind="stable")
        keep, rest = order[:max_terms], order[max_terms:]
        steps = [("base", float(self.base_margin_))]
        steps += [(labels[k], float(contrib[k])) for k in keep]
        if len(rest):
            steps.append((f"outras {len(rest)} árvores", float(contrib[rest].sum())))
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
        ax.set_xlabel(f"logit acumulado (final {pos:+.3f}, P = {p:.3f})")
        return ax


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError("os gráficos precisam do matplotlib: pip install matplotlib") from exc
    return plt
