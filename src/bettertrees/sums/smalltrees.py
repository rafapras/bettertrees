"""Somas de poucas árvores pequenas em logit: K árvores ótimas e FIGS.

Representação comum (``SmallTree``): arrays por nó (feature, threshold em
bins, left, right, value); folha tem left = right = -1. ``x <= t`` (em bins)
vai à esquerda, NaN (bin 0) sempre à esquerda. O custo de complexidade é o
número de cortes (nós internos), somado entre árvores.

- ``SumOfOptimalTrees``: K árvores Newton **ótimas** de profundidade d ∈ {1,2,3}
  ajustadas em sequência no resíduo, depois ``backfit_sweeps`` passadas de
  reestimação conjunta das folhas (cada árvore contra a margem das outras).
- ``FIGSClassifier``: Tan et al. (2022) em Newton/logit: a cada passo, o
  melhor corte em qualquer folha de qualquer árvore, ou a raiz de uma árvore
  nova, contra a margem das outras árvores; para em ``max_splits`` cortes.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ._common import (
    base_margin,
    binned,
    fit_inputs,
    grad_hess,
    predict_input,
    rebin,
    sigmoid,
    teacher_top_features,
    top_features,
)
from ._kernels import (
    best_cut_1d,
    best_depth2,
    best_depth3,
    hist_1d,
    newton_leaf_values,
    node_hist,
    small_tree_leaf_ids,
)
from .explain import InterpretableSumMixin, leaf_rules


@dataclass
class SmallTree:
    feature: list = field(default_factory=lambda: [-1])
    threshold: list = field(default_factory=lambda: [-1])
    left: list = field(default_factory=lambda: [-1])
    right: list = field(default_factory=lambda: [-1])
    value: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def split(self, node, f, t):
        """Transforme a folha ``node`` em corte; os filhos herdam o valor dela."""
        k = len(self.feature)
        self.feature += [-1, -1]
        self.threshold += [-1, -1]
        self.left += [-1, -1]
        self.right += [-1, -1]
        self.feature[node], self.threshold[node] = int(f), int(t)
        self.left[node], self.right[node] = k, k + 1
        self.value = np.append(self.value, [self.value[node], self.value[node]])
        return k, k + 1

    def arrays(self):
        return (np.array(self.feature, dtype=np.int64), np.array(self.threshold, dtype=np.int64),
                np.array(self.left, dtype=np.int64), np.array(self.right, dtype=np.int64))

    def leaf_ids(self, Xb):
        return small_tree_leaf_ids(Xb, *self.arrays())

    @property
    def n_splits(self):
        return sum(1 for v in self.left if v != -1)

    @property
    def leaves(self):
        return [k for k, v in enumerate(self.left) if v == -1]

    @classmethod
    def from_nested(cls, nested):
        """(f, t, esq, dir) aninhado, None = folha."""
        tree = cls()

        def grow(node, spec):
            if spec is None:
                return
            left, right = tree.split(node, spec[0], spec[1])
            grow(left, spec[2])
            grow(right, spec[3])

        grow(0, nested)
        return tree

    def rules(self, edges, names=None, nan_features=None):
        """Folhas como (condições em intervalos por feature, valor); ver ``explain``."""
        return leaf_rules(self, edges, names, nan_features)


def _side(f, t):
    return None if f < 0 else (int(f), int(t), None, None)


def _d2_nested(s):
    """Lado d2 (f2, t2, fl, tl, fr, tr) → aninhado."""
    if s[0] < 0:
        return None
    return (int(s[0]), int(s[1]), _side(s[2], s[3]), _side(s[4], s[5]))


def optimal_tree(Xb, g, h, w, nb, depth, lam, min_weight, features=None):
    """Árvore Newton ótima de profundidade 1–3 (sem valores); None se nada ganha.

    ``features`` restringe a busca (obrigatório na prática para d3 com p grande).
    """
    feats = np.arange(Xb.shape[1]) if features is None else np.asarray(features, dtype=np.int64)
    Xs = np.ascontiguousarray(Xb[:, feats])
    nbs = np.ascontiguousarray(nb[feats])

    def remap(spec):
        if spec is None:
            return None
        return (int(feats[spec[0]]), spec[1], remap(spec[2]), remap(spec[3]))

    if depth == 1:
        gains, cuts = best_cut_1d(hist_1d(Xs, g, h, w, int(nbs.max())), nbs, lam, min_weight)
        f = int(np.argmax(gains))
        spec = (f, int(cuts[f]), None, None) if gains[f] > 0 else None
    elif depth == 2:
        gains, res = best_depth2(Xs, g, h, w, nbs, lam, min_weight)
        f = int(np.argmax(gains))
        spec = (None if gains[f] <= 0 else
                (f, int(res[f, 0]), _side(res[f, 1], res[f, 2]), _side(res[f, 3], res[f, 4])))
    elif depth == 3:
        gains, res = best_depth3(Xs, g, h, w, nbs, lam, min_weight)
        f = int(np.argmax(gains))
        spec = (None if gains[f] <= 0 else
                (f, int(res[f, 0]), _d2_nested(res[f, 1:7]), _d2_nested(res[f, 7:13])))
    else:
        raise ValueError("depth must be 1, 2 or 3.")
    return None if spec is None else SmallTree.from_nested(remap(spec))


def greedy_tree(Xb, g, h, w, nb, depth, lam, min_weight, features=None):
    """Controle guloso de ``optimal_tree``: melhor corte único na raiz, depois em
    cada filho (mesmo ganho Newton, mesmos bins); None se nada ganha."""
    feats = np.arange(Xb.shape[1]) if features is None else np.asarray(features, dtype=np.int64)
    Xs = np.ascontiguousarray(Xb[:, feats])
    nbs = np.ascontiguousarray(nb[feats])
    nmax = int(nbs.max())

    def grow(rows, d):
        if d == 0 or len(rows) == 0:
            return None
        Xr = np.ascontiguousarray(Xs[rows])
        gains, cuts = best_cut_1d(hist_1d(Xr, g[rows], h[rows], w[rows], nmax), nbs, lam,
                                  min_weight)
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            return None
        left = Xr[:, j] <= cuts[j]
        return (int(feats[j]), int(cuts[j]), grow(rows[left], d - 1), grow(rows[~left], d - 1))

    spec = grow(np.arange(len(Xs)), depth)
    return None if spec is None else SmallTree.from_nested(spec)


class _AdditiveTrees(InterpretableSumMixin, ClassifierMixin, BaseEstimator):
    """Base: bins, margem, backfitting e previsão de uma soma de SmallTree."""

    def _prepare(self, X, y, sample_weight, y_soft):
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        Xb, edges, nb = binned(X, self.max_bins)
        self.classes_, self.bin_edges_ = classes, edges
        return X, Xb, nb, target, w

    def _newton_step(self, tree, Xb, target, margin, contrib, w):
        """Passo de Newton incremental nas folhas de ``tree`` a partir dos valores atuais.

        ``margin`` inclui a contribuição atual da árvore; devolve (nova
        contribuição, nova margem). Partir do valor atual (e não de zero) é o
        que mantém o backfitting estável com árvores quase colineares.
        """
        g, h = grad_hess(target, margin, w)
        ids = tree.leaf_ids(Xb)
        tree.value = tree.value + self.learning_rate_ * newton_leaf_values(
            ids, g, h, len(tree.feature), self.lam_)
        new = tree.value[ids]
        return new, margin - contrib + new

    def _backfit(self, trees, contribs, Xb, target, w, sweeps):
        margin = self.base_margin_ + np.sum(contribs, axis=0)
        for _ in range(sweeps):
            for k, tree in enumerate(trees):
                contribs[k], margin = self._newton_step(tree, Xb, target, margin,
                                                        contribs[k], w)
        return margin

    def decision_function(self, X):
        """Logit de ``P(y = classes_[1])``: base + soma das árvores."""
        Xb = rebin(predict_input(self, X, "trees_"), self.bin_edges_)
        m = np.full(len(Xb), self.base_margin_)
        for tree in self.trees_:
            m += tree.value[tree.leaf_ids(Xb)]
        return m


class SumOfOptimalTrees(_AdditiveTrees):
    """Soma em logit de poucas árvores Newton ótimas: o modelo de orçamento fixo.

    ``logit(p) = base + Σ_k árvore_k(x)``. Cada árvore de profundidade
    ``depth`` é a ótima (busca exaustiva nos bins) no resíduo das anteriores;
    depois, ``backfit_sweeps`` passadas reestimam as folhas de todas juntas.
    Orçamento: ``n_trees · (2^depth − 1) + extra_stumps`` cortes (ex.: 16 cortes
    = 5 árvores d2 + 1 toco).

    Parameters
    ----------
    n_trees : int, default=2
        Número de árvores de profundidade ``depth``.
    depth : {1, 2, 3}, default=2
        Profundidade das árvores.
    learning_rate : float or "auto", default="auto"
        Shrinkage dos passos de Newton nas folhas (1 = passo inteiro). "auto" =
        1 / (1 + 0,1 · n_trees), a mediana do tuning no benchmark (0,9 com uma
        árvore, 0,3 com 21).
    lam : float, default=2.0
        Regularização L2 das folhas (no ganho e no passo de Newton).
    min_weight : float, default=20.0
        Massa (hessiana) mínima por folha.
    max_bins : int, default=16
        Bins por feature.
    backfit_sweeps : int, default=2
        Passadas de reestimação conjunta das folhas.
    max_features_d3, max_features_d2 : int, default=24, 128
        Com mais features, a busca fica restrita às de maior importância
        (ótima dentro desse conjunto).
    feature_screen : {"lgbm", "fast"}, default="lgbm"
        Triagem: importância de ganho do LightGBM (uma vez) ou FAST por rodada.
    search : {"optimal", "greedy"}, default="optimal"
        ``"greedy"`` cresce cada árvore pelo melhor corte único (controle).
    extra_stumps : int, default=0
        Tocos (d1) somados depois das árvores, para fechar o orçamento.

    Attributes
    ----------
    trees_ : list of SmallTree
        As árvores, com cortes em bins; ``get_trees()`` dá os valores reais.
    base_margin_ : float
        Logit constante inicial.
    classes_, n_features_in_, feature_names_in_, bin_edges_, nan_features_
        Convenções do sklearn e dos bins.

    Métodos de interpretação: ``explain``, ``rules``, ``to_dict``,
    ``get_trees``, ``export_text``, ``predict_contributions``,
    ``plot_contributions``, ``plot_shapes``.
    """

    def __init__(self, *, n_trees=2, depth=2, learning_rate="auto", lam=2.0,
                 min_weight=20.0, max_bins=16, backfit_sweeps=2, max_features_d3=24,
                 max_features_d2=128, feature_screen="lgbm", search="optimal",
                 extra_stumps=0):
        self.n_trees = n_trees
        self.depth = depth
        self.learning_rate = learning_rate
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.backfit_sweeps = backfit_sweeps
        self.max_features_d3 = max_features_d3
        self.max_features_d2 = max_features_d2
        self.feature_screen = feature_screen
        self.search = search
        self.extra_stumps = extra_stumps

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Ajusta a soma.

        Parameters
        ----------
        X : array-like ou DataFrame, shape (n, p)
            Features numéricas; NaN é aceito (vai sempre para o lado ``<=``).
            Com DataFrame, os nomes das colunas viram ``feature_names_in_``.
        y : array-like, shape (n,)
            Rótulos de duas classes.
        sample_weight : array-like, shape (n,), opcional
            Pesos não negativos das linhas.
        y_soft : array-like, shape (n,), opcional
            Alvo suave em [0, 1] (ex.: probabilidade de um professor) no lugar
            de y; ``classes_`` continua vindo de y.

        Returns
        -------
        self
        """
        if self.search not in ("optimal", "greedy"):
            raise ValueError("search must be 'optimal' or 'greedy'.")
        grow = optimal_tree if self.search == "optimal" else greedy_tree
        self.lam_ = float(self.lam)
        self.learning_rate_ = (1.0 / (1.0 + 0.1 * self.n_trees) if self.learning_rate == "auto"
                               else float(self.learning_rate))
        X, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        margin = np.full(len(Xb), self.base_margin_)
        trees, contribs = [], []
        k = self.max_features_d3 if self.depth == 3 else self.max_features_d2
        fixed = (teacher_top_features(X, (target > 0.5).astype(int), k, w)
                 if self.feature_screen == "lgbm" and self.depth >= 2 else None)
        for depth in [self.depth] * self.n_trees + [1] * self.extra_stumps:
            g, h = grad_hess(target, margin, w)
            if depth == 1:  # toco: busca completa é barata e ótimo = guloso
                feats = None
            else:
                feats = fixed if self.feature_screen == "lgbm" else top_features(
                    Xb, g, h, w, nb, self.lam_, self.min_weight, k)
            tree = grow(Xb, g, h, w, nb, depth, self.lam_, self.min_weight, feats)
            if tree is None:
                break
            contrib, margin = self._newton_step(tree, Xb, target, margin, np.zeros(len(Xb)), w)
            trees.append(tree)
            contribs.append(contrib)
        if trees and self.backfit_sweeps:
            self._backfit(trees, contribs, Xb, target, w, self.backfit_sweeps)
        self.trees_ = trees
        return self


class FIGSClassifier(_AdditiveTrees):
    """FIGS (Tan et al., 2022) em logit: até ``max_splits`` cortes entre árvores
    que crescem juntas.

    A cada passo, o melhor corte (ganho Newton) em qualquer folha de qualquer
    árvore, ou a raiz de uma árvore nova, contra a margem das outras árvores;
    as folhas são reestimadas por backfitting. O número e a forma das árvores
    saem dos dados; o orçamento é o total de cortes.

    Parameters
    ----------
    max_splits : int, default=16
        Total de cortes (pode parar antes se nenhum corte tem ganho).
    max_trees : int ou None, default=None
        Limite de árvores.
    lam : float or "auto", default="auto"
        Regularização L2 das folhas (no ganho e no passo de Newton). "auto" =
        2 · max_splits: o λ escolhido pelo tuning cresce com o orçamento (≈ 5 com
        4 cortes, ≈ 300 com 64); sem isso, orçamentos grandes sobreajustam.
    min_weight : float, default=20.0
        Massa (hessiana) mínima por folha.
    max_bins : int, default=32
        Bins por feature.
    backfit_sweeps : int, default=1
        Passadas de reestimação das folhas após cada corte.

    Attributes
    ----------
    trees_ : list of SmallTree
        As árvores, com cortes em bins; ``get_trees()`` dá os valores reais.
    base_margin_ : float
        Logit constante inicial.
    classes_, n_features_in_, feature_names_in_, bin_edges_, nan_features_
        Convenções do sklearn e dos bins.

    Métodos de interpretação: ``explain``, ``rules``, ``to_dict``,
    ``get_trees``, ``export_text``, ``predict_contributions``,
    ``plot_contributions``, ``plot_shapes``.
    """

    def __init__(self, *, max_splits=16, max_trees=None, lam="auto", min_weight=20.0,
                 max_bins=32, backfit_sweeps=1):
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.backfit_sweeps = backfit_sweeps

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Ajusta a soma.

        Parameters
        ----------
        X : array-like ou DataFrame, shape (n, p)
            Features numéricas; NaN é aceito (vai sempre para o lado ``<=``).
            Com DataFrame, os nomes das colunas viram ``feature_names_in_``.
        y : array-like, shape (n,)
            Rótulos de duas classes.
        sample_weight : array-like, shape (n,), opcional
            Pesos não negativos das linhas.
        y_soft : array-like, shape (n,), opcional
            Alvo suave em [0, 1] (ex.: probabilidade de um professor) no lugar
            de y; ``classes_`` continua vindo de y.

        Returns
        -------
        self
        """
        self.lam_ = 2.0 * self.max_splits if self.lam == "auto" else float(self.lam)
        self.learning_rate_ = 1.0  # FIGS não encolhe: cada folha é o passo de Newton inteiro
        _, Xb, nb, target, w = self._prepare(X, y, sample_weight, y_soft)
        self.base_margin_ = base_margin(target, w)
        n, B = len(Xb), int(nb.max())
        trees, contribs = [], []
        margin = np.full(n, self.base_margin_)
        for _ in range(self.max_splits):
            best = (0.0, None, None, None, None)  # ganho, árvore, nó, f, t
            can_add = self.max_trees is None or len(trees) < self.max_trees
            candidates = list(range(len(trees))) + ([None] if can_add else [])
            for k in candidates:
                if k is None:
                    g, h = grad_hess(target, margin, w)
                    ids = np.zeros(n, dtype=np.int64)
                    leaves, n_nodes = [0], 1
                else:
                    g, h = grad_hess(target, margin - contribs[k], w)
                    ids = trees[k].leaf_ids(Xb)
                    leaves, n_nodes = trees[k].leaves, len(trees[k].feature)
                hist = node_hist(Xb, g, h, w, ids, n_nodes, B)
                for leaf in leaves:
                    gains, cuts = best_cut_1d(hist[leaf], nb, self.lam_, self.min_weight)
                    f = int(np.argmax(gains))
                    if gains[f] > best[0]:
                        best = (float(gains[f]), k, leaf, f, int(cuts[f]))
            _, k, leaf, f, t = best
            if k is None and leaf is None:
                break
            if k is None:
                trees.append(SmallTree())
                contribs.append(np.zeros(n))
                k = len(trees) - 1
            trees[k].split(leaf, f, t)
            contribs[k], margin = self._newton_step(trees[k], Xb, target, margin,
                                                    contribs[k], w)
            if self.backfit_sweeps:
                margin = self._backfit(trees, contribs, Xb, target, w, self.backfit_sweeps)
        self.trees_ = trees
        return self


class BoostedOptimalTrees(_AdditiveTrees):
    """Boosting de árvores Newton ótimas de profundidade 1–3, com early stopping.

    Generaliza ``AdditiveTreeBooster`` (d1/d2) para d3: a cada rodada, a
    árvore ótima de profundidade ``depth`` no resíduo (em d3, restrita às
    ``max_features_d3`` features de maior importância), folhas por passo de
    Newton × ``learning_rate``. Para quando a log-loss da validação interna
    (``validation_fraction``) não melhora em ``patience`` rodadas e mantém as
    árvores até o melhor ponto. Parâmetros como em ``AdditiveTreeBooster``.

    Attributes
    ----------
    trees_ : list of SmallTree
        As árvores, com cortes em bins; ``get_trees()`` dá os valores reais.
    base_margin_ : float
        Logit constante inicial.
    classes_, n_features_in_, feature_names_in_, bin_edges_, nan_features_
        Convenções do sklearn e dos bins.

    Métodos de interpretação: ``explain``, ``rules``, ``to_dict``,
    ``get_trees``, ``export_text``, ``predict_contributions``,
    ``plot_contributions``, ``plot_shapes``.
    """

    def __init__(self, *, depth=3, learning_rate=0.1, max_rounds=300, patience=30, lam=1.0,
                 min_weight=20.0, max_bins=32, validation_fraction=0.15,
                 max_features_d3=32, max_features_d2=128, feature_screen="lgbm",
                 random_state=0):
        self.depth = depth
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.patience = patience
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.validation_fraction = validation_fraction
        self.max_features_d3 = max_features_d3
        self.max_features_d2 = max_features_d2
        self.feature_screen = feature_screen
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Ajusta a soma.

        Parameters
        ----------
        X : array-like ou DataFrame, shape (n, p)
            Features numéricas; NaN é aceito (vai sempre para o lado ``<=``).
            Com DataFrame, os nomes das colunas viram ``feature_names_in_``.
        y : array-like, shape (n,)
            Rótulos de duas classes.
        sample_weight : array-like, shape (n,), opcional
            Pesos não negativos das linhas.
        y_soft : array-like, shape (n,), opcional
            Alvo suave em [0, 1] (ex.: probabilidade de um professor) no lugar
            de y; ``classes_`` continua vindo de y.

        Returns
        -------
        self
        """
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        self.lam_, self.learning_rate_ = float(self.lam), float(self.learning_rate)
        rng = np.random.default_rng(self.random_state)
        val = np.zeros(len(X), dtype=bool)
        val[rng.permutation(len(X))[:int(round(self.validation_fraction * len(X)))]] = True
        Xb, edges, nb = binned(X[~val], self.max_bins)
        Xv = rebin(X[val], edges)
        yt, wt, yv, wv = target[~val], w[~val], target[val], w[val]
        self.classes_, self.bin_edges_ = classes, edges
        self.base_margin_ = base_margin(yt, wt)
        m_tr = np.full(len(yt), self.base_margin_)
        m_val = np.full(len(yv), self.base_margin_)
        trees, history, best, best_round = [], [], np.inf, 0
        k = self.max_features_d3 if self.depth == 3 else self.max_features_d2
        fixed = (teacher_top_features(X[~val], (yt > 0.5).astype(int), k, wt, self.random_state)
                 if self.feature_screen == "lgbm" and self.depth >= 2 else None)
        for _ in range(self.max_rounds):
            g, h = grad_hess(yt, m_tr, wt)
            feats = fixed if self.feature_screen == "lgbm" else top_features(
                Xb, g, h, wt, nb, self.lam_, self.min_weight, k)
            tree = optimal_tree(Xb, g, h, wt, nb, self.depth, self.lam_, self.min_weight, feats)
            if tree is None:
                break
            ids = tree.leaf_ids(Xb)
            tree.value = self.learning_rate * newton_leaf_values(ids, g, h, len(tree.feature),
                                                                 self.lam_)
            m_tr += tree.value[ids]
            trees.append(tree)
            if len(yv):
                m_val += tree.value[tree.leaf_ids(Xv)]
                p = np.clip(sigmoid(m_val), 1e-15, 1 - 1e-15)
                loss = float(-np.sum(wv * (yv * np.log(p) + (1 - yv) * np.log(1 - p))) / wv.sum())
            else:
                loss = -len(trees)
            history.append(loss)
            if loss < best - 1e-12:
                best, best_round = loss, len(trees)
            elif len(trees) - best_round >= self.patience:
                break
        self.trees_ = trees[:best_round] if len(yv) else trees
        self.history_ = np.array(history)
        return self
