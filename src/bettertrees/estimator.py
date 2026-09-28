"""Minimal Python API of the classification tree (scikit-learn compatible).

The exact engine follows CART (Breiman et al., 1984); the histogram engine
follows LightGBM (Ke et al., 2017) and scikit-learn's HistGradientBoosting;
leaf probabilities can use hierarchical shrinkage (Agarwal et al., ICML 2022).
"""

from numbers import Integral, Real
from time import perf_counter

import numpy as np
from numba import get_num_threads, set_num_threads
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from ._data import prepare_training_data, validate_X
from .bins import fit_bin_edges, transform_bins_row_major
from .builder import grow_tree_exact, grow_tree_hist
from .kernels import apply_nodes
from .postprocess import (
    finite_leaf_regions,
    hierarchical_shrinkage_probabilities,
    predict_proba_nodes,
    project_monotonic_leaf_probabilities,
    prune_tree_cost_complexity,
)
from .splitters import resolve_splitter_spec


class FastDecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """Scaffold de classificação binária/multiclasse por Gini ponderado.

    Parameters
    ----------
    splitter : {'hist', 'exact'}, default='hist'
        Motor próprio; nunca delega treino ao scikit-learn.
    objective : {'gini', 'precision'}, default='gini'
        Objetivo que pontua os cortes. A interface é modular para objetivos
        futuros. ``precision`` é **experimental**: pontua cada corte pela
        melhor precision entre os filhos com suporte mínimo, menos a precision
        do pai, sem usar o bound de Gini. Não maximiza TP nem cobertura e
        tende a cortes extremos que isolam o filho mais puro.
        ``feature_importances_`` continua medida em Gini.
    positive_class : scalar, default=None
        Classe positiva explícita quando ``objective='precision'``.
    min_precision : float in [0, 1], default=0.9
        Precisão mínima de uma folha elegível no objetivo de precisão.
    min_support : float > 0, default=1.0
        Massa ponderada mínima de uma folha elegível no objetivo de precisão.
    max_feature_repeats : int >= 1 or None, default=None
        Número máximo de ocorrências de uma mesma feature em cada caminho
        raiz->folha. ``None`` preserva o crescimento sem esse limite; o
        contador é compartilhado pelo caminho e permite, por exemplo, duas
        ocorrências para representar um intervalo.
    monotonic_cst : array-like of {-1, 0, 1} or None, default=None
        Restrições da probabilidade da classe positiva em classificação
        binária. ``+1`` exige que a probabilidade não diminua ao aumentar a
        feature, ``-1`` exige o inverso e ``0`` deixa a feature livre. A
        restrição vale para valores finitos; NaN segue a direção aprendida em
        cada nó e fica fora do contrato monotônico.
    max_depth : int >= 1 or None
        Profundidade máxima; raiz tem profundidade zero.
    min_samples_leaf : int >= 1
        Número mínimo de linhas de peso positivo por filho.
    max_leaf_nodes : int >= 2 or None
        Orçamento de folhas; requer crescimento best-first quando definido.
    random_state : int or None
        Permutação fixa das features para desempate, reproduzível por seed.
    min_impurity_decrease : float >= 0
        Ganho mínimo para aceitar um split. Em Gini é a redução de impureza
        ponderada pela massa relativa à raiz; em precision é o aumento da
        melhor precision filha sobre a precision do pai.
    max_bins : int in [2,255]
        Máximo de bins finitos por feature; NaN usa bin separado 0.
    search_stopping : {'bound', 'off', 'heuristic'}, default='bound'
        'bound': interromper busca apenas por limite superior admissível;
        'off': varrer todos os candidatos; 'heuristic': reservado, bloqueado
        até definir e validar a regra aproximada. Não é parada por holdout.
        O motor ``exact`` sempre faz busca exaustiva; quando ``bound`` é
        solicitado, ``fit_stats_['search_stopping_effective']`` registra
        ``'off'``.
    gain_tolerance : float >= 0, default=0.0
        Incremento local mínimo que ainda vale buscar além do incumbente.
        Zero: bound exato. Positivo: aproximação com tolerância em Gini local,
        sem garantia de perda máxima em métricas fora do treino. Só atua
        quando o bound histogramado está ativo.
    n_jobs : int >= 1, default=1
        Número de threads Numba da variante paralela do motor histogramado.
        ``1`` preserva o baseline serial; a saída deve ser conferida antes de
        promover valores maiores.
    reuse_parent_histograms : bool, default=False
        Variante experimental para crescimento depth-first com pesos unitários.
        Acumula o filho menor e obtém o maior por subtração do histograma pai.
    leaf_smoothing : float >= 0, default=0.0
        Massa de prior adicionada às folhas para prever probabilidades. O
        prior é a distribuição ponderada das classes na raiz. Zero preserva
        exatamente as frequências empíricas; não altera splits ou poda.
    ccp_alpha : float >= 0, default=0.0
        Penalidade por folha na pós-poda de custo-complexidade com risco Gini
        ponderado. Zero preserva a árvore original. Por ora, valores positivos
        só são aceitos para ``objective='gini'``.
    leaf_shrinkage : float >= 0, default=0.0
        Shrinkage hierárquico (Agarwal et al., ICML 2022) das probabilidades:
        cada folha vira uma combinação convexa das frequências dos
        ancestrais, com peso 1/(1 + λ/massa do pai) por nível. Não altera a
        árvore; reduz o sobreajuste das folhas pequenas. Exclusivo com
        ``leaf_smoothing`` e, por ora, com ``monotonic_cst``. Para escolher λ
        e o número de folhas por validação interna, use
        ``FastDecisionTreeClassifierCV``.

    Notes
    -----
    get_params/set_params vêm de BaseEstimator; clone funciona. Isso não
    significa compatibilidade integral com todos os checks do scikit-learn.
    Não há validação interna nem seleção automática da poda por holdout.
    Suavização e pós-poda são opt-in.
    """

    def __init__(self, *, splitter="hist", max_depth=None, min_samples_leaf=1,
                 max_leaf_nodes=None, random_state=None, min_impurity_decrease=0.0,
                 max_bins=255, search_stopping="bound", gain_tolerance=0.0,
                 objective="gini", positive_class=None, min_precision=0.9,
                  min_support=1.0, max_feature_repeats=None, n_jobs=1,
                  reuse_parent_histograms=False, monotonic_cst=None,
                  leaf_smoothing=0.0, ccp_alpha=0.0, leaf_shrinkage=0.0):
        """Guarde parâmetros sem trabalho de treino, permitindo clone/set_params."""
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.max_bins = max_bins
        self.search_stopping = search_stopping
        self.gain_tolerance = gain_tolerance
        self.objective = objective
        self.positive_class = positive_class
        self.min_precision = min_precision
        self.min_support = min_support
        self.max_feature_repeats = max_feature_repeats
        self.n_jobs = n_jobs
        self.reuse_parent_histograms = reuse_parent_histograms
        self.monotonic_cst = monotonic_cst
        self.leaf_smoothing = leaf_smoothing
        self.ccp_alpha = ccp_alpha
        self.leaf_shrinkage = leaf_shrinkage

    def _validate_parameters(self):
        """Rejeite configurações inválidas antes de alocar dados ou compilar."""
        for name, minimum, allow_none in (("max_depth", 1, True),
                                          ("min_samples_leaf", 1, False),
                                          ("max_leaf_nodes", 2, True),
                                          ("max_bins", 2, False)):
            value = getattr(self, name)
            if value is None and allow_none:
                continue
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Integral)
                    or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.max_bins > 255:
            raise ValueError("max_bins must be <= 255.")
        if self.max_feature_repeats is not None and (
                isinstance(self.max_feature_repeats, (bool, np.bool_))
                or not isinstance(self.max_feature_repeats, Integral)
                or self.max_feature_repeats < 1):
            raise ValueError(
                "max_feature_repeats must be a positive integer or None.")
        if (isinstance(self.n_jobs, (bool, np.bool_))
                or not isinstance(self.n_jobs, Integral) or self.n_jobs < 1):
            raise ValueError("n_jobs must be an integer >= 1.")
        if not isinstance(self.reuse_parent_histograms, (bool, np.bool_)):
            raise ValueError("reuse_parent_histograms must be a boolean.")
        if self.reuse_parent_histograms and self.splitter != "hist":
            raise ValueError("Parent-child histogram reuse requires splitter='hist'.")
        if self.reuse_parent_histograms and self.max_leaf_nodes is not None:
            raise ValueError("Parent-child histogram reuse requires max_leaf_nodes=None.")
        if self.monotonic_cst is not None and np.isscalar(self.monotonic_cst):
            raise ValueError("monotonic_cst must be a vector or None.")
        resolve_splitter_spec(self.splitter, self.objective, self.search_stopping)
        for name in ("min_impurity_decrease", "gain_tolerance",
                     "leaf_smoothing", "ccp_alpha", "leaf_shrinkage"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.leaf_shrinkage > 0 and self.leaf_smoothing > 0:
            raise ValueError("Use leaf_shrinkage OR leaf_smoothing, not both.")
        if self.leaf_shrinkage > 0 and self.monotonic_cst is not None:
            raise ValueError("leaf_shrinkage cannot be combined with monotonic_cst yet.")
        if (self.random_state is not None and (isinstance(self.random_state, bool)
                or not isinstance(self.random_state, Integral))):
            raise ValueError("random_state must be an integer or None.")
        check_random_state(self.random_state)
        if self.objective == "precision":
            if self.positive_class is None:
                raise ValueError("positive_class is required for objective='precision'.")
            if self.gain_tolerance != 0:
                raise ValueError("gain_tolerance does not apply to objective='precision'.")
            if self.ccp_alpha > 0:
                raise ValueError("A positive ccp_alpha requires objective='gini'.")
        if (isinstance(self.min_precision, (bool, np.bool_))
                or not isinstance(self.min_precision, Real)
                or not np.isfinite(self.min_precision)
                or not 0 <= self.min_precision <= 1):
            raise ValueError("min_precision must be between 0 and 1.")
        if (isinstance(self.min_support, (bool, np.bool_))
                or not isinstance(self.min_support, Real)
                or not np.isfinite(self.min_support)
                or self.min_support <= 0):
            raise ValueError("min_support must be finite and positive.")

    @staticmethod
    def _feature_names(X):
        """Retorne nomes de colunas textuais, quando X os fornecer.

        O núcleo aceita qualquer matriz numérica densa; a detecção fica na
        fachada para manter DataFrame opcional e não introduzir pandas no
        caminho de treino de arrays NumPy.
        """
        columns = getattr(X, "columns", None)
        if columns is None:
            return None
        try:
            names = tuple(columns)
        except TypeError:
            return None
        if names and all(isinstance(name, str) for name in names):
            return names
        return None

    def _validate_feature_names(self, X):
        """Confira nomes de DataFrame sem rejeitar arrays NumPy no predict."""
        fitted_names = getattr(self, "feature_names_in_", None)
        if fitted_names is None:
            return
        names = self._feature_names(X)
        if names is None:
            return
        if names != tuple(fitted_names):
            raise ValueError(
                "Feature names at prediction time must match the "
                "names and order seen during fit."
            )

    def _validate_predict_X(self, X):
        """Valide X de previsão, incluindo o contrato de nomes de features."""
        self._validate_feature_names(X)
        return validate_X(X, n_features=self.n_features_in_)

    def _validate_monotonic_cst(self, n_features, n_classes):
        """Valide direções e devolva um vetor inteiro estável para o fit."""
        if self.monotonic_cst is None:
            return None
        if n_classes != 2:
            raise ValueError("monotonic_cst is only supported for binary classification.")
        try:
            values = np.asarray(self.monotonic_cst)
        except Exception as exc:  # pragma: no cover - mensagem de contrato
            raise ValueError("monotonic_cst must be a vector of -1, 0 and 1.") from exc
        if values.ndim != 1 or len(values) != n_features:
            raise ValueError(
                "monotonic_cst must have one direction per feature.")
        if any(isinstance(value, (bool, np.bool_)) for value in values.tolist()):
            raise ValueError("monotonic_cst only accepts -1, 0 and 1.")
        try:
            numeric = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("monotonic_cst only accepts -1, 0 and 1.") from exc
        if (not np.isfinite(numeric).all()
                or not np.isin(numeric, [-1.0, 0.0, 1.0]).all()):
            raise ValueError("monotonic_cst only accepts -1, 0 and 1.")
        return numeric.astype(np.int8)

    @staticmethod
    def _finite_leaf_regions(nodes, n_features):
        """Caixas finitas das folhas; ver ``postprocess.finite_leaf_regions``."""
        return finite_leaf_regions(nodes, n_features)

    def _project_monotonic_leaf_probabilities(self, nodes, directions,
                                              positive_class_index):
        """Delegue a projeção monotônica ao pós-processamento."""
        return project_monotonic_leaf_probabilities(
            nodes, directions, positive_class_index, self.leaf_smoothing)

    def fit(self, X, y, sample_weight=None):
        """Ajuste com escopo temporário de threads Numba."""
        self._validate_parameters()
        previous_threads = get_num_threads()
        set_num_threads(int(self.n_jobs))
        try:
            return self._fit_impl(X, y, sample_weight)
        finally:
            set_num_threads(previous_threads)

    def _fit_impl(self, X, y, sample_weight=None):
        """Valide dados, aprenda bins quando necessário e construa a árvore.

        Todo preparo, bins e crescimento pertence ao fit cronometrado.
        Publicar atributos terminados em '_' apenas após o builder retornar;
        treino incompleto não pode aparentar sucesso nem substituir motor.
        """
        fit_start = perf_counter()
        feature_names = self._feature_names(X)
        splitter_spec = resolve_splitter_spec(
            self.splitter, self.objective, self.search_stopping)
        if self.search_stopping == "heuristic":
            raise NotImplementedError("The heuristic stopping rule is not specified/validated yet.")
        prepare_start = perf_counter()
        X, encoded, weights, classes = prepare_training_data(X, y, sample_weight)
        prepare_seconds = perf_counter() - prepare_start
        monotonic_directions = self._validate_monotonic_cst(
            X.shape[1], len(classes))
        positive_class_index = -1
        if self.objective == "precision":
            matches = np.flatnonzero(classes == self.positive_class)
            if len(matches) != 1:
                raise ValueError("positive_class must appear exactly once among the classes of y.")
            positive_class_index = int(matches[0])
        params = dict(n_classes=len(classes), max_depth=self.max_depth,
                      min_samples_leaf=self.min_samples_leaf,
                      max_leaf_nodes=self.max_leaf_nodes,
                      min_impurity_decrease=self.min_impurity_decrease,
                      feature_order=check_random_state(self.random_state).permutation(X.shape[1]),
                      stopping=self.search_stopping, gain_tolerance=self.gain_tolerance,
                      positive_class=positive_class_index,
                      min_precision=float(self.min_precision),
                      min_support=float(self.min_support),
                      max_feature_repeats=self.max_feature_repeats)
        edges = None
        stats = {}
        if self.splitter == "hist":
            effective_max_bins = self.max_bins
            bin_start = perf_counter()
            edges = fit_bin_edges(X, effective_max_bins, n_jobs=self.n_jobs)
            bin_learning_seconds = perf_counter() - bin_start
            transform_start = perf_counter()
            X_binned = transform_bins_row_major(X, edges, n_jobs=self.n_jobs)
            bin_transform_seconds = perf_counter() - transform_start
            builder_start = perf_counter()
            nodes = grow_tree_hist(X, X_binned, encoded, weights, edges,
                                    stats=stats, objective=splitter_spec.name,
                                    parallel=self.n_jobs > 1,
                                    reuse_parent_histograms=self.reuse_parent_histograms,
                                    **params)
        else:
            bin_learning_seconds = 0.0
            bin_transform_seconds = 0.0
            builder_start = perf_counter()
            nodes = grow_tree_exact(X, encoded, weights, stats=stats,
                                    objective=splitter_spec.name, **params)
        builder_seconds = perf_counter() - builder_start
        preprune_nodes = len(nodes.left)
        prune_start = perf_counter()
        if self.ccp_alpha > 0:
            nodes = prune_tree_cost_complexity(nodes, float(self.ccp_alpha))
        prune_seconds = perf_counter() - prune_start
        monotonic_positive_class_index = (
            positive_class_index if self.objective == "precision" else 1)
        if monotonic_directions is None:
            monotonic_leaf_probabilities = None
        else:
            monotonic_leaf_probabilities = (
                self._project_monotonic_leaf_probabilities(
                    nodes, monotonic_directions,
                    monotonic_positive_class_index))
        self.nodes_, self.classes_ = nodes, classes
        self.n_features_in_, self.n_classes_ = X.shape[1], len(classes)
        if feature_names is None:
            self.__dict__.pop("feature_names_in_", None)
        else:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.bin_edges_ = edges
        self.monotonic_cst_ = monotonic_directions
        self.monotonic_positive_class_index_ = (
            monotonic_positive_class_index if monotonic_directions is not None
            else None)
        self.monotonic_leaf_probabilities_ = monotonic_leaf_probabilities
        stats.update({
            "splitter": self.splitter,
            "objective": splitter_spec.name,
            "objective_experimental": splitter_spec.experimental,
            "positive_class": self.positive_class if self.objective == "precision" else None,
            "min_precision": float(self.min_precision),
            "min_support": float(self.min_support),
            "leaf_smoothing": float(self.leaf_smoothing),
            "ccp_alpha": float(self.ccp_alpha),
            "max_feature_repeats": self.max_feature_repeats,
            "monotonic_cst": (None if monotonic_directions is None
                               else monotonic_directions.tolist()),
            "search_stopping": self.search_stopping,
            "search_stopping_effective": (
                "off" if self.splitter == "exact" else self.search_stopping),
            "gain_tolerance": float(self.gain_tolerance),
            "min_impurity_decrease": float(self.min_impurity_decrease),
            "n_jobs": int(self.n_jobs),
            "parallel": bool(self.n_jobs > 1 and self.splitter == "hist"),
            "reuse_parent_histograms": bool(self.reuse_parent_histograms),
            "max_bins_requested": int(self.max_bins),
            "max_bins_effective": int(effective_max_bins) if self.splitter == "hist" else None,
            "prepare_seconds": prepare_seconds,
            "bin_learning_seconds": bin_learning_seconds,
            "bin_transform_seconds": bin_transform_seconds,
            "builder_seconds": builder_seconds,
            "prune_seconds": prune_seconds,
            "preprune_nodes": preprune_nodes,
            "nodes_removed_by_pruning": preprune_nodes - len(nodes.left),
            "fit_complete_seconds": perf_counter() - fit_start,
            "n_nodes": len(nodes.left),
            "n_leaves": int(np.count_nonzero(nodes.left == -1)),
        })
        self.leaf_probabilities_ = (
            hierarchical_shrinkage_probabilities(nodes, float(self.leaf_shrinkage))
            if self.leaf_shrinkage > 0 else None)
        self.fit_stats_ = stats
        return self

    def predict_proba(self, X):
        """Retorne probabilidades nas colunas de classes_, após treino completo."""
        check_is_fitted(self, "nodes_")
        return predict_proba_nodes(
            self._validate_predict_X(X), self.nodes_,
            self.monotonic_leaf_probabilities_,
            positive_class=(self.monotonic_positive_class_index_
                            if self.monotonic_positive_class_index_ is not None
                            else 1),
            leaf_smoothing=self.leaf_smoothing,
            leaf_probabilities=getattr(self, "leaf_probabilities_", None),
        )

    def predict(self, X):
        """Retorne classes originais; empate favorece o menor índice em classes_."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def predict_log_proba(self, X):
        """Retorne o logaritmo das probabilidades previstas."""
        probabilities = self.predict_proba(X)
        with np.errstate(divide="ignore"):
            return np.log(probabilities)

    def apply(self, X):
        """Retorne o índice da folha que recebe cada amostra."""
        check_is_fitted(self, "nodes_")
        X_validated = self._validate_predict_X(X)
        return np.asarray(apply_nodes(
            X_validated, self.nodes_.left, self.nodes_.right,
            self.nodes_.feature, self.nodes_.threshold,
            self.nodes_.missing_left,
        ), dtype=np.intp)

    def get_depth(self):
        """Retorne a profundidade máxima da árvore; a raiz tem profundidade zero."""
        check_is_fitted(self, "nodes_")
        max_depth = 0
        pending = [(0, 0)]
        while pending:
            node_id, depth = pending.pop()
            left = int(self.nodes_.left[node_id])
            if left == -1:
                continue
            max_depth = max(max_depth, depth + 1)
            pending.append((left, depth + 1))
            pending.append((int(self.nodes_.right[node_id]), depth + 1))
        return max_depth

    def get_n_leaves(self):
        """Retorne o número de folhas publicadas na árvore."""
        check_is_fitted(self, "nodes_")
        return int(np.count_nonzero(self.nodes_.left == -1))

    @property
    def feature_importances_(self):
        """Importância normalizada pela redução ponderada de Gini.

        A massa do nó é a soma de ``class_weight``; portanto, quando pesos
        são usados, a redução segue a massa ponderada e não a contagem bruta
        de linhas. Árvores sem cortes retornam zeros, como no sklearn.
        """
        check_is_fitted(self, "nodes_")
        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for node_id, feature in enumerate(self.nodes_.feature):
            feature = int(feature)
            if feature < 0:
                continue
            left = int(self.nodes_.left[node_id])
            right = int(self.nodes_.right[node_id])
            parent_mass = self.nodes_.class_weight[node_id]
            left_mass = self.nodes_.class_weight[left]
            right_mass = self.nodes_.class_weight[right]
            parent_total = float(parent_mass.sum())
            left_total = float(left_mass.sum())
            right_total = float(right_mass.sum())
            if parent_total <= 0:
                continue
            parent_impurity = 1.0 - float(
                np.square(parent_mass / parent_total).sum())
            left_impurity = (0.0 if left_total <= 0 else 1.0 - float(
                np.square(left_mass / left_total).sum()))
            right_impurity = (0.0 if right_total <= 0 else 1.0 - float(
                np.square(right_mass / right_total).sum()))
            reduction = (parent_total * parent_impurity
                         - left_total * left_impurity
                         - right_total * right_impurity)
            importances[feature] += max(0.0, reduction)
        total = float(importances.sum())
        if total > 0.0:
            importances /= total
        return importances
