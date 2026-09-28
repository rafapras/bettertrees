"""Contrato modular entre motor de candidatos e objetivo de split.

Um objetivo é registrado aqui uma única vez e carrega todo o comportamento
de que o builder precisa: quando um nó pode ser dividido, qual busca usar em
cada motor, se o ganho encontrado é aceito e a prioridade no best-first. O
builder não compara nomes de objetivos.

A resolução ocorre uma vez por ``fit``; nada deste módulo roda por candidato.
As buscas são lidas de ``search`` no momento da chamada, de modo que testes
possam substituí-las por referências. Adicionar um objetivo novo começa com
um kernel próprio em ``kernels.py``, uma busca em ``search.py`` e um spec
aqui, sem alterar silenciosamente o Gini de produção.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import search as _search
from .kernels import gini, precision_leaf_score


@dataclass(frozen=True)
class ObjectiveParams:
    """Parâmetros do objetivo, normalizados para tipos estáveis no Numba."""

    positive_class: int = 0
    min_precision: float = 0.0
    min_support: float = 0.0

    @classmethod
    def from_values(cls, positive_class=0, min_precision=0.0, min_support=0.0):
        return cls(int(positive_class), float(min_precision), float(min_support))


@dataclass(frozen=True)
class SplitObjectiveSpec:
    """Metadados e comportamento de um objetivo aceito pelo builder.

    ``node_is_searchable(mass, params)``: o nó ainda pode ganhar com um corte.
    ``accepts_gain(gain)``: o melhor corte encontrado é válido.
    ``growth_priority(gain, mass, root_mass)``: valor comparado com
    ``min_impurity_decrease`` e usado para ordenar o best-first.
    ``search_hist``/``search_exact``: buscas com assinatura uniforme.
    """

    name: str
    supports_hist: bool
    supports_exact: bool
    supports_admissible_bound: bool
    experimental: bool
    node_is_searchable: Callable
    accepts_gain: Callable
    growth_priority: Callable
    search_hist: Callable
    search_exact: Callable


def _gini_node_is_searchable(mass, params):
    return gini(mass) > 0.0


def _gini_accepts_gain(gain):
    return bool(np.isfinite(gain))


def _gini_growth_priority(gain, mass, root_mass):
    # Redução de impureza ponderada pela massa relativa à raiz.
    return (mass.sum() / root_mass) * gain


def _gini_search_hist(X_binned, y, weights, sample_indices, start, end, edges,
                      n_classes, min_samples_leaf, feature_order, *, stopping,
                      gain_tolerance, stats, parent_mass, parallel,
                      prebuilt_histograms, edge_layout, params):
    return _search.find_best_split_hist(
        X_binned, y, weights, sample_indices, start, end, edges, n_classes,
        min_samples_leaf, feature_order, search_stopping=stopping,
        gain_tolerance=gain_tolerance, stats=stats, parent_mass=parent_mass,
        parallel=parallel, prebuilt_histograms=prebuilt_histograms,
        edge_layout=edge_layout)


def _gini_search_exact(X, y, weights, sample_indices, start, end, n_classes,
                       min_samples_leaf, feature_order, *, stopping,
                       gain_tolerance, stats, parent_mass, params):
    return _search.find_best_split_exact(
        X, y, weights, sample_indices, start, end, n_classes,
        min_samples_leaf, feature_order, search_stopping=stopping,
        gain_tolerance=gain_tolerance, stats=stats, parent_mass=parent_mass)


def _precision_node_is_searchable(mass, params):
    total = float(mass.sum())
    if total < params.min_support or mass[params.positive_class] <= 0.0:
        return False
    return precision_leaf_score(
        mass, params.positive_class, params.min_support) < params.min_precision


def _precision_accepts_gain(gain):
    # Sem melhora estrita da métrica não há corte.
    return bool(np.isfinite(gain) and gain > 0.0)


def _precision_growth_priority(gain, mass, root_mass):
    return gain


def _precision_search_hist(X_binned, y, weights, sample_indices, start, end,
                           edges, n_classes, min_samples_leaf, feature_order, *,
                           stopping, gain_tolerance, stats, parent_mass,
                           parallel, prebuilt_histograms, edge_layout, params):
    return _search.find_best_split_hist(
        X_binned, y, weights, sample_indices, start, end, edges, n_classes,
        min_samples_leaf, feature_order, search_stopping=stopping,
        gain_tolerance=gain_tolerance, stats=stats, parent_mass=parent_mass,
        parallel=parallel, prebuilt_histograms=prebuilt_histograms,
        edge_layout=edge_layout, objective="precision",
        positive_class=params.positive_class,
        min_precision=params.min_precision, min_support=params.min_support)


def _precision_search_exact(X, y, weights, sample_indices, start, end,
                            n_classes, min_samples_leaf, feature_order, *,
                            stopping, gain_tolerance, stats, parent_mass,
                            params):
    return _search.find_best_split_exact_precision(
        X, y, weights, sample_indices, start, end, n_classes,
        min_samples_leaf, feature_order, params.positive_class,
        params.min_precision, params.min_support, stats=stats,
        parent_mass=parent_mass)


GINI_OBJECTIVE = SplitObjectiveSpec(
    name="gini",
    supports_hist=True,
    supports_exact=True,
    supports_admissible_bound=True,
    experimental=False,
    node_is_searchable=_gini_node_is_searchable,
    accepts_gain=_gini_accepts_gain,
    growth_priority=_gini_growth_priority,
    search_hist=_gini_search_hist,
    search_exact=_gini_search_exact,
)

# Experimental: maximiza a melhor precision filha menos a do pai, com suporte
# mínimo. Não maximiza TP nem cobertura e tende a cortes extremos (como um
# peeling do PRIM). Ver PLANO_ARVORE_RAPIDA_FASE_3.md.
PRECISION_OBJECTIVE = SplitObjectiveSpec(
    name="precision",
    supports_hist=True,
    supports_exact=True,
    supports_admissible_bound=False,
    experimental=True,
    node_is_searchable=_precision_node_is_searchable,
    accepts_gain=_precision_accepts_gain,
    growth_priority=_precision_growth_priority,
    search_hist=_precision_search_hist,
    search_exact=_precision_search_exact,
)

_OBJECTIVES = {GINI_OBJECTIVE.name: GINI_OBJECTIVE,
               PRECISION_OBJECTIVE.name: PRECISION_OBJECTIVE}
_ENGINES = {"hist", "exact"}


def resolve_objective(objective):
    """Resolva um nome de objetivo sem aceitar fallback silencioso."""
    if not isinstance(objective, str) or objective not in _OBJECTIVES:
        available = ", ".join(sorted(_OBJECTIVES))
        raise ValueError(f"objective must be one of: {available}.")
    return _OBJECTIVES[objective]


def resolve_splitter_spec(splitter, objective="gini", search_stopping="bound"):
    """Valide a combinação motor + objetivo + parada uma vez por fit."""
    if splitter not in _ENGINES:
        raise ValueError("splitter must be 'hist' or 'exact'.")
    spec = resolve_objective(objective)
    if splitter == "hist" and not spec.supports_hist:
        raise ValueError(f"objective={objective!r} does not support splitter='hist'.")
    if splitter == "exact" and not spec.supports_exact:
        raise ValueError(f"objective={objective!r} does not support splitter='exact'.")
    if search_stopping not in ("off", "bound", "heuristic"):
        raise ValueError("search_stopping must be 'off', 'bound' or 'heuristic'.")
    if search_stopping == "bound" and not spec.supports_admissible_bound:
        raise ValueError(f"objective={objective!r} has no admissible bound.")
    return spec
