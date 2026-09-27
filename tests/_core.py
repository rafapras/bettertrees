"""Fachada de compatibilidade: reexporta os módulos do núcleo.

O código vive em módulos por responsabilidade:

- ``_data``: ``NodeArrays``, ``Split``, validação e alocação de nós;
- ``bins``: aprendizado e aplicação dos limites de bins;
- ``kernels``: todos os kernels Numba (um módulo só, por causa do cache);
- ``search``: busca do melhor corte em um nó (fronteira Python -> Numba);
- ``builder``: crescimento depth-first/best-first sobre o contrato do objetivo;
- ``postprocess``: poda, probabilidades e projeção monotônica;
- ``_reference``: implementações Python usadas só em testes.

Scripts antigos continuam importando daqui. Código novo deve importar do
módulo de origem; para instrumentar uma função em teste, substitua-a no
módulo que a chama (por exemplo ``search`` ou ``builder``), não aqui.
"""

import sys as _sys
import types as _types

import _reference
from _reference import (
    _find_best_split_exact_precision_reference,
    _find_best_split_exact_reference,
    _scan_histogram_feature_reference,
)

from bettertrees import _data
from bettertrees import bins as _bins
from bettertrees import builder as _builder
from bettertrees import kernels as _kernels
from bettertrees import postprocess as _postprocess
from bettertrees import search as _search
from bettertrees import splitters as _splitters
from bettertrees._data import NodeArrays, Split, allocate_nodes, prepare_training_data, validate_X
from bettertrees.bins import (
    _count_binary_values,
    _fit_bin_edges_binary,
    _fit_bin_edges_column,
    _transform_bins_row_major_kernel,
    _transform_bins_row_major_parallel_kernel,
    fit_bin_edges,
    transform_bins,
    transform_bins_row_major,
)
from bettertrees.builder import _grow_tree, grow_tree_exact, grow_tree_hist
from bettertrees.kernels import (
    _build_all_histograms_feature_parallel,
    _build_all_histograms_row_major,
    _find_best_split_exact_gini_numba,
    _gini_scalar,
    _hist_candidate_gain,
    _node_class_mass,
    _scan_exact_feature_gini_numba,
    _scan_exact_feature_precision_numba,
    _scan_histogram_feature_numba,
    _scan_histogram_feature_numba_scratch,
    _scan_histogram_feature_precision_numba,
    _scan_histograms_row_major_numba,
    _scan_histograms_row_major_parallel_numba,
    apply_nodes,
    build_feature_histogram,
    build_feature_histogram_into,
    cannot_improve,
    gini,
    partition_samples,
    precision_leaf_score,
    precision_split_gain,
    remaining_gain_upper_bound,
)
from bettertrees.postprocess import (
    finite_leaf_regions,
    predict_proba_nodes,
    project_monotonic_leaf_probabilities,
    prune_tree_cost_complexity,
)
from bettertrees.search import (
    _build_histograms_for_node,
    _find_best_split_hist_feature_major,
    find_best_split_exact,
    find_best_split_exact_precision,
    find_best_split_hist,
    hist_edge_layout,
    scan_histogram_feature,
    scan_histogram_feature_precision,
)
from bettertrees.splitters import resolve_splitter_spec

_SOURCE_MODULES = (_data, _reference, _bins, _builder, _kernels,
                   _postprocess, _search, _splitters)
_MISSING = object()


class _CompatFacade(_types.ModuleType):
    """Repasse ``core.nome = valor`` aos módulos que usam o mesmo objeto.

    Quando tudo vivia em ``core.py``, trocar ``core.find_best_split_hist``
    (ablações, perfis, testes) alterava a função chamada pelo builder. Sem
    este repasse, a troca pararia de valer em silêncio e o script mediria o
    baseline achando que mede a variante.
    """

    def __setattr__(self, name, value):
        current = self.__dict__.get(name, _MISSING)
        if current is not _MISSING:
            for module in _SOURCE_MODULES:
                if module.__dict__.get(name, _MISSING) is current:
                    setattr(module, name, value)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _CompatFacade
