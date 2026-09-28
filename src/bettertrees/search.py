"""Best-cut search in one node: the Python -> Numba boundary.

Each function receives the node as a slice [start, end) of ``sample_indices``
and returns a ``Split``. The goal is one Python -> Numba crossing per node
(hist) or per feature (exact), never per candidate. Kernels are read as
globals of this module at every call, which lets tests instrument them.
"""

import numpy as np

from ._data import Split
from .kernels import (
    _build_all_histograms_feature_parallel,
    _build_all_histograms_row_major,
    _find_best_split_exact_gini_numba,
    _scan_exact_feature_precision_numba,
    _scan_histogram_feature_numba,
    _scan_histogram_feature_precision_numba,
    _scan_histograms_row_major_numba,
    _scan_histograms_row_major_parallel_numba,
    build_feature_histogram,
    build_feature_histogram_into,
    gini,
)


def hist_edge_layout(edges):
    """Devolva (max_bins, comprimentos finitos int64) para reutilizar no fit.

    Evita recalcular em Python, a cada nó, valores que dependem só dos bins.
    """
    lengths = np.ascontiguousarray(
        np.asarray([len(edge) for edge in edges], dtype=np.int64))
    max_bins = max((len(edge) + 2 for edge in edges), default=1)
    return max_bins, lengths


def _parent_mass_or_sum(parent_mass, y, weights, rows, n_classes):
    """Use a massa do nó já somada pelo builder; some só se ela faltar."""
    if parent_mass is not None:
        return np.ascontiguousarray(parent_mass, dtype=np.float64)
    total = np.zeros(n_classes, dtype=np.float64)
    for row in rows:
        total[y[row]] += weights[row]
    return total


def scan_histogram_feature(mass, count, edges, parent_mass, *, min_samples_leaf,
                            incumbent_gain, search_stopping="bound", gain_tolerance=0.0,
                            bound_interval=1, parent_gini=None):
    """Valide a entrada e encaminhe a varredura para o kernel Numba."""
    if search_stopping not in ("off", "bound"):
        raise ValueError("search_stopping must be 'off' or 'bound'.")
    if (isinstance(bound_interval, (bool, np.bool_))
            or not isinstance(bound_interval, (int, np.integer))
            or bound_interval < 1):
        raise ValueError("bound_interval must be a positive integer.")
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    count = np.ascontiguousarray(count, dtype=np.int64)
    parent_mass = np.ascontiguousarray(parent_mass, dtype=np.float64)
    if mass.ndim != 2 or count.ndim != 1 or mass.shape[0] != len(count):
        raise ValueError("Histogram with an inconsistent shape.")
    if mass.shape[1] != len(parent_mass):
        raise ValueError("Histogram and parent weights do not match.")
    if len(mass) == 0:
        return -1, False, -np.inf, 0, 0, 0
    parent_total = float(parent_mass.sum())
    if parent_total <= 0:
        return -1, False, -np.inf, 0, 0, 0
    parent_impurity = (float(gini(parent_mass)) if parent_gini is None
                       else float(parent_gini))
    return _scan_histogram_feature_numba(
        mass, count, parent_mass, int(min_samples_leaf), float(incumbent_gain),
        1 if search_stopping == "bound" else 0, float(gain_tolerance),
        int(bound_interval), parent_impurity)


def scan_histogram_feature_precision(mass, count, parent_mass, *,
                                     min_samples_leaf, positive_class,
                                     min_precision, min_support):
    """Varra um histograma para o objetivo de precisão, sem parada por bound."""
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    count = np.ascontiguousarray(count, dtype=np.int64)
    parent_mass = np.ascontiguousarray(parent_mass, dtype=np.float64)
    if mass.ndim != 2 or count.ndim != 1 or mass.shape[0] != len(count):
        raise ValueError("Histogram with an inconsistent shape.")
    if mass.shape[1] != len(parent_mass):
        raise ValueError("Histogram and parent weights do not match.")
    if (isinstance(positive_class, (bool, np.bool_))
            or not isinstance(positive_class, (int, np.integer))
            or not 0 <= positive_class < mass.shape[1]):
        raise ValueError("positive_class must be a valid class index.")
    return _scan_histogram_feature_precision_numba(
        mass, count, parent_mass, int(min_samples_leaf), int(positive_class),
        float(min_precision), float(min_support))


def find_best_split_exact(X, y, weights, sample_indices, start, end,
                          n_classes, min_samples_leaf, feature_order,
                          search_stopping="off", gain_tolerance=0.0,
                          stats=None, parent_mass=None):
    """Busque o melhor corte Gini exato do nó num único kernel Numba.

    ``parent_mass`` deve ser a massa do nó somada na ordem de
    ``sample_indices[start:end]``; ausente, é somada aqui.
    """
    if search_stopping not in ("off", "bound"):
        raise ValueError("search_stopping must be 'off' or 'bound'.")
    rows = np.asarray(sample_indices[start:end], dtype=np.int64)
    parent_mass = _parent_mass_or_sum(parent_mass, y, weights, rows, n_classes)
    if parent_mass.sum() <= 0.0:
        return Split(-1, np.nan, False, -np.inf, 0, -1)
    parent_impurity = float(gini(parent_mass))
    (feature, threshold, missing_left, gain, n_left, evaluated,
     missing_seen) = _find_best_split_exact_gini_numba(
        X, y, weights, rows, np.asarray(feature_order, dtype=np.int64),
        parent_mass, int(min_samples_leaf), parent_impurity)
    best = (Split(-1, np.nan, False, -np.inf, 0, -1) if feature < 0 else
            Split(int(feature), float(threshold), bool(missing_left),
                  float(gain), int(n_left), -1))

    if stats is not None:
        stats["exact_candidates_evaluated"] = (
            stats.get("exact_candidates_evaluated", 0) + evaluated)
        stats["missing_rows_scanned"] = (
            stats.get("missing_rows_scanned", 0) + missing_seen)
    return best


def find_best_split_exact_precision(
        X, y, weights, sample_indices, start, end, n_classes,
        min_samples_leaf, feature_order, positive_class, min_precision,
        min_support, stats=None, parent_mass=None):
    """Ordene em NumPy e avalie precision em um kernel por feature.

    Experimental: maximiza a melhor precision filha menos a do pai (com
    suporte mínimo), não TP nem cobertura.
    """
    rows = np.asarray(sample_indices[start:end], dtype=np.int64)
    parent_mass = _parent_mass_or_sum(parent_mass, y, weights, rows, n_classes)
    best = Split(-1, np.nan, False, -np.inf, 0, -1)
    evaluated = 0

    for feature in np.asarray(feature_order, dtype=np.int64):
        values = X[rows, feature]
        finite_mask = ~np.isnan(values)
        finite_rows = rows[finite_mask]
        missing_rows = rows[~finite_mask]
        missing_mass = np.zeros(n_classes, dtype=np.float64)
        for row in missing_rows:
            missing_mass[y[row]] += weights[row]
        order = np.argsort(X[finite_rows, feature], kind="mergesort")
        ordered_rows = finite_rows[order]
        threshold, missing_left, gain, n_left, seen = _scan_exact_feature_precision_numba(
            X[ordered_rows, feature], y[ordered_rows], weights[ordered_rows],
            parent_mass, missing_mass, len(rows), min_samples_leaf,
            positive_class, min_support)
        evaluated += seen
        if gain > best.gain and gain > 0.0:
            best = Split(int(feature), float(threshold), bool(missing_left),
                         float(gain), int(n_left), -1)

    if stats is not None:
        stats["exact_candidates_evaluated"] = (
            stats.get("exact_candidates_evaluated", 0) + evaluated)
    return best


def _find_best_split_hist_feature_major(X_binned, y, weights, sample_indices, start, end,
                         edges, n_classes, min_samples_leaf, feature_order,
                         search_stopping="bound", gain_tolerance=0.0,
                         stats=None, bound_interval=1, reuse_histograms=False,
                         parent_mass=None, parallel=False, prebuilt_histograms=None,
                         objective="gini", positive_class=0, min_precision=0.0,
                         min_support=0.0, edge_layout=None):
    """Busque o melhor corte de Gini em histogramas por feature.

    ``edge_layout`` é aceito para ter a mesma assinatura de
    ``find_best_split_hist`` e ignorado: cada histograma é montado por feature.

    Acumular SEPARADAMENTE contagem de linhas e massas ponderadas; prefixos
    avaliam cortes entre bins finitos e as duas direções de NaN (bin 0).
    Aplicar suporte, massa, desempates e corte só de missing como no exato.
    Traduzir bin_threshold=b para edges[feature][b-1] na escala original;
    corte apenas missing usa threshold=+inf e bin_threshold=n_bins_finitos.
    Custo esperado por nó: O(n_node*p + p*B*K), scratch O(B*K) por feature.
    Não criar máscara ou submatriz por candidato. Esta versão mantém a tupla
    de edges no lado Python; compactação para um kernel único é otimização
    posterior.
    """
    if search_stopping not in ("off", "bound"):
        raise ValueError("search_stopping must be 'off' or 'bound'.")
    rows = np.asarray(sample_indices[start:end], dtype=np.int64)
    if parent_mass is None:
        parent_mass = np.zeros(n_classes, dtype=np.float64)
        for row in rows:
            parent_mass[y[row]] += weights[row]
    else:
        parent_mass = np.ascontiguousarray(parent_mass, dtype=np.float64)
    best = Split(-1, np.nan, False, -np.inf, 0, -1)
    parent_gini = float(gini(parent_mass))
    evaluated = 0
    skipped = 0
    histograms = 0
    mass_buffer = None
    count_buffer = None
    if reuse_histograms:
        max_bins = max((len(edge) + 2 for edge in edges), default=1)
        mass_buffer = np.empty((max_bins, n_classes), dtype=np.float64)
        count_buffer = np.empty(max_bins, dtype=np.int64)
    prebuilt_mass = None if prebuilt_histograms is None else prebuilt_histograms[0]
    prebuilt_count = None if prebuilt_histograms is None else prebuilt_histograms[1]
    for feature in np.asarray(feature_order, dtype=np.int64):
        n_bins = int(len(edges[feature]) + 2)
        if prebuilt_histograms is not None:
            mass = prebuilt_mass[feature, :n_bins]
            count = prebuilt_count[feature, :n_bins]
        elif reuse_histograms:
            mass, count = build_feature_histogram_into(
                X_binned[:, feature], y, weights, sample_indices, start, end,
                n_bins, n_classes, mass_buffer, count_buffer)
        else:
            mass, count = build_feature_histogram(
                X_binned[:, feature], y, weights, sample_indices, start, end,
                n_bins, n_classes)
        histograms += 1
        if objective == "precision":
            result = scan_histogram_feature_precision(
                mass, count, parent_mass,
                min_samples_leaf=min_samples_leaf,
                positive_class=positive_class,
                min_precision=min_precision,
                min_support=min_support)
        else:
            result = scan_histogram_feature(
                mass, count, edges[feature], parent_mass,
                min_samples_leaf=min_samples_leaf,
                incumbent_gain=best.gain,
                search_stopping=search_stopping,
                gain_tolerance=gain_tolerance, bound_interval=bound_interval,
                parent_gini=parent_gini)
        bin_threshold, missing_left, gain, n_left, seen, omitted = result
        evaluated += seen
        skipped += omitted
        if bin_threshold < 0 or not np.isfinite(gain):
            continue
        n_finite_bins = len(edges[feature]) + 1
        threshold = (np.inf if bin_threshold == n_finite_bins
                     else float(edges[feature][bin_threshold - 1]))
        if gain > best.gain:
            best = Split(int(feature), threshold, bool(missing_left),
                         float(gain), int(n_left), int(bin_threshold))
    if stats is not None:
        stats["histograms"] = stats.get("histograms", 0) + histograms
        stats["hist_candidates_evaluated"] = stats.get("hist_candidates_evaluated", 0) + evaluated
        stats["hist_candidates_skipped"] = stats.get("hist_candidates_skipped", 0) + skipped
    return best


def _build_histograms_for_node(X_binned, y, weights, sample_indices, start, end,
                               edges, n_classes, parallel=False,
                               edge_layout=None):
    """Acumule o pacote completo de histogramas de um nó.

    Função Python deliberadamente fora do kernel: permite entregar o pacote
    ao filho e medir a variante pai-filho sem duplicar a lógica de layout.
    """
    n_features = X_binned.shape[1]
    max_bins = (hist_edge_layout(edges) if edge_layout is None
                else edge_layout)[0]
    builder = (_build_all_histograms_feature_parallel
               if parallel else _build_all_histograms_row_major)
    return builder(X_binned, y, weights, sample_indices, start, end,
                   n_features, max_bins, n_classes)


def find_best_split_hist(X_binned, y, weights, sample_indices, start, end,
                         edges, n_classes, min_samples_leaf, feature_order,
                         search_stopping="bound", gain_tolerance=0.0,
                         stats=None, bound_interval=1, reuse_histograms=False,
                         parent_mass=None, parallel=False, prebuilt_histograms=None,
                         objective="gini", positive_class=0, min_precision=0.0,
                         min_support=0.0, feature_subset=False,
                         edge_layout=None):
    """Busque o melhor corte com histogramas acumulados em ordem de linhas.

    ``feature_order`` pode ser um subconjunto das features (limite por
    caminho): os histogramas continuam acumulados para todas as colunas em
    ordem de linhas e só o subconjunto é varrido. ``feature_subset=True``
    força o caminho por feature, mantido para comparações reproduzíveis.
    ``edge_layout`` vem de ``hist_edge_layout`` e evita recalcular o layout.
    O objetivo ``precision`` é experimental: maximiza a melhor precision filha
    menos a do pai, não TP nem cobertura.
    """
    if edge_layout is None:
        edge_layout = hist_edge_layout(edges)
    max_bins, finite_edge_lengths = edge_layout
    if feature_subset:
        # A path-specific feature budget must also reduce histogram work.  The
        # row-major builder intentionally remains the fast all-feature path;
        # once features are filtered, build only the surviving columns.
        return _find_best_split_hist_feature_major(
            X_binned, y, weights, sample_indices, start, end, edges,
            n_classes, min_samples_leaf, feature_order,
            search_stopping=search_stopping, gain_tolerance=gain_tolerance,
            stats=stats, bound_interval=bound_interval,
            reuse_histograms=False,
            parent_mass=parent_mass, parallel=False,
            prebuilt_histograms=prebuilt_histograms,
            objective=objective, positive_class=positive_class,
            min_precision=min_precision, min_support=min_support)
    if prebuilt_histograms is not None:
        mass, count = prebuilt_histograms
    elif reuse_histograms:
        return _find_best_split_hist_feature_major(
            X_binned, y, weights, sample_indices, start, end, edges,
            n_classes, min_samples_leaf, feature_order,
            search_stopping=search_stopping, gain_tolerance=gain_tolerance,
            stats=stats, bound_interval=bound_interval, reuse_histograms=True,
            parent_mass=parent_mass, objective=objective,
            positive_class=positive_class, min_precision=min_precision,
            min_support=min_support)
    elif parallel:
        n_features = X_binned.shape[1]
        mass, count = _build_all_histograms_feature_parallel(
            X_binned, y, weights, sample_indices, start, end,
            n_features, max_bins, n_classes)
    else:
        n_features = X_binned.shape[1]
        mass, count = _build_all_histograms_row_major(
            X_binned, y, weights, sample_indices, start, end,
            n_features, max_bins, n_classes)
    if search_stopping not in ("off", "bound"):
        raise ValueError("search_stopping must be 'off' or 'bound'.")
    if parent_mass is None:
        parent_mass = np.zeros(n_classes, dtype=np.float64)
        for row in sample_indices[start:end]:
            parent_mass[y[row]] += weights[row]
    else:
        parent_mass = np.ascontiguousarray(parent_mass, dtype=np.float64)
    if objective == "precision":
        best = Split(-1, np.nan, False, -np.inf, 0, -1)
        evaluated = 0
        for feature in np.asarray(feature_order, dtype=np.int64):
            n_bins = int(len(edges[feature]) + 2)
            result = scan_histogram_feature_precision(
                mass[feature, :n_bins], count[feature, :n_bins], parent_mass,
                min_samples_leaf=min_samples_leaf,
                positive_class=positive_class,
                min_precision=min_precision,
                min_support=min_support)
            bin_threshold, missing_left, gain, n_left, seen, _ = result
            evaluated += seen
            if bin_threshold < 0 or not np.isfinite(gain):
                continue
            n_finite_bins = len(edges[feature]) + 1
            threshold = (np.inf if bin_threshold == n_finite_bins
                         else float(edges[feature][bin_threshold - 1]))
            if gain > best.gain:
                best = Split(int(feature), threshold, bool(missing_left),
                             float(gain), int(n_left), int(bin_threshold))
        if stats is not None:
            stats["histograms"] = stats.get("histograms", 0) + len(edges)
            stats["hist_candidates_evaluated"] = (
                stats.get("hist_candidates_evaluated", 0) + evaluated)
        return best
    parent_gini = float(gini(parent_mass))
    feature_order = np.ascontiguousarray(feature_order, dtype=np.int64)
    scan_kernel = (_scan_histograms_row_major_parallel_numba
                   if parallel else _scan_histograms_row_major_numba)
    (best_feature, bin_threshold, missing_left, gain, n_left,
     evaluated, skipped) = scan_kernel(
        mass, count, finite_edge_lengths, feature_order, parent_mass,
        int(min_samples_leaf), 1 if search_stopping == "bound" else 0,
        float(gain_tolerance), int(bound_interval), parent_gini)
    if best_feature < 0 or not np.isfinite(gain):
        best = Split(-1, np.nan, False, -np.inf, 0, -1)
    else:
        n_finite_bins = len(edges[best_feature]) + 1
        threshold = (np.inf if bin_threshold == n_finite_bins
                     else float(edges[best_feature][bin_threshold - 1]))
        best = Split(int(best_feature), threshold, bool(missing_left),
                     float(gain), int(n_left), int(bin_threshold))
    if stats is not None:
        if prebuilt_histograms is None:
            stats["histograms"] = stats.get("histograms", 0) + n_features
        stats["hist_candidates_evaluated"] = stats.get("hist_candidates_evaluated", 0) + evaluated
        stats["hist_candidates_skipped"] = stats.get("hist_candidates_skipped", 0) + skipped
    return best
