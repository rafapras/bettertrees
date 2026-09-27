"""Aprendizado e aplicação dos limites de bins (NaN -> bin 0).

Kernels Numba deste módulo não chamam kernels de outros módulos: o cache do
Numba invalida por arquivo e não rastreia dependências entre módulos.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange


def _count_balanced_cut_positions(counts, max_bins):
    """Posições i (corte entre o i-ésimo e o (i+1)-ésimo valor distinto).

    Quantis puros desperdiçam orçamento quando poucos valores concentram a
    massa: vários quantis caem no mesmo valor e ``unique`` os funde, e a cauda
    de valores distintos fica com poucos bins. Aqui, valores com contagem >=
    massa média por bin livre viram bins próprios (iterando, pois cada um
    libera orçamento), e o resto é repartido por contagem entre as sequências
    de valores leves que ficam entre eles. Sem empates (contagens 1), reproduz
    bins de mesma contagem, como os quantis.
    """
    n_values = len(counts)
    heavy = np.zeros(n_values, dtype=bool)
    while True:
        free_bins = max_bins - int(heavy.sum())
        rest = float(counts[~heavy].sum())
        if free_bins <= 0 or rest <= 0:
            break
        new = ~heavy & (counts >= rest / free_bins)
        if not new.any():
            break
        heavy |= new
    free_bins = max(1, max_bins - int(heavy.sum()))
    target = max(float(counts[~heavy].sum()) / free_bins, 1.0)
    positions = set()
    for i in np.flatnonzero(heavy):
        if i > 0:
            positions.add(int(i) - 1)
        if i < n_values - 1:
            positions.add(int(i))
    # Sequências de valores leves entre valores pesados (ou nas pontas).
    start = 0
    while start < n_values:
        if heavy[start]:
            start += 1
            continue
        stop = start
        while stop < n_values and not heavy[stop]:
            stop += 1
        run = counts[start:stop]
        run_bins = int(round(float(run.sum()) / target))
        if run_bins > 1 and stop - start > 1:
            cumulative = np.cumsum(run)
            goals = cumulative[-1] * np.arange(1, run_bins) / run_bins
            inner = np.unique(np.searchsorted(cumulative, goals, side="left"))
            positions.update(int(start + i) for i in inner if start + i < stop - 1)
        start = stop
    positions = np.array(sorted(positions), dtype=np.int64)
    # Arredondamentos podem passar do orçamento: funda o par de bins
    # vizinhos de menor contagem até caber.
    while len(positions) > max_bins - 1:
        bounds = np.r_[-1, positions, n_values - 1]
        cumulative = np.r_[0, np.cumsum(counts)]
        sizes = cumulative[bounds[1:] + 1] - cumulative[bounds[:-1] + 1]
        merged = sizes[:-1] + sizes[1:]
        positions = np.delete(positions, int(np.argmin(merged)))
    return positions


def _fit_bin_edges_column(col, max_bins, quantiles):
    """Aprenda cortes de uma coluna; função independente para paralelismo."""
    finite = col[~np.isnan(col)]
    unique, counts = np.unique(finite, return_counts=True)
    unique = unique.astype(np.float64)
    # O bin 0 fica reservado para NaN e uint8 suporta no máximo ids 1..255.
    # Portanto, só preservar todos os intervalos quando eles cabem no
    # orçamento; cardinalidade alta passa pela repartição por contagem.
    if len(unique) <= max_bins:
        cuts = unique[:-1] / 2 + unique[1:] / 2
    else:
        positions = _count_balanced_cut_positions(counts, max_bins)
        cuts = unique[positions] / 2 + unique[positions + 1] / 2
    return np.ascontiguousarray(cuts, dtype=np.float64)


def fit_bin_edges(X, max_bins=255, n_jobs=1):
    """Aprenda limites SOMENTE no treino validado, ignorando NaN.

    Retorna tupla de vetores float64 crescentes, com até max_bins-1 cortes.
    Usa pontos médios quando há poucos valores únicos, preservando colunas
    indicadoras; caso contrário, quantis não ponderados e sem amostragem.
    Constantes e colunas só com NaN recebem vetor vazio. max_bins em [2,255]
    permite uint8 com bin 0 reservado para NaN. É uma referência funcional:
    unique/quantile por coluna pode ser um gargalo a medir antes de otimizar.
    Pesos afetam impureza, mas não a localização destes quantis.
    """
    if (isinstance(max_bins, (bool, np.bool_))
            or not isinstance(max_bins, (int, np.integer))
            or not 2 <= max_bins <= 255):
        raise ValueError("max_bins deve ser inteiro entre 2 e 255.")
    if (isinstance(n_jobs, (bool, np.bool_))
            or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1):
        raise ValueError("n_jobs deve ser inteiro positivo.")
    quantiles = np.arange(1, max_bins) / max_bins
    columns = tuple(X[:, j] for j in range(X.shape[1]))
    if n_jobs == 1:
        return tuple(_fit_bin_edges_column(col, max_bins, quantiles)
                     for col in columns)
    with ThreadPoolExecutor(max_workers=int(n_jobs)) as pool:
        return tuple(pool.map(
            lambda col: _fit_bin_edges_column(col, max_bins, quantiles),
            columns))


@njit(cache=True)
def _count_binary_values(values):
    """Conte zeros em uma coluna 0/1; devolva -1 ao ver outro valor."""
    zeros = 0
    for value in values:
        if value == 0:
            zeros += 1
        elif value != 1:
            return -1
    return zeros


def _fit_bin_edges_binary(X, max_bins=255):
    """Aprenda limites com quantis idênticos à referência em colunas 0/1.

    O atalho evita a partição de grandes colunas indicadoras. Nas demais,
    mantém o caminho original; `fit_bin_edges` segue como baseline de produção.
    """
    if (isinstance(max_bins, (bool, np.bool_))
            or not isinstance(max_bins, (int, np.integer))
            or not 2 <= max_bins <= 255):
        raise ValueError("max_bins deve ser inteiro entre 2 e 255.")
    edges = []
    quantiles = np.arange(1, max_bins) / max_bins
    for col in X.T:
        finite = col[~np.isnan(col)]
        unique = np.unique(finite).astype(np.float64)
        if len(unique) <= max_bins:
            cuts = unique[:-1] / 2 + unique[1:] / 2
        else:
            zeros = _count_binary_values(finite)
            if zeros >= 0:
                if zeros == 0 or zeros == len(finite):
                    cuts = np.empty(0, dtype=np.float64)
                else:
                    positions = (len(finite) - 1) * quantiles
                    lower = np.floor(positions).astype(np.int64)
                    upper = np.ceil(positions).astype(np.int64)
                    cuts = np.unique(np.where(upper < zeros, 0.0,
                                              np.where(lower >= zeros, 1.0,
                                                       positions - lower)))
                    cuts = cuts[cuts < 1.0]
            else:
                finite_min = float(finite.min())
                finite_max = float(finite.max())
                cuts = np.unique(np.quantile(finite, quantiles))
                cuts = cuts[(cuts >= finite_min) & (cuts < finite_max)]
        edges.append(np.ascontiguousarray(cuts, dtype=np.float64))
    return tuple(edges)


def transform_bins(X, edges):
    """Aplique limites congelados e devolva uint8 C-contiguous de shape X.

    NaN -> 0; finitos -> 1..B. Igualdade com um limite fica no bin inferior
    (searchsorted side='left'), exatamente como X <= threshold na previsão.
    Extremos novos entram nos bins externos; não recalcular quantis.
    Contrato interno: X já validado e edges gerado por fit_bin_edges.
    """
    if len(edges) != X.shape[1]:
        raise ValueError("Uma lista de limites é necessária por coluna.")
    result = np.empty(X.shape, dtype=np.uint8)
    for j, cuts in enumerate(edges):
        result[:, j] = np.searchsorted(cuts, X[:, j], side="left") + 1
        result[np.isnan(X[:, j]), j] = 0
    return result


@njit(cache=True)
def _transform_bins_row_major_kernel(X, padded_edges, lengths):
    """Aplique side='left' por linha, respeitando o layout C de X e da saida."""
    result = np.empty(X.shape, dtype=np.uint8)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            value = X[i, j]
            if np.isnan(value):
                result[i, j] = 0
            else:
                lo, hi = 0, lengths[j]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if padded_edges[j, mid] < value:
                        lo = mid + 1
                    else:
                        hi = mid
                result[i, j] = lo + 1
    return result


@njit(cache=True, parallel=True)
def _transform_bins_row_major_parallel_kernel(X, padded_edges, lengths):
    """Transformação row-major paralela por linhas, sem alterar os cortes."""
    result = np.empty(X.shape, dtype=np.uint8)
    for i in prange(X.shape[0]):
        for j in range(X.shape[1]):
            value = X[i, j]
            if np.isnan(value):
                result[i, j] = 0
            else:
                lo, hi = 0, lengths[j]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if padded_edges[j, mid] < value:
                        lo = mid + 1
                    else:
                        hi = mid
                result[i, j] = lo + 1
    return result


def transform_bins_row_major(X, edges, n_jobs=1):
    """Variante row-major; preserva exatamente os bins da referencia NumPy.

    A preparacao dos limites entra no tempo medido. `transform_bins` permanece
    disponivel como baseline comparavel; ambos exigem X previamente validado.
    """
    if len(edges) != X.shape[1]:
        raise ValueError("Uma lista de limites é necessária por coluna.")
    if (isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer))
            or n_jobs < 1):
        raise ValueError("n_jobs deve ser inteiro positivo.")
    lengths = np.fromiter((len(cuts) for cuts in edges), count=len(edges), dtype=np.int64)
    padded = np.zeros((len(edges), int(lengths.max(initial=0))), dtype=np.float64)
    for j, cuts in enumerate(edges):
        padded[j, :len(cuts)] = cuts
    if n_jobs == 1:
        return _transform_bins_row_major_kernel(X, padded, lengths)
    return _transform_bins_row_major_parallel_kernel(X, padded, lengths)
