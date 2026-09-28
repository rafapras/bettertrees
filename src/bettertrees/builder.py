"""Crescimento da árvore: política de expansão sobre o contrato do objetivo.

O builder não conhece objetivos pelo nome. Tudo que depende do objetivo
(quando um nó pode ser dividido, qual busca usar, se um ganho é aceito e a
prioridade no best-first) vem do ``SplitObjectiveSpec`` resolvido uma vez
por fit em ``splitters.py``.
"""

import heapq
from time import perf_counter

import numpy as np

from ._data import NodeArrays, allocate_nodes
from .kernels import _node_class_mass, local_midpoint_threshold, partition_samples
from .search import _build_histograms_for_node, hist_edge_layout
from .splitters import ObjectiveParams, resolve_splitter_spec


def grow_tree_exact(X, y, weights, *, n_classes, max_depth, min_samples_leaf,
                    max_leaf_nodes, min_impurity_decrease, feature_order,
                    stopping="bound", gain_tolerance=0.0, stats=None,
                    objective="gini", positive_class=0, min_precision=0.0,
                    min_support=0.0, max_feature_repeats=None):
    """Cresça uma árvore exata e retorne arrays recortados aos nós usados.

    Uma permutação int64 representa todos os recortes de amostras. Escolher
    motor antes do JIT. Sem max_leaf_nodes: pilha iterativa; com limite de
    folhas: best-first por redução ponderada GLOBAL (compatível com o orçamento
    do comparador). Raiz tem profundidade 0; folha pura ou sem split encerra.
    Para Gini, min_impurity_decrease compara W_node/W_root * gain; para
    precision, compara o aumento da melhor precision filha sobre a do pai.
    Aceitar ganho zero no Gini permite estruturas como XOR; precision só
    aceita ganho positivo, pois não há melhora de métrica.
    Nunca dividir por peso zero nem exceder o orçamento em folhas/profundidade.
    O motor exato sempre varre todos os candidatos: stopping='bound' é
    aceito por compatibilidade com o parâmetro padrão, mas não aplica bound
    nem gain_tolerance. A API registra o modo efetivo em fit_stats_.
    min_impurity_decrease continua controlando apenas o crescimento.
    """
    return _grow_tree(
        X, y, weights, n_classes=n_classes, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease, feature_order=feature_order,
        stopping=stopping, gain_tolerance=gain_tolerance, stats=stats,
        splitter="exact", objective=objective, X_binned=None, edges=None,
        positive_class=positive_class, min_precision=min_precision,
        min_support=min_support, max_feature_repeats=max_feature_repeats)


def grow_tree_hist(X, X_binned, y, weights, edges, *, n_classes, max_depth,
                   min_samples_leaf, max_leaf_nodes, min_impurity_decrease,
                   feature_order, stopping="bound", gain_tolerance=0.0,
                   stats=None, objective="gini", parallel=False,
                   reuse_parent_histograms=False, positive_class=0,
                   min_precision=0.0, min_support=0.0,
                   max_feature_repeats=None):
    """Cresça uma árvore por histogramas e retorne arrays recortados.

    X e bins devem representar as MESMAS linhas na MESMA ordem. X permite
    particionar índices e armazenar cortes na escala original. Discretização
    integra o fit completo; nunca medir só este kernel como tempo de treino.
    O caminho de referência acumula cada nó. A variante experimental
    ``reuse_parent_histograms`` só funciona em crescimento depth-first com
    pesos unitários e deriva o filho maior por subtração do pai.
    """
    X_binned = np.asarray(X_binned)
    if X_binned.shape != X.shape or X_binned.dtype != np.uint8:
        raise ValueError("X_binned must have the same shape as X and dtype uint8.")
    if len(edges) != X.shape[1]:
        raise ValueError("edges must have one entry per feature.")
    return _grow_tree(
        X, y, weights, n_classes=n_classes, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease, feature_order=feature_order,
        stopping=stopping, gain_tolerance=gain_tolerance, stats=stats,
        splitter="hist", objective=objective, X_binned=X_binned, edges=edges,
        parallel=parallel, reuse_parent_histograms=reuse_parent_histograms,
        positive_class=positive_class, min_precision=min_precision,
        min_support=min_support, max_feature_repeats=max_feature_repeats)


def _grow_tree(X, y, weights, *, n_classes, max_depth, min_samples_leaf,
               max_leaf_nodes, min_impurity_decrease, feature_order, stopping,
               gain_tolerance, stats, splitter, objective, X_binned, edges,
               parallel=False, reuse_parent_histograms=False, positive_class=0,
               min_precision=0.0, min_support=0.0,
               max_feature_repeats=None):
    """Implementação comum dos dois builders; índices são a única partição.

    Sem ``max_leaf_nodes``: pilha depth-first. Com orçamento de folhas:
    best-first por ``spec.growth_priority``, desempate pelo menor id de nó.
    """
    if stopping not in ("off", "bound"):
        raise ValueError("stopping must be 'off' or 'bound'.")
    if reuse_parent_histograms and splitter != "hist":
        raise ValueError("Parent-child histogram reuse requires splitter='hist'.")
    if reuse_parent_histograms and max_leaf_nodes is not None:
        raise ValueError("Parent-child histogram reuse (experimental) requires max_leaf_nodes=None.")
    if reuse_parent_histograms and objective != "gini":
        raise ValueError("Parent-child histogram reuse (experimental) requires objective='gini'.")
    if max_feature_repeats is not None and max_feature_repeats < 1:
        raise ValueError("max_feature_repeats must be a positive integer or None.")
    spec = resolve_splitter_spec(splitter, objective, stopping)
    params = ObjectiveParams.from_values(positive_class, min_precision, min_support)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    if len(X) != len(y) or len(y) != len(weights) or len(X) == 0:
        raise ValueError("Training data is empty or misaligned.")
    if X.ndim != 2 or y.ndim != 1 or n_classes < 1:
        raise ValueError("Invalid shape for the tree builder.")
    feature_order = np.ascontiguousarray(feature_order, dtype=np.int64)
    edge_layout = hist_edge_layout(edges) if splitter == "hist" else None

    active_rows = len(X)
    capacity = 2 * active_rows - 1
    if max_depth is not None and max_depth + 1 < capacity.bit_length():
        capacity = min(capacity, (1 << (max_depth + 1)) - 1)
    if max_leaf_nodes is not None:
        capacity = min(capacity, 2 * max_leaf_nodes - 1)
    nodes = allocate_nodes(capacity, n_classes)
    sample_indices = np.arange(active_rows, dtype=np.int64)
    root_mass = float(weights.sum())
    if root_mass <= 0:
        raise ValueError("The total training weight must be positive.")
    if reuse_parent_histograms and not np.all(weights == 1.0):
        raise ValueError("Parent-child histogram reuse (experimental) requires unit sample weights.")
    local_stats = stats if stats is not None else {}
    local_stats.setdefault("nodes_visited", 0)
    local_stats.setdefault("nodes_split", 0)
    local_stats.setdefault("candidate_evaluated", 0)
    local_stats.setdefault("candidate_skipped", 0)
    local_stats.setdefault("node_size_counts", {})
    local_stats.setdefault("feature_repetition_limit", max_feature_repeats)
    local_stats.setdefault("features_skipped_by_path", 0)

    def is_terminal(start, end, depth, mass):
        return (end - start < 2 * min_samples_leaf
                or (max_depth is not None and depth >= max_depth)
                or not spec.node_is_searchable(mass, params))

    def leaf(work, mass, weighted_gain=-np.inf):
        return dict(work=work, mass=mass, split=None,
                    weighted_gain=weighted_gain, histograms=None)

    def evaluate(work, node_histograms=None, precomputed_mass=None):
        start, end, depth, node_id, path_feature_counts = work
        mass = (_node_class_mass(y, weights, sample_indices, start, end, n_classes)
                if precomputed_mass is None else precomputed_mass)
        nodes.class_weight[node_id] = mass
        nodes.n_samples[node_id] = end - start
        local_stats["nodes_visited"] += 1
        bucket = "small" if end - start < 256 else ("medium" if end - start < 4096 else "large")
        sizes = local_stats["node_size_counts"]
        sizes[bucket] = sizes.get(bucket, 0) + 1
        if is_terminal(start, end, depth, mass):
            return leaf(work, mass)
        search_start = perf_counter()
        if max_feature_repeats is None:
            search_features = feature_order
        else:
            # Subconjunto na ordem de desempate; o caminho row-major acumula
            # todas as colunas e varre só estas, então não há penalidade
            # quando o limite ainda não restringe nada.
            search_features = feature_order[
                path_feature_counts[feature_order] < max_feature_repeats]
            local_stats["features_skipped_by_path"] += int(
                len(feature_order) - len(search_features))
            if len(search_features) == 0:
                return leaf(work, mass)
        if splitter == "hist":
            # Em árvores profundas há muitos nós médios; 256 é o ponto de
            # corte experimental que evita paralelizar folhas pequenas sem
            # deixar o trabalho relevante serializado.
            node_parallel = parallel and (end - start >= 256)
            if reuse_parent_histograms and node_histograms is None:
                node_histograms = _build_histograms_for_node(
                    X_binned, y, weights, sample_indices, start, end,
                    edges, n_classes, parallel=node_parallel,
                    edge_layout=edge_layout)
            split = spec.search_hist(
                X_binned, y, weights, sample_indices, start, end, edges,
                n_classes, min_samples_leaf, search_features,
                stopping=stopping, gain_tolerance=gain_tolerance,
                stats=local_stats, parent_mass=mass, parallel=node_parallel,
                prebuilt_histograms=(node_histograms if reuse_parent_histograms
                                     else None),
                edge_layout=edge_layout, params=params)
            if node_parallel:
                local_stats["parallel_nodes"] = local_stats.get("parallel_nodes", 0) + 1
        else:
            split = spec.search_exact(
                X, y, weights, sample_indices, start, end, n_classes,
                min_samples_leaf, search_features, stopping=stopping,
                gain_tolerance=gain_tolerance, stats=local_stats,
                parent_mass=mass, params=params)
        local_stats["search_seconds"] = (local_stats.get("search_seconds", 0.0)
            + (perf_counter() - search_start))
        local_stats["candidate_evaluated"] = (
            local_stats.get("hist_candidates_evaluated", 0)
            + local_stats.get("exact_candidates_evaluated", 0))
        local_stats["candidate_skipped"] = local_stats.get("hist_candidates_skipped", 0)
        if not spec.accepts_gain(split.gain):
            return leaf(work, mass)
        weighted_gain = spec.growth_priority(split.gain, mass, root_mass)
        if weighted_gain + 0.0 < min_impurity_decrease:
            return leaf(work, mass, weighted_gain)
        return dict(work=work, mass=mass, split=split, weighted_gain=weighted_gain,
                    histograms=node_histograms if reuse_parent_histograms else None)

    next_node = 1
    leaf_count = 1

    def expand(current):
        """Particione o nó escolhido, grave o corte e avalie os dois filhos."""
        nonlocal next_node, leaf_count
        split = current["split"]
        start, end, depth, node_id, path_feature_counts = current["work"]
        partition_start = perf_counter()
        if splitter == "hist" and np.isfinite(split.threshold):
            # Mesma partição de treino, limiar na convenção do exact.
            split = split._replace(threshold=float(local_midpoint_threshold(
                X, sample_indices, start, end, split.feature, split.threshold)))
        mid = int(partition_samples(
            X, sample_indices, start, end, split.feature, split.threshold,
            split.missing_left))
        local_stats["partition_seconds"] = (local_stats.get("partition_seconds", 0.0)
            + (perf_counter() - partition_start))
        if mid <= start or mid >= end:
            # A valid split must produce two non-empty support groups. This is
            # a defensive guard against an inconsistent custom Split.
            raise ValueError("The splitter produced an empty partition.")
        nodes.left[node_id] = next_node
        nodes.right[node_id] = next_node + 1
        nodes.feature[node_id] = split.feature
        nodes.threshold[node_id] = split.threshold
        nodes.missing_left[node_id] = split.missing_left
        local_stats["nodes_split"] += 1
        leaf_count += 1
        child_feature_counts = path_feature_counts.copy()
        child_feature_counts[split.feature] += 1
        left_work = (start, mid, depth + 1, next_node,
                     child_feature_counts.copy())
        right_work = (mid, end, depth + 1, next_node + 1,
                      child_feature_counts.copy())
        next_node += 2
        if not reuse_parent_histograms:
            return evaluate(left_work), evaluate(right_work)

        # Reuso pai-filho: acumular o filho menor e obter o maior por
        # subtração. No empate de tamanho, o acumulado é o esquerdo.
        parent_mass_hist, parent_count_hist = current["histograms"]
        left_size = mid - start
        right_size = end - mid
        left_mass = _node_class_mass(y, weights, sample_indices, start, mid, n_classes)
        right_mass = _node_class_mass(y, weights, sample_indices, mid, end, n_classes)
        left_can_search = not is_terminal(start, mid, depth + 1, left_mass)
        right_can_search = not is_terminal(mid, end, depth + 1, right_mass)
        left_histograms = right_histograms = None
        if left_can_search or right_can_search:
            left_is_smaller = left_size <= right_size
            smaller_start, smaller_end = ((start, mid) if left_is_smaller
                                          else (mid, end))
            smaller_parallel = parallel and (smaller_end - smaller_start >= 256)
            smaller = _build_histograms_for_node(
                X_binned, y, weights, sample_indices, smaller_start, smaller_end,
                edges, n_classes, parallel=smaller_parallel,
                edge_layout=edge_layout)
            larger = (parent_mass_hist - smaller[0], parent_count_hist - smaller[1])
            if left_can_search:
                left_histograms = smaller if left_is_smaller else larger
            if right_can_search:
                right_histograms = larger if left_is_smaller else smaller
            local_stats["histogram_reuse_nodes"] = (
                local_stats.get("histogram_reuse_nodes", 0) + 1)
        else:
            local_stats["histogram_reuse_terminal_saves"] = (
                local_stats.get("histogram_reuse_terminal_saves", 0) + 1)
        return (evaluate(left_work, left_histograms, left_mass),
                evaluate(right_work, right_histograms, right_mass))

    root = evaluate((0, active_rows, 0, 0,
                     np.zeros(X.shape[1], dtype=np.int32)))
    if max_leaf_nodes is None:
        stack = [root]
        while stack:
            current = stack.pop()
            if current["split"] is not None:
                stack.extend(expand(current))
    else:
        # Best-first: maior prioridade primeiro; ids únicos desempatam sem
        # comparar os dicionários.
        heap = []

        def push(item):
            if item["split"] is not None:
                heapq.heappush(heap, (-item["weighted_gain"], item["work"][3], item))

        push(root)
        while heap and leaf_count < max_leaf_nodes:
            _, _, current = heapq.heappop(heap)
            for child in expand(current):
                push(child)

    used = next_node
    return NodeArrays(*(array[:used].copy() for array in nodes))
