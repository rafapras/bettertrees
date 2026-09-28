"""All Numba kernels for impurity, histograms, scans and traversal.

They live in a single module on purpose: ``cache=True`` is invalidated by the
file that defines each function and does not track a kernel called from
another module. Keeping every kernel that calls another kernel here avoids
stale caches. No fastmath: reorderings break tie-breaking and NaN handling.

Expected types at the boundary: int64 indices/orders, int32 y, float64
weights and masses, uint8 bins, float32 X.

The histogram subtraction trick (sibling = parent - child) follows LightGBM
(Ke et al., 2017); the admissible bound used to stop a scan early follows the
branch-and-bound literature on optimal trees (e.g. Demirović et al., JMLR 2022).
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def gini(class_weight):
    """Calcule 1-sum(p_k**2) sobre massas não negativas; massa zero -> 0.

    Entrada interna float64[K], previamente validada. Não usar fastmath:
    reordenações agressivas prejudicam comparações de ganho e NaN.
    """
    # Laços explícitos: a forma vetorizada aloca dois temporários por chamada,
    # e o scan exact chama isto duas vezes por candidato.
    total = 0.0
    for k in range(class_weight.shape[0]):
        total += class_weight[k]
    if total <= 0:
        return 0.0
    squares = 0.0
    for k in range(class_weight.shape[0]):
        share = class_weight[k] / total
        squares += share * share
    return max(0.0, 1.0 - squares)


@njit(cache=True)
def precision_leaf_score(class_weight, positive_class, min_support):
    """Retorne a precision da folha quando ela tem suporte suficiente."""
    total = class_weight.sum()
    if total < min_support or total <= 0.0:
        return -np.inf
    return class_weight[positive_class] / total


@njit(cache=True)
def precision_split_gain(parent_mass, left_mass, right_mass, positive_class,
                         min_support):
    """Ganho da melhor precision filha sobre a precision do pai."""
    parent_score = precision_leaf_score(parent_mass, positive_class, min_support)
    left_score = precision_leaf_score(left_mass, positive_class, min_support)
    right_score = precision_leaf_score(right_mass, positive_class, min_support)
    best_child = max(left_score, right_score)
    if not np.isfinite(parent_score) or not np.isfinite(best_child):
        return -np.inf
    return best_child - parent_score


@njit(cache=True)
def partition_samples(X, sample_indices, start, end, feature, threshold, missing_left):
    """Particione IN PLACE apenas índices no intervalo [start,end).

    Retorna mid: esquerda=[start,mid), direita=[mid,end). Usa <= e a mesma
    direção de NaN da previsão. Ordem interna instável; preserva a permutação
    e o restante dos índices. X não muda. O builder verifica filhos válidos.
    """
    left, right = start, end - 1
    while left <= right:
        value = X[sample_indices[left], feature]
        goes_left = missing_left if np.isnan(value) else value <= threshold
        if goes_left:
            left += 1
        else:
            sample_indices[left], sample_indices[right] = (sample_indices[right],
                                                              sample_indices[left])
            right -= 1
    return left


@njit(cache=True)
def local_midpoint_threshold(X, sample_indices, start, end, feature, threshold):
    """Ponto médio entre o maior valor à esquerda e o menor à direita NO NÓ.

    O hist escolhe cortes em bordas globais de bin; a partição de treino é a
    mesma para qualquer limiar entre esses dois valores, mas valores de teste
    ausentes do nó caem de lados diferentes. O ponto médio local é a convenção
    do exact e do sklearn (margem máxima dentro do nó). Sem valor finito de um
    dos lados, devolve o limiar recebido.
    """
    left_max = -np.inf
    right_min = np.inf
    for i in range(start, end):
        value = X[sample_indices[i], feature]
        if np.isnan(value):
            continue
        if value <= threshold:
            if value > left_max:
                left_max = value
        elif value < right_min:
            right_min = value
    if not (np.isfinite(left_max) and np.isfinite(right_min)):
        return threshold
    return (np.float64(left_max) + np.float64(right_min)) / 2.0


@njit(cache=True)
def build_feature_histogram(column_bins, y, weights, sample_indices, start, end,
                            n_bins, n_classes):
    """Some massas e contagens por bin de UMA feature, incluindo NaN no bin 0.

    Entradas internas alinhadas: bins uint8[n], y int32[n], pesos float64[n],
    índices int64. n_bins inclui 0; ids precisam estar em [0,n_bins).
    Retorna (massas float64[n_bins,K], contagens int64[n_bins]). O splitter
    futuro deve reusar buffers; esta referência aloca por chamada. Parar a
    busca de thresholds NÃO elimina o custo O(n_node) desta acumulação.
    """
    mass = np.zeros((n_bins, n_classes), dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    for pos in range(start, end):
        row = sample_indices[pos]
        bin_id = column_bins[row]
        mass[bin_id, y[row]] += weights[row]
        count[bin_id] += 1
    return mass, count


@njit(cache=True)
def build_feature_histogram_into(column_bins, y, weights, sample_indices, start,
                                 end, n_bins, n_classes, mass, count):
    """Preencha buffers de histograma já alocados e devolva suas fatias úteis."""
    for bin_id in range(n_bins):
        count[bin_id] = 0
        for class_id in range(n_classes):
            mass[bin_id, class_id] = 0.0
    for pos in range(start, end):
        row = sample_indices[pos]
        bin_id = column_bins[row]
        mass[bin_id, y[row]] += weights[row]
        count[bin_id] += 1
    return mass[:n_bins], count[:n_bins]


@njit(cache=True)
def _node_class_mass(y, weights, sample_indices, start, end, n_classes):
    """Some a massa de classes de um nó preservando a ordem do baseline."""
    mass = np.zeros(n_classes, dtype=np.float64)
    for pos in range(start, end):
        row = sample_indices[pos]
        mass[y[row]] += weights[row]
    return mass


@njit(cache=True)
def remaining_gain_upper_bound(parent_mass, fixed_left_mass, fixed_right_mass,
                               parent_gini=-1.0):
    """Limite superior admissível de Gini para TODOS os cortes ainda possíveis.

    fixed_left/right são massas por classe irrevogavelmente destinadas a
    cada filho em qualquer corte restante; não podem se sobrepor. Para uma
    varredura crescente, o prefixo já percorrido é fixo à esquerda. Um sufixo
    que nenhum corte admissível poderá mover fica fixo à direita. Fixar a
    direção de NaN antes de calcular; usar o máximo dos dois limites se ambas
    as direções ainda forem candidatas. Avaliar corte só-NaN separadamente.

    Defina Q(c)=sum(c)*Gini(c). Como Q(c+d)>=Q(c) para d>=0, o ganho futuro
    é <= Gini(parent) - [Q(fixed_left)+Q(fixed_right)]/W_parent. Isso relaxa
    a restrição de ordem dos bins: pode ser frouxo, mas não presume que a
    sequência de ganhos seja monótona. Contrato interno: massas válidas e
    W_parent>0. Fórmula válida em aritmética real; use guarda numérica abaixo.
    """
    total = parent_mass.sum()
    if total <= 0:
        return 0.0
    unavoidable = (fixed_left_mass.sum() * gini(fixed_left_mass)
                   + fixed_right_mass.sum() * gini(fixed_right_mass))
    impurity = gini(parent_mass) if parent_gini < 0 else parent_gini
    return impurity - unavoidable / total


@njit(cache=True)
def cannot_improve(upper_bound, incumbent_gain, n_classes, gain_tolerance=0.0):
    """Decida parar a varredura por bound, com folga contra arredondamento.

    incumbent_gain precisa vir de um corte JÁ admissível (suporte e pesos).
    Sem incumbente (-inf), nunca para. gain_tolerance=0 preserva o máximo
    teórico; tolerância >0 permite perder no máximo esse ganho LOCAL frente
    ao melhor corte da mesma discretização, se todos os descartes usarem o
    bound correto. Não garante acurácia, log loss ou árvore global equivalente.
    Usa comparação estrita e guarda 64*K*eps em unidades de Gini; a guarda é
    de engenharia, não uma certificação formal de erro em ponto flutuante.
    """
    if not np.isfinite(incumbent_gain):
        return False
    guard = 64.0 * max(1, n_classes) * np.finfo(np.float64).eps
    return upper_bound + guard < incumbent_gain + gain_tolerance


@njit(cache=True)
def _gini_scalar(values):
    total = 0.0
    sum_sq = 0.0
    for value in values:
        total += value
        sum_sq += value * value
    if total <= 0.0:
        return 0.0
    return max(0.0, 1.0 - sum_sq / (total * total))


@njit(cache=True)
def _hist_candidate_gain(parent_mass, left_mass, right_mass, parent_total,
                         parent_impurity):
    left_total = 0.0
    for value in left_mass:
        left_total += value
    right_total = parent_total - left_total
    if left_total <= 0.0 or right_total <= 0.0:
        return -np.inf
    for class_id in range(len(parent_mass)):
        right_mass[class_id] = parent_mass[class_id] - left_mass[class_id]
    return (parent_impurity
            - (left_total * _gini_scalar(left_mass)
               + right_total * _gini_scalar(right_mass)) / parent_total)


@njit(cache=True)
def _scan_histogram_feature_numba(mass, count, parent_mass, min_samples_leaf,
                                  incumbent_gain, stopping_code, gain_tolerance,
                                  bound_interval, parent_impurity):
    """Kernel escalar da varredura; não aloca arrays por candidato."""
    n_finite_bins = mass.shape[0] - 1
    n_classes = mass.shape[1]
    total_rows = 0
    for value in count:
        total_rows += value
    parent_total = 0.0
    for value in parent_mass:
        parent_total += value
    if parent_total <= 0.0:
        return -1, False, -np.inf, 0, 0, 0

    missing_count = count[0]
    best_bin = -1
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated = 0
    skipped = 0
    missing = mass[0]
    prefix = np.zeros(n_classes, dtype=np.float64)
    left = np.zeros(n_classes, dtype=np.float64)
    right = np.zeros(n_classes, dtype=np.float64)
    fixed_left = np.zeros(n_classes, dtype=np.float64)
    fixed_right = np.zeros(n_classes, dtype=np.float64)

    # Candidate: all finite values left, NaN right.
    if missing_count > 0 and n_finite_bins > 0:
        for class_id in range(n_classes):
            left[class_id] = parent_mass[class_id] - missing[class_id]
        left_count = total_rows - missing_count
        right_count = missing_count
        if left_count >= min_samples_leaf and right_count >= min_samples_leaf:
            gain = _hist_candidate_gain(parent_mass, left, right, parent_total,
                                        parent_impurity)
            if np.isfinite(gain):
                evaluated += 1
                best_bin = n_finite_bins
                best_missing_left = False
                best_gain = gain
                best_n_left = left_count

    prefix_count = 0
    for b in range(1, n_finite_bins):
        for class_id in range(n_classes):
            prefix[class_id] += mass[b, class_id]
        prefix_count += count[b]

        # NaN right is evaluated first to implement the canonical tie-break.
        for direction in range(2):
            if missing_count == 0 and direction == 1:
                continue
            if prefix_count == 0 or prefix_count == total_rows - missing_count:
                # Prefixo finito vazio ou completo NO NÓ: a partição repete o
                # corte só-NaN (ou deixa um lado vazio). Pular mantém o corte
                # canônico, como no motor exato.
                continue
            missing_left = direction == 1
            left_count = prefix_count + (missing_count if missing_left else 0)
            right_count = total_rows - left_count
            for class_id in range(n_classes):
                left[class_id] = prefix[class_id] + (
                    missing[class_id] if missing_left else 0.0)
            if missing_count == 0:
                missing_left = left_count > right_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            gain = _hist_candidate_gain(parent_mass, left, right, parent_total,
                                        parent_impurity)
            if not np.isfinite(gain):
                continue
            evaluated += 1
            tie_break = (gain == best_gain and (
                best_bin < 0 or b < best_bin
                or (b == best_bin and best_missing_left and not missing_left)))
            if gain > best_gain or tie_break:
                best_bin = b
                best_missing_left = missing_left
                best_gain = gain
                best_n_left = left_count

        check_bound = (b % bound_interval == 0 or b == n_finite_bins - 2)
        if stopping_code == 1 and check_bound and b < n_finite_bins - 1:
            target = max(incumbent_gain, best_gain)
            if np.isfinite(target):
                all_directions_cannot = True
                for direction in range(2):
                    if missing_count == 0 and direction == 1:
                        continue
                    missing_left = direction == 1
                    for class_id in range(n_classes):
                        fixed_left[class_id] = prefix[class_id] + (
                            missing[class_id] if missing_left else 0.0)
                        fixed_right[class_id] = mass[n_finite_bins, class_id] + (
                            0.0 if missing_left else missing[class_id])
                    fixed_left_total = 0.0
                    fixed_right_total = 0.0
                    fixed_left_sq = 0.0
                    fixed_right_sq = 0.0
                    for class_id in range(n_classes):
                        fixed_left_total += fixed_left[class_id]
                        fixed_right_total += fixed_right[class_id]
                        fixed_left_sq += fixed_left[class_id] * fixed_left[class_id]
                        fixed_right_sq += fixed_right[class_id] * fixed_right[class_id]
                    q_left = (fixed_left_total - fixed_left_sq / fixed_left_total
                              if fixed_left_total > 0.0 else 0.0)
                    q_right = (fixed_right_total - fixed_right_sq / fixed_right_total
                               if fixed_right_total > 0.0 else 0.0)
                    upper = parent_impurity - (q_left + q_right) / parent_total
                    guard = 64.0 * max(1, n_classes) * np.finfo(np.float64).eps
                    if upper + guard >= target + gain_tolerance:
                        all_directions_cannot = False
                        break
                if all_directions_cannot:
                    remaining_boundaries = n_finite_bins - 1 - b
                    direction_count = 2 if missing_count > 0 else 1
                    skipped += remaining_boundaries * direction_count
                    break

    return (best_bin, best_missing_left, best_gain, best_n_left,
            evaluated, skipped)


@njit(cache=True)
def _scan_histogram_feature_precision_numba(
        mass, count, parent_mass, min_samples_leaf, positive_class,
        min_precision, min_support):
    """Varra cortes pela melhor precision filha menos a do pai, sem bound."""
    n_finite_bins = mass.shape[0] - 1
    n_classes = mass.shape[1]
    total_rows = 0
    for value in count:
        total_rows += value
    missing_count = count[0]
    best_bin = -1
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated = 0
    missing = mass[0]
    prefix = np.zeros(n_classes, dtype=np.float64)
    left = np.zeros(n_classes, dtype=np.float64)
    right = np.zeros(n_classes, dtype=np.float64)

    if missing_count > 0 and n_finite_bins > 0:
        for class_id in range(n_classes):
            left[class_id] = parent_mass[class_id] - missing[class_id]
            right[class_id] = missing[class_id]
        left_count = total_rows - missing_count
        right_count = missing_count
        if left_count >= min_samples_leaf and right_count >= min_samples_leaf:
            gain = precision_split_gain(
                parent_mass, left, right, positive_class, min_support)
            evaluated += 1
            if gain > 0.0:
                best_bin = n_finite_bins
                best_missing_left = False
                best_gain = gain
                best_n_left = left_count

    prefix_count = 0
    for b in range(1, n_finite_bins):
        for class_id in range(n_classes):
            prefix[class_id] += mass[b, class_id]
        prefix_count += count[b]
        for direction in range(2):
            if missing_count == 0 and direction == 1:
                continue
            if prefix_count == 0 or prefix_count == total_rows - missing_count:
                # Prefixo finito vazio ou completo NO NÓ: a partição repete o
                # corte só-NaN (ou deixa um lado vazio). Pular mantém o corte
                # canônico, como no motor exato.
                continue
            missing_left = direction == 1
            left_count = prefix_count + (missing_count if missing_left else 0)
            right_count = total_rows - left_count
            for class_id in range(n_classes):
                left[class_id] = prefix[class_id] + (
                    missing[class_id] if missing_left else 0.0)
                right[class_id] = parent_mass[class_id] - left[class_id]
            if missing_count == 0:
                missing_left = left_count > right_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            gain = precision_split_gain(
                parent_mass, left, right, positive_class, min_support)
            evaluated += 1
            tie_break = (gain == best_gain and (
                best_bin < 0 or b < best_bin
                or (b == best_bin and best_missing_left and not missing_left)))
            if gain > 0.0 and (gain > best_gain or tie_break):
                best_bin = b
                best_missing_left = missing_left
                best_gain = gain
                best_n_left = left_count

    return (best_bin, best_missing_left, best_gain, best_n_left,
            evaluated, 0)


@njit(cache=True)
def _scan_histogram_feature_numba_scratch(
        mass, count, parent_mass, min_samples_leaf, incumbent_gain,
        stopping_code, gain_tolerance, bound_interval, parent_impurity,
        prefix, left, right, fixed_left, fixed_right):
    """Versão da varredura que reutiliza scratch entre features do mesmo nó."""
    n_finite_bins = mass.shape[0] - 1
    n_classes = mass.shape[1]
    total_rows = 0
    for value in count:
        total_rows += value
    parent_total = 0.0
    for value in parent_mass:
        parent_total += value
    if parent_total <= 0.0:
        return -1, False, -np.inf, 0, 0, 0

    missing_count = count[0]
    best_bin = -1
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated = 0
    skipped = 0
    missing = mass[0]

    for class_id in range(n_classes):
        prefix[class_id] = 0.0

    if missing_count > 0 and n_finite_bins > 0:
        for class_id in range(n_classes):
            left[class_id] = parent_mass[class_id] - missing[class_id]
        left_count = total_rows - missing_count
        right_count = missing_count
        if left_count >= min_samples_leaf and right_count >= min_samples_leaf:
            gain = _hist_candidate_gain(parent_mass, left, right, parent_total,
                                        parent_impurity)
            if np.isfinite(gain):
                evaluated += 1
                best_bin = n_finite_bins
                best_missing_left = False
                best_gain = gain
                best_n_left = left_count

    prefix_count = 0
    for b in range(1, n_finite_bins):
        for class_id in range(n_classes):
            prefix[class_id] += mass[b, class_id]
        prefix_count += count[b]

        # NaN right is evaluated first to preserve the canonical tie-break.
        for direction in range(2):
            if missing_count == 0 and direction == 1:
                continue
            if prefix_count == 0 or prefix_count == total_rows - missing_count:
                # Prefixo finito vazio ou completo NO NÓ: a partição repete o
                # corte só-NaN (ou deixa um lado vazio). Pular mantém o corte
                # canônico, como no motor exato.
                continue
            missing_left = direction == 1
            left_count = prefix_count + (missing_count if missing_left else 0)
            right_count = total_rows - left_count
            for class_id in range(n_classes):
                left[class_id] = prefix[class_id] + (
                    missing[class_id] if missing_left else 0.0)
            if missing_count == 0:
                missing_left = left_count > right_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            gain = _hist_candidate_gain(parent_mass, left, right, parent_total,
                                        parent_impurity)
            if not np.isfinite(gain):
                continue
            evaluated += 1
            tie_break = (gain == best_gain and (
                best_bin < 0 or b < best_bin
                or (b == best_bin and best_missing_left and not missing_left)))
            if gain > best_gain or tie_break:
                best_bin = b
                best_missing_left = missing_left
                best_gain = gain
                best_n_left = left_count

        check_bound = (b % bound_interval == 0 or b == n_finite_bins - 2)
        if stopping_code == 1 and check_bound and b < n_finite_bins - 1:
            target = max(incumbent_gain, best_gain)
            if np.isfinite(target):
                all_directions_cannot = True
                for direction in range(2):
                    if missing_count == 0 and direction == 1:
                        continue
                    missing_left = direction == 1
                    for class_id in range(n_classes):
                        fixed_left[class_id] = prefix[class_id] + (
                            missing[class_id] if missing_left else 0.0)
                        fixed_right[class_id] = mass[n_finite_bins, class_id] + (
                            0.0 if missing_left else missing[class_id])
                    fixed_left_total = 0.0
                    fixed_right_total = 0.0
                    fixed_left_sq = 0.0
                    fixed_right_sq = 0.0
                    for class_id in range(n_classes):
                        fixed_left_total += fixed_left[class_id]
                        fixed_right_total += fixed_right[class_id]
                        fixed_left_sq += fixed_left[class_id] * fixed_left[class_id]
                        fixed_right_sq += fixed_right[class_id] * fixed_right[class_id]
                    q_left = (fixed_left_total - fixed_left_sq / fixed_left_total
                              if fixed_left_total > 0.0 else 0.0)
                    q_right = (fixed_right_total - fixed_right_sq / fixed_right_total
                               if fixed_right_total > 0.0 else 0.0)
                    upper = parent_impurity - (q_left + q_right) / parent_total
                    guard = 64.0 * max(1, n_classes) * np.finfo(np.float64).eps
                    if upper + guard >= target + gain_tolerance:
                        all_directions_cannot = False
                        break
                if all_directions_cannot:
                    remaining_boundaries = n_finite_bins - 1 - b
                    direction_count = 2 if missing_count > 0 else 1
                    skipped += remaining_boundaries * direction_count
                    break

    return (best_bin, best_missing_left, best_gain, best_n_left,
            evaluated, skipped)


@njit(cache=True)
def _scan_histograms_row_major_numba(mass, count, finite_edge_lengths,
                                     feature_order, parent_mass,
                                     min_samples_leaf, stopping_code,
                                     gain_tolerance, bound_interval,
                                     parent_impurity):
    """Varra todas as features de um nó em uma única chamada Numba."""
    best_feature = -1
    best_bin = -1
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated_total = 0
    skipped_total = 0
    n_classes = parent_mass.shape[0]
    prefix = np.empty(n_classes, dtype=np.float64)
    left = np.empty(n_classes, dtype=np.float64)
    right = np.empty(n_classes, dtype=np.float64)
    fixed_left = np.empty(n_classes, dtype=np.float64)
    fixed_right = np.empty(n_classes, dtype=np.float64)

    for order_pos in range(len(feature_order)):
        feature = int(feature_order[order_pos])
        n_finite_bins = int(finite_edge_lengths[feature]) + 1
        n_bins = n_finite_bins + 1
        result = _scan_histogram_feature_numba_scratch(
            mass[feature, :n_bins, :], count[feature, :n_bins], parent_mass,
            min_samples_leaf, best_gain, stopping_code, gain_tolerance,
            bound_interval, parent_impurity, prefix, left, right, fixed_left,
            fixed_right)
        bin_threshold, missing_left, gain, n_left, evaluated, skipped = result
        evaluated_total += evaluated
        skipped_total += skipped
        if bin_threshold < 0 or not np.isfinite(gain):
            continue
        # The outer Python loop historically used a strict comparison here;
        # keeping it strict preserves feature_order as the cross-feature tie
        # breaker while the per-feature kernel preserves its local tie-break.
        if gain > best_gain:
            best_feature = feature
            best_bin = bin_threshold
            best_missing_left = missing_left
            best_gain = gain
            best_n_left = n_left

    return (best_feature, best_bin, best_missing_left, best_gain, best_n_left,
            evaluated_total, skipped_total)


@njit(cache=True, parallel=True)
def _scan_histograms_row_major_parallel_numba(
        mass, count, finite_edge_lengths, feature_order, parent_mass,
        min_samples_leaf, stopping_code, gain_tolerance, bound_interval,
        parent_impurity):
    """Varra features em paralelo e reduz o melhor corte em ordem estável.

    O bound paralelo não usa o ganho da feature anterior como incumbente,
    porque as features são independentes. Isso só pode deixar mais candidatos
    por avaliar; cada feature ainda é varrida com o seu próprio bound exato,
    e a redução final preserva ``feature_order`` como desempate.
    """
    n_features = len(feature_order)
    n_classes = parent_mass.shape[0]
    best_bins = np.full(n_features, -1, dtype=np.int64)
    best_missing = np.zeros(n_features, dtype=np.bool_)
    best_gains = np.full(n_features, -np.inf, dtype=np.float64)
    best_counts = np.zeros(n_features, dtype=np.int64)
    evaluated = np.zeros(n_features, dtype=np.int64)
    skipped = np.zeros(n_features, dtype=np.int64)
    prefix = np.empty((n_features, n_classes), dtype=np.float64)
    left = np.empty((n_features, n_classes), dtype=np.float64)
    right = np.empty((n_features, n_classes), dtype=np.float64)
    fixed_left = np.empty((n_features, n_classes), dtype=np.float64)
    fixed_right = np.empty((n_features, n_classes), dtype=np.float64)

    for order_pos in prange(n_features):
        feature = int(feature_order[order_pos])
        n_finite_bins = int(finite_edge_lengths[feature]) + 1
        result = _scan_histogram_feature_numba_scratch(
            mass[feature, :n_finite_bins + 1, :],
            count[feature, :n_finite_bins + 1], parent_mass,
            min_samples_leaf, -np.inf, stopping_code, gain_tolerance,
            bound_interval, parent_impurity, prefix[order_pos], left[order_pos],
            right[order_pos], fixed_left[order_pos], fixed_right[order_pos])
        best_bins[order_pos] = result[0]
        best_missing[order_pos] = result[1]
        best_gains[order_pos] = result[2]
        best_counts[order_pos] = result[3]
        evaluated[order_pos] = result[4]
        skipped[order_pos] = result[5]

    best_feature = -1
    best_bin = -1
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated_total = 0
    skipped_total = 0
    for order_pos in range(n_features):
        evaluated_total += evaluated[order_pos]
        skipped_total += skipped[order_pos]
        if best_bins[order_pos] < 0 or not np.isfinite(best_gains[order_pos]):
            continue
        if best_gains[order_pos] > best_gain:
            feature = int(feature_order[order_pos])
            best_feature = feature
            best_bin = best_bins[order_pos]
            best_missing_left = best_missing[order_pos]
            best_gain = best_gains[order_pos]
            best_n_left = best_counts[order_pos]

    return (best_feature, best_bin, best_missing_left, best_gain, best_n_left,
            evaluated_total, skipped_total)


@njit(cache=True)
def _scan_exact_feature_gini_numba(ordered_values, ordered_y, ordered_weights,
                                   parent_mass, missing_mass, total_rows,
                                   min_samples_leaf, parent_impurity):
    """Varra cortes finitos de uma feature já ordenada, sem voltar a Python."""
    n_classes = len(parent_mass)
    finite_count = len(ordered_values)
    missing_count = total_rows - finite_count
    total_mass = parent_mass.sum()
    prefix_mass = np.zeros(n_classes, dtype=np.float64)
    left_mass = np.zeros(n_classes, dtype=np.float64)
    right_mass = np.zeros(n_classes, dtype=np.float64)
    best_threshold = np.nan
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated = 0

    for pos in range(finite_count - 1):
        prefix_mass[ordered_y[pos]] += ordered_weights[pos]
        if ordered_values[pos] == ordered_values[pos + 1]:
            continue
        threshold = (np.float64(ordered_values[pos])
                     + np.float64(ordered_values[pos + 1])) / 2.0
        for direction in range(2 if missing_count > 0 else 1):
            missing_left = direction == 1
            left_count = pos + 1 + (missing_count if missing_left else 0)
            right_count = total_rows - left_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            for class_id in range(n_classes):
                left_mass[class_id] = (prefix_mass[class_id]
                                       + (missing_mass[class_id] if missing_left else 0.0))
                right_mass[class_id] = parent_mass[class_id] - left_mass[class_id]
            left_total = left_mass.sum()
            right_total = right_mass.sum()
            if left_total <= 0.0 or right_total <= 0.0:
                continue
            if missing_count == 0:
                missing_left = left_count > right_count
            gain = (parent_impurity
                    - (left_total * gini(left_mass)
                       + right_total * gini(right_mass)) / total_mass)
            evaluated += 1
            if gain > best_gain:
                best_threshold = threshold
                best_missing_left = missing_left
                best_gain = gain
                best_n_left = left_count

    if missing_count > 0 and finite_count > 0:
        for class_id in range(n_classes):
            left_mass[class_id] = parent_mass[class_id] - missing_mass[class_id]
            right_mass[class_id] = missing_mass[class_id]
        left_total = left_mass.sum()
        right_total = right_mass.sum()
        if (finite_count >= min_samples_leaf
                and missing_count >= min_samples_leaf
                and left_total > 0.0 and right_total > 0.0):
            gain = (parent_impurity
                    - (left_total * gini(left_mass)
                       + right_total * gini(right_mass)) / total_mass)
            evaluated += 1
            if gain > best_gain:
                best_threshold = np.inf
                best_missing_left = False
                best_gain = gain
                best_n_left = finite_count
    return (best_threshold, best_missing_left, best_gain, best_n_left,
            evaluated)


_INSERTION_MAX_ROWS = 32
_SIGN_BIT = np.uint32(0x80000000)
_MAGNITUDE_BITS = np.uint32(0x7FFFFFFF)


@njit(cache=True)
def _float32_sort_key(bits):
    """Chave uint32 com a mesma ordem do float32 (sem NaN); -0.0 vira 0.0."""
    if (bits & _MAGNITUDE_BITS) == 0:
        return _SIGN_BIT
    if bits & _SIGN_BIT:
        return ~bits
    return bits | _SIGN_BIT


@njit(cache=True)
def _float32_from_sort_key(key):
    """Inverso de ``_float32_sort_key`` (zero volta como +0.0)."""
    if key & _SIGN_BIT:
        bits = key & _MAGNITUDE_BITS
    else:
        bits = ~key
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


@njit(cache=True)
def _stable_sort_keys(keys, order, keys_tmp, order_tmp, counts):
    """Ordene (keys, order) por keys de forma ESTÁVEL; devolva os arrays.

    Radix LSD (dígitos de 8 bits em nós pequenos, 11 bits nos grandes) com as
    contagens de todas as passadas numa só varredura; passada com dígito
    único é pulada. Até 32 linhas, inserção. Estável é o que garante somar a
    massa dos empates na ordem de ``rows``. ``counts`` precisa de 3*2048.
    O resultado pode estar nos buffers ``*_tmp``: use os arrays devolvidos.
    """
    n = keys.shape[0]
    if n <= _INSERTION_MAX_ROWS:
        for i in range(1, n):
            key = keys[i]
            value = order[i]
            j = i - 1
            while j >= 0 and keys[j] > key:
                keys[j + 1] = keys[j]
                order[j + 1] = order[j]
                j -= 1
            keys[j + 1] = key
            order[j + 1] = value
        return keys, order
    if n < 4096:
        digit_bits = 8
        n_passes = 4
    else:
        digit_bits = 11
        n_passes = 3
    n_buckets = 1 << digit_bits
    mask = np.uint32(n_buckets - 1)
    counts[:n_passes * n_buckets] = 0
    for i in range(n):
        key = keys[i]
        for d in range(n_passes):
            counts[d * n_buckets + ((key >> (d * digit_bits)) & mask)] += 1
    src_k, src_o, dst_k, dst_o = keys, order, keys_tmp, order_tmp
    for d in range(n_passes):
        shift = d * digit_bits
        base = d * n_buckets
        if counts[base + ((src_k[0] >> shift) & mask)] == n:
            continue
        total = 0
        for b in range(n_buckets):
            c = counts[base + b]
            counts[base + b] = total
            total += c
        for i in range(n):
            key = src_k[i]
            slot = base + ((key >> shift) & mask)
            j = counts[slot]
            dst_k[j] = key
            dst_o[j] = src_o[i]
            counts[slot] = j + 1
        src_k, dst_k = dst_k, src_k
        src_o, dst_o = dst_o, src_o
    return src_k, src_o


@njit(cache=True)
def _find_best_split_exact_gini_numba(X, y, weights, rows, feature_order,
                                      parent_mass, min_samples_leaf,
                                      parent_impurity):
    """Busca exata do nó inteiro: coleta, ordena e varre cada feature.

    Mesma semântica de ``_scan_exact_feature_gini_numba`` aplicada à ordem do
    mergesort estável: as chaves uint32 preservam a ordem dos float32 (com
    -0.0 igual a 0.0) e a ordenação é estável, então empates acumulam massa
    na mesma ordem de ``rows``.

    O scan ordena candidatos por ``sum(l_k^2)/L + sum(r_k^2)/R``, que é o
    ganho Gini a menos de constantes do nó, e só recalcula o ganho pela
    fórmula de referência (``gini`` das duas massas) quando a pontuação fica
    a menos de 1e-10*W do melhor da feature. Longe disso o ganho de
    referência é estritamente menor e não mudaria a escolha; perto disso a
    decisão é a da fórmula de referência, então o resultado é idêntico.
    Retorna (feature, threshold, missing_left, gain, n_left, avaliados,
    linhas NaN varridas).
    """
    n = rows.shape[0]
    n_classes = parent_mass.shape[0]
    total_mass = parent_mass.sum()
    slack = 1e-10 * total_mass
    node_y = np.empty(n, dtype=np.int64)
    node_w = np.empty(n, dtype=np.float64)
    for i in range(n):
        node_y[i] = y[rows[i]]
        node_w[i] = weights[rows[i]]
    values = np.empty(n, dtype=np.float32)
    bits = values.view(np.uint32)
    keys = np.empty(n, dtype=np.uint32)
    order = np.empty(n, dtype=np.int64)
    keys_tmp = np.empty(n, dtype=np.uint32)
    order_tmp = np.empty(n, dtype=np.int64)
    counts = np.empty(3 * 2048, dtype=np.int64)
    missing_mass = np.zeros(n_classes, dtype=np.float64)
    prefix_mass = np.zeros(n_classes, dtype=np.float64)
    left_mass = np.zeros(n_classes, dtype=np.float64)
    right_mass = np.zeros(n_classes, dtype=np.float64)

    best_feature = -1
    best_threshold = np.nan
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated = 0
    missing_seen = 0

    for order_pos in range(feature_order.shape[0]):
        feature = feature_order[order_pos]
        for k in range(n_classes):
            missing_mass[k] = 0.0
            prefix_mass[k] = 0.0
        finite_count = 0
        for i in range(n):
            value = X[rows[i], feature]
            if np.isnan(value):
                missing_mass[node_y[i]] += node_w[i]
            else:
                values[finite_count] = value
                order[finite_count] = i
                finite_count += 1
        missing_count = n - finite_count
        missing_seen += missing_count
        for j in range(finite_count):
            keys[j] = _float32_sort_key(bits[j])
        sorted_keys, sorted_rows = _stable_sort_keys(
            keys[:finite_count], order[:finite_count], keys_tmp, order_tmp,
            counts)

        feat_gain = -np.inf
        feat_score = -np.inf
        feat_threshold = np.nan
        feat_missing_left = False
        feat_n_left = 0
        n_directions = 2 if missing_count > 0 else 1
        for pos in range(finite_count):
            key = sorted_keys[pos]
            if pos > 0 and key != sorted_keys[pos - 1]:
                for direction in range(n_directions):
                    missing_left = direction == 1
                    left_count = pos + (missing_count if missing_left else 0)
                    right_count = n - left_count
                    if left_count < min_samples_leaf or right_count < min_samples_leaf:
                        continue
                    left_total = 0.0
                    right_total = 0.0
                    left_sq = 0.0
                    right_sq = 0.0
                    for k in range(n_classes):
                        lm = prefix_mass[k] + (missing_mass[k] if missing_left else 0.0)
                        rm = parent_mass[k] - lm
                        left_total += lm
                        right_total += rm
                        left_sq += lm * lm
                        right_sq += rm * rm
                    if left_total <= 0.0 or right_total <= 0.0:
                        continue
                    evaluated += 1
                    score = left_sq / left_total + right_sq / right_total
                    if score < feat_score - slack:
                        continue
                    for k in range(n_classes):
                        left_mass[k] = (prefix_mass[k]
                                        + (missing_mass[k] if missing_left else 0.0))
                        right_mass[k] = parent_mass[k] - left_mass[k]
                    gain = (parent_impurity
                            - (left_mass.sum() * gini(left_mass)
                               + right_mass.sum() * gini(right_mass)) / total_mass)
                    if gain > feat_gain:
                        feat_gain = gain
                        if score > feat_score:
                            feat_score = score
                        feat_threshold = (
                            np.float64(_float32_from_sort_key(sorted_keys[pos - 1]))
                            + np.float64(_float32_from_sort_key(key))) / 2.0
                        feat_missing_left = (missing_left if missing_count > 0
                                             else left_count > right_count)
                        feat_n_left = left_count
            i = sorted_rows[pos]
            prefix_mass[node_y[i]] += node_w[i]
        if missing_count > 0 and finite_count > 0:
            for k in range(n_classes):
                left_mass[k] = parent_mass[k] - missing_mass[k]
                right_mass[k] = missing_mass[k]
            left_total = left_mass.sum()
            right_total = right_mass.sum()
            if (finite_count >= min_samples_leaf
                    and missing_count >= min_samples_leaf
                    and left_total > 0.0 and right_total > 0.0):
                gain = (parent_impurity
                        - (left_total * gini(left_mass)
                           + right_total * gini(right_mass)) / total_mass)
                evaluated += 1
                if gain > feat_gain:
                    feat_gain = gain
                    feat_threshold = np.inf
                    feat_missing_left = False
                    feat_n_left = finite_count
        if feat_gain > best_gain:
            best_feature = feature
            best_threshold = feat_threshold
            best_missing_left = feat_missing_left
            best_gain = feat_gain
            best_n_left = feat_n_left
    return (best_feature, best_threshold, best_missing_left, best_gain,
            best_n_left, evaluated, missing_seen)


@njit(cache=True)
def _scan_exact_feature_precision_numba(
        ordered_values, ordered_y, ordered_weights, parent_mass,
        missing_mass, total_rows, min_samples_leaf, positive_class,
        min_support):
    """Varra cortes para precision sem despachar um kernel por candidato."""
    n_classes = len(parent_mass)
    finite_count = len(ordered_values)
    missing_count = total_rows - finite_count
    prefix_mass = np.zeros(n_classes, dtype=np.float64)
    left_mass = np.zeros(n_classes, dtype=np.float64)
    right_mass = np.zeros(n_classes, dtype=np.float64)
    best_threshold = np.nan
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated = 0

    for pos in range(finite_count - 1):
        prefix_mass[ordered_y[pos]] += ordered_weights[pos]
        if ordered_values[pos] == ordered_values[pos + 1]:
            continue
        threshold = (np.float64(ordered_values[pos])
                     + np.float64(ordered_values[pos + 1])) / 2.0
        for direction in range(2 if missing_count > 0 else 1):
            missing_left = direction == 1
            left_count = pos + 1 + (missing_count if missing_left else 0)
            right_count = total_rows - left_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            for class_id in range(n_classes):
                left_mass[class_id] = (prefix_mass[class_id]
                                       + (missing_mass[class_id] if missing_left else 0.0))
                right_mass[class_id] = parent_mass[class_id] - left_mass[class_id]
            if left_mass.sum() <= 0.0 or right_mass.sum() <= 0.0:
                continue
            if missing_count == 0:
                missing_left = left_count > right_count
            gain = precision_split_gain(parent_mass, left_mass, right_mass,
                                        positive_class, min_support)
            evaluated += 1
            if gain > best_gain and gain > 0.0:
                best_threshold = threshold
                best_missing_left = missing_left
                best_gain = gain
                best_n_left = left_count

    if missing_count > 0 and finite_count > 0:
        for class_id in range(n_classes):
            left_mass[class_id] = parent_mass[class_id] - missing_mass[class_id]
            right_mass[class_id] = missing_mass[class_id]
        if (finite_count >= min_samples_leaf
                and missing_count >= min_samples_leaf
                and left_mass.sum() > 0.0 and right_mass.sum() > 0.0):
            gain = precision_split_gain(parent_mass, left_mass, right_mass,
                                        positive_class, min_support)
            evaluated += 1
            if gain > best_gain and gain > 0.0:
                best_threshold = np.inf
                best_missing_left = False
                best_gain = gain
                best_n_left = finite_count
    return (best_threshold, best_missing_left, best_gain, best_n_left,
            evaluated)


@njit(cache=True)
def _build_all_histograms_row_major(X_binned, y, weights, sample_indices,
                                    start, end, n_features, max_bins, n_classes):
    """Acumule todos os histogramas lendo cada linha de bins uma única vez."""
    mass = np.zeros((n_features, max_bins, n_classes), dtype=np.float64)
    count = np.zeros((n_features, max_bins), dtype=np.int64)
    for pos in range(start, end):
        row = sample_indices[pos]
        class_id = y[row]
        weight = weights[row]
        for feature in range(n_features):
            bin_id = X_binned[row, feature]
            mass[feature, bin_id, class_id] += weight
            count[feature, bin_id] += 1
    return mass, count


@njit(cache=True, parallel=True)
def _build_all_histograms_feature_parallel(X_binned, y, weights, sample_indices,
                                           start, end, n_features, max_bins,
                                           n_classes):
    """Acumule features em paralelo, com uma saída exclusiva por feature."""
    mass = np.zeros((n_features, max_bins, n_classes), dtype=np.float64)
    count = np.zeros((n_features, max_bins), dtype=np.int64)
    for feature in prange(n_features):
        for pos in range(start, end):
            row = sample_indices[pos]
            bin_id = X_binned[row, feature]
            class_id = y[row]
            mass[feature, bin_id, class_id] += weights[row]
            count[feature, bin_id] += 1
    return mass, count


@njit(cache=True)
def apply_nodes(X, left, right, feature, threshold, missing_left):
    """Percorra uma árvore válida e retorne id da folha por linha (int32).

    Contrato interno: X float32 validado; arrays descrevem árvore acíclica,
    conectada, com raiz 0 e filhos válidos. Igualdade segue para a esquerda;
    NaN segue direção gravada no treino. Compartilhado pelos dois motores.
    """
    result = np.empty(len(X), dtype=np.int32)
    for i in range(len(X)):
        node = 0
        while left[node] != -1:
            value = X[i, feature[node]]
            goes_left = missing_left[node] if np.isnan(value) else value <= threshold[node]
            node = left[node] if goes_left else right[node]
        result[i] = node
    return result
@njit(cache=True)
def _solve_depth2_block_hist_numba(X_binned, y, weights, rows,
                                   finite_edge_lengths, feature_order,
                                   n_classes, max_bins, min_samples_leaf,
                                   candidate_limit):
    """Bloco depth 2 inteiro (raiz hist + filhos hist) numa chamada Numba.

    Para cada feature f da raiz, um histograma conjunto
    pair[bin_f, g, bin_g, classe] (uma passada nas linhas do nó) dá o
    histograma do filho esquerdo de qualquer limiar de f como soma de
    prefixos; o direito sai por subtração do histograma do nó. Os filhos são
    varridos por ``_scan_histograms_row_major_numba``, o mesmo kernel da busca
    gulosa. Ordem dos candidatos e desempates iguais aos de
    ``multilevel._candidate_roots`` + ``_solve_block``. Exige pesos
    unitários: as massas são inteiros exatos, então subtrações e somas em
    outra ordem não mudam nenhum ganho.

    Retorna (candidatos, raiz[7], esquerda[5], direita[5]) com raiz =
    (feature, bin, é_só_NaN, missing_left, ganho, n_left, achou) e filhos =
    (feature, bin, missing_left, ganho, n_left); feature=-1 = sem corte.
    candidatos = -1 se ``candidate_limit`` foi excedido.
    """
    n = rows.shape[0]
    n_features = X_binned.shape[1]
    node_mass = np.zeros((n_features, max_bins, n_classes), dtype=np.float64)
    node_count = np.zeros((n_features, max_bins), dtype=np.int64)
    parent = np.zeros(n_classes, dtype=np.float64)
    for pos in range(n):
        row = rows[pos]
        c = y[row]
        w = weights[row]
        parent[c] += w
        for g in range(n_features):
            b = X_binned[row, g]
            node_mass[g, b, c] += w
            node_count[g, b] += 1
    total = parent.sum()
    parent_gini = gini(parent)

    pair_mass = np.zeros((max_bins, n_features, max_bins, n_classes), dtype=np.float64)
    pair_count = np.zeros((max_bins, n_features, max_bins), dtype=np.int64)
    pre_mass = np.empty((n_features, max_bins, n_classes), dtype=np.float64)
    pre_count = np.empty((n_features, max_bins), dtype=np.int64)
    l_mass = np.empty((n_features, max_bins, n_classes), dtype=np.float64)
    l_count = np.empty((n_features, max_bins), dtype=np.int64)
    r_mass = np.empty((n_features, max_bins, n_classes), dtype=np.float64)
    r_count = np.empty((n_features, max_bins), dtype=np.int64)
    lm = np.empty(n_classes, dtype=np.float64)
    rm = np.empty(n_classes, dtype=np.float64)

    best_improvement = 0.0
    best_splits = 0
    found = False
    root = (-1, -1, False, False, -np.inf, 0)
    best_left = (-1, -1, False, -np.inf, 0)
    best_right = (-1, -1, False, -np.inf, 0)
    candidates = 0

    for order_pos in range(feature_order.shape[0]):
        f = feature_order[order_pos]
        n_edges = finite_edge_lengths[f]
        n_bins_f = n_edges + 2
        pair_mass[:n_bins_f] = 0.0
        pair_count[:n_bins_f] = 0
        for pos in range(n):
            row = rows[pos]
            fb = X_binned[row, f]
            c = y[row]
            w = weights[row]
            for g in range(n_features):
                gb = X_binned[row, g]
                pair_mass[fb, g, gb, c] += w
                pair_count[fb, g, gb] += 1
        missing_rows = node_count[f, 0]
        has_missing = missing_rows > 0
        finite_rows = n - missing_rows
        pre_mass[:] = 0.0
        pre_count[:] = 0
        # Partição repetida (limiar em bin vazio no nó, ou só-NaN igual ao
        # último limiar) é descartada como em ``_candidate_roots``: a parte
        # finita à esquerda cresce com o limiar, então basta comparar com o
        # candidato anterior da mesma direção.
        previous_finite_left = np.full(2, -1, dtype=np.int64)
        n_thresholds = n_edges + (1 if has_missing and finite_rows > 0 else 0)
        for t in range(n_thresholds):
            only_missing = t == n_edges
            if not only_missing:
                pre_mass += pair_mass[t + 1]
                pre_count += pair_count[t + 1]
            finite_left = finite_rows
            if not only_missing:
                finite_left = 0
                for b in range(max_bins):
                    finite_left += pre_count[0, b]
            n_directions = 2 if (has_missing and not only_missing) else 1
            for direction in range(n_directions):
                initial_missing_left = direction == 1
                if previous_finite_left[direction] == finite_left:
                    continue
                previous_finite_left[direction] = finite_left
                if only_missing:
                    # finitos à esquerda, NaN à direita
                    for g in range(n_features):
                        for b in range(max_bins):
                            l_count[g, b] = node_count[g, b] - pair_count[0, g, b]
                            for c in range(n_classes):
                                l_mass[g, b, c] = node_mass[g, b, c] - pair_mass[0, g, b, c]
                    left_rows = finite_rows
                else:
                    left_rows = 0
                    for b in range(max_bins):
                        left_rows += pre_count[0, b]
                    if initial_missing_left:
                        left_rows += missing_rows
                right_rows = n - left_rows
                if left_rows < min_samples_leaf or right_rows < min_samples_leaf:
                    continue
                if not only_missing:
                    if initial_missing_left:
                        l_mass[:] = pre_mass + pair_mass[0]
                        l_count[:] = pre_count + pair_count[0]
                    else:
                        l_mass[:] = pre_mass
                        l_count[:] = pre_count
                for c in range(n_classes):
                    lm[c] = 0.0
                for b in range(max_bins):
                    for c in range(n_classes):
                        lm[c] += l_mass[0, b, c]
                for c in range(n_classes):
                    rm[c] = parent[c] - lm[c]
                left_total = lm.sum()
                right_total = rm.sum()
                if left_total <= 0.0 or right_total <= 0.0:
                    continue
                candidates += 1
                if candidates > candidate_limit:
                    return (-1, root, best_left, best_right)
                missing_left = (initial_missing_left if has_missing
                                else left_rows > right_rows)
                root_gain = parent_gini - (left_total * gini(lm)
                                           + right_total * gini(rm)) / total
                r_mass[:] = node_mass - l_mass
                r_count[:] = node_count - l_count
                left_split = (-1, -1, False, -np.inf, 0)
                right_split = (-1, -1, False, -np.inf, 0)
                if left_rows >= 2 * min_samples_leaf:
                    res = _scan_histograms_row_major_numba(
                        l_mass, l_count, finite_edge_lengths, feature_order, lm,
                        min_samples_leaf, 0, 0.0, 1, gini(lm))
                    if res[0] >= 0 and np.isfinite(res[3]) and res[3] > 0.0:
                        left_split = (res[0], res[1], res[2], res[3], res[4])
                if right_rows >= 2 * min_samples_leaf:
                    res = _scan_histograms_row_major_numba(
                        r_mass, r_count, finite_edge_lengths, feature_order, rm,
                        min_samples_leaf, 0, 0.0, 1, gini(rm))
                    if res[0] >= 0 and np.isfinite(res[3]) and res[3] > 0.0:
                        right_split = (res[0], res[1], res[2], res[3], res[4])
                improvement = total * root_gain
                n_splits = 1
                if left_split[0] >= 0:
                    improvement += left_total * left_split[3]
                    n_splits += 1
                if right_split[0] >= 0:
                    improvement += right_total * right_split[3]
                    n_splits += 1
                if (improvement > best_improvement + 1e-12
                        or (found and abs(improvement - best_improvement) <= 1e-12
                            and n_splits < best_splits)):
                    found = True
                    best_improvement = improvement
                    best_splits = n_splits
                    root = (f, t + 1, only_missing, missing_left, root_gain, left_rows)
                    best_left = left_split
                    best_right = right_split
    if not found:
        return (candidates, (-1, -1, False, False, -np.inf, 0), best_left, best_right)
    return (candidates, root, best_left, best_right)


