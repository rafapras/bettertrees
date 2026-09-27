"""Implementações Python de referência, usadas apenas em testes e ablações.

Nenhum caminho de produção importa este módulo.
"""

import numpy as np

from bettertrees._data import Split
from bettertrees.kernels import (
    cannot_improve,
    gini,
    precision_split_gain,
    remaining_gain_upper_bound,
)


def _scan_histogram_feature_reference(mass, count, edges, parent_mass, *, min_samples_leaf,
                                       incumbent_gain, search_stopping="bound", gain_tolerance=0.0,
                                       bound_interval=1, parent_gini=None):
    """Percorra os bins de uma feature e, opcionalmente, use o bound admissível.

    Retorne (bin_threshold, missing_left, gain, n_left, evaluated, skipped).
    1. Testar corte apenas missing antes de excluir essa possibilidade.
    2. Varrer prefixos de massas/contagens; testar suporte e ambas as direções.
    3. Atualizar melhor ganho admissível; construir massas fixas do restante.
    4. Calcular o máximo dos bounds para NaN-left e NaN-right; cannot_improve
       autoriza interromper a feature inteira. Se apenas uma direção falhar,
       descartar somente aquela direção. 'off' sempre termina a varredura.
    Registrar cortes avaliados/pulados e custo dos bounds para a ablação.
    Em ganho empatado manter candidato anterior; não parar só por sequência
    sem melhora. Rejeitar 'heuristic' até especificação estatística explícita.
    """
    if search_stopping not in ("off", "bound"):
        raise ValueError("search_stopping deve ser off ou bound.")
    if (isinstance(bound_interval, (bool, np.bool_))
            or not isinstance(bound_interval, (int, np.integer))
            or bound_interval < 1):
        raise ValueError("bound_interval deve ser inteiro positivo.")
    mass = np.asarray(mass, dtype=np.float64)
    count = np.asarray(count, dtype=np.int64)
    parent_mass = np.asarray(parent_mass, dtype=np.float64)
    if mass.ndim != 2 or count.ndim != 1 or mass.shape[0] != len(count):
        raise ValueError("Histograma com shape inconsistente.")
    if mass.shape[1] != len(parent_mass):
        raise ValueError("Massas do histograma e do pai incompatíveis.")
    if len(mass) == 0:
        return -1, False, -np.inf, 0, 0, 0

    n_finite_bins = len(mass) - 1
    missing_mass = mass[0].copy()
    missing_count = int(count[0])
    finite_mass = mass[1:]
    finite_count = count[1:]
    parent_total = float(parent_mass.sum())
    if parent_total <= 0:
        return -1, False, -np.inf, 0, 0, 0
    parent_impurity = (float(gini(parent_mass)) if parent_gini is None
                       else float(parent_gini))

    best_bin = -1
    best_missing_left = False
    best_gain = -np.inf
    best_n_left = 0
    evaluated = 0
    skipped = 0

    def consider(bin_threshold, missing_left, left_mass, left_count):
        nonlocal best_bin, best_missing_left, best_gain, best_n_left, evaluated
        right_mass = parent_mass - left_mass
        right_count = int(count.sum()) - int(left_count)
        if left_count < min_samples_leaf or right_count < min_samples_leaf:
            return
        if left_mass.sum() <= 0 or right_mass.sum() <= 0:
            return
        gain = float(parent_impurity - (
            left_mass.sum() * gini(left_mass)
            + right_mass.sum() * gini(right_mass)) / parent_total)
        evaluated += 1
        # A scan is in canonical order: lower bin first, NaN right first.
        # Keeping the previous candidate on exact ties implements that rule.
        tie_break = (
            gain == best_gain
            and (best_bin < 0
                 or bin_threshold < best_bin
                 or (bin_threshold == best_bin and best_missing_left and not missing_left))
        )
        if gain > best_gain or tie_break:
            best_bin = int(bin_threshold)
            best_missing_left = bool(missing_left)
            best_gain = gain
            best_n_left = int(left_count)

    # The finite-versus-missing cut is a real candidate and must be considered
    # before a bound can fix the last finite bin on the right.
    if missing_count > 0 and n_finite_bins > 0:
        consider(n_finite_bins, False, finite_mass.sum(axis=0), int(finite_count.sum()))

    prefix_mass = np.zeros_like(parent_mass)
    prefix_count = 0
    total_rows = int(count.sum())
    for b in range(1, n_finite_bins):
        prefix_mass += finite_mass[b - 1]
        prefix_count += int(finite_count[b - 1])
        directions = (False, True) if missing_count > 0 else (False,)
        if prefix_count == 0 or prefix_count == total_rows - missing_count:
            # Prefixo finito vazio ou completo no nó repete o corte só-NaN.
            directions = ()
        for missing_left in directions:
            left_mass = prefix_mass + (missing_mass if missing_left else 0.0)
            left_count = prefix_count + (missing_count if missing_left else 0)
            if missing_count == 0:
                # NaN observed only at prediction time goes to the larger
                # child; ties go right. It has no effect on training gain.
                missing_left = bool(left_count > total_rows - left_count)
            consider(b, missing_left, left_mass, left_count)

        check_bound = (b % int(bound_interval) == 0
                       or b == n_finite_bins - 2)
        if search_stopping == "bound" and check_bound and b < n_finite_bins - 1:
            # For every later finite boundary, the current prefix is fixed on
            # the left and the last finite bin is fixed on the right. Compute
            # one safe bound per still possible NaN direction.
            target = max(float(incumbent_gain), best_gain)
            if np.isfinite(target):
                bounds = []
                for missing_left in ((False, True) if missing_count > 0 else (False,)):
                    fixed_left = prefix_mass + (missing_mass if missing_left else 0.0)
                    fixed_right = finite_mass[-1] + (0.0 if missing_left else missing_mass)
                    bounds.append(float(remaining_gain_upper_bound(
                        parent_mass, fixed_left, fixed_right, parent_impurity)))
                if bounds and all(cannot_improve(bound, target, len(parent_mass), gain_tolerance)
                                  for bound in bounds):
                    remaining_boundaries = n_finite_bins - 1 - b
                    direction_count = 2 if missing_count > 0 else 1
                    skipped += remaining_boundaries * direction_count
                    break

    return (best_bin, best_missing_left, best_gain, best_n_left,
            int(evaluated), int(skipped))


def _find_best_split_exact_reference(X, y, weights, sample_indices, start, end,
                                     n_classes, min_samples_leaf, feature_order,
                                     search_stopping="off", gain_tolerance=0.0,
                                     stats=None, parent_mass=None):
    """Busque exaustivamente o melhor corte de Gini no nó, sem alterar índices.

    Ordenar finitos por feature; percorrer fronteiras entre valores distintos
    atualizando massas por classe. Testar NaN à esquerda e à direita, também
    o corte finitos-versus-NaN (threshold=+inf, missing_left=False).
    Exigir ambos os filhos com min_samples_leaf linhas ativas e massa > 0.
    Limiares são pontos médios float64 de valores X float32. Ausência de NaN
    no treino: encaminhar NaN futuro ao filho com mais linhas; empate direita.
    Empates de ganho: primeiro feature_order, depois menor limiar, depois NaN
    à direita. Sem candidato: Split(-1, nan, False, -inf, 0, -1).
    A versão atual é uma referência serial; a compilação integral fica para a
    etapa de otimização, depois da medição do custo de cada estágio.
    """
    if search_stopping not in ("off", "bound"):
        raise ValueError("search_stopping deve ser off ou bound.")
    rows = np.asarray(sample_indices[start:end], dtype=np.int64)
    parent_mass = np.zeros(n_classes, dtype=np.float64)
    for row in rows:
        parent_mass[y[row]] += weights[row]
    total_rows = len(rows)
    total_mass = float(parent_mass.sum())
    if total_mass <= 0:
        return Split(-1, np.nan, False, -np.inf, 0, -1)

    best = Split(-1, np.nan, False, -np.inf, 0, -1)
    evaluated = 0
    missing_seen = 0

    def better(feature, threshold, missing_left, gain, n_left):
        nonlocal best
        if gain > best.gain:
            best = Split(int(feature), float(threshold), bool(missing_left),
                         float(gain), int(n_left), -1)

    for feature in np.asarray(feature_order, dtype=np.int64):
        values = X[rows, feature]
        finite_mask = ~np.isnan(values)
        finite_rows = rows[finite_mask]
        missing_rows = rows[~finite_mask]
        missing_count = len(missing_rows)
        missing_mass = np.zeros(n_classes, dtype=np.float64)
        for row in missing_rows:
            missing_mass[y[row]] += weights[row]
        missing_seen += missing_count
        order = np.argsort(X[finite_rows, feature], kind="mergesort")
        ordered_rows = finite_rows[order]
        finite_count = len(ordered_rows)
        prefix_mass = np.zeros(n_classes, dtype=np.float64)
        prefix_count = 0

        for pos in range(finite_count - 1):
            row = ordered_rows[pos]
            prefix_mass[y[row]] += weights[row]
            prefix_count += 1
            current = float(X[row, feature])
            next_value = float(X[ordered_rows[pos + 1], feature])
            if current == next_value:
                continue
            threshold = (current + next_value) / 2.0
            directions = (False, True) if missing_count > 0 else (False,)
            for missing_left in directions:
                left_mass = prefix_mass + (missing_mass if missing_left else 0.0)
                left_count = prefix_count + (missing_count if missing_left else 0)
                if missing_count == 0:
                    missing_left = bool(left_count > total_rows - left_count)
                right_mass = parent_mass - left_mass
                right_count = total_rows - left_count
                if (left_count < min_samples_leaf or right_count < min_samples_leaf
                        or left_mass.sum() <= 0 or right_mass.sum() <= 0):
                    continue
                gain = float(gini(parent_mass) - (
                    left_mass.sum() * gini(left_mass)
                    + right_mass.sum() * gini(right_mass)) / total_mass)
                evaluated += 1
                better(feature, threshold, missing_left, gain, left_count)

        # The only useful split when all finite values are on one side of NaN
        # is finite-left / missing-right. A threshold of +inf preserves the
        # same routing rule in partition_samples and apply_nodes.
        if missing_count > 0 and finite_count > 0:
            finite_mass = parent_mass - missing_mass
            left_count = finite_count
            right_count = missing_count
            if (left_count >= min_samples_leaf and right_count >= min_samples_leaf
                    and finite_mass.sum() > 0 and missing_mass.sum() > 0):
                gain = float(gini(parent_mass) - (
                    finite_mass.sum() * gini(finite_mass)
                    + missing_mass.sum() * gini(missing_mass)) / total_mass)
                evaluated += 1
                better(feature, np.inf, False, gain, left_count)

    if stats is not None:
        stats["exact_candidates_evaluated"] = stats.get("exact_candidates_evaluated", 0) + evaluated
        stats["missing_rows_scanned"] = stats.get("missing_rows_scanned", 0) + missing_seen
    return best


def _find_best_split_exact_precision_reference(
        X, y, weights, sample_indices, start, end, n_classes,
        min_samples_leaf, feature_order, positive_class, min_precision,
        min_support, stats=None, parent_mass=None):
    """Referência do objetivo precision (melhor precision filha - pai).

    ``parent_mass`` é aceito para substituir a busca real e ignorado: a
    referência sempre recalcula a massa.
    """
    rows = np.asarray(sample_indices[start:end], dtype=np.int64)
    parent_mass = np.zeros(n_classes, dtype=np.float64)
    for row in rows:
        parent_mass[y[row]] += weights[row]
    best = Split(-1, np.nan, False, -np.inf, 0, -1)
    evaluated = 0

    for feature in np.asarray(feature_order, dtype=np.int64):
        values = X[rows, feature]
        finite_mask = ~np.isnan(values)
        finite_rows = rows[finite_mask]
        missing_rows = rows[~finite_mask]
        missing_count = len(missing_rows)
        missing_mass = np.zeros(n_classes, dtype=np.float64)
        for row in missing_rows:
            missing_mass[y[row]] += weights[row]
        order = np.argsort(X[finite_rows, feature], kind="mergesort")
        ordered_rows = finite_rows[order]
        finite_count = len(ordered_rows)
        prefix_mass = np.zeros(n_classes, dtype=np.float64)
        prefix_count = 0

        for pos in range(finite_count - 1):
            row = ordered_rows[pos]
            prefix_mass[y[row]] += weights[row]
            prefix_count += 1
            current = float(X[row, feature])
            next_value = float(X[ordered_rows[pos + 1], feature])
            if current == next_value:
                continue
            threshold = (current + next_value) / 2.0
            directions = (False, True) if missing_count > 0 else (False,)
            for missing_left in directions:
                left_mass = prefix_mass + (missing_mass if missing_left else 0.0)
                left_count = prefix_count + (missing_count if missing_left else 0)
                if missing_count == 0:
                    missing_left = bool(left_count > len(rows) - left_count)
                right_mass = parent_mass - left_mass
                right_count = len(rows) - left_count
                if (left_count < min_samples_leaf or right_count < min_samples_leaf
                        or left_mass.sum() <= 0 or right_mass.sum() <= 0):
                    continue
                gain = float(precision_split_gain(
                    parent_mass, left_mass, right_mass, positive_class,
                    min_support))
                evaluated += 1
                if gain > best.gain and gain > 0.0:
                    best = Split(int(feature), threshold, bool(missing_left),
                                 gain, int(left_count), -1)

        if missing_count > 0 and finite_count > 0:
            finite_mass = parent_mass - missing_mass
            left_count = finite_count
            right_count = missing_count
            if (left_count >= min_samples_leaf and right_count >= min_samples_leaf
                    and finite_mass.sum() > 0 and missing_mass.sum() > 0):
                gain = float(precision_split_gain(
                    parent_mass, finite_mass, missing_mass, positive_class,
                    min_support))
                evaluated += 1
                if gain > best.gain and gain > 0.0:
                    best = Split(int(feature), np.inf, False, gain,
                                 int(left_count), -1)

    if stats is not None:
        stats["exact_candidates_evaluated"] = (
            stats.get("exact_candidates_evaluated", 0) + evaluated)
    return best
