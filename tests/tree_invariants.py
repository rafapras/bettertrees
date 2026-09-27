"""Invariantes de correção de uma árvore treinada.

Cada verificação aqui é uma propriedade que vale para QUALQUER árvore correta
com os parâmetros dados; uma falha é bug, não questão de qualidade. Qualidade
estatística (log-loss, AUC, tamanho) pertence ao gate de força, não aqui.

Grupos:
- estruturais: árvore conectada, ids coerentes, massas e contagens conservadas;
- restrições: min_samples_leaf, max_depth, max_leaf_nodes, max_feature_repeats;
- regras de parada do próprio objetivo: todo corte mantido era admissível;
- coerência treino->previsão: as linhas de treino caem nas folhas cujas
  massas o builder gravou; probabilidades válidas;
- monotonicidade nas regiões finitas.
"""

import numpy as np

from arvore_rapida.kernels import gini


def _active(X, y, sample_weight):
    """Linhas que participam do treino (peso positivo), como no fit."""
    X = np.asarray(X, dtype=np.float32)
    classes, encoded = np.unique(np.asarray(y), return_inverse=True)
    weights = (np.ones(len(X)) if sample_weight is None
               else np.asarray(sample_weight, dtype=np.float64))
    keep = weights > 0
    return X[keep], encoded[keep], weights[keep], classes


def _depths_and_paths(nodes, n_features):
    """Profundidade e contagem de uso de cada feature no caminho até o nó."""
    n = len(nodes.left)
    depth = np.full(n, -1, dtype=np.int64)
    path_counts = np.zeros((n, n_features), dtype=np.int64)
    depth[0] = 0
    stack = [0]
    while stack:
        node = stack.pop()
        left = int(nodes.left[node])
        if left == -1:
            continue
        right = int(nodes.right[node])
        feature = int(nodes.feature[node])
        for child in (left, right):
            depth[child] = depth[node] + 1
            path_counts[child] = path_counts[node]
            path_counts[child, feature] += 1
            stack.append(child)
    return depth, path_counts


def check_structure(nodes, n_features):
    n = len(nodes.left)
    left, right = nodes.left, nodes.right
    internal = left != -1
    assert np.array_equal(internal, right != -1), "filho esquerdo sem direito"
    parents = np.zeros(n, dtype=np.int64)
    for node in np.flatnonzero(internal):
        for child in (int(left[node]), int(right[node])):
            assert node < child < n, f"id de filho inválido em {node}"
            parents[child] += 1
    assert parents[0] == 0, "raiz tem pai"
    assert np.all(parents[1:] == 1), "nó sem pai ou com dois pais"
    depth, _ = _depths_and_paths(nodes, n_features)
    assert np.all(depth >= 0), "nó inalcançável a partir da raiz"
    assert np.all(nodes.feature[~internal] == -1), "folha com feature"
    assert np.all(np.isnan(nodes.threshold[~internal])), "folha com limiar"
    features = nodes.feature[internal]
    assert np.all((features >= 0) & (features < n_features)), "feature inválida"
    assert not np.isnan(nodes.threshold[internal]).any(), "corte sem limiar"


def check_conservation(nodes):
    root_total = float(nodes.class_weight[0].sum())
    assert root_total > 0
    assert np.all(nodes.class_weight >= 0)
    for node in np.flatnonzero(nodes.left != -1):
        left, right = int(nodes.left[node]), int(nodes.right[node])
        assert nodes.n_samples[node] == nodes.n_samples[left] + nodes.n_samples[right]
        np.testing.assert_allclose(
            nodes.class_weight[node],
            nodes.class_weight[left] + nodes.class_weight[right],
            rtol=1e-10, atol=1e-12 * root_total,
            err_msg=f"massa não conservada no nó {node}")
    assert np.all(nodes.class_weight.sum(axis=1) > 0), "nó com massa zero"


def check_constraints(model, nodes):
    params = model.get_params()
    n_features = model.n_features_in_
    depth, path_counts = _depths_and_paths(nodes, n_features)
    leaves = nodes.left == -1
    msl = params["min_samples_leaf"]
    assert np.all(nodes.n_samples >= msl), "nó abaixo de min_samples_leaf"
    assert np.all(nodes.n_samples[~leaves] >= 2 * msl)
    if params["max_depth"] is not None:
        assert depth.max() <= params["max_depth"]
    if params["max_leaf_nodes"] is not None:
        assert leaves.sum() <= params["max_leaf_nodes"]
    if params["max_feature_repeats"] is not None:
        assert path_counts.max() <= params["max_feature_repeats"]
    assert model.get_depth() == depth.max()
    assert model.get_n_leaves() == leaves.sum()


def check_split_admissibility(model, nodes):
    """Todo corte mantido respeita a regra de parada do objetivo."""
    params = model.get_params()
    root_total = float(nodes.class_weight[0].sum())
    tolerance = 1e-9
    for node in np.flatnonzero(nodes.left != -1):
        parent = nodes.class_weight[node]
        left = nodes.class_weight[int(nodes.left[node])]
        right = nodes.class_weight[int(nodes.right[node])]
        total = parent.sum()
        if params["objective"] == "gini":
            assert gini(parent) > 0.0, f"nó puro dividido: {node}"
            local = gini(parent) - (left.sum() * gini(left)
                                    + right.sum() * gini(right)) / total
            weighted = total / root_total * local
            assert local >= -tolerance, f"ganho Gini negativo no nó {node}"
            assert weighted >= params["min_impurity_decrease"] - tolerance
        else:
            positive = model.classes_.tolist().index(params["positive_class"])
            support = params["min_support"]
            assert total >= support and parent[positive] > 0
            parent_precision = parent[positive] / total
            assert parent_precision < params["min_precision"], (
                f"nó já elegível foi dividido: {node}")
            children = [child[positive] / child.sum()
                        for child in (left, right) if child.sum() >= support]
            assert children, f"nenhum filho com suporte no nó {node}"
            gain = max(children) - parent_precision
            assert gain > 0.0
            assert gain >= params["min_impurity_decrease"] - tolerance


def check_train_coherence(model, X, y, sample_weight=None):
    """As linhas de treino chegam às folhas com as massas gravadas no fit."""
    nodes = model.nodes_
    X_active, encoded, weights, classes = _active(X, y, sample_weight)
    np.testing.assert_array_equal(model.classes_, classes)
    leaf_ids = model.apply(X_active)
    assert np.all(nodes.left[leaf_ids] == -1), "apply devolveu nó interno"
    counts = np.bincount(leaf_ids, minlength=len(nodes.left))
    leaves = np.flatnonzero(nodes.left == -1)
    np.testing.assert_array_equal(counts[leaves], nodes.n_samples[leaves])
    mass = np.zeros_like(nodes.class_weight)
    np.add.at(mass, (leaf_ids, encoded), weights)
    np.testing.assert_allclose(mass[leaves], nodes.class_weight[leaves],
                               rtol=1e-10, atol=1e-12 * weights.sum())
    np.testing.assert_allclose(nodes.class_weight[0],
                               np.bincount(encoded, weights, len(classes)),
                               rtol=1e-10, atol=1e-12 * weights.sum())


def check_probabilities(model, X):
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.predict(X),
                                  model.classes_[proba.argmax(axis=1)])
    params = model.get_params()
    if (params["monotonic_cst"] is None and params["leaf_smoothing"] == 0
            and params.get("leaf_shrinkage", 0) == 0):
        leaf_mass = model.nodes_.class_weight[model.apply(X)]
        np.testing.assert_allclose(
            proba, leaf_mass / leaf_mass.sum(axis=1, keepdims=True),
            rtol=0, atol=1e-12)


def check_monotonic(model, X, rng, n_pairs=400):
    """Aumentar uma feature restrita (valores finitos) não viola a direção."""
    directions = model.monotonic_cst_
    if directions is None:
        return
    positive = model.monotonic_positive_class_index_
    X = np.asarray(X, dtype=np.float32)
    finite_rows = X[np.isfinite(X).all(axis=1)]
    if len(finite_rows) == 0:
        return
    base = finite_rows[rng.integers(0, len(finite_rows), n_pairs)]
    for feature in np.flatnonzero(directions != 0):
        moved = base.copy()
        moved[:, feature] += np.abs(rng.normal(size=n_pairs)).astype(np.float32)
        change = (model.predict_proba(moved)[:, positive]
                  - model.predict_proba(base)[:, positive])
        assert np.all(directions[feature] * change >= -1e-12), (
            f"monotonicidade violada na feature {feature}")


def _node_rows(nodes, X):
    """Índices das linhas (de X) que chegam a cada nó."""
    rows = [None] * len(nodes.left)
    rows[0] = np.arange(len(X))
    stack = [0]
    while stack:
        node = stack.pop()
        left = int(nodes.left[node])
        if left == -1:
            continue
        values = X[rows[node], int(nodes.feature[node])]
        missing = np.isnan(values)
        goes_left = np.where(missing, bool(nodes.missing_left[node]),
                             values <= nodes.threshold[node])
        rows[left] = rows[node][goes_left]
        rows[int(nodes.right[node])] = rows[node][~goes_left]
        stack.extend((left, int(nodes.right[node])))
    return rows


def _gini_rows(mass):
    total = mass.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = mass / total[..., None]
    return np.where(total > 0, 1.0 - np.square(share).sum(axis=-1), 0.0)


def _score(parent, left, right, objective, positive, min_support):
    """Ganho local de cada candidato (vetorizado); -inf quando inválido."""
    if objective == "gini":
        total = parent.sum()
        gain = _gini_rows(parent) - (left.sum(-1) * _gini_rows(left)
                                     + right.sum(-1) * _gini_rows(right)) / total
        return np.where((left.sum(-1) > 0) & (right.sum(-1) > 0), gain, -np.inf)
    parent_precision = parent[positive] / parent.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = [np.where(child.sum(-1) >= min_support,
                           child[..., positive] / child.sum(-1), -np.inf)
                  for child in (left, right)]
    best = np.maximum(*scores)
    return np.where(np.isfinite(best), best - parent_precision, -np.inf)


def _feature_candidates(X, y_enc, w, rows, feature, splitter, edges, n_classes, msl):
    """Todos os cortes do motor numa feature, com massas e linhas à esquerda.

    Candidatos: limiares (bordas de bin no hist, pontos entre valores
    distintos do nó no exact) x direção de NaN, mais o corte só-NaN.
    Devolve ``(left_mass[k, C], left_rows(i))`` só com os cortes que
    respeitam ``min_samples_leaf``. Independente dos kernels; usa só numpy.
    """
    values = X[rows, feature]
    missing = np.isnan(values)
    n_missing = int(missing.sum())
    order = np.argsort(values[~missing], kind="mergesort")
    sorted_values = values[~missing][order]
    finite_rows = rows[~missing][order]
    missing_rows = rows[missing]
    onehot = np.zeros((len(finite_rows), n_classes))
    onehot[np.arange(len(finite_rows)), y_enc[finite_rows]] = w[finite_rows]
    cumulative = np.cumsum(onehot, axis=0)
    missing_mass = np.bincount(y_enc[missing_rows], w[missing_rows], n_classes)
    if splitter == "hist":
        cuts = np.asarray(edges[feature], dtype=np.float64)
        n_left = np.unique(np.searchsorted(
            sorted_values.astype(np.float64), cuts, side="right"))
    else:
        n_left = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    n_left = n_left[(n_left > 0) & (n_left < len(sorted_values))]
    cand_n = [n_left]
    cand_nan = [np.zeros(len(n_left), dtype=bool)]
    if n_missing:
        cand_n.append(n_left)
        cand_nan.append(np.ones(len(n_left), dtype=bool))
        if len(sorted_values):  # só-NaN: finitos à esquerda, NaN à direita
            cand_n.append(np.array([len(sorted_values)]))
            cand_nan.append(np.array([False]))
    cand_n = np.concatenate(cand_n)
    cand_nan = np.concatenate(cand_nan)
    left_count = cand_n + np.where(cand_nan, n_missing, 0)
    keep = (left_count >= msl) & (len(rows) - left_count >= msl)
    cand_n, cand_nan = cand_n[keep], cand_nan[keep]
    left_mass = cumulative[cand_n - 1] + np.where(cand_nan[:, None], missing_mass, 0.0)

    def left_rows(i):
        chosen = finite_rows[:cand_n[i]]
        return np.concatenate([chosen, missing_rows]) if cand_nan[i] else chosen

    return left_mass, left_rows


def _best_candidate_gain(model, X, y_enc, w, rows, allowed_features):
    """Força bruta: melhor ganho local entre todos os candidatos do motor."""
    params = model.get_params()
    n_classes = len(model.classes_)
    positive = (model.classes_.tolist().index(params["positive_class"])
                if params["objective"] == "precision" else -1)
    parent = np.bincount(y_enc[rows], w[rows], n_classes)
    best = -np.inf
    for feature in allowed_features:
        left, _ = _feature_candidates(
            X, y_enc, w, rows, feature, params["splitter"], model.bin_edges_,
            n_classes, params["min_samples_leaf"])
        if len(left):
            gains = _score(parent, left, parent - left, params["objective"],
                           positive, params["min_support"])
            best = max(best, float(gains.max()))
    return best, parent


def check_local_optimality(model, X, y, sample_weight=None, atol=1e-9):
    """Cada corte é o melhor candidato do nó; folhas livres não tinham corte.

    Pega erros que as verificações estruturais não veem: um corte escolhido
    com o histograma errado continua gerando uma árvore bem formada.
    Não vale com ``gain_tolerance > 0`` (busca aproximada por desenho).
    """
    params = model.get_params()
    if params["gain_tolerance"] > 0:
        return
    nodes = model.nodes_
    X_active, y_enc, w, _ = _active(X, y, sample_weight)
    rows = _node_rows(nodes, X_active)
    depth, path_counts = _depths_and_paths(nodes, model.n_features_in_)
    root_total = w.sum()
    order = np.arange(model.n_features_in_)
    repeats = params["max_feature_repeats"]
    n_leaves = int((nodes.left == -1).sum())
    budget_binding = (params["max_leaf_nodes"] is not None
                      and n_leaves >= params["max_leaf_nodes"])
    check_leaves = params["ccp_alpha"] == 0 and not budget_binding
    for node in range(len(nodes.left)):
        allowed = order if repeats is None else order[path_counts[node] < repeats]
        internal = nodes.left[node] != -1
        if not internal and not check_leaves:
            continue
        best, parent = _best_candidate_gain(model, X_active, y_enc, w, rows[node], allowed)
        if internal:
            left = nodes.class_weight[int(nodes.left[node])]
            right = nodes.class_weight[int(nodes.right[node])]
            positive = (model.classes_.tolist().index(params["positive_class"])
                        if params["objective"] == "precision" else -1)
            chosen = float(_score(parent, left[None], right[None], params["objective"],
                                  positive, params["min_support"])[0])
            assert chosen >= best - atol, (
                f"nó {node}: corte escolhido {chosen:.12g} < melhor {best:.12g}")
            continue
        # Folha: se ainda era divisível, nenhum candidato podia ser aceito.
        n_rows = len(rows[node])
        mass = nodes.class_weight[node]
        if n_rows < 2 * params["min_samples_leaf"] or len(allowed) == 0:
            continue
        if params["max_depth"] is not None and depth[node] >= params["max_depth"]:
            continue
        if params["objective"] == "gini":
            if gini(mass) <= 0 or not np.isfinite(best):
                continue
            priority = mass.sum() / root_total * best
        else:
            positive = model.classes_.tolist().index(params["positive_class"])
            if (mass.sum() < params["min_support"] or mass[positive] <= 0
                    or mass[positive] / mass.sum() >= params["min_precision"]
                    or not best > 0):
                continue
            priority = best
        assert priority < params["min_impurity_decrease"] + atol, (
            f"folha {node} tinha corte aceitável (prioridade {priority:.12g})")


def _chosen_priority(model, nodes, node, root_total):
    """Prioridade best-first do corte gravado no nó (a partir das massas)."""
    params = model.get_params()
    parent = nodes.class_weight[node]
    left = nodes.class_weight[int(nodes.left[node])]
    right = nodes.class_weight[int(nodes.right[node])]
    positive = (model.classes_.tolist().index(params["positive_class"])
                if params["objective"] == "precision" else -1)
    gain = float(_score(parent, left[None], right[None], params["objective"],
                        positive, params["min_support"])[0])
    if params["objective"] == "gini":
        return parent.sum() / root_total * gain
    return gain


def _leaf_priority(model, nodes, node, rows, depth, allowed, X, y_enc, w, root_total):
    """Prioridade com que a folha teria entrado no heap; None se não entraria."""
    params = model.get_params()
    mass = nodes.class_weight[node]
    if len(rows) < 2 * params["min_samples_leaf"] or len(allowed) == 0:
        return None
    if params["max_depth"] is not None and depth >= params["max_depth"]:
        return None
    if params["objective"] == "gini":
        if gini(mass) <= 0:
            return None
    else:
        positive = model.classes_.tolist().index(params["positive_class"])
        if (mass.sum() < params["min_support"] or mass[positive] <= 0
                or mass[positive] / mass.sum() >= params["min_precision"]):
            return None
    best, _ = _best_candidate_gain(model, X, y_enc, w, rows, allowed)
    if not np.isfinite(best) or (params["objective"] == "precision" and best <= 0):
        return None
    priority = (mass.sum() / root_total * best if params["objective"] == "gini"
                else best)
    if priority < params["min_impurity_decrease"]:
        return None
    return priority


def check_best_first_order(model, X, y, sample_weight=None, atol=1e-9):
    """Com orçamento de folhas, cada expansão era a de maior prioridade.

    Os ids dos filhos saem na ordem de expansão, então a sequência pode ser
    reconstruída da árvore: no passo t, a fronteira é o conjunto de nós já
    criados e ainda não expandidos que entrariam no heap. O nó expandido
    precisa ter prioridade >= a de todos eles; no empate, o heap prefere o
    menor id. Pega heap invertido, FIFO, prioridade sem o peso da massa ou
    desempate trocado. Não vale após poda (ids deixam de seguir a expansão)
    nem com busca aproximada (``gain_tolerance > 0``).
    """
    params = model.get_params()
    if (params["max_leaf_nodes"] is None or params["ccp_alpha"] > 0
            or params["gain_tolerance"] > 0):
        return
    nodes = model.nodes_
    X_active, y_enc, w, _ = _active(X, y, sample_weight)
    rows = _node_rows(nodes, X_active)
    depth, path_counts = _depths_and_paths(nodes, model.n_features_in_)
    root_total = w.sum()
    order = np.arange(model.n_features_in_)
    repeats = params["max_feature_repeats"]
    internal = np.flatnonzero(nodes.left != -1)
    sequence = internal[np.argsort(nodes.left[internal])]
    expanded_at = {int(node): step for step, node in enumerate(sequence)}
    priority = {}
    for node in range(len(nodes.left)):
        if node in expanded_at:
            priority[node] = _chosen_priority(model, nodes, node, root_total)
        else:
            allowed = order if repeats is None else order[path_counts[node] < repeats]
            priority[node] = _leaf_priority(model, nodes, node, rows[node],
                                            depth[node], allowed, X_active,
                                            y_enc, w, root_total)
    for step, node in enumerate(sequence):
        created_before = int(nodes.left[node])  # ids < isto já existiam
        chosen = priority[int(node)]
        for other in range(created_before):
            if other == node or priority[other] is None:
                continue
            if other in expanded_at and expanded_at[other] < step:
                continue  # já expandido antes
            rival = priority[other]
            assert rival <= chosen + atol, (
                f"passo {step}: expandiu nó {node} (prioridade {chosen:.12g}) "
                f"com nó {other} na fronteira (prioridade {rival:.12g})")
            if abs(rival - chosen) <= 1e-15 * max(1.0, abs(chosen)):
                assert node < other, (
                    f"passo {step}: empate de prioridade resolvido para o "
                    f"nó {node} em vez do menor id {other}")


def check_tree_invariants(model, X, y, sample_weight=None, rng=None):
    """Rode todos os invariantes de correção sobre um modelo treinado."""
    rng = np.random.default_rng(0) if rng is None else rng
    nodes = model.nodes_
    check_structure(nodes, model.n_features_in_)
    check_conservation(nodes)
    check_constraints(model, nodes)
    # A poda só remove cortes; os que ficam continuam admissíveis.
    check_split_admissibility(model, nodes)
    check_train_coherence(model, X, y, sample_weight)
    check_local_optimality(model, X, y, sample_weight)
    check_best_first_order(model, X, y, sample_weight)
    check_probabilities(model, X)
    check_monotonic(model, X, rng)


def _impurity(mass):
    """Gini não normalizado: massa x Gini (0 para nó vazio)."""
    mass = np.asarray(mass, dtype=np.float64)
    return float(mass.sum() * _gini_rows(mass))


def _best_leaf_impurity(X, y_enc, w, rows, splitter, edges, n_classes, msl):
    """Menor impureza da folha após no máximo um corte (força bruta)."""
    parent = np.bincount(y_enc[rows], w[rows], n_classes)
    best = _impurity(parent)
    if len(rows) < 2 * msl:
        return best
    for feature in range(X.shape[1]):
        left, _ = _feature_candidates(X, y_enc, w, rows, feature, splitter,
                                      edges, n_classes, msl)
        if not len(left):
            continue
        right = parent - left
        valid = (left.sum(-1) > 0) & (right.sum(-1) > 0)
        if valid.any():
            split = (left.sum(-1) * _gini_rows(left)
                     + right.sum(-1) * _gini_rows(right))[valid]
            best = min(best, float(split.min()))
    return best


def _best_block_reduction(X, y_enc, w, rows, stats, edges, n_classes):
    """Maior queda de impureza de um bloco de profundidade 2 (força bruta).

    Mesmo espaço do motor: raiz com ``root_splitter``, cada filho com no
    máximo um corte de ``child_splitter``, ``min_samples_leaf`` em todas as
    folhas. Filhos resolvidos de forma independente dada a raiz, o que é
    exato porque a impureza total é a soma das duas metades.
    """
    msl = stats["min_samples_leaf"]
    parent = np.bincount(y_enc[rows], w[rows], n_classes)
    base = _impurity(parent)
    best = 0.0
    if len(rows) < 2 * msl:
        return best
    for feature in range(X.shape[1]):
        left_mass, left_rows = _feature_candidates(
            X, y_enc, w, rows, feature, stats["root_splitter"], edges,
            n_classes, msl)
        for i in range(len(left_mass)):
            if left_mass[i].sum() <= 0 or (parent - left_mass[i]).sum() <= 0:
                continue
            lrows = left_rows(i)
            rrows = np.setdiff1d(rows, lrows, assume_unique=True)
            leaves = sum(_best_leaf_impurity(
                X, y_enc, w, child, stats["child_splitter"], edges,
                n_classes, msl) for child in (lrows, rrows))
            best = max(best, base - leaves)
    return best


def check_multilevel_blocks(model, X, y, sample_weight=None, rtol=1e-9):
    """Cada bloco do multinível é o ótimo de profundidade 2 do seu nó.

    Blocos começam nas profundidades 0 e 2. Para cada um, a queda de
    impureza da subárvore escolhida (raiz + até dois cortes) tem de igualar
    a da melhor subárvore da força bruta: menor é busca incompleta, maior
    é corte fora do espaço de candidatos. Bloco que virou folha não podia
    ter queda positiva. Não diz nada sobre a árvore composta.
    """
    stats = model.fit_stats_
    nodes = model.nodes_
    n_classes = len(model.classes_)
    X_active, y_enc, w, _ = _active(X, y, sample_weight)
    rows = _node_rows(nodes, X_active)
    depth, _ = _depths_and_paths(nodes, model.n_features_in_)
    assert depth.max() <= stats["max_depth"]
    for block in np.flatnonzero((depth % 2 == 0) & (depth < stats["max_depth"])):
        mass = nodes.class_weight[block]
        chosen = 0.0
        if nodes.left[block] != -1:
            leaves = []
            for child in (int(nodes.left[block]), int(nodes.right[block])):
                if nodes.left[child] == -1:
                    leaves.append(child)
                else:
                    leaves.extend((int(nodes.left[child]), int(nodes.right[child])))
            chosen = _impurity(mass) - sum(_impurity(nodes.class_weight[leaf])
                                           for leaf in leaves)
        if stats["max_depth"] - depth[block] == 1:
            # Profundidade ímpar: o último bloco é um corte guloso único.
            base = _impurity(np.bincount(y_enc[rows[block]], w[rows[block]], n_classes))
            best = max(0.0, base - _best_leaf_impurity(
                X_active, y_enc, w, rows[block], stats["root_splitter"],
                model.bin_edges_, n_classes, stats["min_samples_leaf"]))
        else:
            best = _best_block_reduction(X_active, y_enc, w, rows[block], stats,
                                         model.bin_edges_, n_classes)
        tolerance = rtol * max(1.0, float(mass.sum()))
        assert abs(chosen - best) <= tolerance, (
            f"bloco no nó {block} (profundidade {depth[block]}): queda "
            f"escolhida {chosen:.12g} != ótimo do bloco {best:.12g}")
