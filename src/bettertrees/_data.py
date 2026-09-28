"""Node containers and input validation; no Numba here.

``NodeArrays`` and ``Split`` only group arrays/scalars: there is no object per
node. Validation converts X to C-contiguous float32 and drops zero-weight rows
before any binning, support count or split.
"""

from typing import NamedTuple

import numpy as np
from sklearn.utils.multiclass import check_classification_targets


class NodeArrays(NamedTuple):
    """Arrays contíguos; raiz 0, filhos -1 em folhas, feature -1 em folhas.

    Cada posição identifica um nó. class_weight tem shape (capacidade, K)
    e contém massas por classe, não probabilidades. Nenhuma folha publicada
    pode ter massa total zero. threshold é float64; X é float32.
    O builder retorna arrays recortados aos nós efetivamente usados.
    """

    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    missing_left: np.ndarray
    class_weight: np.ndarray
    n_samples: np.ndarray


class Split(NamedTuple):
    """Melhor corte: feature=-1 indica ausência de corte admissível.

    gain é a redução LOCAL de Gini; o builder pondera pela massa relativa
    à raiz ao aplicar min_impurity_decrease. threshold sempre usa a escala
    original; bin_threshold=-1 no motor exato. n_left conta linhas ativas,
    independentemente da magnitude de seus pesos positivos.
    """

    feature: int
    threshold: float
    missing_left: bool
    gain: float
    n_left: int
    bin_threshold: int


def validate_X(X, *, n_features=None):
    """Converta matriz numérica densa em float32 C-contiguous, permitindo NaN.

    Rejeita vazios, complexos, infinito (inclusive overflow da conversão),
    matrizes esparsas e número incorreto de colunas. Não altera o argumento.
    O custo desta função integra o tempo de fit/predict da API pública.
    """
    raw = np.asarray(X)
    if raw.ndim != 2 or 0 in raw.shape or raw.dtype.kind not in "biuf":
        raise ValueError("X must be a non-empty dense numeric array.")
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.ascontiguousarray(raw, dtype=np.float32)
    if np.isinf(out).any():
        raise ValueError("X cannot contain infinity or values outside the float32 range.")
    if n_features is not None and out.shape[1] != n_features:
        raise ValueError("The number of columns differs from training.")
    return out


def prepare_training_data(X, y, sample_weight=None):
    """Valide treino e retorne (X32, y_int32, pesos64, classes_originais).

    Aceita alvo unidimensional binário/multiclasse, inclusive strings.
    classes segue np.unique; seu índice resolve empates na previsão.
    Pesos devem ser finitos, não negativos, alinhados e ter soma positiva.
    Linhas de peso zero não participam de bins, suporte mínimo ou splits;
    classes_ preserva todas as classes observadas antes dessa exclusão.
    Os dados de entrada nunca são modificados e nenhum holdout é usado.
    """
    X = validate_X(X)
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != len(X):
        raise ValueError("y must have one label per row of X.")
    check_classification_targets(target)
    classes, encoded = np.unique(target, return_inverse=True)
    weights = (np.ones(len(X), dtype=np.float64) if sample_weight is None
               else np.asarray(sample_weight, dtype=np.float64))
    if weights.shape != (len(X),) or not np.isfinite(weights).all():
        raise ValueError("sample_weight deve ser um vetor finito alinhado a X.")
    if (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Sample weights must be non-negative with a positive sum.")
    if not np.isfinite(weights.sum()):
        raise ValueError("Soma de pesos excede a capacidade de float64.")
    active = weights > 0
    if not active.all():
        X, encoded, weights = X[active], encoded[active], weights[active]
    return (np.ascontiguousarray(X), np.ascontiguousarray(encoded, dtype=np.int32),
            np.ascontiguousarray(weights), classes)


def allocate_nodes(capacity, n_classes):
    """Aloque arrays de nós ainda NÃO treinados, todos inicialmente folhas.

    O builder deve começar pequeno, ampliar geometricamente e preencher
    massas antes de publicar a árvore; evitar alocar 2**max_depth de saída.
    """
    if capacity < 1 or n_classes < 1:
        raise ValueError("capacity e n_classes devem ser positivos.")
    return NodeArrays(
        np.full(capacity, -1, dtype=np.int32),
        np.full(capacity, -1, dtype=np.int32),
        np.full(capacity, -1, dtype=np.int32),
        np.full(capacity, np.nan, dtype=np.float64),
        np.zeros(capacity, dtype=np.bool_),
        np.zeros((capacity, n_classes), dtype=np.float64),
        np.zeros(capacity, dtype=np.int64),
    )
