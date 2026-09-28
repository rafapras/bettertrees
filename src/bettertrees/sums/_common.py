"""Preparo comum: validação, bins e margem base."""

import numpy as np

from ..bins import fit_bin_edges, transform_bins_row_major
from ._kernels import logistic_grad_hess


def as_float_matrix(X):
    """X float64 C-contiguous (os kernels de combinação leem valores brutos)."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X


def as_target(y, n):
    """Alvo binário 0/1 ou suave em [0, 1], float64."""
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if y.shape != (n,) or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
        raise ValueError("y must be a vector in [0, 1] aligned with X.")
    return y


def as_weights(sample_weight, n):
    if sample_weight is None:
        return np.ones(n)
    w = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float64))
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("sample_weight must be finite, non-negative and aligned with X.")
    return w


def binned(X, max_bins):
    """(Xb uint8, edges, nb int64) com bin 0 = NaN e nb[j] = len(edges[j]) + 2."""
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    edges = fit_bin_edges(X32, max_bins)
    Xb = transform_bins_row_major(X32, edges)
    nb = np.array([len(e) + 2 for e in edges], dtype=np.int64)
    return Xb, edges, nb


def rebin(X, edges):
    return transform_bins_row_major(np.ascontiguousarray(X, dtype=np.float32), edges)


def bin_threshold(edges_j, t):
    """Valor do corte ``x <= valor`` equivalente ao bin t (NaN sempre à esquerda)."""
    if t < 0:
        return np.nan
    if t == 0:
        return -np.inf
    return float(edges_j[t - 1])


def base_margin(y, w):
    """Logit da média ponderada: margem constante inicial."""
    m = float(np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6))
    return np.log(m / (1 - m))


def grad_hess(y, margin, w):
    return logistic_grad_hess(y, np.ascontiguousarray(margin, dtype=np.float64), w)


def sigmoid(m):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(m, dtype=np.float64)))


def top_features(Xb, g, h, w, nb, lam, min_weight, k, probe=8):
    """As k features mais promissoras para a busca d2/d3 (ordem crescente de índice).

    Pontuação = max(ganho do melhor corte único, melhor quadrante FAST que a
    feature forma com as ``probe`` melhores features univariadas). Só o ganho
    univariado não enxerga interação pura (ex.: sign(x1·x2) sem efeito
    principal), que é justamente o que a d2 procura. O resultado da busca é
    ótimo DENTRO desse conjunto. Com p <= k devolve None (busca completa).
    """
    from ._kernels import best_cut_1d, hist_1d, quadrant_scores
    p = Xb.shape[1]
    if k is None or p <= k:
        return None
    gains, _ = best_cut_1d(hist_1d(Xb, g, h, w, int(nb.max())), nb, lam, min_weight)
    score = gains.copy()
    top = np.argsort(-gains, kind="stable")[:probe]
    pairs = np.array([(i, j) for i in top for j in range(p) if j != i], dtype=np.int64)
    q, _, _ = quadrant_scores(Xb, g, h, w, nb, pairs, lam, min_weight)
    for (i, j), v in zip(pairs, q):
        score[i] = max(score[i], v)
        score[j] = max(score[j], v)
    return np.sort(np.argsort(-score, kind="stable")[:k])


def teacher_top_features(X, y, k, w=None, random_state=0):
    """Top-k features pela importância de GANHO de um LightGBM (o clássico).

    Roda uma vez por fit (100 árvores, 31 folhas, taxa 0,1). A importância de
    ganho acumula também os splits dentro de interações, então enxerga
    feature que só age em interação. Com p <= k devolve None. Sem o LightGBM
    instalado (dependência opcional), cai na triagem FAST (``top_features``) no
    gradiente da margem constante.
    """
    if k is None or X.shape[1] <= k:
        return None
    try:
        import lightgbm as lgb
    except ImportError:
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64)
        yf = np.asarray(y, dtype=np.float64)
        Xb, _, nb = binned(X, 32)
        g, h = grad_hess(yf, np.full(len(X), base_margin(yf, w)), w)
        return top_features(Xb, g, h, w, nb, 1.0, 20.0, k)
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                               importance_type="gain", n_jobs=1, random_state=random_state,
                               verbose=-1)
    model.fit(X, y, sample_weight=w)
    return np.sort(np.argsort(-model.feature_importances_, kind="stable")[:k])


_VALIDATE = dict(dtype=np.float64, order="C", ensure_all_finite="allow-nan", accept_sparse=False)


def fit_inputs(est, X, y, sample_weight=None, y_soft=None):
    """Validação de ``fit`` pela API do sklearn (define ``n_features_in_`` e
    ``feature_names_in_``). Devolve (X float64 C, classes, alvo em [0, 1], pesos)
    e grava ``nan_features_``. ``y_soft`` troca o alvo 0/1 por probabilidades,
    mantendo as classes de ``y``."""
    from sklearn.utils.multiclass import check_classification_targets, type_of_target
    from sklearn.utils.validation import _check_sample_weight, validate_data
    X, y = validate_data(est, X, y, reset=True, **_VALIDATE)
    check_classification_targets(y)
    y_type = type_of_target(y, input_name="y", raise_unknown=True)
    if y_type != "binary":  # mensagem padrão do sklearn para estimadores binários
        raise ValueError("Only binary classification is supported. The type of the target "
                         f"is {y_type}.")
    classes, encoded = np.unique(y, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("Binary classification needs two classes; y has only one class.")
    target = encoded.astype(np.float64) if y_soft is None else as_target(y_soft, len(X))
    w = _check_sample_weight(sample_weight, X, dtype=np.float64, ensure_non_negative=True)
    if not w.sum() > 0:
        raise ValueError("sample_weight sums to zero.")
    est.nan_features_ = np.isnan(X).any(axis=0)
    return X, classes, target, np.ascontiguousarray(w)


def predict_input(est, X, attribute):
    """Validação de ``predict``: exige o ajuste e o mesmo número/nomes de colunas."""
    from sklearn.utils.validation import check_is_fitted, validate_data
    check_is_fitted(est, attribute)
    return validate_data(est, X, reset=False, **_VALIDATE)
