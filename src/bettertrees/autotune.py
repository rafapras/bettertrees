"""Escolha de capacidade e de shrinkage por validação interna.

Uma árvore best-first com L folhas é o prefixo das L-1 primeiras expansões
da árvore com L_max folhas: a ordem de expansão não depende do orçamento.
Então cada fold interno precisa de UM fit (com L_max); todas as capacidades
e todos os λ do shrinkage hierárquico são avaliados sobre ele. A escolha
final é refeita no treino inteiro com os parâmetros vencedores.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from .estimator import FastDecisionTreeClassifier
from .postprocess import expansion_steps, hierarchical_shrinkage_probabilities, prefix_leaf_ids

# Grade larga: o sklearn + HS tunado escolhe > 256 folhas e λ > 200 com frequência;
# parar em 256/200 custava 0,7% de log-loss (MENSURACAO_CAPACIDADE_PENDENTE.md).
# Folhas além de n/min_samples_leaf não são alcançadas, então o custo em n pequeno é nulo.
DEFAULT_LEAVES = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_SHRINKAGE = (1.0, 5.0, 20.0, 50.0, 200.0, 500.0, 1000.0)


def _log_loss(y, proba, weights):
    picked = np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.average(np.log(picked), weights=weights))


class FastDecisionTreeClassifierCV(ClassifierMixin, BaseEstimator):
    """Árvore best-first com número de folhas e shrinkage escolhidos por CV.

    Parameters
    ----------
    leaves_grid : sequência de inteiros >= 2
        Capacidades candidatas (máximo de folhas).
    shrinkage_grid : sequência de floats > 0
        Valores de ``leaf_shrinkage`` candidatos.
    cv : int >= 2
        Folds internos estratificados; o critério é o log-loss de validação.
    min_samples_leaf, splitter, max_bins, n_jobs, random_state
        Repassados ao ``FastDecisionTreeClassifier`` (sem poda: a
        truncagem por prefixo exige ids na ordem de expansão).
    """

    def __init__(self, *, leaves_grid=DEFAULT_LEAVES, shrinkage_grid=DEFAULT_SHRINKAGE,
                 cv=3, min_samples_leaf=5, splitter="hist", max_bins=255,
                 n_jobs=1, random_state=0):
        self.leaves_grid = leaves_grid
        self.shrinkage_grid = shrinkage_grid
        self.cv = cv
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.max_bins = max_bins
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _tree(self, **extra):
        return FastDecisionTreeClassifier(
            splitter=self.splitter, min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins, n_jobs=self.n_jobs,
            random_state=self.random_state, **extra)

    def fit(self, X, y, sample_weight=None):
        leaves = sorted({int(v) for v in self.leaves_grid})
        shrinkage = sorted({float(v) for v in self.shrinkage_grid})
        if not leaves or leaves[0] < 2:
            raise ValueError("leaves_grid must contain integers >= 2.")
        if not shrinkage or shrinkage[0] <= 0:
            raise ValueError("shrinkage_grid must contain values > 0.")
        if int(self.cv) < 2:
            raise ValueError("cv must be >= 2.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes, encoded = np.unique(y, return_inverse=True)
        weights = (np.ones(len(y)) if sample_weight is None
                   else np.asarray(sample_weight, dtype=np.float64))
        scores = np.zeros((len(leaves), len(shrinkage)))
        folds = StratifiedKFold(n_splits=int(self.cv), shuffle=True,
                                random_state=self.random_state)
        for fit_rows, val_rows in folds.split(X, encoded):
            model = self._tree(max_leaf_nodes=leaves[-1]).fit(
                X[fit_rows], y[fit_rows], sample_weight=weights[fit_rows])
            nodes = model.nodes_
            steps = expansion_steps(nodes)
            # Classes ausentes no fold interno: coluna com probabilidade ~0.
            columns = np.searchsorted(classes, model.classes_)
            X_val = model._validate_predict_X(X[val_rows])
            y_val, w_val = encoded[val_rows], weights[val_rows]
            node_probs = []
            for lam in shrinkage:
                inner = hierarchical_shrinkage_probabilities(nodes, lam)
                full = np.full((len(inner), len(classes)), 1e-15)
                full[:, columns] = inner
                node_probs.append(full)
            for i, n_leaves in enumerate(leaves):
                ids = prefix_leaf_ids(X_val, nodes, steps, n_leaves)
                for j, probs in enumerate(node_probs):
                    scores[i, j] += _log_loss(y_val, probs[ids], w_val) / int(self.cv)
        i, j = np.unravel_index(int(np.argmin(scores)), scores.shape)
        self.best_params_ = dict(max_leaf_nodes=leaves[i], leaf_shrinkage=shrinkage[j])
        self.cv_scores_ = scores
        self.best_estimator_ = self._tree(**self.best_params_).fit(
            X, y, sample_weight=sample_weight)
        self.classes_ = self.best_estimator_.classes_
        self.n_features_in_ = self.best_estimator_.n_features_in_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def apply(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.apply(X)
