"""Additive boosting of shallow optimal trees (depth 1 or 2) in logit space.

Each round picks the **optimal** Newton tree of depth ``depth`` over the bins
(exhaustive search, not greedy; with more than ``max_features_d2`` features,
optimal within the features with the largest single-cut gain in that round),
with leaf values from a Newton step and shrinkage ``learning_rate``. Early
stopping on a validation fraction. The final model is a sum of readable
terms: each term is a rule of at most two cuts that adds a value to the logit
(one scorecard line).

References
----------
Lou, Caruana, Gehrke, Hooker. "Accurate Intelligible Models with Pairwise
Interactions." KDD 2013 (GA2M), and Nori, Jenkins, Koch, Caruana.
"InterpretML: A Unified Framework for Machine Learning Interpretability." 2019
(EBM): the additive shape of the model.
Chen, Guestrin. "XGBoost." KDD 2016: second-order gain and leaf values.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ._common import (
    base_margin,
    bin_threshold,
    binned,
    fit_inputs,
    grad_hess,
    predict_input,
    rebin,
    sigmoid,
    teacher_top_features,
    top_features,
)
from ._kernels import best_cut_1d, best_depth2, depth2_leaf_ids, hist_1d, newton_leaf_values
from .explain import InterpretableSumMixin
from .smalltrees import SmallTree, _side


def _log_loss(y, m, w):
    p = np.clip(sigmoid(m), 1e-15, 1 - 1e-15)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


class AdditiveTreeBooster(InterpretableSumMixin, ClassifierMixin, BaseEstimator):
    """Soma longa de árvores Newton ótimas rasas (d1/d2), com early stopping.

    Capacidade livre com estrutura simples: cada termo é uma árvore de até dois
    cortes (efeito principal ou par de features) somada ao logit. A cada
    rodada, a árvore ótima no resíduo (busca exaustiva nos bins), folhas por
    passo de Newton × ``learning_rate``; para quando a log-loss da validação
    interna não melhora em ``patience`` rodadas.

    Parameters
    ----------
    depth : {1, 2}, default=2
        Profundidade de cada termo.
    learning_rate : float, default=0.3
        Shrinkage dos valores das folhas.
    max_rounds : int, default=300
        Máximo de termos.
    lam : float, default=1.0
        Regularização L2 das folhas (no ganho e no passo de Newton).
    min_weight : float, default=20.0
        Massa (hessiana) mínima por folha.
    max_bins : int, default=32
        Bins por feature.
    validation_fraction : float, default=0.15
        Fração das linhas para o early stopping (0 desliga).
    patience : int, default=20
        Rodadas sem melhora antes de parar.
    max_features_d2 : int, default=128
        Com mais features, a busca d2 usa as de maior importância (triagem).
    feature_screen : {"lgbm", "fast"}, default="lgbm"
        Triagem: importância de ganho do LightGBM ou FAST por rodada.
    random_state : int, default=0
        Semente do sorteio da validação.

    Attributes
    ----------
    trees_ : list of SmallTree
        Os termos como árvores (ver ``get_trees`` para os valores reais).
    terms_ : list
        Representação interna compacta dos termos (bins).
    base_margin_ : float
        Logit constante inicial.
    classes_, n_features_in_, feature_names_in_, bin_edges_, nan_features_
        Convenções do sklearn e dos bins.
    history_ : ndarray
        Log-loss de validação por rodada.
    """

    def __init__(self, *, depth=2, learning_rate=0.3, max_rounds=300, lam=1.0,
                 min_weight=20.0, max_bins=32, validation_fraction=0.15,
                 patience=20, max_features_d2=128, feature_screen="lgbm", random_state=0):
        self.depth = depth
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.lam = lam
        self.min_weight = min_weight
        self.max_bins = max_bins
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.max_features_d2 = max_features_d2
        self.feature_screen = feature_screen
        self.random_state = random_state

    def _term(self, Xb, g, h, w, nb):
        if self.depth == 1:
            gains, cuts = best_cut_1d(hist_1d(Xb, g, h, w, int(nb.max())), nb,
                                      self.lam, self.min_weight)
            f = int(np.argmax(gains))
            if gains[f] <= 0:
                return None
            return (f, int(cuts[f]), -1, -1, -1, -1)
        feats = (self._fixed_feats if self.feature_screen == "lgbm" else
                 top_features(Xb, g, h, w, nb, self.lam, self.min_weight, self.max_features_d2))
        if feats is not None:  # muitas features: d2 ótima dentro do top-k
            Xs = np.ascontiguousarray(Xb[:, feats])
            gains, res = best_depth2(Xs, g, h, w, np.ascontiguousarray(nb[feats]), self.lam,
                                     self.min_weight)
            f1 = int(np.argmax(gains))
            if gains[f1] <= 0:
                return None
            r = res[f1]
            remap = lambda j: int(feats[j]) if j >= 0 else -1
            return (int(feats[f1]), int(r[0]), remap(r[1]), int(r[2]), remap(r[3]), int(r[4]))
        gains, res = best_depth2(Xb, g, h, w, nb, self.lam, self.min_weight)
        f1 = int(np.argmax(gains))
        if gains[f1] <= 0:
            return None
        return (f1, *[int(v) for v in res[f1]])

    def fit(self, X, y, sample_weight=None, y_soft=None):
        """Ajusta a soma.

        Parameters
        ----------
        X : array-like ou DataFrame, shape (n, p)
            Features numéricas; NaN é aceito (vai sempre para o lado ``<=``).
            Com DataFrame, os nomes das colunas viram ``feature_names_in_``.
        y : array-like, shape (n,)
            Rótulos de duas classes.
        sample_weight : array-like, shape (n,), opcional
            Pesos não negativos das linhas.
        y_soft : array-like, shape (n,), opcional
            Alvo suave em [0, 1] (ex.: probabilidade de um professor) no lugar
            de y; ``classes_`` continua vindo de y.

        Returns
        -------
        self
        """
        if self.depth not in (1, 2):
            raise ValueError("depth must be 1 or 2.")
        X, classes, target, w = fit_inputs(self, X, y, sample_weight, y_soft)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        val = np.zeros(n, dtype=bool)
        if self.validation_fraction > 0:
            val[rng.permutation(n)[:int(round(self.validation_fraction * n))]] = True
        tr = ~val
        Xb_all, edges, nb = binned(X[tr], self.max_bins)
        self._fixed_feats = None
        if self.depth == 2 and self.feature_screen == "lgbm":
            self._fixed_feats = teacher_top_features(X[tr], (target[tr] > 0.5).astype(int),
                                                     self.max_features_d2, w[tr],
                                                     self.random_state)
        Xb_val = rebin(X[val], edges)
        y_tr, w_tr, y_val, w_val = target[tr], w[tr], target[val], w[val]
        base = base_margin(y_tr, w_tr)
        m_tr = np.full(tr.sum(), base)
        m_val = np.full(val.sum(), base)
        terms, history = [], []
        best, best_round = np.inf, 0
        for _ in range(self.max_rounds):
            g, h = grad_hess(y_tr, m_tr, w_tr)
            spec = self._term(Xb_all, g, h, w_tr, nb)
            if spec is None:
                break
            ids = depth2_leaf_ids(Xb_all, *spec)
            values = self.learning_rate * newton_leaf_values(ids, g, h, 4, self.lam)
            m_tr += values[ids]
            terms.append((spec, values))
            if val.any():
                m_val += values[depth2_leaf_ids(Xb_val, *spec)]
                loss = _log_loss(y_val, m_val, w_val)
            else:
                loss = _log_loss(y_tr, m_tr, w_tr)
            history.append(loss)
            if loss < best - 1e-12:
                best, best_round = loss, len(terms)
            elif len(terms) - best_round >= self.patience:
                break
        self.terms_ = terms[:best_round] if val.any() else terms
        self.base_margin_ = base
        self.bin_edges_ = edges
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.history_ = np.array(history)
        self.trees_ = self._terms_to_trees()
        return self

    def _terms_to_trees(self):
        """Termos como SmallTree (folhas 0–1 à esquerda, 2–3 à direita)."""
        trees = []
        for (f1, t1, fl, tl, fr, tr), values in self.terms_:
            tree = SmallTree.from_nested((f1, t1, _side(fl, tl), _side(fr, tr)))
            for child, leaves in ((tree.left[0], (0, 1)), (tree.right[0], (2, 3))):
                if tree.left[child] == -1:
                    tree.value[child] = values[leaves[0]]
                else:
                    tree.value[tree.left[child]] = values[leaves[0]]
                    tree.value[tree.right[child]] = values[leaves[1]]
            trees.append(tree)
        return trees

    def decision_function(self, X):
        """Logit de ``P(y = classes_[1])``: base + soma dos termos."""
        Xb = rebin(predict_input(self, X, "terms_"), self.bin_edges_)
        m = np.full(len(Xb), self.base_margin_)
        for spec, values in self.terms_:
            m += values[depth2_leaf_ids(Xb, *spec)]
        return m

    def scorecard(self, feature_names=None):
        """Termos legíveis: lista de (regra, valor no logit) por folha de cada termo.

        Folhas idênticas de um lado sem corte aparecem uma vez. ``x <= v``
        inclui NaN (NaN vai sempre para a esquerda).
        """
        check_is_fitted(self, "terms_")
        name = (lambda j: feature_names[j]) if feature_names is not None else (lambda j: f"x{j}")
        e = self.bin_edges_

        def cond(f, t, left):
            v = bin_threshold(e[f], t)
            return f"{name(f)} {'<=' if left else '>'} {v:.6g}"

        rows = []
        for k, ((f1, t1, fl, tl, fr, tr), values) in enumerate(self.terms_):
            for side, (fc, tc, leaves) in enumerate(((fl, tl, (0, 1)), (fr, tr, (2, 3)))):
                root = cond(f1, t1, side == 0)
                if fc < 0:
                    rows.append((k, root, float(values[leaves[0]])))
                else:
                    rows.append((k, f"{root} & {cond(fc, tc, True)}", float(values[leaves[0]])))
                    rows.append((k, f"{root} & {cond(fc, tc, False)}", float(values[leaves[1]])))
        return rows
