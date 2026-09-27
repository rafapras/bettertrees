"""API pública das somas interpretáveis: sklearn, regras, contribuições e gráficos."""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from arvore_rapida.capacidade import (
    AdditiveTreeBooster,
    BoostedOptimalTrees,
    FIGSClassifier,
    SumOfOptimalTrees,
)

matplotlib.use("Agg")

ESTIMATORS = [SumOfOptimalTrees(), FIGSClassifier(), AdditiveTreeBooster(max_rounds=20),
              BoostedOptimalTrees(depth=2, max_rounds=20)]


def _expected_failures(est):
    if isinstance(est, AdditiveTreeBooster | BoostedOptimalTrees):
        # early stopping sorteia 15% das LINHAS para validação: repetir uma linha
        # não equivale a dar peso 2 (a linha repetida pode cair nos dois lados)
        return {"check_sample_weight_equivalence_on_dense_data":
                "validação interna por linhas não é invariante a repetição"}
    return {}


@parametrize_with_checks(ESTIMATORS, expected_failed_checks=_expected_failures)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def _data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["idade", "renda", "divida", "score"])
    X.loc[rng.random(n) < 0.1, "renda"] = np.nan
    logit = X.idade + np.nan_to_num(X.renda) * X.divida + 0.5 * np.sign(X.score)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


MODELS = [
    lambda: SumOfOptimalTrees(n_trees=3, depth=2, extra_stumps=2, learning_rate=0.5),
    lambda: SumOfOptimalTrees(n_trees=3, depth=2, search="greedy"),
    lambda: FIGSClassifier(max_splits=8),
    lambda: AdditiveTreeBooster(max_rounds=10),
    lambda: BoostedOptimalTrees(depth=3, max_rounds=10),
]


@pytest.mark.parametrize("make", MODELS)
def test_contributions_reproduce_logit_and_rules_are_consistent(make):
    X, y = _data()
    m = make().fit(X, y)
    c = m.predict_contributions(X)
    np.testing.assert_allclose(m.base_margin_ + c.sum(axis=1), m.decision_function(X),
                               atol=1e-12)
    assert list(m.feature_names_in_) == list(X.columns)
    rules = m.rules()
    # uma linha por folha; as folhas de cada árvore somam a contribuição daquela árvore
    trees = m._explain_trees()
    assert len(rules) == sum(len(t.leaves) for t in trees)
    for k in range(c.shape[1]):
        assert set(np.round(c[:, k], 12)) <= {round(v, 12) for kk, _, v in rules if kk == k}
    d = json.loads(json.dumps(m.to_dict()))
    assert d["link"] == "logit" and len(d["trees"]) == c.shape[1]
    assert m.explain().startswith("logit P(y = 1)")


def test_rules_merge_path_into_intervals_and_flag_missing():
    X, y = _data()
    m = SumOfOptimalTrees(n_trees=4, depth=2, learning_rate=0.5).fit(X, y)
    text = [c for _, conds, _ in m.rules() for c in conds]
    # nunca duas condições da mesma feature no mesmo caminho
    for _, conds, _ in m.rules():
        feats = [next(n for n in X.columns if n in c) for c in conds]
        assert len(feats) == len(set(feats))
    # só renda tinha NaN no treino: só ela pode aparecer como "ou ausente"
    assert all("renda" in c for c in text if "ausente" in c)
    assert not m.nan_features_[0] and m.nan_features_[1]


def test_shape_functions_and_plots():
    X, y = _data()
    m = SumOfOptimalTrees(n_trees=2, depth=2, extra_stumps=3, learning_rate=0.5).fit(X, y)
    shapes = m.shape_functions()
    assert shapes  # os tocos são efeitos principais
    for f, (vals, count) in shapes.items():
        assert len(vals) == len(m.bin_edges_[f]) + 2 and count >= 1
    # toda árvore é efeito principal (conta na forma) ou interação
    assert len(m.interaction_trees()) + sum(c for _, c in shapes.values()) == len(m.trees_)
    fig = m.plot_shapes()
    assert fig.axes
    ax = m.plot_contributions(X.iloc[0].to_numpy())
    assert "P =" in ax.get_xlabel()


def test_predict_rejects_wrong_width_and_unfitted():
    from sklearn.exceptions import NotFittedError
    X, y = _data()
    with pytest.raises(NotFittedError):
        FIGSClassifier().predict(X)
    m = FIGSClassifier(max_splits=4).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X.iloc[:, :3])
