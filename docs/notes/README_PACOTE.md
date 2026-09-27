# arvore-rapida

Árvores de decisão rápidas e **somas interpretáveis de árvores** para
classificação binária, com API do scikit-learn.

```bash
pip install -e .                 # numpy, numba, scikit-learn
pip install -e ".[lightgbm,plot]"  # triagem de features com LightGBM (p > 128) e gráficos
```

## Qual modelo usar

| regime | modelo | forma do resultado |
|---|---|---|
| uma árvore | `FastDecisionTreeClassifierCV` | árvore única; folhas por shrinkage hierárquico escolhido por CV |
| **interpretável (4–16 cortes)** | `SumOfOptimalTrees` ou `FIGSClassifier` | `logit(p) = base + Σ árvore_k(x)`, poucas árvores rasas |
| capacidade média (32–64 cortes) | `SumOfOptimalTrees` | a mesma soma, com mais árvores |
| capacidade livre | `AdditiveTreeBooster` | soma longa de árvores d2 com early stopping (estrutura aditiva + pares) |

`SumOfOptimalTrees(n_trees=K, depth=2, extra_stumps=r)` usa `3K + r` cortes:
cada árvore d2 é a árvore Newton **ótima** (busca exaustiva nos bins) no resíduo,
seguida de backfitting das folhas. `search="greedy"` troca a busca ótima pela
gulosa (o controle pareado dos benchmarks). `FIGSClassifier(max_splits=b)`
cresce várias árvores ao mesmo tempo, um corte por vez.

Os hiperparâmetros que importam são `lam` (regularização L2 das folhas) e
`learning_rate` (shrinkage da soma). Os defaults `"auto"` seguem a mediana do
tuning no benchmark: na soma, `learning_rate = 1/(1 + 0,1·n_trees)` e `lam = 2`;
no FIGS, `lam = 2·max_splits` (orçamentos maiores precisam de mais
regularização). Os valores efetivos ficam em `lam_` e `learning_rate_`. Os estimadores são compatíveis com o
scikit-learn, então `GridSearchCV` e `cross_val_score` funcionam direto:

```python
from sklearn.model_selection import GridSearchCV
from arvore_rapida import SumOfOptimalTrees

grid = GridSearchCV(SumOfOptimalTrees(n_trees=5, depth=2, extra_stumps=1),  # 16 cortes
                    {"lam": [0.3, 3, 30], "learning_rate": [0.1, 0.3, 1.0]},
                    scoring="neg_log_loss", cv=3)
model = grid.fit(X_train, y_train).best_estimator_
```

## Forma interpretável

Toda soma expõe a mesma saída:

```python
print(model.explain())            # scorecard: base + valor da folha de cada árvore
model.rules()                     # [(árvore, condições, valor no logit)]
model.to_dict()                   # o mesmo em JSON (implantação sem o pacote)
model.predict_contributions(X)    # (n, n_árvores); base + soma = decision_function
model.plot_contributions(x)       # waterfall de uma previsão
model.plot_shapes()               # funções de forma das árvores de uma feature só
model.get_trees()                 # cada árvore em arrays (como tree_ do sklearn), cortes em valores reais
print(model.export_text())        # as árvores desenhadas em texto
model.predict_proba(X)            # (n, 2), colunas na ordem de classes_
```

```
logit P(y = 1) = base -0.0213 + soma de 4 árvores (8 cortes)
árvore 1:
  -0.8202  se idade <= -1.11
  -0.3510  se -1.11 < idade <= -0.298
  +0.5747  se idade > 0.5316
árvore 2:
  -0.2236  se (renda <= 1.113 ou ausente) e idade <= -0.1642
  +1.0176  se renda > 1.113 e divida > 0.1527
  ...
```

As condições de um caminho são fundidas em um intervalo por feature. NaN vai
sempre para o lado `<=`; "ou ausente" aparece só nas features que tinham NaN no
treino. Os nomes vêm das colunas do DataFrame (`feature_names_in_`).

## Garantias e limites (v0.1)

- Só classificação binária (as tags do sklearn declaram isso; multiclasse
  levanta `ValueError`).
- Aceita NaN; não aceita matriz esparsa.
- `check_estimator` do scikit-learn passa nos quatro estimadores, exceto a
  equivalência "peso 2 = linha repetida" no `AdditiveTreeBooster` e no
  `BoostedOptimalTrees`, que sorteiam linhas para o early stopping.
- Com mais de 128 features, a busca d2/d3 fica restrita às 128 de maior
  importância de ganho de um LightGBM (sem LightGBM instalado, triagem FAST, que
  não enxerga interação pura sem efeito principal).
- `arvore_rapida.capacidade` contém também módulos experimentais (razões,
  destilação, RuleFit, triagem): fora da API estável.

Resultados de benchmark: `resultados/arvore_rapida/eficiencia/` (relatórios
HTML por camada) e `bench/eff_claims.py` (teses pareadas por dataset).
