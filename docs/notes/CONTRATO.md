# Contrato do `FastDecisionTreeClassifier`

O que cada parâmetro garante, quais combinações são aceitas e onde cada
garantia é testada. Se o código e esta página divergirem, é bug de um dos
dois.

## Garantias de toda árvore treinada

Verificadas em `tests/tree_invariants.py` sobre uma matriz de flags
(`tests/test_invariants.py`). Uma falha é bug, não questão de qualidade.

| Grupo | Garantia |
|---|---|
| Estrutura | Raiz 0; todo nó não-raiz tem um único pai; ids de filhos maiores que o do pai; folhas com `feature=-1` e limiar NaN |
| Conservação | Massa e contagem do pai = soma das dos filhos; nenhum nó com massa zero |
| Restrições | `min_samples_leaf` (linhas de peso positivo) em todo nó; `max_depth`; `max_leaf_nodes`; `max_feature_repeats` por caminho raiz→folha |
| Admissibilidade | Todo corte mantido obedece à regra de parada do objetivo (Gini: nó impuro e prioridade ≥ `min_impurity_decrease`; precision: pai com suporte, abaixo de `min_precision` e ganho > 0) |
| Otimalidade local | Cada corte atinge o melhor ganho entre os candidatos do motor no nó (força bruta independente). Sem `ccp_alpha` e sem orçamento de folhas esgotado, nenhuma folha divisível tinha corte aceitável. Não vale com `gain_tolerance > 0` |
| Coerência treino→previsão | As linhas de treino chegam às folhas cujas massas e contagens o builder gravou; NaN segue a direção gravada |
| Probabilidades | Finitas, em [0, 1], somam 1; `predict` = classe de maior probabilidade; sem suavização/monotonicidade, igual à frequência da folha |
| Monotonicidade | Aumentar uma feature restrita (valores finitos) não viola a direção pedida |

Equivalências (duas execuções que o algoritmo define como iguais):
`off` ≡ `bound`; `n_jobs=1` ≡ `n_jobs>1`; reuso pai-filho ≡ sem reuso;
limite de repetição que não restringe ≡ sem limite; peso 0 ≡ linha removida;
peso inteiro ≡ linha duplicada (exact); hist ≡ exact **na partição do treino**
quando os bins cobrem todos os valores (os limiares diferem por desenho).

## Parâmetros

| Parâmetro | Garante | Restrições |
|---|---|---|
| `splitter` | `hist`: candidatos nas bordas globais dos bins; `exact`: pontos médios entre valores distintos do nó | — |
| `objective` | `gini`: redução de impureza ponderada. `precision` (**experimental**): melhor precision filha menos a do pai, com suporte mínimo; não maximiza TP nem cobertura | `precision` exige `positive_class`, `search_stopping="off"`, `gain_tolerance=0`, `ccp_alpha=0` |
| `max_depth` | Profundidade ≤ valor (raiz = 0) | inteiro ≥ 1 ou `None` |
| `min_samples_leaf` | Toda folha com ≥ valor linhas de peso positivo | inteiro ≥ 1 |
| `max_leaf_nodes` | Crescimento best-first pela prioridade do objetivo; desempate pelo menor id de nó | inteiro ≥ 2 ou `None` |
| `min_impurity_decrease` | Gini: corte só entra se `W_nó/W_raiz · ganho ≥ valor`. Precision: ganho ≥ valor | finito ≥ 0 |
| `max_bins` | ≤ valor bins finitos por feature; bin 0 reservado a NaN | 2..255 |
| `search_stopping` | `bound`: descarta candidatos só com limite superior admissível (mesma árvore que `off`). No `exact`, a busca é sempre exaustiva (`fit_stats_["search_stopping_effective"]="off"`) | `heuristic` → `NotImplementedError`; `bound` rejeitado para `precision` |
| `gain_tolerance` | Tolerância de ganho **local** do bound; > 0 é aproximação sem garantia fora do treino | finito ≥ 0; só `gini` |
| `max_feature_repeats` | Cada feature aparece no máximo k vezes por caminho; k = 2 representa intervalo | inteiro ≥ 1 ou `None` |
| `monotonic_cst` | Projeção pós-crescimento da probabilidade da classe positiva (`classes_[1]`, ou `positive_class` em precision) sobre regiões finitas; NaN fica fora | vetor de −1/0/1, uma entrada por feature; só binário |
| `leaf_smoothing` | Probabilidade `(W·p + s·p₀)/(W + s)`, com p₀ = prevalência ponderada na raiz; não altera cortes | finito ≥ 0 |
| `ccp_alpha` | Poda custo-complexidade com risco Gini ponderado; árvore compacta | finito ≥ 0; > 0 só com `gini` |
| `n_jobs` | Threads Numba no motor hist; mesma árvore que `n_jobs=1` | inteiro ≥ 1 |
| `reuse_parent_histograms` | **Experimental**: filho maior por subtração do histograma do pai; mesma árvore | exige `splitter="hist"`, `max_leaf_nodes=None`, `objective="gini"` e pesos unitários |
| `random_state` | Permutação fixa das features para desempate entre features | inteiro ou `None` |

Ordem do pós-processamento: crescimento → poda (`ccp_alpha`) →
suavização → projeção monotônica.

## Desempates (determinísticos)

- Dentro de uma feature: menor bin/limiar; no mesmo limiar, NaN à direita.
- Entre features: a primeira na permutação de `random_state` com ganho
  estritamente maior.
- O hist não avalia prefixos finitos vazios ou completos **no nó**: eles
  repetiriam o corte só-NaN com outro limiar e enviariam valores finitos
  novos ao lado dos NaN.
- Sem NaN no treino do nó, NaN na previsão vai ao filho com mais linhas
  (empate: direita).

## Arquitetura (para quem mexe no código)

| Módulo | Papel |
|---|---|
| `splitters.py` | Registro dos objetivos: cada `SplitObjectiveSpec` diz quando dividir, qual busca usar, se o ganho é aceito e a prioridade. O builder não compara nomes |
| `builder.py` | Crescimento depth-first (pilha) ou best-first (heap) |
| `search.py` | Busca do melhor corte em um nó; fronteira Python→Numba |
| `kernels.py` | **Todos** os kernels Numba que chamam outros kernels |
| `bins.py` | Limites e transformação de bins (kernels autocontidos) |
| `postprocess.py` | Poda, probabilidades, projeção monotônica |
| `_data.py` | `NodeArrays`, `Split`, validação |
| `_reference.py` | Referências Python só para testes |
| `core.py` | Fachada de compatibilidade; atribuir `core.nome` repassa ao módulo de origem |

Regras de Numba:

1. `cache=True` invalida pelo arquivo que define a função e não rastreia
   kernels chamados de outro módulo. Kernel que chama kernel fica em
   `kernels.py`.
2. Tipos na fronteira: índices e ordens `int64`, `y` `int32`, pesos e massas
   `float64`, bins `uint8`, X `float32`. `ObjectiveParams` normaliza os
   escalares do objetivo.
3. Uma travessia Python→Numba por nó no hist e por feature no exact; nunca
   por candidato (testado por instrumentação em `test_arvore_rapida.py`).
4. Sem `fastmath`.

Para adicionar um objetivo: kernel em `kernels.py` → busca em `search.py` →
`SplitObjectiveSpec` em `splitters.py` → regra de admissibilidade e de
força bruta em `tests/tree_invariants.py`.
