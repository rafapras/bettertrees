# Árvore rápida — primeira implementação funcional

**Status: treino funcional implementado; histogramas por linha validados; fachada
P0 e `feature_importances_` P1 da API sklearn fechadas.**
O pacote já constrói árvores pelos motores `hist` e `exact`, com previsão,
pesos, NaN explícito, limites de profundidade/folhas, bound opcional e
instrumentação básica em `fit_stats_`. A varredura de histogramas usa kernel
Numba; `max_bins=255` é respeitado também em profundidade 1.
O fit hist usa a transformação e a acumulação de bins por linha; as versões
por coluna permanecem como baseline reproduzível. No benchmark atual de 500 mil
× 100, três repetições por profundidade deram ratio candidato/sklearn de
0,595× em profundidade 1 e 0,227× em profundidade 5, sem mudança das métricas
de validação dentro do protocolo. Esses números de profundidade 1 são
históricos: foram medidos antes da correção do cap oculto de 32 bins.
A acumulação de histogramas por linha reduziu adicionalmente o fit mediano de
8,742 para 6,149 s no sintético binário e de 8,363 para 6,237 s no sintético
de sete classes, contra a referência por feature preservada, com cinco pares
comparáveis e saída idêntica. O teste isolado do primeiro fit e da memória
está registrado na revisão; ainda falta validação em Covertype real.
O loteamento da varredura e a massa de classes em Numba reduziram o fit de
6,489 para 5,102 s (21,37%) contra o baseline reconstruído, em cinco pares
`off`, com árvore e probabilidades idênticas. O diagnóstico detalhado está em
[`HOT_PATH_OPORTUNIDADES_2026-09-20.md`](../resultados/arvore_rapida/HOT_PATH_OPORTUNIDADES_2026-09-20.md).

Leia o [parecer e resultados locais](../VIABILIDADE_ARVORE_CLASSIFICACAO_RAPIDA.md)
e o [plano original](../PLANO_ARVORE_CLASSIFICACAO_RAPIDA.md).

Uma repetição posterior com 120 mil × 50 e 255 bins está em
`resultados/arvore_rapida/sprint_depth1_fixed255_2026-09-20.json`.

## Arquivos e estado

| Arquivo | Responsabilidade | Estado |
|---|---|---|
| `estimator.py` | API fit/predict/proba, parâmetros, clone | Implementado; `fit_stats_` registra etapas |
| `CONTRATO.md` | Garantias por parâmetro, combinações aceitas, desempates e regras de Numba | Referência para quem usa e para quem mexe |
| `splitters.py` | Registro dos objetivos com comportamento (quando dividir, busca, aceite, prioridade) | `gini`; `precision` experimental, exige `search_stopping="off"` |
| `builder.py` | Crescimento depth-first (pilha) e best-first (heap) | Não compara nomes de objetivo |
| `search.py` | Busca do melhor corte por nó (fronteira Python→Numba) | hist em lote por nó; exact por feature |
| `kernels.py` / `bins.py` | Kernels Numba (num módulo só, por causa do cache) e bins | — |
| `postprocess.py` | Poda, probabilidades e projeção monotônica | — |
| `_data.py` / `_reference.py` | Contêineres e validação / referências Python só para testes | — |
| `core.py` | Fachada de compatibilidade para scripts antigos | Atribuir `core.nome` repassa ao módulo de origem |
| `benchmark.py` | Referência sklearn, tempo de bins e função do portão | Executável para baseline e nó único |
| `../benchmark_arvore_depth5.py` | Comparação controlada em profundidade 5 | Cinco repetições em 500 mil linhas por 100 colunas |
| `../benchmark_arvore_depth1_5_current.py` | Matriz atual de profundidades 1–5 | Três repetições em 500 mil linhas por 100 colunas |
| `../benchmark_transform_bins_depth5.py` | Ablação da transformação, `off` e `bound` | Dois ensaios de cinco repetições |
| `../benchmark_bin_edges_depth5.py` | Ablação de `np.quantile` versus `np.partition` | Sem ganho; variante não integrada |
| `../benchmark_row_hist_depth5.py` | Ablação do histograma por linha versus feature | Ganho validado em binário e sete classes |
| `../benchmark_parallel_tree.py` | Fit serial versus `n_jobs` paralelo em depth 5/10 | Matriz grande com equivalência da árvore e probabilidades |
| `../benchmark_real_lightgbm.py` | Covertype real versus LightGBM | Split fixo, uma árvore, depth 5/10 |
| `../benchmark_cold_memory_depth5.py` | Primeiro fit e memória em processos isolados | Diagnóstico, uma medição por variante |
| `../benchmark_binary_edges_depth5.py` | Atalho experimental para colunas 0/1 | Rejeitado para fit completo; baseline preservado |
| `../tests/test_arvore_rapida.py` e `../tests/test_row_hist_experiment.py` | Contratos, bounds, splitters e equivalência | Passando em 2026-09-26 |
| `../tests/tree_invariants.py` e `../tests/test_invariants.py` | Invariantes de correção (inclusive otimalidade local por força bruta) em toda a matriz de flags, mais equivalências metamórficas | 565 casos passando em 2026-09-26; 9/9 bugs injetados detectados |

O resultado controlado e o próximo experimento estão em
[`REVISAO_2026-09-20.md`](../resultados/arvore_rapida/REVISAO_2026-09-20.md).
Os resultados de paralelismo e Covertype estão em
[`PARALLEL_LIGHTGBM_2026-09-20.md`](../resultados/arvore_rapida/PARALLEL_LIGHTGBM_2026-09-20.md).
O roadmap da paridade da API está em
[`ROADMAP_API_SKLEARN_2026-09-20.md`](../resultados/arvore_rapida/ROADMAP_API_SKLEARN_2026-09-20.md).
O contrato da fase 2 de precisão está em
[`FASE_2_OBJETIVO_PRECISAO_2026-09-20.md`](../resultados/arvore_rapida/FASE_2_OBJETIVO_PRECISAO_2026-09-20.md).
O comparativo pareado do motor exato está em
[`EXACT_NUMBA_2026-09-22.md`](../resultados/arvore_rapida/EXACT_NUMBA_2026-09-22.md).

## Busca multinível experimental

`fit_multilevel_tree(X, y, sample_weight=None, max_depth=2 ou 4,
root_splitter="hist", child_splitter="exact", ...)` é uma função de treino
separada que reutiliza os splitters Gini e o formato de nós do pacote. Retorna
um modelo com `predict`, `predict_proba`, `apply`, `get_depth`, `get_n_leaves`
e `fit_stats_`. Tanto a raiz quanto os filhos de cada bloco podem usar
`"hist"` ou `"exact"`. Em depth 4, busca um bloco depth 2 na raiz e novos
blocos depth 2 nas folhas que o primeiro bloco produziu.

"Ótimo" significa menor Gini ponderado **dentro de cada bloco e do espaço
de candidatos escolhido**, com `min_samples_leaf`. Não é um ótimo global de
profundidade 4; blocos posteriores não alteram os cortes de seus ancestrais.
O limite `candidate_limit` lança erro se a busca completa de um bloco o
ultrapassar, sem truncamento silencioso. O modo suporta pesos, NaN e
`leaf_smoothing`, mas ainda não oferece `max_leaf_nodes`, `monotonic_cst`,
`objective="precision"`, poda ou compatibilidade `sklearn.clone`. Nenhum
parâmetro/default do `FastDecisionTreeClassifier` foi alterado.

O protocolo reproduzível, resultados preliminares e limitações estão em
[`MULTINIVEL_DEPTH2_2026-09-22.md`](../resultados/arvore_rapida/MULTINIVEL_DEPTH2_2026-09-22.md).
A suíte pareada com guloso hist/exact e PyConTree bruto/balanceado, com runs
SQLite e curvas precision–recall, está em
[`BENCHMARK_MULTINIVEL_PYCONTREE_2026-09-22.md`](../resultados/arvore_rapida/BENCHMARK_MULTINIVEL_PYCONTREE_2026-09-22.md).
O estresse pré-fixado com geradores recursivos, mais linhas, controles sem
interação/sem sinal e oito seeds, seguido de holdouts reais repetidos, está
em [`RESULTADO_ESTRESSE_MULTINIVEL_2026-09-22.md`](../resultados/arvore_rapida/RESULTADO_ESTRESSE_MULTINIVEL_2026-09-22.md).

## Fluxo previsto

```text
fit -> validar/codificar -> [aprender bins no treino, se hist]
    -> escolher builder -> acumular histograma/ordenar valores
    -> buscar corte [off | bound] -> particionar índices -> crescer
    -> publicar arrays de nós

predict_proba -> validar X -> percorrer mesmos arrays -> normalizar massas
```

O bound descarta candidatos durante a busca; `min_impurity_decrease` controla
se a árvore cresce. São decisões distintas. `gain_tolerance` mede tolerância
de ganho local da busca; não é tolerância de acurácia. O motor `exact` ordena
os valores internos `float32` e faz busca exaustiva: `bound` pode ser pedido,
mas `fit_stats_["search_stopping_effective"]` registra `off`, e
`gain_tolerance` não atua. Heurística probabilística permanece bloqueada.
Todos os núcleos mantêm NaN explícito; não usar fastmath.
`splitter` identifica o motor (`hist`/`exact`) e `objective` identifica o
critério de pontuação; a resolução ocorre uma vez por `fit`, fora do hot path.

O objetivo `precision` requer `positive_class`, `min_precision` e
`min_support`. O ganho do corte é o aumento da melhor precision filha sobre a
precision do pai, sempre exigindo suporte suficiente. O suporte é massa
ponderada, enquanto `min_samples_leaf` continua contando linhas ativas.
`min_impurity_decrease` é reutilizado como ganho mínimo de precision nesse
objetivo. `min_precision` é o alvo que encerra a busca naquele ramo; ele não
bloqueia os cortes intermediários que melhoram a métrica. O bound de Gini não
é reutilizado: `search_stopping="off"` é obrigatório.

`max_feature_repeats` limita ocorrências da mesma feature em cada caminho
raiz->folha. O valor `2` é o primeiro limite útil para regras intervalares;
`None` preserva o baseline. Quando o limite está ativo, o hist builder recebe
somente as features ainda disponíveis no caminho, reduzindo a construção de
histogramas de features esgotadas. Esse parâmetro não é um orçamento global:
os contadores são copiados separadamente para os dois filhos.

`monotonic_cst` aceita um vetor `-1/0/+1` somente em classificação binária.
Para valores finitos, a probabilidade da classe positiva é projetada sobre a
ordem global das regiões de folhas que se sobrepõem nas demais features. A
frequência empírica permanece armazenada em `nodes_.class_weight`, mas
`predict_proba` usa a probabilidade projetada quando há restrição. NaN segue a
direção aprendida pelo nó e fica fora da comparação monotônica.

`leaf_smoothing` (padrão `0.0`) encolhe probabilidades das folhas em direção
à prevalência ponderada das classes na raiz: para massa de folha `W`,
probabilidade empírica `p`, prior `p0` e força `s`, devolve
`(W*p + s*p0)/(W+s)`. Não altera splits, nós ou folhas; valores positivos são
opt-in. Quando há `monotonic_cst`, a suavização precede a projeção global.
Selecionar `s` em validação separada; o diagnóstico de loss está em
`../resultados/arvore_rapida/AUDITORIA_LOSS_FOLHAS_2026-09-22.md`.

`ccp_alpha` (padrão `0.0`) faz pós-poda opt-in por custo-complexidade:
minimiza Gini ponderado total das folhas mais `ccp_alpha` por folha da
subárvore. A programação dinâmica é linear nos nós construídos; preserva
as massas, pesos implícitos e direções de NaN dos cortes mantidos. Só vale
para `objective="gini"`. Selecionar `ccp_alpha` em validação separada:
os primeiros testes reduziram folhas, mas não mostraram ganho estável de
log-loss de teste além da suavização. Ver
`../resultados/arvore_rapida/CRITERIOS_PODA_WEIGHTS_MISSING_2026-09-22.md`.

## Executar na raiz do projeto

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
.\.venv-tree\Scripts\python.exe -m pytest tests/test_package_smoke.py -q -p no:cacheprovider
.\.venv-tree\Scripts\python.exe -m pytest tests/test_arvore_rapida.py tests/test_row_hist_experiment.py -q -p no:cacheprovider
.\.venv-tree\Scripts\python.exe -m arvore_rapida.benchmark --samples 200000 --classes 2 --repeats 5 --output resultados/arvore_rapida/baseline_synthetic_binary.json
```

O smoke da API pública cobre fit/previsão dos motores `hist` e `exact`, pesos,
NaN, suavização, poda, `clone` e objetivo `precision`. É o gate rápido antes
de rodar a suíte inteira; não substitui uma instalação externa. Este
repositório ainda não tem metadados de distribuição e o módulo deve ser
executado a partir da raiz do projeto.

Para medir o primeiro experimento do plano em um único nó:

```powershell
.\.venv-tree\Scripts\python.exe -m arvore_rapida.benchmark --samples 20000 --classes 2 --repeats 5 --one-node --node-samples 8192 --output resultados/arvore_rapida/one_node.json
```

O bloco `one_node` separa `histogram_seconds`, `search_seconds` e o pipeline
completo para `search_stopping="off"` e `"bound"`, preservando o mesmo
histograma para a comparação isolada da busca.

Para etapas futuras, o benchmark aceita `--classes 7` no sintético ou
`--dataset covtype --samples 100000` e `--samples 581012`. Covertype pode
baixar dados. Use um `--output` diferente para preservar cada experimento.
O ambiente e suas versões estão no lockfile da raiz. Importação a partir da
raiz é suficiente; nenhum empacotamento/publicação foi introduzido.

Para a rodada de regras/PR/monotonicidade contra CART e uma árvore LightGBM:

```powershell
.\.venv-tree\Scripts\python.exe benchmark_regras_suite.py `
  --depths 2 4 6 8 --repeats 2 --covtype-samples 50000 `
  --db resultados/arvore_rapida/benchmark_runs_reais.sqlite `
  --output resultados/arvore_rapida/benchmark_regras_suite_reais.json
```

Por padrão, essa suíte usa seis datasets reais locais: breast cancer, wine,
digits 0/1, iris setosa/resto, diabetes acima da mediana e Covertype binário.
Os dois sintéticos continuam disponíveis explicitamente em `--datasets` para
stress tests, mas não são misturados à rodada real.

O SQLite guarda o UUID/status de cada rodada, uma linha por configuração,
tempo de fit completo, `fit_stats`, hash do split e os pontos das curvas PR.
Runs interrompidos ficam como `failed` e não são misturados aos completos.

Para verificar se o ganho do lookahead depende do tamanho da amostra,
`benchmark_multilevel_stress.py` cobre seis mecanismos sintéticos independentes
(recursivos, XOR, aditivo, sem sinal e classe rara), com linhas de treino/teste
separadas. `benchmark_multilevel_scale_real.py` faz pares na mesma partição de
teste de Covertype completo e Poker Hand público; este último contém mãos
geradas, não observações. Os resultados e limites de memória estão em
`../resultados/arvore_rapida/PROTOCOLO_ESCALA_MULTINIVEL_2026-09-22.md`.
Os números e a análise de orçamento de folhas estão em
`../resultados/arvore_rapida/RESULTADO_ESCALA_MULTINIVEL_2026-09-23.md`.
Ambas as suítes registram runs, tempos e curvas PR no SQLite. Os tamanhos de
treino podem ser aumentados por CLI somente após verificar memória disponível;
não há repetição artificial de linhas.

## Próxima ordem de trabalho (histórica)

A ordem vigente está em [`PLANO_ARVORE_RAPIDA_FASE_3.md`](../PLANO_ARVORE_RAPIDA_FASE_3.md);
a lista abaixo é o plano anterior, mantido como registro.

1. Fechar o portão de integração sklearn e rodar os checks selecionados de
   `check_estimator`, mantendo explícitas as limitações deliberadas.
2. Comparar em bases reais o limite por caminho `max_feature_repeats=1/2/None`
   com orçamento de folhas e condições por regra; registrar qualidade,
   suporte, repetição e tempo do fit.
3. Comparar a projeção monotônica conservadora com a árvore sem restrição e
   com o comparador sklearn sob orçamento de folhas igual.
4. Adicionar `min_samples_split`; depois avaliar `max_features` e `class_weight`
   separadamente, se houver caso de uso.
5. Retomar os benchmarks de Covertype, reuso de histogramas e binning somente
   em rodada separada, sem alterar o baseline por uma mediana isolada.

A seção **Gargalos prováveis e ordem de investigação** do parecer detalha as
hipóteses, os experimentos que podem refutá-las e a ordem prática de trabalho.

`NodeArrays` e `Split` apenas agrupam arrays/escalares; não há objeto por nó.
As funções internas presumem entradas validadas e árvores consistentes. Não
oferecem carregamento de árvores arbitrárias nem compatibilidade sklearn plena.
