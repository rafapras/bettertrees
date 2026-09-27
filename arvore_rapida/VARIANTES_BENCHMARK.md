# Variantes de otimização da árvore

Esta matriz mantém o caminho padrão intacto. Cada variante precisa preservar o
melhor corte em `gain_tolerance=0` antes de ser considerada para o builder.

| Variante | Estado | Observação |
|---|---|---|
| Gini do pai pré-calculado | Implementada | Ganho isolado pequeno e inconsistente |
| Bound a cada 8 bins | Implementada | Promissora em alguns nós binários; depende de classe/tamanho |
| Bound a cada 16 bins | Implementada | Mais barato, mas pode avaliar mais candidatos |
| Buffers de histograma reutilizados | Implementada | Mais lento no primeiro teste por limpeza do scratch |
| Varredura escalar em kernel Numba | Implementada | Reduziu o custo de busca e mudou o gargalo para binning |
| Varredura de features em lote | Adotada | Uma chamada Numba por nó; 21,37% menos fit completo `off` em 500 mil × 100 |
| Massa de classes do nó em Numba | Adotada | Removeu loop Python por amostra; árvore/probabilidades idênticas |
| Bins adaptativos em profundidade 1 | Removida | Era um cap oculto de 32 bins; o contrato atual respeita `max_bins=255` |
| Transformação dos bins por linha | Adotada | 5 pares `off` e 5 pares `bound`, bins/nós/probabilidades iguais; fit mediano `bound` 10,194 → 8,402 s |
| Layout transposto dos bins | Pendente | Testar apenas se a nova transformação voltar a limitar o fit |
| Acumulação por linha | Adotada | 5 pares binários e 5 multiclasse em 500 mil × 100; nós/probabilidades idênticos e fit 29,7%/25,4% menor |
| Quantis diretos para colunas 0/1 | Não adotada | Etapa de limites mais rápida, mas replicação de 5 pares no fit completo foi 3/5; mediana 6,217 s baseline vs 6,308 s variante |
| Ordem de features promissoras | Próxima | Só pode afetar o momento do bound, não o corte exato |
| Pular bins vazios | Próxima | Especialmente relevante em nós pequenos/concentrados |
| Histograma do filho menor por subtração | Próxima | Requer benchmark no builder completo |
| Caminho especializado para nós pequenos | Próxima | Comparar histograma com busca exata direta |

O cap histórico de 32 bins em profundidade 1 foi removido para que o caminho
de produção respeite `max_bins=255`. A medição corrigida está registrada em
`resultados/arvore_rapida/sprint_depth1_fixed255_2026-09-20.json`; ela é uma
correção de contrato, não evidência de ganho de velocidade.

Em 2026-09-20, o bound padrão a cada bin, frente a `search_stopping="off"`,
preservou exatamente a árvore e as probabilidades no benchmark de 500 mil linhas
por 100 colunas, profundidade 5. Pulou 51.123 avaliações, mas reduziu a mediana
do fit completo apenas de 9,463 s para 9,319 s (1,55%); foi mais lento em dois
dos cinco pares. Isso não justifica promovê-lo como ganho temporal comprovado.
Ver [`REVISAO_2026-09-20.md`](../resultados/arvore_rapida/REVISAO_2026-09-20.md).

O benchmark deve registrar sempre busca `off`, busca `bound`, acumulação,
pipeline completo, número de candidatos avaliados/pulados e igualdade do corte.
Uma melhoria só entra no caminho padrão depois de repetir nos tamanhos 256,
2.048, 8.192 e 12 mil, com 2 e 7 classes, e de passar a comparação de
correção.

O ganho da transformação por linha foi medido no fit completo sob os dois
modos de busca, com o mesmo SHA-256 de dados e mesma qualidade. A função
NumPy por coluna continua em `core.py` como baseline reproduzível. Ver a
continuação da [`REVISAO_2026-09-20.md`](../resultados/arvore_rapida/REVISAO_2026-09-20.md).

A acumulação de histogramas por linha preserva a antiga
`_find_best_split_hist_feature_major` como referência. Os benchmarks pareados
em profundidade 5 estão em `row_hist_depth5_2026-09-20.json` e
`row_hist_multiclass_depth5_2026-09-20.json`; a árvore e as probabilidades
coincidiram em todas as repetições. A suíte passou com 43 testes, inclusive
NaN, pesos e sete classes. O custo de memória transitória do tensor por nó
aumenta; falta medir pico isolado e testar Covertype real.
