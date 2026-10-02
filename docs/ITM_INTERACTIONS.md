# Product terms between trees (lab): one-parameter interactions

`from bettertrees.lab import InteractingTreeClassifier`

Research code (no API or stability guarantee). In the benchmark the product terms tied the plain
sum at the same budget, so the model stays in `bettertrees.lab`.

    InteractingTreeClassifier(max_splits=16, learning_rate=.3, forms=("M1", "M2", "M3", "M4", "M5"),
                              order="interleaved").fit(X, y)

`max_splits` is the total budget: cuts + terms, one unit each. Each term is
`gamma * phi1 * phi2`, with `phi = (raw - center) / scale` fixed (h-weighted) when the term is born.

| form | factors | reads as |
|---|---|---|
| M1 | tree output x tree output | "tree 3 matters more where tree 1 is high" |
| M2 | node region of tree j x tree output k | "inside region R, tree 3 is scaled" |
| M3 | region of tree j x region of tree k | "rows in both regions get +gamma" |
| M4 | new cut [x_f > t] x tree output | "above t, tree 1 changes" |
| M5 | two cuts already in the trees | "if x1 > t1 and x2 > t2, +gamma" |

`order="interleaved"`: at every step cuts and terms compete on the Newton gain G^2/(H + lam) (a tie goes
to the cut); after each accepted operation every tree's leaves and all gammas are refitted.
`order="two_stage"`: b - b//4 additive cuts, then up to b//4 terms with the cuts frozen (GA2M-like).
`forms=()` or `max_interactions=0` delegates to FIGS bit for bit. `from_fitted` adds terms to a fitted
FIGS (cuts frozen); `from_structure` builds a model from given trees and terms (used for the oracle).

Exports: `predict_contributions` (K trees + E terms), `to_sql` (indicator columns plus one line per
term), `to_dict`, `explain`, `export_text`. SHAP and shape plots are blocked when terms exist.
Counts: `n_units_`, `n_splits_`, `n_distinct_splits_`, `conditions_read(X)`.

Tests: `tests/test_itm.py` (no terms = FIGS bit for bit; candidate scores = dense reference for every
form; finite differences for the tree multiplier and the coefficients; training margin; budget and
heredity; SQL for every factor kind with NULLs; a referenced leaf can be split without changing
predictions).
