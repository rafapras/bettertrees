# Changelog

## 0.3.0 (2026-11-26)

- Faster `InterleavedTreeClassifier` (`FIGSClassifier`) via a fused L1-resident
  column-major Numba kernel (`best_leaf_cuts_T`) and single-pass logistic gradient
  computation in `node_hist_margin` (bit-for-bit identical predictions; fits 1.00M × 31
  rows in ~5.3 s and 1.66M × 11 rows in ~4.4 s on 4 cores at 16 cuts).
- Native categorical support via `CategoricalInterleavedTreeClassifier` (alias
  `CategoricalFIGSClassifier`), automatically dispatched by
  `BudgetClassifier(max_splits=b, categorical_features="auto")` on DataFrames with
  categorical or string columns (purely numeric inputs remain bit-for-bit identical):
  - Fisher-optimal Newton ratio ordering with rare-in-leaf pooling (`min_cat_weight`)
    and Rissanen MDL subset complexity regularization (`cat_dof_penalty`).
  - Separate `missing (NaN)` vs `other / unseen` category codes so unseen test-time
    categories route to the majority side of each split instead of inheriting missing risk.
  - Automatic `min_cat_coverage` guardrail so near-unique strings or ISO-8601 timestamps
    keep chronological order via leak-free `searchsorted` binning (and exact SQL string
    comparisons) instead of collapsing into `other`.
  - Optional `max_subset_size` cap for strictly bounded, readable categorical rules.
- `to_sql()` improvements: `null_safe=True` by default (unseen test-time NULLs route left
  like Python `NaN`), customizable `score_col` / `proba_col`, zero-tree edge-case support,
  and full categorical `IN (...)` / `NOT IN (...)` + timestamp SQL export.
- `FastDecisionTreeClassifier` and `FastDecisionTreeClassifierCV`: sklearn `allow_nan`
  tag, `predict_log_proba`, `get_depth`, `get_n_leaves`, and `feature_importances_` on the
  CV wrapper, plus English error messages in `_data.py`.
- `RashomonFIGSClassifier` (in `bettertrees.lab`): leaf-order-invariant structural
  deduplication via `_forest_key`.
- New experimental families in `bettertrees.experimental` (`HingeFIGSClassifier`,
  `CausalFIGS`, `BudgetRegressor`, `FIGSRegressor`, `SumOfOptimalTreesRegressor`,
  `rashomon_mcr`) and research prototypes in `bettertrees.lab` (`LookaheadFIGSClassifier`,
  `CompactTreeBoosterV2`, `L2FIGSClassifier`, `SparseTreeSumClassifier`,
  `ResidualRatioFIGSClassifier`).
- New example notebook `05_categorical_and_sql` and stacked 38-dataset benchmark report
  in `docs/BENCHMARKS.md`.

## 0.2.0 (2026-10-15)

- `as_tree_sum(model, X)`: one entry point that reads LightGBM, XGBoost, CatBoost and
  scikit-learn tree models (`HistGradientBoostingClassifier`, `GradientBoostingClassifier`,
  `DecisionTreeClassifier`) as a bettertrees sum with the same raw score (tested for
  equality), so the whole interpretation API applies to them. Cuts that send missing
  values right are refused when the feature has missing values in `X`; categorical
  splits are refused.
- Scorecard views on every sum: `segments(X)` (the data marked tree by tree: the rule of
  each leaf and its points), `profiles(X)` (best, modal, median and worst client, observed
  or the feasible extremes found by branch and bound, plus any clients chosen with
  `rows=`), `plot_profiles(X)`, `plot_trees()` and `plot_growth()` (the order of the cuts:
  the ITM records it while growing as `growth_`; boosters grow one tree after another).
  `plot_trees(x)` writes the client's value next to every cut on its path. Every plot
  starts from the base, adds one leaf per tree in the same units and says in a legend line
  what each mark means. Points in the credit convention (`offset`, `pdo`) or raw logit;
  `bad` chooses the class the score counts against and `bad_label` names it in the plots
  (e.g. "default").
- New example notebook `04_scorecard_views`.

## 0.1.0 (2026-10-08)

First public release. Tested on Python 3.10 and 3.14 (numpy 2.2 to 2.5, scikit-learn 1.7 to 1.9,
numba 0.68); declared for Python 3.10 to 3.14.

- `BudgetClassifier(max_splits=b)`: the Interleaved Tree Model with a fixed rule per cut
  budget, no tuning (evaluated from 4 to 64 cuts; applied unvalidated above).
- `InterleavedTreeClassifier`, the Interleaved Tree Model (ITM): a logit sum of small trees
  grown together under a budget of cuts (internal nodes), with Newton leaves and a refit of every
  leaf after each cut. `FIGSClassifier` is an alias.
- Other sums: `SumOfOptimalTrees` (optimal depth-1 to depth-3 trees), `CompactTreeBooster`,
  `AdditiveTreeBooster`, and `LightGBMRefitClassifier` / `from_lightgbm` (a LightGBM model
  as an editable sum).
- Single trees: `FastDecisionTreeClassifier` and `FastDecisionTreeClassifierCV` (leaf count
  and hierarchical shrinkage chosen by cross-validation).
- Reading a sum: `explain`, `rules`, `to_dict`, `to_sql`, `get_trees`, `export_text`,
  `predict_contributions`, `plot_contributions`, `plot_shapes`, `to_shap_model`.
- Editing a sum: `prune`, `set_cut`, `split_leaf`, `add_stump`, `drop_tree`,
  `merge_duplicates`, `set_leaf_value`, `refit_leaves` (partial refit, monotone
  constraints), `monotone_violations`, `cut_alternatives`.
- `bettertrees.experimental` (no API guarantee): `ObliqueFIGSClassifier`,
  `PrecisionTreeClassifier`.
- `bettertrees.lab`: research code with negative or inconclusive results, kept so the
  benchmark can be reproduced; not part of the public API.
- Duplicate-tree merging preserves leaf paths after manual edits. LightGBM imports
  retain float64 input precision and reject unsupported missing-value semantics
  and models exceeding the internal bin capacity.
- JSON exports include exact numeric bins and trees with a standalone evaluator;
  rounded rules remain available for reading.
- Zero-weight rows are excluded before binning and cross-validation; conversion
  overflow and invalid cut budgets are rejected explicitly.
- Public type hints for the high-level API, the `py.typed` marker, a `budget.pyi` stub for
  the delegating `BudgetClassifier`, and a downstream `mypy` check in CI; no new runtime
  dependencies.
