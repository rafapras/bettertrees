# Changelog

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
