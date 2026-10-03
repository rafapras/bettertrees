# Changelog

## 0.1.0 (unreleased)

First public release.

- `BudgetClassifier(max_splits=b)`: the Interleaved Tree Model with a fixed rule per cut
  budget, no tuning (evaluated from 4 to 64 cuts; applied unvalidated above).
- `InterleavedTreeClassifier`, the Interleaved Tree Model (ITM): a logit sum of small trees
  grown together under a budget of distinct cuts, with Newton leaves and a refit of every
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
