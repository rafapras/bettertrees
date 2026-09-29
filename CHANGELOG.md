# Changelog

## Unreleased — 0.1.0.dev0

- Add high-level public API type hints, the `py.typed` marker and a downstream
  typing check in CI, without pandas-specific stubs or new runtime dependencies.

- Preserve predictions when merging duplicate trees with different node orders.
- Preserve float64 input comparisons in imported LightGBM models. Reject imports
  with more than 254 thresholds per feature or `zero_as_missing` splits instead
  of returning a silently different model.
- Respect small cut budgets in `LightGBMRefitClassifier`.
- Export schema-1 JSON with exact bins, tree arrays, input precision and missing
  preprocessing, alongside the display rules. Add an independent standard-library
  evaluator and match SHAP input precision to the fitted model.
- Exclude zero-weight rows from sum fitting and inner CV; reject weight-total and
  float32-conversion overflow. Validate CV grids and class support explicitly.
- Fix the experimental LightGBM teacher's validation arguments and make test data
  preserve missing values across supported NumPy/pandas versions.
- Document the sample-weight contract and adapt the sklearn 1.6 binary-label
  equivalence check without suppressing independent regression tests.
- Include test helpers, examples and tools in source distributions. Check wheel
  installation without optional dependencies and source test collection in CI.
- Complete checkout/install instructions and qualify the scope of optimal search
  and benchmark reproducibility.
