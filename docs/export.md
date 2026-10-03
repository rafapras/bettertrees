# Executing an exported model

`model.to_dict()` produces schema 1, a JSON-ready model with exact numeric bins
and tree arrays. Save it with `json.dump(model.to_dict(), stream, allow_nan=False)`.
The `rules` strings are rounded for people to read; never parse them for prediction.

The independent evaluator [`tools/predict_export.py`](../tools/predict_export.py)
uses only the Python standard library, without bettertrees or NumPy:

```bash
python tools/predict_export.py model.json rows.json
```

`rows.json` is a list of numeric rows, in the column order used at training time.
Use JSON `null` for missing values. The result includes probability columns and
their corresponding `classes` labels. You can also import its `decision_function`
and `predict_proba` functions into an application.

The schema records:

- `schema_version: 1`, `link: "logit"`, `base_margin` and the two `classes`;
- `n_features`, optional `feature_names`, and `input_dtype`;
- full-precision `bin_edges` per feature, and optional `nan_fill` preprocessing;
- per-tree `feature`, `threshold_bin`, `children_left`, `children_right`, `value`
  arrays, plus the human-readable `rules`.

Convert inputs to `input_dtype` **before** locating their bins. Native sums use
float32; imported LightGBM models use float64. If `nan_fill` is present, replace
missing values first. Otherwise missing values have bin 0. A finite value has bin
`1 + bisect_left(bin_edges[feature], value)`, so equality goes to the lower bin.
Reject infinities and conversion overflow. At an internal node, choose the left
child when the feature's bin is at most `threshold_bin`; otherwise choose the
right child. A leaf has `children_left == -1`. Add its value to the base, once per
tree, then apply `p = (1 + tanh(margin / 2)) / 2`.

`get_trees()` provides real numeric thresholds and an `input_dtype` entry per tree.
Consumers must cast inputs accordingly. `to_shap_model()` passes this convention
to SHAP; use interventional mode with background data as its docstring describes.

The schema is versioned separately from the package. An evaluator should reject
unknown schema versions instead of guessing. Treat model exports as trusted
artifacts from your training process.
