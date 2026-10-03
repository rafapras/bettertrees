"""Execute a schema-1 bettertrees JSON model using only the Python standard library.

    python tools/predict_export.py model.json rows.json

Rows are numeric lists in training column order; null represents a missing value.
"""

import argparse
import json
import math
import struct
from bisect import bisect_left


def decision_function(model, rows):
    """Return raw logit scores, independently of bettertrees and NumPy."""
    if model.get("schema_version") != 1 or model.get("link") != "logit":
        raise ValueError("Expected a schema-1 bettertrees logit model.")
    dtype = model["input_dtype"]
    if dtype not in ("float32", "float64"):
        raise ValueError("input_dtype must be float32 or float64.")
    scores = []
    for row in rows:
        if len(row) != model["n_features"]:
            raise ValueError("Row width differs from the training data.")
        bins = []
        for j, value in enumerate(row):
            value = math.nan if value is None else float(value)
            fill = model.get("nan_fill")
            if math.isnan(value) and fill is not None:
                value = float(fill[j])
            if dtype == "float32":
                try:
                    value = struct.unpack("f", struct.pack("f", value))[0]
                except OverflowError as exc:
                    raise ValueError("Input outside the float32 range.") from exc
            if math.isinf(value):
                raise ValueError(f"Input contains infinity or is outside the {dtype} range.")
            bins.append(0 if math.isnan(value) else bisect_left(model["bin_edges"][j], value) + 1)
        margin = float(model["base_margin"])
        for tree in model["trees"]:
            node = 0
            for _ in tree["feature"]:
                left = tree["children_left"][node]
                if left == -1:
                    margin += tree["value"][node]
                    break
                feature, threshold = tree["feature"][node], tree["threshold_bin"][node]
                node = left if bins[feature] <= threshold else tree["children_right"][node]
            else:
                raise ValueError("Tree has no reachable leaf.")
        scores.append(margin)
    return scores


def predict_proba(model, rows):
    """Return probability columns in the order given by model['classes']."""
    probabilities = []
    for margin in decision_function(model, rows):
        p = 0.5 * (1.0 + math.tanh(0.5 * margin))
        probabilities.append([1.0 - p, p])
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("rows")
    args = parser.parse_args()
    with open(args.model, encoding="utf-8") as stream:
        model = json.load(stream)
    with open(args.rows, encoding="utf-8") as stream:
        rows = json.load(stream)
    print(json.dumps({"classes": model["classes"], "probabilities": predict_proba(model, rows)},
                     allow_nan=False))


if __name__ == "__main__":
    main()
