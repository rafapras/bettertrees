"""Extraction gate: bettertrees must reproduce the research package's predictions.

    python tools/equivalence.py save  tools/reference.npz  # research ``arvore_rapida`` on the path
    python tools/equivalence.py check tools/reference.npz  # installed ``bettertrees``

Compares ``predict_proba`` of every public estimator on synthetic data (with NaN)
and on sklearn's breast_cancer; tolerance 1e-12.
"""

import sys

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification


def datasets():
    X, y = make_classification(n_samples=4000, n_features=12, n_informative=6,
                               random_state=0)
    rng = np.random.default_rng(0)
    X[rng.random(X.shape) < 0.05] = np.nan
    yield "synthetic_nan", X, y
    Xb, yb = load_breast_cancer(return_X_y=True)
    yield "breast_cancer", Xb, yb


def estimators(module):
    import importlib
    ar = importlib.import_module(module)
    return {
        "tree": lambda: ar.FastDecisionTreeClassifier(max_depth=5, random_state=0),
        "tree_cv": lambda: ar.FastDecisionTreeClassifierCV(),
        "sum_d2": lambda: ar.SumOfOptimalTrees(n_trees=5, depth=2, extra_stumps=1,
                                               feature_screen="fast"),
        "sum_d2_greedy": lambda: ar.SumOfOptimalTrees(n_trees=5, depth=2, search="greedy",
                                                      feature_screen="fast"),
        "figs": lambda: ar.FIGSClassifier(max_splits=16),
        "additive": lambda: ar.AdditiveTreeBooster(),
    }


def predictions(module):
    out = {}
    for dname, X, y in datasets():
        tr, te = slice(0, len(y) * 3 // 4), slice(len(y) * 3 // 4, None)
        for ename, make in estimators(module).items():
            m = make().fit(X[tr], y[tr])
            out[f"{dname}/{ename}"] = m.predict_proba(X[te])[:, 1]
    return out


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    got = predictions("arvore_rapida" if mode == "save" else "bettertrees")
    if mode == "save":
        np.savez(path, **got)
        print(f"{len(got)} references saved to {path}")
    else:
        ref = np.load(path)
        bad = [k for k in ref.files if k not in got or np.max(np.abs(ref[k] - got[k])) > 1e-12]
        for k in ref.files:
            d = np.max(np.abs(ref[k] - got[k])) if k in got else np.inf
            print(f"{'OK ' if k not in bad else 'DIF'} {k}  max_abs_diff={d:.2e}")
        sys.exit(1 if bad else 0)
