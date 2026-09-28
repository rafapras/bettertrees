# bettertrees

Fast decision trees and **interpretable sums of optimal trees** for binary
classification, with a scikit-learn API.

A single CART tree spends most of a small budget copying the same effect into
different branches (the replication problem). A logit sum of a few shallow,
Newton-optimal trees uses the same number of cuts for separate effects, and
stays a readable scorecard.

> Status: pre-release (0.1.0.dev0). The name is provisional.

## Install

```bash
pip install -e .                    # numpy, numba, scikit-learn
pip install -e ".[lightgbm,plot]"   # LightGBM feature screening (p > 128) and plots
```

## Which model

| regime | model | output |
|---|---|---|
| one tree | `FastDecisionTreeClassifierCV` | a single tree; leaf count and hierarchical shrinkage chosen by CV |
| **interpretable (4–16 cuts)** | `FIGSClassifier`, `SumOfOptimalTrees` | `logit(p) = base + Σ tree_k(x)`, a few shallow trees |
| medium capacity (32–64 cuts) | `SumOfOptimalTrees` | the same sum with more trees |
| free capacity | `AdditiveTreeBooster` | a long sum of optimal depth-2 trees with early stopping |

- `FIGSClassifier(max_splits=b)` grows several trees at once, one cut at a time
  (FIGS, Tan et al. 2022), with Newton leaves in logit space.
- `SumOfOptimalTrees(n_trees=K, depth=2, extra_stumps=r)` uses `3K + r` cuts:
  each depth-2 tree is the **optimal** Newton tree on the current residual
  (exhaustive search over bins), followed by backfitting. `search="greedy"`
  swaps the optimal search for a greedy one (the paired control in our benchmarks).

All estimators are scikit-learn compatible (`GridSearchCV`, `cross_val_score`,
pipelines), accept NaN (missing values get their own bin and always go left),
and store the effective hyperparameters in `lam_` and `learning_rate_`.

## Example

```python
from sklearn.datasets import load_breast_cancer
from bettertrees import FIGSClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
model = FIGSClassifier(max_splits=6).fit(X, y)
print(model.explain())
```

```
logit P(y = 1) = base +0.5211 + sum of 5 trees (6 cuts)
tree 1:
  +1.5910  if worst perimeter <= 104.8
  -0.7259  if worst perimeter > 104.8 and worst radius <= 16.32
  -2.4027  if worst perimeter > 104.8 and worst radius > 16.32
tree 2:
  +1.6198  if worst concave points <= 0.141
  -1.6094  if worst concave points > 0.141
tree 3:
  +1.6240  if worst texture <= 23.18
  -1.0371  if worst texture > 23.18
tree 4:
  +0.8500  if radius error <= 0.3858
  -0.8247  if radius error > 0.3858
tree 5:
  +0.4692  if worst smoothness <= 0.1375
  -0.5643  if worst smoothness > 0.1375
```

To score a row, add the base and one value per tree, then apply the sigmoid.

## Interpretation API

Every sum exposes the same output:

```python
model.rules()                   # [(tree, conditions, logit value)] per leaf
model.to_dict()                 # the same as JSON (deploy without the package)
model.predict_contributions(X)  # (n, n_trees); base + row sum = decision_function
model.plot_contributions(x)     # waterfall of one prediction
model.plot_shapes()             # shape functions of the single-feature trees
model.get_trees()               # each tree as arrays (like sklearn's tree_), real thresholds
print(model.export_text())      # the trees drawn as text
model.to_shap_model()           # exact TreeSHAP via shap.TreeExplainer
```

## Benchmark summary

Twenty binary datasets with 10k–100k rows (OpenML/TabArena suites), paired by
dataset, Wilcoxon test. Log-loss change is relative, AUC change in points.

| claim | result |
|---|---|
| FIGS vs a greedy tree with the same number of cuts (4 / 8 / 16) | −1.5% / −1.7% / −1.9% log-loss, +1.1 / +1.6 / +1.7 AUC points, wins 17–18 of 20 datasets (p ≤ 0.001) |
| Optimal vs greedy search inside the same sum (16 / 32 / 64 cuts) | −0.2% / −0.3% / −0.5% log-loss (p < 0.01) |
| Additive booster (free capacity) vs tuned LightGBM | +1.2% log-loss, −0.24 AUC points, in ~3 s vs ~155 s including tuning |

Numbers are preliminary; the final nested-CV run and the paper will replace them.
Where it does not help: hierarchical targets (e.g. `pol`) and high-order
interactions (e.g. `electricity`), where one deep tree or boosting is the right shape.

## Related work

- Tan, Singh, Nasseri, Agarwal, Yu. "Fast Interpretable Greedy-Tree Sums." *PNAS* (2025); arXiv:2201.11931.
- Agarwal, Tan, Ronen, Singh, Yu. "Hierarchical Shrinkage: Improving the Accuracy and Interpretability of Tree-Based Methods." *ICML* (2022).
- Briţa, van der Linden, Demirović. "Optimal Classification Trees for Continuous Feature Data Using Dynamic Programming with Branch-and-Bound." *AAAI* (2025).
- Pagallo, Haussler. "Boolean Feature Discovery in Empirical Learning." *Machine Learning* (1990).

## License

BSD-3-Clause.
