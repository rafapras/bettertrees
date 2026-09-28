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

X, y = load_breast_cancer(return_X_y=True, as_frame=True)  # a DataFrame (needs pandas) gives feature names
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

The Numba kernels compile on first use (about 5-10 s once per environment) and are cached afterwards.

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
| FIGS vs a LightGBM restricted to ⌊b/3⌋ depth-2 trees, tuned (4 / 8 / 16 cuts) | −3.5% / −3.0% / −1.5% log-loss, wins 20/0, 20/0, 19/1; −0.7% at 32, tie at 64 |
| FIGS vs a LightGBM with at most b cuts whose tree shape is tuned too (4 / 8 / 16 / 32 cuts) | −1.3% / −0.8% / −0.8% / −0.5% log-loss (wins 20/0, 18/2, 14/6, 14/6; p ≤ 0.02), tie at 64 |
| FIGS vs rpart (R, cost-complexity pruning) and one LightGBM tree (4 / 8 / 16 cuts) | rpart: −3.9% / −5.5% / −4.5%; LightGBM tree: −2.4% / −3.2% / −3.5% (19–20 wins of 20) |
| Our FIGS (logit, backfitting) vs FIGS from imodels (4 / 8 / 16 cuts) | −0.5% / −2.5% / −4.4% log-loss (15–17 wins of 20) |
| Optimal vs greedy search inside the same sum (16 / 32 / 64 cuts) | −0.2% / −0.3% / −0.5% log-loss (p < 0.01) |
| Additive booster (free capacity) vs tuned LightGBM / XGBoost / CatBoost | median +1.2% / +0.7% / +1.1% log-loss, −0.2 AUC points, 15–50× faster including tuning |
| Single-tree engine vs scikit-learn (same depth) | 3.0× faster (hist) and 2.9× (exact) for n ≥ 10k; for n ≤ 3k use `splitter="exact"` (1.7×) |

Numbers are preliminary; the final nested-CV run and the paper will replace them.
Where it does not help: hierarchical targets (e.g. `pol`) and high-order
interactions (e.g. `electricity`), where one deep tree or boosting is the right
shape — at free capacity the 90th percentile of the gap to LightGBM is large (+46%)
even though the median is +1.2%.

## Credits and inspirations

bettertrees stands on these ideas; what we took from each:

| work | what bettertrees borrows |
|---|---|
| **FIGS** — Tan, Singh, Nasseri, Agarwal, et al. "Fast Interpretable Greedy-Tree Sums." arXiv:2201.11931 (2022); [imodels](https://github.com/csinva/imodels) | The sum-of-trees model that grows several trees at once, one cut at a time. `FIGSClassifier` re-implements it with Newton/logit leaves and backfitting. |
| **Hierarchical Shrinkage** — Agarwal, Tan, Ronen, Singh, Yu. *ICML* (2022) | Leaf values shrunk toward their ancestors; the leaf model of the single tree and of its CV. |
| **XGBoost** — Chen, Guestrin. *KDD* (2016) | The second-order (Newton) gain `G²/(H+λ)` and leaf value `-G/(H+λ)` used by every sum. |
| **LightGBM** — Ke et al. *NeurIPS* (2017); scikit-learn's HistGradientBoosting | Quantile histograms (≤ 255 bins, missing values in their own bin), the (optional) histogram subtraction trick and best-first, leaf-wise growth. |
| **MurTree** — Demirović et al. *JMLR* (2022); **ConTree** — Briţa, van der Linden, Demirović. *AAAI* (2025); [pycontree](https://github.com/ConSol-Lab/contree) | Specialized exhaustive search for depth-two trees, the building block of `SumOfOptimalTrees` (here on bins and with the log-loss Newton gain instead of misclassification). ConTree is also our optimal-tree baseline. |
| **Optimal or greedy?** — van der Linden, Vos, de Weerdt, Verwer, Demirović. *TMLR* (2024) | The finding that optimizing the target objective directly is what makes optimal trees win, and the size–error curve as the evaluation. |
| **GA2M / EBM** — Lou, Caruana, Gehrke, Hooker. *KDD* (2013); Nori et al. InterpretML (2019) | Additive models with pairwise interactions and shape-function plots (`AdditiveTreeBooster`, `plot_shapes`). |
| **CART** — Breiman, Friedman, Olshen, Stone (1984) | The tree engine, cost-complexity pruning and the replication problem (Pagallo & Haussler, *Machine Learning*, 1990) that motivates sums. |
| **SHAP / TreeSHAP** — Lundberg et al. *Nature Machine Intelligence* (2020) | `to_shap_model()` exports the sums for exact TreeSHAP. |
| **scikit-learn** — Pedregosa et al. *JMLR* (2011) | The estimator API and its compatibility checks. |
| **Numba** — Lam, Pitrou, Seibert (2015) | All kernels. |

Related and next: SPLIT (Babbar, McTavish, Rudin, Seltzer, *ICML* 2025) for lookahead
near-optimal trees and TreeFARMS (Xin et al., *NeurIPS* 2022) for Rashomon sets.

## License

BSD-3-Clause.
