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

**Start here:** `BudgetClassifier(max_splits=b)` picks, without tuning, the model our
benchmark validated for a budget of `b` cuts (below), and exposes all of its
interpretation and editing methods.

| regime | model | output |
|---|---|---|
| **any budget, no tuning** | **`BudgetClassifier`** | `InterleavedTreeClassifier` up to 64 cuts, `CompactTreeBooster` above |
| one tree | `FastDecisionTreeClassifierCV` | a single tree; leaf count and hierarchical shrinkage chosen by CV |
| **interpretable (4–16 cuts)** | `InterleavedTreeClassifier`, `SumOfOptimalTrees` | `logit(p) = base + Σ tree_k(x)`, a few shallow trees |
| medium capacity (32–64 cuts) | `InterleavedTreeClassifier(learning_rate=0.3)`, `SumOfOptimalTrees` | the same sum with more trees |
| medium capacity, counted in distinct cuts | `CompactTreeBooster` | shrunken boosting of optimal trees; identical trees merged, so re-used cuts are free |
| free capacity | `AdditiveTreeBooster` | a long sum of optimal depth-2 trees with early stopping |

- `InterleavedTreeClassifier(max_splits=b)` is the **Interleaved Tree Model (ITM)**: a
  logit sum of small trees that grow together. At each step the best cut by Newton
  gain, in any leaf of any tree or as a new root, enters the model and every leaf is
  re-fitted (one backfitting sweep). The growth comes from FIGS (Tan et al. 2022),
  the full leaf re-fit from RGF (Johnson and Zhang 2014); the budget is counted in
  distinct cuts. The constructor's defaults are a full step (`learning_rate=1`),
  `lam="auto"` (2b) and no cap; the validated rule is in `BudgetClassifier`.
  `FIGSClassifier` is the same class under its older name.
- `SumOfOptimalTrees(n_trees=K, depth=2, extra_stumps=r)` uses `3K + r` cuts:
  each depth-2 tree is the **optimal** Newton tree on the current residual
  (exhaustive search over bins), followed by backfitting. `search="greedy"`
  swaps the optimal search for a greedy one (the paired control in our benchmarks).

All estimators are scikit-learn compatible (`GridSearchCV`, `cross_val_score`,
pipelines), accept NaN (missing values get their own bin and always go left),
keep DataFrame column names (and reject a DataFrame whose columns are renamed or
reordered at predict time), and the sums store the effective hyperparameters in
`lam_` and `learning_rate_`. They work inside pipelines, `GridSearchCV`,
`CalibratedClassifierCV` (also on a fitted model through `FrozenEstimator`), stacking,
voting and bagging ensembles, `permutation_importance`, `partial_dependence` and pickle.

The sums are **binary** (they raise on a multiclass target); wrap them in
`OneVsRestClassifier` for more classes. The single tree is natively multiclass.

**The no-tuning rule** (`BudgetClassifier`) for a budget of `b` cuts:
`InterleavedTreeClassifier(max_splits=b, max_delta_step=4.0)` up to 8 cuts,
`InterleavedTreeClassifier(max_splits=b, learning_rate=0.3)` up to 64 (both with
`lam = 2b`), and
`CompactTreeBooster(max_splits=b)` above. `max_delta_step` caps each Newton step on a
leaf; without it, a full step on a nearly pure leaf of rare-class data can diverge.

## What is public, experimental and lab

- **Public** (`import bettertrees`): `BudgetClassifier`, `InterleavedTreeClassifier`
  (alias `FIGSClassifier`), `SumOfOptimalTrees`, `AdditiveTreeBooster`,
  `CompactTreeBooster`, `LightGBMRefitClassifier` / `from_lightgbm`,
  `FastDecisionTreeClassifier` and `FastDecisionTreeClassifierCV`.
- **`bettertrees.experimental`** (no API guarantee): `ObliqueFIGSClassifier`, the ITM with
  cuts on pairs of features, which gained with 8 cuts in the benchmark, and
  `PrecisionTreeClassifier`, a tree whose cuts maximize one class's precision.
- **`bettertrees.lab`** (research code from the benchmark: negative or inconclusive
  results; no API or stability guarantee; not part of the public API): distillation from a
  teacher, pair and ratio features, RuleFit, the ITM with product terms
  (`InteractingTreeClassifier`), bagged and Rashomon structure selection
  (`BaggedFIGSClassifier`, `RashomonFIGSClassifier`) and the multilevel tree
  (`fit_multilevel_tree`). Kept so the results can be reproduced. The old import paths
  (`bettertrees.experimental.distill`, `bettertrees.sums.robust`, `bettertrees.multilevel`,
  ...) still work for the benchmark.

## Examples

Three notebooks in [`examples/`](examples), run by CI on simulated credit data with
known effects (no download):

- [`01_quickstart`](examples/01_quickstart.ipynb): 16 cuts as a CART tree and as a sum
  of small trees, reading the model, scoring a row by hand, choosing the budget.
- [`02_interpretation`](examples/02_interpretation.ipynb): rules, the waterfall of one
  prediction, fitted effects against the true ones, JSON export.
- [`03_editing`](examples/03_editing.ipynb): rounding a threshold, adding a business
  rule, monotone constraints, near-equivalent cuts and the Rashomon view.

## Example

```python
from sklearn.datasets import load_breast_cancer
from bettertrees import InterleavedTreeClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)  # a DataFrame (needs pandas) gives feature names
model = InterleavedTreeClassifier(max_splits=6).fit(X, y)
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

## Editing a model and its Rashomon view

Every sum can be edited mechanically and re-estimated, and all outputs (rules,
contributions, SHAP export) follow the edit:

```python
model.prune(tree, node)                      # collapse a cut into a leaf
model.set_cut(tree, node, "income", 50_000)  # move / replace a cut (snaps to a bin edge)
model.split_leaf(tree, leaf, "debt", 2.5)   # add a cut by hand
model.add_stump("age", 65)                   # add a rule as a new tree
model.drop_tree(k); model.merge_duplicates()
model.set_leaf_value(tree, leaf, 0.0)        # business override
model.refit_leaves(X, y)                     # re-estimate the leaves for the new structure
model.refit_leaves(X, y, trees=[0, 3])       # partial refit: the other trees stay frozen
model.refit_leaves(X, y, monotone={"income": -1, "age": +1})  # monotone constraints
model.monotone_violations("income", increasing=False)         # exact check (empty = ok)
model.cut_alternatives(X_val, y_val, tree, node, epsilon=0.01)  # near-equivalent cuts
```

A LightGBM model becomes an editable sum with `from_lightgbm(lgbm, X, y)` (exact
predictions); `LightGBMRefitClassifier` also refits its leaves jointly, which beats the
same LightGBM at 4-64 cuts in our benchmark.

`RashomonFIGSClassifier` (in `bettertrees.lab`) runs a local search that mutates one cut at a time and keeps
every structure it visits within `epsilon` of the best validation loss. That is a
**sample** of the Rashomon set (the structures the search reached), not the whole set:
`rashomon_models()` returns them as estimators and `rashomon_importance(X)` the range of
each feature's importance across them. `BaggedFIGSClassifier` picks cuts by a bootstrap
vote.

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

The single tree (`FastDecisionTreeClassifier` and its CV version) has
`export_text()`, with the leaf probabilities `predict_proba` returns.

## Benchmark summary

> Numbers to be updated from the paper: the table below comes from earlier runs, before the
> Interleaved Tree Model was named, and will be replaced by the paper's results.

Binary classification, OpenML/TabArena suites plus four Kaggle sets; 28 bases with
10k–100k rows (splits not used in development) and 28 with more than 100k (one
holdout each). Paired by base, median change, Wilcoxon test; log-loss change is
relative, AUC change in points. "No tuning" is the rule above; the controls are tuned
by Optuna (learning rate, λ, leaves per tree, minimum leaf) under the same cut budget.

| claim | result |
|---|---|
| No tuning vs LightGBM with ≤ b cuts, shape tuned, 10k–50k rows (4 / 16 / 128 cuts) | −2.7% / −2.2% log-loss (20/21, 18/21 bases, p ≤ 2e-5), +0.5 / +0.4 AUC points; tie at 128 |
| The same, 50k–100k rows | −0.9% / −0.9% / −0.9% log-loss (7 bases; significant at 4 cuts) |
| The same, more than 100k rows | −1.0% / **−1.6%** / −0.5% log-loss (25/28, **28/28**, 24/28; p ≤ 0.002), +0.4 / **+0.6** / +0.3 AUC points |
| No tuning vs a well-configured CART with the same cuts (scikit-learn tuned, or rpart's probability tree) | +1.1 to +2.5 AUC points at 4–128 cuts in every size group (e.g. +2.2, 27/29 bases above 100k at 16 cuts; p < 0.05 everywhere) |
| Where that gap comes from (63 tasks, CART's own cuts refitted as a sum vs FIGS's cuts) | at 16 cuts: 54% from arranging the same cuts as a sum, 35% from choosing better cuts, 11% from interactions inside FIGS's trees (79 / 34 / −12% at 4 cuts) |
| Optimal vs greedy search inside the same sum of depth-2 trees, 16 cuts | −0.9% log-loss (18/21 bases), +0.2 AUC points; the greedy version only ties LightGBM |
| Our FIGS (logit leaves, backfitting) vs FIGS from imodels, 32–128 cuts | −1% to −5% log-loss (7/7 bases of 50k–100k) |

Where it does not help: at 128 cuts it ties LightGBM; hierarchical targets (`pol`) and
high-order interactions (`electricity`), where one deep tree or boosting is the right
shape; on some bases a well-tuned CART is as good (e.g. Medical-Appointment-No-Shows
at 16 cuts). With very few cuts, FIGS's greedy choice can spend all of them on one deep
tree when the positive class is rare and concentrated in a subgroup (BAF fraud, 4 cuts:
AUC 0.717 against 0.785 for four stumps); across 63 tasks FIGS and a sum of stumps
tie at 4 cuts and FIGS is better at 8. The default rpart (`method="class"`) prunes to zero cuts on rare-class data;
comparisons against it overstate any method's gain, so we do not report them.

Earlier development runs (20 datasets, not re-run on the final splits): the additive
booster at free capacity is within +1% median log-loss of tuned LightGBM / XGBoost /
CatBoost at 15–50× less time including tuning, and the single-tree engine is about
3× faster than scikit-learn for n ≥ 10k.

## Credits and inspirations

bettertrees stands on these ideas; what we took from each:

| work | what bettertrees borrows |
|---|---|
| **FIGS** — Tan, Singh, Nasseri, Agarwal, et al. "Fast Interpretable Greedy-Tree Sums." arXiv:2201.11931 (2022); [imodels](https://github.com/csinva/imodels) | The sum-of-trees model that grows several trees at once, one cut at a time. `InterleavedTreeClassifier` (the ITM) re-implements it with Newton/logit leaves and backfitting. |
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
