# bettertrees

**Interleaved Tree Models: small, readable sums of trees for binary classification, under a
budget of cuts.**

[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
![Python 3.10 to 3.14](https://img.shields.io/badge/python-3.10%20to%203.14-blue.svg)
[![CI](https://github.com/rafapras/bettertrees/actions/workflows/ci.yml/badge.svg)](https://github.com/rafapras/bettertrees/actions/workflows/ci.yml)

![AUC against the number of cuts: the ITM above LightGBM and CART of the same size](https://raw.githubusercontent.com/rafapras/bettertrees/main/docs/images/frontier.png)

## Why

Where a decision must be explained line by line, as in credit scoring and in the biological
sciences, the binding constraint is the size of the model: a scorecard with a few dozen bins, a
tree with a few dozen nodes. bettertrees asks how much predictive power fits in each cut. Its main
model, the Interleaved Tree Model (ITM), is a logit sum of a few small trees that share one budget
of cuts. You set the budget; a fixed rule sets everything else, so there is nothing to
tune per dataset.

## Install

```bash
pip install bettertrees                 # numpy, numba, scikit-learn
pip install "bettertrees[all]"          # + LightGBM screening, matplotlib plots, SHAP, pandas
```

From source: `pip install "git+https://github.com/rafapras/bettertrees"`. Python 3.10 to 3.14.
The Numba kernels compile on first use (about 5 to 10 seconds, once per environment) and are
cached afterwards. The public API has [type hints](https://github.com/rafapras/bettertrees/blob/main/docs/typing.md)
(`py.typed`); pandas inputs stay at the array-like level.

## Quickstart

This example uses a pandas DataFrame; install the `all` extra above. For NumPy-only
use, omit `as_frame=True` when loading the data.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from bettertrees import BudgetClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)  # a DataFrame keeps the column names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

model = BudgetClassifier(max_splits=8).fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"{model.n_splits_} cuts in {len(model.trees_)} trees, test AUC {auc:.3f}")
```

```
8 cuts in 7 trees, test AUC 0.975
```

`BudgetClassifier(max_splits=b)` is the ITM with the rule of the benchmark: `lam = 2b`; up to 8
cuts a full Newton step capped at 4 logits, above 8 cuts a step of 0.3. The rule was evaluated
from 4 to 64 cuts. `InterleavedTreeClassifier` is the same model with every setting exposed.

## Reading the model

The whole model is the printout. To score a row, add the base and one value per tree, then apply
the logistic function.

```python
print(model.explain())
```

```
logit P(y = 1) = base +0.5242 + sum of 7 trees (8 cuts)
tree 1:
  +1.7784  if worst perimeter <= 108.5
  -2.3190  if worst perimeter > 108.5
tree 2:
  +1.5100  if worst concave points <= 0.1415
  -1.8678  if worst concave points > 0.1415
tree 3:
  +1.3691  if worst texture <= 23.07
  -1.2124  if worst texture > 23.07
tree 4:
  +0.8157  if mean concavity <= 0.09267
  -0.8322  if mean concavity > 0.09267
tree 5:
  +0.5433  if mean texture <= 20.15
  -0.6155  if mean texture > 20.15
tree 6:
  +0.7050  if worst radius <= 15.87
  -0.2938  if worst radius > 15.87 and worst texture <= 19.78
  -0.9831  if worst radius > 15.87 and worst texture > 19.78
tree 7:
  +0.4234  if worst symmetry <= 0.2852
  -0.4383  if worst symmetry > 0.2852
```

Six trees are single cuts, one additive effect each; tree 6 is an interaction, the effect of
texture among large radii. The same model as data, per row, and as SQL:

```python
model.rules()[:2]                         # [(tree, conditions, logit value)] per leaf
# [(0, ['worst perimeter <= 108.5'], 1.7783520345848123),
#  (0, ['worst perimeter > 108.5'], -2.3190229737929564)]

model.predict_contributions(X_test[:1])   # one column per tree; base + row sum = logit
# [[-2.31902297 -1.86779044 -1.21240149 -0.8322431  -0.61554356 -0.98310833 -0.43829661]]

print(model.to_sql(table="patients"))     # the model as one query, no Python needed to score
```

```sql
-- logit P(y = 1) = base + sum of 7 tree columns (8 cuts); p = 1 / (1 + EXP(-score))
WITH contributions AS (
  SELECT *,
    CASE  -- tree 1
      WHEN "worst perimeter" <= 108.5 THEN 1.77835
      ELSE -2.31902
    END AS "t1_worst perimeter",
    ...
    CASE  -- tree 6
      WHEN "worst radius" <= 15.870001 THEN 0.704956
      WHEN "worst radius" > 15.870001 AND "worst texture" <= 19.779999 THEN -0.293802
      ELSE -0.983108
    END AS "t6_worst radius_worst texture",
    ...
  FROM patients
), scored AS (
  SELECT *, 0.524249  -- base
    + "t1_worst perimeter"
    ...
    AS score
  FROM contributions
)
SELECT *, 1.0 / (1.0 + EXP(-score)) AS p
FROM scored;
```

Also on every sum: `to_dict()` (JSON), `get_trees()` (arrays, as scikit-learn's `tree_`),
`export_text()`, `plot_contributions(x)` (waterfall of one prediction), `plot_shapes()` (shape
functions of the single-feature trees) and `to_shap_model()` (exact TreeSHAP with
`shap.TreeExplainer`).

For deployment, `to_dict()` stores exact bins and tree arrays alongside the rounded
rules. The [export guide](https://github.com/rafapras/bettertrees/blob/main/docs/export.md) includes an independent evaluator and the
input precision and missing-value conventions. SQL printouts round leaf values to
`precision` significant digits (6 by default); validate the generated query against
the fitted model on deployment inputs.

## Results

From the benchmark of the paper (in preparation): 56 binary classification datasets from
TabArena, the AutoML Benchmark, the suite of Grinsztajn et al. and Kaggle, with 10 thousand to
2.2 million rows. Three evaluation partitions per dataset up to 100 thousand rows, one 80/20
holdout above; the partition used to choose the rule is excluded. Every rival is held to the same
budget in its own unit: CART is grown to b + 1 leaves, LightGBM gets K trees of l leaves with
K(l − 1) ≤ b, RGF is refitted to exactly b cuts.

Mean AUC difference in points (ITM minus rival; one point = 0.01 AUC), with the number of datasets
on which the ITM has the higher AUC. Positive = ITM better.

| rival (same budget unless noted) | 4 cuts | 16 cuts | 32 cuts | 64 cuts |
|---|---|---|---|---|
| CART (scikit-learn, tuned) | +1.89 (50/56) | +2.37 (52/56) | +2.30 (53/56) | +1.80 (25/28) |
| LightGBM, tuned with 16 trials | +0.51 (44/56) | +0.57 (51/56) | +0.32 (20/28) | +0.02 (16/28) |
| LightGBM, tuned with 100 trials | +0.64 (38/49) | +0.45 (43/49) | +0.59 (26/28) | — |
| RGF, tuned | −0.18 (28/56) | −0.03 (23/56) | +0.15 (31/56) | — |
| WoE scorecard (no size limit) | −3.95 (4/56) | +0.41 (37/56) | +1.43 (50/56) | +1.98 (24/28) |

Cells with 28 datasets cover only 10–100 thousand rows (64 cuts, and LightGBM with 16 trials at
32 cuts) or only above 100 thousand rows (LightGBM with 100 trials at 32 cuts); LightGBM with 100
trials was not run on the 7 datasets of 50–100 thousand rows. "—": not run.

In short: the ITM beats CART of the same size by about 2 AUC points at every budget, beats
LightGBM of the same size from 4 to 32 cuts and ties it at 64, and ties a tuned RGF, its closest
ancestor (with its library defaults, RGF is 2.5 to 3.6 points behind). Dataset by dataset at 16
cuts:

![ITM minus CART and minus LightGBM tuned with 100 trials, per dataset, at 16 cuts](https://raw.githubusercontent.com/rafapras/bettertrees/main/docs/images/scoreboard.png)

Against a weight-of-evidence scorecard (up to 10 bins per variable, no size limit; a median of
157 steps, one step per bin boundary), the ITM loses with 4 and 8 cuts and passes the scorecard
at 16:

![ITM minus a WoE scorecard by model size](https://raw.githubusercontent.com/rafapras/bettertrees/main/docs/images/scorecard.png)

**Cost.** The ITM is cheaper to obtain because it is not tuned. At 16 cuts on 10–50 thousand
rows it takes a median of 0.11 seconds against 5.26 seconds for LightGBM tuned with 100 trials
(median per-dataset ratio 47); above 100 thousand rows the ratio is 2.9 (7.7 against 22.3
seconds), because the tuned control searches on a 100 thousand-row sample.

## How it works

The ITM scores a row as `logit P(y = 1) = base + T_1(x) + ... + T_K(x)`, where each `T_k` is a
small tree with values in log-odds. Growth starts with no trees. At each step every leaf of every
tree, and the root of a new tree, is a candidate; the cut with the largest Newton gain
`G_l²/(H_l + λ) + G_r²/(H_r + λ) − G²/(H + λ)`, at the margin of the other trees, is added; its
two leaves take a Newton step `−G/(H + λ)`, and one sweep then refits every leaf of every tree.
Growth stops after b cuts (internal nodes, one per split operation); a (feature, threshold)
pair may repeat, so the model has at most b distinct pairs. The growth across trees comes from FIGS
(Tan et al.), the refit of all leaves from RGF (Johnson and Zhang); the ITM replaces FIGS's
mean-residual leaves with Newton steps in log-odds, refits after every cut, and replaces tuning with a
fixed rule. Features are binned into 32 quantile bins;
missing values get a bin of their own and always go left.

## Limitations

- **Binary classification only** for the sums (they raise on a multiclass target; wrap them in
  `OneVsRestClassifier`). The single trees are multiclass.
- **Evaluated from 4 to 64 cuts.** Above 64 the rule is applied without validation; at 64 the ITM
  only ties LightGBM.
- **Not faster per fit.** One ITM fit is slower than one LightGBM fit of the same size (16 cuts,
  50 features, one thread: 0.15, 1.38 and 15 seconds at 10 thousand, 100 thousand and one
  million rows, against 0.05, 0.60 and 2.62 for LightGBM). The time saved comes from not tuning.
- **Small budgets and many weak effects.** With 4 and 8 cuts a WoE scorecard of about 150 steps
  has the higher AUC; where many variables each carry a weak monotone effect, many cheap bins beat
  a few cuts.
- **Deep interactions.** Where the target is itself a deep interaction (in the benchmark,
  `electricity` and the hierarchical `pol`), a single tree or boosting is the right shape and the
  sum gains little over CART.
- **The rule** was chosen among about fifty variants on development partitions of the same pool
  of datasets, and the step cap at 4 and 8 cuts was added after the first evaluation results. It
  has not been validated on datasets outside the project.
- **Ordered numeric inputs.** Thresholds assume that the order of a feature's values means
  something; integer-coded categories with thousands of levels are better served by native
  categorical splits.

## What else is in the package

| estimator | what it is |
|---|---|
| `BudgetClassifier` | the ITM with the fixed rule per budget; start here |
| `InterleavedTreeClassifier` | the ITM with every setting exposed; `FIGSClassifier` is an alias |
| `SumOfOptimalTrees` | a logit sum of a few optimal (exhaustive search) trees of depth 1–3, with backfitting |
| `CompactTreeBooster` | shrunken boosting of optimal trees, identical trees merged, budget in distinct cuts |
| `AdditiveTreeBooster` | a long sum of optimal depth-1/2 trees with early stopping (no budget) |
| `LightGBMRefitClassifier`, `from_lightgbm` | a LightGBM model imported as an editable sum (exact predictions), with its leaves refitted jointly |
| `FastDecisionTreeClassifier`, `FastDecisionTreeClassifierCV` | a single CART-style tree (Gini), multiclass; the CV version chooses the leaf count and hierarchical shrinkage |

`bettertrees.sums` also exposes `TreeSum` (the editable sum returned by `from_lightgbm`),
`SmallTree` (one tree of a sum, in `trees_`) and `screen_features` (bootstrap screening of
single cuts on a residual).

All estimators follow the scikit-learn API (pipelines, `GridSearchCV`, `CalibratedClassifierCV`,
`permutation_importance`, `partial_dependence`, pickle), accept NaN and `sample_weight`, and keep
DataFrame column names.

`from_lightgbm` supports binary models with numerical splits, at most 254 distinct
thresholds per feature, and missing values routed left. It rejects categorical splits,
`zero_as_missing` and NaN routed right, whose semantics cannot be preserved.
Rows with zero sample weight are excluded from training and binning; positive weights
change the objective mass and do not generally mean repeating rows, especially with
row-based validation or LightGBM binning.

**Editing.** Every sum can be edited and re-estimated, and all outputs follow the edit:

```python
model.set_cut(tree, node, "income", 50_000)   # move or replace a cut (snaps to a bin edge)
model.split_leaf(tree, leaf, "debt", 2.5)     # add a cut by hand
model.add_stump("age", 65)                    # add a rule as a new tree
model.prune(tree, node)                       # collapse a cut into a leaf
model.refit_leaves(X, y, monotone={"income": -1})               # refit, with monotone constraints
model.cut_alternatives(X_val, y_val, tree, node, epsilon=0.01)  # near-equivalent cuts
```

**Experimental and lab.**

- `bettertrees.experimental` (no API stability guarantee): `ObliqueFIGSClassifier`, the ITM with
  cuts on linear combinations of two features, which helped at 4 and 8 cuts and added little at
  16 and 32; `PrecisionTreeClassifier`, a tree whose cuts maximize one class's precision.
- `bettertrees.lab` is research code from the benchmark with negative or inconclusive results:
  product terms between trees, bagged and Rashomon structure selection, distillation from a
  teacher, pair and ratio features, RuleFit, and a multilevel tree of optimal depth-2 blocks.
  **It is not part of the public API and carries no stability guarantee**; it is kept so that the
  results can be reproduced.

**Examples.** Three notebooks in [`examples/`](https://github.com/rafapras/bettertrees/tree/main/examples), run by CI on simulated credit data
with known effects: [`01_quickstart`](https://github.com/rafapras/bettertrees/blob/main/examples/01_quickstart.ipynb),
[`02_interpretation`](https://github.com/rafapras/bettertrees/blob/main/examples/02_interpretation.ipynb) and
[`03_editing`](https://github.com/rafapras/bettertrees/blob/main/examples/03_editing.ipynb).

## Credits

| work | what bettertrees borrows |
|---|---|
| **FIGS**: Tan, Singh, Nasseri, Agarwal, et al. "Fast Interpretable Greedy-Tree Sums." arXiv:2201.11931 (2022); [imodels](https://github.com/csinva/imodels) | Growth across trees: the next cut may deepen any leaf of any tree or start a new one. |
| **RGF**: Johnson, Zhang. "Learning Nonlinear Functions Using Regularized Greedy Forest." *IEEE TPAMI* (2014) | The refit of all leaf values as the forest grows. |
| **XGBoost**: Chen, Guestrin. *KDD* (2016) | The second-order gain `G²/(H+λ)` and leaf value `−G/(H+λ)`. |
| **LightGBM**: Ke et al. *NeurIPS* (2017); scikit-learn's HistGradientBoosting | Quantile histograms with a bin for missing values; best-first growth. |
| **Hierarchical Shrinkage**: Agarwal, Tan, Ronen, Singh, Yu. *ICML* (2022) | Leaf values shrunk toward their ancestors in the single tree. |
| **MurTree**: Demirović et al. *JMLR* (2022); **ConTree**: Briţa, van der Linden, Demirović. *AAAI* (2025) | The depth-two solver behind `SumOfOptimalTrees`, here on bins with the Newton gain. |
| **GA2M / EBM**: Lou, Caruana, Gehrke, Hooker. *KDD* (2013); Nori et al. InterpretML (2019) | Additive models with pairwise interactions and shape plots. |
| **CART**: Breiman, Friedman, Olshen, Stone (1984); Pagallo and Haussler, *Machine Learning* (1990) | The tree engine, cost-complexity pruning and the replication problem that motivates sums. |
| **SHAP / TreeSHAP**: Lundberg et al. *Nature Machine Intelligence* (2020) | `to_shap_model()` for exact TreeSHAP. |
| **scikit-learn** (Pedregosa et al., *JMLR* 2011) and **Numba** (Lam, Pitrou, Seibert, 2015) | The estimator API; all kernels. |

## Citation

A paper on the ITM and the benchmark is in preparation. Until it is out, please cite the software
([`CITATION.cff`](https://github.com/rafapras/bettertrees/blob/main/CITATION.cff)):

```bibtex
@software{bettertrees,
  title   = {bettertrees: Interleaved Tree Models for binary classification under a budget of cuts},
  author  = {rafapras},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/rafapras/bettertrees}
}
```

## License

BSD-3-Clause; see [`LICENSE`](https://github.com/rafapras/bettertrees/blob/main/LICENSE).
