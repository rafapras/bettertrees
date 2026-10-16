"""Build and execute the example notebooks in ``examples/``.

The notebooks are generated from the cells below (so they stay reviewable as code)
and executed in place, so GitHub shows the outputs. CI runs this script too: a
broken example fails the build.

    python tools/build_examples.py            # build + execute
    python tools/build_examples.py --no-run   # build only
"""

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

SETUP = """\
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_data import make_credit_data

X, y = make_credit_data(n=20_000, seed=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)
print(f"{len(X_train):,} training rows, {len(X_test):,} test rows, "
      f"{y.mean():.1%} defaults, {X['income'].isna().mean():.0%} incomes missing")"""

NB = {}

NB["01_quickstart"] = [
    new_markdown_cell("""\
# Quickstart: 16 cuts as a sum of small trees

A single decision tree with a small budget spends its cuts copying the same effect
into different branches. `InterleavedTreeClassifier` spends the **same number of cuts** on a
logit sum of a few small trees, and the result still reads as a scorecard.

The data are simulated credit defaults (`credit_data.py`): the true model is a sum of
one-feature effects plus one interaction, so we know what a good model should find."""),
    new_code_cell(SETUP),
    new_markdown_cell("## Same budget, two shapes\n\nBoth models get 16 cuts."),
    new_code_cell("""\
from sklearn.tree import DecisionTreeClassifier

from bettertrees import BudgetClassifier, InterleavedTreeClassifier, SumOfOptimalTrees

models = {
    "CART, 16 cuts (17 leaves)": DecisionTreeClassifier(
        max_leaf_nodes=17, min_samples_leaf=50, random_state=0),
    "BudgetClassifier, 16 cuts (the no-tuning rule)": BudgetClassifier(max_splits=16),
    "ITM, 16 cuts": InterleavedTreeClassifier(max_splits=16),
    "sum of optimal depth-2 trees, 16 cuts": SumOfOptimalTrees(
        n_trees=5, depth=2, extra_stumps=1),
}
rows = []
for name, model in models.items():
    p = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    rows.append(dict(model=name, auc=roc_auc_score(y_test, p), log_loss=log_loss(y_test, p)))
pd.DataFrame(rows).set_index("model").round(4)"""),
    new_markdown_cell("""\
## Reading the model

`explain()` prints the whole model. To score a row, start from the base, add one
value per tree (the leaf the row falls into) and apply the sigmoid."""),
    new_code_cell("""\
itm = models["ITM, 16 cuts"]
print(itm.explain())"""),
    new_markdown_cell("""\
Tree 1 found the interaction we planted: a high debt ratio matters much more for people
who already paid late. The other trees are mostly single effects (income, employment,
age), which a single tree would have had to repeat in several branches."""),
    new_code_cell("""\
row = X_test.iloc[[0]]
contrib = itm.predict_contributions(row)[0]
logit = itm.base_margin_ + contrib.sum()
print(row.T.rename(columns={row.index[0]: "value"}))
print(f"\\nbase {itm.base_margin_:+.3f} + trees {' '.join(f'{c:+.3f}' for c in contrib)}")
print(f"= logit {logit:+.3f} -> P(default) = {1 / (1 + np.exp(-logit)):.1%}")
print(f"predict_proba: {itm.predict_proba(row)[0, 1]:.1%}")"""),
    new_markdown_cell("""\
## Choosing the budget

The budget is the one knob. It works with the usual scikit-learn tools."""),
    new_code_cell("""\
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(InterleavedTreeClassifier(), {"max_splits": [4, 8, 16, 32]},
                      scoring="neg_log_loss", cv=3).fit(X_train, y_train)
pd.DataFrame(search.cv_results_)[["param_max_splits", "mean_test_score"]]"""),
]

NB["02_interpretation"] = [
    new_markdown_cell("""\
# Interpretation: rules, contributions and shapes

Every sum in `bettertrees` (`InterleavedTreeClassifier`, `SumOfOptimalTrees`, `CompactTreeBooster`,
...) has the same interpretation API. Here we fit one ITM and check it against the
true effects of the simulated data."""),
    new_code_cell(SETUP),
    new_code_cell("""\
from bettertrees import InterleavedTreeClassifier

itm = InterleavedTreeClassifier(max_splits=16).fit(X_train, y_train)"""),
    new_markdown_cell("## Rules\n\nOne entry per leaf: tree, conditions, value added to "
                      "the logit."),
    new_code_cell("""\
rules = pd.DataFrame(itm.rules(), columns=["tree", "conditions", "logit"])
rules["conditions"] = rules["conditions"].str.join(" and ")
rules.round(3)"""),
    new_markdown_cell("""\
## One prediction, term by term

`plot_contributions` draws the waterfall from the base to the prediction."""),
    new_code_cell("""\
i = int(np.argmax(itm.predict_proba(X_test)[:, 1]))  # the riskiest test row
print(X_test.iloc[i])
itm.plot_contributions(X_test.iloc[i]);"""),
    new_markdown_cell("""\
## Effects against the truth

Most trees mix two features, so the effect of one feature is spread over several trees.
Its partial dependence in logit space adds them all up: set the feature to a value for
every row, average the logit. Below, that curve next to the true effect of the
simulation (both centered, since only differences in the logit matter)."""),
    new_code_cell("""\
import matplotlib.pyplot as plt

from credit_data import true_effects

truth = true_effects()
sample = X_train.sample(2000, random_state=0)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, feature in zip(axes, ["age", "debt_ratio", "late_payments", "months_employed"]):
    grid = np.unique(np.quantile(X_train[feature].dropna(), np.linspace(0, 0.99, 60)))
    pd_logit = []
    for v in grid:
        Xv = sample.copy()
        Xv[feature] = v
        pd_logit.append(itm.decision_function(Xv).mean())
    pd_logit, true = np.array(pd_logit), truth[feature](grid)
    ax.plot(grid, true - true.mean(), color="0.6", lw=3, label="true")
    ax.step(grid, pd_logit - pd_logit.mean(), where="post", label="fitted (16 cuts)")
    ax.set_title(feature)
axes[0].legend()
plt.tight_layout()"""),
    new_markdown_cell("""\
With 16 cuts the model draws each effect as a few steps in the right places: the U of
age, the kink of the debt ratio at 0.4, the jump at one year of employment. More cuts
give finer steps."""),
    new_markdown_cell("""\
## Deploying without the package

`to_dict()` is plain JSON: thresholds, leaf values and the base. Scoring is a few lines
in any language."""),
    new_code_cell("""\
import json

spec = itm.to_dict(feature_names=list(X.columns))
print(json.dumps(spec, indent=1)[:800], "...")"""),
    new_code_cell("print(itm.export_text())"),
]

NB["03_editing"] = [
    new_markdown_cell("""\
# Editing a model and its Rashomon view

A small sum is easy to change by hand: round a threshold, add a business rule, force a
monotone effect. After an edit, `refit_leaves` re-estimates the leaf values for the new
structure, and every output (rules, contributions, JSON) follows."""),
    new_code_cell(SETUP),
    new_code_cell("""\
from bettertrees import InterleavedTreeClassifier


def score(model):
    p = model.predict_proba(X_test)[:, 1]
    return f"AUC {roc_auc_score(y_test, p):.4f}, log-loss {log_loss(y_test, p):.4f}"


itm = InterleavedTreeClassifier(max_splits=16).fit(X_train, y_train)
print("fitted:", score(itm))
print(itm.explain())"""),
    new_markdown_cell("""\
## Round a threshold

Say the business wants the first debt-ratio cut at a round 0.45. `set_cut` moves the cut,
snapping it to the nearest edge of the training bins (so the new threshold is close to,
not exactly, 0.45), and `refit_leaves` updates the values."""),
    new_code_cell("""\
import copy

tree0 = itm.get_trees()[0]
node = next(k for k, name in enumerate(tree0["feature_name"]) if name == "debt_ratio")
print("before:", tree0["threshold"][node])

edited = copy.deepcopy(itm)
edited.set_cut(0, node, "debt_ratio", 0.45)
edited.refit_leaves(X_train, y_train)
print("after: ", edited.get_trees()[0]["threshold"][node])
print("edited:", score(edited))"""),
    new_markdown_cell("""\
## Add a rule by hand

A policy says no one over 75 gets credit without review. `add_stump` adds it as a new
one-cut tree; the refit gives it a value learned from the data."""),
    new_code_cell("""\
edited.add_stump("age", 75)
edited.refit_leaves(X_train, y_train)
print(edited.rules()[-2:])
print("with the rule:", score(edited))"""),
    new_markdown_cell("""\
## Monotone constraints

Policy: more months employed must never raise the risk, and more late payments must never
lower it. `monotone_violations` checks the whole model exactly (every combination of
leaves across trees, not a sample of rows); `refit_leaves(..., monotone=...)` enforces
the constraint by re-estimating the leaves, keeping the cuts."""),
    new_code_cell("""\
policy = {"months_employed": -1, "late_payments": +1}
for feature, sign in policy.items():
    print(feature, "violations:", len(edited.monotone_violations(feature, increasing=sign > 0)))
edited.refit_leaves(X_train, y_train, monotone=policy)
print("after the constrained refit:")
for feature, sign in policy.items():
    print(feature, "violations:", len(edited.monotone_violations(feature, increasing=sign > 0)))
print(score(edited))"""),
    new_markdown_cell("""\
## Near-equivalent cuts

`cut_alternatives` lists other cuts that would do almost as well at a node (validation
log-loss within `epsilon`). If a cut has many equally good alternatives, do not read
too much into the particular feature it chose."""),
    new_code_cell("""\
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)
model = InterleavedTreeClassifier(max_splits=16).fit(X_fit, y_fit)
pd.DataFrame(model.cut_alternatives(X_val, y_val, tree=0, node=0, epsilon=0.01))"""),
    new_markdown_cell("""\
## The Rashomon view

`RashomonFIGSClassifier` mutates cuts and keeps every structure within `epsilon` of the
best validation loss. `rashomon_importance` gives the range of each feature's importance
across those equally good models. It is research code (`bettertrees.lab`: no API
guarantee)."""),
    new_code_cell("""\
from bettertrees.lab import RashomonFIGSClassifier

rash = RashomonFIGSClassifier(max_splits=16, n_mutations=60, random_state=0).fit(X_train, y_train)
print(len(rash.rashomon_models()), "models within epsilon of the best validation loss")
imp = pd.DataFrame(rash.rashomon_importance(X_test), index=["min", "mean", "max"]).T
imp.sort_values("mean", ascending=False).round(3)"""),
    new_markdown_cell("""\
Late payments and the debt ratio matter in every equally good model (their minimum stays
high). The mortgage flag can drop to zero: some models that fit just as well do not use
it at all, so its importance in any one model should not be over-read.

(The Rashomon search holds out 20% of the training rows for validation, so its selected
model is fit on less data than the plain ITM above; use it to see the set, not as a
better predictor.)"""),
]


NB["04_scorecard_views"] = [
    new_markdown_cell("""\
# Scorecard views: who is the best, the typical and the worst client?

A logit sum of trees is a scorecard: each tree splits the clients into segments, each
segment is worth some points, and the score is the sum. `as_tree_sum` reads the ITM,
LightGBM, XGBoost, CatBoost and scikit-learn tree models the same way, so the views below
apply to all of them:

- `segments(X)`: the data marked tree by tree (the rule each row falls in and its points);
- `profiles(X)`: the best, the modal and the worst client;
- `plot_profiles(X)` and `plot_trees(x)`.

Points follow the credit convention: 600 points at odds 1:1, every 20 points halve the
odds of default (`offset`, `pdo`; `scale="logit"` keeps the raw logit)."""),
    new_code_cell(SETUP),
    new_markdown_cell("""\
## Three models with 10 cuts

LightGBM and XGBoost send missing values to the side they learn; bettertrees always
sends them left, so for the two boosters the missing incomes are filled below the
minimum before training (the ITM takes them as they are)."""),
    new_code_cell("""\
import lightgbm as lgb
import xgboost as xgb

from bettertrees import BudgetClassifier, as_tree_sum

fill = X_train.min() - 1
X_train_f, X_test_f = X_train.fillna(fill), X_test.fillna(fill)
itm = BudgetClassifier(max_splits=10).fit(X_train, y_train)
lgbm = lgb.LGBMClassifier(n_estimators=5, num_leaves=3, learning_rate=0.5,
                          verbose=-1).fit(X_train_f, y_train)          # 5 trees x 2 cuts
xgbm = xgb.XGBClassifier(n_estimators=10, max_depth=1, learning_rate=0.5
                         ).fit(X_train_f, y_train)                      # 10 stumps
sums = {"ITM": (as_tree_sum(itm, X_train), X_test),
        "LightGBM": (as_tree_sum(lgbm, X_train_f), X_test_f),
        "XGBoost": (as_tree_sum(xgbm, X_train_f), X_test_f)}
pd.DataFrame({name: {"trees": len(s.trees_), "cuts": s.n_splits_,
                     "test AUC": roc_auc_score(y_test, s.predict_proba(Xt)[:, 1])}
              for name, (s, Xt) in sums.items()}).T.round(4)"""),
    new_markdown_cell("""\
## The data marked tree by tree

One column per tree with the rule of the leaf, one with its points; `base` plus the
points is the score, exactly."""),
    new_code_cell("""\
seg = itm.segments(X_test)
seg.iloc[:3].T"""),
    new_markdown_cell(
        "Each tree is a segmentation, so it can be checked against the observed default rate:"),
    new_code_cell("""\
(seg.assign(default=y_test)
    .groupby(["tree_1", "tree_1_points"])["default"].agg(["size", "mean"])
    .round(3))"""),
    new_markdown_cell("""\
## Best, modal and worst client

`kind="observed"` (the default) takes real rows: the best and the worst score in the data.
The modal client is the most frequent combination of leaves."""),
    new_code_cell("""\
fig = itm.plot_profiles(X_test, bad_label="default", title="ITM, 10 cuts")"""),
    new_code_cell("""\
s, Xt = sums["LightGBM"]
fig = s.plot_profiles(Xt, bad_label="default", title="LightGBM, 10 cuts")"""),
    new_markdown_cell("""\
`kind="feasible"` searches the combinations of leaves that the cuts allow. The extreme
combination can be more extreme than any client in the data:"""),
    new_code_cell("""\
pd.concat({"observed": itm.profiles(X_test)[["score", "p"]],
           "feasible": itm.profiles(X_test, kind="feasible")[["score", "p"]]}, axis=1).round(3)"""),
    new_markdown_cell("""\
## Grown in parallel, not in sequence

Boosting finishes one tree before starting the next. The ITM chooses, at every cut,
any leaf of any tree or a new tree. `plot_growth` draws the order of the cuts: a
staircase for the booster, jumps between trees for the ITM (hollow marker = a new tree)."""),
    new_code_cell("""\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
itm.plot_growth(ax=axes[0])
sums["LightGBM"][0].plot_growth(ax=axes[1])
fig.tight_layout()"""),
    new_markdown_cell("""\
## The trees, with the path of the worst client

The orange numbers give each cut's place in the order of growth."""),
    new_code_cell("""\
worst = itm.profiles(X_test).loc["worst", "row"]
fig = itm.plot_trees(X_test.loc[[worst]], X=X_test, bad_label="default")"""),
]


def build(run):
    for name, cells in NB.items():
        nb = new_notebook(cells=cells, metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
        path = EXAMPLES / f"{name}.ipynb"
        if run:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=600, kernel_name="python3",
                           resources={"metadata": {"path": str(EXAMPLES)}}).execute()
        nbformat.write(nb, path)
        print("ok", path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    build(not ap.parse_args().no_run)
