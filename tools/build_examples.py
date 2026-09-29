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
into different branches. `FIGSClassifier` spends the **same number of cuts** on a
logit sum of a few small trees, and the result still reads as a scorecard.

The data are simulated credit defaults (`credit_data.py`): the true model is a sum of
one-feature effects plus one interaction, so we know what a good model should find."""),
    new_code_cell(SETUP),
    new_markdown_cell("## Same budget, two shapes\n\nBoth models get 16 cuts."),
    new_code_cell("""\
from sklearn.tree import DecisionTreeClassifier

from bettertrees import FIGSClassifier, SumOfOptimalTrees

models = {
    "CART, 16 cuts (17 leaves)": DecisionTreeClassifier(
        max_leaf_nodes=17, min_samples_leaf=50, random_state=0),
    "FIGS, 16 cuts": FIGSClassifier(max_splits=16),
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
figs = models["FIGS, 16 cuts"]
print(figs.explain())"""),
    new_markdown_cell("""\
Tree 1 found the interaction we planted: a high debt ratio matters much more for people
who already paid late. The other trees are mostly single effects (income, employment,
age), which a single tree would have had to repeat in several branches."""),
    new_code_cell("""\
row = X_test.iloc[[0]]
contrib = figs.predict_contributions(row)[0]
logit = figs.base_margin_ + contrib.sum()
print(row.T.rename(columns={row.index[0]: "value"}))
print(f"\\nbase {figs.base_margin_:+.3f} + trees {' '.join(f'{c:+.3f}' for c in contrib)}")
print(f"= logit {logit:+.3f} -> P(default) = {1 / (1 + np.exp(-logit)):.1%}")
print(f"predict_proba: {figs.predict_proba(row)[0, 1]:.1%}")"""),
    new_markdown_cell("""\
## Choosing the budget

The budget is the one knob. It works with the usual scikit-learn tools."""),
    new_code_cell("""\
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(FIGSClassifier(), {"max_splits": [4, 8, 16, 32]},
                      scoring="neg_log_loss", cv=3).fit(X_train, y_train)
pd.DataFrame(search.cv_results_)[["param_max_splits", "mean_test_score"]]"""),
]

NB["02_interpretation"] = [
    new_markdown_cell("""\
# Interpretation: rules, contributions and shapes

Every sum in `bettertrees` (`FIGSClassifier`, `SumOfOptimalTrees`, `CompactTreeBooster`,
...) has the same interpretation API. Here we fit one FIGS and check it against the
true effects of the simulated data."""),
    new_code_cell(SETUP),
    new_code_cell("""\
from bettertrees import FIGSClassifier

figs = FIGSClassifier(max_splits=16).fit(X_train, y_train)"""),
    new_markdown_cell("## Rules\n\nOne entry per leaf: tree, conditions, value added to "
                      "the logit."),
    new_code_cell("""\
rules = pd.DataFrame(figs.rules(), columns=["tree", "conditions", "logit"])
rules["conditions"] = rules["conditions"].str.join(" and ")
rules.round(3)"""),
    new_markdown_cell("""\
## One prediction, term by term

`plot_contributions` draws the waterfall from the base to the prediction."""),
    new_code_cell("""\
i = int(np.argmax(figs.predict_proba(X_test)[:, 1]))  # the riskiest test row
print(X_test.iloc[i])
figs.plot_contributions(X_test.iloc[i]);"""),
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
        pd_logit.append(figs.decision_function(Xv).mean())
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

spec = figs.to_dict(feature_names=list(X.columns))
print(json.dumps(spec, indent=1)[:800], "...")"""),
    new_code_cell("print(figs.export_text())"),
]

NB["03_editing"] = [
    new_markdown_cell("""\
# Editing a model and its Rashomon view

A small sum is easy to change by hand: round a threshold, add a business rule, force a
monotone effect. After an edit, `refit_leaves` re-estimates the leaf values for the new
structure, and every output (rules, contributions, JSON) follows."""),
    new_code_cell(SETUP),
    new_code_cell("""\
from bettertrees import FIGSClassifier


def score(model):
    p = model.predict_proba(X_test)[:, 1]
    return f"AUC {roc_auc_score(y_test, p):.4f}, log-loss {log_loss(y_test, p):.4f}"


figs = FIGSClassifier(max_splits=16).fit(X_train, y_train)
print("fitted:", score(figs))
print(figs.explain())"""),
    new_markdown_cell("""\
## Round a threshold

Say the business wants the first debt-ratio cut at a round 0.45. `set_cut` moves the cut,
snapping it to the nearest edge of the training bins (so the new threshold is close to,
not exactly, 0.45), and `refit_leaves` updates the values."""),
    new_code_cell("""\
import copy

tree0 = figs.get_trees()[0]
node = next(k for k, name in enumerate(tree0["feature_name"]) if name == "debt_ratio")
print("before:", tree0["threshold"][node])

edited = copy.deepcopy(figs)
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
model = FIGSClassifier(max_splits=16).fit(X_fit, y_fit)
pd.DataFrame(model.cut_alternatives(X_val, y_val, tree=0, node=0, epsilon=0.01))"""),
    new_markdown_cell("""\
## The Rashomon view

`RashomonFIGSClassifier` mutates cuts and keeps every structure within `epsilon` of the
best validation loss. `rashomon_importance` gives the range of each feature's importance
across those equally good models."""),
    new_code_cell("""\
from bettertrees import RashomonFIGSClassifier

rash = RashomonFIGSClassifier(max_splits=16, n_mutations=60, random_state=0).fit(X_train, y_train)
print(len(rash.rashomon_models()), "models within epsilon of the best validation loss")
imp = pd.DataFrame(rash.rashomon_importance(X_test), index=["min", "mean", "max"]).T
imp.sort_values("mean", ascending=False).round(3)"""),
    new_markdown_cell("""\
Late payments and the debt ratio matter in every equally good model (their minimum stays
high). The mortgage flag can drop to zero: some models that fit just as well do not use
it at all, so its importance in any one model should not be over-read.

(The Rashomon search holds out 20% of the training rows for validation, so its selected
model is fit on less data than the plain FIGS above; use it to see the set, not as a
better predictor.)"""),
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
