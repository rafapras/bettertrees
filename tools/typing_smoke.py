"""Check representative public calls as a downstream typed project would use them."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bettertrees import (
    AdditiveTreeBooster,
    BaggedFIGSClassifier,
    BudgetClassifier,
    CompactTreeBooster,
    FastDecisionTreeClassifier,
    FastDecisionTreeClassifierCV,
    FIGSClassifier,
    LightGBMRefitClassifier,
    RashomonFIGSClassifier,
    SumOfOptimalTrees,
    from_lightgbm,
)
from bettertrees.sums import TreeSum, screen_features


def public_calls(X: NDArray[np.float64], y: NDArray[np.int64], frame: Any,
                 lightgbm_model: Any) -> None:
    """DataFrames and optional external models deliberately stay high level."""
    tree: FastDecisionTreeClassifier = FastDecisionTreeClassifier(max_depth=3).fit(X, y)
    probabilities: NDArray[np.float64] = tree.predict_proba(X)
    leaves: NDArray[np.intp] = tree.apply(X)
    cv: FastDecisionTreeClassifierCV = FastDecisionTreeClassifierCV(
        leaves_grid=[4, 8], shrinkage_grid=[1.0, 5.0], cv=2,
    ).fit(frame, y)
    text: str = cv.export_text(feature_names=["age", "income"])

    figs: FIGSClassifier = FIGSClassifier(max_splits=4).fit(X, y)
    edited: FIGSClassifier = figs.refit_leaves(X, y).enforce_monotone({0: 1})
    rules: list[tuple[int, list[str], float]] = edited.rules()
    exported: dict[str, Any] = figs.to_dict()
    contributions: NDArray[np.float64] = figs.predict_contributions(X)
    bag: BaggedFIGSClassifier = BaggedFIGSClassifier(max_splits=4).fit(X, y)
    rashomon: RashomonFIGSClassifier = RashomonFIGSClassifier(max_splits=4).fit(X, y)
    compact: CompactTreeBooster = CompactTreeBooster(max_splits=8, depth="auto").fit(X, y)
    additive: AdditiveTreeBooster = AdditiveTreeBooster(max_rounds=5).fit(X, y)
    optimal: SumOfOptimalTrees = SumOfOptimalTrees(n_trees=2).fit(X, y)
    refitted: LightGBMRefitClassifier = LightGBMRefitClassifier(max_splits=4).fit(X, y)
    imported: TreeSum = from_lightgbm(lightgbm_model, X, y)
    budget: BudgetClassifier = BudgetClassifier(max_splits=4).fit([[0.0], [1.0]], [0, 1])
    budget_model: FIGSClassifier | CompactTreeBooster = budget.refit_leaves(X, y)
    budget_text: str = budget.explain()
    budget_scores: NDArray[np.float64] = budget.decision_function(X)
    screening: dict[str, NDArray[np.float64]] = screen_features(X, y, top_k=2)

    # Keep every checked result live without actually fitting models in CI here.
    _ = (probabilities, leaves, text, rules, exported, contributions, bag, rashomon,
         compact, additive, optimal, refitted, imported, budget_model, budget_text,
         budget_scores, screening)
