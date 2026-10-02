"""Fast decision trees and interpretable sums of trees for binary classification.

- ``BudgetClassifier(max_splits=b)``: the recommended start, no tuning: the sum of
  trees our benchmark validated for a budget of b cuts.
- ``FastDecisionTreeClassifier`` / ``FastDecisionTreeClassifierCV``: a single tree
  (exact and histogram engines), with leaf count and shrinkage chosen by CV.
- ``SumOfOptimalTrees``: a logit sum of a few Newton-optimal trees (depth 1-3).
- ``InterleavedTreeClassifier`` (the Interleaved Tree Model, ITM): a logit sum of small
  trees; each step adds the best cut of any tree (or a new root) by Newton gain and
  re-fits all leaves. Growth from FIGS (Tan et al., 2022), re-fit from RGF.
  ``FIGSClassifier`` is the same class under its older name.
- ``CompactTreeBooster``: shrunken boosting of optimal trees counted in distinct
  cuts (identical trees merged); the no-tuning choice above 64 cuts.
- ``AdditiveTreeBooster``: a long sum of optimal depth-1/2 trees with early stopping.
- ``LightGBMRefitClassifier`` / ``from_lightgbm``: a LightGBM model imported as an
  editable sum, with its leaves refitted jointly.

The sums (binary classification) expose ``rules()``, ``explain()``, ``to_dict()``,
``predict_contributions()``, ``plot_shapes()``, ``plot_contributions()`` and the
editing API; the single tree is multiclass and has ``export_text()``.
``bettertrees.experimental`` has no API stability guarantee; ``bettertrees.lab`` holds
research code with negative or inconclusive results (no API, no stability, not public).
"""

from .autotune import FastDecisionTreeClassifierCV
from .estimator import FastDecisionTreeClassifier
from .sums import (
    AdditiveTreeBooster,
    BudgetClassifier,
    CompactTreeBooster,
    FIGSClassifier,
    InterleavedTreeClassifier,
    LightGBMRefitClassifier,
    SumOfOptimalTrees,
    from_lightgbm,
)

__version__ = "0.1.0.dev0"

__all__ = ["AdditiveTreeBooster", "BudgetClassifier",
           "CompactTreeBooster", "FIGSClassifier", "FastDecisionTreeClassifier",
           "FastDecisionTreeClassifierCV", "InterleavedTreeClassifier", "LightGBMRefitClassifier",
           "SumOfOptimalTrees", "__version__", "from_lightgbm"]


# moved to bettertrees.lab; still importable from here, resolved on first use (kept for
# the benchmark)
_MOVED = {"BaggedFIGSClassifier": "robust", "RashomonFIGSClassifier": "robust",
          "fit_multilevel_tree": "multilevel"}


def __getattr__(name):
    if name in _MOVED:
        from importlib import import_module

        return getattr(import_module(f".lab.{_MOVED[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
