"""Interpretable sums of small trees fitted in logit space.

- ``SumOfOptimalTrees``: a few Newton-optimal trees (depth 1-3) with backfitting.
- ``InterleavedTreeClassifier`` (Interleaved Tree Model; ``FIGSClassifier`` is the same
  class): growth as in FIGS (Tan et al., 2022), full leaf re-fit as in RGF.
- ``AdditiveTreeBooster``: a long sum of optimal depth-1/2 trees with early stopping.
- ``BudgetClassifier``: the no-tuning choice for a cut budget (FIGS or compact booster).
"""

from .additive import AdditiveTreeBooster
from .budget import BudgetClassifier
from .compact import CompactTreeBooster
from .imported import LightGBMRefitClassifier, TreeSum, from_lightgbm
from .screen import screen_features
from .smalltrees import (
    FIGSClassifier,
    InterleavedTreeClassifier,
    SmallTree,
    SumOfOptimalTrees,
)

__all__ = [
    "AdditiveTreeBooster",
    "BudgetClassifier",
    "CompactTreeBooster",
    "FIGSClassifier",
    "InterleavedTreeClassifier",
    "LightGBMRefitClassifier",
    "SmallTree",
    "SumOfOptimalTrees",
    "TreeSum",
    "from_lightgbm",
    "screen_features",
]


# kept importable for the benchmark, resolved on first use: BoostedOptimalTrees is internal
# (not public), the other two moved to bettertrees.lab.robust
_MOVED = {"BoostedOptimalTrees": "smalltrees", "BaggedFIGSClassifier": "..lab.robust",
          "RashomonFIGSClassifier": "..lab.robust"}


def __getattr__(name):
    if name in _MOVED:
        from importlib import import_module

        mod = _MOVED[name]
        return getattr(import_module(mod if mod.startswith(".") else "." + mod, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
