"""Interpretable sums of small trees fitted in logit space.

- ``SumOfOptimalTrees``: a few Newton-optimal trees (depth 1-3) with backfitting.
- ``InterleavedTreeClassifier`` (Interleaved Tree Model; ``FIGSClassifier`` is the same
  class): growth as in FIGS (Tan et al., 2022), full leaf re-fit as in RGF.
- ``AdditiveTreeBooster``: a long sum of optimal depth-1/2 trees with early stopping.
- ``BaggedFIGSClassifier`` / ``RashomonFIGSClassifier``: FIGS whose structure is chosen
  by a bootstrap vote (optionally distilled from the bag) or by a Rashomon search.
- ``BudgetClassifier``: the no-tuning choice for a cut budget (FIGS or compact booster).
"""

from .additive import AdditiveTreeBooster
from .budget import BudgetClassifier
from .compact import CompactTreeBooster
from .imported import LightGBMRefitClassifier, TreeSum, from_lightgbm
from .robust import BaggedFIGSClassifier, RashomonFIGSClassifier
from .screen import screen_features
from .smalltrees import (
    BoostedOptimalTrees,  # noqa: F401 - internal, importable for the benchmark
    FIGSClassifier,
    InterleavedTreeClassifier,
    SmallTree,
    SumOfOptimalTrees,
)

__all__ = [
    "AdditiveTreeBooster",
    "BaggedFIGSClassifier",
    "BudgetClassifier",
    "CompactTreeBooster",
    "FIGSClassifier",
    "InterleavedTreeClassifier",
    "LightGBMRefitClassifier",
    "RashomonFIGSClassifier",
    "SmallTree",
    "SumOfOptimalTrees",
    "TreeSum",
    "from_lightgbm",
    "screen_features",
]
