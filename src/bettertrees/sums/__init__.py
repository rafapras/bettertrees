"""Interpretable sums of small trees fitted in logit space.

- ``SumOfOptimalTrees``: a few Newton-optimal trees (depth 1-3) with backfitting.
- ``FIGSClassifier``: FIGS (Tan et al., 2022) with Newton leaves.
- ``AdditiveTreeBooster``: a long sum of optimal depth-1/2 trees with early stopping.
- ``BaggedFIGSClassifier`` / ``RashomonFIGSClassifier``: FIGS whose structure is chosen
  by a bootstrap vote (optionally distilled from the bag) or by a Rashomon search.
- ``BoostedOptimalTrees``: plain boosting of optimal trees (no backfitting).
"""

from .additive import AdditiveTreeBooster
from .compact import CompactTreeBooster
from .imported import LightGBMRefitClassifier, TreeSum, from_lightgbm
from .robust import BaggedFIGSClassifier, RashomonFIGSClassifier
from .screen import screen_features
from .smalltrees import BoostedOptimalTrees, FIGSClassifier, SmallTree, SumOfOptimalTrees

__all__ = [
    "AdditiveTreeBooster",
    "BaggedFIGSClassifier",
    "BoostedOptimalTrees",
    "CompactTreeBooster",
    "FIGSClassifier",
    "LightGBMRefitClassifier",
    "RashomonFIGSClassifier",
    "SmallTree",
    "SumOfOptimalTrees",
    "TreeSum",
    "from_lightgbm",
    "screen_features",
]
