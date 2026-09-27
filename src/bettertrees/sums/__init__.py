"""Interpretable sums of small trees fitted in logit space.

- ``SumOfOptimalTrees``: a few Newton-optimal trees (depth 1-3) with backfitting.
- ``FIGSClassifier``: FIGS (Tan et al., 2022) with Newton leaves.
- ``AdditiveTreeBooster``: a long sum of optimal depth-1/2 trees with early stopping.
- ``BoostedOptimalTrees``: plain boosting of optimal trees (no backfitting).
"""

from .additive import AdditiveTreeBooster
from .screen import screen_features
from .smalltrees import BoostedOptimalTrees, FIGSClassifier, SmallTree, SumOfOptimalTrees

__all__ = [
    "AdditiveTreeBooster",
    "BoostedOptimalTrees",
    "FIGSClassifier",
    "SmallTree",
    "SumOfOptimalTrees",
    "screen_features",
]
