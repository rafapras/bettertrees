"""Fast decision trees and interpretable sums of trees for binary classification.

- ``FastDecisionTreeClassifier`` / ``FastDecisionTreeClassifierCV``: a single tree
  (exact and histogram engines), with leaf count and shrinkage chosen by CV.
- ``SumOfOptimalTrees``: a logit sum of a few Newton-optimal trees (depth 1-3).
- ``FIGSClassifier``: FIGS (Tan et al., 2022) in logit space.
- ``AdditiveTreeBooster``: a long sum of optimal depth-1/2 trees with early stopping.

The sums expose ``rules()``, ``explain()``, ``to_dict()``,
``predict_contributions()``, ``plot_shapes()`` and ``plot_contributions()``.
``bettertrees.experimental`` has no API stability guarantee.
"""

from .autotune import FastDecisionTreeClassifierCV
from .estimator import FastDecisionTreeClassifier
from .multilevel import fit_multilevel_tree
from .sums import AdditiveTreeBooster, FIGSClassifier, SumOfOptimalTrees

__version__ = "0.1.0.dev0"

__all__ = ["AdditiveTreeBooster", "FIGSClassifier", "FastDecisionTreeClassifier",
           "FastDecisionTreeClassifierCV", "SumOfOptimalTrees", "__version__",
           "fit_multilevel_tree"]
