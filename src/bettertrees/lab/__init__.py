"""Research code from the benchmark: negative or inconclusive results.

No API or stability guarantee; not part of the public API. Everything here was built to
answer a question in the benchmark and kept so that the result can be reproduced; none of
it beat the public estimators at the same budget in a way that justified a place in the
package. Modules:

- ``distill``: fit trees to a teacher's probabilities (``crossfit_teacher``,
  ``fit_tree_on_target``, ``MixedDepthTree``, ...).
- ``interactions``, ``ratios``: pair screening and ratio features (``RatioVocabulary``).
- ``rulefit``: ``RuleFitLasso``, L1 logistic regression over rules of an additive booster.
- ``itm``: ``InteractingTreeClassifier``, the Interleaved Tree Model plus product terms.
- ``robust``: ``BaggedFIGSClassifier`` and ``RashomonFIGSClassifier`` (bagged and Rashomon
  structure selection).
- ``multilevel``: ``fit_multilevel_tree``, a single tree from locally optimal depth-2 blocks.
"""

from .distill import (
    MixedDepthTree,
    crossfit_teacher,
    fit_tree_on_target,
    restate_leaf_masses,
    soft_label_expand,
)
from .interactions import all_pairs, fast_pair_scores, teacher_path_pairs
from .itm import InteractingFIGSClassifier, InteractingTreeClassifier
from .multilevel import fit_multilevel_tree
from .ratios import RatioVocabulary, pair_shape_scores, top_pairs
from .robust import BaggedFIGSClassifier, RashomonFIGSClassifier
from .rulefit import RuleFitLasso

__all__ = [
    "BaggedFIGSClassifier",
    "InteractingFIGSClassifier",
    "InteractingTreeClassifier",
    "MixedDepthTree",
    "RashomonFIGSClassifier",
    "RatioVocabulary",
    "RuleFitLasso",
    "all_pairs",
    "crossfit_teacher",
    "fast_pair_scores",
    "fit_multilevel_tree",
    "fit_tree_on_target",
    "pair_shape_scores",
    "restate_leaf_masses",
    "soft_label_expand",
    "teacher_path_pairs",
    "top_pairs",
]
