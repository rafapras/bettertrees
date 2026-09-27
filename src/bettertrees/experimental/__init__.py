"""Experimental pieces: no API stability guarantee.

- ``RatioVocabulary``: adds x_i / x_j columns whose pair has a "ratio shape".
- Distillation from a teacher (``crossfit_teacher``, ``fit_tree_on_target``, ...).
- Pair screening (``fast_pair_scores``, ``teacher_path_pairs``).
- ``RuleFitLasso``: L1 logistic regression over rules from an additive booster.
"""

from .distill import (
    MixedDepthTree,
    crossfit_teacher,
    fit_tree_on_target,
    restate_leaf_masses,
    soft_label_expand,
)
from .interactions import all_pairs, fast_pair_scores, teacher_path_pairs
from .ratios import RatioVocabulary, pair_shape_scores, top_pairs
from .rulefit import RuleFitLasso

__all__ = [
    "MixedDepthTree",
    "RatioVocabulary",
    "RuleFitLasso",
    "all_pairs",
    "crossfit_teacher",
    "fast_pair_scores",
    "fit_tree_on_target",
    "pair_shape_scores",
    "restate_leaf_masses",
    "soft_label_expand",
    "teacher_path_pairs",
    "top_pairs",
]
