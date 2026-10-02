"""Experimental pieces: no API stability guarantee.

- ``RatioVocabulary``: adds x_i / x_j columns whose pair has a "ratio shape".
- Distillation from a teacher (``crossfit_teacher``, ``fit_tree_on_target``, ...).
- Pair screening (``fast_pair_scores``, ``teacher_path_pairs``).
- ``RuleFitLasso``: L1 logistic regression over rules from an additive booster.
- ``PrecisionTreeClassifier``: a single tree whose cuts maximize one class's precision.
- ``ObliqueFIGSClassifier``: FIGS whose cuts may be oblique (RO-FIGS-like).
- ``fit_multilevel_tree``: a single tree grown from locally Gini-optimal depth-2 blocks.
"""

from ..multilevel import fit_multilevel_tree
from .distill import (
    MixedDepthTree,
    crossfit_teacher,
    fit_tree_on_target,
    restate_leaf_masses,
    soft_label_expand,
)
from .interactions import all_pairs, fast_pair_scores, teacher_path_pairs
from .itm import InteractingFIGSClassifier, InteractingTreeClassifier
from .oblique import ObliqueFIGSClassifier
from .precision import PrecisionTreeClassifier
from .ratios import RatioVocabulary, pair_shape_scores, top_pairs
from .rulefit import RuleFitLasso

__all__ = [
    "InteractingFIGSClassifier",
    "InteractingTreeClassifier",
    "MixedDepthTree",
    "ObliqueFIGSClassifier",
    "PrecisionTreeClassifier",
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
