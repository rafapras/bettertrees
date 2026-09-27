"""Capacidade de uma árvore única: screening, interações, vocabulário de
razões, destilação de um professor e boosting aditivo de árvores ótimas.

Peças independentes do motor (só usam bins e ``FastDecisionTreeClassifier``).
Plano: ``PLANO_CAPACIDADE_ARVORE_UNICA.md``.
"""

from .additive import AdditiveTreeBooster
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
from .screen import screen_features
from .smalltrees import BoostedOptimalTrees, FIGSClassifier, SmallTree, SumOfOptimalTrees

__all__ = [
                      "AdditiveTreeBooster",
                      "BoostedOptimalTrees",
                      "FIGSClassifier",
                      "MixedDepthTree",
                      "RatioVocabulary",
                      "RuleFitLasso",
                      "SmallTree",
                      "SumOfOptimalTrees",
                      "all_pairs",
                      "crossfit_teacher",
                      "fast_pair_scores",
                      "fit_tree_on_target",
                      "pair_shape_scores",
                      "restate_leaf_masses",
                      "screen_features",
                      "soft_label_expand",
                      "teacher_path_pairs",
                      "top_pairs",
]
