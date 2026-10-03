"""Experimental pieces: no API stability guarantee.

- ``PrecisionTreeClassifier``: a single tree whose cuts maximize one class's precision.
- ``ObliqueFIGSClassifier``: the Interleaved Tree Model whose cuts may be oblique, on
  pairs of features (RO-FIGS-like); in the benchmark it helped at 4 and 8 cuts and added
  little at 16 and 32.

The modules with negative or inconclusive results (distillation, pair and ratio
features, RuleFit, the product-term variant, the multilevel tree) are in
``bettertrees.lab``; their old names under ``bettertrees.experimental`` still import,
for the benchmark.
"""

from .oblique import ObliqueFIGSClassifier
from .precision import PrecisionTreeClassifier

__all__ = ["ObliqueFIGSClassifier", "PrecisionTreeClassifier"]

# names that moved to bettertrees.lab, resolved on first use (kept for the benchmark)
_MOVED = {
    "MixedDepthTree": "distill", "crossfit_teacher": "distill", "fit_tree_on_target": "distill",
    "restate_leaf_masses": "distill", "soft_label_expand": "distill",
    "all_pairs": "interactions", "fast_pair_scores": "interactions",
    "teacher_path_pairs": "interactions",
    "InteractingFIGSClassifier": "itm", "InteractingTreeClassifier": "itm",
    "RatioVocabulary": "ratios", "pair_shape_scores": "ratios", "top_pairs": "ratios",
    "RuleFitLasso": "rulefit", "fit_multilevel_tree": "multilevel",
}


def __getattr__(name):
    if name in _MOVED:
        from importlib import import_module

        return getattr(import_module(f"..lab.{_MOVED[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
