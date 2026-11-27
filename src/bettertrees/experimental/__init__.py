"""Experimental pieces: alternative model families and opt-in tools (no API guarantee).

- ``PrecisionTreeClassifier``: a single tree whose cuts maximize one class's precision.
- ``ObliqueFIGSClassifier``: the Interleaved Tree Model whose cuts may be oblique, on
  pairs of features (RO-FIGS-like); in the benchmark it helped at 4 and 8 cuts and added
  little at 16 and 32.
- ``HingeFIGSClassifier``: continuous 1-knot piecewise-linear hinges + axis trees
  (quantile-clamped, Cauchy-Schwarz L2, Newton-capped). Strong per-term on 10k-100k
  numeric data; under strict 2-cuts-per-hinge cost accounting (``hinge_cost=2``) or at
  ``n > 100k``, pure ``InterleavedTreeClassifier`` matches or leads.
- ``BudgetRegressor``, ``FIGSRegressor``, ``SumOfOptimalTreesRegressor``: squared-error
  and Poisson log-link tree sums.
- ``CausalFIGS``: orthogonal R-learner CATE sums of shallow trees.
- ``rashomon_mcr``, ``feature_importance_shares``: constructive Model Class Reliance.

The modules with negative or inconclusive results (distillation, pair and ratio
features, RuleFit, the product-term variant, multilevel trees, depth-2 lookahead,
stationary L2 backfitting, sparse L1/L2 sums, LightGBM-free compact boosting) are in
``bettertrees.lab``; their old names under ``bettertrees.experimental`` still import,
for compatibility and benchmark reproducibility.
"""

from .causal import CausalFIGS
from .hinge import HingeFIGSClassifier
from .mcr import feature_importance_shares, rashomon_mcr
from .oblique import ObliqueFIGSClassifier
from .precision import PrecisionTreeClassifier
from .regression import BudgetRegressor, FIGSRegressor, SumOfOptimalTreesRegressor

__all__ = [
    "BudgetRegressor",
    "CausalFIGS",
    "FIGSRegressor",
    "HingeFIGSClassifier",
    "ObliqueFIGSClassifier",
    "PrecisionTreeClassifier",
    "SumOfOptimalTreesRegressor",
    "feature_importance_shares",
    "rashomon_mcr",
]

# names that moved to bettertrees.lab or bettertrees.sums, resolved on first use
_MOVED = {
    "CategoricalFIGSClassifier": "..sums.categorical",
    "CategoricalInterleavedTreeClassifier": "..sums.categorical",
    "MixedDepthTree": "..lab.distill",
    "crossfit_teacher": "..lab.distill",
    "fit_tree_on_target": "..lab.distill",
    "restate_leaf_masses": "..lab.distill",
    "soft_label_expand": "..lab.distill",
    "all_pairs": "..lab.interactions",
    "fast_pair_scores": "..lab.interactions",
    "teacher_path_pairs": "..lab.interactions",
    "InteractingFIGSClassifier": "..lab.itm",
    "InteractingTreeClassifier": "..lab.itm",
    "RatioVocabulary": "..lab.ratios",
    "pair_shape_scores": "..lab.ratios",
    "top_pairs": "..lab.ratios",
    "RuleFitLasso": "..lab.rulefit",
    "fit_multilevel_tree": "..lab.multilevel",
    "LookaheadFIGSClassifier": "..lab.lookahead_figs",
    "CompactTreeBoosterV2": "..lab.compact_v2",
    "L2FIGSClassifier": "..lab.l2_figs",
    "refit_leaves_l2": "..lab.l2_figs",
    "SparseTreeSumClassifier": "..lab.sparse_sum",
    "ResidualRatioFIGSClassifier": "..lab.sparse_sum",
}


def __getattr__(name):
    if name in _MOVED:
        from importlib import import_module

        return getattr(import_module(_MOVED[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
