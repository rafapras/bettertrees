"""Every import the benchmark (``bench/`` of the research repository) makes, by its old path.

Modules that moved to ``bettertrees.lab`` leave a bridge at the old path; none of these imports
may warn or fail. The list is what ``bench/*.py`` imports today.
"""

import importlib
import warnings

import pytest

# (module, names imported from it)
BENCH_IMPORTS = [
    ("bettertrees", ["AdditiveTreeBooster", "BaggedFIGSClassifier", "CompactTreeBooster",
                     "FIGSClassifier", "FastDecisionTreeClassifier", "FastDecisionTreeClassifierCV",
                     "LightGBMRefitClassifier", "RashomonFIGSClassifier", "SumOfOptimalTrees",
                     "fit_multilevel_tree"]),
    ("bettertrees.experimental", ["InteractingFIGSClassifier", "ObliqueFIGSClassifier"]),
    ("bettertrees.experimental._itm_kernels", ["best_cut_tree", "cut_tree_hist"]),
    ("bettertrees.experimental.distill", ["crossfit_teacher", "fit_tree_on_target"]),
    ("bettertrees.experimental.interactions", ["all_pairs"]),
    ("bettertrees.experimental.itm", ["CUT", "NODE", "TREE"]),
    ("bettertrees.experimental.ratios", ["RatioVocabulary"]),
    ("bettertrees.experimental.rulefit", ["RuleFitLasso"]),
    ("bettertrees.sums", ["BoostedOptimalTrees", "smalltrees"]),
    ("bettertrees.sums._common", ["base_margin", "bin_threshold", "grad_hess", "predict_input",
                                  "rebin", "teacher_top_features"]),
    ("bettertrees.sums._kernels", ["best_cut_1d", "best_depth2", "hist_1d", "node_hist_margin"]),
    ("bettertrees.sums.prune", ["prune_to_budget"]),
    ("bettertrees.sums.smalltrees", ["SmallTree", "_side", "greedy_tree", "grow_figs"]),
]


@pytest.mark.parametrize("module,names", BENCH_IMPORTS, ids=[m for m, _ in BENCH_IMPORTS])
def test_bench_imports_work_without_warning(module, names):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mod = importlib.import_module(module)
        for name in names:
            assert hasattr(mod, name), f"{module}.{name}"


def test_old_paths_are_the_lab_modules():
    pairs = [("bettertrees.experimental.distill", "bettertrees.lab.distill"),
             ("bettertrees.experimental.interactions", "bettertrees.lab.interactions"),
             ("bettertrees.experimental.ratios", "bettertrees.lab.ratios"),
             ("bettertrees.experimental.rulefit", "bettertrees.lab.rulefit"),
             ("bettertrees.experimental.itm", "bettertrees.lab.itm"),
             ("bettertrees.experimental._itm_kernels", "bettertrees.lab._itm_kernels"),
             ("bettertrees.sums.robust", "bettertrees.lab.robust"),
             ("bettertrees.multilevel", "bettertrees.lab.multilevel")]
    for old, new in pairs:
        assert importlib.import_module(old) is importlib.import_module(new)


def test_moved_names_are_the_same_objects_from_every_path():
    import bettertrees
    import bettertrees.experimental as ex
    import bettertrees.lab as lab
    import bettertrees.sums as sums
    from bettertrees.sums import smalltrees

    assert bettertrees.BaggedFIGSClassifier is lab.BaggedFIGSClassifier is sums.BaggedFIGSClassifier
    assert bettertrees.RashomonFIGSClassifier is lab.RashomonFIGSClassifier
    assert bettertrees.fit_multilevel_tree is lab.fit_multilevel_tree is ex.fit_multilevel_tree
    assert ex.InteractingFIGSClassifier is lab.InteractingTreeClassifier
    assert ex.RuleFitLasso is lab.RuleFitLasso and ex.RatioVocabulary is lab.RatioVocabulary
    assert sums.BoostedOptimalTrees is smalltrees.BoostedOptimalTrees


def test_public_surface_is_trimmed():
    import bettertrees
    import bettertrees.experimental as ex
    import bettertrees.sums as sums

    assert not {"BaggedFIGSClassifier", "RashomonFIGSClassifier", "fit_multilevel_tree"} & set(
        bettertrees.__all__)
    assert "BoostedOptimalTrees" not in sums.__all__
    assert sorted(ex.__all__) == ["ObliqueFIGSClassifier", "PrecisionTreeClassifier"]
    with pytest.raises(AttributeError):
        bettertrees.no_such_name
