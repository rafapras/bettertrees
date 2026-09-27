"""The aggregate must pair only identical splits and never invent evidence."""

import pytest

from summarize_multilevel_scale import paired_summary


def record(engine, *, split="same", seed=0, loss=.5):
    return dict(family="recursive_four", original_dataset="covtype",
                n_train=100000, seed=seed, engine=engine, depth=4,
                split_sha256=split, fit_seconds_median=2 if engine ==
                "block_hist_hist" else 1,
                metrics=dict(log_loss=loss, accuracy=.7,
                             average_precision=.6, leaves=16))


def test_summary_requires_matching_split_and_complete_pair():
    assert paired_summary([record("greedy_exact")], synthetic=True) == []
    with pytest.raises(ValueError, match="incompatível"):
        paired_summary([record("greedy_exact"),
                        record("block_hist_hist", split="different")],
                       synthetic=True)
    group = paired_summary([record("greedy_exact"),
                            record("block_hist_hist", loss=.48)],
                           synthetic=False)[0]
    assert group["mean_log_loss_delta"] == pytest.approx(-.02)
    assert group["log_loss_wins"] == 1
    assert group["median_fit_ratio"] == 2
    assert group["ci95_log_loss_delta"] is None


def test_summary_can_select_posthoc_leaf_budget_control():
    leaf_control = record("greedy_exact")
    leaf_control["engine"] = "greedy_exact_leaf11"
    matched = paired_summary([leaf_control,
                              record("block_hist_hist", loss=.48)],
                             synthetic=False,
                             baseline="greedy_exact_leaf11")[0]
    assert matched["mean_log_loss_delta"] == pytest.approx(-.02)
