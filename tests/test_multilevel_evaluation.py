"""Protocol checks for large-source preparation and independent-unit summaries."""

import numpy as np
import pandas as pd

from benchmark_multilevel_evaluation import (prepare_home_credit,
                                             prepare_ieee, summarize)


def test_ieee_split_is_strictly_temporal_and_selection_uses_earlier_period():
    n = 100
    frame = pd.DataFrame({f"V{i}": np.full(n, i, dtype=float)
                          for i in range(1, 340)})
    frame["V1"] = np.arange(n, dtype=float)
    frame.loc[:50, "V2"] = np.nan
    frame["TransactionID"] = np.arange(500, 500 + n)
    frame["TransactionDT"] = np.arange(n)
    frame["TransactionAmt"] = 1.0
    frame["isFraud"] = np.arange(n) % 2
    pool, y_pool, test, y_test, recipe = prepare_ieee(frame, 4)
    assert len(pool) == len(y_pool) == 80
    assert len(test) == len(y_test) == 20
    assert recipe["boundary_transaction_dt"] == 80
    assert recipe["last_train_transaction_dt"] == 79
    assert recipe["selected_features"] == ["TransactionAmt", "V1", "V3", "V4"]
    assert "TransactionDT" not in recipe["selected_features"]


def test_home_credit_excludes_ids_and_target_from_numeric_features():
    n = 120
    frame = pd.DataFrame({f"f{i}": np.arange(n, dtype=float)
                          for i in range(100)})
    frame["SK_ID_CURR"] = np.arange(n) + 100000
    frame["TARGET"] = np.arange(n) % 5 == 0
    frame["category"] = "a"
    pool, y_pool, test, y_test, recipe = prepare_home_credit(frame)
    assert pool.shape[1] == test.shape[1] == 100
    assert len(y_pool) + len(y_test) == n
    assert "SK_ID_CURR" not in recipe["selected_features"]
    assert "TARGET" not in recipe["selected_features"]


def test_summary_counts_sources_not_training_seeds():
    rows = []
    for dataset, deltas in (("ieee_cis", [-.02, -.01, -.01]),
                            ("home_credit", [.01, .02, .03])):
        for delta in deltas:
            rows.append(dict(dataset=dataset, delta={"log_loss": delta},
                             fit_seconds={"block_hist_hist": 2,
                                          "greedy_hist": 1}))
    result = summarize(rows)
    assert result["completed_datasets"] == 2
    assert result["source_wins"] == 1
    assert result["source_losses"] == 1
    assert result["datasets"][0]["pairs"] == 3
