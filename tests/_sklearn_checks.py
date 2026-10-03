"""Preserve label meaning in both arms of sklearn 1.6's weighted-data check."""

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.utils import estimator_checks


def run_estimator_check(estimator, check, monkeypatch):
    check_name = getattr(getattr(check, "func", check), "__name__", "")
    if sklearn_version.startswith("1.6.") and check_name in {
            "check_sample_weight_equivalence_on_dense_data",
            "check_sample_weight_equivalence_on_sparse_data"}:
        # sklearn maps binary-only targets using the first label, but shuffles
        # only one arm. Keep a common reference label and all original assertions.
        original = estimator_checks._enforce_estimator_tags_y

        def coherent_labels(est, y):
            order = np.argsort(y, kind="stable")
            mapped = original(est, y[order])
            return mapped[np.argsort(order)]

        with monkeypatch.context() as patch:
            patch.setattr(estimator_checks, "_enforce_estimator_tags_y", coherent_labels)
            check(estimator)
    else:
        check(estimator)
