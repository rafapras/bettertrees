# Validation of the public-package review fixes

29 September 2026. Fixes start from commit `407e8ab`; the separate working branch
does not include the pre-existing, uncommitted FIGS performance changes in the
researcher's original checkout.

Windows/Python 3.13.5, NumPy 2.1.0, Numba 0.61.0, scikit-learn 1.6.1,
LightGBM 4.6.0, pandas 2.2.3, SciPy 1.16.2 and SHAP 0.48.0:

- Full suite with Numba enabled: **1,691 passed, 2 expected failures, 5 expected
  failure cases passing**, no unexpected failures (148.30 seconds).
- All three example notebooks executed successfully. Their outputs were refreshed;
  existing notebook cell identities and execution metadata were preserved.
- Independent JSON execution agrees with native margins and probabilities for
  edited models, unseen missing values, full-precision and adjacent thresholds,
  imported float64 models and LightGBM missing preprocessing.
- Regressions cover duplicate node ordering, LightGBM's bin capacity boundary,
  rejected zero-as-missing splits, small cut budgets, inactive sample rows,
  zero-mass CV folds and numerical overflow.

The scikit-learn 1.6 sample-weight equivalence check recodes binary labels from
the first row, despite shuffling only one of its two compared samples. A local
test-only adaptation makes that mapping independent of row order while retaining
the original assertions. Later versions use the original check. LightGBM's row
counts for bins/minimum leaf size and row-based resampling/validation have
documented exceptions to weight-versus-repetition equivalence. Dedicated
regressions still check the removal of zero-weight observations.

The CI keeps one Linux/Python 3.12 job and checks lint, tests, executed notebooks,
isolated build, wheel installation without optional packages, and test collection
from the source distribution. Local clean-wheel validation also passed with
NumPy 2.5.3, Numba 0.67.0 and scikit-learn 1.9.1. Runtime warnings about Windows
core detection, feature-name conversion and third-party deprecations remain.

These checks do not validate the benchmark claims, all supported dependency
combinations, Python 3.10, or macOS. Large LightGBM imports remain explicitly
limited to 254 distinct thresholds per feature. No release tag or public
publication is part of these fixes.
