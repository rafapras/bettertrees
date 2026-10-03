# Public type hints

The wheel includes `py.typed` and high-level annotations for the public model
constructors, fitting, prediction, interpretation and editing methods. Editors
and type checkers can show parameter types and results; `fit` and editing methods
that return the model preserve its concrete type when chained.

Inputs use NumPy's `ArrayLike`: NumPy arrays, ordinary sequences and objects with
an array interface, including pandas DataFrames and Series. There are no pandas
stubs or annotations for individual columns. Shapes, column order, missing values
and supported dtypes are still checked at runtime. Label arrays allow arbitrary
class-label dtypes; probability and score arrays use float64.

Optional external models and plotting objects remain broadly typed. Internal
numerical kernels, `bettertrees.experimental` and `bettertrees.lab` are outside this typing scope. The
`BudgetClassifier` stub describes its delegated public methods for editors without
changing how the wrapper runs. Its delegated edits that return a model return
the underlying `InterleavedTreeClassifier`, as they do at runtime.

To check the representative downstream calls used in CI:

```bash
python -m pip install -e ".[typing]"
python -m mypy
```

This checks the public-use example in `tools/typing_smoke.py`, rather than imposing
strict typing on the numerical implementation.
