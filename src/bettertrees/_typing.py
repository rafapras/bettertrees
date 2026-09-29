"""Small shared aliases for the public API; no pandas or plotting dependencies.

ArrayLike covers NumPy arrays, sequences and objects exposing __array__, including
pandas DataFrames/Series. Shapes and column semantics are validated at runtime.
"""

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike as ArrayLike
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[Any]
IndexArray: TypeAlias = NDArray[np.intp]
FeatureNames: TypeAlias = Sequence[str] | NDArray[Any]
Seed: TypeAlias = int | None
MonotoneConstraints: TypeAlias = Mapping[int | str, int]
SelfT = TypeVar("SelfT")
