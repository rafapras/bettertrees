# Kept for the benchmark: this module moved to bettertrees.lab.itm. The old path is the
# same module object, so every name (private ones included) still imports from here.
import sys

from ..lab import itm as _moved

sys.modules[__name__] = _moved
