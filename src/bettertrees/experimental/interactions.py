# Kept for the benchmark: this module moved to bettertrees.lab.interactions. The old path is the
# same module object, so every name (private ones included) still imports from here.
import sys

from ..lab import interactions as _moved

sys.modules[__name__] = _moved
