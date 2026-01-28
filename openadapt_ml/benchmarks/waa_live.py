"""DEPRECATED: Import from openadapt_evals instead.

This module is kept for backward compatibility only.
All classes are now provided by openadapt_evals.adapters.waa_live.
"""

import warnings

warnings.warn(
    "openadapt_ml.benchmarks.waa_live is deprecated. "
    "Please import from openadapt_evals instead: "
    "from openadapt_evals import WAALiveAdapter, WAALiveConfig",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from openadapt_evals.adapters.waa import (
    WAALiveAdapter,
    WAALiveConfig,
)

__all__ = [
    "WAALiveAdapter",
    "WAALiveConfig",
]
