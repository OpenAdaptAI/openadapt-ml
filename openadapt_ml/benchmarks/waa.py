"""DEPRECATED: Import from openadapt_evals instead.

This module is kept for backward compatibility only.
All classes are now provided by openadapt_evals.adapters.waa.
"""

import warnings

warnings.warn(
    "openadapt_ml.benchmarks.waa is deprecated. "
    "Please import from openadapt_evals instead: "
    "from openadapt_evals import WAAAdapter, WAAMockAdapter, WAAConfig",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from openadapt_evals.adapters.waa import (
    WAA_DOMAINS,
    WAAAdapter,
    WAAConfig,
    WAAMockAdapter,
)

__all__ = [
    "WAA_DOMAINS",
    "WAAAdapter",
    "WAAConfig",
    "WAAMockAdapter",
]
