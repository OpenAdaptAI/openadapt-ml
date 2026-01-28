"""DEPRECATED: Import from openadapt_evals instead.

This module is kept for backward compatibility only.
All classes are now provided by openadapt_evals.benchmarks.live_tracker.
"""

import warnings

warnings.warn(
    "openadapt_ml.benchmarks.live_tracker is deprecated. "
    "Please import from openadapt_evals instead: "
    "from openadapt_evals import LiveEvaluationTracker",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from openadapt_evals.benchmarks.live_tracker import (
    LiveEvaluationTracker,
)

__all__ = [
    "LiveEvaluationTracker",
]
