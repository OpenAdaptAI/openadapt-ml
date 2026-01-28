"""DEPRECATED: Import from openadapt_evals instead.

This module is kept for backward compatibility only.
All classes are now provided by openadapt_evals.benchmarks.data_collection.
"""

import warnings

warnings.warn(
    "openadapt_ml.benchmarks.data_collection is deprecated. "
    "Please import from openadapt_evals instead: "
    "from openadapt_evals import ExecutionTraceCollector, save_execution_trace",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from openadapt_evals.benchmarks.data_collection import (
    ExecutionTraceCollector,
    save_execution_trace,
)

__all__ = [
    "ExecutionTraceCollector",
    "save_execution_trace",
]
