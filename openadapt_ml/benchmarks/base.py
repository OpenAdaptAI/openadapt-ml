"""DEPRECATED: Import from openadapt_evals instead.

This module is kept for backward compatibility only.
All classes are now provided by openadapt_evals.adapters.base.
"""

import warnings

warnings.warn(
    "openadapt_ml.benchmarks.base is deprecated. "
    "Please import from openadapt_evals instead: "
    "from openadapt_evals import BenchmarkAdapter, BenchmarkTask, BenchmarkAction",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from openadapt_evals.adapters.base import (
    BenchmarkAction,
    BenchmarkAdapter,
    BenchmarkObservation,
    BenchmarkResult,
    BenchmarkTask,
    StaticDatasetAdapter,
    UIElement,
)

__all__ = [
    "BenchmarkAction",
    "BenchmarkAdapter",
    "BenchmarkObservation",
    "BenchmarkResult",
    "BenchmarkTask",
    "StaticDatasetAdapter",
    "UIElement",
]
