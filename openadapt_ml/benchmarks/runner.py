"""DEPRECATED: Import from openadapt_evals instead.

This module is kept for backward compatibility only.
All functions are now provided by openadapt_evals.benchmarks.runner.
"""

import warnings

warnings.warn(
    "openadapt_ml.benchmarks.runner is deprecated. "
    "Please import from openadapt_evals instead: "
    "from openadapt_evals import evaluate_agent_on_benchmark, compute_metrics",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from openadapt_evals.benchmarks.runner import (
    EvaluationConfig,
    evaluate_agent_on_benchmark,
)
from openadapt_evals.benchmarks import (
    compute_domain_metrics,
    compute_metrics,
)

__all__ = [
    "EvaluationConfig",
    "compute_domain_metrics",
    "compute_metrics",
    "evaluate_agent_on_benchmark",
]
