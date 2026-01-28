"""Benchmark integration for openadapt-ml.

This module provides benchmark evaluation capabilities by re-exporting from
`openadapt-evals` (the canonical benchmark package) plus ML-specific agents
that wrap openadapt-ml internals.

For standalone benchmark evaluation (no ML training), use openadapt-evals:
    ```python
    from openadapt_evals import ApiAgent, WAAMockAdapter, evaluate_agent_on_benchmark
    ```

For ML-specific agents that use trained models:
    ```python
    from openadapt_ml.benchmarks import PolicyAgent, APIBenchmarkAgent
    ```

ML-specific agents (only available in openadapt-ml):
    - PolicyAgent: Wraps openadapt_ml.runtime.policy.AgentPolicy
    - APIBenchmarkAgent: Uses openadapt_ml.models.api_adapter.ApiVLMAdapter
    - UnifiedBaselineAgent: Uses openadapt_ml.baselines adapters
"""

import warnings

# Emit deprecation warning for users still importing base classes from here
warnings.warn(
    "For standalone benchmark evaluation, prefer importing from openadapt_evals directly. "
    "openadapt_ml.benchmarks re-exports from openadapt_evals for backward compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

# ruff: noqa: E402
# Imports after warning call are intentional

# Re-export base classes from openadapt-evals (canonical location)
from openadapt_evals import (
    # Base classes
    BenchmarkAction,
    BenchmarkAdapter,
    BenchmarkObservation,
    BenchmarkResult,
    BenchmarkTask,
    StaticDatasetAdapter,
    UIElement,
    # Base agent interface
    BenchmarkAgent,
    # Test/mock agents (no ML deps)
    RandomAgent,
    ScriptedAgent,
    SmartMockAgent,
    # Standalone API agent (P0 demo persistence fix)
    ApiAgent,
    # Evaluation utilities
    EvaluationConfig,
    compute_domain_metrics,
    compute_metrics,
    evaluate_agent_on_benchmark,
    # WAA adapters
    WAAAdapter,
    WAAConfig,
    WAAMockAdapter,
    WAALiveAdapter,
    WAALiveConfig,
    # Viewer
    generate_benchmark_viewer,
    # Data collection
    ExecutionTraceCollector,
    LiveEvaluationTracker,
    save_execution_trace,
)

# ML-specific agents (only available in openadapt-ml)
from openadapt_ml.benchmarks.agent import (
    APIBenchmarkAgent,
    PolicyAgent,
    UnifiedBaselineAgent,
)


# Lazy import for Azure classes (avoids requiring azure-ai-ml for basic usage)
def __getattr__(name: str):
    if name in ("AzureConfig", "AzureWAAOrchestrator", "estimate_cost"):
        from openadapt_evals.benchmarks import azure
        return getattr(azure, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes (from openadapt-evals)
    "BenchmarkAdapter",
    "BenchmarkTask",
    "BenchmarkObservation",
    "BenchmarkAction",
    "BenchmarkResult",
    "StaticDatasetAdapter",
    "UIElement",
    # Agents (from openadapt-evals)
    "BenchmarkAgent",
    "ScriptedAgent",
    "RandomAgent",
    "SmartMockAgent",
    "ApiAgent",
    # ML-specific agents (openadapt-ml only)
    "PolicyAgent",
    "APIBenchmarkAgent",
    "UnifiedBaselineAgent",
    # Evaluation (from openadapt-evals)
    "EvaluationConfig",
    "evaluate_agent_on_benchmark",
    "compute_metrics",
    "compute_domain_metrics",
    # WAA (from openadapt-evals)
    "WAAAdapter",
    "WAAConfig",
    "WAAMockAdapter",
    "WAALiveAdapter",
    "WAALiveConfig",
    # Viewer (from openadapt-evals)
    "generate_benchmark_viewer",
    # Data collection (from openadapt-evals)
    "ExecutionTraceCollector",
    "LiveEvaluationTracker",
    "save_execution_trace",
    # Azure (lazy-loaded from openadapt-evals)
    "AzureConfig",
    "AzureWAAOrchestrator",
    "estimate_cost",
]
