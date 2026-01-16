"""Benchmark integration for openadapt-ml.

DEPRECATION NOTICE:
    The canonical benchmark code is now in the `openadapt-evals` package.
    For new projects, prefer importing from `openadapt_evals`:

    ```python
    # Preferred (standalone, no openadapt-ml dependency)
    from openadapt_evals import ApiAgent, WAAMockAdapter, evaluate_agent_on_benchmark

    # Still supported (uses openadapt-ml internals)
    from openadapt_ml.benchmarks import PolicyAgent, APIBenchmarkAgent
    ```

    The following are ONLY available in openadapt-ml (they wrap openadapt-ml internals):
    - PolicyAgent (wraps openadapt_ml.runtime.policy.AgentPolicy)
    - APIBenchmarkAgent (wraps openadapt_ml.models.api_adapter.ApiVLMAdapter)

    The following should be imported from openadapt-evals:
    - ApiAgent (standalone, P0 demo persistence fix)
    - All adapter classes (WAAAdapter, WAALiveAdapter, etc.)
    - Base classes (BenchmarkAdapter, BenchmarkTask, etc.)
    - Evaluation utilities (evaluate_agent_on_benchmark, compute_metrics)

This module provides interfaces and utilities for evaluating GUI agents
on standardized benchmarks like Windows Agent Arena (WAA), OSWorld,
WebArena, and others.

Core classes:
    - BenchmarkAdapter: Abstract interface for benchmark integration
    - BenchmarkAgent: Abstract interface for agents to be evaluated
    - BenchmarkTask, BenchmarkObservation, BenchmarkAction: Data classes

Agent implementations:
    - PolicyAgent: Wraps openadapt-ml AgentPolicy
    - APIBenchmarkAgent: Uses hosted VLM APIs (Claude, GPT-5.1) via openadapt-ml adapters
    - ScriptedAgent: Follows predefined action sequence
    - RandomAgent: Takes random actions (baseline)

Evaluation:
    - evaluate_agent_on_benchmark: Run agent on benchmark tasks
    - compute_metrics: Compute aggregate metrics from results

Example:
    ```python
    from openadapt_ml.benchmarks import (
        BenchmarkAdapter,
        PolicyAgent,
        APIBenchmarkAgent,
        evaluate_agent_on_benchmark,
        compute_metrics,
    )

    # Create adapter for specific benchmark
    adapter = WAAAdapter(waa_repo_path="/path/to/WAA")

    # Wrap policy as benchmark agent
    agent = PolicyAgent(policy)

    # Or use API-backed agent for baselines
    agent = APIBenchmarkAgent(provider="anthropic")  # Claude
    agent = APIBenchmarkAgent(provider="openai")     # GPT-5.1

    # Run evaluation
    results = evaluate_agent_on_benchmark(agent, adapter, max_steps=50)

    # Compute metrics
    metrics = compute_metrics(results)
    print(f"Success rate: {metrics['success_rate']:.1%}")
    ```
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "openadapt_ml.benchmarks is deprecated. "
    "Please use openadapt_evals for standalone benchmark evaluation. "
    "See CLAUDE.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

from openadapt_ml.benchmarks.agent import (
    APIBenchmarkAgent,
    BenchmarkAgent,
    PolicyAgent,
    RandomAgent,
    ScriptedAgent,
    SmartMockAgent,
)
from openadapt_ml.benchmarks.base import (
    BenchmarkAction,
    BenchmarkAdapter,
    BenchmarkObservation,
    BenchmarkResult,
    BenchmarkTask,
    StaticDatasetAdapter,
    UIElement,
)
from openadapt_ml.benchmarks.runner import (
    EvaluationConfig,
    compute_domain_metrics,
    compute_metrics,
    evaluate_agent_on_benchmark,
)
from openadapt_ml.benchmarks.waa import WAAAdapter, WAAConfig, WAAMockAdapter
from openadapt_ml.benchmarks.waa_live import WAALiveAdapter, WAALiveConfig
from openadapt_ml.benchmarks.viewer import generate_benchmark_viewer

# Azure orchestration (lazy import to avoid requiring azure-ai-ml)
def _get_azure_classes():
    from openadapt_ml.benchmarks.azure import (
        AzureConfig,
        AzureWAAOrchestrator,
        estimate_cost,
    )
    return AzureConfig, AzureWAAOrchestrator, estimate_cost


__all__ = [
    # Base classes
    "BenchmarkAdapter",
    "BenchmarkTask",
    "BenchmarkObservation",
    "BenchmarkAction",
    "BenchmarkResult",
    "StaticDatasetAdapter",
    "UIElement",
    # Agents
    "BenchmarkAgent",
    "PolicyAgent",
    "APIBenchmarkAgent",
    "ScriptedAgent",
    "RandomAgent",
    "SmartMockAgent",
    # Evaluation
    "EvaluationConfig",
    "evaluate_agent_on_benchmark",
    "compute_metrics",
    "compute_domain_metrics",
    # WAA
    "WAAAdapter",
    "WAAConfig",
    "WAAMockAdapter",
    "WAALiveAdapter",
    "WAALiveConfig",
    # Viewer
    "generate_benchmark_viewer",
    # Azure (lazy-loaded)
    "AzureConfig",
    "AzureWAAOrchestrator",
    "estimate_cost",
]


# Lazy loading for Azure classes (avoids requiring azure-ai-ml for basic usage)
def __getattr__(name: str):
    if name in ("AzureConfig", "AzureWAAOrchestrator", "estimate_cost"):
        from openadapt_ml.benchmarks.azure import (
            AzureConfig,
            AzureWAAOrchestrator,
            estimate_cost,
        )
        return {"AzureConfig": AzureConfig, "AzureWAAOrchestrator": AzureWAAOrchestrator, "estimate_cost": estimate_cost}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
