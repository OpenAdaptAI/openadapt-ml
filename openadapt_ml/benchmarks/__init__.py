"""Benchmark integration for openadapt-ml.

This module provides ML-specific agents for benchmark evaluation.
These agents wrap openadapt-ml internals (trained policies, API adapters).

For benchmark infrastructure (adapters, runners, viewers), use openadapt-evals:
    ```python
    from openadapt_evals import (
        WAAMockAdapter,
        WAALiveAdapter,
        evaluate_agent_on_benchmark,
    )
    ```

ML-specific agents (only available in openadapt-ml):
    - PolicyAgent: Wraps openadapt_ml.runtime.policy.AgentPolicy
    - APIBenchmarkAgent: Uses openadapt_ml.models.api_adapter.ApiVLMAdapter
    - UnifiedBaselineAgent: Uses openadapt_ml.baselines adapters
"""

from openadapt_ml.benchmarks.agent import (
    APIBenchmarkAgent,
    PolicyAgent,
    UnifiedBaselineAgent,
)

__all__ = [
    "PolicyAgent",
    "APIBenchmarkAgent",
    "UnifiedBaselineAgent",
]
