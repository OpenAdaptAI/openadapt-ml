"""ML-specific agents for benchmark evaluation.

This module provides agents that wrap openadapt-ml ML components
(VLM adapters, policies, baselines) for benchmark evaluation.

For evaluation infrastructure (VM management, pool orchestration, CLI,
adapters, runners, viewers), use openadapt-evals:
    ```python
    from openadapt_evals import (
        WAAMockAdapter,
        WAALiveAdapter,
        evaluate_agent_on_benchmark,
    )
    # VM/pool management CLI:
    #   oa-vm pool-create --workers 4
    #   oa-vm pool-run --tasks 10
    ```

ML agent usage:
    ```python
    from openadapt_ml.benchmarks import PolicyAgent, APIBenchmarkAgent

    agent = APIBenchmarkAgent(provider="anthropic")
    agent = PolicyAgent(policy)
    ```
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
