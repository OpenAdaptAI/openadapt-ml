"""Benchmark integration for openadapt-ml.

This module provides:

1. ML-specific agents for benchmark evaluation (PolicyAgent, APIBenchmarkAgent, etc.)
2. Azure VM management with clean Python API (AzureVMManager)
3. Pool management for parallel WAA evaluation (PoolManager)

For benchmark infrastructure (adapters, runners, viewers), use openadapt-evals:
    ```python
    from openadapt_evals import (
        WAAMockAdapter,
        WAALiveAdapter,
        evaluate_agent_on_benchmark,
    )
    ```

Library usage (programmatic, no CLI):
    ```python
    from openadapt_ml.benchmarks import PoolManager, AzureVMManager

    vm = AzureVMManager(resource_group="my-rg")
    manager = PoolManager(vm_manager=vm)
    pool = manager.create(workers=4)
    manager.wait()
    result = manager.run(tasks=10)
    manager.cleanup(confirm=False)
    ```
"""

from openadapt_ml.benchmarks.agent import (
    APIBenchmarkAgent,
    PolicyAgent,
    UnifiedBaselineAgent,
)
from openadapt_ml.benchmarks.azure_vm import AzureVMManager
from openadapt_ml.benchmarks.pool import PoolManager, PoolRunResult

__all__ = [
    "PolicyAgent",
    "APIBenchmarkAgent",
    "UnifiedBaselineAgent",
    "AzureVMManager",
    "PoolManager",
    "PoolRunResult",
]
