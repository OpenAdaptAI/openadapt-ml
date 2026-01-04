# WAA Task Parallelization Implementation Plan

## Executive Summary

The goal is to reduce the 30-task benchmark time from approximately 5 hours to under 1 hour. Based on the analysis of the existing codebase and the official Windows Agent Arena (WAA) implementation, the **recommended approach is Multiple Azure VMs Running in Parallel**, following the pattern already implemented by Microsoft in their `run_azure.py` script.

## Current State Analysis

### Existing Infrastructure

1. **Single VM Approach** (Currently Implemented in `cli.py`):
   - Location: `/openadapt_ml/benchmarks/cli.py`
   - Commands: `vm setup-waa`, `vm prepare-windows`, `vm run-waa`
   - Uses `Standard_D4ds_v5` VM with nested virtualization
   - Runs one Windows VM inside Docker/QEMU per Azure VM
   - Each task takes approximately 10 minutes on average

2. **Azure ML Orchestrator** (Partially Implemented in `azure.py`):
   - Location: `/openadapt_ml/benchmarks/azure.py`
   - `AzureWAAOrchestrator` class provisions multiple Azure ML compute instances
   - Uses `ThreadPoolExecutor` for parallel VM provisioning
   - Task distribution via round-robin: `batches[i % num_workers].append(task)`
   - **Issue**: Azure ML compute instances do not support nested virtualization

3. **Official WAA Azure Implementation** (`vendor/WindowsAgentArena/scripts/run_azure.py`):
   - Uses Python `multiprocessing.Process` to launch workers in parallel
   - Each worker creates an Azure ML compute instance
   - Workers are automatically cleaned up after completion
   - Reference configuration: 4 workers completing 154 tasks in ~30 minutes

### Key Timing Data (from WAA README)

| Configuration | Tasks | Time | Workers | Cost |
|--------------|-------|------|---------|------|
| Single VM | 30 | ~5 hours | 1 | ~$2.50 |
| 10 workers | 154 | ~20 min | 10 | ~$0.60 |
| 40 workers | 154 | ~8 min | 40 | ~$1.00 |

## Architecture Options Analysis

### Option A: Multiple Azure VMs (Recommended)

**Description**: Create multiple dedicated Azure VMs with nested virtualization, each running one WAA Docker container. Use a coordinator process to distribute tasks and collect results.

**Pros**:
- Directly follows the proven WAA architecture from Microsoft
- True parallelism with isolated Windows environments
- Linear speedup (N workers = N times faster)
- Each VM has full resources for its Windows VM
- Fault isolation (one VM failure does not affect others)

**Cons**:
- Higher fixed cost (each VM has provisioning overhead)
- More complex orchestration logic needed
- Requires adequate vCPU quota (4-8 vCPUs per worker)

**Estimated Time for 30 Tasks**:
- 5 workers: ~30 minutes
- 10 workers: ~15 minutes

### Option B: Single VM with Multiple Docker Containers

**Description**: Run multiple WAA Docker containers on a single large Azure VM, each with its own Windows VM inside QEMU.

**Pros**:
- Lower provisioning overhead (one VM creation)
- Simpler orchestration (single SSH target)
- Potentially lower cost for small task sets

**Cons**:
- Resource contention between QEMU instances
- Single point of failure
- Requires very large VM (32+ vCPUs, 128GB+ RAM)
- Not tested or documented in official WAA
- QEMU/KVM overhead scales poorly with multiple VMs
- Network complexity (port mapping for multiple Windows VMs)

**Estimated Time for 30 Tasks**:
- 4 containers: ~75 minutes (with significant resource contention)

### Option C: Azure Container Instances (ACI)

**Not viable** - ACI does not support `/dev/kvm` device passthrough for QEMU.

### Option D: Kubernetes (AKS)

**Not recommended** for initial implementation - overkill for batch benchmark runs.

## Recommended Implementation: Option A - Multiple Azure VMs

### Phase 1: Basic Multi-VM Parallelization

**Step 1: Extend VM CLI Commands for Parallel Creation**

Modify `openadapt_ml/benchmarks/cli.py` to add:

```python
# New command: vm create-pool
# Creates multiple VMs in parallel using ThreadPoolExecutor
# Arguments: --workers N, --name-prefix, --location
```

**Step 2: Create Task Distributor**

Add new module `openadapt_ml/benchmarks/parallel_runner.py`:

```python
class ParallelWAARunner:
    """Orchestrate WAA benchmark across multiple VMs."""

    def __init__(self, vm_configs: list[VMConfig], api_key: str):
        self.vms = vm_configs
        self.api_key = api_key

    def distribute_tasks(self, task_ids: list[str]) -> dict[str, list[str]]:
        """Round-robin task distribution."""
        distribution = {vm.name: [] for vm in self.vms}
        for i, task_id in enumerate(task_ids):
            vm_name = self.vms[i % len(self.vms)].name
            distribution[vm_name].append(task_id)
        return distribution

    def run_parallel(self, task_ids: list[str]) -> list[BenchmarkResult]:
        """Execute tasks across all VMs in parallel."""
        pass
```

**Step 3: Implement Parallel Execution**

Each worker VM runs a modified version of the WAA benchmark client that:
1. Accepts a subset of task IDs via JSON file
2. Executes tasks sequentially within its VM
3. Writes results to a predictable location
4. Signals completion via a results file

**Step 4: Result Collection and Aggregation**

```python
def collect_results(self, vms: list[VMConfig]) -> list[BenchmarkResult]:
    """SSH into each VM and download results."""
    all_results = []
    for vm in vms:
        results_path = f"/mnt/waa-results/{vm.name}/results.json"
        # SCP download
        # Parse and aggregate
    return all_results
```

### Phase 2: Enhanced Orchestration

**Step 5: Add VM Pool Management CLI**

New commands:

```bash
# Create a pool of N VMs
uv run python -m openadapt_ml.benchmarks.cli vm create-pool --workers 5

# Run benchmark on pool
uv run python -m openadapt_ml.benchmarks.cli vm run-pool --num-tasks 30

# Status of all pool VMs
uv run python -m openadapt_ml.benchmarks.cli vm pool-status

# Cleanup entire pool
uv run python -m openadapt_ml.benchmarks.cli vm delete-pool
```

**Step 6: Implement Progress Tracking**

Extend `LiveEvaluationTracker` in `openadapt_ml/benchmarks/live_tracker.py` to support multi-VM tracking.

**Step 7: Add Fault Tolerance**

- Retry failed VM provisioning (up to 3 attempts)
- Detect and handle task failures
- Support resumption of partially completed runs
- Graceful handling of VM timeout/crash

### Phase 3: Integration and Polish

**Step 8: Dashboard Integration**

Update benchmark viewer to show:
- Per-worker progress
- Aggregated success rate
- Cost tracking per worker
- VNC links for each Windows VM

**Step 9: Cost Optimization**

- Auto-delete VMs upon task completion
- Support spot/preemptible VMs for cost savings
- Implement VM size selection based on task count

## Implementation Steps (Ordered)

1. **Add `--workers` flag to existing `vm setup-waa` command**
   - Accept integer for number of parallel VMs
   - Generate unique names: `waa-eval-1`, `waa-eval-2`, etc.
   - Create VMs in parallel using `ThreadPoolExecutor`

2. **Create `ParallelWAARunner` class**
   - Task distribution logic
   - Parallel SSH execution
   - Result aggregation

3. **Modify Docker run command to accept task subset**
   - Pass task IDs via environment variable or mounted JSON file
   - Ensure results are written to per-worker directories

4. **Implement result collection**
   - SCP results from each VM
   - Aggregate into single results JSON
   - Calculate overall metrics

5. **Add pool management commands**
   - `vm pool-status`: Check all workers
   - `vm delete-pool`: Clean up all workers
   - `vm pool-logs`: Get logs from all workers

6. **Update `VMRegistry` for pool management**
   - Track pools as groups of VMs
   - Support batch operations

7. **Add live progress tracking for parallel runs**
   - Extend `benchmark_live.json` schema for multi-worker
   - Update viewer to display per-worker progress

## Code Changes Summary

| File | Changes |
|------|---------|
| `openadapt_ml/benchmarks/cli.py` | Add `--workers` to `vm setup-waa`, add pool commands |
| `openadapt_ml/benchmarks/parallel_runner.py` | New file: `ParallelWAARunner` class |
| `openadapt_ml/benchmarks/vm_monitor.py` | Extend `VMRegistry` for pool management |
| `openadapt_ml/benchmarks/live_tracker.py` | Add multi-worker support |
| `openadapt_ml/benchmarks/waa/Dockerfile` | Add support for `TASK_IDS` env var |

## Cost Implications

### Assumptions
- VM size: `Standard_D4ds_v5` (~$0.19/hour)
- Average task duration: 10 minutes
- Provisioning overhead: 5 minutes per VM
- Windows golden image: Already prepared

### Cost Estimates for 30 Tasks

| Workers | Time | VM-Hours | Est. Cost |
|---------|------|----------|-----------|
| 1 | ~5 hours | 5.0 | ~$0.95 |
| 3 | ~1.75 hours | 5.25 | ~$1.00 |
| 5 | ~1.1 hours | 5.5 | ~$1.05 |
| 10 | ~35 min | 5.8 | ~$1.10 |

### Quota Requirements

For N workers using `Standard_D4ds_v5` (4 vCPUs each):
- 5 workers: 20 vCPUs
- 10 workers: 40 vCPUs

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Quota insufficient | Start with 3-5 workers; request quota increase |
| VM provisioning failure | Retry logic with fallback locations |
| Task hangs indefinitely | Per-task timeout (15 min default) |
| API rate limits | Staggered VM creation; exponential backoff |
| Storage quota | Use /mnt temp disk (115GB) per VM |
| Cost overrun | Auto-delete VMs on completion; budget alerts |
| Network issues | Health check polling; auto-restart containers |

## Success Criteria

- 30 tasks complete in under 1 hour with 5 workers
- Results are correctly aggregated
- Failed tasks are logged with error details
- VMs are automatically cleaned up
- Cost is predictable and documented
