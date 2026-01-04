# WAA Parallelization Implementation Guide

## Overview

This document provides implementation-ready specifications for parallelizing Windows Agent Arena (WAA) benchmark evaluation across multiple Azure VMs. It extends the architectural plan in [waa_parallelization_plan.md](./waa_parallelization_plan.md).

**Goal**: Reduce 30-task benchmark time from ~5 hours (single VM) to under 1 hour (5+ workers).

**Key Constraint**: Azure ML compute instances do not support nested virtualization. We must use dedicated Azure VMs with nested virt capabilities (Standard_D*ds_v5 series).

---

## 1. Summary of Existing Plan

From `waa_parallelization_plan.md`:

- **Recommended approach**: Multiple Azure VMs running in parallel (Option A)
- **Architecture**: Each Azure VM runs one WAA Docker container with QEMU/Windows VM inside
- **Parallelism**: Linear speedup - N workers = N times faster
- **Target**: 5 workers completing 30 tasks in ~30 minutes

| Workers | Time | Est. Cost |
|---------|------|-----------|
| 1 | ~5 hours | ~$0.95 |
| 5 | ~1.1 hours | ~$1.05 |
| 10 | ~35 min | ~$1.10 |

---

## 2. Phase 1: CLI Changes for `--workers` Flag

### 2.1 Modify `vm setup-waa` Command

**File**: `openadapt_ml/benchmarks/cli.py`

Add `--workers` flag to create multiple VMs in parallel:

```python
# In argparse setup (around line 2784)
p_vm.add_argument("--workers", type=int, default=1,
                  help="Number of worker VMs to create (default: 1)")
```

### 2.2 Implementation in `cmd_vm()` Function

When `args.action == "setup-waa"` and `args.workers > 1`:

```python
# Pseudocode for parallel VM creation
if args.workers > 1:
    print(f"\n=== Creating {args.workers} WAA Worker VMs ===\n")

    # Generate VM names
    vm_names = [f"waa-eval-{i+1}" for i in range(args.workers)]

    # Create VMs in parallel using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, 5)) as executor:
        futures = {
            executor.submit(create_single_vm, name, location, resource_group, api_key): name
            for name in vm_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                ip = future.result()
                results[name] = {"ip": ip, "status": "created"}
            except Exception as e:
                results[name] = {"ip": None, "status": "failed", "error": str(e)}

    # Save to VM registry
    registry = VMPoolRegistry()
    registry.save_pool(results)
```

### 2.3 New Helper Function: `create_single_vm()`

Extract VM creation logic into a reusable function:

```python
def create_single_vm(
    vm_name: str,
    location: str,
    resource_group: str,
    api_key: str | None = None,
) -> str:
    """Create a single WAA-ready VM.

    Args:
        vm_name: Name for the VM (e.g., "waa-eval-1").
        location: Azure region.
        resource_group: Azure resource group.
        api_key: Optional OpenAI API key.

    Returns:
        Public IP address of created VM.

    Raises:
        RuntimeError: If VM creation fails.
    """
    # Try multiple locations if needed
    locations_to_try = [location, "westus2", "centralus", "eastus2"]

    for loc in locations_to_try:
        result = subprocess.run(
            ["az", "vm", "create",
             "--resource-group", resource_group,
             "--name", vm_name,
             "--location", loc,
             "--image", "Ubuntu2204",
             "--size", "Standard_D4ds_v5",
             "--admin-username", "azureuser",
             "--generate-ssh-keys",
             "--public-ip-sku", "Standard"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            vm_info = json.loads(result.stdout)
            ip = vm_info.get("publicIpAddress", "")

            # Install Docker, clone WAA, configure
            _setup_vm_software(ip, api_key)

            return ip

    raise RuntimeError(f"Could not create VM {vm_name} in any region")


def _setup_vm_software(ip: str, api_key: str | None) -> None:
    """Install Docker, clone WAA, and configure on a VM."""
    docker_cmds = [
        "sudo apt-get update -qq",
        "sudo apt-get install -y -qq docker.io",
        "sudo systemctl start docker && sudo systemctl enable docker",
        "sudo usermod -aG docker $USER",
        "sudo systemctl stop docker",
        "sudo mkdir -p /mnt/docker",
        "echo '{\"data-root\": \"/mnt/docker\"}' | sudo tee /etc/docker/daemon.json",
        "sudo systemctl start docker",
        "sudo docker pull dockurr/windows:latest",
        "cd ~ && git clone --depth 1 https://github.com/microsoft/WindowsAgentArena.git 2>/dev/null || true",
    ]

    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
         f"azureuser@{ip}", " && ".join(docker_cmds)],
        capture_output=True, text=True, timeout=600
    )

    # Configure API key
    if api_key:
        config_cmd = f'''cat > ~/WindowsAgentArena/config.json << 'EOF'
{{"OPENAI_API_KEY": "{api_key}", "AZURE_API_KEY": "", "AZURE_ENDPOINT": ""}}
EOF'''
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", config_cmd],
            capture_output=True, text=True
        )
```

---

## 3. VM Naming Convention

### 3.1 Standard Names

| Workers | VM Names |
|---------|----------|
| 1 | `waa-eval-vm` (backward compatible) |
| 2 | `waa-eval-1`, `waa-eval-2` |
| 5 | `waa-eval-1`, ..., `waa-eval-5` |
| N | `waa-eval-1`, ..., `waa-eval-N` |

### 3.2 Pool Identification

Pools are identified by a timestamp prefix stored in the registry:

```json
{
  "pool_id": "20260102_143052",
  "created_at": "2026-01-02T14:30:52Z",
  "workers": [
    {"name": "waa-eval-1", "ip": "52.168.1.100", "status": "ready"},
    {"name": "waa-eval-2", "ip": "52.168.1.101", "status": "ready"}
  ]
}
```

---

## 4. VM Registry File Format

### 4.1 Location

```
benchmark_results/vm_pool_registry.json
```

### 4.2 JSON Schema

```json
{
  "pool_id": "string (YYYYMMDD_HHMMSS)",
  "created_at": "ISO8601 timestamp",
  "resource_group": "string",
  "location": "string",
  "vm_size": "string (e.g., Standard_D4ds_v5)",
  "workers": [
    {
      "name": "waa-eval-1",
      "ip": "52.168.1.100",
      "status": "creating | ready | running | completed | failed | deleted",
      "docker_container": "winarena",
      "waa_ready": false,
      "assigned_tasks": ["notepad_1", "notepad_2"],
      "completed_tasks": [],
      "current_task": null,
      "error": null,
      "created_at": "ISO8601 timestamp",
      "updated_at": "ISO8601 timestamp"
    }
  ],
  "total_tasks": 30,
  "completed_tasks": 0,
  "failed_tasks": 0
}
```

### 4.3 VMPoolRegistry Class

**File**: `openadapt_ml/benchmarks/vm_monitor.py`

```python
@dataclass
class PoolWorker:
    """A single worker in a VM pool."""
    name: str
    ip: str
    status: str = "creating"  # creating, ready, running, completed, failed, deleted
    docker_container: str = "winarena"
    waa_ready: bool = False
    assigned_tasks: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    current_task: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class VMPool:
    """A pool of worker VMs."""
    pool_id: str
    created_at: str
    resource_group: str
    location: str
    vm_size: str
    workers: list[PoolWorker]
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0


class VMPoolRegistry:
    """Manage VM pools for parallel WAA evaluation."""

    REGISTRY_FILE = "benchmark_results/vm_pool_registry.json"

    def __init__(self, registry_file: str | Path | None = None):
        self.registry_file = Path(registry_file or self.REGISTRY_FILE)
        self._pool: VMPool | None = None
        self.load()

    def load(self) -> None:
        """Load pool from registry file."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                workers = [PoolWorker(**w) for w in data.get("workers", [])]
                self._pool = VMPool(
                    pool_id=data["pool_id"],
                    created_at=data["created_at"],
                    resource_group=data["resource_group"],
                    location=data["location"],
                    vm_size=data["vm_size"],
                    workers=workers,
                    total_tasks=data.get("total_tasks", 0),
                    completed_tasks=data.get("completed_tasks", 0),
                    failed_tasks=data.get("failed_tasks", 0),
                )

    def save(self) -> None:
        """Save pool to registry file."""
        if self._pool is None:
            return
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump(asdict(self._pool), f, indent=2)

    def create_pool(
        self,
        workers: list[tuple[str, str]],  # [(name, ip), ...]
        resource_group: str,
        location: str,
        vm_size: str = "Standard_D4ds_v5",
    ) -> VMPool:
        """Create a new pool from created VMs."""
        pool_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._pool = VMPool(
            pool_id=pool_id,
            created_at=datetime.now().isoformat(),
            resource_group=resource_group,
            location=location,
            vm_size=vm_size,
            workers=[PoolWorker(name=name, ip=ip, status="ready") for name, ip in workers],
        )
        self.save()
        return self._pool

    def get_pool(self) -> VMPool | None:
        """Get current pool."""
        return self._pool

    def update_worker(self, name: str, **kwargs) -> None:
        """Update a worker's status."""
        if self._pool is None:
            return
        for worker in self._pool.workers:
            if worker.name == name:
                for key, value in kwargs.items():
                    if hasattr(worker, key):
                        setattr(worker, key, value)
                worker.updated_at = datetime.now().isoformat()
                break
        self.save()

    def delete_pool(self) -> bool:
        """Delete the pool registry (VMs must be deleted separately)."""
        if self.registry_file.exists():
            self.registry_file.unlink()
            self._pool = None
            return True
        return False
```

---

## 5. New CLI Commands

### 5.1 `vm pool-status`

**Purpose**: Show status of all VMs in the current pool.

**Implementation**:

```python
elif args.action == "pool-status":
    registry = VMPoolRegistry()
    pool = registry.get_pool()

    if pool is None:
        print("No active VM pool. Create one with: vm setup-waa --workers N")
        sys.exit(0)

    print(f"\n=== VM Pool: {pool.pool_id} ===\n")
    print(f"Created: {pool.created_at}")
    print(f"Workers: {len(pool.workers)}")
    print(f"Tasks: {pool.completed_tasks}/{pool.total_tasks}")
    print()

    # Table header
    print(f"{'Name':<15} {'IP':<16} {'Status':<12} {'WAA':<6} {'Tasks':<10}")
    print("-" * 60)

    for w in pool.workers:
        waa_status = "Ready" if w.waa_ready else "---"
        task_progress = f"{len(w.completed_tasks)}/{len(w.assigned_tasks)}"
        print(f"{w.name:<15} {w.ip:<16} {w.status:<12} {waa_status:<6} {task_progress:<10}")

    # Probe each VM for live status
    if args.probe:
        print("\nProbing VMs for WAA readiness...")
        for w in pool.workers:
            monitor = VMMonitor(VMConfig(name=w.name, ssh_host=w.ip))
            status = monitor.check_status()
            ready = "READY" if status.waa_ready else "Not ready"
            print(f"  {w.name}: {ready}")
```

**CLI Definition**:

```python
# Add to p_vm choices
p_vm.add_argument("action", choices=[
    "create", "status", "ssh", "delete", "list-sizes", "setup",
    "pull-image", "setup-waa", "run-waa", "prepare-windows",
    "fix-storage", "reset-windows", "screenshot", "probe",
    "pool-status", "delete-pool"  # NEW
], help="Action to perform")

# Add probe flag for pool-status
p_vm.add_argument("--probe", action="store_true",
                  help="For pool-status: Check WAA readiness on each VM")
```

### 5.2 `vm delete-pool`

**Purpose**: Delete all VMs in the current pool.

**Implementation**:

```python
elif args.action == "delete-pool":
    registry = VMPoolRegistry()
    pool = registry.get_pool()

    if pool is None:
        print("No active VM pool.")
        sys.exit(0)

    print(f"\n=== Deleting VM Pool: {pool.pool_id} ===\n")
    print(f"This will delete {len(pool.workers)} VMs:")
    for w in pool.workers:
        print(f"  - {w.name} ({w.ip})")

    if not args.yes:
        confirm = input("\nType 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)

    # Delete VMs in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def delete_vm(name: str) -> tuple[str, bool, str]:
        result = subprocess.run(
            ["az", "vm", "delete", "-g", pool.resource_group, "-n", name, "--yes"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return (name, True, "deleted")
        else:
            return (name, False, result.stderr[:100])

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(delete_vm, w.name): w.name for w in pool.workers}
        for future in as_completed(futures):
            name, success, msg = future.result()
            status = "deleted" if success else f"FAILED: {msg}"
            print(f"  {name}: {status}")

    # Delete registry
    registry.delete_pool()
    print("\nPool deleted.")
```

---

## 6. Task Distribution Approach

### 6.1 Round-Robin Distribution

Tasks are distributed evenly across workers:

```python
def distribute_tasks(
    task_ids: list[str],
    num_workers: int,
) -> dict[int, list[str]]:
    """Distribute tasks across workers using round-robin.

    Args:
        task_ids: List of task IDs to distribute.
        num_workers: Number of workers.

    Returns:
        Dict mapping worker_index -> list of task_ids.

    Example:
        distribute_tasks(["t1", "t2", "t3", "t4", "t5"], 2)
        # Returns: {0: ["t1", "t3", "t5"], 1: ["t2", "t4"]}
    """
    distribution: dict[int, list[str]] = {i: [] for i in range(num_workers)}
    for i, task_id in enumerate(task_ids):
        worker_idx = i % num_workers
        distribution[worker_idx].append(task_id)
    return distribution
```

### 6.2 Saving Task Assignments

Each worker receives a JSON file with its assigned tasks:

```python
def write_task_assignment(
    worker_name: str,
    task_ids: list[str],
    output_dir: Path,
) -> Path:
    """Write task assignment file for a worker.

    Args:
        worker_name: Name of the worker VM.
        task_ids: Task IDs assigned to this worker.
        output_dir: Directory to write assignment file.

    Returns:
        Path to the assignment file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_file = output_dir / f"{worker_name}_tasks.json"

    with open(assignment_file, "w") as f:
        json.dump({
            "worker_name": worker_name,
            "task_ids": task_ids,
            "num_tasks": len(task_ids),
        }, f, indent=2)

    return assignment_file
```

### 6.3 Uploading to Workers

```python
def upload_task_assignment(
    worker_ip: str,
    assignment_file: Path,
    remote_path: str = "/home/azureuser/tasks.json",
) -> bool:
    """Upload task assignment file to a worker VM.

    Args:
        worker_ip: IP address of the worker.
        assignment_file: Local path to assignment JSON.
        remote_path: Remote path on the worker.

    Returns:
        True if upload succeeded.
    """
    result = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no",
         str(assignment_file), f"azureuser@{worker_ip}:{remote_path}"],
        capture_output=True, text=True, timeout=30
    )
    return result.returncode == 0
```

---

## 7. Result Collection from Multiple VMs

### 7.1 Result File Location on Workers

Each worker writes results to:

```
/mnt/waa-results/{worker_name}/results.json
```

### 7.2 Result JSON Schema

```json
{
  "worker_name": "waa-eval-1",
  "run_id": "20260102_143052",
  "completed_at": "2026-01-02T15:45:00Z",
  "tasks": [
    {
      "task_id": "notepad_1",
      "success": true,
      "score": 1.0,
      "num_steps": 5,
      "duration_seconds": 180,
      "error": null,
      "screenshots": ["step_0.png", "step_1.png", ...]
    }
  ],
  "summary": {
    "total": 6,
    "succeeded": 4,
    "failed": 2,
    "success_rate": 0.667
  }
}
```

### 7.3 Result Collection Implementation

```python
def collect_results_from_pool(
    pool: VMPool,
    output_dir: Path,
    timeout_per_vm: int = 120,
) -> dict[str, Any]:
    """Collect results from all workers in a pool.

    Args:
        pool: The VM pool to collect from.
        output_dir: Local directory to save results.
        timeout_per_vm: Timeout in seconds per VM.

    Returns:
        Aggregated results dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for worker in pool.workers:
        if worker.status != "completed":
            print(f"  {worker.name}: skipping (status={worker.status})")
            continue

        # Download results
        local_path = output_dir / f"{worker.name}_results.json"
        remote_path = f"/mnt/waa-results/{worker.name}/results.json"

        result = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no",
             f"azureuser@{worker.ip}:{remote_path}", str(local_path)],
            capture_output=True, text=True, timeout=timeout_per_vm
        )

        if result.returncode == 0:
            with open(local_path) as f:
                worker_results = json.load(f)
                all_results.extend(worker_results.get("tasks", []))
            print(f"  {worker.name}: {len(worker_results.get('tasks', []))} tasks collected")
        else:
            print(f"  {worker.name}: FAILED to collect ({result.stderr[:50]})")

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # Save aggregated results
    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def aggregate_results(task_results: list[dict]) -> dict[str, Any]:
    """Aggregate results from multiple workers.

    Args:
        task_results: List of task result dicts.

    Returns:
        Aggregated summary.
    """
    total = len(task_results)
    succeeded = sum(1 for t in task_results if t.get("success"))
    failed = total - succeeded

    return {
        "total_tasks": total,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": succeeded / total if total > 0 else 0,
        "tasks": task_results,
        "by_domain": _group_by_domain(task_results),
    }


def _group_by_domain(task_results: list[dict]) -> dict[str, dict]:
    """Group results by domain."""
    domains: dict[str, list] = {}
    for task in task_results:
        # Extract domain from task_id (e.g., "notepad_1" -> "notepad")
        domain = task.get("task_id", "").rsplit("_", 1)[0]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(task)

    return {
        domain: {
            "total": len(tasks),
            "succeeded": sum(1 for t in tasks if t.get("success")),
            "success_rate": sum(1 for t in tasks if t.get("success")) / len(tasks) if tasks else 0,
        }
        for domain, tasks in domains.items()
    }
```

---

## 8. Code Structure: Files to Modify

### 8.1 Files to Modify

| File | Changes |
|------|---------|
| `openadapt_ml/benchmarks/cli.py` | Add `--workers` flag to `setup-waa`, add `pool-status` and `delete-pool` actions |
| `openadapt_ml/benchmarks/vm_monitor.py` | Add `VMPoolRegistry`, `VMPool`, `PoolWorker` classes |
| `openadapt_ml/benchmarks/live_tracker.py` | Add multi-worker support to `LiveEvaluationTracker` |

### 8.2 New File: `parallel_runner.py`

**Location**: `openadapt_ml/benchmarks/parallel_runner.py`

```python
"""Parallel WAA benchmark runner.

This module orchestrates WAA evaluation across multiple Azure VMs.

Usage:
    from openadapt_ml.benchmarks.parallel_runner import ParallelWAARunner

    runner = ParallelWAARunner(pool, api_key="...")
    runner.prepare_all_workers()
    results = runner.run(task_ids=["notepad_1", "notepad_2", ...])
"""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openadapt_ml.benchmarks.vm_monitor import VMPool, VMPoolRegistry, PoolWorker


@dataclass
class WorkerExecution:
    """State of a worker's execution."""
    worker: PoolWorker
    assigned_tasks: list[str]
    completed_tasks: list[str]
    current_task: str | None = None
    results: list[dict] = None
    error: str | None = None

    def __post_init__(self):
        if self.results is None:
            self.results = []


class ParallelWAARunner:
    """Run WAA benchmark across multiple VMs in parallel."""

    def __init__(
        self,
        pool: VMPool,
        api_key: str,
        model: str = "gpt-4o",
        results_dir: Path | None = None,
    ):
        self.pool = pool
        self.api_key = api_key
        self.model = model
        self.results_dir = results_dir or Path("benchmark_results")
        self.registry = VMPoolRegistry()

    def prepare_all_workers(self, max_wait_minutes: int = 20) -> bool:
        """Prepare Windows on all workers.

        Runs `prepare-windows` in parallel on all workers.

        Args:
            max_wait_minutes: Max time to wait for Windows to be ready.

        Returns:
            True if all workers are ready.
        """
        print(f"\n=== Preparing {len(self.pool.workers)} Workers ===\n")

        def prepare_worker(worker: PoolWorker) -> tuple[str, bool]:
            # Copy Dockerfile and build
            dockerfile_path = Path(__file__).parent / "waa" / "Dockerfile"
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", str(dockerfile_path),
                 f"azureuser@{worker.ip}:~/build-waa/Dockerfile"],
                capture_output=True, text=True, timeout=30
            )

            # Build and start container
            build_cmd = """
mkdir -p ~/build-waa
cp -r ~/WindowsAgentArena/src/win-arena-container/vm ~/build-waa/
cd ~/build-waa && docker build -t waa-auto:latest . 2>&1 | tail -5
docker stop winarena 2>/dev/null; docker rm -f winarena 2>/dev/null
rm -f /mnt/waa-storage/data.img /mnt/waa-storage/windows.* 2>/dev/null
sudo mkdir -p /mnt/waa-storage /mnt/waa-results
sudo chown azureuser:azureuser /mnt/waa-storage /mnt/waa-results
docker run -d --name winarena --device=/dev/kvm --cap-add NET_ADMIN \
  -p 8006:8006 -p 5000:5000 -p 7100:7100 -p 7200:7200 \
  -v /mnt/waa-storage:/storage -e VERSION=11e -e RAM_SIZE=12G \
  -e CPU_CORES=4 -e DISK_SIZE=64G waa-auto:latest
"""
            subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{worker.ip}",
                 build_cmd],
                capture_output=True, text=True, timeout=600
            )

            # Wait for WAA to be ready
            for _ in range(max_wait_minutes * 6):  # Check every 10s
                time.sleep(10)
                result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no",
                     f"azureuser@{worker.ip}",
                     "curl -s --connect-timeout 3 http://172.30.0.2:5000/probe"],
                    capture_output=True, text=True, timeout=30
                )
                if result.stdout.strip():
                    return (worker.name, True)

            return (worker.name, False)

        # Run preparation in parallel
        ready_count = 0
        with ThreadPoolExecutor(max_workers=len(self.pool.workers)) as executor:
            futures = {executor.submit(prepare_worker, w): w for w in self.pool.workers}
            for future in as_completed(futures):
                worker = futures[future]
                name, ready = future.result()
                status = "READY" if ready else "FAILED"
                print(f"  {name}: {status}")
                if ready:
                    ready_count += 1
                    self.registry.update_worker(name, waa_ready=True, status="ready")
                else:
                    self.registry.update_worker(name, waa_ready=False, status="failed")

        return ready_count == len(self.pool.workers)

    def run(
        self,
        task_ids: list[str] | None = None,
        num_tasks: int | None = None,
    ) -> dict[str, Any]:
        """Run WAA benchmark across all workers.

        Args:
            task_ids: Specific task IDs to run. If None, uses num_tasks.
            num_tasks: Number of tasks to run. If None, runs all.

        Returns:
            Aggregated results dictionary.
        """
        # Get ready workers
        ready_workers = [w for w in self.pool.workers if w.waa_ready]
        if not ready_workers:
            raise RuntimeError("No workers are ready. Run prepare_all_workers() first.")

        # Distribute tasks
        if task_ids is None:
            # Generate task list (would normally come from WAA task config)
            task_ids = [f"task_{i}" for i in range(num_tasks or 30)]

        distribution = distribute_tasks(task_ids, len(ready_workers))

        print(f"\n=== Running {len(task_ids)} Tasks on {len(ready_workers)} Workers ===\n")
        for i, worker in enumerate(ready_workers):
            assigned = distribution[i]
            print(f"  {worker.name}: {len(assigned)} tasks")
            self.registry.update_worker(worker.name, assigned_tasks=assigned)

        # Start benchmark on each worker
        def run_worker_benchmark(worker: PoolWorker, tasks: list[str]) -> tuple[str, list[dict]]:
            # Write tasks.json
            tasks_json = json.dumps({"task_ids": tasks})
            subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{worker.ip}",
                 f"echo '{tasks_json}' > /home/azureuser/tasks.json"],
                capture_output=True, text=True
            )

            # Run benchmark
            run_cmd = f"""
cd ~/WindowsAgentArena
python -m client.run \
  --model {self.model} \
  --tasks-file /home/azureuser/tasks.json \
  --output-dir /mnt/waa-results/{worker.name}
"""
            subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{worker.ip}",
                 run_cmd],
                capture_output=True, text=True, timeout=3600  # 1 hour timeout
            )

            # Collect results
            result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{worker.ip}",
                 f"cat /mnt/waa-results/{worker.name}/results.json"],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                return (worker.name, json.loads(result.stdout).get("tasks", []))
            return (worker.name, [])

        # Run all workers in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=len(ready_workers)) as executor:
            futures = {
                executor.submit(run_worker_benchmark, w, distribution[i]): w
                for i, w in enumerate(ready_workers)
            }
            for future in as_completed(futures):
                worker = futures[future]
                name, results = future.result()
                print(f"  {name}: {len(results)} tasks completed")
                all_results.extend(results)
                self.registry.update_worker(name, status="completed")

        # Aggregate and return
        return aggregate_results(all_results)
```

---

## 9. Quota Considerations

### 9.1 vCPU Requirements

| VM Size | vCPUs | RAM | Workers | Total vCPUs |
|---------|-------|-----|---------|-------------|
| Standard_D4ds_v5 | 4 | 16 GB | 1 | 4 |
| Standard_D4ds_v5 | 4 | 16 GB | 5 | 20 |
| Standard_D4ds_v5 | 4 | 16 GB | 10 | 40 |

### 9.2 Default Azure Quota

- Free tier: 4 vCPUs (1 worker max)
- Pay-as-you-go: 10-20 vCPUs (2-5 workers typical)
- Request increase: Up to 100+ vCPUs possible

### 9.3 Checking Current Quota

```bash
az vm list-usage --location eastus --query "[?contains(localName,'vCPU')]" -o table
```

### 9.4 Requesting Quota Increase

1. Go to Azure Portal > Subscriptions > Usage + quotas
2. Search for "Standard Ddsv5 Family vCPUs"
3. Request increase (typically approved within 24 hours)

### 9.5 Cost Estimation Formula

```python
def estimate_parallel_cost(
    num_tasks: int,
    num_workers: int,
    avg_task_duration_minutes: float = 10.0,
    vm_hourly_cost: float = 0.19,  # Standard_D4ds_v5
    provisioning_overhead_minutes: float = 5.0,
) -> dict[str, float]:
    """Estimate cost for parallel WAA run.

    Returns:
        Dict with estimated_duration_minutes, total_vm_hours, estimated_cost_usd.
    """
    tasks_per_worker = num_tasks / num_workers
    run_duration = tasks_per_worker * avg_task_duration_minutes + provisioning_overhead_minutes
    total_vm_hours = (run_duration / 60) * num_workers

    return {
        "estimated_duration_minutes": run_duration,
        "total_vm_hours": total_vm_hours,
        "estimated_cost_usd": total_vm_hours * vm_hourly_cost,
    }
```

---

## 10. Error Handling and Retry Logic

### 10.1 VM Creation Retries

```python
MAX_CREATE_RETRIES = 3
RETRY_DELAY_SECONDS = 30

def create_vm_with_retry(vm_name: str, location: str, resource_group: str) -> str:
    """Create VM with retry logic."""
    for attempt in range(MAX_CREATE_RETRIES):
        try:
            return create_single_vm(vm_name, location, resource_group)
        except Exception as e:
            if attempt < MAX_CREATE_RETRIES - 1:
                print(f"  {vm_name}: attempt {attempt + 1} failed, retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(f"Failed to create {vm_name} after {MAX_CREATE_RETRIES} attempts: {e}")
```

### 10.2 Task Timeout Handling

```python
TASK_TIMEOUT_SECONDS = 900  # 15 minutes per task

def run_task_with_timeout(worker_ip: str, task_id: str) -> dict:
    """Run a single task with timeout."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{worker_ip}",
             f"cd ~/WindowsAgentArena && timeout {TASK_TIMEOUT_SECONDS} python run_task.py --task {task_id}"],
            capture_output=True, text=True, timeout=TASK_TIMEOUT_SECONDS + 60
        )
        # Parse result...
    except subprocess.TimeoutExpired:
        return {"task_id": task_id, "success": False, "error": "timeout"}
```

### 10.3 Worker Health Checks

```python
def check_worker_health(worker: PoolWorker) -> bool:
    """Check if a worker is healthy and responsive."""
    monitor = VMMonitor(VMConfig(name=worker.name, ssh_host=worker.ip))
    status = monitor.check_status()
    return status.ssh_reachable and status.waa_ready
```

---

## 11. Monitoring and Progress Tracking

### 11.1 Live Progress File

Write progress to `benchmark_results/parallel_live.json`:

```json
{
  "status": "running",
  "pool_id": "20260102_143052",
  "started_at": "2026-01-02T14:30:52Z",
  "total_tasks": 30,
  "completed_tasks": 12,
  "failed_tasks": 2,
  "workers": [
    {
      "name": "waa-eval-1",
      "status": "running",
      "assigned": 6,
      "completed": 4,
      "current_task": "notepad_3"
    },
    {
      "name": "waa-eval-2",
      "status": "running",
      "assigned": 6,
      "completed": 4,
      "current_task": "calc_1"
    }
  ]
}
```

### 11.2 Console Progress Display

```
=== WAA Parallel Benchmark ===

Pool: 20260102_143052 | 5 workers | 30 tasks

Progress: [=========>          ] 47% (14/30)

Workers:
  waa-eval-1: [====]    4/6 tasks | notepad_3
  waa-eval-2: [===]     3/6 tasks | calc_1
  waa-eval-3: [====]    4/6 tasks | paint_2
  waa-eval-4: [===]     3/6 tasks | IDLE
  waa-eval-5: [FAILED]  0/6 tasks | Error: SSH timeout

Elapsed: 18m 32s | ETA: ~12m
```

---

## 12. Complete Workflow Example

```bash
# 1. Create 5-worker pool (takes ~10 min)
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa \
  --workers 5 \
  --api-key $OPENAI_API_KEY

# 2. Check pool status
uv run python -m openadapt_ml.benchmarks.cli vm pool-status

# 3. Prepare Windows on all workers (takes ~20 min)
uv run python -m openadapt_ml.benchmarks.cli vm prepare-windows --all

# 4. Run benchmark
uv run python -m openadapt_ml.benchmarks.cli vm run-waa \
  --num-tasks 30 \
  --model gpt-4o

# 5. View results
uv run python -m openadapt_ml.benchmarks.cli analyze \
  --results-dir benchmark_results/latest

# 6. Clean up (stop billing)
uv run python -m openadapt_ml.benchmarks.cli vm delete-pool --yes
```

---

## 13. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Add `--workers` flag to `vm setup-waa` command
- [ ] Implement `create_single_vm()` helper function
- [ ] Create `VMPoolRegistry` class in `vm_monitor.py`
- [ ] Add `pool-status` command
- [ ] Add `delete-pool` command

### Phase 2: Task Distribution
- [ ] Create `parallel_runner.py` module
- [ ] Implement `distribute_tasks()` function
- [ ] Implement task assignment file creation
- [ ] Implement task upload to workers

### Phase 3: Execution and Collection
- [ ] Implement parallel Windows preparation
- [ ] Implement parallel benchmark execution
- [ ] Implement result collection from workers
- [ ] Implement result aggregation

### Phase 4: Monitoring
- [ ] Add live progress JSON file
- [ ] Update `LiveEvaluationTracker` for multi-worker
- [ ] Add progress display to CLI
- [ ] Add VNC links in status output

### Phase 5: Error Handling
- [ ] Add retry logic for VM creation
- [ ] Add task timeout handling
- [ ] Add worker health checks
- [ ] Add graceful degradation (continue with remaining workers)
