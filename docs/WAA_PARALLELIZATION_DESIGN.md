# WAA Benchmark Parallelization Design

**Last Updated:** 2026-01-29

## Overview

This document describes two approaches for running Windows Agent Arena (WAA) benchmarks:

1. **Dedicated VM Approach** (our current setup) - For development, debugging, small runs
2. **Azure ML Compute Approach** (official WAA) - For full benchmark runs at scale

## Official WAA Approach: Azure ML Compute

The official WAA repository uses Azure ML Compute Instances for parallelization.

**Source:** [README.md](https://github.com/microsoft/WindowsAgentArena/blob/main/README.md)
> "WAA supports the deployment of agents **at scale** using the Azure ML cloud infrastructure, allowing for the parallel running of multiple agents and delivering quick benchmark results for hundreds of tasks in minutes, not days."

**Implementation:** [scripts/run_azure.py](https://github.com/microsoft/WindowsAgentArena/blob/main/scripts/run_azure.py)

```python
# Official WAA creates Azure ML Compute Instances
from azure.ai.ml.entities import ComputeInstance

compute_instance = ComputeInstance(
    name=f"w{worker_id}Exp{exp_name}",
    size="Standard_D8_v3",  # 8 vCPU, nested virtualization
    setup_scripts=setup_scripts,
    idle_time_before_shutdown_minutes=600,
    ssh_public_access_enabled=True
)
ml_client.begin_create_or_update(compute_instance).result()

# Uses multiprocessing.Process for parallel workers
for worker_id in range(num_workers):
    p = Process(target=launch_vm_and_job, args=(worker_id, ...))
    processes.append(p)
    p.start()
```

---

## Our Approach: Dedicated Azure VM

We use a single dedicated Azure VM for development and debugging.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE                                    │
│                                                                          │
│  openadapt-ml CLI                                                        │
│  ├── SSH tunnel for VNC (localhost:8006 → VM:8006)                      │
│  ├── SSH tunnel for WAA API (localhost:5001 → VM:5000)                  │
│  └── Direct SSH for commands                                             │
│                                                                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
                           ┌─────────────┐
                           │ waa-eval-vm │
                           │ D4ds_v4     │
                           │             │
                           │ Docker      │
                           │   └─QEMU    │
                           │     └─Win11 │
                           │       └─WAA │
                           └─────────────┘
```

---

## When to Use Which Approach

| Use Case | Dedicated VM | Azure ML Compute |
|----------|--------------|------------------|
| **Development/debugging** | ✅ Better - VNC, SSH, full control | ❌ Harder to debug |
| **Single task testing** | ✅ Simpler | ❌ Overkill |
| **Quick iteration** | ✅ VM stays running | ❌ Compute instances spin up/down |
| **Cost for small runs** | ✅ One VM, pay as you go | ❌ ML workspace overhead |
| **Parallel at scale (40+ workers)** | ❌ Manual VM management | ✅ Designed for this |
| **Full 154-task benchmark** | ❌ ~5 hours sequential | ✅ ~30 min with 10 workers |

**Recommendation:**
- Use **dedicated VM** for development and debugging
- Use **Azure ML Compute** (official approach) for full benchmark runs

---

## Dedicated VM Details

### Current Setup

- **VM Name:** `waa-eval-vm`
- **Size:** `Standard_D4ds_v4` (4 vCPU, 16GB RAM, nested virtualization)
- **IP:** 20.12.180.208
- **OS:** Ubuntu 22.04 LTS
- **Software:** Docker with `windowsarena/winarena:latest`

### CLI Commands

```bash
# VM management
uv run python -m openadapt_ml.benchmarks.cli create      # Create VM
uv run python -m openadapt_ml.benchmarks.cli status      # Check status
uv run python -m openadapt_ml.benchmarks.cli probe       # Check WAA server
uv run python -m openadapt_ml.benchmarks.cli vnc         # Open VNC tunnel
uv run python -m openadapt_ml.benchmarks.cli logs        # View logs
uv run python -m openadapt_ml.benchmarks.cli deallocate  # Stop billing
uv run python -m openadapt_ml.benchmarks.cli delete      # Delete VM
```

### Access

- **VNC:** http://localhost:8006 (via SSH tunnel)
- **SSH:** `ssh azureuser@20.12.180.208`

---

## Azure ML Compute Details (Official WAA)

### Setup Requirements

1. Azure subscription with ML workspace
2. Storage account for golden image
3. Compute instance startup script
4. vCPU quota (8 vCPU per worker × N workers)

### Running Official WAA at Scale

```bash
cd WindowsAgentArena

# Run with 10 workers
python scripts/run_azure.py \
    --num_workers 10 \
    --agent navi \
    --model_name gpt-4o \
    --json_name evaluation_examples_windows/test_all.json
```

### Cost Estimate (Azure ML)

| Workers | VM Size | vCPUs Each | Total vCPUs | Time for 154 tasks | Est. Cost |
|---------|---------|------------|-------------|-------------------|-----------|
| 1 | D8_v3 | 8 | 8 | ~5 hours | ~$2 |
| 5 | D8_v3 | 8 | 40 | ~1 hour | ~$2 |
| 10 | D8_v3 | 8 | 80 | ~30 min | ~$2 |

---

## Components

### 1. Dedicated Azure VMs

Each VM is identical:
- **Size:** `Standard_D4ds_v4` (4 vCPU, 16GB RAM, nested virtualization)
- **OS:** Ubuntu 22.04 LTS
- **Software:** Docker with `windowsarena/winarena:latest` image
- **Inside Docker:** QEMU running Windows 11 with WAA Flask server

### 2. Task Distribution

- 154 total WAA tasks
- Tasks distributed round-robin across N VMs
- Each VM runs tasks sequentially (WAA limitation - one Windows instance per container)
- No inter-VM communication needed (embarrassingly parallel)

### 3. Orchestration (ThreadPoolExecutor)

```python
# Simplified pseudocode
with ThreadPoolExecutor(max_workers=N) as executor:
    # Phase 1: Create VMs in parallel
    vm_futures = [executor.submit(create_vm, f"waa-eval-vm-{i}") for i in range(N)]
    vms = [f.result() for f in vm_futures]

    # Phase 2: Distribute tasks
    task_assignments = distribute_tasks(tasks, vms)  # round-robin

    # Phase 3: Run tasks in parallel (one thread per VM)
    result_futures = [
        executor.submit(run_tasks_on_vm, vm, assigned_tasks)
        for vm, assigned_tasks in task_assignments
    ]
    results = [f.result() for f in result_futures]

    # Phase 4: Cleanup VMs
    for vm in vms:
        executor.submit(delete_vm, vm)
```

## Tradeoffs: Dedicated VM vs Azure ML Compute

| Aspect | Dedicated VM (Our Approach) | Azure ML Compute (Official WAA) |
|--------|----------------------------|--------------------------------|
| **Best for** | Development, debugging, small runs | Full benchmark at scale |
| **Simplicity** | Simple Azure CLI | Complex ML SDK |
| **Control** | Full control, VNC, SSH | Managed (less visibility) |
| **Debugging** | Easy - VNC shows Windows | Harder - logs only |
| **Parallelization** | Manual (multiple VMs) | Built-in (num_workers flag) |
| **Cost** | Pay for VM only | VM + ML workspace |
| **Dependencies** | Azure CLI | Azure ML SDK, workspace, storage |

**Decision:** Use BOTH approaches for different purposes.

## VM Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  CREATE  │────▶│  SETUP   │────▶│   RUN    │────▶│  DELETE  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
     │                │                │                │
     ▼                ▼                ▼                ▼
  az vm create    docker pull      run.py           az vm delete
  ~2 min          winarena:latest  tasks            ~1 min
                  Windows boot     ~2 min/task
                  ~15 min (first)
                  ~3 min (cached)
```

### Optimization: Pre-warmed VM Pool

To avoid 15-minute first-boot time:
1. Create VMs once with Windows installed
2. **Deallocate** (stops billing, preserves disk)
3. **Start** when needed (~2 min)
4. Run tasks
5. **Deallocate** again (not delete)

```bash
# Initial setup (once)
uv run python -m openadapt_ml.benchmarks.cli create --name waa-eval-vm-1
uv run python -m openadapt_ml.benchmarks.cli create --name waa-eval-vm-2
# ... wait for Windows to install on each ...

# Before benchmark run
uv run python -m openadapt_ml.benchmarks.cli vm-start --name waa-eval-vm-1
uv run python -m openadapt_ml.benchmarks.cli vm-start --name waa-eval-vm-2

# After benchmark run (stops billing, keeps disk)
uv run python -m openadapt_ml.benchmarks.cli deallocate --name waa-eval-vm-1
uv run python -m openadapt_ml.benchmarks.cli deallocate --name waa-eval-vm-2
```

## Scaling Considerations

### Azure vCPU Quota

| VM Size | vCPUs | Max VMs (10 vCPU quota) | Max VMs (40 vCPU quota) |
|---------|-------|-------------------------|-------------------------|
| D4ds_v4 | 4 | 2 | 10 |
| D2ds_v4 | 2 | 5 | 20 |

**Current quota:** 10 vCPUs (Standard D Family)
**Recommended:** Request increase to 40+ vCPUs for 10 parallel VMs

### Cost Estimate

| Workers | VM Size | $/hr each | Total $/hr | 154 tasks @ 2min/task | Total Cost |
|---------|---------|-----------|------------|----------------------|------------|
| 1 | D4ds_v4 | $0.19 | $0.19 | 5.1 hrs | ~$1.00 |
| 5 | D4ds_v4 | $0.19 | $0.95 | 1.0 hr | ~$1.00 |
| 10 | D4ds_v4 | $0.19 | $1.90 | 0.5 hr | ~$1.00 |

**Note:** More workers = faster completion, similar total cost (dominated by compute time, not wall time).

## CLI Commands (Proposed)

```bash
# Create a pool of VMs
uv run python -m openadapt_ml.benchmarks.cli pool create --count 5

# Start all VMs in pool
uv run python -m openadapt_ml.benchmarks.cli pool start

# Run benchmark across pool
uv run python -m openadapt_ml.benchmarks.cli run --parallel --tasks 154

# Deallocate pool (stop billing)
uv run python -m openadapt_ml.benchmarks.cli pool deallocate

# Delete pool entirely
uv run python -m openadapt_ml.benchmarks.cli pool delete
```

## Implementation Plan

### Phase 1: Single Dedicated VM (DONE)
- [x] Create VM with CLI (`uv run python -m openadapt_ml.benchmarks.cli create`)
- [x] Run WAA benchmarks on single VM
- [x] VNC access for debugging
- [x] Results collection

### Phase 2: Scale with Official WAA (TODO)
- [ ] Set up Azure ML workspace (if not exists)
- [ ] Upload golden image to storage account
- [ ] Configure `scripts/run_azure.py` with our credentials
- [ ] Request vCPU quota increase (80+ for 10 workers)
- [ ] Run full 154-task benchmark with `--num_workers 10`

### Phase 3: Integration (OPTIONAL)
- [ ] Wrapper CLI to invoke official `run_azure.py`
- [ ] Results download and analysis
- [ ] Cost tracking

**Note:** We're NOT building our own VM pool management. The official WAA `run_azure.py` already does this well.

## Files

| File | Purpose |
|------|---------|
| `openadapt_ml/benchmarks/cli.py` | CLI for single dedicated VM (dev/debug) |
| `vendor/WindowsAgentArena/scripts/run_azure.py` | Official WAA parallel execution |

## Related Documents

- `docs/WAA_APPROACH_REVIEW.md` - Why vanilla WAA, not custom Dockerfile
- `CLAUDE.md` - CLI-first development guidelines
- `/Users/abrichr/oa/src/STATUS.md` - Project priorities
- [Official WAA README](https://github.com/microsoft/WindowsAgentArena/blob/main/README.md) - Azure ML setup instructions

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-29 | Use dedicated VM for dev/debug | Full control, VNC, easy iteration |
| 2026-01-29 | Use official WAA `run_azure.py` for scale | Don't reinvent the wheel |
| 2026-01-29 | Don't build custom VM pool | Official WAA already handles this |
| 2026-01-29 | ThreadPoolExecutor sufficient | Ray is overkill (agent a7d43c3 analysis) |
