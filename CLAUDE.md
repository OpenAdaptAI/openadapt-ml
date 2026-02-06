# Claude Context for openadapt-ml

## Simplicity Guidelines

**Philosophy**: "Less is more. 80/20 impact/complexity. Working code beats elegant design."

**Before writing code**: Can this be <100 lines? Does this provide 80% of value? Is this the simplest approach?

**Avoid**: Classes when functions work, abstractions before 3rd use, design docs for non-existent code.

See: `/Users/abrichr/oa/src/openadapt-evals/SIMPLICITY_PRINCIPLES.md` for full guidelines.

---

## CRITICAL RULES

### 0. CHECK RESOURCES ON SESSION START

**After context compaction or session start, check for running Azure resources:**

```bash
uv run python -m openadapt_ml.benchmarks.cli resources
```

This prevents:
- Forgetting about running VMs (costs ~$0.19-0.38/hr)
- Creating duplicate resources
- Losing track of what's deployed

See `RESOURCES.md` for current status (auto-updated by the command).

### 1. CLI-FIRST, NEVER RAW COMMANDS

**NEVER run raw commands. ALWAYS use or extend the CLI.**

```bash
# BANNED (require user permission, waste time)
ssh azureuser@IP "anything"
az vm start --name ...
az vm run-command invoke ...
uv run python -c "import subprocess; ..."

# REQUIRED (pre-approved, don't ask permission)
uv run python -m openadapt_ml.benchmarks.cli vm start
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd "command"
uv run python -m openadapt_ml.benchmarks.cli vm diag
uv run python -m openadapt_ml.benchmarks.cli vm logs
```

**If a CLI command doesn't exist**: Edit cli.py to add it, THEN use it. NEVER use raw commands as workaround.

### 2. START DASHBOARD FIRST FOR VM WORK

**Before ANY vm subcommand (probe, diag, logs, etc.):**
```bash
uv run python -m openadapt_ml.benchmarks.cli vm monitor
```

This manages:
- SSH tunnels (VNC at localhost:8006, WAA at localhost:5001)
- Real-time cost tracking
- Azure ML job visibility
- Auto-opens web dashboard

**WRONG**: Running `vm probe` then `vm diag` then telling user to run `vm monitor`
**RIGHT**: Run `vm monitor` FIRST - it handles everything

### 3. VERIFY URLs BEFORE RECOMMENDING

Always test URLs with curl before telling user to access them:
```bash
curl -s --connect-timeout 5 http://localhost:8006/ > /dev/null && echo "accessible" || echo "NOT accessible"
```

---

## Project Status

**IMPORTANT**: Check `/Users/abrichr/oa/src/STATUS.md` at session start for P0 priorities.

## Project Overview

openadapt-ml: Model-agnostic ML engine for GUI automation agents.
- Schemas for GUI trajectories
- VLM adapters (Qwen3-VL, Qwen2.5-VL, API backends)
- Supervised fine-tuning pipeline
- Runtime policy API

## Current Focus: Demo Retrieval

**Validated (Dec 2024)**: Demo-conditioned prompting improves accuracy
- Zero-shot: 33% correct first actions
- With demo: 100% correct first actions
- See `docs/experiments/demo_conditioned_prompting_results.md`

**Validated (Jan 2026)**: Demo persistence fix working in openadapt-evals
- Agent behavior: 6.8 avg steps (random) -> 3.0 avg steps (focused)
- Next: Run full WAA evaluation (154 tasks)

**Key insight**: OpenAdapt's value is trajectory-conditioned disambiguation of UI affordances.

## Benchmark Integration

**Primary**: Windows Agent Arena (WAA)
- 154 tasks across 11 Windows domains
- MIT licensed, runs locally or on Azure
- SOTA: ~19.5% success (GPT-5.1 + OmniParser)

**Future benchmarks** (not yet implemented): WebArena, OSWorld

**Code location**: Benchmark code moved to `openadapt-evals` package. openadapt-ml handles VM management only.

```python
# NEW (preferred)
from openadapt_evals import ApiAgent, WAAMockAdapter, evaluate_agent_on_benchmark

# Backward compat
from openadapt_ml.benchmarks import APIBenchmarkAgent, WAAMockAdapter
```

---

## WAA Workflow

### Two CLIs, Two Purposes

| CLI | Repo | Purpose |
|-----|------|---------|
| `openadapt_ml.benchmarks.cli` | openadapt-ml | VM lifecycle, Docker, tunnels, monitoring |
| `openadapt_evals.benchmarks.cli` | openadapt-evals | Benchmark execution, agents, results |

### API Keys

Auto-loaded from `.env` via `config.py`. No need to pass explicitly.

```bash
# .env file (not committed to git)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Complete Workflow (Pool - Recommended)

**Step 1: Create VM Pool (~10 min)**
```bash
# Single VM for quick tests
uv run python -m openadapt_ml.benchmarks.cli pool-create --workers 1

# Multiple VMs for parallel evaluation
uv run python -m openadapt_ml.benchmarks.cli pool-create --workers 3
```

**Step 2: Wait for WAA Ready (~5-15 min)**
```bash
uv run python -m openadapt_ml.benchmarks.cli pool-wait
```

**Step 3: Run Benchmark**
```bash
# Run 3 tasks for quick validation
uv run python -m openadapt_ml.benchmarks.cli pool-run --tasks 3

# Run all 154 tasks
uv run python -m openadapt_ml.benchmarks.cli pool-run --tasks 154
```

**Step 4: View Progress and VNC**
```bash
# Check status
uv run python -m openadapt_ml.benchmarks.cli pool-status

# Open VNC to view Windows desktops
uv run python -m openadapt_ml.benchmarks.cli pool-vnc

# Stream logs
uv run python -m openadapt_ml.benchmarks.cli pool-logs
```

**Step 5: Cleanup (Stop Billing)**
```bash
uv run python -m openadapt_ml.benchmarks.cli pool-cleanup
```

### CLI Commands Reference

```bash
# === POOL COMMANDS (Parallel VMs - Recommended) ===
pool-create --workers N   # Create N VMs with Docker + WAA image
pool-create --workers N --auto-shutdown-hours 6  # Custom auto-shutdown (default: 4h)
pool-wait                 # Wait for WAA server ready on all workers
pool-run --tasks N        # Run N tasks distributed across workers
pool-status               # Show status of all pool VMs
pool-vnc                  # Open VNC to pool workers (SSH tunnels)
pool-logs                 # Stream logs from all workers
pool-exec --cmd ''        # Execute command on all workers
pool-cleanup -y           # Delete all pool VMs and resources (no prompt)

# === SINGLE VM COMMANDS ===
create --fast             # Create single VM (D8ds_v5)
create --fast --auto-shutdown-hours 6  # Custom auto-shutdown (default: 4h)
delete                    # Delete VM and all resources
status                    # Show VM status
start                     # Start WAA container
stop                      # Stop WAA container
probe                     # Check if WAA server is ready
run --num-tasks N         # Run benchmark on single VM
vm-start                  # Start a deallocated VM
deallocate                # Stop VM (preserves disk, stops billing)
logs                      # Show WAA logs
vnc                       # Open VNC (SSH tunnel)
exec --cmd ''             # Run command in container
docker-exec --cmd ''      # Run command on VM host

# === AZURE ML COMMANDS (Legacy) ===
run-azure-ml --workers N  # Run on Azure ML compute instances
azure-ml-quota            # Check quota status
azure-ml-quota-wait       # Wait for quota approval
```

### Quota Auto-Detection

Wait for quota approval before running evaluation:

```bash
# Wait for quota (polls every 60 seconds, 24h timeout)
uv run python -m openadapt_ml.benchmarks.cli azure-ml-quota-wait

# Wait and automatically run evaluation when quota is approved
uv run python -m openadapt_ml.benchmarks.cli azure-ml-quota-wait --auto-run --tasks 20

# Custom target (e.g., 16 vCPUs for 2 parallel workers)
uv run python -m openadapt_ml.benchmarks.cli azure-ml-quota-wait --target 16

# Run in background (survives terminal close)
nohup uv run python -m openadapt_ml.benchmarks.cli azure-ml-quota-wait --auto-run &
```

See `docs/QUOTA_AUTO_DETECTION_DESIGN.md` for full documentation.

### VM Auto-Shutdown and Orphan Prevention

**Auto-shutdown policy**: All VMs are automatically configured with an Azure auto-shutdown policy as a safety net to prevent orphaned VMs from running indefinitely and consuming quota/money.

- **Default**: 4 hours after VM creation
- **Customizable**: `--auto-shutdown-hours N` (0 to disable)
- **Azure-level enforcement**: Even if SSH connection drops, the VM will still be deallocated

```bash
# Default: auto-shutdown in 4 hours
uv run python -m openadapt_ml.benchmarks.cli pool-create --workers 3

# Custom: auto-shutdown in 8 hours for long-running evaluations
uv run python -m openadapt_ml.benchmarks.cli pool-create --workers 3 --auto-shutdown-hours 8

# Disable auto-shutdown (not recommended)
uv run python -m openadapt_ml.benchmarks.cli pool-create --workers 3 --auto-shutdown-hours 0
```

**Test VM cleanup**: During `pool-create`, a test VM is created to check quota availability. This test VM is always cleaned up via try/finally, even if the command is interrupted or fails.

**Manual cleanup**: Use `pool-cleanup -y` to clean up orphaned resources without confirmation prompts (useful for automation):
```bash
uv run python -m openadapt_ml.benchmarks.cli pool-cleanup -y
```

### Azure ML Automated Workflow

For parallel benchmark execution on Azure ML compute instances:

```bash
# Single command handles everything:
# 1. Create/start VM if needed
# 2. Start Windows container with VERSION=11e
# 3. Wait for WAA server ready (~15-20 min first time)
# 4. Upload golden image to blob storage
# 5. Run Azure ML benchmark with N workers

uv run python -m openadapt_ml.benchmarks.cli run-azure-ml-auto --workers 4

# Setup only (golden image, no benchmark)
uv run python -m openadapt_ml.benchmarks.cli run-azure-ml-auto --skip-benchmark

# Cleanup when done (IMPORTANT - stops billing!)
uv run python -m openadapt_ml.benchmarks.cli run-azure-ml --teardown --confirm
```

See `docs/AZURE_ML_AUTOMATED_WORKFLOW.md` for full documentation.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL MACHINE                             │
│  openadapt-ml CLI         openadapt-evals CLI               │
│  (VM management)          (benchmark execution)              │
│       │                        │                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          SSH TUNNELS (auto-managed by monitor)          │ │
│  │  localhost:5001 → VM:5000 (WAA API)                    │ │
│  │  localhost:8006 → VM:8006 (noVNC)                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                        │ SSH (port 22)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    AZURE VM (Ubuntu)                         │
│  Docker                                                      │
│  └── windowsarena/winarena:latest (Microsoft official)      │
│       └── QEMU (Windows 11 Enterprise)                      │
│            ├── WAA Flask server (port 5000)                 │
│            └── Navi agent (executes tasks)                  │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
1. SSH tunnels required - Azure NSG blocks direct port access
2. WAA server runs INSIDE Windows, not on Ubuntu host
3. Default tunnel port is 5001 (not 5000)
4. Uses vanilla Microsoft WAA image, no custom Dockerfile
5. `VERSION=11e` auto-downloads Windows 11 Enterprise Evaluation

---

## VM Configuration Changes

Delete + recreate (don't try to resize running VMs):
```bash
uv run python -m openadapt_ml.benchmarks.cli vm delete -y
# Update cli.py defaults
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa
```

**Current defaults** (in cli.py):
- Size: `Standard_D8ds_v5` (8 vCPU, 32GB RAM, 300GB temp on /mnt)
- Location: `eastus`
- OS: Ubuntu 22.04 LTS

---

## Key Architecture Decisions

1. **SoM mode** - Element IDs (`CLICK([1])`) instead of coordinates for 100% accuracy on synthetic benchmarks

2. **Grounding module** - Keep but deprioritize. Useful for real UIs without SoM overlays. Located in `openadapt_ml/grounding/`

3. **Schema design** - Actions carry both coordinates AND element grounding when available

4. **Lossless preservation** - Store raw benchmark configs in `raw_config`, `raw_observation`, `raw_action` fields

5. **Schema purity** - Domain-agnostic; external systems adapt TO the schema, not vice versa. See `openadapt_ml/schemas/`

6. **Cloud-first** - Offload heavy compute to cloud GPUs (Azure, Lambda Labs). Everything should feel fast.

7. **Stub training** - Use `--stub` flag for rapid UI iteration without GPU

8. **DOM/AX mandatory in schema** - For evaluator compatibility (WebArena, Mind2Web need DOM), even if agents use vision-only

---

## Azure Automation

`scripts/setup_azure.py` automates 15-step Azure setup:
- Creates resource group, service principal, ML workspace, ACR
- Imports WAA Docker image to ACR
- Configures ACR authentication (AcrPull role)
- Writes credentials to `.env`

```bash
python scripts/setup_azure.py        # Setup
python scripts/setup_azure.py --cleanup  # Cleanup
```

---

## Cloud GPU Training

See `docs/cloud_gpu_training.md` for full documentation.

```bash
# Lambda Labs - automated pipeline
uv run python -m openadapt_ml.cloud.lambda_labs train --capture /path --goal "Task"

# Step by step
uv run python -m openadapt_ml.cloud.lambda_labs launch --type gpu_1x_a10
uv run python -m openadapt_ml.cloud.lambda_labs train-status
uv run python -m openadapt_ml.cloud.lambda_labs terminate <id>
```

---

## Training Commands

```bash
# Train on capture
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_capture.yaml \
  --capture /path/to/capture \
  --open

# Serve dashboard (auto-regenerates HTML)
uv run python -m openadapt_ml.cloud.local serve --port 8080 --open

# Regenerate viewer without serving
uv run python -m openadapt_ml.cloud.local viewer

# Compare human vs model
uv run python -m openadapt_ml.scripts.compare \
  --capture /path/to/capture \
  --checkpoint checkpoints/model \
  --open
```

---

## Code Patterns

### Environment Variables
Use `config.settings`, NOT `os.environ`:
```python
# Good
from openadapt_ml.config import settings
api_key = settings.openai_api_key

# Bad
api_key = os.environ.get("OPENAI_API_KEY")
```

When adding new env vars:
1. Add to `Settings` class in `config.py`
2. Add to `.env.example`

### API Keys for CLI
Priority: `--api-key` flag > `.env` file > environment variable

---

## Dockerfile Testing

Test fixes INSIDE container before rebuilding (saves 30+ min):

```bash
# 1. Start test container
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd \
  'docker run -d --name test-fix --entrypoint /bin/bash windowsarena/winarena:latest -c "sleep 3600"'

# 2. Apply fix
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd \
  "docker exec test-fix sed -i 's/old/new/' /some/file.sh"

# 3. Verify
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd \
  "docker exec test-fix cat /some/file.sh"

# 4. Cleanup
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd 'docker rm -f test-fix'

# 5. ONLY rebuild after fix is verified
```

---

## Files to Know

- `docs/WAA_APPROACH_REVIEW.md` - Full WAA setup documentation
- `docs/cloud_gpu_training.md` - Lambda Labs/Azure training guide
- `docs/azure_waa_setup.md` - Azure quota, costs, troubleshooting
- `docs/design.md` - System design
- `openadapt_ml/benchmarks/cli.py` - VM CLI commands
- `openadapt_ml/cloud/ssh_tunnel.py` - SSH tunnel manager
- `openadapt_ml/config.py` - Settings (pydantic-settings)
- `openadapt_ml/schemas/` - Canonical schema definitions

---

## Git Commit Style (Angular)

```
<type>(<scope>): <subject>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Types**: feat, fix, docs, style, refactor, perf, test, chore, ci

**Rules**: Imperative mood, no period, max 50 chars, lowercase after type

---

## Don't Do

- Don't use `os.environ` - use `config.settings`
- Don't use `pip install` - use `uv add` or `uv sync`
- Don't run VM ops without `vm monitor` first
- Don't use raw SSH/shell commands - use CLI
- Don't tell user to run commands - YOU run them
- Don't use broad pkill patterns (they kill unrelated apps)
- Don't add timelines/estimates to plans
- Don't mention specific clients by name

---

## Safe Process Management

```bash
# WRONG (kills unrelated apps)
pkill -f "openadapt"
pkill -f "python"

# RIGHT (specific)
kill $(lsof -t -i :8765) 2>/dev/null
pkill -f "python.*-m openadapt_ml.cloud.local serve"

# Check before killing
pgrep -f "pattern" -l
```

---

## File Access

Pre-approved read access to `~/oa/src/` (related projects like openadapt-capture).

## Current Capture

Path: `/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift`
Task: Turn off Night Shift in macOS System Settings

---

## TODO / Known Issues

### Benchmark Viewer - Phase 4
**Status**: TODO

Add failure clustering and regression detection. Phases 1-3 done:
- Data collection with ExecutionTraceCollector
- Viewer generation with `view --run-name {name}`
- UI with summary, task list, step replay, playback controls

### Azure ML Experiment ID
**Status**: TODO

Retrieve experiment_id dynamically instead of hardcoded UUID.

### Azure ML Port 80 Conflict
**Status**: INVESTIGATING

Azure ML compute instances have Microsoft infrastructure services on port 80. When vanilla WAA's dockur/windows container starts, nginx tries to bind to port 80 and fails:
```
nginx: [emerg] bind() to 0.0.0.0:80 failed (98: Address already in use)
```

**Key insight**: Port 80 is just nginx redirecting to noVNC on port 8006. **NOT essential for WAA**.
- Port 5000: WAA Flask API (benchmark execution) - ESSENTIAL
- Port 8006: noVNC (browser VNC) - ESSENTIAL
- Port 80: nginx redirect - NOT ESSENTIAL

**What we're testing**:
1. `WEB=N` env var to disable nginx entirely
2. SSH tunnel to access ports 8006 and 5000 for debugging
3. Enhanced diagnostics in run_entry.py to verify Windows boots despite nginx failure

**SSH key support added**: Compute instances now use your local SSH key (~/.ssh/id_rsa) for direct SSH access.

See `docs/AZURE_ML_PORT_80_FIX.md` for full analysis and options.

### Azure ML CLI Commands

```bash
# Status and monitoring
azure-ml-status           # Show compute instances and recent jobs
azure-ml-logs --job NAME  # Stream logs from running job
azure-ml-monitor          # Interactive monitor with VNC tunnel

# Run benchmarks
run-azure-ml-auto --workers N  # Fully automated workflow

# Cleanup (IMPORTANT - stop billing!)
azure-ml-cancel           # Cancel running job (or --job NAME)
azure-ml-delete-compute   # Delete compute instance (--name NAME or --all)
azure-ml-cleanup --yes    # Cancel all jobs + delete all instances

# Resource management
resources                 # Show all Azure resources and costs
```

---

## Troubleshooting

### Dashboard/Viewer Stale Data
After code changes:
1. Regenerate: `uv run python -m openadapt_ml.cloud.local viewer`
2. Hard-refresh browser: Cmd+Shift+R

### WAA Connection Issues
1. Is VM running? `vm status`
2. Are tunnels active? `vm monitor`
3. Check container: `vm diag`

### Windows Not Booting
1. Check VNC via `vm monitor`
2. Check logs: `vm logs`

### Common Issues Table

| Symptom | Fix |
|---------|-----|
| Connection refused localhost:5001 | Run `vm monitor` to start tunnels |
| Windows not booting | Check VNC, check `vm logs` |
| Elapsed time shows 0 | Check training_log.json has elapsed_time |
| No comparison screenshots | Update capture_path in training_log.json |
| Stale data after code change | Hard refresh (Cmd+Shift+R) |

See `docs/` for detailed troubleshooting guides.
