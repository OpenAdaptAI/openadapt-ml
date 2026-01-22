# Claude Context for openadapt-ml

## Simplicity Guidelines

**Philosophy**: "Less is more. 80/20 impact/complexity. Working code beats elegant design."

**Before writing code, ask**:
1. Can this be <100 lines? (ideally <50)
2. Does this provide 80% of value?
3. Is this the simplest approach?

**Red flags to avoid**:
- Classes when functions work
- Abstractions before 3rd use
- Design docs for non-existent code
- Multiple implementations of same thing

**See**: `/Users/abrichr/oa/src/openadapt-evals/SIMPLICITY_PRINCIPLES.md` for full guidelines.

---

## 🚨🚨🚨 CRITICAL: CLI-FIRST, NEVER RAW COMMANDS 🚨🚨🚨

### THIS IS THE #1 RULE. VIOLATIONS FRUSTRATE THE USER.

**NEVER run commands that require user permission. ALWAYS use or extend the CLI.**

❌ **BANNED** (these require permission, waste user's time):
```bash
# Raw Azure CLI
az vm start --name ...
az vm run-command invoke ...

# Raw SSH
ssh azureuser@IP "command"

# Raw Python one-liners
uv run python -c "import subprocess; ..."

# Any command not in the pre-approved CLI
```

✅ **REQUIRED** (these are pre-approved, don't ask permission):
```bash
# ALL VM operations go through the CLI
uv run python -m openadapt_ml.benchmarks.cli vm start
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd "command"
uv run python -m openadapt_ml.benchmarks.cli vm diag
uv run python -m openadapt_ml.benchmarks.cli vm logs
```

### When Functionality Is Missing

**If a CLI command doesn't exist for what you need:**
1. **EDIT the CLI** to add the new command/action
2. **THEN call the CLI** command you just added
3. **NEVER use raw commands** as a workaround

**Example**: Need to restart Docker services?
```python
# 1. Add to cli.py under cmd_vm():
elif action == "fix-docker":
    # Restart containerd and docker
    commands = [
        "sudo systemctl restart containerd",
        "sudo systemctl restart docker",
        "docker ps"
    ]
    for cmd in commands:
        run_on_vm(cmd)

# 2. Then call it:
uv run python -m openadapt_ml.benchmarks.cli vm fix-docker
```

**This rule exists because:**
- Raw commands require user approval every time
- CLI commands are pre-approved and don't interrupt workflow
- CLI commands are documented and reusable
- The user has told you this MANY times - LISTEN

---

## 🔄 STANDARD WORKFLOW: VM Configuration Changes

**When VM config needs to change (disk size, VM size, etc.):**

1. **Delete the current VM** (if running):
   ```bash
   uv run python -m openadapt_ml.benchmarks.cli vm delete -y
   ```

2. **Update the code** that launches the VM (e.g., `cli.py` defaults)

3. **Launch new VM** with the updated code:
   ```bash
   uv run python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key $OPENAI_API_KEY
   ```

**DO NOT** try to resize/modify running VMs. It's simpler and faster to delete + recreate.

**Current VM defaults** (in `cli.py`):
- Size: `Standard_D8ds_v5` (300GB temp storage on /mnt)
- Location: `eastus`
- OS: Ubuntu 22.04 LTS

---

## Project Status & Priorities

**IMPORTANT**: Before starting work, always check the project-wide status document:
- **Location**: `/Users/abrichr/oa/src/STATUS.md`
- **Purpose**: Tracks P0 priorities, active background tasks, blockers, and strategic decisions
- **Action**: Read this file at the start of every session to understand current priorities

This ensures continuity between Claude Code sessions and context compactions.

---

This file helps maintain context across sessions.

---
## ⚠️⚠️⚠️ MANDATORY: START DASHBOARD FIRST ⚠️⚠️⚠️

### STOP. READ THIS BEFORE DOING ANYTHING.

**If ANY of these are true, you MUST run the dashboard command IMMEDIATELY:**
- Session just started or was compacted
- User mentions VMs, Azure, WAA, benchmark, or Windows
- You're about to run ANY `vm` subcommand (probe, diag, logs, run-waa, etc.)
- You want to check benchmark status

**THE COMMAND (run this FIRST, not after other commands):**
```bash
uv run python -m openadapt_ml.benchmarks.cli vm monitor
```

**ENHANCED FEATURES (as of Jan 2026):**
The `vm monitor` command now provides comprehensive VM usage visibility:
- **VM Status**: Real-time VM state, size, and IP
- **Activity Detection**: What the VM is currently doing (idle, benchmark running, setup)
- **Cost Tracking**: Current uptime, hourly rate, and total cost for session
- **Azure ML Jobs**: Recent jobs from last 7 days with status
- **Evaluation History**: Past benchmark runs and success rates (with --details flag)
- **Dashboard & Tunnels**: Auto-starts web dashboard and SSH/VNC tunnels

**Usage:**
```bash
# Basic monitoring
uv run python -m openadapt_ml.benchmarks.cli vm monitor

# With detailed information (costs per day/week, evaluation history)
uv run python -m openadapt_ml.benchmarks.cli vm monitor --details

# With auto-shutdown after 2 hours
uv run python -m openadapt_ml.benchmarks.cli vm monitor --auto-shutdown-hours 2
```

**WHY THIS MATTERS:**
- VNC is ONLY accessible via SSH tunnel at `localhost:8006` (NOT the public IP)
- The dashboard auto-manages SSH tunnels
- Shows real-time costs to prevent budget overruns
- Tracks all Azure ML jobs for visibility into what's running
- Without it, you cannot see what Windows is doing
- The user WILL be frustrated if you keep forgetting this

**WRONG (what you keep doing):**
```bash
# DON'T do this - checking probe/diag/logs WITHOUT dashboard running
uv run python -m openadapt_ml.benchmarks.cli vm probe
uv run python -m openadapt_ml.benchmarks.cli vm diag
# Then telling user to "run vm monitor" - NO! YOU run it FIRST!
```

**RIGHT (what you should do):**
```bash
# ALWAYS start dashboard FIRST, then it handles everything
uv run python -m openadapt_ml.benchmarks.cli vm monitor
```

**After every /compact or session restart, your LITERAL FIRST ACTION must be starting this dashboard if VMs are involved.**

---
## 🔴 MANDATORY: VERIFY URLs BEFORE RECOMMENDING 🔴

**BEFORE telling the user to access ANY URL (localhost:XXXX, VNC, dashboard, etc.):**

1. **MANUALLY VERIFY** the URL is accessible by running a curl/check command
2. **NEVER assume** a service is running just because it was started earlier
3. **NEVER recommend** a URL based on documentation alone - ALWAYS test first

**Example verification:**
```bash
# ALWAYS do this BEFORE telling user to visit localhost:8006
curl -s --connect-timeout 5 http://localhost:8006/ > /dev/null && echo "VNC accessible" || echo "VNC NOT accessible"
```

**If verification fails:**
- Do NOT tell user to access the URL
- Diagnose why it's not working
- Fix it first, THEN provide the URL

**This rule exists because:** The user was told to access localhost:8006 when the container was gone. This is unacceptable.

---
## 🚨🚨🚨 STOP! READ THIS BEFORE EVERY COMMAND 🚨🚨🚨

### ABSOLUTELY NEVER USE RAW SSH COMMANDS

**This is the #1 rule. You have been told this MANY times. STOP IGNORING IT.**

❌ **BANNED** (never type these):
- `ssh azureuser@IP "anything"`
- `ssh $SSH_OPTS ...`
- Any command starting with `ssh` to the VM

✅ **REQUIRED** (always use these instead):
- `uv run python -m openadapt_ml.benchmarks.cli vm exec --cmd "your command"`
- `uv run python -m openadapt_ml.benchmarks.cli vm diag`
- `uv run python -m openadapt_ml.benchmarks.cli vm logs`

**If a CLI command doesn't exist, ADD IT TO THE CLI FIRST, then use it.**

**Before running ANY command involving the VM, ask yourself:**
1. Does this start with `ssh`? → STOP, use CLI instead
2. Is this a raw shell command to the VM? → STOP, use CLI instead
3. Can I use `vm exec --cmd`? → YES, use it

This has been explained to you repeatedly. FOLLOW IT.

---
## 🔧 DOCKERFILE/VM CHANGES: TEST INSIDE CONTAINER FIRST

**Problem**: Each Dockerfile change triggers: rebuild (10 min) → Windows boot (15 min) → test → repeat. Hours wasted on tiny changes.

**Solution**: Test fixes INSIDE a running container BEFORE rebuilding:

```bash
# 1. Start a test container with bash entrypoint (seconds)
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd \
  'docker run -d --name test-fix --entrypoint /bin/bash waa-auto:latest -c "sleep 3600"'

# 2. Apply your fix manually INSIDE the container (seconds)
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd \
  "docker exec test-fix sed -i 's/old/new/' /some/file.sh"

# 3. Verify the fix works (seconds)
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd \
  "docker exec test-fix cat /some/file.sh"

# 4. Test the actual behavior (seconds)
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd \
  "docker exec test-fix /some/script.sh && ls /expected/output"

# 5. Cleanup
uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd 'docker rm -f test-fix'

# 6. ONLY AFTER fix is verified: Update Dockerfile and rebuild ONCE
```

**Why this matters**:
- Testing a fix takes SECONDS instead of 30+ minutes
- Iterate 10x on the fix before committing to a rebuild
- Don't lose context waiting for long builds
- Each rebuild should be the LAST rebuild, not a guess

---

## Project Overview

openadapt-ml is a model-agnostic, domain-agnostic ML engine for GUI automation agents. It provides:
- Schemas for GUI interaction trajectories
- Synthetic UI generation for bootstrapping
- VLM adapters (Qwen3-VL, Qwen2.5-VL, API backends)
- Supervised fine-tuning pipeline
- Runtime policy API

## Current Focus: Demo Retrieval

**Validated**: Demo-conditioned prompting improves action accuracy (Dec 2024)
- Zero-shot: 33% correct first actions
- With demo: 100% correct first actions
- See `docs/experiments/demo_conditioned_prompting_results.md`

**✅ VALIDATED (Jan 17, 2026)**: Demo persistence fix is working
- The P0 fix in `openadapt-evals` ensures demo is included at EVERY step, not just step 1
- Mock test confirms: agent behavior changes from 6.8 avg steps (random) to 3.0 avg steps (focused)
- See `openadapt-evals/CLAUDE.md` for full validation details
- **Next step**: Run full WAA evaluation (154 tasks) to measure episode success improvement

**Next step**: Build demo retrieval to automatically select relevant demos from a library.

**Key insight**: OpenAdapt's value is **trajectory-conditioned disambiguation of UI affordances**, not "better reasoning".

## Benchmark Integration

**Primary benchmark**: Windows Agent Arena (WAA)
- 154 tasks across 11 Windows domains
- MIT licensed, can run locally or on Azure
- SOTA: ~19.5% success (GPT-5.1 + OmniParser)

**Future benchmarks** (not yet implemented):
- WebArena/VisualWebArena (browser)
- OSWorld (cross-platform desktop)

## Key Architecture Decisions

1. **SoM (Set-of-Marks) mode** - Achieves 100% on synthetic benchmarks by using element IDs instead of coordinates (`CLICK([1])` not `CLICK(x=0.42, y=0.31)`)

2. **Grounding module** - Keep but deprioritize. Useful for deployment on real UIs without SoM overlays. Located in `openadapt_ml/grounding/`

3. **Schema design** - Actions should carry both coordinates AND element grounding (node_id, role, name, bbox) when available

4. **Lossless preservation** - Always store raw benchmark configs verbatim in `raw_config`, `raw_observation`, `raw_action` fields

5. **DOM/AX is mandatory in schema, optional at runtime** - Observations must support `accessibility_tree` and `dom_html` fields for evaluator compatibility (WebArena, WorkArena, Mind2Web need DOM for scoring), even if agents choose vision-only

6. **Cloud-First Development** - While features should work locally for testing, immediately build out cloud compatibility (Azure free tier, Lambda Labs) because:
   - Most users won't have 96GB RAM locally for VLM training
   - Developer productivity suffers waiting for long training runs
   - Training should be as short as possible with feedback as quickly as possible
   - **Everything should feel fast** - offload heavy compute to cloud GPUs
   - Cloud providers: Azure (primary, free tier available), Lambda Labs (GPU rental)
   - See `docs/live_inference_design.md` for async inference architecture

7. **Schema Purity** - The schema must remain domain-agnostic and generic:
   - **External systems adapt TO the schema**, not the other way around
   - Never add fields to accommodate specific external data structures
   - Data transformation belongs in importers/exporters, not core schema
   - Use `raw` and `metadata` dict fields for integration-specific data
   - If a proposed field feels specific to one use case, it doesn't belong in the schema
   - This is a standard open-source library: users import and call functions, they don't shape the API
   - See `openadapt_ml/schemas/` for canonical definitions

8. **Stub Training Adapter (HIGH PRIORITY)** - Always implement stub/mock providers first:
   - **Never wait on real training to test UI/code changes**
   - Use `--stub` flag to simulate training progress without GPU
   - Generates fake loss curves, evaluations, checkpoints in seconds
   - Enables rapid iteration on dashboard, viewer, stop button, etc.
   - See `docs/stub_training_adapter.md` for implementation details
   - Usage: `uv run python -m openadapt_ml.cloud.lambda_labs monitor --stub --open`

## Expert Feedback

1. **Prompting first** - Establish baselines with off-the-shelf models before fine-tuning
2. **Prompt engineering matters** - Use structured format: Observation summary → Planning → Possible actions → Action
3. **Element-based actions** - `Click [8]` instead of coordinates, similar to SoM
4. **Larger base models** - They used Gemma3 27B; current 2B/8B might be too small

## Benchmark Integration (MIGRATED TO openadapt-evals)

> **IMPORTANT**: Benchmark code has been consolidated into the `openadapt-evals` package.
> The `openadapt_ml/benchmarks/` directory now contains deprecation stubs that re-export from `openadapt-evals`.
>
> **Use the new package:**
> ```python
> # NEW (preferred)
> from openadapt_evals import ApiAgent, WAAMockAdapter, evaluate_agent_on_benchmark
>
> # Also works (backward compat)
> from openadapt_ml.benchmarks import APIBenchmarkAgent, WAAMockAdapter
> ```
>
> **CLI (now in openadapt-evals):**
> ```bash
> # NEW (preferred)
> uv run python -m openadapt_evals.benchmarks.cli mock --tasks 10
> uv run python -m openadapt_evals.benchmarks.cli live --agent api-claude --server http://vm:5000
>
> # openadapt-ml CLI still works for VM management
> uv run python -m openadapt_ml.benchmarks.cli vm monitor
> ```

The benchmark integration module is now in `openadapt-evals`:
- `openadapt_evals/adapters/` - BenchmarkAdapter, WAAAdapter, WAALiveAdapter
- `openadapt_evals/agents/` - BenchmarkAgent, ApiAgent (with P0 demo persistence fix), PolicyAgent
- `openadapt_evals/benchmarks/` - runner, metrics, viewer, data_collection

### APIBenchmarkAgent

The `APIBenchmarkAgent` wraps hosted VLM APIs (Claude, GPT-5.1) for benchmark evaluation baselines.
This enables comparing fine-tuned models against off-the-shelf VLMs.

```python
from openadapt_ml.benchmarks import APIBenchmarkAgent, evaluate_agent_on_benchmark

# Claude baseline
agent = APIBenchmarkAgent(provider="anthropic")
results = evaluate_agent_on_benchmark(agent, adapter)

# GPT-5.1 baseline
agent = APIBenchmarkAgent(provider="openai")
results = evaluate_agent_on_benchmark(agent, adapter)
```

CLI usage:
```bash
# Run Claude evaluation on mock tasks
uv run python -m openadapt_ml.benchmarks.cli run-api --provider anthropic --tasks 5

# Run GPT-5.1 evaluation
uv run python -m openadapt_ml.benchmarks.cli run-api --provider openai --tasks 5

# Disable accessibility tree in prompts
uv run python -m openadapt_ml.benchmarks.cli run-api --no-a11y --tasks 5
```

The agent:
- Converts BenchmarkObservation to API format (screenshot + structured prompt)
- Parses VLM responses into BenchmarkActions using regex patterns
- Supports CLICK(x,y), CLICK([id]), TYPE("text"), KEY(key), SCROLL(dir), DONE()
- Stores raw VLM responses in `action.raw_action` for debugging

### Azure Automation

`scripts/setup_azure.py` fully automates Azure setup with 15 steps:
1. Check Azure CLI installation
2. Login to Azure
3. Select subscription
4. Register resource providers (Compute, ML, Storage, ContainerRegistry)
5. Create resource group
6. Create service principal with Contributor role
7. Create ML workspace
8. Create Azure Container Registry (ACR)
9. Import WAA Docker image from Docker Hub to ACR
10. Attach ACR to ML workspace
11. Grant AcrPull role to workspace managed identity
12. Sync workspace keys for ACR authentication
13. Request GPU quota
14. Create storage account
15. Create inference queue and blob containers

The script writes all credentials to `.env` including:
- Service principal credentials (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
- Workspace config (AZURE_SUBSCRIPTION_ID, AZURE_ML_RESOURCE_GROUP, AZURE_ML_WORKSPACE_NAME)
- Docker image path (AZURE_DOCKER_IMAGE) pointing to ACR

**Why ACR?** Azure ML cannot pull from Docker Hub or ghcr.io directly. The image must be in ACR.

**ACR Authentication**: The script automatically configures ACR authentication by granting the workspace's managed identity AcrPull role on the ACR. This ensures compute instances can pull Docker images without requiring admin credentials.

CLI usage:
```bash
# Set up Azure (creates resources, ACR, imports image, writes credentials to .env)
python scripts/setup_azure.py

# Clean up all Azure resources
python scripts/setup_azure.py --cleanup

# Estimate Azure costs
python -m openadapt_ml.benchmarks.cli estimate --workers 40

# Test with mock adapter (no Windows required)
python -m openadapt_ml.benchmarks.cli test-mock --tasks 20

# Check Azure status
python -m openadapt_ml.benchmarks.cli status

# Run on Azure (WAA submodule auto-detected)
python -m openadapt_ml.benchmarks.cli run-azure --workers 1
```

Schema extensions completed in `openadapt_ml/schemas/sessions.py`:
- `Action`: `target_node_id`, `target_role`, `target_name`, `answer`, `key`, `modifiers`, `scroll_direction`, `scroll_amount`, `end_x`, `end_y`
- `Observation`: `accessibility_tree`, `dom_html`, `url`, `window_title`, `app_name`, `focused_element`

## Cloud GPU Training

See `docs/cloud_gpu_training.md` for full documentation.

**Quick start:**
```bash
# Lambda Labs - fully automated training pipeline
uv run python -m openadapt_ml.cloud.lambda_labs train \
  --capture /path/to/capture \
  --goal "Task description"

# Or step by step:
uv run python -m openadapt_ml.cloud.lambda_labs launch --type gpu_1x_a10
uv run python -m openadapt_ml.cloud.lambda_labs train-status
uv run python -m openadapt_ml.cloud.lambda_labs terminate <id>
```

**Important**: All cloud operations should be wrapped in CLI commands, not raw SSH. The Lambda Labs module provides:
- `LambdaLabsClient.setup_instance()` - Clone repo, install deps
- `LambdaLabsClient.upload_capture()` - rsync capture data
- `LambdaLabsClient.run_training()` - Execute training
- `LambdaLabsClient.get_training_status()` - Poll training progress

## Training & Visualization Commands

```bash
# Train on a capture recording
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_capture.yaml \
  --capture /path/to/capture \
  --open  # opens dashboard in browser

# Serve dashboard/viewer via HTTP (RECOMMENDED)
# Auto-regenerates dashboard.html and viewer.html before serving
uv run python -m openadapt_ml.cloud.local serve --port 8080 --open

# Skip regeneration if files are already up to date
uv run python -m openadapt_ml.cloud.local serve --port 8080 --open --no-regenerate

# Regenerate viewer/dashboard without serving
# Useful after training completes or to refresh with latest code changes
uv run python -m openadapt_ml.cloud.local viewer

# Compare human vs model predictions
uv run python -m openadapt_ml.scripts.compare \
  --capture /path/to/capture \
  --checkpoint checkpoints/model \
  --open
```

## Benchmark Data Collection & Testing

```bash
# Test benchmark data collection (Phase 1)
# Creates directory structure with screenshots, execution traces, and metadata
uv run python -m openadapt_ml.benchmarks.cli test-collection --tasks 5

# Custom run name and output directory
uv run python -m openadapt_ml.benchmarks.cli test-collection \
  --tasks 10 \
  --run-name my_test_run \
  --output benchmark_results \
  --model-id "my-agent-v1"

# Run the standalone test script (equivalent to test-collection)
uv run python test_data_collection.py
```

**Output directory structure:**
```
benchmark_results/
├── {run_name}/
│   ├── metadata.json        # Benchmark name, model ID, timestamp
│   ├── summary.json         # Aggregate metrics (success rate, avg steps)
│   └── tasks/
│       ├── task_001/
│       │   ├── task.json       # Task definition
│       │   ├── execution.json  # Execution trace with steps
│       │   └── screenshots/
│       │       ├── step_000.png
│       │       ├── step_001.png
│       │       └── ...
│       └── task_002/
│           └── ...
```

**Key files:**
- `execution.json`: Contains step-by-step trace with actions, reasoning, timestamps
- `task.json`: Task definition with instruction, domain, time limits
- `summary.json`: High-level metrics suitable for benchmark viewer
- `screenshots/`: PNG screenshots at each step

## Viewer Setup Troubleshooting

**Problem**: Viewer shows "No model loaded" after training.

**Root cause**: The viewer requires:
1. A base `comparison.html` file (from capture or generated during training)
2. Prediction JSON files (`predictions_*.json`)

**Solution**:
```bash
# If comparison.html is missing, copy from the capture directory:
cp /path/to/capture/comparison.html training_output/

# Then regenerate the viewer:
uv run python -m openadapt_ml.cloud.local viewer

# Serve and open:
uv run python -m openadapt_ml.cloud.local serve --open
```

**Key files in training_output/**:
- `training_log.json` - Training progress, loss curves, evaluations
- `dashboard.html` - Training dashboard (auto-regenerated by serve command)
- `viewer.html` - Capture viewer with predictions (auto-regenerated by serve command)
- `comparison.html` - Base viewer from capture (needed for viewer generation)
- `predictions_*.json` - Model predictions by checkpoint (e.g., `predictions_epoch3.json`)

## Files to Know

- `docs/cloud_gpu_training.md` - Lambda Labs and Azure GPU training guide
- `docs/benchmark_integration_plan.md` - Benchmark integration architecture
- `docs/azure_waa_setup.md` - Azure WAA setup guide (quota increase, costs, troubleshooting)
- `docs/design.md` - Overall system design
- `docs/experiments/demo_conditioned_prompting_results.md` - Demo experiment results (validated Dec 2024)
- `openadapt_ml/cloud/` - Cloud GPU providers (Lambda Labs, Azure)
- `openadapt_ml/benchmarks/` - Benchmark integration module (WAA, base classes)
- `openadapt_ml/experiments/demo_prompt/` - Demo-conditioned prompting experiment
- `openadapt_ml/grounding/` - Grounding module (GeminiGrounder, etc.)
- `openadapt_ml/ingest/capture.py` - Converts openadapt-capture recordings to Episodes
- `scripts/run_demo_experiment.py` - Run demo-conditioned experiment
- `configs/qwen3vl_synthetic_som.yaml` - SoM training config

## Code Patterns

### Environment Variables
Always load env vars through `openadapt_ml/config.py` using pydantic-settings, NOT directly from `os.environ`:

```python
# Good
from openadapt_ml.config import settings
api_key = settings.lambda_api_key

# Bad
api_key = os.environ.get("LAMBDA_API_KEY")
```

This ensures `.env` file is automatically loaded. When adding new env vars:
1. Add to `Settings` class in `config.py`
2. Add to `.env.example` with documentation

## File Access

The user has pre-approved read access to:
- `~/oa/src/` - Parent directory containing related projects (openadapt-capture, etc.)

Related paths:
- Capture recordings: `/Users/abrichr/oa/src/openadapt-capture/`
- Screenshots: `/Users/abrichr/oa/src/openadapt-capture/<capture-name>/screenshots/`

## Shared Dashboard Components

The training dashboard and capture viewer share UI components for visual consistency. When modifying dashboard UI:

**Key files:**
- `openadapt_ml/training/trainer.py` - Contains shared component functions:
  - `_get_shared_header_css()` - CSS for the unified header
  - `_generate_shared_header_html()` - HTML generator for nav tabs + controls

**Pattern:**
1. Define shared CSS/HTML in dedicated functions (prefixed with `_`)
2. Both `generate_training_dashboard()` and `_enhance_comparison_to_unified_viewer()` call these functions
3. Changes to shared functions automatically propagate to all dashboards

**Why this matters:**
- Prevents visual inconsistencies when switching between Training and Viewer tabs
- Single source of truth for styling (no duplicate CSS to maintain)
- Easier to add new dashboards that match existing style

## CRITICAL: Always Start Dashboard When Running Azure Resources

See the ⚠️ MANDATORY section at the TOP of this file. Use:
```bash
uv run python -m openadapt_ml.benchmarks.cli vm monitor
```

## ⚠️ SAFE PROCESS MANAGEMENT ⚠️

**NEVER use broad pkill patterns** - they can kill unrelated applications!

**WRONG (DANGEROUS):**
```bash
# These patterns are TOO BROAD and will kill unrelated apps:
pkill -f "openadapt"      # Kills anything with "openadapt" in path
pkill -f "python"         # Kills ALL Python processes
pkill -9 -f "openadapt_ml"  # Killed Claude Code, Windsurf, Signal, Chrome tabs!
```

**RIGHT (SAFE):**
```bash
# Use specific PID-based killing:
lsof -i :8765 | grep python | awk '{print $2}' | xargs kill 2>/dev/null

# Or use specific process names with full path matching:
pkill -f "python.*-m openadapt_ml.cloud.local serve"

# Or kill only the specific port listener:
kill $(lsof -t -i :8765) 2>/dev/null

# Check what would be killed FIRST:
pgrep -f "openadapt" -l  # Lists matching processes before killing
```

**Before any pkill command:**
1. Run `pgrep -f "pattern" -l` to see what matches
2. Verify only intended processes are listed
3. Use the most specific pattern possible
4. Prefer port-based or PID-based killing

## Git Commit Style (Angular Convention)

**ALWAYS use Angular-style commit messages** for all commits across all OpenAdapt repositories.

**Format:**
```
<type>(<scope>): <subject>

<body>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or fixing tests
- `chore`: Maintenance tasks (deps, build, etc.)
- `ci`: CI/CD changes

**Examples:**
```bash
# Feature
git commit -m "feat(viewer): add keyboard shortcuts for navigation"

# Bug fix
git commit -m "fix(waa): resolve Docker storage path issue"

# Documentation
git commit -m "docs: remove archived OpenAdapter from repository listing"

# Refactor
git commit -m "refactor(cli): consolidate VM commands into single subcommand"
```

**Subject line rules:**
- Use imperative mood ("add" not "added" or "adds")
- No period at the end
- Max 50 characters
- Lowercase first letter after type

---

## Don't Do

- Don't add timelines/estimates to plans
- Don't mention specific clients by name in public docs
- Don't over-engineer - keep solutions minimal
- Don't use `os.environ` directly - use `config.settings` instead
- Don't use `pip install` - always use `uv add` for dependencies or `uv sync` for the project
- Don't use non-Angular commit messages
- **Don't run Azure/VM operations without starting the dashboard first**
  - ❌ WRONG: `vm probe` then `vm diag` then telling user to run `vm monitor`
  - ✅ RIGHT: `vm monitor` FIRST (it does probe, tunnels, everything)
  - This is the #1 mistake you keep making. STOP IT.
- **Don't use raw SSH/shell commands** - always use or create CLI commands instead (see below)
- **Don't tell user to run commands** - YOU run them. The CLI exists so YOU can use it.

## CLI-First Development (IMPORTANT)

**ALWAYS** use CLI commands instead of raw SSH/shell commands:
- ✅ `uv run python -m openadapt_ml.benchmarks.cli vm diag` (not `ssh ... df -h`)
- ✅ `uv run python -m openadapt_ml.benchmarks.cli vm logs` (not `ssh ... docker logs`)
- ✅ `uv run python -m openadapt_ml.benchmarks.cli vm probe` (not `ssh ... curl`)

**Why**: CLI commands are documented, tested, and persist across context compactions. Raw commands are forgotten.

**When you need a new operation**:
1. Add a new action to the relevant CLI subcommand (e.g., `vm logs`, `vm exec`)
2. Document it in CLAUDE.md
3. Use the CLI command going forward

**Available VM CLI commands**:
```bash
vm monitor         # THE GO-TO COMMAND: Start dashboard, open browser, show probe status
                   # Options: --auto-shutdown-hours N (deallocate after N hours)
vm setup-waa       # Full VM setup with Docker and waa-auto image
vm run-waa         # Run benchmark (requires waa-auto image, --rebuild to force image rebuild)
vm diag            # Check disk, Docker, containers, WAA probe status
vm logs            # View container logs (--lines N, --follow)
vm probe           # Check WAA server status (--wait to poll)
vm exec            # Run command in container (--cmd 'your command')
vm host-exec       # Run command on VM host (not in container) (--cmd 'your command')
vm start-windows   # Start Windows container with waa-auto image
vm restart-windows # Stop and restart the Windows container
vm check-build     # Check Docker build status from /tmp/waa_build.log
vm stop-build      # Stop running Docker build and clean build cache
vm fix-oem         # Copy OEM files to Samba share (for manual install.bat)
vm reset-windows   # Delete Windows storage and start fresh installation
vm docker-prune    # Clean Docker images, containers, build cache (free disk space)
vm docker-move     # Move Docker/containerd to /mnt via symlinks (300GB space with D8ds_v5)
vm status          # Azure VM status
vm ssh             # Interactive SSH
vm deallocate      # Stop VM billing (preserves disk), use -y to skip confirmation
vm start           # Start a deallocated VM
vm delete          # Delete VM (use -y to skip confirmation)
```

## TODO / Known Issues

### PyPI Publishing
**Status**: DONE

Completed by background agent:
- Updated `pyproject.toml` with package metadata (description, authors, classifiers, URLs, license)
- Created `LICENSE` (MIT, matching related projects)
- Created `.github/workflows/publish.yml` for automated PyPI publishing on version tags
- Build system: hatchling

To publish:
1. Set up PyPI trusted publishing (PyPI → Account Settings → Publishing)
2. `git tag v0.1.0 && git push origin v0.1.0`

### Azure WAA Evaluation - ACR Auth Issue
**Status**: FIXED - setup_azure.py now configures ACR authentication automatically

**Problem**: Azure ML compute instances cannot pull from ACR even after attaching ACR to workspace.
```
Failed to pull Docker image openadaptacr.azurecr.io/winarena:latest
```

**Root cause**: The workspace's managed identity needed AcrPull role on the ACR, which wasn't being granted automatically.

**Solution implemented**:
1. Added `grant_acr_pull_role()` function to setup_azure.py that:
   - Gets workspace managed identity principal ID
   - Assigns AcrPull role on ACR to that identity
2. Added `sync_workspace_keys()` to refresh workspace credentials
3. Updated setup flow from 12 steps to 15 steps:
   - Step 10: Attach ACR to workspace
   - Step 11: Grant AcrPull role to workspace managed identity
   - Step 12: Sync workspace keys

**For existing installations** (if you already ran setup_azure.py):
```bash
# Run the fix script to update your existing workspace
python scripts/fix_acr_auth.py

# Or manually:
PRINCIPAL_ID=$(az ml workspace show -n openadapt-ml -g openadapt-agents --query identity.principal_id -o tsv)
ACR_ID="/subscriptions/78add6c6-c92a-4a53-b751-eb644ac77e59/resourceGroups/openadapt-agents/providers/Microsoft.ContainerRegistry/registries/openadaptacr"
az role assignment create --assignee $PRINCIPAL_ID --role AcrPull --scope $ACR_ID
az ml workspace sync-keys -n openadapt-ml -g openadapt-agents
```

**Related files**:
- `scripts/setup_azure.py` - Azure setup automation (updated with fix)
- `scripts/fix_acr_auth.py` - Manual fix for existing installations
- `openadapt_ml/benchmarks/azure.py` - Azure orchestration
- `.env` - AZURE_DOCKER_IMAGE setting

**References**:
- [Azure ML Managed Identity ACR Authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication)
- [ACR Pull Role Assignment](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication-managed-identity)

### Azure WAA Evaluation - Dedicated VM Setup
**Status**: WORKING - Fully Automated with waa-auto (Jan 2026)

**IMPORTANT**: See `docs/WAA_APPROACH_REVIEW.md` for full documentation.

**CRITICAL**: NO MANUAL ISO DOWNLOADS. Everything is fully automated using `dockurr/windows`.

**How it works**:
- Our `waa-auto` Dockerfile uses `dockurr/windows:latest` as base
- dockurr/windows **automatically downloads Windows 11** based on `VERSION` env var
- Setting `VERSION=11e` downloads Windows 11 Enterprise Evaluation (~6.6 GB) - **fully unattended, no dialogs** (has built-in GVLK key)
- Note: `VERSION=11` (Pro) may prompt for product key if autounattend.xml is misconfigured
- First run: Downloads ISO + installs Windows (~15-20 min)
- Subsequent runs: Boots from cached disk image (~2-3 min)

**FULLY AUTOMATED - Via CLI**:

```bash
# 1. Setup Azure VM with Docker and build waa-auto image (~10 min)
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key $OPENAI_API_KEY

# 2. Run benchmark (Windows auto-downloads on first run)
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 20

# 3. Monitor (optional, for debugging)
uv run python -m openadapt_ml.benchmarks.cli vm monitor
# Opens browser to VNC at http://localhost:8006

# 4. Delete VM when done (IMPORTANT: stops billing!)
uv run python -m openadapt_ml.benchmarks.cli vm delete -y
```

**Diagnostic commands**:
```bash
uv run python -m openadapt_ml.benchmarks.cli vm diag     # Check disk, Docker, containers
uv run python -m openadapt_ml.benchmarks.cli vm status   # Azure VM status
uv run python -m openadapt_ml.benchmarks.cli vm ssh      # Interactive SSH
uv run python -m openadapt_ml.benchmarks.cli vm probe    # Check WAA server readiness
uv run python -m openadapt_ml.benchmarks.cli vm logs     # View container logs
```

**Key requirements**:
1. **VM Size**: `Standard_D8ds_v5` recommended (8 vCPU, 32GB RAM, 300GB temp storage for nested virtualization)
2. **API key**: `config.json` with OPENAI_API_KEY (or set env var)
3. **Valid model**: Use real OpenAI model name (gpt-4o, gpt-4o-mini)

**Architecture**:
```
Azure VM (Standard_D8ds_v5, nested virt enabled, 300GB /mnt)
  └── Docker (data on /mnt)
       └── waa-auto:latest (based on dockurr/windows)
            └── QEMU running Windows 11 (IP: 172.30.0.2)
                 └── WAA Flask server on port 5000
                 └── Navi agent executing tasks
```

**What waa-auto does**:
1. Uses `dockurr/windows:latest` (auto-downloads Windows Enterprise Eval via `VERSION=11e`)
2. Copies WAA client/server from `windowsarena/winarena:latest`
3. Patches IP addresses (20.20.20.21 -> 172.30.0.2)
4. Injects FirstLogonCommands to run install.bat automatically
5. Installs Python dependencies for benchmark client

**Monitor progress**:
- VNC: `http://localhost:8006` (via SSH tunnel, auto-managed by dashboard)
- Logs: `uv run python -m openadapt_ml.benchmarks.cli vm logs`

**Files**:
- `docs/WAA_APPROACH_REVIEW.md` - Full analysis (updated Jan 2026)
- `openadapt_ml/benchmarks/waa_deploy/Dockerfile` - Custom waa-auto image
- `openadapt_ml/benchmarks/cli.py` - CLI commands

### Docker Disk Space Management
**Status**: FIXED - Automatic cleanup (Jan 2026)

**Problem**: Docker build cache on /mnt was growing to 90+ GB during builds, exhausting disk space and causing builds to fail with "no space left on device". Note: With Standard_D8ds_v5, /mnt is now 300GB which should be sufficient.

**Root cause**: Docker's build cache and containerd snapshotter accumulate data that isn't cleaned by `docker system prune`:
- `/mnt/docker/buildkit/containerd-overlayfs` - BuildKit layer cache
- `/mnt/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots` - Containerd snapshots
- These can grow to 30-40 GB each, even with no images present

**Solution implemented** (3 parts):

1. **Automatic pre-build cleanup**: Before Docker builds, the CLI now runs `docker builder prune -af` and checks available disk space, warning if < 50GB.

2. **Automatic post-build cleanup**: After successful builds, the CLI cleans build cache and dangling images to prevent accumulation.

3. **BuildKit garbage collection**: New VMs are configured with `/etc/buildkit/buildkitd.toml` that limits cache to 30GB max.

4. **Enhanced docker-prune command**: Now includes "deep cleanup" that stops Docker/containerd and removes orphaned snapshots that normal prune misses.

**Usage**:
```bash
# Quick cleanup (standard prune + deep cleanup + configure GC)
uv run python -m openadapt_ml.benchmarks.cli vm docker-prune

# For severe disk issues, delete VM and recreate (comes with GC pre-configured)
uv run python -m openadapt_ml.benchmarks.cli vm delete -y
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key $OPENAI_API_KEY
```

**Files changed**:
- `openadapt_ml/benchmarks/cli.py` - Pre/post build cleanup, enhanced docker-prune
- New VMs get BuildKit GC config during setup

### Windows "Select Operating System" Prompt Fix
**Status**: FIXED (Jan 2026)

**Problem**: Windows installer shows "Select the operating system you want to install" dialog instead of auto-selecting, even with autounattend.xml.

**Root cause**: The autounattend.xml from dockurr/windows lacks an `<InstallFrom>` element with `<MetaData>` to specify which image index to install. When install.wim contains multiple editions (or when Windows can't auto-detect), it prompts the user.

**Solution**: Added `<InstallFrom>` element to autounattend.xml that explicitly selects image index 1:

```xml
<ImageInstall>
  <OSImage>
    <InstallFrom>
      <MetaData wcm:action="add">
        <Key>/IMAGE/INDEX</Key>
        <Value>1</Value>
      </MetaData>
    </InstallFrom>
    <InstallTo>...</InstallTo>
  </OSImage>
</ImageInstall>
```

**Files changed**:
- `openadapt_ml/benchmarks/waa_deploy/Dockerfile` - Adds sed command to inject InstallFrom element

**If you still see the prompt**:
1. Delete cached storage: `uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd 'rm -rf /mnt/waa-storage/*'`
2. Rebuild waa-auto image: `uv run python -m openadapt_ml.benchmarks.cli vm run-waa --rebuild`
3. Check container is using waa-auto (not dockurr/windows directly): `uv run python -m openadapt_ml.benchmarks.cli vm host-exec --cmd 'docker inspect winarena | grep Image'`

### SSH Tunnel Management (VNC/WAA Access)
**Status**: DONE

**Problem**: Azure VMs have Network Security Groups (NSGs) that only expose port 22 (SSH) by default. Ports 8006 (VNC) and 5000 (WAA) are not accessible directly.

**Solution**: Automatic SSH tunnel management via `SSHTunnelManager`:

```
Browser → localhost:8006 → SSH Tunnel → Azure VM:8006 → Docker → noVNC
Browser → localhost:5000 → SSH Tunnel → Azure VM:5000 → WAA Flask
```

**Architecture**:
1. When VM's WAA probe becomes "ready", tunnels auto-start
2. When VM goes offline, tunnels auto-stop
3. Dashboard shows tunnel status next to VNC button
4. VNC button links to localhost:port (tunnel endpoint)

**Files**:
- `openadapt_ml/cloud/ssh_tunnel.py` - SSHTunnelManager class
- `openadapt_ml/cloud/local.py` - Integration with dashboard server
- `openadapt_ml/training/benchmark_viewer.py` - UI showing tunnel status

**API Endpoints**:
- `GET /api/tunnels` - Returns tunnel status for VNC and WAA
- `GET /api/vms` - Includes `tunnels` field with per-tunnel status

**Key features**:
- Auto-start on VM online (idempotent - safe to call repeatedly)
- Auto-stop on VM offline
- Port conflict detection
- Graceful shutdown on process exit
- No manual SSH commands needed

**Manual usage** (if needed):
```python
from openadapt_ml.cloud.ssh_tunnel import get_tunnel_manager

manager = get_tunnel_manager()
manager.start_tunnels_for_vm("172.171.112.41", "azureuser")
status = manager.get_tunnel_status()
manager.stop_all_tunnels()
```

**Why not open NSG ports?**
1. VNC has no authentication by default - anyone can connect
2. SSH tunnel encrypts all traffic
3. Requires SSH key auth - no password guessing
4. No Azure NSG changes needed

**Alternative: Mock evaluation** for testing without Windows:
```bash
uv run python -m openadapt_ml.benchmarks.cli test-mock --tasks 20
```

**References**:
- [Windows Agent Arena GitHub](https://github.com/microsoft/WindowsAgentArena)
- [Azure nested virtualization](https://learn.microsoft.com/en-us/azure/virtual-machines/acu)

### Training Dashboard - Terminal Output Streaming
**Status**: DONE

**Goal**: Show training command line output in the browser dashboard in real-time.

**Implementation**: File-based polling approach
1. Training writes stdout to `training_output/training.log` with timestamps
2. Browser polls training.log every 2 seconds alongside training_log.json
3. Displays last 500 lines in scrollable terminal panel with auto-scroll
4. Terminal panel features:
   - Dark terminal theme (black background, green/colored text)
   - Auto-scroll toggle (on by default)
   - Text wrap toggle
   - Collapse/expand button
   - Line counter
   - Syntax highlighting (errors in red, warnings in orange, success in green)

**Files changed**:
- `openadapt_ml/training/trainer.py`:
  - Added terminal panel CSS styles
  - Added terminal panel HTML section
  - Added JavaScript polling function `fetchTerminalOutput()`
  - Added `TrainingLogger._log_to_terminal()` method
  - Updated `train_supervised()` to log key messages to training.log
- `openadapt_ml/training/stub_provider.py`:
  - Added `_log()` method for dual stdout/file logging
  - All training output now written to training.log
- `openadapt_ml/cloud/local.py`:
  - No changes needed - serve command already serves all files from training_output

**Usage**: Terminal output automatically appears in dashboard during training. Works with both stub and real training.

### Early Termination Controls
**Status**: DONE

**Problem**: Training runs until completion even when loss is low enough. Wastes GPU credits ($0.75/hr for A10).

**Solution implemented**:
1. **Auto-termination**: `early_stop_loss` and `early_stop_patience` in stub_provider.py
2. **Dashboard button**: "Stop Training" button calls `/api/stop` endpoint
3. **Stop signal**: Creates `STOP_TRAINING` file that training loop checks
4. **Termination status**: Dashboard shows termination reason (auto_complete, auto_low_loss, user_stop)

**Files changed**:
- `openadapt_ml/cloud/local.py` - Added `/api/stop` POST endpoint
- `openadapt_ml/training/stub_provider.py` - Added early stop logic, termination status
- `openadapt_ml/training/trainer.py` - Added `updateTerminationStatus()` JS function

### Cloud Cost Estimation in Viewers
**Status**: DONE

Added cost display panel to viewer that shows:
- Running cost based on instance type and elapsed time
- Instance type and hourly rate
- Only visible for cloud training (hidden for local/stub)

Supported rates:
- Lambda Labs: $0.75/hr for A10, $1.29/hr for A100
- Automatic detection from `instance_type` in training_log.json

### Current Working Capture
**Path**: `/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift`
**Task**: Turn off Night Shift in macOS System Settings
**Screenshots**: 20 frames
**Notes**: Real-world macOS settings navigation capture for training/evaluation

### Evaluation Samples Display Enhancement
**Status**: DONE

Enhanced evaluation gallery in dashboard with:
- **Filter controls**: Dropdown filters for epoch and correctness (All/Correct/Incorrect)
- **Visual markers**: H (human) and AI (predicted) click markers on screenshots
- **Expandable model output**: "Show full output" toggle for raw model reasoning
- **Better layout**: Image container with overlay, content section with coordinates
- **Sample count**: "Showing X of Y samples" with filter status

Files changed:
- `openadapt_ml/training/trainer.py` - Enhanced CSS, HTML, and JS for eval gallery

### Viewer Playback Controls
**Status**: DONE

Added full playback controls to the viewer:
- **Buttons**: ⏮ Rewind, ◀ Prev, ▶ Play/Pause, ▶ Next, ⏭ End
- **Speed control**: 0.5x, 1x, 2x, 4x playback speeds
- **Progress bar**: Click-to-seek to any step
- **Keyboard shortcuts**: Space (play/pause), Home/End (jump), Arrow keys (step)
- **Enhanced details panel**: Shows full model output with scrollable raw prediction data

### Viewer Code Consolidation
**Status**: DONE

**Problem**: Viewer code was fragmented across multiple locations:
1. `generate_training_dashboard()` - generates unified viewer template
2. `_enhance_comparison_to_unified_viewer()` - injected checkpoint_script into comparison.html
3. `comparison.html` from capture - had its own display logic

**Solution implemented**:
- `generate_unified_viewer_from_output_dir()` now always uses `_generate_unified_viewer_from_extracted_data()`
- This generates a complete standalone viewer.html without script injection
- `_enhance_comparison_to_unified_viewer()` marked as deprecated
- All viewer display logic is now in one place (`_generate_unified_viewer_from_extracted_data`)
- Changes to viewer code now propagate reliably

### README API Documentation
**Status**: VERIFIED

The README §7.1 API-backed adapters section uses correct model names:
- "Claude Sonnet 4.5" → `claude-sonnet-4-5-20250929` in api_adapter.py ✓
- "GPT-5.1" → `gpt-5.1` in api_adapter.py ✓

Verified:
- API key environment variable names: ANTHROPIC_API_KEY, OPENAI_API_KEY ✓
- Backend flag options: `claude`, `openai` in CLI ✓

### Benchmark Viewer Integration
**Status**: Phases 1-3 DONE, Phase 4 TODO

**Goal**: Integrate benchmark evaluation results (WAA, WebArena, OSWorld) into the unified viewer.

**Design doc**: `docs/benchmark_viewer_integration.md`

**Key features**:
1. **Benchmarks tab**: Third tab alongside Training and Viewer
2. **Task-level view**: List of benchmark tasks with pass/fail status
3. **Step-by-step replay**: Same UI as Viewer tab for benchmark executions
4. **Model comparison**: Side-by-side comparison of different models on same task (TODO)
5. **Aggregate metrics**: Success rate by domain, difficulty rankings

**Implementation phases**:
1. ✅ **Data collection** (DONE): Save screenshots during benchmark runs
   - Created `openadapt_ml/benchmarks/data_collection.py` with `ExecutionTraceCollector`
   - Updated `runner.py` to save execution traces automatically
   - Added CLI command: `uv run python -m openadapt_ml.benchmarks.cli test-collection --tasks 5`
   - Directory structure: `benchmark_results/{run_name}/tasks/{task_id}/`
   - Each task has: `task.json`, `execution.json`, `screenshots/`
   - Test script: `test_data_collection.py` validates all files are created
2. ✅ **Viewer backend** (DONE): `generate_benchmark_viewer()` function
   - Created `openadapt_ml/benchmarks/viewer.py` with viewer generation
   - Added CLI command: `uv run python -m openadapt_ml.benchmarks.cli view --run-name {name}`
   - Generates standalone HTML with same styling as training viewer
   - Uses shared header components via `shared_ui.py`
3. ✅ **UI components** (DONE - Basic): Summary dashboard, task list, replay
   - Summary panel with total tasks, passed/failed, success rate
   - Domain breakdown with per-domain statistics
   - Filter controls (domain, status)
   - Task list with status badges
   - Step-by-step viewer with screenshots, actions, reasoning
   - Playback controls (prev/next, play/pause, speed)
   - Keyboard shortcuts (Space, arrows, Home/End)
4. **Analysis** (TODO): Failure clustering, regression detection

**View benchmark results:**
```bash
# Generate HTML viewer and serve it
uv run python -m openadapt_ml.benchmarks.cli view --run-name {name}

# Options:
# --embed-screenshots  Embed screenshots as base64 (standalone HTML)
# --no-open            Don't auto-open browser
# --port 9000          Use custom port
```

## Preventing Stale Data Issues

**CRITICAL**: When working on dashboard/viewer code, follow this process to avoid showing stale data:

### After Code Changes

1. **Always regenerate HTML files** after modifying trainer.py, viewer.py, or local.py:
   ```bash
   uv run python -m openadapt_ml.cloud.local viewer
   ```

2. **Verify regeneration worked** by checking key values:
   ```bash
   # Check elapsed time was updated (should NOT be 0)
   grep "baseElapsedTime" training_output/current/dashboard.html

   # Check comparison data exists in viewer
   grep "predictionsByCheckpoint" training_output/current/viewer.html
   ```

3. **Hard refresh browser** to bypass cache:
   - macOS: `Cmd+Shift+R`
   - Windows/Linux: `Ctrl+Shift+R`
   - Or use DevTools → Network → "Disable cache" checkbox

4. **Use HTTP serving** (not file://) for auto-refresh:
   ```bash
   uv run python -m openadapt_ml.cloud.local serve --port 8080 --open
   ```

### Before Showing User

Before presenting dashboard/viewer to user, verify:
- [ ] Elapsed time shows correct value (not 0m 0s)
- [ ] Comparison screenshots load (not blank/404)
- [ ] Model predictions appear in dropdown
- [ ] Loss curve shows data
- [ ] Timestamp info panel shows recent dates

### Automatic Data Loading Checklist

The viewer should automatically load:
- [ ] Capture data from `comparison_epoch*.html` files (extracts `window.comparisonData`)
- [ ] Predictions from same comparison HTML files (human + predicted actions per step)
- [ ] Evaluations from `training_log.json` (if present)
- [ ] Recording events from capture data (note: `recording.end` depends on capture source)

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Elapsed time shows 0m 0s | `elapsed_time` not loaded from training_log.json | Check `state.elapsed_time = data.get("elapsed_time", 0.0)` in local.py |
| No comparison screenshots | Paths point to Lambda not local | Update `capture_path` in training_log.json to local path |
| Missing model predictions | No `comparison_epoch*.html` files or wrong data format | Run compare script: `uv run python -m openadapt_ml.scripts.compare --capture ... --checkpoint ...` |
| Predictions not extracted | HTML uses `window.comparisonData` but regex expects `const` | Use regex `(?:const\s+\|window\.)comparisonData` pattern |
| Stale data after code change | Browser caching HTML | Hard refresh (Cmd+Shift+R) or disable cache |
| Screenshots 404 | Screenshot symlink broken | Recreate: `ln -sf /path/to/capture/screenshots training_output/current/screenshots` |

### UI/Display Guidelines

**Placeholder data must be clearly marked** when displaying values that may not reflect actual data:
- If task counts, worker counts, etc. come from local tracking (not synced with Azure), mark them with an asterisk: "3* tasks • 1* worker(s)"
- Add a footnote: "[*: placeholder, actual values may differ]"
- This applies to any data that is locally cached but not confirmed from the authoritative source

### Azure ML Integration Notes

**Experiment ID**: The Azure ML experiments page URL requires an experiment ID which is workspace-specific:
- Current hardcoded ID: `ad29082c-0607-4fda-8cc7-38944eb5a518`
- **TODO**: Retrieve experiment_id dynamically from Azure using `az ml experiment list`
- The experiment name is `openadapt-ml` but the URL requires the UUID format

**Azure ML URL format**:
- Jobs list: `https://ml.azure.com/experiments/id/{experiment_id}?wsid={workspace_id}`
- Specific job: `https://ml.azure.com/experiments/id/{experiment_id}/runs/{run_id}?wsid={workspace_id}`

**WAA Docker command**: Use `python run.py` not `python -m client.run` (the client directory is not a Python package)
