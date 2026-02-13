# Claude Context for openadapt-ml

## Simplicity Guidelines

**Philosophy**: "Less is more. 80/20 impact/complexity. Working code beats elegant design."

**Before writing code**: Can this be <100 lines? Does this provide 80% of value? Is this the simplest approach?

**Avoid**: Classes when functions work, abstractions before 3rd use, design docs for non-existent code.

See: `/Users/abrichr/oa/src/openadapt-evals/SIMPLICITY_PRINCIPLES.md` for full guidelines.

---

## Project Status

**IMPORTANT**: Check `/Users/abrichr/oa/src/STATUS.md` at session start for P0 priorities.

## Project Overview

openadapt-ml: Model-agnostic ML engine for GUI automation agents.
- Schemas for GUI trajectories
- VLM adapters (Qwen3-VL, Qwen2.5-VL, API backends)
- Supervised fine-tuning pipeline
- Runtime policy API
- ML-specific benchmark agents (PolicyAgent, APIBenchmarkAgent)

**NOTE**: All evaluation infrastructure (VM management, pool orchestration, CLI, adapters, runners, viewers) has been migrated to `openadapt-evals`. See the migration guide below.

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

### What lives where

| Package | Purpose |
|---------|---------|
| `openadapt-ml` | ML agents (PolicyAgent, APIBenchmarkAgent, UnifiedBaselineAgent), schemas, training |
| `openadapt-evals` | Everything else: VM infra, pool management, CLI (`oa-vm`), adapters, runners, viewers |

```python
# Evaluation infrastructure (in openadapt-evals)
from openadapt_evals import ApiAgent, WAAMockAdapter, evaluate_agent_on_benchmark

# ML-specific agents (still in openadapt-ml)
from openadapt_ml.benchmarks import PolicyAgent, APIBenchmarkAgent, UnifiedBaselineAgent
```

### Evaluation CLI (in openadapt-evals)

All VM management, pool orchestration, and benchmark execution commands are now in `openadapt-evals`:

```bash
cd /Users/abrichr/oa/src/openadapt-evals

# VM/Pool management
oa-vm pool-create --workers 3
oa-vm pool-wait
oa-vm pool-run --tasks 10
oa-vm pool-status
oa-vm pool-cleanup -y

# Single VM
oa-vm create --fast
oa-vm status
oa-vm delete

# Benchmark execution
uv run python -m openadapt_evals.benchmarks.cli run --agent api-claude --task notepad_1
uv run python -m openadapt_evals.benchmarks.cli mock --tasks 10
```

---

## Migration Guide (from openadapt_ml.benchmarks.cli)

The following imports/commands have moved:

| Old (openadapt-ml) | New (openadapt-evals) |
|---------------------|----------------------|
| `openadapt_ml.benchmarks.cli` | `openadapt_evals.benchmarks.vm_cli` (or `oa-vm` entry point) |
| `openadapt_ml.benchmarks.azure_vm.AzureVMManager` | `openadapt_evals.infrastructure.azure_vm.AzureVMManager` |
| `openadapt_ml.benchmarks.pool.PoolManager` | `openadapt_evals.infrastructure.pool.PoolManager` |
| `openadapt_ml.benchmarks.vm_monitor.VMMonitor` | `openadapt_evals.infrastructure.vm_monitor.VMMonitor` |
| `openadapt_ml.benchmarks.azure_ops_tracker` | `openadapt_evals.infrastructure.azure_ops_tracker` |
| `openadapt_ml.benchmarks.resource_tracker` | `openadapt_evals.infrastructure.resource_tracker` |
| `openadapt_ml.benchmarks.pool_viewer` | `openadapt_evals.benchmarks.pool_viewer` |
| `openadapt_ml.benchmarks.trace_export` | `openadapt_evals.benchmarks.trace_export` |
| `openadapt_ml.benchmarks.waa_deploy` | `openadapt_evals.waa_deploy` |
| `openadapt_ml.config.settings` (Azure VM settings) | `openadapt_evals.config.settings` |

**What stays in openadapt-ml**:
- `openadapt_ml.benchmarks.agent` (PolicyAgent, APIBenchmarkAgent, UnifiedBaselineAgent)
- `openadapt_ml.config.settings` (ML-specific settings: API keys, training config)
- All schemas, training, inference, model adapters

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

## Files to Know

- `docs/WAA_APPROACH_REVIEW.md` - Full WAA setup documentation
- `docs/cloud_gpu_training.md` - Lambda Labs/Azure training guide
- `docs/azure_waa_setup.md` - Azure quota, costs, troubleshooting
- `docs/design.md` - System design
- `openadapt_ml/benchmarks/agent.py` - ML-specific benchmark agents
- `openadapt_ml/cloud/ssh_tunnel.py` - SSH tunnel manager
- `openadapt_ml/config.py` - Settings (pydantic-settings)
- `openadapt_ml/schemas/` - Canonical schema definitions

---

## Git Commit Style (Angular)

```
<type>(<scope>): <subject>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**Types**: feat, fix, docs, style, refactor, perf, test, chore, ci

**Rules**: Imperative mood, no period, max 50 chars, lowercase after type

---

## Don't Do

- Don't use `os.environ` - use `config.settings`
- Don't use `pip install` - use `uv add` or `uv sync`
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

## Troubleshooting

### Dashboard/Viewer Stale Data
After code changes:
1. Regenerate: `uv run python -m openadapt_ml.cloud.local viewer`
2. Hard-refresh browser: Cmd+Shift+R

### Common Issues Table

| Symptom | Fix |
|---------|-----|
| Elapsed time shows 0 | Check training_log.json has elapsed_time |
| No comparison screenshots | Update capture_path in training_log.json |
| Stale data after code change | Hard refresh (Cmd+Shift+R) |

See `docs/` for detailed troubleshooting guides.
