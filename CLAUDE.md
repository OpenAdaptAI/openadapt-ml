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

openadapt-ml (v0.5.0): Pure ML engine for GUI automation agents.

| What | Where |
|------|-------|
| Schemas, VLM adapters, training, inference, grounding, ML agents | `openadapt-ml` |
| Evaluation infrastructure (VM management, pool orchestration, CLI) | `openadapt-evals` |

### Package structure

- `openadapt_ml/schema/` - Canonical trajectory schemas (episode, converters)
- `openadapt_ml/models/` - VLM adapters (Qwen3-VL, Qwen2.5-VL, API backends)
- `openadapt_ml/training/` - Supervised fine-tuning pipeline
- `openadapt_ml/runtime/` - Policy API
- `openadapt_ml/grounding/` - Grounding module (deprioritized; for real UIs without SoM overlays)
- `openadapt_ml/baselines/` - Baseline adapters
- `openadapt_ml/benchmarks/agent.py` - ML-specific agents (PolicyAgent, APIBenchmarkAgent, UnifiedBaselineAgent)
- `openadapt_ml/cloud/` - Cloud GPU training (Lambda Labs, Azure inference, SSH tunnels)
- `openadapt_ml/config.py` - Settings (pydantic-settings)
- `configs/` - Training YAML configs (qwen3vl_capture.yaml, etc.)

---

## Key Architecture Decisions

1. **SoM mode** - Element IDs (`CLICK([1])`) instead of coordinates for accuracy on synthetic benchmarks
2. **Schema purity** - Domain-agnostic; external systems adapt TO the schema, not vice versa
3. **Lossless preservation** - Store raw benchmark configs in `raw_config`, `raw_observation`, `raw_action` fields
4. **DOM/AX mandatory in schema** - For evaluator compatibility (WebArena, Mind2Web need DOM), even if agents use vision-only
5. **Cloud-first** - Offload heavy compute to cloud GPUs (Azure, Lambda Labs)
6. **Stub training** - Use `--stub` flag for rapid UI iteration without GPU

---

## Demo Retrieval (Core Value Proposition)

**Key insight**: OpenAdapt's value is trajectory-conditioned disambiguation of UI affordances.

**Validated**: Demo-conditioned prompting improves accuracy (zero-shot 33% -> with demo 100% correct first actions). See `docs/experiments/demo_conditioned_prompting_results.md`.

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

## Azure Setup

`scripts/setup_azure.py` automates Azure resource creation (resource group, service principal, ML workspace, ACR, WAA Docker image import):

```bash
python scripts/setup_azure.py            # Setup
python scripts/setup_azure.py --cleanup  # Cleanup
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

Pre-approved read access to `~/oa/src/` (related projects like openadapt-capture, openadapt-evals).

---

## Troubleshooting

### Dashboard/Viewer Stale Data
After code changes:
1. Regenerate: `uv run python -m openadapt_ml.cloud.local viewer`
2. Hard-refresh browser: Cmd+Shift+R

| Symptom | Fix |
|---------|-----|
| Elapsed time shows 0 | Check training_log.json has elapsed_time |
| No comparison screenshots | Update capture_path in training_log.json |
| Stale data after code change | Hard refresh (Cmd+Shift+R) |

See `docs/` for detailed troubleshooting guides.
