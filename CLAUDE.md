# Claude Context for openadapt-ml

This file helps maintain context across sessions.

## Project Overview

openadapt-ml is a model-agnostic, domain-agnostic ML engine for GUI automation agents. It provides:
- Schemas for GUI interaction trajectories
- Synthetic UI generation for bootstrapping
- VLM adapters (Qwen3-VL, Qwen2.5-VL, API backends)
- Supervised fine-tuning pipeline
- Runtime policy API

## Current Focus: Benchmark Integration

**Primary benchmark**: Windows Agent Arena (WAA)
- 154 tasks across 11 Windows domains
- MIT licensed, can run locally or on Azure
- SOTA: ~19.5% success (GPT-4V + OmniParser)

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

7. **Stub Training Adapter (HIGH PRIORITY)** - Always implement stub/mock providers first:
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

## Benchmark Integration (Implemented)

The benchmark integration module is implemented in `openadapt_ml/benchmarks/`:
- `base.py` - BenchmarkAdapter interface, data classes
- `agent.py` - BenchmarkAgent, PolicyAgent, ScriptedAgent, RandomAgent
- `runner.py` - evaluate_agent_on_benchmark(), compute_metrics()
- `waa.py` - WAAAdapter (requires WAA repo), WAAMockAdapter (for testing)
- `azure.py` - AzureWAAOrchestrator for parallel VM execution
- `cli.py` - Command-line interface for WAA evaluation

### Azure Automation

`scripts/setup_azure.py` fully automates Azure setup with 9 steps:
1. Check Azure CLI installation
2. Login to Azure
3. Select subscription
4. Register resource providers (Compute, ML, Storage, ContainerRegistry)
5. Create resource group
6. Create service principal with Contributor role
7. Create ML workspace
8. Create Azure Container Registry (ACR)
9. Import WAA Docker image from Docker Hub to ACR

The script writes all credentials to `.env` including:
- Service principal credentials (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
- Workspace config (AZURE_SUBSCRIPTION_ID, AZURE_ML_RESOURCE_GROUP, AZURE_ML_WORKSPACE_NAME)
- Docker image path (AZURE_DOCKER_IMAGE) pointing to ACR

**Why ACR?** Azure ML cannot pull from Docker Hub or ghcr.io directly. The image must be in ACR.

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

# Generate/refresh dashboard manually during training
python3 -c "
import json
from openadapt_ml.training.trainer import TrainingState, TrainingConfig, generate_training_dashboard
from pathlib import Path
with open('training_output/training_log.json') as f:
    data = json.load(f)
state = TrainingState()
state.epoch, state.step, state.loss = data['epoch'], data['step'], data['loss']
state.learning_rate, state.losses = data['learning_rate'], data['losses']
config = TrainingConfig(num_train_epochs=5, learning_rate=5e-5)
Path('training_output/dashboard.html').write_text(generate_training_dashboard(state, config))
" && open training_output/dashboard.html

# Compare human vs model predictions
uv run python -m openadapt_ml.scripts.compare \
  --capture /path/to/capture \
  --checkpoint checkpoints/model \
  --open
```

## Files to Know

- `docs/cloud_gpu_training.md` - Lambda Labs and Azure GPU training guide
- `docs/benchmark_integration_plan.md` - Benchmark integration architecture
- `docs/azure_waa_setup.md` - Azure WAA setup guide (quota increase, costs, troubleshooting)
- `docs/design.md` - Overall system design
- `openadapt_ml/cloud/` - Cloud GPU providers (Lambda Labs, Azure)
- `openadapt_ml/benchmarks/` - Benchmark integration module (WAA, base classes)
- `openadapt_ml/grounding/` - Grounding module (GeminiGrounder, etc.)
- `openadapt_ml/ingest/capture.py` - Converts openadapt-capture recordings to Episodes
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

## Don't Do

- Don't add timelines/estimates to plans
- Don't mention specific clients by name in public docs
- Don't over-engineer - keep solutions minimal
- Don't use `os.environ` directly - use `config.settings` instead
- Don't use `pip install` - always use `uv pip install` or `uv add` for consistency

## TODO / Known Issues

### PyPI Publishing
**Status**: TODO

openadapt-capture and openadapt-privacy are published to PyPI, but openadapt-ml is not yet. Should set up:
- Package metadata in pyproject.toml
- GitHub Actions workflow for publishing
- Version management

### Azure WAA Evaluation - ACR Auth Issue
**Status**: Blocked - needs fix before Azure eval will work

**Problem**: Azure ML compute instances cannot pull from ACR even after attaching ACR to workspace.
```
Failed to pull Docker image openadaptacr.azurecr.io/winarena:latest
```

**What was tried**:
1. Created ACR: `openadaptacr.azurecr.io`
2. Imported WAA image from Docker Hub
3. Attached ACR to ML workspace with `-u` flag

**Next steps to try**:
1. Grant AcrPull role to workspace managed identity:
   ```bash
   az role assignment create --assignee 7f2995f4-f097-45d7-8b5c-189aefdcb3e3 \
     --role AcrPull --scope /subscriptions/.../registries/openadaptacr
   ```
2. Or use ACR admin credentials in environment config
3. Check if workspace keys need sync: `az ml workspace sync-keys`

**Related files**:
- `scripts/setup_azure.py` - Azure setup automation
- `openadapt_ml/benchmarks/azure.py` - Azure orchestration
- `.env` - AZURE_DOCKER_IMAGE setting

### Training Dashboard - Terminal Output Streaming
**Status**: TODO - nice to have

**Goal**: Show training command line output in the browser dashboard in real-time.

**Possible approaches**:
1. **File-based polling** (simplest): Training writes stdout to `training_output/training.log`, browser polls and displays in a `<pre>` element with auto-scroll
2. **WebSocket**: Run training in subprocess, stream stdout via WebSocket server to browser
3. **Server-sent events (SSE)**: Similar to WebSocket but simpler, one-way streaming

**Recommended**: File-based polling is simplest and consistent with current JSON polling approach. Add:
- `--log-stdout` flag to train.py that tees output to training.log
- Add scrollable terminal panel to dashboard.html
- Poll training.log alongside training_log.json

**Priority**: Low - current dashboard shows key metrics (loss, epoch, step). Terminal output mainly useful for debugging.

### Early Termination Controls
**Status**: TODO - HIGH PRIORITY

**Problem**: Training runs until completion even when loss is low enough. Wastes GPU credits ($0.75/hr for A10).

**Requirements**:
1. **Auto-termination**: Stop training when loss drops below threshold (e.g., 0.5 or configurable)
2. **Dashboard button**: "Stop Training" button in dashboard UI that terminates Lambda instance
3. **Checkpoint download**: Auto-download best checkpoint before termination
4. **Cost awareness**: Show running cost and prompt user when approaching budget

**Implementation approach**:
- Add `early_stop_loss` to training config (already exists but may not terminate instance)
- Add terminate endpoint that dashboard can call
- Modify Lambda monitor to download checkpoints on termination
- Add "Stop Training" button to dashboard config section

### Cloud Cost Estimation in Viewers
**Status**: TODO

**Goal**: Show estimated cloud costs in training dashboard and viewer.

**Requirements**:
- Display running cost based on instance type and elapsed time
- Show estimated total cost for completion
- Include cost breakdown by resource type (GPU, storage, transfer)

**Implementation notes**:
- Lambda Labs: $0.75/hr for A10, $1.29/hr for A100
- Azure ML: Variable based on VM type
- Should be visible in both dashboard.html and viewer.html

### Current Working Capture
**Path**: `/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift`
**Task**: Turn off Night Shift in macOS System Settings
**Screenshots**: 20 frames
**Notes**: Real-world macOS settings navigation capture for training/evaluation

### Evaluation Samples Display Enhancement
**Status**: TODO - needs fleshing out

**Current state**: Shows human/predicted coords, model thinking text, legend
**Future improvements**:
- Show the actual screenshot image (need to sync from Lambda or embed base64)
- Visual overlay showing click positions on image
- Side-by-side human vs predicted action comparison
- Full model output (not truncated)
- Filter/search evaluations by epoch or correctness

### README API Documentation
**Status**: TODO - needs review

The README §7.1 API-backed adapters section has placeholder model names that should be updated:
- "Claude Sonnet 4.5" → should reflect actual model (e.g., claude-3-5-sonnet, claude-3-opus)
- "GPT-5.1" → should reflect actual model (e.g., gpt-4-turbo, gpt-4o)

Check `openadapt_ml/models/api_adapter.py` for actual model names used and update README to match.

Also verify:
- API key environment variable names are correct
- Example code snippets work
- Backend flag options in CLI match actual implementations
