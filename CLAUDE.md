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
- `agent.py` - BenchmarkAgent, PolicyAgent, APIBenchmarkAgent, ScriptedAgent, RandomAgent
- `runner.py` - evaluate_agent_on_benchmark(), compute_metrics()
- `waa.py` - WAAAdapter (requires WAA repo), WAAMockAdapter (for testing)
- `azure.py` - AzureWAAOrchestrator for parallel VM execution
- `cli.py` - Command-line interface for WAA evaluation

### APIBenchmarkAgent

The `APIBenchmarkAgent` wraps hosted VLM APIs (Claude, GPT-4V) for benchmark evaluation baselines.
This enables comparing fine-tuned models against off-the-shelf VLMs.

```python
from openadapt_ml.benchmarks import APIBenchmarkAgent, evaluate_agent_on_benchmark

# Claude baseline
agent = APIBenchmarkAgent(provider="anthropic")
results = evaluate_agent_on_benchmark(agent, adapter)

# GPT-4V baseline
agent = APIBenchmarkAgent(provider="openai")
results = evaluate_agent_on_benchmark(agent, adapter)
```

CLI usage:
```bash
# Run Claude evaluation on mock tasks
uv run python -m openadapt_ml.benchmarks.cli run-api --provider anthropic --tasks 5

# Run GPT-4V evaluation
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
**Status**: Phase 1 DONE, Phases 2-4 TODO

**Goal**: Integrate benchmark evaluation results (WAA, WebArena, OSWorld) into the unified viewer.

**Design doc**: `docs/benchmark_viewer_integration.md`

**Key features**:
1. **Benchmarks tab**: Third tab alongside Training and Viewer
2. **Task-level view**: List of benchmark tasks with pass/fail status
3. **Step-by-step replay**: Same UI as Viewer tab for benchmark executions
4. **Model comparison**: Side-by-side comparison of different models on same task
5. **Aggregate metrics**: Success rate by domain, difficulty rankings

**Implementation phases**:
1. ✅ **Data collection** (DONE): Save screenshots during benchmark runs
   - Created `openadapt_ml/benchmarks/data_collection.py` with `ExecutionTraceCollector`
   - Updated `runner.py` to save execution traces automatically
   - Added CLI command: `uv run python -m openadapt_ml.benchmarks.cli test-collection --tasks 5`
   - Directory structure: `benchmark_results/{run_name}/tasks/{task_id}/`
   - Each task has: `task.json`, `execution.json`, `screenshots/`
   - Test script: `test_data_collection.py` validates all files are created
2. **Viewer backend** (TODO): `generate_benchmark_viewer()` function
3. **UI components** (TODO): Summary dashboard, task list, replay
4. **Analysis** (TODO): Failure clustering, regression detection

**Phase 1 verification:**
```bash
# Test data collection
uv run python -m openadapt_ml.benchmarks.cli test-collection --tasks 5

# Verify output
ls -la benchmark_results/{run_name}/tasks/task_001/
# Should contain: task.json, execution.json, screenshots/

# Check JSON structure
cat benchmark_results/{run_name}/summary.json
cat benchmark_results/{run_name}/tasks/task_001/execution.json
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
