# Enterprise Workflow Integration Guide

This guide explains how to export workflow recordings from your enterprise automation platform and use them with openadapt-ml for training and demo-conditioned prompting.

**Ideal for**: Teams with libraries of recorded workflow demonstrations, including multiple examples of the same or similar workflows.

---

## When to Use Episode Format

Use Episode format if your data involves:

- Human or agent interaction with graphical user interfaces
- Multi-step workflows with intermediate UI states
- Action replay, evaluation, or imitation learning

If your data consists only of static images or independent text samples, a generic dataset format may be sufficient.

---

## GUI Automation Data Requirements

GUI automation datasets have requirements that differ from image-only or text-only ML datasets:

- Screenshot + action temporal alignment
- Goal-conditioned retrieval
- Multi-step trajectory support
- Training-ready SFT format
- Verification/outcome annotation (did the action succeed?)

The Episode schema makes these properties explicit by design.

---

## Why Start with Episode Format?

Retrofitting GUI automation data into a trajectory format later is costly and error-prone.

Information such as:
- Action timing
- UI state transitions
- Intent alignment

is often lost once data is flattened into image-only or text-only datasets.

**Once screenshots and actions are decoupled, the original trajectory cannot be reconstructed reliably.**

Capturing trajectories at export time avoids this loss.

**Data Portability**: Episodes serialize to JSON. No proprietary formats or binary dependencies. The Episode format is an open schema and does not require OpenAdapt tooling at training or inference time.

---

## Typical Integration

1. Export your recordings to Episode format
2. Validate with `validate_episodes()`
3. Index with `DemoRetriever` for retrieval
4. Fine-tune with `train_from_json.py`
5. Evaluate on held-out tasks

All components work together—no glue code required.

---

## Quick Start

```python
from openadapt_ml.ingest import load_episodes
from openadapt_ml.schemas import validate_episodes, summarize_episodes

# 1. Load your exported data
episodes = load_episodes("exported_data/")

# 2. Validate
warnings = validate_episodes(episodes, check_images=True)

# 3. Train or use for retrieval
```

## Data Export Format

Your platform should export workflow recordings as JSON files following the Episode schema.

### Required Structure

```json
{
  "id": "workflow_123",
  "goal": "Submit expense report",
  "steps": [
    {
      "t": 0.0,
      "observation": {
        "image_path": "screenshots/step_0.png",
        "window_title": "Expense Portal",
        "app_name": "Chrome"
      },
      "action": {
        "type": "click",
        "x": 0.45,
        "y": 0.32
      }
    },
    {
      "t": 1.5,
      "observation": {
        "image_path": "screenshots/step_1.png"
      },
      "action": {
        "type": "type",
        "text": "Q4 Travel Expenses"
      }
    }
  ]
}
```

### Action Types

| Type | Required Fields | Description |
|------|----------------|-------------|
| `click` | `x`, `y` (0-1 normalized) | Single click |
| `double_click` | `x`, `y` | Double click |
| `right_click` | `x`, `y` | Right/context click |
| `type` | `text` | Type text |
| `key` | `key`, optional `modifiers` | Keyboard shortcut |
| `scroll` | `scroll_direction` | Scroll up/down |
| `drag` | `x`, `y`, `end_x`, `end_y` | Drag operation |
| `done` | (none) | Task complete marker |

### Coordinate System

- All coordinates must be **normalized to [0, 1]**
- `(0, 0)` = top-left corner
- `(1, 1)` = bottom-right corner
- Example: `(0.5, 0.5)` = center of screen

### Screenshots

- Save as PNG files
- Reference with relative or absolute paths in `image_path`
- Recommended: 1920x1080 or native resolution
- Must exist on disk when training

## Option 1: Demo-Conditioned Prompting (No Training)

Use existing workflow recordings to improve VLM performance immediately.

```python
from openadapt_ml.retrieval import DemoIndex, DemoRetriever
from openadapt_ml.ingest import load_episodes
from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo

# Load exported episodes
episodes = load_episodes("workflow_exports/")

# Build retrieval index
index = DemoIndex()
index.add_many(episodes)
index.build()

# Create retriever
retriever = DemoRetriever(index, domain_bonus=0.2)

# At runtime: find similar demo for new task
task = "Submit expense report for conference"
demos = retriever.retrieve(task, app_context="Chrome", top_k=3)

# Format best demo for VLM prompt
if demos:
    demo_text = format_episode_as_demo(demos[0], max_steps=10)
    prompt = f"{demo_text}\n\nNow perform: {task}"
```

**Validated result**: Demo-conditioning improves first-action accuracy from 33% to 100%.

## Option 2: Fine-Tuning

Train a custom model on your workflow data.

### Prepare Training Data

```bash
# Validate data first
python examples/train_from_json.py \
  --data workflow_exports/ \
  --validate-only \
  --check-images
```

### Train Model

```bash
python examples/train_from_json.py \
  --data workflow_exports/ \
  --output training_results/ \
  --config configs/qwen3vl_capture.yaml
```

### Fine-Tuning Approaches

| Approach | Training Input | Use Case |
|----------|---------------|----------|
| **Standard SFT** | (screenshot, task) → action | Baseline model |
| **OpenAdapt SFT** | (screenshot, task, demo) → action | Demo-aware model |

The OpenAdapt approach trains the model to *use* demonstrations, compounding with retrieval.

## Directory Structure

Recommended export structure:

```
workflow_exports/
├── workflows/
│   ├── expense_report.json
│   ├── expense_report_v2.json      # Multiple examples of same workflow
│   ├── expense_report_mobile.json  # Variant for different context
│   ├── timesheet_entry.json
│   └── leave_request.json
├── screenshots/
│   ├── expense_report/
│   │   ├── step_0.png
│   │   ├── step_1.png
│   │   └── ...
│   └── ...
└── metadata.json  # Optional: batch info, export date
```

**Multiple examples of similar workflows**: Having several recordings of the same workflow (e.g., different expense reports) improves retrieval quality and enables the model to generalize across variations.

## Schema Reference

Full schema: `openadapt_ml/schemas/sessions.py`

### Episode

```python
class Episode(BaseModel):
    id: str                    # Unique identifier
    goal: str                  # Task description
    steps: list[Step]          # Ordered list of steps
    summary: str | None        # Optional summary
    success: bool | None       # Whether task completed
    workflow_id: str | None    # Optional workflow reference
```

### Step

```python
class Step(BaseModel):
    t: float                   # Timestamp (seconds from start)
    observation: Observation   # Screen state
    action: Action             # Action taken
    thought: str | None        # Optional reasoning
```

### Observation

```python
class Observation(BaseModel):
    image_path: str | None        # Screenshot path
    window_title: str | None      # Active window title
    app_name: str | None          # Application name
    url: str | None               # For browser tasks
    accessibility_tree: str | None # Optional a11y tree
```

### Action

```python
class Action(BaseModel):
    type: str                  # Action type (click, type, etc.)
    x: float | None            # Normalized X (0-1)
    y: float | None            # Normalized Y (0-1)
    text: str | None           # For type actions
    key: str | None            # For key actions
    modifiers: list[str] | None # Ctrl, Shift, etc.
    scroll_direction: str | None
    end_x: float | None        # For drag
    end_y: float | None        # For drag
```

## Optional Metadata (Extension Pattern)

Episodes and Steps include a free-form `metadata` dict field. This allows attaching domain-specific annotations without modifying the core schema.

Common examples:

| Field | Purpose |
|-------|---------|
| `domain` | Business or application domain (improves retrieval) |
| `quality_label` | Training eligibility flag (`include`/`exclude`) |
| `segment_id` | Workflow segmentation key |
| `source_session_id` | Original recording reference |
| `verified` | Human-verified action flag |

```json
{
  "id": "workflow_123",
  "goal": "Submit expense report",
  "metadata": {
    "domain": "expense_reporting",
    "quality_label": "include",
    "segment_id": "expense_flow_v2"
  },
  "steps": [...]
}
```

## Validation

```python
from openadapt_ml.schemas import validate_episodes

warnings = validate_episodes(
    episodes,
    check_images=True,  # Verify screenshots exist
    strict=False,       # Warn vs error on issues
)

for w in warnings:
    print(f"Warning: {w}")
```

Common validation issues:
- Missing `image_path`
- Coordinates outside [0, 1] range
- Invalid action type
- Missing screenshots on disk

## Example Scripts

| Script | Purpose |
|--------|---------|
| `examples/train_from_json.py` | Validate + train from JSON |
| `examples/retrieval_with_capture.py` | Demo retrieval example |
| `examples/sample_data.json` | Example data format |

## Metrics & Evaluation

After training or configuring retrieval:

```python
# First-action accuracy (validated metric)
from openadapt_ml.experiments.demo_prompt.run_experiment import DemoPromptExperiment

exp = DemoPromptExperiment(provider="anthropic")
result = exp.run_with_demo(
    task="Submit expense report",
    screenshot_path="current_screen.png",
    demo_text=formatted_demo,
)
print(f"Predicted action: {result.action_parsed}")
```

## Support

- Schema questions: See `openadapt_ml/schemas/sessions.py`
- Training issues: Check `docs/cloud_gpu_training.md`
- Experiment results: See `docs/experiments/demo_conditioned_prompting_results.md`
