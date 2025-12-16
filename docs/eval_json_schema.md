# Evaluation JSON Schema Documentation

## Overview

This document defines the stable JSON schema used by all evaluation outputs in OpenAdapt-ML. The schema is designed to be:

- **Stable**: Forward and backward compatible across versions
- **Extensible**: New metrics can be added without breaking existing tools
- **Self-documenting**: Contains metadata about evaluation config and backend
- **Tool-friendly**: Parseable by plotting scripts, analysis tools, and dashboards

## Schema Version

**Current Version**: v1.0

**Last Updated**: December 2024

## Top-Level Structure

All evaluation JSONs contain these top-level fields:

```json
{
  "config_path": "configs/qwen3vl_synthetic_dev.yaml",
  "backend": "qwen3",
  "dsl_mode": "coord",
  "metrics": { ... }
}
```

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `config_path` | string | Yes | Path to YAML config used for evaluation |
| `backend` | string | Yes | Model backend: `qwen3`, `qwen2_5`, `claude`, `openai`, `dummy` |
| `dsl_mode` | string | No | DSL mode: `coord` (coordinates) or `som` (Set-of-Marks). Default: `coord` |
| `metrics` | object | Yes | Evaluation metrics (see below) |

## Metrics Object

The `metrics` object contains all computed evaluation metrics:

```json
{
  "metrics": {
    "num_episodes": 32,
    "num_steps": 224,
    "action_type_accuracy": 0.469,
    "mean_coord_error": 0.051,
    "coord_error_count": 19,
    "episode_success_rate": 0.0,
    "click_hit_rate": 0.850,
    "bbox_hit_rate": 0.900,
    "mean_episode_progress": 0.532,
    "mean_episode_step_score": 0.489,
    "weak_episode_success_rate": 0.125,
    "state_success_rate": null,
    "element_accuracy": null
  }
}
```

### Primary Metrics (Always Present)

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `num_episodes` | integer | No | Total number of episodes evaluated |
| `num_steps` | integer | No | Total number of steps across all episodes |
| `action_type_accuracy` | float | No | Fraction of steps with correct action type (0.0-1.0) |
| `mean_coord_error` | float | Yes | Mean normalized L2 distance for CLICK actions. Null if no clicks with coordinates |
| `coord_error_count` | integer | No | Number of CLICK actions with valid coordinates |
| `episode_success_rate` | float | No | Fraction of episodes with all steps correct (strict) |
| `click_hit_rate` | float | Yes | Fraction of clicks within 5% radius of target center. Null if no clicks with coordinates |

### Auxiliary Metrics (Optional)

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `bbox_hit_rate` | float | Yes | Fraction of clicks landing within element bounding box. Null if bboxes unavailable |
| `mean_episode_progress` | float | Yes | Average fraction of correct steps per episode (partial credit) |
| `mean_episode_step_score` | float | Yes | Average "full step correctness" (type match + click hit when applicable) |
| `weak_episode_success_rate` | float | Yes | Fraction of episodes achieving all semantic milestones |
| `state_success_rate` | float | Yes | Fraction of episodes with successful terminal state. Null if not tracked |
| `element_accuracy` | float | Yes | Fraction of correct element selections (SoM mode only). Null for coordinate mode |

## Null Value Semantics

Metrics are set to `null` when they are not applicable or cannot be computed:

- **`mean_coord_error`**: Null when no CLICK actions have valid coordinates (e.g., SoM mode, or model never predicts clicks)
- **`click_hit_rate`**: Null when no CLICK actions have valid coordinates
- **`bbox_hit_rate`**: Null when ground truth actions lack bounding boxes
- **`element_accuracy`**: Null in coordinate mode (not SoM)
- **`state_success_rate`**: Null when terminal state checking is not implemented

**Important**: Null values are not errors. They indicate the metric is inapplicable to the current evaluation context.

## Schema Evolution

### Adding New Metrics

New metrics can be added to the `metrics` object without breaking existing tools:

1. Add the new field with a default value (typically 0.0 or null)
2. Update documentation in this file
3. Update plotting scripts to handle the new metric (optional)

**Example**: Adding `click_distance_50th_percentile`:

```json
{
  "metrics": {
    "num_episodes": 32,
    "action_type_accuracy": 0.469,
    ...
    "click_distance_50th_percentile": 0.042
  }
}
```

Existing tools will ignore unknown metrics, ensuring backward compatibility.

### Removing Metrics

**Never remove metrics from the schema.** Instead:

1. Mark the metric as deprecated in documentation
2. Continue outputting the metric with a sentinel value (e.g., -1.0 or null)
3. Update tools to ignore the deprecated metric

## Example Evaluations

### Coordinate Mode (Standard)

```json
{
  "config_path": "configs/qwen3vl_synthetic_dev.yaml",
  "backend": "qwen3",
  "metrics": {
    "num_episodes": 32,
    "num_steps": 224,
    "action_type_accuracy": 0.22321428571428573,
    "mean_coord_error": 0.03184812009573501,
    "coord_error_count": 19,
    "episode_success_rate": 0.0,
    "click_hit_rate": 0.9473684210526315
  }
}
```

### Set-of-Marks (SoM) Mode

```json
{
  "config_path": "configs/qwen3vl_synthetic_registration_som.yaml",
  "backend": "qwen3",
  "dsl_mode": "som",
  "metrics": {
    "num_episodes": 32,
    "num_steps": 384,
    "action_type_accuracy": 1.0,
    "mean_coord_error": null,
    "coord_error_count": 0,
    "episode_success_rate": 1.0,
    "click_hit_rate": null,
    "bbox_hit_rate": null,
    "mean_episode_progress": 1.0,
    "mean_episode_step_score": 1.0,
    "weak_episode_success_rate": 0.0,
    "state_success_rate": null,
    "element_accuracy": 1.0
  }
}
```

### API Backend (Claude)

```json
{
  "config_path": "configs/qwen3vl_synthetic_dev.yaml",
  "backend": "claude",
  "dsl_mode": "som",
  "metrics": {
    "num_episodes": 32,
    "num_steps": 192,
    "action_type_accuracy": 1.0,
    "mean_coord_error": null,
    "coord_error_count": 0,
    "episode_success_rate": 1.0,
    "click_hit_rate": null,
    "bbox_hit_rate": null,
    "mean_episode_progress": 1.0,
    "mean_episode_step_score": 1.0,
    "weak_episode_success_rate": 0.0,
    "state_success_rate": null,
    "element_accuracy": 1.0
  }
}
```

## Validation

### Python Validation (Pydantic)

```python
from typing import Optional
from pydantic import BaseModel, Field

class EvalMetrics(BaseModel):
    num_episodes: int = Field(ge=0)
    num_steps: int = Field(ge=0)
    action_type_accuracy: float = Field(ge=0.0, le=1.0)
    mean_coord_error: Optional[float] = Field(default=None, ge=0.0)
    coord_error_count: int = Field(ge=0)
    episode_success_rate: float = Field(ge=0.0, le=1.0)
    click_hit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bbox_hit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mean_episode_progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mean_episode_step_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    weak_episode_success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    state_success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    element_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class EvalOutput(BaseModel):
    config_path: str
    backend: str
    dsl_mode: str = "coord"
    metrics: EvalMetrics
```

### Loading and Parsing

```python
import json
from pathlib import Path

def load_eval_metrics(json_path: Path) -> dict:
    """Load evaluation metrics from JSON file."""
    with json_path.open("r") as f:
        data = json.load(f)
    return data.get("metrics", data)

# Usage
metrics = load_eval_metrics(Path("experiments/qwen_login/eval_base.json"))
print(f"Action accuracy: {metrics['action_type_accuracy']:.1%}")
print(f"Click hit rate: {metrics.get('click_hit_rate', 'N/A')}")
```

## Existing Evaluations

### Verified JSON Outputs

The following evaluation outputs have been verified to conform to the stable schema:

**Standard Coordinate Mode**:
- `/experiments/qwen_login/2b_dev/eval/eval_base.json` - Qwen3-VL-2B base
- `/experiments/qwen_login/2b_dev/eval/eval_ft.json` - Qwen3-VL-2B fine-tuned
- `/experiments/qwen_login/8b_hero/eval/eval_qwen3_8b_base_login_hardened.json` - Qwen3-VL-8B base
- `/experiments/qwen_login/8b_hero/eval/eval_qwen3_8b_ft_login_hardened.json` - Qwen3-VL-8B fine-tuned

**Set-of-Marks Mode**:
- `/experiments/qwen_login/registration_som_eval.json` - Qwen3-VL-2B FT on registration (100% accuracy)
- `/experiments/qwen_login/som_v3/eval_claude_som_fixed.json` - Claude Sonnet 4.5 SoM mode
- `/experiments/qwen_login/som_v3/eval_gpt_som_fixed.json` - GPT-5.1 SoM mode

All files follow the same schema structure and are compatible with `plot_eval_metrics.py`.

## Tools and Scripts

### Plotting

The plotting system automatically handles the stable schema:

```bash
python -m openadapt_ml.evals.plot_eval_metrics \
  --files eval_base.json eval_ft.json \
  --labels "Base" "Fine-tuned" \
  --output comparison.png
```

The plotter:
- Extracts metrics from `data.get("metrics", data)` for backward compatibility
- Handles null values gracefully (treats as 0.0 or omits from plot)
- Supports arbitrary combinations of backends and configs

### Aggregation

Example script for computing aggregate statistics:

```python
import json
from pathlib import Path
from typing import List, Dict

def aggregate_evals(eval_paths: List[Path]) -> Dict[str, float]:
    """Compute aggregate statistics across multiple evaluations."""
    all_metrics = []
    for path in eval_paths:
        with path.open("r") as f:
            data = json.load(f)
            all_metrics.append(data["metrics"])

    # Compute means
    agg = {
        "mean_action_accuracy": sum(m["action_type_accuracy"] for m in all_metrics) / len(all_metrics),
        "mean_click_hit_rate": sum(m.get("click_hit_rate", 0.0) or 0.0 for m in all_metrics) / len(all_metrics),
        "mean_episode_success": sum(m["episode_success_rate"] for m in all_metrics) / len(all_metrics),
    }
    return agg
```

## Migration Guide

If you have existing evaluation JSONs that don't match this schema:

### Legacy Format (Pre-v1.0)

```json
{
  "action_type_accuracy": 0.469,
  "click_hit_rate": 0.850
}
```

**Migration**:
```python
def migrate_legacy_eval(old_json: dict) -> dict:
    """Migrate legacy eval JSON to v1.0 schema."""
    if "metrics" not in old_json:
        # Wrap metrics in "metrics" object
        old_json = {"metrics": old_json}

    if "config_path" not in old_json:
        old_json["config_path"] = "unknown"

    if "backend" not in old_json:
        old_json["backend"] = "unknown"

    return old_json
```

## Future Extensions

Potential future additions to the schema:

1. **Versioning**: Add `schema_version: "1.0"` for explicit version tracking
2. **Timestamps**: Add `eval_timestamp` and `training_timestamp` for reproducibility
3. **Hardware info**: Add `device: "cuda"`, `gpu_model`, `memory_gb` for performance analysis
4. **Checkpoints**: Add `checkpoint_path` to link evals to specific model weights
5. **Per-action metrics**: Add `metrics_by_action_type: {...}` for granular analysis
6. **Confidence intervals**: Add `action_type_accuracy_ci: [0.44, 0.50]` for statistical rigor

These extensions will be added in a backward-compatible manner.

## Contact

For questions about the eval JSON schema or to propose extensions:

- GitHub Issues: https://github.com/OpenAdaptAI/openadapt-ml/issues
- Pull Requests: https://github.com/OpenAdaptAI/openadapt-ml/pulls
