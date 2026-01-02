# Episode Schema

Canonical format for GUI trajectory data. Enables interoperability across training pipelines, benchmarks, and recording tools.

## Installation

```bash
pip install openadapt-ml
# or
uv add openadapt-ml
```

## Quick Start

```python
from openadapt_ml.schema import Episode, Step, Action, Observation, ActionType

episode = Episode(
    episode_id="demo_001",
    instruction="Open Settings and enable Dark Mode",
    steps=[
        Step(
            step_index=0,
            observation=Observation(screenshot_path="screenshots/step_0.png"),
            action=Action(
                type=ActionType.CLICK,
                coordinates={"x": 512, "y": 384},
            ),
            reasoning="Click on Settings icon",
        ),
        Step(
            step_index=1,
            observation=Observation(screenshot_path="screenshots/step_1.png"),
            action=Action(
                type=ActionType.CLICK,
                coordinates={"x": 200, "y": 150},
            ),
            reasoning="Click on Display settings",
        ),
    ],
    success=True,
)

# Save to JSON
from openadapt_ml.schema import save_episode
save_episode(episode, "episode.json")
```

## Schema Overview

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Unique identifier |
| `instruction` | string | Natural language task description |
| `steps` | array | Sequence of Step objects |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version (default: "1.0.0") |
| `task_id` | string | External task identifier |
| `goal` | string | Detailed goal (if different from instruction) |
| `success` | boolean | Whether task completed successfully |
| `source` | enum | Source benchmark (waa, webarena, osworld, human, synthetic) |
| `metadata` | object | Extension point for custom data |

## Action Types

24 supported action types:

**Mouse**: `click`, `double_click`, `right_click`, `drag`, `scroll`, `hover`

**Keyboard**: `type`, `key`, `hotkey`

**Navigation**: `goto`, `back`, `forward`, `refresh`

**System**: `open_app`, `close_app`, `select_monitor`, `window_focus`, `window_resize`, `window_move`

**Meta**: `done`, `fail`, `wait`, `screenshot`

## Coordinate Systems

Two coordinate systems are supported (can use both simultaneously):

```python
# Pixel coordinates (absolute)
Action(
    type=ActionType.CLICK,
    coordinates={"x": 512, "y": 384}
)

# Normalized coordinates (0.0-1.0, resolution-independent)
Action(
    type=ActionType.CLICK,
    normalized_coordinates=(0.5, 0.375)
)
```

**When to use which:**
- **Pixel coordinates**: When screen resolution is fixed/known
- **Normalized coordinates**: For resolution-independent recordings, cross-device transfer

## Validation

```python
from openadapt_ml.schema import validate_episode

data = {"episode_id": "test", "instruction": "Do something", "steps": []}
is_valid, error = validate_episode(data)

if not is_valid:
    print(f"Validation error: {error}")
```

## Format Conversion

Convert from other formats:

```python
from openadapt_ml.schema.converters import from_waa_trajectory, to_waa_trajectory

# From Windows Agent Arena format
trajectory = [
    {"screenshot_path": "step_0.png", "action": "pyautogui.click(100, 200)"},
    {"screenshot_path": "step_1.png", "action": "pyautogui.write('hello')"},
]
task_info = {"id": "task_001", "instruction": "Type hello"}

episode = from_waa_trajectory(trajectory, task_info)

# Back to WAA format
trajectory, task_info = to_waa_trajectory(episode)
```

## JSON Schema

For external tools (TypeScript codegen, JSON validators):

```python
from openadapt_ml.schema import export_json_schema
export_json_schema("episode.schema.json")
```

Or use the pre-generated schema: [`episode.schema.json`](./episode.schema.json)

## Extension Points

Use `raw` and `metadata` fields for custom data without modifying the schema:

```python
Episode(
    episode_id="demo_001",
    instruction="...",
    steps=[...],
    metadata={
        "recording_tool": "my-recorder",
        "screen_resolution": [1920, 1080],
        "custom_field": "any value",
    },
)

Action(
    type=ActionType.CLICK,
    coordinates={"x": 100, "y": 200},
    raw={
        "original_format": "pyautogui.click(100, 200)",
        "confidence": 0.95,
    },
)
```

## Version History

### 1.0.0 (2026-01)
- Initial release
- Core models: Episode, Step, Action, Observation
- 24 action types
- Pixel and normalized coordinate support
- WAA format converter
