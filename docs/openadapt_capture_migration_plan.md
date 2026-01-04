# OpenAdapt-Capture to OpenAdapt-ML Migration Plan

## Executive Summary

This document provides a detailed migration plan for making **openadapt-capture** output data that can be directly ingested by **openadapt-ml**. The goal is to achieve seamless interoperability between the capture tool and the ML training pipeline.

## 1. Current State Analysis

### 1.1 OpenAdapt-Capture Schema (Source)

**Key Models:**

| Model | Description |
|-------|-------------|
| `Capture` | Session metadata (id, started_at, ended_at, platform, screen dimensions) |
| `Event` | Union type of all event types |
| `MouseClickEvent` | Single click with x, y, button, children |
| `MouseDoubleClickEvent` | Double click event |
| `MouseDragEvent` | Drag with x, y, dx, dy |
| `MouseScrollEvent` | Scroll with x, y, dx, dy |
| `KeyTypeEvent` | Typed text with children |
| `ScreenFrameEvent` | Video frame reference |
| `SemanticElementRef` | DOM element with role, name, xpath, bbox |
| `BrowserClickEvent` | Browser click with element reference |

### 1.2 OpenAdapt-ML Schema (Target)

**Key Models:**

| Model | Description |
|-------|-------------|
| `Episode` | Complete trajectory with episode_id, instruction, steps, success |
| `Step` | Single step with step_index, observation, action, reasoning |
| `Action` | Action with type, coordinates, text, element, etc. |
| `Observation` | State with screenshot_path, a11y_tree, dom, window_title |
| `ActionType` | Enum with 24 action types |
| `UIElement` | role, name, value, bounds, xpath, selector |

## 2. Schema Comparison: Event Type Mapping

| openadapt-capture | openadapt-ml | Notes |
|-------------------|--------------|-------|
| `mouse.singleclick` | `ActionType.CLICK` | Direct mapping |
| `mouse.doubleclick` | `ActionType.DOUBLE_CLICK` | Direct mapping |
| `mouse.drag` | `ActionType.DRAG` | Needs coord transform |
| `mouse.scroll` | `ActionType.SCROLL` | Needs direction extraction |
| `key.type` | `ActionType.TYPE` | Direct mapping |
| `key.down` | `ActionType.KEY` | For special keys |
| `browser.click` | `ActionType.CLICK` | With element reference |
| `browser.navigate` | `ActionType.GOTO` | With URL |
| (none) | `ActionType.DONE` | Terminal action needed |

## 3. Migration Tasks

### Phase 1: Add Export Functionality to openadapt-capture

**Task 1.1: Add Episode Export Method**

Add a `to_episode()` method to the `CaptureSession` class:

```python
def to_episode(
    self,
    instruction: str | None = None,
    output_dir: Path | None = None,
    include_moves: bool = False,
) -> "Episode":
    """Export capture as openadapt-ml Episode."""
```

**Task 1.2: Create Schema Converters Module**

Create `openadapt_capture/converters.py` with:

```python
def capture_event_to_action(event: ActionEvent, screen_size: tuple[int, int]) -> Action:
    """Convert capture event to openadapt-ml Action."""

def browser_event_to_action(event: BrowserEvent) -> Action:
    """Convert browser event to openadapt-ml Action."""

def semantic_element_to_ui_element(ref: SemanticElementRef) -> UIElement:
    """Convert browser element reference to UIElement."""
```

**Task 1.3: Add Coordinate Normalization**

```python
def normalize_coordinates(x, y, screen_width, screen_height, pixel_ratio=1.0):
    """Convert pixel coordinates to normalized (0-1) range."""
    logical_x = x / pixel_ratio
    logical_y = y / pixel_ratio
    return (logical_x / screen_width, logical_y / screen_height)
```

### Phase 2: Enhance Observation Data

**Task 2.1: Add Window/App Context Capture**

Add fields to capture window title, app name, and minimal a11y info during recording.

**Task 2.2: Integrate Browser DOM with Observations**

Associate DOM snapshots with action timestamps.

### Phase 3: Update openadapt-ml Ingest Module

**Task 3.1: Update `openadapt_ml/ingest/capture.py`**

Add handling for browser events, element references, and normalized coordinates.

**Task 3.2: Add Browser Event Support**

Convert browser events to openadapt-ml Actions with element references.

### Phase 4: Add CLI Export Command

```bash
openadapt-capture export --format openadapt-ml --output ./episode.json ./my_capture/
```

## 4. Implementation Priority

### High Priority (Must Have)
1. Event type mapping (mouse, keyboard)
2. Coordinate normalization
3. Screenshot extraction
4. Episode/Step structure generation
5. Terminal DONE action addition

### Medium Priority (Should Have)
1. Browser event integration
2. Element reference preservation
3. DOM snapshot association
4. CLI export command

### Lower Priority (Nice to Have)
1. Accessibility tree capture
2. Window/app context
3. Audio integration

## 5. Critical Files

1. **`openadapt_ml/schema/episode.py`** - Target schema definition
2. **`openadapt_ml/ingest/capture.py`** - Existing adapter (needs enhancement)
3. **`openadapt_capture/events.py`** - Source event schemas
4. **`openadapt_capture/browser_events.py`** - Browser event schemas
5. **`openadapt_capture/capture.py`** - Where `to_episode()` should be added
