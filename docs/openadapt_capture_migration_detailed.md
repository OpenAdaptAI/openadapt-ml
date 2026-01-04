# OpenAdapt Capture to OpenAdapt-ML Migration Guide

**Version**: 1.0.0
**Date**: January 2026
**Status**: Implementation Ready

This document provides a comprehensive migration plan for converting openadapt-capture recordings to the openadapt-ml Episode format, including integration of legacy openadapt.record accessibility tree data.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Source Systems Analysis](#2-source-systems-analysis)
3. [Field-by-Field Migration Analysis](#3-field-by-field-migration-analysis)
4. [Staged Implementation Plan](#4-staged-implementation-plan)
5. [Code Changes Required](#5-code-changes-required)
6. [Testing Strategy](#6-testing-strategy)
7. [Appendices](#7-appendices)

---

## 1. Executive Summary

### 1.1 Scope

This migration consolidates three data sources into the unified openadapt-ml `Episode` schema:

| Source | Description | Key Data |
|--------|-------------|----------|
| **openadapt-capture** | Modern capture library | Events, screenshots, video, audio |
| **openadapt.record (legacy)** | Original OpenAdapt recording | Accessibility trees, window state |
| **Chrome Extension** | Browser event capture | DOM, element references, selectors |

### 1.2 Key Decisions

Based on `docs/capture_format_decision.md`, we adopt **Option B: Conversion Layer in openadapt-ml**:

- Converters live in `openadapt_ml/ingest/`
- Raw data preserved in `raw` and `metadata` fields
- Pydantic validation at conversion time
- Schema evolution handled in converters

### 1.3 Value Proposition

The unified Episode format enables:
- **Training**: Consistent format for VLM fine-tuning
- **Benchmarking**: Interoperability with WAA, WebArena, OSWorld
- **Demo Retrieval**: Episode-level similarity for demo selection
- **Replay**: Structured action sequences with element grounding

---

## 2. Source Systems Analysis

### 2.1 Legacy OpenAdapt.Record (openadapt/models.py)

#### 2.1.1 Recording Model

| Field | Type | Description | ML Schema Target |
|-------|------|-------------|------------------|
| `id` | Integer | Primary key | `episode_id` (as string) |
| `timestamp` | Float | Unix start time | `created_at` |
| `monitor_width` | Integer | Screen width | `Observation.screen_size[0]` |
| `monitor_height` | Integer | Screen height | `Observation.screen_size[1]` |
| `double_click_interval_seconds` | Float | System setting | `metadata.double_click_interval` |
| `double_click_distance_pixels` | Float | System setting | `metadata.double_click_distance` |
| `platform` | String | OS identifier | `environment` |
| `task_description` | String | User goal | `instruction` |
| `video_start_time` | Float | Video sync | `metadata.video_start_time` |
| `config` | JSON | Recording config | `metadata.config` |

#### 2.1.2 ActionEvent Model (Accessibility Tree Context)

The **most valuable** legacy data is in `ActionEvent`:

| Field | Type | Description | ML Schema Target |
|-------|------|-------------|------------------|
| `name` | String | Action type ("click", "type", etc.) | `Action.type` |
| `mouse_x`, `mouse_y` | Float | Mouse position | `Action.coordinates` |
| `mouse_dx`, `mouse_dy` | Float | Scroll/drag delta | `Action.scroll_amount`, `end_coordinates` |
| `mouse_button_name` | String | Button ("left", "right") | `Action.raw.button` |
| `key_name`, `key_char`, `key_vk` | String | Key identifiers | `Action.key`, `Action.text` |
| `element_state` | JSON | **Accessibility element at cursor** | `Action.element` |
| `active_segment_description` | String | Target element text | `Action.element.name` |
| `available_segment_descriptions` | String[] | Nearby elements | `Observation.raw.available_elements` |
| `children` | ActionEvent[] | Merged events | `Action.raw.children` |

#### 2.1.3 WindowEvent Model (Full Accessibility Tree)

| Field | Type | Description | ML Schema Target |
|-------|------|-------------|------------------|
| `title` | String | Window title | `Observation.window_title` |
| `left`, `top`, `width`, `height` | Integer | Window bounds | `Observation.raw.window_bounds` |
| `window_id` | String | OS window ID | `Observation.raw.window_id` |
| `state` | JSON | **Full accessibility tree** | `Observation.a11y_tree` |
| `state.data` | JSON | Nested element hierarchy | `Observation.a11y_tree` |
| `state.meta` | JSON | Window metadata | `Observation.raw.window_meta` |

**Accessibility Tree Structure** (macOS example from `_macos.py`):

```json
{
  "AXRole": "AXWindow",
  "AXTitle": "System Preferences",
  "AXPosition": {"x": 100, "y": 100, "type": "CGPoint"},
  "AXSize": {"w": 800, "h": 600, "type": "CGSize"},
  "AXChildren": [
    {
      "AXRole": "AXButton",
      "AXTitle": "Night Shift",
      "AXDescription": "Toggle Night Shift",
      "AXPosition": {"x": 200, "y": 150},
      "AXSize": {"w": 80, "h": 24}
    }
  ]
}
```

#### 2.1.4 BrowserEvent Model

| Field | Type | Description | ML Schema Target |
|-------|------|-------------|------------------|
| `message` | JSON | Chrome extension payload | `Observation.dom` |
| `message.visibleHTMLString` | String | Page HTML | `Observation.dom` |
| `message.targetId` | String | Clicked element ID | `Action.element.element_id` |
| `message.url` | String | Page URL | `Observation.url` |

### 2.2 OpenAdapt-Capture (openadapt_capture/)

#### 2.2.1 Capture Session (storage.py)

| Field | Type | Description | ML Schema Target |
|-------|------|-------------|------------------|
| `id` | String | UUID-8 | `episode_id` |
| `started_at` | Float | Unix timestamp | `created_at` |
| `ended_at` | Float | Unix timestamp | `metadata.ended_at` |
| `platform` | String | OS identifier | `environment` |
| `screen_width`, `screen_height` | Integer | Screen size | `Observation.screen_size` |
| `pixel_ratio` | Float | Display scaling | `metadata.pixel_ratio` |
| `task_description` | String | User goal | `instruction` |
| `double_click_interval_seconds` | Float | System setting | `metadata.double_click_interval` |
| `video_start_time` | Float | Video sync | `metadata.video_start_time` |
| `audio_start_time` | Float | Audio sync | `metadata.audio_start_time` |

#### 2.2.2 Event Types (events.py)

| Event Type | Fields | ML Schema Target |
|------------|--------|------------------|
| `mouse.singleclick` | `x, y, button` | `Action(type=CLICK, coordinates={x,y})` |
| `mouse.doubleclick` | `x, y, button` | `Action(type=DOUBLE_CLICK, coordinates={x,y})` |
| `mouse.drag` | `x, y, dx, dy, button` | `Action(type=DRAG, start/end_coordinates)` |
| `mouse.scroll` | `x, y, dx, dy` | `Action(type=SCROLL, scroll_direction/amount)` |
| `key.type` | `text, children` | `Action(type=TYPE, text=text)` |
| `key.down/up` | `key_name, key_char` | `Action(type=KEY, key=key_name)` |

#### 2.2.3 Browser Events (browser_events.py)

| Model | Fields | ML Schema Target |
|-------|--------|------------------|
| `BrowserClickEvent` | `client_x/y, page_x/y, element` | `Action.coordinates`, `Action.element` |
| `SemanticElementRef` | `role, name, bbox, xpath, css_selector` | `UIElement` |
| `ElementState` | `enabled, focused, visible, value` | `UIElement.raw.state` |
| `DOMSnapshot` | `html, visible_elements, url` | `Observation.dom`, `Observation.a11y_tree` |

### 2.3 OpenAdapt-ML Episode Schema (schema/episode.py)

Target schema version: **1.0.0**

#### 2.3.1 Episode

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | String | Yes (default) | "1.0.0" |
| `episode_id` | String | Yes | Unique identifier |
| `task_id` | String | No | Benchmark task ID |
| `instruction` | String | Yes | Natural language goal |
| `goal` | String | No | Detailed goal |
| `steps` | Step[] | Yes | Action sequence |
| `success` | Boolean | No | Task outcome |
| `final_reward` | Float | No | Score |
| `source` | BenchmarkSource | No | Origin (human, waa, etc.) |
| `source_file` | String | No | Original file path |
| `created_at` | DateTime | No | Creation time |
| `agent_model` | String | No | Model that generated |
| `environment` | String | No | Platform info |
| `tags` | String[] | No | Categories |
| `metadata` | Dict | No | Extension data |

#### 2.3.2 Step

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step_index` | Integer | Yes | 0-indexed position |
| `observation` | Observation | Yes | State before action |
| `action` | Action | Yes | Action taken |
| `reasoning` | String | No | Chain-of-thought |
| `reward` | Float | No | Step reward |
| `done` | Boolean | No | Episode terminated |
| `timestamp` | Float | No | Unix timestamp |
| `duration_ms` | Integer | No | Step duration |

#### 2.3.3 Observation

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `screenshot_path` | String | No | Image file path |
| `screenshot_base64` | String | No | Inline image |
| `a11y_tree` | Dict | No | **Accessibility tree** |
| `dom` | String | No | HTML snapshot |
| `window_title` | String | No | Active window |
| `app_name` | String | No | Application name |
| `url` | String | No | Browser URL |
| `screen_size` | Tuple[int,int] | No | Dimensions |
| `focused_element` | UIElement | No | Current focus |
| `timestamp` | Float | No | Capture time |
| `raw` | Dict | No | Original data |

#### 2.3.4 Action

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | ActionType | Yes | Action type enum |
| `coordinates` | Coordinates | No | Pixel position |
| `start_coordinates` | Coordinates | No | Drag start |
| `end_coordinates` | Coordinates | No | Drag end |
| `scroll_direction` | String | No | up/down/left/right |
| `scroll_amount` | Integer | No | Pixels |
| `text` | String | No | Typed text |
| `key` | String | No | Key name |
| `modifiers` | String[] | No | Ctrl/Alt/Shift/Meta |
| `element` | UIElement | No | Target element |
| `url` | String | No | Navigation URL |
| `app_name` | String | No | Application name |
| `duration` | Float | No | Wait duration |
| `monitor_id` | Integer | No | Display ID |
| `window_title` | String | No | Window focus |
| `normalized_coordinates` | Tuple[float,float] | No | 0-1 range |
| `normalized_start` | Tuple[float,float] | No | Drag start 0-1 |
| `normalized_end` | Tuple[float,float] | No | Drag end 0-1 |
| `raw` | Dict | No | Original action data |

#### 2.3.5 UIElement

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | String | No | Button, textbox, etc. |
| `name` | String | No | Accessible name |
| `value` | String | No | Current value |
| `bounds` | BoundingBox | No | Position/size |
| `element_id` | String | No | Unique ID |
| `xpath` | String | No | XPath selector |
| `selector` | String | No | CSS selector |
| `automation_id` | String | No | Windows automation ID |

---

## 3. Field-by-Field Migration Analysis

### 3.1 Capture to Episode Mapping

Current implementation: `openadapt_ml/ingest/capture.py`

#### 3.1.1 Already Implemented

| Source (openadapt-capture) | Target (Episode) | Status |
|---------------------------|------------------|--------|
| `capture.id` | `episode_id` | Done |
| `capture.task_description` | `instruction` | Done |
| `capture.screen_size` | `screen_width/height` for normalization | Done |
| `capture.started_at` | Timestamp baseline | Done |
| Mouse events | `Action` with coordinates | Done |
| Keyboard events | `Action` with text | Done |
| Screenshots | `Observation.screenshot_path` | Done |
| Scroll events | `Action.scroll_direction/amount` | Done |
| Drag events | `Action` with end coordinates | Done |

#### 3.1.2 Missing (Phase 2-5 Work)

| Source | Target | Priority | Notes |
|--------|--------|----------|-------|
| Browser events | `Observation.dom`, `Action.element` | High | Chrome extension integration |
| Audio transcription | `metadata.transcript` | Medium | Voice commands |
| Video file | Linked in metadata | Low | Space optimization |
| Window bounds | `Observation.raw.window_bounds` | Medium | For cropping |

### 3.2 Legacy OpenAdapt to Episode Mapping

**New work required** - no current converter exists.

#### 3.2.1 Critical Data (Must Have)

| Source (openadapt.record) | Target (Episode) | Implementation |
|--------------------------|------------------|----------------|
| `WindowEvent.state.data` | `Observation.a11y_tree` | Parse AX hierarchy |
| `ActionEvent.element_state` | `Action.element` | Extract role/name/bounds |
| `ActionEvent.active_segment_description` | `Action.element.name` | Direct mapping |
| `BrowserEvent.message.visibleHTMLString` | `Observation.dom` | Store HTML |

#### 3.2.2 Accessibility Tree Normalization

The legacy `state.data` uses platform-specific keys:

**macOS (AX prefix)**:
```python
def normalize_ax_element(ax_data: dict) -> UIElement:
    return UIElement(
        role=ax_data.get("AXRole", "").replace("AX", ""),
        name=ax_data.get("AXTitle") or ax_data.get("AXDescription", ""),
        bounds=BoundingBox(
            x=ax_data.get("AXPosition", {}).get("x", 0),
            y=ax_data.get("AXPosition", {}).get("y", 0),
            width=ax_data.get("AXSize", {}).get("w", 0),
            height=ax_data.get("AXSize", {}).get("h", 0),
        ) if ax_data.get("AXPosition") else None,
        automation_id=ax_data.get("AXIdentifier"),
    )
```

**Windows (UIA)**:
```python
def normalize_uia_element(uia_data: dict) -> UIElement:
    return UIElement(
        role=uia_data.get("ControlType", "").replace("ControlType.", ""),
        name=uia_data.get("Name", ""),
        automation_id=uia_data.get("AutomationId"),
        bounds=BoundingBox(
            x=uia_data.get("BoundingRectangle", {}).get("left", 0),
            y=uia_data.get("BoundingRectangle", {}).get("top", 0),
            width=uia_data.get("BoundingRectangle", {}).get("width", 0),
            height=uia_data.get("BoundingRectangle", {}).get("height", 0),
        ) if uia_data.get("BoundingRectangle") else None,
    )
```

### 3.3 Browser Events to Episode Mapping

From `openadapt_capture/browser_events.py`:

| Browser Event | Episode Field | Notes |
|--------------|---------------|-------|
| `BrowserClickEvent.element` | `Action.element` | Full semantic ref |
| `SemanticElementRef.role` | `UIElement.role` | ARIA role |
| `SemanticElementRef.name` | `UIElement.name` | Accessible name |
| `SemanticElementRef.xpath` | `UIElement.xpath` | For replay |
| `SemanticElementRef.css_selector` | `UIElement.selector` | Minimal CSS |
| `SemanticElementRef.bbox` | `UIElement.bounds` | Position/size |
| `DOMSnapshot.html` | `Observation.dom` | Full HTML |
| `DOMSnapshot.visible_elements` | `Observation.a11y_tree` | As element list |
| `BrowserNavigationEvent.url` | `Observation.url` | Current page |

### 3.4 Gap Analysis Summary

| Category | Available | Missing | Priority |
|----------|-----------|---------|----------|
| **Screenshots** | Full support | - | Complete |
| **Mouse Actions** | Coordinates, type | Element grounding | High |
| **Keyboard Actions** | Text, key names | Modifier combinations | Low |
| **Accessibility Trees** | Legacy only | Integration | Critical |
| **DOM/HTML** | Chrome extension | Merge with native | High |
| **Element Selectors** | Browser events | Native app selectors | Medium |
| **Audio/Video** | Raw files | Transcription link | Low |

---

## 4. Staged Implementation Plan

### Phase 1: Basic Conversion (Current State - Complete)

**Files**: `openadapt_ml/ingest/capture.py`

**Scope**:
- Convert openadapt-capture events to Episode
- Extract screenshots from video
- Normalize coordinates (0-1 range)
- Map event types to ActionType enum

**Validation**:
```bash
uv run python -c "
from openadapt_ml.ingest.capture import capture_to_episode
episode = capture_to_episode('/path/to/turn-off-nightshift')
print(f'Episode: {episode.episode_id}')
print(f'Steps: {episode.num_steps}')
print(f'Actions: {[s.action.type for s in episode.steps]}')
"
```

### Phase 2: Browser Event Integration (Priority: High)

**Timeline**: 1-2 weeks

**Files to Modify**:
- `openadapt_ml/ingest/capture.py` - Add browser event support
- `openadapt_ml/schema/episode.py` - No changes needed (schema ready)

**New Files**:
- `openadapt_ml/ingest/browser.py` - Browser event converter

**Implementation**:

```python
# openadapt_ml/ingest/browser.py
"""Convert browser events from openadapt-capture to Episode format."""

from openadapt_capture.browser_events import (
    BrowserClickEvent,
    BrowserKeyEvent,
    BrowserInputEvent,
    SemanticElementRef,
    DOMSnapshot,
)
from openadapt_ml.schema import UIElement, BoundingBox


def semantic_ref_to_ui_element(ref: SemanticElementRef) -> UIElement:
    """Convert browser SemanticElementRef to UIElement."""
    return UIElement(
        role=ref.role,
        name=ref.name,
        bounds=BoundingBox(
            x=int(ref.bbox.x),
            y=int(ref.bbox.y),
            width=int(ref.bbox.width),
            height=int(ref.bbox.height),
        ) if ref.bbox else None,
        xpath=ref.xpath,
        selector=ref.css_selector,
        element_id=ref.id,
    )


def dom_snapshot_to_a11y_tree(snapshot: DOMSnapshot) -> dict:
    """Convert DOMSnapshot to accessibility tree format."""
    return {
        "type": "browser_dom",
        "url": snapshot.url,
        "title": snapshot.title,
        "elements": [
            {
                "id": elem.som_id,
                "role": elem.element.role,
                "name": elem.element.name,
                "center": {"x": elem.center_x, "y": elem.center_y},
                "bounds": {
                    "x": elem.element.bbox.x,
                    "y": elem.element.bbox.y,
                    "width": elem.element.bbox.width,
                    "height": elem.element.bbox.height,
                } if elem.element.bbox else None,
                "xpath": elem.element.xpath,
                "selector": elem.element.css_selector,
            }
            for elem in snapshot.visible_elements
        ],
    }
```

**Integration with capture.py**:

```python
# In capture_to_episode()
from openadapt_ml.ingest.browser import (
    semantic_ref_to_ui_element,
    dom_snapshot_to_a11y_tree,
)

# After loading capture, check for browser events
browser_events = capture.get_browser_events()  # New method needed
if browser_events:
    for idx, (action, browser_event) in enumerate(
        zip(capture.actions(), browser_events)
    ):
        # Enrich action with element reference
        if hasattr(browser_event, 'element'):
            ml_action.element = semantic_ref_to_ui_element(browser_event.element)

        # Add DOM to observation
        if hasattr(browser_event, 'dom_snapshot'):
            observation.dom = browser_event.dom_snapshot.html
            observation.a11y_tree = dom_snapshot_to_a11y_tree(
                browser_event.dom_snapshot
            )
```

### Phase 3: Accessibility Tree Capture (Priority: Critical)

**Timeline**: 2-3 weeks

**Goal**: Integrate legacy openadapt.record accessibility tree data OR implement native a11y capture in openadapt-capture.

**Option A: Port from Legacy OpenAdapt**

Create converter for legacy openadapt.record:

```python
# openadapt_ml/ingest/legacy.py
"""Convert legacy openadapt.record to Episode format."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

from openadapt_ml.schema import (
    Episode, Step, Action, Observation, ActionType, UIElement, BoundingBox
)


def normalize_ax_tree(ax_data: dict, depth: int = 0, max_depth: int = 5) -> dict:
    """Recursively normalize macOS accessibility tree."""
    if depth >= max_depth or not ax_data:
        return None

    normalized = {
        "role": ax_data.get("AXRole", "").replace("AX", ""),
        "name": ax_data.get("AXTitle") or ax_data.get("AXDescription", ""),
    }

    # Position and size
    if "AXPosition" in ax_data:
        pos = ax_data["AXPosition"]
        normalized["x"] = pos.get("x", 0)
        normalized["y"] = pos.get("y", 0)
    if "AXSize" in ax_data:
        size = ax_data["AXSize"]
        normalized["width"] = size.get("w", 0)
        normalized["height"] = size.get("h", 0)

    # Value for inputs
    if "AXValue" in ax_data:
        normalized["value"] = str(ax_data["AXValue"])

    # Children
    if "AXChildren" in ax_data and ax_data["AXChildren"]:
        normalized["children"] = [
            normalize_ax_tree(child, depth + 1, max_depth)
            for child in ax_data["AXChildren"]
            if child
        ]
        normalized["children"] = [c for c in normalized["children"] if c]

    return normalized


def legacy_recording_to_episode(db_path: str | Path) -> Episode:
    """Convert legacy OpenAdapt recording database to Episode.

    Args:
        db_path: Path to OpenAdapt SQLite database (e.g., openadapt.db)

    Returns:
        Episode with accessibility tree data
    """
    # Import legacy models
    # Note: Requires openadapt package to be installed
    try:
        from openadapt.models import Recording, ActionEvent, WindowEvent, Screenshot
        from openadapt.db import crud
    except ImportError:
        raise ImportError(
            "Legacy openadapt package required. "
            "Install with: pip install openadapt"
        )

    # Load recording
    session = crud.get_new_session(read_only=True)
    recording = session.query(Recording).order_by(Recording.id.desc()).first()

    if not recording:
        raise ValueError(f"No recording found in {db_path}")

    steps = []
    for idx, action_event in enumerate(recording.processed_action_events):
        # Build observation with a11y tree
        window_event = action_event.window_event
        a11y_tree = None
        if window_event and window_event.state:
            data = window_event.state.get("data", {})
            a11y_tree = normalize_ax_tree(data)

        observation = Observation(
            screenshot_path=None,  # Would need to extract from video
            window_title=window_event.title if window_event else None,
            a11y_tree=a11y_tree,
            screen_size=(recording.monitor_width, recording.monitor_height),
            raw={
                "window_bounds": {
                    "left": window_event.left,
                    "top": window_event.top,
                    "width": window_event.width,
                    "height": window_event.height,
                } if window_event else None,
            },
        )

        # Build action with element
        element = None
        if action_event.element_state:
            element = UIElement(
                role=action_event.element_state.get("role"),
                name=action_event.active_segment_description,
            )

        # Map action type
        action_type_map = {
            "click": ActionType.CLICK,
            "singleclick": ActionType.CLICK,
            "doubleclick": ActionType.DOUBLE_CLICK,
            "scroll": ActionType.SCROLL,
            "type": ActionType.TYPE,
            "press": ActionType.KEY,
            "release": ActionType.KEY,
            "move": ActionType.HOVER,
        }
        action_type = action_type_map.get(action_event.name, ActionType.CLICK)

        action = Action(
            type=action_type,
            coordinates={"x": int(action_event.mouse_x), "y": int(action_event.mouse_y)}
            if action_event.mouse_x else None,
            text=action_event.text if action_event.text else None,
            element=element,
            raw={
                "name": action_event.name,
                "mouse_button": action_event.mouse_button_name,
                "key_name": action_event.key_name,
            },
        )

        steps.append(Step(
            step_index=idx,
            observation=observation,
            action=action,
            timestamp=action_event.timestamp,
        ))

    return Episode(
        episode_id=f"legacy_{recording.id}",
        instruction=recording.task_description or "Recorded workflow",
        steps=steps,
        success=True,
        environment=recording.platform,
        metadata={
            "source": "legacy_openadapt",
            "original_id": recording.id,
            "monitor_size": (recording.monitor_width, recording.monitor_height),
        },
    )
```

**Option B: Native A11y in openadapt-capture (Recommended)**

Add accessibility capture to openadapt-capture itself:

```python
# Proposed addition to openadapt_capture/a11y.py

import sys
from typing import Any

if sys.platform == "darwin":
    import ApplicationServices
    import AppKit

    def get_element_at_point(x: int, y: int) -> dict | None:
        """Get accessibility element at screen coordinates."""
        system_wide = ApplicationServices.AXUIElementCreateSystemWide()
        error, element = ApplicationServices.AXUIElementCopyElementAtPosition(
            system_wide, x, y, None
        )
        if error or not element:
            return None

        # Extract attributes
        attrs = {}
        for attr in ["AXRole", "AXTitle", "AXDescription", "AXValue"]:
            err, val = ApplicationServices.AXUIElementCopyAttributeValue(
                element, attr, None
            )
            if not err and val:
                attrs[attr] = str(val)

        # Get position and size
        err, pos = ApplicationServices.AXUIElementCopyAttributeValue(
            element, "AXPosition", None
        )
        err, size = ApplicationServices.AXUIElementCopyAttributeValue(
            element, "AXSize", None
        )

        return {
            "role": attrs.get("AXRole", "").replace("AX", ""),
            "name": attrs.get("AXTitle") or attrs.get("AXDescription"),
            "value": attrs.get("AXValue"),
            "bounds": {
                "x": pos.x if pos else x,
                "y": pos.y if pos else y,
                "width": size.width if size else 0,
                "height": size.height if size else 0,
            },
        }
```

### Phase 4: Full DOM Integration (Priority: High)

**Timeline**: 1-2 weeks

**Goal**: Merge browser extension DOM data with native capture events.

**Chrome Extension Updates** (in `chrome_extension/`):

1. Capture full visible DOM on each action
2. Include Set-of-Marks IDs
3. Sync timestamps with native capture

**Converter Updates**:

```python
# openadapt_ml/ingest/capture.py (additions)

def capture_to_episode(...):
    # ... existing code ...

    # Check for DOM snapshots in capture metadata
    dom_snapshots = capture.get_dom_snapshots()  # New method

    for idx, step in enumerate(steps):
        # Find DOM snapshot closest to action timestamp
        if dom_snapshots:
            closest_dom = find_closest_snapshot(
                step.timestamp, dom_snapshots
            )
            if closest_dom:
                step.observation = step.observation.model_copy(update={
                    "dom": closest_dom.html,
                    "url": closest_dom.url,
                    "a11y_tree": dom_snapshot_to_a11y_tree(closest_dom),
                })
```

### Phase 5: CLI and Tooling (Priority: Medium)

**Timeline**: 1 week

**New CLI Commands**:

```bash
# Convert openadapt-capture to Episode
uv run python -m openadapt_ml.ingest convert-capture \
    /path/to/capture \
    --output episodes/my_episode.json \
    --include-a11y \
    --include-dom

# Convert legacy OpenAdapt recording
uv run python -m openadapt_ml.ingest convert-legacy \
    /path/to/openadapt.db \
    --output episodes/legacy_episode.json

# Validate Episode against schema
uv run python -m openadapt_ml.ingest validate \
    episodes/my_episode.json

# Export Episode to training format
uv run python -m openadapt_ml.export to-parquet \
    episodes/*.json \
    --output training_data.parquet
```

**Implementation**:

```python
# openadapt_ml/ingest/cli.py
"""CLI for capture ingestion."""

import click
from pathlib import Path

from openadapt_ml.ingest.capture import capture_to_episode
from openadapt_ml.schema import save_episode, load_episode, validate_episode


@click.group()
def cli():
    """Capture ingestion commands."""
    pass


@cli.command("convert-capture")
@click.argument("capture_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON path")
@click.option("--include-a11y", is_flag=True, help="Include accessibility tree")
@click.option("--include-dom", is_flag=True, help="Include DOM snapshots")
def convert_capture(capture_path: str, output: str, include_a11y: bool, include_dom: bool):
    """Convert openadapt-capture recording to Episode."""
    capture_path = Path(capture_path)

    episode = capture_to_episode(
        capture_path,
        include_a11y=include_a11y,  # New parameter
        include_dom=include_dom,    # New parameter
    )

    output_path = Path(output) if output else capture_path / "episode.json"
    save_episode(episode, output_path)

    click.echo(f"Saved episode to {output_path}")
    click.echo(f"  ID: {episode.episode_id}")
    click.echo(f"  Steps: {episode.num_steps}")
    click.echo(f"  Instruction: {episode.instruction[:50]}...")


@cli.command("validate")
@click.argument("episode_path", type=click.Path(exists=True))
def validate(episode_path: str):
    """Validate Episode JSON against schema."""
    import json

    with open(episode_path) as f:
        data = json.load(f)

    is_valid, error = validate_episode(data)

    if is_valid:
        click.echo(f"Valid Episode: {episode_path}")
    else:
        click.echo(f"Invalid Episode: {error}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
```

---

## 5. Code Changes Required

### 5.1 Files to Modify

| File | Changes | Phase |
|------|---------|-------|
| `openadapt_ml/ingest/capture.py` | Add a11y, DOM, browser event support | 2-3 |
| `openadapt_ml/ingest/__init__.py` | Export new converters | 2-5 |
| `openadapt_ml/schema/__init__.py` | Already exports all needed types | - |

### 5.2 New Files to Create

| File | Purpose | Phase |
|------|---------|-------|
| `openadapt_ml/ingest/browser.py` | Browser event conversion | 2 |
| `openadapt_ml/ingest/legacy.py` | Legacy OpenAdapt conversion | 3 |
| `openadapt_ml/ingest/a11y.py` | Accessibility tree normalization | 3 |
| `openadapt_ml/ingest/cli.py` | CLI commands | 5 |

### 5.3 Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
ingest = [
    "openadapt-capture>=0.1.0",  # For capture format
]
legacy = [
    "sqlalchemy>=2.0",           # For legacy DB access
    # "openadapt" - for full legacy support
]
```

### 5.4 Key Code Snippets

#### 5.4.1 Enhanced capture_to_episode

```python
# openadapt_ml/ingest/capture.py (enhanced version)

def capture_to_episode(
    capture_path: str | Path,
    output_dir: str | Path | None = None,
    instruction: str | None = None,
    episode_id: str | None = None,
    include_moves: bool = False,
    include_a11y: bool = False,
    include_dom: bool = False,
) -> Episode:
    """Convert openadapt-capture recording to Episode.

    Args:
        capture_path: Path to capture directory
        output_dir: Directory for screenshots
        instruction: Task description
        episode_id: Custom episode ID
        include_moves: Include mouse move events
        include_a11y: Capture accessibility tree for each action
        include_dom: Include DOM snapshots (requires browser extension)

    Returns:
        Episode with full context
    """
    from openadapt_capture import Capture

    capture_path = Path(capture_path)
    capture = Capture.load(capture_path)

    # ... existing setup code ...

    steps = []
    for idx, action in enumerate(capture.actions(include_moves=include_moves)):
        # Screenshot
        screenshot = action.screenshot
        if screenshot is None:
            continue
        screenshot_path = _save_screenshot(screenshot, output_dir, episode_id, idx)

        # Observation with optional a11y
        observation = Observation(
            screenshot_path=screenshot_path,
            screen_size=capture.screen_size,
        )

        if include_a11y and action.x is not None and action.y is not None:
            try:
                from openadapt_ml.ingest.a11y import get_element_at_point
                element_data = get_element_at_point(int(action.x), int(action.y))
                if element_data:
                    observation = observation.model_copy(update={
                        "a11y_tree": element_data,
                    })
            except ImportError:
                pass  # a11y not available

        if include_dom:
            try:
                dom_snapshot = capture.get_dom_at(action.timestamp)
                if dom_snapshot:
                    observation = observation.model_copy(update={
                        "dom": dom_snapshot.html,
                        "url": dom_snapshot.url,
                    })
            except AttributeError:
                pass  # No DOM capture available

        # ... rest of action building ...

        steps.append(Step(
            step_index=idx,
            observation=observation,
            action=ml_action,
            timestamp=action.timestamp - start_time,
        ))

    # ... rest of function ...
```

#### 5.4.2 Platform-Agnostic A11y Normalization

```python
# openadapt_ml/ingest/a11y.py

import sys
from typing import Any


def normalize_a11y_tree(raw_tree: dict, platform: str | None = None) -> dict:
    """Normalize platform-specific accessibility tree to common format.

    Args:
        raw_tree: Platform-specific accessibility data
        platform: 'darwin', 'win32', or 'linux'

    Returns:
        Normalized tree with common keys: role, name, value, bounds, children
    """
    if platform is None:
        platform = sys.platform

    if platform == "darwin":
        return _normalize_macos(raw_tree)
    elif platform == "win32":
        return _normalize_windows(raw_tree)
    elif platform.startswith("linux"):
        return _normalize_linux(raw_tree)
    else:
        return raw_tree  # Return as-is for unknown platforms


def _normalize_macos(data: dict) -> dict:
    """Normalize macOS AX accessibility tree."""
    result = {
        "role": data.get("AXRole", "").replace("AX", ""),
        "name": data.get("AXTitle") or data.get("AXDescription") or "",
    }

    if "AXValue" in data:
        result["value"] = str(data["AXValue"])

    pos = data.get("AXPosition", {})
    size = data.get("AXSize", {})
    if pos or size:
        result["bounds"] = {
            "x": pos.get("x", 0),
            "y": pos.get("y", 0),
            "width": size.get("w", 0),
            "height": size.get("h", 0),
        }

    if "AXChildren" in data and data["AXChildren"]:
        result["children"] = [
            _normalize_macos(child)
            for child in data["AXChildren"]
            if child and isinstance(child, dict)
        ]

    return result


def _normalize_windows(data: dict) -> dict:
    """Normalize Windows UIA accessibility tree."""
    result = {
        "role": data.get("ControlType", "").replace("ControlType.", ""),
        "name": data.get("Name", ""),
    }

    if "Value" in data:
        result["value"] = str(data["Value"])

    if "AutomationId" in data:
        result["automation_id"] = data["AutomationId"]

    rect = data.get("BoundingRectangle", {})
    if rect:
        result["bounds"] = {
            "x": rect.get("left", 0),
            "y": rect.get("top", 0),
            "width": rect.get("width", 0),
            "height": rect.get("height", 0),
        }

    if "Children" in data:
        result["children"] = [
            _normalize_windows(child)
            for child in data["Children"]
            if child and isinstance(child, dict)
        ]

    return result


def _normalize_linux(data: dict) -> dict:
    """Normalize Linux AT-SPI accessibility tree."""
    # AT-SPI uses similar naming to macOS
    result = {
        "role": data.get("role", ""),
        "name": data.get("name", ""),
    }

    if "value" in data:
        result["value"] = str(data["value"])

    pos = data.get("position", {})
    size = data.get("size", {})
    if pos or size:
        result["bounds"] = {
            "x": pos.get("x", 0),
            "y": pos.get("y", 0),
            "width": size.get("width", 0),
            "height": size.get("height", 0),
        }

    if "children" in data:
        result["children"] = [
            _normalize_linux(child)
            for child in data["children"]
            if child and isinstance(child, dict)
        ]

    return result
```

---

## 6. Testing Strategy

### 6.1 Test Data

Primary test capture: `/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift`

Contains:
- 20+ screenshots
- Mouse clicks and keyboard input
- Video file (video.mp4)
- Audio file (audio.flac)
- SQLite database (capture.db)

### 6.2 Unit Tests

```python
# tests/test_ingest_capture.py

import pytest
from pathlib import Path
from openadapt_ml.ingest.capture import capture_to_episode
from openadapt_ml.schema import Episode, ActionType


# Use actual capture path from environment or fixture
CAPTURE_PATH = Path("/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift")


@pytest.fixture
def episode():
    """Load test episode."""
    if not CAPTURE_PATH.exists():
        pytest.skip("Test capture not available")
    return capture_to_episode(CAPTURE_PATH)


def test_episode_basic_structure(episode):
    """Test episode has required fields."""
    assert isinstance(episode, Episode)
    assert episode.episode_id is not None
    assert episode.instruction is not None
    assert len(episode.steps) > 0


def test_episode_steps_have_observations(episode):
    """Test each step has an observation with screenshot."""
    for step in episode.steps:
        assert step.observation is not None
        assert step.observation.screenshot_path is not None


def test_episode_steps_have_actions(episode):
    """Test each step has an action with valid type."""
    for step in episode.steps:
        assert step.action is not None
        assert isinstance(step.action.type, ActionType)


def test_episode_normalized_coordinates(episode):
    """Test coordinates are normalized (0-1 range)."""
    for step in episode.steps:
        if step.action.normalized_coordinates:
            x, y = step.action.normalized_coordinates
            assert 0 <= x <= 1, f"x={x} out of range"
            assert 0 <= y <= 1, f"y={y} out of range"


def test_episode_serialization(episode, tmp_path):
    """Test episode can be serialized and deserialized."""
    from openadapt_ml.schema import save_episode, load_episode

    output_path = tmp_path / "episode.json"
    save_episode(episode, output_path)

    loaded = load_episode(output_path)
    assert loaded.episode_id == episode.episode_id
    assert len(loaded.steps) == len(episode.steps)


def test_episode_validation(episode):
    """Test episode passes validation."""
    from openadapt_ml.schema import validate_episode

    data = episode.model_dump()
    is_valid, error = validate_episode(data)
    assert is_valid, f"Validation failed: {error}"
```

### 6.3 Integration Tests

```python
# tests/test_ingest_integration.py

import pytest
from pathlib import Path


def test_capture_to_training_data():
    """Test full pipeline from capture to training data."""
    from openadapt_ml.ingest.capture import capture_to_episode
    from openadapt_ml.datasets.next_action import NextActionDataset

    capture_path = Path("/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift")
    if not capture_path.exists():
        pytest.skip("Test capture not available")

    # Convert
    episode = capture_to_episode(capture_path)

    # Use in dataset
    dataset = NextActionDataset([episode])
    assert len(dataset) > 0

    # Get a sample
    sample = dataset[0]
    assert "image" in sample or "screenshot_path" in sample
    assert "action" in sample


def test_episode_to_waa_format():
    """Test converting episode to WAA format."""
    from openadapt_ml.ingest.capture import capture_to_episode
    from openadapt_ml.schema.converters import to_waa_trajectory

    capture_path = Path("/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift")
    if not capture_path.exists():
        pytest.skip("Test capture not available")

    episode = capture_to_episode(capture_path)
    trajectory, task_info = to_waa_trajectory(episode)

    assert len(trajectory) == len(episode.steps)
    assert "instruction" in task_info
```

### 6.4 Validation Tests

```python
# tests/test_schema_validation.py

import pytest
from openadapt_ml.schema import (
    Episode, Step, Action, Observation, ActionType,
    validate_episode
)


def test_minimal_valid_episode():
    """Test minimal valid episode."""
    episode = Episode(
        episode_id="test_001",
        instruction="Test task",
        steps=[
            Step(
                step_index=0,
                observation=Observation(screenshot_path="test.png"),
                action=Action(type=ActionType.CLICK),
            )
        ],
    )

    is_valid, error = validate_episode(episode.model_dump())
    assert is_valid, error


def test_episode_missing_required_field():
    """Test episode without required field fails."""
    with pytest.raises(Exception):
        Episode(
            episode_id="test",
            # Missing: instruction
            steps=[],
        )


def test_action_type_validation():
    """Test action requires valid type."""
    with pytest.raises(Exception):
        Action(type="invalid_type")


def test_coordinates_non_negative():
    """Test coordinates validation."""
    from openadapt_ml.schema import Coordinates

    with pytest.raises(Exception):
        Coordinates(x=-1, y=100)
```

### 6.5 Acceptance Criteria

| Phase | Criteria | Test |
|-------|----------|------|
| 1 | Basic capture converts | `test_episode_basic_structure` |
| 1 | Screenshots extracted | `test_episode_steps_have_observations` |
| 1 | Actions have types | `test_episode_steps_have_actions` |
| 1 | Coordinates normalized | `test_episode_normalized_coordinates` |
| 2 | Browser events enriched | `test_browser_element_attached` |
| 2 | DOM in observation | `test_dom_captured` |
| 3 | A11y tree captured | `test_a11y_tree_present` |
| 3 | Cross-platform normalization | `test_normalize_macos`, `test_normalize_windows` |
| 4 | Full DOM integration | `test_dom_snapshot_integration` |
| 5 | CLI converts captures | `test_cli_convert_capture` |
| 5 | CLI validates episodes | `test_cli_validate` |

### 6.6 Running Tests

```bash
# Run all ingest tests
uv run pytest tests/test_ingest*.py -v

# Run with coverage
uv run pytest tests/test_ingest*.py --cov=openadapt_ml.ingest --cov-report=html

# Run integration tests (requires test capture)
uv run pytest tests/test_ingest_integration.py -v --capture=no
```

---

## 7. Appendices

### 7.1 Action Type Mapping Reference

| openadapt-capture | openadapt.record | ActionType |
|------------------|------------------|------------|
| `mouse.singleclick` | `singleclick` | `CLICK` |
| `mouse.doubleclick` | `doubleclick` | `DOUBLE_CLICK` |
| `mouse.click` (button=right) | N/A | `RIGHT_CLICK` |
| `mouse.drag` | `drag` | `DRAG` |
| `mouse.scroll` | `scroll` | `SCROLL` |
| `mouse.move` | `move` | `HOVER` |
| `key.type` | `type` | `TYPE` |
| `key.down` | `press` | `KEY` |
| `key.up` | `release` | `KEY` |
| N/A | N/A | `HOTKEY` |
| N/A | N/A | `DONE` |
| N/A | N/A | `FAIL` |

### 7.2 Accessibility Role Normalization

| macOS (AX) | Windows (UIA) | Linux (AT-SPI) | Normalized |
|------------|---------------|----------------|------------|
| AXButton | ControlType.Button | push-button | button |
| AXTextField | ControlType.Edit | text | textbox |
| AXCheckBox | ControlType.CheckBox | check-box | checkbox |
| AXRadioButton | ControlType.RadioButton | radio-button | radiobutton |
| AXComboBox | ControlType.ComboBox | combo-box | combobox |
| AXList | ControlType.List | list | list |
| AXMenuItem | ControlType.MenuItem | menu-item | menuitem |
| AXWindow | ControlType.Window | frame | window |

### 7.3 Chrome Extension Message Format

Expected message structure from Chrome extension:

```json
{
  "type": "click",
  "timestamp": 1704067200.123,
  "url": "https://example.com",
  "tabId": 123,
  "clientX": 450,
  "clientY": 300,
  "pageX": 450,
  "pageY": 800,
  "element": {
    "role": "button",
    "name": "Submit",
    "xpath": "/html/body/div[1]/form/button",
    "cssSelector": "#submit-btn",
    "bbox": {"x": 400, "y": 280, "width": 100, "height": 40},
    "state": {"enabled": true, "visible": true}
  },
  "domSnapshot": {
    "html": "<html>...</html>",
    "visibleElements": [...]
  }
}
```

### 7.4 File Structure After Migration

```
openadapt-ml/
├── openadapt_ml/
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── capture.py        # Main converter (enhanced)
│   │   ├── browser.py        # Browser event conversion (NEW)
│   │   ├── legacy.py         # Legacy openadapt conversion (NEW)
│   │   ├── a11y.py           # A11y tree normalization (NEW)
│   │   └── cli.py            # CLI commands (NEW)
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── episode.py        # Episode schema (no changes)
│   │   └── converters.py     # Format converters
│   └── ...
├── tests/
│   ├── test_ingest_capture.py
│   ├── test_ingest_browser.py
│   ├── test_ingest_legacy.py
│   └── test_ingest_integration.py
└── docs/
    └── openadapt_capture_migration_detailed.md  # This document
```

### 7.5 Migration Checklist

- [ ] Phase 1: Basic conversion working
  - [ ] Screenshots extracted
  - [ ] Coordinates normalized
  - [ ] Action types mapped
  - [ ] Tests passing

- [ ] Phase 2: Browser events
  - [ ] `browser.py` created
  - [ ] Element references extracted
  - [ ] DOM stored in observation
  - [ ] Tests passing

- [ ] Phase 3: Accessibility trees
  - [ ] `a11y.py` created
  - [ ] macOS normalization
  - [ ] Windows normalization
  - [ ] Legacy converter working
  - [ ] Tests passing

- [ ] Phase 4: DOM integration
  - [ ] Chrome extension updated
  - [ ] Timestamp sync working
  - [ ] Full snapshots captured
  - [ ] Tests passing

- [ ] Phase 5: CLI
  - [ ] `cli.py` created
  - [ ] `convert-capture` command
  - [ ] `convert-legacy` command
  - [ ] `validate` command
  - [ ] Documentation updated

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-03 | OpenAdapt | Initial comprehensive document |

---

**End of Migration Guide**
