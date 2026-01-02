"""
Episode Schema for GUI Trajectory Data

Canonical contract for episode/demonstration data in GUI automation. Designed for
interoperability across training pipelines, benchmarks, and human demonstrations.

Features:
- Pydantic models with runtime validation
- JSON Schema export for language-agnostic tooling
- Supports pixel coordinates AND normalized (0-1) coordinates
- Extensible via `raw` and `metadata` fields
- Converters for common formats (WAA, WebArena, etc.)

Quick Start:
    from openadapt_ml.schema import Episode, Step, Action, Observation, ActionType

    episode = Episode(
        episode_id="demo_001",
        instruction="Open the Settings app and enable Dark Mode",
        steps=[
            Step(
                step_index=0,
                observation=Observation(screenshot_path="step_0.png"),
                action=Action(
                    type=ActionType.CLICK,
                    coordinates={"x": 512, "y": 384},
                    # Or use normalized coords for resolution independence:
                    # normalized_coordinates=(0.5, 0.375),
                ),
                reasoning="Click on Settings icon",
            ),
        ],
        success=True,
    )

    # Validate any dict against the schema
    from openadapt_ml.schema import validate_episode
    is_valid, error = validate_episode(data)

    # Export JSON Schema for external tools
    from openadapt_ml.schema import export_json_schema
    export_json_schema("episode.schema.json")

Schema Version: 1.0.0
- Core models: Episode, Step, Action, Observation
- 24 action types covering mouse, keyboard, navigation, and system actions
- Support for both pixel and normalized coordinates
- Extension points: raw, metadata fields

Evolution Policy (SemVer):
- PATCH (1.0.x): Documentation, bug fixes (no schema changes)
- MINOR (1.x.0): New optional fields with defaults (backward compatible)
- MAJOR (x.0.0): Breaking changes (field removal, type changes, new required fields)

Migration Guide:
- MINOR bumps: No action needed, old data validates
- MAJOR bumps: Use converters or migration scripts (provided in release notes)
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# Schema version - follows semver
SCHEMA_VERSION = "1.0.0"


class ActionType(str, Enum):
    """Supported action types for GUI automation."""

    # Mouse actions
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    HOVER = "hover"

    # Keyboard actions
    TYPE = "type"
    KEY = "key"
    HOTKEY = "hotkey"

    # Combined/special actions
    CLICK_AND_TYPE = "click_and_type"
    WAIT = "wait"
    SCREENSHOT = "screenshot"

    # Navigation (for web)
    GOTO = "goto"
    BACK = "back"
    FORWARD = "forward"
    REFRESH = "refresh"

    # System actions
    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    SELECT_MONITOR = "select_monitor"  # Multi-monitor: focus a specific display
    WINDOW_FOCUS = "window_focus"  # Focus a specific window
    WINDOW_RESIZE = "window_resize"  # Resize window
    WINDOW_MOVE = "window_move"  # Move window

    # Meta actions
    DONE = "done"
    FAIL = "fail"


class BenchmarkSource(str, Enum):
    """Source benchmark/dataset for the episode."""

    WAA = "waa"  # Windows Agent Arena
    WEBARENA = "webarena"
    OSWORLD = "osworld"
    MINIWOB = "miniwob"
    HUMAN = "human"  # Human demonstration
    SYNTHETIC = "synthetic"  # Generated/augmented


class Coordinates(BaseModel):
    """Screen coordinates for mouse actions."""

    x: int = Field(..., description="X coordinate (pixels from left)")
    y: int = Field(..., description="Y coordinate (pixels from top)")

    @field_validator("x", "y")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Coordinates must be non-negative")
        return v


class BoundingBox(BaseModel):
    """Bounding box for UI elements."""

    x: int = Field(..., description="Left edge X coordinate")
    y: int = Field(..., description="Top edge Y coordinate")
    width: int = Field(..., ge=0, description="Width in pixels")
    height: int = Field(..., ge=0, description="Height in pixels")

    @property
    def center(self) -> Coordinates:
        """Get center point of bounding box."""
        return Coordinates(x=self.x + self.width // 2, y=self.y + self.height // 2)


class UIElement(BaseModel):
    """UI element information from accessibility tree or DOM."""

    role: Optional[str] = Field(None, description="Element role (button, textbox, etc.)")
    name: Optional[str] = Field(None, description="Element accessible name")
    value: Optional[str] = Field(None, description="Element value (for inputs)")
    bounds: Optional[BoundingBox] = Field(None, description="Element bounding box")
    element_id: Optional[str] = Field(None, description="Unique element identifier")
    xpath: Optional[str] = Field(None, description="XPath selector (web)")
    selector: Optional[str] = Field(None, description="CSS selector (web)")
    automation_id: Optional[str] = Field(None, description="Automation ID (Windows)")


class Action(BaseModel):
    """An action taken by the agent."""

    type: ActionType = Field(..., description="Type of action")

    # Mouse action parameters
    coordinates: Optional[Coordinates] = Field(
        None, description="Target coordinates for mouse actions"
    )
    start_coordinates: Optional[Coordinates] = Field(
        None, description="Start coordinates for drag actions"
    )
    end_coordinates: Optional[Coordinates] = Field(
        None, description="End coordinates for drag actions"
    )
    scroll_direction: Optional[Literal["up", "down", "left", "right"]] = Field(
        None, description="Scroll direction"
    )
    scroll_amount: Optional[int] = Field(None, description="Scroll amount in pixels")

    # Keyboard action parameters
    text: Optional[str] = Field(None, description="Text to type")
    key: Optional[str] = Field(None, description="Key to press (e.g., 'enter', 'tab')")
    modifiers: Optional[list[str]] = Field(
        None, description="Modifier keys (ctrl, alt, shift, meta)"
    )

    # Element targeting (alternative to coordinates)
    element: Optional[UIElement] = Field(
        None, description="Target element (for element-based actions)"
    )

    # Additional parameters
    url: Optional[str] = Field(None, description="URL for goto action")
    app_name: Optional[str] = Field(None, description="Application name for open/close")
    duration: Optional[float] = Field(None, description="Duration in seconds (for wait)")
    monitor_id: Optional[int] = Field(None, description="Monitor ID for select_monitor action")
    window_title: Optional[str] = Field(None, description="Window title for window_focus action")

    # Normalized coordinates (0.0-1.0) - alternative to pixel coordinates
    # Useful for resolution-independent recordings
    normalized_coordinates: Optional[tuple[float, float]] = Field(
        None, description="Normalized (x, y) coordinates (0.0-1.0 range)"
    )
    normalized_start: Optional[tuple[float, float]] = Field(
        None, description="Normalized start coordinates for drag (0.0-1.0 range)"
    )
    normalized_end: Optional[tuple[float, float]] = Field(
        None, description="Normalized end coordinates for drag (0.0-1.0 range)"
    )

    # Raw/original action data
    raw: Optional[dict[str, Any]] = Field(
        None, description="Original action data from source format"
    )

    @model_validator(mode="after")
    def validate_action_params(self) -> "Action":
        """Validate that required parameters are present for action type."""
        if self.type in {ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK}:
            if self.coordinates is None and self.element is None:
                # Allow missing coordinates - can be inferred from context
                pass

        if self.type == ActionType.TYPE and self.text is None:
            raise ValueError("TYPE action requires 'text' parameter")

        if self.type == ActionType.KEY and self.key is None:
            raise ValueError("KEY action requires 'key' parameter")

        if self.type == ActionType.GOTO and self.url is None:
            raise ValueError("GOTO action requires 'url' parameter")

        return self


class Observation(BaseModel):
    """An observation of the environment state."""

    # Visual observation
    screenshot_path: Optional[str] = Field(
        None, description="Path to screenshot image file"
    )
    screenshot_base64: Optional[str] = Field(
        None, description="Base64-encoded screenshot (for inline storage)"
    )

    # Structured observations
    a11y_tree: Optional[dict[str, Any]] = Field(
        None, description="Accessibility tree snapshot"
    )
    dom: Optional[str] = Field(None, description="DOM HTML snapshot (web)")

    # Window/screen info
    window_title: Optional[str] = Field(None, description="Active window title")
    screen_size: Optional[tuple[int, int]] = Field(
        None, description="Screen dimensions (width, height)"
    )

    # Focused element
    focused_element: Optional[UIElement] = Field(
        None, description="Currently focused UI element"
    )

    # Additional metadata
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    raw: Optional[dict[str, Any]] = Field(
        None, description="Original observation data from source format"
    )


class Step(BaseModel):
    """A single step in an episode (observation -> action pair)."""

    step_index: int = Field(..., ge=0, description="Step number (0-indexed)")

    # Core data
    observation: Observation = Field(..., description="State observation before action")
    action: Action = Field(..., description="Action taken")

    # Agent reasoning (for demos/training)
    reasoning: Optional[str] = Field(
        None, description="Agent's reasoning for the action (chain-of-thought)"
    )

    # Outcome
    reward: Optional[float] = Field(None, description="Reward signal (if available)")
    done: Optional[bool] = Field(None, description="Whether episode ended after this step")

    # Timing
    timestamp: Optional[float] = Field(None, description="Unix timestamp of action")
    duration_ms: Optional[int] = Field(
        None, description="Time taken for this step in milliseconds"
    )


class Episode(BaseModel):
    """A complete episode/demonstration for GUI automation.

    This is the canonical format for storing and exchanging GUI trajectory data.
    All benchmark-specific formats should be converted to/from this format.
    """

    # Schema metadata
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for compatibility checking"
    )

    # Episode identification
    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: Optional[str] = Field(None, description="Task identifier (from benchmark)")

    # Task specification
    instruction: str = Field(..., description="Natural language task instruction")
    goal: Optional[str] = Field(
        None, description="Detailed goal description (if different from instruction)"
    )

    # Episode data
    steps: list[Step] = Field(..., description="Sequence of steps in the episode")

    # Outcome
    success: Optional[bool] = Field(None, description="Whether task was completed successfully")
    final_reward: Optional[float] = Field(None, description="Final reward/score")

    # Provenance
    source: Optional[BenchmarkSource] = Field(
        None, description="Source benchmark/dataset"
    )
    source_file: Optional[str] = Field(
        None, description="Original source file path"
    )

    # Metadata
    created_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="When episode was created/recorded"
    )
    agent_model: Optional[str] = Field(
        None, description="Model that generated this episode (e.g., 'gpt-4o')"
    )
    environment: Optional[str] = Field(
        None, description="Environment info (OS, browser, etc.)"
    )
    tags: Optional[list[str]] = Field(
        None, description="Tags for categorization"
    )

    # Extension point for benchmark-specific data
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional metadata from source"
    )

    @property
    def num_steps(self) -> int:
        """Number of steps in the episode."""
        return len(self.steps)

    @property
    def action_types(self) -> list[ActionType]:
        """List of action types in this episode."""
        return [step.action.type for step in self.steps]

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Episode":
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Get JSON Schema for Episode format."""
        return cls.model_json_schema()


# ============================================================================
# Utility Functions
# ============================================================================

def validate_episode(data: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate episode data against schema.

    Args:
        data: Episode data as dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        Episode.model_validate(data)
        return True, None
    except Exception as e:
        return False, str(e)


def load_episode(path: Union[str, Path]) -> Episode:
    """Load episode from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Episode instance
    """
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    episode = Episode.model_validate(data)

    # Set source_file if not already set
    if episode.source_file is None:
        episode = episode.model_copy(update={"source_file": str(path)})

    return episode


def save_episode(episode: Episode, path: Union[str, Path], indent: int = 2) -> None:
    """Save episode to JSON file.

    Args:
        episode: Episode to save
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(episode.to_json(indent=indent))


def export_json_schema(path: Union[str, Path]) -> None:
    """Export JSON Schema to file for documentation/tooling.

    Args:
        path: Output path for schema file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = Episode.json_schema()

    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
