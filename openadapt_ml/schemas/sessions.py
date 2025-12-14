from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ActionType = Literal[
    "click",
    "double_click",
    "right_click",
    "drag",
    "scroll",
    "type",
    "key",  # Single keypress (e.g., "Enter", "Tab")
    "wait",
    "done",
    "answer",  # For benchmarks that score by final answer
    "failed",
]


@dataclass
class Action:
    """A single GUI action taken by an agent or demonstrator.

    Coordinates are normalized to the range [0, 1] relative to the
    associated screenshot image's width/height.

    Supports both coordinate-based and element-based grounding ("grounding-first"
    approach where both are stored when available).
    """

    type: str
    x: Optional[float] = None
    y: Optional[float] = None
    text: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    # Bounding box for click targets: (x_min, y_min, x_max, y_max) in normalized coords
    bbox: Optional[tuple[float, float, float, float]] = None

    # Element index for Set-of-Marks (SoM) style actions: CLICK([1]), TYPE([2], "text")
    element_index: Optional[int] = None

    # Element grounding (for benchmark compatibility)
    target_node_id: Optional[str] = None  # DOM/AX/UIA node ID
    target_role: Optional[str] = None  # "button", "textfield", etc.
    target_name: Optional[str] = None  # Accessible name

    # Keyboard actions
    key: Optional[str] = None  # Single key: "Enter", "Tab", "Escape"
    modifiers: Optional[List[str]] = None  # ["ctrl", "shift", "alt"]

    # Scroll actions
    scroll_direction: Optional[str] = None  # "up", "down", "left", "right"
    scroll_amount: Optional[float] = None  # Pixels or normalized

    # Drag actions - end coordinates
    end_x: Optional[float] = None
    end_y: Optional[float] = None

    # Answer action (for benchmarks that score by answer)
    answer: Optional[str] = None


@dataclass
class Observation:
    """A single observation of the GUI state.

    Supports multiple observation modalities:
    - Visual: screenshot image
    - Structured UI: accessibility tree (UIA/AXTree/DOM)
    - Context: URL, window title, focused element
    """

    image_path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    # Structured UI (format varies by platform)
    accessibility_tree: Optional[Dict[str, Any]] = None  # UIA/AXTree/DOM
    dom_html: Optional[str] = None  # Raw HTML for web tasks

    # Context
    url: Optional[str] = None  # For web tasks
    window_title: Optional[str] = None  # Active window title
    app_name: Optional[str] = None  # Active application
    focused_element: Optional[Dict[str, Any]] = None  # {node_id, bbox, text}


@dataclass
class Step:
    """One timestep in an episode: observation + action (+ optional thought)."""

    t: float
    observation: Observation
    action: Action
    thought: Optional[str] = None


@dataclass
class Episode:
    """A single workflow instance / task attempt.

    This is the primary training unit used by dataset builders and
    training loops.
    """

    id: str
    goal: str
    steps: List[Step] = field(default_factory=list)
    summary: Optional[str] = None
    success: Optional[bool] = None
    workflow_id: Optional[str] = None


@dataclass
class Session:
    """A container for one or more episodes plus session-level metadata."""

    id: str
    episodes: List[Episode] = field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None
