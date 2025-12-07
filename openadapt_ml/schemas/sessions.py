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
    "key_press",
    "wait",
    "done",
    "failed",
]


@dataclass
class Action:
    """A single GUI action taken by an agent or demonstrator.

    Coordinates are normalized to the range [0, 1] relative to the
    associated screenshot image's width/height.
    """

    type: str
    x: Optional[float] = None
    y: Optional[float] = None
    text: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class Observation:
    """A single observation of the GUI state.

    For v1 this is primarily a path to a screenshot image plus optional
    metadata describing the window, application, URL, etc.
    """

    image_path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


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
