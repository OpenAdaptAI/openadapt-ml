"""Schema validation utilities for openadapt-ml.

Validates that data conforms to the canonical Episode/Session schema
before training or processing.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openadapt_ml.schemas.sessions import Action, Episode, Observation, Session, Step


class ValidationError(Exception):
    """Raised when data fails schema validation."""

    def __init__(self, message: str, path: str = "", details: Optional[List[str]] = None):
        self.message = message
        self.path = path
        self.details = details or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"{self.path}: {self.message}" if self.path else self.message
        if self.details:
            msg += "\n  " + "\n  ".join(self.details)
        return msg


def validate_action(action: Action, path: str = "action") -> List[str]:
    """Validate an Action object.

    Returns list of warnings (non-fatal issues). Raises ValidationError for fatal issues.
    """
    warnings = []

    if not isinstance(action, Action):
        raise ValidationError(f"Expected Action, got {type(action).__name__}", path)

    # Type is required
    if not action.type:
        raise ValidationError("Action type is required", path)

    # Coordinate validation for click/drag actions
    if action.type in ("click", "double_click", "right_click", "drag"):
        if action.x is None or action.y is None:
            # Only warn if no element_index either (SoM mode doesn't need coords)
            if action.element_index is None:
                warnings.append(f"{path}: Click action has no coordinates or element_index")
        else:
            # Validate coordinate range
            if not (0.0 <= action.x <= 1.0):
                warnings.append(f"{path}: x coordinate {action.x} outside [0, 1] range")
            if not (0.0 <= action.y <= 1.0):
                warnings.append(f"{path}: y coordinate {action.y} outside [0, 1] range")

    # Drag requires end coordinates
    if action.type == "drag":
        if action.end_x is None or action.end_y is None:
            if action.element_index is None:
                warnings.append(f"{path}: Drag action missing end coordinates")

    # Type action requires text
    if action.type == "type" and not action.text:
        warnings.append(f"{path}: Type action has no text")

    # Key action requires key
    if action.type == "key" and not action.key:
        warnings.append(f"{path}: Key action has no key specified")

    return warnings


def validate_observation(obs: Observation, path: str = "observation") -> List[str]:
    """Validate an Observation object.

    Returns list of warnings. Raises ValidationError for fatal issues.
    """
    warnings = []

    if not isinstance(obs, Observation):
        raise ValidationError(f"Expected Observation, got {type(obs).__name__}", path)

    # At least one of image_path or accessibility_tree should be present
    if obs.image_path is None and obs.accessibility_tree is None:
        warnings.append(f"{path}: No image_path or accessibility_tree")

    # If image_path is set, check it's a valid path format
    if obs.image_path and not isinstance(obs.image_path, str):
        raise ValidationError(f"image_path must be string, got {type(obs.image_path).__name__}", path)

    return warnings


def validate_step(step: Step, path: str = "step") -> List[str]:
    """Validate a Step object.

    Returns list of warnings. Raises ValidationError for fatal issues.
    """
    warnings = []

    if not isinstance(step, Step):
        raise ValidationError(f"Expected Step, got {type(step).__name__}", path)

    # Timestamp should be non-negative
    if step.t < 0:
        warnings.append(f"{path}: Negative timestamp {step.t}")

    # Validate nested objects
    warnings.extend(validate_observation(step.observation, f"{path}.observation"))
    warnings.extend(validate_action(step.action, f"{path}.action"))

    return warnings


def validate_episode(episode: Episode, check_images: bool = False) -> List[str]:
    """Validate an Episode object.

    Args:
        episode: Episode to validate.
        check_images: If True, verify image files exist on disk.

    Returns:
        List of warnings (non-fatal issues).

    Raises:
        ValidationError: If episode has fatal schema violations.
    """
    warnings = []

    if not isinstance(episode, Episode):
        raise ValidationError(f"Expected Episode, got {type(episode).__name__}")

    # Required fields
    if not episode.id:
        raise ValidationError("Episode id is required")
    if not episode.goal:
        raise ValidationError("Episode goal is required")

    # Steps validation
    if not episode.steps:
        warnings.append(f"Episode '{episode.id}': No steps")
    else:
        for i, step in enumerate(episode.steps):
            step_warnings = validate_step(step, f"Episode '{episode.id}'.steps[{i}]")
            warnings.extend(step_warnings)

            # Optional: check image files exist
            if check_images and step.observation.image_path:
                img_path = Path(step.observation.image_path)
                if not img_path.exists():
                    warnings.append(f"Episode '{episode.id}'.steps[{i}]: Image not found: {img_path}")

    # Check timestamps are monotonic
    if len(episode.steps) > 1:
        for i in range(1, len(episode.steps)):
            if episode.steps[i].t < episode.steps[i - 1].t:
                warnings.append(
                    f"Episode '{episode.id}': Non-monotonic timestamps at steps {i-1} and {i}"
                )

    return warnings


def validate_session(session: Session, check_images: bool = False) -> List[str]:
    """Validate a Session object.

    Args:
        session: Session to validate.
        check_images: If True, verify image files exist on disk.

    Returns:
        List of warnings (non-fatal issues).

    Raises:
        ValidationError: If session has fatal schema violations.
    """
    warnings = []

    if not isinstance(session, Session):
        raise ValidationError(f"Expected Session, got {type(session).__name__}")

    # Required fields
    if not session.id:
        raise ValidationError("Session id is required")

    # Episodes validation
    if not session.episodes:
        warnings.append(f"Session '{session.id}': No episodes")
    else:
        for i, episode in enumerate(session.episodes):
            ep_warnings = validate_episode(episode, check_images=check_images)
            warnings.extend(ep_warnings)

    return warnings


def validate_episodes(episodes: List[Episode], check_images: bool = False) -> List[str]:
    """Validate a list of Episode objects.

    Args:
        episodes: List of episodes to validate.
        check_images: If True, verify image files exist on disk.

    Returns:
        List of warnings (non-fatal issues).

    Raises:
        ValidationError: If any episode has fatal schema violations.
    """
    warnings = []

    if not isinstance(episodes, list):
        raise ValidationError(f"Expected list of Episodes, got {type(episodes).__name__}")

    if not episodes:
        warnings.append("Empty episode list")
        return warnings

    for i, episode in enumerate(episodes):
        ep_warnings = validate_episode(episode, check_images=check_images)
        warnings.extend(ep_warnings)

    return warnings


def summarize_episodes(episodes: List[Episode]) -> Dict[str, Any]:
    """Generate a summary of episode statistics.

    Useful for quick sanity checks after loading data.
    """
    if not episodes:
        return {"count": 0, "total_steps": 0, "action_types": {}}

    action_types: Dict[str, int] = {}
    total_steps = 0

    for ep in episodes:
        total_steps += len(ep.steps)
        for step in ep.steps:
            action_type = step.action.type
            action_types[action_type] = action_types.get(action_type, 0) + 1

    return {
        "count": len(episodes),
        "total_steps": total_steps,
        "avg_steps_per_episode": total_steps / len(episodes),
        "action_types": action_types,
        "goals": [ep.goal for ep in episodes[:5]],  # First 5 goals as sample
    }
