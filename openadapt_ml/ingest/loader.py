"""Episode loading utilities for openadapt-ml.

Load Episodes from JSON files exported by external systems.
This is the primary entry point for users who have their own data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step


def load_episodes(
    path: Union[str, Path],
    validate: bool = True,
    check_images: bool = False,
) -> List[Episode]:
    """Load Episodes from a directory or JSON file.

    Supports two formats:
    1. Single JSON file containing a list of episodes
    2. Directory containing multiple JSON files (one episode per file, or batched)

    Args:
        path: Path to directory or JSON file containing episode data.
        validate: If True, validate episodes against schema (default True).
        check_images: If True, verify image files exist on disk (default False).

    Returns:
        List of Episode objects ready for training.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValidationError: If validate=True and data fails validation.
        ValueError: If JSON format is invalid.

    Example:
        >>> episodes = load_episodes("exported_data/")
        >>> print(f"Loaded {len(episodes)} episodes")
        >>> print(f"Total steps: {sum(len(e.steps) for e in episodes)}")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    episodes: List[Episode] = []

    if path.is_file():
        # Single JSON file
        episodes = _load_episodes_from_file(path, validate=validate)
    elif path.is_dir():
        # Directory of JSON files
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {path}")

        for json_file in json_files:
            file_episodes = _load_episodes_from_file(json_file, validate=validate)
            episodes.extend(file_episodes)
    else:
        raise ValueError(f"Path must be a file or directory: {path}")

    if check_images:
        warnings = _check_episode_images(episodes)
        if warnings:
            print(f"Image warnings ({len(warnings)}):")
            for w in warnings[:10]:  # Show first 10
                print(f"  - {w}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")

    return episodes


def _check_episode_images(episodes: List[Episode]) -> List[str]:
    """Check that all referenced images exist on disk."""
    warnings = []
    for ep in episodes:
        for step in ep.steps:
            if step.observation.screenshot_path:
                if not Path(step.observation.screenshot_path).exists():
                    warnings.append(
                        f"Episode {ep.episode_id}, step {step.step_index}: "
                        f"Image not found: {step.observation.screenshot_path}"
                    )
    return warnings


def _load_episodes_from_file(path: Path, validate: bool = True) -> List[Episode]:
    """Load episodes from a single JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        # List of episodes
        return [_dict_to_episode(ep, validate=validate) for ep in data]
    elif isinstance(data, dict):
        # Single episode or wrapped format
        if "episodes" in data:
            return [_dict_to_episode(ep, validate=validate) for ep in data["episodes"]]
        elif "episode_id" in data or "id" in data:
            # Single episode (support both old and new field names)
            return [_dict_to_episode(data, validate=validate)]
        else:
            raise ValueError(f"Unrecognized JSON format in {path}")
    else:
        raise ValueError(f"Expected list or dict in {path}, got {type(data)}")


def _parse_action_type(type_str: str) -> ActionType:
    """Parse action type string to ActionType enum."""
    # Handle common mappings from old format
    type_map = {
        "unknown": ActionType.CLICK,
        "double_click": ActionType.DOUBLE_CLICK,
        "right_click": ActionType.RIGHT_CLICK,
        "key_press": ActionType.KEY,
    }

    type_lower = type_str.lower()
    if type_lower in type_map:
        return type_map[type_lower]

    # Try direct enum lookup
    try:
        return ActionType(type_lower)
    except ValueError:
        # Default to CLICK for unknown types
        return ActionType.CLICK


def _dict_to_episode(data: Dict[str, Any], validate: bool = True) -> Episode:
    """Convert a dictionary to an Episode object."""
    steps = []
    for step_idx, step_data in enumerate(data.get("steps", [])):
        # Parse observation
        obs_data = step_data.get("observation", {})
        observation = Observation(
            screenshot_path=obs_data.get("screenshot_path") or obs_data.get("image_path"),
            raw=obs_data.get("raw") or obs_data.get("meta"),
            a11y_tree=obs_data.get("a11y_tree") or obs_data.get("accessibility_tree"),
            dom=obs_data.get("dom") or obs_data.get("dom_html"),
            window_title=obs_data.get("window_title"),
            focused_element=obs_data.get("focused_element"),
        )

        # Parse action
        action_data = step_data.get("action", {})

        # Handle action type (string -> enum)
        action_type_raw = action_data.get("type", "click")
        action_type = _parse_action_type(action_type_raw)

        # Handle coordinates: convert x,y to normalized_coordinates tuple
        normalized_coords = None
        if action_data.get("normalized_coordinates"):
            normalized_coords = tuple(action_data["normalized_coordinates"])
        elif action_data.get("x") is not None and action_data.get("y") is not None:
            normalized_coords = (action_data["x"], action_data["y"])

        # Handle end coordinates for drag actions
        normalized_end = None
        if action_data.get("normalized_end"):
            normalized_end = tuple(action_data["normalized_end"])
        elif action_data.get("end_x") is not None and action_data.get("end_y") is not None:
            normalized_end = (action_data["end_x"], action_data["end_y"])

        action = Action(
            type=action_type,
            normalized_coordinates=normalized_coords,
            normalized_end=normalized_end,
            text=action_data.get("text"),
            raw=action_data.get("raw"),
            key=action_data.get("key"),
            modifiers=action_data.get("modifiers"),
            scroll_direction=action_data.get("scroll_direction"),
            scroll_amount=action_data.get("scroll_amount"),
        )

        # Handle step index and timestamp
        step_index = step_data.get("step_index", step_idx)
        timestamp = step_data.get("timestamp") or step_data.get("t")

        step = Step(
            step_index=step_index,
            observation=observation,
            action=action,
            reasoning=step_data.get("reasoning") or step_data.get("thought"),
            timestamp=timestamp,
        )
        steps.append(step)

    # Build episode with field mapping (old -> new)
    episode_data = {
        "episode_id": data.get("episode_id") or data.get("id", "unknown"),
        "instruction": data.get("instruction") or data.get("goal", ""),
        "steps": steps,
        "success": data.get("success"),
        "metadata": {
            "summary": data.get("summary"),
            "workflow_id": data.get("workflow_id"),
        },
    }

    if validate:
        return Episode.model_validate(episode_data)
    else:
        return Episode(**episode_data)


def save_episodes(
    episodes: List[Episode],
    path: Union[str, Path],
    pretty: bool = True,
) -> None:
    """Save Episodes to a JSON file.

    Args:
        episodes: List of Episode objects to save.
        path: Output file path.
        pretty: If True, format JSON with indentation.

    Example:
        >>> save_episodes(episodes, "output/episodes.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [_episode_to_dict(ep) for ep in episodes]

    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2, default=str)
        else:
            json.dump(data, f, default=str)


def _episode_to_dict(episode: Episode) -> Dict[str, Any]:
    """Convert an Episode object to a dictionary."""
    steps = []
    for step in episode.steps:
        step_dict = {
            "step_index": step.step_index,
            "timestamp": step.timestamp,
            "observation": {
                "screenshot_path": step.observation.screenshot_path,
                "raw": step.observation.raw,
                "a11y_tree": step.observation.a11y_tree,
                "dom": step.observation.dom,
                "window_title": step.observation.window_title,
            },
            "action": {
                "type": step.action.type.value,
                "normalized_coordinates": step.action.normalized_coordinates,
                "normalized_end": step.action.normalized_end,
                "text": step.action.text,
                "raw": step.action.raw,
                "key": step.action.key,
                "modifiers": step.action.modifiers,
                "scroll_direction": step.action.scroll_direction,
                "scroll_amount": step.action.scroll_amount,
            },
            "reasoning": step.reasoning,
        }
        steps.append(step_dict)

    return {
        "episode_id": episode.episode_id,
        "instruction": episode.instruction,
        "steps": steps,
        "success": episode.success,
        "metadata": episode.metadata,
    }
