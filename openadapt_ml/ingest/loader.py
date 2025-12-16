"""Episode loading utilities for openadapt-ml.

Load Episodes from JSON files exported by external systems.
This is the primary entry point for users who have their own data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openadapt_ml.schemas.sessions import Action, Episode, Observation, Step
from openadapt_ml.schemas.validation import validate_episodes, summarize_episodes


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
        episodes = _load_episodes_from_file(path)
    elif path.is_dir():
        # Directory of JSON files
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {path}")

        for json_file in json_files:
            file_episodes = _load_episodes_from_file(json_file)
            episodes.extend(file_episodes)
    else:
        raise ValueError(f"Path must be a file or directory: {path}")

    if validate:
        warnings = validate_episodes(episodes, check_images=check_images)
        if warnings:
            print(f"Validation warnings ({len(warnings)}):")
            for w in warnings[:10]:  # Show first 10
                print(f"  - {w}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")

    return episodes


def _load_episodes_from_file(path: Path) -> List[Episode]:
    """Load episodes from a single JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        # List of episodes
        return [_dict_to_episode(ep) for ep in data]
    elif isinstance(data, dict):
        # Single episode or wrapped format
        if "episodes" in data:
            return [_dict_to_episode(ep) for ep in data["episodes"]]
        elif "id" in data and "goal" in data:
            # Single episode
            return [_dict_to_episode(data)]
        else:
            raise ValueError(f"Unrecognized JSON format in {path}")
    else:
        raise ValueError(f"Expected list or dict in {path}, got {type(data)}")


def _dict_to_episode(data: Dict[str, Any]) -> Episode:
    """Convert a dictionary to an Episode object."""
    steps = []
    for step_data in data.get("steps", []):
        # Parse observation
        obs_data = step_data.get("observation", {})
        observation = Observation(
            image_path=obs_data.get("image_path"),
            meta=obs_data.get("meta"),
            accessibility_tree=obs_data.get("accessibility_tree"),
            dom_html=obs_data.get("dom_html"),
            url=obs_data.get("url"),
            window_title=obs_data.get("window_title"),
            app_name=obs_data.get("app_name"),
            focused_element=obs_data.get("focused_element"),
        )

        # Parse action
        action_data = step_data.get("action", {})
        action = Action(
            type=action_data.get("type", "unknown"),
            x=action_data.get("x"),
            y=action_data.get("y"),
            text=action_data.get("text"),
            raw=action_data.get("raw"),
            bbox=tuple(action_data["bbox"]) if action_data.get("bbox") else None,
            element_index=action_data.get("element_index"),
            target_node_id=action_data.get("target_node_id"),
            target_role=action_data.get("target_role"),
            target_name=action_data.get("target_name"),
            key=action_data.get("key"),
            modifiers=action_data.get("modifiers"),
            scroll_direction=action_data.get("scroll_direction"),
            scroll_amount=action_data.get("scroll_amount"),
            end_x=action_data.get("end_x"),
            end_y=action_data.get("end_y"),
            answer=action_data.get("answer"),
        )

        step = Step(
            t=step_data.get("t", 0.0),
            observation=observation,
            action=action,
            thought=step_data.get("thought"),
        )
        steps.append(step)

    return Episode(
        id=data.get("id", "unknown"),
        goal=data.get("goal", ""),
        steps=steps,
        summary=data.get("summary"),
        success=data.get("success"),
        workflow_id=data.get("workflow_id"),
    )


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
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)


def _episode_to_dict(episode: Episode) -> Dict[str, Any]:
    """Convert an Episode object to a dictionary."""
    steps = []
    for step in episode.steps:
        step_dict = {
            "t": step.t,
            "observation": {
                "image_path": step.observation.image_path,
                "meta": step.observation.meta,
                "accessibility_tree": step.observation.accessibility_tree,
                "dom_html": step.observation.dom_html,
                "url": step.observation.url,
                "window_title": step.observation.window_title,
                "app_name": step.observation.app_name,
                "focused_element": step.observation.focused_element,
            },
            "action": {
                "type": step.action.type,
                "x": step.action.x,
                "y": step.action.y,
                "text": step.action.text,
                "raw": step.action.raw,
                "bbox": list(step.action.bbox) if step.action.bbox else None,
                "element_index": step.action.element_index,
                "target_node_id": step.action.target_node_id,
                "target_role": step.action.target_role,
                "target_name": step.action.target_name,
                "key": step.action.key,
                "modifiers": step.action.modifiers,
                "scroll_direction": step.action.scroll_direction,
                "scroll_amount": step.action.scroll_amount,
                "end_x": step.action.end_x,
                "end_y": step.action.end_y,
                "answer": step.action.answer,
            },
            "thought": step.thought,
        }
        steps.append(step_dict)

    return {
        "id": episode.id,
        "goal": episode.goal,
        "steps": steps,
        "summary": episode.summary,
        "success": episode.success,
        "workflow_id": episode.workflow_id,
    }
