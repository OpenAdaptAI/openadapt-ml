"""
Converters for benchmark-specific episode formats.

Supported formats:
- WAA (Windows Agent Arena)
- WebArena (coming soon)
- OSWorld (coming soon)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Union

from openadapt_ml.schema.episode import (
    Action,
    ActionType,
    BenchmarkSource,
    Coordinates,
    Episode,
    Observation,
    Step,
    UIElement,
)


# ============================================================================
# WAA (Windows Agent Arena) Converter
# ============================================================================

def _parse_waa_action(action_str: str) -> tuple[ActionType, dict[str, Any]]:
    """Parse WAA action string into ActionType and parameters.

    WAA action format examples:
    - pyautogui.click(100, 200)
    - pyautogui.write('hello')
    - pyautogui.press('enter')
    - pyautogui.hotkey('ctrl', 'c')
    - pyautogui.scroll(3)
    - DONE
    - FAIL
    """
    action_str = action_str.strip()

    # Meta actions
    if action_str == "DONE":
        return ActionType.DONE, {}
    if action_str == "FAIL":
        return ActionType.FAIL, {}

    # Parse pyautogui calls
    match = re.match(r"pyautogui\.(\w+)\((.*)\)", action_str)
    if not match:
        # Try without pyautogui prefix
        match = re.match(r"(\w+)\((.*)\)", action_str)

    if match:
        func_name = match.group(1).lower()
        args_str = match.group(2)

        # Parse arguments (handle strings with commas inside)
        args = []
        current_arg = ""
        in_string = False
        string_char = None

        for char in args_str:
            if char in "'\"" and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
            elif char == "," and not in_string:
                if current_arg.strip():
                    args.append(current_arg.strip())
                current_arg = ""
                continue
            current_arg += char

        if current_arg.strip():
            args.append(current_arg.strip())

        # Clean up string arguments
        cleaned_args = []
        for arg in args:
            arg = arg.strip()
            if (arg.startswith("'") and arg.endswith("'")) or (
                arg.startswith('"') and arg.endswith('"')
            ):
                cleaned_args.append(arg[1:-1])
            else:
                try:
                    cleaned_args.append(int(arg))
                except ValueError:
                    try:
                        cleaned_args.append(float(arg))
                    except ValueError:
                        cleaned_args.append(arg)

        # Map function to action type
        if func_name == "click":
            params = {}
            if len(cleaned_args) >= 2:
                params["coordinates"] = Coordinates(x=int(cleaned_args[0]), y=int(cleaned_args[1]))
            return ActionType.CLICK, params

        elif func_name == "doubleclick":
            params = {}
            if len(cleaned_args) >= 2:
                params["coordinates"] = Coordinates(x=int(cleaned_args[0]), y=int(cleaned_args[1]))
            return ActionType.DOUBLE_CLICK, params

        elif func_name == "rightclick":
            params = {}
            if len(cleaned_args) >= 2:
                params["coordinates"] = Coordinates(x=int(cleaned_args[0]), y=int(cleaned_args[1]))
            return ActionType.RIGHT_CLICK, params

        elif func_name in ("write", "typewrite"):
            return ActionType.TYPE, {"text": cleaned_args[0] if cleaned_args else ""}

        elif func_name == "press":
            return ActionType.KEY, {"key": cleaned_args[0] if cleaned_args else ""}

        elif func_name == "hotkey":
            if len(cleaned_args) >= 2:
                return ActionType.HOTKEY, {
                    "key": cleaned_args[-1],
                    "modifiers": list(cleaned_args[:-1]),
                }
            return ActionType.KEY, {"key": cleaned_args[0] if cleaned_args else ""}

        elif func_name == "scroll":
            amount = cleaned_args[0] if cleaned_args else 0
            direction = "up" if amount > 0 else "down"
            return ActionType.SCROLL, {
                "scroll_direction": direction,
                "scroll_amount": abs(int(amount)) * 100,  # Convert to pixels
            }

        elif func_name == "moveto":
            params = {}
            if len(cleaned_args) >= 2:
                params["coordinates"] = Coordinates(x=int(cleaned_args[0]), y=int(cleaned_args[1]))
            return ActionType.HOVER, params

        elif func_name == "drag" or func_name == "dragto":
            params = {}
            if len(cleaned_args) >= 2:
                params["end_coordinates"] = Coordinates(
                    x=int(cleaned_args[0]), y=int(cleaned_args[1])
                )
            return ActionType.DRAG, params

    # Fallback - treat as raw text if nothing matched
    return ActionType.TYPE, {"text": action_str, "raw": {"original": action_str}}


def from_waa_trajectory(
    trajectory: list[dict[str, Any]],
    task_info: dict[str, Any],
    episode_id: Optional[str] = None,
) -> Episode:
    """Convert WAA trajectory format to Episode.

    Args:
        trajectory: List of WAA step dictionaries with keys like:
            - screenshot_path: Path to screenshot
            - action: Action string (pyautogui format)
            - a11y_tree: Accessibility tree (optional)
            - thought: Agent reasoning (optional)
        task_info: Task metadata with keys like:
            - id: Task ID
            - instruction: Task instruction
            - domain: Task domain (file_explorer, etc.)

    Returns:
        Episode instance
    """
    steps = []

    for i, step_data in enumerate(trajectory):
        # Parse observation
        observation = Observation(
            screenshot_path=step_data.get("screenshot_path"),
            a11y_tree=step_data.get("a11y_tree"),
            window_title=step_data.get("window_title"),
            raw=step_data.get("observation_raw"),
        )

        # Parse action
        action_str = step_data.get("action", "")
        action_type, action_params = _parse_waa_action(action_str)

        action = Action(
            type=action_type,
            raw={"original": action_str},
            **action_params,
        )

        # Create step
        step = Step(
            step_index=i,
            observation=observation,
            action=action,
            reasoning=step_data.get("thought") or step_data.get("reasoning"),
            reward=step_data.get("reward"),
            done=step_data.get("done"),
        )
        steps.append(step)

    # Extract task info
    task_id = task_info.get("id") or task_info.get("task_id")
    instruction = task_info.get("instruction") or task_info.get("goal", "")

    if episode_id is None:
        episode_id = f"waa_{task_id}" if task_id else f"waa_episode_{id(trajectory)}"

    return Episode(
        episode_id=episode_id,
        task_id=task_id,
        instruction=instruction,
        steps=steps,
        success=task_info.get("success"),
        source=BenchmarkSource.WAA,
        metadata={
            "domain": task_info.get("domain"),
            "difficulty": task_info.get("difficulty"),
            **{k: v for k, v in task_info.items() if k not in ["id", "task_id", "instruction", "goal", "success", "domain", "difficulty"]},
        },
    )


def to_waa_trajectory(episode: Episode) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert Episode to WAA trajectory format.

    Args:
        episode: Episode instance

    Returns:
        Tuple of (trajectory, task_info)
    """
    trajectory = []

    for step in episode.steps:
        step_data = {
            "screenshot_path": step.observation.screenshot_path,
            "a11y_tree": step.observation.a11y_tree,
            "window_title": step.observation.window_title,
        }

        # Convert action back to pyautogui format
        action = step.action
        if action.raw and "original" in action.raw:
            step_data["action"] = action.raw["original"]
        else:
            step_data["action"] = _action_to_pyautogui(action)

        if step.reasoning:
            step_data["thought"] = step.reasoning

        if step.reward is not None:
            step_data["reward"] = step.reward

        if step.done is not None:
            step_data["done"] = step.done

        trajectory.append(step_data)

    task_info = {
        "id": episode.task_id,
        "instruction": episode.instruction,
        "success": episode.success,
    }

    if episode.metadata:
        task_info.update(episode.metadata)

    return trajectory, task_info


def _action_to_pyautogui(action: Action) -> str:
    """Convert Action to pyautogui string format."""
    if action.type == ActionType.DONE:
        return "DONE"
    if action.type == ActionType.FAIL:
        return "FAIL"

    if action.type == ActionType.CLICK:
        if action.coordinates:
            return f"pyautogui.click({action.coordinates.x}, {action.coordinates.y})"
        return "pyautogui.click()"

    if action.type == ActionType.DOUBLE_CLICK:
        if action.coordinates:
            return f"pyautogui.doubleClick({action.coordinates.x}, {action.coordinates.y})"
        return "pyautogui.doubleClick()"

    if action.type == ActionType.RIGHT_CLICK:
        if action.coordinates:
            return f"pyautogui.rightClick({action.coordinates.x}, {action.coordinates.y})"
        return "pyautogui.rightClick()"

    if action.type == ActionType.TYPE:
        text = action.text or ""
        # Escape single quotes
        text = text.replace("'", "\\'")
        return f"pyautogui.write('{text}')"

    if action.type == ActionType.KEY:
        return f"pyautogui.press('{action.key}')"

    if action.type == ActionType.HOTKEY:
        modifiers = action.modifiers or []
        keys = modifiers + [action.key]
        keys_str = ", ".join(f"'{k}'" for k in keys)
        return f"pyautogui.hotkey({keys_str})"

    if action.type == ActionType.SCROLL:
        amount = action.scroll_amount or 100
        if action.scroll_direction in ("down", "right"):
            amount = -amount
        return f"pyautogui.scroll({amount // 100})"

    if action.type == ActionType.HOVER:
        if action.coordinates:
            return f"pyautogui.moveTo({action.coordinates.x}, {action.coordinates.y})"
        return "pyautogui.moveTo()"

    if action.type == ActionType.DRAG:
        if action.end_coordinates:
            return f"pyautogui.dragTo({action.end_coordinates.x}, {action.end_coordinates.y})"
        return "pyautogui.drag()"

    return f"# Unknown action: {action.type}"


def load_waa_result(result_dir: Union[str, Path]) -> Episode:
    """Load episode from WAA result directory.

    WAA result directories contain:
    - result.txt: Final score
    - trajectory.json or similar: Step-by-step data

    Args:
        result_dir: Path to WAA result directory

    Returns:
        Episode instance
    """
    result_dir = Path(result_dir)

    # Try to find trajectory file
    trajectory_files = list(result_dir.glob("*trajectory*.json")) + list(
        result_dir.glob("*steps*.json")
    )

    trajectory = []
    task_info = {}

    if trajectory_files:
        with open(trajectory_files[0]) as f:
            data = json.load(f)
            if isinstance(data, list):
                trajectory = data
            elif isinstance(data, dict):
                trajectory = data.get("steps", data.get("trajectory", []))
                task_info = {k: v for k, v in data.items() if k not in ["steps", "trajectory"]}

    # Try to read result
    result_file = result_dir / "result.txt"
    if result_file.exists():
        with open(result_file) as f:
            result_str = f.read().strip()
            try:
                task_info["success"] = float(result_str) > 0
            except ValueError:
                pass

    # Try to get task info from parent directory name
    task_id = result_dir.name
    if task_id and "task_id" not in task_info:
        task_info["task_id"] = task_id

    return from_waa_trajectory(
        trajectory, task_info, episode_id=f"waa_{task_id}"
    )
