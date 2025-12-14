"""Adapter for converting openadapt-capture recordings to openadapt-ml Episode format.

This module provides functions to ingest real GUI recordings from openadapt-capture
and convert them to the Episode/Step format used by openadapt-ml for training.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from openadapt_ml.schemas.sessions import Action, Episode, Observation, Session, Step

if TYPE_CHECKING:
    from PIL import Image

# Event type mapping from openadapt-capture to openadapt-ml
EVENT_TYPE_MAP = {
    "mouse.singleclick": "click",
    "mouse.click": "click",
    "mouse.doubleclick": "double_click",
    "mouse.drag": "drag",
    "mouse.scroll": "scroll",
    "key.type": "type",
    "key.down": "key_press",
    "key.up": "key_press",
}


def _normalize_coords(
    x: float | None,
    y: float | None,
    screen_width: int,
    screen_height: int,
) -> tuple[float | None, float | None]:
    """Normalize pixel coordinates to [0, 1] range.

    Args:
        x: X coordinate in pixels.
        y: Y coordinate in pixels.
        screen_width: Screen width in pixels.
        screen_height: Screen height in pixels.

    Returns:
        Tuple of (normalized_x, normalized_y).
    """
    if x is None or y is None:
        return None, None
    return x / screen_width, y / screen_height


def _save_screenshot(
    image: "Image",
    output_dir: Path,
    episode_id: str,
    step_idx: int,
) -> str:
    """Save a screenshot and return its path.

    Args:
        image: PIL Image to save.
        output_dir: Directory to save images to.
        episode_id: Episode identifier.
        step_idx: Step index.

    Returns:
        Path to saved image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{episode_id}_step_{step_idx}.png"
    filepath = output_dir / filename
    image.save(filepath)
    return str(filepath)


def capture_to_episode(
    capture_path: str | Path,
    output_dir: str | Path | None = None,
    goal: str | None = None,
    episode_id: str | None = None,
    include_moves: bool = False,
) -> Episode:
    """Convert an openadapt-capture recording to an Episode.

    Args:
        capture_path: Path to the capture directory.
        output_dir: Directory to save extracted screenshots. If None, uses
                    capture_path/screenshots.
        goal: Task description/goal for the episode. If None, uses capture's
              task_description or a generic message.
        episode_id: Identifier for the episode. If None, generates a UUID.
        include_moves: Whether to include mouse move events.

    Returns:
        Episode containing Steps with Observations and Actions.

    Raises:
        ImportError: If openadapt-capture is not installed.
        FileNotFoundError: If capture doesn't exist.
    """
    try:
        from openadapt_capture import Capture
        from openadapt_capture.events import (
            EventType,
            KeyTypeEvent,
            MouseClickEvent,
            MouseDoubleClickEvent,
            MouseDragEvent,
            MouseScrollEvent,
        )
    except ImportError as e:
        raise ImportError(
            "openadapt-capture is required. Install with: pip install openadapt-capture"
        ) from e

    capture_path = Path(capture_path)
    if output_dir is None:
        output_dir = capture_path / "screenshots"
    output_dir = Path(output_dir)

    # Load capture
    capture = Capture.load(capture_path)

    # Generate episode ID if not provided
    if episode_id is None:
        episode_id = f"capture_{capture.id}"

    # Get goal from capture or derive from context
    if goal is None:
        if capture.task_description:
            goal = capture.task_description
        else:
            # Try to derive goal from directory name (e.g., "turn-off-nightshift" -> "Turn off nightshift")
            dir_name = capture_path.name
            if dir_name and dir_name != "capture":
                # Convert kebab-case/snake_case to readable text
                goal = dir_name.replace("-", " ").replace("_", " ").strip().capitalize()
            else:
                goal = "Complete the recorded workflow"

    # Get screen dimensions for coordinate normalization
    screen_width, screen_height = capture.screen_size

    steps: list[Step] = []
    start_time = capture.started_at

    for idx, action in enumerate(capture.actions(include_moves=include_moves)):
        # Get screenshot at action time
        screenshot = action.screenshot
        if screenshot is None:
            continue

        # Save screenshot
        image_path = _save_screenshot(screenshot, output_dir, episode_id, idx)

        # Normalize coordinates
        norm_x, norm_y = _normalize_coords(
            action.x, action.y, screen_width, screen_height
        )

        # Map event type to openadapt-ml action type
        event_type = action.type
        action_type = EVENT_TYPE_MAP.get(event_type, "click")

        # Build Action object
        ml_action = Action(
            type=action_type,
            x=norm_x,
            y=norm_y,
            text=action.text,
        )

        # Handle drag events - add end coordinates
        if isinstance(action.event, MouseDragEvent):
            end_x = action.event.x + action.event.dx
            end_y = action.event.y + action.event.dy
            norm_end_x, norm_end_y = _normalize_coords(
                end_x, end_y, screen_width, screen_height
            )
            ml_action.raw = {
                "end_x": norm_end_x,
                "end_y": norm_end_y,
                "button": action.event.button,
            }

        # Handle scroll events
        if isinstance(action.event, MouseScrollEvent):
            ml_action.raw = {
                "dx": action.event.dx,
                "dy": action.event.dy,
            }

        # Handle keyboard events - include key names for special keys
        if action.keys:
            if ml_action.raw is None:
                ml_action.raw = {}
            ml_action.raw["keys"] = action.keys

        # Create Step
        step = Step(
            t=action.timestamp - start_time,
            observation=Observation(image_path=image_path),
            action=ml_action,
            thought=None,  # Real recordings don't have thoughts
        )
        steps.append(step)

    # Add terminal DONE action if there are steps
    if steps:
        # Use the last screenshot for the done action
        last_step = steps[-1]
        done_step = Step(
            t=last_step.t + 0.1,
            observation=Observation(image_path=last_step.observation.image_path),
            action=Action(type="done"),
            thought="Workflow complete.",
        )
        steps.append(done_step)

    capture.close()

    return Episode(
        id=episode_id,
        goal=goal,
        steps=steps,
        summary=f"Real recording with {len(steps)} steps",
        success=True,
        workflow_id=capture.id,
    )


def capture_to_session(
    capture_path: str | Path,
    output_dir: str | Path | None = None,
    goal: str | None = None,
    session_id: str | None = None,
    include_moves: bool = False,
) -> Session:
    """Convert an openadapt-capture recording to a Session.

    Args:
        capture_path: Path to the capture directory.
        output_dir: Directory to save extracted screenshots.
        goal: Task description/goal for the episode.
        session_id: Identifier for the session. If None, generates a UUID.
        include_moves: Whether to include mouse move events.

    Returns:
        Session containing a single Episode.
    """
    episode = capture_to_episode(
        capture_path=capture_path,
        output_dir=output_dir,
        goal=goal,
        include_moves=include_moves,
    )

    if session_id is None:
        session_id = f"session_{uuid.uuid4().hex[:8]}"

    return Session(
        id=session_id,
        episodes=[episode],
        meta={
            "source": "openadapt-capture",
            "capture_path": str(capture_path),
        },
    )


def load_captures_as_sessions(
    captures_dir: str | Path,
    output_dir: str | Path | None = None,
    include_moves: bool = False,
) -> list[Session]:
    """Load multiple captures from a directory.

    Scans for subdirectories containing capture.db files.

    Args:
        captures_dir: Directory containing capture subdirectories.
        output_dir: Base directory for screenshots. Each capture gets a subdirectory.
        include_moves: Whether to include mouse move events.

    Returns:
        List of Sessions, one per capture.
    """
    captures_dir = Path(captures_dir)
    sessions = []

    # Find all capture.db files
    for db_path in captures_dir.glob("**/capture.db"):
        capture_path = db_path.parent

        # Determine output directory for this capture
        if output_dir is not None:
            capture_output = Path(output_dir) / capture_path.name
        else:
            capture_output = None

        try:
            session = capture_to_session(
                capture_path=capture_path,
                output_dir=capture_output,
                include_moves=include_moves,
            )
            sessions.append(session)
        except Exception as e:
            print(f"Warning: Failed to load {capture_path}: {e}")

    return sessions
