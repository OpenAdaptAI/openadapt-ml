"""Adapter for openadapt-capture SQLite database format.

This adapter loads recordings from the openadapt-capture format
(capture.db SQLite database) and converts them to the format
expected by the segmentation pipeline.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class CaptureAdapter:
    """Adapter for openadapt-capture SQLite format.

    The openadapt-capture tool stores recordings in a SQLite database
    (capture.db) with the following structure:
    - capture table: Recording metadata
    - events table: Action events (click, type, scroll, etc.)
    - screenshots/: Directory with PNG files

    This adapter converts that format to the tuple of (images, events)
    expected by FrameDescriber.
    """

    # Event types to include in segmentation (actual openadapt-capture types)
    RELEVANT_EVENT_TYPES = {
        "mouse.down",
        "mouse.up",
        "key.down",
        "key.up",
        "mouse.move",
        "screen.frame",  # Frame captures (maps to screenshots)
    }

    def __init__(
        self,
        include_moves: bool = False,
        min_move_distance: float = 50.0,
    ):
        """Initialize the adapter.

        Args:
            include_moves: Whether to include mouse move events (can be noisy)
            min_move_distance: Minimum pixel distance for move events
        """
        self.include_moves = include_moves
        self.min_move_distance = min_move_distance

    def load_recording(
        self,
        capture_path: Path,
    ) -> tuple[list[Image.Image], list[dict]]:
        """Load recording from capture.db format.

        Args:
            capture_path: Path to recording directory with capture.db

        Returns:
            Tuple of (images, action_events) where:
            - images: List of PIL Images in chronological order
            - action_events: List of dicts with event data

        Raises:
            FileNotFoundError: If capture.db doesn't exist
            ValueError: If database format is invalid
        """
        db_path = capture_path / "capture.db"
        if not db_path.exists():
            raise FileNotFoundError(f"capture.db not found in {capture_path}")

        screenshots_dir = capture_path / "screenshots"
        if not screenshots_dir.exists():
            raise FileNotFoundError(
                f"screenshots directory not found in {capture_path}"
            )

        # Connect to SQLite
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()

        # Get capture metadata
        cursor.execute("SELECT * FROM capture LIMIT 1")
        capture_row = cursor.fetchone()
        if not capture_row:
            raise ValueError("No capture record found in database")

        capture_metadata = dict(capture_row)
        started_at = capture_metadata["started_at"]

        # Get all screen.frame events (these define our frames)
        cursor.execute(
            """
            SELECT id, timestamp, type, data
            FROM events
            WHERE type = 'screen.frame'
            ORDER BY timestamp
            """
        )

        frame_events = cursor.fetchall()
        logger.info(f"Found {len(frame_events)} screen.frame events")

        # Get all action events (mouse, key)
        cursor.execute(
            """
            SELECT id, timestamp, type, data
            FROM events
            WHERE type IN ('mouse.down', 'mouse.up', 'key.down', 'key.up', 'mouse.move')
            ORDER BY timestamp
            """
        )

        action_events = cursor.fetchall()
        logger.info(f"Found {len(action_events)} action events")

        # Pair action events (down+up → single action)
        paired_actions = self._pair_action_events(action_events, started_at)
        logger.info(f"Paired into {len(paired_actions)} actions")

        # Load screenshot files
        screenshot_files = self._get_screenshot_files(screenshots_dir)
        logger.info(f"Found {len(screenshot_files)} screenshot files")

        # Build frame list with corresponding actions
        images = []
        events = []

        for frame_idx, frame_row in enumerate(frame_events):
            frame_timestamp = frame_row["timestamp"]

            # Find screenshot
            screenshot_path = screenshot_files.get(frame_idx)
            if not screenshot_path:
                logger.warning(f"No screenshot found for frame {frame_idx}")
                continue

            try:
                # Load image
                images.append(Image.open(screenshot_path))

                # Find action closest to this frame (within reasonable window)
                frame_relative_time = frame_timestamp - started_at
                closest_action = self._find_closest_action(
                    paired_actions, frame_relative_time, window=2.0
                )

                if closest_action:
                    # Use action details
                    event = {
                        "timestamp": frame_relative_time,
                        "frame_index": frame_idx,
                        "name": closest_action["type"],
                        **closest_action.get("extra", {}),
                    }
                else:
                    # No action, create a frame-only event
                    event = {
                        "timestamp": frame_relative_time,
                        "frame_index": frame_idx,
                        "name": "frame",
                    }

                events.append(event)

            except Exception as e:
                logger.warning(f"Failed to load screenshot {screenshot_path}: {e}")

        conn.close()

        if not images:
            raise ValueError(f"No screenshots loaded from {capture_path}")

        logger.info(
            f"Loaded {len(images)} frames with {len(events)} events from {capture_path}"
        )
        return images, events

    def _get_screenshot_files(self, screenshots_dir: Path) -> dict[int, Path]:
        """Get mapping of frame indices to screenshot files.

        openadapt-capture uses format: capture_{id}_step_{n}.png

        Args:
            screenshots_dir: Path to screenshots directory

        Returns:
            Dict mapping frame index to file path
        """
        files = {}
        for png_file in screenshots_dir.glob("*.png"):
            # Parse format: capture_31807990_step_0.png
            parts = png_file.stem.split("_")
            if len(parts) >= 4 and parts[-2] == "step":
                try:
                    step_num = int(parts[-1])
                    files[step_num] = png_file
                except ValueError:
                    logger.warning(f"Could not parse step number from {png_file.name}")

        return files

    def _find_screenshot(
        self,
        screenshot_files: dict[int, Path],
        frame_index: int,
        event_id: Optional[int] = None,
    ) -> Optional[Path]:
        """Find screenshot file for frame index.

        Args:
            screenshot_files: Mapping of frame indices to paths
            frame_index: Current frame index
            event_id: Event ID (unused but kept for future)

        Returns:
            Path to screenshot or None if not found
        """
        return screenshot_files.get(frame_index)

    def _convert_event(
        self,
        event_type: str,
        timestamp: float,
        frame_index: int,
        data: dict,
    ) -> dict:
        """Convert openadapt-capture event to segmentation format.

        Args:
            event_type: Event type (click, type, scroll, etc.)
            timestamp: Timestamp in seconds (relative to recording start)
            frame_index: Frame index in sequence
            data: Event data dictionary

        Returns:
            Event dict in expected format
        """
        event = {
            "timestamp": timestamp,
            "frame_index": frame_index,
            "name": event_type,
        }

        # Add coordinates if present
        if "x" in data and "y" in data:
            event["mouse_x"] = data["x"]
            event["mouse_y"] = data["y"]

        # Add text for typing events
        if event_type in ("type", "key"):
            event["text"] = data.get("text") or data.get("key")

        # Add scroll direction
        if event_type == "scroll":
            event["scroll_dx"] = data.get("dx", 0)
            event["scroll_dy"] = data.get("dy", 0)

        # Add drag endpoints
        if event_type == "drag":
            event["start_x"] = data.get("start_x")
            event["start_y"] = data.get("start_y")
            event["end_x"] = data.get("end_x")
            event["end_y"] = data.get("end_y")

        return event

    def _pair_action_events(self, action_events: list, started_at: float) -> list[dict]:
        """Pair mouse.down+up and key.down+up events into single actions.

        Args:
            action_events: List of SQLite Row objects with action events
            started_at: Recording start timestamp

        Returns:
            List of paired action dicts with type, timestamp, duration, and data
        """
        paired = []
        pending_down = {}  # type -> (event, timestamp, data)

        for row in action_events:
            event_type = row["type"]
            timestamp = row["timestamp"] - started_at  # Relative
            data_json = row["data"]

            try:
                data = json.loads(data_json) if data_json else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for event {row['id']}")
                continue

            # Handle down events
            if event_type.endswith(".down"):
                base_type = event_type[:-5]  # Remove '.down' → 'mouse' or 'key'
                pending_down[base_type] = (event_type, timestamp, data)

            # Handle up events
            elif event_type.endswith(".up"):
                base_type = event_type[:-3]  # Remove '.up'

                if base_type in pending_down:
                    # Found matching down event
                    down_type, down_timestamp, down_data = pending_down.pop(base_type)
                    duration = timestamp - down_timestamp

                    # Create paired action
                    if base_type == "mouse":
                        action = {
                            "type": "click",
                            "timestamp": down_timestamp,
                            "duration": duration,
                            "extra": {
                                "mouse_x": down_data.get("x"),
                                "mouse_y": down_data.get("y"),
                                "button": down_data.get("button", "left"),
                            },
                        }
                    elif base_type == "key":
                        action = {
                            "type": "key",
                            "timestamp": down_timestamp,
                            "duration": duration,
                            "extra": {
                                "text": down_data.get("key") or down_data.get("text"),
                                "key": down_data.get("key"),
                            },
                        }
                    else:
                        continue

                    paired.append(action)
                else:
                    # Unpaired up event (shouldn't happen, but log it)
                    logger.debug(f"Unpaired {event_type} event at {timestamp}")

            # Handle mouse.move (if configured to include)
            elif event_type == "mouse.move" and self.include_moves:
                action = {
                    "type": "move",
                    "timestamp": timestamp,
                    "duration": 0.0,
                    "extra": {
                        "mouse_x": data.get("x"),
                        "mouse_y": data.get("y"),
                    },
                }
                paired.append(action)

        # Log any unpaired down events
        for base_type, (down_type, down_timestamp, down_data) in pending_down.items():
            logger.debug(f"Unpaired {down_type} event at {down_timestamp}")

        return paired

    def _find_closest_action(
        self, paired_actions: list[dict], frame_time: float, window: float = 2.0
    ) -> Optional[dict]:
        """Find action closest to a given frame time.

        Args:
            paired_actions: List of paired action dicts
            frame_time: Frame timestamp (relative to recording start)
            window: Maximum time distance in seconds to consider

        Returns:
            Closest action dict or None if no action within window
        """
        closest_action = None
        closest_distance = float("inf")

        for action in paired_actions:
            distance = abs(action["timestamp"] - frame_time)
            if distance < closest_distance and distance <= window:
                closest_distance = distance
                closest_action = action

        return closest_action

    def get_capture_metadata(self, capture_path: Path) -> dict:
        """Get recording metadata from capture.db.

        Args:
            capture_path: Path to recording directory

        Returns:
            Dict with capture metadata (task_description, platform, etc.)
        """
        db_path = capture_path / "capture.db"
        if not db_path.exists():
            raise FileNotFoundError(f"capture.db not found in {capture_path}")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM capture LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError("No capture record found")

        metadata = dict(row)

        # Parse JSON metadata field if present
        if "metadata" in metadata and metadata["metadata"]:
            try:
                extra_metadata = json.loads(metadata["metadata"])
                metadata.update(extra_metadata)
            except json.JSONDecodeError:
                pass

        return metadata
