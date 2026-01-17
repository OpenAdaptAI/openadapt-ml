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

    # Event types to include in segmentation
    RELEVANT_EVENT_TYPES = {
        "click",
        "double_click",
        "right_click",
        "key",
        "type",
        "scroll",
        "drag",
        "move",
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

        # Query events
        cursor.execute(
            """
            SELECT id, timestamp, type, data
            FROM events
            WHERE type IN ({})
            ORDER BY timestamp
            """.format(",".join("?" * len(self.RELEVANT_EVENT_TYPES))),
            tuple(self.RELEVANT_EVENT_TYPES),
        )

        images = []
        events = []
        screenshot_files = self._get_screenshot_files(screenshots_dir)

        last_move_pos = None
        frame_index = 0

        for row in cursor.fetchall():
            event_id = row["id"]
            timestamp = row["timestamp"]
            event_type = row["type"]
            data_json = row["data"]

            try:
                data = json.loads(data_json) if data_json else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for event {event_id}")
                continue

            # Skip moves if not including or too close to last move
            if event_type == "move":
                if not self.include_moves:
                    continue
                if last_move_pos:
                    x, y = data.get("x"), data.get("y")
                    if x is not None and y is not None:
                        dx = x - last_move_pos[0]
                        dy = y - last_move_pos[1]
                        distance = (dx**2 + dy**2) ** 0.5
                        if distance < self.min_move_distance:
                            continue
                last_move_pos = (data.get("x"), data.get("y"))

            # Find corresponding screenshot
            screenshot_path = self._find_screenshot(
                screenshot_files, frame_index, event_id
            )

            if screenshot_path:
                try:
                    images.append(Image.open(screenshot_path))

                    # Convert to expected format
                    event = self._convert_event(
                        event_type=event_type,
                        timestamp=timestamp - started_at,  # Relative to start
                        frame_index=frame_index,
                        data=data,
                    )
                    events.append(event)

                    frame_index += 1

                except Exception as e:
                    logger.warning(f"Failed to load screenshot {screenshot_path}: {e}")

        conn.close()

        if not images:
            raise ValueError(f"No screenshots loaded from {capture_path}")

        logger.info(f"Loaded {len(images)} frames from {capture_path}")
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
