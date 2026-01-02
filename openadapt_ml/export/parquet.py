"""Parquet export utilities for Episode trajectories.

Parquet is a derived format for analytics and governance.
Episode JSON remains the canonical representation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openadapt_ml.schema import Episode


def to_parquet(
    episodes: list[Episode],
    output_path: str,
    flatten_steps: bool = True,
    include_summary: bool = False,
) -> None:
    """Export Episodes to Parquet for analytics.

    Creates a step-level Parquet file with one row per step.
    Episode-level fields are repeated for each step.

    Args:
        episodes: List of Episode objects to export.
        output_path: Path to output .parquet file.
        flatten_steps: If True, one row per step. If False, one row per episode
            with steps as nested structure (not yet implemented).
        include_summary: If True, also generate {output_path}_summary.parquet
            with episode-level aggregations.

    Raises:
        ImportError: If pyarrow is not installed.
        ValueError: If flatten_steps is False (not yet implemented).

    Example:
        >>> from openadapt_ml.ingest import load_episodes
        >>> from openadapt_ml.export import to_parquet
        >>> episodes = load_episodes("workflow_exports/")
        >>> to_parquet(episodes, "episodes.parquet")
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "Parquet export requires pyarrow. "
            "Install with: pip install openadapt-ml[parquet]"
        )

    if not flatten_steps:
        raise ValueError(
            "flatten_steps=False is not yet implemented. "
            "Use flatten_steps=True for step-level rows."
        )

    rows = []
    for episode in episodes:
        episode_metadata = None
        if hasattr(episode, "metadata") and episode.metadata:
            episode_metadata = json.dumps(episode.metadata)

        for step in episode.steps:
            # Extract normalized coordinates if available
            x, y = None, None
            if step.action and step.action.normalized_coordinates:
                x, y = step.action.normalized_coordinates

            # Extract action type value (enum -> string)
            action_type = None
            if step.action:
                action_type = step.action.type.value if hasattr(step.action.type, 'value') else step.action.type

            row = {
                "episode_id": episode.episode_id,
                "instruction": episode.instruction,
                "task_id": getattr(episode, "task_id", None),
                "step_index": step.step_index,
                "timestamp": step.timestamp,
                "action_type": action_type,
                "x": x,
                "y": y,
                "end_x": step.action.normalized_end[0] if step.action and step.action.normalized_end else None,
                "end_y": step.action.normalized_end[1] if step.action and step.action.normalized_end else None,
                "text": getattr(step.action, "text", None) if step.action else None,
                "key": getattr(step.action, "key", None) if step.action else None,
                "scroll_direction": (
                    getattr(step.action, "scroll_direction", None)
                    if step.action
                    else None
                ),
                "screenshot_path": (
                    step.observation.screenshot_path if step.observation else None
                ),
                "window_title": (
                    getattr(step.observation, "window_title", None)
                    if step.observation
                    else None
                ),
                "app_name": (
                    None  # Not in new schema at Observation level
                ),
                "url": (
                    None  # Not in new schema at Observation level
                ),
                "reasoning": getattr(step, "reasoning", None),
                "episode_metadata": episode_metadata,
            }
            rows.append(row)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)

    if include_summary:
        _write_summary(episodes, output_path)


def _write_summary(episodes: list[Episode], output_path: str) -> None:
    """Write episode-level summary Parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return

    summary_rows = []
    for episode in episodes:
        first_t = episode.steps[0].timestamp if episode.steps else None
        last_t = episode.steps[-1].timestamp if episode.steps else None
        duration = (last_t - first_t) if first_t is not None and last_t is not None else None

        # Extract action type values (enum -> string)
        first_action_type = None
        last_action_type = None
        if episode.steps and episode.steps[0].action:
            t = episode.steps[0].action.type
            first_action_type = t.value if hasattr(t, 'value') else t
        if episode.steps and episode.steps[-1].action:
            t = episode.steps[-1].action.type
            last_action_type = t.value if hasattr(t, 'value') else t

        summary_rows.append({
            "episode_id": episode.episode_id,
            "instruction": episode.instruction,
            "task_id": getattr(episode, "task_id", None),
            "step_count": len(episode.steps),
            "duration": duration,
            "success": getattr(episode, "success", None),
            "first_action_type": first_action_type,
            "last_action_type": last_action_type,
            "metadata": (
                json.dumps(episode.metadata)
                if hasattr(episode, "metadata") and episode.metadata
                else None
            ),
        })

    summary_table = pa.Table.from_pylist(summary_rows)
    summary_path = str(output_path).replace(".parquet", "_summary.parquet")
    pq.write_table(summary_table, summary_path)


def from_parquet(parquet_path: str) -> list[Episode]:
    """Load Episodes from Parquet (inverse of to_parquet).

    This is a lossy reconstruction. For full fidelity, always keep
    Episode JSON as the source of truth.

    Args:
        parquet_path: Path to the Parquet file created by to_parquet().

    Returns:
        List of reconstructed Episode objects.

    Raises:
        ImportError: If pyarrow is not installed.

    Note:
        - Metadata fields are deserialized from JSON strings
        - Step ordering is recovered from step_index
        - Episode boundaries are recovered from episode_id grouping
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "Parquet import requires pyarrow. "
            "Install with: pip install openadapt-ml[parquet]"
        )

    from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step

    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    episodes = []
    for episode_id, group in df.groupby("episode_id"):
        group = group.sort_values("step_index")

        steps = []
        for _, row in group.iterrows():
            observation = Observation(
                screenshot_path=row.get("screenshot_path") or row.get("image_path"),
                window_title=row.get("window_title"),
            )

            action = None
            if row.get("action_type"):
                # Convert string action type to ActionType enum
                action_type_str = row["action_type"]
                try:
                    action_type = ActionType(action_type_str)
                except ValueError:
                    action_type = ActionType.CLICK  # Default fallback

                # Build normalized coordinates tuple if x and y are present
                normalized_coords = None
                if row.get("x") is not None and row.get("y") is not None:
                    normalized_coords = (float(row["x"]), float(row["y"]))

                # Build normalized end coordinates for drag
                normalized_end = None
                if row.get("end_x") is not None and row.get("end_y") is not None:
                    normalized_end = (float(row["end_x"]), float(row["end_y"]))

                action = Action(
                    type=action_type,
                    normalized_coordinates=normalized_coords,
                    normalized_end=normalized_end,
                    text=row.get("text"),
                    key=row.get("key"),
                    scroll_direction=row.get("scroll_direction"),
                )

            step = Step(
                step_index=int(row.get("step_index", 0)),
                observation=observation,
                action=action,
                reasoning=row.get("reasoning") or row.get("thought"),
                timestamp=row.get("timestamp"),
            )
            steps.append(step)

        # Parse metadata if present
        metadata = None
        if group.iloc[0].get("episode_metadata"):
            try:
                metadata = json.loads(group.iloc[0]["episode_metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        episode = Episode(
            episode_id=str(episode_id),
            instruction=group.iloc[0].get("instruction") or group.iloc[0].get("goal", ""),
            steps=steps,
            task_id=group.iloc[0].get("task_id"),
            metadata=metadata,
        )
        episodes.append(episode)

    return episodes
