"""Parquet export utilities for Episode trajectories.

Parquet is a derived format for analytics and governance.
Episode JSON remains the canonical representation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openadapt_ml.schemas.sessions import Episode


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

        for step_idx, step in enumerate(episode.steps):
            row = {
                "episode_id": episode.id,
                "goal": episode.goal,
                "workflow_id": getattr(episode, "workflow_id", None),
                "step_index": step_idx,
                "timestamp": step.t,
                "action_type": step.action.type if step.action else None,
                "x": step.action.x if step.action else None,
                "y": step.action.y if step.action else None,
                "end_x": getattr(step.action, "end_x", None) if step.action else None,
                "end_y": getattr(step.action, "end_y", None) if step.action else None,
                "text": getattr(step.action, "text", None) if step.action else None,
                "key": getattr(step.action, "key", None) if step.action else None,
                "scroll_direction": (
                    getattr(step.action, "scroll_direction", None)
                    if step.action
                    else None
                ),
                "image_path": (
                    step.observation.image_path if step.observation else None
                ),
                "window_title": (
                    getattr(step.observation, "window_title", None)
                    if step.observation
                    else None
                ),
                "app_name": (
                    getattr(step.observation, "app_name", None)
                    if step.observation
                    else None
                ),
                "url": (
                    getattr(step.observation, "url", None)
                    if step.observation
                    else None
                ),
                "thought": getattr(step, "thought", None),
                "episode_metadata": episode_metadata,
                "step_metadata": (
                    json.dumps(step.metadata)
                    if hasattr(step, "metadata") and step.metadata
                    else None
                ),
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
        first_t = episode.steps[0].t if episode.steps else None
        last_t = episode.steps[-1].t if episode.steps else None
        duration = (last_t - first_t) if first_t is not None and last_t is not None else None

        summary_rows.append({
            "episode_id": episode.id,
            "goal": episode.goal,
            "workflow_id": getattr(episode, "workflow_id", None),
            "step_count": len(episode.steps),
            "duration": duration,
            "success": getattr(episode, "success", None),
            "first_action_type": (
                episode.steps[0].action.type
                if episode.steps and episode.steps[0].action
                else None
            ),
            "last_action_type": (
                episode.steps[-1].action.type
                if episode.steps and episode.steps[-1].action
                else None
            ),
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

    from openadapt_ml.schemas.sessions import Action, Episode, Observation, Step

    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    episodes = []
    for episode_id, group in df.groupby("episode_id"):
        group = group.sort_values("step_index")

        steps = []
        for _, row in group.iterrows():
            observation = Observation(
                image_path=row.get("image_path"),
                window_title=row.get("window_title"),
                app_name=row.get("app_name"),
                url=row.get("url"),
            )

            action = None
            if row.get("action_type"):
                action = Action(
                    type=row["action_type"],
                    x=row.get("x"),
                    y=row.get("y"),
                    end_x=row.get("end_x"),
                    end_y=row.get("end_y"),
                    text=row.get("text"),
                    key=row.get("key"),
                    scroll_direction=row.get("scroll_direction"),
                )

            step = Step(
                t=row.get("timestamp", 0.0),
                observation=observation,
                action=action,
                thought=row.get("thought"),
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
            id=str(episode_id),
            goal=group.iloc[0].get("goal", ""),
            steps=steps,
            workflow_id=group.iloc[0].get("workflow_id"),
            metadata=metadata,
        )
        episodes.append(episode)

    return episodes
