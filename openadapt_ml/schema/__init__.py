"""Episode schema module for standardized GUI trajectory data.

This module provides a canonical contract for GUI trajectory/episode data,
enabling interoperability between different benchmarks (WAA, WebArena, OSWorld).

Usage:
    from openadapt_ml.schema import Episode, Step, Action, Observation

    # Create an episode
    episode = Episode(
        episode_id="demo_001",
        instruction="Open Notepad",
        steps=[...]
    )

    # Save/load
    save_episode(episode, "episode.json")
    episode = load_episode("episode.json")

    # Convert from WAA format
    from openadapt_ml.schema.converters import from_waa_trajectory
    episode = from_waa_trajectory(trajectory, task_info)
"""

from openadapt_ml.schema.episode import (
    SCHEMA_VERSION,
    Episode,
    Step,
    Action,
    Observation,
    ActionType,
    BenchmarkSource,
    Coordinates,
    BoundingBox,
    UIElement,
    validate_episode,
    load_episode,
    save_episode,
    export_json_schema,
)

__all__ = [
    # Version
    "SCHEMA_VERSION",
    # Core models
    "Episode",
    "Step",
    "Action",
    "Observation",
    # Supporting models
    "ActionType",
    "BenchmarkSource",
    "Coordinates",
    "BoundingBox",
    "UIElement",
    # Utilities
    "validate_episode",
    "load_episode",
    "save_episode",
    "export_json_schema",
]
