"""
Episode Schema - Canonical format for GUI trajectory data.

A standardized contract for representing GUI automation episodes, enabling
interoperability across training pipelines, benchmarks, and recording tools.

Installation:
    pip install openadapt-ml
    # or: uv add openadapt-ml

Basic Usage:
    from openadapt_ml.schema import Episode, Step, Action, Observation, ActionType

    # Create an episode
    episode = Episode(
        episode_id="demo_001",
        instruction="Open Notepad and type Hello World",
        steps=[
            Step(
                step_index=0,
                observation=Observation(screenshot_path="step_0.png"),
                action=Action(type=ActionType.CLICK, coordinates={"x": 100, "y": 200}),
            ),
            Step(
                step_index=1,
                observation=Observation(screenshot_path="step_1.png"),
                action=Action(type=ActionType.TYPE, text="Hello World"),
            ),
        ],
        success=True,
    )

    # Save/load JSON
    save_episode(episode, "episode.json")
    episode = load_episode("episode.json")

    # Validate external data
    is_valid, error = validate_episode({"episode_id": "x", ...})

Coordinate Systems:
    # Pixel coordinates (absolute)
    Action(type=ActionType.CLICK, coordinates={"x": 512, "y": 384})

    # Normalized coordinates (0.0-1.0, resolution-independent)
    Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.375))

    # Both can coexist - use whichever fits your pipeline

Converting from Other Formats:
    from openadapt_ml.schema.converters import from_waa_trajectory

    # Convert Windows Agent Arena format
    episode = from_waa_trajectory(trajectory_list, task_info_dict)

    # Convert back
    trajectory, task_info = to_waa_trajectory(episode)

JSON Schema Export:
    # For external validation tools (e.g., JSON Schema validators, TypeScript codegen)
    export_json_schema("episode.schema.json")

See Also:
    - docs/schema/episode.schema.json - Full JSON Schema
    - openadapt_ml.schema.episode - Model definitions
    - openadapt_ml.schema.converters - Format converters
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
