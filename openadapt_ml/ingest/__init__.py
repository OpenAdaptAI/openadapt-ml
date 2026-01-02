"""Data ingestion modules for openadapt-ml.

This package provides adapters for loading GUI interaction data from various sources
and converting them to the format used for training.

Data Model:
    - Episode: A single task attempt (e.g., "log into the app"). Contains a sequence
      of Steps, each with an Observation (screenshot) and Action (click/type/etc).

Functions:
    - load_episodes(): Load Episodes from JSON files (primary entry point)
    - save_episodes(): Save Episodes to JSON file
    - capture_to_episode(): Converts one openadapt-capture recording → one Episode
    - capture_to_session(): Converts one recording → Session containing one Episode
    - load_captures_as_sessions(): Loads multiple recordings → list of Sessions
    - generate_synthetic_episodes(): Creates synthetic training data
"""

from openadapt_ml.ingest.loader import load_episodes, save_episodes
from openadapt_ml.ingest.synthetic import generate_synthetic_episodes

__all__ = [
    "load_episodes",
    "save_episodes",
    "generate_synthetic_episodes",
]

# Conditionally export capture functions if openadapt-capture is installed
try:
    from openadapt_ml.ingest.capture import (
        capture_to_episode,
        capture_to_session,
        load_captures_as_sessions,
    )

    __all__.extend([
        "capture_to_episode",
        "capture_to_session",
        "load_captures_as_sessions",
    ])
except ImportError:
    pass
