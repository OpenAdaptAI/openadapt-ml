"""Schema definitions and validation for openadapt-ml.

Core data structures:
    - Action: A single GUI action (click, type, scroll, etc.)
    - Observation: GUI state observation (screenshot, accessibility tree, etc.)
    - Step: One timestep containing observation + action
    - Episode: A single task attempt / workflow instance
    - Session: Container for multiple episodes

Validation:
    - validate_episode(): Validate an Episode object
    - validate_session(): Validate a Session object
    - validate_episodes(): Validate a list of Episodes
    - ValidationError: Raised on schema violations
"""

from openadapt_ml.schemas.sessions import (
    Action,
    ActionType,
    Episode,
    Observation,
    Session,
    Step,
)
from openadapt_ml.schemas.validation import (
    ValidationError,
    summarize_episodes,
    validate_action,
    validate_episode,
    validate_episodes,
    validate_observation,
    validate_session,
    validate_step,
)

__all__ = [
    # Core types
    "Action",
    "ActionType",
    "Episode",
    "Observation",
    "Session",
    "Step",
    # Validation
    "ValidationError",
    "validate_action",
    "validate_episode",
    "validate_episodes",
    "validate_observation",
    "validate_session",
    "validate_step",
    "summarize_episodes",
]
