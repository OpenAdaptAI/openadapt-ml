"""Configuration for baseline adapters.

Defines track types, model registry, and configuration dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrackType(str, Enum):
    """Baseline evaluation track types.

    TRACK_A: Direct coordinate prediction (CLICK(x, y))
    TRACK_B: ReAct-style reasoning with coordinates
    TRACK_C: Set-of-Mark element selection (CLICK([id]))
    """

    TRACK_A = "direct_coords"
    TRACK_B = "react_coords"
    TRACK_C = "set_of_mark"


@dataclass
class TrackConfig:
    """Configuration for a specific evaluation track.

    Attributes:
        track_type: The track type (A, B, or C).
        output_format: Expected output format string.
        use_som: Whether to use Set-of-Mark overlay.
        use_a11y_tree: Whether to include accessibility tree.
        max_a11y_elements: Max elements in a11y tree (truncation).
        include_reasoning: Whether to request reasoning steps.
        include_history: Whether to include action history.
        max_history_steps: Max history steps to include.
    """

    track_type: TrackType
    output_format: str
    use_som: bool = False
    use_a11y_tree: bool = True
    max_a11y_elements: int = 50
    include_reasoning: bool = False
    include_history: bool = True
    max_history_steps: int = 5

    @classmethod
    def track_a(cls) -> "TrackConfig":
        """Create Track A (Direct Coordinates) config."""
        return cls(
            track_type=TrackType.TRACK_A,
            output_format='{"action": "CLICK", "x": float, "y": float}',
            use_som=False,
            use_a11y_tree=True,
            include_reasoning=False,
        )

    @classmethod
    def track_b(cls) -> "TrackConfig":
        """Create Track B (ReAct with Coordinates) config."""
        return cls(
            track_type=TrackType.TRACK_B,
            output_format='{"thought": str, "action": "CLICK", "x": float, "y": float}',
            use_som=False,
            use_a11y_tree=True,
            include_reasoning=True,
        )

    @classmethod
    def track_c(cls) -> "TrackConfig":
        """Create Track C (Set-of-Mark) config."""
        return cls(
            track_type=TrackType.TRACK_C,
            output_format='{"action": "CLICK", "element_id": int}',
            use_som=True,
            use_a11y_tree=True,
            include_reasoning=False,
        )


@dataclass
class ModelSpec:
    """Specification for a supported model.

    Attributes:
        provider: Provider name (anthropic, openai, google).
        model_id: Full model identifier for the API.
        display_name: Human-readable name.
        is_default: Whether this is the default for the provider.
        max_tokens: Default max tokens for this model.
        supports_vision: Whether the model supports images.
    """

    provider: str
    model_id: str
    display_name: str
    is_default: bool = False
    max_tokens: int = 1024
    supports_vision: bool = True


# Model registry
MODELS: dict[str, ModelSpec] = {
    # Anthropic Claude
    "claude-opus-4.5": ModelSpec(
        provider="anthropic",
        model_id="claude-opus-4-5-20251101",
        display_name="Claude Opus 4.5",
        is_default=True,
        max_tokens=4096,
    ),
    "claude-sonnet-4.5": ModelSpec(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        display_name="Claude Sonnet 4.5",
        max_tokens=4096,
    ),
    # OpenAI GPT
    "gpt-5.2": ModelSpec(
        provider="openai",
        model_id="gpt-5.2",
        display_name="GPT-5.2",
        is_default=True,
        max_tokens=4096,
    ),
    "gpt-5.1": ModelSpec(
        provider="openai",
        model_id="gpt-5.1",
        display_name="GPT-5.1",
        max_tokens=4096,
    ),
    "gpt-4o": ModelSpec(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        max_tokens=4096,
    ),
    # Google Gemini
    "gemini-3-pro": ModelSpec(
        provider="google",
        model_id="gemini-3-pro",
        display_name="Gemini 3 Pro",
        is_default=True,
        max_tokens=4096,
    ),
    "gemini-3-flash": ModelSpec(
        provider="google",
        model_id="gemini-3-flash",
        display_name="Gemini 3 Flash",
        max_tokens=4096,
    ),
    "gemini-2.5-pro": ModelSpec(
        provider="google",
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        max_tokens=4096,
    ),
    "gemini-2.5-flash": ModelSpec(
        provider="google",
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        max_tokens=4096,
    ),
}


def get_model_spec(model_alias: str) -> ModelSpec:
    """Get model specification by alias.

    Args:
        model_alias: Model alias (e.g., 'claude-opus-4.5').

    Returns:
        ModelSpec for the model.

    Raises:
        ValueError: If alias not recognized.
    """
    if model_alias not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_alias}. Available: {available}")
    return MODELS[model_alias]


def get_default_model(provider: str) -> ModelSpec:
    """Get default model for a provider.

    Args:
        provider: Provider name (anthropic, openai, google).

    Returns:
        Default ModelSpec for the provider.

    Raises:
        ValueError: If no default found.
    """
    for spec in MODELS.values():
        if spec.provider == provider and spec.is_default:
            return spec
    raise ValueError(f"No default model for provider: {provider}")


@dataclass
class BaselineConfig:
    """Configuration for a baseline adapter run.

    Attributes:
        provider: Provider name or model alias.
        model: Model identifier (full ID or alias).
        track: Track configuration.
        api_key: Optional API key (defaults to env).
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        demo: Optional demo text to include.
        verbose: Whether to log verbose output.
    """

    provider: str
    model: str
    track: TrackConfig = field(default_factory=TrackConfig.track_a)
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1024
    demo: str | None = None
    verbose: bool = False

    def __post_init__(self):
        """Resolve model alias if needed."""
        # If provider is actually a model alias, resolve it
        if self.provider in MODELS:
            spec = MODELS[self.provider]
            self.provider = spec.provider
            self.model = spec.model_id
        # If model is an alias, resolve it
        elif self.model in MODELS:
            spec = MODELS[self.model]
            self.model = spec.model_id

    @classmethod
    def from_alias(
        cls,
        model_alias: str,
        track: TrackConfig | None = None,
        **kwargs: Any,
    ) -> "BaselineConfig":
        """Create config from model alias.

        Args:
            model_alias: Model alias (e.g., 'claude-opus-4.5').
            track: Track config (defaults to Track A).
            **kwargs: Additional config options.

        Returns:
            BaselineConfig instance.
        """
        spec = get_model_spec(model_alias)
        return cls(
            provider=spec.provider,
            model=spec.model_id,
            track=track or TrackConfig.track_a(),
            **kwargs,
        )
