"""Configuration for baseline adapters.

Defines track types, model registry, and configuration dataclasses.
Based on SOTA patterns from:
- Claude Computer Use API
- Microsoft UFO/UFO2
- OSWorld benchmark
- Agent-S/Agent-S2
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


class CoordinateSystem(str, Enum):
    """Coordinate system for action output.

    NORMALIZED: Coordinates in 0.0-1.0 range (relative to screen)
    PIXEL: Absolute pixel coordinates
    PERCENTAGE: Coordinates as percentages (0-100)
    """

    NORMALIZED = "normalized"
    PIXEL = "pixel"
    PERCENTAGE = "percentage"


class ActionOutputFormat(str, Enum):
    """Output format style for model responses.

    JSON: Structured JSON object
    FUNCTION_CALL: Function-style like CLICK(x, y)
    PYAUTOGUI: PyAutoGUI-style Python code (OSWorld compatible)
    """

    JSON = "json"
    FUNCTION_CALL = "function_call"
    PYAUTOGUI = "pyautogui"


@dataclass
class SoMConfig:
    """Configuration for Set-of-Mark (SoM) overlay.

    Controls how UI elements are labeled and displayed.
    Based on patterns from SoM paper and OMNI-parser.

    Attributes:
        overlay_enabled: Whether to draw element overlays on screenshot.
        label_format: Format for element labels ("[{id}]", "{id}", "e{id}").
        font_size: Font size for labels in pixels.
        label_background_color: RGBA tuple for label background.
        label_text_color: RGB tuple for label text.
        max_elements: Maximum elements to include (0=unlimited).
        include_roles: Element roles to include (None=all).
        exclude_roles: Element roles to exclude.
        min_element_area: Minimum element area in pixels to include.
        include_invisible: Whether to include non-visible elements.
    """

    overlay_enabled: bool = True
    label_format: str = "[{id}]"  # "[1]", "1", "e1"
    font_size: int = 12
    label_background_color: tuple[int, int, int, int] = (0, 120, 255, 200)  # Blue
    label_text_color: tuple[int, int, int] = (255, 255, 255)  # White
    max_elements: int = 100
    include_roles: list[str] | None = None  # None = include all
    exclude_roles: list[str] = field(
        default_factory=lambda: ["group", "generic", "static_text", "separator"]
    )
    min_element_area: int = 100  # Minimum bbox area in pixels
    include_invisible: bool = False


@dataclass
class ReActConfig:
    """Configuration for ReAct-style reasoning.

    Controls the observation-thought-action cycle used in Track B.
    Based on ReAct paper and UFO's Observation->Thought->Action pattern.

    Attributes:
        require_observation: Whether to require explicit observation.
        require_thought: Whether to require reasoning explanation.
        require_plan: Whether to require multi-step plan.
        max_plan_steps: Maximum steps in plan output.
        thinking_budget: Token budget for thinking (Claude extended thinking).
    """

    require_observation: bool = True
    require_thought: bool = True
    require_plan: bool = False
    max_plan_steps: int = 5
    thinking_budget: int | None = None  # For Claude extended thinking


@dataclass
class ScreenConfig:
    """Screen/display configuration for coordinate handling.

    Attributes:
        width: Display width in pixels.
        height: Display height in pixels.
        coordinate_system: How coordinates are represented.
        scale_factor: DPI scale factor (1.0 = standard, 2.0 = retina).
    """

    width: int = 1920
    height: int = 1080
    coordinate_system: CoordinateSystem = CoordinateSystem.NORMALIZED
    scale_factor: float = 1.0

    def normalize_coords(self, x: float, y: float) -> tuple[float, float]:
        """Convert pixel coordinates to normalized (0-1)."""
        return (x / self.width, y / self.height)

    def denormalize_coords(self, x: float, y: float) -> tuple[int, int]:
        """Convert normalized coordinates to pixels."""
        return (int(x * self.width), int(y * self.height))


@dataclass
class TrackConfig:
    """Configuration for a specific evaluation track.

    Attributes:
        track_type: The track type (A, B, or C).
        output_format: Expected output format string.
        action_format: Style of action output (JSON, function, pyautogui).
        use_som: Whether to use Set-of-Mark overlay.
        som_config: Configuration for SoM (Track C).
        use_a11y_tree: Whether to include accessibility tree.
        max_a11y_elements: Max elements in a11y tree (truncation).
        include_reasoning: Whether to request reasoning steps.
        react_config: Configuration for ReAct (Track B).
        include_history: Whether to include action history.
        max_history_steps: Max history steps to include.
        screen_config: Screen/coordinate configuration.
        verify_after_action: Request screenshot verification after actions.
    """

    track_type: TrackType
    output_format: str
    action_format: ActionOutputFormat = ActionOutputFormat.JSON
    use_som: bool = False
    som_config: SoMConfig | None = None
    use_a11y_tree: bool = True
    max_a11y_elements: int = 50
    include_reasoning: bool = False
    react_config: ReActConfig | None = None
    include_history: bool = True
    max_history_steps: int = 5
    screen_config: ScreenConfig = field(default_factory=ScreenConfig)
    verify_after_action: bool = False  # Claude computer use best practice

    @classmethod
    def track_a(cls, **kwargs: Any) -> "TrackConfig":
        """Create Track A (Direct Coordinates) config.

        Simplest track: screenshot + goal -> coordinates.
        No reasoning or element IDs.
        """
        return cls(
            track_type=TrackType.TRACK_A,
            output_format='{"action": "CLICK", "x": float, "y": float}',
            action_format=ActionOutputFormat.JSON,
            use_som=False,
            use_a11y_tree=True,
            include_reasoning=False,
            **kwargs,
        )

    @classmethod
    def track_b(cls, **kwargs: Any) -> "TrackConfig":
        """Create Track B (ReAct with Coordinates) config.

        Includes observation->thought->action cycle.
        Based on ReAct, UFO, and Claude thinking patterns.
        """
        react_config = kwargs.pop("react_config", None) or ReActConfig()
        return cls(
            track_type=TrackType.TRACK_B,
            output_format='{"observation": str, "thought": str, "action": "CLICK", "x": float, "y": float}',
            action_format=ActionOutputFormat.JSON,
            use_som=False,
            use_a11y_tree=True,
            include_reasoning=True,
            react_config=react_config,
            **kwargs,
        )

    @classmethod
    def track_c(cls, **kwargs: Any) -> "TrackConfig":
        """Create Track C (Set-of-Mark) config.

        Uses numbered element labels instead of coordinates.
        Based on SoM paper and OMNI-parser patterns.
        """
        som_config = kwargs.pop("som_config", None) or SoMConfig()
        return cls(
            track_type=TrackType.TRACK_C,
            output_format='{"action": "CLICK", "element_id": int}',
            action_format=ActionOutputFormat.JSON,
            use_som=True,
            som_config=som_config,
            use_a11y_tree=True,
            include_reasoning=False,
            **kwargs,
        )

    @classmethod
    def osworld_compatible(cls, **kwargs: Any) -> "TrackConfig":
        """Create OSWorld-compatible config.

        Uses PyAutoGUI-style action format for OSWorld benchmark.
        """
        return cls(
            track_type=TrackType.TRACK_A,
            output_format="pyautogui.click(x, y)",
            action_format=ActionOutputFormat.PYAUTOGUI,
            use_som=False,
            use_a11y_tree=True,
            include_reasoning=False,
            **kwargs,
        )

    @classmethod
    def ufo_compatible(cls, **kwargs: Any) -> "TrackConfig":
        """Create UFO-compatible config.

        Uses UFO's AppAgent output format with observation/thought/plan.
        """
        react_config = kwargs.pop("react_config", None) or ReActConfig(
            require_observation=True,
            require_thought=True,
            require_plan=True,
        )
        return cls(
            track_type=TrackType.TRACK_B,
            output_format='{"Observation": str, "Thought": str, "ControlLabel": int, "Function": str, "Args": list}',
            action_format=ActionOutputFormat.JSON,
            use_som=True,
            som_config=SoMConfig(),
            use_a11y_tree=True,
            include_reasoning=True,
            react_config=react_config,
            **kwargs,
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
