"""Configuration dataclasses for the Representation Shootout experiment.

This module defines all configuration structures using dataclasses with
sensible defaults for the experiment comparing Coordinates vs Marks approaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConditionName(str, Enum):
    """Experimental conditions for representation comparison."""

    RAW_COORDS = "raw_coords"  # Condition A: Raw coordinate regression
    COORDS_CUES = "coords_cues"  # Condition B: Coordinates + visual cues
    MARKS = "marks"  # Condition C: Element ID classification (SoM-style)


class DriftType(str, Enum):
    """Types of distribution drift for evaluation."""

    RESOLUTION = "resolution"  # Scale the UI resolution
    TRANSLATION = "translation"  # Shift the window position
    THEME = "theme"  # Change UI theme (light/dark/high-contrast)
    SCROLL = "scroll"  # Change scroll offset


class MetricName(str, Enum):
    """Metrics computed during evaluation."""

    CLICK_HIT_RATE = "click_hit_rate"  # Clicks within target bbox
    GROUNDING_TOP1_ACCURACY = (
        "grounding_top1_accuracy"  # Correct element ID (marks only)
    )
    EPISODE_SUCCESS_RATE = "episode_success_rate"  # Episodes reaching goal
    COORD_DISTANCE = "coord_distance"  # L2 distance to target (normalized)
    ROBUSTNESS_SCORE = "robustness_score"  # Performance ratio: drift / canonical


class OutputFormat(str, Enum):
    """Output format for model predictions."""

    COORDINATES = "coordinates"  # {"type": "CLICK", "x": float, "y": float}
    ELEMENT_ID = "element_id"  # {"type": "CLICK", "element_id": str}


@dataclass
class VisualCuesConfig:
    """Configuration for visual cues in Condition B.

    Attributes:
        marker_enabled: Whether to draw a marker at click target.
        marker_radius: Radius of the marker circle in pixels.
        marker_color: RGB color tuple for the marker.
        zoom_enabled: Whether to include a zoomed inset patch.
        zoom_factor: Magnification factor for zoom patch.
        zoom_patch_size: Size of the zoom patch in pixels (square).
        zoom_position: Where to place zoom ("auto", "top-left", "top-right", etc.).
    """

    marker_enabled: bool = True
    marker_radius: int = 8
    marker_color: tuple[int, int, int] = (255, 0, 0)  # Red

    zoom_enabled: bool = True
    zoom_factor: float = 2.0
    zoom_patch_size: int = 100
    zoom_position: str = "auto"  # "auto" places opposite to click location


@dataclass
class MarksConfig:
    """Configuration for SoM-style marks in Condition C.

    Attributes:
        overlay_enabled: Whether to draw element ID overlays on screenshot.
        font_size: Font size for element ID labels.
        label_background: Background color for labels (RGBA).
        label_text_color: Text color for labels.
        max_elements: Maximum number of elements to include.
        include_roles: Element roles to include (None = all).
        exclude_roles: Element roles to exclude.
    """

    overlay_enabled: bool = True
    font_size: int = 12
    label_background: tuple[int, int, int, int] = (
        0,
        0,
        255,
        200,
    )  # Blue, semi-transparent
    label_text_color: tuple[int, int, int] = (255, 255, 255)  # White

    max_elements: int = 50
    include_roles: list[str] | None = None  # None = include all
    exclude_roles: list[str] = field(
        default_factory=lambda: ["group", "generic", "static_text"]
    )


@dataclass
class ConditionConfig:
    """Configuration for a single experimental condition.

    Attributes:
        name: Condition identifier (raw_coords, coords_cues, marks).
        output_format: Expected model output format.
        include_history: Whether to include action history in prompt.
        max_history_steps: Maximum number of history steps to include.
        visual_cues: Configuration for visual cues (Condition B only).
        marks: Configuration for element marks (Condition C only).
        loss_type: Loss function type ("mse" for coords, "cross_entropy" for marks).
    """

    name: ConditionName
    output_format: OutputFormat
    include_history: bool = True
    max_history_steps: int = 5

    # Condition-specific configs
    visual_cues: VisualCuesConfig | None = None
    marks: MarksConfig | None = None

    # Training config
    loss_type: str = (
        "mse"  # "mse" for coordinate regression, "cross_entropy" for classification
    )

    @classmethod
    def raw_coords(cls, **kwargs: Any) -> ConditionConfig:
        """Create Condition A (Raw Coordinates) config."""
        return cls(
            name=ConditionName.RAW_COORDS,
            output_format=OutputFormat.COORDINATES,
            loss_type="mse",
            **kwargs,
        )

    @classmethod
    def coords_cues(cls, **kwargs: Any) -> ConditionConfig:
        """Create Condition B (Coordinates + Visual Cues) config."""
        visual_cues = kwargs.pop("visual_cues", None) or VisualCuesConfig()
        return cls(
            name=ConditionName.COORDS_CUES,
            output_format=OutputFormat.COORDINATES,
            visual_cues=visual_cues,
            loss_type="mse",
            **kwargs,
        )

    @classmethod
    def marks(cls, **kwargs: Any) -> ConditionConfig:  # noqa: F811
        """Create Condition C (Marks/Element IDs) config."""
        marks_config = kwargs.pop("marks", None) or MarksConfig()
        return cls(
            name=ConditionName.MARKS,
            output_format=OutputFormat.ELEMENT_ID,
            marks=marks_config,
            loss_type="cross_entropy",
            **kwargs,
        )


@dataclass
class ResolutionDriftParams:
    """Parameters for resolution scaling drift."""

    scale: float  # 0.75, 1.0, 1.25, 1.5


@dataclass
class TranslationDriftParams:
    """Parameters for window translation drift."""

    offset_x: int  # Pixels to shift horizontally
    offset_y: int  # Pixels to shift vertically


@dataclass
class ThemeDriftParams:
    """Parameters for UI theme drift."""

    theme: str  # "light", "dark", "high_contrast"


@dataclass
class ScrollDriftParams:
    """Parameters for scroll offset drift."""

    offset_y: int  # Pixels scrolled down from top


@dataclass
class DriftConfig:
    """Configuration for a drift evaluation test.

    Attributes:
        name: Human-readable name for this drift test.
        drift_type: Type of drift (resolution, translation, theme, scroll).
        params: Drift-specific parameters.
        is_canonical: Whether this is the canonical (no-drift) baseline.
    """

    name: str
    drift_type: DriftType
    params: (
        ResolutionDriftParams
        | TranslationDriftParams
        | ThemeDriftParams
        | ScrollDriftParams
    )
    is_canonical: bool = False

    @classmethod
    def canonical(cls) -> DriftConfig:
        """Create canonical (no-drift) baseline config."""
        return cls(
            name="canonical",
            drift_type=DriftType.RESOLUTION,
            params=ResolutionDriftParams(scale=1.0),
            is_canonical=True,
        )

    @classmethod
    def resolution(cls, scale: float) -> DriftConfig:
        """Create resolution scaling drift config."""
        return cls(
            name=f"resolution_{scale}x",
            drift_type=DriftType.RESOLUTION,
            params=ResolutionDriftParams(scale=scale),
        )

    @classmethod
    def translation(cls, offset_x: int, offset_y: int) -> DriftConfig:
        """Create window translation drift config."""
        return cls(
            name=f"translation_{offset_x}_{offset_y}",
            drift_type=DriftType.TRANSLATION,
            params=TranslationDriftParams(offset_x=offset_x, offset_y=offset_y),
        )

    @classmethod
    def theme(cls, theme_name: str) -> DriftConfig:
        """Create UI theme drift config."""
        return cls(
            name=f"theme_{theme_name}",
            drift_type=DriftType.THEME,
            params=ThemeDriftParams(theme=theme_name),
        )

    @classmethod
    def scroll(cls, offset_y: int) -> DriftConfig:
        """Create scroll offset drift config."""
        return cls(
            name=f"scroll_{offset_y}px",
            drift_type=DriftType.SCROLL,
            params=ScrollDriftParams(offset_y=offset_y),
        )


@dataclass
class DatasetConfig:
    """Configuration for training/evaluation datasets.

    Attributes:
        train_path: Path to training data directory or file.
        eval_path: Path to evaluation data directory or file.
        canonical_resolution: Expected resolution for canonical data.
        min_train_samples: Minimum training samples required.
        min_eval_samples: Minimum evaluation samples per drift condition.
    """

    train_path: str | None = None
    eval_path: str | None = None
    canonical_resolution: tuple[int, int] = (1920, 1080)
    min_train_samples: int = 1000
    min_eval_samples: int = 100


@dataclass
class ExperimentConfig:
    """Top-level configuration for the Representation Shootout experiment.

    Attributes:
        name: Experiment name for logging and results.
        conditions: List of conditions to evaluate (A, B, C).
        drift_tests: List of drift conditions for evaluation.
        metrics: Metrics to compute during evaluation.
        decision_tolerance: Tolerance for decision rule (default 5%).
        dataset: Dataset configuration.
        output_dir: Directory for experiment outputs.
        seed: Random seed for reproducibility.
    """

    name: str = "representation_shootout"
    conditions: list[ConditionConfig] = field(default_factory=list)
    drift_tests: list[DriftConfig] = field(default_factory=list)
    metrics: list[MetricName] = field(default_factory=list)
    decision_tolerance: float = 0.05  # 5% tolerance for decision rule
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output_dir: str = "experiment_results/representation_shootout"
    seed: int = 42

    @classmethod
    def default(cls) -> ExperimentConfig:
        """Create default experiment configuration with all conditions and drifts."""
        return cls(
            name="representation_shootout_default",
            conditions=[
                ConditionConfig.raw_coords(),
                ConditionConfig.coords_cues(),
                ConditionConfig.marks(),
            ],
            drift_tests=[
                # Canonical baseline
                DriftConfig.canonical(),
                # Resolution scaling
                DriftConfig.resolution(0.75),
                DriftConfig.resolution(1.25),
                DriftConfig.resolution(1.5),
                # Window translation
                DriftConfig.translation(200, 0),
                DriftConfig.translation(0, 100),
                DriftConfig.translation(200, 100),
                # Theme changes
                DriftConfig.theme("dark"),
                DriftConfig.theme("high_contrast"),
                # Scroll offset
                DriftConfig.scroll(300),
                DriftConfig.scroll(600),
            ],
            metrics=[
                MetricName.CLICK_HIT_RATE,
                MetricName.GROUNDING_TOP1_ACCURACY,
                MetricName.EPISODE_SUCCESS_RATE,
                MetricName.COORD_DISTANCE,
                MetricName.ROBUSTNESS_SCORE,
            ],
        )

    @classmethod
    def minimal(cls) -> ExperimentConfig:
        """Create minimal configuration for quick testing."""
        return cls(
            name="representation_shootout_minimal",
            conditions=[
                ConditionConfig.raw_coords(),
                ConditionConfig.marks(),
            ],
            drift_tests=[
                DriftConfig.canonical(),
                DriftConfig.resolution(0.75),
                DriftConfig.resolution(1.5),
            ],
            metrics=[
                MetricName.CLICK_HIT_RATE,
                MetricName.COORD_DISTANCE,
            ],
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid).
        """
        issues = []

        if not self.conditions:
            issues.append("At least one condition must be specified")

        if not self.drift_tests:
            issues.append("At least one drift test must be specified")

        # Check for canonical baseline
        has_canonical = any(d.is_canonical for d in self.drift_tests)
        if not has_canonical:
            issues.append("At least one canonical (no-drift) baseline must be included")

        # Check metrics
        if MetricName.GROUNDING_TOP1_ACCURACY in self.metrics:
            has_marks = any(c.name == ConditionName.MARKS for c in self.conditions)
            if not has_marks:
                issues.append("GROUNDING_TOP1_ACCURACY metric requires MARKS condition")

        return issues
