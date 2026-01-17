"""Unified baseline adapters for VLM comparison.

This module provides tools for comparing different VLM providers
(Claude, GPT, Gemini) across multiple evaluation tracks:

- Track A: Direct coordinate prediction
- Track B: ReAct-style reasoning with coordinates
- Track C: Set-of-Mark element selection

Based on SOTA patterns from:
- Claude Computer Use (Anthropic)
- Microsoft UFO/UFO2
- OSWorld benchmark
- Agent-S/Agent-S2 (Simular AI)

Usage:
    from openadapt_ml.baselines import UnifiedBaselineAdapter, BaselineConfig, TrackConfig

    # Quick start with model alias
    adapter = UnifiedBaselineAdapter.from_alias("claude-opus-4.5")
    action = adapter.predict(screenshot, "Click the submit button")

    # With explicit configuration
    config = BaselineConfig(
        provider="anthropic",
        model="claude-opus-4-5-20251101",
        track=TrackConfig.track_c(),
    )
    adapter = UnifiedBaselineAdapter(config)

    # OSWorld-compatible configuration
    config = BaselineConfig(
        provider="openai",
        model="gpt-5.2",
        track=TrackConfig.osworld_compatible(),
    )

    # Parse responses directly
    from openadapt_ml.baselines import UnifiedResponseParser, ElementRegistry

    parser = UnifiedResponseParser()
    action = parser.parse('{"action": "CLICK", "x": 0.5, "y": 0.3}')

    # With element ID to coordinate conversion
    registry = ElementRegistry.from_a11y_tree(tree)
    parser = UnifiedResponseParser(element_registry=registry)
    action = parser.parse_and_resolve('{"action": "CLICK", "element_id": 17}')
"""

from openadapt_ml.baselines.adapter import UnifiedBaselineAdapter
from openadapt_ml.baselines.config import (
    # Enums
    ActionOutputFormat,
    CoordinateSystem,
    TrackType,
    # Config dataclasses
    BaselineConfig,
    ModelSpec,
    ReActConfig,
    ScreenConfig,
    SoMConfig,
    TrackConfig,
    # Registry
    MODELS,
    # Helper functions
    get_default_model,
    get_model_spec,
)
from openadapt_ml.baselines.parser import (
    ElementRegistry,
    ParsedAction,
    UIElement,
    UnifiedResponseParser,
)
from openadapt_ml.baselines.prompts import (
    # System prompts
    FORMAT_PROMPTS,
    SYSTEM_PROMPT_OSWORLD,
    SYSTEM_PROMPT_TRACK_A,
    SYSTEM_PROMPT_TRACK_B,
    SYSTEM_PROMPT_TRACK_C,
    SYSTEM_PROMPT_UFO,
    SYSTEM_PROMPTS,
    # Builder class
    PromptBuilder,
)

__all__ = [
    # Main adapter
    "UnifiedBaselineAdapter",
    # Configuration - Enums
    "ActionOutputFormat",
    "CoordinateSystem",
    "TrackType",
    # Configuration - Dataclasses
    "BaselineConfig",
    "ModelSpec",
    "ReActConfig",
    "ScreenConfig",
    "SoMConfig",
    "TrackConfig",
    # Configuration - Registry
    "MODELS",
    # Configuration - Functions
    "get_default_model",
    "get_model_spec",
    # Parsing
    "ElementRegistry",
    "ParsedAction",
    "UIElement",
    "UnifiedResponseParser",
    # Prompts
    "FORMAT_PROMPTS",
    "PromptBuilder",
    "SYSTEM_PROMPT_OSWORLD",
    "SYSTEM_PROMPT_TRACK_A",
    "SYSTEM_PROMPT_TRACK_B",
    "SYSTEM_PROMPT_TRACK_C",
    "SYSTEM_PROMPT_UFO",
    "SYSTEM_PROMPTS",
]
