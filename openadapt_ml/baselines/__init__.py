"""Unified baseline adapters for VLM comparison.

This module provides tools for comparing different VLM providers
(Claude, GPT, Gemini) across multiple evaluation tracks:

- Track A: Direct coordinate prediction
- Track B: ReAct-style reasoning with coordinates
- Track C: Set-of-Mark element selection

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
"""

from openadapt_ml.baselines.adapter import UnifiedBaselineAdapter
from openadapt_ml.baselines.config import (
    BaselineConfig,
    ModelSpec,
    TrackConfig,
    TrackType,
    MODELS,
    get_model_spec,
    get_default_model,
)
from openadapt_ml.baselines.parser import ParsedAction, UnifiedResponseParser
from openadapt_ml.baselines.prompts import PromptBuilder

__all__ = [
    # Main adapter
    "UnifiedBaselineAdapter",
    # Configuration
    "BaselineConfig",
    "TrackConfig",
    "TrackType",
    "ModelSpec",
    "MODELS",
    "get_model_spec",
    "get_default_model",
    # Parsing
    "ParsedAction",
    "UnifiedResponseParser",
    # Prompts
    "PromptBuilder",
]
