"""Unified baseline adapter for comparing VLMs across tracks.

Main entry point for baseline evaluations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from openadapt_ml.baselines.config import BaselineConfig, TrackConfig
from openadapt_ml.baselines.parser import ParsedAction, UnifiedResponseParser
from openadapt_ml.baselines.prompts import PromptBuilder
from openadapt_ml.config import settings
from openadapt_ml.models.providers import get_provider

if TYPE_CHECKING:
    from PIL import Image


class UnifiedBaselineAdapter:
    """Adapter for running baseline evaluations across VLM providers.

    Provides a unified interface for Claude, GPT, and Gemini models
    across multiple evaluation tracks (coordinates, ReAct, SoM).

    Example:
        adapter = UnifiedBaselineAdapter(BaselineConfig.from_alias("claude-opus-4.5"))
        result = adapter.predict(screenshot, "Click the submit button")
        print(result.x, result.y)

        # With Track C (SoM)
        adapter = UnifiedBaselineAdapter(BaselineConfig.from_alias(
            "gemini-3-pro",
            track=TrackConfig.track_c(),
        ))
        result = adapter.predict(screenshot, "Click the login button", a11y_tree=tree)
        print(result.element_id)
    """

    def __init__(self, config: BaselineConfig):
        """Initialize the baseline adapter.

        Args:
            config: Baseline configuration including model, track, etc.
        """
        self.config = config
        self._provider = get_provider(config.provider)
        self._client = None
        self._prompt_builder = PromptBuilder(config.track)
        self._parser = UnifiedResponseParser()

    @property
    def client(self) -> Any:
        """Lazy-load API client."""
        if self._client is None:
            api_key = self._resolve_api_key()
            self._client = self._provider.create_client(api_key)
        return self._client

    def _resolve_api_key(self) -> str:
        """Resolve API key from config, settings, or environment."""
        if self.config.api_key:
            return self.config.api_key

        # Try settings first
        if self.config.provider == "anthropic":
            key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        elif self.config.provider == "openai":
            key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        elif self.config.provider == "google":
            key = settings.google_api_key or os.getenv("GOOGLE_API_KEY")
        else:
            key = None

        if not key:
            raise RuntimeError(
                f"API key for {self.config.provider} not found. "
                "Set in .env file or environment variable."
            )
        return key

    def predict(
        self,
        screenshot: "Image",
        goal: str,
        a11y_tree: str | dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ParsedAction:
        """Predict the next action given current state.

        Args:
            screenshot: Current screenshot as PIL Image.
            goal: Task goal/instruction.
            a11y_tree: Optional accessibility tree (string or dict).
            history: Optional list of previous actions.

        Returns:
            ParsedAction with predicted action.
        """
        # Build system prompt
        system_prompt = self._prompt_builder.get_system_prompt(self.config.demo)

        # Build user content
        content = self._prompt_builder.build_user_content(
            goal=goal,
            screenshot=screenshot,
            a11y_tree=a11y_tree,
            history=history,
            encode_image_fn=self._provider.encode_image,
        )

        # Call API
        response = self._provider.send_message(
            client=self.client,
            model=self.config.model,
            system=system_prompt,
            content=content,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        if self.config.verbose:
            print(f"[{self.config.provider}] Response: {response[:200]}...")

        # Parse response
        action = self._parser.parse(response)

        return action

    def predict_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> list[ParsedAction]:
        """Predict actions for multiple samples.

        Note: Currently runs sequentially. Future optimization could
        use async/parallel calls for providers that support it.

        Args:
            samples: List of dicts with keys: screenshot, goal, a11y_tree, history.

        Returns:
            List of ParsedActions.
        """
        results = []
        for sample in samples:
            action = self.predict(
                screenshot=sample.get("screenshot"),
                goal=sample.get("goal", ""),
                a11y_tree=sample.get("a11y_tree"),
                history=sample.get("history"),
            )
            results.append(action)
        return results

    @classmethod
    def from_alias(
        cls,
        model_alias: str,
        track: TrackConfig | None = None,
        **kwargs: Any,
    ) -> "UnifiedBaselineAdapter":
        """Create adapter from model alias.

        Convenience constructor that resolves model aliases.

        Args:
            model_alias: Model alias (e.g., 'claude-opus-4.5', 'gpt-5.2').
            track: Track config (defaults to Track A).
            **kwargs: Additional config options.

        Returns:
            UnifiedBaselineAdapter instance.
        """
        config = BaselineConfig.from_alias(model_alias, track=track, **kwargs)
        return cls(config)

    def __repr__(self) -> str:
        return (
            f"UnifiedBaselineAdapter("
            f"provider={self.config.provider}, "
            f"model={self.config.model}, "
            f"track={self.config.track.track_type.value})"
        )
