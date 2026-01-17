"""Anthropic (Claude) API provider.

Supports Claude Opus 4.5, Sonnet 4.5, and other Claude models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openadapt_ml.models.providers.base import BaseAPIProvider

if TYPE_CHECKING:
    from PIL import Image


class AnthropicProvider(BaseAPIProvider):
    """Provider for Anthropic's Claude models.

    Supported models:
        - claude-opus-4-5-20251101 (SOTA computer use)
        - claude-sonnet-4-5-20250929 (fast, cheaper)

    Example:
        provider = AnthropicProvider()
        client = provider.create_client(api_key)
        response = provider.send_message(
            client,
            model="claude-opus-4-5-20251101",
            system="You are a GUI agent.",
            content=[
                {"type": "text", "text": "Click the submit button"},
                provider.encode_image(screenshot),
            ],
        )
    """

    @property
    def name(self) -> str:
        return "anthropic"

    def create_client(self, api_key: str) -> Any:
        """Create Anthropic client.

        Args:
            api_key: Anthropic API key.

        Returns:
            Anthropic client instance.

        Raises:
            ImportError: If anthropic package not installed.
        """
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            ) from e

        return Anthropic(api_key=api_key)

    def send_message(
        self,
        client: Any,
        model: str,
        system: str,
        content: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Send message using Anthropic Messages API.

        Args:
            client: Anthropic client.
            model: Model ID (e.g., 'claude-opus-4-5-20251101').
            system: System prompt.
            content: List of content blocks.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.

        Returns:
            Model response text.
        """
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or None,
            messages=[{"role": "user", "content": content}],
        )

        # Extract text from content blocks
        parts = getattr(response, "content", [])
        texts = [
            getattr(p, "text", "")
            for p in parts
            if getattr(p, "type", "") == "text"
        ]
        return "\n".join([t for t in texts if t]).strip()

    def encode_image(self, image: "Image") -> dict[str, Any]:
        """Encode image for Anthropic API.

        Anthropic uses base64-encoded images with explicit source type.

        Args:
            image: PIL Image.

        Returns:
            Image content block for Anthropic API.
        """
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": self.image_to_base64(image, "PNG"),
            },
        }
