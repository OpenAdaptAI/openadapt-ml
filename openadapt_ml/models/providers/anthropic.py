"""Anthropic (Claude) API provider.

Supports Claude Opus 4.5, Sonnet 4.5, and other Claude models.
Implements the BaseAPIProvider interface for the Anthropic Messages API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from openadapt_ml.models.providers.base import (
    BaseAPIProvider,
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Default models
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

# Supported models with their context windows
SUPPORTED_MODELS = {
    "claude-opus-4-5-20251101": {
        "context": 200_000,
        "description": "SOTA computer use",
    },
    "claude-sonnet-4-5-20250929": {"context": 200_000, "description": "Fast, cheaper"},
    "claude-sonnet-4-20250514": {"context": 200_000, "description": "Previous Sonnet"},
    "claude-haiku-3-5-20241022": {
        "context": 200_000,
        "description": "Fastest, cheapest",
    },
}


class AnthropicProvider(BaseAPIProvider):
    """Provider for Anthropic's Claude models.

    Implements vision support via base64-encoded images in the Messages API format.
    Claude models natively support screenshots and UI analysis for computer use tasks.

    Supported models:
        - claude-opus-4-5-20251101: Most capable, best for complex GUI tasks
        - claude-sonnet-4-5-20250929: Fast and cost-effective
        - claude-haiku-3-5-20241022: Fastest, lowest cost

    Example:
        >>> provider = AnthropicProvider()
        >>> client = provider.create_client(api_key)
        >>> response = provider.send_message(
        ...     client,
        ...     model="claude-opus-4-5-20251101",
        ...     system="You are a GUI agent.",
        ...     content=[
        ...         {"type": "text", "text": "Click the submit button"},
        ...         provider.encode_image(screenshot),
        ...     ],
        ... )

    Attributes:
        name: Returns 'anthropic'.
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    @property
    def env_key_name(self) -> str:
        """Environment variable name for API key."""
        return "ANTHROPIC_API_KEY"

    @property
    def default_model(self) -> str:
        """Default model to use."""
        return DEFAULT_MODEL

    @property
    def supported_models(self) -> dict[str, dict[str, Any]]:
        """Dictionary of supported models and their properties."""
        return SUPPORTED_MODELS

    def create_client(self, api_key: str) -> Any:
        """Create Anthropic client.

        Args:
            api_key: Anthropic API key.

        Returns:
            Anthropic client instance.

        Raises:
            ImportError: If anthropic package not installed.
            AuthenticationError: If API key format is invalid.
        """
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for provider='anthropic'. "
                "Install with: uv add anthropic"
            ) from e

        if not api_key or not api_key.strip():
            raise AuthenticationError(
                "Anthropic API key cannot be empty. "
                "Get a key from https://console.anthropic.com/"
            )

        logger.debug("Creating Anthropic client")
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
            client: Anthropic client from create_client().
            model: Model ID (e.g., 'claude-opus-4-5-20251101').
            system: System prompt.
            content: List of content blocks (text and images).
            max_tokens: Max response tokens.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            Model response text.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded.
            ModelNotFoundError: If model doesn't exist.
            ProviderError: For other API errors.
        """
        logger.debug(f"Sending message to {model} with {len(content)} content blocks")

        try:
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
            result = "\n".join([t for t in texts if t]).strip()

            logger.debug(f"Received response: {len(result)} chars")
            return result

        except Exception as e:
            error_str = str(e).lower()

            # Map common errors to specific exceptions
            if "authentication" in error_str or "api_key" in error_str:
                raise AuthenticationError(
                    f"Anthropic authentication failed: {e}"
                ) from e
            elif "rate_limit" in error_str or "429" in error_str:
                raise RateLimitError(f"Anthropic rate limit exceeded: {e}") from e
            elif "model_not_found" in error_str or "not found" in error_str:
                raise ModelNotFoundError(f"Model '{model}' not found: {e}") from e
            else:
                raise ProviderError(f"Anthropic API error: {e}") from e

    def encode_image(self, image: "Image") -> dict[str, Any]:
        """Encode image for Anthropic API.

        Anthropic uses base64-encoded images with explicit source type.
        PNG format is used for lossless quality.

        Args:
            image: PIL Image.

        Returns:
            Image content block for Anthropic API in format:
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "<base64_string>"
                }
            }
        """
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": self.image_to_base64(image, "PNG"),
            },
        }

    def encode_image_from_bytes(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
    ) -> dict[str, Any]:
        """Encode raw image bytes for Anthropic API.

        Useful when you already have image bytes and don't need PIL.

        Args:
            image_bytes: Raw image bytes.
            media_type: MIME type of the image.

        Returns:
            Image content block for Anthropic API.
        """
        import base64

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            },
        }

    def encode_image_from_url(self, url: str) -> dict[str, Any]:
        """Create image content block from URL.

        Note: Anthropic doesn't support URL-based images directly.
        This method fetches the URL and encodes the image.

        Args:
            url: Image URL to fetch and encode.

        Returns:
            Image content block for Anthropic API.

        Raises:
            ProviderError: If URL fetch fails.
        """
        import urllib.request

        try:
            with urllib.request.urlopen(url) as response:
                image_bytes = response.read()
                content_type = response.headers.get("Content-Type", "image/png")
                return self.encode_image_from_bytes(image_bytes, content_type)
        except Exception as e:
            raise ProviderError(f"Failed to fetch image from URL: {e}") from e
