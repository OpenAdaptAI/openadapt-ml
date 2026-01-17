"""Base provider abstraction for API-backed VLMs.

This module defines the interface that all API providers must implement.
Providers handle client creation, message sending, and image encoding
in a provider-specific way.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class AuthenticationError(ProviderError):
    """Raised when API authentication fails."""

    pass


class RateLimitError(ProviderError):
    """Raised when API rate limit is exceeded."""

    pass


class ModelNotFoundError(ProviderError):
    """Raised when the specified model is not available."""

    pass


class BaseAPIProvider(ABC):
    """Abstract base class for API providers (Anthropic, OpenAI, Google).

    Each provider implements client creation, message sending, and image encoding
    in a provider-specific way.

    Attributes:
        name: Provider identifier ('anthropic', 'openai', 'google').

    Example:
        >>> provider = get_provider("anthropic")
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
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'anthropic', 'openai', 'google')."""
        ...

    @property
    def env_key_name(self) -> str:
        """Environment variable name for API key.

        Returns:
            Environment variable name (e.g., 'ANTHROPIC_API_KEY').
        """
        return f"{self.name.upper()}_API_KEY"

    def get_api_key(self, api_key: str | None = None) -> str:
        """Get API key from parameter, settings, or environment.

        Args:
            api_key: Optional explicit API key.

        Returns:
            API key string.

        Raises:
            AuthenticationError: If no API key is available.
        """
        if api_key:
            return api_key

        # Try settings
        from openadapt_ml.config import settings

        settings_key = getattr(settings, f"{self.name}_api_key", None)
        if settings_key:
            return settings_key

        # Try environment
        env_key = os.getenv(self.env_key_name)
        if env_key:
            return env_key

        raise AuthenticationError(
            f"{self.env_key_name} is required but not found. "
            f"Set it in .env file, environment variable, or pass api_key parameter."
        )

    @abstractmethod
    def create_client(self, api_key: str) -> Any:
        """Create and return an API client.

        Args:
            api_key: The API key for authentication.

        Returns:
            Provider-specific client object.

        Raises:
            ImportError: If required package is not installed.
            AuthenticationError: If API key is invalid.
        """
        ...

    @abstractmethod
    def send_message(
        self,
        client: Any,
        model: str,
        system: str,
        content: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Send a message to the API and return the response text.

        Args:
            client: The API client from create_client().
            model: Model identifier (e.g., 'claude-opus-4-5-20251101').
            system: System prompt.
            content: List of content items (text and images).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            The model's text response.

        Raises:
            RateLimitError: If rate limit is exceeded.
            ModelNotFoundError: If model is not available.
            ProviderError: For other API errors.
        """
        ...

    @abstractmethod
    def encode_image(self, image: "Image") -> dict[str, Any]:
        """Encode a PIL Image for the API.

        Args:
            image: PIL Image to encode.

        Returns:
            Provider-specific image representation for inclusion in content.
        """
        ...

    def image_to_base64(self, image: "Image", format: str = "PNG") -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image to convert.
            format: Image format (PNG, JPEG, etc.).

        Returns:
            Base64-encoded string.
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_media_type(self, format: str = "PNG") -> str:
        """Get MIME type for image format.

        Args:
            format: Image format string.

        Returns:
            MIME type string.
        """
        format_map = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "GIF": "image/gif",
            "WEBP": "image/webp",
        }
        return format_map.get(format.upper(), "image/png")

    def create_text_content(self, text: str) -> dict[str, Any]:
        """Create a text content block.

        Args:
            text: Text content.

        Returns:
            Text content block.
        """
        return {"type": "text", "text": text}

    def build_content(
        self,
        text: str | None = None,
        image: "Image | None" = None,
        additional_content: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a content list from text and/or image.

        Convenience method for building content lists in the correct format.

        Args:
            text: Optional text content.
            image: Optional PIL Image.
            additional_content: Optional additional content blocks.

        Returns:
            List of content blocks.

        Example:
            >>> content = provider.build_content(
            ...     text="Click the button",
            ...     image=screenshot,
            ... )
        """
        content = []

        if text:
            content.append(self.create_text_content(text))

        if image is not None:
            content.append(self.encode_image(image))

        if additional_content:
            content.extend(additional_content)

        return content

    def quick_message(
        self,
        api_key: str,
        model: str,
        prompt: str,
        image: "Image | None" = None,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Send a quick message without managing client lifecycle.

        Convenience method that creates a client, sends a message, and returns
        the response in one call. Useful for one-off requests.

        Args:
            api_key: API key for authentication.
            model: Model identifier.
            prompt: User prompt text.
            image: Optional image to include.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Model response text.

        Example:
            >>> response = provider.quick_message(
            ...     api_key=key,
            ...     model="claude-opus-4-5-20251101",
            ...     prompt="What's in this image?",
            ...     image=screenshot,
            ... )
        """
        client = self.create_client(api_key)
        content = self.build_content(text=prompt, image=image)
        return self.send_message(
            client=client,
            model=model,
            system=system,
            content=content,
            max_tokens=max_tokens,
            temperature=temperature,
        )
