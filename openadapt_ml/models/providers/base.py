"""Base provider abstraction for API-backed VLMs.

This module defines the interface that all API providers must implement.
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


class BaseAPIProvider(ABC):
    """Abstract base class for API providers (Anthropic, OpenAI, Google).

    Each provider implements client creation, message sending, and image encoding
    in a provider-specific way.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'anthropic', 'openai', 'google')."""
        ...

    @abstractmethod
    def create_client(self, api_key: str) -> Any:
        """Create and return an API client.

        Args:
            api_key: The API key for authentication.

        Returns:
            Provider-specific client object.
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
