"""OpenAI (GPT) API provider.

Supports GPT-5.2, GPT-5.1, GPT-4o, and other OpenAI models with vision.
Implements the BaseAPIProvider interface for the Chat Completions API.
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
DEFAULT_MODEL = "gpt-4o"

# Supported models with their properties
SUPPORTED_MODELS = {
    "gpt-5.2": {"context": 128_000, "description": "Latest GPT model"},
    "gpt-5.1": {"context": 128_000, "description": "Previous GPT-5"},
    "gpt-4o": {"context": 128_000, "description": "Vision-capable, fast"},
    "gpt-4o-mini": {"context": 128_000, "description": "Cheaper, fast"},
    "gpt-4-turbo": {"context": 128_000, "description": "Previous gen turbo"},
}


class OpenAIProvider(BaseAPIProvider):
    """Provider for OpenAI's GPT models.

    Implements vision support via data URL encoded images in the Chat Completions API.
    Supports both standard chat and vision-enabled models.

    Supported models:
        - gpt-5.2: Latest and most capable
        - gpt-5.1: Previous generation GPT-5
        - gpt-4o: Fast, vision-capable
        - gpt-4o-mini: Cost-effective, vision-capable

    Example:
        >>> provider = OpenAIProvider()
        >>> client = provider.create_client(api_key)
        >>> response = provider.send_message(
        ...     client,
        ...     model="gpt-5.2",
        ...     system="You are a GUI agent.",
        ...     content=[
        ...         {"type": "text", "text": "Click the submit button"},
        ...         provider.encode_image(screenshot),
        ...     ],
        ... )

    Note:
        OpenAI uses data URLs for images (data:image/png;base64,...).
        This differs from Anthropic's explicit source object format.

    Attributes:
        name: Returns 'openai'.
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    @property
    def env_key_name(self) -> str:
        """Environment variable name for API key."""
        return "OPENAI_API_KEY"

    @property
    def default_model(self) -> str:
        """Default model to use."""
        return DEFAULT_MODEL

    @property
    def supported_models(self) -> dict[str, dict[str, Any]]:
        """Dictionary of supported models and their properties."""
        return SUPPORTED_MODELS

    def create_client(self, api_key: str) -> Any:
        """Create OpenAI client.

        Args:
            api_key: OpenAI API key.

        Returns:
            OpenAI client instance.

        Raises:
            ImportError: If openai package not installed.
            AuthenticationError: If API key is empty.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for provider='openai'. "
                "Install with: uv add openai"
            ) from e

        if not api_key or not api_key.strip():
            raise AuthenticationError(
                "OpenAI API key cannot be empty. "
                "Get a key from https://platform.openai.com/api-keys"
            )

        logger.debug("Creating OpenAI client")
        return OpenAI(api_key=api_key)

    def send_message(
        self,
        client: Any,
        model: str,
        system: str,
        content: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Send message using OpenAI Chat Completions API.

        Args:
            client: OpenAI client from create_client().
            model: Model ID (e.g., 'gpt-5.2', 'gpt-4o').
            system: System prompt.
            content: List of content blocks (text and images).
            max_tokens: Max response tokens.
            temperature: Sampling temperature (0.0-2.0 for OpenAI).

        Returns:
            Model response text.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded.
            ModelNotFoundError: If model doesn't exist.
            ProviderError: For other API errors.
        """
        logger.debug(f"Sending message to {model} with {len(content)} content blocks")

        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": content})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )

            result = response.choices[0].message.content or ""
            logger.debug(f"Received response: {len(result)} chars")
            return result

        except Exception as e:
            error_str = str(e).lower()

            # Map common errors to specific exceptions
            if (
                "authentication" in error_str
                or "api_key" in error_str
                or "invalid_api_key" in error_str
            ):
                raise AuthenticationError(f"OpenAI authentication failed: {e}") from e
            elif "rate_limit" in error_str or "429" in error_str:
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}") from e
            elif "model_not_found" in error_str or "does not exist" in error_str:
                raise ModelNotFoundError(f"Model '{model}' not found: {e}") from e
            else:
                raise ProviderError(f"OpenAI API error: {e}") from e

    def encode_image(self, image: "Image") -> dict[str, Any]:
        """Encode image for OpenAI API.

        OpenAI uses data URLs for images in the format:
        data:image/<type>;base64,<data>

        Args:
            image: PIL Image.

        Returns:
            Image content block for OpenAI API in format:
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,..."
                }
            }
        """
        base64_data = self.image_to_base64(image, "PNG")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_data}",
            },
        }

    def encode_image_with_detail(
        self,
        image: "Image",
        detail: str = "auto",
    ) -> dict[str, Any]:
        """Encode image with detail level specification.

        OpenAI supports different detail levels for vision processing:
        - "low": Fixed 512x512, 85 tokens, fast
        - "high": Scaled up to 2048x2048, more tokens, detailed
        - "auto": Let the model decide based on image size

        Args:
            image: PIL Image.
            detail: Detail level ("low", "high", "auto").

        Returns:
            Image content block with detail specification.
        """
        base64_data = self.image_to_base64(image, "PNG")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_data}",
                "detail": detail,
            },
        }

    def encode_image_from_url(
        self,
        url: str,
        detail: str = "auto",
    ) -> dict[str, Any]:
        """Create image content block from URL.

        OpenAI natively supports URL-based images, so no fetching needed.

        Args:
            url: Image URL.
            detail: Detail level ("low", "high", "auto").

        Returns:
            Image content block for OpenAI API.
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": url,
                "detail": detail,
            },
        }

    def encode_image_from_bytes(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
    ) -> dict[str, Any]:
        """Encode raw image bytes for OpenAI API.

        Args:
            image_bytes: Raw image bytes.
            media_type: MIME type of the image.

        Returns:
            Image content block for OpenAI API.
        """
        import base64

        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{base64_data}",
            },
        }

    def send_with_tools(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Any:
        """Send message with function calling/tools support.

        OpenAI supports function calling which can be useful for structured
        action extraction in GUI automation.

        Args:
            client: OpenAI client.
            model: Model ID.
            messages: Chat messages.
            tools: Tool definitions.
            tool_choice: Tool choice strategy.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.

        Returns:
            Raw API response (for tool call handling).

        Example:
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "click",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "x": {"type": "number"},
            ...                 "y": {"type": "number"}
            ...             }
            ...         }
            ...     }
            ... }]
            >>> response = provider.send_with_tools(client, model, messages, tools)
        """
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            raise ProviderError(f"OpenAI tools API error: {e}") from e
