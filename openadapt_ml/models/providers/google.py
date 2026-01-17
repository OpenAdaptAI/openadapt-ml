"""Google (Gemini) API provider.

Supports Gemini 3 Pro, Gemini 3 Flash, and other Gemini models.
Implements the BaseAPIProvider interface for the Generative AI API.
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
DEFAULT_MODEL = "gemini-2.5-flash"

# Supported models with their properties
SUPPORTED_MODELS = {
    "gemini-3-pro": {"context": 2_000_000, "description": "Most capable Gemini"},
    "gemini-3-flash": {"context": 1_000_000, "description": "Fast inference"},
    "gemini-2.5-pro": {"context": 2_000_000, "description": "Previous pro"},
    "gemini-2.5-flash": {"context": 1_000_000, "description": "Fast previous gen"},
    "gemini-2.0-flash": {"context": 1_000_000, "description": "Stable flash"},
    "gemini-1.5-pro": {"context": 2_000_000, "description": "Legacy pro"},
    "gemini-1.5-flash": {"context": 1_000_000, "description": "Legacy flash"},
}


class GoogleProvider(BaseAPIProvider):
    """Provider for Google's Gemini models.

    Implements vision support with native PIL Image handling. Unlike Anthropic
    and OpenAI which require base64 encoding, Gemini accepts PIL Images directly.

    Supported models:
        - gemini-3-pro: Most capable, 2M context window
        - gemini-3-flash: Fast inference, 1M context
        - gemini-2.5-pro/flash: Previous generation
        - gemini-2.0-flash: Stable release

    Note:
        Gemini supports PIL Images directly without base64 encoding.
        The encode_image method returns the image wrapped in a dict for
        consistency with other providers.

    Example:
        >>> provider = GoogleProvider()
        >>> client = provider.create_client(api_key)
        >>> response = provider.send_message(
        ...     client,
        ...     model="gemini-3-pro",
        ...     system="You are a GUI agent.",
        ...     content=[
        ...         {"type": "text", "text": "Click the submit button"},
        ...         provider.encode_image(screenshot),
        ...     ],
        ... )

    Attributes:
        name: Returns 'google'.
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return "google"

    @property
    def env_key_name(self) -> str:
        """Environment variable name for API key."""
        return "GOOGLE_API_KEY"

    @property
    def default_model(self) -> str:
        """Default model to use."""
        return DEFAULT_MODEL

    @property
    def supported_models(self) -> dict[str, dict[str, Any]]:
        """Dictionary of supported models and their properties."""
        return SUPPORTED_MODELS

    def create_client(self, api_key: str) -> Any:
        """Create Google Generative AI client.

        Unlike Anthropic/OpenAI, Gemini uses a global configure call.
        We return a dict containing the configured genai module.

        Args:
            api_key: Google API key.

        Returns:
            Dict containing api_key and configured genai module.

        Raises:
            ImportError: If google-generativeai package not installed.
            AuthenticationError: If API key is empty.
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai package is required for provider='google'. "
                "Install with: uv add google-generativeai"
            ) from e

        if not api_key or not api_key.strip():
            raise AuthenticationError(
                "Google API key cannot be empty. "
                "Get a key from https://makersuite.google.com/app/apikey"
            )

        logger.debug("Configuring Google Generative AI")
        genai.configure(api_key=api_key)
        return {"api_key": api_key, "genai": genai}

    def send_message(
        self,
        client: Any,
        model: str,
        system: str,
        content: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Send message using Gemini Generate Content API.

        Args:
            client: Client dict from create_client().
            model: Model ID (e.g., 'gemini-3-pro').
            system: System prompt (prepended to content as text).
            content: List of content blocks.
            max_tokens: Max response tokens.
            temperature: Sampling temperature (0.0-2.0 for Gemini).

        Returns:
            Model response text.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded.
            ModelNotFoundError: If model doesn't exist.
            ProviderError: For other API errors.
        """
        logger.debug(f"Sending message to {model} with {len(content)} content blocks")

        genai = client["genai"]
        model_instance = genai.GenerativeModel(model)

        # Build content list for Gemini
        gemini_content = []

        # Add system prompt as first text if provided
        if system:
            gemini_content.append(f"System: {system}\n\n")

        # Process content items
        for item in content:
            if item.get("type") == "text":
                gemini_content.append(item.get("text", ""))
            elif item.get("type") == "image":
                # Gemini accepts PIL Images directly
                image = item.get("image")
                if image is not None:
                    gemini_content.append(image)

        try:
            response = model_instance.generate_content(
                gemini_content,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            result = response.text
            logger.debug(f"Received response: {len(result)} chars")
            return result

        except Exception as e:
            error_str = str(e).lower()

            # Map common errors to specific exceptions
            if (
                "api_key" in error_str
                or "authentication" in error_str
                or "invalid" in error_str
            ):
                raise AuthenticationError(f"Google authentication failed: {e}") from e
            elif "quota" in error_str or "rate" in error_str or "429" in error_str:
                raise RateLimitError(f"Google rate limit/quota exceeded: {e}") from e
            elif "not found" in error_str or "does not exist" in error_str:
                raise ModelNotFoundError(f"Model '{model}' not found: {e}") from e
            else:
                raise ProviderError(f"Google API error: {e}") from e

    def encode_image(self, image: "Image") -> dict[str, Any]:
        """Encode image for Gemini API.

        Gemini accepts PIL Images directly, no base64 encoding needed.
        We wrap the image in a dict for API consistency.

        Args:
            image: PIL Image.

        Returns:
            Image content block containing the PIL Image:
            {
                "type": "image",
                "image": <PIL.Image>
            }
        """
        return {
            "type": "image",
            "image": image,
        }

    def encode_image_from_bytes(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
    ) -> dict[str, Any]:
        """Encode raw image bytes for Gemini API.

        Converts bytes to PIL Image for Gemini's native format.

        Args:
            image_bytes: Raw image bytes.
            media_type: MIME type (used to verify format).

        Returns:
            Image content block with PIL Image.
        """
        import io

        from PIL import Image as PILImage

        image = PILImage.open(io.BytesIO(image_bytes))
        return self.encode_image(image)

    def encode_image_from_url(self, url: str) -> dict[str, Any]:
        """Create image content block from URL.

        Fetches the image and converts to PIL Image.

        Args:
            url: Image URL to fetch.

        Returns:
            Image content block with PIL Image.

        Raises:
            ProviderError: If URL fetch fails.
        """
        import io
        import urllib.request

        from PIL import Image as PILImage

        try:
            with urllib.request.urlopen(url) as response:
                image_bytes = response.read()
                image = PILImage.open(io.BytesIO(image_bytes))
                return self.encode_image(image)
        except Exception as e:
            raise ProviderError(f"Failed to fetch image from URL: {e}") from e

    def encode_image_as_base64(self, image: "Image") -> dict[str, Any]:
        """Encode image as base64 for Gemini API.

        While Gemini prefers PIL Images, it can also accept base64.
        Use this for cases where you need to serialize the content.

        Args:
            image: PIL Image.

        Returns:
            Image content block with base64 data.
        """
        return {
            "type": "image",
            "inline_data": {
                "mime_type": "image/png",
                "data": self.image_to_base64(image, "PNG"),
            },
        }

    def send_with_grounding(
        self,
        client: Any,
        model: str,
        prompt: str,
        image: "Image",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Send message with grounding/bounding box detection.

        Uses Gemini's native vision capabilities to detect UI elements
        and return bounding boxes. Useful for Set-of-Marks processing.

        Args:
            client: Client dict from create_client().
            model: Model ID.
            prompt: Detection prompt.
            image: Screenshot to analyze.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.

        Returns:
            Dict with response text and any detected bounding boxes.

        Example:
            >>> result = provider.send_with_grounding(
            ...     client,
            ...     model="gemini-2.5-flash",
            ...     prompt="Find the login button",
            ...     image=screenshot,
            ... )
            >>> print(result["boxes"])  # List of bounding boxes
        """
        genai = client["genai"]
        model_instance = genai.GenerativeModel(model)

        grounding_prompt = f"""Analyze this screenshot and {prompt}

Return a JSON object with:
- "elements": array of detected elements with "label", "bbox" [x1,y1,x2,y2], "confidence"
- "description": brief description of what you found

Use pixel coordinates based on image dimensions: {image.width}x{image.height}

Return ONLY valid JSON."""

        try:
            response = model_instance.generate_content(
                [grounding_prompt, image],
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            text = response.text

            # Try to parse JSON response
            import json
            import re

            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return {
                        "text": text,
                        "elements": data.get("elements", []),
                        "description": data.get("description", ""),
                    }
                except json.JSONDecodeError:
                    pass

            return {"text": text, "elements": [], "description": ""}

        except Exception as e:
            raise ProviderError(f"Google grounding error: {e}") from e
