"""OpenAI (GPT) API provider.

Supports GPT-5.2, GPT-5.1, and other OpenAI models with vision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openadapt_ml.models.providers.base import BaseAPIProvider

if TYPE_CHECKING:
    from PIL import Image


class OpenAIProvider(BaseAPIProvider):
    """Provider for OpenAI's GPT models.

    Supported models:
        - gpt-5.2 (latest)
        - gpt-5.1 (previous)
        - gpt-4o (vision capable)

    Example:
        provider = OpenAIProvider()
        client = provider.create_client(api_key)
        response = provider.send_message(
            client,
            model="gpt-5.2",
            system="You are a GUI agent.",
            content=[
                {"type": "text", "text": "Click the submit button"},
                provider.encode_image(screenshot),
            ],
        )
    """

    @property
    def name(self) -> str:
        return "openai"

    def create_client(self, api_key: str) -> Any:
        """Create OpenAI client.

        Args:
            api_key: OpenAI API key.

        Returns:
            OpenAI client instance.

        Raises:
            ImportError: If openai package not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

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
            client: OpenAI client.
            model: Model ID (e.g., 'gpt-5.2').
            system: System prompt.
            content: List of content blocks.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.

        Returns:
            Model response text.
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content or ""

    def encode_image(self, image: "Image") -> dict[str, Any]:
        """Encode image for OpenAI API.

        OpenAI uses data URLs for images.

        Args:
            image: PIL Image.

        Returns:
            Image content block for OpenAI API.
        """
        base64_data = self.image_to_base64(image, "PNG")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_data}",
            },
        }
