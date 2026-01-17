"""Google (Gemini) API provider.

Supports Gemini 3 Pro, Gemini 3 Flash, and other Gemini models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openadapt_ml.models.providers.base import BaseAPIProvider

if TYPE_CHECKING:
    from PIL import Image


class GoogleProvider(BaseAPIProvider):
    """Provider for Google's Gemini models.

    Supported models:
        - gemini-3-pro (most capable)
        - gemini-3-flash (fast inference)
        - gemini-2.5-pro (previous gen)
        - gemini-2.5-flash (fast previous gen)

    Note:
        Gemini supports PIL Images directly without base64 encoding.

    Example:
        provider = GoogleProvider()
        client = provider.create_client(api_key)
        response = provider.send_message(
            client,
            model="gemini-3-pro",
            system="You are a GUI agent.",
            content=[
                {"type": "text", "text": "Click the submit button"},
                provider.encode_image(screenshot),
            ],
        )
    """

    @property
    def name(self) -> str:
        return "google"

    def create_client(self, api_key: str) -> Any:
        """Create Google Generative AI client.

        Unlike Anthropic/OpenAI, Gemini uses a global configure call.
        We return a dict with the API key for later use.

        Args:
            api_key: Google API key.

        Returns:
            Dict containing api_key for model creation.

        Raises:
            ImportError: If google-generativeai package not installed.
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai"
            ) from e

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
            system: System prompt (prepended to content).
            content: List of content blocks.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.

        Returns:
            Model response text.
        """
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

        response = model_instance.generate_content(
            gemini_content,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        return response.text

    def encode_image(self, image: "Image") -> dict[str, Any]:
        """Encode image for Gemini API.

        Gemini accepts PIL Images directly, no base64 needed.

        Args:
            image: PIL Image.

        Returns:
            Image content block containing the PIL Image.
        """
        return {
            "type": "image",
            "image": image,
        }
