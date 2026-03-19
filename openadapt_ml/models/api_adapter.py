from __future__ import annotations

from typing import Any, Dict, List, Optional

import base64
import os

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from openadapt_ml.config import settings
from openadapt_ml.models.base_adapter import BaseVLMAdapter, get_default_device


class ApiVLMAdapter(BaseVLMAdapter):
    """Inference-only adapter for hosted VLM APIs (Anthropic, OpenAI).

    This adapter implements `generate` only; `prepare_inputs` and
    `compute_loss` are not supported and will raise NotImplementedError.
    """

    def __init__(
        self,
        provider: str,
        device: Optional[Any] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize an API-backed adapter.

        Parameters
        ----------
        provider:
            "anthropic" or "openai".
        device:
            Unused for remote APIs but kept for BaseVLMAdapter compatibility.
        api_key:
            Optional API key override. If not provided, keys are loaded from:
            1. Settings (.env file)
            2. Environment variables (ANTHROPIC_API_KEY / OPENAI_API_KEY)
            3. Error if not found
        model_name:
            Override the default model for this provider.
            Defaults to ``claude-sonnet-4-5-20250929`` (Anthropic) or
            ``gpt-4.1`` (OpenAI).
        """

        self.provider = provider
        self._model_name = model_name

        if provider == "anthropic":
            try:
                from anthropic import Anthropic  # type: ignore[import]
            except Exception as exc:  # pragma: no cover - import-time failure
                raise RuntimeError(
                    "anthropic package is required for provider='anthropic'. "
                    "Install with `uv sync --extra api`."
                ) from exc

            key = (
                api_key or settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is required but not found. "
                    "Please set it in .env file, environment variable, or pass api_key parameter."
                )
            client = Anthropic(api_key=key)
        elif provider == "openai":
            try:
                from openai import OpenAI  # type: ignore[import]
            except Exception as exc:  # pragma: no cover - import-time failure
                raise RuntimeError(
                    "openai package is required for provider='openai'. "
                    "Install with `uv sync --extra api`."
                ) from exc

            key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required but not found. "
                    "Please set it in .env file, environment variable, or pass api_key parameter."
                )
            client = OpenAI(api_key=key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Store client separately; BaseVLMAdapter expects a model + processor, so
        # we pass a tiny dummy module and the client as the "processor".
        self._client = client
        if torch is not None:
            if device is None:
                device = get_default_device()
            model = torch.nn.Identity()
            processor: Any = client
            super().__init__(model=model, processor=processor, device=device)
        else:
            # Lightweight mode: skip torch-based init for API-only usage
            self.model = None
            self.processor = client
            self.device = None

    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:  # type: ignore[override]
        raise NotImplementedError(
            "ApiVLMAdapter does not support training (prepare_inputs)"
        )

    def compute_loss(self, inputs: Dict[str, Any]) -> Any:  # type: ignore[override]
        raise NotImplementedError(
            "ApiVLMAdapter does not support training (compute_loss)"
        )

    def generate(self, sample: Dict[str, Any], max_new_tokens: int = 64) -> str:  # type: ignore[override]
        images = sample.get("images", [])
        if not images:
            raise ValueError("Sample is missing image paths")
        image_path = images[0]

        messages = sample.get("messages", [])
        system_text = ""
        user_text = ""
        for m in messages:
            role = m.get("role")
            if role == "system":
                system_text = m.get("content", "")
            elif role == "user":
                user_text = m.get("content", "")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        if self.provider == "anthropic":
            client: Any = self._client
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            content: List[Dict[str, Any]] = []
            if user_text:
                content.append({"type": "text", "text": user_text})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                }
            )

            resp = client.messages.create(
                model=self._model_name or "claude-sonnet-4-5-20250929",
                max_tokens=max_new_tokens,
                system=system_text or None,
                messages=[{"role": "user", "content": content}],
            )

            # Anthropic messages API returns a list of content blocks.
            parts = getattr(resp, "content", [])
            texts = [
                getattr(p, "text", "")
                for p in parts
                if getattr(p, "type", "") == "text"
            ]
            return "\n".join([t for t in texts if t]).strip()

        if self.provider == "openai":
            client: Any = self._client
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            messages_payload: List[Dict[str, Any]] = []
            if system_text:
                messages_payload.append({"role": "system", "content": system_text})

            user_content: List[Dict[str, Any]] = []
            if user_text:
                user_content.append({"type": "text", "text": user_text})
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
            messages_payload.append({"role": "user", "content": user_content})

            resp = client.chat.completions.create(
                model=self._model_name or "gpt-4.1",
                messages=messages_payload,
                max_completion_tokens=max_new_tokens,
            )
            return resp.choices[0].message.content or ""

        # Should be unreachable because provider is validated in __init__.
        raise ValueError(f"Unsupported provider: {self.provider}")


def get_api_adapter(model_name: str, **kwargs: Any) -> ApiVLMAdapter:
    """Create an ApiVLMAdapter from a model name.

    Maps common model name prefixes to providers:
    - ``gpt-*``, ``o1-*``, ``o3-*``, ``o4-*`` → ``"openai"``
    - ``claude-*`` → ``"anthropic"``
    - Unknown prefixes default to ``"openai"``

    Args:
        model_name: Model identifier, e.g. ``"gpt-4o"`` or
            ``"claude-sonnet-4-5-20250929"``.
        **kwargs: Passed through to :class:`ApiVLMAdapter` (e.g. ``api_key``).

    Returns:
        Configured :class:`ApiVLMAdapter` instance.
    """
    name = model_name.lower()

    if name.startswith("claude"):
        provider = "anthropic"
    elif any(name.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
        provider = "openai"
    else:
        provider = "openai"

    return ApiVLMAdapter(provider=provider, model_name=model_name, **kwargs)
