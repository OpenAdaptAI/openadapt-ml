from __future__ import annotations

from typing import Any, Dict, List, Optional

import base64
import os

import torch

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
        device: Optional[torch.device] = None,
        api_key: Optional[str] = None,
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
        """

        self.provider = provider

        if provider == "anthropic":
            try:
                from anthropic import Anthropic  # type: ignore[import]
            except Exception as exc:  # pragma: no cover - import-time failure
                raise RuntimeError(
                    "anthropic package is required for provider='anthropic'. "
                    "Install with `uv sync --extra api`."
                ) from exc

            key = api_key or settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
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

        if device is None:
            device = get_default_device()

        # Store client separately; BaseVLMAdapter expects a model + processor, so
        # we pass a tiny dummy module and the client as the "processor".
        self._client = client
        model = torch.nn.Identity()
        processor: Any = client
        super().__init__(model=model, processor=processor, device=device)

    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:  # type: ignore[override]
        raise NotImplementedError("ApiVLMAdapter does not support training (prepare_inputs)")

    def compute_loss(self, inputs: Dict[str, Any]) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError("ApiVLMAdapter does not support training (compute_loss)")

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
                model="claude-sonnet-4-5-20250929",
                max_tokens=max_new_tokens,
                system=system_text or None,
                messages=[{"role": "user", "content": content}],
            )

            # Anthropic messages API returns a list of content blocks.
            parts = getattr(resp, "content", [])
            texts = [getattr(p, "text", "") for p in parts if getattr(p, "type", "") == "text"]
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
                model="gpt-5.1",
                messages=messages_payload,
                max_completion_tokens=max_new_tokens,
            )
            return resp.choices[0].message.content or ""

        # Should be unreachable because provider is validated in __init__.
        raise ValueError(f"Unsupported provider: {self.provider}")
