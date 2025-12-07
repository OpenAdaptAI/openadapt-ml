from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from openadapt_ml.models.base_adapter import BaseVLMAdapter, get_default_device


class DummyAdapter(BaseVLMAdapter):
    """Minimal adapter used to validate the training loop.

    - Ignores images/messages content.
    - Uses a tiny linear model and returns a simple MSE loss.
    - generate() returns a fixed string.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = get_default_device()
        # Tiny dummy model with a few parameters
        model = nn.Linear(4, 1)
        processor: Any = None
        super().__init__(model=model, processor=processor, device=device)

    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:  # type: ignore[override]
        batch_size = len(batch)
        # Create a dummy input tensor; real adapters will encode images + text.
        x = torch.zeros(batch_size, 4, device=self.device)
        # Target is a constant zero tensor; loss will be ||model(x)||^2.
        y = torch.zeros(batch_size, 1, device=self.device)
        return {"inputs": x, "targets": y}

    def compute_loss(self, inputs: Dict[str, Any]) -> torch.Tensor:  # type: ignore[override]
        x = inputs["inputs"]
        y = inputs["targets"]
        preds = self.model(x)
        return torch.mean((preds - y) ** 2)

    def generate(self, sample: Dict[str, Any], max_new_tokens: int = 64) -> str:  # type: ignore[override]
        return "DONE()"
