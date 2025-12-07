from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


def get_default_device() -> torch.device:
    """Select cuda, then mps, then cpu.

    This is used as a fallback when no explicit device is provided.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


class BaseVLMAdapter(ABC):
    """Abstract wrapper around a vision-language model + processor.

    Concrete implementations are responsible for:
    - converting SFT-style samples into model inputs (tokenization, image processing)
    - computing supervised training loss
    - generating assistant text given a single sample at inference time
    """

    def __init__(self, model: torch.nn.Module, processor: Any, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.processor = processor
        self.device = device or get_default_device()
        self.model.to(self.device)

    @abstractmethod
    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert a batch of SFT samples into model inputs.

        The batch is a list of samples of the form produced by
        `build_next_action_sft_samples` (images + messages).
        Implementations should return a dict suitable for passing to the
        underlying HF model, including `labels` for supervised loss.
        """

    @abstractmethod
    def compute_loss(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Run the model forward and return a scalar loss tensor."""

    @abstractmethod
    def generate(self, sample: Dict[str, Any], max_new_tokens: int = 64) -> str:
        """Generate assistant text for a single SFT-style sample."""

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()
