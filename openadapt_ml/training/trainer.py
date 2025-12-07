from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from openadapt_ml.models.base_adapter import BaseVLMAdapter


@dataclass
class TrainingConfig:
    # Model / LoRA-related fields are handled elsewhere; this covers loop hyperparams.
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10


def _create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    # Use an identity collate_fn so that each batch is a List[Dict], matching
    # the expectations of adapters that operate on SFT-style samples.
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)


def train_supervised(
    adapter: BaseVLMAdapter,
    dataset: Dataset,
    config: TrainingConfig,
    optimizer: Optional[Optimizer] = None,
) -> None:
    """Minimal supervised training loop skeleton.

    This assumes that `adapter.prepare_inputs` and `adapter.compute_loss` are
    implemented. For now, it is primarily a placeholder to validate the
    control flow; it will raise if those methods are not implemented.
    """

    device = adapter.device  # type: ignore[attr-defined]
    dataloader = _create_dataloader(dataset, batch_size=config.per_device_train_batch_size)

    if optimizer is None:
        optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=config.learning_rate)  # type: ignore[arg-type]

    total_steps = 0
    adapter.train()

    for epoch in range(config.num_train_epochs):
        for step, batch in enumerate(dataloader):
            # Batch is a List[Dict[str, Any]] of SFT-style samples; adapter is
            # responsible for converting it into model inputs.
            samples: List[Dict[str, Any]] = batch

            inputs = adapter.prepare_inputs(samples)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            loss = adapter.compute_loss(inputs)
            loss.backward()

            if (total_steps + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), config.max_grad_norm)  # type: ignore[arg-type]
                optimizer.step()
                optimizer.zero_grad()

            total_steps += 1

            if config.logging_steps and total_steps % config.logging_steps == 0:
                print(f"epoch={epoch} step={total_steps} loss={loss.item():.4f}")
