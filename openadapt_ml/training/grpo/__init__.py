"""GRPO (Group Relative Policy Optimization) training module.

Provides online RL training for GUI agent VLMs using the GRPO algorithm.
Connects to openadapt-evals RLEnvironment for rollout collection and
task evaluation against live Windows Agent Arena VMs.

Supports two training backends (set via GRPOConfig.backend):
    - "standalone" (default): Built-in trainer using HuggingFace + PEFT.
      Good for single-GPU prototyping and debugging. See trainer.py.
    - "verl": Integration with verl-agent/VAGEN for GiGPO and multi-GPU
      distributed training. See verl_backend.py.

Key components:
    - GRPOConfig: Training configuration dataclass (includes backend field)
    - GRPOTrainer: Main training loop (standalone backend)
    - GRPORolloutCollector: Collects rollouts via RLEnvironment
    - reward functions: Binary task success + group-relative advantages
    - CoT warm-up: Chain-of-thought SFT before GRPO
    - verl_backend: verl-agent/VAGEN integration (verl backend)

Example (standalone):
    from openadapt_ml.training.grpo import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        task_ids=["notepad_1", "settings_1"],
        num_training_steps=100,
    )
    trainer = GRPOTrainer(config)
    trainer.train()

Example (verl backend):
    from openadapt_ml.training.grpo import GRPOConfig
    from openadapt_ml.training.grpo.verl_backend import train_with_verl

    config = GRPOConfig(
        backend="verl",
        task_ids=["notepad_1", "settings_1"],
        num_training_steps=100,
    )
    train_with_verl(config)  # Prints instructions; raises NotImplementedError
"""

from __future__ import annotations

# Lightweight imports (no torch required)
from openadapt_ml.training.grpo.config import GRPOConfig
from openadapt_ml.training.grpo.reward import (
    binary_task_success,
    compute_group_advantages,
)
from openadapt_ml.training.grpo.rollout_collector import (
    GRPORolloutCollector,
    Rollout,
)
from openadapt_ml.training.grpo.cot_warmup import (
    build_cot_sft_samples,
    generate_cot_annotations,
)
from openadapt_ml.training.grpo.verl_backend import (
    build_vagen_config,
    train_with_verl,
)

# Lazy imports for torch-dependent modules
_TRAINER_NAMES = {
    "GRPOTrainer",
    "policy_gradient_loss",
    "grpo_loss",
    "parse_vlm_output_to_action",
    "format_action_as_text",
}


def __getattr__(name: str):
    if name in _TRAINER_NAMES:
        from openadapt_ml.training.grpo import trainer as _trainer

        return getattr(_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GRPOConfig",
    "GRPOTrainer",
    "GRPORolloutCollector",
    "Rollout",
    "binary_task_success",
    "compute_group_advantages",
    "policy_gradient_loss",
    "grpo_loss",
    "parse_vlm_output_to_action",
    "format_action_as_text",
    "build_cot_sft_samples",
    "generate_cot_annotations",
    "build_vagen_config",
    "train_with_verl",
]
