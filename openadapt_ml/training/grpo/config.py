"""GRPO training configuration.

Follows the same pattern as TRLTrainingConfig in trl_trainer.py, with
additional fields for GRPO-specific hyperparameters and environment setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training.

    Groups model/LoRA defaults with TRLTrainingConfig for consistency.

    Attributes:
        model_name: HuggingFace model identifier.
        load_in_4bit: Whether to use 4-bit quantization.
        max_seq_length: Maximum sequence length for the model.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        num_rollouts_per_step: Group size N for GRPO advantage computation.
        max_steps_per_episode: Maximum actions per rollout episode.
        temperature: Sampling temperature for action generation during rollouts.
        kl_coef: KL divergence penalty coefficient against reference policy.
        server_url: URL of the WAA server for live environment interaction.
        task_ids: List of WAA task IDs to train on.
        learning_rate: Optimizer learning rate for LoRA parameter updates.
        num_training_steps: Total number of GRPO training steps (outer loop).
        save_every_steps: Checkpoint frequency.
        output_dir: Directory for saving checkpoints and logs.
        stuck_window: Number of identical screenshots before early termination.
    """

    # Model (same defaults as TRLTrainingConfig)
    model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct"
    load_in_4bit: bool = True
    max_seq_length: int = 4096

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32

    # GRPO-specific
    num_rollouts_per_step: int = 8  # Group size N
    max_steps_per_episode: int = 15
    temperature: float = 0.7  # Sampling temperature for rollouts
    kl_coef: float = 0.01  # KL divergence penalty

    # Environment
    server_url: str = "http://localhost:5001"
    task_ids: list[str] = field(default_factory=list)
    screen_size: tuple[int, int] = (1920, 1080)  # (width, height)

    # Training
    learning_rate: float = 5e-6
    num_training_steps: int = 1000
    save_every_steps: int = 50
    output_dir: str = "checkpoints/grpo"

    # Stuck detection
    stuck_window: int = 3
