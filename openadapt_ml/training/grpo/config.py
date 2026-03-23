"""GRPO training configuration.

Follows the same pattern as TRLTrainingConfig in trl_trainer.py, with
additional fields for GRPO-specific hyperparameters and environment setup.

Supports two training backends:
    - "standalone" (default): Built-in GRPO trainer using HuggingFace + PEFT.
    - "verl": Integration point for verl-agent/VAGEN, which provides GiGPO
      and multi-GPU support. See verl_backend.py for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training.

    Groups model/LoRA defaults with TRLTrainingConfig for consistency.

    Attributes:
        backend: Training backend to use. "standalone" for the built-in
            HuggingFace + PEFT trainer, or "verl" for verl-agent/VAGEN
            integration (requires separate installation).
        model_name: HuggingFace model identifier.
        load_in_4bit: Whether to use 4-bit quantization.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        num_rollouts_per_step: Group size N for GRPO advantage computation.
        max_steps_per_episode: Maximum actions per rollout episode.
        lora_checkpoint: Path to an existing LoRA adapter to resume from.
            If set, loads the adapter via PeftModel.from_pretrained() instead
            of creating a fresh LoRA. Useful for GRPO on top of an SFT LoRA.
        temperature: Sampling temperature for action generation during rollouts.
        server_url: URL of the WAA server for live environment interaction.
        evaluate_url: URL of the evaluate server. If None, defaults to server_url.
        task_ids: List of WAA task IDs to train on.
        learning_rate: Optimizer learning rate for LoRA parameter updates.
        num_training_steps: Total number of GRPO training steps (outer loop).
        save_every_steps: Checkpoint frequency.
        output_dir: Directory for saving checkpoints and logs.
        stuck_window: Number of identical screenshots before early termination.
    """

    # Backend: "standalone" (built-in HF+PEFT) or "verl" (verl-agent/VAGEN)
    backend: str = "standalone"

    # Model
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_checkpoint: str | None = None  # Path to existing LoRA adapter to resume from

    # GRPO-specific
    num_rollouts_per_step: int = 8  # Group size N
    max_steps_per_episode: int = 15
    temperature: float = 0.7  # Sampling temperature for rollouts

    # Environment
    server_url: str = "http://localhost:5001"
    evaluate_url: str | None = (
        None  # Separate evaluate endpoint; defaults to server_url
    )
    task_ids: list[str] = field(default_factory=list)
    screen_size: tuple[int, int] = (1920, 1080)  # (width, height)

    # Training
    learning_rate: float = 5e-6
    num_training_steps: int = 1000
    save_every_steps: int = 50
    output_dir: str = "checkpoints/grpo"

    # Generation
    max_new_tokens: int = 2048  # Token budget per step. Reasoning models need
    # 1000+ tokens (thought + action). 100 truncates mid-reasoning → unparseable.

    # Task configs
    task_dir: str | None = None  # Directory of TaskConfig YAMLs for milestone rewards

    # Stuck detection
    stuck_window: int = 3
