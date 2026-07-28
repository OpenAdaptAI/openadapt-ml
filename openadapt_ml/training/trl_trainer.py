"""Simplified training using TRL SFTTrainer + Unsloth.

This module provides a minimal, efficient training path for VLMs:
- Unsloth for 2x speed, 50% less VRAM
- TRL SFTTrainer for production-grade training
- Direct integration with openadapt-ml data format

Usage:
    from openadapt_ml.training.trl_trainer import train_with_trl

    # Train on episodes
    train_with_trl(
        episodes=episodes,
        model_name="unsloth/Qwen2.5-VL-7B-Instruct",
        output_dir="checkpoints/my_model",
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class TRLTrainingConfig:
    """Configuration for TRL-based training."""

    # Model
    model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct"
    load_in_4bit: bool = True
    max_seq_length: int = 4096

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = False  # Set True if grounding needs improvement

    # LoRA target modules
    target_modules: list = (
        None  # None = default ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "checkpoints"
    logging_steps: int = 10
    save_strategy: str = "epoch"

    # Early stopping
    early_stop_loss: float = 0.0  # Stop when loss drops below this (0 = disabled)
    early_stop_patience: int = 5  # Steps below threshold before stopping
    early_stop_min_delta: float = (
        0.0  # Min improvement to count as progress (0 = disabled)
    )
    early_stop_plateau_patience: int = 5  # Steps without improvement before stopping


class _OpenAdaptCallback:
    """HuggingFace TrainerCallback for training_log.json and early stopping.

    Writes step/epoch/loss/lr to training_log.json on each log event.
    Stops training when loss drops below early_stop_loss threshold.
    """

    def __init__(self, config: TRLTrainingConfig):
        import json

        self._json = json
        self._config = config
        self._log_path = Path(config.output_dir) / "training_log.json"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._losses: list = []
        self._below_threshold_count = 0
        self._best_loss: float | None = None
        self._last_loss: float = 0
        self._plateau_count = 0
        self._start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        import time

        self._start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        import time

        if logs is None:
            return

        loss = logs.get("loss")
        lr = logs.get("learning_rate", 0)
        epoch = logs.get("epoch", 0)
        elapsed = time.time() - self._start_time if self._start_time else 0

        if loss is not None:
            self._last_loss = loss
            self._losses.append(
                {
                    "epoch": epoch,
                    "step": state.global_step,
                    "loss": loss,
                    "lr": lr,
                    "time": elapsed,
                }
            )

        # Write training_log.json (use last known loss for non-loss log events)
        log_data = {
            "epoch": int(epoch),
            "step": state.global_step,
            "loss": self._last_loss,
            "learning_rate": lr,
            "total_epochs": args.num_train_epochs,
            "elapsed_time": elapsed,
            "losses": self._losses,
            "model_name": self._config.model_name,
        }
        self._log_path.write_text(self._json.dumps(log_data, indent=2))

        # Early stopping: absolute threshold
        if (
            self._config.early_stop_loss > 0
            and loss is not None
            and loss <= self._config.early_stop_loss
        ):
            self._below_threshold_count += 1
            if self._below_threshold_count >= self._config.early_stop_patience:
                print(
                    f"\nEarly stopping: loss {loss:.4f} <= {self._config.early_stop_loss} "
                    f"for {self._below_threshold_count} steps"
                )
                control.should_training_stop = True
        else:
            self._below_threshold_count = 0

        # Early stopping: plateau detection
        if self._config.early_stop_min_delta > 0 and loss is not None:
            if (
                self._best_loss is None
                or loss < self._best_loss - self._config.early_stop_min_delta
            ):
                self._best_loss = loss
                self._plateau_count = 0
            else:
                self._plateau_count += 1
                if self._plateau_count >= self._config.early_stop_plateau_patience:
                    print(
                        f"\nEarly stopping: loss plateaued at {loss:.4f} "
                        f"(best: {self._best_loss:.4f}, delta: {self._config.early_stop_min_delta}, "
                        f"no improvement for {self._plateau_count} steps)"
                    )
                    control.should_training_stop = True


def _make_callback(config: TRLTrainingConfig):
    """Create a TrainerCallback instance (import deferred to avoid top-level dep)."""
    from transformers import TrainerCallback

    # _OpenAdaptCallback must come BEFORE TrainerCallback so its on_log/on_train_begin
    # override TrainerCallback's no-op implementations (Python MRO).
    class OpenAdaptCallback(_OpenAdaptCallback, TrainerCallback):
        def __init__(self, cfg):
            _OpenAdaptCallback.__init__(self, cfg)

    return OpenAdaptCallback(config)


def _load_unsloth_model(config: TRLTrainingConfig):
    """Load model with Unsloth optimizations.

    Returns:
        tuple: (model, tokenizer, is_unsloth) - is_unsloth indicates if Unsloth was used
    """
    # Check if Unsloth is explicitly disabled via environment variable
    if os.environ.get("OPENADAPT_DISABLE_UNSLOTH", "").lower() in ("1", "true", "yes"):
        print("Unsloth disabled via OPENADAPT_DISABLE_UNSLOTH environment variable")
        return _load_standard_model(config)

    try:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            config.model_name,
            load_in_4bit=config.load_in_4bit,
            use_gradient_checkpointing="unsloth",
            max_seq_length=config.max_seq_length,
        )

        # Apply LoRA
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=config.finetune_vision_layers,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            random_state=42,
        )

        # Enable training mode
        FastVisionModel.for_training(model)

        print(
            f"✓ Loaded {config.model_name} with Unsloth (4-bit: {config.load_in_4bit})"
        )
        return model, tokenizer, True

    except ImportError:
        print("⚠ Unsloth not installed, falling back to standard transformers")
        return _load_standard_model(config)


def _load_standard_model(config: TRLTrainingConfig):
    """Fallback: Load model with standard transformers + peft.

    Automatically detects vision-language models and uses the appropriate
    model class (Qwen2VLForConditionalGeneration for VL models,
    AutoModelForCausalLM for text-only models).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig

    # 4-bit quantization config
    quant_config = None
    if config.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if quant_config:
        load_kwargs["quantization_config"] = quant_config

    # Check if this is a vision-language model
    model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
    is_vl_model = (
        "VL" in config.model_name.upper()
        or "vision" in config.model_name.lower()
        or hasattr(model_config, "vision_config")
    )

    if is_vl_model:
        # Vision-language model - prefer AutoModelForImageTextToText (supports Qwen3-VL)
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                config.model_name,
                **load_kwargs,
            )
            print("  Using AutoModelForImageTextToText for VL model")
        except (ImportError, ValueError, RuntimeError, TypeError):
            from transformers import AutoModelForVision2Seq

            model = AutoModelForVision2Seq.from_pretrained(
                config.model_name,
                **load_kwargs,
            )
            print("  Using AutoModelForVision2Seq for VL model")
    else:
        # Text-only model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs,
        )
        print("  Using AutoModelForCausalLM for text-only model")

    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    # Apply LoRA - CAUSAL_LM for all models (Qwen-VL is decoder-only, not encoder-decoder)
    modules = config.target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    print(f"✓ Loaded {config.model_name} with standard transformers")
    return model, processor, False


def _convert_samples_to_trl_format(
    samples: List[Dict[str, Any]],
    base_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Convert openadapt-ml samples to TRL format.

    The only change is loading image paths as PIL Images.

    Args:
        samples: List of samples from build_next_action_sft_samples()
        base_path: Optional base path to resolve relative image paths

    Returns:
        List of samples with PIL Images instead of paths
    """
    trl_samples = []

    for sample in samples:
        # Load images as PIL
        pil_images = []
        for img_path in sample["images"]:
            path = Path(img_path)
            if base_path and not path.is_absolute():
                path = base_path / path

            if path.exists():
                pil_images.append(Image.open(path).convert("RGB"))
            else:
                print(f"⚠ Image not found: {path}")
                continue

        if not pil_images:
            continue  # Skip samples with missing images

        trl_samples.append(
            {
                "images": pil_images,
                "messages": sample["messages"],
            }
        )

    return trl_samples


def _run_sft_training(
    dataset,
    config: TRLTrainingConfig,
    model,
    tokenizer,
    is_unsloth: bool,
    num_samples: int,
    label: str = "",
) -> str:
    """Shared SFTTrainer setup, training, and checkpoint saving.

    Args:
        dataset: HuggingFace Dataset with images + messages columns.
        config: Training configuration.
        model: Loaded model (Unsloth or standard).
        tokenizer: Tokenizer/processor.
        is_unsloth: Whether model was loaded with Unsloth.
        num_samples: Number of training samples (for logging).
        label: Optional label for log messages (e.g. "JSONL").

    Returns:
        Path to saved checkpoint.
    """
    from trl import SFTConfig, SFTTrainer

    callback = _make_callback(config)

    if is_unsloth:
        from unsloth.trainer import UnslothVisionDataCollator

        training_args = SFTConfig(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_epochs,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            logging_steps=config.logging_steps,
            save_strategy=config.save_strategy,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=dataset,
            args=training_args,
            callbacks=[callback],
        )
    else:
        import torch

        has_cuda = torch.cuda.is_available()
        has_mps = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        training_args = SFTConfig(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_epochs,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            logging_steps=config.logging_steps,
            save_strategy=config.save_strategy,
            max_length=None,  # Critical for VLMs
            assistant_only_loss=False,  # Not supported for VL models yet
            use_cpu=not (has_cuda or has_mps),
            bf16=has_cuda and torch.cuda.is_bf16_supported(),
            fp16=False,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            callbacks=[callback],
        )

    tag = f" ({label})" if label else ""
    print(f"\n{'=' * 50}")
    print(f"Starting training{tag}:")
    print(f"  Model: {config.model_name}")
    print(f"  Samples: {num_samples}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Unsloth: {is_unsloth}")
    print(f"  Output: {config.output_dir}")
    if config.early_stop_loss > 0:
        print(
            f"  Early stop: loss <= {config.early_stop_loss} (patience: {config.early_stop_patience})"
        )
    print(f"{'=' * 50}\n")

    trainer.train()

    checkpoint_path = Path(config.output_dir) / "final"
    trainer.save_model(str(checkpoint_path))
    print(f"\n✓ Saved checkpoint to {checkpoint_path}")

    # Co-locate training log and plots with checkpoint
    import shutil

    log_path = Path(config.output_dir) / "training_log.json"
    if log_path.exists():
        shutil.copy2(log_path, checkpoint_path / "training_log.json")

    try:
        from openadapt_ml.training.plot_training import plot_training_curves

        plots = plot_training_curves(log_path, Path(config.output_dir))
        for plot_path in plots:
            shutil.copy2(plot_path, checkpoint_path / plot_path.name)
        if plots:
            print(f"  Generated {len(plots)} training plots")
    except Exception as e:
        print(f"  Warning: plot generation failed: {e}")

    return str(checkpoint_path)


def train_with_trl(
    episodes: List,
    config: Optional[TRLTrainingConfig] = None,
    use_som: bool = False,
    base_path: Optional[Path] = None,
) -> str:
    """Train a VLM using TRL SFTTrainer + Unsloth.

    This is the simplified training entry point that replaces the legacy
    custom training loop. It:
    1. Converts episodes to TRL format
    2. Loads model with Unsloth (or fallback)
    3. Trains with TRL's SFTTrainer
    4. Saves LoRA adapter

    Args:
        episodes: List of Episode objects from openadapt-ml schema
        config: Training configuration (uses defaults if None)
        use_som: If True, use Set-of-Marks DSL instead of coordinates
        base_path: Base path for resolving relative image paths

    Returns:
        Path to saved checkpoint
    """
    from datasets import Dataset

    from openadapt_ml.datasets.next_action import build_next_action_sft_samples

    config = config or TRLTrainingConfig()

    # Convert episodes to SFT samples
    print(f"Converting {len(episodes)} episodes to training samples...")
    raw_samples = build_next_action_sft_samples(episodes, use_som=use_som)
    print(f"  Generated {len(raw_samples)} training samples")

    # Convert to TRL format (load images as PIL)
    print("Loading images...")
    trl_samples = _convert_samples_to_trl_format(raw_samples, base_path)
    print(f"  Loaded {len(trl_samples)} samples with images")

    if not trl_samples:
        raise ValueError("No valid training samples after loading images")

    dataset = Dataset.from_list(trl_samples)
    model, tokenizer, is_unsloth = _load_unsloth_model(config)

    return _run_sft_training(
        dataset,
        config,
        model,
        tokenizer,
        is_unsloth,
        len(trl_samples),
    )


def train_from_jsonl(
    jsonl_path: str,
    config: Optional[TRLTrainingConfig] = None,
) -> str:
    """Train from a JSONL file in internal SFT format (images + messages).

    Each line must have {"images": ["path/to/img.png"], "messages": [...]}.
    Relative image paths are resolved from the JSONL file's parent directory.

    Args:
        jsonl_path: Path to JSONL training data file.
        config: Training configuration.

    Returns:
        Path to saved checkpoint.
    """
    import json

    from datasets import Dataset

    config = config or TRLTrainingConfig()
    base_path = Path(jsonl_path).parent

    # Load JSONL lines
    raw_samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    print(f"Loaded {len(raw_samples)} samples from {jsonl_path}")

    # Convert to TRL format (load images as PIL)
    trl_samples = _convert_samples_to_trl_format(raw_samples, base_path)
    print(f"  {len(trl_samples)} samples with valid images")

    if not trl_samples:
        raise ValueError("No valid training samples after loading images")

    dataset = Dataset.from_list(trl_samples)
    model, tokenizer, is_unsloth = _load_unsloth_model(config)

    return _run_sft_training(
        dataset,
        config,
        model,
        tokenizer,
        is_unsloth,
        len(trl_samples),
        label="JSONL",
    )


def train_from_parquet(
    parquet_path: str,
    config: Optional[TRLTrainingConfig] = None,
    use_som: bool = False,
) -> str:
    """Train from a parquet file exported by openadapt-ml.

    Args:
        parquet_path: Path to parquet file with episode data
        config: Training configuration
        use_som: Use Set-of-Marks DSL

    Returns:
        Path to saved checkpoint
    """
    from openadapt_ml.export import from_parquet

    print(f"Loading episodes from {parquet_path}...")
    episodes = from_parquet(parquet_path)

    base_path = Path(parquet_path).parent

    return train_with_trl(
        episodes=episodes,
        config=config,
        use_som=use_som,
        base_path=base_path,
    )


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Train VLM with TRL + Unsloth")
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--output", default="checkpoints", help="Output directory")
    parser.add_argument(
        "--model", default="unsloth/Qwen2.5-VL-7B-Instruct", help="Model name"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--use-som", action="store_true", help="Use Set-of-Marks DSL")

    args = parser.parse_args()

    config = TRLTrainingConfig(
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
    )

    checkpoint = train_from_parquet(
        parquet_path=args.parquet,
        config=config,
        use_som=args.use_som,
    )

    print(f"\nTraining complete! Checkpoint: {checkpoint}")
