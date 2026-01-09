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

    # Training
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03

    # Output
    output_dir: str = "checkpoints"
    logging_steps: int = 10
    save_strategy: str = "epoch"


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

        print(f"✓ Loaded {config.model_name} with Unsloth (4-bit: {config.load_in_4bit})")
        return model, tokenizer, True

    except ImportError:
        print("⚠ Unsloth not installed, falling back to standard transformers")
        return _load_standard_model(config)


def _load_standard_model(config: TRLTrainingConfig):
    """Fallback: Load model with standard transformers + peft."""
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import LoraConfig, get_peft_model
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    # Apply LoRA
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
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

        trl_samples.append({
            "images": pil_images,
            "messages": sample["messages"],
        })

    return trl_samples


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

    # Step 1: Convert episodes to SFT samples
    print(f"Converting {len(episodes)} episodes to training samples...")
    raw_samples = build_next_action_sft_samples(episodes, use_som=use_som)
    print(f"  Generated {len(raw_samples)} training samples")

    # Step 2: Convert to TRL format (load images as PIL)
    print("Loading images...")
    trl_samples = _convert_samples_to_trl_format(raw_samples, base_path)
    print(f"  Loaded {len(trl_samples)} samples with images")

    if not trl_samples:
        raise ValueError("No valid training samples after loading images")

    # Step 3: Create HuggingFace Dataset
    dataset = Dataset.from_list(trl_samples)

    # Step 4: Load model with Unsloth (or fallback)
    model, tokenizer, is_unsloth = _load_unsloth_model(config)

    # Step 5: Configure and run training
    try:
        from trl import SFTTrainer, SFTConfig

        if is_unsloth:
            # Unsloth-specific configuration
            from unsloth.trainer import UnslothVisionDataCollator

            training_args = SFTConfig(
                output_dir=config.output_dir,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                warmup_ratio=config.warmup_ratio,
                lr_scheduler_type="cosine",
                logging_steps=config.logging_steps,
                save_strategy=config.save_strategy,
                # Unsloth-specific settings
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
            )
        else:
            # Standard TRL configuration
            training_args = SFTConfig(
                output_dir=config.output_dir,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                warmup_ratio=config.warmup_ratio,
                lr_scheduler_type="cosine",
                logging_steps=config.logging_steps,
                save_strategy=config.save_strategy,
                max_length=None,  # Critical for VLMs
                assistant_only_loss=True,
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                args=training_args,
            )

        print(f"\n{'='*50}")
        print(f"Starting training:")
        print(f"  Model: {config.model_name}")
        print(f"  Samples: {len(trl_samples)}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Unsloth: {is_unsloth}")
        print(f"  Output: {config.output_dir}")
        print(f"{'='*50}\n")

        trainer.train()

        # Save the LoRA adapter
        checkpoint_path = Path(config.output_dir) / "final"
        trainer.save_model(str(checkpoint_path))
        print(f"\n✓ Saved checkpoint to {checkpoint_path}")

        return str(checkpoint_path)

    except ImportError as e:
        raise ImportError(
            f"TRL not installed. Install with: pip install trl\n"
            f"Original error: {e}"
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
    parser.add_argument("--model", default="unsloth/Qwen2.5-VL-7B-Instruct", help="Model name")
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
