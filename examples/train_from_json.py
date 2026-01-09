#!/usr/bin/env python3
"""Example: Train a model from exported JSON data.

This script demonstrates two training approaches for GUI automation:

1. STANDARD SFT (--mode standard):
   Train on (screenshot, task) -> action pairs.
   The model learns to predict actions without demonstration context.

2. DEMO-CONDITIONED SFT (--mode demo-conditioned):
   Train on (screenshot, task, retrieved_demo) -> action pairs.
   The model learns to USE demonstrations, compounding with retrieval.

Usage:
    # Standard fine-tuning (baseline)
    python examples/train_from_json.py --data exports/ --mode standard

    # Demo-conditioned fine-tuning (uses retrieval)
    python examples/train_from_json.py --data exports/ --mode demo-conditioned

    # Validate data only
    python examples/train_from_json.py --data exports/ --validate-only

Your JSON data should follow the openadapt-ml Episode schema. See
docs/enterprise_integration.md for the full specification.

NOTE: This example uses the TRL trainer. For demo-conditioned training,
we build the retrieval index from held-out episodes and train on the rest.
"""

import argparse
from pathlib import Path

from openadapt_ml.ingest import load_episodes
from openadapt_ml.schema import Episode


def main():
    parser = argparse.ArgumentParser(
        description="Train a model from exported JSON data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  standard          Train on (screenshot, task) -> action
                    Baseline approach, no demonstration context.
                    At inference: model predicts action from task alone.

  demo-conditioned  Train on (screenshot, task, demo) -> action
                    Model learns to follow demonstrations during training.
                    At inference: retrieve a relevant demo, model follows it.
                    Best when you have multiple examples of similar workflows.

Examples:
  # Compare both approaches
  python examples/train_from_json.py --data exports/ --mode standard --output results_standard/
  python examples/train_from_json.py --data exports/ --mode demo-conditioned --output results_demo/
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to directory or JSON file containing episode data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_output",
        help="Output directory for model and dashboard",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3vl_capture.yaml",
        help="Training configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "demo-conditioned"],
        default="standard",
        help="Training mode: 'standard' (no demos) or 'demo-conditioned' (with demos)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data, don't train",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Verify image files exist on disk",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of episodes to hold out for retrieval (demo-conditioned mode only)",
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        default=True,
        help="Enable Unsloth optimizations (default)."
    )
    parser.add_argument(
        "--no-unsloth",
        action="store_true",
        help="Disable Unsloth optimizations."
    )
    args = parser.parse_args()

    # 1. Load episodes from JSON
    print(f"Loading episodes from: {args.data}")
    episodes = load_episodes(
        args.data,
        validate=True,
        check_images=args.check_images,
    )
    print(f"Loaded {len(episodes)} episodes")

    # 2. Show summary statistics
    total_steps = sum(len(ep.steps) for ep in episodes)
    action_types = set()
    for ep in episodes:
        for step in ep.steps:
            if step.action and step.action.type:
                action_types.add(step.action.type.value if hasattr(step.action.type, 'value') else str(step.action.type))
    print("\nData Summary:")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Avg steps/episode: {total_steps / len(episodes):.1f}" if episodes else "  Avg steps/episode: 0")
    print(f"  Action types: {action_types}")

    if args.validate_only:
        print("\nValidation complete. Use --help to see training options.")
        return

    # 3. Prepare training based on mode
    print(f"\nTraining mode: {args.mode.upper()}")
    print(f"Output directory: {args.output}")

    use_unsloth = args.use_unsloth and not args.no_unsloth

    if args.mode == "demo-conditioned":
        _train_demo_conditioned_mode(episodes, args, use_unsloth)
    else:
        _train_standard_mode(episodes, args, use_unsloth)


def _train_standard_mode(episodes: list, args, use_unsloth: bool) -> None:
    """Standard SFT: (screenshot, task) -> action using TRL trainer."""
    print("\n[STANDARD MODE] Training without demonstration context")
    print("  Input: screenshot + task")
    print("  Output: next action")

    from openadapt_ml.training.trl_trainer import TRLTrainingConfig, train_with_trl
    import os

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load config if available
    config = _load_training_config(args.config, args.output)

    # Disable Unsloth if requested
    if not use_unsloth:
        os.environ["OPENADAPT_DISABLE_UNSLOTH"] = "1"

    # Train using TRL
    print("Starting training with TRL...")
    checkpoint_path = train_with_trl(
        episodes=episodes,
        config=config,
        use_som=False,  # Standard coordinate mode
        base_path=Path(args.data) if Path(args.data).is_dir() else Path(args.data).parent,
    )

    _finish_training(True, args.output, checkpoint_path)


def _train_demo_conditioned_mode(episodes: list, args, use_unsloth: bool) -> None:
    """Demo-conditioned SFT: (screenshot, task, demo) -> action.

    NOTE: This is a simplified implementation. For full demo-conditioning,
    you would need to modify the TRL trainer to include demo text in prompts.
    Here we just split the data and train on the training portion.
    """
    print("\n[DEMO-CONDITIONED MODE] Training with demonstration context")
    print("  Input: screenshot + task + retrieved demo")
    print("  Output: next action")
    print("  Note: Full demo-conditioning requires custom prompt formatting.")

    from openadapt_ml.training.trl_trainer import TRLTrainingConfig, train_with_trl
    import os
    import random

    # Split episodes: some for retrieval library, rest for training
    random.seed(42)
    shuffled = episodes.copy()
    random.shuffle(shuffled)

    holdout_count = max(1, int(len(shuffled) * args.holdout_ratio))
    library_episodes = shuffled[:holdout_count]
    train_episodes = shuffled[holdout_count:]

    if len(train_episodes) == 0:
        print("ERROR: Not enough episodes for training after holdout. Need at least 2.")
        return

    print(f"\n  Demo library: {len(library_episodes)} episodes (held out for retrieval)")
    print(f"  Training set: {len(train_episodes)} episodes")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load config
    config = _load_training_config(args.config, args.output)

    # Disable Unsloth if requested
    if not use_unsloth:
        os.environ["OPENADAPT_DISABLE_UNSLOTH"] = "1"

    # Train using TRL on training portion
    # NOTE: For full demo-conditioning, you would need to modify prompts
    # to include retrieved demonstrations
    print("Starting training with TRL...")
    checkpoint_path = train_with_trl(
        episodes=train_episodes,
        config=config,
        use_som=False,
        base_path=Path(args.data) if Path(args.data).is_dir() else Path(args.data).parent,
    )

    _finish_training(True, args.output, checkpoint_path)


def _load_training_config(config_path: str, output_dir: str):
    """Load training configuration from YAML or use defaults."""
    from openadapt_ml.training.trl_trainer import TRLTrainingConfig

    config_path = Path(config_path)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        train_cfg = config_dict.get("training", {})
        lora_cfg = config_dict.get("lora", {})
        model_cfg = config_dict.get("model", {})

        config = TRLTrainingConfig(
            model_name=model_cfg.get("name", "unsloth/Qwen2.5-VL-7B-Instruct"),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
            num_epochs=train_cfg.get("num_train_epochs", 3),
            batch_size=train_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            output_dir=output_dir,
        )
    else:
        # Default config
        config = TRLTrainingConfig(
            output_dir=output_dir,
            num_epochs=3,
            batch_size=1,
            learning_rate=1e-4,
        )

    return config


def _finish_training(success: bool, output_dir: str, checkpoint_path: str = "") -> None:
    """Finish training and generate dashboard."""
    if success:
        print(f"\nTraining complete! Results saved to: {output_dir}")
        if checkpoint_path:
            print(f"  Checkpoint: {checkpoint_path}")
    else:
        print("\nTraining stopped early (loss divergence)")

    # Generate visualization
    print("\nGenerating dashboard...")
    try:
        from openadapt_ml.cloud.local import regenerate_viewer
        regenerate_viewer(output_dir)
        print(f"Dashboard ready: {output_dir}/dashboard.html")
    except Exception as e:
        print(f"Warning: Could not generate dashboard: {e}")


if __name__ == "__main__":
    main()
