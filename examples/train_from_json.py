#!/usr/bin/env python3
"""Example: Train a model from exported JSON data.

This script demonstrates two training approaches for GUI automation:

1. STANDARD SFT (--mode standard):
   Train on (screenshot, task) → action pairs.
   The model learns to predict actions without demonstration context.

2. DEMO-CONDITIONED SFT (--mode demo-conditioned):
   Train on (screenshot, task, retrieved_demo) → action pairs.
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
"""

import argparse
from pathlib import Path

from openadapt_ml.ingest import load_episodes
from openadapt_ml.schemas import validate_episodes, summarize_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Train a model from exported JSON data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  standard          Train on (screenshot, task) → action
                    Baseline approach, no demonstration context.
                    At inference: model predicts action from task alone.

  demo-conditioned  Train on (screenshot, task, demo) → action
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
        help="Fraction of episodes to hold out for retrieval (openadapt mode only)",
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
    summary = summarize_episodes(episodes)
    print("\nData Summary:")
    print(f"  Episodes: {summary['count']}")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Avg steps/episode: {summary['avg_steps_per_episode']:.1f}")
    print(f"  Action types: {summary['action_types']}")

    if args.validate_only:
        print("\nValidation complete. Use --help to see training options.")
        return

    # 3. Prepare training based on mode
    print(f"\nTraining mode: {args.mode.upper()}")
    print(f"Output directory: {args.output}")

    if args.mode == "demo-conditioned":
        _train_demo_conditioned_mode(episodes, args)
    else:
        _train_standard_mode(episodes, args)


def _train_standard_mode(episodes: list, args) -> None:
    """Standard SFT: (screenshot, task) → action."""
    print("\n[STANDARD MODE] Training without demonstration context")
    print("  Input: screenshot + task")
    print("  Output: next action")

    # Import training modules only when needed
    from openadapt_ml.models.qwen_vl import QwenVLAdapter
    from openadapt_ml.training.trainer import (
        TrainingConfig,
        TrainingLogger,
        train_supervised,
    )
    from openadapt_ml.datasets.capture import CaptureDataset

    # Load config
    config = _load_training_config(args.config, args.output)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    adapter = QwenVLAdapter.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda",
    )

    # Create dataset from episodes (standard mode - no demos)
    print("Creating dataset...")
    dataset = CaptureDataset(episodes, adapter.processor)

    # Create logger for dashboard visualization
    logger = TrainingLogger(args.output, config)

    # Train
    print("Starting training...")
    success = train_supervised(
        adapter=adapter,
        dataset=dataset,
        config=config,
        logger=logger,
        episode=episodes[0] if episodes else None,
    )

    _finish_training(success, args.output)


def _train_demo_conditioned_mode(episodes: list, args) -> None:
    """Demo-conditioned SFT: (screenshot, task, demo) → action."""
    print("\n[DEMO-CONDITIONED MODE] Training with demonstration context")
    print("  Input: screenshot + task + retrieved demo")
    print("  Output: next action")
    print("  The model learns to USE demonstrations, compounding with retrieval.")

    from openadapt_ml.retrieval import DemoIndex, DemoRetriever
    from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo

    # Split episodes: some for retrieval library, rest for training
    import random
    random.seed(42)
    shuffled = episodes.copy()
    random.shuffle(shuffled)

    holdout_count = max(1, int(len(shuffled) * args.holdout_ratio))
    library_episodes = shuffled[:holdout_count]
    train_episodes = shuffled[holdout_count:]

    if len(train_episodes) == 0:
        print("ERROR: Not enough episodes for training after holdout. Need at least 2.")
        return

    print(f"\n  Demo library: {len(library_episodes)} episodes")
    print(f"  Training set: {len(train_episodes)} episodes")

    # Build retrieval index from library episodes
    print("\nBuilding retrieval index...")
    index = DemoIndex()
    index.add_many(library_episodes)
    index.build()
    retriever = DemoRetriever(index, domain_bonus=0.2)

    # Create demo-conditioned training samples
    print("Creating demo-conditioned training samples...")
    demo_samples = []
    for episode in train_episodes:
        # For each training episode, retrieve a relevant demo
        demos = retriever.retrieve(episode.goal, top_k=1)
        if demos:
            demo_text = format_episode_as_demo(demos[0], max_steps=5)
        else:
            demo_text = ""  # Fallback if no demo found

        demo_samples.append({
            "episode": episode,
            "demo_text": demo_text,
        })

    print(f"  Created {len(demo_samples)} demo-conditioned samples")

    # Import training modules
    from openadapt_ml.models.qwen_vl import QwenVLAdapter
    from openadapt_ml.training.trainer import (
        TrainingConfig,
        TrainingLogger,
        train_supervised,
    )
    from openadapt_ml.datasets.capture import CaptureDataset

    # Load config
    config = _load_training_config(args.config, args.output)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    adapter = QwenVLAdapter.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda",
    )

    # Create dataset with demo context
    # Note: This passes demo_text that will be prepended to prompts
    print("Creating dataset with demo context...")
    dataset = CaptureDataset(
        [s["episode"] for s in demo_samples],
        adapter.processor,
        demo_texts=[s["demo_text"] for s in demo_samples],
    )

    # Create logger
    logger = TrainingLogger(args.output, config)

    # Train
    print("Starting training...")
    success = train_supervised(
        adapter=adapter,
        dataset=dataset,
        config=config,
        logger=logger,
        episode=train_episodes[0] if train_episodes else None,
    )

    _finish_training(success, args.output)


def _load_training_config(config_path: str, output_dir: str):
    """Load training configuration from YAML or use defaults."""
    from openadapt_ml.training.trainer import TrainingConfig

    config_path = Path(config_path)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict.get("training", {}))
    else:
        # Default config
        config = TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
        )

    config.output_dir = output_dir
    return config


def _finish_training(success: bool, output_dir: str) -> None:
    """Finish training and generate dashboard."""
    if success:
        print(f"\nTraining complete! Results saved to: {output_dir}")
        print(f"  Dashboard: {output_dir}/dashboard.html")
        print(f"  Model: {output_dir}/checkpoints/")
    else:
        print("\nTraining stopped early (loss divergence)")

    # Generate visualization
    print("\nGenerating dashboard...")
    from openadapt_ml.cloud.local import regenerate_viewer

    regenerate_viewer(output_dir)
    print(f"Dashboard ready: {output_dir}/dashboard.html")


if __name__ == "__main__":
    main()
