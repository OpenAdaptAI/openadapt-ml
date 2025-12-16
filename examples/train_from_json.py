#!/usr/bin/env python3
"""Example: Train a model from exported JSON data.

This script demonstrates the standard workflow for training a GUI automation
model using openadapt-ml:

1. Load episodes from JSON files (exported by your data pipeline)
2. Validate the data against the schema
3. Train a model with supervised fine-tuning
4. Generate a visualization dashboard

Usage:
    python examples/train_from_json.py --data exported_data/ --output results/

Your JSON data should follow the openadapt-ml Episode schema. See
docs/schema.md for the full specification.
"""

import argparse
from pathlib import Path

from openadapt_ml.ingest import load_episodes
from openadapt_ml.schemas import validate_episodes, summarize_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Train a model from exported JSON data"
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
        "--validate-only",
        action="store_true",
        help="Only validate data, don't train",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Verify image files exist on disk",
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

    # 3. Train the model
    print(f"\nTraining with config: {args.config}")
    print(f"Output directory: {args.output}")

    # Import training modules only when needed
    from openadapt_ml.models.qwen_vl import QwenVLAdapter
    from openadapt_ml.training.trainer import (
        TrainingConfig,
        TrainingLogger,
        train_supervised,
    )
    from openadapt_ml.datasets.capture import CaptureDataset

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict.get("training", {}))
    else:
        # Default config
        config = TrainingConfig(
            output_dir=args.output,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
        )

    config.output_dir = args.output
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    adapter = QwenVLAdapter.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda",
    )

    # Create dataset from episodes
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
        episode=episodes[0] if episodes else None,  # For periodic evaluation
    )

    if success:
        print(f"\nTraining complete! Results saved to: {args.output}")
        print(f"  Dashboard: {args.output}/dashboard.html")
        print(f"  Model: {args.output}/checkpoints/")
    else:
        print("\nTraining stopped early (loss divergence)")

    # 4. Generate visualization
    print("\nGenerating dashboard...")
    from openadapt_ml.cloud.local import regenerate_viewer

    regenerate_viewer(args.output)
    print(f"Dashboard ready: {args.output}/dashboard.html")


if __name__ == "__main__":
    main()
