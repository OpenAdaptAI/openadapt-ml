"""Train a VLM using TRL SFTTrainer + Unsloth.

This script provides the main training entry point for openadapt-ml.
It uses TRL's SFTTrainer with optional Unsloth optimizations for
efficient VLM fine-tuning.

Usage:
    # Train on synthetic data
    python -m openadapt_ml.scripts.train --config configs/qwen3vl_synthetic_som.yaml

    # Train on capture recording
    python -m openadapt_ml.scripts.train --config configs/qwen3vl_capture.yaml \
        --capture /path/to/capture --goal "Task description" --open
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from openadapt_ml.ingest.synthetic import generate_synthetic_episodes
from openadapt_ml.training.trl_trainer import TRLTrainingConfig, train_with_trl


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_capture_episodes(capture_path: str | Path, goal: str | None = None) -> list:
    """Load episodes from an openadapt-capture recording."""
    from openadapt_ml.ingest.capture import capture_to_episode

    capture_path = Path(capture_path)
    episode = capture_to_episode(capture_path, goal=goal)
    return [episode]


def main(
    config_path: str,
    capture_path: str | None = None,
    goal: str | None = None,
    output_dir: str | None = None,
    open_dashboard: bool = False,
    use_unsloth: bool = True,
) -> None:
    """Train a VLM using TRL SFTTrainer.

    Args:
        config_path: Path to YAML config file
        capture_path: Optional path to openadapt-capture recording
        goal: Task goal/description (overrides recording's task description)
        output_dir: Output directory for logs and dashboard
        open_dashboard: Open training dashboard in browser after training
        use_unsloth: Enable Unsloth optimizations (default True)
    """
    cfg = _load_config(config_path)

    model_name = cfg["model"]["name"]
    load_in_4bit = cfg["model"].get("load_in_4bit", False)

    # LoRA config
    raw_lora_cfg = cfg.get("lora")
    lora_cfg: Optional[Dict[str, Any]] = None
    if isinstance(raw_lora_cfg, dict):
        lora_cfg = {k: v for k, v in raw_lora_cfg.items() if k != "weights_path"}
    else:
        lora_cfg = raw_lora_cfg

    # Load data - either from capture or synthetic
    use_som = cfg.get("synthetic_data", {}).get("use_som", False)

    if capture_path:
        # Load from real openadapt-capture recording
        print(f"Loading capture from: {capture_path}")
        episodes = _load_capture_episodes(capture_path, goal=goal)
        data_source = f"capture '{Path(capture_path).name}'"
    else:
        # Generate synthetic data
        synth_cfg = cfg.get("synthetic_data", {})
        num_sessions = synth_cfg.get("num_sessions", 10)
        seed = synth_cfg.get("seed")
        default_output_dir = str(Path("synthetic") / "train")
        synth_output = synth_cfg.get("output_dir", default_output_dir)
        use_som = synth_cfg.get("use_som", False)
        scenario = synth_cfg.get("scenario", "login")

        episodes = generate_synthetic_episodes(
            num_episodes=num_sessions,
            seed=seed,
            output_dir=synth_output,
            use_som=use_som,
            scenario=scenario,
        )
        data_source = f"synthetic '{scenario}'"

    # Determine output directory
    train_cfg_raw = cfg.get("training", {})
    if output_dir is None:
        output_dir = train_cfg_raw.get("output_dir", "training_output")

    print(f"Using TRL trainer (Unsloth: {use_unsloth})")

    # Build TRL config from YAML config
    lora_dict = lora_cfg if isinstance(lora_cfg, dict) else {}
    trl_config = TRLTrainingConfig(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_seq_length=train_cfg_raw.get("max_seq_length", 4096),
        lora_r=lora_dict.get("r", 16),
        lora_alpha=lora_dict.get("lora_alpha", 32),
        lora_dropout=lora_dict.get("lora_dropout", 0.0),
        finetune_vision_layers=lora_dict.get("finetune_vision_layers", False),
        num_epochs=train_cfg_raw.get("num_train_epochs", 3),
        batch_size=train_cfg_raw.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg_raw.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg_raw.get("learning_rate", 2e-4),
        warmup_ratio=train_cfg_raw.get("warmup_ratio", 0.03),
        output_dir=output_dir,
        logging_steps=train_cfg_raw.get("logging_steps", 10),
        save_strategy=train_cfg_raw.get("save_strategy", "epoch"),
    )

    # Disable Unsloth if requested
    if not use_unsloth:
        import os
        os.environ["OPENADAPT_DISABLE_UNSLOTH"] = "1"

    base_path = Path(capture_path).parent if capture_path else None
    print(f"Training on {len(episodes)} episodes from {data_source}")

    checkpoint_path = train_with_trl(
        episodes=episodes,
        config=trl_config,
        use_som=use_som,
        base_path=base_path,
    )
    print(f"Training complete. Checkpoint saved to: {checkpoint_path}")

    # Open dashboard in browser if requested
    if open_dashboard:
        import webbrowser
        dashboard_path = Path(output_dir) / "dashboard.html"
        if dashboard_path.exists():
            webbrowser.open(f"file://{dashboard_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Qwen-VL adapter on synthetic data or openadapt-capture recordings."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--capture", type=str, help="Path to openadapt-capture recording directory.")
    parser.add_argument("--goal", type=str, help="Task goal/description (overrides recording's task description).")
    parser.add_argument("--output-dir", type=str, help="Output directory for logs and dashboard.")
    parser.add_argument("--open", action="store_true", help="Open training dashboard in browser.")

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

    # Determine effective flags
    use_unsloth = args.use_unsloth and not args.no_unsloth

    main(
        args.config,
        capture_path=args.capture,
        goal=args.goal,
        output_dir=args.output_dir,
        open_dashboard=args.open,
        use_unsloth=use_unsloth,
    )
