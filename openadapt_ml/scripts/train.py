from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml

from openadapt_ml.datasets.next_action import NextActionDataset, build_next_action_sft_samples
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions
from openadapt_ml.models.qwen_vl import QwenVLAdapter
from openadapt_ml.training.trainer import TrainingConfig, TrainingLogger, train_supervised


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
) -> None:
    cfg = _load_config(config_path)

    model_name = cfg["model"]["name"]
    load_in_4bit = cfg["model"].get("load_in_4bit", False)
    max_pixels = cfg["model"].get("max_pixels")  # For faster training with smaller images
    min_pixels = cfg["model"].get("min_pixels")

    # LoRA config may include an optional weights_path where the trained
    # adapter should be saved. We pass a cleaned config (without
    # weights_path) to the adapter loader.
    raw_lora_cfg = cfg.get("lora")
    lora_weights_path: Optional[str] = None
    lora_cfg: Optional[Dict[str, Any]] = None
    if isinstance(raw_lora_cfg, dict):
        lora_weights_path = raw_lora_cfg.get("weights_path")
        lora_cfg = {k: v for k, v in raw_lora_cfg.items() if k != "weights_path"}
    else:
        lora_cfg = raw_lora_cfg

    # Load data - either from capture or synthetic
    use_som = cfg.get("synthetic_data", {}).get("use_som", False)

    if capture_path:
        # Load from real capture
        print(f"Loading capture from: {capture_path}")
        episodes = _load_capture_episodes(capture_path, goal=goal)
        data_source = f"capture '{Path(capture_path).name}'"
    else:
        # Generate synthetic data
        synth_cfg = cfg.get("synthetic_data", {})
        num_sessions = synth_cfg.get("num_sessions", 10)
        seed = synth_cfg.get("seed")
        default_output_dir = str(Path("synthetic") / "train")
        output_dir = synth_cfg.get("output_dir", default_output_dir)
        use_som = synth_cfg.get("use_som", False)
        scenario = synth_cfg.get("scenario", "login")

        sessions = generate_synthetic_sessions(
            num_sessions=num_sessions,
            seed=seed,
            output_dir=output_dir,
            use_som=use_som,
            scenario=scenario,
        )
        episodes = [ep for sess in sessions for ep in sess.episodes]
        data_source = f"synthetic '{scenario}'"

    samples = build_next_action_sft_samples(episodes, use_som=use_som)
    dataset = NextActionDataset(samples)

    # Adapter + model
    adapter = QwenVLAdapter.from_pretrained(
        model_name=model_name,
        lora_config=lora_cfg,
        load_in_4bit=load_in_4bit,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )

    # Training config
    train_cfg_raw = cfg.get("training", {})
    # Determine output directory
    if output_dir is None:
        output_dir = train_cfg_raw.get("output_dir", "training_output")
    train_cfg = TrainingConfig(
        num_train_epochs=train_cfg_raw.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg_raw.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg_raw.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg_raw.get("learning_rate", 2e-4),
        warmup_ratio=train_cfg_raw.get("warmup_ratio", 0.03),
        weight_decay=train_cfg_raw.get("weight_decay", 0.0),
        max_grad_norm=train_cfg_raw.get("max_grad_norm", 1.0),
        logging_steps=train_cfg_raw.get("logging_steps", 10),
        early_stop_loss=train_cfg_raw.get("early_stop_loss", 1e-4),
        early_stop_patience=train_cfg_raw.get("early_stop_patience", 10),
        output_dir=output_dir,
        # Evaluation settings
        eval_every_epoch=train_cfg_raw.get("eval_every_epoch", True),
        eval_samples=train_cfg_raw.get("eval_samples", 3),
    )

    som_label = " (SoM mode)" if use_som else " (coordinate mode)"
    print(f"Loaded {len(episodes)} episodes and {len(samples)} SFT samples{som_label} from {data_source}.")
    print("Starting training...")

    # Create logger with metadata for dashboard
    logger = TrainingLogger(
        output_dir=train_cfg.output_dir,
        config=train_cfg,
        capture_path=str(capture_path) if capture_path else "",
        config_path=str(config_path),
    )

    # Pass the first episode for periodic evaluation (if available)
    eval_episode = episodes[0] if episodes else None
    training_success = train_supervised(adapter, dataset, train_cfg, logger=logger, episode=eval_episode)

    # Persist the trained adapter if a weights_path was provided and training succeeded.
    if lora_weights_path:
        if training_success:
            save_path = Path(lora_weights_path)
            save_path.mkdir(parents=True, exist_ok=True)
            adapter.model.save_pretrained(save_path)  # type: ignore[arg-type]
            print(f"Saved LoRA adapter to {save_path}")
        else:
            print("Training aborted due to invalid loss. Skipping checkpoint save to avoid corrupted weights.")

    # Open dashboard in browser if requested
    if open_dashboard:
        import webbrowser
        dashboard_path = Path(output_dir) / "dashboard.html"
        if dashboard_path.exists():
            webbrowser.open(f"file://{dashboard_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Qwen-VL adapter on data.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--capture", type=str, help="Path to openadapt-capture recording directory.")
    parser.add_argument("--goal", type=str, help="Task goal/description for the capture.")
    parser.add_argument("--output-dir", type=str, help="Output directory for logs and dashboard.")
    parser.add_argument("--open", action="store_true", help="Open training dashboard in browser.")
    args = parser.parse_args()

    main(args.config, capture_path=args.capture, goal=args.goal, output_dir=args.output_dir, open_dashboard=args.open)
