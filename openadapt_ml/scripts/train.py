from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml

from openadapt_ml.datasets.next_action import NextActionDataset, build_next_action_sft_samples
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions
from openadapt_ml.models.qwen_vl import QwenVLAdapter
from openadapt_ml.training.trainer import TrainingConfig, train_supervised


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    cfg = _load_config(config_path)

    model_name = cfg["model"]["name"]
    load_in_4bit = cfg["model"].get("load_in_4bit", False)

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

    # Data generation
    synth_cfg = cfg.get("synthetic_data", {})
    num_sessions = synth_cfg.get("num_sessions", 10)
    seed = synth_cfg.get("seed")
    output_dir = synth_cfg.get("output_dir", "synthetic_train")

    sessions = generate_synthetic_sessions(
        num_sessions=num_sessions,
        seed=seed,
        output_dir=output_dir,
    )
    episodes = [ep for sess in sessions for ep in sess.episodes]

    samples = build_next_action_sft_samples(episodes)
    dataset = NextActionDataset(samples)

    # Adapter + model
    adapter = QwenVLAdapter.from_pretrained(
        model_name=model_name,
        lora_config=lora_cfg,
        load_in_4bit=load_in_4bit,
    )

    # Training config
    train_cfg_raw = cfg.get("training", {})
    train_cfg = TrainingConfig(
        num_train_epochs=train_cfg_raw.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg_raw.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg_raw.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg_raw.get("learning_rate", 2e-4),
        warmup_ratio=train_cfg_raw.get("warmup_ratio", 0.03),
        weight_decay=train_cfg_raw.get("weight_decay", 0.0),
        max_grad_norm=train_cfg_raw.get("max_grad_norm", 1.0),
        logging_steps=train_cfg_raw.get("logging_steps", 10),
    )

    print(f"Loaded {len(episodes)} episodes and {len(samples)} SFT samples.")
    print("Starting training (adapter.prepare_inputs/compute_loss must be implemented)...")

    train_supervised(adapter, dataset, train_cfg)

    # Persist the trained adapter if a weights_path was provided.
    if lora_weights_path:
        save_path = Path(lora_weights_path)
        save_path.mkdir(parents=True, exist_ok=True)
        adapter.model.save_pretrained(save_path)  # type: ignore[arg-type]
        print(f"Saved LoRA adapter to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Qwen-VL adapter on synthetic data.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    main(args.config)
