from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from openadapt_ml.datasets.next_action import build_next_action_sft_samples, parse_action_som
from openadapt_ml.evals.trajectory_matching import evaluate_policy_on_episodes
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions
from openadapt_ml.models.dummy_adapter import DummyAdapter
from openadapt_ml.models.qwen_vl import QwenVLAdapter
from openadapt_ml.models.api_adapter import ApiVLMAdapter
from openadapt_ml.runtime.policy import AgentPolicy


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
    config_path: str,
    backend: str,
    output_json: str | None,
    ignore_lora: bool = False,
    log_samples: Optional[str] = None,
    log_limit: Optional[int] = None,
    dsl_mode: str = "coord",
    eval_on_training_data: bool = False,
    no_jitter: bool = False,
) -> None:
    cfg = _load_config(config_path)

    # Determine if using Set-of-Marks (SoM) mode
    use_som = dsl_mode == "som"

    # Synthetic data config
    synth_cfg: Dict[str, Any] = cfg.get("synthetic_data", {})
    num_sessions = synth_cfg.get("num_sessions", 4)
    seed = synth_cfg.get("seed", 999)

    # Determine output directory and jitter setting
    if eval_on_training_data:
        # Use the SAME data directory as training to test memorization
        output_dir = synth_cfg.get("output_dir", "synthetic_train")
        # When evaluating on training data, use same jitter setting as training
        # (default True unless explicitly set)
        jitter = synth_cfg.get("jitter", True) and not no_jitter
        print(f"[INFO] Evaluating on TRAINING data from: {output_dir}")
    else:
        # Generate fresh data for generalization testing
        output_dir = synth_cfg.get("output_dir", "synthetic_eval") + "_eval"
        jitter = not no_jitter
        print(f"[INFO] Evaluating on FRESH data in: {output_dir}")

    if no_jitter:
        print("[INFO] Jitter disabled - using deterministic layouts")

    # Generate sessions with SoM if requested
    sessions = generate_synthetic_sessions(
        num_sessions=num_sessions,
        seed=seed,
        output_dir=output_dir,
        use_som=use_som,
        jitter=jitter,
    )
    episodes = [ep for sess in sessions for ep in sess.episodes]

    # Build samples with appropriate DSL mode
    samples = build_next_action_sft_samples(episodes, use_som=use_som)

    # Backend / adapter selection
    if backend == "dummy":
        adapter = DummyAdapter()
    elif backend == "qwen3":
        model_cfg = cfg.get("model", {})
        model_name = model_cfg.get("name", "Qwen/Qwen3-VL-8B-Instruct")
        load_in_4bit = model_cfg.get("load_in_4bit", False)

        # Optionally ignore LoRA to evaluate the base model only.
        if ignore_lora:
            lora_cfg = None
        else:
            lora_cfg = cfg.get("lora")

        adapter = QwenVLAdapter.from_pretrained(
            model_name,
            lora_config=lora_cfg,
            load_in_4bit=load_in_4bit,
        )
    elif backend == "qwen2_5":
        adapter = QwenVLAdapter.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            lora_config=None,
            load_in_4bit=False,
        )
    elif backend == "claude":
        adapter = ApiVLMAdapter(provider="anthropic")
    elif backend == "openai":
        adapter = ApiVLMAdapter(provider="openai")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    policy = AgentPolicy(adapter)

    log_fn: Optional[callable] = None
    log_file_handle = None
    if log_samples is not None:
        log_path = Path(log_samples)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = open(log_path, "w", encoding="utf-8")

        def _log(record: Dict[str, Any]) -> None:
            assert log_file_handle is not None
            log_file_handle.write(json.dumps(record) + "\n")

        log_fn = _log

    try:
        metrics = evaluate_policy_on_episodes(
            policy,
            episodes,
            samples,
            log_fn=log_fn,
            log_limit=log_limit,
            use_som=use_som,
        )
    finally:
        if log_file_handle is not None:
            log_file_handle.close()

    print(f"Evaluation results (DSL mode: {dsl_mode}):")
    print(f"  num_episodes: {metrics.num_episodes}")
    print(f"  num_steps: {metrics.num_steps}")
    print(f"  action_type_accuracy: {metrics.action_type_accuracy:.4f}")
    if metrics.mean_coord_error is not None:
        print(
            "  mean_coord_error (normalized): "
            f"{metrics.mean_coord_error:.4f} (n={metrics.coord_error_count})"
        )
    else:
        print("  mean_coord_error (normalized): N/A")
    if metrics.episode_success_rate is not None:
        print(f"  episode_success_rate: {metrics.episode_success_rate:.4f}")
    else:
        print("  episode_success_rate: N/A")
    if metrics.click_hit_rate is not None:
        print(f"  click_hit_rate: {metrics.click_hit_rate:.4f}")
    else:
        print("  click_hit_rate: N/A")
    if metrics.mean_episode_progress is not None:
        print(f"  mean_episode_progress: {metrics.mean_episode_progress:.4f}")
    else:
        print("  mean_episode_progress: N/A")
    if metrics.mean_episode_step_score is not None:
        print(f"  mean_episode_step_score: {metrics.mean_episode_step_score:.4f}")
    else:
        print("  mean_episode_step_score: N/A")
    if metrics.weak_episode_success_rate is not None:
        print(f"  weak_episode_success_rate: {metrics.weak_episode_success_rate:.4f}")
    else:
        print("  weak_episode_success_rate: N/A")
    if metrics.state_success_rate is not None:
        print(f"  state_success_rate: {metrics.state_success_rate:.4f}")
    else:
        print("  state_success_rate: N/A")
    if metrics.bbox_hit_rate is not None:
        print(f"  bbox_hit_rate: {metrics.bbox_hit_rate:.4f}")
    else:
        print("  bbox_hit_rate: N/A")
    if metrics.element_accuracy is not None:
        print(f"  element_accuracy: {metrics.element_accuracy:.4f}")
    else:
        print("  element_accuracy: N/A")

    if output_json is not None:
        payload = {
            "config_path": str(config_path),
            "backend": backend,
            "dsl_mode": dsl_mode,
            "metrics": {
                "num_episodes": metrics.num_episodes,
                "num_steps": metrics.num_steps,
                "action_type_accuracy": metrics.action_type_accuracy,
                "mean_coord_error": metrics.mean_coord_error,
                "coord_error_count": metrics.coord_error_count,
                "episode_success_rate": metrics.episode_success_rate,
                "click_hit_rate": metrics.click_hit_rate,
                "bbox_hit_rate": metrics.bbox_hit_rate,
                "mean_episode_progress": metrics.mean_episode_progress,
                "mean_episode_step_score": metrics.mean_episode_step_score,
                "weak_episode_success_rate": metrics.weak_episode_success_rate,
                "state_success_rate": metrics.state_success_rate,
                "element_accuracy": metrics.element_accuracy if hasattr(metrics, 'element_accuracy') else None,
            },
        }
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Metrics written to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a policy on synthetic episodes.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["dummy", "qwen3", "qwen2_5", "claude", "openai"],
        default="qwen2_5",
        help="Backend adapter to use for evaluation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write metrics as JSON.",
    )
    parser.add_argument(
        "--ignore-lora",
        action="store_true",
        help="Ignore any LoRA config in the YAML and evaluate the base model only.",
    )
    parser.add_argument(
        "--log-samples",
        type=str,
        default=None,
        help="Optional path to write per-step eval logs as JSONL.",
    )
    parser.add_argument(
        "--log-limit",
        type=int,
        default=None,
        help="Maximum number of steps to log (default: no limit).",
    )
    parser.add_argument(
        "--dsl-mode",
        type=str,
        choices=["coord", "som"],
        default="coord",
        help="DSL mode: 'coord' for coordinate-based (CLICK(x=..., y=...)), "
             "'som' for Set-of-Marks index-based (CLICK([1])). Default: coord.",
    )
    parser.add_argument(
        "--eval-on-training-data",
        action="store_true",
        help="Evaluate on the exact training data (tests memorization). "
             "If not set, generates fresh data (tests generalization).",
    )
    parser.add_argument(
        "--no-jitter",
        action="store_true",
        help="Disable jitter for deterministic UI layouts. "
             "Useful for testing memorization of fixed layouts.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        backend=args.backend,
        output_json=args.output_json,
        ignore_lora=args.ignore_lora,
        log_samples=args.log_samples,
        log_limit=args.log_limit,
        dsl_mode=args.dsl_mode,
        eval_on_training_data=args.eval_on_training_data,
        no_jitter=args.no_jitter,
    )
