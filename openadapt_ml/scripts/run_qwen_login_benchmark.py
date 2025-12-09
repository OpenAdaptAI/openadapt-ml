from __future__ import annotations

from pathlib import Path
import argparse

from openadapt_ml.scripts.train import main as train_main
from openadapt_ml.scripts.eval_policy import main as eval_main
from openadapt_ml.evals.plot_eval_metrics import plot_eval_metrics


def run_qwen_login_benchmark(config_path: str, out_dir: str) -> None:
    """Run end-to-end synthetic login benchmark (train → eval base/FT → plot).

    This is a thin orchestrator over existing train/eval/plot utilities. It:
    - trains a LoRA adapter using the given config
    - evaluates the base (no LoRA) and fine-tuned models on fresh synthetic data
    - writes eval JSONs and a comparison plot under the given output directory
    """

    config = Path(config_path)
    out_root = Path(out_dir)

    eval_dir = out_root / "eval"
    plots_dir = out_root / "plots"

    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Train LoRA adapter according to config.
    train_main(str(config))

    # 2) Evaluate base model (ignoring any LoRA config).
    base_json = eval_dir / "eval_base.json"
    eval_main(
        config_path=str(config),
        backend="qwen3",
        output_json=str(base_json),
        ignore_lora=True,
        log_samples=None,
        log_limit=None,
    )

    # 3) Evaluate fine-tuned model (LoRA-enabled).
    ft_json = eval_dir / "eval_ft.json"
    eval_main(
        config_path=str(config),
        backend="qwen3",
        output_json=str(ft_json),
        ignore_lora=False,
        log_samples=None,
        log_limit=None,
    )

    # 4) Plot base vs FT metrics.
    plot_path = plots_dir / "base_vs_ft.png"
    plot_eval_metrics(
        metric_files=[base_json, ft_json],
        labels=["base", "ft"],
        output_path=plot_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic login benchmark end-to-end (train → eval base/FT → plot)."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/qwen3vl_synthetic_dev.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help=(
            "Output directory for eval JSONs and plots "
            "(e.g. experiments/qwen_login/2b_dev)"
        ),
    )
    args = parser.parse_args()

    run_qwen_login_benchmark(config_path=args.config, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
