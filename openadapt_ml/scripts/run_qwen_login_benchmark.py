from __future__ import annotations

from pathlib import Path
import argparse
import os

from openadapt_ml.config import settings
from openadapt_ml.scripts.train import main as train_main
from openadapt_ml.scripts.eval_policy import main as eval_main
from openadapt_ml.evals.plot_eval_metrics import plot_eval_metrics


def _require_env(var_name: str) -> None:
    """Raise a clear error if a required API key is missing."""

    # Check settings first, then fall back to os.getenv
    if var_name == "ANTHROPIC_API_KEY":
        key = settings.anthropic_api_key or os.getenv(var_name)
    elif var_name == "OPENAI_API_KEY":
        key = settings.openai_api_key or os.getenv(var_name)
    else:
        key = os.getenv(var_name)

    if not key:
        raise RuntimeError(
            f"API key {var_name} is required for this benchmark but is not set. "
            "Please set it in .env file, as an environment variable, or configure "
            "it before including the corresponding API backend."
        )


def run_qwen_login_benchmark(
    config_path: str,
    out_dir: str,
    include_claude: bool = False,
    include_openai: bool = False,
    skip_train: bool = False,
) -> None:
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

    # Validate API keys up front if needed.
    if include_claude:
        _require_env("ANTHROPIC_API_KEY")
    if include_openai:
        _require_env("OPENAI_API_KEY")

    # 1) Train LoRA adapter according to config unless explicitly skipped.
    if not skip_train:
        train_main(str(config))

    metric_files = []
    labels = []

    # 2) Evaluate Qwen base model (ignoring any LoRA config).
    base_json = eval_dir / "eval_qwen_base.json"
    eval_main(
        config_path=str(config),
        backend="qwen3",
        output_json=str(base_json),
        ignore_lora=True,
        log_samples=None,
        log_limit=None,
    )
    metric_files.append(base_json)
    labels.append("Qwen3-2B base")

    # 3) Evaluate fine-tuned Qwen model (LoRA-enabled).
    ft_json = eval_dir / "eval_qwen_ft.json"
    eval_main(
        config_path=str(config),
        backend="qwen3",
        output_json=str(ft_json),
        ignore_lora=False,
        log_samples=None,
        log_limit=None,
    )
    metric_files.append(ft_json)
    labels.append("Qwen3-2B FT")

    # 4) Optionally evaluate API backends.
    if include_claude:
        claude_json = eval_dir / "eval_claude.json"
        eval_main(
            config_path=str(config),
            backend="claude",
            output_json=str(claude_json),
            ignore_lora=False,
            log_samples=None,
            log_limit=None,
        )
        metric_files.append(claude_json)
        labels.append("Claude Sonnet 4.5")

    if include_openai:
        gpt_json = eval_dir / "eval_gpt51.json"
        eval_main(
            config_path=str(config),
            backend="openai",
            output_json=str(gpt_json),
            ignore_lora=False,
            log_samples=None,
            log_limit=None,
        )
        metric_files.append(gpt_json)
        labels.append("GPT-5.1")

    # 5) Plot metrics for whichever backends were evaluated.
    if include_claude or include_openai:
        plot_name = "qwen_vs_apis.png"
    else:
        plot_name = "qwen_base_vs_ft.png"

    plot_path = plots_dir / plot_name
    plot_eval_metrics(
        metric_files=metric_files,
        labels=labels,
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
    parser.add_argument(
        "--include-claude",
        action="store_true",
        help="Include Claude Sonnet 4.5 API backend in the evaluation.",
    )
    parser.add_argument(
        "--include-openai",
        action="store_true",
        help="Include GPT-5.1 API backend in the evaluation.",
    )
    parser.add_argument(
        "--include-all-apis",
        action="store_true",
        help="Shorthand to include both Claude and GPT-5.1 backends.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip LoRA training and only run evaluations.",
    )
    args = parser.parse_args()

    include_claude = args.include_claude or args.include_all_apis
    include_openai = args.include_openai or args.include_all_apis

    run_qwen_login_benchmark(
        config_path=args.config,
        out_dir=args.out_dir,
        include_claude=include_claude,
        include_openai=include_openai,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
