from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import json

import matplotlib.pyplot as plt


METRIC_KEYS = [
    ("action_type_accuracy", "Action Type Accuracy"),
    ("mean_coord_error", "Mean Coord Error"),
    ("click_hit_rate", "Click Hit Rate"),
    ("episode_success_rate", "Episode Success Rate"),
]


def _load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", payload)


def plot_eval_metrics(
    metric_files: List[Path],
    labels: List[str],
    output_path: Path,
) -> None:
    if len(metric_files) != len(labels):
        raise ValueError("Number of labels must match number of metric files")

    metrics_list = [_load_metrics(p) for p in metric_files]

    num_models = len(metrics_list)
    num_metrics = len(METRIC_KEYS)

    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4))
    fig.suptitle(
        "Qwen-VL base vs fine-tuned (FT = LoRA fine-tuned model)",
        fontsize=10,
    )
    if num_metrics == 1:
        axes = [axes]

    for idx, (key, title) in enumerate(METRIC_KEYS):
        ax = axes[idx]
        values: List[float] = []
        for m in metrics_list:
            v = m.get(key)
            if v is None:
                values.append(0.0)
            else:
                values.append(float(v))

        x = range(num_models)
        ax.bar(x, values, tick_label=labels)
        ax.set_title(title)
        ax.set_ylabel(key)
        ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot evaluation metrics (base vs fine-tuned or cross-model).",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to one or more JSON metric files produced by eval_policy.py.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Labels for each metrics file (e.g. base ft).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG path for the plot.",
    )
    args = parser.parse_args()

    files = [Path(p) for p in args.files]
    labels = list(args.labels)
    output_path = Path(args.output)

    plot_eval_metrics(files, labels, output_path)


if __name__ == "__main__":
    main()
