from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METRIC_KEYS = [
    ("action_type_accuracy", "Action Type Accuracy"),
    ("mean_coord_error", "Mean Coord Error"),
    ("click_hit_rate", "Click Hit Rate"),
    ("episode_success_rate", "Strict Episode Success"),
    ("mean_episode_progress", "Episode Progress"),
    ("mean_episode_step_score", "Step Score (Type+Click)"),
    ("weak_episode_success_rate", "Weak Episode Success"),
]


def _load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", payload)


def _get_bar_style(label: str) -> tuple[str, str]:
    """Determine bar color and hatch pattern based on model label.

    Returns:
        (color, hatch): color string and hatch pattern
    """
    label_lower = label.lower()

    # Determine color based on model type
    if "claude" in label_lower:
        color = "#FF6B35"  # Orange for Claude
    elif "gpt" in label_lower or "openai" in label_lower:
        color = "#C1121F"  # Red for GPT
    elif "2b" in label_lower:
        color = "#4A90E2"  # Light blue for 2B
    elif "8b" in label_lower:
        color = "#2E5C8A"  # Dark blue for 8B
    else:
        color = "#6C757D"  # Gray for unknown

    # Determine hatch pattern for fine-tuned models
    if "ft" in label_lower or "fine" in label_lower or "finetuned" in label_lower:
        hatch = "///"  # Diagonal lines for fine-tuned
    else:
        hatch = ""  # Solid for base/API models

    return color, hatch


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

    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 5))
    fig.suptitle(
        "VLM Model Comparison (Offline fine-tuned vs API models)",
        fontsize=12,
        fontweight='bold',
    )
    if num_metrics == 1:
        axes = [axes]

    for idx, (key, title) in enumerate(METRIC_KEYS):
        ax = axes[idx]
        values: List[float] = []
        colors: List[str] = []
        hatches: List[str] = []

        for m, label in zip(metrics_list, labels):
            v = m.get(key)
            if v is None:
                values.append(0.0)
            else:
                values.append(float(v))

            color, hatch = _get_bar_style(label)
            colors.append(color)
            hatches.append(hatch)

        x = range(num_models)
        bars = ax.bar(x, values, tick_label=labels, color=colors, edgecolor='black', linewidth=1.2)

        # Apply hatch patterns
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(key, fontsize=9)
        ax.set_ylim(bottom=0.0)
        # Rotate x-axis labels to prevent crowding
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        # Align labels to the right for better readability when rotated
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')

    fig.tight_layout()

    # Add legend explaining color coding and hatch patterns
    legend_elements = [
        Patch(facecolor='#4A90E2', edgecolor='black', label='Qwen3-VL-2B'),
        Patch(facecolor='#2E5C8A', edgecolor='black', label='Qwen3-VL-8B'),
        Patch(facecolor='#FF6B35', edgecolor='black', label='Claude (API)'),
        Patch(facecolor='#C1121F', edgecolor='black', label='GPT (API)'),
        Patch(facecolor='gray', edgecolor='black', hatch='///', label='Fine-tuned'),
        Patch(facecolor='gray', edgecolor='black', label='Base/Pretrained'),
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=9,
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
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
