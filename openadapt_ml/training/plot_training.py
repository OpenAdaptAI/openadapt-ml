"""Generate training loss and learning rate plots from training_log.json.

Usage:
    python -m openadapt_ml.training.plot_training training_output/training_log.json
    python -m openadapt_ml.training.plot_training training_log.json --output docs/images/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(
    log_path: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate training plots from a training_log.json file.

    Args:
        log_path: Path to training_log.json.
        output_dir: Directory for output PNGs. Defaults to log_path's parent.

    Returns:
        List of paths to generated PNG files.
    """
    data = json.loads(log_path.read_text())
    losses = data.get("losses", [])
    if not losses:
        print(f"No loss data in {log_path}")
        return []

    if output_dir is None:
        output_dir = log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [entry["step"] for entry in losses]
    loss_vals = [entry["loss"] for entry in losses]
    lr_vals = [entry.get("lr", 0) for entry in losses]
    epochs = [entry.get("epoch", 0) for entry in losses]

    generated = []

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, loss_vals, color="#2E5C8A", linewidth=2)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Mark epoch boundaries
    seen_epochs = set()
    for i, ep in enumerate(epochs):
        ep_int = int(ep)
        if ep_int not in seen_epochs and ep_int > 0:
            seen_epochs.add(ep_int)
            ax.axvline(
                x=steps[i], color="gray", linestyle="--", alpha=0.4, linewidth=0.8
            )

    fig.tight_layout()
    loss_path = output_dir / "training_loss.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(loss_path)

    # --- Learning rate schedule ---
    if any(lr > 0 for lr in lr_vals):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(steps, lr_vals, color="#FF6B35", linewidth=2)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-2, -2))
        fig.tight_layout()
        lr_path = output_dir / "training_lr.png"
        fig.savefig(lr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(lr_path)

    # --- Combined (loss + LR on dual axis) ---
    if any(lr > 0 for lr in lr_vals):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color_loss = "#2E5C8A"
        color_lr = "#FF6B35"

        ax1.set_xlabel("Step", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12, color=color_loss)
        ax1.plot(steps, loss_vals, color=color_loss, linewidth=2, label="Loss")
        ax1.tick_params(axis="y", labelcolor=color_loss)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Learning Rate", fontsize=12, color=color_lr)
        ax2.plot(
            steps, lr_vals, color=color_lr, linewidth=1.5, linestyle="--", label="LR"
        )
        ax2.tick_params(axis="y", labelcolor=color_lr)
        ax2.ticklabel_format(axis="y", style="scientific", scilimits=(-2, -2))

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        model_name = data.get("model_name", "")
        title = "Training Loss & Learning Rate"
        if model_name:
            title += f" ({model_name})"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        combined_path = output_dir / "training_combined.png"
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(combined_path)

    for p in generated:
        print(f"  Generated: {p}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training curves from training_log.json"
    )
    parser.add_argument("log_path", type=str, help="Path to training_log.json")
    parser.add_argument("--output", type=str, help="Output directory for PNGs")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    output_dir = Path(args.output) if args.output else None
    plot_training_curves(log_path, output_dir)


if __name__ == "__main__":
    main()
