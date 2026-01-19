"""Adapter module for openadapt-viewer components.

This module provides wrapper functions that adapt openadapt-viewer components
for openadapt-ml specific use cases, particularly for training visualization.

Migration Approach:
------------------
Phase 1 (Foundation): Create this adapter module to establish patterns
Phase 2 (Integration): Gradually migrate viewer.py to use these adapters
Phase 3 (Consolidation): Remove duplicate code from viewer.py
Phase 4 (Completion): Full dependency on openadapt-viewer

Design Principles:
-----------------
1. Each function wraps openadapt-viewer components with ML-specific context
2. Functions accept openadapt-ml data structures (TrainingState, predictions, etc.)
3. No breaking changes to existing viewer.py code
4. Can be incrementally adopted in future phases
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Import openadapt-viewer components
from openadapt_viewer.components import (
    screenshot_display as _screenshot_display,
    playback_controls as _playback_controls,
    metrics_grid as _metrics_grid,
    badge as _badge,
)


def screenshot_with_predictions(
    screenshot_path: str | Path,
    human_action: dict[str, Any] | None = None,
    predicted_action: dict[str, Any] | None = None,
    step_number: int | None = None,
    show_difference: bool = True,
) -> str:
    """Generate screenshot display with human and AI action overlays."""
    overlays = []

    if human_action:
        overlays.append(
            {
                "type": human_action.get("type", "click"),
                "x": human_action.get("x", 0),
                "y": human_action.get("y", 0),
                "label": "H",
                "variant": "human",
                "color": "#34d399",
            }
        )

    if predicted_action:
        overlays.append(
            {
                "type": predicted_action.get("type", "click"),
                "x": predicted_action.get("x", 0),
                "y": predicted_action.get("y", 0),
                "label": "AI",
                "variant": "predicted",
                "color": "#00d4aa",
            }
        )

    caption = f"Step {step_number}" if step_number is not None else None

    return _screenshot_display(
        image_path=str(screenshot_path),
        overlays=overlays,
        caption=caption,
    )


def training_metrics(
    epoch: int | None = None,
    loss: float | None = None,
    accuracy: float | None = None,
    elapsed_time: float | None = None,
    learning_rate: float | None = None,
    **additional_metrics: Any,
) -> str:
    """Generate metrics grid for training statistics."""
    metrics = []

    if epoch is not None:
        metrics.append({"label": "Epoch", "value": epoch})

    if loss is not None:
        color = "success" if loss < 0.1 else "warning" if loss < 0.5 else "error"
        metrics.append({"label": "Loss", "value": f"{loss:.4f}", "color": color})

    if accuracy is not None:
        color = (
            "success" if accuracy > 0.9 else "warning" if accuracy > 0.7 else "error"
        )
        metrics.append(
            {"label": "Accuracy", "value": f"{accuracy:.2%}", "color": color}
        )

    if elapsed_time is not None:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{hours}h {minutes}m {seconds}s"
        metrics.append({"label": "Elapsed", "value": time_str})

    if learning_rate is not None:
        metrics.append({"label": "LR", "value": f"{learning_rate:.2e}"})

    for key, value in additional_metrics.items():
        label = key.replace("_", " ").title()
        metrics.append({"label": label, "value": str(value)})

    return _metrics_grid(metrics)


def playback_controls(
    step_count: int,
    initial_step: int = 0,
) -> str:
    """Generate playback controls for step-by-step viewer."""
    return _playback_controls(
        step_count=step_count,
        initial_step=initial_step,
    )


def correctness_badge(is_correct: bool, show_label: bool = True) -> str:
    """Generate a badge indicating prediction correctness."""
    if is_correct:
        text = "Correct" if show_label else "✓"
        color = "success"
    else:
        text = "Incorrect" if show_label else "✗"
        color = "error"

    return _badge(text=text, color=color)


def generate_comparison_summary(
    total_steps: int,
    correct_steps: int,
    model_name: str | None = None,
) -> str:
    """Generate a summary card for model comparison results."""
    accuracy = correct_steps / total_steps if total_steps > 0 else 0
    incorrect_steps = total_steps - correct_steps

    metrics = [
        {"label": "Total Steps", "value": total_steps},
        {"label": "Correct", "value": correct_steps, "color": "success"},
        {
            "label": "Incorrect",
            "value": incorrect_steps,
            "color": "error" if incorrect_steps > 0 else "muted",
        },
        {
            "label": "Accuracy",
            "value": f"{accuracy:.1%}",
            "color": "success" if accuracy > 0.9 else "warning",
        },
    ]

    if model_name:
        metrics.insert(0, {"label": "Model", "value": model_name})

    return _metrics_grid(metrics)


__all__ = [
    "screenshot_with_predictions",
    "training_metrics",
    "playback_controls",
    "correctness_badge",
    "generate_comparison_summary",
]
