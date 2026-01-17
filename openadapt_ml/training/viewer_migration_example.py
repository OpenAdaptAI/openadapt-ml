"""Validation example demonstrating viewer component migration.

This script generates a simple viewer using the new openadapt-viewer components
through the viewer_components adapter module.

Usage:
    uv run python -m openadapt_ml.training.viewer_migration_example
"""

from pathlib import Path

from openadapt_viewer.builders import PageBuilder
from openadapt_ml.training.viewer_components import (
    training_metrics,
    playback_controls,
    generate_comparison_summary,
)


def generate_example_viewer(output_path: Path) -> None:
    """Generate example viewer demonstrating new components."""
    builder = PageBuilder(title="Viewer Migration Example", include_alpine=True)

    builder.add_header(
        title="Viewer Component Migration Example",
        subtitle="Demonstrating Phase 1 Foundation",
        nav_tabs=[
            {"href": "#", "label": "Training", "active": False},
            {"href": "#", "label": "Viewer", "active": True},
        ],
    )

    # Section 1: Training Metrics
    builder.add_section(
        content=training_metrics(
            epoch=3,
            loss=0.045,
            accuracy=0.95,
            elapsed_time=3600,
            learning_rate=1e-4,
        ),
        title="Training Metrics",
    )

    # Section 2: Comparison Summary
    builder.add_section(
        content=generate_comparison_summary(
            total_steps=20,
            correct_steps=18,
            model_name="qwen3-vl-2b",
        ),
        title="Model Comparison Summary",
    )

    # Section 3: Playback Controls
    builder.add_section(
        content=playback_controls(step_count=20),
        title="Playback Controls",
    )

    # Render to file
    html = builder.render()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"✓ Generated example viewer: {output_path}")


if __name__ == "__main__":
    output = Path("viewer_migration_example.html")
    generate_example_viewer(output)
    print(f"\nView the example: file://{output.resolve()}")
