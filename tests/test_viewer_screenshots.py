"""Screenshot regression tests for viewer components.

Running tests:
    uv run pytest tests/test_viewer_screenshots.py -v
"""

import pytest

# openadapt-viewer is an optional local development dependency
pytest.importorskip("openadapt_viewer", reason="openadapt-viewer not installed (optional dependency)")

from openadapt_ml.training.viewer_components import (
    screenshot_with_predictions,
    training_metrics,
    playback_controls,
    generate_comparison_summary,
    correctness_badge,
)


def test_component_generation():
    """Test that components generate valid HTML."""
    html = screenshot_with_predictions(
        screenshot_path="test.png",
        human_action={"type": "click", "x": 0.5, "y": 0.5},
        predicted_action={"type": "click", "x": 0.5, "y": 0.5},
    )
    assert "oa-screenshot" in html

    html = training_metrics(epoch=1, loss=0.1)
    assert "oa-metrics" in html

    html = playback_controls(step_count=10)
    assert "oa-playback" in html

    html = generate_comparison_summary(total_steps=10, correct_steps=8)
    assert "oa-metrics" in html

    html = correctness_badge(is_correct=True)
    assert "oa-badge" in html
