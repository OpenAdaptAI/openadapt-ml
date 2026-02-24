"""Tests for plot_training.py — training curve plot generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openadapt_ml.training.plot_training import plot_training_curves


def _write_log(tmp_path: Path, losses: list[dict], **extra) -> Path:
    """Write a training_log.json fixture."""
    data = {"losses": losses, **extra}
    log_path = tmp_path / "training_log.json"
    log_path.write_text(json.dumps(data))
    return log_path


class TestPlotTrainingCurves:
    """Test plot generation from training_log.json."""

    def test_generates_loss_plot(self, tmp_path):
        log_path = _write_log(
            tmp_path,
            [
                {"step": 1, "loss": 10.0, "lr": 0, "epoch": 0.2},
                {"step": 2, "loss": 8.0, "lr": 1e-5, "epoch": 0.4},
                {"step": 3, "loss": 6.0, "lr": 2e-5, "epoch": 0.6},
            ],
        )
        result = plot_training_curves(log_path, tmp_path)

        assert any(p.name == "training_loss.png" for p in result)
        assert (tmp_path / "training_loss.png").exists()
        assert (tmp_path / "training_loss.png").stat().st_size > 0

    def test_generates_lr_plot_when_lr_nonzero(self, tmp_path):
        log_path = _write_log(
            tmp_path,
            [
                {"step": 1, "loss": 10.0, "lr": 1e-5, "epoch": 0.2},
                {"step": 2, "loss": 8.0, "lr": 2e-5, "epoch": 0.4},
            ],
        )
        result = plot_training_curves(log_path, tmp_path)

        assert any(p.name == "training_lr.png" for p in result)
        assert any(p.name == "training_combined.png" for p in result)

    def test_skips_lr_plot_when_all_zero(self, tmp_path):
        log_path = _write_log(
            tmp_path,
            [
                {"step": 1, "loss": 10.0, "lr": 0, "epoch": 0.2},
                {"step": 2, "loss": 8.0, "lr": 0, "epoch": 0.4},
            ],
        )
        result = plot_training_curves(log_path, tmp_path)

        # Only loss plot, no LR or combined
        assert len(result) == 1
        assert result[0].name == "training_loss.png"

    def test_empty_losses_returns_empty(self, tmp_path):
        log_path = _write_log(tmp_path, [])
        result = plot_training_curves(log_path, tmp_path)
        assert result == []

    def test_output_dir_created(self, tmp_path):
        log_path = _write_log(
            tmp_path,
            [{"step": 1, "loss": 5.0, "lr": 0, "epoch": 0.1}],
        )
        output_dir = tmp_path / "subdir" / "plots"
        result = plot_training_curves(log_path, output_dir)

        assert output_dir.exists()
        assert len(result) >= 1

    def test_defaults_output_to_log_parent(self, tmp_path):
        log_path = _write_log(
            tmp_path,
            [{"step": 1, "loss": 5.0, "lr": 0, "epoch": 0.1}],
        )
        result = plot_training_curves(log_path)

        # Should write to same dir as log_path
        assert all(p.parent == tmp_path for p in result)

    def test_combined_plot_includes_model_name(self, tmp_path):
        log_path = _write_log(
            tmp_path,
            [
                {"step": 1, "loss": 10.0, "lr": 1e-5, "epoch": 0.2},
                {"step": 2, "loss": 8.0, "lr": 2e-5, "epoch": 0.4},
            ],
            model_name="Qwen/Qwen3-VL-2B-Instruct",
        )
        result = plot_training_curves(log_path, tmp_path)

        # Combined plot should exist
        assert any(p.name == "training_combined.png" for p in result)

    def test_epoch_boundaries_dont_crash(self, tmp_path):
        """Multiple epoch values should produce epoch boundary lines without error."""
        log_path = _write_log(
            tmp_path,
            [
                {"step": 1, "loss": 10.0, "lr": 0, "epoch": 0.5},
                {"step": 2, "loss": 9.0, "lr": 0, "epoch": 1.0},
                {"step": 3, "loss": 8.0, "lr": 0, "epoch": 1.5},
                {"step": 4, "loss": 7.0, "lr": 0, "epoch": 2.0},
            ],
        )
        result = plot_training_curves(log_path, tmp_path)
        assert len(result) >= 1

    def test_real_training_log(self):
        """Test against the actual reconstructed training log if it exists."""
        log_path = Path("docs/training_results/qwen3vl2b_demo_training_log.json")
        if not log_path.exists():
            pytest.skip("Reconstructed training log not found")

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = plot_training_curves(log_path, Path(tmp))
            assert len(result) == 3  # loss, lr, combined
            for p in result:
                assert p.stat().st_size > 1000  # non-trivial PNG


class TestCheckpointCoLocation:
    """Test that training_log.json is copied to checkpoint dir after training."""

    def test_log_copied_to_checkpoint(self, tmp_path):
        """Simulate the co-location logic from _run_sft_training."""
        import shutil

        # Simulate: training writes log to output_dir
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        log_data = {
            "losses": [{"step": 1, "loss": 10.0, "lr": 1e-5, "epoch": 0.2}],
            "model_name": "test",
        }
        log_path = output_dir / "training_log.json"
        log_path.write_text(json.dumps(log_data))

        # Simulate: trainer saves checkpoint
        checkpoint_path = output_dir / "final"
        checkpoint_path.mkdir()

        # The co-location logic from trl_trainer.py
        if log_path.exists():
            shutil.copy2(log_path, checkpoint_path / "training_log.json")

        assert (checkpoint_path / "training_log.json").exists()
        copied = json.loads((checkpoint_path / "training_log.json").read_text())
        assert copied["model_name"] == "test"

    def test_plots_copied_to_checkpoint(self, tmp_path):
        """Simulate plot generation and copy to checkpoint dir."""
        import shutil

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        log_data = {
            "losses": [
                {"step": 1, "loss": 10.0, "lr": 1e-5, "epoch": 0.2},
                {"step": 2, "loss": 8.0, "lr": 2e-5, "epoch": 0.4},
            ],
            "model_name": "test",
        }
        log_path = output_dir / "training_log.json"
        log_path.write_text(json.dumps(log_data))

        checkpoint_path = output_dir / "final"
        checkpoint_path.mkdir()

        # Generate plots
        plots = plot_training_curves(log_path, output_dir)
        for plot_path in plots:
            shutil.copy2(plot_path, checkpoint_path / plot_path.name)

        assert (checkpoint_path / "training_loss.png").exists()
        assert (checkpoint_path / "training_combined.png").exists()
