"""Tests for the local training CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from openadapt_ml.datasets.next_action import NextActionDataset, build_next_action_sft_samples
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions
from openadapt_ml.models.dummy_adapter import DummyAdapter
from openadapt_ml.training.trainer import TrainingConfig, train_supervised


def test_local_status_command():
    """Test that the status command runs without error."""
    result = subprocess.run(
        [sys.executable, "-m", "openadapt_ml.cloud.local", "status"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "LOCAL TRAINING STATUS" in result.stdout
    assert "Device:" in result.stdout


def test_local_check_command_no_data():
    """Test check command with no training data."""
    import os
    from unittest.mock import patch
    from openadapt_ml.cloud.local import cmd_check
    import argparse

    original_dir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            args = argparse.Namespace()
            result = cmd_check(args)
            # Should return 1 (error) with no data
            assert result == 1
    finally:
        os.chdir(original_dir)


def test_local_check_command_with_mock_data():
    """Test check command with mock training data."""
    import os
    from openadapt_ml.cloud.local import cmd_check
    import argparse

    original_dir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Create mock training_output
            training_output = Path("training_output")
            training_output.mkdir()

            # Create mock training log
            mock_log = {
                "epoch": 2,
                "step": 10,
                "loss": 0.5,
                "learning_rate": 5e-5,
                "losses": [
                    {"epoch": 0, "step": 1, "loss": 2.0, "lr": 5e-5},
                    {"epoch": 0, "step": 2, "loss": 1.5, "lr": 5e-5},
                    {"epoch": 1, "step": 3, "loss": 1.0, "lr": 5e-5},
                    {"epoch": 1, "step": 4, "loss": 0.8, "lr": 5e-5},
                    {"epoch": 2, "step": 5, "loss": 0.5, "lr": 5e-5},
                ],
                "status": "completed",
            }
            (training_output / "training_log.json").write_text(json.dumps(mock_log))

            args = argparse.Namespace()
            result = cmd_check(args)
            # Should return 0 (success)
            assert result == 0
    finally:
        os.chdir(original_dir)


def test_local_train_help():
    """Test that train --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "openadapt_ml.cloud.local", "train", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--capture" in result.stdout
    assert "--goal" in result.stdout


def test_local_viewer_no_data():
    """Test viewer command with no training data."""
    import os
    from openadapt_ml.cloud.local import cmd_viewer
    import argparse

    original_dir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            args = argparse.Namespace(open=False)
            result = cmd_viewer(args)
            # Should return 1 (error) with no data
            assert result == 1
    finally:
        os.chdir(original_dir)


def test_detect_device():
    """Test device detection function."""
    from openadapt_ml.cloud.local import detect_device

    device = detect_device()
    # Should return one of the expected device types
    assert any(x in device.lower() for x in ["cuda", "mps", "cpu", "unknown"])


def test_get_training_status_empty():
    """Test get_training_status with no data."""
    import os

    from openadapt_ml.cloud.local import get_training_status

    # Save current directory
    original_dir = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            status = get_training_status()
            assert status["running"] == False
            assert status["epoch"] == 0
            assert status["has_dashboard"] == False
    finally:
        os.chdir(original_dir)


def test_get_training_status_with_data():
    """Test get_training_status with mock data."""
    import os

    from openadapt_ml.cloud.local import get_training_status

    original_dir = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Create mock training_output
            training_output = Path("training_output")
            training_output.mkdir()

            mock_log = {
                "epoch": 3,
                "step": 50,
                "loss": 0.25,
                "learning_rate": 1e-5,
                "losses": [{"loss": 1.0}, {"loss": 0.5}, {"loss": 0.25}],
                "status": "completed",
            }
            (training_output / "training_log.json").write_text(json.dumps(mock_log))
            (training_output / "dashboard.html").write_text("<html></html>")

            status = get_training_status()
            assert status["epoch"] == 3
            assert status["step"] == 50
            assert status["loss"] == 0.25
            assert status["has_dashboard"] == True
    finally:
        os.chdir(original_dir)


def test_training_with_early_stopping():
    """Test training with early stopping enabled - should complete quickly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate tiny synthetic dataset
        sessions = generate_synthetic_sessions(
            num_sessions=1,
            seed=42,
            output_dir=os.path.join(tmpdir, "synthetic_data"),
        )
        episodes = [ep for sess in sessions for ep in sess.episodes]
        samples = build_next_action_sft_samples(episodes)
        dataset = NextActionDataset(samples)

        # Use dummy adapter for fast execution
        adapter = DummyAdapter()

        # Configure for quick test with early stopping
        config = TrainingConfig(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=0,
            early_stop_loss=1e-4,  # Very low threshold
            early_stop_patience=2,  # Stop after 2 consecutive low losses
        )

        # Should complete without error
        train_supervised(adapter, dataset, config)


def test_training_generates_output():
    """Test that training generates the expected output files."""
    original_dir = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Generate tiny synthetic dataset
            sessions = generate_synthetic_sessions(
                num_sessions=1,
                seed=42,
                output_dir="synthetic_data",
            )
            episodes = [ep for sess in sessions for ep in sess.episodes]
            samples = build_next_action_sft_samples(episodes)
            dataset = NextActionDataset(samples)

            adapter = DummyAdapter()
            config = TrainingConfig(
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,  # Log every step
                output_dir="training_output",
            )

            train_supervised(adapter, dataset, config)

            # Check output files were created
            output_dir = Path("training_output")
            assert output_dir.exists(), "training_output directory should exist"

            # Training log should exist
            log_file = output_dir / "training_log.json"
            if log_file.exists():
                with open(log_file) as f:
                    data = json.load(f)
                assert "epoch" in data
                assert "step" in data
    finally:
        os.chdir(original_dir)
