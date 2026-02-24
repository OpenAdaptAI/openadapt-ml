"""Unit tests for _OpenAdaptCallback (training logging and early stopping).

Tests for:
- Training log JSON writing
- Early stopping: absolute threshold
- Early stopping: plateau detection
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openadapt_ml.training.trl_trainer import TRLTrainingConfig, _OpenAdaptCallback


def _make_config(tmp_path, **overrides) -> TRLTrainingConfig:
    """Create a TRLTrainingConfig pointing to tmp_path."""
    defaults = dict(
        model_name="test-model",
        output_dir=str(tmp_path),
    )
    defaults.update(overrides)
    return TRLTrainingConfig(**defaults)


def _make_state(global_step=1):
    """Create a mock trainer state."""
    state = MagicMock()
    state.global_step = global_step
    return state


def _make_args(num_train_epochs=10):
    """Create a mock training args."""
    args = MagicMock()
    args.num_train_epochs = num_train_epochs
    return args


def _make_control():
    """Create a mock trainer control."""
    control = MagicMock()
    control.should_training_stop = False
    return control


class TestTrainingLogWriting:
    """Test that training_log.json is written correctly."""

    def test_log_written_on_loss(self, tmp_path):
        config = _make_config(tmp_path)
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        cb.on_log(
            _make_args(),
            _make_state(global_step=1),
            _make_control(),
            logs={"loss": 10.5, "learning_rate": 5e-5, "epoch": 0.2},
        )

        log_path = tmp_path / "training_log.json"
        assert log_path.exists()

        data = json.loads(log_path.read_text())
        assert data["step"] == 1
        assert data["loss"] == 10.5
        assert data["model_name"] == "test-model"
        assert len(data["losses"]) == 1
        assert data["losses"][0]["loss"] == 10.5

    def test_multiple_logs(self, tmp_path):
        config = _make_config(tmp_path)
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        for step in range(1, 4):
            cb.on_log(
                _make_args(),
                _make_state(global_step=step),
                _make_control(),
                logs={"loss": 20.0 - step, "learning_rate": 1e-5, "epoch": step * 0.2},
            )

        data = json.loads((tmp_path / "training_log.json").read_text())
        assert len(data["losses"]) == 3
        assert data["step"] == 3

    def test_no_log_on_none_logs(self, tmp_path):
        config = _make_config(tmp_path)
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        cb.on_log(_make_args(), _make_state(), _make_control(), logs=None)
        assert not (tmp_path / "training_log.json").exists()

    def test_non_loss_log_uses_last_loss(self, tmp_path):
        """Non-loss log events (e.g. eval metrics) should use last known loss, not 0."""
        config = _make_config(tmp_path)
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        # First: a real loss event
        cb.on_log(
            _make_args(),
            _make_state(global_step=1),
            _make_control(),
            logs={"loss": 8.5, "learning_rate": 5e-5, "epoch": 0.1},
        )

        # Second: a non-loss log event (e.g. eval metrics)
        cb.on_log(
            _make_args(),
            _make_state(global_step=2),
            _make_control(),
            logs={"eval_loss": 9.0, "learning_rate": 4e-5, "epoch": 0.2},
        )

        data = json.loads((tmp_path / "training_log.json").read_text())
        # Should report last known loss (8.5), NOT 0
        assert data["loss"] == 8.5
        # Only 1 entry in losses array (non-loss events don't add entries)
        assert len(data["losses"]) == 1


class TestEarlyStopThreshold:
    """Test early stopping via absolute loss threshold."""

    def test_stops_after_patience(self, tmp_path):
        config = _make_config(
            tmp_path,
            early_stop_loss=5.0,
            early_stop_patience=3,
        )
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        # Loss above threshold — no stop
        control = _make_control()
        cb.on_log(_make_args(), _make_state(1), control, logs={"loss": 10.0})
        assert not control.should_training_stop

        # Loss below threshold, step 1 of 3
        control = _make_control()
        cb.on_log(_make_args(), _make_state(2), control, logs={"loss": 4.0})
        assert not control.should_training_stop

        # Step 2 of 3
        control = _make_control()
        cb.on_log(_make_args(), _make_state(3), control, logs={"loss": 3.0})
        assert not control.should_training_stop

        # Step 3 of 3 — should stop
        control = _make_control()
        cb.on_log(_make_args(), _make_state(4), control, logs={"loss": 2.0})
        assert control.should_training_stop

    def test_resets_on_high_loss(self, tmp_path):
        config = _make_config(
            tmp_path,
            early_stop_loss=5.0,
            early_stop_patience=3,
        )
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        # Below threshold
        control = _make_control()
        cb.on_log(_make_args(), _make_state(1), control, logs={"loss": 4.0})
        assert not control.should_training_stop

        # Back above threshold — resets counter
        control = _make_control()
        cb.on_log(_make_args(), _make_state(2), control, logs={"loss": 6.0})
        assert not control.should_training_stop

        # Below again — only 1 step, not 3
        control = _make_control()
        cb.on_log(_make_args(), _make_state(3), control, logs={"loss": 4.0})
        assert not control.should_training_stop

    def test_disabled_when_zero(self, tmp_path):
        config = _make_config(tmp_path, early_stop_loss=0.0)
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        control = _make_control()
        cb.on_log(_make_args(), _make_state(1), control, logs={"loss": 0.001})
        assert not control.should_training_stop


class TestEarlyStopPlateau:
    """Test early stopping via plateau detection."""

    def test_stops_on_plateau(self, tmp_path):
        config = _make_config(
            tmp_path,
            early_stop_min_delta=0.1,
            early_stop_plateau_patience=3,
        )
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        # Initial loss
        control = _make_control()
        cb.on_log(_make_args(), _make_state(1), control, logs={"loss": 10.0})
        assert not control.should_training_stop

        # Improvement (resets counter)
        control = _make_control()
        cb.on_log(_make_args(), _make_state(2), control, logs={"loss": 9.5})
        assert not control.should_training_stop

        # No improvement (within delta) — plateau step 1
        control = _make_control()
        cb.on_log(_make_args(), _make_state(3), control, logs={"loss": 9.45})
        assert not control.should_training_stop

        # Plateau step 2
        control = _make_control()
        cb.on_log(_make_args(), _make_state(4), control, logs={"loss": 9.42})
        assert not control.should_training_stop

        # Plateau step 3 — should stop
        control = _make_control()
        cb.on_log(_make_args(), _make_state(5), control, logs={"loss": 9.41})
        assert control.should_training_stop

    def test_resets_on_improvement(self, tmp_path):
        config = _make_config(
            tmp_path,
            early_stop_min_delta=0.1,
            early_stop_plateau_patience=3,
        )
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        cb.on_log(_make_args(), _make_state(1), _make_control(), logs={"loss": 10.0})
        cb.on_log(
            _make_args(), _make_state(2), _make_control(), logs={"loss": 9.95}
        )  # plateau 1
        cb.on_log(
            _make_args(), _make_state(3), _make_control(), logs={"loss": 9.92}
        )  # plateau 2

        # Big improvement — resets
        cb.on_log(_make_args(), _make_state(4), _make_control(), logs={"loss": 9.0})

        # Now need 3 more plateau steps, not 1
        control = _make_control()
        cb.on_log(_make_args(), _make_state(5), control, logs={"loss": 8.95})
        assert not control.should_training_stop

    def test_disabled_when_zero(self, tmp_path):
        config = _make_config(tmp_path, early_stop_min_delta=0.0)
        cb = _OpenAdaptCallback(config)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        # Even with flat loss, shouldn't stop
        for step in range(1, 20):
            control = _make_control()
            cb.on_log(_make_args(), _make_state(step), control, logs={"loss": 10.0})
            assert not control.should_training_stop
