"""Tests for training with TRL trainer.

Note: The legacy train_supervised has been replaced by train_with_trl.
These tests verify the new TRL-based training works correctly.
"""

from __future__ import annotations

import pytest

from openadapt_ml.ingest.synthetic import generate_synthetic_episodes


def test_trl_training_config() -> None:
    """Test TRLTrainingConfig creation and defaults."""
    from openadapt_ml.training.trl_trainer import TRLTrainingConfig

    config = TRLTrainingConfig()
    assert config.num_epochs == 3
    assert config.batch_size == 1
    assert config.learning_rate == 2e-4
    assert config.output_dir == "checkpoints"


def test_convert_samples_to_trl_format() -> None:
    """Test sample conversion to TRL format."""
    from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format
    from openadapt_ml.datasets.next_action import build_next_action_sft_samples
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic episodes with images
        episodes = generate_synthetic_episodes(
            num_episodes=1,
            seed=123,
            output_dir=str(Path(tmpdir) / "synthetic")
        )

        # Build SFT samples
        samples = build_next_action_sft_samples(episodes)

        if samples:
            # Convert to TRL format (loads images as PIL)
            trl_samples = _convert_samples_to_trl_format(samples, base_path=Path(tmpdir))

            # Verify samples have expected structure
            for sample in trl_samples:
                assert "images" in sample
                assert "messages" in sample
                # Images should be PIL Images (not paths)
                for img in sample["images"]:
                    assert hasattr(img, "convert"), "Expected PIL Image"


@pytest.mark.skip(reason="Requires TRL and GPU - run manually for integration testing")
def test_train_with_trl_integration() -> None:
    """Integration test for train_with_trl.

    This test is skipped by default as it requires:
    - TRL library installed
    - GPU for efficient execution
    - Significant time to run

    Run manually with: pytest tests/test_training_dummy.py -k test_train_with_trl_integration --no-skip
    """
    from openadapt_ml.training.trl_trainer import train_with_trl, TRLTrainingConfig
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        episodes = generate_synthetic_episodes(
            num_episodes=1,
            seed=123,
            output_dir=str(Path(tmpdir) / "synthetic")
        )

        config = TRLTrainingConfig(
            num_epochs=1,
            batch_size=1,
            output_dir=str(Path(tmpdir) / "checkpoints"),
        )

        checkpoint = train_with_trl(
            episodes=episodes,
            config=config,
            base_path=Path(tmpdir),
        )

        assert Path(checkpoint).exists()
