#!/usr/bin/env python3
"""Test script to verify TRL trainer batching implementation.

NOTE: The legacy train_supervised has been removed. This test now
uses the TRL trainer and verifies TRLTrainingConfig works correctly.
For actual batching tests, run:

    pytest tests/test_training_dummy.py -v

Or for integration testing with GPU:

    pytest tests/test_training_dummy.py -k test_train_with_trl_integration --no-skip
"""

from __future__ import annotations


def test_trl_config_batching():
    """Test TRLTrainingConfig supports different batch sizes."""
    from openadapt_ml.training.trl_trainer import TRLTrainingConfig

    # Test batch_size=2
    cfg = TRLTrainingConfig(
        num_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        output_dir="test_batching_output",
    )
    assert cfg.batch_size == 2
    print("Config with batch_size=2: OK")

    # Test batch_size=4
    cfg2 = TRLTrainingConfig(
        num_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        output_dir="test_batching_output_4",
    )
    assert cfg2.batch_size == 4
    assert cfg2.gradient_accumulation_steps == 2
    print("Config with batch_size=4: OK")

    print("\n" + "="*60)
    print("All batching config tests passed!")
    print("="*60)
    return True


if __name__ == "__main__":
    test_trl_config_batching()
