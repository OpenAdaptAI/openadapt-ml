from __future__ import annotations

from openadapt_ml.datasets.next_action import NextActionDataset, build_next_action_sft_samples
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions
from openadapt_ml.models.dummy_adapter import DummyAdapter
from openadapt_ml.training.trainer import TrainingConfig, train_supervised


def test_training_loop_with_dummy_adapter() -> None:
    # Generate a tiny synthetic dataset
    sessions = generate_synthetic_sessions(num_sessions=1, seed=123, output_dir="synthetic_test_training")
    episodes = [ep for sess in sessions for ep in sess.episodes]
    samples = build_next_action_sft_samples(episodes)
    dataset = NextActionDataset(samples)

    adapter = DummyAdapter()
    cfg = TrainingConfig(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=0,
        lr_scheduler_type="linear",
    )

    # Should run without raising; we don't assert on loss values here.
    train_supervised(adapter, dataset, cfg)
