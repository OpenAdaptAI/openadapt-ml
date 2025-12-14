# Stub Training Adapter

## Problem

Testing UI components (dashboard, viewer, stop button, etc.) requires waiting for actual training:
- Local training: slow, requires GPU
- Lambda training: costs money, requires internet, takes minutes to start

This blocks rapid iteration on the codebase.

## Solution

Implement a stub/mock training adapter that:
1. Simulates training progress without actual computation
2. Generates fake loss curves, evaluations, checkpoints
3. Responds to stop signals
4. Runs instantly for UI testing

## Usage

```bash
# Start stub training (simulates 5 epochs in ~30 seconds)
uv run python -m openadapt_ml.cloud.lambda_labs train --stub --capture /path/to/capture

# Or with monitor
uv run python -m openadapt_ml.cloud.lambda_labs monitor --stub --open

# Serve dashboard for stub data
uv run python -m openadapt_ml.cloud.lambda_labs serve --open
```

## Implementation

### StubTrainingProvider

```python
class StubTrainingProvider:
    """Simulates training without actual computation."""

    def __init__(self, output_dir: Path, epochs: int = 5, steps_per_epoch: int = 10):
        self.output_dir = output_dir
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.current_step = 0
        self.losses = []

    def simulate_step(self) -> dict:
        """Simulate one training step."""
        # Generate decreasing loss with noise
        base_loss = 2.5 * (1 - self.current_step / (self.epochs * self.steps_per_epoch))
        noise = random.uniform(-0.2, 0.2)
        loss = max(0.1, base_loss + noise)

        self.losses.append({
            "epoch": self.current_epoch,
            "step": self.current_step,
            "loss": loss,
            "lr": 5e-5,
            "time": self.current_step * 2.0  # 2 sec per step
        })

        self.current_step += 1
        if self.current_step % self.steps_per_epoch == 0:
            self.current_epoch += 1

        return self.get_status()

    def get_status(self) -> dict:
        """Return current training status."""
        return {
            "job_id": f"stub_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "epoch": self.current_epoch,
            "step": self.current_step,
            "total_epochs": self.epochs,
            "loss": self.losses[-1]["loss"] if self.losses else 0,
            "losses": self.losses,
            "evaluations": self._generate_fake_evaluations(),
            "elapsed_time": self.current_step * 2.0,
        }

    def _generate_fake_evaluations(self) -> list:
        """Generate fake evaluation samples."""
        if self.current_epoch == 0:
            return []
        return [
            {
                "epoch": e,
                "sample_idx": 0,
                "image_path": "screenshots/sample.png",
                "human_action": {"type": "click", "x": 0.5, "y": 0.5},
                "predicted_action": {
                    "type": "click",
                    "x": 0.5 + random.uniform(-0.1, 0.1),
                    "y": 0.5 + random.uniform(-0.1, 0.1),
                    "raw_output": "Thought: This is a stub prediction.\nAction: CLICK(x=0.5, y=0.5)"
                },
                "distance": random.uniform(0.05, 0.2),
                "correct": random.random() > 0.3,
            }
            for e in range(self.current_epoch)
        ]
```

### Integration Points

1. **lambda_labs.py**: Add `--stub` flag to `train` and `monitor` commands
2. **trainer.py**: Add `StubTrainer` class that uses `StubTrainingProvider`
3. **Dashboard**: Works unchanged (just reads training_log.json)

### Benefits

- **Fast iteration**: Test UI changes in seconds, not minutes
- **No GPU required**: Run on any machine
- **No cost**: No Lambda credits consumed
- **Deterministic**: Reproducible test scenarios
- **CI/CD friendly**: Can run in automated tests

## Priority

**HIGH** - This unblocks rapid development of:
- Dashboard improvements
- Viewer enhancements
- Stop button functionality
- Checkpoint download flow
- Any UI/UX work

## Files to Modify

1. `openadapt_ml/cloud/lambda_labs.py` - Add `--stub` flag and stub provider
2. `openadapt_ml/training/trainer.py` - Add `StubTrainer` class
3. `openadapt_ml/training/stub_provider.py` - New file for `StubTrainingProvider`
