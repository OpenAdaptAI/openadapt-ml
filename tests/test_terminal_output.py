#!/usr/bin/env python3
"""Test terminal output streaming functionality."""

from pathlib import Path
from openadapt_ml.training.stub_provider import StubTrainingProvider

# Create stub provider and run short training
output_dir = Path("training_output")
provider = StubTrainingProvider(
    output_dir=output_dir,
    epochs=2,
    steps_per_epoch=3,
    step_delay=0.1,
)

print("Starting stub training to test terminal output...")
provider.run()

# Check if training.log was created
log_file = output_dir / "training.log"
if log_file.exists():
    print(f"\n✓ Log file created: {log_file}")
    print("\nLog file contents:")
    print("=" * 60)
    print(log_file.read_text())
    print("=" * 60)
else:
    print(f"\n✗ Log file not found: {log_file}")
