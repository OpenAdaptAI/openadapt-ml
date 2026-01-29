"""Test script for live evaluation progress tracking.

This script demonstrates the live evaluation tracking feature by running
a mock benchmark evaluation and generating live progress updates that can
be viewed in the benchmark viewer.

Usage:
    # Terminal 1: Run this test (generates benchmark_live.json)
    uv run python test_live_eval.py

    # Terminal 2: Serve the viewer
    uv run python -m openadapt_ml.cloud.local serve --open

    # Navigate to the Benchmarks tab to see live progress
"""

import time
from pathlib import Path

from openadapt_evals import (
    EvaluationConfig,
    RandomAgent,
    WAAMockAdapter,
    evaluate_agent_on_benchmark,
)


def main():
    print("Starting live evaluation tracking test...")
    print("This will generate benchmark_live.json that the viewer can poll.")
    print()

    # Use mock adapter (no Windows required)
    adapter = WAAMockAdapter()

    # Use random agent
    agent = RandomAgent()

    # Configure evaluation with live tracking
    config = EvaluationConfig(
        max_steps=10,
        verbose=True,
        model_id="test-random-agent",
        enable_live_tracking=True,
        live_tracking_file="training_output/benchmark_live.json",  # Write to training_output
        save_execution_traces=False,  # Disable trace collection for this test
    )

    print(f"Live tracking file: {config.live_tracking_file}")
    print(f"Evaluating 5 mock tasks with random agent...")
    print()

    # Create output directory
    Path("training_output").mkdir(exist_ok=True)

    # Run evaluation (this will update benchmark_live.json in real-time)
    results = evaluate_agent_on_benchmark(
        agent=agent,
        adapter=adapter,
        task_ids=None,  # All tasks
        config=config,
    )

    print()
    print("Evaluation complete!")
    print(f"Live tracking file: {Path(config.live_tracking_file).absolute()}")
    print()
    print("To view results:")
    print("  1. uv run python -m openadapt_ml.cloud.local serve --open")
    print("  2. Navigate to the Benchmarks tab")
    print()

    # Print summary
    success_count = sum(1 for r in results if r.success)
    print(f"Results: {success_count}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
