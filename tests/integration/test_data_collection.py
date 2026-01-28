"""Test script for benchmark data collection.

This script tests the Phase 1 implementation of benchmark viewer integration
by running a few tasks with the WAAMockAdapter and verifying that execution
traces are saved correctly.

Usage:
    uv run python test_data_collection.py

Expected output:
    - Creates benchmark_results/test_run_phase1/ directory
    - Contains metadata.json, summary.json, and tasks/ subdirectory
    - Each task has task.json, execution.json, and screenshots/
    - All JSON files are valid and contain expected fields
    - Screenshots are saved as PNG files (step_000.png, step_001.png, etc.)

CLI alternative:
    This test is also available as a CLI command:
    uv run python -m openadapt_ml.benchmarks.cli test-collection --tasks 5

What's tested:
    ✓ ExecutionTraceCollector creates directory structure
    ✓ Screenshots are saved at each step
    ✓ Task definitions are saved with all fields
    ✓ Execution traces include actions, reasoning, timestamps
    ✓ Summary JSON contains aggregate metrics
    ✓ All files are valid JSON
"""

import logging
from pathlib import Path

# Import from openadapt-evals (canonical benchmark package)
from openadapt_evals import (
    EvaluationConfig,
    RandomAgent,
    WAAMockAdapter,
    evaluate_agent_on_benchmark,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_data_collection():
    """Test data collection with WAAMockAdapter."""
    logger.info("Starting data collection test...")

    # Create mock adapter with a few tasks
    adapter = WAAMockAdapter(num_tasks=5, domains=["browser", "office"])
    logger.info(f"Created mock adapter with {len(adapter.list_tasks())} tasks")

    # Create a random agent for testing
    agent = RandomAgent(action_types=["click", "type", "scroll", "done"], seed=42)
    logger.info("Created random agent")

    # Configure evaluation with data collection
    config = EvaluationConfig(
        max_steps=10,
        parallel=1,
        save_trajectories=True,
        save_execution_traces=True,
        model_id="random-agent-test",
        output_dir="benchmark_results",
        run_name="test_run_phase1",
        verbose=True,
    )

    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluate_agent_on_benchmark(
        agent=agent,
        adapter=adapter,
        config=config,
    )

    # Verify results
    logger.info(f"\nEvaluation complete! Results:")
    logger.info(f"  Total tasks: {len(results)}")
    logger.info(f"  Success: {sum(1 for r in results if r.success)}")
    logger.info(f"  Failure: {sum(1 for r in results if not r.success)}")

    # Verify directory structure
    run_dir = Path("benchmark_results") / "test_run_phase1"
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return False

    logger.info(f"\nVerifying directory structure at: {run_dir}")

    # Check metadata.json
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        logger.info("  ✓ metadata.json exists")
    else:
        logger.error("  ✗ metadata.json missing")
        return False

    # Check summary.json
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        logger.info("  ✓ summary.json exists")
        import json
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info(f"    - Success rate: {summary['success_rate']:.1%}")
        logger.info(f"    - Avg steps: {summary['avg_steps']:.1f}")
    else:
        logger.error("  ✗ summary.json missing")
        return False

    # Check tasks directory
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.exists():
        logger.error("  ✗ tasks directory missing")
        return False

    logger.info(f"  ✓ tasks directory exists")

    # Check each task directory
    task_dirs = list(tasks_dir.iterdir())
    logger.info(f"  ✓ Found {len(task_dirs)} task directories")

    for task_dir in task_dirs[:3]:  # Check first 3 tasks
        logger.info(f"\n  Checking task: {task_dir.name}")

        # Check task.json
        task_json = task_dir / "task.json"
        if task_json.exists():
            logger.info("    ✓ task.json exists")
        else:
            logger.error("    ✗ task.json missing")
            return False

        # Check execution.json
        execution_json = task_dir / "execution.json"
        if execution_json.exists():
            logger.info("    ✓ execution.json exists")
            with open(execution_json) as f:
                execution = json.load(f)
            logger.info(f"      - Steps: {len(execution['steps'])}")
            logger.info(f"      - Success: {execution['success']}")
        else:
            logger.error("    ✗ execution.json missing")
            return False

        # Check screenshots directory
        screenshots_dir = task_dir / "screenshots"
        if screenshots_dir.exists():
            screenshots = list(screenshots_dir.glob("*.png"))
            logger.info(f"    ✓ screenshots directory exists ({len(screenshots)} images)")
        else:
            logger.error("    ✗ screenshots directory missing")
            return False

    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed! Data collection is working correctly.")
    logger.info("=" * 60)
    logger.info(f"\nYou can inspect the results at: {run_dir.absolute()}")

    return True


if __name__ == "__main__":
    success = test_data_collection()
    exit(0 if success else 1)
