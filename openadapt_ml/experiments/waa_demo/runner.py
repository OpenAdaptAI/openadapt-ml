"""Runner for WAA demo-conditioned experiment.

Usage:
    # List all tasks and demo status
    python -m openadapt_ml.experiments.waa_demo.runner list

    # Show a specific demo
    python -m openadapt_ml.experiments.waa_demo.runner show 8

    # Run experiment (requires WAA environment)
    python -m openadapt_ml.experiments.waa_demo.runner run --condition demo
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from openadapt_ml.experiments.waa_demo.demos import (
    DEMOS,
    format_demo_for_prompt,
    get_complete_demos,
    get_demo,
    get_placeholder_demos,
)
from openadapt_ml.experiments.waa_demo.tasks import (
    TASKS,
    get_manual_tasks,
    get_recorded_tasks,
    get_task,
)


def cmd_list(args: argparse.Namespace) -> int:
    """List all tasks with their demo status."""
    print("WAA Demo Experiment - Task List")
    print("=" * 80)
    print()

    complete = get_complete_demos()
    placeholder = get_placeholder_demos()

    print(f"Tasks: {len(TASKS)} total")
    print(f"  Manual demos written: {len(complete)}")
    print(f"  Recorded demos needed: {len(placeholder)}")
    print()
    print("-" * 80)
    print(f"{'#':<3} {'Domain':<18} {'Difficulty':<8} {'Demo':<10} {'Instruction'}")
    print("-" * 80)

    for num, task in TASKS.items():
        demo_status = "Ready" if num in complete else "NEEDS REC"
        print(
            f"{num:<3} {task.domain.value:<18} {task.difficulty.value:<8} "
            f"{demo_status:<10} {task.instruction[:45]}..."
        )

    print()
    print("Tasks needing recorded demos on Windows:")
    for task in get_recorded_tasks():
        print(f"  - #{list(TASKS.keys())[list(TASKS.values()).index(task)]}: {task.instruction}")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show a specific demo."""
    task_num = args.task
    task = get_task(task_num)
    demo = get_demo(task_num)

    if not task:
        print(f"Error: Task {task_num} not found (valid: 1-10)")
        return 1

    print(f"Task #{task_num}: {task.instruction}")
    print(f"Domain: {task.domain.value}")
    print(f"Difficulty: {task.difficulty.value}")
    print(f"Demo method: {task.demo_method}")
    print()
    print("=" * 80)
    print("DEMO:")
    print("=" * 80)
    print(demo or "No demo available")

    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    """Generate a prompt for a task with optional demo."""
    task_num = args.task
    task = get_task(task_num)
    demo = get_demo(task_num) if args.with_demo else None

    if not task:
        print(f"Error: Task {task_num} not found")
        return 1

    print("=" * 80)
    print("GENERATED PROMPT")
    print("=" * 80)
    print()

    if demo and "[PLACEHOLDER" not in demo:
        prompt = format_demo_for_prompt(demo, task.instruction)
        print(prompt)
    else:
        print(f"Task: {task.instruction}")
        print()
        print("Analyze the screenshot and provide the next action to complete this task.")
        if demo and "[PLACEHOLDER" in demo:
            print()
            print("[Note: Demo not available - this would be zero-shot]")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the experiment (placeholder - requires WAA environment)."""
    print("WAA Demo Experiment Runner")
    print("=" * 80)
    print()
    print(f"Condition: {args.condition}")
    print(f"Tasks: {args.tasks or 'all'}")
    print()

    if not args.waa_url:
        print("Error: WAA environment not configured")
        print()
        print("To run the experiment:")
        print("  1. Start the Azure VM: az vm start -g openadapt-agents -n waa-eval-vm")
        print("  2. Wait for WAA server: curl http://<vm-ip>:5000/health")
        print("  3. Run: python -m openadapt_ml.experiments.waa_demo.runner run \\")
        print("       --condition demo --waa-url http://<vm-ip>:5000")
        return 1

    # TODO: Implement actual experiment runner with WAA integration
    print("Experiment execution not yet implemented.")
    print("This will integrate with openadapt_ml.benchmarks.waa when ready.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WAA Demo-Conditioned Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List all tasks")
    list_parser.set_defaults(func=cmd_list)

    # show command
    show_parser = subparsers.add_parser("show", help="Show a specific demo")
    show_parser.add_argument("task", help="Task number (1-10)")
    show_parser.set_defaults(func=cmd_show)

    # prompt command
    prompt_parser = subparsers.add_parser("prompt", help="Generate prompt for a task")
    prompt_parser.add_argument("task", help="Task number (1-10)")
    prompt_parser.add_argument("--with-demo", action="store_true", help="Include demo")
    prompt_parser.set_defaults(func=cmd_prompt)

    # run command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument(
        "--condition",
        choices=["zero-shot", "demo"],
        default="demo",
        help="Experiment condition",
    )
    run_parser.add_argument("--tasks", help="Comma-separated task numbers (default: all)")
    run_parser.add_argument("--waa-url", help="WAA server URL (e.g., http://vm-ip:5000)")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
