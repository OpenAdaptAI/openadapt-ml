#!/usr/bin/env python
"""Run multi-step execution experiment.

Tests whether demo-following holds beyond the first action by executing
a full task trajectory and measuring accuracy at each step.

Usage:
    uv run python scripts/run_multistep_experiment.py

Results are saved to: openadapt_ml/experiments/demo_prompt/results/
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openadapt_ml.experiments.demo_prompt.run_experiment import (
    DemoPromptExperiment,
)


# The full 5-step demo for "Turn off Night Shift"
# Using behavior-only format (no explanatory annotations)
DEMO_TURN_OFF_NIGHTSHIFT = """DEMONSTRATION:
Goal: Turn off Night Shift in macOS System Settings

Step 1:
  Screen: Desktop
  Action: CLICK(0.01, 0.01)
  Result: Apple menu opened

Step 2:
  Screen: Apple menu visible
  Action: CLICK(0.05, 0.15)
  Result: System Settings opened

Step 3:
  Screen: System Settings with sidebar
  Action: CLICK(0.12, 0.45)
  Result: Displays panel shown

Step 4:
  Screen: Displays panel
  Action: CLICK(0.75, 0.65)
  Result: Night Shift popup appeared

Step 5:
  Screen: Night Shift popup
  Action: CLICK(0.50, 0.35)
  Result: Schedule set to Off

---"""


# Screenshots for each step of the trajectory
# These are from the actual turn-off-nightshift capture
SCREENSHOT_DIR = Path("/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift/screenshots")

# Map step numbers to screenshot files
# We'll use available screenshots that represent each stage
STEP_SCREENSHOTS = {
    1: "capture_31807990_step_0.png",   # Desktop (starting state)
    2: "capture_31807990_step_2.png",   # After Apple menu click
    3: "capture_31807990_step_5.png",   # System Settings open
    4: "capture_31807990_step_10.png",  # Displays panel
    5: "capture_31807990_step_15.png",  # Night Shift area
}

# Expected actions for each step (ground truth)
# These are the "correct" actions based on the demo
EXPECTED_ACTIONS = {
    1: {"type": "click", "target": "Apple menu", "region": "top-left", "x_range": (0, 0.05), "y_range": (0, 0.05)},
    2: {"type": "click", "target": "System Settings", "region": "menu", "x_range": (0, 0.15), "y_range": (0.1, 0.3)},
    3: {"type": "click", "target": "Displays", "region": "sidebar", "x_range": (0.05, 0.25), "y_range": (0.3, 0.6)},
    4: {"type": "click", "target": "Night Shift button", "region": "main-panel", "x_range": (0.5, 0.9), "y_range": (0.5, 0.8)},
    5: {"type": "click", "target": "Schedule dropdown", "region": "popup", "x_range": (0.3, 0.7), "y_range": (0.2, 0.5)},
}

# Task instructions for each step
STEP_TASKS = {
    1: "Turn off Night Shift in macOS System Settings. You are at the desktop.",
    2: "Turn off Night Shift. The Apple menu is now open. Select System Settings.",
    3: "Turn off Night Shift. System Settings is open. Navigate to the Displays section.",
    4: "Turn off Night Shift. You are in the Displays panel. Open the Night Shift settings.",
    5: "Turn off Night Shift. The Night Shift popup is open. Set the Schedule to Off.",
}


def parse_click_coordinates(action_str: str) -> tuple[float, float] | None:
    """Extract coordinates from a CLICK action string."""
    import re

    if not action_str:
        return None

    # Match CLICK(x, y) pattern
    match = re.search(r"CLICK\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", action_str, re.IGNORECASE)
    if match:
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            # Normalize if needed (coordinates might be pixels or normalized)
            # If > 1, assume pixels and normalize assuming 1920x1200 screen
            if x > 1:
                x = x / 1920
            if y > 1:
                y = y / 1200
            return (x, y)
        except ValueError:
            return None
    return None


def evaluate_action(action_str: str, expected: dict, step: int) -> dict:
    """Evaluate if an action matches the expected action for a step."""
    result = {
        "step": step,
        "action": action_str,
        "expected_target": expected["target"],
        "expected_region": expected["region"],
        "correct": False,
        "reason": None,
    }

    if not action_str:
        result["reason"] = "No action parsed"
        return result

    # Check if it's a CLICK action
    if "CLICK" not in action_str.upper():
        result["reason"] = f"Expected CLICK, got: {action_str[:30]}"
        return result

    coords = parse_click_coordinates(action_str)
    if not coords:
        result["reason"] = "Could not parse coordinates"
        return result

    x, y = coords
    x_min, x_max = expected["x_range"]
    y_min, y_max = expected["y_range"]

    # Check if coordinates are in expected range
    if x_min <= x <= x_max and y_min <= y <= y_max:
        result["correct"] = True
        result["reason"] = f"Coordinates ({x:.3f}, {y:.3f}) in expected region"
    else:
        result["reason"] = f"Coordinates ({x:.3f}, {y:.3f}) outside expected region ({x_min}-{x_max}, {y_min}-{y_max})"

    return result


def run_multistep_experiment(
    provider: str = "anthropic",
    max_steps: int = 5,
    conditions: list[str] | None = None,
):
    """Run multi-step execution experiment.

    Args:
        provider: API provider to use.
        max_steps: Maximum number of steps to execute.
        conditions: Which conditions to run. Default: ["zero_shot", "with_demo"]
    """
    if conditions is None:
        conditions = ["zero_shot", "with_demo"]

    print("=" * 70)
    print("MULTI-STEP EXECUTION EXPERIMENT")
    print("=" * 70)
    print(f"\nProvider: {provider}")
    print(f"Max steps: {max_steps}")
    print(f"Conditions: {conditions}")
    print()

    # Verify screenshots exist
    available_steps = []
    for step, filename in STEP_SCREENSHOTS.items():
        if step > max_steps:
            continue
        path = SCREENSHOT_DIR / filename
        if path.exists():
            available_steps.append(step)
        else:
            print(f"WARNING: Screenshot not found for step {step}: {path}")

    if not available_steps:
        print("ERROR: No screenshots available. Cannot run experiment.")
        return

    print(f"Available steps: {available_steps}")
    print()

    experiment = DemoPromptExperiment(provider=provider, verbose=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "max_steps": max_steps,
        "demo": DEMO_TURN_OFF_NIGHTSHIFT,
        "conditions": {},
    }

    for condition in conditions:
        print("-" * 70)
        print(f"CONDITION: {condition.upper()}")
        print("-" * 70)

        condition_results = {
            "steps": [],
            "total_correct": 0,
            "total_attempted": 0,
        }

        for step in available_steps:
            screenshot_path = str(SCREENSHOT_DIR / STEP_SCREENSHOTS[step])
            task = STEP_TASKS[step]
            expected = EXPECTED_ACTIONS[step]

            print(f"\n  Step {step}: {expected['target']}")
            print(f"    Task: {task[:60]}...")

            # Run the appropriate condition
            if condition == "zero_shot":
                result = experiment.run_zero_shot(
                    task=task,
                    screenshot_path=screenshot_path,
                )
            elif condition == "with_demo":
                result = experiment.run_with_demo(
                    task=task,
                    screenshot_path=screenshot_path,
                    demo_text=DEMO_TURN_OFF_NIGHTSHIFT,
                )
            else:
                print(f"    Unknown condition: {condition}")
                continue

            # Evaluate the action
            evaluation = evaluate_action(result.action_parsed, expected, step)

            step_result = {
                "step": step,
                "task": task,
                "screenshot": screenshot_path,
                "action_parsed": result.action_parsed,
                "response": result.response,
                "error": result.error,
                "evaluation": evaluation,
            }

            condition_results["steps"].append(step_result)
            condition_results["total_attempted"] += 1
            if evaluation["correct"]:
                condition_results["total_correct"] += 1

            status = "✓" if evaluation["correct"] else "✗"
            print(f"    Action: {result.action_parsed}")
            print(f"    Result: {status} {evaluation['reason']}")

        # Calculate accuracy
        if condition_results["total_attempted"] > 0:
            accuracy = condition_results["total_correct"] / condition_results["total_attempted"]
            condition_results["accuracy"] = accuracy
        else:
            condition_results["accuracy"] = 0.0

        all_results["conditions"][condition] = condition_results

        print(f"\n  {condition.upper()} ACCURACY: {condition_results['total_correct']}/{condition_results['total_attempted']} ({condition_results['accuracy']*100:.1f}%)")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Condition':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 50)

    for condition, results in all_results["conditions"].items():
        print(f"{condition:<20} {results['total_correct']:<10} {results['total_attempted']:<10} {results['accuracy']*100:.1f}%")

    # Step-by-step comparison
    print("\n" + "-" * 70)
    print("STEP-BY-STEP COMPARISON")
    print("-" * 70)
    print()
    print(f"{'Step':<6} {'Target':<20} ", end="")
    for condition in conditions:
        print(f"{condition:<15}", end="")
    print()
    print("-" * (26 + 15 * len(conditions)))

    for step in available_steps:
        expected = EXPECTED_ACTIONS[step]
        print(f"{step:<6} {expected['target']:<20} ", end="")
        for condition in conditions:
            cond_results = all_results["conditions"].get(condition, {})
            step_results = [s for s in cond_results.get("steps", []) if s["step"] == step]
            if step_results:
                status = "✓" if step_results[0]["evaluation"]["correct"] else "✗"
                print(f"{status:<15}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    if "zero_shot" in all_results["conditions"] and "with_demo" in all_results["conditions"]:
        zs_acc = all_results["conditions"]["zero_shot"]["accuracy"]
        demo_acc = all_results["conditions"]["with_demo"]["accuracy"]
        delta = demo_acc - zs_acc

        print(f"Zero-shot accuracy:  {zs_acc*100:.1f}%")
        print(f"With-demo accuracy:  {demo_acc*100:.1f}%")
        print(f"Delta:               {delta*100:+.1f} percentage points")
        print()

        if delta > 0:
            print("FINDING: Demo-conditioning improves multi-step execution.")
        elif delta < 0:
            print("FINDING: Demo-conditioning hurts multi-step execution (unexpected).")
        else:
            print("FINDING: No difference between conditions.")

        # Identify where divergence happens
        print("\nDivergence analysis:")
        for step in available_steps:
            zs_steps = [s for s in all_results["conditions"]["zero_shot"]["steps"] if s["step"] == step]
            demo_steps = [s for s in all_results["conditions"]["with_demo"]["steps"] if s["step"] == step]

            if zs_steps and demo_steps:
                zs_correct = zs_steps[0]["evaluation"]["correct"]
                demo_correct = demo_steps[0]["evaluation"]["correct"]

                if demo_correct and not zs_correct:
                    print(f"  Step {step}: Demo helps (zero-shot fails, demo succeeds)")
                elif not demo_correct and zs_correct:
                    print(f"  Step {step}: Demo hurts (zero-shot succeeds, demo fails)")
                elif not demo_correct and not zs_correct:
                    print(f"  Step {step}: Both fail - potential grounding issue")

    # Save results
    output_dir = Path("openadapt_ml/experiments/demo_prompt/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"multistep_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-step execution experiment")
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="API provider (default: anthropic)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of steps to execute (default: 5)",
    )
    parser.add_argument(
        "--zero-shot-only",
        action="store_true",
        help="Only run zero-shot condition",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only run with-demo condition",
    )

    args = parser.parse_args()

    conditions = None
    if args.zero_shot_only:
        conditions = ["zero_shot"]
    elif args.demo_only:
        conditions = ["with_demo"]

    run_multistep_experiment(
        provider=args.provider,
        max_steps=args.max_steps,
        conditions=conditions,
    )
