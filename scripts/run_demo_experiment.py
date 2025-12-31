#!/usr/bin/env python
"""Run the demo-conditioned prompt experiment.

This script tests whether including a demonstration improves VLM performance
on similar GUI tasks.

Usage:
    uv run python scripts/run_demo_experiment.py

Results are saved to: openadapt_ml/experiments/demo_prompt/results/
"""

import json
from datetime import datetime
from pathlib import Path

from openadapt_ml.experiments.demo_prompt.run_experiment import (
    DemoPromptExperiment,
)


# Hand-crafted demo for "Turn off Night Shift" on macOS
# This represents what the human actually did
DEMO_TURN_OFF_NIGHTSHIFT = """DEMONSTRATION:
Goal: Turn off Night Shift in macOS System Settings

The following shows the step-by-step procedure:

Step 1:
  [Screen: Desktop with Terminal window visible]
  [Action: CLICK(0.01, 0.01) - Click Apple menu icon in top-left]
  [Result: Apple menu dropdown opened]

Step 2:
  [Screen: Apple menu visible with options]
  [Action: CLICK on "System Settings..." menu item]
  [Result: System Settings application opened]

Step 3:
  [Screen: System Settings window with sidebar]
  [Action: CLICK on "Displays" in the sidebar]
  [Result: Displays panel shown in main area]

Step 4:
  [Screen: Displays panel showing display settings]
  [Action: CLICK on "Night Shift..." button]
  [Result: Night Shift popup/sheet appeared]

Step 5:
  [Screen: Night Shift popup with Schedule dropdown]
  [Action: CLICK on Schedule dropdown, select "Off"]
  [Result: Night Shift schedule set to Off, Night Shift disabled]

---"""


# Test cases: (test_task, expected_similarity_to_demo)
TEST_CASES = [
    {
        "name": "near_toggle",
        "task": "Turn ON Night Shift in macOS System Settings",
        "similarity": "near",
        "notes": "Same procedure, just toggle to opposite state",
    },
    {
        "name": "medium_same_panel",
        "task": "Adjust Night Shift color temperature to warmer setting",
        "similarity": "medium",
        "notes": "Same navigation path, different final action",
    },
    {
        "name": "far_different_setting",
        "task": "Turn on True Tone display in macOS System Settings",
        "similarity": "far",
        "notes": "Same app but different panel (True Tone is in Displays)",
    },
]


def run_experiment(
    provider: str = "anthropic",
    screenshot_path: str | None = None,
):
    """Run the full experiment.

    Args:
        provider: API provider to use.
        screenshot_path: Path to screenshot for testing. If None, uses default.
    """
    # Default to first screenshot from nightshift capture
    if screenshot_path is None:
        screenshot_path = "/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift/screenshots/capture_31807990_step_0.png"

    # Verify screenshot exists
    if not Path(screenshot_path).exists():
        print(f"ERROR: Screenshot not found: {screenshot_path}")
        return

    print("=" * 70)
    print("DEMO-CONDITIONED PROMPT EXPERIMENT")
    print("=" * 70)
    print(f"\nProvider: {provider}")
    print(f"Screenshot: {screenshot_path}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    experiment = DemoPromptExperiment(provider=provider, verbose=True)

    # Generate control text
    from openadapt_ml.experiments.demo_prompt.format_demo import (
        generate_length_matched_control,
    )
    control_text = generate_length_matched_control(DEMO_TURN_OFF_NIGHTSHIFT)

    all_results = []

    for i, test_case in enumerate(TEST_CASES, 1):
        print("-" * 70)
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"Task: {test_case['task']}")
        print(f"Similarity: {test_case['similarity']}")
        print("-" * 70)

        case_results = {
            "test_case": test_case,
            "results": {},
        }

        # Run zero-shot
        print("\n[1/3] Zero-shot...")
        result_zero = experiment.run_zero_shot(
            task=test_case["task"],
            screenshot_path=screenshot_path,
        )
        case_results["results"]["zero_shot"] = {
            "action": result_zero.action_parsed,
            "response": result_zero.response,
            "error": result_zero.error,
        }
        print(f"      Action: {result_zero.action_parsed}")

        # Run with demo
        print("\n[2/3] With demo...")
        result_demo = experiment.run_with_demo(
            task=test_case["task"],
            screenshot_path=screenshot_path,
            demo_text=DEMO_TURN_OFF_NIGHTSHIFT,
        )
        case_results["results"]["with_demo"] = {
            "action": result_demo.action_parsed,
            "response": result_demo.response,
            "error": result_demo.error,
        }
        print(f"      Action: {result_demo.action_parsed}")

        # Run control
        print("\n[3/3] Control (length-matched)...")
        result_control = experiment.run_control(
            task=test_case["task"],
            screenshot_path=screenshot_path,
            control_text=control_text,
        )
        case_results["results"]["control"] = {
            "action": result_control.action_parsed,
            "response": result_control.response,
            "error": result_control.error,
        }
        print(f"      Action: {result_control.action_parsed}")

        all_results.append(case_results)
        print()

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Task':<40} {'Zero-shot':<20} {'With Demo':<20} {'Control':<20}")
    print("-" * 100)

    for result in all_results:
        task_name = result["test_case"]["name"]
        zero = result["results"]["zero_shot"]["action"] or "ERROR"
        demo = result["results"]["with_demo"]["action"] or "ERROR"
        ctrl = result["results"]["control"]["action"] or "ERROR"

        # Truncate for display
        zero = zero[:18] + ".." if len(zero) > 20 else zero
        demo = demo[:18] + ".." if len(demo) > 20 else demo
        ctrl = ctrl[:18] + ".." if len(ctrl) > 20 else ctrl

        print(f"{task_name:<40} {zero:<20} {demo:<20} {ctrl:<20}")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("Compare the actions across conditions:")
    print("- If 'With Demo' produces better/more relevant actions than 'Zero-shot',")
    print("  the demonstration is helping.")
    print("- If 'With Demo' ≈ 'Control', the benefit is just from longer context,")
    print("  not from the demonstration content.")
    print()

    # Save results
    output_dir = Path("openadapt_ml/experiments/demo_prompt/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "provider": provider,
            "screenshot": screenshot_path,
            "demo": DEMO_TURN_OFF_NIGHTSHIFT,
            "test_cases": all_results,
        }, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
    )
    parser.add_argument(
        "--screenshot",
        help="Path to test screenshot",
    )

    args = parser.parse_args()
    run_experiment(provider=args.provider, screenshot_path=args.screenshot)
