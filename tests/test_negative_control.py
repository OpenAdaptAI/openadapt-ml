"""Negative control test for demo-conditioned prompting experiment.

Tests that IRRELEVANT demos (file management task) do NOT improve performance
on Night Shift tasks, proving that retrieval quality matters.

Expected outcome: irrelevant demo should perform same as or worse than zero-shot.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from openadapt_ml.experiments.demo_prompt.run_experiment import (
    DemoPromptExperiment,
    ExperimentResult,
)


# Irrelevant demo - completely unrelated file management task
IRRELEVANT_DEMO = """DEMONSTRATION:
Goal: Create a new folder on Desktop

Step 1:
  Screen: Desktop
  Action: RIGHT_CLICK(0.5, 0.5)
  Result: Context menu appeared

Step 2:
  Screen: Context menu visible
  Action: CLICK on "New Folder"
  Result: New folder created
---"""


# Test cases
TEST_CASES = [
    {
        "name": "near_toggle",
        "task": "Turn ON Night Shift in macOS System Settings",
        "screenshot": "/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift/screenshots/capture_31807990_step_0.png",
    },
    {
        "name": "medium_same_panel",
        "task": "Adjust Night Shift color temperature to warmer setting",
        "screenshot": "/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift/screenshots/capture_31807990_step_0.png",
    },
]


def run_negative_control_test(
    provider: str = "anthropic",
    output_dir: str = "negative_control_results",
) -> dict[str, any]:
    """Run negative control test.

    Args:
        provider: API provider ("anthropic" or "openai").
        output_dir: Output directory for results.

    Returns:
        Dict with results and summary.
    """
    print("=" * 80)
    print("NEGATIVE CONTROL TEST: Irrelevant Demo vs Zero-Shot")
    print("=" * 80)
    print(f"\nProvider: {provider}")
    print(f"Output directory: {output_dir}")
    print(f"\nIrrelevant demo:\n{IRRELEVANT_DEMO}")
    print("\n" + "=" * 80)

    # Initialize experiment
    experiment = DemoPromptExperiment(provider=provider, verbose=True)

    # Store all results
    all_results = {}

    # Run each test case
    for i, test_case in enumerate(TEST_CASES, 1):
        name = test_case["name"]
        task = test_case["task"]
        screenshot = test_case["screenshot"]

        print(f"\n[{i}/{len(TEST_CASES)}] Test case: {name}")
        print(f"Task: {task}")
        print(f"Screenshot: {screenshot}")
        print("-" * 80)

        # Run zero-shot
        print("\nCondition 1/2: ZERO-SHOT")
        zero_shot_result = experiment.run_zero_shot(task, screenshot)

        # Run with irrelevant demo
        print("\nCondition 2/2: WITH IRRELEVANT DEMO")
        irrelevant_result = experiment.run_with_demo(
            task=task,
            screenshot_path=screenshot,
            demo_text=IRRELEVANT_DEMO,
            demo_screenshots=None,  # No demo screenshots
        )

        # Store results
        all_results[name] = {
            "task": task,
            "zero_shot": zero_shot_result,
            "with_irrelevant_demo": irrelevant_result,
        }

        # Print comparison
        print("\n" + "=" * 80)
        print(f"RESULTS FOR: {name}")
        print("=" * 80)
        print(f"\nZERO-SHOT:")
        print(f"  Action: {zero_shot_result.action_parsed}")
        print(f"  Response preview: {zero_shot_result.response[:300]}...")
        if zero_shot_result.error:
            print(f"  Error: {zero_shot_result.error}")

        print(f"\nWITH IRRELEVANT DEMO:")
        print(f"  Action: {irrelevant_result.action_parsed}")
        print(f"  Response preview: {irrelevant_result.response[:300]}...")
        if irrelevant_result.error:
            print(f"  Error: {irrelevant_result.error}")

        print()

    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = {
        "test_name": "negative_control_irrelevant_demo",
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "num_test_cases": len(TEST_CASES),
        "results": [],
    }

    for name, results in all_results.items():
        zero_shot = results["zero_shot"]
        irrelevant = results["with_irrelevant_demo"]

        # Compare actions
        same_action = (
            zero_shot.action_parsed == irrelevant.action_parsed
            if zero_shot.action_parsed and irrelevant.action_parsed
            else None
        )

        case_summary = {
            "name": name,
            "task": results["task"],
            "zero_shot_action": zero_shot.action_parsed,
            "irrelevant_demo_action": irrelevant.action_parsed,
            "same_action": same_action,
            "zero_shot_error": zero_shot.error,
            "irrelevant_demo_error": irrelevant.error,
        }

        summary["results"].append(case_summary)

        print(f"\nTest case: {name}")
        print(f"  Task: {results['task']}")
        print(f"  Zero-shot action: {zero_shot.action_parsed}")
        print(f"  Irrelevant demo action: {irrelevant.action_parsed}")
        print(f"  Same action? {same_action}")
        if zero_shot.error or irrelevant.error:
            print(f"  Errors: zero_shot={zero_shot.error}, irrelevant={irrelevant.error}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_path / f"negative_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": summary,
                "detailed_results": {
                    name: {
                        "task": results["task"],
                        "zero_shot": {
                            "condition": results["zero_shot"].condition,
                            "action_parsed": results["zero_shot"].action_parsed,
                            "response": results["zero_shot"].response,
                            "error": results["zero_shot"].error,
                            "timestamp": results["zero_shot"].timestamp,
                        },
                        "with_irrelevant_demo": {
                            "condition": results["with_irrelevant_demo"].condition,
                            "action_parsed": results["with_irrelevant_demo"].action_parsed,
                            "response": results["with_irrelevant_demo"].response,
                            "error": results["with_irrelevant_demo"].error,
                            "timestamp": results["with_irrelevant_demo"].timestamp,
                        },
                    }
                    for name, results in all_results.items()
                },
            },
            f,
            indent=2,
        )

    print(f"\n\nResults saved to: {results_file}")

    # Print conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nExpected outcome: Irrelevant demo should NOT improve performance.")
    print("This proves that retrieval quality matters - wrong demos don't help.")

    same_count = sum(1 for r in summary["results"] if r.get("same_action") is True)
    different_count = sum(1 for r in summary["results"] if r.get("same_action") is False)

    print(f"\nActions comparison:")
    print(f"  Same action: {same_count}/{len(TEST_CASES)}")
    print(f"  Different action: {different_count}/{len(TEST_CASES)}")

    if same_count == len(TEST_CASES):
        print("\nResult: Irrelevant demo produced SAME actions as zero-shot.")
        print("Interpretation: Demo was ignored/unhelpful (expected behavior).")
    elif different_count == len(TEST_CASES):
        print("\nResult: Irrelevant demo produced DIFFERENT actions from zero-shot.")
        print("Interpretation: Demo confused the model (negative transfer).")
    else:
        print("\nResult: Mixed - some same, some different.")
        print("Interpretation: Inconsistent effect of irrelevant demo.")

    print("\n" + "=" * 80)

    return {
        "summary": summary,
        "all_results": all_results,
        "output_file": str(results_file),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run negative control test for demo-conditioned prompting"
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="API provider (default: anthropic)",
    )
    parser.add_argument(
        "--output",
        default="negative_control_results",
        help="Output directory for results (default: negative_control_results)",
    )

    args = parser.parse_args()

    run_negative_control_test(
        provider=args.provider,
        output_dir=args.output,
    )
