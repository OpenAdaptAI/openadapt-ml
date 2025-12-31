#!/usr/bin/env python
"""Run demo-conditioned prompt experiment at scale (n≥30).

This script expands the original 3-test-case experiment to 30+ test cases
covering various macOS System Settings tasks. All tasks share the same
first action: click Apple menu at top-left (~0.01, 0.01).

Usage:
    uv run python scripts/run_demo_experiment_n30.py
    uv run python scripts/run_demo_experiment_n30.py --provider openai
    uv run python scripts/run_demo_experiment_n30.py --dry-run  # Show test cases only

Results are saved to: openadapt_ml/experiments/demo_prompt/results/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from openadapt_ml.experiments.demo_prompt.run_experiment import (
    DemoPromptExperiment,
)


# Hand-crafted demo for "Turn off Night Shift" on macOS
# All test tasks share the same first action: Apple menu click
DEMO_TURN_OFF_NIGHTSHIFT = """DEMONSTRATION:
Goal: Turn off Night Shift in macOS System Settings

Step 1:
  Screen: Desktop with application window visible
  Action: CLICK(0.01, 0.01)
  Result: Apple menu opened

Step 2:
  Screen: Apple menu visible with options
  Action: CLICK on "System Settings..." menu item
  Result: System Settings application opened

Step 3:
  Screen: System Settings window with sidebar
  Action: CLICK on "Displays" in the sidebar
  Result: Displays panel shown in main area

Step 4:
  Screen: Displays panel showing display settings
  Action: CLICK on "Night Shift..." button
  Result: Night Shift popup/sheet appeared

Step 5:
  Screen: Night Shift popup with Schedule dropdown
  Action: CLICK on Schedule dropdown, select "Off"
  Result: Night Shift schedule set to Off, Night Shift disabled

---"""


# 30+ test cases covering various macOS System Settings
# All share the same first action: click Apple menu
TEST_CASES = [
    # === DISPLAYS (6 cases) ===
    {"name": "displays_night_shift_on", "task": "Turn ON Night Shift in macOS System Settings", "category": "Displays"},
    {"name": "displays_night_shift_off", "task": "Turn OFF Night Shift in macOS System Settings", "category": "Displays"},
    {"name": "displays_night_shift_schedule", "task": "Set Night Shift to turn on at sunset in System Settings", "category": "Displays"},
    {"name": "displays_night_shift_temp", "task": "Adjust Night Shift color temperature to warmer setting", "category": "Displays"},
    {"name": "displays_true_tone_on", "task": "Enable True Tone display in System Settings", "category": "Displays"},
    {"name": "displays_resolution", "task": "Change display resolution in System Settings", "category": "Displays"},

    # === SOUND (5 cases) ===
    {"name": "sound_output_device", "task": "Change sound output device in System Settings", "category": "Sound"},
    {"name": "sound_input_device", "task": "Change microphone input device in System Settings", "category": "Sound"},
    {"name": "sound_alert_volume", "task": "Adjust alert sound volume in System Settings", "category": "Sound"},
    {"name": "sound_startup_sound", "task": "Disable startup sound in System Settings", "category": "Sound"},
    {"name": "sound_output_volume", "task": "Set system volume to 50% in System Settings", "category": "Sound"},

    # === NOTIFICATIONS (4 cases) ===
    {"name": "notifications_dnd", "task": "Enable Do Not Disturb in System Settings", "category": "Notifications"},
    {"name": "notifications_app", "task": "Disable notifications for Messages app in System Settings", "category": "Notifications"},
    {"name": "notifications_preview", "task": "Hide notification previews when locked in System Settings", "category": "Notifications"},
    {"name": "notifications_sound", "task": "Turn off notification sounds in System Settings", "category": "Notifications"},

    # === FOCUS (3 cases) ===
    {"name": "focus_enable", "task": "Enable Focus mode in System Settings", "category": "Focus"},
    {"name": "focus_schedule", "task": "Schedule Focus mode for work hours in System Settings", "category": "Focus"},
    {"name": "focus_share", "task": "Share Focus status across devices in System Settings", "category": "Focus"},

    # === GENERAL (4 cases) ===
    {"name": "general_about", "task": "Open About This Mac in System Settings", "category": "General"},
    {"name": "general_update", "task": "Check for software updates in System Settings", "category": "General"},
    {"name": "general_storage", "task": "View storage usage in System Settings", "category": "General"},
    {"name": "general_login_items", "task": "Manage login items in System Settings", "category": "General"},

    # === PRIVACY & SECURITY (4 cases) ===
    {"name": "privacy_location", "task": "Disable location services in System Settings", "category": "Privacy"},
    {"name": "privacy_camera", "task": "Check which apps have camera access in System Settings", "category": "Privacy"},
    {"name": "privacy_microphone", "task": "Manage microphone permissions in System Settings", "category": "Privacy"},
    {"name": "security_filevault", "task": "Check FileVault encryption status in System Settings", "category": "Security"},

    # === KEYBOARD (3 cases) ===
    {"name": "keyboard_shortcuts", "task": "Customize keyboard shortcuts in System Settings", "category": "Keyboard"},
    {"name": "keyboard_input_sources", "task": "Add a new input language in System Settings", "category": "Keyboard"},
    {"name": "keyboard_dictation", "task": "Enable dictation in System Settings", "category": "Keyboard"},

    # === TRACKPAD/MOUSE (3 cases) ===
    {"name": "trackpad_tap_click", "task": "Enable tap to click on trackpad in System Settings", "category": "Trackpad"},
    {"name": "trackpad_scroll", "task": "Change scroll direction in System Settings", "category": "Trackpad"},
    {"name": "mouse_speed", "task": "Adjust mouse tracking speed in System Settings", "category": "Mouse"},

    # === NETWORK (3 cases) ===
    {"name": "wifi_connect", "task": "Connect to a WiFi network in System Settings", "category": "Network"},
    {"name": "wifi_forget", "task": "Forget a saved WiFi network in System Settings", "category": "Network"},
    {"name": "network_dns", "task": "Change DNS settings in System Settings", "category": "Network"},

    # === BLUETOOTH (2 cases) ===
    {"name": "bluetooth_enable", "task": "Turn on Bluetooth in System Settings", "category": "Bluetooth"},
    {"name": "bluetooth_pair", "task": "Pair a new Bluetooth device in System Settings", "category": "Bluetooth"},

    # === ACCESSIBILITY (3 cases) ===
    {"name": "accessibility_zoom", "task": "Enable zoom accessibility feature in System Settings", "category": "Accessibility"},
    {"name": "accessibility_voiceover", "task": "Turn on VoiceOver in System Settings", "category": "Accessibility"},
    {"name": "accessibility_display", "task": "Increase contrast in System Settings", "category": "Accessibility"},

    # === BATTERY (2 cases) ===
    {"name": "battery_percentage", "task": "Show battery percentage in menu bar via System Settings", "category": "Battery"},
    {"name": "battery_low_power", "task": "Enable Low Power Mode in System Settings", "category": "Battery"},

    # === DESKTOP & DOCK (3 cases) ===
    {"name": "dock_size", "task": "Change Dock size in System Settings", "category": "Desktop & Dock"},
    {"name": "dock_autohide", "task": "Enable auto-hide for Dock in System Settings", "category": "Desktop & Dock"},
    {"name": "desktop_wallpaper", "task": "Change desktop wallpaper in System Settings", "category": "Desktop & Dock"},
]


def run_experiment(
    provider: str = "anthropic",
    screenshot_path: str | None = None,
    dry_run: bool = False,
    max_cases: int | None = None,
):
    """Run the expanded n≥30 experiment.

    Args:
        provider: API provider to use.
        screenshot_path: Path to screenshot for testing. If None, uses default.
        dry_run: If True, only print test cases without running.
        max_cases: Maximum number of test cases to run (for testing).
    """
    # Default to first screenshot from nightshift capture
    if screenshot_path is None:
        screenshot_path = "/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift/screenshots/capture_31807990_step_0.png"

    # Verify screenshot exists
    if not Path(screenshot_path).exists():
        print(f"ERROR: Screenshot not found: {screenshot_path}")
        print("Please provide a valid screenshot path with --screenshot")
        return

    cases_to_run = TEST_CASES[:max_cases] if max_cases else TEST_CASES

    print("=" * 70)
    print("DEMO-CONDITIONED PROMPT EXPERIMENT (n≥30)")
    print("=" * 70)
    print(f"\nProvider: {provider}")
    print(f"Screenshot: {screenshot_path}")
    print(f"Test cases: {len(cases_to_run)}")
    print()

    # Show category breakdown
    categories = {}
    for case in cases_to_run:
        cat = case.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print()

    if dry_run:
        print("\n[DRY RUN] Test cases that would be run:")
        for i, case in enumerate(cases_to_run, 1):
            print(f"  {i:2}. [{case.get('category', 'Unknown'):15}] {case['name']}")
            print(f"      Task: {case['task']}")
        print(f"\nTotal: {len(cases_to_run)} test cases")
        return

    experiment = DemoPromptExperiment(provider=provider, verbose=False)

    # Generate control text
    from openadapt_ml.experiments.demo_prompt.format_demo import (
        generate_length_matched_control,
    )
    control_text = generate_length_matched_control(DEMO_TURN_OFF_NIGHTSHIFT)

    all_results = []
    correct_counts = {"zero_shot": 0, "with_demo": 0, "control": 0}

    # Expected correct action: Apple menu click at top-left
    # Accept any x < 50 and y < 20 as correct (top-left region)
    def is_correct_action(action_str: str) -> bool:
        """Check if action is a click in the top-left region (Apple menu)."""
        if not action_str or "CLICK" not in action_str:
            return False
        try:
            # Parse CLICK(x, y) format
            import re
            match = re.search(r'CLICK\((\d+),\s*(\d+)\)', action_str)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                # Accept clicks in top-left region (x < 100, y < 50)
                return x < 100 and y < 50
        except Exception:
            pass
        return False

    for i, test_case in enumerate(cases_to_run, 1):
        print(f"\r[{i}/{len(cases_to_run)}] {test_case['name'][:40]:<40}", end="", flush=True)

        case_results = {
            "test_case": test_case,
            "results": {},
        }

        # Run zero-shot
        result_zero = experiment.run_zero_shot(
            task=test_case["task"],
            screenshot_path=screenshot_path,
        )
        case_results["results"]["zero_shot"] = {
            "action": result_zero.action_parsed,
            "correct": is_correct_action(result_zero.action_parsed),
            "error": result_zero.error,
        }
        if is_correct_action(result_zero.action_parsed):
            correct_counts["zero_shot"] += 1

        # Run with demo
        result_demo = experiment.run_with_demo(
            task=test_case["task"],
            screenshot_path=screenshot_path,
            demo_text=DEMO_TURN_OFF_NIGHTSHIFT,
        )
        case_results["results"]["with_demo"] = {
            "action": result_demo.action_parsed,
            "correct": is_correct_action(result_demo.action_parsed),
            "error": result_demo.error,
        }
        if is_correct_action(result_demo.action_parsed):
            correct_counts["with_demo"] += 1

        # Run control
        result_control = experiment.run_control(
            task=test_case["task"],
            screenshot_path=screenshot_path,
            control_text=control_text,
        )
        case_results["results"]["control"] = {
            "action": result_control.action_parsed,
            "correct": is_correct_action(result_control.action_parsed),
            "error": result_control.error,
        }
        if is_correct_action(result_control.action_parsed):
            correct_counts["control"] += 1

        all_results.append(case_results)

    print()  # Newline after progress

    # Print summary
    n = len(cases_to_run)
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Condition':<25} {'Correct':>10} {'Accuracy':>12}")
    print("-" * 50)
    print(f"{'Zero-shot':<25} {correct_counts['zero_shot']:>10}/{n} {correct_counts['zero_shot']/n*100:>10.1f}%")
    print(f"{'With Demo':<25} {correct_counts['with_demo']:>10}/{n} {correct_counts['with_demo']/n*100:>10.1f}%")
    print(f"{'Control (length-matched)':<25} {correct_counts['control']:>10}/{n} {correct_counts['control']/n*100:>10.1f}%")
    print()

    # Print by category
    print("\nBy Category:")
    print("-" * 70)
    for category in sorted(set(c.get("category", "Unknown") for c in cases_to_run)):
        cat_results = [r for r in all_results if r["test_case"].get("category") == category]
        cat_n = len(cat_results)
        cat_zero = sum(1 for r in cat_results if r["results"]["zero_shot"]["correct"])
        cat_demo = sum(1 for r in cat_results if r["results"]["with_demo"]["correct"])
        print(f"  {category:<20} Zero: {cat_zero}/{cat_n} ({cat_zero/cat_n*100:.0f}%)  Demo: {cat_demo}/{cat_n} ({cat_demo/cat_n*100:.0f}%)")

    # Calculate improvement
    zero_acc = correct_counts["zero_shot"] / n
    demo_acc = correct_counts["with_demo"] / n
    improvement = demo_acc - zero_acc

    print()
    print("=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    print(f"\nDemo-conditioning improvement: +{improvement*100:.1f} percentage points")
    print(f"  Zero-shot: {zero_acc*100:.1f}%")
    print(f"  With demo: {demo_acc*100:.1f}%")

    if improvement > 0:
        print(f"\n✓ Demo-conditioning IMPROVES first-action accuracy")
    elif improvement < 0:
        print(f"\n✗ Demo-conditioning DECREASES first-action accuracy")
    else:
        print(f"\n= Demo-conditioning has NO EFFECT on first-action accuracy")

    # Save results
    output_dir = Path("openadapt_ml/experiments/demo_prompt/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_n30_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "screenshot": screenshot_path,
            "demo": DEMO_TURN_OFF_NIGHTSHIFT,
            "n_test_cases": len(cases_to_run),
            "summary": {
                "zero_shot_accuracy": correct_counts["zero_shot"] / n,
                "with_demo_accuracy": correct_counts["with_demo"] / n,
                "control_accuracy": correct_counts["control"] / n,
                "improvement": improvement,
            },
            "correct_counts": correct_counts,
            "test_cases": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run demo-conditioned experiment at scale (n≥30)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="API provider to use",
    )
    parser.add_argument(
        "--screenshot",
        help="Path to test screenshot",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print test cases without running",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Maximum number of test cases to run (for testing)",
    )

    args = parser.parse_args()
    run_experiment(
        provider=args.provider,
        screenshot_path=args.screenshot,
        dry_run=args.dry_run,
        max_cases=args.max_cases,
    )
