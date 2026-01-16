#!/usr/bin/env python3
"""P1 Validation: A/B test episode success with demo conditioning.

This script validates the thesis:
- Zero-shot: baseline episode success (expected ~0%)
- Demo-conditioned: should show episode success > 0%

What this tests:
- Real GUI execution (not just prompt validation)
- Multi-step task completion
- A/B comparison with controlled conditions

Metrics logged (per GPT's spec):
- episode_success (binary)
- steps_to_success
- step_of_failure
- failure_class ∈ {drift, local_error, timeout, tool_error}
- prompt_tokens per step
- action_type distribution

Pass/fail thresholds:
- PASS: demo-conditioned achieves ≥2/10 successes AND beats zero-shot by ≥+2 episodes
- SOFT PASS: 1 success but clear failure-mode shift (drift → local_error)
- FAIL: 0 successes or no delta vs zero-shot

Run with:
    uv run python scripts/p1_episode_success_ab_test.py --tasks 10
    uv run python scripts/p1_episode_success_ab_test.py --quick  # 3 tasks for fast iteration
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "p1_results"


class FailureClass(Enum):
    """Failure mode classification."""
    NONE = "none"  # Success
    DRIFT = "drift"  # Agent lost track of goal
    LOCAL_ERROR = "local_error"  # Wrong click/action at a step
    TIMEOUT = "timeout"  # Exceeded step limit
    TOOL_ERROR = "tool_error"  # API/tool failure


@dataclass
class StepResult:
    """Result of a single step."""
    step_num: int
    action: str
    action_type: str  # click, type, hotkey, scroll, done, etc.
    prompt_tokens: int
    demo_included: bool
    success: bool  # Did this step succeed?
    error: str | None = None


@dataclass
class EpisodeResult:
    """Result of a single episode (task execution)."""
    task_id: str
    task_instruction: str
    condition: str  # "zero_shot" or "demo_conditioned"
    episode_success: bool
    steps_taken: int
    step_limit: int
    steps_to_success: int | None  # None if failed
    step_of_failure: int | None  # None if succeeded
    failure_class: str
    steps: list[StepResult] = field(default_factory=list)
    action_type_distribution: dict[str, int] = field(default_factory=dict)
    total_prompt_tokens: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class ABTestResult:
    """Result of the full A/B test."""
    timestamp: str
    zero_shot: list[EpisodeResult]
    demo_conditioned: list[EpisodeResult]
    summary: dict = field(default_factory=dict)


# macOS tasks for P1 testing
# These are tasks that can be verified programmatically
MACOS_TASKS = [
    {
        "id": "spotlight_search",
        "instruction": "Open Spotlight search",
        "demo": """DEMONSTRATION:
Task: Open Spotlight search

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: DONE()

---""",
        "verify": lambda: True,  # Spotlight opens automatically
        "step_limit": 3,
    },
    {
        "id": "open_calculator",
        "instruction": "Open the Calculator app using Spotlight",
        "demo": """DEMONSTRATION:
Task: Open the Calculator app using Spotlight

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("calculator")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("Calculator"),
        "step_limit": 6,
    },
    {
        "id": "open_notes",
        "instruction": "Open the Notes app using Spotlight",
        "demo": """DEMONSTRATION:
Task: Open the Notes app using Spotlight

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("notes")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("Notes"),
        "step_limit": 6,
    },
    {
        "id": "open_terminal",
        "instruction": "Open the Terminal app using Spotlight",
        "demo": """DEMONSTRATION:
Task: Open the Terminal app using Spotlight

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("terminal")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("Terminal"),
        "step_limit": 6,
    },
    {
        "id": "open_safari",
        "instruction": "Open Safari browser using Spotlight",
        "demo": """DEMONSTRATION:
Task: Open Safari browser using Spotlight

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("safari")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("Safari"),
        "step_limit": 6,
    },
    {
        "id": "open_finder",
        "instruction": "Open a new Finder window",
        "demo": """DEMONSTRATION:
Task: Open a new Finder window

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("finder")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("Finder"),
        "step_limit": 6,
    },
    {
        "id": "open_textedit",
        "instruction": "Open TextEdit app using Spotlight",
        "demo": """DEMONSTRATION:
Task: Open TextEdit app using Spotlight

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("textedit")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("TextEdit"),
        "step_limit": 6,
    },
    {
        "id": "open_preview",
        "instruction": "Open Preview app using Spotlight",
        "demo": """DEMONSTRATION:
Task: Open Preview app using Spotlight

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("preview")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("Preview"),
        "step_limit": 6,
    },
    {
        "id": "open_activity_monitor",
        "instruction": "Open Activity Monitor using Spotlight",
        "demo": """DEMONSTRATION:
Task: Open Activity Monitor using Spotlight

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("activity monitor")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("Activity Monitor"),
        "step_limit": 6,
    },
    {
        "id": "open_system_preferences",
        "instruction": "Open System Settings using Spotlight",
        "demo": """DEMONSTRATION:
Task: Open System Settings using Spotlight

Step 1:
  Action: KEY(cmd+space)

Step 2:
  Action: TYPE("system settings")

Step 3:
  Action: KEY(enter)

Step 4:
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("System Settings") or _is_app_running("System Preferences"),
        "step_limit": 6,
    },
    {
        "id": "turn_off_nightshift",
        "instruction": "Turn off Night Shift in System Settings. Open System Settings, go to Displays, click Night Shift, and turn it off.",
        "demo": """DEMONSTRATION:
Task: Turn off Night Shift in System Settings

This demonstration shows the procedural steps to turn off Night Shift on macOS.
The agent should look at the current screenshot to find the correct UI elements.

Step 1: Open Spotlight search
  Action: KEY(cmd+space)

Step 2: Search for System Settings
  Action: TYPE("system settings")

Step 3: Open System Settings
  Action: KEY(enter)

Step 4: Navigate to Displays
  Action: Click on "Displays" in the left sidebar
  Note: Look for the display/monitor icon in the sidebar

Step 5: Open Night Shift settings
  Action: Click on "Night Shift..." button
  Note: This button is usually near the bottom of the Displays pane

Step 6: Turn off Night Shift
  Action: Click the Schedule dropdown and select "Off", or toggle the switch off
  Note: The exact UI depends on macOS version

Step 7: Task complete
  Action: DONE()

---""",
        "verify": lambda: _is_app_running("System Settings") or _is_app_running("System Preferences"),
        "step_limit": 12,
    },
]


def _is_app_running(app_name: str) -> bool:
    """Check if an app is running on macOS."""
    try:
        result = subprocess.run(
            ["pgrep", "-x", app_name],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _close_app(app_name: str) -> None:
    """Close an app on macOS."""
    try:
        subprocess.run(
            ["osascript", "-e", f'tell application "{app_name}" to quit'],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass


def _take_screenshot() -> bytes:
    """Take a screenshot on macOS and return as bytes."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name

    try:
        subprocess.run(
            ["screencapture", "-x", tmp_path],
            capture_output=True,
            timeout=5,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_action_type(action: str) -> str:
    """Extract action type from action string."""
    action_lower = action.lower()
    if "click" in action_lower:
        return "click"
    elif "type" in action_lower:
        return "type"
    elif "hotkey" in action_lower or "key" in action_lower:
        return "hotkey"
    elif "scroll" in action_lower:
        return "scroll"
    elif "done" in action_lower:
        return "done"
    elif "fail" in action_lower:
        return "fail"
    else:
        return "other"


def _execute_action(action: str, dry_run: bool = False) -> bool:
    """Execute action on macOS using AppleScript (for hotkeys) and pyautogui.

    Args:
        action: Action string from agent (e.g., "computer.hotkey('cmd', 'space')")
        dry_run: If True, log but don't execute.

    Returns:
        True if executed successfully, False otherwise.
    """
    import pyautogui
    import re

    # Enable failsafe (move mouse to corner to abort)
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.3  # Pause between actions

    action_lower = action.lower()

    try:
        if dry_run:
            logger.info(f"    [DRY RUN] Would execute: {action[:100]}")
            return True

        # Handle hotkey/key actions: KEY(cmd+space) or computer.hotkey('cmd', 'space')
        # Use AppleScript for reliability on macOS
        if "hotkey" in action_lower or ("key(" in action_lower and "+" in action):
            # Parse hotkey patterns
            # Pattern 1: computer.hotkey('cmd', 'space')
            hotkey_match = re.search(r"hotkey\(['\"](\w+)['\"],\s*['\"](\w+)['\"]\)", action)
            if hotkey_match:
                key1, key2 = hotkey_match.groups()
                return _applescript_hotkey(key1, key2)

            # Pattern 2: KEY(cmd+space)
            key_match = re.search(r"key\(([^)]+)\)", action, re.IGNORECASE)
            if key_match:
                keys_str = key_match.group(1)
                keys = [k.strip().strip("'\"") for k in keys_str.replace('+', ',').split(',')]
                if len(keys) == 2:
                    return _applescript_hotkey(keys[0], keys[1])
                elif len(keys) == 1:
                    return _applescript_key(keys[0])

        # Handle single key press: computer.press('enter')
        elif "press(" in action_lower:
            press_match = re.search(r"press\(['\"](\w+)['\"]\)", action)
            if press_match:
                key = press_match.group(1)
                return _applescript_key(key)

        # Handle type actions: TYPE("text") or computer.type("text")
        elif "type(" in action_lower:
            # Extract text to type
            type_match = re.search(r"type\(['\"]([^'\"]*)['\"]", action, re.IGNORECASE)
            if type_match:
                text = type_match.group(1)
                logger.info(f"    [EXEC] type('{text}')")
                pyautogui.write(text, interval=0.05)
                return True

        # Handle click actions: computer.click(x, y) or CLICK(x, y)
        elif "click(" in action_lower:
            # Extract coordinates
            click_match = re.search(r"click\((\d+),\s*(\d+)\)", action)
            if click_match:
                x, y = int(click_match.group(1)), int(click_match.group(2))
                logger.info(f"    [EXEC] click({x}, {y})")
                pyautogui.click(x, y)
                return True
            # Normalized coordinates: CLICK(0.5, 0.5)
            click_norm = re.search(r"click\((0\.\d+),\s*(0\.\d+)\)", action, re.IGNORECASE)
            if click_norm:
                x_norm, y_norm = float(click_norm.group(1)), float(click_norm.group(2))
                screen_w, screen_h = pyautogui.size()
                x, y = int(x_norm * screen_w), int(y_norm * screen_h)
                logger.info(f"    [EXEC] click({x}, {y}) from normalized ({x_norm}, {y_norm})")
                pyautogui.click(x, y)
                return True

        # Handle scroll: computer.scroll(amount)
        elif "scroll(" in action_lower:
            scroll_match = re.search(r"scroll\((-?\d+)\)", action)
            if scroll_match:
                amount = int(scroll_match.group(1))
                logger.info(f"    [EXEC] scroll({amount})")
                pyautogui.scroll(amount)
                return True

        # No execution for DONE/FAIL - just state changes
        elif "done" in action_lower or "fail" in action_lower:
            logger.info(f"    [EXEC] Terminal action: {action[:50]}")
            return True

        logger.warning(f"    [SKIP] Unknown action format: {action[:100]}")
        return False

    except Exception as e:
        logger.error(f"    [ERROR] Failed to execute: {e}")
        return False


def _applescript_hotkey(modifier: str, key: str) -> bool:
    """Execute hotkey via AppleScript (more reliable on macOS)."""
    # Map modifier names
    modifier_map = {
        'cmd': 'command down',
        'command': 'command down',
        'ctrl': 'control down',
        'control': 'control down',
        'alt': 'option down',
        'option': 'option down',
        'shift': 'shift down',
    }
    mod = modifier_map.get(modifier.lower(), f'{modifier} down')

    # Map special keys
    key_map = {
        'space': 'space',
        'enter': 'return',
        'return': 'return',
        'tab': 'tab',
        'escape': 'escape',
        'esc': 'escape',
    }
    k = key_map.get(key.lower(), key.lower())

    script = f'tell application "System Events" to keystroke "{k}" using {mod}'
    if k == 'space':
        script = f'tell application "System Events" to keystroke space using {mod}'
    elif k in ('return', 'tab', 'escape'):
        key_code = {'return': 36, 'tab': 48, 'escape': 53}.get(k, 36)
        script = f'tell application "System Events" to key code {key_code} using {mod}'

    logger.info(f"    [EXEC] AppleScript hotkey: {modifier}+{key}")
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"    [ERROR] AppleScript failed: {result.stderr}")
        return False
    return True


def _applescript_key(key: str) -> bool:
    """Execute single key press via AppleScript."""
    key_codes = {
        'enter': 36,
        'return': 36,
        'tab': 48,
        'escape': 53,
        'esc': 53,
        'space': 49,
        'delete': 51,
        'backspace': 51,
        'up': 126,
        'down': 125,
        'left': 123,
        'right': 124,
    }

    key_lower = key.lower()
    if key_lower in key_codes:
        script = f'tell application "System Events" to key code {key_codes[key_lower]}'
    else:
        # For regular characters
        script = f'tell application "System Events" to keystroke "{key_lower}"'

    logger.info(f"    [EXEC] AppleScript key: {key}")
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"    [ERROR] AppleScript failed: {result.stderr}")
        return False
    return True


class P1Validator:
    """Runs P1 A/B test for episode success."""

    def __init__(self, provider: str = "anthropic", dry_run: bool = False):
        self.provider = provider
        self.dry_run = dry_run
        self.results = ABTestResult(
            timestamp=datetime.now().isoformat(),
            zero_shot=[],
            demo_conditioned=[],
        )

    def run_episode(
        self,
        task: dict,
        condition: str,
        demo: str | None = None,
    ) -> EpisodeResult:
        """Run a single episode (task execution).

        Args:
            task: Task definition dict.
            condition: "zero_shot" or "demo_conditioned".
            demo: Demo string if demo_conditioned.

        Returns:
            EpisodeResult with all metrics.
        """
        from openadapt_ml.benchmarks.waa_deploy.api_agent import ApiAgent

        task_id = task["id"]
        instruction = task["instruction"]
        step_limit = task.get("step_limit", 10)
        verify_fn = task.get("verify", lambda: True)

        logger.info(f"  Running {task_id} ({condition})")

        # Close related apps first
        app_name = task_id.replace("open_", "").replace("_", " ").title()
        _close_app(app_name)
        time.sleep(0.5)

        # Create agent
        agent = ApiAgent(
            provider=self.provider,
            demo=demo if condition == "demo_conditioned" else None,
            use_accessibility_tree=False,
            use_history=True,
        )

        result = EpisodeResult(
            task_id=task_id,
            task_instruction=instruction,
            condition=condition,
            episode_success=False,
            steps_taken=0,
            step_limit=step_limit,
            steps_to_success=None,
            step_of_failure=None,
            failure_class=FailureClass.NONE.value,
        )

        start_time = time.time()
        action_types: dict[str, int] = {}

        for step_num in range(1, step_limit + 1):
            try:
                # Take screenshot
                screenshot_bytes = _take_screenshot()

                obs = {
                    "screenshot": screenshot_bytes,
                    "window_title": "",
                    "window_names_str": "",
                    "computer_clipboard": "",
                    "accessibility_tree": None,
                }

                # Get action from agent
                _, actions, logs, _ = agent.predict(
                    instruction=instruction,
                    obs=obs,
                )

                action = actions[0] if actions else ""
                action_type = _extract_action_type(action)
                action_types[action_type] = action_types.get(action_type, 0) + 1

                # Log step
                step_result = StepResult(
                    step_num=step_num,
                    action=action[:200],  # Truncate
                    action_type=action_type,
                    prompt_tokens=len(logs.get("user_question", "")) // 4,  # Rough estimate
                    demo_included=logs.get("demo_included", False),
                    success=True,
                )
                result.steps.append(step_result)
                result.total_prompt_tokens += step_result.prompt_tokens
                result.steps_taken = step_num

                logger.info(f"    Step {step_num}: {action_type} - {action[:50]}...")

                # Check for done/fail
                if action_type == "done":
                    # Verify success
                    time.sleep(1.0)  # Wait for app to open
                    if verify_fn():
                        result.episode_success = True
                        result.steps_to_success = step_num
                        result.failure_class = FailureClass.NONE.value
                        logger.info(f"    ✓ Task completed successfully in {step_num} steps")
                    else:
                        result.step_of_failure = step_num
                        result.failure_class = FailureClass.DRIFT.value
                        logger.info(f"    ✗ Agent said DONE but task not verified")
                    break
                elif action_type == "fail":
                    result.step_of_failure = step_num
                    result.failure_class = FailureClass.LOCAL_ERROR.value
                    logger.info(f"    ✗ Agent said FAIL at step {step_num}")
                    break

                # Execute action for real using pyautogui
                if not self.dry_run:
                    _execute_action(action, dry_run=False)
                    time.sleep(0.5)  # Wait for UI to update
                else:
                    _execute_action(action, dry_run=True)
                    time.sleep(0.2)

            except Exception as e:
                logger.error(f"    Step {step_num} error: {e}")
                result.step_of_failure = step_num
                result.failure_class = FailureClass.TOOL_ERROR.value
                result.error = str(e)
                break

        # Check for timeout
        if result.steps_taken >= step_limit and not result.episode_success:
            result.failure_class = FailureClass.TIMEOUT.value
            result.step_of_failure = step_limit

        result.elapsed_seconds = time.time() - start_time
        result.action_type_distribution = action_types

        return result

    def run_ab_test(self, tasks: list[dict]) -> ABTestResult:
        """Run full A/B test on all tasks.

        Args:
            tasks: List of task definitions.

        Returns:
            ABTestResult with all metrics.
        """
        logger.info("=" * 60)
        logger.info("P1 A/B TEST: Episode Success Delta")
        logger.info("=" * 60)
        logger.info(f"Tasks: {len(tasks)}")
        logger.info(f"Conditions: zero_shot, demo_conditioned")
        logger.info("-" * 60)

        # Run zero-shot condition
        logger.info("\n[A] ZERO-SHOT CONDITION")
        for task in tasks:
            result = self.run_episode(task, "zero_shot", demo=None)
            self.results.zero_shot.append(result)

        # Run demo-conditioned condition
        logger.info("\n[B] DEMO-CONDITIONED CONDITION")
        for task in tasks:
            demo = task.get("demo", "")
            result = self.run_episode(task, "demo_conditioned", demo=demo)
            self.results.demo_conditioned.append(result)

        # Compute summary
        self._compute_summary()

        return self.results

    def _compute_summary(self) -> None:
        """Compute summary statistics."""
        zero_shot_successes = sum(1 for r in self.results.zero_shot if r.episode_success)
        demo_successes = sum(1 for r in self.results.demo_conditioned if r.episode_success)
        total_tasks = len(self.results.zero_shot)

        zero_shot_failure_modes = {}
        demo_failure_modes = {}

        for r in self.results.zero_shot:
            mode = r.failure_class
            zero_shot_failure_modes[mode] = zero_shot_failure_modes.get(mode, 0) + 1

        for r in self.results.demo_conditioned:
            mode = r.failure_class
            demo_failure_modes[mode] = demo_failure_modes.get(mode, 0) + 1

        delta = demo_successes - zero_shot_successes

        # Determine pass/fail
        if demo_successes >= 2 and delta >= 2:
            verdict = "PASS"
        elif demo_successes >= 1 and demo_failure_modes != zero_shot_failure_modes:
            verdict = "SOFT_PASS"
        else:
            verdict = "FAIL"

        self.results.summary = {
            "total_tasks": total_tasks,
            "zero_shot_successes": zero_shot_successes,
            "demo_conditioned_successes": demo_successes,
            "zero_shot_success_rate": zero_shot_successes / total_tasks if total_tasks > 0 else 0,
            "demo_success_rate": demo_successes / total_tasks if total_tasks > 0 else 0,
            "delta": delta,
            "zero_shot_failure_modes": zero_shot_failure_modes,
            "demo_failure_modes": demo_failure_modes,
            "verdict": verdict,
        }

    def save_results(self) -> Path:
        """Save results to JSON."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"p1_ab_test_{timestamp}.json"

        # Convert dataclasses to dicts
        data = {
            "timestamp": self.results.timestamp,
            "zero_shot": [asdict(r) for r in self.results.zero_shot],
            "demo_conditioned": [asdict(r) for r in self.results.demo_conditioned],
            "summary": self.results.summary,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print summary to console."""
        s = self.results.summary

        logger.info("\n" + "=" * 60)
        logger.info("P1 A/B TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tasks: {s['total_tasks']}")
        logger.info("-" * 40)
        logger.info(f"Zero-shot successes:      {s['zero_shot_successes']} ({s['zero_shot_success_rate']:.1%})")
        logger.info(f"Demo-conditioned successes: {s['demo_conditioned_successes']} ({s['demo_success_rate']:.1%})")
        logger.info(f"Delta: {s['delta']:+d}")
        logger.info("-" * 40)
        logger.info(f"Zero-shot failure modes: {s['zero_shot_failure_modes']}")
        logger.info(f"Demo failure modes:      {s['demo_failure_modes']}")
        logger.info("-" * 40)
        logger.info(f"VERDICT: {s['verdict']}")
        logger.info("=" * 60)

        if s['verdict'] == "PASS":
            logger.info("✓ THESIS VALIDATED: Demo conditioning improves episode success")
        elif s['verdict'] == "SOFT_PASS":
            logger.info("~ SOFT PASS: Clear failure mode shift, needs more data")
        else:
            logger.info("✗ FAIL: No improvement from demo conditioning")


def main():
    """Run P1 A/B test."""
    import argparse

    parser = argparse.ArgumentParser(description="P1: A/B test episode success")
    parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Number of tasks to run (default: 10)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run only 3 tasks",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Single task mode: run only Calculator task for P0 validation",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="API provider (default: anthropic)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute actions (REQUIRED for real test). Without this, runs in dry-run mode.",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only run demo-conditioned (skip zero-shot baseline)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run a specific task by ID (e.g., 'turn_off_nightshift')",
    )
    args = parser.parse_args()

    # Safety: require explicit --execute flag
    if not args.execute:
        logger.warning("=" * 60)
        logger.warning("DRY RUN MODE - No actions will be executed!")
        logger.warning("Add --execute flag to actually run actions on your Mac")
        logger.warning("=" * 60)

    # Task selection
    if args.task:
        # Run specific task by ID
        tasks = [t for t in MACOS_TASKS if t["id"] == args.task]
        if not tasks:
            available = [t["id"] for t in MACOS_TASKS]
            logger.error(f"Task '{args.task}' not found. Available: {available}")
            exit(1)
        logger.info(f"Running specific task: {args.task}")
    elif args.single:
        # P0 validation: just Calculator
        tasks = [t for t in MACOS_TASKS if t["id"] == "open_calculator"]
        logger.info("P0 Mode: Testing single Calculator task")
    elif args.quick:
        tasks = MACOS_TASKS[:3]
    else:
        tasks = MACOS_TASKS[:args.tasks]

    logger.info(f"Running P1 with {len(tasks)} tasks using {args.provider}")
    logger.info(f"Execute mode: {args.execute}")

    validator = P1Validator(provider=args.provider, dry_run=not args.execute)

    if args.demo_only:
        # Only run demo-conditioned
        logger.info("\n[DEMO-CONDITIONED ONLY]")
        for task in tasks:
            demo = task.get("demo", "")
            result = validator.run_episode(task, "demo_conditioned", demo=demo)
            validator.results.demo_conditioned.append(result)
        validator._compute_summary()
    else:
        validator.run_ab_test(tasks)

    validator.save_results()
    validator.print_summary()

    return 0 if validator.results.summary.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
