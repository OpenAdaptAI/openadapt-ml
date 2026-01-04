"""Windows Agent Arena Live adapter.

This module provides a live HTTP-based adapter for WAA that connects to the
WAA Flask server running inside a Windows VM. Unlike WAAAdapter which imports
WAA's DesktopEnv locally, this adapter talks to the server remotely.

Architecture:
    The adapter uses WAA's element-based execution model:
    1. Fetch accessibility tree from /accessibility endpoint
    2. Extract element bboxes and POST to /update_computer as rects dict
    3. Agent outputs actions with target_node_id (element-based grounding)
    4. Execute via /execute_windows using computer.mouse.move_id(id) commands

    This keeps grounding authority on WAA side - we send element IDs,
    not pixel coordinates. WAA's Computer class handles the grounding.

Example:
    from openadapt_ml.benchmarks.waa_live import WAALiveAdapter, WAALiveConfig

    adapter = WAALiveAdapter(WAALiveConfig(server_url="http://vm-ip:5000"))
    agent = DemoConditionedAgent(base_agent, retriever)
    results = evaluate_agent_on_benchmark(agent, adapter, max_steps=15)
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from openadapt_ml.benchmarks.base import (
    BenchmarkAction,
    BenchmarkAdapter,
    BenchmarkObservation,
    BenchmarkResult,
    BenchmarkTask,
)

logger = logging.getLogger(__name__)


@dataclass
class WAALiveConfig:
    """Configuration for WAALiveAdapter.

    Attributes:
        server_url: URL of WAA Flask server (e.g., "http://172.171.112.41:5000").
        a11y_backend: Accessibility backend ("uia" or "win32").
        screen_width: Screen width in pixels.
        screen_height: Screen height in pixels.
        max_steps: Default maximum steps per task.
        action_delay: Delay after actions in seconds (for UI to settle).
        timeout: Request timeout in seconds.
    """

    server_url: str = "http://localhost:5000"
    a11y_backend: str = "uia"
    screen_width: int = 1920
    screen_height: int = 1200
    max_steps: int = 15
    action_delay: float = 0.5
    timeout: float = 90.0


class WAALiveAdapter(BenchmarkAdapter):
    """Live WAA adapter that connects to WAA Flask server over HTTP.

    Unlike WAAAdapter which imports WAA's DesktopEnv locally, this adapter
    talks to the WAA server remotely via HTTP. This enables:
    - Running DemoConditionedAgent from local machine
    - Using our own VLM (Claude/GPT) instead of WAA's built-in navi agent
    - Injecting demos into prompts before each action

    Args:
        config: WAALiveConfig with server URL and settings.
    """

    def __init__(self, config: WAALiveConfig | None = None):
        self.config = config or WAALiveConfig()
        self._current_task: BenchmarkTask | None = None
        self._step_count = 0
        self._current_a11y: dict | None = None
        self._current_rects: dict[str, list[int]] = {}  # element_id -> [l, t, r, b]
        self._current_screenshot: bytes | None = None
        self._actions: list[BenchmarkAction] = []

    @property
    def name(self) -> str:
        """Benchmark name."""
        return "waa-live"

    @property
    def benchmark_type(self) -> str:
        """Benchmark type (interactive)."""
        return "interactive"

    @property
    def supports_parallel(self) -> bool:
        """Whether parallel execution is supported."""
        return False  # Single VM for now

    def check_connection(self) -> bool:
        """Check if WAA server is reachable.

        Returns:
            True if server responds to /probe endpoint.
        """
        try:
            resp = requests.get(
                f"{self.config.server_url}/probe",
                timeout=5.0
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_tasks(self, domain: str | None = None) -> list[BenchmarkTask]:
        """List available WAA tasks.

        For live adapter, tasks are typically loaded on-demand.
        Returns empty list - use load_task() directly.
        """
        return []

    def load_task(self, task_id: str) -> BenchmarkTask:
        """Load a specific task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            BenchmarkTask object.
        """
        # For now, create a minimal task - actual task configs should be
        # loaded from WAA repo if needed
        return BenchmarkTask(
            task_id=task_id,
            instruction=f"Task {task_id}",
            domain=task_id.split("_")[0] if "_" in task_id else "unknown",
            time_limit_steps=self.config.max_steps,
        )

    def reset(self, task: BenchmarkTask) -> BenchmarkObservation:
        """Reset environment to task's initial state.

        Args:
            task: Task to initialize.

        Returns:
            Initial observation (screenshot + accessibility tree).

        Raises:
            RuntimeError: If server is not reachable.
        """
        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to WAA server at {self.config.server_url}. "
                f"Ensure Windows VM is running and server is started."
            )

        self._current_task = task
        self._step_count = 0
        self._actions = []

        # Try to close all windows for clean state
        try:
            requests.post(
                f"{self.config.server_url}/setup/close_all",
                timeout=30.0
            )
            logger.info("Closed all windows for clean state")
        except requests.RequestException as e:
            logger.warning(f"Failed to close windows: {e}")

        # If task has setup commands in raw_config, execute them
        if task.raw_config:
            self._run_task_setup(task.raw_config)

        # Small delay for UI to settle
        time.sleep(1.0)

        return self._get_observation()

    def step(
        self, action: BenchmarkAction
    ) -> tuple[BenchmarkObservation, bool, dict[str, Any]]:
        """Execute action and return new observation.

        Uses element-based grounding via WAA's Computer class. Click actions
        are translated to computer.mouse.move_id(id) commands that WAA executes
        using the rects we POSTed to /update_computer.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (observation, done, info).
        """
        self._step_count += 1
        self._actions.append(action)

        # Translate action to element-based command for WAA's Computer
        command = self._translate_action(action)

        # Execute command via /execute_windows (has access to computer object)
        if command:
            try:
                resp = requests.post(
                    f"{self.config.server_url}/execute_windows",
                    json={"command": command},
                    timeout=self.config.timeout
                )
                if resp.status_code != 200:
                    logger.error(f"Execute failed ({resp.status_code}): {resp.text}")
                else:
                    result = resp.json()
                    if result.get("stderr"):
                        logger.warning(f"Command stderr: {result['stderr']}")
                    logger.debug(f"Executed: {command}")
            except requests.RequestException as e:
                logger.error(f"Execute request failed: {e}")

        # Wait for UI to settle
        time.sleep(self.config.action_delay)

        # Check if done
        done = (
            action.type == "done" or
            self._step_count >= self.config.max_steps
        )

        obs = self._get_observation()
        info = {
            "step": self._step_count,
            "command": command,
        }

        return obs, done, info

    def evaluate(self, task: BenchmarkTask) -> BenchmarkResult:
        """Evaluate current state against task success criteria.

        For live adapter, full evaluation requires running WAA's evaluators.
        Currently returns a placeholder result.

        Args:
            task: Task to evaluate.

        Returns:
            BenchmarkResult with success/score.
        """
        # TODO: Implement proper evaluation by calling WAA evaluators
        # For now, check if agent took any actions
        has_actions = len(self._actions) > 0
        called_done = any(a.type == "done" for a in self._actions)

        return BenchmarkResult(
            task_id=task.task_id,
            success=False,  # Can't determine without evaluator
            score=0.5 if has_actions and called_done else 0.0,
            num_steps=self._step_count,
            reason="Evaluation requires WAA evaluators (not yet implemented)",
        )

    def close(self) -> None:
        """Clean up resources."""
        self._current_task = None
        self._current_a11y = None
        self._actions = []

    def _get_observation(self) -> BenchmarkObservation:
        """Fetch current observation from WAA server.

        Also extracts element rects from a11y tree and updates WAA's Computer
        so element-based grounding works for subsequent actions.

        Returns:
            BenchmarkObservation with screenshot and accessibility tree.
        """
        screenshot = None
        a11y_tree = None

        # Get screenshot
        try:
            resp = requests.get(
                f"{self.config.server_url}/screenshot",
                timeout=30.0
            )
            if resp.status_code == 200:
                screenshot = resp.content
                self._current_screenshot = screenshot
                logger.debug(f"Got screenshot: {len(screenshot)} bytes")
            else:
                logger.warning(f"Screenshot request failed: {resp.status_code}")
        except requests.RequestException as e:
            logger.error(f"Screenshot request error: {e}")

        # Get accessibility tree
        try:
            resp = requests.get(
                f"{self.config.server_url}/accessibility",
                params={"backend": self.config.a11y_backend},
                timeout=30.0
            )
            if resp.status_code == 200:
                result = resp.json()
                a11y_tree = result.get("AT", {})
                self._current_a11y = a11y_tree
                # Extract rects for element-based grounding
                self._current_rects = self._extract_rects_from_a11y(a11y_tree)
                logger.debug("Got accessibility tree with %d elements", len(self._current_rects))
            else:
                logger.warning(f"A11y request failed: {resp.status_code}")
        except requests.RequestException as e:
            logger.error(f"A11y request error: {e}")

        # Update WAA's Computer with current rects for element grounding
        if self._current_rects:
            self._update_waa_computer()

        return BenchmarkObservation(
            screenshot=screenshot,
            viewport=(self.config.screen_width, self.config.screen_height),
            accessibility_tree=a11y_tree,
            window_title=self._extract_window_title(a11y_tree),
        )

    def _extract_window_title(self, a11y_tree: dict | str | None) -> str | None:
        """Extract window title from accessibility tree."""
        if not a11y_tree:
            return None
        # Handle XML string - can't extract title easily
        if isinstance(a11y_tree, str):
            return None
        # Try common field names
        for key in ["Name", "name", "title", "Title"]:
            if key in a11y_tree:
                return a11y_tree[key]
        return None

    def _extract_rects_from_a11y(self, a11y_tree: dict | None) -> dict[str, list[int]]:
        """Extract element ID -> bounding box mapping from accessibility tree.

        This produces the `rects` dict that WAA's Computer class expects.
        The rects are then POSTed to /update_computer so WAA can handle grounding.

        Args:
            a11y_tree: Accessibility tree from /accessibility endpoint.

        Returns:
            Dict mapping element IDs to [left, top, right, bottom] bounding boxes.
        """
        rects: dict[str, list[int]] = {}

        def visit(node: dict) -> None:
            # Get element ID
            elem_id = None
            for id_field in ["id", "Id", "ID", "AutomationId"]:
                if id_field in node and node[id_field]:
                    elem_id = str(node[id_field])
                    break

            # Get bounding box
            bbox = None
            for bbox_field in ["bbox", "BoundingRectangle", "Rect", "rect"]:
                if bbox_field in node:
                    bbox = node[bbox_field]
                    break

            # Store if we have both ID and bbox
            if elem_id is not None and bbox is not None:
                # Normalize bbox to [left, top, right, bottom]
                if isinstance(bbox, list) and len(bbox) == 4:
                    # Could be [l, t, r, b] or [l, t, w, h] - assume [l, t, r, b]
                    rects[elem_id] = [int(x) for x in bbox]
                elif isinstance(bbox, dict):
                    x = bbox.get("x", 0)
                    y = bbox.get("y", 0)
                    w = bbox.get("width", 0)
                    h = bbox.get("height", 0)
                    rects[elem_id] = [x, y, x + w, y + h]
                elif isinstance(bbox, str):
                    parts = [int(p) for p in bbox.split(",")]
                    if len(parts) == 4:
                        rects[elem_id] = parts

            # Visit children
            for child_field in ["children", "Children"]:
                children = node.get(child_field, [])
                if isinstance(children, list):
                    for child in children:
                        if isinstance(child, dict):
                            visit(child)

        if a11y_tree:
            # Handle case where a11y_tree is XML string (WAA returns XML)
            if isinstance(a11y_tree, str):
                # TODO: Parse XML to dict if needed for element grounding
                logger.debug("A11y tree is XML string, skipping rect extraction")
                return rects
            visit(a11y_tree)

        logger.debug(f"Extracted {len(rects)} element rects from a11y tree")
        return rects

    def _update_waa_computer(self) -> None:
        """POST current rects and screenshot to WAA's /update_computer endpoint.

        This syncs WAA's Computer object with our current element state,
        allowing computer.mouse.move_id(id) to work correctly.
        """
        if not self._current_rects:
            logger.warning("No rects to update - skipping /update_computer")
            return

        # Encode screenshot as base64
        screenshot_b64 = ""
        if self._current_screenshot:
            screenshot_b64 = base64.b64encode(self._current_screenshot).decode("utf-8")

        # Window rect (full screen for now)
        window_rect = [0, 0, self.config.screen_width, self.config.screen_height]

        payload = {
            "rects": self._current_rects,
            "window_rect": window_rect,
            "screenshot": screenshot_b64,
            "scale": [1.0, 1.0],
        }

        try:
            resp = requests.post(
                f"{self.config.server_url}/update_computer",
                json=payload,
                timeout=30.0
            )
            if resp.status_code == 200:
                logger.debug("Updated WAA computer with %d rects", len(self._current_rects))
            else:
                logger.warning(f"update_computer failed: {resp.status_code} - {resp.text}")
        except requests.RequestException as e:
            logger.error(f"update_computer request error: {e}")

    def _run_task_setup(self, raw_config: dict) -> None:
        """Run task setup commands from raw_config.

        Args:
            raw_config: Task configuration with setup commands.
        """
        # Handle different setup command formats
        setup = raw_config.get("setup", raw_config.get("init", {}))

        if isinstance(setup, dict):
            # Launch application if specified
            if "app" in setup or "application" in setup:
                app = setup.get("app") or setup.get("application")
                try:
                    requests.post(
                        f"{self.config.server_url}/setup/launch",
                        json={"app": app},
                        timeout=30.0
                    )
                    logger.info(f"Launched app: {app}")
                except requests.RequestException as e:
                    logger.warning(f"Failed to launch app: {e}")

            # Run shell commands if specified
            if "commands" in setup:
                for cmd in setup["commands"]:
                    try:
                        requests.post(
                            f"{self.config.server_url}/execute_windows",
                            json={"command": cmd, "shell": "powershell"},
                            timeout=60.0
                        )
                        logger.info(f"Ran setup command: {cmd[:50]}...")
                    except requests.RequestException as e:
                        logger.warning(f"Setup command failed: {e}")

    def _translate_action(self, action: BenchmarkAction) -> str | None:
        """Translate BenchmarkAction to element-based command for WAA's Computer.

        Uses WAA's Computer class via /execute_windows endpoint. Click actions
        use computer.mouse.move_id(id) for element-based grounding - the actual
        coordinates are resolved by WAA's Computer class using the rects we
        POSTed to /update_computer.

        Args:
            action: The action to translate.

        Returns:
            Python command string to execute via /execute_windows endpoint,
            or None for actions that don't need execution.
        """
        if action.type == "done":
            return None

        if action.type == "wait":
            return "import time; time.sleep(1)"

        if action.type == "click":
            return self._translate_click_action(action, "single_click")

        if action.type == "double_click":
            return self._translate_click_action(action, "double_click")

        if action.type == "right_click":
            return self._translate_click_action(action, "right_click")

        if action.type == "type":
            text = action.text or ""
            # Escape special characters
            text = text.replace("\\", "\\\\").replace("'", "\\'")
            # Use pyautogui for typing (no grounding needed)
            return f"import pyautogui; pyautogui.write('{text}', interval=0.02)"

        if action.type == "key":
            return self._translate_key_action(action)

        if action.type == "scroll":
            direction = action.scroll_direction or "down"
            return f"computer.mouse.scroll('{direction}')"

        if action.type == "drag":
            # Drag requires start and end - use element IDs or coordinates
            if action.target_node_id is not None:
                elem_id = str(action.target_node_id)
                if elem_id in self._current_rects:
                    # Start at element, drag to end coords
                    end_x = action.end_x or 0
                    end_y = action.end_y or 0
                    if isinstance(end_x, float) and 0 <= end_x <= 1:
                        end_x = int(end_x * self.config.screen_width)
                    if isinstance(end_y, float) and 0 <= end_y <= 1:
                        end_y = int(end_y * self.config.screen_height)
                    return (
                        f"computer.mouse.move_id('{elem_id}'); "
                        f"computer.mouse.drag({int(end_x)}, {int(end_y)})"
                    )
            logger.warning("Drag requires target_node_id with valid element")
            return None

        logger.warning(f"Unknown action type: {action.type}")
        return None

    def _translate_click_action(self, action: BenchmarkAction, click_method: str) -> str:
        """Translate click-type action to element-based command.

        Args:
            action: The click action.
            click_method: "single_click", "double_click", or "right_click".

        Returns:
            Python command string using computer.mouse.move_id() for grounding.
        """
        # Prefer element ID for grounding (SoM mode)
        if action.target_node_id is not None:
            elem_id = str(action.target_node_id)
            if elem_id in self._current_rects:
                return f"computer.mouse.move_id('{elem_id}'); computer.mouse.{click_method}()"
            else:
                logger.warning(f"Element ID '{elem_id}' not found in rects, falling back to coordinates")

        # Fallback: use coordinates if provided (less precise)
        x = action.x if action.x is not None else 0
        y = action.y if action.y is not None else 0

        # Normalize coordinates
        if isinstance(x, float) and 0 <= x <= 1:
            x = x  # Keep normalized - move_abs handles it
        if isinstance(y, float) and 0 <= y <= 1:
            y = y  # Keep normalized

        return f"computer.mouse.move_abs({x}, {y}); computer.mouse.{click_method}()"

    def _translate_key_action(self, action: BenchmarkAction) -> str:
        """Translate key press action using pyautogui (no grounding needed)."""
        key = action.key or ""

        # Map common key names to pyautogui names
        key_map = {
            "Enter": "enter",
            "Return": "enter",
            "Tab": "tab",
            "Escape": "escape",
            "Esc": "escape",
            "Backspace": "backspace",
            "Delete": "delete",
            "Del": "delete",
            "Space": "space",
            "Up": "up",
            "Down": "down",
            "Left": "left",
            "Right": "right",
            "Home": "home",
            "End": "end",
            "PageUp": "pageup",
            "PageDown": "pagedown",
            "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
            "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
            "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12",
        }
        key = key_map.get(key, key.lower())

        # Handle modifiers with hotkey
        if action.modifiers:
            mods = [m.lower() for m in action.modifiers]
            mod_map = {"control": "ctrl", "command": "win", "meta": "win"}
            mods = [mod_map.get(m, m) for m in mods]
            all_keys = mods + [key]
            keys_str = ", ".join(f"'{k}'" for k in all_keys)
            return f"import pyautogui; pyautogui.hotkey({keys_str})"

        return f"import pyautogui; pyautogui.press('{key}')"
