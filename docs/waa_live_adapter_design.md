# WAALiveAdapter Design: DemoConditionedAgent + WAA Integration

**Goal**: Enable DemoConditionedAgent to execute tasks on the live WAA Windows VM via HTTP API.

**Status**: Implemented (`openadapt_ml/benchmarks/waa_live.py`).

---

## 1. Architecture Overview

**Key Principle**: Element-based grounding is handled by WAA, not locally.

The adapter uses WAA's existing element-based execution model:
1. Fetch accessibility tree from `/accessibility` endpoint
2. Extract element IDs and bboxes, POST to `/update_computer`
3. Agent outputs actions with `target_node_id` (element-based grounding)
4. Execute via `/execute_windows` using `computer.mouse.move_id(id)` commands

This keeps grounding authority on WAA's side - we send element IDs, not pixel coordinates.

```
┌─────────────────────────────────────────────────────────────────┐
│  Local Machine                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ DemoConditionedAgent                                      │  │
│  │   ├── VLM (Claude/GPT)                                    │  │
│  │   ├── DemoRetriever → retrieves relevant demo             │  │
│  │   └── Prompt builder → injects demo into context          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              │ BenchmarkAction (target_node_id)  │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ WAALiveAdapter                                            │  │
│  │   ├── Connects to WAA Flask server via HTTP               │  │
│  │   ├── Extracts rects from a11y → POSTs to /update_computer│  │
│  │   ├── Translates CLICK([id]) → computer.mouse.move_id(id) │  │
│  │   └── Implements BenchmarkAdapter interface               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP (SSH tunnel or direct)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Azure VM (172.171.112.41)                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Docker (dockurr/windows)                                  │  │
│  │   ┌───────────────────────────────────────────────────┐   │  │
│  │   │ Windows 11 VM (QEMU)                              │   │  │
│  │   │   ┌─────────────────────────────────────────────┐ │   │  │
│  │   │   │ WAA Flask Server (port 5000)                │ │   │  │
│  │   │   │   /screenshot      → GET screenshot PNG     │ │   │  │
│  │   │   │   /accessibility   → GET a11y tree JSON     │ │   │  │
│  │   │   │   /update_computer → POST element rects     │ │   │  │
│  │   │   │   /execute_windows → POST Computer commands │ │   │  │
│  │   │   │   /probe           → GET health check       │ │   │  │
│  │   │   └─────────────────────────────────────────────┘ │   │  │
│  │   │   ┌─────────────────────────────────────────────┐ │   │  │
│  │   │   │ Computer Class (grounding)                  │ │   │  │
│  │   │   │   mouse.move_id(id) → looks up rects[id]    │ │   │  │
│  │   │   │   mouse.single_click() → pyautogui.click()  │ │   │  │
│  │   │   └─────────────────────────────────────────────┘ │   │  │
│  │   └───────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. WAA Flask Server API

From `vendor/WindowsAgentArena/src/win-arena-container/vm/setup/server/main.py`:

### 2.1 Key Endpoints

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/probe` | GET | - | `{"status": "Probe successful", "message": "Service is operational"}` |
| `/screenshot` | GET | - | PNG image bytes |
| `/accessibility` | GET | `?backend=uia` | `{"AT": {...accessibility tree...}}` |
| `/execute` | POST | `{"command": "pyautogui.click(x, y)"}` | `{"output": "...", "stdout": "...", "stderr": "..."}` |
| `/execute_windows` | POST | `{"command": "...", "shell": "powershell"}` | Executes shell command |

### 2.2 Example Usage

```python
import requests

WAA_SERVER = "http://172.171.112.41:5000"

# Health check
resp = requests.get(f"{WAA_SERVER}/probe")
assert resp.status_code == 200

# Get screenshot
resp = requests.get(f"{WAA_SERVER}/screenshot")
screenshot_png = resp.content

# Get accessibility tree
resp = requests.get(f"{WAA_SERVER}/accessibility?backend=uia")
a11y_tree = resp.json()["AT"]

# Execute click at (500, 300)
resp = requests.post(f"{WAA_SERVER}/execute", json={
    "command": "import pyautogui; pyautogui.click(500, 300)"
})
```

---

## 3. Action Translation

### 3.1 Element-Based Grounding Architecture

**Key Principle**: Grounding (element ID → coordinates) happens on WAA's side, not locally.

The workflow is:
1. Adapter extracts element rects from a11y tree
2. Adapter POSTs rects to WAA's `/update_computer` endpoint
3. WAA's Computer class stores the rects
4. Agent outputs action with `target_node_id`
5. Adapter sends `computer.mouse.move_id(id)` command
6. WAA's Computer looks up the rect and clicks the center

### 3.2 SoM Actions → WAA Commands

| Agent Action | Adapter Translation | WAA Execution |
|--------------|---------------------|---------------|
| `CLICK([id])` | `computer.mouse.move_id('id'); computer.mouse.single_click()` | Computer looks up rects[id], clicks center |
| `CLICK(x, y)` | `computer.mouse.move_abs(x, y); computer.mouse.single_click()` | Fallback for coordinate-based clicks |
| `TYPE("text")` | `import pyautogui; pyautogui.write("text")` | Direct keyboard input (no grounding) |
| `KEY("Enter")` | `import pyautogui; pyautogui.press("enter")` | Direct keyboard input (no grounding) |
| `SCROLL("down")` | `computer.mouse.scroll("down")` | Scroll at current position |
| `DONE()` | None | No-op, marks task complete |

### 3.3 Why Not Local Coordinate Resolution?

The previous implementation did local coordinate resolution:
```python
# BAD: Local grounding (what we removed)
x, y = self._resolve_element_coords(element_id)  # Local lookup
return f"pyautogui.click({x}, {y})"  # Sends coordinates
```

This reintroduced the grounding failure that SoM was meant to solve. The fix:
```python
# GOOD: WAA-side grounding (current implementation)
return f"computer.mouse.move_id('{element_id}'); computer.mouse.single_click()"
```

Now grounding is authoritative because:
- We extract rects from the live a11y tree
- We POST rects to WAA's Computer class
- WAA's `move_id()` uses those rects to compute center coordinates
- Coordinates are computed from authoritative a11y data, not visual estimation

---

## 4. WAALiveAdapter Implementation

### 4.1 Class Definition

```python
from dataclasses import dataclass
import re
import requests
from pathlib import Path
from typing import Any

from openadapt_ml.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkAction,
    BenchmarkObservation,
    BenchmarkResult,
    BenchmarkTask,
)


@dataclass
class WAALiveConfig:
    """Configuration for WAALiveAdapter."""
    server_url: str = "http://172.171.112.41:5000"
    a11y_backend: str = "uia"  # "uia" or "win32"
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

    Example:
        adapter = WAALiveAdapter(WAALiveConfig(server_url="http://vm-ip:5000"))
        agent = DemoConditionedAgent(base_agent, retriever)
        results = evaluate_agent_on_benchmark(agent, adapter, max_steps=15)
    """

    def __init__(self, config: WAALiveConfig | None = None):
        self.config = config or WAALiveConfig()
        self._current_task: BenchmarkTask | None = None
        self._step_count = 0
        self._current_a11y: dict | None = None  # Cache for element resolution

    @property
    def name(self) -> str:
        return "waa-live"

    @property
    def benchmark_type(self) -> str:
        return "interactive"

    def _check_connection(self) -> bool:
        """Check if WAA server is reachable."""
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

        For live adapter, we typically run specific tasks rather than
        listing all 154. Tasks are loaded from local WAA repo if available.
        """
        # TODO: Load from vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/
        return []

    def load_task(self, task_id: str) -> BenchmarkTask:
        """Load a specific WAA task."""
        # TODO: Load task config from WAA repo
        raise NotImplementedError("load_task not implemented for live adapter")

    def reset(self, task: BenchmarkTask) -> BenchmarkObservation:
        """Reset to task's initial state.

        For live adapter, this:
        1. Clears any open windows via /setup/close_all
        2. Runs task setup via /setup endpoints
        3. Returns initial observation
        """
        if not self._check_connection():
            raise RuntimeError(f"Cannot connect to WAA server at {self.config.server_url}")

        self._current_task = task
        self._step_count = 0

        # Close all windows first
        try:
            requests.post(
                f"{self.config.server_url}/setup/close_all",
                timeout=30.0
            )
        except requests.RequestException:
            pass  # Best effort

        # TODO: Execute task setup commands from task.raw_config

        return self._get_observation()

    def step(self, action: BenchmarkAction) -> tuple[BenchmarkObservation, bool, dict[str, Any]]:
        """Execute action and return new observation."""
        self._step_count += 1

        # Translate action to WAA command
        command = self._translate_action(action)

        # Execute command
        if command:
            try:
                resp = requests.post(
                    f"{self.config.server_url}/execute",
                    json={"command": command},
                    timeout=self.config.timeout
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Execute failed: {resp.text}")
            except requests.RequestException as e:
                raise RuntimeError(f"Execute request failed: {e}")

        # Wait for UI to settle
        import time
        time.sleep(self.config.action_delay)

        # Check if done
        done = (
            action.type == "done" or
            self._step_count >= self.config.max_steps
        )

        return self._get_observation(), done, {"step": self._step_count}

    def evaluate(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run WAA's evaluation.

        For live adapter, we need to call WAA's evaluator endpoints
        or run evaluation logic locally.
        """
        # TODO: Implement evaluation via WAA server
        # For now, return placeholder
        return BenchmarkResult(
            task_id=task.task_id,
            success=False,
            score=0.0,
            num_steps=self._step_count,
            reason="Evaluation not implemented for live adapter",
        )

    def close(self) -> None:
        """Clean up resources."""
        pass

    def _get_observation(self) -> BenchmarkObservation:
        """Get current observation from WAA server."""
        # Get screenshot
        try:
            resp = requests.get(
                f"{self.config.server_url}/screenshot",
                timeout=30.0
            )
            screenshot = resp.content if resp.status_code == 200 else None
        except requests.RequestException:
            screenshot = None

        # Get accessibility tree
        try:
            resp = requests.get(
                f"{self.config.server_url}/accessibility?backend={self.config.a11y_backend}",
                timeout=30.0
            )
            if resp.status_code == 200:
                self._current_a11y = resp.json().get("AT", {})
            else:
                self._current_a11y = None
        except requests.RequestException:
            self._current_a11y = None

        return BenchmarkObservation(
            screenshot=screenshot,
            viewport=(self.config.screen_width, self.config.screen_height),
            accessibility_tree=self._current_a11y,
            window_title=None,  # Could be extracted from a11y tree
        )

    def _translate_action(self, action: BenchmarkAction) -> str | None:
        """Translate BenchmarkAction to pyautogui command string."""
        if action.type == "done":
            return None

        if action.type == "click":
            x, y = self._resolve_click_coords(action)
            return f"import pyautogui; pyautogui.click({x}, {y})"

        elif action.type == "double_click":
            x, y = self._resolve_click_coords(action)
            return f"import pyautogui; pyautogui.doubleClick({x}, {y})"

        elif action.type == "right_click":
            x, y = self._resolve_click_coords(action)
            return f"import pyautogui; pyautogui.rightClick({x}, {y})"

        elif action.type == "type":
            # Escape special characters in text
            text = action.text or ""
            text = text.replace("\\", "\\\\").replace("'", "\\'")
            return f"import pyautogui; pyautogui.write('{text}')"

        elif action.type == "key":
            key = action.key or ""
            # Map common key names
            key_map = {
                "Enter": "enter",
                "Tab": "tab",
                "Escape": "escape",
                "Backspace": "backspace",
                "Delete": "delete",
                "Space": "space",
                "Up": "up",
                "Down": "down",
                "Left": "left",
                "Right": "right",
            }
            key = key_map.get(key, key.lower())

            # Handle modifiers
            if action.modifiers:
                mods = "+".join(m.lower() for m in action.modifiers)
                return f"import pyautogui; pyautogui.hotkey('{mods}', '{key}')"
            return f"import pyautogui; pyautogui.press('{key}')"

        elif action.type == "scroll":
            direction = action.scroll_direction or "down"
            amount = action.scroll_amount or 3
            clicks = -amount if direction == "down" else amount
            return f"import pyautogui; pyautogui.scroll({clicks})"

        elif action.type == "wait":
            return "import time; time.sleep(1)"

        else:
            raise ValueError(f"Unknown action type: {action.type}")

    def _resolve_click_coords(self, action: BenchmarkAction) -> tuple[int, int]:
        """Resolve click coordinates, handling SoM element IDs."""
        # Check if we have a target_node_id (SoM format)
        if action.target_node_id and self._current_a11y:
            return self._resolve_element_coords(str(action.target_node_id))

        # Use raw coordinates
        x = action.x or 0
        y = action.y or 0

        # Convert normalized (0-1) to pixel coords
        if 0 <= x <= 1 and 0 <= y <= 1:
            x = int(x * self.config.screen_width)
            y = int(y * self.config.screen_height)

        return (int(x), int(y))

    def _resolve_element_coords(self, element_id: str) -> tuple[int, int]:
        """Find element by ID and return center coordinates."""
        def find_element(node: dict, target_id: str) -> dict | None:
            if str(node.get("id")) == target_id:
                return node
            for child in node.get("children", []):
                result = find_element(child, target_id)
                if result:
                    return result
            return None

        element = find_element(self._current_a11y, element_id)
        if element is None:
            raise KeyError(f"Element ID {element_id} not found in accessibility tree")

        bbox = element.get("bbox") or element.get("BoundingRectangle")
        if not bbox:
            raise KeyError(f"Element {element_id} has no bounding box")

        # Handle different bbox formats
        if isinstance(bbox, list) and len(bbox) == 4:
            x = (bbox[0] + bbox[2]) // 2
            y = (bbox[1] + bbox[3]) // 2
        elif isinstance(bbox, dict):
            x = bbox.get("x", 0) + bbox.get("width", 0) // 2
            y = bbox.get("y", 0) + bbox.get("height", 0) // 2
        else:
            raise ValueError(f"Unknown bbox format: {bbox}")

        return (int(x), int(y))
```

### 4.2 File Location

`openadapt_ml/benchmarks/waa_live.py`

---

## 5. Integration with DemoConditionedAgent

### 5.1 End-to-End Flow

```python
from openadapt_ml.benchmarks import DemoConditionedAgent, evaluate_agent_on_benchmark
from openadapt_ml.benchmarks.waa_live import WAALiveAdapter, WAALiveConfig
from openadapt_ml.retrieval import DemoRetriever
from openadapt_ml.models.api_adapter import ApiVLMAdapter

# 1. Setup components
retriever = DemoRetriever(demo_dir="demos/", method="hybrid")
base_vlm = ApiVLMAdapter(provider="claude")

# 2. Create demo-conditioned agent
agent = DemoConditionedAgent(
    vlm=base_vlm,
    retriever=retriever,
    max_demo_steps=10,
)

# 3. Create live WAA adapter
adapter = WAALiveAdapter(WAALiveConfig(
    server_url="http://172.171.112.41:5000",
    max_steps=15,
))

# 4. Load and run task
task = BenchmarkTask(
    task_id="settings_1",
    instruction="Open Windows Settings and enable Dark Mode",
    domain="settings",
)

# 5. Execute
for step_num in range(15):
    obs = adapter.reset(task) if step_num == 0 else obs

    # Agent selects action (with demo context)
    action = agent.act(obs)

    # Execute on real Windows
    obs, done, info = adapter.step(action)

    if done:
        break

# 6. Evaluate
result = adapter.evaluate(task)
print(f"Task {task.task_id}: {'SUCCESS' if result.success else 'FAILED'}")
```

### 5.2 Demo-Conditioned Prompt Format

When a relevant demo is retrieved, it's injected into the VLM prompt:

```
## Reference Demonstration

The following demonstration shows how to complete a similar task.
Use it as a guide for the current task.

Task: Enable True Tone on macOS
Steps:
1. Screenshot: [System Settings - Displays panel visible]
   Action: CLICK([12])  # "Night Shift..." button
2. Screenshot: [Night Shift popup open]
   Action: CLICK([8])   # Schedule dropdown
3. Screenshot: [Schedule dropdown expanded]
   Action: CLICK([3])   # "Off" option
4. Action: DONE()

---

## Current Task

Goal: Enable Dark Mode on Windows
Current Screenshot: [attached]
Accessibility Tree: [attached]

Based on the reference demonstration and current state, what is the next action?
```

---

## 6. CLI Integration

### 6.1 New Command

```bash
# Run DemoConditionedAgent on live WAA
uv run python -m openadapt_ml.benchmarks.cli waa-demo \
    --server-url http://172.171.112.41:5000 \
    --demo-dir demos/ \
    --provider claude \
    --task-id settings_1 \
    --max-steps 15

# Run on multiple tasks
uv run python -m openadapt_ml.benchmarks.cli waa-demo \
    --server-url http://172.171.112.41:5000 \
    --demo-dir demos/ \
    --provider claude \
    --task-ids settings_1,browser_3,file_explorer_2
```

### 6.2 Existing Command Enhancement

Enhance `vm run-waa` to optionally use DemoConditionedAgent:

```bash
# Run with default navi agent (existing)
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 30

# Run with demo-conditioned agent (new)
uv run python -m openadapt_ml.benchmarks.cli vm run-waa \
    --num-tasks 30 \
    --agent demo-conditioned \
    --demo-dir demos/ \
    --provider claude
```

---

## 7. Implementation Plan

### Phase 1: Basic Adapter (Day 1)

| Task | Output |
|------|--------|
| Create `WAALiveAdapter` class skeleton | `openadapt_ml/benchmarks/waa_live.py` |
| Implement `_get_observation()` | Screenshot + a11y tree fetching |
| Implement `_translate_action()` | Action → pyautogui command |
| Add `reset()` and `step()` | Basic episode loop |
| Unit tests | `tests/test_waa_live.py` |

### Phase 2: Element Resolution (Day 1-2)

| Task | Output |
|------|--------|
| Implement `_resolve_element_coords()` | SoM ID → (x, y) translation |
| Handle edge cases (no bbox, nested elements) | Robust error handling |
| Integration test with mock server | Verify full flow |

### Phase 3: CLI & Integration (Day 2)

| Task | Output |
|------|--------|
| Add `waa-demo` CLI command | `cli.py` update |
| Integrate with DemoConditionedAgent | End-to-end test |
| Add VM connection management | SSH tunnel option |

### Phase 4: Evaluation (Day 2-3)

| Task | Output |
|------|--------|
| Implement `evaluate()` method | Run WAA evaluators |
| Add metrics collection | Success rate, step count |
| Document usage | Update `docs/waa_setup.md` |

---

## 8. Open Questions

1. **SoM Overlay**: Should we add visual SoM markers to screenshots before sending to VLM?
   - Pros: VLM can reference `[id]` directly
   - Cons: Requires image processing, may clutter UI

2. **Task Setup**: How to run task-specific setup commands?
   - Option A: Parse task.raw_config and call /setup endpoints
   - Option B: Require user to manually set up Windows before running

3. **Evaluation**: Can we call WAA's Python evaluators from local machine?
   - Need to analyze WAA evaluator code
   - May need to run evaluators on VM via SSH

4. **Error Recovery**: What happens when VLM returns unparseable action?
   - Current: Fail immediately
   - Alternative: Retry with clarification prompt

---

## 9. Related Files

- `openadapt_ml/benchmarks/waa.py` - Existing WAAAdapter (local, uses DesktopEnv)
- `vendor/WindowsAgentArena/src/win-arena-container/vm/setup/server/main.py` - Flask server code
- `vendor/WindowsAgentArena/src/win-arena-container/client/desktop_env/controllers/python.py` - Client controller
- `docs/waa_setup.md` - VM setup guide
- `docs/demo_retrieval_design.md` - Demo retrieval system design
