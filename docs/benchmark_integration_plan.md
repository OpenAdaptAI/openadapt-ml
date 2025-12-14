# Benchmark Integration Plan

## Overview

This document outlines the plan for integrating GUI agent benchmarks into openadapt-ml, enabling standardized evaluation of fine-tuned models on real-world tasks.

### Goals

1. **Validate fine-tuning pipeline** - Confirm that models trained on recordings generalize to benchmark tasks
2. **Establish baselines** - Compare fine-tuned models against off-the-shelf APIs (GPT-4V, Claude, Gemini)
3. **Support multiple benchmarks** - Unified interface for WAA, OSWorld, WebArena, and future benchmarks

### Current Scope

**Primary focus**: Windows Agent Arena (WAA)
- 154 tasks across 11 Windows domains
- MIT licensed, runs locally or on Azure
- Well-documented evaluation protocol

**Future work** (not in current scope):
- WebArena/VisualWebArena (browser)
- OSWorld (cross-platform desktop)
- AndroidWorld (mobile)

---

## Part 1: Architecture

### 1.1 Two Benchmark Types

Benchmarks fall into two categories:

| Type | Description | Examples |
|------|-------------|----------|
| **Interactive** | Run environment, step through tasks, execution-based scoring | WAA, OSWorld, WebArena |
| **Static** | Load trajectories, train/eval offline with provided labels | Mind2Web, MiniWoB++ demos |

The adapter interface must support both patterns.

### 1.2 BenchmarkAdapter Interface

```python
# openadapt_ml/benchmarks/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Any

@dataclass
class BenchmarkTask:
    """Canonical task representation."""
    task_id: str
    instruction: str
    domain: str  # "web", "desktop", "mobile"

    # Environment setup
    initial_state_ref: str | None  # VM snapshot, storage_state, start URL
    time_limit_steps: int | None

    # Preserve original config losslessly
    raw_config: dict[str, Any]

    # Evaluation spec (benchmark-native)
    evaluation_spec: dict[str, Any] | None


@dataclass
class BenchmarkObservation:
    """Canonical observation at each step."""
    # Visual
    screenshot: bytes | None  # PNG image bytes
    screenshot_path: str | None
    viewport: tuple[int, int] | None  # (width, height)

    # Structured UI (format varies by platform)
    accessibility_tree: dict | None  # UIA (Windows), AXTree (macOS), DOM (web)
    dom_html: str | None  # Raw HTML for web

    # Context
    url: str | None  # For web tasks
    window_title: str | None  # For desktop tasks
    focused_element: dict | None  # {node_id, bbox, text}

    # Raw benchmark-specific data (lossless)
    raw_observation: dict[str, Any] | None


@dataclass
class BenchmarkAction:
    """Canonical action representation."""
    type: str  # "click", "type", "scroll", "key", "drag", "answer", "done"

    # Pointer actions - coordinates
    x: float | None  # Normalized [0,1] or pixel
    y: float | None

    # Element grounding (when available)
    target_node_id: str | None  # DOM/AX/UIA node ID
    target_bbox: tuple[float, float, float, float] | None
    target_role: str | None  # "button", "textfield", etc.
    target_name: str | None  # Accessible name

    # Keyboard actions
    text: str | None  # For "type" action - text to type
    key: str | None  # For "key" action - single key (e.g., "Enter", "Tab", "Escape")
    modifiers: list[str] | None  # ["ctrl", "shift", "alt"]

    # Scroll actions
    scroll_direction: str | None  # "up", "down", "left", "right"
    scroll_amount: float | None  # Pixels or normalized

    # Drag actions
    end_x: float | None
    end_y: float | None

    # Answer action (some benchmarks score by final answer)
    answer: str | None

    # Raw benchmark-specific format (lossless)
    raw_action: dict[str, Any] | None


@dataclass
class BenchmarkResult:
    """Result of a single task evaluation."""
    task_id: str
    success: bool
    score: float  # 0.0 to 1.0

    # Trajectory
    steps: list[tuple[BenchmarkObservation, BenchmarkAction]]
    num_steps: int

    # Diagnostics
    error: str | None
    reason: str | None  # Why success/fail

    # Timing
    total_time_seconds: float


class BenchmarkAdapter(ABC):
    """Abstract interface for benchmark integration."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name (e.g., 'waa', 'osworld', 'webarena')."""
        pass

    @property
    @abstractmethod
    def benchmark_type(self) -> str:
        """'interactive' or 'static'."""
        pass

    @abstractmethod
    def list_tasks(self, domain: str | None = None) -> list[BenchmarkTask]:
        """List available tasks, optionally filtered by domain."""
        pass

    @abstractmethod
    def load_task(self, task_id: str) -> BenchmarkTask:
        """Load a specific task by ID."""
        pass

    @abstractmethod
    def reset(self, task: BenchmarkTask) -> BenchmarkObservation:
        """Reset environment to task's initial state, return first observation."""
        pass

    @abstractmethod
    def step(self, action: BenchmarkAction) -> tuple[BenchmarkObservation, bool, dict]:
        """Execute action, return (observation, done, info)."""
        pass

    @abstractmethod
    def evaluate(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run benchmark's native evaluation on current state."""
        pass

    def close(self) -> None:
        """Clean up resources (VMs, browser, etc.)."""
        pass


class StaticDatasetAdapter(BenchmarkAdapter):
    """Base for static trajectory datasets (Mind2Web, demos)."""

    @property
    def benchmark_type(self) -> str:
        return "static"

    @abstractmethod
    def load_trajectories(self, split: str = "test") -> Iterator[list[tuple[BenchmarkObservation, BenchmarkAction]]]:
        """Iterate over expert trajectories."""
        pass

    def reset(self, task: BenchmarkTask) -> BenchmarkObservation:
        raise NotImplementedError("Static datasets don't support interactive reset")

    def step(self, action: BenchmarkAction) -> tuple[BenchmarkObservation, bool, dict]:
        raise NotImplementedError("Static datasets don't support interactive stepping")
```

### 1.3 Agent Interface

```python
# openadapt_ml/benchmarks/agent.py

from abc import ABC, abstractmethod

class BenchmarkAgent(ABC):
    """Interface for agents evaluated on benchmarks."""

    @abstractmethod
    def act(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None = None,
    ) -> BenchmarkAction:
        """Given observation and task, return next action."""
        pass

    def reset(self) -> None:
        """Reset agent state between episodes."""
        pass


class PolicyAgent(BenchmarkAgent):
    """Wraps openadapt-ml AgentPolicy for benchmark evaluation."""

    def __init__(self, policy: "AgentPolicy"):
        self.policy = policy

    def act(self, observation, task, history=None):
        # Convert BenchmarkObservation → SFT sample format
        # Call policy.predict()
        # Convert Action → BenchmarkAction
        ...
```

### 1.4 Evaluation Runner

```python
# openadapt_ml/benchmarks/runner.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

def evaluate_agent_on_benchmark(
    agent: BenchmarkAgent,
    adapter: BenchmarkAdapter,
    task_ids: list[str] | None = None,
    max_steps: int = 50,
    parallel: int = 1,
) -> list[BenchmarkResult]:
    """Run agent on benchmark tasks and collect results.

    Args:
        agent: Agent to evaluate.
        adapter: Benchmark adapter.
        task_ids: Specific tasks to run (None = all tasks).
        max_steps: Maximum steps per task.
        parallel: Number of parallel workers (requires adapter support).

    Returns:
        List of BenchmarkResult for each task.
    """
    tasks = adapter.list_tasks() if task_ids is None else [adapter.load_task(t) for t in task_ids]

    if parallel > 1 and hasattr(adapter, 'supports_parallel') and adapter.supports_parallel:
        return _evaluate_parallel(agent, adapter, tasks, max_steps, parallel)

    return _evaluate_sequential(agent, adapter, tasks, max_steps)


def _evaluate_sequential(agent, adapter, tasks, max_steps):
    results = []
    for task in tasks:
        result = _run_single_task(agent, adapter, task, max_steps)
        results.append(result)
    return results


def _run_single_task(agent, adapter, task, max_steps):
    obs = adapter.reset(task)
    history = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.act(obs, task, history)
        history.append((obs, action))
        obs, done, info = adapter.step(action)
        steps += 1

    return adapter.evaluate(task)
```

---

## Part 2: Schema Extensions

### 2.1 Current Schema Gaps

Comparing current `openadapt_ml.schemas` to benchmark requirements:

| Required Field | Current Schema | Status |
|----------------|----------------|--------|
| `target_node_id` | ❌ Missing | Add to Action |
| `target_role/name` | ❌ Missing | Add to Action |
| `key` (single keypress) | ❌ Missing | Add to Action |
| `scroll_direction/amount` | ❌ Missing | Add to Action |
| `drag end coords` | ❌ Missing | Add to Action |
| `accessibility_tree` | ❌ Missing | Add to Observation |
| `dom_html` | ❌ Missing | Add to Observation |
| `url` | ❌ Missing | Add to Observation |
| `window_title` | ❌ Missing | Add to Observation |
| `answer` action type | ❌ Missing | Add to Action.type |
| `raw_config` preservation | ❌ Missing | Add to Episode/Task |

### 2.2 Proposed Schema Updates

```python
# openadapt_ml/schemas/sessions.py - Extended Action

@dataclass
class Action:
    type: str  # "click", "type", "scroll", "key", "drag", "answer", "done"
    x: float | None = None
    y: float | None = None
    text: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    # Element grounding
    target_node_id: str | None = None  # DOM/AX/UIA reference
    target_role: str | None = None  # "button", "textfield", etc.
    target_name: str | None = None  # Accessible name

    # Keyboard
    key: str | None = None  # Single key: "Enter", "Tab", "Escape"
    modifiers: list[str] | None = None  # ["ctrl", "shift"]

    # Scroll
    scroll_direction: str | None = None  # "up", "down"
    scroll_amount: float | None = None

    # Drag
    end_x: float | None = None
    end_y: float | None = None

    # Answer (for benchmarks that score by answer)
    answer: str | None = None

    raw: dict | None = None


# openadapt_ml/schemas/sessions.py - Extended Observation

@dataclass
class Observation:
    image_path: str | None = None

    # Structured UI
    accessibility_tree: dict | None = None  # UIA/AXTree/DOM
    dom_html: str | None = None  # Raw HTML

    # Context
    url: str | None = None
    window_title: str | None = None
    app_name: str | None = None
    focused_element: dict | None = None  # {node_id, bbox, text}

    raw: dict | None = None
```

### 2.3 Accessibility Tree Normalization

Different platforms provide accessibility trees in different formats:
- **Windows**: UI Automation (UIA) tree
- **macOS/Linux**: AXTree
- **Web**: DOM with ARIA attributes

For policy consumption, extract a minimal common subset:

```python
@dataclass
class UIElement:
    """Normalized UI element for cross-platform use."""
    node_id: str
    role: str  # "button", "textfield", "link", etc.
    name: str | None  # Accessible name/label
    bbox: tuple[float, float, float, float] | None  # Normalized
    text: str | None  # Text content
    children: list["UIElement"] | None
```

The full platform-specific tree is preserved in `raw_observation` for debugging.

### 2.4 Grounding Module Integration

The existing grounding module (`openadapt_ml/grounding/`) complements benchmark integration:

| Scenario | Approach |
|----------|----------|
| Benchmark provides accessibility tree | Use tree for element selection |
| Benchmark provides only screenshots | Use `GeminiGrounder` for visual grounding |
| SoM-annotated screenshots | Model selects element by number |
| Real deployment (no overlays) | Grounding module finds elements by description |

```python
def act_with_grounding(self, observation, task):
    # Policy decides WHAT to do
    thought, target_description = self.policy.reason(observation, task)

    # Grounder finds WHERE
    if observation.accessibility_tree:
        # Use structured UI if available (preferred)
        element = find_in_tree(observation.accessibility_tree, target_description)
    else:
        # Fall back to visual grounding
        candidates = self.grounder.ground(observation.screenshot, target_description)
        element = candidates[0] if candidates else None

    return BenchmarkAction(
        type="click",
        x=element.centroid[0],
        y=element.centroid[1],
        target_node_id=element.node_id,
        target_bbox=element.bbox,
    )
```

---

## Part 3: Windows Agent Arena (WAA) Adapter

### 3.1 Overview

WAA is the first large-scale benchmark for agents on real Windows OS:
- **154 tasks** across 11 domains (browser, Office, coding, media, system apps)
- **MIT licensed** - open source
- **Azure parallelization** - optional, can run locally
- **SOTA**: 19.5% success (GPT-4V + OmniParser) vs 74.5% human

### 3.2 Adapter Implementation

```python
# openadapt_ml/benchmarks/waa.py

class WAAAdapter(BenchmarkAdapter):
    """Windows Agent Arena adapter."""

    def __init__(
        self,
        waa_repo_path: str,  # Path to cloned WAA repo
        use_azure: bool = False,  # Local vs Azure VMs
        vm_snapshot: str | None = None,
    ):
        self.waa_repo = Path(waa_repo_path)
        self.use_azure = use_azure
        self._vm_controller = None
        ...

    @property
    def name(self) -> str:
        return "waa"

    @property
    def benchmark_type(self) -> str:
        return "interactive"

    @property
    def supports_parallel(self) -> bool:
        return self.use_azure  # Azure supports parallel VMs

    def list_tasks(self, domain=None) -> list[BenchmarkTask]:
        """List WAA tasks.

        Domains: browser, office, coding, media, notepad, paint,
                 file_explorer, clock, settings, edge, vscode
        """
        tasks = self._load_task_configs()
        if domain:
            tasks = [t for t in tasks if t.domain == domain]
        return tasks

    def reset(self, task: BenchmarkTask) -> BenchmarkObservation:
        """Reset Windows VM to task's initial state."""
        # Load VM snapshot
        self._vm_controller.load_snapshot(task.initial_state_ref)

        # Apply task-specific setup (open apps, create files, etc.)
        self._apply_task_setup(task)

        # Capture initial observation
        return self._capture_observation()

    def step(self, action: BenchmarkAction) -> tuple[BenchmarkObservation, bool, dict]:
        """Execute action on Windows VM."""
        # Convert canonical action to WAA format
        waa_action = self._to_waa_action(action)

        # Execute via VM controller
        self._vm_controller.execute(waa_action)

        # Capture new observation
        obs = self._capture_observation()

        # Check if done (task-specific termination)
        done = self._check_done()

        return obs, done, {}

    def evaluate(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run WAA's execution-based evaluator."""
        # WAA evaluators check OS state (files, settings, app state)
        success = self._run_evaluator(task)
        return BenchmarkResult(
            task_id=task.task_id,
            success=success,
            score=1.0 if success else 0.0,
            ...
        )

    def _capture_observation(self) -> BenchmarkObservation:
        """Capture screenshot and UIA tree from VM."""
        screenshot = self._vm_controller.capture_screen()
        uia_tree = self._vm_controller.get_uia_tree()

        return BenchmarkObservation(
            screenshot=screenshot,
            accessibility_tree=uia_tree,
            window_title=self._vm_controller.get_active_window_title(),
            raw_observation={"uia_tree_full": uia_tree},
        )

    def _to_waa_action(self, action: BenchmarkAction) -> dict:
        """Convert canonical action to WAA format."""
        if action.type == "click":
            return {"type": "click", "x": action.x, "y": action.y}
        elif action.type == "type":
            return {"type": "type", "text": action.text}
        elif action.type == "key":
            return {"type": "key", "key": action.key, "modifiers": action.modifiers}
        # ... etc
```

### 3.3 Integration Requirements

1. **Clone WAA repo**: `git clone https://github.com/microsoft/WindowsAgentArena`
2. **Windows VM**: Local Windows 10/11 or Azure VM
3. **VM Controller**: WAA provides Python API for VM interaction
4. **OmniParser** (optional): For UI parsing if not using raw UIA

### 3.4 Action Space Mapping

WAA uses low-level mouse/keyboard events. Mapping to canonical actions:

| WAA Action | Canonical Action |
|------------|------------------|
| `click(x, y)` | `type="click", x=x, y=y` |
| `double_click(x, y)` | `type="double_click", x=x, y=y` |
| `type_text(text)` | `type="type", text=text` |
| `press_key(key)` | `type="key", key=key` |
| `scroll(direction, amount)` | `type="scroll", scroll_direction=..., scroll_amount=...` |
| `drag(x1, y1, x2, y2)` | `type="drag", x=x1, y=y1, end_x=x2, end_y=y2` |

---

## Part 4: Evaluation Strategy

### 4.1 Baseline-First Approach

Before fine-tuning, establish baselines with off-the-shelf models:

1. **GPT-4V** (via API) on WAA tasks
2. **Claude** (via API) on WAA tasks
3. **Qwen-VL base** (no fine-tuning) on WAA tasks

This establishes:
- Upper bound (what's possible with best available models)
- Gap to close (how much fine-tuning might help)
- Whether fine-tuning is even necessary for the domain

### 4.2 Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Task Success Rate** | Fraction of tasks completed | `successful_tasks / total_tasks` |
| **Step Accuracy** | Fraction of correct actions | `correct_steps / total_steps` |
| **Grounding Accuracy** | Clicks on correct element | `correct_clicks / total_clicks` |

### 4.3 Domain-Specific Evaluation

WAA tasks span 11 domains. Evaluate per-domain to identify strengths/weaknesses:

| Domain | Tasks | Description |
|--------|-------|-------------|
| browser | ~20 | Edge/Chrome navigation, settings |
| office | ~25 | Word, Excel, Outlook |
| coding | ~15 | VSCode, terminal |
| settings | ~15 | Windows Settings app |
| file_explorer | ~15 | File operations |
| notepad | ~10 | Text editing |
| paint | ~10 | Drawing operations |
| media | ~10 | Video/audio playback |
| clock | ~10 | Alarms, timers |
| edge | ~12 | Browser-specific |
| vscode | ~12 | IDE-specific |

---

## Part 5: Implementation Phases

### Phase 1: Foundation

**Deliverables:**
- `openadapt_ml/benchmarks/base.py` - BenchmarkAdapter interface
- `openadapt_ml/benchmarks/agent.py` - BenchmarkAgent interface
- `openadapt_ml/benchmarks/runner.py` - Evaluation runner
- Schema extensions in `openadapt_ml/schemas/sessions.py`

**Acceptance criteria:**
- Interfaces are importable and type-check
- Unit tests pass for dataclasses

### Phase 2: WAA Integration

**Deliverables:**
- `openadapt_ml/benchmarks/waa.py` - WAAAdapter implementation
- Local Windows VM setup documentation
- UIA tree extraction working
- Can run at least 10 WAA tasks end-to-end

**Acceptance criteria:**
- `WAAAdapter.list_tasks()` returns 154 tasks
- `WAAAdapter.reset()` initializes VM correctly
- `WAAAdapter.step()` executes actions
- `WAAAdapter.evaluate()` returns correct success/fail
- At least 1 task passes with a simple scripted agent

### Phase 3: Baseline Evaluation

**Deliverables:**
- API agent wrappers (GPT-4V, Claude)
- Baseline results on WAA tasks
- Analysis of failure modes

**Acceptance criteria:**
- Can run GPT-4V on all 154 WAA tasks
- Results match or exceed published SOTA (~19.5%)

---

## Appendix A: Benchmark Comparison

| Benchmark | Platform | Tasks | SOTA Success | Human Success | License |
|-----------|----------|-------|--------------|---------------|---------|
| WAA | Windows | 154 | 19.5% | 74.5% | MIT |
| OSWorld | Linux/Win/Mac | 369 | 12.2% | 72.4% | Apache-2.0 |
| WebArena | Browser | 812 | 35.8% | 78.2% | MIT |
| VisualWebArena | Browser | 910 | 16.4% | 88.7% | MIT |
| AndroidWorld | Android | 116 | 30.6% | ~100% | Apache-2.0 |

---

## Appendix B: References

- [Windows Agent Arena](https://github.com/microsoft/WindowsAgentArena)
- [OSWorld](https://os-world.github.io/)
- [WebArena](https://webarena.dev/)
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena)
- [UIPro](https://arxiv.org/html/2509.17328v1)
- [Ponder & Press](https://aclanthology.org/2025.findings-acl.76/)
