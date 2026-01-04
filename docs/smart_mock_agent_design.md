# Smart Mock Agent Design for WAA Benchmark Validation

## Problem Statement

Running mock adapter tests with `RandomAgent` produces a **0% success rate**:

```
Success rate: 0/20 (0.0%)
clicked=[], typed=True, done=True
```

This is **expected but not useful** for pipeline validation:

1. **RandomAgent behavior**: Types random text ("test") but clicks at random coordinates (0.0-1.0 normalized), never targeting specific UI elements
2. **Mock adapter evaluation**: Requires clicking specific elements by ID (Submit=4, OK=1) or coordinate-to-element mapping
3. **Mismatch**: Random clicks don't resolve to element IDs, so `clicked_ids` is always empty

The evaluation pipeline works correctly - it correctly identifies RandomAgent's actions as failures. But this doesn't validate that a **real agent** could succeed, nor does it test the full evaluation flow end-to-end.

## What the Mock Adapter Evaluates

From `WAAMockAdapter.evaluate()` in `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa.py`:

```python
# Success criteria:
# 1. Clicked Submit (ID 4) - primary success path
# 2. Typed something AND clicked OK (ID 1) - form submission path
# 3. Called DONE after at least 2 actions - reasonable completion
clicked_submit = "4" in clicked_ids
clicked_ok = "1" in clicked_ids
form_submitted = typed_text and clicked_ok
reasonable_completion = called_done and len(self._actions) >= 2
```

The mock UI has these elements:
| ID | Role | Name |
|----|------|------|
| 1 | button | OK |
| 2 | textfield | Input |
| 3 | button | Cancel |
| 4 | button | Submit |

## Solution: Smart Mock Agent

Create an agent that **knows how to succeed** on the mock adapter's evaluation criteria. This validates:
- Action routing works (click, type, done)
- Element targeting works (target_node_id)
- Evaluation criteria are correctly checked
- The full pipeline produces expected results

### Option A: ScriptedMockAgent

A hardcoded agent that always performs the "winning" action sequence:

```python
class ScriptedMockAgent(BenchmarkAgent):
    """Agent that performs actions guaranteed to pass mock evaluation."""

    def __init__(self):
        self._step = 0
        # Actions that satisfy mock adapter evaluation:
        # Click input field, type text, click Submit, done
        self._actions = [
            BenchmarkAction(type="click", target_node_id="2"),  # Click Input field
            BenchmarkAction(type="type", text="test input"),    # Type something
            BenchmarkAction(type="click", target_node_id="4"),  # Click Submit
            BenchmarkAction(type="done"),                        # Finish
        ]

    def act(self, observation, task, history=None):
        if self._step < len(self._actions):
            action = self._actions[self._step]
            self._step += 1
            return action
        return BenchmarkAction(type="done")

    def reset(self):
        self._step = 0
```

**Pros**: Simple, deterministic, 100% success rate on mock adapter
**Cons**: Doesn't exercise observation parsing

### Option B: ObservationAwareMockAgent

An agent that reads the accessibility tree and chooses correct actions:

```python
class ObservationAwareMockAgent(BenchmarkAgent):
    """Agent that parses observations to find correct elements."""

    def __init__(self):
        self._done_actions = ["type", "click_submit"]
        self._current_action = 0

    def act(self, observation, task, history=None):
        a11y = observation.accessibility_tree
        if not a11y:
            return BenchmarkAction(type="done")

        # Find elements
        elements = self._flatten_tree(a11y)
        input_field = next((e for e in elements if e.get("role") == "textfield"), None)
        submit_btn = next((e for e in elements if e.get("name") == "Submit"), None)

        if self._current_action == 0 and input_field:
            self._current_action += 1
            return BenchmarkAction(type="type", text="test", target_node_id=input_field.get("id"))

        if self._current_action == 1 and submit_btn:
            self._current_action += 1
            return BenchmarkAction(type="click", target_node_id=submit_btn.get("id"))

        return BenchmarkAction(type="done")

    def _flatten_tree(self, node, result=None):
        """Flatten accessibility tree to list of elements."""
        if result is None:
            result = []
        result.append(node)
        for child in node.get("children", []):
            self._flatten_tree(child, result)
        return result

    def reset(self):
        self._current_action = 0
```

**Pros**: Exercises observation parsing, more realistic
**Cons**: More complex, still mock-specific

### Option C: DemoConditionedMockAgent (Recommended)

An agent that uses a "demo trajectory" to guide actions - this validates the demo-conditioned prompting approach:

```python
class DemoConditionedMockAgent(BenchmarkAgent):
    """Agent that follows a demo trajectory for mock tasks.

    This validates the demo-conditioned prompting approach works
    by replaying a known-good trajectory.
    """

    # Demo trajectory for mock tasks
    DEMO_TRAJECTORY = {
        "instruction": "Mock task in browser domain",
        "actions": [
            {"type": "click", "target_node_id": "2", "reasoning": "Click input field"},
            {"type": "type", "text": "hello world", "reasoning": "Enter text"},
            {"type": "click", "target_node_id": "4", "reasoning": "Click Submit"},
            {"type": "done", "reasoning": "Task complete"},
        ]
    }

    def __init__(self, demo_trajectory=None):
        self._demo = demo_trajectory or self.DEMO_TRAJECTORY
        self._step = 0

    def act(self, observation, task, history=None):
        if self._step < len(self._demo["actions"]):
            action_spec = self._demo["actions"][self._step]
            self._step += 1
            return BenchmarkAction(
                type=action_spec["type"],
                target_node_id=action_spec.get("target_node_id"),
                text=action_spec.get("text"),
                raw_action={"reasoning": action_spec.get("reasoning")},
            )
        return BenchmarkAction(type="done")

    def reset(self):
        self._step = 0
```

**Pros**:
- Validates demo-conditioned approach end-to-end
- Can load real demo trajectories for more complex testing
- Demonstrates the pattern for production use

## Implementation Plan

### Step 1: Add SmartMockAgent to agent.py

File: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/agent.py`

Add after `RandomAgent` class (~line 392):

```python
class SmartMockAgent(BenchmarkAgent):
    """Agent designed to pass WAAMockAdapter evaluation.

    Performs a fixed sequence of actions that satisfy the mock adapter's
    success criteria. Use for validating the benchmark pipeline locally.

    Args:
        variant: Action sequence variant ("submit", "ok", "minimal").
    """

    VARIANTS = {
        "submit": [
            BenchmarkAction(type="click", target_node_id="2"),  # Click Input
            BenchmarkAction(type="type", text="test input"),
            BenchmarkAction(type="click", target_node_id="4"),  # Click Submit
            BenchmarkAction(type="done"),
        ],
        "ok": [
            BenchmarkAction(type="type", text="hello"),
            BenchmarkAction(type="click", target_node_id="1"),  # Click OK
            BenchmarkAction(type="done"),
        ],
        "minimal": [
            BenchmarkAction(type="click", target_node_id="4"),  # Just Submit
            BenchmarkAction(type="done"),
        ],
    }

    def __init__(self, variant: str = "submit"):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Options: {list(self.VARIANTS.keys())}")
        self._actions = self.VARIANTS[variant]
        self._step = 0

    def act(self, observation, task, history=None):
        if self._step < len(self._actions):
            action = self._actions[self._step]
            self._step += 1
            return action
        return BenchmarkAction(type="done")

    def reset(self):
        self._step = 0
```

### Step 2: Export in __init__.py

File: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/__init__.py`

Add to imports and `__all__`:
```python
from openadapt_ml.benchmarks.agent import SmartMockAgent
__all__ = [..., "SmartMockAgent"]
```

### Step 3: Add CLI Command

File: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py`

Add command:
```python
@cli.command()
@click.option("--tasks", default=5, help="Number of tasks")
@click.option("--variant", default="submit", help="Action variant: submit, ok, minimal")
def test-smart(tasks: int, variant: str):
    """Test mock adapter with SmartMockAgent (expected 100% success)."""
    from openadapt_ml.benchmarks import SmartMockAgent, WAAMockAdapter, evaluate_agent_on_benchmark

    adapter = WAAMockAdapter(num_tasks=tasks)
    agent = SmartMockAgent(variant=variant)
    results = evaluate_agent_on_benchmark(agent, adapter, max_steps=10)

    success_count = sum(1 for r in results if r.success)
    print(f"Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.0f}%)")

    if success_count != len(results):
        print("WARNING: Expected 100% success with SmartMockAgent")
        for r in results:
            if not r.success:
                print(f"  FAIL {r.task_id}: {r.reason}")
```

### Step 4: Add Tests

File: `/Users/abrichr/oa/src/openadapt-ml/tests/benchmarks/test_waa.py`

Add test class:
```python
class TestSmartMockAgent:
    """Tests for SmartMockAgent with mock adapter."""

    def test_submit_variant_succeeds(self):
        """Test submit variant achieves 100% success."""
        adapter = WAAMockAdapter(num_tasks=5)
        agent = SmartMockAgent(variant="submit")
        results = evaluate_agent_on_benchmark(agent, adapter, max_steps=10)

        assert len(results) == 5
        assert all(r.success for r in results), "SmartMockAgent should always succeed"

    def test_ok_variant_succeeds(self):
        """Test ok variant achieves 100% success."""
        adapter = WAAMockAdapter(num_tasks=3)
        agent = SmartMockAgent(variant="ok")
        results = evaluate_agent_on_benchmark(agent, adapter, max_steps=10)

        assert all(r.success for r in results)

    def test_minimal_variant_succeeds(self):
        """Test minimal variant achieves 100% success."""
        adapter = WAAMockAdapter(num_tasks=3)
        agent = SmartMockAgent(variant="minimal")
        results = evaluate_agent_on_benchmark(agent, adapter, max_steps=10)

        assert all(r.success for r in results)
```

## Value Proposition

| Benefit | Description |
|---------|-------------|
| **Local validation** | Test entire evaluation pipeline without Azure VMs or API costs |
| **Fast iteration** | Validate changes to evaluation logic in seconds |
| **Deterministic** | Known success rate (100%) for regression testing |
| **Pipeline verification** | Confirms action routing, element targeting, and evaluation work |
| **Demo approach validation** | Proves demo-conditioned pattern works end-to-end |

## Testing Verification

After implementation, run:

```bash
# Should show 100% success
uv run python -m openadapt_ml.benchmarks.cli test-smart --tasks 10

# Compare with RandomAgent (should be ~0%)
uv run python -m openadapt_ml.benchmarks.cli test-mock --tasks 10

# Run unit tests
uv run pytest tests/benchmarks/test_waa.py::TestSmartMockAgent -v
```

Expected output:
```
test-smart:  Success rate: 10/10 (100%)
test-mock:   Success rate: 0/10 (0%)
```

## Relationship to Real Evaluation

The SmartMockAgent validates the **pipeline mechanics**, not agent intelligence:

| Component | Mock Testing | Real WAA Testing |
|-----------|--------------|------------------|
| Action types | Click, type, done | Same |
| Element targeting | target_node_id | Same |
| Observation format | Mock a11y tree | Real UIA tree |
| Evaluation | Scripted criteria | WAA evaluators |
| Success rate | 100% (by design) | ~20% SOTA |

Once SmartMockAgent shows 100% success, you can trust that pipeline bugs aren't causing low success rates on real benchmarks.
