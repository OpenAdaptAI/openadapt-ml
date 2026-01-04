# WAAMockAdapter Evaluation Fix Design

## 1. Problem Statement

The `WAAMockAdapter.evaluate()` method uses non-deterministic random evaluation instead of evaluating actual agent actions. This prevents meaningful testing of agent behavior and produces unreliable benchmark results.

**Current code** (`openadapt_ml/benchmarks/waa.py:620-629`):
```python
def evaluate(self, task: BenchmarkTask) -> BenchmarkResult:
    # Random success for testing
    import random
    success = random.random() < 0.2  # ~20% success rate like WAA SOTA
    return BenchmarkResult(
        task_id=task.task_id,
        success=success,
        score=1.0 if success else 0.0,
        num_steps=self._step_count,
    )
```

**Issues:**
1. Success is random, not based on agent actions
2. Cannot validate that agents correctly parse and execute tasks
3. Test results are unreliable and non-reproducible
4. No way to verify task completion logic

## 2. Current Behavior Analysis

### Mock Environment State

The `WAAMockAdapter` simulates a simple UI with these elements:

```python
accessibility_tree={
    "role": "window",
    "name": "Mock Window",
    "children": [
        {"role": "button", "name": "OK", "id": "1"},
        {"role": "textfield", "name": "Input", "id": "2"},
        {"role": "button", "name": "Cancel", "id": "3"},
        {"role": "button", "name": "Submit", "id": "4"},
    ],
}
```

### Task Structure

Mock tasks are generated with:
- `task_id`: `"{domain}_{n}"` (e.g., `"browser_1"`)
- `instruction`: `"Mock task {n} in {domain} domain"`
- `time_limit_steps`: 15

### Action Flow

1. Agent receives observation with screenshot + accessibility tree
2. Agent parses task instruction and chooses action
3. `step()` tracks actions via `_step_count`
4. `done` when action type is "done" or step limit reached
5. `evaluate()` ignores actions entirely

## 3. Proposed Solution

### 3.1 Track Actions During Execution

Add state tracking in `WAAMockAdapter`:

```python
class WAAMockAdapter(BenchmarkAdapter):
    def __init__(self, num_tasks: int = 20, domains: list[str] | None = None):
        # ... existing init ...
        self._actions_taken: list[BenchmarkAction] = []
        self._typed_text: list[str] = []
        self._clicked_elements: list[str] = []

    def reset(self, task: BenchmarkTask) -> BenchmarkObservation:
        self._current_task = task
        self._step_count = 0
        self._actions_taken = []
        self._typed_text = []
        self._clicked_elements = []
        return self._mock_observation()

    def step(self, action: BenchmarkAction) -> tuple[BenchmarkObservation, bool, dict]:
        self._step_count += 1
        self._actions_taken.append(action)

        # Track action details
        if action.type == "type" and action.text:
            self._typed_text.append(action.text)
        if action.type == "click" and action.target_node_id:
            self._clicked_elements.append(action.target_node_id)

        done = action.type == "done" or self._step_count >= 15
        return self._mock_observation(), done, {"step": self._step_count}
```

### 3.2 Define Task-Specific Success Criteria

Create task templates with explicit success conditions:

```python
@dataclass
class MockTaskTemplate:
    """Template for mock tasks with success criteria."""
    task_id_prefix: str
    instruction: str
    domain: str
    success_criteria: Callable[[list[BenchmarkAction], list[str], list[str]], bool]

MOCK_TASK_TEMPLATES = {
    "submit_form": MockTaskTemplate(
        task_id_prefix="browser",
        instruction="Fill in the form and click Submit",
        domain="browser",
        success_criteria=lambda actions, typed, clicked: (
            len(typed) > 0 and  # Typed something
            "4" in clicked and  # Clicked Submit button (id="4")
            actions[-1].type == "done"  # Ended with DONE
        ),
    ),
    "click_ok": MockTaskTemplate(
        task_id_prefix="notepad",
        instruction="Click the OK button",
        domain="notepad",
        success_criteria=lambda actions, typed, clicked: (
            "1" in clicked and  # Clicked OK button (id="1")
            actions[-1].type == "done"
        ),
    ),
    "type_and_cancel": MockTaskTemplate(
        task_id_prefix="office",
        instruction="Type 'hello' in the input field and click Cancel",
        domain="office",
        success_criteria=lambda actions, typed, clicked: (
            "hello" in " ".join(typed).lower() and
            "3" in clicked and  # Clicked Cancel (id="3")
            actions[-1].type == "done"
        ),
    ),
}
```

### 3.3 Deterministic Evaluation

```python
def evaluate(self, task: BenchmarkTask) -> BenchmarkResult:
    """Evaluate based on actual actions taken."""

    # No actions taken = failure
    if not self._actions_taken:
        return BenchmarkResult(
            task_id=task.task_id,
            success=False,
            score=0.0,
            num_steps=self._step_count,
            reason="No actions taken",
        )

    # Find matching template
    template = self._get_task_template(task)

    if template:
        # Use template-specific criteria
        success = template.success_criteria(
            self._actions_taken,
            self._typed_text,
            self._clicked_elements,
        )
        reason = f"Template '{template.task_id_prefix}' evaluation"
    else:
        # Fallback: generic success criteria
        success = self._evaluate_generic(task)
        reason = "Generic evaluation"

    # Calculate partial score based on actions
    score = self._calculate_score()

    return BenchmarkResult(
        task_id=task.task_id,
        success=success,
        score=score if success else score * 0.5,  # Partial credit
        num_steps=self._step_count,
        reason=reason,
    )

def _evaluate_generic(self, task: BenchmarkTask) -> bool:
    """Generic evaluation for tasks without specific templates."""
    # Must end with DONE action
    if not self._actions_taken or self._actions_taken[-1].type != "done":
        return False

    # Must have taken at least one substantive action
    substantive_actions = [a for a in self._actions_taken if a.type in ("click", "type")]
    return len(substantive_actions) > 0

def _calculate_score(self) -> float:
    """Calculate score based on action quality."""
    if not self._actions_taken:
        return 0.0

    score = 0.0

    # Points for substantive actions
    for action in self._actions_taken:
        if action.type == "click" and action.target_node_id:
            score += 0.2  # Element-based click
        elif action.type == "click":
            score += 0.1  # Coordinate-based click
        elif action.type == "type" and action.text:
            score += 0.2
        elif action.type == "done":
            score += 0.1

    return min(1.0, score)  # Cap at 1.0
```

## 4. Success Criteria by Task Type

| Task Pattern | Required Actions | Success Condition |
|--------------|------------------|-------------------|
| "click {button}" | CLICK([id]) | Clicked correct element ID |
| "type {text}" | TYPE("text") | Typed expected text |
| "fill form and submit" | TYPE + CLICK([4]) | Typed in input + clicked Submit |
| Generic | Any substantive action + DONE | At least one click/type + DONE() |

### Element ID Mapping

| ID | Role | Name |
|----|------|------|
| 1 | button | OK |
| 2 | textfield | Input |
| 3 | button | Cancel |
| 4 | button | Submit |

## 5. Implementation Steps

### Phase 1: Core Changes
1. Add `_actions_taken`, `_typed_text`, `_clicked_elements` tracking to `WAAMockAdapter.__init__()`
2. Update `reset()` to clear tracking state
3. Update `step()` to record actions
4. Implement deterministic `evaluate()`

### Phase 2: Task Templates
1. Create `MockTaskTemplate` dataclass
2. Define templates for common task patterns
3. Match tasks to templates by instruction parsing
4. Use template-specific success criteria

### Phase 3: Enhanced Mock Tasks
1. Update `_generate_mock_tasks()` to use templates
2. Generate instructions that match success criteria
3. Add instruction variants for robustness testing

### File Changes

**Primary file**: `openadapt_ml/benchmarks/waa.py`
- Lines 557-629: `WAAMockAdapter` class

**Test file**: `tests/benchmarks/test_waa.py`
- Add tests for deterministic evaluation
- Add tests for success criteria

## 6. Testing Approach

### 6.1 Unit Tests

```python
class TestWAAMockAdapterEvaluation:
    """Tests for deterministic evaluation."""

    def test_evaluate_success_on_correct_actions(self):
        """Test that correct actions lead to success."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.load_task("browser_1")
        adapter.reset(task)

        # Simulate correct actions
        adapter.step(BenchmarkAction(type="click", target_node_id="2"))  # Click input
        adapter.step(BenchmarkAction(type="type", text="test"))           # Type
        adapter.step(BenchmarkAction(type="click", target_node_id="4"))  # Click Submit
        adapter.step(BenchmarkAction(type="done"))                        # Done

        result = adapter.evaluate(task)
        assert result.success is True
        assert result.score > 0.5

    def test_evaluate_failure_on_no_actions(self):
        """Test that no actions lead to failure."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.list_tasks()[0]
        adapter.reset(task)

        result = adapter.evaluate(task)
        assert result.success is False
        assert result.score == 0.0

    def test_evaluate_failure_without_done(self):
        """Test that missing DONE() leads to failure."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.list_tasks()[0]
        adapter.reset(task)

        adapter.step(BenchmarkAction(type="click", target_node_id="1"))

        result = adapter.evaluate(task)
        assert result.success is False

    def test_evaluate_deterministic(self):
        """Test that same actions always produce same result."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.list_tasks()[0]

        results = []
        for _ in range(10):
            adapter.reset(task)
            adapter.step(BenchmarkAction(type="click", target_node_id="1"))
            adapter.step(BenchmarkAction(type="done"))
            results.append(adapter.evaluate(task))

        # All results should be identical
        assert all(r.success == results[0].success for r in results)
        assert all(r.score == results[0].score for r in results)
```

### 6.2 Integration Tests

```python
def test_api_agent_with_deterministic_eval():
    """Test that APIBenchmarkAgent can achieve success on mock tasks."""
    adapter = WAAMockAdapter(num_tasks=3)
    agent = APIBenchmarkAgent(provider="anthropic")

    results = evaluate_agent_on_benchmark(agent, adapter, max_steps=10)

    # With proper parsing, agent should achieve >0% success
    success_rate = sum(r.success for r in results) / len(results)
    # Note: actual success depends on agent quality, but should be >0
    assert len(results) == 3
```

### 6.3 Manual Verification

```bash
# Test with scripted agent (guaranteed success)
uv run python -c "
from openadapt_ml.benchmarks import WAAMockAdapter, ScriptedAgent, BenchmarkAction, evaluate_agent_on_benchmark

adapter = WAAMockAdapter(num_tasks=3)
agent = ScriptedAgent(actions=[
    BenchmarkAction(type='click', target_node_id='2'),
    BenchmarkAction(type='type', text='test'),
    BenchmarkAction(type='click', target_node_id='4'),
    BenchmarkAction(type='done'),
])

results = evaluate_agent_on_benchmark(agent, adapter, max_steps=10)
for r in results:
    print(f'{r.task_id}: success={r.success}, score={r.score:.2f}')
"
```

## 7. Migration Notes

### Backward Compatibility

- Random evaluation behavior will be removed
- Tests relying on random success rate must be updated
- `test_evaluate_returns_result` should verify deterministic behavior

### Deprecation

The old random evaluation simulated WAA SOTA (~20% success rate). This was useful for testing the benchmark infrastructure but prevents validating agent correctness. The new deterministic evaluation replaces this entirely.

## References

- `openadapt_ml/benchmarks/waa.py`: WAAMockAdapter implementation
- `openadapt_ml/benchmarks/agent.py`: APIBenchmarkAgent parses VLM responses
- `openadapt_ml/benchmarks/base.py`: BenchmarkAction, BenchmarkResult schemas
- `tests/benchmarks/test_waa.py`: Existing mock adapter tests
