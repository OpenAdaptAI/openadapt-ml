# Demo-in-Prompt Experiment Plan

## Objective

Test whether including a human demonstration in the prompt improves VLM agent performance on similar tasks.

**Core Hypothesis**: If a VLM sees a step-by-step demonstration of "Turn off Night Shift", it should perform better on variations like "Turn on Night Shift" or "Adjust Night Shift schedule".

## Experiment Design

### Phase 1: Extract Demo from Existing Capture

**Source**: macOS screen recording of Night Shift settings toggle

**Contents**:
- 22 screenshots (steps)
- 13 click actions
- 16 key events
- Task: "Turn off Night Shift" in macOS System Settings

**Extraction**:
```python
from openadapt_ml.ingest.capture import capture_to_episode

episode = capture_to_episode(
    "/path/to/capture",
    goal="Turn off Night Shift"
)
```

### Phase 2: Format Demo for Few-Shot Prompting

Convert the Episode into a prompt format:

```
DEMONSTRATION:
The following shows how to navigate macOS Settings.

Step 1:
[Screenshot 1]
ACTION: CLICK(0.05, 0.02)  # Click Apple menu

Step 2:
[Screenshot 2]
ACTION: CLICK(0.15, 0.25)  # Click System Settings

Step 3:
[Screenshot 3]
ACTION: CLICK(0.12, 0.45)  # Click Displays
...

---
NOW PERFORM THIS TASK:
Goal: Turn ON Night Shift
[Current Screenshot]
What is the next action?
```

### Phase 3: Define Test Cases

| Demo Task | Test Task (Variation) | Similarity |
|-----------|----------------------|------------|
| Turn off Night Shift | Turn ON Night Shift | Near (toggle) |
| Turn off Night Shift | Adjust Night Shift schedule | Medium (same panel) |
| Turn off Night Shift | Turn off True Tone | Far (different setting) |

### Phase 4: Run Comparisons

For each test task, run:

1. **Zero-shot baseline**: Just the task instruction + current screenshot
2. **With demo**: Task instruction + demo trajectory + current screenshot

### Phase 5: Metrics

- **Task completion**: Did the model reach the goal state?
- **Action accuracy**: Did it take the correct next action at each step?
- **Step efficiency**: How many steps to complete vs. optimal?

## Implementation

### File Structure

```
openadapt_ml/
├── experiments/
│   └── demo_prompt/
│       ├── __init__.py
│       ├── run_experiment.py      # Main experiment runner
│       ├── format_demo.py         # Demo formatting utilities
│       └── results/               # Output directory
```

### Key Components

#### 1. Demo Formatter (`format_demo.py`)

```python
def format_episode_as_demo(episode: Episode, max_steps: int = 10) -> str:
    """Convert Episode to few-shot demo format."""
    lines = ["DEMONSTRATION:", ""]

    for i, step in enumerate(episode.steps[:max_steps]):
        lines.append(f"Step {i+1}:")
        # Include screenshot reference
        lines.append(f"[Screenshot: {step.observation.screenshot_path}]")
        # Format action
        action_str = format_action(step.action)
        lines.append(f"ACTION: {action_str}")
        lines.append("")

    return "\n".join(lines)
```

#### 2. Test Runner (`run_experiment.py`)

```python
class DemoPromptExperiment:
    def __init__(self, provider: str = "anthropic"):
        self.adapter = ApiVLMAdapter(provider=provider)

    def run_zero_shot(self, task: str, screenshot_path: str) -> str:
        """Run without demo context."""
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Goal: {task}\nWhat is the next action?"},
            ],
            "images": [screenshot_path],
        }
        return self.adapter.generate(sample)

    def run_with_demo(self, task: str, screenshot_path: str, demo: str) -> str:
        """Run with demo context."""
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{demo}\n---\nNOW: {task}\nWhat is the next action?"},
            ],
            "images": [screenshot_path],  # + demo screenshots
        }
        return self.adapter.generate(sample)
```

### CLI Command

```bash
# Run the experiment
uv run python -m openadapt_ml.experiments.demo_prompt.run_experiment \
    --demo-capture /path/to/turn-off-nightshift \
    --test-task "Turn on Night Shift" \
    --provider anthropic \
    --output results/
```

## Execution Plan

### Step 1: Create experiment directory (5 min)
```bash
mkdir -p openadapt_ml/experiments/demo_prompt
```

### Step 2: Implement demo formatter (15 min)
- Extract Episode from capture
- Format as few-shot prompt
- Handle screenshot embedding (base64 or paths)

### Step 3: Implement test runner (20 min)
- Zero-shot baseline
- With-demo variant
- Results logging

### Step 4: Run on 3 test cases (30 min)
- Near variation (toggle same setting)
- Medium variation (same panel, different setting)
- Far variation (different panel)

### Step 5: Analyze results (20 min)
- Compare zero-shot vs with-demo
- Identify patterns
- Document findings

## Expected Outcomes

### If demo helps significantly:
- Validates OpenAdapt's core value proposition
- Next step: Invest in training on demonstrations
- Consider: How many demos needed? How similar must they be?

### If demo doesn't help:
- Investigate why:
  - Is the demo format wrong?
  - Is the variation too different?
  - Is the model not learning from in-context examples?
- Consider: Need different prompting strategy or actual fine-tuning

## Constraints

- **No Azure VM needed**: Run locally with screenshots
- **No training**: Pure prompting experiment
- **Cost**: ~$0.50 per test case (3 cases = ~$1.50)
- **Time**: ~2 hours total

## Success Criteria

**Minimum viable signal**: With-demo performs noticeably better than zero-shot on at least 2/3 test cases.

**Strong signal**: With-demo completes tasks that zero-shot fails entirely.

## Notes

- Claude Sonnet 4.5 supports up to 20 images per request
- We can include multiple demo screenshots in context
- Start with text descriptions of actions, add images if helpful
