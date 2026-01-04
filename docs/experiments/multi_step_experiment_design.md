# Multi-Step Demo-Conditioning Experiment Design

## Executive Summary

This document presents an improved experimental design for validating demo-conditioned prompting that addresses the three key limitations of the current n=45 experiment:
1. **Single screenshot problem**: All 45 tasks were evaluated on the same screenshot (step 0)
2. **Shared first action problem**: All 45 tasks have the same correct first action (click Apple menu)
3. **First-action only evaluation**: No assessment of multi-step trajectory following

## Analysis of Current Experiment Limitations

### Current State
The existing experiment (`scripts/run_demo_experiment_n30.py`) demonstrates a 53.3 percentage point improvement (46.7% to 100%) with demo-conditioning. However:

| Limitation | Impact |
|------------|--------|
| Single screenshot (step_0.png) | Cannot measure whether demo helps at different UI states |
| Same first action | Cannot distinguish demo transfer from spatial memorization |
| First-action only | Cannot measure trajectory coherence or multi-step success |

### Available Data
From `/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift/`:
- **20 screenshots** (step_0 through step_19)
- **13 click events** in the recording
- **5 distinct visual states**: Terminal, System Settings sidebar, Displays panel, Night Shift popup, Schedule dropdown

## Experiment Goals and Hypotheses

### Primary Hypothesis (H1)
Demo-conditioning improves action accuracy across multiple steps, not just the first action.

### Secondary Hypotheses
- **H2**: Demo-conditioning benefit varies by visual context complexity
- **H3**: Demo-conditioning helps more when UI state differs from demo screenshots
- **H4**: Multi-step trajectory coherence is higher with demo-conditioning

### Null Hypothesis (H0)
Demo-conditioning only helps on first action; subsequent actions show no improvement.

## Test Case Design

### Design Principle: Diverse Entry Points
To eliminate the "same first action" confound, we design test cases that start from different screenshots (different UI states) with different correct first actions.

### Screenshot-to-Action Mapping (from capture data)

| Screenshot | Visual State | Example Tasks Starting Here |
|------------|--------------|----------------------------|
| step_0 | Terminal (initial desktop) | "Turn off Night Shift" (full workflow) |
| step_1 | System Settings > Displays | "Click Night Shift button" |
| step_7 | System Settings > General | "Navigate to Displays" |
| step_8 | Screen Saver panel | "Go to Displays panel" |
| step_9 | Displays panel (pre-popup) | "Open Night Shift settings" |
| step_10 | Night Shift popup (Sunset schedule) | "Change schedule to Off" |
| step_11 | Night Shift dropdown open | "Select Off option" |
| step_12 | Night Shift popup (Off selected) | "Click Done button" |

### Test Case Categories

#### Category A: Full Workflow (from step_0)
Starting from Terminal screenshot, complete multi-step tasks:

```
A1: "Turn off Night Shift" (5+ steps)
A2: "Turn on Night Shift" (5+ steps)
A3: "Set Night Shift to sunset schedule" (5+ steps)
A4: "Adjust Night Shift color temperature" (5+ steps)
```

**Correct first action**: Click in System Settings sidebar or use menu

#### Category B: Mid-Workflow (from System Settings)
Starting from step_7 (General panel) or step_8 (Screen Saver):

```
B1: "Navigate to Displays panel" (start: General)
B2: "Open Night Shift settings" (start: Screen Saver)
B3: "Click on Displays in sidebar" (start: General)
```

**Correct first action**: Click "Displays" in sidebar

#### Category C: Final Steps (from Night Shift popup)
Starting from step_10 (Night Shift popup visible):

```
C1: "Turn off Night Shift schedule" (start: popup with Sunset active)
C2: "Click the Schedule dropdown" (start: popup)
C3: "Change color temperature to warmer" (start: popup)
```

**Correct first action**: Varies - dropdown click, slider drag, or Done button

#### Category D: Different Panels (transfer test)
Tasks that go to different settings areas:

```
D1: "Enable True Tone display" (start: Displays panel)
D2: "Change screen brightness" (start: Displays panel)
D3: "View screen resolution" (start: Displays panel)
```

**Correct first action**: Different from Night Shift workflow

### Diversity Matrix

| Category | Start Screenshot | Correct First Action | Tests |
|----------|-----------------|---------------------|-------|
| A (Full) | step_0 | Menu bar or sidebar | 4 |
| B (Mid) | step_7, step_8 | Sidebar click | 3 |
| C (Final) | step_10, step_11 | Popup controls | 3 |
| D (Transfer) | step_1, step_9 | Different UI elements | 3 |

**Total: 13 test cases with 4+ distinct first actions**

## Evaluation Metrics

### Step-Level Metrics

1. **Step Accuracy**: Proportion of steps where predicted action matches ground truth
   ```
   step_accuracy = correct_steps / total_steps
   ```

2. **Position Error**: Euclidean distance between predicted and actual click coordinates
   ```
   position_error = sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2)
   ```

3. **Action Type Accuracy**: Whether the action type (click, type, scroll) is correct

### Trajectory-Level Metrics

4. **Task Completion Rate**: Proportion of tasks where all N steps are correct
   ```
   completion_rate = fully_correct_tasks / total_tasks
   ```

5. **Prefix Accuracy**: Longest correct prefix of steps before first error
   ```
   prefix_length = steps_until_first_error
   ```

6. **Recovery Rate**: Whether model gets back on track after an error

### Aggregate Metrics

7. **Cumulative Step Accuracy by Position**:
   ```
   step_1_acc = correct_step1 / total_tasks
   step_2_acc = correct_step2 / tasks_that_reached_step2
   ...
   ```

8. **Demo Transfer Score**: Performance delta between tasks similar vs. dissimilar to demo

## Implementation Approach

### Phase 1: Multi-Screenshot Test Framework

Modify the existing experiment to:
1. Accept a list of `(screenshot_path, task, expected_action)` tuples
2. Support evaluating multiple steps per task
3. Track per-step metrics

Key code changes to `openadapt_ml/experiments/demo_prompt/run_experiment.py`:

```python
@dataclass
class MultiStepTestCase:
    """Test case for multi-step evaluation."""
    name: str
    task: str
    start_screenshot: str  # Path to starting screenshot
    expected_actions: list[dict]  # [{type, x, y, ...}, ...]
    category: str  # A, B, C, or D

@dataclass
class MultiStepResult:
    """Result of multi-step evaluation."""
    test_case: MultiStepTestCase
    condition: str  # zero_shot, with_demo, control
    predicted_actions: list[str]
    step_accuracies: list[bool]
    position_errors: list[float]
    prefix_length: int
```

### Phase 2: Sequential Screenshot Execution

For true multi-step evaluation:
1. Start with the designated screenshot
2. Get model's predicted action
3. Find the **next screenshot** in the sequence (simulating execution)
4. Repeat for N steps

Screenshot transition map:
```python
SCREENSHOT_TRANSITIONS = {
    ("step_0", "CLICK(sidebar_displays)"): "step_1",
    ("step_1", "CLICK(night_shift_button)"): "step_10",
    ("step_10", "CLICK(schedule_dropdown)"): "step_11",
    ("step_11", "CLICK(off_option)"): "step_12",
    ("step_12", "CLICK(done_button)"): "step_13",
}
```

### Phase 3: Ground Truth Construction

Extract action sequences from the capture data:

```python
def extract_ground_truth_actions(capture_path: str) -> list[dict]:
    """Extract the ground truth action sequence from a capture."""
    episode = capture_to_episode(capture_path)

    actions = []
    for step in episode.steps:
        if step.action.type != "done":
            actions.append({
                "type": step.action.type,
                "x": step.action.x,
                "y": step.action.y,
                "screenshot": step.observation.image_path,
            })
    return actions
```

## Detailed Test Cases

### Category A: Full Workflow Tests

**A1: Turn off Night Shift (5 steps)**
```yaml
name: full_workflow_off
task: "Turn off Night Shift in macOS System Settings"
start_screenshot: step_0
expected_steps:
  - screenshot: step_0
    action: {type: click, description: "Click Displays in sidebar"}
  - screenshot: step_1
    action: {type: click, description: "Click Night Shift button"}
  - screenshot: step_10
    action: {type: click, description: "Click Schedule dropdown"}
  - screenshot: step_11
    action: {type: click, description: "Select Off"}
  - screenshot: step_12
    action: {type: click, description: "Click Done"}
```

### Category B: Mid-Workflow Tests

**B1: Navigate to Displays**
```yaml
name: mid_nav_displays
task: "Navigate to the Displays settings panel"
start_screenshot: step_7  # General panel
expected_steps:
  - screenshot: step_7
    action: {type: click, description: "Click Displays in sidebar"}
```

### Category C: Final Step Tests

**C1: Turn off schedule**
```yaml
name: final_turn_off
task: "Set the Night Shift schedule to Off"
start_screenshot: step_10  # Night Shift popup visible
expected_steps:
  - screenshot: step_10
    action: {type: click, description: "Click Schedule dropdown"}
  - screenshot: step_11
    action: {type: click, description: "Select Off option"}
```

### Category D: Transfer Tests

**D1: Enable True Tone**
```yaml
name: transfer_true_tone
task: "Enable True Tone display"
start_screenshot: step_1  # Displays panel
expected_steps:
  - screenshot: step_1
    action: {type: click, description: "Click True Tone toggle"}
```

## Using Existing Capture Data Effectively

### What We Have
The Night Shift recording provides:
- 20 screenshots covering the full workflow
- 13 distinct click actions
- 5+ distinct UI states

### How to Maximize Utility

1. **Use different screenshots as starting points**: Test from step_0, step_1, step_7, step_8, step_9, step_10

2. **Create sub-workflows**: Instead of always testing full workflow, test:
   - Steps 1-3 only (navigation)
   - Steps 3-5 only (final interaction)
   - Single step (action selection)

3. **Counterfactual tasks**: Use same screenshots but different tasks
   - From Displays panel: "Turn on True Tone" vs "Open Night Shift" vs "Change brightness"

4. **Leverage existing screenshots for demo images**: Include actual demo screenshots in the prompt

## Expected Outcomes

### Best Case (H1 confirmed)
- Demo-conditioned: 80%+ multi-step accuracy
- Zero-shot: <50% multi-step accuracy
- Clear benefit at all step positions

### Moderate Case (partial confirmation)
- Demo helps on steps 1-2, diminishes for later steps
- Benefit concentrated on navigation, not final interactions

### Null Case (H0 not rejected)
- Demo only helps first action, no multi-step benefit
- Zero-shot catches up after first step

## File Structure

```
openadapt_ml/experiments/multi_step_demo/
    __init__.py
    test_cases.py          # TestCase definitions
    evaluator.py           # MultiStepEvaluator class
    metrics.py             # Metric computation
    run_experiment.py      # Main runner

scripts/
    run_multi_step_experiment.py  # CLI entry point

docs/experiments/
    multi_step_experiment_design.md  # This document
    multi_step_experiment_results.md # Results (future)
```

## Summary

This design addresses all three limitations of the current experiment:

| Limitation | Solution |
|------------|----------|
| Single screenshot | Test cases starting from 6+ different screenshots |
| Same first action | 4+ distinct correct first actions across categories |
| First-action only | Multi-step evaluation with trajectory metrics |

The experiment uses the existing Night Shift recording data creatively by treating different screenshots as "entry points" and creating sub-workflow test cases.
