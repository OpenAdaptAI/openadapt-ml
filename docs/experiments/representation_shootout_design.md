# Representation Shootout Experiment Design

**Date**: January 2026
**Author**: OpenAdapt Team
**Status**: Framework Complete - Pending Model Integration

## Executive Summary

This document defines the **Representation Shootout** experiment comparing three approaches for GUI action prediction under distribution drift conditions:

1. **Condition A: Raw Coordinates** - Direct coordinate regression
2. **Condition B: Coordinates + Visual Cues** - Enhanced coordinate regression with visual markers
3. **Condition C: Marks (Element IDs)** - Element classification using SoM-style markers

The experiment determines which representation is most robust to common deployment scenarios where the UI differs from training data (resolution changes, window movement, theme changes, scroll offsets).

---

## Motivation

GUI automation agents must predict where to click or interact. Two fundamental approaches exist:

### Coordinate-Based Approaches (A, B)
- **Output**: `{"type": "CLICK", "x": 0.42, "y": 0.31}` (normalized coordinates)
- **Advantages**: Works without element detection, simpler training pipeline
- **Challenges**: Brittle to resolution changes, window movement, layout shifts

### Element-Based Approach (C)
- **Output**: `{"type": "CLICK", "element_id": "e17"}` (element reference)
- **Advantages**: Resolution-independent, robust to position changes
- **Challenges**: Requires element detection/grounding pipeline, may fail if elements not detected

### Research Question

> Under what conditions do coordinate-based approaches match or exceed element-based approaches in robustness to UI drift?

The hypothesis is that visual cues (Condition B) can bridge the gap by providing the model with explicit visual anchors at the target location during training.

---

## Experimental Conditions

### Condition A: Raw Coordinates

**Input**:
- Screenshot (unmodified)
- Goal instruction (text)
- Action history (optional)

**Output**:
```json
{"type": "CLICK", "x": 0.423, "y": 0.156}
```

**Training Signal**: Mean squared error (MSE) between predicted and ground-truth coordinates.

**Prompt Structure**:
```
GOAL: {instruction}

PREVIOUS ACTIONS:
1. CLICK(0.5, 0.3)
2. TYPE("username")

What is the next action?
```

### Condition B: Coordinates + Visual Cues

**Input**:
- Screenshot with visual markers:
  - Red circular marker at click target
  - Zoomed inset patch (2x magnification, 100x100 px) at target location
- Goal instruction (text)
- Action history (optional)

**Output**:
```json
{"type": "CLICK", "x": 0.423, "y": 0.156}
```

**Training Signal**: MSE between predicted and ground-truth coordinates.

**Augmentation Details**:
- Red marker: 8px radius circle, RGB(255, 0, 0)
- Zoomed patch: Positioned in corner opposite to click location
- Both overlays applied to training images only (not test images)

**Prompt Structure**:
```
GOAL: {instruction}

The red marker and zoomed inset show the target click location.
Learn to identify this location based on the UI context.

What is the next action?
```

### Condition C: Marks (Element IDs)

**Input**:
- Screenshot with Set-of-Marks (SoM) overlay
- UIElementGraph (structured element list with IDs, roles, names, bboxes)
- Goal instruction (text)

**Output**:
```json
{"type": "CLICK", "element_id": "e17"}
```

**Training Signal**: Cross-entropy loss for element classification.

**UIElementGraph Format**:
```json
{
  "elements": [
    {"id": "e1", "role": "button", "name": "Submit", "bbox": [0.4, 0.8, 0.6, 0.85]},
    {"id": "e17", "role": "textfield", "name": "Username", "bbox": [0.3, 0.4, 0.7, 0.45]},
    ...
  ]
}
```

**Prompt Structure**:
```
GOAL: {instruction}

UI ELEMENTS:
[e1] button "Submit" at (0.4, 0.8)-(0.6, 0.85)
[e17] textfield "Username" at (0.3, 0.4)-(0.7, 0.45)
...

Which element should be clicked?
```

---

## Drift Evaluation Conditions

All models are trained on a canonical dataset (1920x1080, centered window, light theme, no scroll). Evaluation tests robustness to:

### 1. Resolution Scaling

| Scale | Resolution | Test Description |
|-------|------------|------------------|
| 0.75x | 1440x810   | Downscaled UI    |
| 1.0x  | 1920x1080  | Original (control) |
| 1.25x | 2400x1350  | Upscaled UI      |
| 1.5x  | 2880x1620  | 4K-like scaling  |

**Implementation**: Resize screenshot, scale all coordinates/bboxes proportionally.

### 2. Window Translation

| Offset | Description |
|--------|-------------|
| (0, 0) | Original position (control) |
| (+200, 0) | Shifted right 200px |
| (0, +100) | Shifted down 100px |
| (+200, +100) | Diagonal shift |

**Implementation**: Add offset to all coordinates, crop/pad screenshot accordingly.

### 3. UI Drift (Theme Changes)

| Theme | Description |
|-------|-------------|
| Light | Original light theme (control) |
| Dark  | Dark mode equivalent |
| High Contrast | Accessibility theme |

**Implementation**: Use pre-recorded theme variants or synthetic theme transformation.

### 4. Scroll Offset

| Offset | Description |
|--------|-------------|
| 0px    | Top of page (control) |
| 300px  | Scrolled down 300px |
| 600px  | Scrolled down 600px |

**Implementation**: Adjust element bboxes based on scroll, provide different scroll states of same page.

---

## Metrics

### Primary Metrics

#### 1. Click-Hit Rate
```
hit_rate = (clicks_within_target_bbox) / (total_clicks)
```
A click is a "hit" if the predicted coordinate falls within the target element's bounding box.

#### 2. Grounding Top-1 Accuracy (Condition C only)
```
accuracy = (correct_element_id_predictions) / (total_predictions)
```

#### 3. Episode Success Rate
```
success_rate = (episodes_reaching_goal) / (total_episodes)
```
Requires multi-step execution with success criteria.

### Secondary Metrics

#### 4. Coordinate Distance to Target
```
distance = sqrt((pred_x - target_x)^2 + (pred_y - target_y)^2)
```
Measured in normalized coordinates (0-1 scale). Lower is better.

#### 5. Robustness Score
```
robustness = metric_under_drift / metric_at_canonical
```
Ratio of performance under drift vs. canonical conditions. 1.0 = no degradation.

---

## Decision Rule

The experiment produces a recommendation based on comparative performance:

```python
TOLERANCE = 0.05  # 5 percentage points

# Calculate average metrics across all drift conditions
marks_avg = mean(marks_results[drift_condition] for drift_condition in ALL_DRIFTS)
coords_cues_avg = mean(coords_cues_results[drift_condition] for drift_condition in ALL_DRIFTS)

# Decision logic
if coords_cues_avg >= marks_avg - TOLERANCE:
    recommendation = "COORDINATES"
    reason = f"Coords+Cues within {TOLERANCE*100}% of Marks under drift"
else:
    recommendation = "MARKS"
    reason = f"Marks outperforms Coords+Cues by >{TOLERANCE*100}% under drift"
```

### Rationale

- **If Coordinates+Cues is within 5% of Marks** → Choose Coordinates
  - Simpler deployment (no grounding pipeline)
  - No dependency on element detection quality
  - Easier to collect training data (just screenshots + clicks)

- **If Marks exceeds by >5%** → Choose Marks
  - Robustness benefit justifies additional complexity
  - Investment in grounding pipeline worthwhile

---

## Dataset Requirements

### Training Data (Canonical)
- Resolution: 1920x1080
- Window: Centered, default size
- Theme: Light mode
- Scroll: Top of page
- N samples: Minimum 1000 click actions

### Evaluation Data (Per Drift Condition)
- At least 100 samples per drift condition
- Same UI elements/tasks as training (different drift)
- Ground truth: Element ID and click coordinates

### Recommended Data Sources
1. **Synthetic UI Generator** - Controlled generation with known ground truth
2. **WAA Benchmark Tasks** - Real Windows UI with element annotations
3. **Human Demonstrations** - Recorded desktop interactions

---

## Implementation Notes

### Directory Structure

```
openadapt_ml/experiments/representation_shootout/
    __init__.py
    config.py        # Experiment configurations (dataclasses)
    conditions.py    # Condition implementations (A, B, C)
    evaluator.py     # Drift evaluator and metrics
    runner.py        # Main experiment runner
```

### Key Classes

```python
@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str
    conditions: list[ConditionConfig]
    drift_tests: list[DriftConfig]
    metrics: list[str]
    decision_tolerance: float = 0.05

@dataclass
class ConditionConfig:
    """Configuration for a single experimental condition."""
    name: str  # "raw_coords", "coords_cues", "marks"
    model_type: str
    input_augmentation: dict | None
    output_format: str

@dataclass
class DriftConfig:
    """Configuration for a drift test."""
    name: str
    drift_type: str  # "resolution", "translation", "theme", "scroll"
    parameters: dict

class ConditionBase(ABC):
    """Abstract base for experimental conditions."""

    @abstractmethod
    def prepare_input(self, observation, goal, history) -> dict:
        """Prepare model input from raw observation."""
        pass

    @abstractmethod
    def parse_output(self, model_output) -> dict:
        """Parse model output to action dict."""
        pass

    @abstractmethod
    def compute_loss(self, prediction, ground_truth) -> float:
        """Compute training loss."""
        pass
```

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Framework | 1 day | Implement config, conditions, evaluator, runner scaffolding |
| 2. Synthetic Data | 2-3 days | Generate synthetic training/eval datasets |
| 3. Model Training | 3-5 days | Train models for each condition |
| 4. Evaluation | 1-2 days | Run drift tests, compute metrics |
| 5. Analysis | 1 day | Analyze results, make recommendation |

**Total**: ~2 weeks

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Insufficient synthetic data variety | Use multiple UI domains (web, desktop, forms) |
| Theme transformation artifacts | Use recorded theme variants instead of synthetic |
| Element detection failures (Condition C) | Use ground-truth elements for fair comparison |
| Overfitting to canonical distribution | Ensure no data leakage between train/eval |

---

## Success Criteria

1. **Framework functional**: All conditions can train and evaluate
2. **Statistically significant**: N >= 100 samples per drift condition
3. **Clear recommendation**: One approach wins decisively or tie is documented
4. **Reproducible**: All configs and data sources documented

---

## Appendix: Prior Work

### Set-of-Marks (SoM)
- **Reference**: OmniParser, SeeClick
- **Approach**: Overlay numbered markers on UI elements
- **Benefit**: Reduces coordinate prediction to element selection

### Visual Grounding
- **Reference**: Ferret-UI, CogAgent
- **Approach**: Point to elements via coordinate prediction
- **Benefit**: Works without explicit element detection

### Coordinate Regression
- **Reference**: Many VLM-based agents
- **Approach**: Directly predict (x, y) from screenshot
- **Challenge**: Sensitive to resolution and position changes

---

## References

1. Windows Agent Arena (WAA) - Microsoft Research, 2024
2. OmniParser for Pure Vision Based GUI Agent - Microsoft Research, 2024
3. SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents - 2024
4. Ferret-UI: Grounded Mobile UI Understanding - Apple ML Research, 2024
