# Coordinates vs Marks: Ablation Study Design

## Purpose

This document defines the ablation study to answer: **Is element abstraction (marks) actually buying robustness, or are we over-engineering?**

## Conditions

### Condition A: Coordinate Policy

- **Input**: screenshot + goal + history + optional retrieved demo
- **Output**: `{action_type, x, y, ...}` (plus click_type, scroll params)
- **Training label**: recorded click coordinates

### Condition B: Marks Policy

- **Input**: screenshot + UIElementGraph + goal + history + retrieval
- **Output**: `{action_type, element_id, ...}`
- **Training label**: clicked element_id derived from click ∈ bbox

## Why Coordinate Fine-tuning is Worth Testing

1. **Cheaper data**: labels already exist (x, y)
2. **No perception dependency**: can start before parser quality is known
3. **Baseline for ROI**: if coordinates are close to marks, the marks pipeline isn't justified

## Why Marks Usually Win (What We're Testing)

Coordinates tend to break on:
- Resolution / DPI changes
- Window reposition / multi-monitor
- Minor UI layout drift (updates, A/B tests)
- Scrolling lists (same element moves)
- Modal overlays

Marks tend to be stable under all of the above **if perception recall is high**.

## Coordinate Variants (Not Strawman)

### Variant 1: Grid / Tokenized Coordinates (Recommended)

- Discretize screen into WxH bins (e.g., 100x100 or 200x200)
- Output a token pair (x_bin, y_bin) or single index token
- Turns regression into classification, which SFT handles cleanly

### Variant 2: Heatmap-Style Target (Optional)

- Predict a distribution over grid cells
- Use argmax as click target
- Better calibration, but requires model head changes

### Coordinate Normalization

- Relative to active window bounds when available (better generalization)
- Otherwise relative to screen size

## Required Evaluation

Test coordinate-only under controlled perturbations:

| Perturbation | Description |
|--------------|-------------|
| **Resolution scaling** | Replay on different resolutions (0.75x, 1.0x, 1.25x, 1.5x, 2.0x) |
| **Window translation** | Shift window position ±200px |
| **UI drift** | Newer app version or theme change |
| **Scroll offsets** | Same list, different scroll position |

### Metrics

| Metric | Description |
|--------|-------------|
| **Click-hit rate** | Click lands within intended bbox (use perception as evaluator) |
| **Episode success rate** | Task completed end-to-end |
| **Calibration** | Distance-to-target vs confidence |
| **Grounding top-1** | Correct element selected (marks only) |

## Decision Rule

| Condition | Decision |
|-----------|----------|
| Coordinate-only within ~5% of marks on drift tests | Deprioritize marks for that workflow class |
| Coordinate-only collapses under drift (common) | Marks becomes mandatory |

## Implementation

Most of the policy trainer can be shared:
- Only swap the action schema and target representation
- Keep retrieval ablations identical (none/oracle/top-k/random)

## Conclusion

**Yes, train coordinates directly and compare.** It's a cheap, high-leverage experiment that prevents months of building the wrong abstraction.

## References

- V-Star: Visual coordinate augmentation for grounding
- Set-of-Marks (SoM): Element-based action specification
- OpenCUA: Multi-level reasoning with coordinate grounding
- GUI-Actor: Attention-based coordinate-free grounding
