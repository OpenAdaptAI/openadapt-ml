# OpenAdapt-ML: Grounding Architecture & Next Steps

## Executive Summary

**Key Insight**: The policy problem (action type + sequencing) is solved on synthetic data.
The remaining challenge is **grounding generalization**—making localization work on real, unseen UIs.

This document proposes a clean architectural separation between **policy** and **grounding**,
incorporating insights from GUI-Actor (Microsoft Research, NeurIPS 2025) and our own findings.

---

## 1. Current State Analysis

### What Works (100% on Synthetic)

| Approach | Action Type | Element/Coord | Episode Success |
|----------|-------------|---------------|-----------------|
| SoM + Action History | 100% | 100% | 100% |
| Coordinate + Action History | 100% | 100% (0.007 error) | 100% |

Both approaches achieve saturation on synthetic benchmarks when action history is included.

### What This Means

1. **Policy is solved** for synthetic scenarios
   - Model correctly predicts WHEN to click/type/done
   - Model correctly predicts WHICH element (SoM) or WHERE (coordinates)
   - Action history removes workflow position ambiguity

2. **The unsolved problem is grounding generalization**
   - Real UIs have variable layouts, themes, element counts
   - Coordinate models overfit to training resolution/positions
   - SoM requires element detection on unseen screenshots

3. **Synthetic benchmarks have hit their ceiling**
   - 100% accuracy means we need harder tests
   - Real recordings are the path forward

---

## 2. GUI-Actor Insights

### Core Contribution

GUI-Actor reframes grounding from **coordinate regression** to **attention-based region selection**:

- Adds an `<ACTOR>` token that attends over visual patch tokens
- Attention map defines candidate action regions (coordinate-free)
- Can emit multiple candidates with confidence scores
- Optional verifier selects best candidate

### Why This Matters for OpenAdapt-ML

| Approach | Grounding Type | Generalization | Training Cost |
|----------|----------------|----------------|---------------|
| Coordinate | Point regression | Poor (overfits) | High (26B tokens for VideoAgentTrek) |
| SoM | Element index | Good (if detection works) | Low |
| Attention (GUI-Actor) | Region selection | Good (patch-aligned) | Medium |

**Key architectural insight**: GUI-Actor's attention mechanism can be used as:
- An **auxiliary training signal** (supervise attention alignment to bboxes)
- A **grounding module** (replace coordinate regression with region selection)
- A **verification layer** (multi-candidate with reranking)

---

## 3. Proposed Architecture: Policy/Grounding Separation

### Current (Implicit)

```
VLM(screen, goal, history) → ActionDSL (CLICK(x,y) or CLICK([N]))
                                       ↓
                              Execute action
```

Policy and grounding are entangled in the single VLM output.

### Proposed (Explicit)

```
Policy: VLM(screen, goal, history) → ActionIntent
        {action_type: "click", target: "login button", reasoning: "..."}
                                       ↓
Grounding: GroundingModule(screen, target_description) → ExecutableAction
           {type: "click", x: 0.42, y: 0.73, confidence: 0.95}
                                       ↓
                              Execute action
```

This separation enables:
- Training policy and grounding separately or jointly
- Swapping grounding strategies without retraining policy
- Evaluating each layer independently
- Composing different grounding modules per platform

### GroundingModule Interface

```python
# openadapt_ml/grounding/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image

@dataclass
class RegionCandidate:
    """A candidate region for action execution."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    centroid: tuple[float, float]  # click point
    confidence: float
    element_label: str | None = None
    metadata: dict | None = None

class GroundingModule(ABC):
    """Abstract base for grounding strategies."""

    @abstractmethod
    def ground(
        self,
        image: Image.Image,
        target_description: str,
        k: int = 1,
    ) -> list[RegionCandidate]:
        """
        Locate regions matching the target description.

        Args:
            image: Screenshot to search
            target_description: Natural language target (e.g., "login button")
            k: Number of candidates to return

        Returns:
            List of candidate regions, sorted by confidence descending
        """
        pass
```

### Grounding Implementations

| Module | Strategy | Latency | Best For |
|--------|----------|---------|----------|
| `SoMGrounder` | Pre-labeled element indices | ~0ms | Synthetic, controlled UIs |
| `CoordinateGrounder` | Fine-tuned VLM regression | ~50ms | Fixed-resolution tasks |
| `DetectorGrounder` | OmniParser/Gemini bbox extraction | ~100ms | Real UIs, zero-shot |
| `AttentionGrounder` | GUI-Actor style patch attention | ~50ms | Real UIs, fine-tuned |

---

## 4. Implementation Roadmap

### Phase 0: Capture Integration (Immediate)

**Goal**: Get real data flowing from openadapt-capture to openadapt-ml.

```
openadapt-capture record ./my_workflow
        ↓
openadapt-ml ingest capture ./my_workflow
        ↓
Episode/Step format with real screenshots
```

**Tasks**:
1. [ ] Write `openadapt_ml/ingest/capture.py` adapter
2. [ ] Map capture events (click, type, drag) to Action schema
3. [ ] Extract screenshots at action timestamps
4. [ ] Test on calculator demo recording

### Phase 1: Grounding Abstraction (This Week)

**Goal**: Make grounding pluggable without breaking existing code.

**Tasks**:
1. [ ] Create `openadapt_ml/grounding/base.py` with `GroundingModule` interface
2. [ ] Implement `SoMGrounder` (wrap existing SoM logic)
3. [ ] Implement `CoordinateGrounder` (wrap existing coord logic)
4. [ ] Add grounding-specific eval: `eval_grounding.py`
   - IoU with ground-truth bbox
   - Hit rate (centroid inside bbox)
   - Multi-candidate oracle accuracy

### Phase 2: Real UI Grounding (Next Week)

**Goal**: Ground actions on unseen real screenshots.

**Tasks**:
1. [ ] Implement `DetectorGrounder` with OmniParser
2. [ ] Alternatively: Implement `GeminiGrounder` using Gemini API bbox extraction
3. [ ] Record 5-10 real workflows (calculator, file ops, browser)
4. [ ] Evaluate grounding modules on real data
5. [ ] Compare: SoM vs detector vs coordinate on real recordings

### Phase 3: Attention-Based Grounding (Research Track)

**Goal**: Implement GUI-Actor insights for better generalization.

**Tasks**:
1. [ ] Extract attention maps from Qwen backbone on grounding task
2. [ ] Implement `AttentionGrounder` that converts attention → region scores
3. [ ] Add bbox supervision to training (auxiliary loss on attention alignment)
4. [ ] Compare attention grounding vs coordinate regression vs SoM
5. [ ] Measure sample efficiency (how much data to generalize?)

### Phase 4: Policy/Grounding Joint Training (Future)

**Goal**: End-to-end system that generalizes to unseen UIs.

**Tasks**:
1. [ ] Define ActionIntent schema (action type + target description)
2. [ ] Train policy to output ActionIntent (not coordinates)
3. [ ] Chain policy → grounding at inference
4. [ ] Evaluate on held-out real workflows
5. [ ] Compare joint vs separate training

---

## 5. Evaluation Strategy

### Current Metrics (Entangled)

- `action_type_accuracy` — policy + grounding mixed
- `coord_error` / `click_hit_rate` — grounding only
- `episode_success` — end-to-end

### Proposed Metrics (Separated)

**Policy Metrics**:
- `action_type_accuracy` — given ground-truth grounding
- `target_description_accuracy` — semantic match to intended element
- `termination_accuracy` — correct DONE timing

**Grounding Metrics**:
- `bbox_iou` — IoU with ground-truth element bbox
- `centroid_hit_rate` — click point inside correct element
- `oracle_hit_rate@k` — any of top-k candidates correct
- `grounding_latency` — ms per grounding call

**End-to-End Metrics**:
- `episode_success` — task completion
- `step_efficiency` — steps taken vs optimal

---

## 6. Research Questions

1. **Attention vs Coordinates**: Does supervising attention alignment improve generalization over coordinate regression? (GUI-Actor claims yes)

2. **Multi-Candidate Value**: For GUI automation, is multi-candidate grounding + verification better than single-best prediction?

3. **Sample Efficiency**: How many real recordings are needed for grounding to generalize? (VideoAgentTrek needed 1.5M steps for coordinates)

4. **Cross-Platform Transfer**: Does grounding trained on macOS generalize to Windows/Linux?

5. **Compositional Policy**: Can we train policy once (action types + sequencing) and swap grounding modules per platform/resolution?

---

## 7. File Organization

```
openadapt_ml/
├── grounding/
│   ├── __init__.py
│   ├── base.py           # GroundingModule interface, RegionCandidate
│   ├── som.py            # SoMGrounder (index-based)
│   ├── coordinate.py     # CoordinateGrounder (VLM regression)
│   ├── detector.py       # DetectorGrounder (OmniParser/Gemini)
│   └── attention.py      # AttentionGrounder (GUI-Actor style)
├── ingest/
│   ├── synthetic.py      # existing
│   └── capture.py        # NEW: openadapt-capture adapter
├── evals/
│   ├── trajectory_matching.py  # existing
│   └── grounding.py      # NEW: grounding-specific metrics
└── ...
```

---

## 8. Dependencies and Integration

### With openadapt-capture

- Use openadapt-capture for recording real workflows
- Adapter converts capture.db → Episode format
- Screenshots extracted from video at action timestamps

### With openadapt-privacy

- Apply privacy scrubbing to real recordings before training
- PII detection on screenshots + action text

### External Dependencies

| Dependency | Purpose | Optional? |
|------------|---------|-----------|
| OmniParser | UI element detection | Yes (alternative to Gemini) |
| Gemini API | Bbox extraction | Yes (alternative to OmniParser) |
| Florence-2 | Vision grounding | Yes (research comparison) |

---

## 9. Success Criteria

### Phase 0-1 (Infrastructure)
- [ ] Real recordings flow through pipeline without errors
- [ ] GroundingModule interface implemented with 2+ backends
- [ ] Grounding-specific eval produces meaningful metrics

### Phase 2-3 (Grounding Quality)
- [ ] At least one grounding module achieves >80% hit rate on real recordings
- [ ] Clear ranking of grounding strategies on held-out data
- [ ] Attention-based grounding shows measurable benefit over coordinates

### Phase 4 (End-to-End)
- [ ] Policy + grounding achieves >50% episode success on real workflows
- [ ] System generalizes to workflows not seen during training
- [ ] Grounding modules can be swapped without retraining policy

---

## 10. Immediate Next Actions

1. **Commit openadapt-capture changes** and push to GitHub
2. **Publish openadapt-capture to PyPI** (configure trusted publishing)
3. **Record calculator demo** using openadapt-capture
4. **Write capture adapter** (`openadapt_ml/ingest/capture.py`)
5. **Create GroundingModule interface** (`openadapt_ml/grounding/base.py`)
6. **Evaluate grounding** on calculator recording

---

*Document created: 2025-12-12*
*Status: Planning*
