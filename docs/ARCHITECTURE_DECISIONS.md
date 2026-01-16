# OpenAdapt-ML Architecture Decisions

**Status**: Living document
**Last Updated**: January 2026
**Scope**: Comprehensive technical direction for openadapt-ml and related packages

---

## 1. Repository Ecosystem

OpenAdapt is structured as a family of focused packages:

| Package | Responsibility | Status |
|---------|---------------|--------|
| **openadapt-capture** | Desktop/browser screen recording | ✅ Exists |
| **openadapt-privacy** | PII detection and redaction | ✅ Exists |
| **openadapt-grounding** | Perception (OmniParser/UI-TARS → Elements) | ✅ Exists |
| **openadapt-ml** | Training, Policy, Benchmarks, Retrieval | ✅ This repo |
| **openadapt-evals** | Benchmark adapters, metrics, regression | 🔄 To extract |

### Decision: Modular Package Architecture
**Rationale**:
- Clear separation of concerns enables independent development
- Enterprise customers can use subsets (e.g., just grounding + privacy)
- Easier testing and deployment of individual components

---

## 2. System Architecture (SAC-Aligned)

We adopt the **Shared Architecture Contract (SAC)** framework with explicit stage separation:

```
Observe → Perceive → Retrieve → Ground → Decide → Resolve → Safety → Execute → Trace
```

### 2.1 Stage Definitions

| Stage | Input | Output | Owner |
|-------|-------|--------|-------|
| **Perceive** | Screenshot | UIElementGraph | openadapt-grounding |
| **Retrieve** | Goal + Screenshot | Demo snippets | openadapt-ml/retrieval |
| **Ground (Stage A)** | Intent + UIElementGraph | element_id | openadapt-ml/grounding |
| **Decide (Stage B)** | State + History + Retrieval | Action | openadapt-ml/runtime |
| **Resolve** | Action + element_id | Coordinates | openadapt-ml/runtime |
| **Safety** | Proposed action | Allow/Block/Confirm | openadapt-ml/runtime |

### Decision: Reject Monolithic End-to-End Agents
**Rationale**: Modular stages ensure failures are **localizable**:
- Perception failure: missing/incorrect UI elements
- Grounding failure: wrong element selected
- Policy failure: incorrect action sequence
- This is essential for debugging, iteration speed, and trust

---

## 3. The Representation Question

### 3.1 Coordinates vs. Marks

A critical architectural decision is whether actions target:
- **Coordinates**: `CLICK(x=0.42, y=0.31)` - direct pixel/normalized output
- **Marks (Elements)**: `CLICK(element_id="e17")` - abstract element reference

### Decision: Run Ablation Before Committing
**Status**: Pending experiment
**Experiment**: `experiments/representation_shootout/`

| Condition | Description | Pros | Cons |
|-----------|-------------|------|------|
| **A: Raw Coords** | Model outputs (x,y) directly | Simple, no perception needed | Breaks on resolution/layout changes |
| **B: Coords + Cues** | Visual markers + zoomed patches | Better than raw, still simple | May not generalize |
| **C: Marks** | element_id via UIElementGraph | Resolution-independent, robust | Requires perception pipeline |

**Decision Rule**:
- If Coords+Cues within 5% of Marks under drift tests → use Coordinates (simpler)
- If Marks significantly better → use full UIElementGraph pipeline

---

## 4. Training Architecture

### 4.1 Two-Stage Training (SAC §7)

**Stage A - Grounding** (Required first):
- Task: `(intent, screenshot, UIElementGraph) → element_id`
- Acceptance: ≥70% top-1 for assistive, ≥85% for autopilot
- Minimum eval set: ≥200 labeled grounding instances

**Stage B - Policy** (After grounding):
- Task: `(goal, history, observation, retrieval) → action`
- Acceptance: ≥99% schema validity, stable action sequencing

### Decision: Grounding-First Training
**Rationale**:
- Grounding is the highest-leverage learning signal
- Policy depends on accurate element selection
- Matches literature (SeeClick, UIPro findings)

### 4.2 Demo Retrieval

**Decision**: Retrieval is per-episode, not per-frame
- Coarse embedding search + fine reranking
- Inject as structured context to policy
- Validated: +67pp first-action accuracy in Dec 2025 experiments

---

## 5. Safety Architecture

### Decision: Deterministic Safety Gate (Not Learned)
**Rationale**: Learned safety is unpredictable; deterministic rules are auditable.

**Required Checks** (MVP):
1. **Blocklist**: Regex on destructive keywords (delete, format, reset)
2. **Irreversibility**: Confirm for submit/send/apply/confirm
3. **Confidence threshold**: Low confidence → require human confirmation
4. **Loop detection**: Same state 3x → halt
5. **Credential guard**: Block typing into password fields

**Location**: `openadapt_ml/runtime/safety_gate.py`

---

## 6. Integration with openadapt-grounding

### 6.1 Interface

openadapt-grounding provides:
```python
class Parser(Protocol):
    def parse(self, image: Image) -> List[Element]

@dataclass
class Element:
    bounds: Tuple[float, float, float, float]  # normalized (x, y, w, h)
    text: Optional[str]
    element_type: str
    confidence: float
```

### 6.2 Integration Layer

openadapt-ml wraps this as UIElementGraph:
```python
class UIElementGraph:
    graph_id: str
    elements: list[UIElement]
    source: str  # "omniparser" | "uitars" | "ax"
```

**Location**: `openadapt_ml/perception/integration.py`

---

## 7. Package Extraction Plan

### 7.1 openadapt-evals (Priority 1)

Extract evaluation infrastructure:
```
openadapt-evals/
├── openadapt_evals/
│   ├── metrics/          # grounding.py, trajectory.py
│   ├── benchmarks/       # base.py, runner.py, waa.py, viewer.py
│   └── __init__.py
```

**Rationale**:
- Evals are used across projects
- Clear dependency boundary
- Enables standalone benchmark development

### 7.2 openadapt-core (Priority 2, Optional)

Consider extracting shared schemas if needed by multiple packages.

---

## 8. Deferred Items (Not in 90-Day Scope)

Per SAC/enterprise docs, explicitly deferred:
- Subgoal discovery and management
- Reinforcement learning
- End-to-end joint training
- Cloud orchestration in OSS
- Cross-frame element identity tracking

---

## 9. Key Files Reference

| Concept | Location |
|---------|----------|
| Episode/Action/Observation schemas | `openadapt_ml/schema/episode.py` |
| Grounding base interface | `openadapt_ml/grounding/base.py` |
| Grounding metrics | `openadapt_ml/evals/grounding.py` |
| Demo retrieval | `openadapt_ml/retrieval/demo_retriever.py` |
| Policy runtime | `openadapt_ml/runtime/policy.py` |
| Benchmark runner | `openadapt_ml/benchmarks/runner.py` |
| Training pipeline | `openadapt_ml/training/trl_trainer.py` |

---

## 10. Prompting Paradigms (Track Taxonomy)

Based on research survey, GUI agents use distinct paradigms across two dimensions:

### 10.1 Grounding Methods

| Method | Target Output | Pros | Cons |
|--------|---------------|------|------|
| **Pure Coordinates** | `(x, y)` pixel | Fast, simple | Fragile under UI drift |
| **Bounding Box** | `(x1, y1, x2, y2)` | Multi-point tolerance | More tokens |
| **Set-of-Mark (SoM)** | Element ID `[14]` | Intuitive for LLMs | Requires pre-processing |
| **Element Selector** | CSS/XPath | Precise | Platform-dependent |
| **Attention-based** | Patch weights | No coord generation | Custom head required |

### 10.2 Reasoning Formats

| Format | Structure | Tokens | Error Recovery |
|--------|-----------|--------|----------------|
| **Act-Only** | Action only | ~50 | None |
| **ReAct** | Thought → Action | ~100-200 | Moderate |
| **CoT (L2)** | Observation → Thought → Action | ~200-400 | Strong |
| **System-2** | Decomposition + milestones + reflection | ~400-800 | Very strong |

### 10.3 Track Definitions

**Track A (Default)**: Direct Coordinate JSON
- Input: screenshot + goal
- Output: `{"action": "click", "coordinate": [x, y]}`
- ~50 tokens, no error recovery

**Track B (If Needed)**: ReAct + Coordinates
- Input: screenshot + history + goal
- Output: Thought trace → Action
- ~100-200 tokens, moderate error recovery

**Track C (Reference)**: Set-of-Mark + Element Selection
- Requires parser pre-processing
- Good for web, struggles on desktop

**Track D (Reference)**: Two-Stage (Plan → Ground)
- Separates planning from grounding
- Higher latency, higher accuracy

### Decision: Support All Tracks
OpenAdapt provides infrastructure for all tracks. Integrators choose based on their requirements:
- Track A for simple, fast automation
- Track B for error recovery needs
- Track C/D/E for maximum robustness

The platform enables experimentation across approaches.

---

## 11. Data Quality (Reference)

OpenAdapt supports various data quality levels. Integrators define their own quality gates based on use case requirements.

### 11.1 Key Data Quality Dimensions

- **Action completeness**: All action types and parameters resolved
- **Temporal alignment**: Screenshots and actions properly synchronized
- **Replayability**: Episodes can be deterministically replayed
- **Label confidence**: High-confidence action labels

---

## 12. Evaluation Strategy

### 12.1 Baseline-First Approach

```
0-shot baseline (all APIs) → n-shot baseline → Fine-tuning experiments
```

**Key insight**: Baseline rankings don't predict fine-tuning outcomes. Run parallel experiments.

### 12.2 Robustness Testing (Drift Suite)

| Drift Type | Test |
|------------|------|
| Window shift | ±200px |
| DPI change | 1x → 1.5x → 2x |
| Resolution change | 1080p → 1440p |
| Theme change | Light → Dark |

**Decision Rule**:
- <5% SR drop → Stay with coordinates
- 5-15% drop → Add visual cues
- >15% drop → Activate parser (SoM)

### 12.3 Success Metrics

| Metric | Definition |
|--------|------------|
| **Success Rate (SR)** | Tasks completed / Tasks attempted |
| **Action Recovery Confidence** | Correctly recovered / Total actions |
| **Click-Hit Rate** | Clicks within target bbox |
| **Schema Validity** | Valid JSON outputs |

---

## 13. Decision Log

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2026-01 | Adopt SAC architecture | Aligns with enterprise contracts | Active |
| 2026-01 | Grounding-first training | Highest leverage signal | Active |
| 2026-01 | Deterministic safety gate | Auditable, predictable | Planned |
| 2026-01 | Run coords vs marks ablation | Avoid over-engineering | Pending |
| 2026-01 | Extract openadapt-evals | Reusability | Planned |
| 2026-01 | Support all tracks (A-E) | Enable diverse enterprise needs | Active |
| 2026-01 | Baselines before fine-tuning | Establish feasibility first | Recommended |

---

## Appendix: Enterprise Document References

Full enterprise contracts stored in `docs/enterprise/`:
- `SAC_v0.1.md` - Shared Architecture Contract
- `DESIGN_ROADMAP_v0.1.md` - Design & Roadmap
- `ENTERPRISE_PLAN_v1.0.md` - Enterprise Evaluation & Deployment Plan
