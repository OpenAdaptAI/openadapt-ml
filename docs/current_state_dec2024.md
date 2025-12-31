# OpenAdapt-ML: Current State & Roadmap

**Date**: December 31, 2024
**Status**: Validated Demo-Conditioning, Ready for Multi-Step Testing

---

## Executive Summary

We have **empirically validated** that demo-conditioned prompting improves VLM action selection from 33% to 100% accuracy on first-action selection. The retrieval module is built and tested. The next step is multi-step execution testing before fine-tuning.

---

## 1. The Core Realization

Analysis of WAA failures (22% success rate on file_explorer tasks) revealed that the bottleneck is **grounding and search**, not reasoning. The model struggles with Windows-specific UI idioms because it doesn't know where to look.

**OpenAdapt's fundamental value**: Not "zero-shot intelligence," but **trajectory-conditioned disambiguation**. Demonstrations provide a spatial and procedural map that collapses the search space.

---

## 2. Experimental Validation

### Demo-Conditioned Prompting Experiment (Dec 31, 2024)

| Condition | Success Rate | Key Finding |
|-----------|-------------|-------------|
| **Zero-shot** | 33% (1/3) | Systematic spatial bias (clicking right-side status icons) |
| **Control (Length-matched)** | 67% (2/3) | More tokens alone isn't sufficient |
| **Demo-Conditioned** | 100% (3/3) | Demo content corrects spatial bias |

**Generalization**: Demo for "Turn OFF Night Shift" transferred to:
- Opposite toggle (Turn ON)
- Parameter change (Adjust temperature)
- Different setting (True Tone)

### Negative Control Test (Dec 31, 2024)

| Condition | Result |
|-----------|--------|
| Irrelevant demo | No improvement over zero-shot |

**Conclusion**: Retrieval quality matters. Random demos don't help.

---

## 3. Current Infrastructure

### Validated Components

| Component | Status | Location |
|-----------|--------|----------|
| Demo-conditioned prompting | ✅ Validated | `openadapt_ml/experiments/demo_prompt/` |
| Negative control | ✅ Validated | `negative_control_results/` |
| Retrieval module | ✅ Built & tested | `openadapt_ml/retrieval/` |
| Episode schema | ✅ Stable | `openadapt_ml/schemas/sessions.py` |
| Capture ingestion | ✅ Working | `openadapt_ml/ingest/capture.py` |

### Retrieval Module (v1)

- **DemoIndex**: Stores episodes with metadata (app, domain, task)
- **DemoRetriever**: TF-IDF + cosine similarity with domain bonus
- **Tests**: 17/17 passing
- **Ready for**: Integration with prompt experiment

---

## 4. The Experimental Matrix

What we've measured vs. what remains:

| | No Retrieval | With Retrieval |
|---|---|---|
| **No Fine-tuning** | 33% (measured) | 100% (measured) |
| **Fine-tuning** | ? (baseline needed) | ? (OpenAdapt's value) |

**Next steps fill the bottom row.**

---

## 5. Recommended Sequence

### Phase 1: Multi-Step Execution (This Week)

Before fine-tuning, test if demo-following holds beyond first action.

**Experiment**:
1. Run 5-step execution with demo-conditioning
2. Measure: How many steps correct? Where does it diverge?
3. Add lightweight verification ("did expected panel appear?")

**Why first**: If demo-following breaks at step 2, fine-tuning design changes.

### Phase 2: Fine-Tuning Comparison (Week 2)

Run **dual-track** fine-tuning to compare approaches:

#### Track A: Standard Fine-Tuning (Baseline)
```python
# Training sample
{"input": [screenshot, "Turn off Night Shift"], "output": "CLICK(20, 8)"}
```
- No demos at training or inference
- Answers: "Could we just train our way out?"

#### Track B: OpenAdapt Fine-Tuning (Differentiation)
```python
# Training sample
{"input": [screenshot, "Turn on Night Shift", retrieved_demo], "output": "CLICK(20, 8)"}
```
- Train model to **use** demonstrations
- Hypothesis: Model learns to follow *new* demos it hasn't seen

### Phase 3: Full Comparison (Week 3)

| Condition | Training | Inference | Expected |
|-----------|----------|-----------|----------|
| C1: Zero-shot | None | Task only | 33% (baseline) |
| C2: Retrieval-only | None | Task + demo | 100% (validated) |
| C3: Fine-tuned | Standard SFT | Task only | ? |
| C4: Fine-tuned + Retrieval | OpenAdapt SFT | Task + demo | ? |

**Key hypothesis**: C4 > C3 proves OpenAdapt's unique value.

---

## 6. Enterprise Value Proposition

By combining **Retrieval** and **Demo-Conditioned Fine-Tuning**, we offer:

1. **Fast Rollout**: Existing workflow recordings become immediate performance boosters
2. **No Training Required**: Retrieval alone provides 67 percentage point improvement
3. **Auditability**: Actions grounded in explicit human demonstrations
4. **Compounding Accuracy**: Fine-tuning makes the model a better "demo-follower"

**The value**: Reuse prior workflow recordings to improve performance without retraining.

---

## 7. Technical Decisions Made

### Demo Format (Validated)

Behavior-only format (no explanations):
```
Step 1:
  Screen: Desktop
  Action: CLICK(0.01, 0.01)
  Result: Apple menu opened
```

Not:
```
[Action: CLICK(0.01, 0.01) - Click Apple menu icon in top-left]
```

### Retrieval Strategy (v1)

1. Text similarity (task ↔ demo goal) via TF-IDF
2. Domain match bonus (configurable)
3. Top-K = 3 demos concatenated

### What's Deferred

- Full multi-step autonomy (until Phase 1 validates)
- WAA benchmark re-runs (different capability being tested)
- Sentence-transformers / CLIP embeddings (v2)
- FAISS/Qdrant scaling (v3)

---

## 8. Files & Artifacts

### Experiment Results
- `docs/experiments/demo_conditioned_prompting_results.md` - Full experiment writeup
- `openadapt_ml/experiments/demo_prompt/results/` - Raw JSON results
- `negative_control_results/` - Negative control test results

### Code
- `openadapt_ml/experiments/demo_prompt/` - Experiment module
- `openadapt_ml/retrieval/` - Retrieval module (index, retriever, embeddings)
- `scripts/run_demo_experiment.py` - Experiment runner

### Documentation
- `RETRIEVAL_QUICKSTART.md` - Retrieval usage guide
- `openadapt_ml/retrieval/README.md` - Module architecture
- `openadapt_ml/retrieval/USAGE.md` - Detailed patterns

---

## 9. Immediate Next Action

**Multi-step execution test**:

1. Take a 5-step task (e.g., full Night Shift toggle)
2. Run with retrieved demo
3. Execute steps sequentially, recording each action
4. Measure: Steps correct / Steps attempted
5. Document where and why divergence occurs

This determines whether fine-tuning should focus on:
- First-action accuracy (if later steps are fine)
- Trajectory consistency (if model drifts mid-task)
- Error recovery (if model can self-correct)

---

## 10. Success Criteria

### Short-term (2 weeks)
- [ ] Multi-step execution maintains >80% accuracy through 5 steps
- [ ] Standard fine-tuning baseline established
- [ ] C4 > C3 validated (OpenAdapt approach beats standard SFT)

### Medium-term (1 month)
- [ ] Pilot with 5-10 enterprise workflows
- [ ] Retrieval working on real agent recordings
- [ ] Paper-ready experimental results

### Long-term
- [ ] Production deployment on enterprise workflows
- [ ] Continuous improvement from new recordings
- [ ] Cross-domain transfer validated
