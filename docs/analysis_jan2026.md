# OpenAdapt-ML: Comprehensive Analysis & Strategic Options

**Date**: January 7, 2026
**Status**: Pre-alpha
**Last Updated**: After WAA benchmark evaluation (12.5% on 8 tasks)

---

## Executive Summary

OpenAdapt-ML is a model-agnostic ML engine for GUI automation agents. The core thesis is that **trajectory-conditioned disambiguation** (showing demos of similar tasks) dramatically improves agent performance compared to zero-shot approaches.

**Key Validated Finding**: Demo-conditioned prompting improves first-action accuracy from 33% to 100% (+67 percentage points).

**Critical Blocker**: Despite 100% first-action accuracy, episode success remains at 0%. The agent fails to complete multi-step trajectories.

**Root Cause Identified**: Demo context was only injected at step 1, dropped at subsequent steps. Fix implemented but untested.

---

## Table of Contents

1. [Current Capabilities](#1-current-capabilities)
2. [Validated Results](#2-validated-results)
3. [Critical Blockers](#3-critical-blockers)
4. [State of the Art](#4-state-of-the-art)
5. [Gap Analysis](#5-gap-analysis)
6. [Strategic Options](#6-strategic-options)
7. [Recommendations](#7-recommendations)
8. [Appendix: Key Files](#appendix-key-files)

---

## 1. Current Capabilities

### 1.1 Core Infrastructure

| Component | Status | Description |
|-----------|--------|-------------|
| **Schema** | ✅ Complete | Episode/Step/Action/Observation data model |
| **Synthetic UI Generation** | ✅ Working | Procedural login/registration forms with SoM |
| **Demo Retrieval** | ✅ Implemented | Embedding-based retrieval (TF-IDF, sentence-transformers, OpenAI) |
| **Training Pipeline** | ✅ Ready | TRL SFTTrainer + Unsloth (2x speed, 50% less VRAM) |
| **API Agents** | ✅ Working | Claude Sonnet 4.5, GPT-5.1 wrappers |
| **WAA Integration** | ✅ Working | Azure VM orchestration, custom Docker image |
| **Benchmark Viewer** | ✅ Working | HTML viewer for execution traces |

### 1.2 Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ openadapt-      │────▶│ Episode Schema  │────▶│ SFT Samples     │
│ capture         │     │ (JSON + images) │     │ (TRL format)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Real recordings        Synthetic data          Training data
   from human demos       from procedural         for VLM fine-tuning
                          generation
```

### 1.3 Benchmark Infrastructure

| Benchmark | Integration Status | Notes |
|-----------|-------------------|-------|
| **WAA (Windows Agent Arena)** | ✅ Full | Azure VM, custom Docker, API agents |
| **OSWorld** | ❌ Not started | Linux desktop benchmark |
| **WebArena** | ❌ Not started | Web browser benchmark |
| **Synthetic (internal)** | ✅ Complete | 100% accuracy with SoM |

### 1.4 Demo-Conditioned Prompting

The core differentiating feature:

```python
# Without demo (zero-shot)
prompt = f"Task: {instruction}\nScreenshot: [image]"
# Result: 33% first-action accuracy

# With demo (trajectory-conditioned)
prompt = f"""
DEMONSTRATION:
{demo_trajectory}

Now perform: {instruction}
Screenshot: [image]
"""
# Result: 100% first-action accuracy
```

---

## 2. Validated Results

### 2.1 Demo-Conditioned Prompting (December 2025)

**Experiment**: 45 macOS tasks with Qwen2.5-VL-7B

| Condition | First-Action Accuracy | Δ vs Zero-shot |
|-----------|----------------------|----------------|
| Zero-shot | 33.3% (15/45) | — |
| Demo-conditioned | **100.0% (45/45)** | **+66.7 pp** |
| Length-matched control | 57.8% (26/45) | +24.5 pp |

**Key insight**: The improvement is from semantic trajectory information, not just longer prompts.

### 2.2 Synthetic Benchmark (Set-of-Marks)

| Mode | Accuracy | Notes |
|------|----------|-------|
| Coordinate-based | ~60% | `CLICK(x=0.42, y=0.31)` |
| SoM (element IDs) | **100%** | `CLICK([1])` |

### 2.3 WAA Benchmark (January 2026)

**Run**: 8 of 154 tasks attempted (SSH timeout at ~1.5 hours)

| Metric | Value |
|--------|-------|
| Tasks Passed | 1 of 8 (12.5%) |
| Published SOTA | ~19.5% (GPT-5.1 + OmniParser) |
| Agent | Navi (buggy) |

**Failure Analysis**:
- 4 tasks: Hit step limit (15), agent hallucinated completion
- 1 task: Planning crash (NoneType bug in Navi)
- 1 task: Passed (simple 2-click task)
- 2 tasks: SSH timeout

**Root cause**: Navi agent has fundamental bugs. Our API agent bypasses these.

---

## 3. Critical Blockers

### 3.1 The 100%/0% Paradox

**Symptom**: Demo-conditioned prompting achieves 100% first-action accuracy, but 0% episode success.

**Root Cause Identified**: Demo context was only injected at step 1.

```python
# BEFORE (broken)
def predict(self, instruction, obs):
    if self.step_counter == 1:
        prompt = f"Demo: {demo}\nTask: {instruction}"  # Demo only here
    else:
        prompt = f"Task: {instruction}"  # No demo!
```

**Fix Implemented** (January 7, 2026): Demo now persists across all steps.

```python
# AFTER (fixed)
def predict(self, instruction, obs):
    prompt = f"Demo: {self.demo}\nTask: {instruction}"  # Demo at EVERY step
```

**Status**: Fix implemented, not yet validated.

### 3.2 Demo Data Gap

| What We Have | What We Need |
|--------------|--------------|
| macOS captures | Windows captures for WAA |
| Synthetic login/registration | Real Windows app UIs |
| Text-only demos | Screenshots + actions |

### 3.3 Navi Agent Bugs

The default WAA agent (Navi) has critical bugs:

```python
# Bug: plan_result can be None, crashes regex
TypeError: expected string or bytes-like object, got 'NoneType'
# Location: navi/agent.py line ~287
```

**Workaround**: Our API agent (`api_agent.py`) bypasses Navi entirely.

---

## 4. State of the Art

### 4.1 GUI Automation Benchmarks

| Benchmark | Platform | Tasks | SOTA | SOTA Agent |
|-----------|----------|-------|------|------------|
| **WAA** | Windows 11 | 154 | 19.5% | GPT-5.1 + OmniParser |
| **WAA-V2** | Windows 11 | 141 | 36.0% | PC Agent-E |
| **OSWorld** | Ubuntu | 369 | 22.7% | Claude 3.5 Sonnet |
| **WebArena** | Browser | 812 | 35.8% | GPT-4V + SoM |
| **Mind2Web** | Browser | 2,350 | 41.1% | GPT-4 + HTML |

### 4.2 Key Approaches

#### OmniParser (Microsoft, 2024)
- Vision-based UI element detection
- Generates Set-of-Marks overlays
- Enables `CLICK([5])` instead of coordinates
- **Limitation**: Still requires good planning

#### PC Agent-E (GAIR-NLP, 2025)
- 312 human trajectories + "Trajectory Boost" augmentation
- Thought completion (reconstructs reasoning)
- **Result**: 36% on WAA-V2 (SOTA)
- **Key insight**: Small data + good augmentation beats large models

#### UI-TARS (ByteDance, 2025)
- Native GUI agent trained on diverse data
- DPO for preference learning
- **Result**: 26.2% on WAA-V2

#### Claude Computer Use (Anthropic, 2024-2025)
- Built-in computer control capability
- Coordinates-based clicking
- **Limitation**: No explicit demo conditioning

### 4.3 Retrieval-Augmented Approaches

| Approach | Description | Results |
|----------|-------------|---------|
| **RAG for code** | Retrieve similar code examples | +15-30% on coding tasks |
| **Demo retrieval (ours)** | Retrieve similar UI demos | +67pp first-action |
| **In-context learning** | Few-shot examples | Diminishing returns past 3-5 |

### 4.4 Training Approaches

| Method | Data Required | Compute | Results |
|--------|---------------|---------|---------|
| **Zero-shot** | None | Inference only | 10-20% |
| **Few-shot prompting** | 3-5 demos | Inference only | 20-35% |
| **SFT (Supervised Fine-Tuning)** | 1K-10K trajectories | 1-8 GPU days | 30-50% |
| **RLHF/DPO** | 10K+ preferences | 2-16 GPU days | +5-10pp over SFT |
| **Demo retrieval (ours)** | 10-100 demos | Inference only | TBD |

---

## 5. Gap Analysis

### 5.1 What's Missing

| Gap | Impact | Effort to Close |
|-----|--------|-----------------|
| **Validated demo persistence** | Blocking | Low (test existing fix) |
| **Windows demo data** | Blocking for WAA | Medium (capture or generate) |
| **Multi-step evaluation** | Can't measure progress | Low (infrastructure exists) |
| **OSWorld/WebArena integration** | Can't compare to SOTA | Medium |
| **Accessibility tree support** | Better grounding | Medium |
| **Production deployment** | Can't ship to users | High |

### 5.2 Comparison to SOTA

| Capability | OpenAdapt-ML | PC Agent-E | Claude Computer Use |
|------------|--------------|------------|---------------------|
| Demo retrieval | ✅ | ❌ | ❌ |
| Thought completion | ❌ | ✅ | ❌ |
| Trajectory augmentation | ❌ | ✅ | ❌ |
| Native training | ✅ | ✅ | ❌ (API only) |
| Cross-platform | ✅ | Windows only | ✅ |
| WAA performance | 12.5%* | 36% | ~20% |

*On 8 tasks with buggy agent, not comparable

### 5.3 Unique Differentiators

1. **Demo retrieval at inference**: No one else does this for GUI agents
2. **Trajectory-conditioned prompting**: Validated 67pp improvement
3. **Platform-agnostic schema**: Works across macOS, Windows, Linux, web
4. **Open source training pipeline**: TRL + Unsloth, reproducible

---

## 6. Strategic Options

### Option A: Validate Demo Persistence Fix

**Goal**: Prove the core thesis works end-to-end

**Tasks**:
1. Test with existing macOS demo
2. Run 5-10 WAA tasks with fixed agent
3. Measure episode success (not just first-action)

**Effort**: 1-2 days
**Risk**: Low
**Upside**: If works, validates entire approach

### Option B: Generate Synthetic Demos via LLM

**Goal**: Scalably create demos without manual capture

**Tasks**:
1. Use Claude/GPT to generate text trajectories for all 154 WAA tasks
2. Format as demo strings (no screenshots)
3. Test retrieval + prompting

**Effort**: 1-2 days
**Risk**: Medium (may not work without screenshots)
**Upside**: Scalable to any benchmark

### Option C: Bootstrap from Successful Runs

**Goal**: Self-improving system

**Tasks**:
1. Run WAA zero-shot with API agent
2. Extract traces from successful tasks
3. Use as demos for retry
4. Iterate

**Effort**: 2-3 days
**Risk**: Low (worst case: no improvement)
**Upside**: No manual demo creation needed

### Option D: Integrate OSWorld/WebArena

**Goal**: Validate on benchmarks with published baselines

**Tasks**:
1. Implement OSWorld adapter
2. Run Claude/GPT-5.1 zero-shot
3. Compare to published numbers
4. Identify if agent loop is fundamentally broken

**Effort**: 3-5 days
**Risk**: Low
**Upside**: Diagnostic clarity

### Option E: Implement Thought Completion (PC Agent-E style)

**Goal**: Improve trajectory quality via reasoning reconstruction

**Tasks**:
1. Use LLM to add reasoning to existing trajectories
2. Train on augmented data
3. Evaluate improvement

**Effort**: 1-2 weeks
**Risk**: Medium
**Upside**: Proven technique (+141% in PC Agent-E)

### Option F: Full Training Pipeline

**Goal**: Train custom agent on collected data

**Tasks**:
1. Collect 500+ Windows trajectories
2. Apply thought completion + trajectory boost
3. Fine-tune Qwen2.5-VL-7B or similar
4. Evaluate on WAA

**Effort**: 2-4 weeks
**Risk**: High (data collection bottleneck)
**Upside**: Could match or beat SOTA

### Option G: Production Demo Retrieval System

**Goal**: Ship retrieval as product feature

**Tasks**:
1. Complete CLI integration
2. Create demo library structure
3. Add hybrid BM25+embedding retriever
4. Document and test

**Effort**: 1-2 weeks
**Risk**: Low
**Upside**: Differentiating feature for enterprise

---

## 7. Recommendations

### Immediate (This Week)

| Priority | Action | Rationale |
|----------|--------|-----------|
| **P0** | Validate demo persistence fix | Can't proceed without knowing if core thesis works |
| **P1** | Generate LLM text demos for WAA | Scalable, no manual work |
| **P1** | Run WAA with fixed agent + demos | Measure actual episode success |

### Short-term (2-4 Weeks)

| Priority | Action | Rationale |
|----------|--------|-----------|
| **P1** | Implement thought completion | Proven technique, low risk |
| **P2** | Add OSWorld integration | Diagnostic benchmark |
| **P2** | Bootstrap demo collection | Self-improving system |

### Medium-term (1-3 Months)

| Priority | Action | Rationale |
|----------|--------|-----------|
| **P2** | Full training pipeline | Required for SOTA |
| **P3** | Production retrieval system | Enterprise feature |
| **P3** | Accessibility tree support | Better grounding |

### Decision Framework

```
                        ┌─────────────────────────┐
                        │ Test demo persistence   │
                        │ fix with existing demo  │
                        └───────────┬─────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                   WORKS                    DOESN'T WORK
                        │                       │
            ┌───────────┴───────┐       ┌───────┴───────────┐
            ▼                   ▼       ▼                   ▼
    Generate LLM demos    Bootstrap   Run OSWorld      Debug agent
    for WAA tasks         from runs   for diagnosis    loop/prompts
            │                   │           │               │
            └─────────┬─────────┘           └───────┬───────┘
                      ▼                             ▼
              Run WAA with demos            Fix fundamental
                      │                     issue first
                      ▼
              Episode success > 0%?
              /                    \
            YES                     NO
            /                         \
    Scale up:                   Investigate:
    - More demos                - State representation
    - Training                  - Action execution
    - Production                - Prompt structure
```

---

## Appendix: Key Files

### Core Modules

| File | Lines | Purpose |
|------|-------|---------|
| `openadapt_ml/schemas/sessions.py` | ~400 | Episode/Step/Action/Observation schemas |
| `openadapt_ml/datasets/next_action.py` | 526 | Episodes → SFT training samples |
| `openadapt_ml/retrieval/demo_retriever.py` | 817 | Embedding-based demo retrieval |
| `openadapt_ml/training/trl_trainer.py` | 355 | TRL + Unsloth training pipeline |
| `openadapt_ml/ingest/synthetic.py` | 1,164 | Procedural UI generation |

### Benchmark Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| `openadapt_ml/benchmarks/cli.py` | ~2,000 | CLI for Azure VM, WAA, benchmarks |
| `openadapt_ml/benchmarks/waa_deploy/api_agent.py` | 540 | Claude/GPT-5.1 agent for WAA |
| `openadapt_ml/benchmarks/runner.py` | ~600 | Benchmark evaluation loop |
| `openadapt_ml/benchmarks/waa.py` | ~800 | WAA adapter and mock |

### Experiments

| File | Purpose |
|------|---------|
| `openadapt_ml/experiments/demo_prompt/` | Demo-conditioned prompting experiment |
| `openadapt_ml/experiments/waa_demo/` | WAA-specific demo experiments |
| `docs/experiments/demo_conditioned_prompting_results.md` | December 2025 results |
| `docs/experiments/waa_benchmark_results_jan2026.md` | January 2026 WAA run |

### Configuration

| File | Purpose |
|------|---------|
| `configs/qwen3vl_synthetic_som.yaml` | Synthetic SoM training |
| `configs/qwen3vl_capture.yaml` | Real capture training |
| `CLAUDE.md` | Developer context and conventions |

---

## References

1. [Windows Agent Arena](https://github.com/microsoft/WindowsAgentArena) - Microsoft, 2024
2. [PC Agent-E](https://github.com/GAIR-NLP/PC-Agent-E) - GAIR-NLP, 2025
3. [OSWorld](https://os-world.github.io/) - CMU, 2024
4. [WebArena](https://webarena.dev/) - CMU, 2023
5. [OmniParser](https://github.com/microsoft/OmniParser) - Microsoft, 2024
6. [Set-of-Marks Prompting](https://arxiv.org/abs/2310.11441) - Microsoft, 2023

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-07 | Initial comprehensive analysis |
| 2026-01-07 | Demo persistence fix implemented |
| 2026-01-06 | WAA benchmark run (1/8 passed) |
| 2025-12 | Demo-conditioned prompting validated |
