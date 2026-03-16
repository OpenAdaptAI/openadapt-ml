# Design: Task-Specific LoRA Adapters with Runtime Routing

**Status**: Proposal (Experiment Track)
**Authors**: Richard Abrich, Claude
**Date**: 2026-03-16
**Related**: `verl_agent_decision.md`, `demo_retrieval_design.md`, `grpo_architecture_analysis.md`
**Parent**: `openadapt-evals/docs/design/experiment_framework.md`

> **Context**: This document describes one experiment track within the broader OpenAdapt experimentation framework. See `openadapt-evals/docs/design/experiment_framework.md` for the umbrella architecture and how this track compares to others (demo-conditioning, GRPO, SFT, API baselines). OpenAdapt is a general-purpose computer use framework — demo-conditioning, LoRA-per-task, and GRPO are all experiments within it.

---

## 1. Problem Statement

OpenAdapt's current demo-conditioning approach injects task knowledge into the context window at inference time via the DemoController pipeline: retrieve demo → parse into plan → execute step-by-step with VLM verification → retry/replan on failure. This works but has significant drawbacks:

- **High latency**: Multi-turn VLM verification loops per step
- **Fragile infrastructure**: VLM verification, replanning, and socat proxies introduce failure modes
- **Context window bottleneck**: Demo text competes with observation tokens for limited context
- **No lift observed yet**: DC=14% vs ZS=18% in 7 prior trials (infra issues were root cause, but the pipeline complexity contributed)

Meanwhile, the literature consistently shows that **gradient-based task adaptation (LoRA fine-tuning) outperforms in-context learning** for well-defined, repeatable tasks — which is exactly OpenAdapt's target domain.

## 2. Core Idea

**Overfit a dedicated LoRA adapter per task (or per skill). Reduce inference to: classify instruction → load LoRA → execute.**

A LoRA adapter is a compressed, weight-space encoding of the same task knowledge currently provided via demo text. Weight deltas have higher bandwidth than context tokens — the model doesn't need to parse a demo description and map it to actions; the mapping is in the weights.

### 2.1 Overfitting Is a Feature, Not a Bug

Enterprise desktop automation workflows are repetitive by design. The deployment distribution *is* the training distribution. When a user says "clear my browsing data," they want the model to execute the exact same 8-12 click sequence it saw in training, adapting only for minor UI variations (window position, resolution, theme). An overfit LoRA on a good foundation model gives you:

- **The LoRA memorizes the sequence** — which buttons to click, in what order
- **The base VLM handles variation** — different resolutions, slightly moved buttons, dark mode vs light mode
- Reliability of a macro with robustness of a vision model

### 2.2 The Abstraction Ladder

Per-task LoRA occupies a sweet spot on the automation spectrum:

| Level | Approach | Robustness | Reliability | Data Needed |
|-------|----------|------------|-------------|-------------|
| **0** | Literal pixel replay (macro) | Brittle — breaks on any UI change | Very high when UI is static | 1 recording |
| **1** | **Per-task overfit LoRA** | Moderate — VLM base handles variation | **High — memorized + flexible** | **~50 trajectories** |
| **2** | General LoRA + demo retrieval | Good — handles similar tasks | Moderate — accuracy degrades | All task data combined |
| **3** | Raw VLM reasoning (zero-shot) | Excellent — handles anything | Low — unreliable execution | None |

Level 1 is the sweet spot: more robust than macros, more reliable than general-purpose reasoning.

### 2.3 Cross-Task Confusion in Combined Training

When training one LoRA on a combined multi-workflow dataset, thin-tail workflows get drowned out by dominant ones. In a typical enterprise dataset:

- A few high-frequency workflows may have 30-60+ episodes
- Many workflows have only 4-10 episodes
- The model can't reliably distinguish "workflow A step 3" from "workflow B step 3" when both involve clicking dialog boxes

A per-task LoRA eliminates this confusion entirely — it only sees one flow. Thin workflows get proportionally more attention (5 epochs on 7 episodes vs. being 6% of a combined dataset).

## 3. Literature Support

### 3.1 Task-Specific LoRA Works for GUI Agents

| Model | Method | Key Result | Data |
|-------|--------|------------|------|
| ShowUI-Aloha | SFT on demo traces | +26.6pp from teaching traces (60.1% vs 36.7% without) | Human demos |
| SeeClick | LoRA grounding | +20pp over Qwen-VL baseline (67.0% vs 48.4%) | 2.8K samples |
| Qwen2.5-VL-32B | LoRA fine-tune | 48% → 66% exact-match | 1.7K samples |
| UI-TARS-2 | SFT + multi-turn RL | +10.5% on OSWorld (43.0% → 47.5%) | Data flywheel |
| LoRA Land | 310 task-specific LoRAs | Beat GPT-4 by 10pts average; 224/310 exceeded GPT-4 | Task-specific |
| OS-Atlas-7B | SFT on 13M+ GUI elements | SOTA ScreenSpot cross-platform | Large corpus |

**Takeaway**: Fine-tuning consistently delivers 10-26pp improvement over prompted baselines for GUI tasks.

### 3.2 Few Demonstrations Suffice

| Source | Finding |
|--------|---------|
| MeTA-LoRA (2025) | Competitive with full-data models using **50 examples per task** |
| SeeClick | +20pp grounding improvement with 2.8K total samples |
| Qwen2.5-VL LoRA | +18pp with 1.7K samples, 3 epochs, 2.3 hours on A100 |
| General empirical | LoRA outperforms full fine-tuning below 1,000 examples (less overfitting) |
| LoRAHub | Composes LoRAs for new tasks with just a few examples |

**Takeaway**: 50-100 trajectories per task is likely sufficient. Below that, LoRA still outperforms prompting for well-defined tasks.

### 3.3 Runtime LoRA Routing Is Solved

| System | Approach | Result |
|--------|----------|--------|
| LORAUTER (2026) | Task embeddings from validation sets | Matches Oracle (101.2%) when aligned adapters exist |
| vLLM Semantic Router | AI-semantic intent classification | +66-80% over keyword matching |
| Adaptive Minds | LLM-as-router (LoRA-as-Tools) | Open-source LangGraph implementation |
| MeteoRA | Trainable gating network | Auto-senses task intention |
| Vector DB routing (2026) | Similarity retrieval for LoRA composition | Scales to large adapter collections |
| LoRA on the Go (LoGo, 2025) | Training-free per-instance selection via forward-pass scoring | Outperforms trained routers by 3.6% on some benchmarks; no router training needed |
| LoraRetriever (2024) | Retrieve-then-compose: adaptively retrieves and blends LoRAs per prompt | Dynamic LoRA pool management |
| LD-MoLE (2025) | Differentiable routing replaces non-differentiable TopK, adaptive per-layer expert counts | Fine-grained routing control |

**Takeaway**: The routing problem is isomorphic to the demo retrieval problem OpenAdapt already solves via `MultimodalDemoRetriever`. Same embedding space, same nearest-neighbor lookup — just pointing to a LoRA instead of a demo file.

### 3.4 Production Serving at Scale Is Mature

| System | Scale | Overhead |
|--------|-------|----------|
| S-LoRA | 2,000 concurrent adapters | Up to 4x throughput over naive PEFT |
| Punica (SGMV kernel) | Concurrent per-request adapters | 12x throughput over vLLM baseline; latency constant with adapter count |
| vLLM (native) | Per-request LoRA selection, LRU cache | First-class API feature since Jan 2024 |
| LoRAX (Predibase) | 60+ concurrent models in production | Sub-2s latency, 10x cost reduction vs OpenAI |
| Compress then Serve | 1,000 LoRAs | 80% of single-model throughput |

**Takeaway**: vLLM (already in our stack) can serve thousands of LoRA adapters with negligible overhead.

### 3.5 No Catastrophic Forgetting (When Adapters Are Loaded/Unloaded)

Biderman et al. (TMLR 2024), "LoRA Learns Less and Forgets Less":
- LoRA substantially mitigates forgetting vs full fine-tuning
- But it still introduces "intruder dimensions" that can disrupt capabilities

**Critical point**: Task-specific adapters that are loaded/unloaded at runtime (never merged into base weights) **completely avoid** catastrophic forgetting. The base model is never modified. This is the strongest argument for adapter-routing over model-merging.

Additional VLM-specific findings:
- **VLM2VLA** (2025): LoRA fine-tuning VLMs into vision-language-action models without catastrophic forgetting when data representation aligns with pretraining distribution
- **Comp-LoRA** (2025): Complementary subspace regularization prevents forgetting during VLM LoRA tuning — can train specialists safely

### 3.6 Optimal Rank and Regularization

| Technique | Finding |
|-----------|---------|
| LoRA Dropout (ICLR 2025) | Theoretical generalization bound; especially effective with limited data |
| AutoLoRA (NAACL 2024) | Per-layer optimal ranks via meta-learning; uniform rank is suboptimal |
| NormAL LoRA (EMNLP 2025) | Rank-norm regularization for per-layer rank optimization |
| Image-LoRA (Dec 2025) | VLM-specific: LoRA only on visual-token attention path reduces FLOPs, preserves text reasoning |
| General guidance | Rank 4-8 for narrow tasks; rank 16-64 for complex domains; dropout essential for <100 examples |

## 4. Architecture

### 4.1 System Overview

```
┌──────────────────────────────────────────────────────────┐
│                     LoRA Registry                        │
│  skill_id → {lora_path, embedding, score, task_ids}      │
└────────────────────────┬─────────────────────────────────┘
                         │
    Instruction ──→ Embed ──→ Nearest Neighbor Lookup
                                      │
                      ┌───────────────┴────────────────┐
                      │ sim > τ                        │ sim ≤ τ
                      ▼                                ▼
              Load LoRA adapter                Fallback: DemoController
              PolicyAgent.predict()            or base model + retrieval
                      │                                │
                      └────────────┬───────────────────┘
                                   │
                             [Success?]
                              ├─ YES → Store trajectory → training queue
                              └─ NO  → Log for review
```

### 4.2 Key Components

#### LoRA Registry

A JSONL manifest (consistent with `demo_retrieval_design.md` format):

```jsonl
{"skill_id": "libreoffice_writer_font", "lora_path": "loras/writer_font/", "embedding": [...], "base_model": "Qwen/Qwen2.5-VL-7B-Instruct", "rank": 8, "score": 0.92, "task_ids": ["0e763496"], "created": "2026-03-16", "trajectories": 50}
{"skill_id": "libreoffice_calc_formulas", "lora_path": "loras/calc_formulas/", "embedding": [...], "base_model": "Qwen/Qwen2.5-VL-7B-Instruct", "rank": 8, "score": 0.85, "task_ids": ["04d9aeaf"], "created": "2026-03-16", "trajectories": 75}
```

#### LoRA Router

**Routing options, ordered by simplicity:**

1. **Explicit routing (simplest).** User/system selects task from a menu. Task ID → LoRA name. No ML needed. This is the standard enterprise RPA model — and for most deployments, it's sufficient.

2. **Instruction-based classifier.** Lightweight classifier (or embeddings) on user's text instruction. "Clear my browser cache" → `clear_browsing_lora`. Can be as simple as cosine similarity on sentence embeddings.

3. **Screenshot + instruction embedding (OpenAdapt-style).** CLIP-embed the first screenshot + instruction, compare against stored task embeddings. Repurpose `MultimodalDemoRetriever` for LoRA selection instead of demo selection.

4. **Base VLM as router.** Before loading any LoRA, ask the base VLM: "Given this screenshot and instruction, which of these tasks is the user performing? [list]". One inference call, then load the LoRA.

5. **LoRA on the Go (LoGo).** Training-free: run a single forward pass through all candidate LoRAs, score which one "reacts" most strongly to the input. No router training needed. Validated across 27 benchmarks and 3 model families.

Reference implementation using option 3:

```python
class LoRARouter:
    def __init__(self, registry_path: str, embedder: str = "Qwen3-VL"):
        self.registry = load_jsonl(registry_path)
        self.embedder = load_embedder(embedder)
        self.index = build_index([r["embedding"] for r in self.registry])

    def route(self, instruction: str, threshold: float = 0.7) -> Optional[str]:
        """Return lora_path if confident match, else None (fallback)."""
        embedding = self.embedder.encode(instruction)
        sim, idx = self.index.nearest(embedding)
        if sim >= threshold:
            return self.registry[idx]["lora_path"]
        return None
```

#### PolicyAgent with Dynamic LoRA

Extend existing `PolicyAgent` to accept per-request LoRA:

```python
class PolicyAgent:
    def predict(self, observation, lora_path: Optional[str] = None):
        if lora_path and lora_path != self._current_lora:
            self.model.load_lora(lora_path)
            self._current_lora = lora_path
        return self.model.generate(observation)
```

For vLLM serving, use the native per-request adapter API:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    extra_body={"lora_name": "libreoffice_writer_font"},
    messages=[...],
)
```

### 4.3 Training Pipeline

```
1. Collect trajectories (WAA recording pipeline or RL rollouts)
       ↓
2. Convert to SFT format (convert_demos.py)
       ↓
3. Train LoRA (QwenVLAdapter + PEFT, rank 8, dropout 0.1, 3 epochs)
       ↓
4. Evaluate on held-out configs (RLEnvironment)
       ↓
5. Register in LoRA Registry (with embedding + score)
       ↓
6. (Optional) GRPO refinement against evaluator
```

Per-task training cost: ~30-60 min on A100, ~$1-2 compute.

### 4.4 Task Granularity: Skills, Not Tasks

The right abstraction is **application-skill pairs**, not individual task instances:

| Skill ID | Application | Covers |
|----------|-------------|--------|
| `writer_font` | LibreOffice Writer | Any font change (Arial, Times, size, bold, etc.) |
| `calc_formulas` | LibreOffice Calc | Any formula entry (SUM, AVERAGE, custom) |
| `calc_formatting` | LibreOffice Calc | Zero-pad, decimal places, cell formatting |
| `vscode_settings` | VS Code | Any settings.json modification |
| `windows_settings_search` | Windows Settings | Any settings search + toggle |

An enterprise likely needs 50-200 skills to cover its automation portfolio. Each skill-LoRA handles variations within the skill via training data diversity.

## 5. Comparison: LoRA-per-Task vs. Demo-Conditioning

| Dimension | LoRA-per-task | Demo-conditioning (current) |
|-----------|---------------|-----------------------------|
| Known, repeated tasks | Much better (weights encode task) | Redundant overhead (context parsing) |
| Novel/unseen tasks | Fallback required | Can attempt with any demo |
| Inference latency | Single forward pass per step | Multi-turn VLM verification |
| Accuracy ceiling | Limited by training data quality | Limited by context window / parsing |
| Data requirement | ~50 trajectories per skill | 1 demo per task |
| Compositionality | Via LoRA composition (LoRAHub, MixLoRA) | Natural (text in context) |
| Error recovery | Implicit in weights | Explicit VLM verification + replan |
| Infrastructure complexity | vLLM + registry (simple) | DemoController + VLM + replan (complex) |
| Catastrophic forgetting | None (adapters loaded/unloaded) | N/A |
| Serving cost | Negligible per S-LoRA | High (multi-turn API calls) |

## 6. Hybrid Strategy: LoRA + Demo-Conditioning

Don't choose one exclusively. Use both at their strengths:

### 6.1 LoRA for Grounding, Demo for Planning

- **Demo-conditioning** provides the high-level **plan** (step decomposition)
- **Task-specific LoRA** provides low-level **grounding** (clicking the right pixel, typing in the right field)

This is what ShowUI-Aloha implicitly does: teaching traces handle planning, fine-tuned weights handle execution.

### 6.2 Graduated Migration

```
Phase 1: Demo-conditioning only (current)
    ↓ Collect trajectories from successful runs
Phase 2: LoRA for high-frequency skills + demo fallback for the rest
    ↓ Flywheel: successful executions → training data → new LoRAs
Phase 3: Mostly LoRA-driven, demo-conditioning for rare/novel tasks
```

### 6.3 LoRA as Compressed Demo

A LoRA adapter IS a compressed representation of demo trajectories — same information, but in a format directly usable by the model without parsing or in-context injection. The thesis shifts from "trajectory-conditioned disambiguation" to **"trajectory-conditioned weight specialization"**: same demos, same trajectories, compiled into weights instead of interpreted at runtime.

## 7. LoRA Composition for Novel Tasks

When no single LoRA matches, compose multiple:

| Method | How | When |
|--------|-----|------|
| **LoRAHub** | Gradient-free weight optimization with few examples | New task similar to existing skills |
| **MixLoRA** | MoE-style top-k routing across LoRA experts | Multi-skill tasks (e.g., format + formula) |
| **TIES-Merging** | Trim + resolve sign conflicts + merge | Offline combination of related skills |
| **DARE** | Drop 90-99% of delta params + rescale | Aggressive compression before merge |
| **AdapterFusion** | Attention-based fusion layer | When you have validation data for the composition |

For OpenAdapt, **LoRAHub** is the most practical starting point: given a novel task instruction and 3-5 example observations, compose weights from existing skill-LoRAs without gradient computation.

## 8. Data Collection Strategy

### 8.1 Sources of Training Trajectories

1. **Correction flywheel** (PR #116): When an agent fails a step and a human corrects it, the correction is stored as a `{think, action, expect}` dict — exactly the format SFT training expects. Each correction is a labeled training example generated at zero marginal cost during normal usage. A library of 50 corrections for one task IS 50 training examples for a task-specific LoRA.
2. **WAA recording pipeline**: VNC screenshot API → before/after PNGs → VLM annotate → SFT format (existing)
3. **Successful RL rollouts**: GRPO training generates trajectories for free
4. **Successful DemoController runs**: Every successful demo-conditioned execution is a training example
5. **Programmatic variation**: Record 1 trajectory, script state variations (different documents, cell values, fonts)
6. **Data augmentation**: Screenshot augmentation (crop, resize, color jitter), instruction rephrasing

### 8.2 Per-Skill Budget

| Skill complexity | Trajectories needed | Collection time | Training time |
|-----------------|--------------------|--------------------|---------------|
| Simple (1-3 steps) | 30-50 | ~3-5 hours | ~20 min |
| Moderate (4-10 steps) | 50-100 | ~8-15 hours | ~30-60 min |
| Complex (10+ steps) | 100-200 | ~15-30 hours | ~1-2 hours |

Programmatic state variation can 5-10x the effective dataset size from a single recorded trajectory.

### 8.3 The Flywheel: Corrections → LoRA

The correction flywheel (PR #116) is the natural data collection mechanism for LoRA-per-task:

```
Phase 1 (Human-in-the-loop):
  Agent fails → human corrects → correction stored as {think, action, expect}
  Next run: correction retrieved → agent succeeds

Phase 2 (Training):
  Correction library has 50+ entries for a task
  → Export as SFT training data (format already matches)
  → Train task-specific LoRA (rank 8, 3 epochs, ~30 min)

Phase 3 (Graduated):
  LoRA-PolicyAgent executes without correction retrieval
  → Task knowledge is in the weights, not the correction store
  → Correction store becomes fallback for edge cases the LoRA misses
```

This solves the data collection cost problem from Section 8.2. Instead of manually recording 50 trajectories per task upfront, corrections accumulate organically during normal usage. The cost is amortized across production runs — every failure becomes a training example.

## 9. Update Economics

Per-task LoRAs have dramatically better update economics than a single combined LoRA:

| Operation | Single General LoRA | Per-Task LoRAs |
|-----------|-------------------|----------------|
| Adding a new workflow | Retrain from scratch on all data | Train 1 new LoRA (~15-30 min) |
| Updating an existing workflow | Retrain from scratch | Retrain 1 LoRA (~15-30 min) |
| Removing a workflow | Retrain from scratch | Delete 1 adapter file |
| UI update breaks 1 workflow | Retrain from scratch | Retrain 1 LoRA with new demos |
| Storage at 100 workflows | ~50MB | ~5GB |
| Serving overhead | None | <4% (S-LoRA) |

This is the key operational argument: **changes are O(1) not O(N)**. At enterprise scale (50-200 workflows), this is the difference between a 15-minute fix and a multi-hour retraining cycle.

## 10. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overfit to specific UI state (pixel memorization) | High | Data augmentation, state variation, moderate rank (4-8) with dropout |
| Misrouting → complete failure on wrong LoRA | High | Confidence threshold with fallback; "no match" returns to base model |
| Combinatorial explosion of skills | Medium | Hierarchical skill taxonomy; LoRA composition for novel combinations |
| Stale LoRAs after UI updates | Medium | Version-stamp LoRAs; automated regression testing; retrain triggers |
| Training data quality | Medium | Filter trajectories by evaluator score; only train on score ≥ 0.8 |
| VLM-specific LoRA challenges (multimodal alignment) | Low | Image-LoRA technique; target visual attention only if text reasoning degrades |

## 11. Validation Plan

### Phase 0: Single-Task Proof of Concept

1. **Task**: `0e763496` (Writer font — already scored 1.0 in Trial 1)
2. **Collect**: 50 trajectories with varied initial states
3. **Train**: LoRA rank 8, dropout 0.1, 3 epochs on Qwen2.5-VL-7B
4. **Evaluate**: LoRA-PolicyAgent vs. DemoController on 10 held-out configurations
5. **Measure**: Accuracy, step count, latency, failure modes
6. **Success criterion**: LoRA matches or beats DemoController accuracy with ≤50% of the steps

### Phase 1: Core4 Expansion

Train LoRAs for all 4 Core4 tasks. Compare LoRA-only, demo-only, and hybrid (LoRA+plan) approaches.

### Phase 2: Routing Validation

Implement LoRA router. Test with mixed instructions across all 4 skills. Measure routing accuracy and fallback rate.

### Phase 3: Flywheel

Run 50 evaluation episodes. Measure trajectory yield (successful executions that generate usable training data). Retrain LoRAs with augmented data. Measure improvement.

## 12. Implementation in Existing Codebase

Most infrastructure already exists:

| Component | Status | Location |
|-----------|--------|----------|
| LoRA loading | ✅ Exists | `openadapt_ml/models/qwen_vl.py` (`weights_path` param) |
| SFT data conversion | ✅ Exists | `openadapt_ml/training/convert_demos.py` |
| PolicyAgent | ✅ Exists | `openadapt_ml/benchmarks/agent.py` |
| GRPO training | ✅ Exists | `openadapt_ml/training/grpo/` |
| RL environment | ✅ Exists | `openadapt_evals/adapters/rl_env.py` |
| Demo retrieval (embedding) | ✅ Exists | `openadapt-retrieval` / `RetrievalAgent` |
| vLLM serving | ✅ Exists | Already in dependency stack |
| LoRA registry | ❌ New | JSONL manifest + embedding index |
| LoRA router | ❌ New | Thin wrapper around existing retrieval |
| Training automation | ❌ New | Script to train LoRA from trajectory set |
| Fallback logic | ❌ New | Router → LoRA or DemoController |

**Estimated new code**: ~300-500 lines (registry, router, training script, fallback logic).

## 13. Open Questions

1. **Optimal skill granularity**: How broad can a skill-LoRA be before accuracy degrades? Empirical question for Phase 1.
2. **Cross-application transfer**: Does a "menu navigation" LoRA trained on Writer transfer to Calc? If yes, fewer LoRAs needed.
3. **LoRA composition quality**: How well does LoRAHub/MixLoRA work for GUI tasks specifically? No existing benchmarks.
4. **Rank selection**: Should we use AutoLoRA for per-layer rank optimization, or is uniform rank 8 sufficient?
5. **Retraining frequency**: How quickly do LoRAs go stale after OS/application updates?

## References

- ShowUI-Aloha (2025): [arXiv 2601.07181](https://arxiv.org/abs/2601.07181)
- SeeClick (ACL 2024): [arXiv 2401.10935](https://arxiv.org/abs/2401.10935)
- UI-TARS-2 (2025): [arXiv 2509.02544](https://arxiv.org/abs/2509.02544)
- LoRA Land (2024): [arXiv 2405.00732](https://arxiv.org/abs/2405.00732)
- LoRAHub (COLM 2024): [arXiv 2307.13269](https://arxiv.org/abs/2307.13269)
- MixLoRA (2024): [arXiv 2404.15159](https://arxiv.org/abs/2404.15159)
- LORAUTER (2026): [arXiv 2601.21795](https://arxiv.org/abs/2601.21795)
- S-LoRA (2023): [arXiv 2311.03285](https://arxiv.org/abs/2311.03285)
- Punica (2023): [arXiv 2310.18547](https://arxiv.org/abs/2310.18547)
- LoRA Learns Less and Forgets Less (TMLR 2024): [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- LoRA Dropout (ICLR 2025): [arXiv 2404.09610](https://arxiv.org/abs/2404.09610)
- AutoLoRA (NAACL 2024): [arXiv 2403.09113](https://arxiv.org/abs/2403.09113)
- Image-LoRA (2025): [arXiv 2512.19219](https://arxiv.org/abs/2512.19219)
- TIES-Merging (NeurIPS 2023): [arXiv 2306.01708](https://arxiv.org/abs/2306.01708)
- DARE (ICLR 2025): [arXiv 2311.03099](https://arxiv.org/abs/2311.03099)
- AdapterFusion (EACL 2021): [arXiv 2005.00247](https://arxiv.org/abs/2005.00247)
- MeTA-LoRA (2025): [arXiv 2510.11598](https://arxiv.org/abs/2510.11598)
- OS-Atlas (2024): [arXiv 2410.23218](https://arxiv.org/abs/2410.23218)
- Task Arithmetic (ICLR 2023): [arXiv 2212.04089](https://arxiv.org/abs/2212.04089)
- LoRA on the Go / LoGo (2025): [arXiv 2511.07129](https://arxiv.org/abs/2511.07129)
- LoraRetriever (2024): [arXiv 2402.09997](https://arxiv.org/abs/2402.09997)
- LD-MoLE (2025): [OpenReview](https://openreview.net/forum?id=4ST2YyTjI7)
- VLM2VLA (2025): [arXiv 2509.22195](https://arxiv.org/abs/2509.22195)
- Comp-LoRA (2025): [arXiv 2501.15040](https://arxiv.org/abs/2501.15040)
- Instance-Level Dynamic LoRA Composition (EMNLP 2024): [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.326/)
