# Research Thesis: Demo-Conditioned Action Selection for GUI Agents

## The Problem

Zero-shot VLMs fail on GUI tasks not due to lack of capability, but due to **ambiguity in UI affordances**. Given a screenshot and instruction, frontier models exhibit systematic spatial biases (e.g., clicking menu bar right instead of left). The observation-to-action mapping is learnable; the model simply does not know which element to click first.

## The Hypothesis

Demo-conditioning resolves this ambiguity by providing **procedural priors**. A single relevant demonstration—showing the correct navigation path—dramatically improves episode success. The demo acts as an entropy-reducing signal over the action space, not as additional reasoning capacity.

## Why This Sequencing

| Phase | Purpose |
|-------|---------|
| **1. Zero-shot baseline** | Calibrate evaluation harness. Establish that failures are real, not artifacts. |
| **2. Demo-conditioned** | The punchline. Prompt-level upper bound. Same model, same eval, no training. |
| **3. Fine-tuning** | Distillation—only after Phase 2 proves the signal exists. |

**Why prompt-level results must precede fine-tuning:**
- Fine-tuning collapses representation, prompt engineering, retrieval strategy, and model capacity into one opaque blob
- Gains become non-attributable—you cannot isolate what drove improvement
- Demo-conditioning isolates the causal factor: **trajectory priors**

## The Minimum Shocking Artifact

First-action accuracy on macOS System Settings tasks (n=45, Claude Sonnet 4.5):

| Condition | Accuracy | Delta |
|-----------|----------|-------|
| Zero-shot | 46.7% | — |
| **Demo-conditioned** | **100%** | **+53.3 pp** |
| Length-matched control | 57.8% | +11.1 pp |

Same model. Same prompt structure. Same evaluation harness. No fine-tuning.

The length-matched control rules out prompt verbosity—the benefit is semantic, not token-length.

> **Source**: [Demo-Conditioned Prompting Results](experiments/demo_conditioned_prompting_results.md)

## What This Proves

1. **Action space is executable.** The model can produce valid CLICK(x, y) actions that hit the correct target.

2. **Observation-to-action mapping is learnable.** Given the right context, accuracy reaches 100%.

3. **Failure mode was ambiguity, not capacity.** Zero-shot errors show consistent spatial bias (clicking right side of menu bar). With demo, model consistently identifies correct entry point.

4. **Retrieval is a control knob; fine-tuning is a sledgehammer.** Before investing in training infrastructure, demonstrate that prompt-level conditioning suffices for the target task distribution.

## Implications for Benchmarks

This framing applies directly to standard benchmarks:

- **OSWorld / WAA**: Desktop automation with complex navigation paths
- **WebArena / VisualWebArena**: Web tasks requiring procedural knowledge
- **Mind2Web / TTI**: Multi-step web navigation with branching decisions

The prediction: benchmarks showing low zero-shot success will exhibit large gains from demo-conditioning on the subset of tasks where the failure mode is "wrong first action" rather than "wrong goal understanding."

## Next Steps

1. **WAA Baseline**: Zero-shot evaluation on Windows Agent Arena (154 tasks)
2. **Demo Retrieval**: Given a new task, retrieve the most relevant demo from a library
3. **Episode Success**: Extend from first-action to full trajectory completion
4. **Fine-tuning (Phase 3)**: Distill demo-conditioned behavior into model weights—only after Phases 1-2 establish the signal

---

*December 2025 | OpenAdapt*
