# OpenAdapt Vision

## What OpenAdapt Is

OpenAdapt is the **API layer between your desktop and AI**.

It captures human-computer interactions, extracts workflow structure, and provides
the context that makes AI actually useful for desktop work.

## Core Thesis

**For GUI automation, you don't need AGI. You need a small model that's seen your workflows.**

A 2B parameter model, fine-tuned on domain-specific workflows, outperforms
general-purpose frontier models (Claude, GPT) on GUI action prediction. This is
because:

1. GUI interaction is a narrow, well-defined domain with constrained action spaces
2. Specialization compounds: the model learns *your* apps, *your* patterns
3. The workflow structure is known — the model just navigates it

## Architecture

OpenAdapt treats workflows as **state machines**, not pixel sequences:

```
Recording → Process Mining (pm4py) → Process Graph (DFG)
                                           ↓
                              States + Valid Transitions
                                           ↓
Screenshot + Goal + Current State → Transformer → Next State + Grounded Action
```

The process graph, mined from human recordings, acts as both curriculum and
constraint. The transformer learns to execute the state machine while grounding
abstract states in visual reality.

## Why Specialization Wins

| Model | Action Accuracy | Click Hit Rate |
|-------|-----------------|----------------|
| Qwen 2B Fine-tuned | 42.9% | 100% |
| Claude Sonnet 4.5 | 11.2% | 0% |
| GPT-5.1 | 23.2% | 66.7% |

General-purpose models must infer workflow structure from scratch on every query.
OpenAdapt agents *know* the structure — they just navigate it.

## Two Modes

1. **Automation mode**: Record workflow → fine-tune → replay autonomously
2. **Copilot mode**: Continuous context stream → AI understands what you're doing → assists in real-time

## The Flywheel

```
More recordings → Better process graphs → Better agents → More users → More recordings
```

## What's Built

- Recording and replay infrastructure (OpenAdapt main repo)
- Process mining integration via pm4py (PRs #560, #852)
- Fine-tuning pipeline proving specialization thesis (openadapt-ml)
- Privacy controls for enterprise deployment

## References

- [Design Document](design.md) — technical architecture
- [Roadmap](roadmap.md) — prioritized build plan
- [Action DSL](design.md#action-dsl) — canonical action format
