# Repository layout

Where things live in `openadapt_ml/`, and which other packages this one sits
next to.

## Package tree

```
openadapt_ml/
├── schema/          # Episode, Step, Action, Observation (Pydantic) + converters
├── models/          # VLM adapters (Qwen3-VL, Qwen2.5-VL, API backends, dummy)
│   └── providers/   #   Provider-specific client wiring
├── training/        # Fine-tuning + RL
│   ├── trl_trainer.py  #   TRL SFTTrainer (+ optional Unsloth)
│   ├── trainer.py      #   Training orchestration
│   ├── grpo/           #   GRPO online RL (standalone default; verl = stub)
│   └── viewer.py       #   Training dashboard (HTML)
├── runtime/         # Inference: AgentPolicy + action safety gate
├── datasets/        # Episodes -> SFT chat samples (next_action)
├── ingest/          # Synthetic UI, openadapt-capture loader, generic loader
├── grounding/       # UI element localization (oracle, vision API, SoM)
├── perception/      # Perception integration helpers
├── retrieval/       # Demo-conditioned retrieval for RAG-style prompting
├── segmentation/    # Recording -> described, deduplicated segments
├── baselines/       # Baseline agents and prompt/parse utilities
├── benchmarks/      # ML-specific benchmark agents (PolicyAgent, API, unified)
├── evals/           # Evaluation metrics (grounding, trajectory matching)
├── export/          # Dataset export (Parquet, CLI)
├── cloud/           # Cloud GPU training (Lambda Labs, Modal, Azure, vast.ai)
├── config.py        # Settings via pydantic-settings
└── scripts/         # CLI entry points (train, eval, compare, demo)
```

## What each area does

**Schemas.** Pydantic models for `Episode`, `Step`, `Action`, and `Observation`,
with JSON Schema export and converters for WAA and WebArena formats. Everything
else in the package reads and writes this shape.

**VLM adapters.** One interface over Qwen3-VL and Qwen2.5-VL running locally,
plus Claude, GPT, and Gemini for inference only. Device selection across CUDA,
MPS, and CPU is automatic. Build a local adapter with
`QwenVLAdapter.from_pretrained(model_name)`.

**Supervised fine-tuning.** TRL's `SFTTrainer` training LoRA adapters, with
optional Unsloth optimizations. This is the most exercised training path.

**Online RL.** A Group Relative Policy Optimization module that collects
rollouts against a live environment. `GRPOConfig.backend` defaults to
`"standalone"`, a HuggingFace plus PEFT trainer for single-GPU prototyping.
`backend="verl"` is an integration point for verl-agent and VAGEN (GiGPO,
multi-GPU) that currently prints setup instructions and raises
`NotImplementedError`.

**Runtime policy.** `AgentPolicy.predict_action_from_sample(sample)` returns a
4-tuple of `(Action, thought, state, raw_text)`. Action types include `CLICK`,
`TYPE`, `WAIT`, and `DONE`.

**Demo-conditioned inference.** Retrieval-augmented prompting that conditions on
recorded demonstrations, so a step can be disambiguated by what the human did in
the same situation.

**Grounding.** Locate a UI element by vision API, by oracle bounding box, or
through Set-of-Marks overlays. See [gemini_grounding.md](gemini_grounding.md).

**Recording segmentation.** Turn a raw recording into described, deduplicated
segments.

**Cloud GPU training.** One-command pipelines for Lambda Labs, Modal, Azure, and
vast.ai, plus local CUDA and Apple Silicon. See
[cloud_gpu_training.md](cloud_gpu_training.md).

**Synthetic data.** Configurable login and registration scenarios with layout
jitter, for iterating without recording anything.

## Neighbouring packages

| Package | Purpose |
|---|---|
| [OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt) | Desktop automation platform and launcher (`pip install openadapt`) |
| [openadapt-flow](https://github.com/OpenAdaptAI/openadapt-flow) | The demonstration compiler: deterministic, zero-model-call replay on the healthy path |
| [openadapt-ml](https://github.com/OpenAdaptAI/openadapt-ml) | This repository: schemas, VLM adapters, training, inference, grounding |
| [openadapt-evals](https://github.com/OpenAdaptAI/openadapt-evals) | Evaluation infrastructure: VM management, pool orchestration, benchmark runners, `oa-vm` CLI |
| [openadapt-capture](https://github.com/OpenAdaptAI/openadapt-capture) | Lightweight GUI recording and demo sharing |

Lifecycle labels for every repository are in the
[repository lifecycle registry](https://github.com/OpenAdaptAI/.github/blob/main/REPOSITORY_LIFECYCLE.md).
