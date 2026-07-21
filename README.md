# OpenAdapt-ML

> [!IMPORTANT]
> **Status: experimental research. Not required by the product.** This package
> explores training and running demo-conditioned vision-language model (VLM)
> agents for GUI automation. It is evidence-generating research work with an
> unstable API, and it is not required to record, compile, or replay a
> workflow.
>
> The OpenAdapt product is a **governed demonstration compiler**:
> [`openadapt-flow`](https://github.com/OpenAdaptAI/openadapt-flow), installed
> via the [`OpenAdapt`](https://github.com/OpenAdaptAI/OpenAdapt) launcher
> (`pip install openadapt`). You record a workflow once, it compiles the
> demonstration into a deterministic, locally executable program, and it replays
> that program with **zero model calls on the healthy path**, halting instead of
> guessing when verification fails. Model training and grounding live here as a
> **research and cost-optimization surface (Phase 2)**, not as part of that
> deterministic replay path. Lifecycle labels for every repository are in the
> [repository lifecycle registry](https://github.com/OpenAdaptAI/.github/blob/main/REPOSITORY_LIFECYCLE.md).

[![Tests](https://github.com/OpenAdaptAI/openadapt-ml/actions/workflows/test.yml/badge.svg)](https://github.com/OpenAdaptAI/openadapt-ml/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/openadapt-ml.svg)](https://pypi.org/project/openadapt-ml/)
[![Downloads](https://img.shields.io/pypi/dm/openadapt-ml.svg)](https://pypi.org/project/openadapt-ml/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenAdapt-ML is the research ML layer for [OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt).
It provides the GUI-specific machinery for experimenting with vision-language
model (VLM) agents that automate desktop tasks: canonical schemas for GUI
trajectories, VLM adapters, supervised fine-tuning, visual grounding, online RL
(GRPO) experiments, and demo-conditioned inference.

## How this fits the product

OpenAdapt is a governed demonstration compiler. All substrates are first-class,
with honest maturity: Browser is in beta (the full record, compile, and replay
loop runs in CI); Windows, macOS, and RDP are early access; Citrix and VDI are
exploratory. That deterministic replay loop lives in
[`openadapt-flow`](https://github.com/OpenAdaptAI/openadapt-flow) and makes no
model calls when a run is healthy.

This repository sits deliberately upstream of that path. Everything here is
research aimed at the surfaces where a model may help: repairing or generalizing
a compiled step, grounding UI elements when structural cues are missing, and
reducing cost over time. Treat it as a lab, not a supported API. The APIs,
configs, and results below can and do change.

## Demos

**Synthetic Login** (Qwen3-VL-2B fine-tuned on synthetic UI scenarios):

![Login Demo](experiments/qwen_login/login_demo.gif)
![Registration Demo](experiments/qwen_login/registration_demo.gif)

## What is here

- **GUI trajectory schemas.** Pydantic models for `Episode`, `Step`, `Action`,
  and `Observation` with JSON Schema export and format converters (WAA,
  WebArena).
- **VLM adapters.** A unified interface for Qwen3-VL and Qwen2.5-VL (local) plus
  Claude, GPT, and Gemini (inference-only, API-backed), with automatic device
  selection (CUDA / MPS / CPU).
- **Supervised fine-tuning (SFT).** TRL `SFTTrainer` with optional Unsloth
  optimizations, training LoRA adapters.
- **Online RL (GRPO), experimental.** A Group Relative Policy Optimization
  training module that collects rollouts against a live environment. See the
  training status note below for what actually runs today.
- **Runtime policy API.** `AgentPolicy` predicts the next GUI action (`CLICK`,
  `TYPE`, `DONE`, and related types) from a screenshot and goal.
- **Demo-conditioned inference.** Retrieval-augmented prompting that conditions
  on recorded demonstrations for trajectory-aware disambiguation.
- **Grounding.** Locate UI elements via a vision API, oracle bounding boxes, or
  Set-of-Marks (SoM) overlays.
- **Recording segmentation.** Turn raw recordings into described, deduplicated
  segments.
- **Cloud GPU training.** One-command training pipelines for Lambda Labs, Modal,
  and Azure, plus local training.
- **Synthetic data generation.** Configurable UI scenarios (login, registration)
  with layout jitter for rapid iteration.

## Training status (read before you train)

Model training here is research and cost-optimization work, not the product's
healthy replay path. Two facts matter most:

- **A base VLM cannot operate Windows out of the box.** In practice you need an
  SFT checkpoint (or distillation) before online RL produces any signal.
  Un-fine-tuned base models yield near-zero reward on real GUI tasks.
- **The GRPO module has two backends at different maturity.**
  `GRPOConfig.backend="standalone"` (the default) is a built-in HuggingFace plus
  PEFT trainer intended for single-GPU prototyping and debugging.
  `backend="verl"` is an integration point for verl-agent / VAGEN
  (GiGPO, multi-GPU); it currently prints setup instructions and raises
  `NotImplementedError` rather than running a training job. Supervised
  fine-tuning uses TRL's `SFTTrainer` and is the most exercised training path.

Expect rough edges. This is where experiments happen.

## Installation

```bash
# Core package
pip install openadapt-ml

# With training dependencies (torch, transformers, TRL, PEFT, datasets)
pip install openadapt-ml[training]

# With API-backed VLMs (Claude, GPT)
pip install openadapt-ml[api]

# From source
git clone https://github.com/OpenAdaptAI/openadapt-ml.git
cd openadapt-ml
uv sync
```

Unsloth is optional and installed separately; see the
[Unsloth install guide](https://docs.unsloth.ai/get-started/installation).

## Quick start

### Run a smoke test (no GPU)

```bash
uv run python -m openadapt_ml.scripts.demo_policy --backend dummy
```

### Train on synthetic data

```bash
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_synthetic.yaml
```

### Train on real recordings

```bash
# Record a workflow with openadapt-capture, then train
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_capture.yaml \
  --capture ~/captures/my-workflow \
  --open  # Opens the training dashboard in a browser
```

### End-to-end benchmark (train, eval, plot)

```bash
uv run python -m openadapt_ml.scripts.run_qwen_login_benchmark \
  --config configs/qwen3vl_synthetic_dev.yaml \
  --out-dir experiments/qwen_login/2b_dev
```

### Use the policy API

```python
from openadapt_ml.runtime.policy import AgentPolicy
from openadapt_ml.models.qwen_vl import QwenVLAdapter

adapter = QwenVLAdapter(model_name="Qwen/Qwen3-VL-2B-Instruct")
policy = AgentPolicy(adapter)

# Given an SFT-style sample (screenshot, goal, chat history):
output = policy.predict(sample)
print(output.action)   # Action(type=CLICK, coordinates={"x": 0.45, "y": 0.71})
print(output.thought)  # "Click the Login button"
```

### Use the schema

```python
from openadapt_ml.schema import Episode, Step, Action, Observation, ActionType

episode = Episode(
    episode_id="demo_001",
    instruction="Open Notepad and type Hello World",
    steps=[
        Step(
            step_index=0,
            observation=Observation(screenshot_path="step_0.png"),
            action=Action(type=ActionType.CLICK, coordinates={"x": 100, "y": 200}),
        ),
        Step(
            step_index=1,
            observation=Observation(screenshot_path="step_1.png"),
            action=Action(type=ActionType.TYPE, text="Hello World"),
        ),
    ],
    success=True,
)
```

## Architecture

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
├── datasets/        # Episodes -> SFT chat samples
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

## Benchmark results

These are controlled synthetic results. They show that the training pipeline
runs end to end, not real-world performance.

### Synthetic Login (Qwen3-VL-2B with Set-of-Marks)

| Metric               | Score    |
|----------------------|----------|
| Action Type Accuracy | **100%** |
| Element Accuracy     | **100%** |
| Episode Success Rate | **100%** |

### Multi-model comparison (Synthetic Login, coordinate mode)

| Model             | Action Accuracy | Coord Error | Click Hit Rate |
|-------------------|-----------------|-------------|----------------|
| Qwen3-VL-2B FT    | 0.469           | 0.051       | 0.850          |
| Qwen3-VL-8B FT    | 0.286           | 0.004       | 1.000          |
| Claude Sonnet 4.5 | 0.121           | 0.757       | 0.000          |
| GPT-5.1           | 0.183           | 0.057       | 0.600          |

> This is a controlled synthetic benchmark with roughly three UI elements. It
> validates that the training pipeline works, not real-world accuracy.
> Evaluation on standard benchmarks (WAA, WebArena) is ongoing via
> [openadapt-evals](https://github.com/OpenAdaptAI/openadapt-evals).

## Cloud GPU training

### Lambda Labs

```bash
export LAMBDA_API_KEY=your_key_here

# Launch, train, download, and terminate in one command
uv run python -m openadapt_ml.cloud.lambda_labs train \
  --capture ~/captures/my-workflow \
  --goal "Turn off Night Shift in System Settings"
```

### Local (CUDA / Apple Silicon)

```bash
uv run python -m openadapt_ml.cloud.local train \
  --capture ~/captures/my-workflow --open
```

## Ecosystem

OpenAdapt-ML is one component in the OpenAdapt stack:

| Package | Purpose |
|---------|---------|
| **[OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt)** | Desktop automation platform and launcher (`pip install openadapt`) |
| **[openadapt-flow](https://github.com/OpenAdaptAI/openadapt-flow)** | The demonstration compiler: deterministic, zero-model-call replay on the healthy path |
| **[openadapt-ml](https://github.com/OpenAdaptAI/openadapt-ml)** | This repo: research ML (schemas, VLM adapters, training, inference, grounding) |
| **[openadapt-evals](https://github.com/OpenAdaptAI/openadapt-evals)** | Evaluation infrastructure: VM management, pool orchestration, benchmark runners, `oa-vm` CLI |
| **[openadapt-capture](https://github.com/OpenAdaptAI/openadapt-capture)** | Lightweight GUI recording and demo sharing |

> Looking for benchmark evaluation, Azure VM management, or the `oa-vm` CLI?
> Those live in [openadapt-evals](https://github.com/OpenAdaptAI/openadapt-evals).

## Documentation

- [docs.openadapt.ai](https://docs.openadapt.ai) for the product and the overall
  stack.
- [`docs/design.md`](docs/design.md) for system design (schemas, adapters,
  training, runtime).
- [`docs/cloud_gpu_training.md`](docs/cloud_gpu_training.md) for the Lambda Labs
  and Azure training guide.
- [`docs/qwen_login_experiment.md`](docs/qwen_login_experiment.md) for synthetic
  benchmark reproduction.
- [`docs/gemini_grounding.md`](docs/gemini_grounding.md) for the grounding
  module.

## Contributing

```bash
git clone https://github.com/OpenAdaptAI/openadapt-ml.git
cd openadapt-ml
uv sync --extra dev --extra training

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

We use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
`fix:`, `docs:`, and so on) with
[Python Semantic Release](https://python-semantic-release.readthedocs.io/) for
automated versioning and PyPI publishing.

## License

[MIT](LICENSE). OpenAdapt is open core: this repository is permissively licensed,
while private hardening corpora, tuned parameters, and deployment-derived recipes
are intentionally kept out of it.
