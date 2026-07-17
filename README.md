# OpenAdapt-ML

> [!IMPORTANT]
> **Status: Research — not required by the product.** This package explores
> training and running demo-conditioned vision-language model (VLM) agents for
> GUI automation. It is evidence-generating research work and is not required
> to record, compile, or replay a workflow.
>
> The OpenAdapt product is the demonstration compiler,
> [`openadapt-flow`](https://github.com/OpenAdaptAI/openadapt-flow), installed
> via the [`OpenAdapt`](https://github.com/OpenAdaptAI/OpenAdapt) launcher
> (`pip install openadapt`): it compiles a demonstrated GUI workflow into a
> deterministic, locally executable program. Healthy runs make no model calls,
> and it halts instead of guessing when verification fails. Lifecycle labels for
> every repository are in the
> [repository lifecycle registry](https://github.com/OpenAdaptAI/.github/blob/main/REPOSITORY_LIFECYCLE.md).

[![Tests](https://github.com/OpenAdaptAI/openadapt-ml/actions/workflows/test.yml/badge.svg)](https://github.com/OpenAdaptAI/openadapt-ml/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/openadapt-ml.svg)](https://pypi.org/project/openadapt-ml/)
[![Downloads](https://img.shields.io/pypi/dm/openadapt-ml.svg)](https://pypi.org/project/openadapt-ml/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The ML engine for [OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt) -- open-source desktop automation with demo-conditioned AI agents.**

OpenAdapt-ML provides the GUI-specific ML layer for training and running vision-language model (VLM) agents that automate desktop tasks. It handles everything between raw screen recordings and a production policy API: canonical schemas for GUI trajectories, VLM adapters, supervised fine-tuning with TRL + Unsloth, grounding, and demo-conditioned inference.

## Demos

**Synthetic Login** -- Qwen3-VL-2B fine-tuned on synthetic UI scenarios:

![Login Demo](experiments/qwen_login/login_demo.gif)
![Registration Demo](experiments/qwen_login/registration_demo.gif)

## Key Features

- **GUI trajectory schemas** -- Pydantic models for Episodes, Steps, Actions, and Observations with JSON Schema export and format converters (WAA, WebArena)
- **VLM adapters** -- Unified interface for Qwen3-VL, Qwen2.5-VL, Claude, GPT, and Gemini with automatic device selection (CUDA / MPS / CPU)
- **Supervised fine-tuning** -- TRL SFTTrainer with Unsloth optimizations for 2x faster training and 50% less VRAM via LoRA adapters
- **Runtime policy API** -- `AgentPolicy` that predicts the next GUI action (`CLICK`, `TYPE`, `DONE`) from a screenshot and goal
- **Demo-conditioned inference** -- Retrieval-augmented prompting using recorded demonstrations for trajectory-conditioned disambiguation
- **Grounding module** -- Locate UI elements via Gemini vision API, oracle bounding boxes, or Set-of-Marks (SoM) overlays
- **Cloud GPU training** -- One-command training pipelines for Lambda Labs and Azure
- **Synthetic data generation** -- Configurable UI scenarios (login, registration) with layout jitter for rapid iteration

## Installation

```bash
# Core package
pip install openadapt-ml

# With training dependencies (TRL + datasets)
pip install openadapt-ml[training]

# With API-backed VLMs (Claude, GPT)
pip install openadapt-ml[api]

# Development (from source)
git clone https://github.com/OpenAdaptAI/openadapt-ml.git
cd openadapt-ml
uv sync
```

## Quick Start

### Run a smoke test

```bash
# Model-free policy demo (no GPU required)
uv run python -m openadapt_ml.scripts.demo_policy --backend dummy
```

### Train on synthetic data

```bash
# Fine-tune Qwen3-VL on synthetic login scenario
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_synthetic.yaml
```

### Train on real recordings

```bash
# Record a workflow with openadapt-capture, then train
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_capture.yaml \
  --capture ~/captures/my-workflow \
  --open  # Opens training dashboard in browser
```

### End-to-end benchmark (train + eval + plot)

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

# Given an SFT-style sample (screenshot + goal + chat history):
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
├── schema/              # Episode, Step, Action, Observation (Pydantic models)
│   ├── episode.py       #   Core dataclasses + JSON Schema export
│   └── converters.py    #   WAA/WebArena format converters
├── models/              # VLM adapters
│   ├── base_adapter.py  #   BaseVLMAdapter ABC
│   ├── qwen_vl.py       #   Qwen3-VL, Qwen2.5-VL
│   ├── api_adapter.py   #   Claude, GPT (inference-only)
│   └── dummy_adapter.py #   Fake adapter for testing
├── training/            # Fine-tuning pipeline
│   ├── trl_trainer.py   #   TRL SFTTrainer + Unsloth
│   ├── trainer.py       #   Training orchestration
│   └── viewer.py        #   Training dashboard (HTML)
├── runtime/             # Inference
│   ├── policy.py        #   AgentPolicy (screenshot -> action)
│   └── safety_gate.py   #   Action safety checks
├── datasets/            # Data loading
│   └── next_action.py   #   Episodes -> SFT chat samples
├── ingest/              # Data ingestion
│   ├── synthetic.py     #   Synthetic UI generation
│   ├── capture.py       #   openadapt-capture loader
│   └── loader.py        #   Generic episode loader
├── grounding/           # UI element localization
│   ├── base.py          #   OracleGrounder, GroundingModule ABC
│   └── detector.py      #   GeminiGrounder, SoM overlays
├── retrieval/           # Demo-conditioned inference
│   ├── retriever.py     #   Demo retrieval for RAG prompting
│   └── embeddings.py    #   Screenshot/action embeddings
├── benchmarks/          # ML-specific benchmark agents
│   └── agent.py         #   PolicyAgent, APIBenchmarkAgent, UnifiedBaselineAgent
├── cloud/               # Cloud GPU training
│   ├── lambda_labs.py   #   Lambda Labs integration
│   ├── local.py         #   Local training (CUDA/MPS)
│   └── ssh_tunnel.py    #   SSH tunnel management
├── segmentation/        # Recording segmentation pipeline
├── evals/               # Evaluation metrics (grounding, trajectory matching)
├── config.py            # Settings via pydantic-settings
└── scripts/             # CLI entry points (train, eval, compare, demo)
```

## Benchmark Results

### Synthetic Login (Qwen3-VL-2B with Set-of-Marks)

| Metric                | Score    |
|-----------------------|----------|
| Action Type Accuracy  | **100%** |
| Element Accuracy      | **100%** |
| Episode Success Rate  | **100%** |

### Multi-Model Comparison (Synthetic Login, coordinate mode)

| Model               | Action Accuracy | Coord Error | Click Hit Rate |
|----------------------|-----------------|-------------|----------------|
| Qwen3-VL-2B FT      | 0.469           | 0.051       | 0.850          |
| Qwen3-VL-8B FT      | 0.286           | 0.004       | 1.000          |
| Claude Sonnet 4.5    | 0.121           | 0.757       | 0.000          |
| GPT-5.1              | 0.183           | 0.057       | 0.600          |

> These are results on a controlled synthetic benchmark with ~3 UI elements. They validate that the training pipeline works, not real-world performance. Evaluation on standard benchmarks (WAA, WebArena) is ongoing via [openadapt-evals](https://github.com/OpenAdaptAI/openadapt-evals).

## Cloud GPU Training

### Lambda Labs

```bash
export LAMBDA_API_KEY=your_key_here

# One-command: launch, train, download, terminate
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
| **[openadapt-ml](https://github.com/OpenAdaptAI/openadapt-ml)** | ML engine: schemas, VLM adapters, training, inference, grounding |
| **[openadapt-evals](https://github.com/OpenAdaptAI/openadapt-evals)** | Evaluation infrastructure: VM management, pool orchestration, benchmark runners, `oa-vm` CLI |
| **[openadapt-capture](https://github.com/OpenAdaptAI/openadapt-capture)** | Lightweight GUI recording and demo sharing |
| **[OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt)** | Desktop automation platform (end-user application) |

> Looking for benchmark evaluation, Azure VM management, or the `oa-vm` CLI? Those live in [openadapt-evals](https://github.com/OpenAdaptAI/openadapt-evals).

## Documentation

- [`docs/design.md`](docs/design.md) -- System design (schemas, adapters, training, runtime)
- [`docs/cloud_gpu_training.md`](docs/cloud_gpu_training.md) -- Lambda Labs and Azure training guide
- [`docs/qwen_login_experiment.md`](docs/qwen_login_experiment.md) -- Synthetic benchmark reproduction
- [`docs/gemini_grounding.md`](docs/gemini_grounding.md) -- Grounding module documentation

## Contributing

```bash
# Clone and install dev dependencies
git clone https://github.com/OpenAdaptAI/openadapt-ml.git
cd openadapt-ml
uv sync --extra dev --extra training

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

We use [Angular-style commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.) with [Python Semantic Release](https://python-semantic-release.readthedocs.io/) for automated versioning and PyPI publishing.

## License

[MIT](LICENSE)
