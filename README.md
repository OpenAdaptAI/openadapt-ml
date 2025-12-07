# OpenAdapt-ML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

OpenAdapt-ML is a **model-agnostic, domain-agnostic ML engine** for GUI
automation agents.

It provides:

- **Schemas** for GUI interaction trajectories (screens + actions + goals).
- **Synthetic semantic UI generation** for bootstrapping datasets.
- **Dataset builders** that turn episodes into next-action SFT samples.
- **VLM adapters** (Qwen3-VL, Qwen2.5-VL) using Hugging Face + PEFT.
- A minimal **supervised training loop** for fine-tuning.
- A simple **runtime policy** API that predicts the next GUI action.

The design is described in detail in [`docs/design.md`](docs/design.md).

---

## 1. Repository Structure

Key modules:

- `openadapt_ml/schemas/`
  - Canonical dataclasses for `Session`, `Episode`, `Step`, `Observation`,
    `Action`.
- `openadapt_ml/ingest/synthetic.py`
  - Synthetic semantic UI generator (e.g. login screen) that produces PNG
    screenshots and scripted episodes.
- `openadapt_ml/datasets/next_action.py`
  - Converts episodes into goal-conditioned, chat-style next-action SFT
    samples suitable for VLM fine-tuning.
- `openadapt_ml/models/base_adapter.py`
  - `BaseVLMAdapter` abstraction shared by all VLM backends.
- `openadapt_ml/models/qwen_vl.py`
  - `QwenVLAdapter` implementing support for **Qwen3-VL** and
    **Qwen2.5-VL**.
- `openadapt_ml/models/dummy_adapter.py`
  - Tiny fake adapter used to validate training and runtime flows without
    loading a real VLM.
- `openadapt_ml/training/trainer.py`
  - Minimal supervised training loop (`train_supervised`) with gradient
    accumulation and logging.
- `openadapt_ml/runtime/policy.py`
  - `AgentPolicy` that formats inputs for a VLM and parses textual actions
    like `CLICK(x=..., y=...)` and `DONE()` into structured `Action`s.
- `openadapt_ml/scripts/train.py`
  - CLI entry point for running synthetic-data training with a chosen
    model/config.
- `openadapt_ml/scripts/demo_policy.py`
  - CLI demo showing how to use `AgentPolicy` with different backends
    (dummy, Qwen3-VL, Qwen2.5-VL).

Configs and docs:

- `configs/qwen3vl_synthetic.yaml`
  - Synthetic training config for **Qwen3-VL-8B-Instruct**.
- `configs/qwen2_5vl_synthetic.yaml`
  - Synthetic training config for **Qwen2.5-VL-7B-Instruct**.
- `docs/design.md`
  - High-level design document (scope, architecture, schemas, adapters,
    training, runtime, and evaluation strategy).

---

## 2. Environment Setup

OpenAdapt-ML targets **Python 3.12** and uses [`uv`](https://github.com/astral-sh/uv)
for dependency management.

### 2.1 Install and sync

From the repository root:

```bash
# Ensure uv is installed (see uv docs for platform-specific install)
# Then:
uv sync
```

This will create a virtual environment (e.g. `.venv/`) and install all
packages declared in `pyproject.toml`.

### 2.2 Working inside the environment

Use `uv run` to execute Python modules and scripts with the synced
environment:

```bash
uv run python -m openadapt_ml.scripts.train --help
```

You can also run `pytest` or other tools via `uv run`.

---

## 3. Synthetic Data & Datasets

The v1 pipeline is validated on **synthetic, semantic UIs**, starting with a
simple login flow.

### 3.1 Synthetic sessions

Synthetic data is generated on the fly by `generate_synthetic_sessions` in
`openadapt_ml/ingest/synthetic.py` and used internally by the training
scripts.

You can also call it directly from Python:

```python
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions

sessions = generate_synthetic_sessions(num_sessions=2, seed=123, output_dir="synthetic_examples")
print(len(sessions), "sessions")
```

Each session contains episodes with:

- A **goal** (e.g. "Log in as demo user").
- A sequence of **steps**, each with:
  - An observation (screenshot path).
  - An action (e.g. `CLICK`, `TYPE`, `DONE`).

### 3.2 Next-action SFT samples

Episodes are converted into SFT-style samples by
`build_next_action_sft_samples` in `openadapt_ml/datasets/next_action.py`.

Each sample has the form:

```python
{
  "images": ["/path/to/screenshot.png"],
  "messages": [
    {"role": "system", "content": ...},
    {"role": "user", "content": "Goal: ...\nCurrent screen: ..."},
    {"role": "assistant", "content": "CLICK(x=..., y=...)"},
  ],
}
```

These samples are wrapped in a simple `NextActionDataset` for use with the
training loop.

---

## 4. Training

Training is driven by `openadapt_ml/scripts/train.py` and YAML configs under
`configs/`.

The training script:

1. Loads a config file (YAML).
2. Generates synthetic sessions.
3. Flattens to episodes and builds SFT samples.
4. Wraps them in a `NextActionDataset`.
5. Instantiates a VLM adapter (e.g. `QwenVLAdapter`).
6. Runs `train_supervised` over the dataset.

### 4.1 Qwen3-VL synthetic training

Config: `configs/qwen3vl_synthetic.yaml`

Key fields:

```yaml
model:
  name: Qwen/Qwen3-VL-8B-Instruct
  load_in_4bit: false  # 4-bit quantization is disabled on macOS / Apple Silicon

# LoRA config and training hyperparameters are also defined in the YAML.
```

Run:

```bash
uv run python -m openadapt_ml.scripts.train --config configs/qwen3vl_synthetic.yaml
```

This will:

- Download and load `Qwen/Qwen3-VL-8B-Instruct`.
- Generate a small synthetic dataset.
- Run a single-epoch supervised fine-tuning loop.
- Print loss values as training progresses.

### 4.2 Qwen2.5-VL synthetic training

Config: `configs/qwen2_5vl_synthetic.yaml`

Key fields:

```yaml
model:
  name: Qwen/Qwen2.5-VL-7B-Instruct
  load_in_4bit: false
```

Run:

```bash
uv run python -m openadapt_ml.scripts.train --config configs/qwen2_5vl_synthetic.yaml
```

This exercises the **Qwen2.5-VL** path in `QwenVLAdapter`, using a
`process_vision_info`-style helper internally to pack image inputs in the
format expected by the Qwen2.5-VL processor.

> Note: Both configs are sized for **small synthetic smoke runs**, not
> large-scale production training.

---

## 5. VLM Adapters

All VLM backends implement the `BaseVLMAdapter` interface in
`openadapt_ml/models/base_adapter.py`:

- `prepare_inputs(batch) -> dict`
- `compute_loss(inputs) -> torch.Tensor`
- `generate(sample, max_new_tokens=...) -> str`

### 5.1 QwenVLAdapter

`openadapt_ml/models/qwen_vl.py` implements `QwenVLAdapter`, which supports
both **Qwen3-VL** and **Qwen2.5-VL**:

- Detects the family from `model_name`.
- Uses `AutoProcessor` for chat + vision.
- For training:
  - Converts SFT samples into Qwen-style multimodal `messages` where the user
    turn includes both the image and the text.
  - Uses full-sequence supervision (`labels = input_ids`) for v1 synthetic
    experiments.
- For generation:
  - Feeds the screenshot + goal text as a user message.
  - Lets the model generate the assistant continuation, which the runtime
    policy then parses into an `Action`.

> See `docs/design.md` §8.3 for more details on the adapter design.

### 5.2 DummyAdapter

`openadapt_ml/models/dummy_adapter.py` provides a trivial baseline adapter
that ignores inputs and returns a fixed loss / text. It is used for:

- Validating that the training loop runs without loading a large VLM.
- Simple runtime policy demos.

---

## 6. Runtime Policy & Demos

The runtime policy is implemented in `openadapt_ml/runtime/policy.py` as
`AgentPolicy`.

### 6.1 AgentPolicy

`AgentPolicy` is initialized with a VLM adapter (dummy or real). Given an
SFT-style sample, it:

1. Calls `adapter.generate(sample)` to obtain assistant text.
2. Parses actions from strings like:
   - `CLICK(x=0.45, y=0.71)`
   - `DONE()`
3. Returns a structured `Action` plus an optional free-form `thought`.

### 6.2 Demo script

`openadapt_ml/scripts/demo_policy.py` demonstrates how to use
`AgentPolicy` with different backends.

Run with a **dummy** backend (fast, no model load):

```bash
uv run python -m openadapt_ml.scripts.demo_policy --backend dummy
```

Run with **Qwen3-VL** backend:

```bash
uv run python -m openadapt_ml.scripts.demo_policy --backend qwen3
```

Run with **Qwen2.5-VL** backend:

```bash
uv run python -m openadapt_ml.scripts.demo_policy --backend qwen2_5
```

Each invocation will:

- Generate a synthetic login episode and select one step.
- Build an SFT-style sample from that step.
- Use `AgentPolicy` to predict the next action.
- Print the raw messages and the parsed action/thought.

---

## 7. Testing

Basic tests are provided under `tests/`.

Run the test suite with:

```bash
uv run pytest
```

In particular:

- `tests/test_training_dummy.py` runs a smoke test over the training loop
  using `DummyAdapter`.

---

## 8. Limitations & Notes

- **Apple Silicon / bitsandbytes**:
  - On Apple Silicon (M1/M2/M3, macOS), `bitsandbytes` does not currently
    provide GPU-based 4-bit quantization (QLoRA).
  - All example configs set `load_in_4bit: false` for local runs.
  - 4-bit QLoRA is expected to run on CUDA-capable GPUs in remote
    environments.
- **Batching**:
  - For v1, `QwenVLAdapter` is implemented assuming `batch_size=1` for
    simplicity when handling multimodal inputs. The training configs are
    sized accordingly.
- **Evaluation**:
  - v1 focuses on smoke tests and qualitative behavior on synthetic data.
    More formal evaluation scripts and metrics are planned.

For deeper architectural details, see [`docs/design.md`](docs/design.md).

---

## 9. Roadmap (summary)

Planned near-term improvements include:

- **Evaluation CLI** to measure next-action accuracy on synthetic episodes.
- **Stronger prompting and examples** to stabilize `CLICK(...)` / `DONE()`
  style outputs.
- **Assistant-only label masking** and support for small batches > 1 in
  `QwenVLAdapter` when running on GPUs.
- **Additional synthetic UI scenarios** beyond login.
- **GPU / QLoRA config examples** for CUDA environments, while keeping
  Apple Silicon configs in full/mixed precision.
 - **CI and style tooling** such as a minimal GitHub Actions workflow
   running `pytest`, plus documented formatter/linter expectations.
 - **Plots and comparisons** such as simple training/evaluation curves and
   high-level comparisons between different models/datasets (including base
   vs fine-tuned Qwen-VL checkpoints on the same synthetic benchmark, and,
   in separate analysis code, external APIs like OpenAI/Claude/Gemini).
