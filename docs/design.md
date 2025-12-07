# OpenAdapt-ML: High-Level Design (v1)

## 1. Purpose

OpenAdapt-ML is the **model-agnostic, domain-agnostic ML engine** for GUI automation agents. It:

- Turns **GUI interaction trajectories** (screens + actions + goals) into datasets.
- Fine-tunes **vision-language models (VLMs)** via LoRA/QLoRA.
- Exposes a simple **runtime policy API** for predicting the next GUI action.
- Is validated end-to-end on **synthetic, semantic UIs** before integrating real data.

This repo contains only **generic** ML and data logic.


## 2. Scope (v1)

v1 focuses on a minimal but complete pipeline:

1. **Canonical schema** for GUI interaction data:
   - `Session`, `Episode`, `Step`, `Observation`, `Action`.
   - Episodes are **goal-conditioned**.
   - Steps optionally include **thought/reasoning**.
2. **Synthetic semantic UIs**:
   - Login screen, settings panel, browser-style screens, etc.
   - Each episode encodes a concrete goal (e.g. "Log in as alice", "Clear browser cache").
3. **Dataset builder** for **goal-conditioned next-action SFT**:
   - Produces chat-style samples with images + messages.
   - Assistant outputs parseable textual actions (e.g. `CLICK(x=0.42, y=0.73)`).
4. **VLM adapter abstraction** with a **Qwen-VL implementation**:
   - Uses Hugging Face `transformers` + `peft` (LoRA/QLoRA).
   - Designed primarily for **Qwen3-VL** but extensible.
5. **Minimal training loop / CLI**:
   - Loads a base VLM, attaches LoRA, trains on synthetic data, saves the adapter.
   - Training hyperparameters align with common Qwen SFT recipes.
6. **Runtime policy**:
   - Given a screenshot and a goal, predicts the next action.
   - Parses textual actions into a structured `Action` object.


## 3. Non-goals (v1)

v1 explicitly does **not** implement:

- Real OS hooks, replay mechanisms, or system tray integration.
- Application-specific ingestion logic, custom taxonomies, or integration layers
- Large-scale distributed training or production deployment stacks.
- Advanced workflow mining (clustering, summarization, segmentation).


## 4. Architecture Overview

Directory layout (planned):

- `openadapt_ml/`
  - `schemas/`
    - `sessions.py` — canonical dataclasses.
  - `ingest/`
    - `synthetic.py` — semantic synthetic UIs and scripted episodes.
  - `datasets/`
    - `next_action.py` — goal-conditioned next-action SFT builder.
  - `models/`
    - `base_adapter.py` — `BaseVLMAdapter`, device utils.
    - `qwen_vl.py` — Qwen-family adapter with LoRA/QLoRA.
  - `training/`
    - `trainer.py` — minimal training loop over adapters + datasets.
  - `runtime/`
    - `policy.py` — `AgentPolicy` for inference.
  - `scripts/`
    - `train.py` — config-driven training entrypoint.
    - `demo_policy.py` — simple end-to-end demo on synthetic data.

This structure keeps **schemas**, **ingest**, **datasets**, **models**, **training**, and **runtime** cleanly separated.


## 5. Core Schema

All data is normalized into the following entities (Python dataclasses, type-hinted):

- **ActionType** (string or `Literal`):
  - For v1: `"click"`, `"double_click"`, `"right_click"`, `"drag"`, `"scroll"`, `"type"`, `"key_press"`, `"wait"`, `"done"`, `"failed"`.
  - We start by implementing `click`, `type`, and `done`; others are reserved for future use.

- **Action**:
  - `type: str` — one of the action types.
  - `x: float | None` — normalized [0, 1] horizontal coordinate (optional).
  - `y: float | None` — normalized [0, 1] vertical coordinate (optional).
  - `text: str | None` — text payload for `type` actions.
  - `raw: dict[str, Any] | None` — raw extra metadata (element ids, key names, etc.).

- **Observation**:
  - `image_path: str | None` — path to screenshot (PNG, etc.).
  - `meta: dict[str, Any] | None` — optional metadata (window title, app name, URL, etc.).

- **Step**:
  - `t: float` — timestamp or relative time.
  - `observation: Observation`.
  - `action: Action` — the action taken at this step.
  - `thought: str | None` — optional reasoning behind the action.

- **Episode**:
  - `id: str`.
  - `goal: str` — **required**, natural-language description of the task.
  - `steps: list[Step]`.
  - `summary: str | None` — optional high-level description.
  - `success: bool | None` — optional outcome flag (ground-truth goal completion).
  - `workflow_id: str | None` — optional cluster / taxonomy label (workflow type).

- **Session**:
  - `id: str`.
  - `episodes: list[Episode]`.
  - `meta: dict[str, Any] | None` — session-level metadata.

This schema is the **contract** between ingest, datasets, models, and runtime.

- **Training unit**: dataset builders and training loops operate on `list[Episode]`.
  `Session` is primarily an ingest/container type for grouping episodes and metadata.

- **DONE vs success semantics**:
  - Synthetic generators will include a terminal `Action(type="done")` once the scripted goal is achieved.
  - `Episode.success` records whether the goal was actually achieved in that episode.
  - This lets us evaluate both task success and whether the policy learns to emit `DONE()` at the right time.

- **Thought usage in v1**:
  - `Step.thought` is present for future ReAct-style supervision.
  - **v1 training ignores `thought`** and only supervises the final action string.
  - Thought supervision is a v1.1+ extension.


## 6. Synthetic Data: Semantic UIs

### 6.1 Motivation

We want synthetic data that:

- Looks like real desktop UIs (not abstract grids).
- Has **semantically meaningful elements** (username field, login button, etc.).
- Encodes **clear goals** and **scripted sequences** that achieve them.

This follows the spirit of `omnimcp`'s `synthetic_ui.py` and matches patterns in VideoAgentTrek (task-oriented episodes).

### 6.2 Planned synthetic scenarios

Initial synthetic UIs (v1):

1. **Login screen**
   - Fields: username, password; checkbox: "Remember Me"; link: "Forgot Password?"; button: "Login".
   - Example goal: `"Log in with username 'alice' and password 'hunter2'."`

2. **Settings panel**
   - Toggles: e.g. "Enable notifications", "Send usage data".
   - Buttons: "Save", "Cancel".
   - Example goal: `"Disable usage data and save settings."`

3. **Browser-style UI** (later in v1 or v1.1)
   - Toolbar: back, forward, refresh, settings.
   - Settings subpanel: "Clear cache", "Clear cookies".
   - Example goal: `"Clear the browser cache and cookies."`

Each scenario will be generated by dedicated helpers under `openadapt_ml/ingest/synthetic.py`.

### 6.3 Synthetic episodes

For each scenario, we script an **Episode** with:

- A **goal** string.
- A sequence of **Steps** with:
  - An `Observation.image_path` for each intermediate UI state.
  - An `Action` with normalized coordinates or text.
  - An optional `thought` summarizing the intent.

Example (conceptual only):

- Goal: `"Clear the browser cache and cookies."`
- Steps:
  - Step 0: click settings icon.
  - Step 1: click "Privacy" section.
  - Step 2: click "Clear data" button.
  - Step 3: ensure cookies checkbox is enabled.
  - Step 4: click "Confirm".
  - Step 5: `done` action when goal complete.

### 6.4 API

```python
# openadapt_ml/ingest/synthetic.py

def generate_synthetic_sessions(num_sessions: int = 10, seed: int | None = None) -> list[Session]:
    """Generate a list of synthetic Sessions with semantic UIs and goals.

    For v1, each Session may contain a small number of Episodes
    (e.g. 1–3) drawn from a set of scenario templates.
    """
```

We may later wrap this in a small metadata carrier if helpful for logging and reproducibility:

```python
@dataclass
class SyntheticDataset:
    sessions: list[Session]
    scenarios: dict[str, int]  # scenario_name -> count
    seed: int | None
```

All action coordinates are normalized **relative to the screenshot image** (range `[0, 1]`).
If needed, the original screen resolution can be stored in `Observation.meta`.
Synthetic generators save PNGs to disk and store their paths in `Observation.image_path` for simplicity and easy inspection.


## 7. Dataset Builder: Goal-Conditioned Next-Action SFT

We train the VLM via **supervised fine-tuning (SFT)** on chat-style samples.

### 7.1 Action text format

For v1, we encode actions as simple textual commands, e.g.:

- `CLICK(x=0.42, y=0.73)`
- `TYPE(text="alice")`
- `DONE()`

This keeps things:

- **Parseable** via regex.
- **Human-readable**.
- Compatible with standard Qwen SFT setups (text-only supervision on top of images).

### 7.2 SFT sample structure

Internal SFT sample (conceptual):

```python
{
    "images": ["/path/to/screen.png"],
    "messages": [
        {
            "role": "system",
            "content": "You are a GUI automation agent. Given a screen and a user goal, predict the single next action. Actions include CLICK, TYPE, SCROLL, DRAG, KEY, and DONE."
        },
        {
            "role": "user",
            "content": (
                "Goal: Clear the browser cache and cookies.\n"
                # Optional: "Previous actions: CLICK(x=0.95, y=0.10), CLICK(x=0.5, y=0.6)\n"
                "Current screen: see the attached image.\n"
                "Predict the next action."
            ),
        },
        {
            "role": "assistant",
            "content": "CLICK(x=0.95, y=0.10)",
        },
    ],
}
```

Later we may optionally include **history** and/or **thoughts** in the user content, but v1 can start with:

- Goal + current screen → next action.

### 7.3 Dataset components

Under `openadapt_ml/datasets/next_action.py` we will provide:

- A function to convert `list[Episode]` → `list[SFTSample]`.
- A small `torch.utils.data.Dataset` implementation wrapping these samples.
- Utilities for formatting `Action` objects into action strings.

### 7.4 Mapping Episodes to SFT Samples

For an `Episode` with steps `[s0, s1, ..., sN]` we create **one SFT sample per step** that has both an observation and an action, including the terminal `DONE()` step (if present). This allows the model to learn termination behavior.

For each step `k` in `range(len(steps))`:

- `image` = `steps[k].observation.image_path`
- `goal` = `episode.goal`
- `target_action` = textual serialization of `steps[k].action`, e.g. `"CLICK(x=0.42, y=0.73)"` or `"DONE()"`.

v1 deliberately uses a **one-step Markov assumption**:

- Input: `(goal, current screen)`
- Output: `next action`

History and thought are not used in v1 training, but the SFT sample format leaves room to add them later (by augmenting the user message).


## 8. VLM Adapter Abstraction

### 8.1 Device selection

`openadapt_ml/models/base_adapter.py` will provide a helper:

1. Use `"cuda"` if `torch.cuda.is_available()`.
2. Else `"mps"` if `torch.backends.mps.is_available()`.
3. Else `"cpu"`.

Adapters take an explicit `device` but default to this helper.

### 8.2 BaseVLMAdapter

The base class encodes the common API:

- Construction: wraps a Hugging Face model + processor + device.
- `prepare_inputs(batch: list[dict]) -> dict`:
  - Takes a list of SFT samples and returns HF model inputs
    (`input_ids`, `pixel_values`, `attention_mask`, `labels`, etc.).
- `compute_loss(inputs: dict) -> torch.Tensor`:
  - Runs the model and returns the loss (labels already in `inputs`).
- `generate(sample: dict, max_new_tokens: int = 64) -> str`:
  - Single-sample generation for runtime; returns assistant text.

Adapters are **stateless** model wrappers: they do not own optimizers, schedulers, or training state. All training state lives in the training loop.

### 8.3 QwenVLAdapter

`openadapt_ml/models/qwen_vl.py` will implement a `QwenVLAdapter` targeting
the Qwen-VL family via Hugging Face `transformers` + `peft`:

- **Primary target:** Qwen3-VL (e.g. `Qwen/Qwen3-VL-8B-Instruct`).
- **Secondary / compatibility target (future):** Qwen2.5-VL
  (e.g. `Qwen/Qwen2.5-VL-7B-Instruct` or compatible derivatives).

- `from_pretrained(model_name: str, lora_config: LoraConfig | dict | None, load_in_4bit: bool = True, device: torch.device | None = None)`:
  - Loads a base Qwen-VL model + processor.
  - Applies 4-bit loading (QLoRA) when requested (on CUDA-capable GPUs).
  - Attaches a LoRA adapter using `peft`.
- `prepare_inputs` (v1, Qwen3-VL path):
  - Reinterprets generic SFT samples into Qwen-style multimodal `messages`
    where the user content includes both an `image` entry and a `text` entry.
  - Uses `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")` to obtain model inputs.
  - For v1 synthetic experiments, uses **full-sequence supervision**
    (`labels = input_ids`) for simplicity; assistant-only masking is a planned
    refinement.
- `compute_loss` and `generate` delegate to the underlying HF model.


## 9. Training Loop & Config

### 9.1 Training configuration

We follow common Qwen-VL LoRA/QLoRA conventions, exposing a similar set of knobs. At a high level:

- Model:
  - `model_name`: e.g. `"Qwen/Qwen3-VL-8B-Instruct"`.
  - `load_in_4bit`: bool flag for QLoRA.
- LoRA:
  - `r` (rank): e.g. 16.
  - `alpha`: e.g. 32.
  - `dropout`: e.g. 0.05.
  - `target_modules`: e.g. `["q_proj", "v_proj"]`.
- Training:
  - `num_train_epochs`: 1–3.
  - `per_device_train_batch_size`: small (1–4).
  - `gradient_accumulation_steps`: 4–8.
  - `learning_rate`: ~2e-4 (LoRA typical range).
  - `warmup_ratio`: 0.03–0.1.
  - `weight_decay`: 0.0–0.1.
  - `lr_scheduler_type`: `"cosine"` or `"linear"`.
  - `max_grad_norm`: 1.0.
- Logging / saving:
  - `logging_steps`, `save_steps`, `save_total_limit`.

- Evaluation (optional in v1):
  - `eval_steps`: run a small synthetic eval every N training steps, if set.
  - `eval_episodes`: number of synthetic episodes to sample for validation.
  - `metric`: name of the primary metric (e.g. `"action_accuracy"`, `"coordinate_error"`).

These will be represented in a small `TrainingConfig` dataclass and a corresponding YAML/JSON schema used by `scripts/train.py`.

### 9.2 Training loop

`openadapt_ml/training/trainer.py` will expose a simple helper that:

1. Instantiates an optimizer (e.g. `AdamW` or `adamw_bnb_8bit` when available).
2. Creates a scheduler from the config.
3. Iterates over a `DataLoader` built from the SFT dataset.
4. Uses the adapter to prepare inputs and compute loss.
5. Supports gradient accumulation and basic logging.
6. Saves LoRA weights and config at the end.

We keep the loop intentionally compact and explicit so it can later be swapped for Hugging Face `Trainer` or `TRL` `SFTTrainer` if needed.

v1 assumes **single-device training** (one CUDA GPU, MPS, or CPU) with small per-device batch sizes and gradient accumulation. Multi-GPU / distributed training and `accelerate` are future extensions.

### 9.4 Quantization and Apple Silicon

On Apple Silicon (M1/M2/M3, macOS), `bitsandbytes` does **not** currently
provide GPU-based 4-bit quantization (QLoRA). Attempts to use
`load_in_4bit=True` on an M-series Mac will fail or fall back to
unsupported paths, since there is no CUDA backend.

For v1 this implies:

- **Local development on M2:** use standard LoRA (no 4-bit) with
  full-/mixed-precision weights and keep synthetic datasets and training
  configs small for speed.
- **GPU training (e.g., cloud):** enable `load_in_4bit: true` and use
  `bitsandbytes` on CUDA-capable GPUs where 4-bit QLoRA is supported.

### 9.3 `scripts/train.py`

The CLI script will:

1. Load a config file (YAML or JSON).
2. Generate synthetic sessions (`generate_synthetic_sessions`).
3. Flatten to episodes and build SFT samples.
4. Create a `NextActionDataset`.
5. Instantiate `QwenVLAdapter` with LoRA.
6. Run the training loop.
7. Save:
   - LoRA adapter weights.
   - Training config.
   - Dataset / synthetic scenario metadata.


## 10. Runtime Policy

`openadapt_ml/runtime/policy.py` defines an `AgentPolicy` that wraps a trained adapter:

- `__init__(adapter: BaseVLMAdapter, device: torch.device)`.
- `predict_action(image: Image.Image, goal: str, history: list[Action] | None = None, think: bool = False) -> tuple[Action, str | None]`:
  - Builds a single SFT-style sample with:
    - System prompt (instructions + supported actions).
    - User prompt including the **goal** and reference to the current screen.
    - (Optional) short history summary, if provided.
  - Calls `adapter.generate` to get an assistant text string.
  - Parses the text into:
    - an `Action` object (type, x/y, text), and
    - an optional `thought` string (if we adopt a `Thought: ...\nAction: ...` pattern later).

If parsing fails (e.g. malformed text), `predict_action` returns an `Action` with `type="failed"` and attaches the raw text and error info in `Action.raw`; the `thought` return value is `None`. This lets callers handle failures gracefully (retry, fallback, or logging).

v1 expects the assistant output to be an **action-only** string (e.g. `"CLICK(x=0.5, y=0.5)"`). In later versions we may adopt a combined `Thought: ...\nAction: ...` format; the parser will be designed to optionally extract a thought prefix if present.

A simple `scripts/demo_policy.py` will:

1. Load a trained LoRA adapter onto a base Qwen-VL model.
2. Load a synthetic screenshot from disk.
3. Call `AgentPolicy.predict_action` with a goal string.
4. Print the resulting `Action` and optionally compare it to ground truth.


## 11. Future Extensions

The v1 design intentionally leaves room for:

- **Thought supervision**:
  - Store `thought` in `Step` and optionally supervise a reasoning prefix before the action (ReAct-style).
- **Action history in prompts**:
  - Include a short natural-language history of previous actions.
- **Element-based actions**:
  - Move from pure coordinates to actions referencing specific UI elements (ids, bounds).
- **Multi-frame context**:
  - Use multiple images per sample to capture temporal transitions.
- **Real data ingest**:
  - Implement arbitrary external interaction trace ingestors **outside** this repo, mapping their traces into the `Session`/`Episode` schema.
- **Cloud runtime**:
  - Separate repo (e.g. `openadapt-cloud`) providing generic Docker images and job specs for training and serving adapters on Azure or other clouds.

For now, the priority is to get the v1 pipeline **working end-to-end** on synthetic semantic UIs, so we can demonstrate:

- Data schema coherence.
- Qwen-VL LoRA/QLoRA integration.
- Goal-conditioned next-action prediction.
- A clean runtime API for GUI agents.


## 12. Evaluation Strategy (Conceptual)

Although v1 focuses on getting the training pipeline working, the design supports two complementary evaluation modes:

### 12.1 Interactive Environment Evaluation

In an executable environment (e.g., a desktop automation arena or sandbox), we can:

- Execute actions predicted by `AgentPolicy` step-by-step.
- Observe the resulting UI state.
- Determine whether the **goal** was achieved by inspecting the final state.

Typical metrics:

- **Task success rate**: fraction of episodes where the agent achieves the goal within a step budget.
- **Step efficiency**: number of steps taken vs. scripted or optimal.

This mode depends on an external environment and is not required for the initial synthetic, offline validation, but the `AgentPolicy` interface is designed to plug into such environments.

### 12.2 Offline Trajectory Matching

For recorded interaction data (synthetic or real), where we have:

- An `Episode` with `goal` and a sequence of `Step`s.
- Each `Step` has an `Observation.image_path` and ground-truth `Action`.

We can evaluate the policy **without executing actions**, by:

1. For each step `k` in an episode:
   - Load the screenshot from `steps[k].observation.image_path`.
   - Call `AgentPolicy.predict_action(image, goal)`.
   - Compare the predicted `Action` to the ground-truth `steps[k].action`.
2. Aggregate metrics across steps and episodes.

Representative metrics:

- **Action type accuracy**: percentage of steps where `pred.type == gt.type`.
- **Coordinate error**: mean pixel distance between predicted and ground-truth coordinates for spatial actions (e.g., `click`, `drag`).
- **Sequence match rate**: fraction of episodes where all steps match within a tolerance.
- **Early termination rate**: fraction of episodes where the model predicts `DONE()` too early.

This can be implemented in a small evaluation helper (e.g., `openadapt_ml.evals.trajectory_matching`) and driven by a CLI script (e.g., `scripts/eval_trajectory.py`). Synthetic episodes provide an immediate, environment-free sanity check; other recorded episodes can be evaluated via the same interface once they are mapped into the `Episode` schema.

A key goal is to **compare base vs. fine-tuned models** on the same synthetic
benchmark (e.g., login), using identical evaluation code. For example, we can
evaluate a frozen Qwen-VL checkpoint and a LoRA-tuned checkpoint on the same
test episodes and visualize the difference in action accuracy and coordinate
error.

Over time, we can also:

- Log **training and evaluation curves** (loss, accuracy) and generate simple
  plots under `docs/` to visualize convergence across different models and
  datasets.
- Perform **cross-model comparisons** by running the same evaluation scripts
  against different backends (e.g. Qwen-VL variants, or external APIs such as
  OpenAI / Claude / Gemini via thin adapter wrappers) and summarizing results
  in tables/figures. These external comparisons would live in separate
  analysis notebooks or repos, while OpenAdapt-ML focuses on the common
  data/model/runtime interfaces.


## 13. Quickstart (Conceptual)

**Generate synthetic data (optional script):**

```bash
python -m openadapt_ml.scripts.prepare_synthetic \
  --num-sessions 100 \
  --output data/synthetic
```

**Train a LoRA adapter on synthetic data:**

```bash
python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_synthetic.yaml
```

**Run inference with a trained policy:**

```python
from PIL import Image
from openadapt_ml.models.qwen_vl import QwenVLAdapter
from openadapt_ml.runtime.policy import AgentPolicy

adapter = QwenVLAdapter.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    lora_config=...,            # or load saved adapter config
    # lora_weights_path will be handled by adapter/model-loading logic
)
policy = AgentPolicy(adapter, device=None)  # uses default device helper

image = Image.open("screenshot.png")
action, thought = policy.predict_action(
    image=image,
    goal="Clear the browser cache",
)
print(action, thought)
```

## 14. Roadmap (near-term)

The initial v1 implementation focuses on validating the synthetic pipeline and
Qwen-VL integration. Near-term priorities include:

- **Evaluation CLI**
  - Add a script (e.g. `scripts/eval_policy.py`) that runs `AgentPolicy` over
    synthetic episodes and reports next-action accuracy and simple
    episode-level success metrics. In later iterations this CLI can also
    emit metrics logs suitable for plotting training/eval curves.
- **Prompting & action format**
  - Strengthen the system prompt and in-prompt examples so models reliably
    emit a single `CLICK(...)` / `TYPE(...)` / `DONE()` command.
  - Slightly harden `AgentPolicy` parsing to better handle noisy outputs.
- **Label masking & batching**
  - Move from full-sequence supervision to assistant-only masking in
    `QwenVLAdapter.prepare_inputs`.
  - Relax the `batch_size=1` constraint to support small batches when
    running on GPUs.
- **Additional synthetic UIs**
  - Add at least one more semantic UI scenario (e.g. settings panel,
    search/filter flow) to stress-test generality.
- **GPU / QLoRA configs**
  - Provide example configs for CUDA environments that enable
    `load_in_4bit: true` and more realistic batch sizes, while keeping
    Apple Silicon configs full/mixed-precision.
- **CI and style tooling**
  - Add a minimal CI workflow running `uv sync` and `pytest` on pushes/PRs.
  - Standardize formatting/linting (e.g. `ruff`, `black`) and document the
    expected style in the README.
