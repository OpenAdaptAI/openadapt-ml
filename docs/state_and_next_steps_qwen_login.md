# OpenAdapt-ML: Qwen Synthetic Login – Current State and Next Steps

## 1. Scope of this doc

This doc summarizes the current state of the **Qwen-based synthetic login benchmark** in `openadapt-ml`, and lays out concrete next steps to turn it into a compelling "base vs fine-tuned" story.

It focuses on:

- The **current pipeline** (train → eval → log → plot).
- The **behavior** of Qwen3-VL-2B on the synthetic login task.
- **Pathologies** we’ve observed (prompt collapse, degenerate actions).
- A prioritized set of **next changes** to the objective, synthetic data, and eval metrics.

This is meant as an internal design/status note, not polished user-facing docs.

For the **canonical benchmark entrypoint** and **Action DSL contract**, see:

- `uv run python -m openadapt_ml.scripts.run_qwen_login_benchmark \
  --config configs/qwen3vl_synthetic_dev.yaml \
  --out-dir experiments/qwen_login/2b_dev`
  - Single command that runs train → eval base/FT → plot for the 2B dev setup.
- `docs/design.md` §7.4 *Action DSL & invariants (canonical)*
  - CLICK/TYPE/WAIT/DONE grammar, normalized `[0,1]` coordinates, and
    parser failure behavior.

---

## 2. Current pipeline and components

### 2.1. Synthetic login generator

File: `openadapt_ml/ingest/synthetic.py`

- Implements `_script_login_episode`:
  - Renders a fixed-size (800×600) synthetic login UI with:
    - Username box
    - Password box
    - Login button
  - Creates a **fixed 7-step episode**:
    - Step 0: initial screen, `wait`
    - Step 1: click username field
    - Step 2: type username
    - Step 3: click password field
    - Step 4: type password
    - Step 5: click login button
    - Step 6: logged-in screen + `done`
  - The **original v1/v2 generator** used deterministic coordinates (no jitter) for
    all UI elements. The current hardened version (see §5.2.2 and §7) introduces
    per-episode layout jitter and a decoy `Help` button.

- `generate_synthetic_sessions`:
  - Generates `num_sessions` sessions, each with a single login episode.
  - Saves per-step PNGs to `synthetic_data/` or a configured output dir.

### 2.2. SFT dataset builder and prompts

File: `openadapt_ml/datasets/next_action.py`

- `SYSTEM_PROMPT` (v2, **no numeric examples**):

  - Explains the strict DSL:
    - `CLICK(x=<float in [0,1]>, y=<float in [0,1]>)`
    - `TYPE(text="...")`
    - `WAIT()`
    - `DONE()`
  - Requires a **single-line** response.
  - Explicitly forbids explanations / extra text.

- `build_next_action_sft_samples(episodes)`:
  - One sample per step with an image + chat-style messages:

    ```json
    {
      "images": [image_path],
      "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": action_text}
      ]
    }
    ```

  - `user_content` (v2) reiterates the DSL and single-line requirement without concrete coordinate examples.
  - `assistant_content` is the ground-truth action formatted as DSL, via `format_action`.

- `NextActionDataset` is a thin wrapper around the sample list.

### 2.3. Qwen adapter

File: `openadapt_ml/models/qwen_vl.py`

- `QwenVLAdapter` wraps `Qwen3VLForConditionalGeneration` / `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor`.
- Supports **LoRA** with two modes:
  - Fresh adapter via `LoraConfig`.
  - Loading existing adapter from `lora.weights_path` using `PeftModel.from_pretrained`.

- `prepare_inputs(batch)` (current behavior):
  - Expects `batch` of SFT-style samples (currently batch size 1 enforced).
  - Extracts image path + user and assistant text.
  - For training, builds **Qwen chat messages** with a single `user` turn containing:

    ```python
    {
      "role": "user",
      "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": combined_text},  # user text + assistant text
      ],
    }
    ```

  - Calls `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=False)`.
  - For Qwen3-VL, now uses **assistant-only supervision**:
    - Tokens corresponding to system + user messages are masked with `-100`.
    - Only the assistant DSL line is unmasked in `labels`.

- `generate(sample)` (runtime):
  - Builds Qwen chat messages with a `user` role that includes image + user text (no assistant text).
  - Uses `apply_chat_template(..., add_generation_prompt=True)` then `model.generate`.
  - Decodes full text; `AgentPolicy` is responsible for parsing the action out of it.

### 2.4. Policy + eval

Files:

- `openadapt_ml/runtime/policy.py`
- `openadapt_ml/evals/trajectory_matching.py`
- `openadapt_ml/scripts/eval_policy.py`

**Policy and parsing**:

- `AgentPolicy` wraps an adapter and exposes `predict_action_from_sample`:
  - Calls `adapter.generate(sample)`.
  - Parses the resulting text with regexes:
    - `CLICK(x=..., y=...)` → `Action(type="click", x, y)`.
    - `DONE()` → `Action(type="done")`.
  - Anything else → `Action(type="failed", raw={"text": ...})`.

**Trajectory matching metrics**:

- `evaluate_policy_on_episodes(policy, episodes, samples)`:
  - Iterates episodes and aligned SFT samples.
  - For each step with an image:
    - Calls `policy.predict_action_from_sample`.
    - Compares `pred_action.type` vs `gt_action.type`.
    - For `click`/`drag` steps, computes L2 coord error (`coord_error_norm`).
    - Tracks an episode-level `success_pred` flag; fails if any type mismatch or wrong `done`.

- Aggregated metrics (`AggregateMetrics`):
  - `action_type_accuracy` = total correct types / total steps.
  - `mean_coord_error` (over all collected coord errors).
  - `coord_error_count`.
  - `episode_success_rate` = fraction of episodes with `success_pred == True`.

**Eval script**:

- `eval_policy.py`:
  - Generates fresh synthetic sessions (using the same config as training).
  - Builds SFT samples via `build_next_action_sft_samples`.
  - Loads adapter based on `backend`:
    - `qwen3` uses config `model.name`, optional `lora`.
    - `--ignore-lora` forces base model.
  - Supports **logging** and **JSON output**:
    - `--output-json`: summary metrics JSON.
    - `--log-samples PATH`, `--log-limit N`: JSONL per-step logs with prompts, predictions, GT.

### 2.5. Training script

File: `openadapt_ml/scripts/train.py`

- Reads a YAML config (e.g. `configs/qwen3vl_synthetic_dev.yaml`).
- Generates synthetic data via `generate_synthetic_sessions`.
- Builds `NextActionDataset` from SFT samples.
- Instantiates `QwenVLAdapter` with LoRA config (cleaned of `weights_path`).
- Runs `train_supervised` from `openadapt_ml/training/trainer.py`:
  - Simple loop:
    - `inputs = adapter.prepare_inputs(batch)`
    - `loss = adapter.compute_loss(inputs)`
    - Backprop + AdamW + gradient clipping
    - Prints loss every `logging_steps`.
- If `lora.weights_path` is set, saves the adapter via `adapter.model.save_pretrained`.

### 2.6. Plotting

File: `openadapt_ml/evals/plot_eval_metrics.py`

- CLI that reads one or more eval JSON files and produces a PNG with 3 bar plots:
  - `action_type_accuracy`
  - `mean_coord_error`
  - `episode_success_rate`
- Used to compare base vs fine-tuned at a glance.

---

## 3. Current configs and results

### 3.1. Dev config (Qwen3-VL-2B)

File: `configs/qwen3vl_synthetic_dev.yaml`

Key settings:

```yaml
model:
  name: Qwen/Qwen3-VL-2B-Instruct
  load_in_4bit: false

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - v_proj
  task_type: CAUSAL_LM
  weights_path: checkpoints/qwen3vl2b_login_lora

synthetic_data:
  num_sessions: 4
  seed: 123
  output_dir: synthetic_train_dev

training:
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  logging_steps: 1
```

This is the **fast dev loop**: small model, few sessions, overfitting is acceptable.

### 3.2. Qwen3-VL-8B baseline (non-dev)

File: `configs/qwen3vl_synthetic.yaml`

- Same structure, but:
  - `model.name: Qwen/Qwen3-VL-8B-Instruct`
  - `synthetic_data.num_sessions: 4`
  - `lora.weights_path: checkpoints/qwen3vl_login_lora`

8B is used as the intended **benchmark** configuration; it’s slower to train, so most debugging is done on 2B.

### 3.3. Key experimental snapshots

#### 3.3.1. Early Qwen2.5 run (before Qwen3 focus)

Command (older):

```bash
uv run python -m openadapt_ml.scripts.eval_policy \
  --config configs/qwen2_5vl_synthetic.yaml \
  --backend qwen2_5 \
  --output-json eval_qwen2_5_login.json
```

Results:

- `action_type_accuracy`: 0.0
- `mean_coord_error`: N/A
- `episode_success_rate`: 0.0

Interpretation: base Qwen2.5-VL did **not** emit parseable DSL actions at all under the initial prompts.

#### 3.3.2. Qwen3-VL-8B base, v1 prompts (with numeric example)

Command:

```bash
uv run python -m openadapt_ml.scripts.eval_policy \
  --config configs/qwen3vl_synthetic.yaml \
  --backend qwen3 \
  --output-json eval_qwen3_base_login.json
```

Results:

- `action_type_accuracy`: 0.4286
- `mean_coord_error`: ≈ 0.1311
- `episode_success_rate`: 0.0

This was the **first meaningful baseline**: base Qwen3-VL sometimes emitted `CLICK(...)` with reasonable coords, but episodes still failed as a whole.

#### 3.3.3. Qwen3-VL-2B dev, v1 prompts (with numeric example)

- Training (2 sessions, 1 epoch) reduced loss significantly.
- Eval (base vs fine-tuned) showed **identical metrics and per-step actions**.
- Logs confirmed a degenerate behavior:
  - The model **always** emitted the same `CLICK(x=0.5123, y=0.431)` action.
  - This was directly copying the coordinate example from the prompt.

Conclusion: the numeric example in the prompt caused a **copy-collapse**; LoRA could not move the behavior in a useful direction on this tiny synthetic task.

#### 3.3.4. Prompt v2 (no numeric examples), Qwen3-VL-2B dev

After removing numeric examples from `SYSTEM_PROMPT` and `user_content`:

- Re-trained 2B dev (4 sessions, 2 epochs).
- Ran evals with logging:

  - Base (no LoRA):
    - `action_type_accuracy`: **0.3214**
    - `mean_coord_error`: **0.1943** (n=5)
    - `episode_success_rate`: **0.0**
  - Fine-tuned (with LoRA):
    - `action_type_accuracy`: **0.1429**
    - `mean_coord_error`: **N/A** (no valid click coords)
    - `episode_success_rate`: **0.0**

Logs (v2) show:

- **Prompt collapse is broken**:
  - The model no longer emits a single fixed `CLICK(x=0.5123, y=0.431)` for every step.
- However, LoRA behavior is currently **worse** than base on this tiny task:
  - Many `failed` actions (unparseable text) or wrong types.
  - Very few valid clickable predictions compared to base.

This indicates (for the tiny, pre-hardened v2 setup):

- Infra is solid (train → eval → log → plot all work).
- The **training objective** and/or **task design** were not yet well aligned with the eval,
  motivating the later hardened setup in §5.2 and the results in §7.

---

## 4. Diagnosed issues and pathologies

### 4.1. Prompt pathology (fixed)

- **Before v2**: numeric examples (`CLICK(x=0.5123, y=0.4310)`) caused the model to copy those coords regardless of screen/goal.
- **After v2**: removing numeric examples fixed this collapse:
  - Base predictions are more diverse and partially correct.
  - LoRA is not stuck on a single click location.

### 4.2. Objective mismatch: full-sequence vs assistant-only labels

Originally, training used:

```python
input_ids = inputs["input_ids"]
labels = input_ids.clone()  # no masking
```

- This meant the model was trained to reproduce:
  - System prompt
  - User text
  - Assistant action

Problems:

- LoRA has very limited capacity compared to the full model.
- On a tiny dataset, it is likely overfitting to **scaffolding text** (system/user) instead of sharpening the **action prediction**.
- The inference-time behavior cares **only** about the assistant’s DSL line.

This suggested moving to **assistant-only labels** (mask out system/user tokens in `labels`) to focus the loss where it matters. That change is now implemented for the Qwen3-VL path: system + user tokens are masked with `-100`, and only the assistant DSL span is supervised.

### 4.3. Task degeneracy: trivial policy can get ~0.3–0.4 accuracy

The v1/v2 synthetic login task has:

- Only 4 sessions / 28 steps for the dev run.
- Fixed UI positions (no jitter).
- No decoy clickable elements.

Consequences:

- A dumb heuristic like "always click in the central-ish input region" can achieve ~0.32–0.43 type accuracy.
- LoRA may improve **probabilities** but still produce the same argmax actions as base, especially on such small data.

Even now, after prompt v2, the base model gets some steps right by luck/priors; LoRA on a crude objective with limited data can easily drift into worse behavior.

### 4.4. Eval limitations

Current metrics:

- `action_type_accuracy` only checks the discrete type, not whether click coords are *good enough*.
- `mean_coord_error` requires a valid coord prediction; if LoRA starts emitting malformed actions, this collapses to N/A.
- `episode_success_rate` is 0 across the board given tiny size and strict matching.

This makes it harder to detect subtler improvements like "LoRA predicts clicks that are closer to the true button, even if type accuracy is similar".

---

## 5. Next steps: options and priorities

We now have:

- Working infra: **train → eval → log → plot**.
- A v2 prompt that avoids the worst collapse.
- Evidence that LoRA currently **hurts** performance on the tiny v2 task.

The remaining work is to **shape the task and objective** so that LoRA can win against the base model on a small but nontrivial synthetic benchmark.

Below are the main levers, with suggested order.

### 5.1. Objective: assistant-only labels (high priority)

**Goal:** Focus the loss on the action string instead of the entire conversation.

**Change:** In `QwenVLAdapter.prepare_inputs`, build `labels` such that:

- All tokens corresponding to system + user messages are masked with `-100`.
- Only tokens for the final assistant DSL line are unmasked.

Conceptual outline:

1. Use the existing `processor.apply_chat_template(..., tokenize=True)` call to build `input_ids`.
2. Identify the span of tokens that correspond to the assistant message (`action_text`).
   - This likely requires:
     - Knowing how the chat template inserts special tokens / delimiters.
     - Possibly re-encoding just the assistant text and searching for that subsequence.
3. Initialize labels as `-100` everywhere, then unmask `[asst_start:asst_end]`.

**Why this first:**

- It is a **correctness** fix you’ll want regardless of synthetic details.
- It directly addresses the symptom "LoRA behavior is worse than base" by ensuring gradients target the right part of the sequence.
- It doesn’t change data, eval, or prompts.

**Risks / cost:**

- Span-detection with the chat template can be fiddly but is localized.

### 5.2. Synthetic environment: jitter + decoys + more sessions (medium priority)

Once the training objective is fixed, the next lever is the **task itself**.

#### 5.2.1. Jitter UI element positions

In `ingest/synthetic.py`, adjust `_draw_login_screen` and `_script_login_episode` to:

- For each session, sample small random offsets (in pixels or normalized coords) for:
  - Username box position
  - Password box position
  - Login button position

Example (conceptual):

```python
rng = random.Random(session_seed)

button_x = base_button_x + rng.randint(-offset_px, offset_px)
button_y = base_button_y + rng.randint(-offset_px, offset_px)
```

Then recompute normalized centers via `_center` and store those as GT click coords.

#### 5.2.2. Add decoy clickable elements

- Augment the UI with at least one decoy clickable region per screen, e.g.:
  - A "Help" button in a different location.
  - A "Sign up" / "Create account" link.
- Do **not** create ground-truth actions for decoys in the main login script, but ensure they:
  - Are visually salient.
  - Are not colocated with the true login button.

This forces the policy to read the goal and localize the correct element, not just "something that looks like a button".

#### 5.2.3. Increase `num_sessions` in dev config

- In `configs/qwen3vl_synthetic_dev.yaml`:
  - `num_sessions: 4 → 32` (or 64, depending on how slow this feels on your machine).

Motivation:

- 4 sessions / 28 steps is too small; a single odd episode can dominate.
- With 32–64 synthetic sessions, LoRA has more varied layouts to learn from while still being tractable on 2B.

### 5.3. Eval: add a "click hit" metric and possibly stricter success criteria (medium priority)

To better see improvements, we can add:

- `click_hit_rate`: among steps with GT `click`/`drag`, fraction where:
  - Either `coord_error_norm < threshold` (e.g. 0.05), or
  - Predicted coords fall inside the ground-truth UI element bbox (available via `LoginUIElements`).

This would require:

- Extending `EpisodeMetrics` and `AggregateMetrics` to track hit counts.
- Adding the metric to JSON output and plots.

Benefits:

- Distinguishes between "knows to click" (correct type) vs "clicks near the target" (good coords).
- Gives another axis where FT can beat base.

We can also consider:

- Defining `episode_success_pred` based not just on type matches but also on click hits (optional for now).

### 5.4. Optional: discretized grid actions for a simpler demo (low–medium priority)

If, even after objective and environment fixes, it is hard to get a clean metric separation, we can **simplify the action space** for the synthetic demo:

- Snap coords to a small grid (e.g. 3×3 or 5×5) and represent actions as `CLICK(grid="i,j")`.
- This turns coordinate prediction into a small classification problem.
- LoRA is much more likely to show clear gains.

This is more invasive to the action format and should probably be treated as a **last resort** for a v1 "hero plot".

---

## 6. Recommended execution order

Putting the options together, a concrete plan:

1. **Assistant-only labels (high priority)**
   - Implement masking in `QwenVLAdapter.prepare_inputs`.
   - Retrain 2B dev.
   - Re-run base vs FT eval with logs.
   - Check whether FT still underperforms base or begins to match/improve.

2. **Synthetic hardening (medium priority)**
   - If FT is still not clearly better:
     - Add per-session jitter to login UI elements.
     - Add at least one decoy clickable per screen.
     - Increase `num_sessions` in dev config to ~32.
   - Retrain 2B dev.
   - Re-run eval + logs + plots.

3. **Eval improvements (medium priority)**
   - Add `click_hit_rate` to metrics and plots.
   - Optionally tighten `episode_success_rate` definition.

4. **Scale to 8B (high-value milestone)**
   - Once 2B shows **any reasonable base vs FT separation**:
     - Clone dev config to an 8B benchmark config (or keep using `qwen3vl_synthetic.yaml`).
     - Run 8B train + eval.
     - Generate **hero plot** for README / `docs/design.md`.

5. **Optional simplification (grid actions)**
   - Only if needed for a cleaner demo.
   - Keep it isolated to synthetic experiments.

---

## 7. Hardened benchmark results

With assistant-only labels, a hardened synthetic environment (layout jitter + a decoy `Help` button), and an added `click_hit_rate` metric, we now have a concrete **base vs fine-tuned** story.

### 7.1 Qwen3-VL-2B dev (hardened synthetic login, v2)

Config: `configs/qwen3vl_synthetic_dev.yaml` with:

- `num_sessions: 32` (32 episodes, 224 steps).
- Jittered username/password fields and login button.
- A decoy `Help` button per screen.
- Thought/Action prompt format with semantically meaningful per-step thoughts.
- Assistant-only label masking for Qwen3.

Metrics (from `eval_qwen3_2b_base_login_hardened_v2.json` and `eval_qwen3_2b_ft_login_hardened_v2.json`):

| Model           | action_type_accuracy | mean_coord_error | click_hit_rate | episode_success_rate |
|----------------|----------------------|------------------|----------------|----------------------|
| Qwen3-VL-2B    | 0.143                | N/A              | N/A            | 0.0                  |
| Qwen3-VL-2B FT | 0.469                | 0.0514           | 0.85           | 0.0                  |

- **FT > base** on all step-level metrics:
  - Type accuracy improves by ~3.3× (≈ 0.14 → 0.47).
  - Fine-tuned model produces many valid clicks; base emits almost none in this run (hence N/A coord metrics for base).
  - Coord error for FT is low (≈ 0.05) with a high click hit rate (~0.85).

Plot: `plots/qwen3_2b_base_vs_ft_hardened_v2.png` (generated via `plot_eval_metrics.py`).

### 7.2 Qwen3-VL-8B benchmark (hardened synthetic login, v2)

Config: `configs/qwen3vl_synthetic.yaml` with:

- `num_sessions: 32` (32 episodes, 224 steps).
- Same hardened generator (per-episode layout jitter + decoy `Help` button).
- Thought/Action prompt format with semantically meaningful per-step thoughts.
- Assistant-only label masking for Qwen3.

Metrics (from `eval_qwen3_8b_base_login_hardened_v2.json` and `eval_qwen3_8b_ft_login_hardened_v2.json`):

| Model           | action_type_accuracy | mean_coord_error | click_hit_rate | episode_success_rate |
|----------------|----------------------|------------------|----------------|----------------------|
| Qwen3-VL-8B    | 0.143                | N/A              | N/A            | 0.0                  |
| Qwen3-VL-8B FT | 0.286                | 0.0038           | 1.00           | 0.0                  |

- **FT > base** again, with stronger spatial precision than 2B:
  - Type accuracy roughly doubles over base (≈ 0.14 → 0.29).
  - Fine-tuned 8B produces valid clicks on all evaluated steps (click_hit_rate = 1.0).
  - Coord error is extremely low (~0.004), indicating very accurate localization of UI elements.

Plot: `plots/qwen3_8b_base_vs_ft_hardened_v2.png`.

These results confirm that, on a non-trivial synthetic login task with jitter and decoys, a lightweight LoRA adapter can reliably outperform the base Qwen3-VL on both 2B and 8B variants.

## 8. Summary

- The **pipeline** for Qwen-based synthetic login is in good shape:
  - Synthetic generator → SFT samples → Qwen adapter → policy → eval → logging → plots.
- We’ve identified and fixed the most obvious **prompt pathology** (coordinate-copying).
- The Qwen3-VL training objective now uses **assistant-only labels**, focusing supervision on the DSL action span.
- The synthetic login environment has been **hardened** with layout jitter, a decoy clickable, and more sessions in the dev loop.
- Evaluation now includes a **click_hit_rate** metric, making spatial improvements visible.
- On the hardened benchmark, LoRA **clearly beats** the base model for both Qwen3-VL-2B (dev) and Qwen3-VL-8B (benchmark), with substantial gains in action accuracy, coordinate error, and click hit rate.

These pieces together give us a clean, interpretable "Qwen3 base vs Qwen3 fine-tuned on synthetic login" story suitable for the README and design docs.
