# OpenAdapt-ML — Roadmap (Public + Agent-Executable)

This roadmap defines what OpenAdapt-ML is today and what will be built next, with concrete implementation targets.
It is written to guide both human contributors and autonomous coding agents.

## 1. Current Architecture Overview

OpenAdapt-ML provides:

- Canonical trajectory schema (Session → Episode → Step → Observation + Action)
- Synthetic UI generators (currently hardened login scenario)
- Next-action SFT dataset builder (strict CLICK/TYPE/DONE DSL)
- Model adapters (Qwen3-VL, Qwen2.5-VL, LoRA-enabled)
- Training loop (simple LoRA SFT)
- Offline evaluation (action accuracy, coord error, click hit rate)
- Runtime policy (regex-parsed Thought/Action output)

This stack is correct but minimal.
The next steps expand scale, generality, and real-world usefulness.

## 2. Roadmap (Prioritized Build Plan)

This section is the canonical list of what to build, in order, with crisp acceptance criteria.

### 2.0 Priority 0 — Fix Episode Success Rate (Critical Bug) ✅

**Why**
Episode success rate was 0% across ALL models (base, fine-tuned, and API) despite
fine-tuned models achieving up to 100% click hit rate. This was a critical
evaluation bug that masked the true performance of the system.

**Root Causes Identified**

1. **Missing TYPE and WAIT action parsing (BUG)**
   - `AgentPolicy._parse_action()` in `runtime/policy.py` only handled `CLICK` and `DONE`
   - `TYPE(text="...")` and `WAIT()` actions fell through to `Action(type="failed")`
   - Evidence from logs: 0% accuracy on TYPE (64 steps) and WAIT (32 steps)

2. **Overly strict episode success criterion**
   - Any single action type mismatch fails the entire 7-step episode
   - Even with perfect CLICK accuracy, TYPE/WAIT failures guarantee 0% episode success

**Fixes Implemented**

- ✅ Added `_TYPE_RE` regex pattern: `r'TYPE\(text="([^"\\]*(?:\\.[^"\\]*)*)"\)'`
- ✅ Added `_WAIT_RE` regex pattern: `r"\bWAIT\s*\(\s*\)"`
- ✅ Updated `_parse_action()` to handle all four DSL actions
- ✅ Added `tests/test_action_parsing.py` with comprehensive regex and parsing tests

**Acceptance Criteria**

- ✅ `AgentPolicy._parse_action()` correctly parses all DSL actions: CLICK, TYPE, WAIT, DONE
- ✅ TYPE action text is properly unescaped (handles `\"` and `\\`)
- ✅ Unit tests cover all action types and edge cases
- ✅ Re-run evaluation to measure true episode success rate

**Post-Fix Evaluation Results (2B Fine-tuned)**

| Metric | Before Fix | After Fix | Change |
|--------|-----------|-----------|--------|
| action_type_accuracy | 0.2545 | **0.4330** | +70% |
| mean_coord_error | 0.0138 | 0.0112 | -19% |
| click_hit_rate | 0.9737 | **1.0000** | Perfect |
| episode_success_rate | 0.0 | 0.0 | No change |

**Impact**

The parser fix was **confirmed successful**:
- Click hit rate improved to 100% (was 97.4%)
- Action type accuracy improved by 70% (0.25 → 0.43)

Episode success rate remains 0% because this is now a **model learning problem**, not a
parsing problem. The model is not predicting the correct action type sequences (e.g.,
predicting CLICK when ground truth is TYPE). With 43% action type accuracy, roughly
3 of 7 steps match per episode, which is insufficient for complete episode success.

**Next Steps**

The remaining episode success issue requires:
1. Analysis of per-action-type accuracy to identify which types the model struggles with
2. Potential improvements to training data, loss weighting, or training duration
3. See Priority 1 (batching/schedulers) for training infrastructure improvements

---

### 2.1 Priority 1 — Training + Adapters Upgrade (Batching, Schedulers, Logging)

**Why**  
Current Qwen3 trainer enforces `batch_size=1`, blocking GPU throughput and scaling.

**Build Targets**

- **True batching in `QwenVLAdapter.prepare_inputs`**
  - Accept `list[dict]` batch input.
  - Use `processor.apply_chat_template([...], padding=True, truncation=True)` for multi-sample tokenization.
  - Compute assistant-only labels per sample.
  - Ensure correct padding masks and label alignment.

- **Learning rate schedulers**
  - Add `lr_scheduler_type: [linear, cosine, none]`.
  - Compute warmup steps from `warmup_ratio`.

- **Run-directory logging**
  - Every training run creates `runs/<timestamp>_<config>/` with:
    - Config snapshot
    - Step-wise loss JSONL
    - Optional periodic eval metrics

**Acceptance Criteria**

- Qwen3-VL trains with `per_device_train_batch_size>1`.
- Loss curve stable.
- Configurable schedulers functional.
- Each run produces a self-contained log directory.

### 2.2 Priority 2 — Hardened Login Benchmark → Publishable Artifact

**Why**  
We need a clean, reproducible, public example that demonstrates LoRA fine-tuning improving GUI grounding.

**Build Targets**

- **Stable eval JSON schema**
  - Versioned output containing: metrics, run metadata, backend, config path.

- **Golden benchmark results**
  - Commit eval outputs for:
    - Qwen3-VL-2B base vs LoRA-FT
    - Qwen3-VL-8B base vs LoRA-FT

- **Plotting upgrade** ✅ (implemented and exceeded)
  - Comprehensive multi-model comparison plots with legends
  - Color-coded bars: blue (Qwen 2B/8B), orange (Claude API), red (GPT API)
  - Hatching patterns: solid (base/pretrained), diagonal stripes (fine-tuned)
  - Four key metrics per plot: action type accuracy, coord error, click hit rate, episode success
  - Supports arbitrary model combinations (base vs FT, offline vs API, comprehensive comparisons)

- **Documentation page**
  - `docs/qwen_login_experiment.md` describing:
    - Scenario
    - Training setup
    - Evaluation metrics
    - LoRA improvement plots

**Acceptance Criteria**

- Running:
  - `uv run python -m openadapt_ml.scripts.run_qwen_login_benchmark \
    --config configs/qwen3vl_synthetic_dev.yaml \
    --out-dir experiments/qwen_login/2b_dev`
  completes without error on a supported environment (e.g. CUDA GPU or Apple
  Silicon / CPU) using the documented config.
- The command above produces at least:
  - `experiments/qwen_login/2b_dev/eval/eval_base.json`
  - `experiments/qwen_login/2b_dev/eval/eval_ft.json`
  - `experiments/qwen_login/2b_dev/plots/base_vs_ft.png`
- Each eval JSON contains a top-level `metrics` object with:
  - `num_episodes`, `num_steps`, `action_type_accuracy`, `mean_coord_error`,
    `coord_error_count`, `episode_success_rate`, `click_hit_rate`.
- For the hardened 2B dev config, `action_type_accuracy_ft - action_type_accuracy_base`
  is **non-negative and typically >= 0.20` (LoRA does not regress vs. base).
- Documentation of the login benchmark is linked from the README.

### 2.3 Priority 3 — Add Second Synthetic Scenario (Generalization Test)

**Why**  
Today the system only tests login. A second scenario demonstrates robustness and multi-task capacity.

**Build Targets**

- **Settings Panel Generator**
  - Multiple toggles.
  - Save/Cancel buttons.
  - Layout jitter + decoys like login.

- **Scenario mixing**
  - Extend `generate_synthetic_sessions` with:
    - `scenario: ["login", "settings", "mixed"]`
    - `workflow_id` tagging

- **Multi-scenario training configs**
  - `qwen3vl_multi_scenario.yaml`

- **Cross-scenario evaluation matrix**
  - Train on: login-only, settings-only, mixed.
  - Eval on both scenarios.
  - Produce generalization heatmaps.

**Acceptance Criteria**

- Synthetic generator produces both scenarios deterministically.
- Eval matrix visualizes cross-scenario performance.
- Mixed model shows measurable generalization:
  - On held-out settings episodes, a model trained on login+settings achieves
    at least **0.05** higher `action_type_accuracy` than a login-only model,
    and symmetrically for settings-only vs mixed.

### 2.4 Priority 4 — Real Capture Data Bridge

**Why**
Synthetic-only is useful for unit tests; real world workflows are the end goal.

**Status**: DONE

Implementation uses openadapt-capture recordings (the modern capture tool) rather than the deprecated legacy OpenAdapt database.

**Completed**

- **`openadapt_ml/ingest/capture.py` ingestion module**
  - Maps openadapt-capture recordings → Episode/Step/Action
  - Extracts screenshots from video or screenshots/ directory
  - Maps events to CLICK/TYPE/DONE actions
  - Goals derived from directory name or specified via --goal

- **Training integration**
  - `--capture` flag in train.py to train on real recordings
  - Viewer and comparison tools work with real captures

**Acceptance Criteria**

- Real openadapt-capture recordings load cleanly into canonical schema. ✓
- Training pipeline works end-to-end with captures. ✓

### 2.5 Priority 5a — API VLM Adapter + Local CLI

**Why**  
Before introducing cloud orchestration, we want a clean way to run the same
benchmarks against hosted VLM APIs.

**Status**
Implementation complete:

- **Configuration System**
  - Pydantic-settings based configuration (`openadapt_ml/config.py`)
  - `.env` file support for API key management (`.env.example` provided)
  - Priority chain: explicit parameter > `.env` settings > environment variables > raise error
  - API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- **API Adapters**
  - `ApiVLMAdapter` (`openadapt_ml/models/api_adapter.py`) wraps:
    - Anthropic Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
    - OpenAI GPT-5.1 (`gpt-5.1`)
  - Inference-only adapters implementing `generate()` method
- **CLI Integration**
  - `scripts/eval_policy.py` supports `--backend claude` / `--backend openai`
  - `scripts/run_qwen_login_benchmark.py` supports `--include-claude`,
    `--include-openai`, or `--include-all-apis`
- **Visualization**
  - Comprehensive comparison plots with legends (`plot_eval_metrics.py`)
  - Color-coded bars: blue (Qwen 2B/8B), orange (Claude), red (GPT)
  - Hatching patterns: solid (base/pretrained), diagonal stripes (fine-tuned)
  - All evaluation plots support multi-model comparison

**Acceptance Criteria (all met)**

- ✅ `ApiVLMAdapter` can be dropped into `AgentPolicy` without code changes
- ✅ Local API eval CLI produces metrics JSONs compatible with `plot_eval_metrics.py`
- ✅ `ApiVLMAdapter` implements `generate(sample: dict) -> str` and returns the
  raw model text (no post-processing beyond what the remote API already performs)
- ✅ Configuration system with `.env` support and clear priority chain
- ✅ Comprehensive comparison plots with legends for multi-model evaluation

**Future Extensions (optional)**

- Add support for additional API providers as needed (e.g., Gemini, other Claude/GPT versions)
- Provider-specific configuration options (temperature, top_p, etc.)
- Richer logging for API calls (token usage, latency metrics)

### 2.6 Priority 5b — AWS Lambda Orchestration (Stretch)

**Why**  
Lambda is useful for lightweight compute orchestration and API-backed
inference, but not for GPU training.

**Build Targets (stretch)**

- **Lambda inference endpoint**
  - Input: `{goal, image_s3_uri}`.
  - Lambda:
    - Downloads image.
    - Builds SFT-style prompt.
    - Calls API-backed adapter.
    - Returns parsed Action JSON.

- **Synthetic generation Lambda (optional)**
  - Parallel generation of synthetic batches → S3.

- **Training orchestration Lambda (optional)**
  - Trigger ECS/SageMaker GPU jobs from configs.

**Non-goal**

- No local model loading in Lambda (no GPUs, slow cold starts).

**Acceptance Criteria (stretch)**

- Public Lambda endpoint returns structured `Action` for any uploaded
  screenshot.
- Adapters work interchangeably: Qwen local vs API remote.

### 2.7 Priority 6 — Tests, CI, Style (Continuous Workstream)

**Build Targets**

- **CI (GitHub Actions)**
  - `uv sync`.
  - `pytest`.
  - `ruff` lint.

- **Critical tests**
  - Action parser regex.
  - Adapter `prepare_inputs` (mock tokens).
  - Metric correctness tests.

- **Style consistency**
  - Enforced `ruff` + `black`.
  - `CONTRIBUTING.md` updated.

**Acceptance Criteria**

- Every PR triggers CI pipeline.
- Adapters + metrics covered by unit tests.

## 3. Execution Order (Authoritative)

This is the order coding agents should follow unless explicitly overridden:

### Critical Path (must achieve >0% episode success before other work)

0. **Priority 0: Fix Episode Success Rate** ✅ (parsing fix DONE, but 0% success persists)
1. **Priority 0.1: Validate prompts on known benchmark** ⚠️ NEW
   - Test on one OSWorld or WebVoyager task to compare against published numbers
   - Ensure prompts and action extraction are working correctly
   - Reference: TTI repo (`scripts/prompts/create_prompt_json.py`)
2. **Priority 0.2: Establish upper bound with larger models** ⚠️ NEW
   - Prompt Qwen 32B and frontier APIs (Claude, GPT) on synthetic benchmark
   - If larger models also fail, the problem is in prompts/action format
   - If larger models succeed, smaller models need more training data or better architecture
3. **Priority 0.3: Achieve >0% episode success** ⚠️ BLOCKING
   - This is the gate for all other work
   - Without task completion, all other metrics are noise

### Post-Validation Priorities

4. Priority 1: Batching + schedulers + logging.
5. Priority 2: Publishable login benchmark (only after >0% success).
6. Priority 3: Second synthetic scenario + generalization.
7. Priority 4: Real-data ingestion + eval.
8. Priority 5a: API adapter + local CLI ✅ (DONE).
9. Priority 5b: Lambda orchestration (stretch).
10. Priority 6: CI + tests + repo hygiene (continuous).

### Lesson Learned

We skipped the essential first step: validating prompts work on known benchmarks
before fine-tuning. The correct order is:

```
prompts → API baselines → base model comparison → fine-tuning
```

See `docs/internal/vision-notes.md` for expert feedback details.

## 4. Agent Implementation Notes (Guardrails)

These rules are explicit so agents behave predictably and avoid breaking core contracts:

- **DSL stability**
  - Do not change the DSL grammar (`CLICK`, `TYPE`, `WAIT`, `DONE`) or argument
    names without:
    - updating all adapters and the runtime parser, and
    - extending parser tests to cover the new forms.
  - Backward-incompatible changes must bump a `dsl_version` field wherever it is
    serialized.
- **Schema stability**
  - Always use the canonical schema (`Session`/`Episode`/`Step`/`Observation`/`Action`).
  - Do not rename these types or their core fields; extensions must be additive
    (new optional fields) rather than destructive.
- **Adapter contract**
  - All VLM backends must implement the `BaseVLMAdapter` interface
    (`prepare_inputs`, `compute_loss`, `generate`).
  - Do not change method signatures; add new behavior via kwargs or new helper
    methods instead.
- **Synthetic scenario invariants**
  - All new scenarios must use:
    - Layout jitter.
    - At least one decoy element.
    - Deterministic random seeds for reproducible benchmarks.
- **Eval invariants**
  - All new eval CLIs must reuse the existing trajectory-matching metrics
    (action type accuracy, coord error, episode success rate, click hit rate),
    or extend them in a strictly additive way.
  - Policies and adapters must not rewrite or normalize DSL text (no JSON
    wrapping, added prefixes like `Action: CLICK(...)`, or whitespace
    rewriting) beyond strict parsing into an `Action`; the original output
    must be preserved in logs / `Action.raw`.
