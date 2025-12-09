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

### 2.4 Priority 4 — Real OpenAdapt Data Bridge

**Why**  
Synthetic-only is useful for unit tests; real world workflows are the end goal.

**Build Targets**

- **`openadapt_bridge.py` ingestion module**
  - Map OpenAdapt recordings → Session/Episode/Step/Action.
  - Extract screenshot paths.
  - Map low-level events to CLICK/TYPE/DONE.
  - Heuristics or annotations for episode-level goals.

- **Real-data evaluation CLI**
  - `scripts/eval_openadapt_policy.py`.
  - Reuse existing offline metrics.

- **Baseline experiment**
  - Compare:
    - Qwen base.
    - Qwen synthetic-FT.
  - Report metrics on real traces.

**Acceptance Criteria**

- Real OpenAdapt sessions load cleanly into canonical schema.
- Offline eval runs without modification.
- Synthetic-FT model shows any signal > baseline.

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

1. Priority 1: Batching + schedulers + logging.
2. Priority 2: Publishable login benchmark.
3. Priority 3: Second synthetic scenario + generalization.
4. Priority 4: Real-data ingestion + eval.
5. Priority 5a: API adapter + local CLI.
6. Priority 5b: Lambda orchestration (stretch).
7. Priority 6: CI + tests + repo hygiene (continuous).

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
