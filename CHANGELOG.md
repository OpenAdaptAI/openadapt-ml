# CHANGELOG


## v0.16.3 (2026-07-28)

### Bug Fixes

- Preserve benchmark agent failures
  ([`4090310`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4090310a38ee91394d1a04bb1ef635727a0868a0))

Preserve provider and parser failures as failures across the benchmark agents.

### Chores

- **release**: Enforce source-policy boundary
  ([`7246d98`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7246d984730ba811ebf69684a086e123cad95a43))

Add the generated fail-closed source-policy guard. Validate the tracked repository tree and built
  wheel and sdist artifacts before release.


## v0.16.2 (2026-07-28)

### Bug Fixes

- Stop reporting evaluation and grounding failures as empty results
  ([#70](https://github.com/OpenAdaptAI/openadapt-ml/pull/70),
  [`91c7c48`](https://github.com/OpenAdaptAI/openadapt-ml/commit/91c7c484173519abf60dd38dfba251cacaea5acd))

A failure rendered as a successful empty result is the defect class this project exists to catch in
  other people's software. This repository produces GRPO rewards, grounding metrics and exported
  artifacts, so here the same shape does not raise -- it yields a wrong number.

Each site below could not tell "I looked and the answer is empty/zero" from "I could not look", and
  returned the first for the second. Each fix makes the distinction representable rather than
  logged.

GRPO milestone reward (openadapt_ml/training/grpo/reward.py) -- highest blast radius, it is the
  training signal: * openadapt-evals not installed -> logged a warning and returned reward 0.0. *
  VLM judge raised -> caught per milestone; the milestone stayed in the denominator, so a 429
  depressed the reward exactly like an unmet milestone. * Milestone with no description ->
  `continue`d past but still counted, so the task could never score 1.0 and nothing said why. * No
  milestones / none of type "screenshot" -> returned 0.0. All now raise MilestoneEvaluationError.
  GRPOTrainer catches it at the one place that can decide, and returns None ("no measurement, keep
  the existing reward"), which the caller already handles -- rather than feeding an invented 0.0
  into compute_group_advantages and taking a gradient step on it. GRPOTrainer's unused
  `_compute_milestone_reward` raised the same 0.0 for an unknown task_id; it now raises KeyError,
  matching its sibling.

Grounding (openadapt_ml/grounding/): `GeminiGrounder.ground` printed the error and returned [];
  `_parse_bbox_response` swallowed JSONDecodeError/KeyError/ TypeError; `extract_ui_elements` did
  both. [] is the legitimate answer for "I looked and matched nothing", and evaluate_grounder scores
  it as best_iou 0.0 / centroid_hit False -- so a rate limit or a bad key was recorded as a
  grounding miss the model never made. These now raise GroundingError; the honest empty result still
  returns [].

Grounding metrics (openadapt_ml/evals/grounding.py): a step whose screenshot could not be opened was
  `continue`d past. Every GroundingMetrics property divides by len(results), so the sample left the
  denominator and the reported hit rate stayed high over a silently smaller sample. Now raises.

Parquet summary (openadapt_ml/export/parquet.py): `except ImportError: return` made
  `include_summary=True` a no-op that returned None exactly like success.

Azure login probe (scripts/setup_azure.py): an unrecognised az error returned True -- "I could not
  tell, so assume you are logged in". Now re-raises.

Also: PR #69 fixed `task_dir` being declared twice in GRPOConfig (PIE794) and

kept the correct, documented declaration. Nothing in the configured rule set `B,E,F,I,W` catches a
  duplicated class field, so the defect could return the moment the audit stopped. PIE794 is adopted
  as a single rule (repository count: 0) and tests/test_failure_is_not_success.py adds an AST guard
  over every dataclass in the package.

Every test in tests/test_failure_is_not_success.py is mutation-checked: with the production file
  reverted to its pre-fix form the matching test fails with "DID NOT RAISE" (behaviour, not a
  missing import -- the new exception types are resolved dynamically for exactly that reason).

Claude-Session: https://claude.ai/code/session_01NyCHrzA1psrKMFfroYbzaM

Co-authored-by: Claude Opus 5 <noreply@anthropic.com>

### Build System

- Bound ruff to >=0.16,<0.17 and make lint config explicit
  ([#69](https://github.com/OpenAdaptAI/openadapt-ml/pull/69),
  [`ab43a9d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ab43a9dd6a7b5f07ae335679ed1ae092a121c02a))

`ruff>=0.1.0` with no `[tool.ruff]` section at all meant every lint decision in this repository was
  inherited from whatever ruff version a runner happened to resolve. ruff 0.16.0 (2026-07-23) then
  changed both halves of that implicit contract at once:

* the default rule set grew from 59 rules to 413, taking `ruff check openadapt_ml/` from 0 findings
  to 1035; and * the default `include` gained `*.md`, so `ruff format` began rewriting Python code
  blocks inside README/USAGE documentation (3 files).

Neither needed a commit here to land. `main` is still green only because uv.lock happens to pin ruff
  0.14.9.

Changes:

* dev dependency bounded to `ruff>=0.16,<0.17`, and uv.lock relocked so CI actually runs 0.16.0
  against the new gate. The relock also repairs a lockfile that predated the `build` and
  `openadapt-types` dependencies; the only version changes are ruff and this package's own stale
  self-version, with no downgrades and no removals. * explicit `[tool.ruff]`: `line-length = 88`
  (what the tree already uses), `target-version = "py310"`, and `include` pinned to 0.16's default
  list minus `*.md`, so a future addition to either default is inert until someone edits this file.
  * explicit `select = ["B", "E", "F", "I", "W"]` with `ignore = ["E501"]`. This is the OpenAdapt
  house set plus flake8-bugbear, and is wider than what this repository actually enforced before
  (ruff's old default was effectively `E4,E7,E9,F`). Full 0.16 defaults were rejected: 1035 findings
  is not honestly reviewable, and ~90% of it is `UP` typing churn, `BLE001` on deliberately blind
  status probes, and `PLW1510`, whose fixes are all behaviour changes. Reasons are recorded inline.
  * `ruff check` in CI widened from `openadapt_ml/` to the whole repository, because every genuine
  defect the audit surfaced was in scripts/ or tests/, outside the old gate.

Defects fixed:

* scripts/generate_vm_screenshots_simple.py: two bare `except:` around PIL font loading also
  swallowed KeyboardInterrupt and SystemExit, so Ctrl-C during font load silently fell through to
  the fallback font instead of exiting. Narrowed to `except OSError`. *
  openadapt_ml/training/grpo/config.py: `task_dir` was declared twice in the same dataclass, with
  the documented declaration shadowed by a later bare one. Both defaulted to None so behaviour was
  unchanged, but editing the documented default would silently have had no effect. *
  scripts/p0_validate_demo_persistence.py: three `"Episode"` annotations referenced a name the
  module never imports. `from __future__ import annotations` keeps this from raising at import time,
  but the annotations were unresolvable to any runtime introspection. Imported under TYPE_CHECKING.
  * 10 x B904: exception chaining restored (`from e`) so a friendly "install X" ImportError no
  longer hides the underlying cause. * 16 x B905: every `zip()` now declares its intent.
  `strict=True` only where equality is already guaranteed by an adjacent guard or by construction,
  so it can never fire; `strict=False` with an inline reason where truncation is genuinely
  tolerated. The one substantive case is representation_shootout/evaluator.py, where a short
  `model_outputs` could silently shrink the sample set the aggregated metrics are computed over; its
  docstring already declared the correspondence, which is now enforced. * 5 x F811 redundant `import
  os` shadowing the module-level import, 4 x F841 dead bindings, 3 x E712 `== True/False` in
  assertions.

Reviewed and deliberately not changed: 33 S110/S112 sites are all best-effort dashboard, telemetry
  and VM-status probes that degrade to a default; none swallows a credential or decision-path
  failure. 7 RUF012 sites are read-only class-level registries. No B006 mutable default arguments
  exist in this tree.

`build:` rather than `fix:`: nothing in the shipped `openadapt_ml` package changes behaviour for a
  user. The bare-except and undefined-name defects are both in `scripts/`, which is not packaged,
  and the duplicate dataclass field had identical defaults. This should not burn a release.

Verified locally with ruff 0.16.0: `ruff check .` clean repository-wide, `ruff format --check
  openadapt_ml/` clean, 454 passed / 2 skipped.

Claude-Session: https://claude.ai/code/session_01NyCHrzA1psrKMFfroYbzaM

Co-authored-by: Claude Opus 5 <noreply@anthropic.com>

### Chores

- Gitignore .private/ ([#68](https://github.com/OpenAdaptAI/openadapt-ml/pull/68),
  [`e1ff03e`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e1ff03eadf8900cda26c3928f387da03ae9b5d06))

`.private/` is the workspace-wide convention for material that must never be published. It was not
  ignored here, so a directory created inside this checkout was one stray `git add` from being
  committed. Ignore it mechanically rather than relying on that never happening.

Claude-Session: https://claude.ai/code/session_01NyCHrzA1psrKMFfroYbzaM

Co-authored-by: Claude Opus 5 <noreply@anthropic.com>

### Documentation

- Add lifecycle status banner ([#66](https://github.com/OpenAdaptAI/openadapt-ml/pull/66),
  [`1e295f8`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1e295f800662aa7f79ee6de503029401c65dc92d))

Adds the standard lifecycle banner used across the org (matching the banner wave on
  openadapt-capture, openadapt-grounding, etc.), derived from this repo's classification in the
  repository lifecycle registry (OpenAdaptAI/.github repository-lifecycle.yml, reviewed 2026-07-15):
  Research.

Existing README content is unchanged below the banner.

Co-authored-by: Claude Fable 5 <noreply@anthropic.com>

- Refresh README with honest research positioning
  ([#67](https://github.com/OpenAdaptAI/openadapt-ml/pull/67),
  [`0a1d05c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/0a1d05c4348a5aec062ca46c3f9f7006eb6fa72a))

Align openadapt-ml's README with the OpenAdapt house positioning and its actual, current maturity as
  experimental research code.

- Frame the product as a governed demonstration compiler (openadapt-flow) whose healthy replay path
  makes zero model calls; position this repo as the upstream research and cost-optimization (Phase
  2) surface. - Add an honest training status section: base VLMs cannot operate Windows without SFT;
  GRPO standalone backend is the default prototyping path while the verl backend is an integration
  stub raising NotImplementedError; SFT via TRL SFTTrainer is the most exercised path. - Update the
  architecture tree to match the current package layout (grpo, perception, export, baselines,
  segmentation, evals, providers). - Cross-link docs.openadapt.ai and the OpenAdaptAI/openadapt
  launcher; note private hardening corpora and recipes are kept out of this repo. - Remove
  em-dash-style separators throughout.

Claude-Session: https://claude.ai/code/session_01NyCHrzA1psrKMFfroYbzaM

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

### Refactoring

- Make openadapt-ml a leaf; break ml<->evals import cycle
  ([#65](https://github.com/OpenAdaptAI/openadapt-ml/pull/65),
  [`7f62c18`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7f62c1812c7f920befeb5a84277bb893c4864962))

* refactor: make openadapt-ml a leaf; break ml<->evals cycle

Phase 1 of the evals->ml refactor.

Fork A: import the Benchmark* vocabulary (Task/Observation/Action/Agent) from openadapt-types
  instead of openadapt-evals. Removes the one hard, unconditional module-level ml->evals import in
  benchmarks/agent.py.

Fork B: add a thin RolloutEnv Protocol (reset/step/observe/collect_rollout /screen_size) in
  training/grpo/rollout_env.py. The GRPO collector/trainer type against this interface; the concrete
  WAA env/adapters stay in openadapt-evals and implement it structurally.

Also convert the remaining guarded module-level openadapt-evals imports (rollout_collector,
  verl_backend, trainer TaskConfig) to lazy in-function imports so openadapt-ml has zero
  module-level openadapt-evals imports.

Adds openadapt-types as a dependency (>=0.2.0) with a local editable uv source until the
  openadapt-types release is published.

Tests: tests/ (excl. integration) 451 passed, 5 skipped.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01CKrVJJy5jWVCkXAqgUqtqZ

* chore: pin openadapt-types>=0.3.0 (published w/ benchmark), drop local editable source

* style: ruff format verl_backend.py (lazy-import refactor)

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>


## v0.16.1 (2026-06-12)

### Bug Fixes

- Repair 8 latent import/kwarg bugs; add import-integrity and packaging guards
  ([#64](https://github.com/OpenAdaptAI/openadapt-ml/pull/64),
  [`04e9e67`](https://github.com/OpenAdaptAI/openadapt-ml/commit/04e9e678f754c51e8670bd439d1c396ff9ee22cd))

Follow-up to OpenAdaptAI/OpenAdapt#999: green CI shipped a fully broken CLI because no test
  exercised lazy imports, internal call seams, or the built wheel. This adds three guards and fixes
  everything they caught on current main:

New tests (dependency-free, <3s total): - tests/test_import_integrity.py::test_no_phantom_imports —
  AST-walks every `from openadapt_ml.x import y` (including inside function bodies) and verifies y
  exists in x - tests/test_import_integrity.py::test_no_phantom_kwargs — verifies keyword args
  passed to internal functions exist in their signatures - tests/test_packaging.py — builds the
  wheel and asserts bundled configs, core modules, and version match

Latent bugs the new tests caught on main, fixed here: - cloud/local.py: cmd_serve called
  regenerate_local_dashboard with a keep_polling kwarg that no longer exists (dashboard regeneration
  failed on every serve, downgraded to a warning) - scripts/compare.py +
  experiments/demo_prompt/run_experiment.py: capture_to_episode(goal=) — same kwarg bug as #999 bug
  3, two more call sites - ingest/__init__.py: re-exported capture_to_session and
  load_captures_as_sessions, which don't exist; the except ImportError guard meant
  capture_to_episode was silently never exported either - evals/grounding.py: TYPE_CHECKING import
  from missing module openadapt_ml.data.types (Episode lives in openadapt_ml.schema) -
  cloud/azure_inference.py: imported QwenVLAdapter from missing module openadapt_ml.adapters.qwen
  (lives in models.qwen_vl) and constructed it with a non-existent model_name kwarg (needs
  from_pretrained); imported generate_comparison which was refactored away — restored as a thin
  wrapper over generate_comparison_data/_html in compare.py

Also: - release.yml: file/append a GitHub issue when the release workflow fails. Releases failed
  silently Mar-Jun 2026 while PyPI went stale, which is what forced #999's reporter onto git
  installs - pyproject: add build to the dev extra for the packaging tests

Co-authored-by: Claude Fable 5 <noreply@anthropic.com>


## v0.16.0 (2026-06-12)

### Bug Fixes

- Align GRPO prompt format with SFT training format
  ([#61](https://github.com/OpenAdaptAI/openadapt-ml/pull/61),
  [`04e6e9f`](https://github.com/OpenAdaptAI/openadapt-ml/commit/04e6e9f8321334e3aac6a701a8c68e29e53cb0c6))

The GRPO rollout prompt was missing the "Thought:" line and action history that the SFT training
  uses. Models fine-tuned via SFT output "Thought: ...\nAction: CLICK(...)" but the GRPO prompt
  didn't prompt for this format, causing verbose free-form output that couldn't be parsed → reward
  0.0 → zero gradients.

Changes: - Add "Thought:" and "Action:" prompt lines matching SFT format - Add action_history
  parameter for step context - Parser extracts action from "Action: ..." line before regex matching
  - Parser handles JSON format {"action_type": "click", "coordinate": [x,y]} - Debug logging of raw
  VLM output for zero-reward diagnosis

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Include image placeholder in chat template for VLM GRPO
  ([#59](https://github.com/OpenAdaptAI/openadapt-ml/pull/59),
  [`6617e02`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6617e026e564c2b7e04cbeb1bb63e8c8b7a348c7))

Qwen2.5-VL requires <|image_pad|> tokens in the input. These are inserted by apply_chat_template
  only when messages include {"type": "image"} content blocks.

Fixed both agent_fn and _compute_rollout_loss.

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Increase max_new_tokens to 2048 and make configurable via GRPOConfig
  ([#62](https://github.com/OpenAdaptAI/openadapt-ml/pull/62),
  [`fecf461`](https://github.com/OpenAdaptAI/openadapt-ml/commit/fecf4619ce9ebe3763b90d3f6007a0a31689afe3))

* fix: align GRPO prompt format with SFT training format

The GRPO rollout prompt was missing the "Thought:" line and action history that the SFT training
  uses. Models fine-tuned via SFT output "Thought: ...\nAction: CLICK(...)" but the GRPO prompt
  didn't prompt for this format, causing verbose free-form output that couldn't be parsed → reward
  0.0 → zero gradients.

Changes: - Add "Thought:" and "Action:" prompt lines matching SFT format - Add action_history
  parameter for step context - Parser extracts action from "Action: ..." line before regex matching
  - Parser handles JSON format {"action_type": "click", "coordinate": [x,y]} - Debug logging of raw
  VLM output for zero-reward diagnosis

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: increase max_new_tokens to 2048 and make configurable

The default of 100 tokens truncated reasoning models mid-thought, producing unparseable output →
  DONE → reward 0.0 → zero gradients. Caused 4 failed training runs (~20 GPU-hours wasted).

- Add max_new_tokens to GRPOConfig (default 2048) - Use config value instead of hardcoded 100 - Add
  truncation warning when generation hits the limit

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Repair broken internal imports, CPU training, and config packaging (OpenAdapt#999)
  ([#63](https://github.com/OpenAdaptAI/openadapt-ml/pull/63),
  [`9b8f3cd`](https://github.com/OpenAdaptAI/openadapt-ml/commit/9b8f3cd8a2e0f95203813338917bc89bae474001))

* fix: repair broken internal imports, CPU training, and config packaging

Fixes the openadapt-ml side of OpenAdaptAI/OpenAdapt#999:

- Add missing update_current_symlink_to_latest() to training/trainer.py; cmd_serve imported it but
  it never existed, so `openadapt serve` failed with a misleading "openadapt-ml not installed" error
  (bug 2) - scripts/train.py: capture_to_episode(goal=) -> instruction=, matching the actual
  signature (bug 3) - scripts/demo_policy.py: import generate_synthetic_episodes instead of the
  non-existent generate_synthetic_sessions (bug 4) - Ship configs/ inside the wheel (hatch
  force-include) and add resolve_config_path() so installed packages find bundled configs instead of
  failing on repo-relative paths (bug 5) - training/trl_trainer.py: pass use_cpu/bf16/fp16 to
  SFTConfig based on CUDA availability so CPU-only training no longer raises ValueError (bug 7)

Verified: py_compile on all changed files, isolated functional test of

the symlink helper, and wheel build confirming configs are included.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

* fix: respect MPS in CPU fallback, prefer run-like dirs for symlink

- use_cpu now stays False on Apple Silicon (MPS) so the previous accelerated path isn't regressed;
  only true CPU-only setups get use_cpu=True - update_current_symlink_to_latest prefers directories
  containing training_log.json or dashboard.html so a stray top-level "checkpoints" directory from
  the old flat layout can't win

* style: ruff format grpo modules (pre-existing violations blocking CI)

These two files fail 'ruff format --check' on main as well; formatting them here so the test matrix
  can actually run.

---------

Co-authored-by: Claude Fable 5 <noreply@anthropic.com>

### Features

- Add --task-dir support for milestone-based rewards in standalone GRPO trainer
  ([#60](https://github.com/OpenAdaptAI/openadapt-ml/pull/60),
  [`7d095da`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7d095dabbcf39b4146eb3bb5a28d12bde95a6bd7))

* fix: include image placeholder in chat template for VLM GRPO

Qwen2.5-VL requires <|image_pad|> tokens in the input. These are inserted by apply_chat_template
  only when messages include {"type": "image"} content blocks.

Fixed both agent_fn and _compute_rollout_loss.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: add --task-dir support for milestone-based rewards in standalone GRPO trainer

- GRPOConfig: add task_dir field - reward.py: evaluate_milestones_screenshot() for client-side
  reward - trainer.py: load TaskConfigs, auto-populate task_ids, override rewards -
  rollout_collector.py: pass task_configs to env - No WAA evaluate endpoint needed — rewards
  computed via VLM judge

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>


## v0.15.1 (2026-03-21)

### Bug Fixes

- Use keyword args for Qwen VL processor call
  ([#58](https://github.com/OpenAdaptAI/openadapt-ml/pull/58),
  [`d196dce`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d196dce5c3f472fae2e8f830cafd5f2e565f5b13))

* fix: replace AutoModelForVision2Seq with AutoModelForImageTextToText for transformers 5.x

AutoModelForVision2Seq was removed in transformers 5.x (shipped on AWS DL AMI). Use
  AutoModelForImageTextToText as the primary import with a fallback to AutoModelForVision2Seq for
  older transformers versions.

Files updated: - openadapt_ml/training/grpo/trainer.py - openadapt_ml/cloud/modal_cloud.py -
  docs/grpo_trl_rewrite_draft.py (comment only)

Note: openadapt_ml/training/trl_trainer.py already had the correct

try/except pattern and was not modified.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: use keyword args for Qwen VL processor to avoid positional conflict

Qwen2_5_VLProcessor.__call__() expects text= and images= as keyword args. Passing text as positional
  arg conflicts with images kwarg: TypeError: got multiple values for argument 'images'

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Chores

- Trigger release
  ([`7f8833c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7f8833cd7143323d807ed791b4a98a37e266ce4a))

### Documentation

- Add first scored trace (Notepad Hello World, score 0.5)
  ([`ba44eaa`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ba44eaae7e626358b5910ac265beb29c0c414f5f))

6 steps, 91s, GPT-5.4-mini planner+grounder, lightweight mode. VLM judge passed milestone 2 (Hello
  World typed, confidence 1.00). Milestone 1 (process check) timed out during /execute_windows eval.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>


## v0.15.0 (2026-03-19)

### Bug Fixes

- Make heavy ML dependencies optional for lightweight installs
  ([#57](https://github.com/OpenAdaptAI/openadapt-ml/pull/57),
  [`aa954ba`](https://github.com/OpenAdaptAI/openadapt-ml/commit/aa954ba9b49024b834d4c6bbc33e16acf2ad51e7))

* fix: make heavy ML dependencies optional for lightweight installs

Move torch, torchvision, bitsandbytes, peft, and transformers from required dependencies to
  [project.optional-dependencies.training]. Wrap all top-level imports of these packages in
  try/except ImportError so the package can be imported without them installed.

This unblocks lightweight consumers (e.g. Wright worker installing openadapt-evals) that don't need
  local model training/inference. Users who need training can install with: pip install
  openadapt-ml[training]

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* style: fix ruff formatting in qwen_vl.py

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Use ResetConfig for RLEnvironment.reset() in validation script
  ([#56](https://github.com/OpenAdaptAI/openadapt-ml/pull/56),
  [`942a2f3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/942a2f3b8c3654e4b4d4bf59977f4015e856e8e5))

The validation script called env.reset(task_id=...) but the actual API is
  env.reset(config=ResetConfig(task_id=...)). This caused Phase 2 to fail with TypeError.

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Chores

- Trigger release
  ([`c9da079`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c9da0791ab05051582096041bda9973d30f9f657))

### Documentation

- Add LoRA-per-task design document ([#54](https://github.com/OpenAdaptAI/openadapt-ml/pull/54),
  [`d6d63b6`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d6d63b63a34446bfc87ac0a7b934079a185f6874))

Literature-backed design for task-specific LoRA adapters with runtime routing. Covers architecture,
  training pipeline, data collection (including correction flywheel as training data source), update
  economics, and validation plan. Positioned as one experiment track within the broader OpenAdapt
  experimentation framework.

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Features

- Add GRPO validation infrastructure and LoRA checkpoint support
  ([#55](https://github.com/OpenAdaptAI/openadapt-ml/pull/55),
  [`1b8ae78`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1b8ae78dd1ba0a996b0dd6e9f21727cd049e5864))

* feat: add evaluate_url, lora_checkpoint, validation script, and CLI for GRPO training

- Add evaluate_url field to GRPOConfig for separate evaluate endpoint - Add lora_checkpoint field to
  resume GRPO from existing SFT LoRA adapter - Pass evaluate_url through rollout collector to
  WAALiveConfig - Load existing LoRA via PeftModel.from_pretrained() when lora_checkpoint set -
  Update verl_backend.py error message with actionable instructions - Add 5-phase validation script
  (connectivity → rollout → inference → train → multi-step) - Add CLI entry point
  (scripts/run_grpo.py) for running GRPO without writing Python

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* style: fix ruff formatting in config and validation script

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>


## v0.14.1 (2026-03-04)

### Bug Fixes

- Lower PyTorch minimum to 2.8.0 for vLLM compatibility
  ([#53](https://github.com/OpenAdaptAI/openadapt-ml/pull/53),
  [`60bd60c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/60bd60c66ef4a81699d6e85a42e8af0860ad37b6))

vLLM 0.11.0 pins torch==2.8.0. The GPU E2E validation (openadapt-evals PR #87) confirmed the full ML
  stack works with PyTorch 2.8.0+cu128. The previous >=2.9.1 constraint prevented installing
  openadapt-ml alongside vLLM in the same environment.

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.14.0 (2026-03-04)

### Features

- Add dual training backend support (standalone + verl-agent)
  ([#51](https://github.com/OpenAdaptAI/openadapt-ml/pull/51),
  [`9cccc49`](https://github.com/OpenAdaptAI/openadapt-ml/commit/9cccc49af780d54a0c155320aea1f5d14bcd19a4))

* feat: add dual training backend support (standalone + verl-agent)

Add `backend` field to GRPOConfig ("standalone" or "verl") to support switching between training
  backends:

- standalone: existing trainer.py (single-GPU, episode-level rewards) - verl: verl-agent/VAGEN
  integration (multi-GPU, GiGPO per-step credit)

New verl_backend.py provides build_vagen_config() to map GRPOConfig to VAGEN-compatible config, and
  train_with_verl() as the integration point (placeholder until full end-to-end is wired up).

No existing function signatures or behavior modified.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

* style: format verl_backend.py with ruff

---------

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.13.0 (2026-03-03)

### Features

- Add docs sync trigger ([#52](https://github.com/OpenAdaptAI/openadapt-ml/pull/52),
  [`8cb5e8c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/8cb5e8c8f24f21445b5377fe031e0e521f03fc46))


## v0.12.0 (2026-03-03)

### Features

- Add GRPO training module with minimal TRL bridge
  ([#34](https://github.com/OpenAdaptAI/openadapt-ml/pull/34),
  [`40d2f73`](https://github.com/OpenAdaptAI/openadapt-ml/commit/40d2f732119c461a69814ba4118244072ae8b757))

* docs: add experimental roadmap and evidence context to vision

- Add 2x2 experimental matrix (retrieval × fine-tuning) to Core Thesis - Add evidence context to
  benchmark table: note it's an internal synthetic benchmark (~3 UI elements) that validates the
  pipeline, not real-world performance. Link to openadapt-evals for ongoing WAA/OSWorld evaluation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

* fix: use 46.7% consistently in 2x2 matrix

Was showing 33-47% range which conflated preliminary (n=3) and full (n=45) results. The validated
  number is 46.7%.

* feat: add GRPO training module for online RL

Add openadapt_ml/training/grpo/ package with: - GRPOConfig for training hyperparameters -
  GRPORolloutCollector connecting to openadapt-evals RLEnvironment - GRPOTrainer implementing custom
  GRPO loop for multimodal VLMs - Binary reward function and group-relative advantage computation -
  Chain-of-thought warm-up pipeline for SFT pre-training - 20 unit tests passing without GPU

* fix: address review findings in GRPO module

- Replace copy.deepcopy(model) with LoRA state dict snapshot (prevents OOM) - Mark
  _compute_rollout_loss as scaffold with dummy forward pass for grad flow - Fix collect_rollout call
  to match RLEnvironment API (task_id in signature) - Add model.eval()/model.train() toggling around
  rollout/training phases - Remove unused gradient_accumulation_steps config field - Use actual
  screen_size from RLEnvironment instead of hardcoded 1920x1200 - Clamp CLICK coordinates to [0.0,
  1.0] to prevent invalid pixel values - Validate task_ids non-empty at start of train() - Export
  CoT warmup functions from package __init__ - Add BenchmarkAction fallback when openadapt-evals not
  installed - Add 9 new tests: action parser (8) + empty task_ids validation (1) - All 29 tests
  passing

* feat: implement GRPO loss computation and fix cot_warmup dependency

Implement the core _compute_rollout_loss method that was previously a NotImplementedError scaffold.
  The implementation:

- Reconstructs VLM prompts from rollout observations - Formats actions back to DSL text via new
  _format_action_as_text helper - Computes log-probabilities of action tokens under current policy -
  Computes reference policy log-probs via PEFT disable_adapter() with fallback to manual LoRA weight
  swapping - Returns GRPO loss: -advantage * log_prob + kl_coef * KL penalty

Also adds get_api_adapter() factory function to api_adapter.py, fixing the broken import in
  cot_warmup.py's generate_cot_annotations().

Additional review fixes from prior session: - Initialize _is_unsloth and _ref_lora_state in __init__
  - Remove dead else branch for task_id selection - Fix total_loss device placement - LoRA-only
  fallback save in checkpoint - TYPE regex accepts single quotes - Coordinate clamping in
  _parse_vlm_output_to_action

40 tests passing (10 new: 8 format_action + 1 roundtrip + 1 api_adapter).

* refactor: deduplicate GRPO prompts via shared _build_agent_messages

Extract prompt construction into _build_agent_messages() which imports SYSTEM_PROMPT from
  next_action.py (the SFT training prompt). This ensures the GRPO agent uses the same prompt
  distribution the model was warm-started on, and guarantees _make_agent_fn and
  _compute_rollout_loss use identical prompts (critical for correct log-prob computation).

* fix(grpo): address critical review findings in GRPO loss computation

- C-01: Store raw model output on action._grpo_raw_text for accurate loss - C-02: Separate
  tokenization of prompt/action with concatenation to fix BPE boundary alignment - I-01: Prefer LoRA
  weight swapping over disable_adapter() for reference policy (captures initial LoRA state after SFT
  warm-start) - I-03: Per-step gradient accumulation via immediate backward() to prevent OOM from
  building computation graph over all rollout steps - I-04: Fix unescape order in TYPE parser
  (backslash before quotes) - M-03: Pass model_name through get_api_adapter to ApiVLMAdapter - M-07:
  Case-insensitive CLICK/TYPE regex in _parse_vlm_output_to_action - L-01: Extract
  DEFAULT_SCREEN_SIZE constant, replace all hardcoded values

* fix(grpo): fix instruction propagation, screen size, weight swap safety

- CR-01: Task instruction was never populated during GRPO rollouts.
  WAALiveAdapter._get_observation() does not populate raw_observation, so the agent prompt said
  "Goal: " with nothing after it. Fix: store instruction on Rollout dataclass (populated from
  env._current_task in collector), use it in both agent_fn and _compute_rollout_loss. - IM-01:
  Change DEFAULT_SCREEN_SIZE from 1920x1200 to 1920x1080 for consistency with baselines module and
  standard VM configurations. Add screen_size field to GRPOConfig so it is configurable. - IM-02:
  Add try/finally around LoRA weight swap in _compute_ref_log_probs. Without this, an exception
  during the reference forward pass permanently corrupts the model state.

* fix(grpo): remove unused torch import in _setup_model

The import torch at line 121 was flagged by ruff (F401) as unused. The surrounding code only calls
  .detach().clone() on tensor objects, which does not require the torch module directly.

* style(grpo): apply ruff formatting to GRPO module files

Run ruff format on cot_warmup.py, rollout_collector.py, and trainer.py to satisfy the CI ruff
  formatter check.

* refactor(grpo): replace custom trainer with minimal TRL bridge

Replace 809-line custom GRPO trainer with ~280 lines that: - Use standard HuggingFace
  AutoModelForVision2Seq + AutoProcessor + PEFT LoraConfig instead of Unsloth monkey-patching -
  Implement standalone GRPO loss in ~15 lines of PyTorch (clipped surrogate) instead of custom
  policy gradient + KL penalty - Use beta=0.0 (no KL penalty, no reference model) per DAPO/Open-
  Reasoner-Zero literature, eliminating weight-swap complexity - Keep per-step backward to avoid OOM
  on long trajectories - Use standard model.save_pretrained() for checkpointing - Document WHY
  standalone GRPO math vs TRL GRPOTrainer (VLM multi-turn image pixel_values not stored in token
  IDs) and WHEN to switch

Preserves all public API: GRPOTrainer, _parse_vlm_output_to_action, _format_action_as_text,
  _build_agent_messages, DEFAULT_SCREEN_SIZE. All 50 tests pass (44 existing + 6 new for grpo_loss
  and trainer internals).

* feat(grpo): add E2E tests with artifact generation and architecture docs

- tests/test_grpo_e2e.py: 5 E2E tests (training loop, rollout collection, loss convergence, weight
  diff, mathematical properties) using tiny mock VLM. Produces 65+ artifacts (JSON traces, PNGs,
  checkpoints, summaries). - scripts/grpo_e2e_report.py: CLI report generator for test artifacts
  (text + optional HTML output). - docs/grpo_e2e_test_design.md: design rationale for E2E test
  approach - docs/grpo_architecture_analysis.md: analysis of custom vs TRL-based GRPO -
  docs/grpo_trl_rewrite_draft.py: TRL v0.29.0 integration research -
  docs/strategic_analysis_evals_ml_synergy.md: business/economics analysis

* fix(grpo): address self-review findings (BUG-01, CLEAN-01 through -05)

- Rename grpo_loss to policy_gradient_loss with honest docstring: single-epoch on-policy means
  ratio=1.0, clipping never fires, this is REINFORCE with group-relative advantages. Keep grpo_loss
  as backwards-compatible alias. - Add public aliases: parse_vlm_output_to_action,
  format_action_as_text (drop underscore prefix for public API) - Export policy_gradient_loss and
  public functions from __init__.py - Remove unused config fields: kl_coef (was 0.01 but never used
  with beta=0), max_seq_length (never referenced) - Fix model_name default:
  Qwen/Qwen2.5-VL-7B-Instruct (not unsloth variant) - Fix trivial test assertion: grad_norm > 0 (was
  >= 0, always true) - Update loss tests to verify gradient direction, not just loss sign - Add
  test_public_api_exports for new public names

56 tests pass (51 unit + 5 E2E).

---------

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.2 (2026-02-25)

### Bug Fixes

- **docs**: Require conventional commit format for PR titles
  ([#32](https://github.com/OpenAdaptAI/openadapt-ml/pull/32),
  [`729c289`](https://github.com/OpenAdaptAI/openadapt-ml/commit/729c289e3f26b58be0d2e2d7a2efe7de147bed62))

PR titles become squash merge commit messages. Without the fix:/feat: prefix,
  python-semantic-release skips the release. Document this requirement prominently in CLAUDE.md.

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation

- Enforce branch protection rules ([#30](https://github.com/OpenAdaptAI/openadapt-ml/pull/30),
  [`3f065a1`](https://github.com/OpenAdaptAI/openadapt-ml/commit/3f065a129441d0f67b731539860ba4b922996153))

* docs: add mandatory branch/PR rule to CLAUDE.md

Adds explicit instruction that all changes must go through feature branches and pull requests.
  enforce_admins has been enabled on GitHub to prevent admin bypass of branch protection.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

* fix(modal): remove unused os import

Fixes ruff F401 lint error on modal_cloud.py.

---------

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.1 (2026-02-24)

### Bug Fixes

- **modal**: Fix inference container image and multi-modal message handling
  ([`6aef712`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6aef712b443aec9c2b52b82c16cafed065c51fd0))

- Pin transformers==4.57.3 (matches local, has Qwen3-VL support) - Add torchvision dependency
  (required by AutoVideoProcessor) - Add fallback: AutoModelForVision2Seq ->
  Qwen2_5_VLForConditionalGeneration - Add fallback: AutoProcessor -> Qwen2_5_VLProcessor -
  Reconstruct multi-modal messages with {"type": "image"} placeholders for proper vision token
  generation in apply_chat_template - Rename container_idle_timeout -> scaledown_window (Modal API
  update)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.0 (2026-02-24)

### Features

- **modal**: Add inference serving with call_inference API
  ([`f45c524`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f45c524f92c8b7c28651e936aae983d8316e86ee))

- Add _build_inference_app() for Modal GPU inference with PEFT adapter - Add
  upload_adapter_to_volume() for uploading adapters to Modal volume - Add call_inference() as the
  primary API for remote inference - Add 'serve' CLI command for interactive model serving -
  Container caches model in memory across calls (container_idle_timeout=600) - Support --no-adapter
  for zero-shot base model serving

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.10.1 (2026-02-24)

### Bug Fixes

- **modal**: Apply fixes from first successful Modal training run
  ([`3b38b77`](https://github.com/OpenAdaptAI/openadapt-ml/commit/3b38b776bb8ace98840b8bba45e7b773507185b9))

- Add `serialized=True` to @app.function for non-global-scope support - Auto-create volume before
  upload, add `--force` for overwrites - Fix variable scoping (`vol = training_volume`) inside
  remote function - Add `openadapt-ml[training]` to container image dependencies - Use `--jsonl`
  flag in train subprocess for correct data path - Add `modal` to project dependencies - Update test
  to verify create+put two-call pattern

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.10.0 (2026-02-24)

### Features

- **cloud**: Add Vast.ai and Modal GPU providers
  ([`dd0aad2`](https://github.com/OpenAdaptAI/openadapt-ml/commit/dd0aad23b69d66d1b8759513ece1fc97bf73a753))

Vast.ai (~$0.17/hr A10): SSH+rsync marketplace model with full CLI (list, launch, terminate, train)
  matching lambda_labs.py pattern. Includes GPU search, --gpu-wait retry, auto-convert --demo-dir
  flow.

Modal ($30/mo free, $1.10/hr A10G): Python-native cloud with zero-ops training via decorated
  functions and Modal Volumes for data transfer. CLI: train, status, download, list-volumes.

Both support the same --demo-dir end-to-end pipeline as Lambda Labs.

53 new tests (34 Vast.ai + 19 Modal), all passing.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.9.0 (2026-02-24)

### Features

- **train**: Add end-to-end pipeline automation with --demo-dir flag
  ([`d883e39`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d883e39d26d0aa45c780176a52f9e9d9329fa848))

Add prepare_bundle() and generate_screenshot_mapping() to convert_demos.py for single-call demo
  conversion. Extend both train.py and lambda_labs.py train commands with --demo-dir,
  --captures-dir, --mapping flags so the full pipeline (mapping → conversion → bundle → upload →
  train) runs as one command. Add --gpu-wait for Lambda GPU availability retry loop.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.8.0 (2026-02-24)

### Features

- Sft training pipeline with demo conversion, Lambda Labs integration, and data persistence
  ([#29](https://github.com/OpenAdaptAI/openadapt-ml/pull/29),
  [`e56c9e4`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e56c9e4a581b46c697188b3b2674b850d8225a4f))

* feat(training): add demo conversion pipeline for ms-swift SFT format

Convert annotated demo JSON files to JSONL training data compatible with ms-swift for Qwen3-VL
  fine-tuning. Handles coordinate conversion from [0,1] to [0,1000] range, generates <think> blocks
  from observation and intent fields, and accumulates action history across steps.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

* feat(training): add screenshot linking via mapping file

Support --mapping flag for pre-computed screenshot mapping JSON that maps task_id -> {step_index ->
  screenshot_path}. This correctly handles the coalesced step-to-raw-capture mapping (where step
  indices skip due to merging). Also adds --captures-dir with DB-based fallback and
  DOUBLE_CLICK/RIGHT_CLICK parsing.

* fix(training): align SFT format with inference prompt and add validation

- Add system role message to training conversations matching qwen3vl_agent.SYSTEM_PROMPT - Add
  "Output exactly one action" / thinking instruction to user message matching _build_prompt() output
  - Add coordinate range validation warning for values outside [0, 1] - Add input schema validation
  for required demo/step fields - Remove broken _resolve_screenshots_from_db() and
  _resolve_screenshots_direct() fallbacks that produced silently wrong mappings for coalesced demos
  - Remove --screenshot-dir CLI arg (unreliable for coalesced demos) - Keep --mapping (recommended)
  and capture API as screenshot resolution

* feat(training): add JSONL training pipeline with bundle support

Align convert_demos output with internal SFT format (images + messages), add train_from_jsonl()
  loader, --jsonl flag to train.py, --bundle flag to convert_demos and Lambda Labs train command.
  Enables training on annotated demo data without Episode objects.

* fix(training): add TRL callback, 4-bit quantization, early stopping

- Add OpenAdaptCallback for training_log.json output + early stop on loss - Fix _load_standard_model
  to use BitsAndBytesConfig for 4-bit quantization - Use AutoModelForImageTextToText (supports
  Qwen3-VL) instead of Qwen2VL class - Switch demo config to 2B model for fast iteration on A10 -
  Hide Azure ML Jobs panel when cloud_provider is not azure - Fix Lambda setup: remove uv.sources
  before uv sync on remote

* fix(cloud): use git archive for code sync, fix callback MRO

Replace rsync with `git archive HEAD | ssh tar` in sync_local_code() to send only committed tracked
  files (~10MB vs ~1.8GB with binary artifacts).

Fix callback class MRO: _OpenAdaptCallback must precede TrainerCallback so our on_log/on_train_begin
  override the no-op base implementations.

* refactor(training): consolidate duplicated SFTTrainer setup

Extract _run_sft_training() shared by train_with_trl() and train_from_jsonl(), eliminating ~80 lines
  of duplicated SFTConfig, SFTTrainer instantiation, and training loop code.

* feat(training): add plateau-based early stopping

Add early_stop_min_delta and early_stop_plateau_patience to stop training when loss stops improving
  by at least min_delta for N consecutive steps. Works alongside the existing absolute threshold.

* docs: add GPU hosting options and training pipeline gap analysis

GPU hosting options covers 24 platforms ranked by value for open-source projects needing
  free/credited GPU compute for VLM fine-tuning.

* docs: add Qwen3-VL-2B training results analysis

Detailed analysis of first fine-tuning run: 27.24 → 9.77 loss (64% reduction) over 50 steps on 20
  annotated WAA demo samples. Includes per-epoch breakdown, compute efficiency metrics, and
  recommendations for future training runs.

* fix: remove absolute paths from repo, fix LoRA task_type, add tests

- Remove screenshot_mapping.json (has absolute local paths), add to .gitignore, add
  screenshot_mapping.example.json instead - Fix LoRA task_type: always use CAUSAL_LM (Qwen-VL is
  decoder-only, not encoder-decoder like T5/BART that needs SEQ_2_SEQ_LM) - Add 57 tests for
  convert_demos (action parsing, coordinate conversion, step conversion, validation, bundle
  creation) and training callback (log writing, threshold early stopping, plateau detection)

* fix: address self-review issues (config wiring, security, tests)

- Wire lr_scheduler_type, weight_decay, max_grad_norm, target_modules from YAML config through to
  SFTConfig (were silently ignored) - Fix command injection in lambda_labs.py via shlex.quote() -
  Fix callback writing loss=0 on non-loss log events (track _last_loss) - Fix WAIT() mapping to
  wait() instead of finished() in convert_demos - Fix CI: add --no-sources to uv sync for uv.sources
  compatibility - Add test for non-loss log event callback behavior - Update SYSTEM_PROMPT comment
  (remove stale cross-reference)

* ci: fix uv.sources with UV_NO_SOURCES env var, skip integration tests

UV_NO_SOURCES=1 covers uv sync, uv run ruff, and uv run pytest. Integration tests require
  openadapt_evals which is not a dependency.

* style: format annotate.py with ruff

* feat: auto-generate training plots, persist data with checkpoint

- Add plot_training.py: generates loss curve, LR schedule, and combined plots from training_log.json
  using matplotlib - Copy training_log.json + plots into checkpoint directory after training so
  artifacts are self-contained and never lost - Add periodic rsync of training_log.json during
  Lambda training (every 5 min) so data survives instance interruption - Replace ASCII loss curve in
  training results doc with real PNG plots - Add reconstructed training_log.json from Qwen3-VL-2B
  demo run

* test: add tests for plot generation and checkpoint co-location

11 tests covering: - Loss plot generation (with/without LR data) - Output directory creation and
  defaults - Empty data handling - Epoch boundary rendering - Real training log validation -
  Checkpoint co-location of log + plots

---------

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.1 (2026-02-18)

### Bug Fixes

- **ci**: Use v9 branch config for python-semantic-release
  ([#28](https://github.com/OpenAdaptAI/openadapt-ml/pull/28),
  [`bfe7092`](https://github.com/OpenAdaptAI/openadapt-ml/commit/bfe7092f3405a66cd058fd1523f23658dabbc076))

Replace `branch = "main"` (v7/v8 key) with `[tool.semantic_release.branches.main]` table (v9 key).
  The old key is silently ignored by v9, causing releases to never trigger on the main branch.

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.0 (2026-02-18)

### Features

- **demo-prompt**: Add VLM-annotated traces for 3 recorded demos
  ([`6d2e0a9`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6d2e0a9b27e3f3d8537aa0e277166f8377cd9b36))

Ran annotation pipeline (GPT-4o) on all 3 recorded captures: - 37e10fc4 (notifications): 5 steps —
  turn off system notifications - 0c9dda13 (archive): 9 steps — create Archive folder, move .docx
  files - 366de66e (notepad): 6 steps — open Notepad, create draft.txt

These grounded traces replace fabricated hand-written demos for demo-conditioned evaluation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.6.0 (2026-02-17)

### Bug Fixes

- **deps**: Bump openadapt-capture to >=0.3.0, add uv.sources
  ([`c120b0e`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c120b0e0ea4d3999ef42fea6749130a52f57aed7))

The new recording format uses recording.db (not capture.db). Local editable source ensures lockfile
  resolves correctly.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Code Style

- Fix ruff formatting for migrated import references
  ([#27](https://github.com/OpenAdaptAI/openadapt-ml/pull/27),
  [`36fd74c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/36fd74c115ac042224e9766aba942d8802bb4310))

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation

- Add demo GIFs back to README
  ([`f725872`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f7258720cd8d382ccc7916cb08e1922821255166))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Rewrite CLAUDE.md — remove migration guide, match pure ML scope
  ([`392bd0e`](https://github.com/OpenAdaptAI/openadapt-ml/commit/392bd0e4665a1851927f51ce5c24a229360290f4))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Rewrite README for professional open-source style
  ([`e26032b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e26032b0457946d1ad0465434f6d4f4161e366ec))

Replace 1100-line README containing stale VM/pool references with clean 220-line README reflecting
  what the package actually contains post-migration. Use test.yml badge instead of release.yml for
  accurate build status.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Update README references to migrated CLI
  ([#26](https://github.com/OpenAdaptAI/openadapt-ml/pull/26),
  [`b56a245`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b56a2452342691f4a42ca167e5ca6b39d41d818e))

* feat: remove evaluation infrastructure (moved to openadapt-evals)

All evaluation infrastructure (~13,000 lines) has been migrated to openadapt-evals (PR #29). This PR
  removes the now-redundant code from openadapt-ml, making it a pure ML package.

Deleted files: - benchmarks/cli.py (8,503 lines - VM/pool CLI) - benchmarks/azure_vm.py
  (AzureVMManager) - benchmarks/pool.py (PoolManager) - benchmarks/vm_monitor.py,
  azure_ops_tracker.py, resource_tracker.py - benchmarks/azure.py, viewer.py, pool_viewer.py,
  trace_export.py - benchmarks/waa_deploy/ (Docker agent deployment) -
  tests/test_quota_auto_detection.py, test_demo_persistence.py - tests/benchmarks/test_api_agent.py,
  test_waa.py

Updated: - benchmarks/__init__.py: Only exports ML agents (PolicyAgent, etc.) - pyproject.toml:
  Removed azure-ai-ml, azureml-core, azure-mgmt-* - CLAUDE.md: Removed CLI/VM/pool docs, added
  migration guide

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

* fix: update stale references to migrated benchmark modules

Update all remaining references to deleted benchmark modules across source code, scripts, and tests:

- cloud/local.py: azure_ops_tracker, session_tracker, CLI subprocess calls - scripts/: p0/p1
  validation scripts, screenshot generators, quota checker - training/benchmark_viewer.py: HTML
  template CLI references - experiments/waa_demo/runner.py: docstring and print references -
  deprecated/waa_deploy/__init__.py: import path

All now point to openadapt_evals equivalents.

* docs: update README references to migrated CLI

All VM/pool CLI commands moved from openadapt_ml.benchmarks.cli to openadapt-evals (oa-vm). Update
  all README references.

---------

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **demo-prompt**: Add VLM annotation pipeline for recorded demos
  ([`d1c1bc7`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d1c1bc7832431541e31749f3545f04bb0c0fbd1e))

Converts raw recordings (coordinates + screenshots) into structured text traces matching the
  hand-written demo format. Uses VLM to annotate each step with screen observation, intent, semantic
  action, and result.

Pipeline: capture → episode → coalesce → annotate (VLM) → validate → format

Key components: - Step coalescing (500 raw actions → 5-30 meaningful steps) - Click marker rendering
  on screenshots for VLM - Before+after frame pairs for grounded result descriptions - Sequential
  context (previous step annotation feeds into next) - Compact formatting matching hand-written demo
  shape - Runner integration with annotated > hand-written priority

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.5.0 (2026-02-13)

### Features

- Remove evaluation infrastructure (moved to openadapt-evals)
  ([#25](https://github.com/OpenAdaptAI/openadapt-ml/pull/25),
  [`ca50a77`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ca50a77b7e397ca5e5d8e72a4f15dfbad21b61d4))

* feat: remove evaluation infrastructure (moved to openadapt-evals)

All evaluation infrastructure (~13,000 lines) has been migrated to openadapt-evals (PR #29). This PR
  removes the now-redundant code from openadapt-ml, making it a pure ML package.

Deleted files: - benchmarks/cli.py (8,503 lines - VM/pool CLI) - benchmarks/azure_vm.py
  (AzureVMManager) - benchmarks/pool.py (PoolManager) - benchmarks/vm_monitor.py,
  azure_ops_tracker.py, resource_tracker.py - benchmarks/azure.py, viewer.py, pool_viewer.py,
  trace_export.py - benchmarks/waa_deploy/ (Docker agent deployment) -
  tests/test_quota_auto_detection.py, test_demo_persistence.py - tests/benchmarks/test_api_agent.py,
  test_waa.py

Updated: - benchmarks/__init__.py: Only exports ML agents (PolicyAgent, etc.) - pyproject.toml:
  Removed azure-ai-ml, azureml-core, azure-mgmt-* - CLAUDE.md: Removed CLI/VM/pool docs, added
  migration guide

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

* fix: update stale references to migrated benchmark modules

Update all remaining references to deleted benchmark modules across source code, scripts, and tests:

- cloud/local.py: azure_ops_tracker, session_tracker, CLI subprocess calls - scripts/: p0/p1
  validation scripts, screenshot generators, quota checker - training/benchmark_viewer.py: HTML
  template CLI references - experiments/waa_demo/runner.py: docstring and print references -
  deprecated/waa_deploy/__init__.py: import path

All now point to openadapt_evals equivalents.

---------

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.4.2 (2026-02-13)

### Bug Fixes

- **cli**: Write correct dict format for --task in run command
  ([#24](https://github.com/OpenAdaptAI/openadapt-ml/pull/24),
  [`97f55a7`](https://github.com/OpenAdaptAI/openadapt-ml/commit/97f55a7fdf2d4bb79f96c8b917934380c6f6130d))

WAA's run.py expects test config as {domain: [task_ids...]} dict, but --task wrote a bare JSON array
  [task_id] causing TypeError when run.py indexes by domain string key.

Now looks up the task's domain from test_all.json inside the container and writes the correct
  {domain: [task_id]} format.

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.4.1 (2026-02-13)

### Bug Fixes

- Remove broken WAA submodule, embed startup script
  ([#22](https://github.com/OpenAdaptAI/openadapt-ml/pull/22),
  [`d5e0548`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d5e05488a08421d4e1170e64cf11244b7d42be77))

The vendor/WindowsAgentArena submodule pointed to unpushed local commits (a956c5b) that don't exist
  upstream, breaking git-based pip installs.

- Remove submodule entirely (not a runtime dependency) - Embed the 9-line
  compute-instance-startup.sh as a constant in cli.py - Update path references in Azure ML commands
  to not depend on vendor/

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Use ADMIN_TOKEN for release automation on protected branches
  ([#23](https://github.com/OpenAdaptAI/openadapt-ml/pull/23),
  [`93b079d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/93b079d4fed68c6547d2ea27e11d8d8725a5fb3e))

GITHUB_TOKEN cannot push version-bump commits to branches with PR protection. Use org-level
  ADMIN_TOKEN instead, with skip-check to prevent infinite loops on release commits.

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **benchmarks**: Extract library API from CLI
  ([#21](https://github.com/OpenAdaptAI/openadapt-ml/pull/21),
  [`f8cac6c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f8cac6c7c61a251f76df42a4d66f762b517ad897))

* refactor(benchmarks): extract library API from CLI for programmatic usage

Extract core VM management and pool lifecycle logic from cli.py into importable modules
  (azure_vm.py, pool.py) with clean Python APIs.

- Add AzureVMManager class with Azure SDK primary path + az CLI fallback - Add PoolManager class for
  pool create/wait/run/cleanup lifecycle - Add configurable resource_group via Settings, env var, or
  --resource-group flag - Support DefaultAzureCredential for enterprise SSO/service principals - CLI
  handlers become thin wrappers delegating to library classes - Add agent_factory parameter stub on
  PoolManager.run() for pluggable agents

All 327 tests pass, CLI surface unchanged.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

* style: fix pre-existing ruff lint errors in pool_viewer and resource_tracker

Remove unused import `json` and unused variable `worker_re` in pool_viewer.py, and unused import
  `Optional` in resource_tracker.py.

* style: run ruff formatter on benchmarks modules

* fix(azure_vm): add SDK path for set_auto_shutdown via generic resource API

Auto-shutdown schedules are Microsoft.DevTestLab/schedules resources. Use azure-mgmt-resource
  (already a dependency) to create them via the generic resource client, with az CLI fallback if SDK
  fails.

---------

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.4.0 (2026-02-06)

### Code Style

- **cli**: Run ruff formatter
  ([`714268c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/714268c1386129dcc2a63256a9e5c88a8120a7d1))

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- **benchmarks**: Add pool viewer and auto-shutdown
  ([#20](https://github.com/OpenAdaptAI/openadapt-ml/pull/20),
  [`36c7206`](https://github.com/OpenAdaptAI/openadapt-ml/commit/36c720679a0e0c8265cd1409d7258a008a72fe39))

* feat(benchmarks): add HTML viewer for WAA pool benchmark results

Add pool_viewer.py module and CLI command for generating interactive HTML viewers from WAA parallel
  benchmark runs.

Features: - Parse waa-pool-*.log files to extract task results - Summary stats (total tasks, success
  rate, avg time per task) - Per-worker breakdown showing tasks per worker - Task list with
  pass/fail status and step counts - Domain breakdown with per-domain success rates - Interactive
  filters for domain and status

Usage: uv run python -m openadapt_ml.benchmarks.cli view-pool uv run python -m
  openadapt_ml.benchmarks.cli view-pool --run-name pool_run_20260204 uv run python -m
  openadapt_ml.benchmarks.cli view-pool --no-open

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* docs(claude): document VM auto-shutdown and orphan prevention

Add documentation for the auto-shutdown feature: - Explain auto-shutdown policy (default 4 hours) -
  Document --auto-shutdown-hours flag for pool-create and create - Document -y flag for pool-cleanup
  (skip confirmation) - Document test VM cleanup via try/finally

* docs(readme): update CLI commands to use pool-* workflow

Update documentation to reflect the current working CLI: - Replace outdated `vm monitor` with
  `pool-status/pool-vnc/pool-logs` - Update single VM workflow to use `pool-create --workers 1` -
  Add analyze_pool_logs.py script for parsing benchmark results

* fix(cli): prevent orphaned test VMs during pool-create

Remove --no-wait flag from test VM creation so the VM fully exists before we attempt to delete it.
  Previously, the test VM would still be provisioning when delete was called, causing delete to fail
  silently and leave orphaned VMs consuming quota.

* fix(cli): use waa-auto image in pool-wait, wait for apt lock

Critical fixes for end-to-end pool workflow:

1. Use waa-auto:latest in pool-wait (not windowsarena/winarena) - pool-create builds waa-auto with
  modern dockurr/windows v5.14 - pool-wait was incorrectly using vanilla windowsarena/winarena
  (v0.00) - v0.00 doesn't support VERSION=11e auto-download - This caused "ISO file not found"
  errors

2. Wait for apt lock before Docker install - Fresh Azure VMs run unattended-upgrades - apt-get
  install failed with "unable to locate package" - Added wait loop for /var/lib/apt/lists/lock

* fix(pool): match working waa command parameters exactly

- Use vanilla windowsarena/winarena:latest with --entrypoint /bin/bash - Add --prepare-image false
  --start-client false flags (skips ISO download) - Use 172.30.0.2 for probe and emulator_ip
  (matching working waa command)

The pool-wait command was broken because it used waa-auto:latest without the proper entrypoint and
  flags. The working 'waa' command (line 5404-5454) uses these exact parameters successfully.

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>


## v0.3.1 (2026-02-05)

### Bug Fixes

- **cli**: Resolve ruff linter errors
  ([`6084161`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6084161b47a628a94ada31e35fa4589680ac9de2))

- Replace bare `except:` with `except Exception:` - Remove unused f-string prefixes - Remove unused
  variable assignments - Remove unused imports

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Documentation

- **readme**: Add parallel WAA evaluation, fix build badge
  ([#19](https://github.com/OpenAdaptAI/openadapt-ml/pull/19),
  [`3f74b55`](https://github.com/OpenAdaptAI/openadapt-ml/commit/3f74b552f6306a70c21f083a50f02ca053f8d897))

* docs(readme): add parallel WAA evaluation section, fix build badge

- Fix broken build badge (publish.yml → release.yml) - Add prominent "Parallel WAA Benchmark
  Evaluation" section near top - Add detailed "WAA Benchmark Workflow" section (#14) with: - Single
  VM and parallel pool workflows - VNC access instructions - Architecture diagram - Cost estimates -
  Update section numbering (Limitations → 15, Roadmap → 16)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* fix(readme): address self-review feedback

- Fix anchor placement (move before heading for proper navigation) - Correct pool-delete →
  pool-cleanup (actual command name) - Add pool-status example for getting worker IPs - Add "prices
  vary by region" caveat

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>


## v0.3.0 (2026-02-05)

### Bug Fixes

- **cli**: Improve pool-create reliability and error handling
  ([`6ead5ff`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6ead5ffc73aa8ed9648526bfaa48c8d5ec83198f))

- Properly clean up test VM and associated resources during quota check - Use sudo for docker pull
  (usermod not effective in same session) - Add pool-cleanup command for orphaned resources - Show
  full error messages in pool creation failures

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **pool**: Use WAA native task distribution with --worker_id/--num_workers
  ([`69082e4`](https://github.com/OpenAdaptAI/openadapt-ml/commit/69082e4dfe48f1e2d20e5917864ca3ad736d4c79))

- Fixed task distribution: WAA ignores --start_idx/--num_tasks, use native --worker_id and
  --num_workers parameters instead - Worker 0 gets tasks 0, N, 2N... Worker 1 gets tasks 1, N+1,
  2N+1... - Use vanilla windowsarena/winarena image with correct IP (20.20.20.21) - Add container
  reuse check (skip restart if already running) - Pass API key via env var instead of config file -
  Fix QMP port exposure (7200) for QEMU control - Store Windows disk on /mnt for 300GB temp storage
  (D8ds_v5)

Tested: 2-worker pool running 4 tasks in parallel successfully

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **waa**: Use D4ds_v4 VM size for quota compatibility
  ([`8dfa40d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/8dfa40d7aeb7e0266f81546438c519cb945626ce))

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **waa**: Use D8ds_v5 VM size for Azure ML workers
  ([`e9dc820`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e9dc820b2cec517a8e4cdb5cc74189af080c4788))

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Documentation

- Add Azure ML log streaming and cost tracking guides
  ([`728f274`](https://github.com/OpenAdaptAI/openadapt-ml/commit/728f2747f105a7691dff6889294f0a1d84bd60b4))

Document the new CLI commands for: - Live log streaming from Azure ML jobs - Cost tracking for
  compute instances - Teardown procedures

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- **cli**: Add Azure ML log streaming, cost tracking, and teardown
  ([`401e36d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/401e36dd2f50a959ba0b022c2b0bef9f5a9a015b))

Add comprehensive Azure ML management commands: - azure-ml-stream: Stream logs from running jobs
  using Python SDK with account key auth (works around DefaultAzureCredential permission issues) -
  azure-ml-cost: Track compute instance uptime and estimated costs - azure-ml-teardown: Cancel jobs
  and delete compute instances

Also improves: - azure-ml-quota: Shows both ML Dedicated quota (what Azure ML actually uses) and
  regular VM quota - Better error handling and logging throughout

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **cli**: Add Azure ML status, VNC, and monitor commands
  ([`055ecc3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/055ecc33027f560a06459cd65b4ed3446124c0c4))

New commands for end-to-end Azure ML automation: - azure-ml-status: Show jobs and compute instances
  - azure-ml-vnc: Set up VNC tunnel to compute instance - azure-ml-monitor: Monitor jobs with auto
  VNC setup

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **cli**: Add azure-ml-quota command for quota management
  ([`5b34170`](https://github.com/OpenAdaptAI/openadapt-ml/commit/5b34170dd5c35969ea1c89733ccaf8b41f91a0a5))

Semi-automated quota increase workflow: - Checks current quota for WAA-compatible VM families -
  Shows which families have sufficient quota - Opens Azure Portal quota page with instructions -
  Guides user through the request process

Usage: uv run python -m openadapt_ml.benchmarks.cli azure-ml-quota

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **cli**: Add multi-VM pool commands for parallel WAA evaluation
  ([`d988e56`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d988e56d8954519ed28e01b47e88432b8c7e5672))

Add pool-create, pool-wait, and pool-run commands for running WAA benchmarks across multiple VMs in
  parallel:

- pool-create --workers N: Create N VMs with Docker and WAA image - Parallel VM creation using
  ThreadPoolExecutor - Auto-selects available region and VM size - Configures Docker with /mnt
  storage - Registers pool for tracking

- pool-wait: Wait for WAA to be ready on all workers - Starts WAA containers on each worker - Polls
  /probe endpoint until ready - Configurable timeout

- pool-run --tasks N: Distribute tasks across pool - Round-robin task distribution - Parallel
  execution on all workers - Progress tracking in registry

This enables ~5x faster benchmark completion with 5 workers, or full 154-task evaluation in ~10min
  with 10+ workers.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Refactoring

- **waa**: Update submodule with SDK v2 migration
  ([`241ddf8`](https://github.com/OpenAdaptAI/openadapt-ml/commit/241ddf85a36c55396969925301aa7889d017ffbb))

Updates WindowsAgentArena submodule to include Azure ML SDK v2 migration that enables job submission
  from macOS ARM64.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.2.2 (2026-01-29)

### Bug Fixes

- **ci**: Remove build_command from semantic-release config
  ([`6bd7ded`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6bd7ded5b6eda6631ebe34e5a98009fb0cd22120))

The python-semantic-release action runs in a Docker container where uv is not available. Let the
  workflow handle building instead.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Continuous Integration

- Add auto-release workflow
  ([`e6d067b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e6d067bbc31924be6e87feff5ffca292664b86b4))

Automatically bumps version and creates tags on PR merge: - feat: minor version bump - fix/perf:
  patch version bump - docs/style/refactor/test/chore/ci/build: patch version bump

Triggers publish.yml which deploys to PyPI.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Switch to python-semantic-release for automated versioning
  ([`404f26f`](https://github.com/OpenAdaptAI/openadapt-ml/commit/404f26f09cc5851659611795cbe902a43e44f2ad))

Replaces manual commit parsing with python-semantic-release: - Automatic version bumping based on
  conventional commits - feat: -> minor, fix:/perf: -> patch - Creates GitHub releases automatically
  - Publishes to PyPI on release

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.2.1 (2026-01-29)

### Bug Fixes

- **ci**: Resolve ruff linter and format errors
  ([#15](https://github.com/OpenAdaptAI/openadapt-ml/pull/15),
  [`cfbbed9`](https://github.com/OpenAdaptAI/openadapt-ml/commit/cfbbed97ce7699b86a78ba15e9cacbc125832509))

- Move warnings.warn() after imports to fix E402 in viewer files - Remove unused imports (Any,
  base64, os, Service) to fix F401 - Remove f-string without placeholders to fix F541 - Apply ruff
  formatting to 5 files

Files changed (7): - benchmarks/viewer.py - E402 fix - benchmarks/waa_deploy/api_agent.py - F401 +
  format - benchmarks/azure_ops_tracker.py - format only - benchmarks/vm_monitor.py - format only -
  cloud/local.py - format only - scripts/capture_screenshots.py - F401, F541 + format -
  training/viewer.py - E402 fix

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>

- **training**: Support VL models in standard transformers fallback
  ([#18](https://github.com/OpenAdaptAI/openadapt-ml/pull/18),
  [`84c3838`](https://github.com/OpenAdaptAI/openadapt-ml/commit/84c383803674979c5b8be62a415e39e0b5bbbbac))

* fix(training): support VL models in standard transformers fallback

Auto-detect vision-language models (Qwen2-VL, Qwen2.5-VL) and use the appropriate model class
  instead of always using AutoModelForCausalLM.

Detection criteria: - "VL" in model name (case-insensitive) - "vision" in model name - vision_config
  attribute in model config

Model class selection: - VL models: Qwen2VLForConditionalGeneration (with AutoModelForVision2Seq
  fallback) - Text-only models: AutoModelForCausalLM

Also sets task_type to SEQ_2_SEQ_LM for VL models in LoRA config.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* test(training): simplify VL tests to avoid model downloads

* fix(training): improve VL model support - catch RuntimeError, disable assistant_only_loss

- Add RuntimeError and TypeError to exception handling in _load_standard_model() to catch errors
  when loading Qwen2.5-VL with Qwen2VLForConditionalGeneration - Disable assistant_only_loss in
  standard TRL config as it's not supported for VL models yet

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>

### Chores

- Bump version to 0.2.1
  ([`7a1c054`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7a1c054101a164b01098ead5b078ce5647c32656))

Includes VL model support fix (PR #18): - Auto-detect VL models and use correct model class - Handle
  Qwen2VLForConditionalGeneration properly - Set assistant_only_loss=False for VL models

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Remove dead code and legacy fix scripts
  ([#16](https://github.com/OpenAdaptAI/openadapt-ml/pull/16),
  [`0da2f5b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/0da2f5be09eabb67831d316a4f9f1498fefb76e9))

Delete unused files: - training/viewer_migration_example.py (72 lines) - only self-referential -
  scripts/fix_acr_auth.py (212 lines) - one-time fix now baked into setup_azure.py -
  docs/azure_acr_authentication.md - docs for removed script

Update CLAUDE.md to remove references to deleted fix script.

Verified safe to delete: - None of these files are imported by cli.py - fix_acr_auth.py
  functionality is now in setup_azure.py (steps 10-12)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>

- Update gitignore and module exports
  ([`4ab39ea`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4ab39ea307ab0aa19d2b11cbf992d4d032c4e856))

- Add patterns for training output, synthetic data, experiment results - Add .jsonl,
  benchmark_live.json, external/, demos/ to gitignore - Export new runtime and schema types in
  module __init__.py

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Documentation

- Add architecture decisions, analysis, and design documentation
  ([`4c6859f`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4c6859f7ee241973b73796e887e6bac3ec5b4fda))

Key documents: - ARCHITECTURE_DECISIONS.md: Technical direction and decision records -
  analysis_jan2026.md: Comprehensive analysis and strategic options - enterprise/: SAC, Design
  Roadmap, Coords vs Marks ablation research

Design docs: - safety_gate_design.md: Safety gate architecture - perception_integration.md:
  Grounding integration design - representation_shootout_design.md: Coords vs Marks experiment
  design - viewer_consolidation_design.md, viewer_redesign_proposal.md

Experiment results: - waa_benchmark_results_jan2026.md: WAA benchmark analysis -
  grpo_training_report.md: GRPO training experiments - trl_unsloth_integration_analysis.md: Training
  integration analysis

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add ecosystem planning documents
  ([`f3afda5`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f3afda5d9837c1b6f37d1b6cce8db9dc794cac94))

- github_org_update_plan.md: GitHub org profile update strategy - desktop_app_plan.md: Desktop app
  distribution (pywebview + PyInstaller) - openadapt_integration_plan.md: Core openadapt integration
  roadmap

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add GitHub organization profile content recommendations
  ([`afaa1dc`](https://github.com/OpenAdaptAI/openadapt-ml/commit/afaa1dccd062e535dc3a47d819cc235fb4907cd2))

Add comprehensive recommendations for updating the OpenAdaptAI GitHub organization profile
  including:

- Organization bio (160 char max) - Organization README content for .github/profile/README.md -
  Pinned repositories recommendation (6 repos) - Repository descriptions for each package in the
  modular ecosystem

Focuses on the new modular architecture with openadapt as the unified entry point, highlighting
  openadapt-ml, openadapt-capture, openadapt-evals, openadapt-viewer, openadapt-grounding, and
  openadapt-retrieval packages.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add Qwen3-VL embedding research and design documentation
  ([`30e85c5`](https://github.com/OpenAdaptAI/openadapt-ml/commit/30e85c56fe85c36aed5eed36b44450a7a7f38528))

Add comprehensive documentation for Qwen3-VL vision-language embedding: -
  qwen3_vl_embedding_research.md: Literature review of VLM embedding extraction methods, including
  early exit strategies, hidden state extraction, and multimodal representation learning -
  qwen3_vl_embedding_design.md: Technical design document for extracting and using Qwen3-VL
  embeddings for GUI element retrieval and similarity-based action prediction

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add viewer architecture survey and comparison
  ([`21cc0fe`](https://github.com/OpenAdaptAI/openadapt-ml/commit/21cc0fedad47a270f70db564697b6dd21405fe12))

Survey of viewer technologies and frameworks for training/benchmark visualization, comparing options
  like Gradio, Streamlit, Panel, and custom HTML solutions for the unified viewer architecture.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add website redesign plan
  ([`6710e77`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6710e775b8dbfed9db260efd87584d86ce14ae3f))

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Pivot desktop app to uv-first distribution and propose meta-package architecture
  ([`98e4b2c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/98e4b2cee9c646072c4c04f2e09474338beec11a))

- desktop_app_plan.md: Switch from PyInstaller to uv-based installation - Tier 1: Single command
  install via uv tool - Tier 2: Optional uv bundled installer (~15MB) - Tier 3: PyInstaller full
  bundle (deferred) - Reduces annual cost from $500-700 to $0

- new_openadapt_architecture.md: Propose Option B+ thin CLI wrapper - Create unified 'openadapt'
  meta-package - Re-export common items from sub-packages - Unified CLI (openadapt
  capture/train/eval) - Phase-based implementation over 2 weeks

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Update openadapt-web repository reference to new name
  ([`f4176c7`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f4176c7dd197685bc2555172f07c197a1f806f43))

Update repository link from OpenAdapt.web to openadapt-web following the rename to match the
  lowercase-hyphen naming convention.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- Add safety gate, perception integration, and representation experiments
  ([`5778323`](https://github.com/OpenAdaptAI/openadapt-ml/commit/57783232ffbd8c8a3faf984e51c3559c4bf4673b))

New modules: - runtime/safety_gate.py: Deterministic safety gate for action validation -
  perception/integration.py: Bridge between openadapt-grounding and openadapt-ml -
  experiments/representation_shootout/: Coords vs Marks ablation framework -
  benchmarks/trace_export.py: Export benchmark traces to various formats

Tests: - Reorganize tests from root to tests/ directory - Add integration tests in
  tests/integration/ - Add test_gemini_grounding_imports.py for grounding module

Scripts: - p1_episode_success_ab_test.py: A/B test for demo-conditioned episode success

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add unified baseline adapters for VLM comparison
  ([`4921030`](https://github.com/OpenAdaptAI/openadapt-ml/commit/49210308e20e2498f8a4c7670c2819622b351563))

Implements a provider abstraction layer and unified baseline system for comparing Claude, GPT, and
  Gemini across multiple evaluation tracks.

New modules: - openadapt_ml/models/providers/ - API provider implementations - base.py:
  BaseAPIProvider ABC - anthropic.py: Claude support - openai.py: GPT support - google.py: Gemini
  support

- openadapt_ml/baselines/ - Unified baseline system - config.py: TrackConfig, BaselineConfig, MODELS
  registry - prompts.py: Track-specific prompt templates - parser.py: Response parsing with JSON and
  regex fallback - adapter.py: UnifiedBaselineAdapter main class - cli.py: CLI commands (run,
  compare, list-models)

Tracks supported: - Track A: Direct coordinate prediction - Track B: ReAct-style reasoning with
  coordinates - Track C: Set-of-Mark element selection

Usage: uv run python -m openadapt_ml.baselines.cli list-models uv run python -m
  openadapt_ml.baselines.cli run --model claude-opus-4.5 --track A --image screenshot.png --goal
  "Click submit"

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **experiments**: Add representation shootout and SOM evaluation results
  ([`e6aca89`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e6aca89d9f8f9f5c94bee28ee091d17f0d8858f6))

Add experiment results and artifacts: - representation_shootout results comparing embedding
  extraction methods - qwen_login 2b_dev_fixed plots showing base vs fine-tuned comparison -
  registration_som_eval.json evaluation metrics for SOM-based action prediction

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **waa**: Refactor CLI and fix Python 3.9 compatibility
  ([#14](https://github.com/OpenAdaptAI/openadapt-ml/pull/14),
  [`e55b610`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e55b610524300ba3665854c837327434329d489c))

- Refactor CLI from 6800 to ~1300 lines with flat command structure - Add analyze command to parse
  and summarize benchmark results - Add --num-tasks flag to limit number of tasks to run - Fix
  Python 3.9 compatibility by copying Python from vanilla WAA image (fixes transformers 4.46.2
  compatibility with GroundingDINO) - Add coverage and analysis artifacts to .gitignore

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>

### Refactoring

- **benchmarks**: Consolidate to re-export from openadapt-evals
  ([#17](https://github.com/OpenAdaptAI/openadapt-ml/pull/17),
  [`4232546`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4232546292a13b6a45e1210a9452933135e740ca))

* docs: add verified repo consolidation plan

- Two-package architecture: openadapt-evals (foundation) + openadapt-ml (ML) - Verified audit
  findings: 10 dead files confirmed, 3 previously marked dead but used - CLI namespacing: oa evals
  <cmd>, oa ml <cmd> - Dependency direction: openadapt-ml depends on openadapt-evals (not circular)
  - Agents with ML deps (PolicyAgent, BaselineAgent) move to openadapt-ml - adapters/waa/
  subdirectory pattern for benchmark organization

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

* feat: add openadapt-evals as optional dependency

Add [benchmarks] optional dependency for benchmark evaluation: - pip install
  openadapt-ml[benchmarks]

This is part of the repo consolidation to establish: - openadapt-evals: Foundation for benchmarks +
  infrastructure - openadapt-ml: ML training (depends on evals for benchmarks)

* docs(cli): clarify serve vs dashboard command naming

- oa ml serve: serve trained models for inference - oa ml dashboard: training dashboard for
  monitoring

This distinguishes the two use cases clearly: - serve = model inference endpoint - dashboard =
  training progress UI

* refactor(benchmarks): consolidate to re-export from openadapt-evals

Migrate benchmark infrastructure to two-package architecture: - openadapt-evals: Foundation package
  with all adapters, agents, runner - openadapt-ml: ML-specific agents that wrap openadapt-ml
  internals

Changes: - Convert base.py, waa.py, waa_live.py, runner.py, data_collection.py, live_tracker.py to
  deprecation stubs that re-export from openadapt-evals - Keep only ML-specific agents in agent.py:
  PolicyAgent, APIBenchmarkAgent, UnifiedBaselineAgent - Update __init__.py to import from
  openadapt-evals with deprecation warning - Update tests to import from correct locations - Remove
  test_waa_live.py (tests belong in openadapt-evals)

Net: -3540 lines of duplicate code removed

* refactor(benchmarks): delete deprecation stubs, import from openadapt-evals

Remove deprecation stubs since there are no external users. Tests now import directly from
  openadapt-evals (canonical location).

Deleted: - base.py, waa.py, waa_live.py, runner.py, data_collection.py, live_tracker.py

Kept: - agent.py (ML-specific agents: PolicyAgent, APIBenchmarkAgent, UnifiedBaselineAgent) -
  __init__.py (simplified to only export ML-specific agents)

* docs(readme): add WAA benchmark results section with placeholders

Add section 15 for Windows Agent Arena benchmark results with clearly marked placeholders. Results
  will be filled in when full evaluation completes. Warning banner indicates PR should not merge
  until placeholders are replaced.

Sections added: - 15.1 Benchmark Overview - 15.2 Baseline Reproduction (paper vs our run) - 15.3
  Model Comparison (GPT-4o, Claude, Qwen variants) - 15.4 Domain Breakdown

* docs(readme): move WAA benchmark results to openadapt-evals

WAA benchmark results belong in openadapt-evals (the benchmark infrastructure package) rather than
  openadapt-ml (the training package).

See: https://github.com/OpenAdaptAI/openadapt-evals/pull/22

* feat(cli): add VNC auto-launch and --fast VM option

- Add setup_vnc_tunnel_and_browser() helper for automatic VNC access - Add VM_SIZE_FAST constants
  with D8 series sizes - Add VM_SIZE_FAST_FALLBACKS for automatic region/size retry - Add --fast
  flag to create command for faster installations - Add --fast flag to start command for more QEMU
  resources (6 cores, 16GB) - Opens browser automatically after container starts

* docs: add WAA speedup options documentation

- Document --fast VM flag usage - Explain parallelization options - Detail golden image approach for
  future optimization

* docs(readme): add benchmark execution logs section

- Add section 13.5 with log viewing commands - Add benchmark run commands with examples - Renumber
  screenshot capture tool section to 13.6

* docs(readme): clarify --run flag for benchmark execution logs

- Add logs --run command for viewing task progress - Add logs --run -f for live streaming - Add logs
  --run --tail N for last N lines

* docs(readme): add example output for logs commands

- Add example output for `logs` (container status) - Add example output for `logs --run -f`
  (benchmark execution)

* feat(cli): add --progress flag for benchmark ETA

- Add _show_benchmark_progress() function - Parse run logs for completed task count - Calculate
  elapsed time and estimated remaining - Show progress percentage

Example usage: uv run python -m openadapt_ml.benchmarks.cli logs --progress

* docs(research): add cua.ai vs openadapt-ml WAA comparison

Comprehensive analysis of Cua (YC X25) computer-use agent platform: - Architecture comparison
  (composite agents, sandbox-first) - Benchmark framework differences (cua-bench vs openadapt-evals)
  - Training data generation (trajectory replotting) - Recommendations: adopt patterns, not full
  migration

Key findings: - Cua's parallelization uses multiple sandboxes (like our multi-VM plan) - Composite
  agent pattern could reduce API costs - HTML capture enables training data diversity

* feat(cli): add parallelization support with --worker-id and --num-workers

WAA natively supports parallel execution by distributing tasks across workers.

Usage: # Run on single VM (default) run --num-tasks 154

# Run in parallel on multiple VMs VM1: run --num-tasks 154 --worker-id 0 --num-workers 3

VM2: run --num-tasks 154 --worker-id 1 --num-workers 3

VM3: run --num-tasks 154 --worker-id 2 --num-workers 3

Tasks auto-distribute: worker 0 gets tasks 0-51, worker 1 gets 52-103, etc.

* docs(research): add market positioning and strategic differentiation

Expand cua_waa_comparison.md with: - Success rate gap analysis (38.1% vs 19.5%) - Market positioning
  comparison (TAM, buyers, value props) - Where sandbox approach fails (Citrix, licensed SW,
  compliance) - Shell applications convergence opportunities - Bottom line: Windows enterprise
  automation is hard, validates OpenAdapt approach

* docs(waa): add parallelization and scalable benchmark design docs

- Add WAA_PARALLELIZATION_DESIGN.md documenting: - Official WAA approach (Azure ML Compute) - Our
  dedicated VM approach (dev/debug) - When to use each approach

- Add WAA_UNATTENDED_SCALABLE.md documenting: - Goal: unattended, scalable, programmatic WAA -
  Synthesized approach using official run_azure.py - Implementation plan and cost estimates

- Update Dockerfile comments to clarify: - API agents (api-claude, api-openai) run externally -
  openadapt-evals CLI connects via SSH tunnel - No internal run.py patching needed

* style: fix ruff formatting

* fix(imports): update internal code to import from openadapt-evals

Replace imports from deleted benchmark files with direct imports from openadapt-evals:

- azure.py: BenchmarkResult, BenchmarkTask, WAAAdapter - waa_demo/runner.py: BenchmarkAction,
  WAAMockAdapter, etc.

This completes the migration to the two-package architecture where openadapt-evals is the canonical
  source for benchmark infrastructure.

* fix(imports): add missing EvaluationConfig import

- Update azure.py to import BenchmarkAgent from openadapt_evals - Add EvaluationConfig to runner.py
  imports

Fixes CI failure: F821 Undefined name `EvaluationConfig`

* fix(deps): require openadapt-evals>=0.1.1

v0.1.0 uses task ID format "browser_1" but tests expect "mock_browser_001" which was added in
  v0.1.1.

---------

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>

- **benchmarks**: Migrate to openadapt-evals package
  ([`e6f63c7`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e6f63c79430cd965efe76c65407caba43973d449))

BREAKING CHANGE: Benchmark code moved to openadapt-evals.

- Update CLAUDE.md with migration guide - Add deprecation warning to benchmarks/__init__.py - Old
  imports still work but emit DeprecationWarning

Migration: # OLD (deprecated) from openadapt_ml.benchmarks import WAAMockAdapter

# NEW (preferred) from openadapt_evals import WAAMockAdapter

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Testing

- Add unit tests for providers and baselines modules
  ([`a56ec04`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a56ec0424482d3a5107fafd74d4cc7b6e729454a))

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.2.0 (2026-01-09)

### Bug Fixes

- Resolve test failures and SSE dashboard state conflicts
  ([`041912d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/041912d4c1962655844a381c77d940b99cc21761))

Test fixes: - test_action_parsing.py: Handle 4-value return from predict_action_from_sample() -
  test_api_adapter.py: Fix mock patch locations (openai.OpenAI, anthropic.Anthropic) - trainer.py:
  Change logger.save() to logger._save_log() - policy.py: Allow negative coords in CLICK regex for
  clamping tests

SSE dashboard fixes: - Add phase: "ready" to Azure VM Host tasks to prevent Starting+completed
  conflict - Improve frontend phase inference from status when phase is missing - Add debug console
  logging for SSE troubleshooting

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **cli**: Use localhost for VNC URLs via SSH tunnel
  ([`947816d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/947816daf563d3f7333d62a58e1e9df9c81d02ed))

Probe output now correctly shows localhost:8006 instead of public IP which is not accessible without
  SSH tunnel.

- **waa**: Add full Python dependencies for benchmark client
  ([`0544199`](https://github.com/OpenAdaptAI/openadapt-ml/commit/05441990b839a2b76471e60298e35605cd451f46))

- Add build-essential, ffmpeg, and X11 libs for package compilation - Install core packages:
  gymnasium, fabric, transformers, torch (CPU) - Install ML packages: opencv, easyocr, matplotlib,
  accelerate - Create python -> python3 symlink for compatibility - Separate pip installs into
  layers for better caching

- **waa**: Add missing pydrive and other client dependencies
  ([`238ff91`](https://github.com/OpenAdaptAI/openadapt-ml/commit/238ff9109ec19be6f9b4566564eb39f43ba68b7d))

- **waa**: Add remaining WAA client dependencies (openpyxl, docx, etc.)
  ([`678df44`](https://github.com/OpenAdaptAI/openadapt-ml/commit/678df440aadccd4cc48d41bd2924c6fc42728055))

- **waa**: Copy OEM files to Samba share at container startup
  ([`04d5f94`](https://github.com/OpenAdaptAI/openadapt-ml/commit/04d5f949dac9e2356b162e19430cd007411bef52))

Add /copy-oem.sh startup script that copies OEM files from /oem to /tmp/smb (Samba share) at
  container startup. This fixes Windows not finding setup scripts because smb.conf is generated at
  runtime.

Also update experiment doc to remove timeline estimates and add WAA baseline as in-progress.

- **waa**: Copy Python env from official image to avoid 3.13 compat issues
  ([`4a27a22`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4a27a229400e9de11c5bdc733f812b7e04a77643))

### Chores

- Bump version to 0.2.0 for PyPI release
  ([`6aedda3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6aedda38e77b79a7efdec94e3e8366166a83a8b0))

Features in this release: - TRL + Unsloth training integration (2x faster, 50% less VRAM) -
  Standardized on uv for package management - Enhanced VM CLI and WAA deployment - Comprehensive
  documentation updates

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Remove old waa/Dockerfile (moved to waa_deploy/)
  ([`ad78e78`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ad78e78357a5748c2a395b042b072af59182cb38))

- Standardize on uv for package management
  ([`eb3aecd`](https://github.com/OpenAdaptAI/openadapt-ml/commit/eb3aecd2f84f871e139f564708b7d196e21aa917))

- Replace all `pip install` with `uv add` in docs - Update cloud GPU training to use `curl ... | sh`
  for uv install - Update CLAUDE.md with enhanced VM operations guidance - Consistent `uv sync` for
  local development

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Update CLAUDE.md and minor fixes
  ([`84507df`](https://github.com/OpenAdaptAI/openadapt-ml/commit/84507df6dccc0f2c2585a5aae8ed4b7b8e5138ca))

- Updated CLAUDE.md with new features and documentation - trainer.py: minor improvements -
  eval_policy.py: updated for new schema - uv.lock: dependency updates

- Update uv.lock
  ([`63797af`](https://github.com/OpenAdaptAI/openadapt-ml/commit/63797af3a938867e92587bb0f8bb36219aa824aa))

### Documentation

- Add benchmark viewer screenshot to README
  ([`88889c3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/88889c3a71fecb6b4b7fce98e32aef53096a0a96))

- Add capture format decision framework
  ([`d57c27a`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d57c27a58d478a439471756e2b44c29c1b9fe38e))

Explores options for data format interoperability: - Option A: Native Episode output from
  openadapt-capture - Option B: Conversion layer in openadapt-ml (recommended) - Option C: Shared
  schema package - Option D: Dual output

Recommends Option B with clear guidelines for the conversion layer. Includes text demo format
  specification for WAA experiment.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add comprehensive capture migration guide with a11y integration
  ([`779ba58`](https://github.com/OpenAdaptAI/openadapt-ml/commit/779ba588b5d01d704bee689297c12cc135c8081e))

- Add design documents for SSE, retrieval, and parallelization
  ([`464861c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/464861c9f67028f4d0125fce1e13c7bcc502e759))

- SSE architecture and integration guides - Demo retrieval design and experiments - WAA
  parallelization and live adapter plans - Chrome extension design for capture - Benchmark viewer UX
  improvements

- Add openadapt-capture to openadapt-ml migration plan
  ([`447bd7e`](https://github.com/OpenAdaptAI/openadapt-ml/commit/447bd7e7d85d269ef8d3be897a4aec4ec1ad3803))

- Add schema consolidation plan
  ([`4d7c18a`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4d7c18ad5323f7f4dac0275a605a9659c88c7108))

Detailed migration plan for consolidating from two schema modules to one: - DELETE:
  openadapt_ml/schemas/ (dataclass-based, legacy) - KEEP: openadapt_ml/schema/ (Pydantic-based,
  canonical)

Includes: - Dependency analysis (22 files affected) - Field mapping between old and new - 7-phase
  migration strategy - Testing strategy - Rollback plan - Timeline estimate (~8-10 hours)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add staged export hierarchy to enterprise guide
  ([`f093304`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f0933044796ceef8ff4c28bdcf841c030acda88f))

- Episode JSON as canonical, Parquet/WebDataset as projections - Expand data loss table for flat
  formats - Mark exporters as Planned with design doc links - Add multi-step evaluation caveat

- Add WAA demo recording guide for Windows captures
  ([`6d8bdb1`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6d8bdb17b32369a1d5c9c6db2dff88dbcd301316))

Step-by-step instructions for recording the 3 complex demos: - Task #4: Fill blank cells
  (LibreOffice Calc) - Task #5: Create chart (LibreOffice Calc) - Task #9: Archive folder (File
  Explorer)

Includes setup, recording steps, export, and transfer instructions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Consolidate WAA CLI workflow documentation
  ([`dbe00d0`](https://github.com/OpenAdaptAI/openadapt-ml/commit/dbe00d07d92e4d45ed4c3edad2a19894ade6c496))

- Add Quick Start: Single VM section to azure_waa_setup.md - Replace manual SSH steps in CLAUDE.md
  with CLI commands - Document custom waa-auto Docker image that fixes OEM folder issue - Add vm
  probe, vm reset-windows, and other useful commands

- Strengthen enterprise integration guide positioning
  ([`c093f93`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c093f93d4b8971afba26498c9930ff129e691f84))

Add decision boundary, requirements, retrofitting cost sections. Add data portability note
  addressing vendor lock-in concern. Add optional metadata extension pattern. Add typical
  integration workflow. Add open schema independence statement.

- Update README for TRL training and PyPI installation
  ([`1c8e899`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1c8e8996cb8b633719e4185af1cc90c3b38489a6))

- Add Installation section with PyPI instructions (uv add openadapt-ml) - Update training section to
  reflect TRL + Unsloth integration - Update repository structure with trl_trainer.py reference -
  Add PyPI badge - Fix section numbering throughout - Update test descriptions for TRL trainer

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Validate demo-conditioning at n=45 (46.7% → 100%)
  ([`bce5a11`](https://github.com/OpenAdaptAI/openadapt-ml/commit/bce5a11173b2199217baa9163b19644f932b2fd7))

- Zero-shot: 46.7% (21/45), Demo: 100% (45/45), Control: 57.8% - Improvement: +53.3 percentage
  points across 15 macOS categories - Add Parquet export design doc (derived analytics format) -
  Update enterprise guide with validated result

- **experiment**: Strengthen demo-conditioning doc for expert review
  ([`b56faa8`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b56faa8960b49ef5ae119ce2643a24715be86f29))

- Add interpretation note framing result as "trajectory-conditioned disambiguation" not general
  task-solving - Highlight length-matched control in executive summary - Frame shared first action
  as intentional controlled variable - Add "Positioning Relative to Fine-Tuning" section connecting
  to prompting-first methodology - Expand limitations with actionable specifics (WAA running, SoM
  conventions, episode success vs first-action)

- **schema**: Comprehensive documentation for Episode schema
  ([`0963b75`](https://github.com/OpenAdaptAI/openadapt-ml/commit/0963b7585fc46be9b3d665a25ce0126246a13e92))

- Add detailed Quick Start with code examples - Document all 24 action types with categories -
  Explain pixel vs normalized coordinate systems - Add validation and format conversion examples -
  Document extension points (raw, metadata fields) - Add docs/schema/README.md as standalone
  reference

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- Add demo-conditioned prompting experiment and retrieval module
  ([`f80921e`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f80921e26e3c4a12231eab4e2ad55eca2f5abbd6))

Validated that demo-conditioning improves first-action accuracy from 33% to 100% (n=3, preliminary
  signal). Key findings: - Benefit is semantic, not token-length (length-matched control: 67%) -
  Demos generalize across task variations (toggle polarity, parameters) - Zero-shot has systematic
  spatial bias that demos correct

Added retrieval module (TF-IDF + domain bonus) for automatic demo selection. Added demo-conditioned
  training mode to train_from_json.py. Added enterprise integration guide for workflow data export.

Statistical note: n=3 is insufficient for significance. Validation at n≥30 on expanded task set in
  progress.

- Add Episode JSON schema and polish benchmark viewer
  ([`d9e669e`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d9e669e1a3e4e8150c8d7458912de27632f09ca3))

Episode Schema (openadapt_ml/schema/): - Pydantic models for Episode, Step, Action, Observation -
  Schema version 1.0.0 with semver evolution policy - WAA format converter (from_waa_trajectory,
  to_waa_trajectory) - JSON Schema export for documentation/tooling - 20 action types (click, type,
  key, hotkey, scroll, drag, etc.)

Benchmark Viewer Improvements: - Fix SSE memory leak (clearAllIntervals on reconnect) - Add
  ThreadedTCPServer for concurrent request handling - Polish UI with color-coded status, loading
  spinners, error banners - Add refresh buttons to all panels with feedback - Prominent VNC button
  with copy-to-clipboard IP

CLI Enhancements: - Add --auto-shutdown flag to deallocate VM after benchmark - Add --timeout flag
  for Azure ML job auto-cancellation - Add vm cleanup-stale command for finding stale jobs/VMs - Add
  refresh button support in Azure Jobs API

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add WAA demo-conditioned experiment with 7 manual demos
  ([`4f9799a`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4f9799abe00c32ef89fb4ed8731b356f7e79013e))

Implements hybrid demo approach for WAA benchmark: - 7 manual demos for simple tasks (settings,
  toggles, linear flows) - 3 placeholders for complex tasks needing recorded demos

Tasks covered: 1. Do Not Track (Edge) - manual 2. Bookmark to bar (Edge) - manual 3. Font size
  (Edge) - manual 4. Fill blank cells (Calc) - needs recording 5. Create chart (Calc) - needs
  recording 6. Center align (Writer) - manual 7. Notifications (Settings) - manual 8. Night Light
  schedule (Settings) - manual 9. Archive folder (Explorer) - needs recording 10. Details view
  (Explorer) - manual

Includes runner CLI for listing tasks and viewing demos.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Enhanced VM CLI and WAA deployment
  ([`cd95c43`](https://github.com/OpenAdaptAI/openadapt-ml/commit/cd95c436da928791293df23ec4e00485e4082ff9))

CLI improvements: - Add vm deallocate, start, exec, fix-oem, docker-prune, stop-build actions - SSH
  keepalive settings (60s interval) to prevent timeouts - Docker startup check after VM restart -
  Better probe checking (curl from inside container)

WAA deployment: - Move Dockerfile to waa_deploy/ with api_agent.py - Add api-claude and api-openai
  agent support - P0 demo persistence validation script

Demo persistence validated: - scripts/p0_validate_demo_persistence.py confirms demo included at all
  steps - ApiAgent properly passes demo through multi-step episodes

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Trl + Unsloth training integration
  ([`c006f2b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c006f2b00b0d010eec3864fcc81b10b5e1b1f2a9))

- Add trl_trainer.py with SFTTrainer + Unsloth optimizations - Update train_from_json.py to use TRL
  trainer (2x faster, 50% less VRAM) - Remove legacy custom training loop from trainer.py - Add
  [training] optional dependencies (trl, datasets) - Support --use-unsloth / --no-unsloth flags

Training command: uv run python examples/train_from_json.py --data episodes/ --output results/

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **benchmark**: Add Run Benchmark UI panel and fix VNC/Docker issues
  ([`a86269e`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a86269e2e2a7f328d39de990917a671c440a3654))

- Add Run Benchmark panel to benchmark viewer (model, tasks, agent, domain selection) - Add POST
  /api/benchmark/start endpoint to launch benchmarks from UI - Add --domain and --task-ids CLI flags
  for filtered benchmark runs - Fix VNC link to use localhost:8006 (SSH tunnel) instead of direct VM
  IP - Fix Docker build to use --no-cache --pull to prevent stale dockurr/windows layers - Add
  docs/waa_network_architecture.md explaining the localhost-based network topology - Add
  docs/benchmark_run_ui_design.md with UI design specification

The Docker cache issue caused dockurr/windows v0.00 scripts (no auto-download) to be used instead of
  v5.14 (with auto-download). Fixed by adding --no-cache --pull.

- **benchmarks**: Waa CLI improvements, result analysis, and viewer enhancements
  ([`c8d1ce0`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c8d1ce08226185c9ff0b9a777f31bd549a677b2f))

WAA CLI: - Add `analyze` command for programmatic WAA result analysis - Remote analysis via SSH
  (--vm-ip --remote) - Local directory analysis (--results-dir) - Per-domain success rates, JSON
  export - Fix invalid model name: gpt-5.2 → gpt-4o - Add --skip-build tip for faster reruns - Add
  vm_monitor.py for VM status tracking - Add live_tracker.py for real-time benchmark progress

Documentation: - Add docs/waa_setup.md - WAA setup guide - Add docs/GEMINI_GROUNDING_QUICKSTART.md -
  Add docs/background_task_visibility.md - Add implementation summaries

Viewer: - Add benchmark_viewer.py for WAA result visualization - Enhance local.py serve command -
  Integrate benchmarks into unified viewer

- **export**: Add Parquet exporter and toolbox positioning
  ([`bb11d62`](https://github.com/OpenAdaptAI/openadapt-ml/commit/bb11d6210443ea1c3133ccfeda3112bb8ad5ffd8))

Add first-class Parquet export support: - to_parquet() / from_parquet() for Episode serialization -
  CLI: python -m openadapt_ml.export parquet --input <dir> --output <path> - Optional summary table
  generation - pyarrow as optional dependency

Update enterprise integration guide: - Add "What is openadapt-ml?" toolbox section - Frame as
  composable utilities, not monolithic framework - Update Parquet section with real implementation
  examples - Scope canonical claim to within openadapt-ml

- **retrieval**: Add demo retrieval system and WAA live adapter
  ([`b62b6f1`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b62b6f1eb7dc81b99eb7787ec019f8ba1045c288))

Demo Retrieval: - embeddings.py: improved embedding generation with caching - demo_retriever.py:
  semantic search for relevant demonstrations - Support for goal-based and screenshot-based
  retrieval

Benchmark Viewer: - viewer.py: standalone HTML viewer for benchmark results - waa_live.py: live
  evaluation adapter for WAA benchmarks - Integrated with dashboard for real-time monitoring

- **schema**: Add converters between internal and external formats
  ([`3124992`](https://github.com/OpenAdaptAI/openadapt-ml/commit/312499279b1d3e369c0abfa7db0aaba2b6e3ea37))

- from_internal_episode(): Convert schemas.sessions.Episode to schema.Episode -
  to_internal_episode(): Convert schema.Episode back to internal dict format - Document field
  mapping in README

This enables bidirectional conversion between: - Internal training format (schemas.sessions): id,
  goal, t, image_path, x/y - External interop format (schema.episode): episode_id, instruction,
  step_index, etc.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **schema**: Add select_monitor and normalized coordinates
  ([`da9fc16`](https://github.com/OpenAdaptAI/openadapt-ml/commit/da9fc16fbc03f0c319f5e5f1453c73df05445161))

- Add action types: select_monitor, window_focus, window_resize, window_move - Add monitor_id field
  for select_monitor action - Add window_title field for window_focus action - Add
  normalized_coordinates (0.0-1.0) as alternative to pixel coords - Add normalized_start/end for
  resolution-independent drag actions

This enables cu-episode-v1 alignment without loss of information.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **waa**: Auto-build Docker image and fix tunnel detection
  ([`f71e556`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f71e55672da192f8372d164d3119b8b1360f114a))

- CLI run-waa now automatically builds waa-auto image if missing - Added --rebuild flag to force
  image rebuild - Dockerfile: fixed IP patching, added playwright for web automation -
  ssh_tunnel.py: fixed tunnel status to check actual port state instead of just internal tracking,
  correctly reports external tunnels

Closes #XX

### Refactoring

- Consolidate schema to single Pydantic-based Episode module
  ([`c05ad6b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c05ad6bb8a2c3682b6beed430f5a3f96ad5fcf4c))

- Migrate from dual schema modules (schemas/ dataclass-based) to single canonical Pydantic schema
  (schema/episode.py) - Delete old openadapt_ml/schemas/ directory (sessions.py, validation.py) -
  Update all imports across 27 files to use openadapt_ml.schema

Schema field mappings (old -> new): - Episode.id -> Episode.episode_id - Episode.goal ->
  Episode.instruction (required), Episode.goal (optional) - Step.t -> Step.step_index (int) +
  Step.timestamp (float) - Step.thought -> Step.reasoning - Observation.image_path ->
  Observation.screenshot_path - Action.x, Action.y -> Action.normalized_coordinates (tuple) -
  Action.type (str) -> Action.type (ActionType enum) - Action.element_index ->
  Action.element.element_id (via UIElement)

Added fields to Observation: app_name, url for benchmark compatibility

Converters in schema/converters.py handle legacy format conversion.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Testing

- Add comprehensive tests for WAA demo experiment module
  ([`ae5930b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ae5930be3ccf91ee2bcb860f2527cffa3e262c1c))

28 tests covering: - Task definitions (10 tasks, domains, difficulties) - Demo content (7 complete,
  3 placeholder) - Integration (task/demo consistency, retrieval) - Format validation (DEMONSTRATION
  header, Goal line)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add tests for demo retrieval and WAA live adapter
  ([`d9b6eab`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d9b6eabbf22a7d7a5c6cd66cefd128b0f6e98c2d))

- test_demo_retrieval.py: comprehensive tests for demo retriever - test_waa_live.py: tests for live
  WAA adapter - Updated test_retrieval.py for new embedding features - demo_retrieval_example.py:
  usage examples

- Update tests for TRL trainer refactor
  ([`0694d8d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/0694d8db61cba4181f6aae425e16292c09d06fa3))


## v0.1.0 (2025-12-16)

### Bug Fixes

- Consistent 'Viewer' label in nav tabs
  ([`4a6f627`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4a6f627f13e939e9fb9ba594d1346e9364e3b8cc))

- **dashboard**: Improve viewer/dashboard consistency and CLI commands
  ([`27a9857`](https://github.com/OpenAdaptAI/openadapt-ml/commit/27a9857cf731d34d6b0a53ed3bc38db4c0260768))

- **dashboard**: Remove duplicate job ID and make header full-width
  ([`5c3c22b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/5c3c22b6cc9abd5de95df9e34ed551f601522ac8))

- **dashboard**: Use subprocess for simpler http server startup
  ([`4e74142`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4e74142f2d900927bfdff24f88c81d375d738b9e))

- **plots**: Add model size labels to hardened benchmark plots
  ([`4c828f2`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4c828f20066533f1b0f7b74ac35ce20c46deedec))

- **serve**: Remove stale refresh args, stop button now works
  ([`b7206bc`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b7206bc8dc70e71adc236e213ff6a8e3acbdcc63))

- **stub**: Auto-copy real screenshot for evaluation samples
  ([`53c9f50`](https://github.com/OpenAdaptAI/openadapt-ml/commit/53c9f5074bb5235f9264b6665568d6ce6ab2ac01))

- **viewer**: Extract predictions from window.comparisonData and fix elapsed time loading
  ([`475c38d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/475c38d624617108aa558239971f84b32ad4c061))

- **viewer**: Sync audio speed with playback, add visual feedback to overlay toggles
  ([`e58cc22`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e58cc22f561be8ade47d3e9ad259128035d1c154))

### Chores

- **gitignore**: Ignore synthetic and ephemeral training artifacts
  ([`ff3a6e3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ff3a6e30e406fd534b90c6f07380fd52f09d9933))

- **plots**: Track hardened v2 experiment plots and scope ignore to top-level
  ([`1511e6d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1511e6d0f69731a2e80ae262b12e53b7ff2d5105))

- **readme**: Point synthetic plots at hardened v2 experiment artifacts
  ([`6f35079`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6f35079a9828b8fdbd267aa83a2d0113950788b9))

### Documentation

- Add benchmark viewer integration design and update TODOs
  ([`f21c828`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f21c8289a5a88162b44a9102b0ac15a1eae4ff7c))

- Add early termination controls as high priority TODO
  ([`6ed8042`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6ed80420690dc60b3a1d2874d3d78a6501a4601a))

- Document need for auto-termination, dashboard stop button, checkpoint download - Fix shared header
  in unified viewer template (trainer.py) - Remove 'Dashboards:' label from compare.py nav

- Add GUI-Actor integration plan for coordinate-free grounding
  ([`a67abc3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a67abc3f9be7912ce0f3d3a425dc321f43282cbf))

- Add training feedback UX critical path analysis
  ([`a87b9bb`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a87b9bba5dee22fdb66e60a196005271e94e6903))

- Add unified compute architecture design and PyPI TODO
  ([`e2abfe3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e2abfe329a552c06064123227d157edc8204536c))

- **readme**: Add 2b training log snippet and clarify qwen3 masking roadmap
  ([`f02ebfd`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f02ebfd557dd01a520ee3f73608da6cc7f0c0039))

- **roadmap**: Mark Priority 5a complete and update plotting achievements
  ([`2da03f2`](https://github.com/OpenAdaptAI/openadapt-ml/commit/2da03f2ee9a1f7780e336323f036b21d293776ca))

- **viewer**: Add timeline visualizer and eval integration design
  ([`d3e93ed`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d3e93ede05513c2ee92a6a51c2ef4139fbc493ef))

### Features

- Initial commit of openadapt-ml pipeline (synthetic login, qwen adapters, eval + training)
  ([`ec92d6b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ec92d6b250dc6e4c184e9b2ff07c12781979ba1d))

- V0.1.0 release with benchmark integration, grounding module, and cloud training
  ([`b29d558`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b29d5586967e0673ee45417ca11797f75c46e00f))

- **benchmark**: Add qwen login orchestrator and refine docs
  ([`efcce00`](https://github.com/OpenAdaptAI/openadapt-ml/commit/efcce008ecf4839857cae154b94b29499d959809))

- **cloud**: Add Lambda Labs training, benchmarks, and training visualization
  ([`2063aea`](https://github.com/OpenAdaptAI/openadapt-ml/commit/2063aea43501253296d4ac6a09c3c56fc72614b3))

- **config**: Add pydantic-settings configuration and API benchmarks
  ([`2e7bfd1`](https://github.com/OpenAdaptAI/openadapt-ml/commit/2e7bfd1923b15223e7bf59181244f465cd300d7a))

- **dashboard**: Add early termination controls and /api/stop endpoint
  ([`7c27f47`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7c27f476792914c3b1115891c2c98b2296fa7e71))

- **dashboard**: Enhance evaluation samples with model thinking display
  ([`a0e2b09`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a0e2b09250bbcc83257072674197c46e63ed3845))

- **dashboard**: Show model thinking by default, add legend
  ([`fd26952`](https://github.com/OpenAdaptAI/openadapt-ml/commit/fd26952b6c12c6b2a72844f9820f295860e145a9))

- **docs**: Add dashboard screenshots and fix viewer predictions
  ([`8ea00d0`](https://github.com/OpenAdaptAI/openadapt-ml/commit/8ea00d0df44f3ffcc98be08ee3b7b77a571a061d))

- **lambda**: Add early termination controls with auto-stop, checkpoint download, and dashboard stop
  button
  ([`4936fe0`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4936fe03d729f407c58e1d5c4bb725f8637f74a2))

- **lambda**: Auto-symlink capture screenshots and rewrite paths
  ([`1bef82c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1bef82c0a980efbfaa3497d4a0d8b9576584e1bd))

- **local**: Add local training CLI for CUDA/Apple Silicon
  ([`1efb175`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1efb175dacb965e267f403c9c79603c738e53f67))

- **plots**: Add legend to comprehensive comparison and streamline README
  ([`b710d49`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b710d4939e4aee97d9553838cf78947c3c99590e))

- **plots**: Add legend to qwen_vs_apis comparison plot
  ([`eee3f25`](https://github.com/OpenAdaptAI/openadapt-ml/commit/eee3f254f2d8900dc5dc92628fc534e7d3d69040))

- **plots**: Update individual plots with consistent color coding and legend
  ([`e934e05`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e934e050cccbffb34cbdaa41fd34f84136d1c784))

- **qwen-login**: Harden benchmark and add plots, GIF, and output docs
  ([`1b29003`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1b290031a0402cb77b5db1aae4c0e0d0bd5a1a17))

- **synthetic-login**: Harden jitter, prompts, and Qwen3-VL 2B/8B results
  ([`9b27555`](https://github.com/OpenAdaptAI/openadapt-ml/commit/9b275558dc77fdcac1abc1ddaf5bacb8f9b9c2d8))

- **training**: Add job-scoped directories and HTTP server for dashboards
  ([`cb69e6a`](https://github.com/OpenAdaptAI/openadapt-ml/commit/cb69e6a7366a1a4c3dec1c8db4c9c9ee27a8e948))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **training**: Add stub adapter for rapid UI testing without GPU
  ([`5f236aa`](https://github.com/OpenAdaptAI/openadapt-ml/commit/5f236aa53faa467692613ff4e1dc4273dab0ac61))

- **viewer**: Add benchmark tab with WAA integration WIP state
  ([`a707acb`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a707acb780e2996467e49c849817a96eb4761ee8))

- **viewer**: Add parseModelOutput for SoM parsing and truncation
  ([`1217f64`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1217f6400cce64ba0afe7e411479cbe16010d1dc))

- **viewer**: Add screenshots to README and smart auto-scroll
  ([`e147f09`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e147f091382c9d871a306b63cde6e84bdd90f4ee))

- **viewer**: Add SoM action parsing for model predictions
  ([`6a89164`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6a89164fdb4826fa7064c76612bcd7a857e366cf))

- **viewer**: Add transcript/audio sync, copy-all button, and extract shared UI
  ([`6beff41`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6beff41e32ed064a81de05db62b3dfbd7e9ea67c))

- **viewer**: Extract viewer module with evaluation gallery and badges
  ([`911faac`](https://github.com/OpenAdaptAI/openadapt-ml/commit/911faac01b923a1ab23c69b3cb2ed3499d3a0c57))

### Refactoring

- Rename --eval-on-training-data to --overfit
  ([`3cb9dc1`](https://github.com/OpenAdaptAI/openadapt-ml/commit/3cb9dc1e9efab45a13ec91dca53f5c5524aef9bd))

- **viewer**: Consolidate to standalone HTML generation
  ([`bbda9c9`](https://github.com/OpenAdaptAI/openadapt-ml/commit/bbda9c9c30f205369caf89099240b65f9b7ace54))

### Testing

- **local**: Add tests for local CLI with early stopping
  ([`e371687`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e37168798f4b0a2469299f05f202f08954d6648c))
