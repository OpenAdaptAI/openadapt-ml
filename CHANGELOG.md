# CHANGELOG


## v0.2.2 (2026-01-29)

### Bug Fixes

- **ci**: Remove build_command from semantic-release config
  ([`c0d455a`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c0d455a395ec27f9705b15661cf978b092772a35))

The python-semantic-release action runs in a Docker container where uv is not available. Let the
  workflow handle building instead.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Continuous Integration

- Add auto-release workflow
  ([`46c91fe`](https://github.com/OpenAdaptAI/openadapt-ml/commit/46c91fe137d807a738a122a60c512470612ea708))

Automatically bumps version and creates tags on PR merge: - feat: minor version bump - fix/perf:
  patch version bump - docs/style/refactor/test/chore/ci/build: patch version bump

Triggers publish.yml which deploys to PyPI.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Switch to python-semantic-release for automated versioning
  ([`4b5ab9a`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4b5ab9aaeb970b0ef4798fed9aa0a7d7d7854e01))

Replaces manual commit parsing with python-semantic-release: - Automatic version bumping based on
  conventional commits - feat: -> minor, fix:/perf: -> patch - Creates GitHub releases automatically
  - Publishes to PyPI on release

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.2.1 (2026-01-29)

### Bug Fixes

- **ci**: Resolve ruff linter and format errors
  ([#15](https://github.com/OpenAdaptAI/openadapt-ml/pull/15),
  [`af64fe3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/af64fe3c710b747baab7866a21b1bd87993ab426))

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
  [`2b2c1df`](https://github.com/OpenAdaptAI/openadapt-ml/commit/2b2c1df49753f9498b793925772f349f4a66c00a))

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
  ([`cd969f8`](https://github.com/OpenAdaptAI/openadapt-ml/commit/cd969f87a66379bad37c14e15fdd39386eb8613b))

Includes VL model support fix (PR #18): - Auto-detect VL models and use correct model class - Handle
  Qwen2VLForConditionalGeneration properly - Set assistant_only_loss=False for VL models

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Remove dead code and legacy fix scripts
  ([#16](https://github.com/OpenAdaptAI/openadapt-ml/pull/16),
  [`6d808ea`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6d808ea481a327027e319112359ece07fc5012b0))

Delete unused files: - training/viewer_migration_example.py (72 lines) - only self-referential -
  scripts/fix_acr_auth.py (212 lines) - one-time fix now baked into setup_azure.py -
  docs/azure_acr_authentication.md - docs for removed script

Update CLAUDE.md to remove references to deleted fix script.

Verified safe to delete: - None of these files are imported by cli.py - fix_acr_auth.py
  functionality is now in setup_azure.py (steps 10-12)

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>

- Update gitignore and module exports
  ([`81f19d7`](https://github.com/OpenAdaptAI/openadapt-ml/commit/81f19d70af9bf52df566ebe60e9537f4f960404a))

- Add patterns for training output, synthetic data, experiment results - Add .jsonl,
  benchmark_live.json, external/, demos/ to gitignore - Export new runtime and schema types in
  module __init__.py

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Documentation

- Add architecture decisions, analysis, and design documentation
  ([`178e6e5`](https://github.com/OpenAdaptAI/openadapt-ml/commit/178e6e534674e281a2ab4590e7fd69b0813cdc13))

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
  ([`897dedd`](https://github.com/OpenAdaptAI/openadapt-ml/commit/897dedd46cef94a2e8500850337e7187bdc61bf3))

- github_org_update_plan.md: GitHub org profile update strategy - desktop_app_plan.md: Desktop app
  distribution (pywebview + PyInstaller) - openadapt_integration_plan.md: Core openadapt integration
  roadmap

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add GitHub organization profile content recommendations
  ([`fac58bb`](https://github.com/OpenAdaptAI/openadapt-ml/commit/fac58bb0ab0c6f2b463d86d5139bbbd2f90d84ce))

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
  ([`a2e81a0`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a2e81a076db53ac9e83536dcac5f8cf89937c28f))

Add comprehensive documentation for Qwen3-VL vision-language embedding: -
  qwen3_vl_embedding_research.md: Literature review of VLM embedding extraction methods, including
  early exit strategies, hidden state extraction, and multimodal representation learning -
  qwen3_vl_embedding_design.md: Technical design document for extracting and using Qwen3-VL
  embeddings for GUI element retrieval and similarity-based action prediction

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add viewer architecture survey and comparison
  ([`894d03d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/894d03da6f86906a10fe8e49121a4818639a3a28))

Survey of viewer technologies and frameworks for training/benchmark visualization, comparing options
  like Gradio, Streamlit, Panel, and custom HTML solutions for the unified viewer architecture.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add website redesign plan
  ([`1a25483`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1a254837a92bf4ff892a305dc2648678ab46f33c))

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Pivot desktop app to uv-first distribution and propose meta-package architecture
  ([`21a05ac`](https://github.com/OpenAdaptAI/openadapt-ml/commit/21a05acf83347b7fed2496ad17009df1e0a0f2c1))

- desktop_app_plan.md: Switch from PyInstaller to uv-based installation - Tier 1: Single command
  install via uv tool - Tier 2: Optional uv bundled installer (~15MB) - Tier 3: PyInstaller full
  bundle (deferred) - Reduces annual cost from $500-700 to $0

- new_openadapt_architecture.md: Propose Option B+ thin CLI wrapper - Create unified 'openadapt'
  meta-package - Re-export common items from sub-packages - Unified CLI (openadapt
  capture/train/eval) - Phase-based implementation over 2 weeks

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Update openadapt-web repository reference to new name
  ([`ca60945`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ca6094581801c9aab3bd0b8b86a6dfe7e3c3448e))

Update repository link from OpenAdapt.web to openadapt-web following the rename to match the
  lowercase-hyphen naming convention.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- Add safety gate, perception integration, and representation experiments
  ([`5ef0ef8`](https://github.com/OpenAdaptAI/openadapt-ml/commit/5ef0ef8455e97441b963f3ccff3c94fc06c2c63c))

New modules: - runtime/safety_gate.py: Deterministic safety gate for action validation -
  perception/integration.py: Bridge between openadapt-grounding and openadapt-ml -
  experiments/representation_shootout/: Coords vs Marks ablation framework -
  benchmarks/trace_export.py: Export benchmark traces to various formats

Tests: - Reorganize tests from root to tests/ directory - Add integration tests in
  tests/integration/ - Add test_gemini_grounding_imports.py for grounding module

Scripts: - p1_episode_success_ab_test.py: A/B test for demo-conditioned episode success

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add unified baseline adapters for VLM comparison
  ([`9c323db`](https://github.com/OpenAdaptAI/openadapt-ml/commit/9c323db0fbb17beb33b975a2120eece30e014e79))

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
  ([`9951c40`](https://github.com/OpenAdaptAI/openadapt-ml/commit/9951c4078edcd1a409a38d82c9667785bf5df3f1))

Add experiment results and artifacts: - representation_shootout results comparing embedding
  extraction methods - qwen_login 2b_dev_fixed plots showing base vs fine-tuned comparison -
  registration_som_eval.json evaluation metrics for SOM-based action prediction

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **waa**: Refactor CLI and fix Python 3.9 compatibility
  ([#14](https://github.com/OpenAdaptAI/openadapt-ml/pull/14),
  [`5557130`](https://github.com/OpenAdaptAI/openadapt-ml/commit/5557130255cc23fa07841ef89520e35dd14f4464))

- Refactor CLI from 6800 to ~1300 lines with flat command structure - Add analyze command to parse
  and summarize benchmark results - Add --num-tasks flag to limit number of tasks to run - Fix
  Python 3.9 compatibility by copying Python from vanilla WAA image (fixes transformers 4.46.2
  compatibility with GroundingDINO) - Add coverage and analysis artifacts to .gitignore

Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>

### Refactoring

- **benchmarks**: Consolidate to re-export from openadapt-evals
  ([#17](https://github.com/OpenAdaptAI/openadapt-ml/pull/17),
  [`7f171e4`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7f171e41abfc6a3d6065ec2abb11f73e34b11abe))

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
  ([`2e81378`](https://github.com/OpenAdaptAI/openadapt-ml/commit/2e8137830d5e42abfbd2847973684ee9313284ae))

BREAKING CHANGE: Benchmark code moved to openadapt-evals.

- Update CLAUDE.md with migration guide - Add deprecation warning to benchmarks/__init__.py - Old
  imports still work but emit DeprecationWarning

Migration: # OLD (deprecated) from openadapt_ml.benchmarks import WAAMockAdapter

# NEW (preferred) from openadapt_evals import WAAMockAdapter

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Testing

- Add unit tests for providers and baselines modules
  ([`4942982`](https://github.com/OpenAdaptAI/openadapt-ml/commit/49429822ad5abc57eec40971eaf13558087255bf))

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.2.0 (2026-01-09)

### Bug Fixes

- Resolve test failures and SSE dashboard state conflicts
  ([`76fa63c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/76fa63c42c31991016eca9559811cfe11b487ccd))

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
  ([`03b23fc`](https://github.com/OpenAdaptAI/openadapt-ml/commit/03b23fcd08b58dab58d760f60f95f5af568346ae))

Probe output now correctly shows localhost:8006 instead of public IP which is not accessible without
  SSH tunnel.

- **waa**: Add full Python dependencies for benchmark client
  ([`a2cd826`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a2cd8262b2f1e6a4e582ef2f293a4e5878887f78))

- Add build-essential, ffmpeg, and X11 libs for package compilation - Install core packages:
  gymnasium, fabric, transformers, torch (CPU) - Install ML packages: opencv, easyocr, matplotlib,
  accelerate - Create python -> python3 symlink for compatibility - Separate pip installs into
  layers for better caching

- **waa**: Add missing pydrive and other client dependencies
  ([`ebdc4f6`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ebdc4f6a6d266d392a1a981b7b5101147919680d))

- **waa**: Add remaining WAA client dependencies (openpyxl, docx, etc.)
  ([`02e5e2f`](https://github.com/OpenAdaptAI/openadapt-ml/commit/02e5e2f40bec8fd5d880048af0bbfd37c66f612d))

- **waa**: Copy OEM files to Samba share at container startup
  ([`4fb26fe`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4fb26fe2162996b28c2c3e1ce2b54f27eed74230))

Add /copy-oem.sh startup script that copies OEM files from /oem to /tmp/smb (Samba share) at
  container startup. This fixes Windows not finding setup scripts because smb.conf is generated at
  runtime.

Also update experiment doc to remove timeline estimates and add WAA baseline as in-progress.

- **waa**: Copy Python env from official image to avoid 3.13 compat issues
  ([`e5b3dc0`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e5b3dc04997ce1b2a6a86bf69ee09c30aaa3b3b2))

### Chores

- Bump version to 0.2.0 for PyPI release
  ([`3fac13d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/3fac13de1f593676cf3aba896799bca4965db2c2))

Features in this release: - TRL + Unsloth training integration (2x faster, 50% less VRAM) -
  Standardized on uv for package management - Enhanced VM CLI and WAA deployment - Comprehensive
  documentation updates

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Remove old waa/Dockerfile (moved to waa_deploy/)
  ([`e3bd4e3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e3bd4e3f0ffcaaeaacd02ba2ad6c36f8d359d000))

- Standardize on uv for package management
  ([`7d5c2fe`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7d5c2feaeff9a0f20e136ffae6f7499f0b775379))

- Replace all `pip install` with `uv add` in docs - Update cloud GPU training to use `curl ... | sh`
  for uv install - Update CLAUDE.md with enhanced VM operations guidance - Consistent `uv sync` for
  local development

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Update CLAUDE.md and minor fixes
  ([`8dbe154`](https://github.com/OpenAdaptAI/openadapt-ml/commit/8dbe15434b8474552e57e82ca657824b9f1c6efe))

- Updated CLAUDE.md with new features and documentation - trainer.py: minor improvements -
  eval_policy.py: updated for new schema - uv.lock: dependency updates

- Update uv.lock
  ([`6b82505`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6b82505a453f8b972235b82c8cbab689970ba4c0))

### Documentation

- Add benchmark viewer screenshot to README
  ([`e5c0516`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e5c0516171f79ec7504e19329e0eb355d456a721))

- Add capture format decision framework
  ([`c60088f`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c60088f147d966bd7ffe27fb569b6c804aa0460d))

Explores options for data format interoperability: - Option A: Native Episode output from
  openadapt-capture - Option B: Conversion layer in openadapt-ml (recommended) - Option C: Shared
  schema package - Option D: Dual output

Recommends Option B with clear guidelines for the conversion layer. Includes text demo format
  specification for WAA experiment.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add comprehensive capture migration guide with a11y integration
  ([`9307ef6`](https://github.com/OpenAdaptAI/openadapt-ml/commit/9307ef690437c28334d2442a32d15d61253a4d80))

- Add design documents for SSE, retrieval, and parallelization
  ([`fca9c94`](https://github.com/OpenAdaptAI/openadapt-ml/commit/fca9c945836a3a45973ba07de32019fe3dbdc2a2))

- SSE architecture and integration guides - Demo retrieval design and experiments - WAA
  parallelization and live adapter plans - Chrome extension design for capture - Benchmark viewer UX
  improvements

- Add openadapt-capture to openadapt-ml migration plan
  ([`f3d6194`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f3d6194a06844c5c9c4b53ccc72c40010494b9a0))

- Add schema consolidation plan
  ([`b24ea42`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b24ea4260264ccc115147a23e7b68472cd138b9b))

Detailed migration plan for consolidating from two schema modules to one: - DELETE:
  openadapt_ml/schemas/ (dataclass-based, legacy) - KEEP: openadapt_ml/schema/ (Pydantic-based,
  canonical)

Includes: - Dependency analysis (22 files affected) - Field mapping between old and new - 7-phase
  migration strategy - Testing strategy - Rollback plan - Timeline estimate (~8-10 hours)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add staged export hierarchy to enterprise guide
  ([`e6b985b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e6b985b29f21dfe31a01851b941d6220cd4fedea))

- Episode JSON as canonical, Parquet/WebDataset as projections - Expand data loss table for flat
  formats - Mark exporters as Planned with design doc links - Add multi-step evaluation caveat

- Add WAA demo recording guide for Windows captures
  ([`213ab7b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/213ab7bb7b89294501049b7f71c31b584cdfedc0))

Step-by-step instructions for recording the 3 complex demos: - Task #4: Fill blank cells
  (LibreOffice Calc) - Task #5: Create chart (LibreOffice Calc) - Task #9: Archive folder (File
  Explorer)

Includes setup, recording steps, export, and transfer instructions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Consolidate WAA CLI workflow documentation
  ([`263ca42`](https://github.com/OpenAdaptAI/openadapt-ml/commit/263ca42e97491a4564f98cca2e9862e0051440a5))

- Add Quick Start: Single VM section to azure_waa_setup.md - Replace manual SSH steps in CLAUDE.md
  with CLI commands - Document custom waa-auto Docker image that fixes OEM folder issue - Add vm
  probe, vm reset-windows, and other useful commands

- Strengthen enterprise integration guide positioning
  ([`e6ea3ba`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e6ea3ba1001a3a3707c95a47bcc81c6b2baad0b4))

Add decision boundary, requirements, retrofitting cost sections. Add data portability note
  addressing vendor lock-in concern. Add optional metadata extension pattern. Add typical
  integration workflow. Add open schema independence statement.

- Update README for TRL training and PyPI installation
  ([`a8a6055`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a8a60559c020ccc127bb6375b9de4f95e7e87330))

- Add Installation section with PyPI instructions (uv add openadapt-ml) - Update training section to
  reflect TRL + Unsloth integration - Update repository structure with trl_trainer.py reference -
  Add PyPI badge - Fix section numbering throughout - Update test descriptions for TRL trainer

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Validate demo-conditioning at n=45 (46.7% → 100%)
  ([`49d5a20`](https://github.com/OpenAdaptAI/openadapt-ml/commit/49d5a206bc710a430155e5339107543fa60d9b7d))

- Zero-shot: 46.7% (21/45), Demo: 100% (45/45), Control: 57.8% - Improvement: +53.3 percentage
  points across 15 macOS categories - Add Parquet export design doc (derived analytics format) -
  Update enterprise guide with validated result

- **experiment**: Strengthen demo-conditioning doc for expert review
  ([`d49af59`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d49af596ea96fe058640a2d86c94e8a53af231d3))

- Add interpretation note framing result as "trajectory-conditioned disambiguation" not general
  task-solving - Highlight length-matched control in executive summary - Frame shared first action
  as intentional controlled variable - Add "Positioning Relative to Fine-Tuning" section connecting
  to prompting-first methodology - Expand limitations with actionable specifics (WAA running, SoM
  conventions, episode success vs first-action)

- **schema**: Comprehensive documentation for Episode schema
  ([`b20b879`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b20b8794fc142a619d485022b62c2da56f64450a))

- Add detailed Quick Start with code examples - Document all 24 action types with categories -
  Explain pixel vs normalized coordinate systems - Add validation and format conversion examples -
  Document extension points (raw, metadata fields) - Add docs/schema/README.md as standalone
  reference

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- Add demo-conditioned prompting experiment and retrieval module
  ([`8745752`](https://github.com/OpenAdaptAI/openadapt-ml/commit/874575278f9886afb1acfc1be5508228438193a6))

Validated that demo-conditioning improves first-action accuracy from 33% to 100% (n=3, preliminary
  signal). Key findings: - Benefit is semantic, not token-length (length-matched control: 67%) -
  Demos generalize across task variations (toggle polarity, parameters) - Zero-shot has systematic
  spatial bias that demos correct

Added retrieval module (TF-IDF + domain bonus) for automatic demo selection. Added demo-conditioned
  training mode to train_from_json.py. Added enterprise integration guide for workflow data export.

Statistical note: n=3 is insufficient for significance. Validation at n≥30 on expanded task set in
  progress.

- Add Episode JSON schema and polish benchmark viewer
  ([`c18ae67`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c18ae67eecccee331704fe04abde4b805f92f87a))

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
  ([`c54b261`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c54b261107600725a5ddcf6f48f73d9ee5729862))

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
  ([`1689ab4`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1689ab41a34991099b3313d1f0882cfe4dec172e))

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
  ([`99e34b2`](https://github.com/OpenAdaptAI/openadapt-ml/commit/99e34b2fb0cb98933683da4c3d7e845a644987aa))

- Add trl_trainer.py with SFTTrainer + Unsloth optimizations - Update train_from_json.py to use TRL
  trainer (2x faster, 50% less VRAM) - Remove legacy custom training loop from trainer.py - Add
  [training] optional dependencies (trl, datasets) - Support --use-unsloth / --no-unsloth flags

Training command: uv run python examples/train_from_json.py --data episodes/ --output results/

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **benchmark**: Add Run Benchmark UI panel and fix VNC/Docker issues
  ([`8316bda`](https://github.com/OpenAdaptAI/openadapt-ml/commit/8316bdacbdc4ff7a3eca727de3a26921fffb0365))

- Add Run Benchmark panel to benchmark viewer (model, tasks, agent, domain selection) - Add POST
  /api/benchmark/start endpoint to launch benchmarks from UI - Add --domain and --task-ids CLI flags
  for filtered benchmark runs - Fix VNC link to use localhost:8006 (SSH tunnel) instead of direct VM
  IP - Fix Docker build to use --no-cache --pull to prevent stale dockurr/windows layers - Add
  docs/waa_network_architecture.md explaining the localhost-based network topology - Add
  docs/benchmark_run_ui_design.md with UI design specification

The Docker cache issue caused dockurr/windows v0.00 scripts (no auto-download) to be used instead of
  v5.14 (with auto-download). Fixed by adding --no-cache --pull.

- **benchmarks**: Waa CLI improvements, result analysis, and viewer enhancements
  ([`8483233`](https://github.com/OpenAdaptAI/openadapt-ml/commit/84832335f1cc75203300245f28c0d3fdc49db9c7))

WAA CLI: - Add `analyze` command for programmatic WAA result analysis - Remote analysis via SSH
  (--vm-ip --remote) - Local directory analysis (--results-dir) - Per-domain success rates, JSON
  export - Fix invalid model name: gpt-5.2 → gpt-4o - Add --skip-build tip for faster reruns - Add
  vm_monitor.py for VM status tracking - Add live_tracker.py for real-time benchmark progress

Documentation: - Add docs/waa_setup.md - WAA setup guide - Add docs/GEMINI_GROUNDING_QUICKSTART.md -
  Add docs/background_task_visibility.md - Add implementation summaries

Viewer: - Add benchmark_viewer.py for WAA result visualization - Enhance local.py serve command -
  Integrate benchmarks into unified viewer

- **export**: Add Parquet exporter and toolbox positioning
  ([`e0f93f2`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e0f93f2bd567916ab805b5395378446c5dae25dc))

Add first-class Parquet export support: - to_parquet() / from_parquet() for Episode serialization -
  CLI: python -m openadapt_ml.export parquet --input <dir> --output <path> - Optional summary table
  generation - pyarrow as optional dependency

Update enterprise integration guide: - Add "What is openadapt-ml?" toolbox section - Frame as
  composable utilities, not monolithic framework - Update Parquet section with real implementation
  examples - Scope canonical claim to within openadapt-ml

- **retrieval**: Add demo retrieval system and WAA live adapter
  ([`21d48cf`](https://github.com/OpenAdaptAI/openadapt-ml/commit/21d48cf824a837f03d692c0ed448430d19ad77ae))

Demo Retrieval: - embeddings.py: improved embedding generation with caching - demo_retriever.py:
  semantic search for relevant demonstrations - Support for goal-based and screenshot-based
  retrieval

Benchmark Viewer: - viewer.py: standalone HTML viewer for benchmark results - waa_live.py: live
  evaluation adapter for WAA benchmarks - Integrated with dashboard for real-time monitoring

- **schema**: Add converters between internal and external formats
  ([`f992029`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f992029b4e296f404772a588cad7da56e7109bba))

- from_internal_episode(): Convert schemas.sessions.Episode to schema.Episode -
  to_internal_episode(): Convert schema.Episode back to internal dict format - Document field
  mapping in README

This enables bidirectional conversion between: - Internal training format (schemas.sessions): id,
  goal, t, image_path, x/y - External interop format (schema.episode): episode_id, instruction,
  step_index, etc.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **schema**: Add select_monitor and normalized coordinates
  ([`e398412`](https://github.com/OpenAdaptAI/openadapt-ml/commit/e398412b2ec7e44d8adf71e15170568836afeab8))

- Add action types: select_monitor, window_focus, window_resize, window_move - Add monitor_id field
  for select_monitor action - Add window_title field for window_focus action - Add
  normalized_coordinates (0.0-1.0) as alternative to pixel coords - Add normalized_start/end for
  resolution-independent drag actions

This enables cu-episode-v1 alignment without loss of information.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **waa**: Auto-build Docker image and fix tunnel detection
  ([`0a04998`](https://github.com/OpenAdaptAI/openadapt-ml/commit/0a0499802371781d1a9f4cd18f7ab623676bc07d))

- CLI run-waa now automatically builds waa-auto image if missing - Added --rebuild flag to force
  image rebuild - Dockerfile: fixed IP patching, added playwright for web automation -
  ssh_tunnel.py: fixed tunnel status to check actual port state instead of just internal tracking,
  correctly reports external tunnels

Closes #XX

### Refactoring

- Consolidate schema to single Pydantic-based Episode module
  ([`8c3d7c9`](https://github.com/OpenAdaptAI/openadapt-ml/commit/8c3d7c97de82263e05873ee7da8a5e1abe303d70))

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
  ([`b5a5d02`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b5a5d02e4b2fb6b3107ee848df5dead339281577))

28 tests covering: - Task definitions (10 tasks, domains, difficulties) - Demo content (7 complete,
  3 placeholder) - Integration (task/demo consistency, retrieval) - Format validation (DEMONSTRATION
  header, Goal line)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add tests for demo retrieval and WAA live adapter
  ([`93565d0`](https://github.com/OpenAdaptAI/openadapt-ml/commit/93565d0cab4263ddbafba66439fcd4c190f39921))

- test_demo_retrieval.py: comprehensive tests for demo retriever - test_waa_live.py: tests for live
  WAA adapter - Updated test_retrieval.py for new embedding features - demo_retrieval_example.py:
  usage examples

- Update tests for TRL trainer refactor
  ([`b17e22d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/b17e22d75ee93fb357049aabb585547e374d721b))


## v0.1.0 (2025-12-16)

### Bug Fixes

- Consistent 'Viewer' label in nav tabs
  ([`5e2f847`](https://github.com/OpenAdaptAI/openadapt-ml/commit/5e2f8470c3b014e1ba7cef34fc2f7978e3ee4387))

- **dashboard**: Improve viewer/dashboard consistency and CLI commands
  ([`a23a881`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a23a88194fbcbd18de5330d28b3de9da8581d597))

- **dashboard**: Remove duplicate job ID and make header full-width
  ([`15260ec`](https://github.com/OpenAdaptAI/openadapt-ml/commit/15260ecc0885c92c66ea96170bac055045f98222))

- **dashboard**: Use subprocess for simpler http server startup
  ([`0f9c9f9`](https://github.com/OpenAdaptAI/openadapt-ml/commit/0f9c9f9bf7e147d6cd1a532506b5083b9c1024e2))

- **plots**: Add model size labels to hardened benchmark plots
  ([`4c828f2`](https://github.com/OpenAdaptAI/openadapt-ml/commit/4c828f20066533f1b0f7b74ac35ce20c46deedec))

- **serve**: Remove stale refresh args, stop button now works
  ([`c190ea5`](https://github.com/OpenAdaptAI/openadapt-ml/commit/c190ea567ac1d8ee7a950d2dc55f8c88288beab4))

- **stub**: Auto-copy real screenshot for evaluation samples
  ([`700f56a`](https://github.com/OpenAdaptAI/openadapt-ml/commit/700f56a60556bd50547cb017588776cb9f2b3139))

- **viewer**: Extract predictions from window.comparisonData and fix elapsed time loading
  ([`afb5bb8`](https://github.com/OpenAdaptAI/openadapt-ml/commit/afb5bb8128d0a48fe956ac54ef339e9eee6aa2e9))

- **viewer**: Sync audio speed with playback, add visual feedback to overlay toggles
  ([`bfafcba`](https://github.com/OpenAdaptAI/openadapt-ml/commit/bfafcba05188a354c3c0cff0c9ab9b9dbfed0505))

### Chores

- **gitignore**: Ignore synthetic and ephemeral training artifacts
  ([`ff3a6e3`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ff3a6e30e406fd534b90c6f07380fd52f09d9933))

- **plots**: Track hardened v2 experiment plots and scope ignore to top-level
  ([`1511e6d`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1511e6d0f69731a2e80ae262b12e53b7ff2d5105))

- **readme**: Point synthetic plots at hardened v2 experiment artifacts
  ([`6f35079`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6f35079a9828b8fdbd267aa83a2d0113950788b9))

### Documentation

- Add benchmark viewer integration design and update TODOs
  ([`f62ba9c`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f62ba9c1ef302bec660a728f73f5c7f1700965f0))

- Add early termination controls as high priority TODO
  ([`454d137`](https://github.com/OpenAdaptAI/openadapt-ml/commit/454d137aa7a15cc695f99b019b70114b4a94903a))

- Document need for auto-termination, dashboard stop button, checkpoint download - Fix shared header
  in unified viewer template (trainer.py) - Remove 'Dashboards:' label from compare.py nav

- Add GUI-Actor integration plan for coordinate-free grounding
  ([`964d92a`](https://github.com/OpenAdaptAI/openadapt-ml/commit/964d92acc2069eb0e6b14c409cb5ab1ac085cfa2))

- Add training feedback UX critical path analysis
  ([`8c8edef`](https://github.com/OpenAdaptAI/openadapt-ml/commit/8c8edef14e2ca37d38a106c47af23bd794d003fc))

- Add unified compute architecture design and PyPI TODO
  ([`db9e008`](https://github.com/OpenAdaptAI/openadapt-ml/commit/db9e008ef8d3b3912c688d5b3312912815274406))

- **readme**: Add 2b training log snippet and clarify qwen3 masking roadmap
  ([`f02ebfd`](https://github.com/OpenAdaptAI/openadapt-ml/commit/f02ebfd557dd01a520ee3f73608da6cc7f0c0039))

- **roadmap**: Mark Priority 5a complete and update plotting achievements
  ([`2da03f2`](https://github.com/OpenAdaptAI/openadapt-ml/commit/2da03f2ee9a1f7780e336323f036b21d293776ca))

- **viewer**: Add timeline visualizer and eval integration design
  ([`27c4130`](https://github.com/OpenAdaptAI/openadapt-ml/commit/27c4130b8753f4fe2817feb4b77c72b3a4a93b4a))

### Features

- Initial commit of openadapt-ml pipeline (synthetic login, qwen adapters, eval + training)
  ([`ec92d6b`](https://github.com/OpenAdaptAI/openadapt-ml/commit/ec92d6b250dc6e4c184e9b2ff07c12781979ba1d))

- V0.1.0 release with benchmark integration, grounding module, and cloud training
  ([`7887890`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7887890a6903aa3f7edb32fcf2e41f53b252a537))

- **benchmark**: Add qwen login orchestrator and refine docs
  ([`efcce00`](https://github.com/OpenAdaptAI/openadapt-ml/commit/efcce008ecf4839857cae154b94b29499d959809))

- **cloud**: Add Lambda Labs training, benchmarks, and training visualization
  ([`7eece80`](https://github.com/OpenAdaptAI/openadapt-ml/commit/7eece80e9974b0b6c5414db9cb597408487cd596))

- **config**: Add pydantic-settings configuration and API benchmarks
  ([`2e7bfd1`](https://github.com/OpenAdaptAI/openadapt-ml/commit/2e7bfd1923b15223e7bf59181244f465cd300d7a))

- **dashboard**: Add early termination controls and /api/stop endpoint
  ([`d6f5df4`](https://github.com/OpenAdaptAI/openadapt-ml/commit/d6f5df4704d45301a7ee9820ea5ba17c538b3c7b))

- **dashboard**: Enhance evaluation samples with model thinking display
  ([`340fe36`](https://github.com/OpenAdaptAI/openadapt-ml/commit/340fe36b6f715aca5480e77709801b2c029541e3))

- **dashboard**: Show model thinking by default, add legend
  ([`5828d67`](https://github.com/OpenAdaptAI/openadapt-ml/commit/5828d6764891261d688fd741dcbbbecc4250369d))

- **docs**: Add dashboard screenshots and fix viewer predictions
  ([`5b729a9`](https://github.com/OpenAdaptAI/openadapt-ml/commit/5b729a9c1f01d304260f266081900f4cae04c8a7))

- **lambda**: Add early termination controls with auto-stop, checkpoint download, and dashboard stop
  button
  ([`1157af8`](https://github.com/OpenAdaptAI/openadapt-ml/commit/1157af89f00041fc0bafe8e468f9edea9e626dc6))

- **lambda**: Auto-symlink capture screenshots and rewrite paths
  ([`40af8db`](https://github.com/OpenAdaptAI/openadapt-ml/commit/40af8dbfb424de02dd7cb3fe18bd0f023853083c))

- **local**: Add local training CLI for CUDA/Apple Silicon
  ([`6edee26`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6edee26b1bf251cac3f61d866867b6e1b839f9f5))

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
  ([`73fb601`](https://github.com/OpenAdaptAI/openadapt-ml/commit/73fb6011ead9690807db497c128186d2c7b36794))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **training**: Add stub adapter for rapid UI testing without GPU
  ([`79a95a9`](https://github.com/OpenAdaptAI/openadapt-ml/commit/79a95a91efa0fca926991598f51a8755cb7ebbde))

- **viewer**: Add benchmark tab with WAA integration WIP state
  ([`a31dbf9`](https://github.com/OpenAdaptAI/openadapt-ml/commit/a31dbf98e4cd48faa28ef41ec67e2a0a77640d3a))

- **viewer**: Add parseModelOutput for SoM parsing and truncation
  ([`3396594`](https://github.com/OpenAdaptAI/openadapt-ml/commit/3396594aaac19e9b0ff9e1b79d5eb2070b790721))

- **viewer**: Add screenshots to README and smart auto-scroll
  ([`3a5a256`](https://github.com/OpenAdaptAI/openadapt-ml/commit/3a5a2568888b493ccaab12e3d00242336df650e4))

- **viewer**: Add SoM action parsing for model predictions
  ([`2dd29fb`](https://github.com/OpenAdaptAI/openadapt-ml/commit/2dd29fbe0ac33a1971c1507610372819b29df428))

- **viewer**: Add transcript/audio sync, copy-all button, and extract shared UI
  ([`bd54110`](https://github.com/OpenAdaptAI/openadapt-ml/commit/bd54110b457558ecb4f3a50114cc194d338fede7))

- **viewer**: Extract viewer module with evaluation gallery and badges
  ([`fc003b0`](https://github.com/OpenAdaptAI/openadapt-ml/commit/fc003b06179a33a140840881159a9912f112a7eb))

### Refactoring

- Rename --eval-on-training-data to --overfit
  ([`19b1c94`](https://github.com/OpenAdaptAI/openadapt-ml/commit/19b1c9447031e654ec8bbc68ba0fe0d10785db57))

- **viewer**: Consolidate to standalone HTML generation
  ([`6126aeb`](https://github.com/OpenAdaptAI/openadapt-ml/commit/6126aeb864b4b4529f9df34e1e93758fd8c73475))

### Testing

- **local**: Add tests for local CLI with early stopping
  ([`51f7f32`](https://github.com/OpenAdaptAI/openadapt-ml/commit/51f7f326b4a986467b108971098570498e361a17))
