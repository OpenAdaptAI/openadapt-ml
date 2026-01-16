# Codebase Cleanup Notes

**Date**: January 16, 2026
**Purpose**: Document cleanup actions to prepare codebase for production-ready commits

---

## 1. Overview

This document tracks the cleanup actions performed to organize the codebase, improve `.gitignore`, and prepare files for production commits.

---

## 2. .gitignore Updates

### Patterns Added

The following patterns have been added to `.gitignore` to exclude ephemeral/generated content:

```
# Training output directories
training_output*/

# Synthetic data directories
synthetic_*/

# JSONL data files (except specific needed ones)
*.jsonl

# Live benchmark tracking
benchmark_live.json

# Experiment results directories
p0_results/
p1_results/

# External dependencies (cloned repos, vendored code)
external/

# Demo recordings
demos/

# Python cache
__pycache__/
*.pyc

# Pytest cache
.pytest_cache/
```

### Rationale

- `training_output*/` - Generated during training runs, contains checkpoints and logs
- `synthetic_*/` - Generated synthetic UI data for testing/training
- `*.jsonl` - Data files generated during experiments (large, reproducible)
- `benchmark_live.json` - Real-time benchmark tracking data
- `p0_results/`, `p1_results/` - Experiment result directories
- `external/` - External cloned repositories (e.g., pc-agent-e)
- `demos/` - Demo recording files
- `__pycache__/`, `*.pyc` - Python bytecode cache
- `.pytest_cache/` - Pytest cache directory

---

## 3. Test File Reorganization

### Files Moved from Root to tests/

| Original Location | New Location |
|-------------------|--------------|
| `test_batching.py` | `tests/test_batching.py` |
| `test_mock_labeling.py` | `tests/test_mock_labeling.py` |
| `test_negative_control.py` | `tests/test_negative_control.py` |
| `test_terminal_output.py` | `tests/test_terminal_output.py` |

### Files Moved to tests/integration/

| Original Location | New Location |
|-------------------|--------------|
| `test_benchmark_viewer.py` | `tests/integration/test_benchmark_viewer.py` |
| `test_data_collection.py` | `tests/integration/test_data_collection.py` |
| `test_live_eval.py` | `tests/integration/test_live_eval.py` |
| `test_sse_endpoint.py` | `tests/integration/test_sse_endpoint.py` |

### Rationale

- Unit tests (fast, isolated) remain in `tests/`
- Integration tests (require external systems/long-running) go to `tests/integration/`
- Keeps root directory clean

---

## 4. Files Ready to Commit

### New Features (Recommended for Commit)

| File | Description | Status |
|------|-------------|--------|
| `openadapt_ml/benchmarks/trace_export.py` | Benchmark trace export functionality | New feature, untracked |
| `openadapt_ml/benchmarks/live_tracker.py` | Live benchmark tracking | New feature (already tracked but may need commit) |

### New Tests

| File | Description | Status |
|------|-------------|--------|
| `tests/test_gemini_grounding_imports.py` | Tests for Gemini grounding module imports | New test, untracked |

### Experiment Scripts

| File | Description | Status |
|------|-------------|--------|
| `scripts/p1_episode_success_ab_test.py` | A/B test for episode success rates | Experiment script, untracked |

### Documentation

| File | Description | Status |
|------|-------------|--------|
| `docs/analysis_jan2026.md` | January 2026 analysis results | Untracked |
| `docs/trl_unsloth_integration_analysis.md` | TRL/Unsloth integration analysis | Untracked |
| `docs/grpo_training_report.md` | GRPO training experiment report | Untracked |

---

## 5. Other Uncommitted Production-Ready Files

### Benchmarks Module

| File | Description | Notes |
|------|-------------|-------|
| `openadapt_ml/benchmarks/trace_export.py` | Export benchmark traces to various formats | Production-ready, well-documented |
| `openadapt_ml/benchmarks/live_tracker.py` | Real-time benchmark progress tracking | Production-ready |

### Runtime Module - New Features

| File | Description | Notes |
|------|-------------|-------|
| `openadapt_ml/runtime/safety_gate.py` | Safety gate for action validation before execution | Production-ready, well-documented |

### Perception Module (New)

| File | Description | Notes |
|------|-------------|-------|
| `openadapt_ml/perception/__init__.py` | Perception module init | New module |
| `openadapt_ml/perception/integration.py` | Perception integration | New feature |

### Experiments Module - New Features

| File | Description | Notes |
|------|-------------|-------|
| `openadapt_ml/experiments/representation_shootout/__init__.py` | Representation comparison experiments | New module |
| `openadapt_ml/experiments/representation_shootout/config.py` | Configuration for representation experiments | New feature |

### Documentation

| File | Description | Notes |
|------|-------------|-------|
| `docs/ARCHITECTURE_DECISIONS.md` | Architecture decision records | Production-ready documentation |
| `docs/viewer_consolidation_design.md` | Viewer consolidation design doc | Design documentation |
| `docs/viewer_redesign_proposal.md` | Viewer redesign proposal | Design documentation |
| `docs/live_benchmark_monitoring_fix.md` | Live benchmark monitoring fix notes | Technical documentation |
| `docs/perception_integration.md` | Perception module integration design | Design documentation |
| `docs/safety_gate_design.md` | Safety gate design documentation | Design documentation |
| `docs/experiments/representation_shootout_design.md` | Representation experiment design | Experiment documentation |
| `docs/experiments/waa_benchmark_results_jan2026.md` | WAA benchmark results | Results documentation |

### Modified Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Project configuration (modified) |
| `uv.lock` | Dependency lock file (modified) |

---

## 6. Files NOT Recommended for Commit

### Ephemeral/Generated

- `benchmark_live.json` - Real-time tracking data
- `training_output*/` - Training artifacts
- `synthetic_*/` - Generated synthetic data
- `p0_results/`, `p1_results/` - Experiment results
- `experiments/qwen_login/` subdirectories - Experiment outputs

### HTML Test Files

- `test_cost_dashboard.html` - Local test artifact
- `test_local_dashboard.html` - Local test artifact

### Summary Documents (Consider for Commit)

- `LIVE_EVAL_SUMMARY.md` - May be useful for documentation
- `SSE_IMPLEMENTATION_SUMMARY.md` - May be useful for documentation

---

## 7. Recommended Commit Strategy

### Commit 1: .gitignore Updates
```bash
git add .gitignore
git commit -m "chore: update .gitignore with training, synthetic, and cache patterns"
```

### Commit 2: Test File Reorganization
```bash
git add tests/
git rm test_batching.py test_benchmark_viewer.py test_data_collection.py \
       test_live_eval.py test_mock_labeling.py test_negative_control.py \
       test_sse_endpoint.py test_terminal_output.py
git commit -m "refactor: move test files from root to tests/ directory"
```

### Commit 3: New Features
```bash
git add openadapt_ml/benchmarks/trace_export.py
git add openadapt_ml/benchmarks/live_tracker.py
git commit -m "feat: add benchmark trace export and live tracking"
```

### Commit 4: Documentation
```bash
git add docs/analysis_jan2026.md
git add docs/trl_unsloth_integration_analysis.md
git add docs/grpo_training_report.md
git add docs/ARCHITECTURE_DECISIONS.md
git commit -m "docs: add January 2026 analysis and architecture documentation"
```

### Commit 5: Experiment Scripts
```bash
git add scripts/p1_episode_success_ab_test.py
git commit -m "feat: add episode success A/B test experiment script"
```

### Commit 6: Safety Gate Module
```bash
git add openadapt_ml/runtime/safety_gate.py
git add docs/safety_gate_design.md
git commit -m "feat: add safety gate for action validation before execution"
```

### Commit 7: Perception Module
```bash
git add openadapt_ml/perception/
git add docs/perception_integration.md
git commit -m "feat: add perception module for visual understanding"
```

### Commit 8: Representation Experiments
```bash
git add openadapt_ml/experiments/representation_shootout/
git add docs/experiments/representation_shootout_design.md
git commit -m "feat: add representation comparison experiment framework"
```

### Commit 9: Additional Documentation
```bash
git add docs/viewer_consolidation_design.md
git add docs/viewer_redesign_proposal.md
git add docs/live_benchmark_monitoring_fix.md
git add docs/experiments/waa_benchmark_results_jan2026.md
git commit -m "docs: add viewer redesign and benchmark documentation"
```

---

## 8. Notes

- DO NOT commit files in `external/` - these are cloned external repositories
- DO NOT commit files in `demos/` - these are demo recordings with potentially sensitive data
- DO NOT commit `*.jsonl` files in experiment directories - large data files
- The `vendor/WindowsAgentArena` submodule has local modifications - review before committing

---

*Generated during codebase cleanup on January 16, 2026*
