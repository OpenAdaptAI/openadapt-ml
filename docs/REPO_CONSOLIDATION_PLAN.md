# Repository Consolidation Plan

## Overview

Clean up the existing **two-package architecture** by moving code to the right place:

```
openadapt-evals       # Foundation: benchmarks + infrastructure (standalone)
    └── MOVE HERE: VM management, waa_deploy/, session tracking
    └── Zero ML dependencies
    └── Supports multiple benchmarks (WAA, OSWorld, WebArena, etc.)

openadapt-ml          # Extension: ML training
    └── KEEP: training/, vlm/, baselines/, grounding/
    └── ADD DEPENDENCY: openadapt-evals
    └── DELETE: duplicate benchmark code
```

**What this consolidation does:**
1. Moves VM/benchmark infrastructure from openadapt-ml → openadapt-evals
2. Deletes ~1000 lines of duplicate code between repos
3. Establishes proper dependency: openadapt-ml depends on openadapt-evals
4. Cleans up ~1500 lines of dead code (server patches never deployed)

**Why `openadapt-evals` not `openadapt-waa`?**
- Avoids repo proliferation (no need for openadapt-osworld, openadapt-webarena, etc.)
- Single package supports all benchmarks with shared infrastructure
- Discoverability via README, PyPI keywords, GitHub topics instead of package name

---

## Part 0: Current State (as of Jan 2026)

### openadapt-ml Current Structure

```
openadapt_ml/
├── benchmarks/                    # VM + Benchmark code (mostly from PR #14)
│   ├── cli.py                     # ⭐ PR #14: VM lifecycle CLI (~1300 lines)
│   │                              #    Commands: create, delete, status, build,
│   │                              #    start, stop, probe, run, deallocate,
│   │                              #    logs, exec, docker-exec, vnc, tasks, etc.
│   ├── waa_deploy/                # ⭐ PR #14: Docker deployment
│   │   ├── Dockerfile             #    Custom WAA image build
│   │   ├── api_agent.py           #    Agent running inside container
│   │   ├── install.bat            #    Windows setup script
│   │   └── start_waa_server.bat   #    Server startup script
│   ├── vm_monitor.py              # VM status monitoring
│   ├── azure_ops_tracker.py       # Azure operation logging
│   ├── session_tracker.py         # Cost/time tracking
│   ├── disk_manager.py            # Disk space management
│   ├── dashboard.py               # Dashboard generation
│   ├── viewer.py                  # Benchmark results viewer
│   │
│   ├── # --- Duplicates (also in openadapt-evals) ---
│   ├── agent.py                   # → DELETE (use openadapt-evals)
│   ├── base.py                    # → DELETE (use openadapt-evals)
│   ├── runner.py                  # → DELETE (use openadapt-evals)
│   ├── waa.py                     # → DELETE (use openadapt-evals)
│   ├── waa_live.py                # → DELETE (use openadapt-evals)
│   ├── data_collection.py         # → DELETE (use openadapt-evals)
│   ├── live_tracker.py            # → DELETE (use openadapt-evals)
│   ├── azure.py                   # → DELETE (use openadapt-evals)
│   └── trace_export.py            # → DELETE (use openadapt-evals)
│
├── cloud/                         # Cloud infrastructure
│   ├── local.py                   # Dashboard server (~3700 lines, 90% benchmark)
│   ├── ssh_tunnel.py              # SSH tunnel management
│   ├── lambda_labs.py             # Lambda Labs GPU training
│   └── azure_inference.py         # Azure ML inference
│
├── training/                      # ML Training (KEEP in openadapt-ml)
│   ├── trainer.py                 # Core trainer
│   ├── trl_trainer.py             # TRL-based trainer
│   ├── stub_provider.py           # Mock training for testing
│   ├── benchmark_viewer.py        # Training benchmark viewer
│   ├── azure_ops_viewer.py        # Azure ops viewer
│   ├── shared_ui.py               # Shared UI components
│   ├── viewer.py                  # Training viewer
│   └── viewer_components.py       # Viewer components
│
├── models/                        # VLM Adapters (KEEP in openadapt-ml)
│   ├── api_adapter.py             # API-based VLM
│   ├── base_adapter.py            # Base adapter interface
│   ├── qwen_vl.py                 # Qwen adapter
│   ├── dummy_adapter.py           # Testing
│   └── providers/                 # Provider implementations
│       ├── anthropic.py
│       ├── openai.py
│       └── google.py
│
├── baselines/                     # Baseline adapters (KEEP in openadapt-ml)
│   ├── adapter.py
│   ├── cli.py
│   ├── config.py
│   ├── parser.py
│   └── prompts.py
│
├── grounding/                     # UI grounding (KEEP in openadapt-ml)
│   ├── base.py
│   └── detector.py
│
├── ingest/                        # Data ingestion (KEEP in openadapt-ml)
│   ├── capture.py                 # OpenAdapt capture ingestion
│   ├── loader.py
│   └── synthetic.py
│
├── retrieval/                     # Demo retrieval (KEEP in openadapt-ml)
│   ├── retriever.py
│   ├── demo_retriever.py
│   ├── embeddings.py
│   └── index.py
│
├── experiments/                   # Research experiments (KEEP in openadapt-ml)
│   ├── demo_prompt/               # Demo-conditioned prompting
│   ├── representation_shootout/   # Representation experiments
│   └── waa_demo/                  # WAA demo experiments
│
├── segmentation/                  # Workflow segmentation (KEEP in openadapt-ml)
│   ├── cli.py
│   ├── pipeline.py
│   ├── annotator.py
│   └── ...
│
├── runtime/                       # Runtime policy (KEEP in openadapt-ml)
│   ├── policy.py
│   └── safety_gate.py
│
├── schema/                        # Data schemas
│   ├── episode.py                 # Episode schema
│   └── converters.py
│
├── evals/                         # Evaluation metrics (KEEP in openadapt-ml)
│   ├── grounding.py
│   ├── trajectory_matching.py
│   └── plot_eval_metrics.py
│
├── export/                        # Data export (KEEP in openadapt-ml)
│   ├── cli.py
│   └── parquet.py
│
├── scripts/                       # CLI scripts (KEEP in openadapt-ml)
│   ├── train.py
│   ├── compare.py
│   ├── capture_screenshots.py
│   └── ...
│
└── config.py                      # Configuration
```

### openadapt-evals Current Structure

```
openadapt_evals/
├── adapters/                      # Benchmark adapters (KEEP in openadapt-evals)
│   ├── base.py                    # BenchmarkAdapter interface
│   ├── waa.py                     # WAAMockAdapter
│   └── waa_live.py                # WAALiveAdapter
│
├── agents/                        # Benchmark agents (KEEP in openadapt-evals)
│   ├── base.py                    # BenchmarkAgent interface
│   ├── api_agent.py               # Claude/GPT API agent (P0 demo fix)
│   ├── retrieval_agent.py         # Demo retrieval agent
│   ├── scripted_agent.py          # Scripted agent for testing
│   ├── baseline_agent.py          # → MOVE to openadapt-ml (uses VLM)
│   └── policy_agent.py            # → MOVE to openadapt-ml (uses trained model)
│
├── benchmarks/                    # Benchmark framework (KEEP in openadapt-evals)
│   ├── cli.py                     # Evaluation CLI
│   ├── runner.py                  # evaluate_agent_on_benchmark()
│   ├── data_collection.py         # ExecutionTraceCollector
│   ├── live_tracker.py            # LiveEvaluationTracker
│   ├── monitoring.py              # Benchmark monitoring
│   ├── dashboard_server.py        # Dashboard HTTP server
│   ├── viewer.py                  # Results viewer
│   ├── config.py                  # Configuration
│   ├── health_checker.py          # Health checking
│   ├── auto_screenshot.py         # Screenshot automation
│   ├── generate_synthetic_demos.py
│   ├── validate_demos.py
│   ├── validate_screenshots.py
│   ├── agent.py                   # → Duplicate
│   ├── base.py                    # → Duplicate
│   ├── waa.py                     # → Duplicate
│   ├── waa_live.py                # → Duplicate
│   ├── azure.py                   # → Duplicate
│   └── live_api.py
│
├── evaluation/                    # Evaluation framework (KEEP)
│   ├── client.py                  # → REVIEW (may be dead code)
│   └── discovery.py               # VM IP auto-discovery (KEEP)
│
├── server/                        # Server patches → DELETE (unused)
│   ├── evaluate_endpoint.py       # → DELETE (never deployed)
│   └── waa_server_patch.py        # → DELETE (never deployed)
│
├── shared_ui/                     # UI components (KEEP)
│   └── keyboard_shortcuts.py
│
├── metrics/                       # Metrics (KEEP)
│   └── __init__.py
│
└── tests/                         # Tests (KEEP)
    ├── test_api_agent_p0_fix.py
    ├── test_api_agent_parsing.py
    ├── test_cost_optimization.py
    ├── test_evaluate_endpoint.py
    ├── test_mock_adapter.py
    ├── test_retrieval_agent.py
    ├── test_runner.py
    └── test_synthetic_demos.py
```

### PR #14 Code Summary

PR #14 (merged Jan 2026) added the VM management CLI to openadapt-ml:

**Files Added/Modified:**
- `openadapt_ml/benchmarks/cli.py` - ~1300 lines of VM lifecycle commands
- `openadapt_ml/benchmarks/waa_deploy/Dockerfile` - Custom WAA Docker image
- `openadapt_ml/benchmarks/waa_deploy/api_agent.py` - Agent inside container
- `openadapt_ml/benchmarks/waa_deploy/install.bat` - Windows setup
- `openadapt_ml/benchmarks/waa_deploy/start_waa_server.bat` - Server startup

**CLI Commands (from PR #14):**
```
create      - Create Azure VM with nested virtualization
delete      - Delete VM and ALL associated resources
status      - Show VM state and IP
build       - Build WAA image from waa_deploy/Dockerfile
start       - Start WAA container
stop        - Stop container
probe       - Check if WAA server is ready
run         - Run benchmark tasks
deallocate  - Stop VM (preserves disk, stops billing)
logs        - Show WAA status and logs
exec        - Run command in container
docker-exec - Run docker command on host
vnc         - Open VNC viewer
tasks       - List available tasks
download    - Download results
analyze     - Analyze results
```

**Destination in openadapt-evals:**
- `cli.py` → `openadapt_evals/cli/vm.py` (merge with existing evals CLI)
- `waa_deploy/` → `openadapt_evals/waa_deploy/`

---

## Part 0.5: Code Audit Results (VERIFIED 2026-01-28)

> **✅ VERIFIED**: These findings have been confirmed by comprehensive import analysis.

### Audit Methodology

Verified by:
1. Checking all imports across the codebase (`grep -r "from.*module\|import.*module"`)
2. Checking exports in `__init__.py` files and `__all__`
3. Checking CLI command references
4. Checking test file imports

### Dead Code (VERIFIED - 10 files)

| File | Status | Evidence |
|------|--------|----------|
| `benchmarks/agent.py` | ✅ DEAD (deprecated shim) | Deprecation warning, zero imports |
| `benchmarks/base.py` | ✅ DEAD (deprecated shim) | Deprecation warning, zero imports |
| `benchmarks/waa.py` | ✅ DEAD (deprecated shim) | Deprecation warning, zero imports |
| `benchmarks/waa_live.py` | ✅ DEAD (deprecated shim) | Deprecation warning, zero imports |
| `benchmarks/auto_screenshot.py` | ✅ DEAD | Zero imports, no CLI command |
| `benchmarks/dashboard_server.py` | ✅ DEAD | Zero imports, no CLI command |
| `benchmarks/generate_synthetic_demos.py` | ✅ DEAD | Zero imports, no CLI command |
| `benchmarks/live_api.py` | ✅ DEAD | Zero imports, no CLI command |
| `benchmarks/validate_demos.py` | ✅ DEAD | Zero imports, no CLI command |
| `benchmarks/validate_screenshots.py` | ✅ DEAD | Zero imports, no CLI command |

**Total: ~1000 lines of dead code to remove**

### Previously Marked Dead But Actually Used (3 files)

| File | Status | Evidence |
|------|--------|----------|
| `agents/baseline_agent.py` | ✅ USED | Lazy-exported in `agents/__init__.py` |
| `server/waa_server_patch.py` | ✅ USED | Referenced in `scripts/patch_waa_evaluate.py` |
| `server/evaluate_endpoint.py` | ✅ USED | Exported and tested (100+ tests) |

### Agents Analysis (VERIFIED)

Agents directory split based on ML dependencies:

| Agent | ML Deps | Key Imports | Recommendation |
|-------|---------|-------------|----------------|
| `BenchmarkAgent` (base.py) | ❌ None | `abc`, `re`, `dataclasses` | Keep in openadapt-evals |
| `ScriptedAgent`, `RandomAgent`, `SmartMockAgent` | ❌ None | `random` | Keep in openadapt-evals |
| `ApiAgent` | ❌ None | `anthropic`, `openai` (API clients only) | Keep in openadapt-evals |
| `RetrievalAugmentedAgent` | ⚠️ `openadapt_retrieval` | Embedding models | Keep w/ lazy load |
| `PolicyAgent` | ✅ `openadapt_ml.vlm` | torch, transformers | **MOVE to openadapt-ml** |
| `BaselineAgent` | ✅ `openadapt_ml.baselines` | torch, transformers | **MOVE to openadapt-ml** |

**Key Insight**: `ApiAgent` does NOT need ML deps - it just wraps hosted API clients (Claude, GPT).

### Duplicates Between Repos (7 file pairs)

These files exist in both openadapt-ml and openadapt-evals:

| openadapt_evals/ | openadapt_ml/benchmarks/ | Notes |
|------------------|--------------------------|-------|
| `adapters/base.py` | `base.py` | Core schemas |
| `adapters/waa.py` | `waa.py` | WAA adapter |
| `adapters/waa_live.py` | `waa_live.py` | Live adapter |
| `benchmarks/runner.py` | `runner.py` | Eval loop |
| `benchmarks/data_collection.py` | `data_collection.py` | Trace saving |
| `benchmarks/live_tracker.py` | `live_tracker.py` | Progress tracking |
| `benchmarks/azure.py` | `azure.py` | Azure orchestration |

**Recommendation**: Pick one canonical location, delete the other, update imports.

### Genuine Value-Add (TENTATIVE - 10 files)

These files provide functionality not available elsewhere:

| File | Value | Confidence |
|------|-------|------------|
| `agents/api_agent.py` | **P0 demo persistence fix** - critical | High |
| `agents/retrieval_agent.py` | Demo retrieval feature | High |
| `agents/scripted_agent.py` | Testing utilities (RandomAgent, SmartMockAgent) | High |
| `evaluation/discovery.py` | VM IP auto-discovery from multiple sources | High |
| `benchmarks/cli.py` | Evaluation-focused CLI | High |
| `benchmarks/config.py` | Task loading utilities | High |
| `benchmarks/runner.py` | Core evaluation loop | High |
| `benchmarks/viewer.py` | Results viewer | High |
| `benchmarks/health_checker.py` | Used by azure.py | Medium |
| `benchmarks/monitoring.py` | Cost tracking (used by tests) | Medium |

### Revised Migration Recommendation

Based on this audit, the approach is **simpler than originally planned**:

**openadapt-evals already exists** - we're consolidating INTO it, not creating a new repo.

**Move FROM openadapt-ml TO openadapt-evals:**
- `benchmarks/cli.py` (VM commands) → merge into `openadapt_evals/cli/`
- `benchmarks/waa_deploy/` → `openadapt_evals/waa_deploy/`
- `benchmarks/vm_monitor.py` → `openadapt_evals/infrastructure/`
- `benchmarks/session_tracker.py` → `openadapt_evals/infrastructure/`
- `cloud/ssh_tunnel.py` → `openadapt_evals/infrastructure/`

**Delete FROM openadapt-evals (VERIFIED):**
- Deprecated shims (4): `benchmarks/agent.py`, `benchmarks/base.py`, `benchmarks/waa.py`, `benchmarks/waa_live.py`
- Dead code (6): `auto_screenshot.py`, `dashboard_server.py`, `generate_synthetic_demos.py`, `live_api.py`, `validate_demos.py`, `validate_screenshots.py`

**KEEP in openadapt-evals (previously marked for deletion but actually used):**
- `server/waa_server_patch.py` - used by `scripts/patch_waa_evaluate.py`
- `server/evaluate_endpoint.py` - exported and tested
- `agents/baseline_agent.py` - lazy-exported in public API

**Delete FROM openadapt-ml (duplicates):**
- `benchmarks/agent.py`, `base.py`, `runner.py`, `waa.py`, `waa_live.py`
- `benchmarks/data_collection.py`, `live_tracker.py`, `azure.py`

**Move FROM openadapt-evals TO openadapt-ml (FIXES circular dependency):**
- `agents/policy_agent.py` - currently imports `openadapt_ml.vlm` (circular!)
- `agents/baseline_agent.py` - currently imports `openadapt_ml.baselines` (circular!)
- Moving them to openadapt-ml fixes the dependency direction:
  - Before: evals → ml (wrong, creates circular dep)
  - After: ml has the agents, depends on evals (correct)
- Keep backward-compat lazy imports in openadapt-evals (optional, for API compat)

**Keep in openadapt-evals (no ML deps):**
- `agents/base.py` - abstract interface
- `agents/api_agent.py` - just API clients (anthropic, openai)
- `agents/scripted_agent.py` - test agents
- `agents/retrieval_agent.py` - keep with lazy load for openadapt_retrieval

---

## Part 1: Architecture

### Package Layering

```
┌─────────────────────────────────────────────────────────────┐
│                      openadapt-ml                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Training    │  VLM Inference  │  Policy Agent      │   │
│  │  Fine-tuning │  Qwen, etc.     │  Trained models    │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│                    depends on                               │
│                         ▼                                   │
├─────────────────────────────────────────────────────────────┤
│                      openadapt-evals                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Adapters    │ │ Agents      │ │ Evaluation          │   │
│  │ WAA, OS-    │ │ API (GPT,   │ │ Runner, metrics     │   │
│  │ World, etc  │ │ Claude)     │ │ Data collection     │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Infra       │ │ Dashboard   │ │ CLI                 │   │
│  │ VM, Docker  │ │ Monitoring  │ │ evals command       │   │
│  │ SSH, Azure  │ │ Viewers     │ │                     │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Schemas    │  Config       │  Utilities            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### User Journeys

**Journey 1: Benchmark Researcher (WAA, OSWorld, etc.)**
```bash
pip install openadapt-evals
oa evals vm setup
oa evals run --agent gpt-4o --tasks 10
oa evals view --run my_eval
oa evals vm stop
```
- No ML dependencies (no PyTorch, no transformers)
- Lightweight install
- Supports multiple benchmarks (WAA, OSWorld, WebArena, etc.)

**Journey 2: ML Engineer (Training + Benchmarks)**
```bash
pip install openadapt-ml  # Also installs openadapt-evals
oa ml train --capture /path/to/recording --goal "Open Notepad"
oa evals run --agent policy --checkpoint ./model
oa ml serve --checkpoint ./model  # Serve model for inference
oa ml dashboard  # Training dashboard
```
- Full ML training capabilities
- Uses `oa evals` for evaluation
- Trains custom agents with `oa ml`, evaluates on benchmarks

**CLI Namespacing**: `oa evals <cmd>` for benchmarks, `oa ml <cmd>` for training. Clear ownership of commands.

---

## Part 2: Package Structures

### openadapt-evals (Foundation)

```
openadapt-evals/
├── openadapt_evals/
│   │
│   ├── ══════════════════════════════════════════
│   ├── # BENCHMARK FRAMEWORK
│   ├── ══════════════════════════════════════════
│   │
│   ├── schemas/                    # Shared data structures
│   │   ├── __init__.py
│   │   ├── actions.py              # BenchmarkAction
│   │   ├── observations.py         # BenchmarkObservation
│   │   ├── tasks.py                # BenchmarkTask
│   │   └── results.py              # BenchmarkResult
│   │
│   ├── adapters/                   # Benchmark environment adapters
│   │   ├── __init__.py
│   │   ├── base.py                 # BenchmarkAdapter interface
│   │   └── waa/                    # Windows Agent Arena
│   │       ├── __init__.py
│   │       ├── mock.py             # WAAMockAdapter
│   │       └── live.py             # WAALiveAdapter
│   │
│   ├── agents/                     # Benchmark agents
│   │   ├── __init__.py
│   │   ├── base.py                 # BenchmarkAgent interface
│   │   ├── api_agent.py            # Claude/GPT API agent (P0 demo fix)
│   │   ├── retrieval_agent.py      # Demo retrieval agent
│   │   ├── scripted_agent.py       # For testing
│   │   └── random_agent.py         # Baseline
│   │
│   ├── evaluation/                 # Evaluation framework
│   │   ├── __init__.py
│   │   ├── runner.py               # evaluate_agent_on_benchmark()
│   │   ├── metrics.py              # compute_metrics()
│   │   ├── data_collection.py      # ExecutionTraceCollector
│   │   └── live_tracker.py         # LiveEvaluationTracker
│   │
│   ├── ══════════════════════════════════════════
│   ├── # INFRASTRUCTURE
│   ├── ══════════════════════════════════════════
│   │
│   ├── infrastructure/             # VM & cloud infrastructure
│   │   ├── __init__.py
│   │   ├── azure_vm.py             # Azure VM lifecycle
│   │   ├── vm_monitor.py           # VM status monitoring
│   │   ├── session_tracker.py      # Cost/time tracking
│   │   ├── ssh_tunnel.py           # SSH tunnel management
│   │   ├── disk_manager.py         # Disk management
│   │   └── docker.py               # Docker management
│   │
│   ├── waa_deploy/                 # WAA Docker deployment
│   │   ├── Dockerfile
│   │   ├── api_agent.py            # Agent for inside container
│   │   └── install.bat
│   │
│   ├── ══════════════════════════════════════════
│   ├── # USER INTERFACE
│   ├── ══════════════════════════════════════════
│   │
│   ├── cli/                        # CLI commands
│   │   ├── __init__.py
│   │   ├── main.py                 # Entry point: oa evals
│   │   ├── vm.py                   # oa evals vm <cmd>
│   │   ├── run.py                  # oa evals run <cmd>
│   │   ├── view.py                 # oa evals view
│   │   └── tasks.py                # oa evals tasks
│   │
│   ├── dashboard/                  # Monitoring dashboard
│   │   ├── __init__.py
│   │   ├── server.py               # HTTP server
│   │   ├── api.py                  # REST endpoints
│   │   └── viewers/                # HTML generation
│   │       ├── benchmark.py
│   │       └── azure_ops.py
│   │
│   ├── ══════════════════════════════════════════
│   ├── # UTILITIES
│   ├── ══════════════════════════════════════════
│   │
│   └── config.py                   # Configuration (API keys, Azure, etc.)
│
├── tests/
│   ├── test_adapters.py
│   ├── test_agents.py
│   ├── test_runner.py
│   ├── test_vm.py
│   └── test_cli.py
│
├── docs/
│   ├── getting_started.md
│   ├── cli_reference.md
│   └── vm_setup.md
│
├── pyproject.toml
├── README.md                       # Benchmark-focused marketing
└── CLAUDE.md
```

**pyproject.toml (openadapt-evals):**
```toml
[project]
name = "openadapt-evals"
description = "GUI agent benchmark toolkit - WAA, OSWorld, WebArena evaluations"
dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
    "pillow>=10.0.0",
    "azure-cli>=2.50.0",
    # NO torch, NO transformers, NO heavy ML deps
]

[project.scripts]
oa = "openadapt_evals.cli.main:main"  # Provides: oa evals <cmd>
```

### openadapt-ml (Extension)

```
openadapt-ml/
├── openadapt_ml/
│   │
│   ├── ══════════════════════════════════════════
│   ├── # ML TRAINING
│   ├── ══════════════════════════════════════════
│   │
│   ├── training/                   # Model training
│   │   ├── __init__.py
│   │   ├── trainer.py              # Core trainer
│   │   ├── trl_trainer.py          # TRL-based trainer
│   │   ├── stub_provider.py        # Mock training for testing
│   │   └── dashboard.py            # Training dashboard generation
│   │
│   ├── vlm/                        # VLM inference
│   │   ├── __init__.py
│   │   ├── qwen.py                 # Qwen adapter
│   │   ├── api_adapter.py          # API-based VLM
│   │   └── base.py
│   │
│   ├── baselines/                  # Baseline model adapters
│   │   ├── __init__.py
│   │   ├── unified_adapter.py
│   │   └── providers/
│   │
│   ├── grounding/                  # UI element grounding
│   │   ├── __init__.py
│   │   └── gemini_grounder.py
│   │
│   ├── ══════════════════════════════════════════
│   ├── # AGENTS & INTEGRATION
│   ├── ══════════════════════════════════════════
│   │
│   ├── agents/                     # ML-specific agents
│   │   ├── __init__.py
│   │   ├── policy_agent.py         # Uses trained VLM policy
│   │   └── baseline_agent.py       # Unified baseline agent
│   │
│   ├── ══════════════════════════════════════════
│   ├── # CLI EXTENSION
│   ├── ══════════════════════════════════════════
│   │
│   ├── cli/                        # Extended CLI
│   │   ├── __init__.py
│   │   ├── main.py                 # Entry point: oa ml <cmd>
│   │   ├── train.py                # oa ml train
│   │   ├── serve.py                # oa ml serve (model inference server)
│   │   └── dashboard.py            # oa ml dashboard (training dashboard)
│   │
│   ├── ══════════════════════════════════════════
│   ├── # DATA & UTILITIES
│   ├── ══════════════════════════════════════════
│   │
│   ├── ingest/                     # Data ingestion
│   │   ├── __init__.py
│   │   └── capture.py              # OpenAdapt capture ingestion
│   │
│   ├── cloud/                      # Cloud GPU training
│   │   ├── __init__.py
│   │   ├── lambda_labs.py
│   │   └── azure_ml.py
│   │
│   ├── experiments/                # Research experiments
│   │   ├── demo_prompt/
│   │   └── waa_demo/
│   │
│   └── config.py                   # ML-specific config (extends evals config)
│
├── tests/
│   ├── test_training.py
│   ├── test_vlm.py
│   ├── test_policy_agent.py
│   └── test_cli.py
│
├── docs/
│   ├── training_guide.md
│   ├── model_development.md
│   └── cloud_training.md
│
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

**pyproject.toml (openadapt-ml):**
```toml
[project]
name = "openadapt-ml"
description = "ML training toolkit for OpenAdapt GUI automation agents"
dependencies = [
    "openadapt-evals>=0.1.0",       # Foundation dependency
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "trl>=0.8.0",
    "accelerate>=0.27.0",
    # Heavy ML deps here
]

# Note: oa entry point is registered by openadapt-evals
# openadapt-ml extends it by registering additional subcommands
# Implementation: oa ml <cmd> routes to openadapt_ml.cli
```

---

## Part 3: CLI Design

CLI uses namespaced subcommands: `oa evals <cmd>` for benchmarks, `oa ml <cmd>` for training.

### oa evals (openadapt-evals)

```bash
# VM Management
oa evals vm create         # Create Azure VM
oa evals vm delete         # Delete VM
oa evals vm start / stop   # Start/stop VM
oa evals vm deallocate     # Deallocate (stop billing)
oa evals vm status         # Show VM status
oa evals vm setup          # Full setup (Docker + benchmark image)
oa evals vm probe          # Check benchmark server status
oa evals vm diag           # Diagnostic info
oa evals vm logs           # Container logs
oa evals vm ssh            # Interactive SSH
oa evals vm vnc            # Open VNC viewer

# Evaluation
oa evals mock              # Mock evaluation (no VM)
oa evals live              # Live evaluation against server
oa evals run               # Shorthand for common evaluation

# Results & Monitoring
oa evals view              # Generate results viewer
oa evals dashboard         # Start monitoring dashboard
oa evals tasks             # List available tasks

# Configuration
oa evals config            # Show/edit configuration
oa evals config set KEY VALUE
```

### oa ml (openadapt-ml)

```bash
# Training
oa ml train                # Start training
oa ml train --capture /path --goal "description"
oa ml train --config config.yaml
oa ml train status         # Training status
oa ml train stop           # Stop training

# Model Serving (Inference)
oa ml serve                       # Serve trained model for inference
oa ml serve --checkpoint ./model  # Serve specific checkpoint
oa ml serve --port 8080           # Custom port

# Training Dashboard
oa ml dashboard            # Start training dashboard
oa ml dashboard --port 8080

# Cloud Training
oa ml cloud launch         # Launch cloud GPU instance
oa ml cloud status         # Check cloud training
oa ml cloud terminate      # Terminate instance

# ML-specific evaluation (uses oa evals under the hood)
oa evals run --agent policy --checkpoint ./model
```

---

## Part 4: Migration Steps

### Phase 1: Consolidate into openadapt-evals (Existing Repo)

Since openadapt-evals already exists, we consolidate INTO it rather than creating a new repo.

1. **Restructure openadapt-evals** for multi-benchmark support:
   - Move `adapters/waa*.py` → `adapters/waa/` (subdirectory per benchmark)
   - Move `benchmarks/cli.py` → restructure into `cli/`
   - Move `benchmarks/dashboard_server.py` → `dashboard/`
   - Create `infrastructure/` directory for VM/cloud code
2. **Copy from openadapt-ml**:
   - `benchmarks/cli.py` (VM commands) → `openadapt_evals/cli/vm.py`
   - `benchmarks/waa_deploy/` → `openadapt_evals/waa_deploy/`
   - `benchmarks/vm_monitor.py` → `openadapt_evals/infrastructure/vm_monitor.py`
   - `benchmarks/session_tracker.py` → `openadapt_evals/infrastructure/session_tracker.py`
   - `benchmarks/azure_ops_tracker.py` → `openadapt_evals/infrastructure/azure_ops_tracker.py`
   - `cloud/ssh_tunnel.py` → `openadapt_evals/infrastructure/ssh_tunnel.py`
3. **Clean up dead code** (after verification):
   - Delete deprecated shims: `benchmarks/agent.py`, `benchmarks/base.py`, etc.
   - Delete unused server patch: `server/waa_server_patch.py`, `server/evaluate_endpoint.py`
4. **Write CLI entry point**: `evals`
5. **Write tests**
6. **Write README** with multi-benchmark marketing

### Phase 2: Refactor openadapt-ml

1. **Add dependency**: `openadapt-evals>=0.1.0`
2. **Delete moved code**:
   - `benchmarks/` (most of it)
   - `cloud/local.py` (dashboard moved to evals)
   - `cloud/ssh_tunnel.py` (moved to evals)
3. **Keep ML-specific code**:
   - `training/`
   - `vlm/`
   - `baselines/`
   - `grounding/`
   - `ingest/`
   - `cloud/lambda_labs.py`, `cloud/azure_ml.py`
   - `experiments/`
4. **Add ML-specific agents**:
   - `agents/policy_agent.py`
   - `agents/baseline_agent.py`
5. **Create extended CLI**: `oa` that imports from evals and adds training
6. **Update imports** to use `openadapt_evals`
7. **Update tests**

### Phase 3: Update Documentation

1. **Update openadapt-evals README**: Multi-benchmark focus
   - "GUI agent benchmark toolkit - WAA, OSWorld, WebArena evaluations"
2. **Update openadapt-ml README**: Training focus
   - Links to openadapt-evals for evaluation
3. **Update CLAUDE.md** in both repos

### Phase 4: Publishing & Marketing

1. **openadapt-evals README**: Multi-benchmark-focused
   - "GUI agent benchmark toolkit - WAA, OSWorld, WebArena evaluations"
   - One-liner install
   - Quick start examples for each supported benchmark
2. **openadapt-ml README**: Training-focused
   - "Train custom GUI automation agents"
   - Links to openadapt-evals for evaluation
3. **PyPI publishing**: Publish both packages
4. **Update main OpenAdapt docs** to reference both

---

## Part 5: File Mapping (Detailed)

### openadapt-evals Internal Restructuring

These files stay in openadapt-evals but may be reorganized:

| Current Location | New Location | Notes |
|------------------|--------------|-------|
| **Adapters** | | |
| `adapters/base.py` | `adapters/base.py` | BenchmarkAdapter interface (keep) |
| `adapters/waa.py` | `adapters/waa/mock.py` | WAAMockAdapter |
| `adapters/waa_live.py` | `adapters/waa/live.py` | WAALiveAdapter |
| **Agents** | | |
| `agents/base.py` | `agents/base.py` | BenchmarkAgent interface (keep) |
| `agents/api_agent.py` | `agents/api_agent.py` | Claude/GPT agent (P0 demo fix) |
| `agents/retrieval_agent.py` | `agents/retrieval_agent.py` | Demo retrieval |
| `agents/scripted_agent.py` | `agents/scripted_agent.py` | For testing |
| `agents/baseline_agent.py` | → MOVE to openadapt-ml | Uses VLM (ML dep) |
| `agents/policy_agent.py` | → MOVE to openadapt-ml | Uses trained model (ML dep) |
| **Evaluation** | | |
| `benchmarks/runner.py` | `evaluation/runner.py` | Core evaluation |
| `benchmarks/data_collection.py` | `evaluation/data_collection.py` | Trace collector |
| `benchmarks/live_tracker.py` | `evaluation/live_tracker.py` | Live tracking |
| `benchmarks/monitoring.py` | `evaluation/monitoring.py` | Monitoring |
| `benchmarks/health_checker.py` | `evaluation/health_checker.py` | Health checks |
| `evaluation/client.py` | `evaluation/client.py` | Eval client |
| `evaluation/discovery.py` | `evaluation/discovery.py` | Service discovery |
| **CLI** | | |
| `benchmarks/cli.py` | `cli/eval.py` | Evaluation commands |
| **Dashboard** | | |
| `benchmarks/dashboard_server.py` | `dashboard/server.py` | HTTP server |
| `benchmarks/viewer.py` | `dashboard/viewer.py` | Results viewer |
| **Config** | | |
| `benchmarks/config.py` | `config.py` | Configuration |
| **Delete (dead code)** | | |
| `server/evaluate_endpoint.py` | DELETE | Never deployed |
| `server/waa_server_patch.py` | DELETE | Never deployed |
| `benchmarks/auto_screenshot.py` | DELETE | Never imported |
| `benchmarks/generate_synthetic_demos.py` | DELETE | Never imported |
| `benchmarks/validate_demos.py` | DELETE | Never imported |
| `benchmarks/validate_screenshots.py` | DELETE | Never imported |
| `benchmarks/live_api.py` | DELETE | Never imported |
| **Delete (duplicates)** | | |
| `benchmarks/agent.py` | DELETE | Duplicate shim |
| `benchmarks/base.py` | DELETE | Duplicate shim |
| `benchmarks/waa.py` | DELETE | Duplicate shim |
| `benchmarks/waa_live.py` | DELETE | Duplicate shim |
| `benchmarks/azure.py` | DELETE | Duplicate |
| **UI Components** | | |
| `shared_ui/keyboard_shortcuts.py` | `shared_ui/keyboard_shortcuts.py` | UI shortcuts |
| **Tests** | | |
| `tests/test_api_agent_*.py` | `tests/test_api_agent_*.py` | Agent tests |
| `tests/test_runner.py` | `tests/test_runner.py` | Runner tests |
| `tests/test_mock_adapter.py` | `tests/test_mock_adapter.py` | Adapter tests |
| `tests/test_retrieval_agent.py` | `tests/test_retrieval_agent.py` | Retrieval tests |

### From openadapt-ml → openadapt-evals

| Source (openadapt_ml/) | Destination (openadapt_evals/) | Notes |
|------------------------|--------------------------------|-------|
| **PR #14 Code** | | |
| `benchmarks/cli.py` | `cli/vm.py` | ⭐ VM lifecycle commands (1300 lines) |
| `benchmarks/waa_deploy/` | `waa_deploy/` | ⭐ Docker deployment files |
| `benchmarks/waa_deploy/Dockerfile` | `waa_deploy/Dockerfile` | WAA image build |
| `benchmarks/waa_deploy/api_agent.py` | `waa_deploy/api_agent.py` | In-container agent |
| `benchmarks/waa_deploy/install.bat` | `waa_deploy/install.bat` | Windows setup |
| `benchmarks/waa_deploy/start_waa_server.bat` | `waa_deploy/start_waa_server.bat` | Server startup |
| **Infrastructure** | | |
| `benchmarks/vm_monitor.py` | `infrastructure/vm_monitor.py` | VM status monitoring |
| `benchmarks/session_tracker.py` | `infrastructure/session_tracker.py` | Cost/time tracking |
| `benchmarks/azure_ops_tracker.py` | `infrastructure/azure_ops_tracker.py` | Azure op logging |
| `benchmarks/disk_manager.py` | `infrastructure/disk_manager.py` | Disk management |
| `benchmarks/dashboard.py` | `dashboard/panels.py` | Dashboard panels |
| `cloud/ssh_tunnel.py` | `infrastructure/ssh_tunnel.py` | SSH tunnels |
| **Dashboard Server** | | |
| `cloud/local.py` (partial) | `dashboard/server.py` | ~90% is benchmark (extract) |
| | | Training parts stay in openadapt-ml |
| **Viewers** | | |
| `benchmarks/viewer.py` | `dashboard/benchmark_viewer.py` | Benchmark viewer |
| `training/azure_ops_viewer.py` | `dashboard/azure_ops_viewer.py` | Azure ops viewer |
| **Skip (Duplicates - already in openadapt-evals)** | | |
| `benchmarks/agent.py` | Skip | Already in openadapt-evals |
| `benchmarks/base.py` | Skip | Already in openadapt-evals |
| `benchmarks/runner.py` | Skip | Already in openadapt-evals |
| `benchmarks/waa.py` | Skip | Already in openadapt-evals |
| `benchmarks/waa_live.py` | Skip | Already in openadapt-evals |
| `benchmarks/data_collection.py` | Skip | Already in openadapt-evals |
| `benchmarks/live_tracker.py` | Skip | Already in openadapt-evals |
| `benchmarks/azure.py` | Skip | Already in openadapt-evals |
| `benchmarks/trace_export.py` | Skip | Not needed |

### Stays in openadapt-ml (After Migration)

| Directory | Contents | Notes |
|-----------|----------|-------|
| `training/` | trainer.py, trl_trainer.py, stub_provider.py, etc. | Core ML training |
| `models/` | api_adapter.py, qwen_vl.py, providers/ | VLM inference |
| `baselines/` | adapter.py, cli.py, config.py, etc. | Baseline models |
| `grounding/` | base.py, detector.py | UI grounding |
| `ingest/` | capture.py, loader.py, synthetic.py | Data ingestion |
| `retrieval/` | retriever.py, demo_retriever.py, etc. | Demo retrieval |
| `experiments/` | demo_prompt/, waa_demo/, etc. | Research |
| `segmentation/` | cli.py, pipeline.py, etc. | Workflow segmentation |
| `runtime/` | policy.py, safety_gate.py | Runtime policy |
| `evals/` | grounding.py, trajectory_matching.py | Eval metrics |
| `export/` | cli.py, parquet.py | Data export |
| `scripts/` | train.py, compare.py, etc. | CLI scripts |
| `schema/` | episode.py, converters.py | OR move to openadapt-evals |
| `cloud/lambda_labs.py` | GPU training | Keep |
| `cloud/azure_inference.py` | Azure ML | Keep |
| `config.py` | Configuration | Extend openadapt_evals.config |

### New Files in openadapt-ml (After Migration)

| File | Purpose |
|------|---------|
| `agents/policy_agent.py` | Move from openadapt-evals (ML dep) |
| `agents/baseline_agent.py` | Move from openadapt-evals (ML dep) |
| `cli/main.py` | `oa` CLI entry point (extends `evals`) |
| `cli/train.py` | Training commands |
| `cli/serve.py` | Model inference server |
| `cli/dashboard.py` | Training dashboard |

### Delete from openadapt-ml (After Migration)

| File | Reason |
|------|--------|
| `benchmarks/` (entire directory) | Moved to openadapt-evals |
| `cloud/local.py` | Dashboard parts moved to openadapt-evals |
| `cloud/ssh_tunnel.py` | Moved to openadapt-evals |
| `training/azure_ops_viewer.py` | Moved to openadapt-evals |
| `training/benchmark_viewer.py` | Moved to openadapt-evals |

---

## Part 6: Effort Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| 1. Restructure openadapt-evals | Reorganize files, create cli/, infrastructure/ | 3-4 hrs |
| 2. Copy VM code from openadapt-ml | Move PR #14 code to evals | 2-3 hrs |
| 3. Write evals CLI | Entry point, subcommands | 2-3 hrs |
| 4. Clean up dead code | Delete unused files (after verification) | 1-2 hrs |
| 5. Refactor openadapt-ml | Delete moved code, add dependency | 2-3 hrs |
| 6. Write oa CLI extension | Extends evals, adds training | 1-2 hrs |
| 7. Update tests | Fix imports in both repos | 2-3 hrs |
| 8. Documentation | READMEs, CLAUDE.md, docs | 2-3 hrs |

**Total: ~16-22 hours (2-3 days)**

---

## Part 7: Success Criteria

### openadapt-evals

- [ ] `pip install openadapt-evals` works
- [ ] `oa evals --help` shows all commands
- [ ] `oa evals vm status` works (no ML deps imported)
- [ ] `oa evals mock --tasks 5` works
- [ ] `oa evals run --agent gpt-4o` works (with VM running)
- [ ] All tests pass
- [ ] No PyTorch/transformers in dependencies
- [ ] README has multi-benchmark quick start (WAA, OSWorld, WebArena)

### openadapt-ml

- [ ] `pip install openadapt-ml` installs openadapt-evals too
- [ ] `oa ml --help` shows training commands
- [ ] `oa ml train --help` works
- [ ] `oa evals run --agent policy` works with trained model
- [ ] All tests pass
- [ ] Imports from openadapt_evals work correctly
- [ ] Dependency direction: openadapt-ml → openadapt-evals (not circular)

---

## Part 8: Marketing Positioning

### openadapt-evals

**Tagline**: "GUI agent benchmark toolkit - evaluate agents on WAA, OSWorld, WebArena"

**README opener**:
```markdown
# openadapt-evals

The easiest way to run GUI agent benchmarks.

## Quick Start

```bash
pip install openadapt-evals
oa evals vm setup                    # One-time Azure VM setup
oa evals run --agent gpt-4o --tasks 10
oa evals view                        # See results
```

No ML dependencies. No complex setup. Just benchmarks.

## Supported Benchmarks

- **Windows Agent Arena (WAA)** - 154 Windows desktop tasks
- **OSWorld** - Cross-platform desktop (coming soon)
- **WebArena/VisualWebArena** - Browser tasks (coming soon)
```

**Target audience**: Researchers evaluating agents, teams benchmarking LLM capabilities

### openadapt-ml

**Tagline**: "Train custom GUI automation agents"

**README opener**:
```markdown
# openadapt-ml

Train and fine-tune VLMs for GUI automation. Built on openadapt-evals.

## Quick Start

```bash
pip install openadapt-ml
oa ml train --capture ./recording --goal "Open Notepad and type Hello"
oa evals run --agent policy --checkpoint ./model
```

Full ML training pipeline with benchmark evaluation built in.
```

**Target audience**: ML engineers building GUI agents, researchers training custom models

---

## Part 9: Future Considerations

### Adding New Benchmarks

To add a new benchmark (e.g., OSWorld, WebArena):

1. Create adapter in `openadapt_evals/adapters/{benchmark}/`
2. Add CLI commands in `openadapt_evals/cli/{benchmark}.py`
3. Add VM/container setup if needed in `infrastructure/`
4. Update README with benchmark-specific quick start

No new repos needed - openadapt-evals supports all benchmarks.

### If We Split Again

The two-package structure is already clean. If further splitting needed:

- **openadapt-evals-azure**: Azure-specific infrastructure (for non-Azure users)
- **openadapt-evals-local**: Local-only running (Docker on local machine)

### Integration with Main OpenAdapt

```
OpenAdapt (main)          # Capture/recording
    ↓ recordings
openadapt-ml              # Training
    ↓ trained models
openadapt-evals           # Evaluation
    ↓ benchmark results
```

The full pipeline: Capture → Train → Evaluate

### openadapt-viewer Integration

Both packages can use openadapt-viewer for HTML generation:
```toml
# Optional dependency
[project.optional-dependencies]
viewer = ["openadapt-viewer>=0.1.0"]
```
