# What Should the New "OpenAdapt" Package Be?

A comprehensive analysis of package architecture options for the OpenAdapt ecosystem.

**Date**: January 2026
**Status**: Architecture Proposal

---

## Executive Summary

After reviewing how major ML/automation ecosystems structure their packages and analyzing our current codebase, I recommend **Option B+: Thin CLI Wrapper with Progressive Enhancement**. This provides a unified entry point (`pip install openadapt`) without requiring a complex full application upfront, while maintaining a clear path to evolve into a full product.

---

## 1. Literature Review: How Other Ecosystems Do It

### 1.1 HuggingFace Ecosystem

**Structure**: Hub-and-spoke model with a central "transformers" package

| Package | Role | Installation |
|---------|------|--------------|
| `transformers` | **Core** - model definitions, unified API | `pip install transformers` |
| `datasets` | Data loading and processing | `pip install datasets` |
| `accelerate` | Distributed training utilities | `pip install accelerate` |
| `evaluate` | Metrics and evaluation | `pip install evaluate` |
| `tokenizers` | Fast tokenization (Rust) | `pip install tokenizers` |
| `diffusers` | Image/video generation | `pip install diffusers` |
| `peft` | Parameter-efficient fine-tuning | `pip install peft` |
| `trl` | Reinforcement learning for LLMs | `pip install trl` |

**Key Insights** (from [Transformers v5](https://huggingface.co/blog/transformers-v5)):
- `transformers` is the **pivot** that all other tools build around
- 3M+ daily pip installs in 2025 (up from 20k in v4)
- Unified abstractions: `PreTrainedModel`, `PreTrainedConfig`, `PreTrainedTokenizerBase`
- **"Model-definition framework"** - it defines, others use
- Strong interoperability: Axolotl, Unsloth, DeepSpeed, vLLM, etc. all leverage transformers

**Lesson**: A single "core" package that defines the fundamental abstractions works well when it's clear what the core abstraction is.

### 1.2 LangChain Ecosystem

**Structure**: Core/Community/Integration split

| Package | Role | Installation |
|---------|------|--------------|
| `langchain-core` | Base interfaces and abstractions | Required dependency |
| `langchain` | Chains, agents, retrieval strategies | `pip install langchain` |
| `langchain-community` | Third-party integrations | `pip install langchain-community` |
| `langchain-openai` | OpenAI provider | `pip install langchain-openai` |
| `langchain-anthropic` | Anthropic provider | `pip install langchain-anthropic` |
| `langgraph` | Stateful multi-actor apps | `pip install langgraph` |
| `langsmith` | Observability/tracing | `pip install langsmith` |

**Key Insights** (from [LangChain Architecture](https://python.langchain.com/docs/concepts/architecture/)):
- Started monolithic, refactored into modular packages
- `langchain-core` holds **stable abstractions** with backward-compat guarantees
- Provider packages are versioned separately (critical for API changes)
- `langchain` (main) depends on `langchain-core` but NOT provider packages

**Lesson**: Separate stable core abstractions from volatile integrations. Provider packages should be opt-in.

### 1.3 PyTorch Ecosystem

**Structure**: Domain-specific libraries around a single core

| Package | Role | Installation |
|---------|------|--------------|
| `torch` | **Core** - tensor ops, autograd, training | `pip install torch` |
| `torchvision` | Computer vision models/transforms | `pip install torchvision` |
| `torchaudio` | Audio processing | `pip install torchaudio` |
| `torchtext` | NLP utilities | `pip install torchtext` |
| `torchserve` | Model serving | Separate install |

**Key Insights** (from [PyTorch Ecosystem](https://pytorch.org/)):
- `torch` is the undisputed core - everything depends on it
- Domain libraries (vision, audio) follow same philosophy but are independent
- Version coupling is explicit: `torchaudio 2.9` requires `torch 2.9`
- Recently: `torchaudio` moved to "maintenance phase" to reduce redundancy

**Lesson**: Domain libraries should be tightly version-coupled to core. Pruning redundant packages is healthy.

### 1.4 Agent Frameworks (AutoGPT, AgentGPT)

**Structure**: Platform-centric with toolkit separation

| Component | Role |
|-----------|------|
| `autogpt_platform` | Full platform (server, marketplace, GUI) |
| `AutoGPT Classic` | Original standalone agent |
| `Forge` | Toolkit for building custom agents |
| `agbenchmark` | Evaluation framework |

**Key Insights** (from [AutoGPT Docs](https://docs.agpt.co/)):
- Evolved from single agent to full platform
- **Forge** = reusable components, **Platform** = complete product
- Different licenses: Platform (Polyform Shield), Rest (MIT)
- Memory architecture: short-term (queue) + long-term (vector DB)

**Lesson**: Separation between toolkit (for developers) and platform (for end users) allows different licenses and evolution speeds.

### 1.5 ComfyUI / Stable Diffusion

**Structure**: Node-based plugin architecture

| Component | Role |
|-----------|------|
| `ComfyUI` | Core graph/node execution engine |
| `custom_nodes/` | Plugin directory (community extensions) |
| `ComfyUI Manager` | Package manager for extensions |
| `workflows/` | Shareable DAG definitions |

**Key Insights** (from [ComfyUI Docs](https://github.com/comfyanonymous/ComfyUI)):
- **Everything is a node** - maximum composability
- Lazy DAG evaluation - only run what changed
- Smart memory management (works with 1GB VRAM)
- Extensions via `custom_nodes/` directory - no core changes needed
- Workflows are JSON - shareable, versionable

**Lesson**: Node/plugin architecture enables massive community contribution without touching core. Clear extension points matter.

---

## 2. Analysis of Our Current Ecosystem

### 2.1 Package Inventory

| Package | Purpose | Key Exports | CLI Entry |
|---------|---------|-------------|-----------|
| **openadapt-ml** | ML engine, training, models, runtime | `AgentPolicy`, `QwenVLAdapter`, `train_with_trl` | `python -m openadapt_ml.scripts.train` |
| **openadapt-capture** | Screen recording, events | `CaptureSession`, event streams | `capture` command |
| **openadapt-grounding** | UI element localization | `OmniParser`, `UITarsGrounder` | Deploy commands |
| **openadapt-evals** | Benchmark evaluation | `ApiAgent`, `WAAAdapter`, `evaluate_agent_on_benchmark` | `openadapt-evals` command |
| **openadapt-viewer** | HTML viewer generation | `PageBuilder`, `screenshot_display`, components | `openadapt-viewer` command |
| **openadapt-retrieval** | Demo retrieval | `MultimodalDemoRetriever`, `Qwen3VLEmbedder` | `openadapt-retrieval` command |

### 2.2 Dependency Graph

```
                    +-----------------+
                    |  openadapt-ml   |  (THE CORE)
                    |    (v0.2.0)     |
                    +--------+--------+
                             |
              +--------------+---------------+
              |              |               |
              v              v               v
    +----------------+ +-----------+ +----------------+
    |openadapt-capture| |openadapt-| |openadapt-evals |
    |    (v0.1.0)    | |grounding  | |   (v0.1.0)     |
    +----------------+ |(v0.1.0)   | +----------------+
                       +-----------+

                +-------------------+
                | openadapt-viewer  |  (UI components)
                |     (v0.1.0)      |
                +-------------------+

                +-------------------+
                |openadapt-retrieval|  (Demo search)
                |     (v0.1.0)      |
                +-------------------+
```

**Current dependency from pyproject.toml**:
- `openadapt-ml` depends on `openadapt-capture>=0.1.0`
- `openadapt-evals` is standalone (can use openadapt-ml optionally)
- Other packages are standalone

### 2.3 What Each Package Actually Does

#### openadapt-ml (THE CORE)

**Primary responsibility**: Model-agnostic, domain-agnostic ML engine for GUI automation

**Key modules**:
```
openadapt_ml/
├── schema/           # Episode, Step, Action, Observation
├── models/           # QwenVLAdapter, APIAdapter, DummyAdapter
├── training/         # TRL trainer, dashboard generation
├── runtime/          # AgentPolicy, SafetyGate
├── ingest/           # Capture converter, synthetic data
├── datasets/         # Next-action SFT samples
├── benchmarks/       # WAA integration, VM management (shared with evals)
├── retrieval/        # Demo retriever (shared with retrieval package)
├── cloud/            # Lambda Labs, Azure, local serving
└── export/           # Parquet export
```

**CLI entry points**:
- `python -m openadapt_ml.scripts.train` - Train models
- `python -m openadapt_ml.scripts.compare` - Compare predictions
- `python -m openadapt_ml.benchmarks.cli vm monitor` - VM management
- `python -m openadapt_ml.cloud.local serve` - Serve dashboard

#### openadapt-capture

**Primary responsibility**: Platform-agnostic event capture with time-aligned media

**Key features**:
- Keyboard/mouse events via pynput
- Screen recording via av/mss
- Audio capture via sounddevice
- Whisper transcription
- Privacy scrubbing (optional)

**CLI**: `capture` command

#### openadapt-grounding

**Primary responsibility**: Robust UI element localization

**Key features**:
- OmniParser integration
- UI-TARS VLM grounding
- VLM provider adapters (Claude, GPT, Gemini)
- AWS deployment automation

**CLI**: `python -m openadapt_grounding.deploy`

#### openadapt-evals

**Primary responsibility**: Benchmark evaluation infrastructure

**Key features**:
- `ApiAgent` with P0 demo persistence fix
- `WAAAdapter`, `WAALiveAdapter`, `WAAMockAdapter`
- `evaluate_agent_on_benchmark()` runner
- Azure parallel evaluation

**CLI**: `openadapt-evals mock`, `openadapt-evals live`

#### openadapt-viewer

**Primary responsibility**: Reusable HTML visualization components

**Key features**:
- Screenshot displays with overlays
- Playback controls
- Metrics grids
- PageBuilder for composing views

**CLI**: `openadapt-viewer demo`

#### openadapt-retrieval

**Primary responsibility**: Multimodal demo retrieval

**Key features**:
- Qwen3-VL embeddings
- CLIP fallback
- FAISS vector index
- `MultimodalDemoRetriever`

**CLI**: `openadapt-retrieval embed`, `search`, `index`

### 2.4 What's Missing for a Complete Product?

| Gap | Description | Which Package? |
|-----|-------------|----------------|
| **Unified CLI** | No single `openadapt` command that ties it all together | NEW package |
| **GUI for recording** | End users need a GUI, not just CLI | NEW or openadapt-capture |
| **Example workflows** | No end-to-end examples showing packages working together | NEW package |
| **Documentation hub** | Docs scattered across repos | NEW package |
| **Agent orchestration** | No loop that runs capture -> train -> deploy -> eval | openadapt-ml or NEW |
| **Model registry** | No central place to publish/share trained models | NEW or external (HF Hub) |

---

## 3. Options with Detailed Pros/Cons

### Option A: openadapt = Examples/Docs Only

**Description**: The `openadapt` repo contains only examples, tutorials, and documentation. No code.

```
openadapt/
├── README.md
├── docs/
│   ├── getting-started.md
│   ├── architecture.md
│   └── tutorials/
├── examples/
│   ├── basic_capture_train_eval/
│   ├── demo_retrieval_augmented/
│   └── custom_grounding/
└── mkdocs.yml
```

**User experience**:
```bash
# No pip install openadapt
# Just visit docs site or clone repo for examples
```

| Pros | Cons |
|------|------|
| Zero maintenance burden | Not pip-installable |
| Clear that openadapt-ml is the core | Fragmented experience |
| No version conflicts | No unified CLI |
| Simple | Confusing for newcomers |

**Verdict**: Too minimal. Users expect `pip install openadapt` to work.

---

### Option B: openadapt = Thin CLI Wrapper

**Description**: `pip install openadapt` installs all packages as dependencies and provides a unified CLI.

```
openadapt/
├── pyproject.toml  # depends on all openadapt-* packages
├── src/openadapt/
│   ├── __init__.py  # re-exports common items
│   └── cli.py       # unified CLI
└── README.md
```

**pyproject.toml**:
```toml
[project]
name = "openadapt"
version = "0.1.0"
description = "GUI automation with ML"

dependencies = [
    "openadapt-ml>=0.2.0",
    "openadapt-capture>=0.1.0",
    "openadapt-evals>=0.1.0",
    "openadapt-viewer>=0.1.0",
]

[project.optional-dependencies]
grounding = ["openadapt-grounding>=0.1.0"]
retrieval = ["openadapt-retrieval>=0.1.0"]
all = ["openadapt[grounding,retrieval]"]

[project.scripts]
openadapt = "openadapt.cli:main"
```

**User experience**:
```bash
pip install openadapt

# Unified CLI
openadapt capture --name my-task
openadapt train --capture my-task
openadapt eval --checkpoint model.pt --benchmark waa
openadapt serve --port 8080
```

| Pros | Cons |
|------|------|
| Single pip install | Another package to maintain |
| Unified CLI | Version coordination needed |
| Easy for newcomers | Heavy install (all deps) |
| Clear entry point | May pull unused packages |

**Verdict**: Good balance. This is the LangChain approach.

---

### Option C: openadapt = Full Application

**Description**: Full GUI application with bundled everything. Like the legacy openadapt.

```
openadapt/
├── pyproject.toml
├── src/openadapt/
│   ├── __init__.py
│   ├── cli.py
│   ├── app/           # GUI application
│   │   ├── main.py
│   │   ├── windows/
│   │   └── dialogs/
│   ├── orchestrator/  # Agent loop
│   └── server/        # Web dashboard
└── README.md
```

**User experience**:
```bash
pip install openadapt

# GUI app
openadapt app  # Opens GUI

# Or headless
openadapt capture
openadapt train
```

| Pros | Cons |
|------|------|
| Complete product | Lots of work |
| Best for end users | Premature optimization |
| Single install | Hard to maintain |
| Clear vision | Delays shipping |

**Verdict**: This is the goal, but premature now. Build towards it.

---

### Option D: No "openadapt" Package

**Description**: Users install individual packages. openadapt-ml is the "main" one.

```bash
# Users install what they need
pip install openadapt-ml
pip install openadapt-capture
pip install openadapt-evals
```

| Pros | Cons |
|------|------|
| Simplest | Confusing for newcomers |
| No coordination needed | No unified entry point |
| Minimal overhead | Have to know package names |
| Flexible | Fragmented experience |

**Verdict**: Works for developers but bad UX for newcomers.

---

### Option B+: Thin CLI Wrapper with Progressive Enhancement (RECOMMENDED)

**Description**: Start with Option B but design it to evolve toward Option C.

**Phase 1 (Now)**: Thin wrapper
```
openadapt/
├── pyproject.toml
├── src/openadapt/
│   ├── __init__.py      # Re-exports
│   ├── cli.py           # Unified CLI
│   └── config.py        # Shared config
└── README.md
```

**Phase 2 (When ready)**: Add orchestration
```
openadapt/
├── src/openadapt/
│   ├── ...
│   ├── orchestrator/    # Agent loop
│   │   ├── loop.py      # Capture -> train -> deploy
│   │   └── scheduler.py
│   └── server/          # REST API
│       ├── app.py
│       └── routes.py
```

**Phase 3 (Product launch)**: Add GUI
```
openadapt/
├── src/openadapt/
│   ├── ...
│   ├── app/             # Desktop GUI
│   │   ├── main.py
│   │   └── ...
│   └── web/             # Web interface
│       ├── frontend/
│       └── backend/
```

**Key design principles**:
1. **CLI first**: Everything accessible via CLI
2. **Server optional**: `openadapt serve` exposes REST API
3. **GUI optional**: `openadapt app` opens GUI (when ready)
4. **Progressive disclosure**: Basic use is simple, power features available

---

## 4. Recommended Approach

### 4.1 Final Recommendation: Option B+

**Summary**: Create a thin `openadapt` meta-package that:
1. Depends on core packages (openadapt-ml, openadapt-capture, openadapt-evals, openadapt-viewer)
2. Provides a unified CLI
3. Re-exports common items for convenience
4. Has optional dependencies for grounding and retrieval
5. Is designed to grow into a full application over time

### 4.2 Proposed Package Structure

```
openadapt/
├── pyproject.toml
├── README.md
├── LICENSE (MIT)
├── CHANGELOG.md
├── src/openadapt/
│   ├── __init__.py          # Re-exports
│   ├── cli.py               # Unified CLI
│   ├── config.py            # Shared configuration
│   └── version.py           # Version info
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── architecture.md
│   └── tutorials/
├── examples/
│   ├── 01_basic_capture/
│   ├── 02_train_model/
│   ├── 03_evaluate/
│   └── 04_demo_retrieval/
└── tests/
    └── test_cli.py
```

### 4.3 Proposed CLI Design

```bash
# Installation
pip install openadapt              # Core packages
pip install openadapt[all]         # Everything
pip install openadapt[grounding]   # Add grounding
pip install openadapt[retrieval]   # Add retrieval

# Capture workflow
openadapt capture start --name "my-task"
openadapt capture stop
openadapt capture list
openadapt capture view my-task

# Training workflow
openadapt train --capture my-task --model qwen3vl-2b
openadapt train status
openadapt train stop

# Evaluation workflow
openadapt eval --checkpoint model.pt --benchmark waa --tasks 10
openadapt eval --agent api-claude --benchmark waa

# Serving
openadapt serve --port 8080        # Web dashboard
openadapt serve --api-only         # REST API only

# Utilities
openadapt version                  # Show all package versions
openadapt doctor                   # Check system requirements
openadapt config show              # Show configuration
```

### 4.4 Proposed `__init__.py` Re-exports

```python
"""OpenAdapt - GUI automation with ML."""

# Version
from openadapt.version import __version__

# From openadapt-ml (core)
from openadapt_ml.runtime import AgentPolicy, SafetyGate
from openadapt_ml.models import QwenVLAdapter, APIAdapter
from openadapt_ml.schema import Episode, Step, Action, Observation

# From openadapt-capture
from openadapt_capture import CaptureSession

# From openadapt-evals
from openadapt_evals import (
    evaluate_agent_on_benchmark,
    ApiAgent,
    WAAAdapter,
)

# From openadapt-viewer
from openadapt_viewer import PageBuilder, generate_benchmark_html

# Optional: grounding
try:
    from openadapt_grounding import OmniParser, UITarsGrounder
except ImportError:
    pass

# Optional: retrieval
try:
    from openadapt_retrieval import MultimodalDemoRetriever
except ImportError:
    pass

__all__ = [
    "__version__",
    # Core
    "AgentPolicy",
    "SafetyGate",
    "QwenVLAdapter",
    "APIAdapter",
    "Episode",
    "Step",
    "Action",
    "Observation",
    # Capture
    "CaptureSession",
    # Evals
    "evaluate_agent_on_benchmark",
    "ApiAgent",
    "WAAAdapter",
    # Viewer
    "PageBuilder",
    "generate_benchmark_html",
]
```

### 4.5 Proposed `pyproject.toml`

```toml
[project]
name = "openadapt"
version = "0.1.0"
description = "GUI automation with ML - record, train, deploy, evaluate"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [
    {name = "MLDSAI Inc.", email = "richard@mldsai.com"}
]
keywords = ["gui", "automation", "ml", "rpa", "agent", "vlm"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "openadapt-ml>=0.2.0",
    "openadapt-capture>=0.1.0",
    "openadapt-evals>=0.1.0",
    "openadapt-viewer>=0.1.0",
]

[project.optional-dependencies]
grounding = [
    "openadapt-grounding>=0.1.0",
]
retrieval = [
    "openadapt-retrieval>=0.1.0",
]
all = [
    "openadapt[grounding,retrieval]",
]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
openadapt = "openadapt.cli:main"

[project.urls]
Homepage = "https://openadapt.ai"
Documentation = "https://docs.openadapt.ai"
Repository = "https://github.com/OpenAdaptAI/openadapt"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 5. Migration Path from Legacy OpenAdapt

### 5.1 Current State

```
OpenAdapt (legacy)
├── openadapt/         # Monolithic codebase
│   ├── capture/       # -> openadapt-capture
│   ├── models/        # -> openadapt-ml
│   ├── training/      # -> openadapt-ml
│   └── privacy/       # -> openadapt-privacy
└── ...
```

### 5.2 Migration Steps

1. **Archive legacy** as `openadapt-legacy`
   ```bash
   # Rename repo
   gh repo rename OpenAdaptAI/OpenAdapt OpenAdaptAI/openadapt-legacy

   # Update README
   echo "This repo is archived. Use pip install openadapt instead." > README.md
   ```

2. **Create new `openadapt` repo**
   ```bash
   gh repo create OpenAdaptAI/openadapt --public
   ```

3. **Publish packages to PyPI** (in order)
   ```bash
   # 1. Core packages (no inter-dependencies)
   cd openadapt-capture && uvx twine upload dist/*
   cd openadapt-viewer && uvx twine upload dist/*
   cd openadapt-grounding && uvx twine upload dist/*
   cd openadapt-retrieval && uvx twine upload dist/*

   # 2. Depends on capture
   cd openadapt-ml && uvx twine upload dist/*

   # 3. Depends on nothing (but optionally uses ml)
   cd openadapt-evals && uvx twine upload dist/*

   # 4. Meta-package
   cd openadapt && uvx twine upload dist/*
   ```

4. **Update documentation**
   - Create `docs.openadapt.ai` with unified docs
   - Add migration guide for legacy users

5. **Communication**
   - Blog post announcing new architecture
   - GitHub Discussions announcement
   - Update all READMEs to point to new structure

### 5.3 For Legacy Users

```python
# Old (legacy openadapt)
from openadapt.capture import capture_session
from openadapt.models import train_model

# New (modular openadapt)
from openadapt import CaptureSession  # From openadapt-capture
from openadapt_ml.training import train_with_trl  # From openadapt-ml

# Or using meta-package
from openadapt import AgentPolicy, CaptureSession
```

---

## 6. Timeline Considerations

### 6.1 What's Ready Now

| Package | PyPI Status | Production Ready? |
|---------|-------------|-------------------|
| openadapt-ml | Published (v0.2.0) | Yes (core) |
| openadapt-capture | Published (v0.1.0) | Yes |
| openadapt-evals | Ready | Yes |
| openadapt-viewer | Ready | Yes |
| openadapt-grounding | Ready | Beta |
| openadapt-retrieval | Ready | Beta |

### 6.2 MVP Timeline

**Week 1-2: Foundation**
- [ ] Create `openadapt` repo
- [ ] Implement thin CLI wrapper
- [ ] Set up PyPI publishing
- [ ] Basic documentation

**Week 3-4: Polish**
- [ ] End-to-end examples
- [ ] Integration tests
- [ ] Unified docs site
- [ ] Blog post / announcement

### 6.3 What Can Wait

| Feature | When | Why Wait |
|---------|------|----------|
| GUI application | After product-market fit | Need to validate workflows first |
| Agent orchestration | After eval framework stable | Need benchmark results first |
| Model registry | After training stable | Need fine-tuned models first |
| Web dashboard | After CLI validated | CLI-first approach |

### 6.4 Is Building the Frontend Premature?

**Yes, a full GUI is premature.** Here's why:

1. **We don't know the workflows yet**: Until we have more real users, we don't know what the ideal workflow is. CLI lets us iterate faster.

2. **Core ML isn't done**: Training pipeline, evaluation, and demo retrieval are still evolving. GUI would lock us into current abstractions.

3. **Developer focus**: Our current users are developers who prefer CLI/API over GUI.

**What we DO need now**:
- Unified CLI for discoverability
- Web dashboard for viewing results (openadapt-viewer handles this)
- REST API for integration (can add to openadapt later)

---

## 7. Decision Matrix

| Criteria | Option A | Option B | Option C | Option D | **Option B+** |
|----------|----------|----------|----------|----------|---------------|
| User onboarding | Poor | Good | Best | Poor | Good |
| Maintenance burden | None | Low | High | None | Low |
| Developer experience | Poor | Good | Good | Best | Good |
| Newcomer experience | Poor | Good | Best | Poor | Good |
| Time to ship | Instant | 2 weeks | 3+ months | Instant | 2 weeks |
| Scalability | N/A | Good | Good | N/A | Best |
| Future GUI path | No | Yes | Yes | No | **Yes** |

**Recommendation**: **Option B+** provides the best balance of quick shipping, good UX, and future extensibility.

---

## 8. Sources

- [HuggingFace Transformers v5 Blog](https://huggingface.co/blog/transformers-v5)
- [LangChain Architecture Documentation](https://python.langchain.com/docs/concepts/architecture/)
- [PyTorch Ecosystem Overview](https://pytorch.org/)
- [AutoGPT Documentation](https://docs.agpt.co/)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [Transformers InfoQ Article](https://www.infoq.com/news/2025/12/transformers-hugging-face/)

---

## Appendix A: Package Comparison Table

| Package | HF Equivalent | LangChain Equivalent | PyTorch Equivalent |
|---------|---------------|---------------------|-------------------|
| openadapt | transformers (hub) | langchain (main) | torch |
| openadapt-ml | transformers | langchain-core | torch |
| openadapt-capture | datasets | N/A | torchaudio |
| openadapt-evals | evaluate | N/A | torchmetrics |
| openadapt-viewer | gradio | N/A | tensorboard |
| openadapt-grounding | N/A | langchain-community | torchvision |
| openadapt-retrieval | faiss | langchain-community | N/A |

---

## Appendix B: Alternative Considered - Monorepo

We considered a monorepo structure (all packages in one repo) but rejected it because:

1. **Different release cadences**: openadapt-ml changes faster than openadapt-capture
2. **Different dependencies**: openadapt-grounding needs AWS deps, others don't
3. **Team specialization**: Different contributors focus on different packages
4. **CI/CD complexity**: Monorepo requires complex build matrix

The current multi-repo approach with a thin meta-package provides the flexibility we need while maintaining a unified user experience.
