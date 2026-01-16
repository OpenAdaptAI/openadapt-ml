# OpenAdapt Integration Plan

## Executive Summary

This document outlines a comprehensive plan to integrate the main [OpenAdapt repository](https://github.com/OpenAdaptAI/OpenAdapt) with the new modular package ecosystem. The goal is to have the core `openadapt` package import from specialized libraries rather than duplicating functionality, enabling better maintainability, clearer separation of concerns, and independent release cycles.

## 1. Current Architecture of OpenAdapt

### 1.1 Overview

OpenAdapt is an open-source Generative Process Automation (GPA) framework that serves as an adapter between Large Multimodal Models (LMMs) and traditional desktop/web GUIs. It records user interactions, processes them through AI models, and generates executable process descriptions.

### 1.2 Core Components

The main `openadapt` package (version 0.46.0) contains the following key modules:

| Module | Description | Lines of Code |
|--------|-------------|---------------|
| `record.py` | Screen/input recording with screenshot capture | ~1,400 |
| `models.py` | SQLAlchemy database models (ActionEvent, Screenshot, etc.) | ~1,200 |
| `events.py` | Event merging and processing | ~900 |
| `vision.py` | Computer vision utilities (mask processing, SSIM) | ~500 |
| `visualize.py` | Recording visualization | ~400 |
| `scrub.py` | PII/PHI privacy scrubbing | ~350 |
| `browser.py` | Chrome extension integration | ~1,000 |
| `utils.py` | General utilities | ~1,000 |
| `plotting.py` | Plotting utilities for visualization | ~800 |

### 1.3 Replay Strategies

Located in `openadapt/strategies/`:

| Strategy | Description | Key Dependencies |
|----------|-------------|-----------------|
| `NaiveReplayStrategy` | Direct action replay | None |
| `StatefulReplayStrategy` | GPT-4 API with OS-level prompts | OpenAI API |
| `VanillaReplayStrategy` | Direct model reasoning | OpenAI/Anthropic |
| `VisualReplayStrategy` | FastSAM segmentation for GUI | FastSAM, SAM |
| `VisualBrowserReplayStrategy` | DOM-based browser segments | Chrome extension |
| `SegmentReplayStrategy` | Segment Anything integration | SAM, Replicate |

### 1.4 Adapters (LLM/Segmentation Providers)

Located in `openadapt/adapters/`:

| Adapter | Purpose |
|---------|---------|
| `prompt.py` | LLM prompt construction |
| `replicate.py` | Replicate API for SAM |
| `som.py` | Set-of-Mark prompting |
| `ultralytics.py` | Ultralytics YOLO/FastSAM |

### 1.5 Key Dependencies

From `pyproject.toml`:
- **AI/ML**: openai, anthropic, torch, transformers, spacy, segment-anything, ultralytics
- **Vision/OCR**: pytesseract, rapidocr-onnxruntime, easyocr, pillow
- **Database**: alembic, dictalchemy3, SQLAlchemy
- **Web**: fastapi, uvicorn, gradio-client
- **Privacy**: presidio_analyzer, presidio_anonymizer, boto3

---

## 2. New Package Ecosystem

### 2.1 openadapt-ml

**Purpose**: Model-agnostic, domain-agnostic ML engine for GUI automation agents

**Key Features**:
- Schemas for GUI interaction trajectories (`openadapt_ml/schema/`)
- VLM adapters (Qwen3-VL, Qwen2.5-VL, API backends)
- Supervised fine-tuning pipeline (`openadapt_ml/training/`)
- Runtime policy API (`openadapt_ml/runtime/`)
- Cloud GPU training (Lambda Labs, Azure) (`openadapt_ml/cloud/`)
- Benchmark integration (WAA) (`openadapt_ml/benchmarks/`)

**Python Version**: >=3.12

### 2.2 openadapt-grounding

**Purpose**: Robust UI element localization for automation

**Key Features**:
- OmniParser integration (`parsers/omniparser.py`)
- UI-TARS integration (`parsers/uitars.py`)
- VLM providers (Anthropic, OpenAI, Google) (`providers/`)
- AWS EC2 deployment automation (`deploy/`)
- Evaluation framework (`eval/`)

**Python Version**: >=3.10

### 2.3 openadapt-evals

**Purpose**: Benchmark evaluation infrastructure for GUI agents

**Key Features**:
- Unified benchmark adapters (`adapters/`)
- Agent implementations (API, Policy, Scripted) (`agents/`)
- WAA (Windows Agent Arena) integration
- Azure parallel evaluation
- Benchmark viewer generation (`benchmarks/viewer.py`)

**Python Version**: >=3.10

### 2.4 openadapt-viewer

**Purpose**: Standalone HTML viewer generation for dashboards and benchmarks

**Key Features**:
- Training dashboard generation
- Benchmark result visualization
- Step-by-step screenshot playback
- Jinja2 templates with Plotly charts
- Component library (`components/`)

**Python Version**: >=3.10

### 2.5 openadapt-retrieval

**Purpose**: Multimodal demo retrieval using VLM embeddings

**Key Features**:
- Qwen3-VL-Embedding support (`embeddings/qwen3vl.py`)
- CLIP fallback (`embeddings/clip.py`)
- FAISS vector index (`retriever/index.py`)
- Demo library management (`retriever/demo_retriever.py`)
- MRL (Matryoshka Representation Learning) for flexible dimensions

**Python Version**: >=3.10

---

## 3. Integration Opportunities

### 3.1 openadapt + openadapt-ml

| OpenAdapt Component | Integration Opportunity |
|---------------------|------------------------|
| `strategies/visual.py` | Replace custom segmentation with `openadapt_ml.grounding` |
| `strategies/stateful.py` | Use `openadapt_ml.runtime` for model inference |
| `adapters/prompt.py` | Migrate to `openadapt_ml.models.adapters` |
| `vision.py` | Share mask processing with `openadapt_ml.perception` |
| Training (not present) | Add training via `openadapt_ml.training` |

**Proposed Changes**:

```python
# Before (openadapt/strategies/visual.py)
from openadapt import adapters, vision
# Custom segmentation and prompting

# After
from openadapt_ml.runtime import PolicyRunner
from openadapt_ml.grounding import GroundingEngine
```

### 3.2 openadapt + openadapt-grounding

| OpenAdapt Component | Integration Opportunity |
|---------------------|------------------------|
| `adapters/som.py` | Replace with `openadapt_grounding.parsers.omniparser` |
| `adapters/ultralytics.py` | Replace with `openadapt_grounding.parsers` |
| `adapters/replicate.py` | Replace with `openadapt_grounding.deploy` |
| `strategies/segment.py` | Use unified `GroundingEngine` |
| EC2 SAM deployment | Migrate to `openadapt_grounding.deploy` |

**Proposed Changes**:

```python
# Before (openadapt/adapters/__init__.py)
def get_default_segmentation_adapter():
    return {"som": som, "replicate": replicate, "ultralytics": ultralytics}[...]

# After
from openadapt_grounding import get_grounder
grounder = get_grounder("omniparser")  # or "uitars", "vlm"
```

### 3.3 openadapt + openadapt-viewer

| OpenAdapt Component | Integration Opportunity |
|---------------------|------------------------|
| `visualize.py` | Replace with `openadapt_viewer.viewers.benchmark` |
| `plotting.py` | Replace with `openadapt_viewer.components` |
| Web dashboard (app/) | Share components with viewer |

**Proposed Changes**:

```python
# Before (openadapt/visualize.py)
def main():
    # Generate custom HTML

# After
from openadapt_viewer import generate_recording_html
html = generate_recording_html(recording, embed_screenshots=True)
```

### 3.4 openadapt + openadapt-retrieval

| OpenAdapt Component | Integration Opportunity |
|---------------------|------------------------|
| Demo replay (strategies/demo.py) | Use `MultimodalDemoRetriever` for demo selection |
| Visual similarity (vision.py) | Use VLM embeddings for matching |
| Recording search (none) | Add semantic search via retrieval |

**Proposed Changes**:

```python
# Before: No automatic demo selection

# After
from openadapt_retrieval import MultimodalDemoRetriever

retriever = MultimodalDemoRetriever(embedding_dim=512)
retriever.load("demo_library/")
relevant_demos = retriever.retrieve(
    task=current_task,
    screenshot=current_screenshot,
    top_k=3
)
```

### 3.5 openadapt + openadapt-evals

| OpenAdapt Component | Integration Opportunity |
|---------------------|------------------------|
| No benchmarking | Add evaluation via `openadapt_evals` |
| Replay strategies | Wrap as `BenchmarkAgent` for evaluation |
| Metrics (none) | Use `openadapt_evals.metrics` |

**Proposed Changes**:

```python
# New capability
from openadapt_evals import evaluate_agent_on_benchmark, WAALiveAdapter

class OpenAdaptAgent(BenchmarkAgent):
    def __init__(self, strategy: BaseReplayStrategy):
        self.strategy = strategy

    def predict_action(self, observation):
        return self.strategy.get_next_action(observation)

agent = OpenAdaptAgent(VisualReplayStrategy())
results = evaluate_agent_on_benchmark(agent, WAALiveAdapter(server_url))
```

---

## 4. Migration Path

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Establish package boundaries and backward compatibility layer

1. **Version Alignment**
   - Update openadapt to Python >=3.10 (current: 3.10-3.11)
   - Ensure all new packages have compatible version ranges

2. **Add Dependencies**
   ```toml
   # pyproject.toml additions
   openadapt-grounding = ">=0.1.0"
   openadapt-viewer = ">=0.1.0"
   # Optional:
   openadapt-ml = {version = ">=0.2.0", python = ">=3.12", optional = true}
   openadapt-retrieval = {version = ">=0.1.0", optional = true}
   openadapt-evals = {version = ">=0.1.0", optional = true}
   ```

3. **Create Compatibility Shims**
   ```python
   # openadapt/adapters/som.py
   import warnings
   warnings.warn(
       "openadapt.adapters.som is deprecated. "
       "Use openadapt_grounding instead.",
       DeprecationWarning
   )
   from openadapt_grounding.parsers.omniparser import OmniParser as SoMAdapter
   ```

### Phase 2: Grounding Integration (Weeks 3-4)

**Goal**: Replace segmentation adapters with openadapt-grounding

1. **Migrate Adapters**
   - `openadapt/adapters/som.py` -> `openadapt_grounding.parsers.omniparser`
   - `openadapt/adapters/ultralytics.py` -> `openadapt_grounding.parsers.uitars`
   - `openadapt/adapters/replicate.py` -> `openadapt_grounding.deploy`

2. **Update Strategies**
   - Modify `VisualReplayStrategy` to use `openadapt_grounding`
   - Modify `SegmentReplayStrategy` to use unified interface

3. **Deploy Script Migration**
   - Move EC2 SAM deployment to `openadapt_grounding.deploy`
   - Update CLI commands

### Phase 3: Viewer Integration (Weeks 5-6)

**Goal**: Consolidate visualization

1. **Migrate Visualization**
   - Replace `visualize.py` with `openadapt_viewer` calls
   - Migrate `plotting.py` utilities

2. **Dashboard Unification**
   - Share header/nav components between viewer and dashboard
   - Standardize CSS/JS

3. **Remove Duplicated Code**
   - Remove redundant HTML generation
   - Update imports in `openadapt/app/`

### Phase 4: ML/Training Integration (Weeks 7-9)

**Goal**: Add training and fine-tuning capability

1. **Schema Alignment**
   - Map OpenAdapt `ActionEvent` to `openadapt_ml.schema.Action`
   - Create converters for `Screenshot` -> `openadapt_ml.schema.Observation`

2. **Training Integration**
   - Add `openadapt train` command using `openadapt_ml.training`
   - Support cloud training via `openadapt_ml.cloud`

3. **Runtime Integration**
   - Add `PolicyReplayStrategy` using `openadapt_ml.runtime`
   - Support fine-tuned model inference

### Phase 5: Retrieval Integration (Weeks 10-11)

**Goal**: Add demo retrieval capability

1. **Demo Library**
   - Create `openadapt/demo_library/` directory structure
   - Index existing recordings as demos

2. **Retrieval Integration**
   - Add demo retrieval to replay strategies
   - Support "find similar demo" in dashboard

3. **Demo-Conditioned Prompting**
   - Integrate retrieval with `VisualReplayStrategy`
   - Include retrieved demo in LLM prompts

### Phase 6: Evaluation Integration (Weeks 12-13)

**Goal**: Add benchmarking capability

1. **Agent Wrapper**
   - Create `OpenAdaptBenchmarkAgent` wrapping replay strategies
   - Support WAA, WebArena evaluation

2. **CLI Integration**
   - Add `openadapt eval` command
   - Support `--benchmark waa --strategy visual`

3. **Results Dashboard**
   - Integrate benchmark viewer into OpenAdapt dashboard
   - Show evaluation metrics alongside recordings

---

## 5. Breaking Changes

### 5.1 API Changes

| Component | Breaking Change | Migration Path |
|-----------|-----------------|----------------|
| `adapters.get_default_segmentation_adapter()` | Returns new type | Use `openadapt_grounding.get_grounder()` |
| `adapters.som` | Module deprecated | Import from `openadapt_grounding` |
| `adapters.replicate` | Module deprecated | Use `openadapt_grounding.deploy` |
| `visualize.main()` | Signature change | Use `openadapt_viewer.generate_recording_html()` |

### 5.2 Configuration Changes

```python
# Before (config.py)
DEFAULT_SEGMENTATION_ADAPTER = "som"

# After
GROUNDING_PROVIDER = "omniparser"  # or "uitars", "vlm"
GROUNDING_ENDPOINT = "http://localhost:8000"  # if self-hosted
```

### 5.3 Import Path Changes

```python
# Deprecated imports (still work with warnings)
from openadapt.adapters import som
from openadapt import visualize

# New imports
from openadapt_grounding import OmniParser
from openadapt_viewer import generate_recording_html
```

---

## 6. Backward Compatibility Strategy

### 6.1 Deprecation Warnings

All deprecated modules will emit `DeprecationWarning` for at least 2 minor versions:

```python
# openadapt/adapters/som.py
import warnings
warnings.warn(
    "openadapt.adapters.som is deprecated and will be removed in v0.50.0. "
    "Use openadapt_grounding.parsers.omniparser instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### 6.2 Version Timeline

| Version | Status |
|---------|--------|
| v0.47.0 | Add new package dependencies, deprecation warnings |
| v0.48.0 | Grounding + Viewer integration complete |
| v0.49.0 | ML + Retrieval integration complete |
| v0.50.0 | Remove deprecated modules |
| v0.51.0 | Evaluation integration complete |

### 6.3 Optional Dependencies

Heavy dependencies (ML, retrieval) are optional to keep core package lightweight:

```toml
[project.optional-dependencies]
ml = ["openadapt-ml>=0.2.0"]
retrieval = ["openadapt-retrieval>=0.1.0"]
eval = ["openadapt-evals>=0.1.0"]
full = ["openadapt[ml,retrieval,eval]"]
```

---

## 7. Blockers and Dependencies

### 7.1 Immediate Blockers

| Blocker | Package | Resolution |
|---------|---------|------------|
| Python version mismatch | openadapt-ml requires >=3.12, openadapt requires <3.12 | Update openadapt to support 3.12 |
| Schema incompatibility | Different Action/Observation definitions | Create schema mapping layer |
| Package not on PyPI | New packages not published | Publish to PyPI before integration |

### 7.2 Circular Dependency Concerns

**Potential Issue**: If `openadapt-ml` imports from `openadapt` for database models, circular dependency occurs.

**Resolution**:
1. `openadapt-ml` defines its own schema (`openadapt_ml/schema/`)
2. `openadapt` provides converters to/from `openadapt-ml` schema
3. No direct import of `openadapt` from any sub-package

**Dependency Graph** (allowed):
```
openadapt
  ├── imports openadapt-grounding
  ├── imports openadapt-viewer
  ├── imports openadapt-retrieval
  ├── imports openadapt-ml (optional)
  └── imports openadapt-evals (optional)

openadapt-evals
  └── imports openadapt-viewer (for HTML generation)
```

### 7.3 Version Compatibility

| Package | Current Python | Target Python |
|---------|---------------|---------------|
| openadapt | 3.10-3.11 | 3.10-3.12 |
| openadapt-ml | 3.12+ | 3.12+ |
| openadapt-grounding | 3.10+ | 3.10+ |
| openadapt-viewer | 3.10+ | 3.10+ |
| openadapt-retrieval | 3.10+ | 3.10+ |
| openadapt-evals | 3.10+ | 3.10+ |

**Action Required**: Update `openadapt` to support Python 3.12 before ML integration.

---

## 8. Technical Debt to Address

### 8.1 In OpenAdapt

1. **Vision module** (`vision.py`): Contains mask processing that should move to `openadapt-grounding`
2. **Plotting module** (`plotting.py`): Should migrate to `openadapt-viewer`
3. **Adapters directory**: Should be refactored to thin wrappers around `openadapt-grounding`

### 8.2 In New Packages

1. **openadapt-ml**:
   - `benchmarks/` module is being migrated to `openadapt-evals` (deprecation stubs in place)
   - WAA Docker setup should be shared with core openadapt

2. **openadapt-grounding**:
   - Deployment scripts assume AWS; should support other clouds

3. **openadapt-viewer**:
   - Currently focused on benchmarks; needs recording visualization support

---

## 9. Testing Strategy

### 9.1 Integration Tests

```python
# tests/integration/test_grounding_integration.py
def test_visual_replay_with_openadapt_grounding():
    """Verify VisualReplayStrategy works with openadapt-grounding."""
    from openadapt.strategies import VisualReplayStrategy
    from openadapt_grounding import OmniParser

    strategy = VisualReplayStrategy(grounder=OmniParser())
    result = strategy.get_next_action(screenshot)
    assert result is not None
```

### 9.2 Backward Compatibility Tests

```python
# tests/compatibility/test_deprecated_imports.py
def test_deprecated_som_import():
    """Verify deprecated imports still work with warning."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from openadapt.adapters import som
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()
```

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Code duplication | <10% overlap | LOC comparison |
| Test coverage | >80% on integration points | pytest-cov |
| Import time | <2s for core, <5s with all extras | timeit |
| Memory usage | No regression | pytest-memray |
| Documentation | 100% of public APIs | sphinx coverage |

---

## 11. Appendix: Package Responsibility Matrix

| Capability | openadapt | openadapt-ml | openadapt-grounding | openadapt-viewer | openadapt-retrieval | openadapt-evals |
|------------|-----------|--------------|--------------------|-----------------|--------------------|-----------------|
| Recording | **Primary** | - | - | - | - | - |
| Replay | **Primary** | Runtime | Grounding | - | Demo selection | - |
| Training | - | **Primary** | - | - | - | - |
| Segmentation | - | - | **Primary** | - | - | - |
| Visualization | Dashboard | - | - | **Primary** | - | - |
| Embeddings | - | - | - | - | **Primary** | - |
| Benchmarking | - | - | - | - | - | **Primary** |
| Privacy/Scrub | **Primary** | - | - | - | - | - |
| Database | **Primary** | Schema | - | - | - | - |
| CLI | **Primary** | CLI | CLI | CLI | CLI | CLI |

---

## 12. Conclusion

This integration plan provides a structured approach to consolidating the OpenAdapt ecosystem. By following this phased approach:

1. **Phase 1-2** (Foundation + Grounding): Establishes the groundwork with backward compatibility
2. **Phase 3** (Viewer): Unifies visualization across all packages
3. **Phase 4** (ML): Adds training capability without breaking existing workflows
4. **Phase 5** (Retrieval): Enhances replay with demo retrieval
5. **Phase 6** (Evaluation): Enables benchmarking and continuous improvement

The key principles are:
- **Maintain backward compatibility** through deprecation warnings
- **Keep core lightweight** with optional dependencies
- **Avoid circular dependencies** through schema separation
- **Enable independent releases** for each package

This architecture positions OpenAdapt for:
- Enterprise integrations with clear API boundaries
- Research experimentation with modular components
- Community contributions with well-defined package responsibilities
