# OpenAdaptAI GitHub Organization Profile Content

This document contains recommended content for updating the OpenAdaptAI GitHub organization profile to highlight the new modular ecosystem.

---

## 1. Organization Bio (max 160 chars)

**Recommended:**
```
Open source AI-powered desktop automation. Record workflows, train models, replay intelligently. Modular ecosystem for GUI automation agents.
```
(138 characters)

**Alternative (shorter):**
```
Open source AI desktop automation. Learn from demonstrations, automate intelligently.
```
(86 characters)

---

## 2. Organization README Content

The following content is recommended for `.github/profile/README.md`:

```markdown
# Welcome to OpenAdapt.AI

**OpenAdapt.AI** is an open source **Generative Process Automation** platform that transforms desktop automation through machine learning. Record human demonstrations, train vision-language models, and deploy agents that adapt to any software environment.

## What is OpenAdapt.AI?

OpenAdapt.AI leverages advanced machine learning models to:

- **Observe and Learn**: Automatically learns from user interactions to generate automation scripts
- **Automate Intelligently**: Adapts to software changes, varying workflows, and complex environments
- **Ensure Privacy**: Built-in data protection with support for AWS Rekognition, Microsoft Presidio, and Private-ai.com

## Key Features

- **Generative Process Automation**: Automates tasks using generative AI models that learn from demonstrations
- **Cross-Platform Compatibility**: Works with browser apps (Chrome) and desktop applications
- **Audio Narration Transcription**: Converts spoken instructions into automation scripts
- **Flexible Model Providers**: Supports OpenAI, Anthropic, Google (online) and Ollama, vLLM (offline)
- **Open Source**: Developed transparently and collaboratively by the community
- **Privacy by Design**: Robust data handling mechanisms to protect user data

## Package Ecosystem

OpenAdapt.AI is built as a **modular ecosystem** of specialized packages:

| Package | Description | Status |
|---------|-------------|--------|
| [**openadapt**](https://github.com/OpenAdaptAI/openadapt) | Core platform - unified CLI, recording, training, evaluation | Production |
| [**openadapt-ml**](https://github.com/OpenAdaptAI/openadapt-ml) | ML engine for training GUI automation models | Active Development |
| [**openadapt-capture**](https://github.com/OpenAdaptAI/openadapt-capture) | Cross-platform screen and input event recording | Active Development |
| [**openadapt-grounding**](https://github.com/OpenAdaptAI/openadapt-grounding) | Visual grounding with OmniParser and UI-TARS | Active Development |
| [**openadapt-evals**](https://github.com/OpenAdaptAI/openadapt-evals) | Benchmark evaluation (WAA, WebArena) | Active Development |
| [**openadapt-viewer**](https://github.com/OpenAdaptAI/openadapt-viewer) | Reusable UI components for visualization | Active Development |
| [**openadapt-retrieval**](https://github.com/OpenAdaptAI/openadapt-retrieval) | Multimodal demo retrieval (Qwen3-VL, CLIP, FAISS) | Active Development |

### Architecture Overview

```
                    +-----------------+
                    |    openadapt    |  <-- Unified CLI and meta-package
                    +-----------------+
                             |
              +--------------+---------------+
              |              |               |
              v              v               v
    +----------------+ +-----------+ +----------------+
    |openadapt-capture| |openadapt-| |openadapt-evals |
    |   (recording)   | |    ml    | |  (benchmarks)  |
    +----------------+ | (training)| +----------------+
                       +-----------+
                             |
              +--------------+---------------+
              |              |               |
              v              v               v
    +----------------+ +----------------+ +----------------+
    |openadapt-viewer| |openadapt-      | |openadapt-      |
    |(visualization) | |   grounding    | |   retrieval    |
    +----------------+ | (perception)   | | (demo search)  |
                       +----------------+ +----------------+
```

## Getting Started

### Quick Start

```bash
# Install with pip
pip install openadapt

# Or use uv (recommended)
uv tool install openadapt

# Record a demonstration
openadapt capture start --name my-task
# ... perform the task ...
openadapt capture stop

# Train a model
openadapt train --capture ./captures/my-task

# Evaluate on benchmarks
openadapt eval run --benchmark waa-mock --tasks 20
```

### Individual Packages

Each package can be installed independently:

```bash
pip install openadapt           # Core CLI and unified interface
pip install openadapt-ml        # ML training engine
pip install openadapt-capture   # Screen recording
pip install openadapt-grounding # UI element localization
pip install openadapt-evals     # Benchmark evaluation
pip install openadapt-retrieval # Demo similarity search
```

See each package's README for detailed documentation.

## Contributing

We welcome contributions from everyone. Whether you are a developer, data scientist, tester, or just passionate about automation, there is a place for you here.

- **[Contributing Guide](https://github.com/OpenAdaptAI/openadapt/blob/main/CONTRIBUTING.md)**
- **[Code of Conduct](https://github.com/OpenAdaptAI/openadapt/blob/main/CODE_OF_CONDUCT.md)**

Join our [Community](https://github.com/OpenAdaptAI/openadapt/discussions) to discuss new ideas, report bugs, or seek help.

**Areas of interest:**
- Frontend development (React, visualization)
- Machine learning (model training, evaluation)
- Desktop integration (Windows, macOS, Linux)
- Benchmark development and evaluation
- Documentation and tutorials

## Implementation Consulting

Looking to implement OpenAdapt.AI in your organization? We offer expert consulting services for deployment, customization, and optimization.

Contact us at [sales@openadapt.ai](mailto:sales@openadapt.ai).

## Sponsors and Partners

OpenAdapt.AI is made possible by the support of our contributors and partners. If you are interested in sponsoring development, check out our [GitHub Sponsors page](https://github.com/sponsors/OpenAdaptAI).

## Contact

For general questions, feedback, or partnership inquiries: [contact@openadapt.ai](mailto:contact@openadapt.ai)

## Disclaimer

OpenAdapt.AI is currently in alpha. While functional, some features are still under development. Use with caution in production environments.
```

---

## 3. Pinned Repositories Recommendation

Pin the following 6 repositories in this order:

| Order | Repository | Rationale |
|-------|------------|-----------|
| 1 | **openadapt** | Main entry point, unified CLI, meta-package for the ecosystem |
| 2 | **openadapt-ml** | Core ML engine, foundation for training GUI automation models |
| 3 | **openadapt-capture** | Recording component, essential for data collection |
| 4 | **openadapt-evals** | Benchmark evaluation, demonstrates research credibility |
| 5 | **openadapt-grounding** | Visual perception layer, key differentiator with OmniParser/UI-TARS |
| 6 | **openadapt-retrieval** | Demo retrieval capability, enables demo-conditioned prompting |

**Note:** The legacy monolithic `OpenAdapt` repository (1,470 stars) should be renamed or archived once migration is complete. The new `openadapt` package (currently `openadapt-new`) becomes the primary entry point.

---

## 4. Repository Descriptions (max 350 chars each)

### openadapt-ml
```
ML engine for training multimodal GUI automation models. Provides VLM adapters (Qwen3-VL, API backends), supervised fine-tuning pipelines, GRPO training, and runtime policy APIs. Part of the OpenAdapt ecosystem.
```
(211 characters)

### openadapt-capture
```
Cross-platform screen and input event recording for GUI automation. Captures screenshots, mouse events, keyboard input, and accessibility data. Exports to training-ready formats. Part of the OpenAdapt ecosystem.
```
(213 characters)

### openadapt-evals
```
Benchmark evaluation framework for GUI automation agents. Supports Windows Agent Arena (WAA), WebArena, and custom benchmarks. Includes mock adapters for testing and data collection. Part of the OpenAdapt ecosystem.
```
(217 characters)

### openadapt-viewer
```
React component library for GUI automation visualization. Provides recording playback, action annotation, step-by-step replay, and benchmark result dashboards. Part of the OpenAdapt ecosystem.
```
(193 characters)

### openadapt-grounding
```
Visual grounding module for precise UI element localization. Integrates OmniParser for element detection and UI-TARS for action grounding. Temporal smoothing for stable tracking. Part of the OpenAdapt ecosystem.
```
(214 characters)

### openadapt-retrieval
```
Multimodal demo retrieval using VLM embeddings. Enables semantic search over demonstration libraries with Qwen3-VL embeddings and FAISS indexing. Supports MRL for flexible dimensions. Part of the OpenAdapt ecosystem.
```
(219 characters)

### openadapt-new (will become openadapt)
```
Unified CLI and meta-package for the OpenAdapt ecosystem. Record demonstrations, train vision-language models, and evaluate GUI automation agents. Integrates capture, ML, evaluation, and visualization packages.
```
(213 characters)

---

## Implementation Notes

### GitHub Topics

Add consistent topics across all repositories for discoverability:

**Core topics (all repos):**
- `openadapt`
- `gui-automation`
- `desktop-automation`
- `ai`
- `machine-learning`
- `python`

**Package-specific topics:**

| Repository | Additional Topics |
|------------|-------------------|
| openadapt | `rpa`, `process-automation`, `vlm`, `llm`, `cli` |
| openadapt-ml | `training`, `fine-tuning`, `vlm`, `grpo`, `qwen` |
| openadapt-capture | `screen-recording`, `accessibility`, `macos`, `windows`, `linux` |
| openadapt-grounding | `omniparser`, `ui-tars`, `visual-grounding`, `object-detection` |
| openadapt-evals | `benchmarks`, `evaluation`, `webarena`, `windows-agent-arena` |
| openadapt-viewer | `react`, `visualization`, `components`, `dashboard` |
| openadapt-retrieval | `retrieval`, `embeddings`, `qwen`, `clip`, `faiss` |

### Social Media Links

Verify these are set in organization settings:
- Website: https://openadapt.ai/
- Email: info@openadapt.ai
- Twitter/X: @OpenAdaptAI
- LinkedIn: company/openadapt-ai

### Migration Path

1. Rename `openadapt-new` to `openadapt` (or transfer package name)
2. Archive the legacy `OpenAdapt` repository with a deprecation notice pointing to the new ecosystem
3. Update all cross-references and documentation
4. Announce the ecosystem migration via blog post and social media
