# OpenAdaptAI GitHub Organization Profile Update Plan

## Current State Analysis

### Organization Profile (as of January 2026)

**URL:** https://github.com/openadaptai

| Field | Current Value |
|-------|---------------|
| **Name** | OpenAdapt.AI |
| **Description** | "AI for Desktops" - Generative process automation that learns from human demonstrations |
| **Website** | https://openadapt.ai/ |
| **Email** | info@openadapt.ai |
| **Twitter/X** | @OpenAdaptAI |
| **LinkedIn** | company/openadapt-ai |
| **Followers** | 85 |
| **Total Repos** | 41 |

### Current Pinned Repository

Only **1 pinned repository**:
- **OpenAdapt** (1,470 stars, 216 forks) - "Open Source Generative Process Automation (i.e. Generative RPA)"

### Organization README Status

The `.github` repository **does have a profile README** with comprehensive content including:
- Welcome message and project description
- Key features and capabilities
- Getting started guide
- Contributing links
- Implementation consulting section
- Sponsors section
- Contact information
- Alpha disclaimer

**Current README content:**

```markdown
# Welcome to OpenAdapt.AI 👋

**OpenAdapt.AI** is an open-source **Generative Process Automation** framework that aims to transform task automation across applications and platforms. Designed as a baseline for evaluating agents, OpenAdapt.AI learns from human demonstrations to intelligently automate tasks, adapting dynamically to evolving workflows.

## 🚀 What is OpenAdapt.AI?

OpenAdapt.AI leverages advanced machine learning models to:
- **Observe and Learn**: Automatically learns from user interactions to generate automation scripts.
- **Automate Intelligently**: Adapts to software changes, varying workflows, and complex environments.
- **Ensure Privacy**: Privacy is a core focus, with robust data handling mechanisms to protect user data.

## 🌟 Key Features

- **Generative Process Automation**: Automates tasks using generative AI models that learn from demonstrations.
- **Cross-Platform Compatibility**: Works seamlessly with browser apps (e.g., Chrome) and desktop applications.
- **Audio Narration Transcription**: Converts spoken instructions or feedback into automation scripts.
- **Flexible LLM Model Providers**: Supports multiple language model providers, including:
  - Online: **OpenAI**, **Anthropic**, **Google**
  - Offline (Work in Progress): **Ollama**, **vLLM**
- **Open Source**: Developed transparently and collaboratively by the community.
- **Privacy by Design**: Uses tools like AWS Rekognition, Microsoft Presidio, and Private-ai.com to ensure data privacy.

## 📚 Getting Started

To get started with OpenAdapt.AI:

1. **Clone the Repository**: `git clone https://github.com/OpenAdaptAI/OpenAdapt`
2. **Install Dependencies**: Follow the [Installation Guide](https://github.com/OpenAdaptAI/OpenAdapt/blob/main/docs/installation.md)
3. **Run Your First Automation**: Check out our [Quick Start Guide](https://github.com/OpenAdaptAI/OpenAdapt/blob/main/docs/quickstart.md)

## 🛠️ Contributing

We welcome contributions from everyone. Whether you're a developer, data scientist, tester, or just passionate about automation, there's a place for you here!

- **[Contributing Guide](https://github.com/OpenAdaptAI/OpenAdapt/blob/main/CONTRIBUTING.md)**
- **[Code of Conduct](https://github.com/OpenAdaptAI/OpenAdapt/blob/main/CODE_OF_CONDUCT.md)**

Join our [Community](https://github.com/OpenAdaptAI/OpenAdapt/discussions) to discuss new ideas, report bugs, or seek help.

## 🤝 Implementation Consulting

Looking to implement OpenAdapt.AI in your organization or need help tailoring it to your specific use case? We offer expert consulting services to assist with deployment, customization, and optimization of OpenAdapt.AI for your unique needs.

For more information or to schedule a consultation, contact us at [sales@openadapt.ai](mailto:sales@openadapt.ai).

## 🤝 Sponsors and Partners

OpenAdapt.AI is made possible by the support of our contributors and partners. If you're interested in sponsoring the development, check out our [GitHub Sponsors page](https://github.com/sponsors/OpenAdaptAI).

##  Contact

For general questions, feedback, or partnership inquiries, please reach out at [contact@openadapt.ai](mailto:contact@openadapt.ai).

## ⚠️ Disclaimer

OpenAdapt.AI is currently in alpha. While it's functional, some features are still under development. Use with caution in production environments.
```

### New Package Ecosystem (Not Yet Highlighted)

| Repository | Description | Stars | Last Updated |
|------------|-------------|-------|--------------|
| **openadapt** | Main app - desktop recording/playback | 1,470 | Active |
| **openadapt-ml** | ML toolkit for training multimodal GUI-action models | 2 | Jan 9, 2026 |
| **openadapt-grounding** | Visual grounding with OmniParser, UI-TARS integration | 0 | Jan 15, 2026 |
| **openadapt-evals** | Benchmark evaluation (WAA, WebArena) | 0 | Jan 16, 2026 |
| **openadapt-viewer** | UI component library for visualization | 0 | Jan 16, 2026 |
| **openadapt-retrieval** | Multimodal demo retrieval with Qwen3-VL/CLIP/FAISS | 0 | Jan 16, 2026 |

---

## Proposed Changes

### 1. Organization Description Update

**Current:**
> "AI for Desktops" - Generative process automation that learns from human demonstrations

**Proposed:**
> AI-powered desktop automation platform. Record workflows, train models, replay intelligently. Modular ecosystem: recording, ML training, visual grounding, evaluation benchmarks, and visualization.

**Alternative (shorter):**
> Open source AI desktop automation. Learn from demonstrations, automate intelligently.

### 2. Organization Profile README

Update `.github/profile/README.md` to **evolve** the existing README by adding the package ecosystem section while preserving the existing structure, tone, and valuable content.

#### Summary of Changes

| Section | Current | Proposed Change |
|---------|---------|-----------------|
| Header | Welcome message | Keep as-is |
| What is OpenAdapt.AI | 3 bullet points | Keep as-is |
| Key Features | 6 bullet points | Keep as-is |
| **NEW: Package Ecosystem** | N/A | Add after Key Features |
| **NEW: Architecture** | N/A | Add after Package Ecosystem |
| Getting Started | Clone, install, run | Update to mention modular packages |
| Contributing | Guide links + community | Keep as-is |
| Implementation Consulting | sales@openadapt.ai | Keep as-is |
| Sponsors and Partners | GitHub Sponsors link | Keep as-is |
| Contact | contact@openadapt.ai | Keep as-is |
| Disclaimer | Alpha warning | Keep as-is |

#### Full Proposed README (copy-paste ready)

```markdown
# Welcome to OpenAdapt.AI 👋

**OpenAdapt.AI** is an open-source **Generative Process Automation** framework that aims to transform task automation across applications and platforms. Designed as a baseline for evaluating agents, OpenAdapt.AI learns from human demonstrations to intelligently automate tasks, adapting dynamically to evolving workflows.

## 🚀 What is OpenAdapt.AI?

OpenAdapt.AI leverages advanced machine learning models to:
- **Observe and Learn**: Automatically learns from user interactions to generate automation scripts.
- **Automate Intelligently**: Adapts to software changes, varying workflows, and complex environments.
- **Ensure Privacy**: Privacy is a core focus, with robust data handling mechanisms to protect user data.

## 🌟 Key Features

- **Generative Process Automation**: Automates tasks using generative AI models that learn from demonstrations.
- **Cross-Platform Compatibility**: Works seamlessly with browser apps (e.g., Chrome) and desktop applications.
- **Audio Narration Transcription**: Converts spoken instructions or feedback into automation scripts.
- **Flexible LLM Model Providers**: Supports multiple language model providers, including:
  - Online: **OpenAI**, **Anthropic**, **Google**
  - Offline (Work in Progress): **Ollama**, **vLLM**
- **Open Source**: Developed transparently and collaboratively by the community.
- **Privacy by Design**: Uses tools like AWS Rekognition, Microsoft Presidio, and Private-ai.com to ensure data privacy.

## 📦 Package Ecosystem

OpenAdapt.AI is built as a **modular ecosystem** of specialized packages. Use what you need:

| Package | Description | Status |
|---------|-------------|--------|
| [**openadapt**](https://github.com/OpenAdaptAI/OpenAdapt) | Core platform - recording, playback, visualization | ⭐ Production |
| [**openadapt-ml**](https://github.com/OpenAdaptAI/openadapt-ml) | ML engine for training GUI automation models | 🔧 Active Development |
| [**openadapt-grounding**](https://github.com/OpenAdaptAI/openadapt-grounding) | Visual grounding with OmniParser & UI-TARS | 🔧 Active Development |
| [**openadapt-evals**](https://github.com/OpenAdaptAI/openadapt-evals) | Benchmark evaluation (WAA, WebArena) | 🔧 Active Development |
| [**openadapt-viewer**](https://github.com/OpenAdaptAI/openadapt-viewer) | Reusable UI components for visualization | 🔧 Active Development |
| [**openadapt-retrieval**](https://github.com/OpenAdaptAI/openadapt-retrieval) | Multimodal demo retrieval (Qwen3-VL, CLIP, FAISS) | 🔧 Active Development |

### 🏗️ Architecture Overview

```
                     ┌──────────────────┐
                     │    openadapt     │  ← Main application
                     │ (record/replay)  │
                     └────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  openadapt-ml   │ │   openadapt-    │ │   openadapt-    │
│   (training)    │ │   grounding     │ │   retrieval     │
└─────────────────┘ │  (perception)   │ │    (demos)      │
                    └─────────────────┘ └─────────────────┘
          │
          ▼
┌─────────────────┐ ┌─────────────────┐
│  openadapt-     │ │   openadapt-    │
│     evals       │ │     viewer      │
│  (benchmarks)   │ │ (UI components) │
└─────────────────┘ └─────────────────┘
```

## 📚 Getting Started

### Quick Start (Main Application)

```bash
# Clone the repository
git clone https://github.com/OpenAdaptAI/OpenAdapt

# Install dependencies
# Follow the Installation Guide: https://github.com/OpenAdaptAI/OpenAdapt/blob/main/docs/installation.md

# Run your first automation
# See the Quick Start Guide: https://github.com/OpenAdaptAI/OpenAdapt/blob/main/docs/quickstart.md
```

### Individual Packages

Each package can be installed independently via pip:

```bash
# Core application
pip install openadapt

# ML training engine
pip install openadapt-ml

# Visual grounding
pip install openadapt-grounding

# Benchmark evaluation
pip install openadapt-evals

# Demo retrieval
pip install openadapt-retrieval
```

📖 See each package's README for detailed documentation.

## 🛠️ Contributing

We welcome contributions from everyone. Whether you're a developer, data scientist, tester, or just passionate about automation, there's a place for you here!

- **[Contributing Guide](https://github.com/OpenAdaptAI/OpenAdapt/blob/main/CONTRIBUTING.md)**
- **[Code of Conduct](https://github.com/OpenAdaptAI/OpenAdapt/blob/main/CODE_OF_CONDUCT.md)**

Join our [Community](https://github.com/OpenAdaptAI/OpenAdapt/discussions) to discuss new ideas, report bugs, or seek help.

**Areas of interest:**
- Frontend development (React, visualization)
- Machine learning (model training, evaluation)
- Desktop integration (Windows, macOS, Linux)
- Benchmark development and evaluation
- Documentation and tutorials

## 🤝 Implementation Consulting

Looking to implement OpenAdapt.AI in your organization or need help tailoring it to your specific use case? We offer expert consulting services to assist with deployment, customization, and optimization of OpenAdapt.AI for your unique needs.

For more information or to schedule a consultation, contact us at [sales@openadapt.ai](mailto:sales@openadapt.ai).

## 🤝 Sponsors and Partners

OpenAdapt.AI is made possible by the support of our contributors and partners. If you're interested in sponsoring the development, check out our [GitHub Sponsors page](https://github.com/sponsors/OpenAdaptAI).

## 📬 Contact

For general questions, feedback, or partnership inquiries, please reach out at [contact@openadapt.ai](mailto:contact@openadapt.ai).

## ⚠️ Disclaimer

OpenAdapt.AI is currently in alpha. While it's functional, some features are still under development. Use with caution in production environments.
```

#### Diff View (Key Changes Only)

```diff
 # Welcome to OpenAdapt.AI 👋
 [... unchanged ...]

 ## 🌟 Key Features
 [... unchanged ...]

+## 📦 Package Ecosystem
+
+OpenAdapt.AI is built as a **modular ecosystem** of specialized packages. Use what you need:
+
+| Package | Description | Status |
+|---------|-------------|--------|
+| [**openadapt**](https://github.com/OpenAdaptAI/OpenAdapt) | Core platform - recording, playback, visualization | ⭐ Production |
+| [**openadapt-ml**](https://github.com/OpenAdaptAI/openadapt-ml) | ML engine for training GUI automation models | 🔧 Active Development |
+| [**openadapt-grounding**](https://github.com/OpenAdaptAI/openadapt-grounding) | Visual grounding with OmniParser & UI-TARS | 🔧 Active Development |
+| [**openadapt-evals**](https://github.com/OpenAdaptAI/openadapt-evals) | Benchmark evaluation (WAA, WebArena) | 🔧 Active Development |
+| [**openadapt-viewer**](https://github.com/OpenAdaptAI/openadapt-viewer) | Reusable UI components for visualization | 🔧 Active Development |
+| [**openadapt-retrieval**](https://github.com/OpenAdaptAI/openadapt-retrieval) | Multimodal demo retrieval (Qwen3-VL, CLIP, FAISS) | 🔧 Active Development |
+
+### 🏗️ Architecture Overview
+
+[ASCII diagram showing package relationships]

 ## 📚 Getting Started

-To get started with OpenAdapt.AI:
+### Quick Start (Main Application)

-1. **Clone the Repository**: `git clone https://github.com/OpenAdaptAI/OpenAdapt`
-2. **Install Dependencies**: Follow the [Installation Guide](...)
-3. **Run Your First Automation**: Check out our [Quick Start Guide](...)
+```bash
+# Clone the repository
+git clone https://github.com/OpenAdaptAI/OpenAdapt
+# ... etc
+```
+
+### Individual Packages
+
+Each package can be installed independently via pip:
+
+```bash
+pip install openadapt
+pip install openadapt-ml
+# ... etc
+```

 ## 🛠️ Contributing
 [... unchanged, but added "Areas of interest" section ...]
+
+**Areas of interest:**
+- Frontend development (React, visualization)
+- Machine learning (model training, evaluation)
+- Desktop integration (Windows, macOS, Linux)
+- Benchmark development and evaluation
+- Documentation and tutorials

 ## 🤝 Implementation Consulting
 [... unchanged ...]

 ## 🤝 Sponsors and Partners
 [... unchanged ...]

-##  Contact
+## 📬 Contact
 [... unchanged, fixed missing emoji ...]

 ## ⚠️ Disclaimer
 [... unchanged ...]
```

### 3. Suggested Pinned Repositories Order

Pin **6 repositories** in this order (GitHub allows up to 6 pinned repos):

| Order | Repository | Rationale |
|-------|------------|-----------|
| 1 | **OpenAdapt** | Main entry point, most stars (1,470), production-ready |
| 2 | **openadapt-ml** | Core ML engine, foundation for training |
| 3 | **openadapt-grounding** | Visual perception layer, key differentiator |
| 4 | **openadapt-evals** | Benchmark evaluation, credibility builder |
| 5 | **openadapt-viewer** | UI components, shows polish |
| 6 | **openadapt-retrieval** | Demo retrieval, novel capability |

### 4. Repository Description Updates

Update repository descriptions to be consistent and informative:

| Repository | Current Description | Proposed Description |
|------------|--------------------|-----------------------|
| openadapt | Open Source Generative Process Automation... | AI desktop automation platform. Record workflows, train models, replay intelligently. |
| openadapt-ml | Open-source ML toolkit... | Train and evaluate multimodal GUI-action models for desktop automation |
| openadapt-grounding | Temporal smoothing for UI element detection... | Visual grounding with OmniParser and UI-TARS for precise UI element detection |
| openadapt-evals | Benchmark evaluation framework... | Evaluate GUI automation agents against WAA, WebArena, and other benchmarks |
| openadapt-viewer | Reusable UI component library... | React components for recording playback, action visualization, and metrics |
| openadapt-retrieval | Multimodal demo retrieval... | Retrieve relevant demonstrations using Qwen3-VL embeddings and FAISS indexing |

---

## Additional Recommendations

### 1. Add Topics to All Repositories

Add consistent GitHub topics across all repos for discoverability:

**Core topics for all repos:**
- `openadapt`
- `gui-automation`
- `desktop-automation`
- `ai`
- `machine-learning`

**Package-specific topics:**

| Repository | Additional Topics |
|------------|-------------------|
| openadapt | `rpa`, `process-automation`, `multimodal`, `llm`, `vlm` |
| openadapt-ml | `training`, `fine-tuning`, `vlm`, `grpo`, `unsloth` |
| openadapt-grounding | `omniparser`, `ui-tars`, `visual-grounding`, `object-detection` |
| openadapt-evals | `benchmarks`, `evaluation`, `webarena`, `windows-agent-arena` |
| openadapt-viewer | `react`, `visualization`, `components`, `ui-library` |
| openadapt-retrieval | `retrieval`, `embeddings`, `qwen`, `clip`, `faiss` |

### 2. Create a Documentation Hub

Consider creating `openadapt-docs` repository or using GitHub Wiki more extensively:
- Unified installation guide across packages
- Architecture diagrams
- API reference
- Tutorials and examples

### 3. Add Social Preview Images

Create consistent social preview images (1280x640px) for each repository showing:
- OpenAdapt logo
- Package name
- Brief tagline
- Visual representation of functionality

### 4. Establish Release Cadence

- Set up GitHub Releases for all packages
- Use semantic versioning consistently
- Publish changelogs

### 5. Add Status Badges to READMEs

```markdown
![GitHub stars](https://img.shields.io/github/stars/OpenAdaptAI/OpenAdapt)
![License](https://img.shields.io/github/license/OpenAdaptAI/OpenAdapt)
![Python Version](https://img.shields.io/badge/python-3.10+-blue)
![Build Status](https://img.shields.io/github/actions/workflow/status/OpenAdaptAI/OpenAdapt/ci.yml)
```

### 6. Cross-Link Repositories

Each package README should include:
- Link to main OpenAdapt repo
- Links to related packages
- Architecture diagram showing package relationships

---

## Implementation Checklist

### Immediate Actions (This Week)

- [ ] Update organization description
- [ ] Create/update `.github/profile/README.md`
- [ ] Pin 6 repositories in recommended order
- [ ] Update repository descriptions for all packages
- [ ] Add consistent topics to all repositories

### Short-Term Actions (Next 2 Weeks)

- [ ] Create social preview images
- [ ] Add status badges to all READMEs
- [ ] Cross-link repositories in documentation
- [ ] Set up GitHub Releases for new packages

### Medium-Term Actions (Next Month)

- [ ] Create comprehensive documentation hub
- [ ] Record demo videos for each package
- [ ] Write blog post announcing ecosystem
- [ ] Update website to reflect new architecture

---

## Metrics to Track

After implementing changes, monitor:

1. **GitHub Stars** - Growth rate for new packages
2. **Organization Followers** - Currently 85, target 150+ in 3 months
3. **Repository Traffic** - Views and clones for each package
4. **Issue/PR Activity** - Community engagement
5. **Search Rankings** - "gui automation", "desktop automation", "rpa open source"

---

## Summary

The OpenAdaptAI GitHub organization has a strong foundation with the main OpenAdapt repository (1,470 stars) but lacks visibility for the new modular package ecosystem. By updating the organization profile, creating a comprehensive README, pinning the key repositories, and maintaining consistent branding across packages, the organization can better communicate:

1. **What OpenAdapt is** - AI-powered desktop automation platform
2. **How the ecosystem works** - Modular packages with clear responsibilities
3. **How to get started** - Clear entry points for different use cases

This will improve developer discoverability, encourage contributions, and establish OpenAdapt as a serious open-source alternative in the desktop automation space.
