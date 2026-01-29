# Cua vs OpenAdapt-ML Windows Agent Arena (WAA) Implementation Comparison

**Date**: 2026-01-28 (Updated)
**Status**: Research Analysis
**Author**: Research Agent

---

## Quick Reference: Key Metrics

| Metric | Cua/OpenAI CUA | OpenAdapt-ML | Microsoft WAA (Navi) |
|--------|----------------|--------------|----------------------|
| WAA Success Rate | N/A (OSWorld: 38.1%) | In progress | 19.5% (GPT-4V) |
| OSWorld Success Rate | 38.1% (OpenAI CUA) | Not implemented | N/A |
| Human Baseline | 72-74.5% | 74.5% (WAA) | 74.5% |
| VM Setup Time | Minutes (Lume) | ~15-20 min (Azure) | ~20 min |
| Primary Platform | macOS (Apple Silicon) | Windows (Azure) | Windows (Azure) |

---

## Executive Summary

This document analyzes [Cua (trycua/cua)](https://github.com/trycua/cua), a YC X25-backed open-source platform for Computer-Use Agents, and compares it with our OpenAdapt-Evals/OpenAdapt-ML two-package architecture.

**Key Finding**: Cua represents a significantly more comprehensive infrastructure platform that addresses many problems we've been solving piecemeal. However, adopting Cua wholesale would require substantial architectural changes and has notable trade-offs around Windows/Azure focus, Apple Silicon dependency, and our training pipeline integration.

**Recommendation**: Consider incremental adoption of Cua components, starting with cua-bench adapters for benchmark standardization, rather than full migration.

---

## 1. What is Cua?

### Overview

Cua ("koo-ah") is an open-source infrastructure platform for developing, evaluating, and deploying Computer-Use Agents. According to their [Hacker News launch](https://news.ycombinator.com/item?id=46768906) and [HuggingFace blog](https://huggingface.co/blog/cua-ai/cua-bench):

> "Cua is Docker for Computer-Use AI Agents - it enables AI agents to control full operating systems in virtual containers and deploy them locally or to the cloud."

### Core Components

The Cua ecosystem is organized as a monorepo with these key packages:

| Package | Purpose | Tech Stack |
|---------|---------|------------|
| **cua-agent** | AI agent framework for computer-use tasks | Python |
| **cua-computer** | SDK for controlling desktop environments | Python |
| **cua-computer-server** | Sandbox driver for UI interactions | Python/FastAPI |
| **cua-bench** | Benchmarks and RL environments | Python |
| **lume** | macOS/Linux VM management on Apple Silicon | Swift/CLI |
| **lumier** | Docker-compatible interface for Lume VMs | Python |
| **som** | Set-of-Mark for OmniParser integration | Python |
| **pylume** | Python bindings for Lume | Python |
| **mcp-server** | Multi-Modal Control Protocol server for Claude Desktop | Python |

### Key Capabilities

1. **Multi-Platform Virtualization**:
   - macOS/Linux via Apple Virtualization Framework (97% native CPU speed on Apple Silicon)
   - Windows via Docker/QEMU
   - Cloud deployment support

2. **Composite Agents Architecture**:
   - Separate grounding model (fast, small) from reasoning model (large)
   - Model-agnostic: supports Anthropic, OpenAI, Google, Ollama, LM Studio

3. **Unified Benchmark Framework (cua-bench)**:
   - Adapters for OSWorld, ScreenSpot, WindowsArena
   - Trajectory export for training
   - RL environment support

4. **Training Data Generation**:
   - "Trajectory replotting": Record 1 demo, render across 10 OS themes = 10 training trajectories
   - HTML snapshots with bounding boxes, not just screenshots
   - Multi-resolution (640x480 to 3440x1440)

---

## 2. Cua's Approach to Computer Use Automation

### Architecture Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cua Platform                             │
├─────────────────────────────────────────────────────────────────┤
│  Agent Layer (cua-agent)                                        │
│  ├── ComputerAgent - Main agent class                          │
│  ├── Provider adapters (Anthropic, OpenAI, Ollama, etc.)       │
│  └── Composite agents (grounding + reasoning split)            │
├─────────────────────────────────────────────────────────────────┤
│  Computer Layer (cua-computer)                                  │
│  ├── Computer class - Unified interface                         │
│  ├── Display drivers (screen capture, coordinates)             │
│  └── Input drivers (mouse, keyboard)                            │
├─────────────────────────────────────────────────────────────────┤
│  Sandbox Layer                                                   │
│  ├── Lume (Apple Silicon VMs - macOS/Linux)                    │
│  ├── Docker/QEMU (Windows, Linux)                               │
│  └── Cloud containers (cua-cloud)                               │
├─────────────────────────────────────────────────────────────────┤
│  Benchmark Layer (cua-bench)                                    │
│  ├── OSWorld adapter                                             │
│  ├── WindowsArena adapter                                        │
│  ├── ScreenSpot adapter                                          │
│  └── Custom task definitions                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Technical Decisions

1. **Sandbox-First**: Every agent runs in an isolated VM/container. This is non-negotiable for safety.

2. **Playwright-Like API**: Tasks defined with declarative Python decorators:
   ```python
   @cb.setup_task
   async def setup(env, scenario):
       await env.spotify.open()
       await env.spotify.create_playlist(scenario["playlist_name"])

   @cb.solve_task
   async def solve(env, scenario):
       await env.spotify.search(scenario["song"])
   ```

3. **HTML + Screenshots**: Captures full HTML with bounding boxes, accessibility labels, and CSS - not just screenshots. This enables:
   - Element-level grounding
   - Style variation generation
   - More robust training data

4. **Shell Applications**: Simulated apps (Spotify, Slack clones) that run in lightweight webtops without VM overhead. Enables rapid iteration.

---

## 3. Comparison with Our WAA-Based Evaluation Setup

### Our Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenAdapt Ecosystem                           │
├─────────────────────────────────────────────────────────────────┤
│  openadapt-ml (Training)                                        │
│  ├── training/ - VLM fine-tuning pipeline                      │
│  ├── vlm/ - Model adapters (Qwen, API-based)                   │
│  ├── baselines/ - Baseline model adapters                      │
│  ├── benchmarks/cli.py - VM lifecycle management               │
│  └── cloud/ - Lambda Labs, Azure ML                            │
├─────────────────────────────────────────────────────────────────┤
│  openadapt-evals (Evaluation)                                   │
│  ├── agents/ - BenchmarkAgent implementations                  │
│  │   ├── ApiAgent (Claude, GPT-5.1)                            │
│  │   ├── PolicyAgent (trained models)                          │
│  │   └── RetrievalAgent (demo-conditioned)                     │
│  ├── adapters/ - Benchmark adapters                            │
│  │   ├── WAAMockAdapter                                         │
│  │   └── WAALiveAdapter                                         │
│  └── benchmarks/ - Runner, viewer, Azure orchestration         │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                  │
│  ├── Azure VMs (Standard_D4ds_v5 with nested virt)             │
│  ├── Docker + QEMU (Windows 11 Enterprise via WAA image)       │
│  └── SSH tunnels for VNC/API access                            │
└─────────────────────────────────────────────────────────────────┘
```

### Side-by-Side Comparison

| Aspect | Cua | OpenAdapt-Evals/ML |
|--------|-----|-------------------|
| **Scope** | Full platform (sandboxes, SDKs, benchmarks, training) | Focused on evaluation + ML training |
| **Sandbox Technology** | Lume (Apple Silicon) + Docker/QEMU | Azure VMs + Docker/QEMU |
| **Primary Platform** | macOS first, then Linux/Windows | Windows first (WAA-focused) |
| **Local Dev Experience** | Native macOS VMs on Apple Silicon | Requires Azure VM or local Docker |
| **Benchmark Support** | OSWorld, ScreenSpot, WAA via adapters | WAA only (others planned) |
| **Training Data Gen** | Built-in trajectory replotting | Manual demo collection |
| **Agent Architecture** | Composite (grounding + reasoning) | Monolithic (single API call) |
| **VM Performance** | 97% native on Apple Silicon | Nested virtualization overhead |
| **Cloud Support** | cua-cloud (managed service coming) | Azure VMs, Lambda Labs for training |
| **RL Support** | Native RL environments in cua-bench | Not implemented |
| **Model Agnostic** | Yes (100+ providers) | Yes (Anthropic, OpenAI, local VLMs) |
| **Package Count** | 8+ packages in monorepo | 2 packages |
| **Dependencies** | Python 3.12+ required | Python 3.10+ |
| **Lines of Code** | ~15K+ (estimated) | ~8K |
| **Documentation** | Extensive (cua.ai/docs) | CLAUDE.md + README |
| **Community** | YC-backed, active development | Internal OpenAdapt project |

### Benchmark Framework Comparison

#### cua-bench

```python
# Task definition
@cb.tasks_config
def config():
    return {"scenarios": [{"playlist_name": "Workout", "song": "Eye of the Tiger"}, ...]}

@cb.setup_task
async def setup(env, scenario):
    await env.spotify.create_playlist(scenario["playlist_name"])

@cb.solve_task
async def solve(env, scenario):
    await env.spotify.search(scenario["song"])
    await env.spotify.add_to_playlist(scenario["playlist_name"])

@cb.evaluate_task
async def evaluate(env, scenario):
    playlist = await env.spotify.get_playlist(scenario["playlist_name"])
    return scenario["song"] in playlist.songs
```

**Key Features**:
- Declarative task definition
- Scenario variation injection
- Automatic trajectory recording
- Shell application support (simulated apps)

#### openadapt-evals

```python
# Task loaded from JSON
adapter = WAALiveAdapter(server_url="http://vm:5000")
task = adapter.load_task("notepad_1")

# Agent interaction
agent = ApiAgent(provider="anthropic")
obs = adapter.reset(task)
action = agent.act(obs, task)
obs, done, info = adapter.step(action)
result = adapter.evaluate(task)
```

**Key Features**:
- Uses upstream WAA task definitions
- HTTP adapter to WAA server
- Execution trace collection
- P0 demo persistence fix in ApiAgent

---

## 4. Key Differences in Architecture

### 4.1 Sandbox Philosophy

| Cua | OpenAdapt |
|-----|-----------|
| Sandboxes are the core primitive | VMs are infrastructure detail |
| Local-first (Apple Silicon VMs) | Cloud-first (Azure VMs) |
| Multiple sandbox types unified | Single sandbox type (WAA Docker) |
| Safety is architectural constraint | Safety via SSH/isolation |

**Implication**: Cua's sandbox-first design makes it safer and more portable, but requires Lume infrastructure which is Apple Silicon-only.

### 4.2 Training Data Generation

| Cua | OpenAdapt |
|-----|-----------|
| Trajectory replotting (1 demo → N variants) | Manual demo collection |
| HTML + screenshots captured | Screenshots only in WAA |
| Built-in visual diversity generation | No automatic variation |
| Shell apps for fast iteration | Full VM required |

**Implication**: Cua can generate significantly more diverse training data from fewer human demonstrations. This addresses the "10x performance variance across UI changes" problem they identified.

### 4.3 Agent Architecture

| Cua | OpenAdapt |
|-----|-----------|
| Composite agents (grounding + reasoning) | Monolithic agents |
| Explicit OmniParser/SoM integration | SoM mode supported but not primary |
| Cost-optimized (small model for grounding) | Full API call for each decision |

**Implication**: Cua's composite approach could reduce API costs and improve grounding accuracy by using specialized models for each subtask.

### 4.4 Benchmark Integration

| Cua | OpenAdapt |
|-----|-----------|
| Unified adapter interface across benchmarks | WAA-specific adapter |
| Native adapters for OSWorld, ScreenSpot, WAA | WAA only (others TODO) |
| Benchmark-agnostic task format | BenchmarkTask dataclass |
| RL environment support | Evaluation only |

**Implication**: Cua already has the multi-benchmark support we're planning in REPO_CONSOLIDATION_PLAN.md.

---

## 5. Should We Adopt Cua or Parts of It?

### Arguments FOR Adoption

1. **Multi-Benchmark Support**: They've already built adapters for OSWorld, ScreenSpot, WAA - exactly what we need.

2. **Training Data Generation**: Trajectory replotting would dramatically improve our training data diversity.

3. **Active Development**: YC-backed with active community. They're solving the same problems we are.

4. **Better Local Dev**: macOS VMs on Apple Silicon would enable faster iteration for Mac users.

5. **RL Support**: Native RL environments would enable future research directions.

6. **MCP Integration**: Claude Desktop integration via MCP server.

### Arguments AGAINST Full Adoption

1. **Apple Silicon Dependency**: Lume requires Apple Silicon. Our team uses Azure VMs which have no Apple Silicon equivalent.

2. **Windows Focus Mismatch**: We're focused on Windows (WAA) for enterprise use cases. Cua is macOS-first.

3. **Training Pipeline Integration**: Our training pipeline (openadapt-ml) is tightly integrated with openadapt-evals. Switching to cua-bench would require significant refactoring.

4. **Operational Complexity**: 8+ packages vs our 2. More to learn and maintain.

5. **Python 3.12+ Requirement**: We support Python 3.10+. Migration could break user environments.

6. **Unproven at Scale**: Despite YC backing, it's still early-stage. Our WAA setup is battle-tested.

7. **Azure VM Investment**: We've invested significant effort in Azure VM automation (PR #14). This would be partially wasted.

---

## 6. Trade-offs Analysis

### Scenario A: Full Migration to Cua

**Effort**: High (3-6 months)

**Benefits**:
- Unified multi-benchmark support
- Training data generation
- Active community support
- MCP/Claude Desktop integration

**Costs**:
- Significant refactoring of openadapt-ml training pipeline
- Azure VM automation work partially wasted
- New learning curve for team
- Potential compatibility issues with Python 3.10 users

**Risk**: Medium-High (depending on Cua's stability and our ability to extend it)

### Scenario B: Adopt cua-bench Adapters Only

**Effort**: Medium (1-2 months)

**Benefits**:
- Standardized benchmark interface
- Access to OSWorld, ScreenSpot adapters
- Can still use our Azure VM infrastructure
- Incremental migration path

**Costs**:
- Must maintain compatibility layer
- Miss out on sandbox/Lume benefits
- Partial adoption may cause confusion

**Risk**: Low-Medium

### Scenario C: Adopt Architectural Patterns Only

**Effort**: Low (2-4 weeks)

**Benefits**:
- No external dependencies
- Learn from their solutions
- Can implement selectively

**What to Adopt**:
- Composite agent pattern (grounding + reasoning)
- Trajectory replotting concept
- Declarative task definition style
- HTML capture alongside screenshots

**Costs**:
- Must implement ourselves
- No community support

**Risk**: Low

### Scenario D: Stay Current Course

**Effort**: None

**Benefits**:
- Known system, no learning curve
- REPO_CONSOLIDATION_PLAN.md already addresses multi-benchmark support
- Full control over architecture

**Costs**:
- Slower to add OSWorld, other benchmarks
- No training data generation automation
- Potentially duplicating work

**Risk**: Low (but higher opportunity cost)

---

## 7. Recommendations

### Immediate (Next 2-4 Weeks)

1. **Do NOT migrate to Cua wholesale**. The Azure VM investment is too recent, and we have a working system.

2. **Adopt the composite agent pattern** in ApiAgent:
   - Add optional grounding model (OmniParser/SoM)
   - Use small model for element detection, large model for reasoning
   - This is an incremental change to existing code

3. **Add HTML capture** to WAALiveAdapter:
   - Capture accessibility tree alongside screenshots
   - Enables future training data diversity

### Medium-Term (Next 2-3 Months)

4. **Evaluate cua-bench integration**:
   - Test if cua-bench adapters can work with our evaluation runner
   - If compatible, adopt their OSWorld/ScreenSpot adapters
   - Keep our WAALiveAdapter for Azure VM compatibility

5. **Implement trajectory replotting prototype**:
   - Record demos with HTML + screenshots
   - Test re-rendering across Windows themes
   - Measure training data quality improvement

### Long-Term (6+ Months)

6. **Consider Lume for local development**:
   - If team has Apple Silicon Macs
   - Would enable faster local iteration
   - Keep Azure VMs for CI/production

7. **Contribute back to Cua**:
   - Our Azure VM automation could benefit the community
   - Windows-focused improvements

---

## 8. Specific Recommendations for REPO_CONSOLIDATION_PLAN.md

Our current consolidation plan is **still valid** but should incorporate these learnings:

1. **Keep the two-package split** (openadapt-evals + openadapt-ml). Cua's monorepo with 8+ packages is more complex than necessary for our use case.

2. **Add benchmark adapter interface** compatible with cua-bench:
   ```python
   class BenchmarkAdapter(ABC):
       # Our current interface is similar to cua-bench
       # Add optional HTML capture in observations
       # Add evaluation spec support
   ```

3. **Prioritize OSWorld adapter** as second benchmark (after WAA). Cua's OSWorld-Verified work validates this as the next target.

4. **Consider shell applications** for testing:
   - Simulated apps for unit tests
   - No VM overhead for CI
   - This is orthogonal to our VM-based evaluation

5. **Document composite agent pattern** in CLAUDE.md for future implementation.

---

## 9. Conclusion

Cua is an impressive and comprehensive platform that addresses many problems we're solving. However, full migration is not recommended at this time due to:

1. Our recent Azure VM automation investment
2. Apple Silicon dependency in Lume
3. Windows-first focus vs their macOS-first approach

Instead, we should:
- **Learn from their architecture** (composite agents, trajectory replotting)
- **Evaluate cua-bench adapters** for multi-benchmark support
- **Stay on our current consolidation path** while incorporating their patterns

The OpenAdapt ecosystem can achieve similar capabilities through incremental improvements rather than wholesale migration.

---

## 10. Market Positioning and Strategic Differentiation

### 10.1 The Success Rate Gap

| Agent | Benchmark | Success Rate | Gap to Human |
|-------|-----------|--------------|--------------|
| OpenAI CUA | OSWorld | 38.1% | ~36 pts below human (74.5%) |
| Microsoft Navi | WAA | 19.5% | ~55 pts below human (74.5%) |

**Key insight**: The problem is far from solved. Both approaches have runway—the technology isn't mature enough for either to dominate yet.

The 38.1% vs 19.5% gap is significant:
- OSWorld is macOS/Linux focused
- WAA is Windows focused
- **Windows automation appears harder** (more legacy complexity, more app diversity)

This validates OpenAdapt's focus: Windows enterprise workflows are the harder problem.

### 10.2 Market Positioning

| Aspect | Cua | OpenAdapt |
|--------|-----|-----------|
| **Primary TAM** | AI Agents / Developer Tools (~$500M-1B, 40%+ CAGR) | Enterprise RPA + Legacy Automation (~$8-10B, 20% CAGR) |
| **Buyer** | ML engineers, AI researchers | Ops, IT, compliance, support |
| **Value Prop** | "Build computer-use agents faster" | "Learn automation from how you already work" |

### 10.3 Why These Markets Don't Fully Overlap

- Cua assumes synthetic, controlled environments
- OpenAdapt captures real workflows from production systems
- Enterprise compliance requirements (HIPAA, SOX) favor retrospective capture

### 10.4 Where Cua's Sandbox Approach Breaks Down

Cua's sandbox-first design assumes you can:
- Spin up a clean VM with the target app
- Control the environment end-to-end
- Reproduce the workflow deterministically

**This fails for:**

| Scenario | Why Sandboxes Fail | OpenAdapt Alternative |
|----------|-------------------|----------------------|
| **Citrix/RDP apps** | No local install possible | Capture remote session natively |
| **Licensed enterprise software** | SAP, Epic, Oracle—can't sandbox without licensing | Record from licensed desktop |
| **Policy-controlled desktops** | Enterprise IT won't allow arbitrary VMs | Capture from existing desktop |
| **Compliance-restricted environments** | Healthcare, finance—can't replicate production | Retrospective recording allowed |
| **Multi-app workflows** | Spanning 5+ apps that can't all be sandboxed together | Single recording captures all |

**OpenAdapt's retrospective recording doesn't have these constraints.**

### 10.5 Shell Applications: Where Cua and OpenAdapt Could Converge

Shell apps (simulated Spotify, Slack clones) serve different purposes:

| Use Case | Cua's Approach | OpenAdapt's Approach |
|----------|---------------|---------------------|
| Unit tests | Primary use case | Could adopt for CI |
| Training data | Synthetic generation | Not applicable (need real data) |
| Fast iteration | Core workflow | Could speed up agent logic dev |
| Production eval | Not representative | Azure VMs remain primary |

**Recommendation**: Adopt shell apps for regression testing agent logic, but never train on them. Real behavioral data from enterprise workflows remains the moat.

### 10.6 Bottom Line

The 19.5% WAA success rate validates OpenAdapt's approach:
- Windows enterprise automation is hard
- Current agents fail often
- Learning from real human demonstrations is one path to improvement

Cua's strength (macOS VMs at 97% native speed) doesn't help with SAP, Citrix, or legacy Win32 apps—exactly where OpenAdapt focuses.

---

## 12. Appendix: Agent Loop Types in Cua

Cua provides multiple agent loop implementations optimized for different use cases:

| Agent Loop | Best For | Model Support |
|------------|----------|---------------|
| **AgentLoop.OPENAI** | Web-based tasks, browser automation | OpenAI models (requires Tier 3 access) |
| **AgentLoop.ANTHROPIC** | Strong reasoning + computer-use | claude-3-5-sonnet, claude-3-7-sonnet |
| **AgentLoop.UITARS** | OS/desktop tasks, latency-sensitive | UI-TARS-1.5 (local or HuggingFace) |
| **AgentLoop.OMNI** | Maximum flexibility | Any vision-language model |

### Composite Agent Example

```python
# Pair a grounding model with a reasoning model
model = "huggingface-local/GTA1-7B+openai/gpt-4o"
# GTA1-7B: precise click coordinates
# GPT-4o: action planning and reasoning
```

---

## 13. Appendix: OpenAdapt-ML Docker Setup Details

Our current implementation uses a custom Dockerfile that:

1. **Base**: `dockurr/windows:latest` (modern Windows ISO auto-download)
2. **WAA Components**: Copied from `windowsarena/winarena:latest`
3. **IP Patching**: Changes `20.20.20.21` to `172.30.0.2` for dockurr compatibility
4. **Python**: Uses Python 3.9 from vanilla WAA for GroundingDINO compatibility
5. **Automation**: FirstLogonCommands for firewall, WAA server auto-start

Key environment variables:
- `VERSION=11e` - Windows 11 Enterprise Evaluation
- `RAM_SIZE=8G` / `16G` (fast mode)
- `CPU_CORES=4` / `6` (fast mode)

---

## References

- [Cua GitHub Repository](https://github.com/trycua/cua)
- [Cua-Bench HuggingFace Blog](https://huggingface.co/blog/cua-ai/cua-bench)
- [Show HN: Cua-Bench Discussion](https://news.ycombinator.com/item?id=46768906)
- [Launch HN: Cua (YC X25)](https://news.ycombinator.com/item?id=43773563)
- [Cua Documentation](https://cua.ai/docs)
- [Cua Composite Agents Blog](https://www.trycua.com/blog/composite-agents)
- [What is Lume?](https://cua.ai/docs/lume/guide/getting-started/introduction)
- [OSWorld-Verified](https://xlang.ai/blog/osworld-verified)
- [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)
- [Windows Agent Arena Paper](https://arxiv.org/abs/2409.08264)
- [OpenAI Computer-Using Agent](https://openai.com/index/computer-using-agent/)
- [OpenAdapt REPO_CONSOLIDATION_PLAN.md](/Users/abrichr/oa/src/openadapt-ml/docs/REPO_CONSOLIDATION_PLAN.md)
