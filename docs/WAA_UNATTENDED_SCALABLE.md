# Unattended Scalable Programmatic WAA

**Last Updated:** 2026-01-29

## Goal

Run Windows Agent Arena (WAA) benchmark with:
- **Unattended**: No manual intervention (Windows auto-installs, server auto-starts)
- **Scalable**: N parallel workers (10+ for full 154-task benchmark in ~30 min)
- **Programmatic**: Single command execution

## Current State

### What Official WAA Provides

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LOCAL: python scripts/run_azure.py --num_workers 10                    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     AZURE ML WORKSPACE                                   │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐           │
│  │ Compute  │  │ Compute  │  │ Compute  │  ...  │ Compute  │           │
│  │ Instance │  │ Instance │  │ Instance │       │ Instance │           │
│  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │       │ Worker N │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘           │
│       │             │             │                   │                 │
│       ▼             ▼             ▼                   ▼                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Each instance runs: Docker → QEMU → Windows → WAA → Navi       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Pros:**
- ✅ Parallelization built-in (`--num_workers N`)
- ✅ Azure ML handles compute lifecycle
- ✅ Auto-shutdown on idle
- ✅ Results to Azure Storage

**Cons:**
- ❌ Only supports Navi agent (not our API agents)
- ❌ Requires pre-uploaded golden image to Azure Storage
- ❌ Complex Azure ML setup (workspace, storage, startup script)
- ❌ Limited debugging (no VNC)

### What We Built

| Component | Purpose | Useful? |
|-----------|---------|---------|
| `waa_deploy/Dockerfile` | Auto-download Windows, API agent support | ✅ For dev |
| `waa_deploy/api_agent.py` | Claude/OpenAI agent (alternative to Navi) | ✅ Key differentiator |
| `cli.py` | Dedicated VM management | ✅ For dev/debug |
| `WAALiveAdapter` | Connects to WAA server API | ✅ Portable |
| `ApiAgent` | Structured actions via LLM API | ✅ Portable |

---

## Synthesized Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LOCAL: uv run python -m openadapt_ml.benchmarks.cli scale              │
│           --workers 10 --agent api-openai --tasks 154                   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   Use official run_azure.py│
                    │   for compute orchestration│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     AZURE ML COMPUTE INSTANCES                           │
│                                                                          │
│  Each instance runs our modified Docker image:                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  waa-auto:latest (our Dockerfile)                                │   │
│  │  ├── dockurr/windows (auto-downloads Windows 11)                │   │
│  │  ├── windowsarena/winarena components                           │   │
│  │  ├── api_agent.py (Claude/OpenAI support)                       │   │
│  │  └── Auto-start WAA server on boot                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Insight

**Don't reinvent parallelization.** Use official `run_azure.py` for compute orchestration, but:
1. Replace their Docker image with ours (`waa-auto:latest`)
2. Add our API agent to the agent options

---

## Implementation Plan

### Phase 1: Validate Single Worker (DONE)

- [x] Dedicated VM working (`waa-eval-vm`)
- [x] VNC access for debugging
- [x] WAA server auto-starts
- [x] Benchmark runs with Navi agent

### Phase 2: Add API Agent to Official WAA

**Goal:** Run `python run_azure.py --agent api-openai`

**Steps:**

1. **Create NaviAgent-compatible wrapper:**
   ```python
   # mm_agents/openadapt/agent.py
   class OpenAdaptAgent:
       """Wrapper to use our ApiAgent with official WAA runner."""

       def __init__(self, model="gpt-4o", provider="openai"):
           self.provider = provider
           self.model = model
           # Initialize API client

       def predict(self, instruction: str, obs: Dict) -> List[str]:
           """Convert observation → API call → action code."""
           # 1. Extract screenshot from obs
           # 2. Call OpenAI/Claude API
           # 3. Parse response to action
           # 4. Return as Python code string
           return [f"computer.mouse.click(x={x}, y={y})"]

       def reset(self):
           self.history = []
   ```

2. **Modify official `run.py` to support new agent:**
   ```python
   # In run.py, add:
   elif cfg_args["agent_name"] == "api-openai":
       from mm_agents.openadapt.agent import OpenAdaptAgent
       agent = OpenAdaptAgent(provider="openai", model=cfg_args["model"])
   elif cfg_args["agent_name"] == "api-claude":
       from mm_agents.openadapt.agent import OpenAdaptAgent
       agent = OpenAdaptAgent(provider="anthropic", model=cfg_args["model"])
   ```

3. **Test locally first:**
   ```bash
   # On dedicated VM
   cd /client
   python run.py --agent api-openai --model gpt-4o --test_all_meta_path ...
   ```

### Phase 3: Push Custom Image to Azure

**Goal:** Azure ML uses our `waa-auto:latest` instead of `windowsarena/winarena:latest`

**Steps:**

1. **Push to Azure Container Registry:**
   ```bash
   # Build locally
   docker build -t waa-auto:latest -f waa_deploy/Dockerfile .

   # Tag for ACR
   docker tag waa-auto:latest openadaptacr.azurecr.io/waa-auto:latest

   # Push
   az acr login --name openadaptacr
   docker push openadaptacr.azurecr.io/waa-auto:latest
   ```

2. **Modify `run_azure.py` to use our image:**
   ```python
   # Change default:
   parser.add_argument('--docker_img_name',
       default='openadaptacr.azurecr.io/waa-auto:latest',  # Was: windowsarena/winarena:latest
       help='Docker image name')
   ```

### Phase 4: Wrapper CLI

**Goal:** Single command for everything

```bash
# Full benchmark with 10 workers
uv run python -m openadapt_ml.benchmarks.cli scale \
    --workers 10 \
    --agent api-openai \
    --model gpt-4o \
    --tasks all

# Subset for testing
uv run python -m openadapt_ml.benchmarks.cli scale \
    --workers 2 \
    --agent api-claude \
    --tasks notepad_1,notepad_2,browser_1
```

**Implementation:**
```python
# In cli.py, add 'scale' command that:
# 1. Ensures Azure ML workspace exists
# 2. Ensures our image is in ACR
# 3. Calls run_azure.py with appropriate args
# 4. Monitors progress
# 5. Downloads results when done
```

---

## File Changes Required

| File | Change | Effort |
|------|--------|--------|
| `mm_agents/openadapt/agent.py` | NEW: NaviAgent-compatible wrapper | ~100 lines |
| `run.py` | MODIFY: Add api-openai/api-claude agent options | ~10 lines |
| `waa_deploy/Dockerfile` | EXISTING: Already has api_agent.py | Done |
| `cli.py` | ADD: `scale` command | ~200 lines |
| `run_azure.py` | MODIFY: Default to our Docker image | ~5 lines |

---

## Prerequisites

### Azure Setup (One-time)

1. **Azure ML Workspace** (if not exists)
   ```bash
   az ml workspace create -n openadapt-ml -g openadapt-agents
   ```

2. **Azure Container Registry**
   ```bash
   az acr create -n openadaptacr -g openadapt-agents --sku Basic
   ```

3. **vCPU Quota** (request increase)
   - Standard_D8_v3: 8 vCPUs per worker
   - 10 workers = 80 vCPUs needed
   - Request via Azure Portal → Quotas

4. **Upload startup script** to Azure ML Notebooks
   - Path: `Users/<user>/compute-instance-startup.sh`
   - Content: From `scripts/azure_files/compute-instance-startup.sh`

### Environment Variables

```bash
# .env file
AZURE_SUBSCRIPTION_ID=...
AZURE_ML_RESOURCE_GROUP=openadapt-agents
AZURE_ML_WORKSPACE_NAME=openadapt-ml
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Cost Estimate

| Workers | VM Size | Time for 154 tasks | Compute Cost | API Cost (GPT-4o) | Total |
|---------|---------|-------------------|--------------|-------------------|-------|
| 1 | D8_v3 | ~5 hours | ~$2.50 | ~$5 | ~$7.50 |
| 5 | D8_v3 | ~1 hour | ~$2.50 | ~$5 | ~$7.50 |
| 10 | D8_v3 | ~30 min | ~$2.50 | ~$5 | ~$7.50 |

**Note:** More workers = faster, same total cost (compute + API calls are constant).

---

## Summary

| Aspect | Approach |
|--------|----------|
| **Parallelization** | Use official `run_azure.py` (Azure ML Compute) |
| **Docker Image** | Our `waa-auto:latest` (auto-download Windows, API agents) |
| **Agent** | Our `OpenAdaptAgent` wrapper (uses Claude/OpenAI) |
| **CLI** | Wrapper command `cli.py scale` |
| **Development** | Dedicated VM with VNC for debugging |

**Total new code:** ~300 lines
**Reused from official WAA:** Parallelization, compute management, task distribution
**Reused from our work:** Dockerfile, api_agent.py, WAALiveAdapter concepts

---

## Next Steps

1. [ ] Create `mm_agents/openadapt/agent.py` wrapper (~100 lines)
2. [ ] Test on dedicated VM with `--agent api-openai`
3. [ ] Push `waa-auto:latest` to Azure Container Registry
4. [ ] Modify `run_azure.py` to use our image
5. [ ] Add `scale` command to CLI
6. [ ] Request vCPU quota increase (80+ for 10 workers)
7. [ ] Run full 154-task benchmark
