# Infrastructure Refactor Design Document

**Author:** OpenAdapt Team
**Status:** Draft
**Last Updated:** January 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State](#2-current-state)
3. [Phase 1 - Internal Modularization](#3-phase-1---internal-modularization)
4. [Phase 2 - Extraction Criteria](#4-phase-2---extraction-criteria)
5. [Target API](#5-target-api)
6. [Market Context](#6-market-context)
7. [Migration Path](#7-migration-path)
8. [Appendix](#appendix)

---

## 1. Executive Summary

This document outlines a two-phase approach to refactor OpenAdapt's infrastructure code:

1. **Phase 1 (Internal):** Modularize existing infra code within `openadapt-ml` under a new `openadapt_ml/infra/` directory with clean abstractions.

2. **Phase 2 (Extraction):** When specific criteria are met, extract the infra module into a standalone `openadapt-infra` repository that can be used independently.

The goal is to establish OpenAdapt as a picks-and-shovels provider for agent infrastructure, specifically targeting Windows desktop automation - an underserved niche in the current market dominated by browser-focused solutions.

---

## 2. Current State

Infrastructure code is currently distributed across multiple locations within the `openadapt-ml` repository:

### 2.1 File Inventory

| File | Location | Purpose | Lines |
|------|----------|---------|-------|
| `cli.py` | `openadapt_ml/benchmarks/cli.py` | CLI for VM management, benchmark orchestration | ~1500+ |
| `azure.py` | `openadapt_ml/benchmarks/azure.py` | Azure ML orchestration, VM provisioning | ~791 |
| `waa_live.py` | `openadapt_ml/benchmarks/waa_live.py` | HTTP adapter for live WAA Windows VM connection | ~612 |
| `waa.py` | `openadapt_ml/benchmarks/waa.py` | WAA benchmark adapter with local/mock modes | ~762 |
| `waa/Dockerfile` | `openadapt_ml/benchmarks/waa/Dockerfile` | Custom Docker image combining dockurr/windows + WAA | ~181 |
| `lambda_labs.py` | `openadapt_ml/cloud/lambda_labs.py` | Lambda Labs GPU instance management | ~500+ |
| `local.py` | `openadapt_ml/cloud/local.py` | Local training/dashboard server | ~400+ |

### 2.2 Current Architecture

```
openadapt-ml/
├── openadapt_ml/
│   ├── benchmarks/
│   │   ├── cli.py           # VM subcommands (setup-waa, run-waa, etc.)
│   │   ├── azure.py         # AzureMLClient, AzureWAAOrchestrator
│   │   ├── waa.py           # WAAAdapter (local), WAAMockAdapter
│   │   ├── waa_live.py      # WAALiveAdapter (HTTP to Windows VM)
│   │   ├── base.py          # BenchmarkAdapter interface
│   │   └── waa/
│   │       └── Dockerfile   # Custom WAA Docker image
│   └── cloud/
│       ├── lambda_labs.py   # GPU instance management
│       ├── azure_inference.py
│       └── local.py         # Local training server
```

### 2.3 Key Classes and Responsibilities

#### Azure Infrastructure (`azure.py`)

```python
@dataclass
class AzureConfig:
    """Azure configuration for WAA deployment."""
    subscription_id: str
    resource_group: str
    workspace_name: str
    vm_size: str = "Standard_D2_v3"
    docker_image: str = "ghcr.io/microsoft/windowsagentarena:latest"

@dataclass
class WorkerState:
    """State of a single worker VM."""
    worker_id: int
    compute_name: str
    status: str  # pending, running, completed, failed
    assigned_tasks: list[str]

class AzureMLClient:
    """Wrapper around Azure ML SDK for compute management."""
    def create_compute_instance(self, name: str) -> str
    def delete_compute_instance(self, name: str) -> None
    def submit_job(self, compute_name: str, command: str, ...) -> str
    def wait_for_job(self, job_name: str, timeout_seconds: int) -> dict

class AzureWAAOrchestrator:
    """Orchestrates WAA evaluation across multiple Azure VMs."""
    def run_evaluation(self, agent, num_workers, task_ids, ...) -> list[BenchmarkResult]
```

#### Live Adapter (`waa_live.py`)

```python
@dataclass
class WAALiveConfig:
    """Configuration for WAALiveAdapter."""
    server_url: str = "http://localhost:5000"
    a11y_backend: str = "uia"
    screen_width: int = 1920
    screen_height: int = 1200

class WAALiveAdapter(BenchmarkAdapter):
    """Live WAA adapter that connects to WAA Flask server over HTTP."""
    def check_connection(self) -> bool
    def reset(self, task: BenchmarkTask) -> BenchmarkObservation
    def step(self, action: BenchmarkAction) -> tuple[BenchmarkObservation, bool, dict]
```

#### CLI VM Commands (`cli.py`)

```bash
# Current CLI structure
python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key KEY
python -m openadapt_ml.benchmarks.cli vm prepare-windows
python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5
python -m openadapt_ml.benchmarks.cli vm status
python -m openadapt_ml.benchmarks.cli vm ssh
python -m openadapt_ml.benchmarks.cli vm delete
```

### 2.4 Current Pain Points

1. **Tight Coupling:** Infrastructure code is interleaved with benchmark evaluation logic.
2. **No Abstraction Layer:** Azure and local implementations share no common interface.
3. **Import Dependencies:** Infra code imports from ML modules (`openadapt_ml.benchmarks.agent`).
4. **Testing Difficulty:** Hard to test infra independently of benchmark logic.
5. **Reusability:** Cannot use infra components without pulling in entire ML stack.

---

## 3. Phase 1 - Internal Modularization

### 3.1 Goals

- Create clean abstractions (Sandbox, Pool, Lease) that hide implementation details
- Enable swapping backends (Azure, Local, AWS, GCP) without changing client code
- Establish import boundaries - infra MUST NOT import from ML logic
- Prepare for potential extraction while maintaining internal usability

### 3.2 New Directory Structure

```
openadapt-ml/
├── openadapt_ml/
│   ├── infra/                    # NEW: Infrastructure module
│   │   ├── __init__.py           # Public API exports
│   │   ├── base.py               # Abstract interfaces
│   │   ├── config.py             # Infrastructure-specific config
│   │   │
│   │   ├── azure/                # Azure backend
│   │   │   ├── __init__.py
│   │   │   ├── client.py         # AzureMLClient (from azure.py)
│   │   │   ├── pool.py           # AzurePoolBackend
│   │   │   └── sandbox.py        # AzureSandbox implementation
│   │   │
│   │   ├── local/                # Local/Docker backend
│   │   │   ├── __init__.py
│   │   │   ├── pool.py           # LocalPoolBackend
│   │   │   └── sandbox.py        # DockerSandbox implementation
│   │   │
│   │   └── windows/              # Windows-specific utilities
│   │       ├── __init__.py
│   │       ├── adapter.py        # WAALiveAdapter (from waa_live.py)
│   │       ├── docker.py         # Windows Docker image management
│   │       └── Dockerfile        # Moved from benchmarks/waa/
│   │
│   ├── benchmarks/               # EXISTING: Slimmed down
│   │   ├── cli.py                # Calls into infra module
│   │   ├── base.py               # BenchmarkAdapter (unchanged)
│   │   ├── waa.py                # WAAAdapter (uses infra.windows)
│   │   └── ...
│   │
│   └── cloud/                    # EXISTING: Training-focused
│       ├── lambda_labs.py        # GPU training (separate concern)
│       └── local.py              # Local training/dashboard
```

### 3.3 Abstract Interfaces (`base.py`)

```python
"""Abstract interfaces for sandbox infrastructure.

These abstractions enable provider-agnostic sandbox management.
Implementations exist for Azure, Local Docker, AWS, etc.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class SandboxType(Enum):
    """Type of sandbox environment."""
    WINDOWS_11 = "windows-11"
    WINDOWS_10 = "windows-10"
    LINUX = "linux"
    MACOS = "macos"  # Future


class SandboxState(Enum):
    """Lifecycle state of a sandbox."""
    PENDING = "pending"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class SandboxConfig:
    """Configuration for a sandbox instance.

    Attributes:
        sandbox_type: Type of OS environment.
        image: Docker/VM image reference.
        cpu_cores: Number of CPU cores.
        memory_gb: RAM in gigabytes.
        disk_gb: Disk size in gigabytes.
        ports: Port mappings {host_port: container_port}.
        env_vars: Environment variables.
        labels: Metadata labels for filtering/grouping.
    """
    sandbox_type: SandboxType = SandboxType.WINDOWS_11
    image: str | None = None
    cpu_cores: int = 4
    memory_gb: int = 8
    disk_gb: int = 30
    ports: dict[int, int] = field(default_factory=lambda: {8006: 8006, 5000: 5000})
    env_vars: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Lease:
    """Represents exclusive access to a sandbox.

    A lease grants exclusive use of a sandbox for a bounded duration.
    Must be released when done to return sandbox to pool.

    Attributes:
        lease_id: Unique identifier for this lease.
        sandbox: The leased sandbox instance.
        expires_at: Unix timestamp when lease expires.
        metadata: Additional lease metadata.
    """
    lease_id: str
    sandbox: "Sandbox"
    expires_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def release(self) -> None:
        """Release the lease, returning sandbox to pool."""
        self.sandbox._pool.release(self)

    def renew(self, duration_seconds: float = 3600) -> None:
        """Extend the lease duration."""
        self.sandbox._pool.renew_lease(self, duration_seconds)


class Sandbox(ABC):
    """Abstract sandbox interface.

    A sandbox is an isolated execution environment (VM, container)
    where agent actions can be performed safely.
    """

    @property
    @abstractmethod
    def sandbox_id(self) -> str:
        """Unique identifier for this sandbox."""
        pass

    @property
    @abstractmethod
    def state(self) -> SandboxState:
        """Current lifecycle state."""
        pass

    @property
    @abstractmethod
    def config(self) -> SandboxConfig:
        """Sandbox configuration."""
        pass

    @abstractmethod
    def execute(self, command: str, timeout: float = 60.0) -> tuple[str, str, int]:
        """Execute a command in the sandbox.

        Args:
            command: Command to execute.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (stdout, stderr, exit_code).
        """
        pass

    @abstractmethod
    def screenshot(self) -> bytes:
        """Capture screenshot of sandbox display.

        Returns:
            PNG image bytes.
        """
        pass

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload file to sandbox."""
        pass

    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> None:
        """Download file from sandbox."""
        pass

    def get_accessibility_tree(self) -> dict | None:
        """Get UI accessibility tree (Windows/macOS).

        Returns:
            Accessibility tree dict, or None if not supported.
        """
        return None


class SandboxPool(ABC):
    """Abstract pool of sandboxes.

    Manages lifecycle of multiple sandboxes, handles scaling,
    and provides lease-based access to sandboxes.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the backend provider (azure, local, aws)."""
        pass

    @abstractmethod
    def spawn(
        self,
        count: int = 1,
        config: SandboxConfig | None = None,
        wait: bool = True,
    ) -> list[Sandbox]:
        """Spawn new sandbox instances.

        Args:
            count: Number of sandboxes to spawn.
            config: Configuration for new sandboxes.
            wait: Whether to wait for sandboxes to be ready.

        Returns:
            List of Sandbox instances.
        """
        pass

    @abstractmethod
    def lease(
        self,
        count: int = 1,
        duration_seconds: float = 3600,
        config: SandboxConfig | None = None,
    ) -> list[Lease]:
        """Acquire exclusive leases on sandboxes.

        If not enough sandboxes available, spawns new ones.

        Args:
            count: Number of leases needed.
            duration_seconds: Lease duration.
            config: Required sandbox configuration.

        Returns:
            List of Lease objects.
        """
        pass

    @abstractmethod
    def release(self, lease: Lease) -> None:
        """Release a lease, returning sandbox to available pool."""
        pass

    @abstractmethod
    def list_sandboxes(
        self,
        state: SandboxState | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[Sandbox]:
        """List sandboxes matching criteria.

        Args:
            state: Filter by state.
            labels: Filter by labels.

        Returns:
            List of matching sandboxes.
        """
        pass

    @abstractmethod
    def teardown(self, sandboxes: list[Sandbox] | None = None) -> None:
        """Terminate and cleanup sandboxes.

        Args:
            sandboxes: Specific sandboxes to teardown, or None for all.
        """
        pass

    def scale_to(self, count: int, config: SandboxConfig | None = None) -> list[Sandbox]:
        """Scale pool to target count.

        Spawns or terminates sandboxes as needed.
        """
        current = self.list_sandboxes(state=SandboxState.READY)
        current_count = len(current)

        if count > current_count:
            return current + self.spawn(count - current_count, config)
        elif count < current_count:
            to_terminate = current[count:]
            self.teardown(to_terminate)
            return current[:count]
        return current


class PoolBackend(ABC):
    """Factory for creating SandboxPool instances.

    Each cloud provider implements this to create pools.
    """

    @classmethod
    @abstractmethod
    def create_pool(cls, **kwargs) -> SandboxPool:
        """Create a new sandbox pool.

        Args:
            **kwargs: Provider-specific configuration.

        Returns:
            Configured SandboxPool instance.
        """
        pass
```

### 3.4 Import Rules

**Critical Rule:** The `openadapt_ml/infra/` module MUST NOT import from:
- `openadapt_ml/benchmarks/` (except `base.py` types)
- `openadapt_ml/models/`
- `openadapt_ml/training/`
- `openadapt_ml/retrieval/`

**Allowed imports:**
- Standard library
- Third-party packages (requests, azure-*, boto3, etc.)
- `openadapt_ml/config.py` (settings only)

**Enforcement:**
```python
# In openadapt_ml/infra/__init__.py
import sys

# Validate no forbidden imports at module load time
_FORBIDDEN_PREFIXES = [
    "openadapt_ml.benchmarks.agent",
    "openadapt_ml.benchmarks.runner",
    "openadapt_ml.models",
    "openadapt_ml.training",
    "openadapt_ml.retrieval",
]

def _check_imports():
    for name in sys.modules:
        for prefix in _FORBIDDEN_PREFIXES:
            if name.startswith(prefix):
                raise ImportError(
                    f"openadapt_ml.infra must not import {name}. "
                    f"This violates the infra module boundary."
                )

# Run check on import (can be disabled in production)
if __debug__:
    _check_imports()
```

### 3.5 Module Breakdown

#### `infra/azure/client.py`
Extracted from `benchmarks/azure.py`:
- `AzureMLClient` class
- Authentication logic
- Compute instance management

#### `infra/azure/pool.py`
New implementation:
- `AzurePoolBackend(PoolBackend)` factory
- `AzureSandboxPool(SandboxPool)` implementation

#### `infra/azure/sandbox.py`
New implementation:
- `AzureSandbox(Sandbox)` wrapping Azure compute instance
- Execute via SSH/Azure CLI
- Screenshot via VNC proxy

#### `infra/local/pool.py`
New implementation:
- `LocalPoolBackend(PoolBackend)` factory
- `LocalSandboxPool(SandboxPool)` using Docker

#### `infra/local/sandbox.py`
New implementation:
- `DockerSandbox(Sandbox)` wrapping Docker container
- Execute via docker exec
- Screenshot via dockurr/windows VNC

#### `infra/windows/adapter.py`
Moved from `benchmarks/waa_live.py`:
- `WAALiveAdapter` refactored to use `Sandbox` interface
- HTTP communication with WAA Flask server

---

## 4. Phase 2 - Extraction Criteria

The infrastructure module should be extracted into a standalone `openadapt-infra` repository when **ANY** of the following criteria are met:

### 4.1 External User Request

**Trigger:** A user or organization requests to use the infra module independently of the ML stack.

**Rationale:** External demand validates that the abstraction has value beyond our internal use case.

**Evidence:**
- GitHub issue requesting standalone package
- Direct inquiry from potential user/customer
- Fork of repo using only infra components

### 4.2 Multiple Backend Implementations

**Trigger:** Two or more fundamentally different sandbox backends are implemented and production-ready.

**Current state:**
- Azure: Implemented (azure.py)
- Local Docker: Partially implemented (waa/Dockerfile)
- AWS: Not implemented
- GCP: Not implemented

**Threshold:** 2+ backends with:
- All Sandbox interface methods implemented
- Integration tests passing
- Production usage for > 1 month

### 4.3 Scale Reliability

**Trigger:** Reliable spawning of 100+ sandboxes in a single pool.

**Rationale:** At scale, the infra becomes a critical path component worthy of independent versioning and release cycles.

**Metrics:**
- 100+ sandboxes spawned in single operation
- 95%+ success rate for spawn operations
- Mean time to ready < 5 minutes
- Demonstrated in production (not just test)

### 4.4 Monetization Path

**Trigger:** Clear revenue opportunity from infra-as-a-service.

**Possible paths:**
1. **Managed Service:** OpenAdapt-hosted sandbox pools (usage-based pricing)
2. **Enterprise License:** On-prem deployment for regulated industries
3. **Partnership:** Integration with existing cloud providers
4. **Consulting:** Professional services for custom deployments

**Threshold:** Any of:
- LOI or contract for infra service
- $10K+ in committed revenue
- Strategic partnership signed

---

## 5. Target API

The following API represents the aspirational interface for `openadapt-infra`:

### 5.1 Basic Usage

```python
from openadapt_infra import SandboxPool

# Create pool with Azure backend
pool = SandboxPool(backend="azure", sandbox_type="windows")

# Spawn 100 Windows sandboxes
sandboxes = pool.spawn(count=100)

# Execute commands on each sandbox
for sb in sandboxes:
    # Run application
    stdout, stderr, code = sb.execute("notepad.exe")

    # Capture screenshot
    screenshot = sb.screenshot()  # Returns PNG bytes

    # Get UI accessibility tree
    a11y_tree = sb.get_accessibility_tree()

# Clean up all sandboxes
pool.teardown()
```

### 5.2 Lease-Based Access

```python
from openadapt_infra import SandboxPool

pool = SandboxPool(backend="azure", sandbox_type="windows")

# Get exclusive lease on a sandbox
leases = pool.lease(count=5, duration_seconds=3600)

for lease in leases:
    sb = lease.sandbox
    try:
        sb.execute("myapp.exe")
        result = sb.screenshot()
    finally:
        # Release lease - sandbox returns to pool
        lease.release()
```

### 5.3 Configuration

```python
from openadapt_infra import SandboxPool, SandboxConfig, SandboxType

# Custom configuration
config = SandboxConfig(
    sandbox_type=SandboxType.WINDOWS_11,
    cpu_cores=8,
    memory_gb=16,
    disk_gb=100,
    ports={8006: 8006, 5000: 5000, 3389: 3389},
    env_vars={"OPENAI_API_KEY": "..."},
    labels={"project": "benchmark", "team": "ml"},
)

pool = SandboxPool(
    backend="azure",
    region="eastus",
    subscription_id="...",
)

sandboxes = pool.spawn(count=10, config=config)
```

### 5.4 Backend-Specific Options

```python
# Azure backend
from openadapt_infra.azure import AzurePoolBackend

pool = AzurePoolBackend.create_pool(
    subscription_id="...",
    resource_group="openadapt-agents",
    workspace_name="openadapt-ml",
    vm_size="Standard_D4ds_v5",  # Nested virt required
)

# Local Docker backend
from openadapt_infra.local import LocalPoolBackend

pool = LocalPoolBackend.create_pool(
    docker_host="unix:///var/run/docker.sock",
    network_mode="bridge",
    storage_path="/mnt/sandboxes",
)

# AWS backend (future)
from openadapt_infra.aws import AWSPoolBackend

pool = AWSPoolBackend.create_pool(
    region="us-east-1",
    instance_type="m5.xlarge",
    ami_id="ami-windows-2022",
)
```

### 5.5 Async Support

```python
import asyncio
from openadapt_infra import SandboxPool

async def main():
    pool = SandboxPool(backend="azure", sandbox_type="windows")

    # Async spawn
    sandboxes = await pool.spawn_async(count=100)

    # Parallel execution
    async def run_task(sb, task):
        await sb.execute_async(task.command)
        return await sb.screenshot_async()

    results = await asyncio.gather(*[
        run_task(sb, task)
        for sb, task in zip(sandboxes, tasks)
    ])

    await pool.teardown_async()

asyncio.run(main())
```

---

## 6. Market Context

### 6.1 The Picks-and-Shovels Opportunity

The agent infrastructure market is analogous to cloud computing in its early days:
- **Gold Rush:** Everyone building AI agents
- **Picks and Shovels:** Infrastructure to run those agents

Just as AWS provided compute infrastructure that enabled countless startups, agent infrastructure providers will enable the next wave of automation companies.

### 6.2 Existing Players

| Company | Focus | Strengths | Limitations |
|---------|-------|-----------|-------------|
| **CUA.ai** | Desktop automation sandboxes | Open-source, multi-OS (Windows/macOS/Linux), cloud + local, PyAutoGUI-compatible API, pay-per-use pricing | Cloud Windows sandboxes "coming soon" (currently Linux only); local Windows uses built-in Windows Sandbox (requires Windows host) |
| **Modal** | Serverless GPU/CPU | Simple API, fast cold starts | No desktop/VM support |
| **E2B** | Code execution sandboxes | Developer-friendly, secure | Linux-focused, no Windows |
| **Browserbase** | Browser automation | Playwright integration, scaling | Browser-only, no desktop apps |
| **Scrapfly** | Web scraping | Anti-bot handling | Browser-only |
| **AgentOps** | Observability | Good monitoring | Not an execution environment |

**Note on CUA.ai:** CUA (Containers for Computer-Use AI Agents) is the closest direct competitor to our infrastructure vision. Key capabilities:
- **Local providers:** Windows Sandbox (Windows host required), Lume (macOS VMs on macOS host), Docker (any OS)
- **Cloud:** Currently Linux-only sandboxes with noVNC access; Windows/macOS cloud "coming soon"
- **API:** Simple Python SDK for spawning sandboxes and executing actions
- **Pricing:** Credit-based, pay-per-use with three tiers (Small: 1 vCPU/4GB, Medium: 2 vCPU/8GB, Large: 8 vCPU/32GB)
- **Open-source:** Core framework is MIT-licensed (https://github.com/trycua/cua)

### 6.3 Our Differentiation

**Windows Desktop Focus:**
- **Scalable cloud Windows sandboxes today** - CUA.ai has announced Windows cloud support as "coming soon," but we offer production-ready Azure-based Windows VMs now
- Enterprise software is overwhelmingly Windows-based (SAP, Oracle, legacy apps)
- RPA market ($2.9B) is Windows-dominated but lacks modern agent support

**Competitive Positioning vs CUA.ai:**

CUA.ai is the closest competitor with overlapping goals. Our key differentiators:

| Capability | CUA.ai | OpenAdapt Infra |
|------------|--------|-----------------|
| Cloud Windows sandboxes | Coming soon (Linux only today) | **Available now** (Azure VMs) |
| Windows local | Requires Windows host (Windows Sandbox) | Docker-based (runs on any host with nested virt) |
| Accessibility tree (UIA) | Not mentioned | **Full UIA integration** |
| Benchmark integration | General-purpose | **Native WAA support** (154 tasks) |
| Enterprise cloud providers | Proprietary cloud only | Azure, AWS, GCP (your cloud) |
| On-prem deployment | Docker | Docker + VMs |

**Key Differentiators:**

1. **Production-Ready Cloud Windows**
   - Windows 11 VMs on Azure available today (not "coming soon")
   - Docker-based Windows (dockurr/windows) runs on any host with nested virtualization
   - Not limited to machines with Windows Sandbox pre-installed

2. **Accessibility Tree Integration**
   - Full UIA accessibility tree extraction (critical for reliable element targeting)
   - Element-based action grounding beyond just coordinates
   - This is table-stakes for enterprise RPA but missing from CUA.ai's described capabilities

3. **Agent-First Design**
   - Built for LLM agents, not legacy RPA
   - Screenshot + accessibility tree observations
   - Evaluation infrastructure included

4. **Benchmark Integration**
   - Native WAA (Windows Agent Arena) support
   - 154 pre-built evaluation tasks
   - Reproducible benchmarking at scale

5. **Enterprise Cloud Flexibility**
   - Deploy in YOUR cloud (Azure, AWS, GCP) - not locked to vendor cloud
   - On-premise (Docker) option for regulated industries
   - Hybrid (cloud control plane, on-prem execution)

### 6.4 Market Sizing

**Total Addressable Market (TAM):**
- RPA Market: $2.9B (2023), 23.4% CAGR
- AI Agent Infrastructure: $1.2B (2024), 45% CAGR (estimated)

**Serviceable Addressable Market (SAM):**
- Windows Desktop Automation: ~$800M
- Enterprise Agent Deployment: ~$400M

**Serviceable Obtainable Market (SOM):**
- Initial target: Research/benchmark users
- 5-year target: 1-2% of Windows automation = $8-16M ARR

---

## 7. Migration Path

### 7.1 Phase 1 Migration Steps

#### Step 1: Create Directory Structure (Day 1)

```bash
mkdir -p openadapt_ml/infra/{azure,local,windows}
touch openadapt_ml/infra/__init__.py
touch openadapt_ml/infra/base.py
touch openadapt_ml/infra/config.py
```

#### Step 2: Define Interfaces (Week 1)

1. Copy `Sandbox`, `SandboxPool`, `Lease` interfaces to `infra/base.py`
2. Define `SandboxConfig` dataclass
3. Add import boundary enforcement

#### Step 3: Extract Azure Code (Week 2)

1. Copy `AzureMLClient` to `infra/azure/client.py`
2. Remove benchmark-specific logic
3. Create `AzureSandbox` implementing `Sandbox` interface
4. Create `AzureSandboxPool` implementing `SandboxPool`
5. Update `AzureConfig` to use `SandboxConfig`

**Mapping:**
```
benchmarks/azure.py::AzureConfig      -> infra/azure/config.py::AzurePoolConfig
benchmarks/azure.py::AzureMLClient    -> infra/azure/client.py::AzureMLClient
benchmarks/azure.py::WorkerState      -> infra/azure/pool.py::AzureSandboxState
benchmarks/azure.py::EvaluationRun    -> (removed - benchmark-specific)
benchmarks/azure.py::AzureWAAOrchestrator -> (stays in benchmarks, uses infra)
```

#### Step 4: Extract Windows/WAA Code (Week 3)

1. Move `waa_live.py` to `infra/windows/adapter.py`
2. Move `waa/Dockerfile` to `infra/windows/Dockerfile`
3. Create `WindowsSandbox` wrapping HTTP adapter
4. Remove benchmark-specific evaluation logic

**Mapping:**
```
benchmarks/waa_live.py::WAALiveConfig   -> infra/windows/config.py::WindowsConfig
benchmarks/waa_live.py::WAALiveAdapter  -> infra/windows/adapter.py::WindowsAdapter
benchmarks/waa/Dockerfile               -> infra/windows/Dockerfile
```

#### Step 5: Create Local Backend (Week 4)

1. Create `DockerSandbox` for local Docker execution
2. Create `LocalSandboxPool` for Docker container management
3. Test with existing `waa/Dockerfile`

#### Step 6: Update Benchmarks to Use Infra (Week 5)

1. Update `cli.py` VM commands to use `infra.SandboxPool`
2. Update `waa.py` to use `infra.windows.WindowsAdapter`
3. Update `azure.py` orchestrator to use `infra.azure.AzureSandboxPool`
4. Deprecate direct imports of moved classes

#### Step 7: Add Tests and Documentation (Week 6)

1. Unit tests for all interface implementations
2. Integration tests for Azure and Local backends
3. Update CLAUDE.md with new architecture
4. Add migration guide for existing code

### 7.2 Backwards Compatibility

During migration, maintain backwards compatibility:

```python
# benchmarks/azure.py
from openadapt_ml.infra.azure import AzureMLClient as _AzureMLClient

# Deprecated: Use openadapt_ml.infra.azure.AzureMLClient
class AzureMLClient(_AzureMLClient):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "AzureMLClient moved to openadapt_ml.infra.azure. "
            "Update imports to: from openadapt_ml.infra.azure import AzureMLClient",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

### 7.3 Phase 2 Extraction Process

When extraction criteria are met:

1. **Create New Repository**
   ```bash
   gh repo create openadapt/openadapt-infra --public
   ```

2. **Copy Infrastructure Module**
   ```bash
   cp -r openadapt_ml/infra openadapt-infra/openadapt_infra
   ```

3. **Set Up Package**
   - Create `pyproject.toml` with proper metadata
   - Add CI/CD workflows
   - Configure PyPI publishing

4. **Update openadapt-ml**
   - Add `openadapt-infra` as dependency
   - Replace `openadapt_ml/infra/` with import redirects
   - Update documentation

5. **Versioning**
   - Initial release: `openadapt-infra==0.1.0`
   - Follow semver for API stability

---

## Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| **Sandbox** | Isolated execution environment (VM, container) for running agent actions |
| **Pool** | Collection of sandboxes managed together with scaling/lifecycle |
| **Lease** | Exclusive access grant to a sandbox for bounded duration |
| **Backend** | Cloud provider or runtime (Azure, AWS, Docker, etc.) |
| **WAA** | Windows Agent Arena - benchmark with 154 Windows tasks |

### B. Related Documents

- `docs/benchmark_integration_plan.md` - Benchmark integration architecture
- `docs/azure_waa_setup.md` - Azure WAA setup guide
- `docs/waa_setup.md` - WAA setup guide
- `docs/unified_compute_architecture.md` - Compute architecture design
- `CLAUDE.md` - Project context and conventions

### C. References

- [Windows Agent Arena](https://github.com/microsoft/WindowsAgentArena)
- [CUA.ai](https://cua.ai/) - Desktop automation sandboxes (closest competitor)
- [CUA GitHub](https://github.com/trycua/cua) - Open-source CUA framework
- [Modal](https://modal.com/) - Serverless compute
- [E2B](https://e2b.dev/) - Code sandboxes
- [Browserbase](https://browserbase.com/) - Browser automation
- [Azure ML Compute](https://docs.microsoft.com/azure/machine-learning/concept-compute-instance)

### D. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01 | Phase 1 before Phase 2 | Internal modularization validates abstractions before extraction |
| 2026-01 | Lease-based access model | Enables resource sharing and cost optimization |
| 2026-01 | Windows-first strategy | Underserved market, aligns with WAA benchmark |
| 2026-01 | Four extraction criteria | Prevents premature optimization while ensuring readiness |
| 2026-01 | Updated competitive analysis (CUA.ai) | CUA.ai emerged as closest competitor; differentiation updated to emphasize production-ready cloud Windows (vs their "coming soon"), UIA integration, enterprise cloud flexibility |
