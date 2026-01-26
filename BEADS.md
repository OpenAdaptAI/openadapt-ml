# BEADS - Build Evolution And Decision Summary

This document tracks significant project decisions, architecture choices, and key learnings that persist across conversation compactions. Each "bead" represents a notable change worth remembering.

---

## 2026-01-24: Windows Product Key Prompt Fix (CRITICAL)

**Category**: bugfix, docker, regression

**Summary**: Fixed Windows asking for product key during installation. The Dockerfile was REPLACING dockurr/windows's autounattend.xml with windowsarena's version, which broke the OOBE flow.

**Root Cause**: Dockerfile line `COPY --from=windowsarena/winarena:latest /run/assets/win11x64-enterprise-eval.xml /run/assets/win11x64.xml` replaced dockurr/windows's native autounattend.xml which handles OOBE properly.

**The Fix** (from commits 914513e and 6b9f744):
- DO NOT replace dockurr/windows's autounattend.xml
- Instead, PATCH it to add InstallFrom element (prevents "Select OS" dialog)
- VERSION="11e" is CORRECT - it downloads Enterprise Evaluation with built-in GVLK key

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile` (lines 62-70)

**Key Learning**: Never replace dockurr/windows's autounattend.xml - only patch it. The base image's XML handles OOBE properly for the editions it downloads.

---

## 2026-01-24: Automatic Disk Space Management

**Category**: feature, reliability

**Summary**: Added automatic disk space check and cleanup to `vm run-waa` command. Before starting Windows, the CLI now checks if /mnt has at least 15GB free. If not, it automatically runs Docker cleanup (`docker system prune -af --volumes`) to reclaim space. This prevents the recurring "Not enough free space in /storage" errors during Windows extraction.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py`
  - Added `ensure_sufficient_disk_space()` function (lines 246-305)
  - Called from `run-waa` action before starting container (lines 3130-3135)

**Why It Matters**: Windows extraction requires ~9GB, and Docker images/containers accumulate over time. Previously, users would hit disk space errors mid-setup, requiring manual intervention. Now the CLI proactively ensures space is available.

**Behavior**:
1. Check /mnt free space
2. If < 15GB, run `docker system prune -af --volumes`
3. Re-check space
4. If still < 15GB, provide manual cleanup instructions and exit

---

## 2026-01-24: Dockerfile CMD Fix - Missing /copy-oem.sh Script

**Category**: bugfix, docker

**Summary**: The waa-auto Dockerfile had a CMD that referenced a non-existent `/copy-oem.sh` script, causing containers to crash immediately on startup with "No such file or directory". Fixed by removing the reference since OEM files are already copied via modified `samba.sh`.

**Root Cause**: The Dockerfile CMD was `["/copy-oem.sh /entry.sh --start-client false"]` but no script at `/copy-oem.sh` was ever created. The OEM file copying was actually handled by a `sed` modification to `/run/samba.sh` (line ~59) that injects `cp -r /oem/* /tmp/smb/` before the return statement.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile` (line 230)

**Fix**: Changed CMD from `["/copy-oem.sh /entry.sh --start-client false"]` to `["/entry.sh --start-client false"]`

**Workaround**: To start an existing (broken) image, override the command:
```bash
docker run ... waa-auto:latest /entry.sh --start-client false
```

**Why It Matters**: Containers would crash immediately after creation, making it impossible to run the WAA benchmark without manual intervention.

---

## 2026-01-24: Windows CMD UNC Path Limitation

**Category**: learning, windows

**Summary**: Windows CMD.EXE does not support UNC paths (like `\\host.lan\Data`) as the current directory. When running batch scripts from a network share, `cd /d \\host.lan\Data\server` silently fails and the current directory remains unchanged.

**Symptoms**:
- Script runs from `\\host.lan\Data\script.bat` via Win+R
- `cd /d \\host.lan\Data\server` appears to succeed
- But `%CD%` shows `C:\Windows` (unchanged)
- Python tries to run `main.py` from wrong directory

**Fix**: Map the network share to a drive letter first:
```batch
net use Z: \\host.lan\Data /persistent:no
cd /d Z:\server
```

**Key Learning**: Always use `net use` to map UNC paths to drive letters before using `cd` in batch scripts that need to change to a network directory.

---

## 2026-01-24: VNC View Only Mode Toggle

**Category**: feature, ux

**Summary**: Added "View Only" toggle switch to the VNC controls in the Azure Operations dashboard. When enabled, an overlay blocks keyboard and mouse events from reaching the embedded VNC iframe, allowing users to watch the VM without accidentally sending input. The toggle state persists in localStorage.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/training_output/azure_ops.html` - Added CSS for toggle switch and input blocker overlay, HTML toggle control, JavaScript `toggleViewOnly()` function with localStorage persistence
- `/Users/abrichr/oa/src/openadapt-ml/training_output/current/azure_ops.html` - Copy for served version

**Implementation Details**:
1. CSS toggle switch styled to match dashboard theme (accent color when active)
2. Transparent overlay div positioned over the VNC iframe to intercept all pointer/keyboard events
3. Visual indicator in top-right corner when view-only mode is active ("View Only Mode - Input Disabled")
4. Preference persisted in localStorage (`vnc_view_only_mode`) and restored on page load

**Why It Matters**: Users need to monitor Windows VM activity during benchmark runs without accidentally interfering. This was previously listed as BUG-006 in the dashboard bugs document - the toggle was mentioned but never implemented.

---

## 2026-01-24: Auto-Shutdown to Prevent Runaway VM Costs

**Category**: feature, cost-optimization

**Summary**: Implemented default auto-shutdown for `vm monitor` command to prevent forgotten VMs from accumulating costs. The default is now 2 hours (was disabled). Users receive warnings at 15 minutes and 5 minutes before shutdown. Can be extended with `--auto-shutdown-hours N` or disabled with `--auto-shutdown-hours 0`.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py` - Updated default and added warnings
- `/Users/abrichr/oa/src/openadapt-ml/docs/auto_shutdown_design.md` - Design doc for future Azure Automation
- `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md` - Documentation update

**Why It Matters**: A forgotten VM costs ~$0.42/hr ($10/day, $70/week). This change saves money by default while still allowing users to extend or disable as needed. Future Phase 2/3 will add server-side Azure Automation for shutdown even when client disconnects.

**Cost Reference**:
- Standard_D4ds_v5: $0.422/hr
- 4 hours idle (common oversight): $1.69
- 24 hours idle (weekend forgotten): $10.13
- 1 week idle: $70.90

---

## 2026-01-24: Client-Side Evaluation Module (openadapt-evals)

**Category**: feature, architecture

**Summary**: Created client-side evaluation infrastructure in the `openadapt-evals` repository. The `EvaluatorClient` runs WAA evaluators locally by making HTTP calls to the WAA server's `/execute` endpoint. Includes `VMIPDiscovery` for auto-detecting VM IP from multiple sources (Azure CLI, SSH config, environment variables).

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/evaluation/client.py` - EvaluatorClient
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/evaluation/ip_discovery.py` - VMIPDiscovery
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/evaluation/__init__.py`

**Why It Matters**: Eliminates complexity of running evaluation as a separate service inside Docker. Evaluators just make HTTP calls and can run from anywhere with network access. Follows WAA's own design pattern (`run.py` uses client-side evaluation).

---

## 2026-01-24: Install Script Progress Indicators

**Category**: feature, ux

**Summary**: Created `install.bat` with step-by-step progress indicators for Windows installation inside the WAA Docker container. Shows window title updates for 14 installation steps (Python, ChromeDriver, LibreOffice, VSCode, WAA server, etc.). The script is copied into the Dockerfile to replace the original silent install.bat.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/install.bat`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile`

**Why It Matters**: Previously, Windows installation appeared to hang with no progress feedback. Now VNC viewers can see exactly which step is running via the window title (e.g., "Step 3/14: Installing ChromeDriver...").

---

## 2026-01-24: Dashboard Consolidation (benchmark.html to ops.html)

**Category**: refactor, ux

**Summary**: Renamed `benchmark.html` to `ops.html` and updated the shared header from "Benchmarks" to "Operations" to better reflect the dashboard's broader purpose (VM management, Azure ops, not just benchmarks). Converted legacy `viewer.py` to a deprecation shim that re-exports from shared_ui.py.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/training/shared_ui.py` - Updated header text
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/cloud/local.py` - Updated filename references
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/benchmark_viewer.py` - Updated references
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py` - Updated CLI output messages
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/viewer.py` - Now deprecation shim

**Why It Matters**: Clearer naming reflects actual usage. The dashboard manages VM operations, Azure resources, and SSH tunnels - not just benchmarks.

---

## 2026-01-24: Azure Ops Status API Enhancement

**Category**: feature, api

**Summary**: Enhanced the `/api/azure-ops-status` endpoint with benchmark detection, log fetching from VM, and better phase/operation state detection. The API now returns richer status information for the Azure operations dashboard.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/cloud/local.py`

**Why It Matters**: Dashboard can now show more detailed status including whether benchmarks are running, container logs, and accurate phase detection (setup, ready, benchmark, error states).

---

## 2026-01-24: VNC Iframe Flicker Fix

**Category**: bug-fix, ux

**Summary**: Fixed VNC iframe flickering on the Azure operations dashboard (azure_ops.html). Added state tracking variables (`vncIframeLoaded`, `lastKnownVmState`) to prevent unnecessary iframe reloads. The VNC iframe now only reacts to actual state transitions (offline to online, or vice versa) rather than reloading on every status poll.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/training_output/azure_ops.html`

**Why It Matters**: VNC was flickering every few seconds during status polling, making it unusable for monitoring. Now maintains stable connection while still updating status displays.

---

## 2026-01-16: Unified Baseline Adapters for VLM Comparison

**Category**: feature, architecture

**Summary**: Implemented a comprehensive baseline adapter system for comparing VLM providers (Claude, GPT, Gemini) across three evaluation tracks: direct coordinate prediction (Track A), ReAct-style reasoning (Track B), and Set-of-Mark element selection (Track C).

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/baselines/__init__.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/baselines/adapter.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/baselines/config.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/baselines/parser.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/baselines/prompts.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/models/providers/`

**Why It Matters**: Enables systematic comparison of off-the-shelf VLMs before fine-tuning. Based on SOTA patterns from Claude Computer Use, Microsoft UFO/UFO2, OSWorld benchmark, and Agent-S/Agent-S2.

---

## 2026-01-16: Benchmark Migration to openadapt-evals Package

**Category**: architecture, refactor

**Summary**: Consolidated benchmark code into a separate `openadapt-evals` package. The `openadapt_ml/benchmarks/` directory now contains deprecation stubs that re-export from `openadapt-evals`.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/__init__.py`
- `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md` (updated documentation)

**Why It Matters**: Separates benchmark infrastructure from training code, enabling cleaner dependency management and allowing benchmarks to be used independently.

---

## 2026-01-16: Safety Gate and Perception Integration

**Category**: feature

**Summary**: Added a safety gate for runtime action validation and perception integration module for enhanced UI understanding.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/runtime/safety_gate.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/perception/integration.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/trace_export.py`

**Why It Matters**: Safety gate prevents potentially dangerous actions from executing. Perception integration provides richer UI understanding for agents.

---

## 2026-01-16: Representation Shootout Experiment Framework

**Category**: feature, experiment

**Summary**: Created a framework for systematically comparing different UI representation approaches (coordinates vs marks vs hybrid).

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/experiments/representation_shootout/`
- `/Users/abrichr/oa/src/openadapt-ml/docs/experiments/representation_shootout_design.md`

**Why It Matters**: Enables data-driven decisions about which UI representation approach works best for different scenarios.

---

## 2026-01-09: TRL + Unsloth Training Integration

**Category**: feature, architecture

**Summary**: Replaced custom training implementation with TRL (Transformer Reinforcement Learning) and Unsloth for faster, more efficient fine-tuning.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/training/trl_trainer.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/scripts/train.py`
- `/Users/abrichr/oa/src/openadapt-ml/tests/test_trl_trainer.py`

**Why It Matters**: TRL provides battle-tested training loops. Unsloth provides 2-4x speedup for LoRA fine-tuning. Version bumped to 0.2.0 for PyPI release.

---

## 2026-01-09: Enhanced VM CLI and WAA Deployment

**Category**: feature

**Summary**: Significantly enhanced the VM CLI with new commands for WAA deployment, diagnostics, and monitoring. Added custom `waa-auto` Docker image that auto-downloads Windows 11.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/api_agent.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/cloud/ssh_tunnel.py`

**New CLI Commands**:
- `vm monitor` - Dashboard with auto-SSH tunnels
- `vm setup-waa` - Full VM setup with Docker
- `vm run-waa` - Run benchmark with agent options
- `vm diag` - Check disk, Docker, containers
- `vm logs` - View container logs
- `vm probe` - Check WAA server status
- `vm exec` - Run command in container
- `vm docker-prune` - Clean Docker resources
- `vm deallocate` / `vm start` - VM power management

**Why It Matters**: Eliminates need for manual SSH commands. All VM operations now have CLI wrappers for consistency and documentation.

---

## 2026-01-04: VM Monitor Dashboard with Auto-Shutdown

**Category**: feature

**Summary**: Added `vm monitor` command that opens a real-time dashboard in the browser, automatically manages SSH tunnels for VNC access, and supports auto-shutdown to prevent runaway costs.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py`
- `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md` (mandatory dashboard note moved to TOP)

**Why It Matters**: The dashboard is the single entry point for VM operations. Auto-shutdown prevents billing surprises when VMs are left running.

---

## 2026-01-03: Demo Retrieval System and WAA Live Adapter

**Category**: feature

**Summary**: Implemented demo retrieval system for automatically selecting relevant demonstrations from a library based on task similarity.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/retrieval/` (if exists)
- `/Users/abrichr/oa/src/openadapt-ml/docs/experiments/demo_conditioned_prompting_results.md`

**Why It Matters**: Validated that demo-conditioned prompting improves action accuracy from 33% (zero-shot) to 100% (with demo).

---

## 2026-01-03: WAA Docker Image Fixes

**Category**: bug-fix

**Summary**: Fixed multiple issues with WAA Docker image: Python 3.13 compatibility, missing client dependencies (pydrive, openpyxl, docx), OEM files for Windows installation.

**Key Commits**:
- `e5b3dc0` - Copy Python env from official image to avoid 3.13 compat issues
- `02e5e2f` - Add remaining WAA client dependencies
- `ebdc4f6` - Add missing pydrive and other client dependencies

**Why It Matters**: The official `windowsarena/winarena:latest` image is broken. Custom `waa-auto` image is required for automated Windows setup.

---

## 2026-01-02: Schema Consolidation

**Category**: architecture, refactor

**Summary**: Consolidated multiple schema variants into a single Pydantic-based Episode module with converters for external formats.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/schemas/`
- `/Users/abrichr/oa/src/openadapt-ml/docs/schema_consolidation_plan.md`

**Why It Matters**: Maintains schema purity - external systems adapt to the schema, not vice versa. Simplifies data flow through the system.

---

## 2026-01-02: WAA Demo-Conditioned Experiment

**Category**: feature, experiment

**Summary**: Added demo-conditioned experiment infrastructure with 7 manual demos recorded on Windows.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/agent.py` (DemoConditionedAgent)
- `/Users/abrichr/oa/src/openadapt-ml/docs/waa_demo_recording_guide.md`

**Why It Matters**: Proves the core value proposition - trajectory-conditioned disambiguation of UI affordances improves accuracy significantly.

---

## Uncommitted Changes (as of 2026-01-24)

### Session Tracker
**Category**: feature (uncommitted)

**File**: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/session_tracker.py`

**Summary**: Persists VM runtime and cost across page refreshes. Tracks session start time, elapsed time, accumulated seconds (for pause/resume), and computes cost based on VM hourly rates.

**Why It Matters**: Dashboard shows consistent cost/time values even after page refresh. Handles VM deallocate/start cycles correctly.

---

### Disk Manager
**Category**: feature (uncommitted)

**File**: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/disk_manager.py`

**Summary**: Automatic disk space monitoring and cleanup for Azure VM. Proactively cleans Docker resources when disk space drops below thresholds.

**Thresholds**:
- Warning: < 20GB free on /mnt
- Critical: < 10GB free on /mnt
- Auto-cleanup trigger: < 15GB free

**Cleanup Priority**:
1. Docker build cache
2. Unused Docker images
3. Stopped containers
4. Old Windows storage files
5. Old benchmark results

**Why It Matters**: Prevents disk full errors during Windows VM operations which can corrupt the Windows image.

---

### Dashboard Bug Inventory
**Category**: documentation (uncommitted)

**File**: `/Users/abrichr/oa/src/openadapt-ml/docs/DASHBOARD_BUGS.md`

**Summary**: Documented 11 known bugs in the Azure Ops dashboard:

| Bug ID | Description | Priority | Status |
|--------|-------------|----------|--------|
| BUG-001 | IP Address Flickering | P1 | Open |
| BUG-002 | Activity Detection Wrong State | P1 | Open |
| BUG-003 | SSE Connection Memory Leaks | P2 | **Fixed** |
| BUG-004 | Azure Jobs Polling Pauses | P2 | **Fixed** |
| BUG-005 | Session Tracker State Corruption | P2 | Open |
| BUG-006 | VNC Input Toggle Missing | P3 | **Fixed** |
| BUG-007 | Panel States Reset on Polling | P3 | Open |
| BUG-008 | Debug Console Logs | P3 | **Fixed** |
| BUG-009 | Hardcoded Azure Values | P4 | Open |
| BUG-010 | SSH Commands Not Using CLI | P4 | Open |

**Why It Matters**: Provides systematic tracking of dashboard issues with root cause analysis and fix approaches.

---

### Ingest Module
**Category**: feature (uncommitted)

**Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/ingest/__main__.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/ingest/base.py`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/ingest/external_example.py`

**Summary**: Data ingestion infrastructure for importing external data sources into the Episode schema.

---

## Key Architecture Decisions

### CLI-First Development
All VM operations should use CLI commands, not raw SSH. This ensures:
- Commands are documented
- Commands are tested
- Commands persist across context compactions
- Commands can be run by users or agents consistently

### Dashboard-First for VM Operations
Always start `vm monitor` before any VM operations. The dashboard:
- Auto-manages SSH tunnels (VNC at localhost:8006)
- Shows real-time VM status
- Provides all diagnostic information in one place

### Schema Purity
The schema must remain domain-agnostic:
- External systems adapt TO the schema
- Never add fields for specific integrations
- Use `raw` and `metadata` dicts for integration-specific data
- Data transformation belongs in importers/exporters

### Test in Container First
Before rebuilding Docker images (~30 min), test fixes inside running containers (~seconds):
```bash
docker run -d --name test-fix --entrypoint /bin/bash waa-auto:latest -c "sleep 3600"
docker exec test-fix sed -i 's/old/new/' /some/file.sh
docker exec test-fix /some/script.sh && ls /expected/output
docker rm -f test-fix
```

---

## Key Architectural Decision: Client-Side Evaluation (2026-01-24)

**Category**: architecture, decision

**File**: `/Users/abrichr/oa/src/openadapt-ml/docs/EVALUATION_ARCHITECTURE.md`

**Summary**: After analyzing three approaches (sidecar service, client-side, volume mount), client-side evaluation was chosen as the recommended architecture for benchmark evaluation.

**Decision**: Evaluators run client-side by importing from WAA vendor submodule and making HTTP calls to the WAA server's `/execute` endpoint.

**Rationale**:
1. Follows WAA's own design (`run.py` uses client-side evaluation)
2. Simplest architecture (no extra services or ports)
3. Already proven working (`StandaloneEvaluator` in `evaluate_endpoint.py`)
4. Generalizes to other benchmarks (WebArena, OSWorld use same pattern)

**Why It Matters**: Eliminates complexity of running evaluation as a separate service inside the Docker container. The key insight is that evaluators just make HTTP calls - they can run from anywhere with network access.

---

## 2026-01-24: Synthetic Task ID Validation (a9c8bae)

**Category**: bug-fix, architecture

**Summary**: Fixed misleading evaluation results caused by synthetic task IDs. Mock task IDs like `notepad_1` were being accepted by the live adapter, producing fake evaluation scores that appeared legitimate.

**Solution**:
- Added `is_real_waa_task_id()` validation function that checks task IDs against the real WAA task database
- Renamed mock task IDs from `notepad_1`, `chrome_1` to `mock_notepad_1`, `mock_chrome_1` prefix
- Live adapter now rejects synthetic IDs with a helpful error message pointing to the mock adapter
- Clear separation between mock testing and live evaluation

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/adapters/waa_live.py` - Added ID validation
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/adapters/waa.py` - Updated mock ID prefix
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/benchmarks/cli.py` - Updated CLI references

**Why It Matters**: Prevents false confidence from fake evaluation scores. Users now get clear feedback when using mock vs live evaluation.

---

## 2026-01-24: Unified Dashboard Command (a0eea20)

**Category**: feature, refactor

**Summary**: Replaced the crash-prone `vm monitor` command with a new `vm dashboard` command that uses subprocess-based SSH tunnels and generates a single HTML page with an embedded VNC iframe.

**Problem**: The original `vm monitor` crashed due to signal handling in threads, and multiple stale dashboard instances would accumulate.

**Solution**:
- Created new `dashboard.py` module (637 lines) with:
  - `SubprocessTunnel` class for reliable SSH tunnel management
  - Single-page HTML dashboard with VNC iframe
  - Real-time status polling
  - Clean process management (no orphan processes)
- Updated CLI to use new dashboard command
- Updated CLAUDE.md with new command documentation

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/dashboard.py` - New 637-line module
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py` - Added dashboard command
- `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md` - Updated documentation

**Why It Matters**: Reliable dashboard is essential for VM monitoring. The subprocess-based approach avoids Python threading issues with signal handling.

---

## 2026-01-24: Safe Process Management (ae8d78e)

**Category**: documentation, safety

**Summary**: Added comprehensive safe process management guidelines to CLAUDE.md after broad `pkill` patterns accidentally killed unrelated applications (Windsurf, Chrome tabs, Signal).

**Problem**: Commands like `pkill -9 -f "openadapt"` matched too broadly and killed important user applications.

**Solution**: Enhanced CLAUDE.md with:
- Banned patterns section (what NOT to do)
- Safe alternatives using port-based or PID-based killing
- Decision checklist: always run `pgrep -f "pattern" -l` first to see matches
- Examples of specific vs broad patterns

**Key File**:
- `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md` - Added "Safe Process Management" section

**Why It Matters**: Prevents accidental termination of user applications. Establishes pattern of checking before killing.

---

## 2026-01-24: WAA Server Diagnosis (a9eaa03)

**Category**: diagnosis, bug-fix

**Summary**: Diagnosed why WAA `/probe` endpoint was not responding. The status dashboard was showing incorrect information.

**Root Cause**: Container was running `dockurr/windows:latest` instead of `waa-auto:latest`. The base dockurr image doesn't have WAA server installed.

**Solution**: Identified issue and recommended rebuilding with `--rebuild` flag to ensure correct image is used.

**Diagnostic Steps**:
1. `vm diag` showed container running but probe failing
2. `vm exec --cmd "docker images"` revealed wrong image
3. `vm run-waa --rebuild` forces image rebuild

**Why It Matters**: Demonstrates importance of checking actual container image, not just container status.

---

## 2026-01-24: Disk Manager SSH Warning Fix (adf86b7)

**Category**: bug-fix

**Summary**: Fixed parsing errors in disk_manager.py caused by SSH warning messages like "Permanently added host to known hosts".

**Problem**: SSH outputs warning messages to stderr which were being mixed with stdout, causing JSON/number parsing to fail.

**Solution**: Updated disk_manager.py to filter out SSH warning messages before parsing command output.

**Key File**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/disk_manager.py` - Fixed SSH warning filtering

**Why It Matters**: Ensures reliable disk space monitoring even when SSH adds hosts to known_hosts file.

---

## 2026-01-24: Client-Side Evaluation Module (a63112b)

**Category**: feature, architecture

**Summary**: Added comprehensive client-side evaluation infrastructure to openadapt-evals, including VM IP auto-discovery, config management, and proper WAA task loading.

**Key Components**:
- `EvaluatorClient` - Runs WAA evaluators locally via HTTP calls to WAA server `/execute` endpoint
- `VMIPDiscovery` - Auto-detects VM IP from Azure CLI, SSH config, or environment variables
- Config management - Centralized benchmark configuration

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/evaluation/client.py` - EvaluatorClient
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/evaluation/discovery.py` - VMIPDiscovery
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/benchmarks/config.py` - Config management

**Why It Matters**: Enables running benchmark evaluations from the client machine without needing to deploy evaluator code inside the Docker container.

---

## 2026-01-24: Dockerfile Python Ordering Fix

**Category**: bug-fix

**Summary**: Fixed Docker build failure "python3: not found" at step 16/32. The Dockerfile had Python used (for api_agent patching) before it was installed.

**Problem**:
- Python installation was at lines 169-182
- Python used (python3 -c) at lines 95-100 for api_agent patching
- Build failed with "python3: not found"

**Solution**: Moved Python installation section to lines 84-127, before api_agent patching.

**Key File**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile`

**Why It Matters**: Without this fix, the waa-auto image cannot be built.

---

## 2026-01-24: Auto-Shutdown Design for Azure VMs

**Category**: feature, design

**Summary**: Designed auto-shutdown mechanism to prevent wasted VM idle time (~$0.42/hr).

**Problem**: VM was running for 4+ hours idle costing $1.74 because there was no automatic shutdown when not in use.

**Recommended Solution**: Azure Automation + CPU Alert
- Server-side execution (works even if client disconnects)
- Monitors CPU < 5% for 30 min, then deallocates
- Costs ~$0.10/month
- See `docs/auto_shutdown_design.md` for full design

**Quick Win**: Make `--auto-shutdown-hours 2` the default instead of opt-in.

**Key File**:
- `/Users/abrichr/oa/src/openadapt-ml/docs/auto_shutdown_design.md`

**Why It Matters**: Prevents billing surprises when VMs are forgotten.

---

## 2026-01-24: Disk Space Threshold Adjustment

**Category**: bug-fix

**Summary**: Lowered disk space requirement from 50GB to 35GB for WAA benchmark runs.

**Problem**: Benchmark was failing with "Insufficient disk space: 43GB (need 50GB)" even though Windows only needs ~30GB.

**Solution**: Changed `MIN_DISK_GB` from 50 to 35 in cli.py.

**Key File**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py` (lines 3155-3168)

**Why It Matters**: Allows benchmarks to run on VMs with smaller disks.

---

## 2026-01-24: Viewer Screenshot Embedding Default

**Category**: feature, ux

**Summary**: Changed viewer generation to embed screenshots as base64 by default.

**Problem**: viewer.html files referenced relative screenshot paths that broke when copied elsewhere.

**Solution**: Changed `--embed-screenshots` to `--no-embed-screenshots` (inverted default). Viewers are now ~20MB but fully portable.

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/benchmarks/cli.py` (cmd_view function)
- `/Users/abrichr/oa/src/openadapt-evals/openadapt_evals/benchmarks/viewer.py` (embed_screenshots parameter)

**Why It Matters**: Viewers can now be copied/shared without breaking screenshot references.

---

## Key Learnings from 2026-01-24 Session

1. **Always check Dockerfile ordering** when adding new build steps that use tools
2. **Auto-shutdown should be default**, not opt-in, for cloud VMs
3. **Conservative disk checks can block valid operations** - 35GB is sufficient for WAA
4. **Viewer portability matters** - embedding screenshots by default prevents broken references
5. **Multiple concurrent Docker builds cause issues** - ensure only one build runs at a time

---

---

## 2026-01-24: WAA Server Startup Script Fix

**Category**: bug-fix

**Summary**: Fixed WAA server not starting after install.bat completes. Root cause was Python not on PATH (`PrependPath=0`) and `start_waa_server.bat` using just `python` instead of full path.

**Problem**:
1. install.bat used `PrependPath=0` when installing Python
2. `start_waa_server.bat` called `python main.py` which failed (python not found)
3. WAA server never started, container showed "Waiting for response from windows server" forever

**Solution**:
1. Updated `install.bat` to use `PrependPath=1` (line 92)
2. Updated `start_waa_server.bat` to use full Python path: `C:\Users\Docker\AppData\Local\Programs\Python\Python310\python.exe`

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/install.bat`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/start_waa_server.bat`

**Why It Matters**: Critical for WAA benchmark to run end-to-end. Without this fix, manual VNC intervention required.

**Important Note**: Docker builds use files at build START. If you modify files after build starts, you must rebuild with `--rebuild`.

---

## 2026-01-24: Dashboard Liveness Indicators

**Category**: feature, ux

**Summary**: Added heartbeat indicator, current action display, and log freshness tracking to the Azure Ops dashboard.

**New Features**:
- Green pulsing dot (●) next to "Live Logs" shows connection is active
- "Current action" display shows what's happening (Downloading, Installing, etc.)
- "Log updated Xs ago" shows freshness of log data
- API returns `server_time`, `dockerfile_step`, `current_action`, `log_file_mtime`

**Key Files**:
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/cloud/local.py` - `_detect_docker_build_on_vm()` enhanced
- `/Users/abrichr/oa/src/openadapt-ml/training_output/azure_ops.html` - UI additions

**Known Issue**: Stale Docker build data causes contradictory display ("Dockerfile step 25/31" + "Log stale 36m ago"). Background agent ae154ac launched to fix this.

---

## 2026-01-24: Training Output File Serving

**Category**: bug-fix

**Summary**: Dashboard server serves from `training_output/current/`, not `training_output/`. Changes to azure_ops.html must be copied to the current directory.

**Problem**: Edits to `/training_output/azure_ops.html` weren't reflected in served dashboard.

**Solution**: Copy updated files: `cp training_output/azure_ops.html training_output/current/azure_ops.html`

**Why It Matters**: Prevents confusion when dashboard changes don't appear.

---

## Key Learnings from 2026-01-24 Evening Session

1. **Docker builds capture files at START** - modifying files during build has no effect
2. **Dashboard serves from training_output/current/** - must copy changes there
3. **PrependPath=1 is essential** for Python to be findable
4. **SSH timeouts cause stale detection** - API may not detect running containers if SSH is slow
5. **Contradictory UI states confuse users** - need to clear stale data when not applicable

*Last updated: 2026-01-25*

---

## 2026-01-25: Standalone WAA Build with Dev Mode - VERIFIED WORKING

**Category**: architecture, verification

**Summary**: After multiple failed approaches, found a **verified working** solution using vanilla WAA's "dev mode" with a standalone Dockerfile. The key insight: use Samba share (`\\host.lan\Data`) instead of OEM folder patching.

**What Works**:
1. Standalone Dockerfile copies from LOCAL vendor submodule (no COPY --from circular dependency)
2. Uses vanilla WAA's `dev_win11x64-enterprise-eval.xml` unattend (expects files at `\\host.lan\Data`)
3. Injects file copy into `samba.sh` to populate `/tmp/smb/` at container startup
4. Only patches IP addresses (20.20.20.21 → 172.30.0.2 for modern dockurr/windows)
5. Image size: **1.3GB** (vs 45GB official)

**Verification Results** (2026-01-25):
- [x] Windows ISO downloads automatically (VERSION=11e)
- [x] Windows installs unattended (no license key prompt)
- [x] FirstLogonCommands find scripts at `\\host.lan\Data`
- [x] setup.ps1 installs Python, Git, dependencies
- [x] WAA server starts on port 5000
- [x] `/probe` endpoint returns 200

**Key Files**:
- `/openadapt_ml/benchmarks/waa_deploy/Dockerfile` - Standalone build (159 lines)
- `/openadapt_ml/benchmarks/cli.py` - Updated to copy vendor files and build
- `/docs/waa_automation_checklist.md` - End-to-end verification checklist

**Why Previous Approaches Failed**:
1. **3-step vanilla build**: Required 50GB+ disk, downloads 15GB models
2. **COPY --from official image**: Circular dependency, old dockurr/windows base
3. **OEM folder with script patching**: Scripts expected `\\host.lan\Data`, not `C:\oem`

**The Fix**: Use vanilla WAA's dev mode as-is. The scripts already expect `\\host.lan\Data` (Samba share). Just copy files to `/tmp/smb/` when container starts.

**CLI Commands**:
```bash
# Full setup (builds 1.3GB image, ~5 min)
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa

# Run benchmark (first run downloads Windows, ~15 min)
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5
```

---
