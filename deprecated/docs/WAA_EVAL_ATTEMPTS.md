# WAA Evaluation Attempts - Comprehensive History

**Document Created**: 2026-01-20
**Purpose**: Track all attempts to run Windows Agent Arena (WAA) benchmark evaluations, including what was tried, what failed, and lessons learned.

---

## 1. Problem Statement

### Goal
Run the Windows Agent Arena (WAA) benchmark on Azure VMs to evaluate GUI automation agents and establish baseline performance metrics.

### What WAA Requires
1. **Azure VM with nested virtualization** - Windows 11 runs inside QEMU inside Docker
2. **Docker container with Windows 11** - Using dockurr/windows for auto-download
3. **WAA server running inside Windows** - Flask server on port 5000 with `/probe`, `/execute`, `/screenshot` endpoints
4. **Benchmark client on Linux host** - Sends tasks to Windows, captures results

### State-of-the-Art
- SOTA on WAA: ~19.5% success rate (GPT-5.1 + OmniParser)
- Our best run: 12.5% (1/8 tasks) on January 6, 2026

### What Keeps Failing
1. **Windows ISO not downloading** - Container stuck on "ISO file not found"
2. **Autounattend.xml issues** - Windows prompts for image selection or gets stuck
3. **OEM files not available** - install.bat not accessible via \\host.lan\Data
4. **WAA server never starts** - FirstLogonCommands fail silently
5. **Disk space exhaustion** - Docker images fill up /dev/sda1 (30GB OS disk)
6. **Docker builds cancelled/failed** - Low disk space causes build failures
7. **VM idle/waste** - VMs left running without auto-shutdown

---

## 2. Timeline of Attempts

### January 6, 2026 - First Successful Benchmark Run (Partial)
**Attempt**: Run WAA benchmark with Navi agent (GPT-4o)

**Outcome**:
- 8 of 19 tasks attempted (42%)
- 1 of 8 passed (12.5%) - "Open Details view in Explorer"
- SSH timeout at ~1.5 hours terminated the run

**Key Issues**:
- Navi agent has fundamental bugs (`TypeError: expected string or bytes-like object, got 'NoneType'`)
- Command parsing failures on malformed action strings
- SSH connection instability

**Files Created**:
- `/Users/abrichr/oa/src/openadapt-ml/docs/experiments/waa_benchmark_results_jan2026.md`

---

### January 8, 2026 - VM Created
**Attempt**: Create Azure VM `waa-eval-vm` for dedicated WAA evaluation

**Outcome**: VM created successfully
- Size: Standard_D4ds_v5
- Location: westus2
- Nested virtualization: enabled
- Cost: ~$0.20/hour

---

### January 17, 2026 - Strategic Pivot to Validation
**Decision**: Stop all polish work (viewers, docs) and focus on WAA validation

**Rationale**:
- Built excellent infrastructure but no validation at scale
- Last evaluation: 0/1 success on live WAA
- Need quantitative performance data

**Files Created**:
- `/Users/abrichr/oa/src/STATUS.md` - Updated priorities

---

### January 18, 2026 (Afternoon) - Azure ML Job Stuck
**Attempt**: Run WAA evaluation via Azure ML orchestration

**Job ID**: `waa-waa3718w0-1768743963-20a88242`

**Outcome**: Job stuck in "Running" state for 8+ hours with 0/13 tasks completed

**Root Causes Identified**:
1. **TrustedLaunch security type** - Azure default since 2024 may disable nested virtualization
2. **Container startup failure** - Docker image never pulled successfully
3. **Silent failure mode** - Azure ML doesn't report container startup failures
4. **No health checks** - No verification between compute provisioning and job execution

**Files Created**:
- `/Users/abrichr/oa/src/openadapt-evals/AZURE_JOB_DIAGNOSIS.md`
- `/Users/abrichr/oa/src/openadapt-evals/AZURE_LONG_TERM_SOLUTION.md` (44KB)

---

### January 18, 2026 (Evening) - WAA Integration Fix
**Attempt**: Fix task loading and evaluator integration issues in openadapt-evals

**Outcome**: Code fixed, validation in progress

**Issues Fixed**:
- Task loading: Fixed relative path issues in evaluator registry
- Evaluator integration: Simplified by removing session dependency
- Path resolution: Made paths absolute and consistent

**Files Modified**:
- `openadapt_evals/adapters/waa_adapter.py`
- `openadapt_evals/tasks/registry.py`
- `openadapt_evals/evaluators/waa_evaluator.py`

---

### January 19, 2026 (Early Morning) - VM Idle Investigation
**Attempt**: Investigate why VM was running idle for 3+ hours

**Outcome**: Container stuck on "ISO file not found or is empty"

**Key Findings**:
1. Container waiting indefinitely for Windows ISO
2. Uses outdated `dockurr/windows v0.00` that does NOT auto-download Windows
3. No auto-shutdown configured on regular Azure VMs
4. Cost: ~$0.73 wasted in current session

**Immediate Action**: Deallocate VM to stop billing

**Files Created**:
- `/Users/abrichr/oa/src/openadapt-evals/VM_IDLE_INVESTIGATION.md`
- `/Users/abrichr/oa/src/openadapt-evals/VM_IDLE_ACTION_ITEMS.md`

---

### January 19, 2026 - Custom waa-auto Dockerfile Created
**Attempt**: Create custom Docker image that properly auto-downloads Windows

**Solution**: Build `waa-auto` image combining:
1. `dockurr/windows:latest` (auto-downloads Windows 11)
2. `windowsarena/winarena:latest` (WAA client/server scripts)

**Dockerfile Location**: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile`

**Key Features**:
- Uses modern dockurr/windows:latest base
- Copies OEM files from official WAA image
- Patches IP addresses (20.20.20.21 -> 172.30.0.2)
- Adds InstallFrom element to autounattend.xml for automatic image selection
- Adds FirstLogonCommands to auto-run install.bat and start WAA server
- Port forwards 5000 from container to Windows VM

---

### January 20, 2026 (Morning) - Multiple Docker Build Attempts
**Attempt**: Build waa-auto Docker image on Azure VM

**Outcome**: Multiple build failures and cancellations

**Issues**:
1. **Disk space critically low** (15GB available, need ~20GB for build)
2. **Build cache filling disk** - Need to clear before each build
3. **ISO download failing** - Windows 11 ISO ~6.6GB

**Evidence from task outputs**:
```
WARNING: Low disk space (15G). Build may fail.
Consider running: uv run python -m openadapt_ml.benchmarks.cli vm docker-prune
```

**Commands Run**:
- `docker-prune` to clear images and build cache
- `--rebuild` flag to force image rebuild

**Status**: Build cancelled mid-way multiple times

---

### January 20, 2026 (Midday) - Ongoing Attempts
**Current Status**: Multiple background agents attempting to:
1. Clear disk space
2. Rebuild waa-auto image
3. Monitor VM status
4. Run benchmark with 5 tasks

**Visible Patterns**:
- Builds starting but not completing
- Disk space constantly an issue
- Container layer extraction slow (14.37GB layer for winarena:latest)

---

## 3. Technical Issues Encountered

### Docker/Build Issues
| Issue | Description | Status |
|-------|-------------|--------|
| Disk space exhaustion | /dev/sda1 (30GB) fills up with Docker images | Recurring |
| Build cache growth | Build cache grows unboundedly | Mitigated with docker-prune |
| Large image layers | winarena:latest has 14.37GB layer | Inherent |
| Build cancellation | Builds cancelled due to space/time | Recurring |
| waa-auto vs winarena | Official winarena uses outdated dockurr/windows | Fixed with custom image |

### Windows Installation Issues
| Issue | Description | Status |
|-------|-------------|--------|
| ISO not found | dockurr/windows v0.00 doesn't auto-download | Fixed with latest dockurr |
| Image selection prompt | Multiple editions in install.wim | Fixed with InstallFrom element |
| Product key dialog | Non-enterprise editions need key | Fixed with VERSION=11e |
| AutoLogon not working | Windows stays at login screen | Fixed with password setting |
| Hardware checks | TPM/SecureBoot/RAM checks | Fixed in autounattend.xml |

### Autounattend.xml Issues
| Issue | Description | Status |
|-------|-------------|--------|
| Missing InstallFrom | "Select operating system" prompt | Fixed with sed patch |
| Empty password | AutoLogon fails without password | Fixed with docker password |
| FirstLogonCommands | Commands not running | Fixed with Python XML patcher |
| Multiple XML files | VERSION detection uses different files | Fixed by patching both |

### Dashboard/Viewer Issues
| Issue | Description | Status |
|-------|-------------|--------|
| Live monitoring broken | No task progress shown | Fixed - infrastructure works |
| Stale data display | Dashboard shows old elapsed time | Fixed with data loading |
| VNC access | Port 8006 not accessible directly | Fixed with SSH tunnel |

### Network/Tunnel Issues
| Issue | Description | Status |
|-------|-------------|--------|
| NSG blocking ports | 8006, 5000 not exposed | Fixed with SSH tunnel |
| SSH timeouts | Long-running benchmarks drop | Needs keepalive config |
| IP address mismatch | Official uses 20.20.20.21, dockurr uses 172.30.0.2 | Fixed with sed patches |
| Port forwarding | 5000 not forwarded to Windows VM | Fixed with nc loop |

### Disk Space Issues
| Issue | Description | Status |
|-------|-------------|--------|
| OS disk full | /dev/sda1 only 30GB | Use /mnt (147GB) |
| Docker data location | Docker uses OS disk by default | docker-move command |
| Windows ISO size | 6.6GB for Win11 | Inherent |
| qcow2 disk image | Grows to DISK_SIZE setting | Reduced to 20GB |

---

## 4. Fixes Applied

### Dockerfile Fixes (waa-auto)
1. **Base image**: Changed from `windowsarena/winarena:latest` to `dockurr/windows:latest`
2. **OEM files**: Copy from official image with `COPY --from=windowsarena/winarena:latest /oem /oem`
3. **IP patching**: `sed -i 's|20.20.20.21|172.30.0.2|g'` on all entry scripts
4. **Port forwarding**: Added nc loop script to forward 5000 to Windows VM
5. **InstallFrom**: Added XML element for automatic image selection
6. **Password**: Set `docker` password for AutoLogon
7. **FirstLogonCommands**: Added commands via Python XML patcher
8. **Environment**: Set VERSION=11e, DISK_SIZE=20G, RAM_SIZE=6G

### CLI Additions
- `vm setup-waa` - Full setup with Docker and waa-auto image
- `vm run-waa` - Run benchmark with --rebuild option
- `vm monitor` - Dashboard with SSH tunnels and VNC
- `vm diag` - Check disk, Docker, containers
- `vm docker-prune` - Clean images and build cache
- `vm docker-move` - Move Docker data to /mnt
- `vm probe --wait` - Check WAA server with polling

### Infrastructure Fixes
- SSH tunnel manager for VNC/WAA access
- Auto-shutdown recommendations documented
- Container health check timeout (15 min)

---

## 5. Current Status

### Working
- [x] Azure VM creation with nested virtualization
- [x] SSH tunnel management for VNC/WAA access
- [x] Dashboard with real-time VM status
- [x] Custom Dockerfile with all fixes
- [x] CLI commands for VM management
- [x] Mock evaluation pipeline (no Windows)

### In Progress
- [ ] Docker image build (disk space issues)
- [ ] Windows 11 auto-installation
- [ ] WAA server startup automation
- [ ] Full benchmark run (5+ tasks)

### Remaining Work
- [ ] Complete single successful benchmark run
- [ ] Validate WAA server responds to /probe
- [ ] Run 20-50 task evaluation
- [ ] Analyze failure modes
- [ ] Compare against SOTA (19.5%)

---

## 6. Lessons Learned

### Patterns That Didn't Work
1. **Using official winarena image directly** - Outdated base image doesn't auto-download Windows
2. **Assuming Azure ML handles containers** - Silent failures, no health checks
3. **Building on OS disk** - 30GB not enough for Docker images + Windows
4. **Manual ISO downloads** - Breaks automation, requires VNC interaction
5. **Assuming auto-shutdown** - Regular Azure VMs run indefinitely

### Patterns That Worked
1. **Custom Dockerfile combining images** - Modern dockurr + official WAA components
2. **SSH tunnels for port access** - Secure, works through NSG restrictions
3. **Python for XML patching** - More reliable than shell sed loops
4. **Multiple XML file patches** - VERSION detection uses different files
5. **Disk space monitoring** - Proactive cleanup before builds
6. **CLI-first development** - Document commands, not manual steps

### Key Technical Insights
1. **dockurr/windows version matters** - v0.00 vs latest is critical difference
2. **Autounattend.xml complexity** - Multiple elements needed for full automation
3. **Nested virtualization + TrustedLaunch** - May conflict, need Standard security type
4. **Windows 11 Enterprise Evaluation** - Best choice for automated setup
5. **Port 5000 forwarding** - dockurr doesn't auto-forward to QEMU guest

### Process Improvements Needed
1. **Test inside container first** - Don't rebuild for small changes
2. **Monitor disk space continuously** - Build failures are costly
3. **Document every attempt** - Context compactions lose history
4. **Use deallocate, not stop** - Stops billing completely
5. **Configure auto-shutdown** - Prevent waste from forgotten VMs

---

## 7. Reference Commands

### Start Fresh WAA Evaluation
```bash
# 1. Setup VM with Docker and build waa-auto image
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key $OPENAI_API_KEY

# 2. Run benchmark
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5

# 3. Monitor (opens dashboard with VNC)
uv run python -m openadapt_ml.benchmarks.cli vm monitor

# 4. Delete when done (IMPORTANT!)
uv run python -m openadapt_ml.benchmarks.cli vm delete -y
```

### Debugging Commands
```bash
# Check VM status
uv run python -m openadapt_ml.benchmarks.cli vm status

# Check disk, Docker, containers
uv run python -m openadapt_ml.benchmarks.cli vm diag

# View container logs
uv run python -m openadapt_ml.benchmarks.cli vm logs --lines 100

# Check WAA server
uv run python -m openadapt_ml.benchmarks.cli vm probe --wait

# Clean disk space
uv run python -m openadapt_ml.benchmarks.cli vm docker-prune

# SSH into VM
uv run python -m openadapt_ml.benchmarks.cli vm ssh
```

### Testing Without Windows (Mock)
```bash
uv run python -m openadapt_ml.benchmarks.cli test-mock --tasks 20
```

---

## 8. Files Referenced

### Key Documentation
- `/Users/abrichr/oa/src/STATUS.md` - Project-wide status
- `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md` - CLI and VM instructions
- `/Users/abrichr/oa/src/openadapt-ml/docs/azure_waa_setup.md` - Azure setup guide
- `/Users/abrichr/oa/src/openadapt-ml/docs/waa_setup.md` - WAA setup guide

### Investigation Reports
- `/Users/abrichr/oa/src/openadapt-evals/VM_IDLE_INVESTIGATION.md`
- `/Users/abrichr/oa/src/openadapt-evals/VM_IDLE_ACTION_ITEMS.md`
- `/Users/abrichr/oa/src/openadapt-evals/AZURE_JOB_DIAGNOSIS.md`
- `/Users/abrichr/oa/src/openadapt-evals/AZURE_LONG_TERM_SOLUTION.md`

### Benchmark Results
- `/Users/abrichr/oa/src/openadapt-ml/docs/experiments/waa_benchmark_results_jan2026.md`

### Docker Configuration
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile`

---

## 9. Update Log

| Date | Update |
|------|--------|
| 2026-01-20 | Document created with full history |
| 2026-01-20 | **ROOT CAUSE FOUND: Storage disk full** - /mnt (32GB) was 100% full, preventing Windows install |

---

## January 20, 2026 - Storage Disk Root Cause (CRITICAL FIX)

**Problem**: Windows installation stuck at "Installing 0%" for 40+ minutes

**Investigation**:
1. Checked container logs - endless "Waiting for a response from the windows server"
2. Checked disk space inside container: `/dev/sdb1 32G 32G 0 100% /storage`
3. Windows ISO (7GB) + data.img (20GB) = 27GB, filling 32GB disk completely
4. Windows couldn't write to disk during installation

**Root Cause**:
- CLI was hardcoded to use `/mnt/waa-storage` for Docker storage mount
- `/mnt` is Azure's 32GB ephemeral temp disk (sdb)
- The 128GB data disk we attached is mounted at `/data` (sdc)
- Container was using the wrong disk!

**Fix Applied**:
1. Changed all CLI references from `/mnt/waa-storage` to `/data/waa-storage`
2. Created new storage directory: `sudo mkdir -p /data/waa-storage`
3. Restarted container with correct mount: `-v /data/waa-storage:/storage`
4. Container now has 69GB free on storage (was 0GB)

**Verification**:
```bash
docker exec winarena df -h /storage
# Output: /dev/sdc1 126G 51G 69G 43% /storage  (previously: 100% full)
```

**Files Modified**:
- `openadapt_ml/benchmarks/cli.py` - Changed all `/mnt/waa-storage` to `/data/waa-storage`

**Lesson Learned**:
- Always verify disk space INSIDE the container, not just on host
- Azure VMs have multiple disks: OS (sda), temp (sdb/mnt), data (sdc/data)
- The CLI needs to target the correct disk for large workloads like WAA

---

**Next Update**: After 5-task WAA evaluation completes
