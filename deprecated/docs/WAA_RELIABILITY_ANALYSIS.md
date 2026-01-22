# WAA Benchmark System Reliability Analysis

**Document Created**: 2026-01-20
**Author**: Claude Code Agent
**Purpose**: Meta-analysis of WAA benchmark system fragility with root causes and solutions

---

## Executive Summary

After reviewing 2-3 weeks of failed WAA evaluation attempts documented in `WAA_EVAL_ATTEMPTS.md`, CLAUDE.md, and related investigation reports, this analysis identifies **systemic root causes** of the fragility and proposes **concrete solutions**.

**Key Finding**: The WAA benchmark system has **7 layers of complexity**, each with its own failure modes. The combination creates a "fragility cascade" where a small issue at any layer can cause complete system failure with unclear error messages.

**Reliability Score**: ~15% chance of success on first attempt (estimated from documented attempts)

---

## 1. Architecture Complexity Analysis

### The 7-Layer Stack

```
Layer 7: Benchmark Client (Python on Linux)
         |
Layer 6: SSH/Network (tunnels, ports 5000, 8006)
         |
Layer 5: Docker Container (waa-auto image)
         |
Layer 4: QEMU Virtualization (nested virt)
         |
Layer 3: Windows 11 OS Installation
         |
Layer 2: Azure VM (D4ds_v5 + nested virt)
         |
Layer 1: Azure Resource Management (RG, Storage, NSG)
```

### Failure Points Per Layer

| Layer | Component | Common Failures | Time to Detect |
|-------|-----------|-----------------|----------------|
| 7 | Benchmark Client | Model name typos, API errors | Seconds |
| 6 | SSH/Network | Port conflicts, tunnel failures, timeouts | Minutes |
| 5 | Docker | Build failures, disk space, image corruption | 10-30 min |
| 4 | QEMU/KVM | Nested virt disabled, CPU features | 5-15 min |
| 3 | Windows 11 | ISO download, install.wim selection, activation | 15-45 min |
| 2 | Azure VM | TrustedLaunch conflicts, wrong size | 5-10 min |
| 1 | Azure Resources | Quota limits, NSG rules, disk provisioning | 2-5 min |

**Total potential wait time before failure detection: 52-110 minutes**

This is the core problem: a failure at Layer 3 (Windows installation) requires waiting through Layers 1-4 before you even know something is wrong.

---

## 2. Root Cause Analysis

### RC1: Excessive Architectural Complexity

**Evidence**:
- 7 layers of abstraction between "run benchmark" and "benchmark running"
- Each layer has independent failure modes
- Error messages often don't propagate up the stack

**Specific Issues**:
1. Azure VM needs nested virtualization (specific VM sizes only)
2. Azure ML can't use nested virt (TrustedLaunch conflict)
3. Docker inside Azure VM needs KVM access
4. QEMU inside Docker needs CPU passthrough
5. Windows inside QEMU needs unattend.xml automation
6. WAA server inside Windows needs network bridging
7. Benchmark client needs SSH tunnels to reach WAA

**Impact**: A 1% failure rate at each layer compounds to ~7% chance of full-stack failure.

### RC2: Upstream Dependencies Are Brittle

**Evidence from attempts**:
1. `windowsarena/winarena:latest` uses outdated `dockurr/windows v0.00`
2. `dockurr/windows v0.00` doesn't auto-download Windows (breaks automation)
3. Microsoft's Windows ISOs change URLs/checksums periodically
4. dockurr/windows network config differs from official WAA (IP: 172.30.0.2 vs 20.20.20.21)

**Impact**: Upstream changes break our automation without warning.

### RC3: Silent Failure Modes

**Documented cases**:
1. Container stuck on "ISO file not found" - loops forever without exiting
2. Azure ML job "Running" for 8+ hours with 0 tasks - no error logged
3. Windows install at 0% for 40+ minutes - actually disk full, no error
4. FirstLogonCommands fail silently - no logs accessible

**Impact**: Hours wasted waiting for systems that will never recover.

### RC4: Resource Constraints Not Validated Upfront

**Documented cases**:
1. `/dev/sda1` (30GB OS disk) filled by Docker images
2. `/mnt` (32GB temp disk) filled by Windows ISO + disk image
3. `/data` disk exists but wasn't being used (CLI hardcoded wrong path)
4. Docker build cache grows unboundedly

**The storage disk confusion pattern repeated 3+ times**:
- Jan 19: "ISO file not found" - actually disk full
- Jan 20: "Installing 0%" stuck - storage disk 100% full
- Jan 20: Build cancelled - OS disk full from cache

**Impact**: Same failure mode reoccurred because resource validation wasn't systematic.

### RC5: Documentation Drift

**Evidence**:
- CLAUDE.md has 1147+ lines with many outdated sections
- Multiple fix attempts documented but fixes not applied
- CLI commands evolved but docs lagged behind
- "Status: WORKING" in docs, but actually broken

**Impact**: Each session starts with wrong assumptions, repeating past mistakes.

### RC6: Missing Health Checks

**What's missing**:
1. **Pre-flight checks**: Is the VM the right size? Is nested virt enabled? Is there enough disk space?
2. **Build-time checks**: Did the Docker build actually succeed? Is the image valid?
3. **Runtime checks**: Is Windows actually installing? Is the WAA server starting?
4. **Post-install checks**: Did install.bat run successfully? Are dependencies installed?

**Impact**: Problems discovered late in the pipeline when recovery is expensive.

### RC7: Context Loss Between Sessions

**Pattern observed**:
1. Session starts, previous context compacted
2. Agent doesn't know about previous fixes
3. Same debugging steps repeated
4. Same errors encountered
5. Same fixes re-discovered

**Evidence**: Multiple investigation documents created for the same issues:
- `VM_IDLE_INVESTIGATION.md`
- `VM_IDLE_ACTION_ITEMS.md`
- `AZURE_JOB_DIAGNOSIS.md`
- `AZURE_LONG_TERM_SOLUTION.md`

**Impact**: Engineering time wasted rediscovering known issues.

---

## 3. Failure Pattern Taxonomy

### Category A: Windows Installation Failures

| Failure | Symptom | Root Cause | Fix |
|---------|---------|------------|-----|
| ISO not found | Container loops on "waiting for response" | dockurr v0.00 doesn't auto-download | Use waa-auto with dockurr:latest |
| Edition picker | VNC shows "Select operating system" | Missing InstallFrom in unattend.xml | Add MetaData with IMAGE/INDEX |
| Product key dialog | VNC shows "Enter product key" | Using non-Enterprise ISO | Set VERSION=11e |
| AutoLogon fails | VNC shows login screen | Empty password in unattend.xml | Set password to "docker" |

### Category B: Disk Space Failures

| Failure | Symptom | Root Cause | Fix |
|---------|---------|------------|-----|
| Build fails | "no space left on device" | OS disk too small (30GB) | Move Docker to /mnt |
| Install stuck | Windows at 0% forever | Storage disk full | Use /data disk |
| Container won't start | Docker errors on start | Image layers fill disk | docker-prune before build |

### Category C: Network/Connectivity Failures

| Failure | Symptom | Root Cause | Fix |
|---------|---------|------------|-----|
| VNC inaccessible | Connection refused on 8006 | NSG blocks port | Use SSH tunnel |
| WAA unreachable | /probe returns nothing | Port 5000 not forwarded to QEMU | nc port forwarder in container |
| SSH timeout | Command hangs | Long operations, no keepalive | Add SSH keepalive |

### Category D: Azure/Cloud Failures

| Failure | Symptom | Root Cause | Fix |
|---------|---------|------------|-----|
| Nested virt fails | QEMU can't start | TrustedLaunch security | Use Standard security |
| Job stuck | "Running" but 0 tasks | Container never started | Use regular VM, not Azure ML |
| VM costs money idle | $4.80/day waste | No auto-shutdown | Add timeout to container startup |

---

## 4. Concrete Solutions

### Solution 1: Pre-Flight Validation Command

Create `vm preflight` command that validates ALL requirements before any work begins:

```bash
uv run python -m openadapt_ml.benchmarks.cli vm preflight
```

**Checks to perform**:
1. Azure subscription has quota for D4ds_v5
2. SSH key exists and is valid
3. VM size supports nested virtualization (not all do!)
4. No TrustedLaunch security type
5. Disk sizes adequate (OS: 30GB, data: 128GB+)
6. Docker data directory is on large disk
7. Required ports (22, 5000, 8006) not blocked by NSG rules
8. dockurr/windows:latest is accessible
9. OPENAI_API_KEY or ANTHROPIC_API_KEY is set

**Exit codes**:
- 0: All checks pass
- 1: Critical failure (cannot proceed)
- 2: Warning (might work but risky)

### Solution 2: Health Check Pipeline

Add continuous health monitoring at each layer:

```python
class WAAPipeline:
    def run_with_health_checks(self):
        # Layer 1: Azure resources
        if not self.check_azure_resources():
            raise AzureResourceError("Failed to provision Azure resources")

        # Layer 2: VM running
        if not self.check_vm_running(timeout=300):  # 5 min
            raise VMError("VM did not start in time")

        # Layer 3: Docker ready
        if not self.check_docker_ready(timeout=120):  # 2 min
            raise DockerError("Docker not responding")

        # Layer 4: Container started
        if not self.check_container_started(timeout=600):  # 10 min
            raise ContainerError("Container failed to start")

        # Layer 5: Windows booting
        if not self.check_windows_booting(timeout=1800):  # 30 min
            raise WindowsError("Windows installation stuck")

        # Layer 6: WAA server responding
        if not self.check_waa_probe(timeout=900):  # 15 min
            raise WAAError("WAA server never became ready")

        # Layer 7: Ready for benchmark
        return True
```

**Key design points**:
- Each layer has explicit timeout
- Failure at any layer stops the pipeline immediately
- Clear error messages indicate which layer failed
- Logs captured at each transition

### Solution 3: Simplified Architecture Option

Create a "lite" mode that trades some isolation for reliability:

**Current (complex)**:
```
Azure VM -> Docker -> QEMU -> Windows 11 -> WAA Server -> Benchmark
```

**Lite mode (simpler)**:
```
Windows Azure VM -> WAA Server -> Benchmark
```

**Implementation**:
1. Use Azure Windows 11 VM directly (no nested virtualization)
2. Install WAA server on startup via CustomScriptExtension
3. Eliminates Docker, QEMU layers entirely
4. ~3x faster startup, ~2x more reliable

**Trade-offs**:
- Loses some isolation (Windows state persists)
- Slightly higher cost (Windows VM licensing)
- But: dramatically simpler, more reliable

### Solution 4: Smart Retry with Checkpointing

Instead of starting from scratch on failure, checkpoint progress:

```bash
# Checkpoints saved after each major milestone
~/.waa-checkpoints/
  vm_created        # Azure VM provisioned
  docker_installed  # Docker daemon running
  image_built       # waa-auto image ready
  windows_installed # Windows disk image exists
  waa_ready         # WAA server responding
```

**Resume from checkpoint**:
```bash
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --resume
```

This skips completed steps, reducing retry time from 45 min to 5-10 min.

### Solution 5: Disk Space Guardian

Add automatic disk space management:

```bash
# Before any operation that uses disk
def ensure_disk_space(required_gb: int, path: str) -> bool:
    available = shutil.disk_usage(path).free / (1024**3)
    if available < required_gb:
        # Try automatic cleanup
        run_docker_prune()
        available = shutil.disk_usage(path).free / (1024**3)
        if available < required_gb:
            raise DiskSpaceError(
                f"Need {required_gb}GB but only {available:.1f}GB available on {path}. "
                f"Run 'vm docker-prune' or delete old containers."
            )
    return True

# Check before critical operations
ensure_disk_space(20, "/mnt")  # For Windows ISO + disk image
ensure_disk_space(15, "/var/lib/docker")  # For Docker build
```

### Solution 6: Timeout-Based Auto-Recovery

Add intelligent timeouts with recovery actions:

```python
LAYER_TIMEOUTS = {
    "windows_install": {"timeout": 2400, "action": "reset_windows"},  # 40 min
    "waa_server_start": {"timeout": 900, "action": "restart_container"},  # 15 min
    "benchmark_task": {"timeout": 600, "action": "skip_task"},  # 10 min
}

def run_with_timeout(layer: str, func: Callable) -> Any:
    config = LAYER_TIMEOUTS[layer]
    try:
        return timeout(config["timeout"])(func)()
    except TimeoutError:
        recovery_action = config["action"]
        print(f"Timeout at {layer}, attempting recovery: {recovery_action}")
        getattr(self, recovery_action)()
        # Retry once
        return timeout(config["timeout"])(func)()
```

### Solution 7: Unified Status Dashboard

Create real-time status view showing all layers:

```
WAA Benchmark Status
====================
[OK] Azure VM:     waa-eval-vm running (172.171.112.41)
[OK] Docker:       Daemon responding
[OK] Container:    winarena (Up 15 minutes)
[..] Windows:      Installing... 45% (ETA: 8 min)
[--] WAA Server:   Waiting for Windows
[--] Benchmark:    Not started

Recent Events:
  03:45:22  Container started
  03:46:15  Windows ISO download complete (6.6GB)
  03:47:02  Windows installation started
  03:52:45  Windows at 45%

[Refresh] [View Logs] [VNC] [Stop]
```

**Implementation**: Extend existing `vm monitor` to show layer-by-layer status.

### Solution 8: Failure Pattern Detection

Add automatic detection of known failure patterns:

```python
KNOWN_FAILURE_PATTERNS = [
    {
        "pattern": "ISO file not found or is empty",
        "diagnosis": "Using outdated dockurr/windows v0.00",
        "fix": "Rebuild with waa-auto which uses dockurr/windows:latest",
        "command": "vm run-waa --rebuild"
    },
    {
        "pattern": "no space left on device",
        "diagnosis": "Disk full, usually from Docker cache",
        "fix": "Clean Docker cache and retry",
        "command": "vm docker-prune && vm run-waa"
    },
    {
        "pattern": "Waiting for a response.*repeated 10+ times",
        "diagnosis": "Windows installation stuck, usually disk issue",
        "fix": "Check disk space inside container",
        "command": "vm exec --cmd 'df -h /storage'"
    },
]

def analyze_logs(logs: str) -> Optional[FailureAnalysis]:
    for pattern in KNOWN_FAILURE_PATTERNS:
        if re.search(pattern["pattern"], logs):
            return FailureAnalysis(
                pattern=pattern["pattern"],
                diagnosis=pattern["diagnosis"],
                fix=pattern["fix"],
                command=pattern["command"]
            )
    return None
```

---

## 5. Implementation Priority

### P0 (Do immediately - highest impact)

1. **Pre-flight checks** - Prevent starting doomed runs
2. **Disk space guardian** - Most common failure mode
3. **Timeout-based recovery** - Stop waiting for stuck systems

### P1 (Do this week)

4. **Health check pipeline** - Layer-by-layer monitoring
5. **Failure pattern detection** - Auto-diagnose known issues
6. **Checkpointing** - Reduce retry time

### P2 (Do this month)

7. **Simplified architecture option** - For reliability-critical runs
8. **Unified status dashboard** - Better visibility

---

## 6. Success Metrics

Track these metrics to measure improvement:

| Metric | Current (Estimated) | Target |
|--------|---------------------|--------|
| First-attempt success rate | ~15% | 70%+ |
| Time to first task execution | 45-60 min | 20 min |
| Time to detect failure | 30-60 min | 5 min |
| Recovery time after failure | 45 min (full restart) | 10 min (from checkpoint) |
| Wasted compute cost per failure | $1-2 | <$0.25 |

---

## 7. Recommended Immediate Actions

### Today

1. Add disk space check before Docker build (5 min)
2. Add timeout to container startup loop (5 min)
3. Add `--preflight` flag to `vm run-waa` (15 min)

### This Week

4. Implement checkpointing for major milestones
5. Add failure pattern detection with automatic suggestions
6. Update CLAUDE.md with streamlined troubleshooting section

### Before Next Major Evaluation Run

7. Test full pipeline end-to-end with all checks enabled
8. Document the "known good" configuration that worked
9. Create runbook for common failure recovery

---

## 8. Conclusion

The WAA benchmark system's fragility stems from **architectural complexity**, **upstream brittleness**, and **missing validation**. The solution is not to make each layer more robust individually, but to:

1. **Fail fast** - Detect problems at the earliest possible layer
2. **Fail clearly** - Provide actionable error messages
3. **Recover quickly** - Checkpoint progress, don't restart from zero
4. **Validate upfront** - Pre-flight checks prevent doomed runs

With these changes, the system can achieve 70%+ first-attempt success rate and reduce debugging time from hours to minutes.

---

## Appendix A: Complete Failure Timeline

From `WAA_EVAL_ATTEMPTS.md`:

| Date | Attempt | Outcome | Root Cause |
|------|---------|---------|------------|
| Jan 6 | First benchmark | 1/8 tasks (12.5%) | Navi bugs, SSH timeout |
| Jan 8 | VM created | Success | - |
| Jan 17 | Strategic pivot | Decision | - |
| Jan 18 (PM) | Azure ML job | 0/13 tasks | Job stuck, TrustedLaunch |
| Jan 18 (Eve) | Code fixes | Partial | Path issues fixed |
| Jan 19 (AM) | VM idle | Waste | ISO not found, no auto-shutdown |
| Jan 19 | waa-auto Dockerfile | Created | - |
| Jan 20 (AM) | Docker builds | Failed | Disk space |
| Jan 20 (Mid) | Multiple retries | Failed | Storage disk confusion |
| Jan 20 | Storage path fix | Found | /mnt vs /data disk |

**Pattern**: Most failures were resource/configuration issues detectable before Windows installation even started.

## Appendix B: Related Documentation

- `/Users/abrichr/oa/src/openadapt-ml/docs/WAA_EVAL_ATTEMPTS.md` - Full attempt history
- `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md` - CLI commands and patterns
- `/Users/abrichr/oa/src/openadapt-ml/docs/waa_setup.md` - Setup guide
- `/Users/abrichr/oa/src/openadapt-evals/VM_IDLE_INVESTIGATION.md` - Idle VM analysis
- `/Users/abrichr/oa/src/openadapt-evals/AZURE_JOB_DIAGNOSIS.md` - Azure ML issues

## Appendix C: Key Files

- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile` - Custom waa-auto image
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py` - VM management CLI
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/cloud/ssh_tunnel.py` - SSH tunnel management
