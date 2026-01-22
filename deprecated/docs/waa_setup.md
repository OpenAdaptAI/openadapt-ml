# Windows Agent Arena (WAA) Setup Guide

This document describes how to set up and run the Windows Agent Arena benchmark for evaluating GUI automation agents.

## Status
Legacy. Use the vanilla flow in `docs/waa_vanilla_automation.md` instead.

## Overview

Windows Agent Arena (WAA) is a benchmark with 154 tasks across 11 Windows application domains. It runs Windows 11 inside a Docker container using QEMU virtualization.

**Repository:** https://github.com/microsoft/WindowsAgentArena

## FULLY AUTOMATED Setup

**CRITICAL**: NO manual ISO downloads. Everything is automated using `dockurr/windows`.

Our `waa-auto` Docker image:
1. Uses `dockurr/windows:latest` which **automatically downloads Windows 11** based on `VERSION` env var
2. Combines WAA client/server from `windowsarena/winarena:latest`
3. Handles all automation via unattend.xml

## Architecture

```
Azure VM (Standard_D4ds_v5, nested virtualization required)
  └── Docker (data on /mnt)
       └── waa-auto:latest (based on dockurr/windows)
            └── QEMU running Windows 11 (IP: 172.30.0.2)
                 └── WAA Server (Flask on port 5000)
                      ├── /probe - Health check
                      ├── /execute - Run commands
                      └── /screenshot - Capture screen
```

## Time & Cost Estimates

| Phase | Duration | Notes |
|-------|----------|-------|
| Azure VM creation | 5-10 min | One-time |
| Docker image build | 5-10 min | One-time, cached |
| Windows ISO download | 5-10 min | ~6.6GB, **automatic** via dockurr |
| Windows installation | 10-15 min | First time only, cached after |
| Benchmark execution | 5-15 min/task | Varies by task complexity |
| **Total first run** | **~30-45 min** | Subsequent runs: ~3 min startup |

**Azure costs:** `Standard_D4ds_v5` ≈ $0.19/hour. **Remember to delete the VM when done.**

---

## Quick Start (FULLY AUTOMATED)

```bash
# 1. Setup Azure VM with Docker and build waa-auto image (~10 min)
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key $OPENAI_API_KEY

# 2. Run benchmark (Windows auto-downloads on first run)
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5

# 3. Monitor progress (optional, for debugging)
uv run python -m openadapt_ml.benchmarks.cli vm monitor
# Opens browser to VNC at http://localhost:8006

# 4. Delete VM when done (IMPORTANT: stops billing!)
uv run python -m openadapt_ml.benchmarks.cli vm delete -y
```

**Alternative: Mock evaluation (no Windows required):**
```bash
uv run python -m openadapt_ml.benchmarks.cli test-mock --tasks 20
```

---

## How Auto-Download Works

The [dockurr/windows](https://github.com/dockur/windows) project handles Windows installation automatically:

1. **Set VERSION environment variable:**
   - `VERSION=11e` - Windows 11 Enterprise (6.6 GB, recommended)
   - `VERSION=11` - Windows 11 Pro (7.2 GB)
   - `VERSION=10e` - Windows 10 Enterprise (5.2 GB)

2. **First run behavior:**
   - dockurr/windows downloads Windows ISO from Microsoft
   - QEMU installs Windows using unattend.xml (unattended)
   - Disk image saved to `/storage/data.qcow2`

3. **Subsequent runs:**
   - Boots from existing disk image (~2-3 min)
   - No re-download needed

**Why Windows 11 Enterprise (`11e`)?**
- Accepts GVLK keys (no "product key" dialog during setup)
- 90-day evaluation period (sufficient for benchmarks)
- Most compatible with WAA test applications

---

## Manual Docker Commands (Advanced)

If you prefer direct Docker commands instead of CLI:

### Start Windows VM (First Run)

```bash
# This downloads Windows 11 and installs it (~20 min first run)
docker run -d \
  --name winarena \
  --device=/dev/kvm \
  --cap-add NET_ADMIN \
  -p 8006:8006 \
  -p 5000:5000 \
  -v /mnt/waa-storage:/storage \
  -e VERSION=11e \
  -e RAM_SIZE=12G \
  -e CPU_CORES=4 \
  -e DISK_SIZE=64G \
  waa-auto:latest \
  "/waa-entry.sh --start-client false"
```

### Run Benchmarks

```bash
docker run -d \
  --name winarena \
  --device=/dev/kvm \
  --cap-add NET_ADMIN \
  -p 8006:8006 \
  -p 5000:5000 \
  -v /mnt/waa-storage:/storage \
  -e OPENAI_API_KEY="your-key" \
  waa-auto:latest \
  "/waa-entry.sh --start-client true --model gpt-4o --num-tasks 5"
```

---

## CLI Commands

```bash
# Full setup (creates Azure VM, installs Docker, builds waa-auto)
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key $OPENAI_API_KEY

# Run WAA benchmark
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5

# Start monitoring dashboard (VNC, logs, status)
uv run python -m openadapt_ml.benchmarks.cli vm monitor

# Check VM and WAA status
uv run python -m openadapt_ml.benchmarks.cli vm status

# Check if WAA server is ready
uv run python -m openadapt_ml.benchmarks.cli vm probe --wait

# View container logs
uv run python -m openadapt_ml.benchmarks.cli vm logs --lines 100

# SSH into VM for debugging
uv run python -m openadapt_ml.benchmarks.cli vm ssh

# Check disk space, Docker status
uv run python -m openadapt_ml.benchmarks.cli vm diag

# Clean Docker images/containers (free disk space)
uv run python -m openadapt_ml.benchmarks.cli vm docker-prune

# Reset Windows (delete disk image, forces fresh install)
uv run python -m openadapt_ml.benchmarks.cli vm reset-windows

# Delete VM when done (IMPORTANT: stops billing)
uv run python -m openadapt_ml.benchmarks.cli vm delete -y
```

---

## Understanding Results

Benchmark results are saved to `~/waa-results/` with this structure:

```
waa-results/
├── task_001/
│   ├── screenshots/       # Step-by-step screenshots
│   ├── actions.json       # Actions taken by agent
│   └── result.json        # Success/failure, reasoning
├── task_002/
│   └── ...
└── summary.json           # Aggregate metrics
```

**Key metrics in summary.json:**
- `success_rate`: Percentage of tasks completed correctly
- `avg_steps`: Average actions per task
- `avg_time`: Average time per task

---

## How Windows Automation Works

### unattend.xml

Windows installs automatically using an unattend.xml answer file that:

1. **Skips product key dialog** - Enterprise Evaluation ISO with `VERSION=11e`
2. **Bypasses hardware checks** - TPM, SecureBoot, RAM checks disabled
3. **Configures user account** - Creates "Docker" user with password
4. **Enables AutoLogon** - User logs in automatically after install
5. **Runs FirstLogonCommands** - Executes setup scripts on first login

### FirstLogonCommands (in waa-auto)

Our waa-auto Dockerfile injects additional FirstLogonCommands:

1. Disable Windows Firewall
2. Disable sleep and monitor timeout
3. Disable lock screen
4. Run `\\host.lan\Data\install.bat` (installs Python, Chrome, etc.)
5. Create scheduled task for WAA server auto-start
6. Start WAA server immediately

### WAA Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/probe` | GET | Health check - returns 200 when ready |
| `/execute` | POST | Execute commands (pyautogui, etc.) |
| `/screenshot` | GET | Capture current screen |

---

## Troubleshooting

### Known Flaky Behaviors

| Issue | Symptom | Mitigation |
|-------|---------|------------|
| AutoLogon race | Windows boots but stays at login screen | Wait 2-3 min, or VNC and click user |
| Screenshot black frames | `/screenshot` returns black image | Wait 30s after boot for display init |
| QEMU clock skew | Tasks timeout unexpectedly | Restart container |
| FirstLogonCommands failure | Server never starts | Check `C:\Users\Docker\Desktop\*.log` via VNC |

### WAA server not responding on /probe

**Cause:** Windows still booting or Flask server failed

**Diagnosis:**
1. Check VNC at `http://localhost:8006` (via SSH tunnel)
2. Wait 15-20 minutes for first boot
3. Run `uv run python -m openadapt_ml.benchmarks.cli vm logs` to see container output

### Container won't start - disk space

**Cause:** Storage on OS disk (~10GB free) instead of temp disk (~147GB)

**Fix:**
```bash
uv run python -m openadapt_ml.benchmarks.cli vm docker-prune
# Or move Docker data to /mnt
uv run python -m openadapt_ml.benchmarks.cli vm docker-move
```

### Windows stuck at "Product key" dialog

This should NOT happen with `VERSION=11e` (Enterprise Evaluation).

If it does:
1. Connect via VNC: `http://localhost:8006`
2. Click "I don't have a product key"
3. Select "Windows 11 Enterprise" edition

**Better fix:** Delete disk image and let it reinstall:
```bash
uv run python -m openadapt_ml.benchmarks.cli vm reset-windows
```

---

## Technical Notes

### Why waa-auto Instead of Official Image?

The official `windowsarena/winarena:latest` is built on `dockurr/windows v0.00` (November 2024) which does **NOT** auto-download Windows. It expects a manual ISO.

Our `waa-auto` image uses `dockurr/windows:latest` which auto-downloads Windows based on `VERSION` env var.

### Network Configuration

- **waa-auto (dockurr/windows):** Windows VM at `172.30.0.2`
- **Official WAA:** Windows VM at `20.20.20.21`
- Our Dockerfile patches IP addresses in all entry scripts

### Azure VM Sizing

| Size | vCPUs | RAM | Cost/hr | Notes |
|------|-------|-----|---------|-------|
| D4ds_v5 | 4 | 16GB | ~$0.19 | Minimum for WAA |
| D8ds_v5 | 8 | 32GB | ~$0.38 | Recommended: faster task execution |
| D16ds_v5 | 16 | 64GB | ~$0.77 | For parallel task evaluation |

Larger VMs reduce screenshot→action loop latency and improve overall throughput.

---

## Security Considerations

The WAA setup exposes several ports:

| Port | Service | Risk | Recommendation |
|------|---------|------|----------------|
| 8006 | VNC (noVNC web) | Medium | Restrict via NSG to your IP |
| 5000 | WAA Flask API | High | SSH tunnel or NSG restrict |

**Recommended:** Access via SSH tunnel rather than exposing ports publicly:
```bash
ssh -L 8006:localhost:8006 -L 5000:localhost:5000 azureuser@<vm-ip>
```

The CLI's `vm monitor` command automatically sets up SSH tunnels.

---

## References

- [Windows Agent Arena GitHub](https://github.com/microsoft/WindowsAgentArena)
- [WAA Paper (arXiv)](https://arxiv.org/abs/2409.08264)
- [dockur/windows](https://github.com/dockur/windows) - Auto-downloads Windows
- [Microsoft KMS Keys](https://learn.microsoft.com/en-us/windows-server/get-started/kms-client-activation-keys)
