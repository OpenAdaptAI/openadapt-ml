# Windows Agent Arena (WAA) Setup Guide

This document describes how to set up and run the Windows Agent Arena benchmark for evaluating GUI automation agents.

## Overview

Windows Agent Arena (WAA) is a benchmark with 154 tasks across 11 Windows application domains. It runs Windows 11 inside a Docker container using QEMU virtualization.

**Repository:** https://github.com/microsoft/WindowsAgentArena

> **Cost & Complexity Warning:** WAA infrastructure is the dominant cost factor, not modeling. VM uptime + human babysitting time dominate total cost. A single benchmark run requires ~30GB storage, ~30 min setup, and $0.19+/hour Azure VM costs. For rapid iteration, consider using our **mock evaluation mode** (`test-mock`) which replays recorded sessions and generates synthetic results without Windows virtualization.

## Architecture

```
Azure VM (Standard_D4ds_v5, nested virtualization required)
  └── Docker
       └── windowsarena/winarena container
            └── QEMU running Windows 11
                 └── WAA Server (Flask on port 5000)
                      ├── /probe - Health check
                      ├── /execute - Run commands
                      └── /screenshot - Capture screen
```

## Time & Cost Estimates

| Phase | Duration | Notes |
|-------|----------|-------|
| Azure VM creation | 5-10 min | One-time |
| Windows ISO download | 5-15 min | ~6GB, depends on bandwidth |
| Windows installation | 20-30 min | First time only, cached after |
| Benchmark execution | 5-15 min/task | Varies by task complexity |
| **Total first run** | **~45-60 min** | Subsequent runs: ~5 min startup |

**Azure costs:** `Standard_D4ds_v5` ≈ $0.19/hour. **Remember to delete the VM when done.**

---

## Quick Start

```bash
# 1. Set up Azure VM with WAA
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa

# 2. Prepare Windows (downloads ISO, installs Windows)
uv run python -m openadapt_ml.benchmarks.cli vm prepare-windows
# ✓ Success: VNC at http://<vm-ip>:8006 shows Windows desktop

# 3. Run benchmark
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5
# ✓ Success: Results saved to ~/waa-results/

# 4. Delete VM when done (stops billing)
uv run python -m openadapt_ml.benchmarks.cli vm delete
```

**Alternative: Mock evaluation (no Windows required):**
```bash
uv run python -m openadapt_ml.benchmarks.cli test-mock --tasks 20
```

---

## Detailed Setup

### Prerequisites

- Azure subscription with nested virtualization support
- VM size: `Standard_D4ds_v5` or larger (D8+ recommended for faster task execution)
- At least 50GB disk space (use `/mnt` temp disk, not OS disk)

### Security Considerations

The WAA setup exposes several ports:

| Port | Service | Risk | Recommendation |
|------|---------|------|----------------|
| 8006 | VNC (noVNC web) | Medium | Restrict via NSG to your IP |
| 5000 | WAA Flask API | High | SSH tunnel or NSG restrict |

**Recommended:** Access via SSH tunnel rather than exposing ports publicly:
```bash
ssh -L 8006:localhost:8006 -L 5000:localhost:5000 azureuser@<vm-ip>
```

### Step 1: Download Windows ISO

The official WAA image requires a Windows ISO. Two options:

**Option A: Enterprise Evaluation ISO (recommended)**
- No product key required
- 90-day evaluation period (sufficient for benchmarks)
- Download from [Microsoft Evaluation Center](https://www.microsoft.com/en-us/evalcenter/download-windows-11-enterprise)

**Option B: Volume-licensed Enterprise ISO + GVLK**
- Requires GVLK key in unattend.xml: `NPPR9-FWDCX-D2C8J-H872K-2YT43`
- This is [Microsoft's published KMS client key](https://learn.microsoft.com/en-us/windows-server/get-started/kms-client-activation-keys) for volume licensing scenarios
- For organizations with volume licensing

**Automated download (Evaluation ISO):**
```bash
mkdir -p ~/waa-iso
curl -L -o ~/waa-iso/setup.iso \
  'https://go.microsoft.com/fwlink/?linkid=2334167&clcid=0x409&culture=en-us&country=us'
```

> **Note:** This URL may change. If it fails, download manually from the Evaluation Center.

### Step 2: Prepare Windows Image

Run the official WAA container with `--prepare-image true`:

```bash
docker run --rm \
  --name waa-prepare \
  --device=/dev/kvm \
  --cap-add NET_ADMIN \
  -p 8006:8006 \
  -v ~/waa-storage:/storage \
  -v ~/waa-iso:/iso \
  windowsarena/winarena:latest \
  "/entry.sh --prepare-image true --start-client false"
```

**Verify success:**
1. VNC at `http://<vm-ip>:8006` shows Windows desktop (not installer)
2. `~/waa-storage/` contains `data.img` (~30GB)

### Step 3: Run Benchmarks

```bash
docker run --rm \
  --name waa-benchmark \
  --device=/dev/kvm \
  --cap-add NET_ADMIN \
  -p 8006:8006 \
  -p 5000:5000 \
  -v ~/waa-storage:/storage \
  -v ~/waa-results:/results \
  -e OPENAI_API_KEY="your-key" \
  windowsarena/winarena:latest \
  "/entry.sh --start-client true --model gpt-4o --agent navi --result-dir /results"
  # Note: --model must be a valid OpenAI model name (e.g., gpt-4o, gpt-4o-mini)
```

**Model options:** The `--model` flag must be a valid OpenAI model name (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`). Invalid model names will cause the benchmark to hang on retries. For local VLMs or proxies, set `OPENAI_API_BASE` accordingly.

**Verify success:**
1. `curl http://localhost:5000/probe` returns `{"status": "Probe successful"}`
2. Results appear in `~/waa-results/`

---

## CLI Commands

```bash
# Full setup (creates Azure VM, installs Docker)
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa

# Prepare Windows (download ISO, install Windows)
uv run python -m openadapt_ml.benchmarks.cli vm prepare-windows

# Run WAA benchmark (uses OPENAI_API_KEY from .env)
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5

# Check VM and WAA status
uv run python -m openadapt_ml.benchmarks.cli vm status

# SSH into VM for debugging
uv run python -m openadapt_ml.benchmarks.cli vm ssh

# Fix storage (move to larger temp disk)
uv run python -m openadapt_ml.benchmarks.cli vm fix-storage

# Reset Windows (fresh install)
uv run python -m openadapt_ml.benchmarks.cli vm reset-windows

# Delete VM when done (IMPORTANT: stops billing)
uv run python -m openadapt_ml.benchmarks.cli vm delete
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

1. **Skips product key dialog** - Either Evaluation ISO (no key needed) or GVLK
2. **Bypasses hardware checks** - TPM, SecureBoot, RAM checks disabled
3. **Configures user account** - Creates "Docker" user with password
4. **Enables AutoLogon** - User logs in automatically after install
5. **Runs FirstLogonCommands** - Executes setup scripts on first login

### FirstLogonCommands

After Windows installs and auto-logs in, these scripts run:

1. `C:\oem\install.bat` - Entry point
2. `C:\oem\setup.ps1` - Main PowerShell setup (installs Python, dependencies)
3. `C:\oem\on-logon.ps1` - Starts WAA Flask server

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

### "ISO file not found"

The Windows ISO must be mounted at `/iso/setup.iso`. Either:
- Mount with `-v ~/waa-iso:/iso`, OR
- Use our CLI which handles this automatically

### Windows stuck at "Product key" dialog

**Cause:** Using wrong ISO type without matching configuration

**Solution:**
- Use Enterprise Evaluation ISO (no key needed), OR
- Use Enterprise ISO + add GVLK to unattend.xml
- Fallback: VNC to port 8006, click "I don't have a product key"

### Container won't start - disk space

**Cause:** Storage on OS disk (~10GB free) instead of temp disk (~147GB)

**Fix:**
```bash
uv run python -m openadapt_ml.benchmarks.cli vm fix-storage
```

### WAA server not responding on /probe

**Cause:** Windows still booting or Flask server failed

**Diagnosis:**
1. Check VNC at `http://<vm-ip>:8006`
2. Wait 15-20 minutes for first boot
3. Look for `waa_setup.log` on Windows desktop

---

## Technical Notes

### Official Image Limitation

The official `windowsarena/winarena:latest` is built on `dockurr/windows v0.00` (November 2024) which does **not** auto-download Windows.

> **Warning:** The dockurr/windows repo updates frequently and may break KVM flags. Consider pinning to a specific digest for production use.

### Network Configuration

- Official WAA uses IP `20.20.20.21` inside the QEMU VM
- Newer dockurr/windows versions use `172.30.0.2`
- The official image's scripts are hardcoded to `20.20.20.21`

### Azure VM Sizing

| Size | vCPUs | RAM | Cost/hr | Notes |
|------|-------|-----|---------|-------|
| D4ds_v5 | 4 | 16GB | ~$0.19 | Minimum for WAA |
| D8ds_v5 | 8 | 32GB | ~$0.38 | Recommended: faster task execution |
| D16ds_v5 | 16 | 64GB | ~$0.77 | For parallel task evaluation |

Larger VMs reduce screenshot→action loop latency and improve overall throughput.

---

## References

- [Windows Agent Arena GitHub](https://github.com/microsoft/WindowsAgentArena)
- [WAA Paper (arXiv)](https://arxiv.org/abs/2409.08264)
- [Microsoft Evaluation Center](https://www.microsoft.com/en-us/evalcenter/download-windows-11-enterprise)
- [Microsoft KMS Keys](https://learn.microsoft.com/en-us/windows-server/get-started/kms-client-activation-keys)
- [dockur/windows](https://github.com/dockur/windows)
