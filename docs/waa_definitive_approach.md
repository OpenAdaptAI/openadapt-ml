# WAA Definitive Approach - Stop the Flip-Flopping

**Date**: January 25, 2026
**Status**: DESIGN DOCUMENT - READ BEFORE ANY MORE CHANGES

---

## The Problem: We Keep Flip-Flopping

We've been switching between approaches without fully committing to or testing any of them:

| Date | Commit | Change | Result |
|------|--------|--------|--------|
| Jan 9 | 1689ab4 | Created custom `waa-auto` Dockerfile | Worked initially |
| Jan 21 | 8fe7e6f | Switched to vanilla WAA build | Claimed to work |
| Jan 21 | 6b9f744 | Fixed unattended installation | Claimed to work |
| Jan 24 | 9a37bb4 | Merged "Vanilla WAA bootstrap" PR | NOT VERIFIED |
| Jan 25 | Today | Back to custom Dockerfile | Failed - disk space |
| Jan 25 | Today | Pull official image | Failed - uses old dockurr |

**Every switch was made without fully verifying the previous approach worked.**

---

## Root Cause Analysis

### Why Custom Dockerfile Fails

Our `waa_deploy/Dockerfile` uses:
```dockerfile
COPY --from=windowsarena/winarena:latest /entry.sh /entry.sh
COPY --from=windowsarena/winarena:latest /client /client
COPY --from=windowsarena/winarena:latest /oem /oem
```

This requires `windowsarena/winarena:latest` to exist. Options:
1. **Pull from Docker Hub**: Image uses OLD dockurr/windows v0.00 (doesn't auto-download Windows)
2. **Build locally**: Needs 50GB+ disk space (winarena-base is 15GB)

**Circular dependency**: We need the image to build the image.

### Why Vanilla Build Fails

The 3-step vanilla build:
```bash
docker build -t windowsarena/windows-local docker/windows-local/
docker build -t windowsarena/winarena-base ...  # 15GB
docker build -t windowsarena/winarena ...
```

Fails because:
- winarena-base downloads 15GB of ML models
- Total disk needed: ~50GB
- D4s_v3 temp disk: 32GB
- D4ds_v4 temp disk: 150GB (but we didn't finish testing it)

### Why Runtime Override Fails

Even if we pull the official image, `VERSION=11e` may not work because:
- The base image (old dockurr/windows) may have VERSION baked in
- Environment variables may be ignored if the base handles Windows download at build time

---

## The Three Actual Options

### Option A: Build Vanilla with Big Disk

**Approach**: Use D4ds_v4 (150GB temp disk), build official 3-step WAA

**Pros**:
- Uses upstream code exactly
- Any fixes from Microsoft come automatically
- Already partially implemented in `vm setup-waa`

**Cons**:
- Slow (30+ min build)
- Higher VM cost ($0.38/hr vs $0.19/hr)
- 15GB model download every time

**Unattended?**: Need to verify vendor submodule uses modern dockurr/windows that auto-downloads

### Option B: Standalone Custom Build (NO COPY --from)

**Approach**: Build from scratch without depending on any pre-existing image

```dockerfile
FROM dockurr/windows:latest

# Copy files from LOCAL vendor submodule (not from Docker image)
COPY vendor/WindowsAgentArena/src/win-arena-container/entry.sh /entry.sh
COPY vendor/WindowsAgentArena/src/win-arena-container/client /client
COPY vendor/WindowsAgentArena/src/win-arena-container/vm/setup /oem

# Set VERSION for auto Windows download
ENV VERSION="11e"
```

**Pros**:
- No circular dependency
- Small image (no 15GB models baked in)
- Fast build
- Completely controlled

**Cons**:
- Need to maintain parity with upstream
- Models loaded at runtime (slower first run)
- May miss upstream fixes

**Unattended?**: Yes, `VERSION=11e` on modern dockurr/windows auto-downloads Windows 11 Enterprise Eval

### Option C: Pre-built Golden Image

**Approach**: Build once, snapshot, reuse

1. Build winarena image once (any method)
2. Run container, let Windows install complete
3. Snapshot the Windows disk (`data.img`)
4. Store in Azure Blob or ship with repo
5. Future runs use pre-built snapshot

**Pros**:
- Fast startup (Windows already installed)
- No download wait
- Consistent environment

**Cons**:
- 30GB snapshot to store/transfer
- Maintenance burden when Windows updates needed
- May violate Windows licensing for redistribution

**Unattended?**: Yes (after initial setup)

---

## Recommendation: Option B (Standalone Custom Build)

**Why**: It's the only approach that:
1. Doesn't require pulling 15GB image
2. Doesn't require 50GB disk space
3. Uses modern dockurr/windows for auto-download
4. Is fully unattended

### Implementation Plan

1. **Create new Dockerfile** that copies from LOCAL vendor submodule
2. **Test on D4s_v3** (32GB is enough without winarena-base)
3. **Verify unattended**: Windows downloads, installs, WAA server starts
4. **Add to CLI**: New command or modify existing
5. **Document**: Update CLAUDE.md with ONE way to do things
6. **Remove alternatives**: Delete Dockerfile.simplified, Dockerfile.backup, etc.

### Required Files from Vendor Submodule

```
vendor/WindowsAgentArena/
├── src/win-arena-container/
│   ├── entry.sh              → /entry.sh
│   ├── entry_setup.sh        → /entry_setup.sh
│   ├── start_client.sh       → /start_client.sh
│   ├── client/               → /client/
│   └── vm/
│       └── setup/            → /oem/
│           ├── install.bat
│           ├── setup.ps1
│           └── on-logon.ps1
```

### Key Configuration

```dockerfile
# Modern base that auto-downloads Windows
FROM dockurr/windows:latest

# Environment for unattended setup
ENV VERSION="11e"           # Enterprise Eval (no license key)
ENV RAM_SIZE="8G"
ENV DISK_SIZE="30G"
ENV XRES="1440"
ENV YRES="900"
```

---

## What NOT To Do

1. **Don't switch approaches again** without fully testing current one
2. **Don't use COPY --from** windowsarena/winarena (circular dependency)
3. **Don't assume "it works"** without end-to-end verification:
   - Windows boots automatically (no manual ISO)
   - No license key prompt
   - WAA server starts (port 5000 responds)
   - Benchmark task completes
4. **Don't run one-off SSH commands** - use CLI
5. **Don't have multiple Dockerfiles** - ONE definitive approach

---

## Verification Checklist

Before declaring "it works", verify ALL of these:

- [ ] VM created successfully
- [ ] Docker installed with /mnt storage
- [ ] Docker image built successfully
- [ ] Container started without manual intervention
- [ ] Windows ISO downloaded automatically (check VNC)
- [ ] Windows installed without product key prompt
- [ ] Windows booted to desktop
- [ ] WAA server started (curl localhost:5000/probe returns 200)
- [ ] SSH tunnel to 8006 shows Windows desktop
- [ ] SSH tunnel to 5000 forwards to WAA server
- [ ] At least ONE benchmark task completes successfully

---

## CLI Commands (Final State)

```bash
# Create VM (use D4ds_v4 if need big disk, D4s_v3 if using Option B)
uv run python -m openadapt_ml.benchmarks.cli vm create --size Standard_D4s_v3

# Setup everything (Docker, image build, container start)
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa

# Monitor (dashboard with VNC)
uv run python -m openadapt_ml.benchmarks.cli vm dashboard

# Run benchmark
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5

# Clean up
uv run python -m openadapt_ml.benchmarks.cli vm delete -y
```

---

## Action Items

1. [ ] Review this document with user
2. [ ] Agree on Option B (standalone custom build)
3. [ ] Create new Dockerfile.standalone (copies from vendor, no COPY --from)
4. [ ] Update cli.py to use new Dockerfile
5. [ ] Delete old Dockerfiles (simplified, backup, complex)
6. [ ] Test end-to-end with verification checklist
7. [ ] Update CLAUDE.md with final approach
8. [ ] Commit and document in BEADS.md

---

## Files to Clean Up After Decision

```
DELETE:
- openadapt_ml/benchmarks/waa_deploy/Dockerfile.simplified
- openadapt_ml/benchmarks/waa_deploy/Dockerfile.backup
- openadapt_ml/benchmarks/waa_deploy/Dockerfile.complex

KEEP (after rewrite):
- openadapt_ml/benchmarks/waa_deploy/Dockerfile  (the ONE true Dockerfile)
- openadapt_ml/benchmarks/waa_deploy/start_waa_server.bat
- openadapt_ml/benchmarks/waa_deploy/api_agent.py
```

---

*This document must be reviewed and approved before any more WAA changes are made.*
