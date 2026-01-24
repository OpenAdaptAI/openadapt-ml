# WAA ACR Design (Unattended + Vanilla)

## Goals
- Make WAA image pulls reliable (no Docker Hub throttling/timeouts).
- Preserve unattended Windows install (no license prompts).
- Use existing CLI/scripts wherever possible.

## Constraints
- Windows install must be fully unattended (VERSION=11e + OEM Azure mode).
- Prefer vanilla WAA components; no dev-mode UNC paths.
- Avoid custom tooling if existing commands cover the flow.

## Proposed ACR Naming
ACR names must be globally unique and <= 50 chars. Use a deterministic pattern tied to the subscription and region:

```
openadapt-evals-<region>-<suffix>
```

Suggested suffix: last 6 chars of the Azure subscription ID.

Example (eastus + sub id ...1234ab):

```
openadapt-evals-eastus-1234ab
```

If name is taken, append `-01`, `-02`, etc.

## Implementation Plan

### 1) Create ACR + import WinArena image
Use the existing helper script in `openadapt-evals`:

```bash
cd /Users/abrichr/oa/src/openadapt-evals
./scripts/setup_acr.sh \
  --acr-name openadapt-evals-eastus-1234ab \
  --resource-group openadapt-agents \
  --workspace openadapt-ml \
  --location eastus
```

This script:
- Creates the registry (Basic tier).
- Imports `docker.io/windowsarena/winarena:latest`.
- Grants `AcrPull` to the Azure ML workspace identity.

### 2) Use ACR image for Azure ML runs
No new code needed; use the existing config/env support:

```bash
export AZURE_DOCKER_IMAGE="openadapt-evals-eastus-1234ab.azurecr.io/winarena:latest"
```

Then run Azure evals as usual:

```bash
uv run python -m openadapt_evals.benchmarks.cli azure --workers 1 --task-ids notepad_1 --waa-path /path/to/WAA
```

### 3) Use ACR image for dedicated VM builds
The VM flow already supports ACR via existing CLI commands:

```bash
uv run python -m openadapt_ml.benchmarks.cli vm pull-image --acr openadapt-evals-eastus-1234ab
```

When building the custom `waa-auto` image on the VM, set:

```bash
export WAA_SOURCE_IMAGE="openadapt-evals-eastus-1234ab.azurecr.io/winarena:latest"
uv run python -m openadapt_ml.benchmarks.cli vm prepare-windows
```

This uses the simplified Dockerfile (OEM Azure mode) and keeps installs unattended.

## Verification Checklist
- ACR import succeeded (`az acr repository show --name <acr> --repository winarena`).
- Azure ML run logs show pulls from the ACR login server.
- VM `prepare-windows` completes without product key prompts.
- WAA `/probe` endpoint responds on port 5000 after boot.

## Notes
- The simplified Dockerfile copies OEM assets from the source image and uses `VERSION=11e` for unattended installs.
- If Windows prompts for a product key, treat it as a regression and follow `docs/RECURRING_ISSUES.md`.
- Keep Azure ML and ACR in the same region to avoid throttling and reduce pull time.
