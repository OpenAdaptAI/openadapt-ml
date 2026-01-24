# Vanilla WAA Automation (No Repo Modifications)

## Goal
Run Windows Agent Arena exactly as published, with automation handled outside the WAA repo.

## Approach
1. Place `setup.iso` at the expected location.
2. Run the official `run-local.sh --prepare-image true`.
3. Use the golden image for all subsequent runs.

This keeps the WAA repo pristine and avoids custom Dockerfiles or internal patches.

## One-Time Local Bootstrap
Use the wrapper script in this repo to download/copy the ISO and run the official prep command.
If `--waa-path` is omitted, the script will auto-detect WAA in standard locations
(`vendor/WindowsAgentArena` relative to the repo root) or clone it if not found.

```bash
./scripts/waa_bootstrap_local.sh \
  --iso-path /path/to/Windows11_Enterprise_Eval.iso
```

If you have a direct ISO URL (pre-authorized download):

```bash
./scripts/waa_bootstrap_local.sh \
  --iso-url "https://example.com/Windows11_Enterprise_Eval.iso"
```

If Docker requires root:

```bash
./scripts/waa_bootstrap_local.sh --iso-path /path/to/Windows11.iso --sudo
```

If you need a guided manual download step, open the Microsoft Eval Center page:

```bash
./scripts/waa_bootstrap_local.sh --open-iso-page
```

## Helper Check
Use the helper to verify the repo path, `setup.iso`, and `config.json`:

```bash
./scripts/waa_bootstrap_helper.sh --clone
```

## Subsequent Local Runs
Once the golden image is created, you can use vanilla WAA commands:

```bash
cd /path/to/WindowsAgentArena/scripts
./run-local.sh
```

## Fully Unattended Note
Microsoft's evaluation center download often requires a manual acceptance step.
For fully unattended runs, host the ISO internally and pass a direct URL with
`--iso-url`, or use a prebuilt golden image stored in Azure blob.

## Azure (Future)
- Upload `src/win-arena-container/vm/storage` to Azure blob as described in the official WAA README.
- Run `run_azure.py` with `datastore_input_path` pointing at the uploaded storage.
- TODO: automate blob upload and use a pre-hosted ISO.

## Deprecations
The following custom paths are considered legacy under this design:
- Custom `waa-auto` Dockerfile flows.
- Dev-mode UNC/samba bootstraps.
- Any non-WAA wrappers that reimplement `run-local.sh` or `run_azure.py`.

Legacy materials have been moved to `deprecated/` for review.
