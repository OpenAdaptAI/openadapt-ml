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
If `--waa-path` is omitted, the script will auto-clone WAA into `vendor/WindowsAgentArena`.

```bash
./scripts/waa_bootstrap_local.sh \
  --iso-path /path/to/Windows11_Enterprise_Eval.iso
```

If you have a direct ISO URL:

```bash
./scripts/waa_bootstrap_local.sh \
  --iso-url "https://example.com/Windows11_Enterprise_Eval.iso"
```

If Docker requires root:

```bash
./scripts/waa_bootstrap_local.sh --iso-path /path/to/Windows11.iso --sudo
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
