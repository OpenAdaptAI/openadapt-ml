# Auto-Update Documentation System

This document describes the automatic documentation update system for OpenAdapt.

## Overview

When code is committed to OpenAdapt sub-packages (openadapt-ml, openadapt-evals, etc.), the main documentation site at https://docs.openadapt.ai/ is automatically updated.

**Current Status**: docs.openadapt.ai is hosted on **GitHub Pages** using **MkDocs Material**, deployed from the main [OpenAdaptAI/OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt) repository.

## Architecture

```
┌─────────────────────┐     repository_dispatch     ┌─────────────────────┐
│  openadapt-ml       │ ─────────────────────────▶  │  OpenAdapt (main)   │
│  push to main       │                             │                     │
└─────────────────────┘                             │  sync-package-docs  │
                                                    │  workflow runs      │
┌─────────────────────┐     repository_dispatch     │                     │
│  openadapt-evals    │ ─────────────────────────▶  │  Updates docs/      │
│  push to main       │                             │  packages/*.md      │
└─────────────────────┘                             │                     │
                                                    │  Triggers docs.yml  │
     ... other repos ...                            │  (deploys to GH     │
                                                    │   Pages)            │
                                                    └─────────────────────┘
```

## Components

### 1. Trigger Workflow (in sub-packages)

Located at `.github/workflows/trigger-docs-update.yml` in each sub-package.

When README.md or docs/ changes are pushed to main, this workflow sends a `repository_dispatch` event to the main OpenAdapt repo.

**Required secret**: `DOCS_UPDATE_TOKEN` - A GitHub PAT with `contents:write` permission on OpenAdaptAI/OpenAdapt.

### 2. Sync Script (in main OpenAdapt repo)

Located at `docs/_scripts/sync_package_docs.py` in the main repo.

This Python script:
- Fetches README.md from each sub-package via GitHub API
- Transforms content to MkDocs-compatible format
- Updates `docs/packages/*.md` files
- Handles badges, relative links, and formatting

### 3. Sync Workflow (in main OpenAdapt repo)

Located at `.github/workflows/sync-package-docs.yml` in the main repo.

Triggers on:
- `repository_dispatch` events from sub-packages
- Manual `workflow_dispatch`
- Daily schedule at 6 AM UTC

Creates a PR with documentation changes and optionally auto-merges.

### 4. Deploy Workflow (existing)

The existing `docs.yml` workflow in the main repo deploys to GitHub Pages when docs are updated.

## Setup Instructions

### Step 1: Set Up the Main OpenAdapt Repo

1. **Copy sync script**:
   ```bash
   # In the main OpenAdapt repo
   cp path/to/openadapt-ml/scripts/sync_package_docs.py docs/_scripts/
   ```

2. **Copy sync workflow**:
   ```bash
   cp path/to/openadapt-ml/docs/workflows/sync-package-docs.yml .github/workflows/
   ```

3. **Configure repository settings**:
   - Go to Settings > Actions > General
   - Enable "Allow GitHub Actions to create and approve pull requests"

### Step 2: Set Up Each Sub-Package

1. **Create a Personal Access Token (PAT)**:
   - Go to GitHub Settings > Developer settings > Personal access tokens > Fine-grained tokens
   - Create token with:
     - Repository access: `OpenAdaptAI/OpenAdapt`
     - Permissions: `contents: write`
   - Save the token securely

2. **Add as organization secret** (recommended) or repo secret:
   - Organization: Settings > Secrets and variables > Actions > New organization secret
   - Name: `DOCS_UPDATE_TOKEN`
   - Value: The PAT you created
   - Repository access: Select repos that need it

3. **Add the trigger workflow** to each sub-package:
   - Copy `.github/workflows/trigger-docs-update.yml`
   - Update `client-payload.package` to match the package name

### Packages Currently Configured

| Package | Trigger Workflow | Status |
|---------|-----------------|--------|
| openadapt-ml | `.github/workflows/trigger-docs-update.yml` | Ready |
| openadapt-evals | Needs to be added | Pending |
| openadapt-capture | Needs to be added | Pending |
| openadapt-viewer | Needs to be added | Pending |
| openadapt-grounding | Needs to be added | Pending |
| openadapt-retrieval | Needs to be added | Pending |
| openadapt-privacy | Needs to be added | Pending |

## Manual Operations

### Trigger a sync manually

```bash
# From any sub-package repo (requires workflow file)
gh workflow run trigger-docs-update.yml

# From the main OpenAdapt repo
gh workflow run sync-package-docs.yml

# Sync a specific package
gh workflow run sync-package-docs.yml -f package=openadapt-ml
```

### Run sync script locally

```bash
# In the main OpenAdapt repo
cd /path/to/OpenAdapt

# Dry run (preview changes)
python docs/_scripts/sync_package_docs.py --dry-run

# Sync all packages
python docs/_scripts/sync_package_docs.py

# Sync specific package
python docs/_scripts/sync_package_docs.py --package openadapt-ml
```

## Troubleshooting

### Workflow not triggering

1. Check that paths filter matches your changes (README.md, docs/**, CHANGELOG.md)
2. Verify `DOCS_UPDATE_TOKEN` secret is set and not expired
3. Check Actions tab for workflow run logs
4. Ensure the workflow file is on the default branch

### Sync script failing

1. Check GitHub API rate limits (authenticated requests get 5000/hour)
2. Verify the package README exists and is accessible
3. Review error messages in workflow logs
4. Test locally with `--dry-run`

### Docs not deploying

1. Check that `docs.yml` workflow is enabled
2. Verify GitHub Pages is configured (Settings > Pages)
3. Check MkDocs build output for errors
4. Ensure `mkdocs.yml` nav includes the package pages

### Permission errors

1. Verify PAT has correct permissions (`contents:write` on main repo)
2. Check PAT hasn't expired
3. Ensure secret is accessible to the workflow

## How It Works (Technical Details)

### Transform Process

The sync script applies these transformations to README content:

1. **Title normalization**: Replaces `# PackageName` with standardized `# package-name`
2. **Badge removal**: Strips shields.io badges that don't render well in MkDocs
3. **Link fixing**: Converts relative links (`[link](docs/file.md)`) to absolute GitHub URLs
4. **Metadata addition**: Adds repository link at the top
5. **Footer addition**: Adds auto-generation notice with source link

### Package Configuration

Packages are configured in `sync_package_docs.py`:

```python
PACKAGES = {
    "openadapt-ml": {
        "repo": "OpenAdaptAI/openadapt-ml",
        "doc_file": "docs/packages/ml.md",
        "title": "openadapt-ml",
        "description": "Policy learning, training, and inference for GUI automation agents.",
    },
    # ... more packages
}
```

To add a new package, add an entry to this dict and add the trigger workflow to the package repo.

## Best Practices

### README Format

For best results, structure package READMEs as:

```markdown
# Package Name

Brief description (1-2 sentences).

## Installation

\`\`\`bash
pip install package-name
\`\`\`

## Quick Start

...

## API Reference

...
```

### What to Include

- Installation instructions
- Basic usage examples
- API documentation or links
- Configuration options

### What NOT to Include

- CI badges (handled automatically)
- Development setup (belongs in CONTRIBUTING.md)
- Internal implementation details

## Future Improvements

- [ ] Add support for CHANGELOG.md syncing
- [ ] Add API reference auto-generation from docstrings
- [ ] Add versioned documentation support
- [ ] Add link validation in CI
- [ ] Add preview deployments for PRs
