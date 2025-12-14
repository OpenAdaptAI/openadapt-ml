# Dashboard Architecture - Design Decision

## Problem Statement

The training dashboard and viewer have inconsistent navigation because:
1. Multiple HTML files each manage their own nav
2. `discoverDashboards()` dynamically finds files and builds nav
3. Old files get discovered and linked
4. Regeneration doesn't consistently update all files

## Decision: Two Pages with Hardcoded Nav

### Architecture

We have exactly **2 pages**:

| Page | Purpose | URL |
|------|---------|-----|
| `dashboard.html` | Training progress (loss, ETA, config) | `/dashboard.html` |
| `viewer.html` | Unified capture/comparison viewer | `/viewer.html` |

### Navigation

Nav is **hardcoded** in both files:
```html
<nav class="nav-bar">
    <span class="nav-label">Dashboards:</span>
    <a href="dashboard.html" class="nav-link">Training</a>
    <a href="viewer.html" class="nav-link">Viewer</a>
</nav>
```

### Key Changes

1. **Remove `discoverDashboards()`** - No dynamic nav discovery
2. **Hardcode nav in generators** - Both `generate_training_dashboard()` and `_enhance_comparison_to_unified_viewer()` include the same nav
3. **Delete old comparison files** - No `comparison_epoch*.html`, `comparison_preview.html`, etc.
4. **Checkpoint selection via dropdown** - `viewer.html` has a checkpoint dropdown to switch between predictions

### Viewing Options

| Method | Command | When to Use |
|--------|---------|-------------|
| **Server (recommended)** | `uv run python -m openadapt_ml.cloud.lambda_labs serve` | Development, live updates |
| **File** | `open training_output/dashboard.html` | Quick local viewing |
| **Export** | (future) `export-snapshot` | Sharing with others |

### Why Not Dynamic Nav?

| Approach | Problem |
|----------|---------|
| `discoverDashboards()` JS | Finds old files, creates inconsistent nav |
| Server-side includes | Doesn't work with file:// protocol |
| Shared nav.html | Requires build step or server |

With only 2 pages, hardcoded nav is simpler and more reliable.

### File Structure

```
training_output/
├── dashboard.html      # Training dashboard (nav: Training | Viewer)
├── viewer.html         # Unified viewer with checkpoint dropdown
├── training_log.json   # Training state
└── archive/            # Old comparison files (kept for reference)
    ├── comparison_epoch3.html
    ├── comparison_preview.html
    └── ...
```

### Implementation Checklist

- [x] Consolidate comparison tabs into single `viewer.html`
- [x] Add checkpoint dropdown to viewer
- [x] Archive old comparison_*.html files
- [x] Remove `discoverDashboards()` from dashboard generator
- [x] Hardcode nav in dashboard generator
- [x] Test both pages have consistent nav
