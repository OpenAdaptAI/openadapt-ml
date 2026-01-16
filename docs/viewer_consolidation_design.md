# Viewer Consolidation Design

## Executive Summary

This document outlines the consolidation of `build_monitor.html` into the unified viewer system. The goal is to eliminate standalone viewers and provide a cohesive UX across all monitoring surfaces while following the 80/20 principle: maximum impact with minimal complexity.

## Current State

### Existing Viewers (3 Tabs)

| Tab | File | Purpose | Generator |
|-----|------|---------|-----------|
| **Training** | `dashboard.html` | Loss curves, metrics, stop button | `trainer.py:generate_training_dashboard()` |
| **Viewer** | `viewer.html` | Step playback, predictions, comparison | `viewer.py:generate_unified_viewer_from_output_dir()` |
| **Benchmarks** | `benchmark.html` | Task results, domain stats, execution replay | `benchmark_viewer.py:generate_benchmark_viewer()` |

### Standalone Build Monitor

| File | Purpose | Data Source |
|------|---------|-------------|
| `build_monitor.html` | Docker build progress, logs | `/api/build-log` endpoint |

**Problem**: The build monitor is disconnected from the main navigation. Users must manually navigate to it, and it doesn't share styling/behavior patterns with other tabs.

## Proposed Solution

### Option A: Fourth Tab (RECOMMENDED)

Add "Infrastructure" tab that consolidates:
1. Docker build progress and logs
2. VM status (Azure VMs)
3. SSH tunnel status
4. Resource monitoring

```
[Training]  [Viewer]  [Benchmarks]  [Infrastructure]
```

**Why "Infrastructure" not "Build"?**
- More extensible for future additions (VM monitoring, GPU stats, storage)
- Better semantic grouping of "operational" vs "data" views
- Avoids confusion with benchmark "runs"

### Option B: Sub-tabs Under Benchmarks

```
[Training]  [Viewer]  [Benchmarks ▼]
                      ├── Results
                      ├── Infrastructure
                      └── Run History
```

**Rejected because**: Adds complexity, breaks existing URL patterns, not discoverable.

### Option C: Collapsible Status Panel

Add infrastructure status as collapsible panel at top of all tabs.

**Rejected because**: Clutters UI, duplicates information, inconsistent with tab pattern.

## Implementation Plan (80/20)

### Phase 1: Quick Win (Day 1)
**Impact: High | Complexity: Low**

1. **Add 4th tab to shared header** (`shared_ui.py`)
   - Add "Infrastructure" to `generate_shared_header_html()`
   - Support `active_page="infrastructure"`

2. **Create infrastructure.html generator**
   - Port existing `build_monitor.html` styles/JS
   - Use shared header components
   - Keep existing `/api/build-log` endpoint

**Files to modify:**
- `openadapt_ml/training/shared_ui.py` (~20 lines)
- `openadapt_ml/training/benchmark_viewer.py` (~300 lines new function)

### Phase 2: Enhanced Functionality (Day 2-3)
**Impact: Medium | Complexity: Medium**

1. **Add VM status panel**
   - Show Azure VM state, IP, tunnels
   - Use existing `/api/vms` endpoint

2. **Add build history**
   - List previous builds with timestamps
   - Click to view logs for each build
   - Store build logs in `training_output/builds/`

3. **Add resource indicators**
   - Disk usage (important for Docker builds)
   - Docker image list and sizes

### Phase 3: Polish (Optional)
**Impact: Low | Complexity: Medium**

1. **Unified status indicators**
   - Consistent pulse/solid animations
   - Color coding across all views

2. **Cross-tab notifications**
   - Build complete notification on Training tab
   - VM ready notification on Benchmarks tab

3. **Dark/light mode toggle**
   - Persist preference in localStorage

## Detailed Design

### Shared UI Updates (`shared_ui.py`)

```python
def generate_shared_header_html(
    active_page: str,  # "training", "viewer", "benchmarks", "infrastructure"
    controls_html: str = "",
    meta_html: str = "",
) -> str:
    """Generate unified header with 4 navigation tabs."""

    # Tab definitions
    tabs = [
        ("dashboard.html", "Training", "training"),
        ("viewer.html", "Viewer", "viewer"),
        ("benchmark.html", "Benchmarks", "benchmarks"),
        ("infrastructure.html", "Infrastructure", "infrastructure"),
    ]

    # Generate nav HTML
    nav_html = ""
    for href, label, page_id in tabs:
        active_class = "active" if page_id == active_page else ""
        nav_html += f'<a href="{href}" class="nav-tab {active_class}">{label}</a>\n'

    return f'''
    <header class="unified-header">
        <div class="nav-tabs">
            {nav_html}
        </div>
        <div class="controls-section">
            {controls_html}
            {meta_html}
        </div>
    </header>
    '''
```

### Infrastructure Page Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Infrastructure | OpenAdapt ML</title>
    <style>
        /* Shared variables from shared_ui.py */
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --accent: #00d4aa;
            --text-primary: #f0f0f0;
            --text-secondary: #888;
            --success: #34d399;
            --error: #ff5f5f;
            --warning: #f59e0b;
        }

        /* Status banner - prominent at top */
        .status-banner {
            position: fixed;
            top: 60px; /* Below header */
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 16px 24px;
            z-index: 999;
        }

        /* Status indicator with animation */
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--warning);
        }
        .status-indicator.building { animation: pulse 1s infinite; }
        .status-indicator.complete { background: var(--success); }
        .status-indicator.error { background: var(--error); }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Progress bar */
        .progress-bar-container {
            width: 250px;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
        }
        .progress-bar {
            height: 100%;
            background: var(--accent);
            transition: width 0.3s ease;
        }

        /* Main content grid */
        .main-content {
            margin-top: 140px; /* Below header + status banner */
            padding: 24px;
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 24px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Resource panels */
        .resource-panel {
            background: var(--bg-secondary);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 20px;
        }

        /* Log terminal */
        .log-terminal {
            background: #000;
            border-radius: 8px;
            padding: 16px;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 12px;
            line-height: 1.6;
            max-height: calc(100vh - 220px);
            overflow-y: auto;
            white-space: pre-wrap;
        }

        /* Log line coloring */
        .log-line.download { color: #60a5fa; }
        .log-line.extract { color: #a78bfa; }
        .log-line.done { color: var(--success); }
        .log-line.error { color: var(--error); }
        .log-line.step { color: var(--warning); }
    </style>
</head>
<body>
    <!-- Shared header from shared_ui.py -->
    {shared_header_html}

    <!-- Status banner -->
    <div class="status-banner">
        <div class="status-content">
            <div class="status-left">
                <div class="status-indicator" id="statusIndicator"></div>
                <div class="status-text">
                    <h1 id="statusTitle">Docker Build</h1>
                    <p id="statusDetail">Checking status...</p>
                </div>
            </div>
            <div class="progress-section">
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progressBar"></div>
                </div>
                <span class="progress-text" id="progressText">--%</span>
            </div>
        </div>
    </div>

    <!-- Main content -->
    <div class="main-content">
        <!-- Left sidebar: Resources -->
        <div class="sidebar">
            <div class="resource-panel">
                <h3>VMs</h3>
                <div id="vmList">Loading...</div>
            </div>
            <div class="resource-panel">
                <h3>Docker</h3>
                <div id="dockerStatus">Loading...</div>
            </div>
            <div class="resource-panel">
                <h3>Build History</h3>
                <div id="buildHistory">Loading...</div>
            </div>
        </div>

        <!-- Right: Log terminal -->
        <div class="log-section">
            <div class="log-terminal" id="logOutput">
                Connecting to build logs...
            </div>
        </div>
    </div>

    <script>
        // Configuration
        const POLL_INTERVAL = 2000;

        // Log line classification
        function classifyLine(line) {
            if (line.includes('sha256:') && line.includes('/')) return 'download';
            if (line.includes('extracting')) return 'extract';
            if (line.includes('DONE') || line.includes('done')) return 'done';
            if (line.includes('ERROR') || line.includes('error') || line.includes('failed')) return 'error';
            if (line.match(/^\s*#\d+\s+\[/)) return 'step';
            return '';
        }

        // Parse Docker download progress
        function parseProgress(text) {
            const matches = text.match(/(\d+\.?\d*)\s*GB\s*\/\s*(\d+\.?\d*)\s*GB/g);
            if (matches && matches.length > 0) {
                const lastMatch = matches[matches.length - 1];
                const parts = lastMatch.match(/(\d+\.?\d*)\s*GB\s*\/\s*(\d+\.?\d*)\s*GB/);
                if (parts) {
                    const current = parseFloat(parts[1]);
                    const total = parseFloat(parts[2]);
                    return Math.min(100, (current / total) * 100);
                }
            }
            if (text.includes('extracting') && !text.includes('GB /')) return 100;
            return 0;
        }

        // Update build logs
        async function updateLogs() {
            try {
                const response = await fetch('/api/build-log?' + Date.now());
                if (!response.ok) return;

                const data = await response.json();
                const text = data.content || '';

                // Update log terminal
                const lines = text.split('\n').slice(-300);
                const logOutput = document.getElementById('logOutput');
                logOutput.innerHTML = lines.map(line => {
                    const cls = classifyLine(line);
                    return `<span class="log-line ${cls}">${escapeHtml(line)}</span>`;
                }).join('\n');
                logOutput.scrollTop = logOutput.scrollHeight;

                // Update progress
                const progress = parseProgress(text);
                document.getElementById('progressBar').style.width = progress + '%';
                document.getElementById('progressText').textContent = progress.toFixed(0) + '%';

                // Update status indicator
                const indicator = document.getElementById('statusIndicator');
                const title = document.getElementById('statusTitle');

                if (text.includes('ERROR') || text.includes('failed to build')) {
                    indicator.className = 'status-indicator error';
                    title.textContent = 'Build Failed';
                } else if (text.includes('exporting to image') && text.includes('done')) {
                    indicator.className = 'status-indicator complete';
                    title.textContent = 'Build Complete';
                } else if (text.length > 0) {
                    indicator.className = 'status-indicator building';
                    title.textContent = 'Building...';
                }
            } catch (e) {
                console.log('Poll error:', e);
            }
        }

        // Update VM status
        async function updateVMs() {
            try {
                const response = await fetch('/api/vms?' + Date.now());
                if (!response.ok) return;

                const vms = await response.json();
                const vmList = document.getElementById('vmList');

                if (vms.length === 0) {
                    vmList.innerHTML = '<p class="text-muted">No VMs configured</p>';
                    return;
                }

                vmList.innerHTML = vms.map(vm => `
                    <div class="vm-item">
                        <span class="vm-name">${vm.name || 'VM'}</span>
                        <span class="vm-status ${vm.state}">${vm.state}</span>
                        ${vm.ip ? `<span class="vm-ip">${vm.ip}</span>` : ''}
                    </div>
                `).join('');
            } catch (e) {
                // VM endpoint may not exist
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Start polling
        updateLogs();
        updateVMs();
        setInterval(updateLogs, POLL_INTERVAL);
        setInterval(updateVMs, POLL_INTERVAL * 5); // VMs update less frequently
    </script>
</body>
</html>
```

### API Endpoint (Already Exists)

The `/api/build-log` endpoint in `local.py` (line 821) already provides:

```python
elif self.path.startswith('/api/build-log'):
    import glob
    output_files = sorted(
        glob.glob('/tmp/claude/-Users-*/tasks/*.output'),
        key=os.path.getmtime,
        reverse=True
    )
    content = ""
    if output_files:
        with open(output_files[0], 'r') as f:
            lines = f.readlines()
            content = ''.join(lines[-500:])

    self.send_json({"content": content, "file": output_files[0] if output_files else None})
```

**Enhancement needed**: Return build metadata (start time, source file, status).

## UX Considerations

### Navigation Flow

1. **During active build**: Infrastructure tab shows animated indicator
2. **Build complete**: Success indicator, logs preserved
3. **No active build**: Shows last build logs, "No active build" message

### Status Indicators Across Tabs

| Tab | Indicator Location | Shows |
|-----|-------------------|-------|
| Training | Header | Training status (running/complete/stopped) |
| Viewer | Header | Prediction count, capture name |
| Benchmarks | Header | Task progress, success rate |
| Infrastructure | Banner | Build progress, VM status |

### Keyboard Shortcuts

- `1-4`: Switch between tabs
- `Ctrl+L`: Clear log terminal
- `Ctrl+S`: Scroll to bottom
- `Escape`: Collapse panels

## File Changes Summary

| File | Changes | LOC |
|------|---------|-----|
| `shared_ui.py` | Add 4th tab, update active_page handling | ~20 |
| `benchmark_viewer.py` | Add `generate_infrastructure_viewer()` | ~300 |
| `local.py` | No changes (endpoint exists) | 0 |
| `trainer.py` | Call infrastructure generator during dashboard gen | ~10 |
| **Total** | | ~330 |

## Testing Plan

1. **Unit tests**: Verify HTML generation with all 4 tabs
2. **Integration test**: Start server, verify all tabs load
3. **Manual test**:
   - Run Docker build, verify live logs
   - Verify progress parsing
   - Test tab navigation
   - Test mobile responsiveness

## Rollout

1. **Phase 1**: Add tab + basic logs (this PR)
2. **Phase 2**: VM status integration (follow-up)
3. **Phase 3**: Build history persistence (follow-up)

## Open Questions

1. **Tab naming**: "Infrastructure" vs "Build" vs "Operations"?
2. **Build log storage**: Keep in `/tmp` or persist to `training_output/builds/`?
3. **Multi-user**: How to handle concurrent builds?

## Appendix: Current Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Shared header CSS | `shared_ui.py` | 10-104 |
| Shared header HTML | `shared_ui.py` | 107-144 |
| Training dashboard | `trainer.py` | 380-2029 |
| Viewer generation | `viewer.py` | 62-257 |
| Benchmark viewer | `benchmark_viewer.py` | 3556-3747 |
| Build log endpoint | `local.py` | 821-843 |
| Build monitor HTML | `training_output/current/build_monitor.html` | 1-276 |
