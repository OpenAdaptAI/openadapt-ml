"""Azure Operations Dashboard HTML generation.

Generates a real-time dashboard for monitoring Azure VM operations
(Docker builds, Windows boot, benchmark runs, etc.).

Usage:
    from openadapt_ml.training.azure_ops_viewer import generate_azure_ops_dashboard

    # Generate and write HTML
    generate_azure_ops_dashboard(Path("training_output/current/azure_ops.html"))
"""

from __future__ import annotations

from pathlib import Path

from openadapt_ml.training.shared_ui import (
    get_shared_header_css as _get_shared_header_css,
)


def generate_azure_ops_dashboard(output_path: Path | str | None = None) -> str:
    """Generate Azure Operations Dashboard HTML.

    Args:
        output_path: Optional path to write the HTML file.

    Returns:
        HTML string.
    """
    shared_header_css = _get_shared_header_css()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Azure Operations Dashboard</title>
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --border-color: rgba(255, 255, 255, 0.06);
            --text-primary: #f0f0f0;
            --text-secondary: #888;
            --text-muted: #555;
            --accent: #00d4aa;
            --accent-dim: rgba(0, 212, 170, 0.15);
            --success: #34d399;
            --error: #ff5f5f;
            --warning: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}
        {shared_header_css}

        /* Header with nav */
        .unified-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 24px;
            background: linear-gradient(180deg, rgba(18,18,26,0.98) 0%, rgba(26,26,36,0.98) 100%);
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 20px;
            gap: 16px;
            flex-wrap: wrap;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .nav-tabs {{
            display: flex;
            align-items: center;
            gap: 4px;
            background: rgba(0,0,0,0.3);
            padding: 4px;
            border-radius: 8px;
        }}
        .nav-tab {{
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            color: var(--text-secondary);
            background: transparent;
            border: none;
            transition: all 0.2s;
            cursor: pointer;
        }}
        .nav-tab:hover {{
            color: var(--text-primary);
            background: rgba(255,255,255,0.05);
        }}
        .nav-tab.active {{
            color: var(--bg-primary);
            background: var(--accent);
            font-weight: 600;
        }}

        /* Status Grid */
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        /* Status Cards */
        .status-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }}
        .status-card h3 {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }}
        .status-card .value {{
            font-size: 1.5rem;
            font-weight: 600;
            font-family: "SF Mono", Monaco, monospace;
        }}
        .status-card .value.success {{ color: var(--success); }}
        .status-card .value.warning {{ color: var(--warning); }}
        .status-card .value.error {{ color: var(--error); }}
        .status-card .value.accent {{ color: var(--accent); }}
        .status-card .sub {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        /* Progress Section */
        .progress-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .progress-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .progress-header h2 {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        .progress-header .operation-badge {{
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .progress-header .operation-badge.running {{
            background: rgba(0, 212, 170, 0.2);
            color: var(--accent);
        }}
        .progress-header .operation-badge.idle {{
            background: rgba(136, 136, 136, 0.2);
            color: var(--text-secondary);
        }}
        .progress-header .operation-badge.complete {{
            background: rgba(52, 211, 153, 0.2);
            color: var(--success);
        }}
        .progress-header .operation-badge.failed {{
            background: rgba(255, 95, 95, 0.2);
            color: var(--error);
        }}

        .progress-bar-container {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            height: 24px;
            overflow: hidden;
            margin-bottom: 12px;
        }}
        .progress-bar {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent) 0%, #00f5c4 100%);
            border-radius: 8px;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .progress-bar span {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--bg-primary);
        }}
        .progress-info {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        /* VNC Button */
        .vnc-button {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.2s;
        }}
        .vnc-button:hover {{
            background: #00f5c4;
            transform: translateY(-1px);
        }}
        .vnc-button:disabled {{
            background: var(--text-muted);
            cursor: not-allowed;
            transform: none;
        }}

        /* Log Viewer */
        .log-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }}
        .log-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
        }}
        .log-header h3 {{
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .log-controls {{
            display: flex;
            gap: 8px;
        }}
        .log-controls button {{
            padding: 4px 12px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background: transparent;
            color: var(--text-secondary);
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .log-controls button:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        .log-controls button.active {{
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }}
        .log-viewer {{
            font-family: "SF Mono", Monaco, "Courier New", monospace;
            font-size: 0.8rem;
            line-height: 1.6;
            padding: 16px;
            background: #000;
            color: #ccc;
            height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .log-viewer .log-line {{
            padding: 2px 0;
        }}
        .log-viewer .log-line.error {{
            color: var(--error);
        }}
        .log-viewer .log-line.success {{
            color: var(--success);
        }}
        .log-viewer .log-line.warning {{
            color: var(--warning);
        }}
        .log-viewer .log-line.step {{
            color: var(--accent);
            font-weight: bold;
        }}

        /* Error Banner */
        .error-banner {{
            background: rgba(255, 95, 95, 0.15);
            border: 1px solid var(--error);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            display: none;
        }}
        .error-banner.visible {{
            display: block;
        }}
        .error-banner h4 {{
            color: var(--error);
            font-size: 0.9rem;
            margin-bottom: 8px;
        }}
        .error-banner p {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        /* VNC Embed Section */
        .vnc-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 24px;
            overflow: hidden;
        }}
        .vnc-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            user-select: none;
        }}
        .vnc-header:hover {{
            background: rgba(26,26,36,0.9);
        }}
        .vnc-header h3 {{
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .vnc-header .toggle-icon {{
            font-size: 0.7rem;
            color: var(--text-muted);
            transition: transform 0.2s;
        }}
        .vnc-header .toggle-icon.expanded {{
            transform: rotate(90deg);
        }}
        .vnc-controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .vnc-controls button {{
            padding: 4px 12px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background: transparent;
            color: var(--text-secondary);
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .vnc-controls button:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        .vnc-controls .size-select {{
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background: var(--bg-secondary);
            color: var(--text-secondary);
            font-size: 0.75rem;
            cursor: pointer;
        }}
        .vnc-container {{
            display: none;
            background: #000;
            position: relative;
        }}
        .vnc-container.visible {{
            display: block;
        }}
        .vnc-iframe {{
            width: 100%;
            border: none;
            background: #000;
        }}
        .vnc-placeholder {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 400px;
            color: var(--text-muted);
            gap: 12px;
        }}
        .vnc-placeholder .icon {{
            font-size: 3rem;
            opacity: 0.3;
        }}
        .vnc-placeholder .message {{
            font-size: 0.9rem;
        }}
        .vnc-status {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        .vnc-status.connected {{
            color: var(--success);
        }}
        .vnc-status.disconnected {{
            color: var(--error);
        }}

        /* VM Info Row */
        .vm-info {{
            display: flex;
            gap: 24px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}
        .vm-info-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
        }}
        .vm-info-item .label {{
            color: var(--text-muted);
        }}
        .vm-info-item .value {{
            color: var(--text-primary);
            font-family: "SF Mono", Monaco, monospace;
        }}
        .vm-info-item .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
        }}
        .vm-info-item .status-dot.running {{
            background: var(--success);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}

        /* Refresh indicator */
        .refresh-indicator {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        .refresh-indicator.loading {{
            color: var(--accent);
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <div class="unified-header">
        <div class="nav-tabs">
            <a href="dashboard.html" class="nav-tab">Training</a>
            <a href="viewer.html" class="nav-tab">Viewer</a>
            <a href="benchmark.html" class="nav-tab">Benchmarks</a>
            <a href="azure_ops.html" class="nav-tab active">Azure Ops</a>
        </div>
        <div class="refresh-indicator" id="refresh-indicator">Connecting...</div>
    </div>

    <div class="container">
        <!-- Error Banner -->
        <div class="error-banner" id="error-banner">
            <h4>Operation Error</h4>
            <p id="error-message"></p>
        </div>

        <!-- VM Info Row -->
        <div class="vm-info">
            <div class="vm-info-item">
                <span class="status-dot" id="vm-status-dot"></span>
                <span class="label">VM:</span>
                <span class="value" id="vm-state">Unknown</span>
            </div>
            <div class="vm-info-item">
                <span class="label">IP:</span>
                <span class="value" id="vm-ip">-</span>
            </div>
            <div class="vm-info-item">
                <span class="label">Size:</span>
                <span class="value" id="vm-size">-</span>
            </div>
            <div class="vm-info-item">
                <a href="http://localhost:8006" target="_blank" class="vnc-button" id="vnc-button" disabled>
                    Open VNC Desktop
                </a>
            </div>
        </div>

        <!-- Status Cards -->
        <div class="status-grid">
            <div class="status-card">
                <h3>Running Cost</h3>
                <div class="value accent" id="cost-value">$0.00</div>
                <div class="sub" id="cost-rate">$0.00/hr</div>
            </div>
            <div class="status-card">
                <h3>Elapsed Time</h3>
                <div class="value" id="elapsed-value">0m 0s</div>
                <div class="sub" id="started-at">Not started</div>
            </div>
            <div class="status-card">
                <h3>ETA</h3>
                <div class="value" id="eta-value">-</div>
                <div class="sub" id="projected-cost">-</div>
            </div>
        </div>

        <!-- Progress Section -->
        <div class="progress-section">
            <div class="progress-header">
                <h2 id="operation-title">Waiting for operation...</h2>
                <span class="operation-badge idle" id="operation-badge">Idle</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progress-bar" style="width: 0%">
                    <span id="progress-text">0%</span>
                </div>
            </div>
            <div class="progress-info">
                <span id="phase-text">-</span>
                <span id="step-text">Step 0 / 0</span>
            </div>
        </div>

        <!-- VNC Embed Section -->
        <div class="vnc-section">
            <div class="vnc-header" onclick="toggleVNC()">
                <h3>
                    <span class="toggle-icon" id="vnc-toggle-icon">&#9654;</span>
                    Windows VM Screen
                    <span class="vnc-status" id="vnc-status">Checking...</span>
                </h3>
                <div class="vnc-controls" onclick="event.stopPropagation()">
                    <select class="size-select" id="vnc-size" onchange="updateVNCSize()">
                        <option value="400">400px</option>
                        <option value="500">500px</option>
                        <option value="600" selected>600px</option>
                        <option value="800">800px</option>
                        <option value="1000">1000px</option>
                    </select>
                    <button onclick="refreshVNC()">Refresh</button>
                    <button onclick="openVNCExternal()">Open in New Tab</button>
                </div>
            </div>
            <div class="vnc-container" id="vnc-container">
                <div class="vnc-placeholder" id="vnc-placeholder">
                    <span class="icon">&#128421;</span>
                    <span class="message">VNC not available - VM may not be running</span>
                    <span class="message" style="font-size: 0.8rem; color: var(--text-muted);">Start the VM and ensure SSH tunnel is active (localhost:8006)</span>
                </div>
                <iframe
                    id="vnc-iframe"
                    class="vnc-iframe"
                    src=""
                    style="display: none; height: 600px;"
                    allow="clipboard-read; clipboard-write"
                    sandbox="allow-scripts allow-same-origin allow-forms allow-pointer-lock"
                ></iframe>
            </div>
        </div>

        <!-- Log Viewer -->
        <div class="log-section">
            <div class="log-header">
                <h3>Live Logs</h3>
                <div class="log-controls">
                    <button id="auto-scroll-btn" class="active" onclick="toggleAutoScroll()">Auto-scroll</button>
                    <button onclick="clearLogs()">Clear</button>
                    <button id="copy-logs-btn" onclick="copyLogs()">Copy Logs</button>
                </div>
            </div>
            <div class="log-viewer" id="log-viewer">
                <div class="log-line">Waiting for logs...</div>
            </div>
        </div>
    </div>

    <script>
    let autoScroll = true;
    let lastLogLength = 0;
    let pollInterval = null;
    let eventSource = null;
    let useSSE = true;  // Try SSE first, fallback to polling
    // Note: elapsed_seconds and cost_usd are now computed server-side
    // No client-side timer needed - server sends fresh values on each request

    function formatDuration(seconds) {{
        if (!seconds || seconds <= 0) return '-';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${{h}}h ${{m}}m ${{s}}s`;
        if (m > 0) return `${{m}}m ${{s}}s`;
        return `${{s}}s`;
    }}

    function formatCost(usd) {{
        if (usd === null || usd === undefined) return '$0.00';
        return `$${{usd.toFixed(2)}}`;
    }}

    // Note: Client-side timer removed. Server now computes elapsed_seconds
    // and cost_usd fresh on every request, ensuring accuracy without
    // client clock dependencies.

    function classifyLogLine(line) {{
        const lower = line.toLowerCase();
        if (lower.includes('error') || lower.includes('failed') || lower.includes('exception')) {{
            return 'error';
        }}
        if (lower.includes('success') || lower.includes('complete') || lower.includes('done')) {{
            return 'success';
        }}
        if (lower.includes('warning') || lower.includes('warn')) {{
            return 'warning';
        }}
        if (/step\\s+\\d+\\/\\d+/i.test(line)) {{
            return 'step';
        }}
        return '';
    }}

    function updateUI(status) {{
        // Server now sends pre-computed elapsed_seconds and cost_usd
        // No client-side storage needed

        // Operation badge
        const badge = document.getElementById('operation-badge');
        const operation = status.operation || 'idle';
        badge.textContent = operation.replace(/_/g, ' ').toUpperCase();
        badge.className = 'operation-badge ' + (
            operation === 'idle' ? 'idle' :
            operation === 'complete' ? 'complete' :
            operation === 'failed' ? 'failed' : 'running'
        );

        // Operation title
        const titles = {{
            'idle': 'Waiting for operation...',
            'vm_create': 'Creating Azure VM',
            'docker_install': 'Installing Docker',
            'docker_build': 'Building Docker Image',
            'windows_boot': 'Booting Windows VM',
            'benchmark': 'Running Benchmark',
            'complete': 'Operation Complete',
            'failed': 'Operation Failed'
        }};
        document.getElementById('operation-title').textContent = titles[operation] || `Running: ${{operation}}`;

        // Progress bar
        const pct = Math.min(100, Math.max(0, status.progress_pct || 0));
        document.getElementById('progress-bar').style.width = `${{pct}}%`;
        document.getElementById('progress-text').textContent = `${{pct.toFixed(0)}}%`;

        // Phase and step
        document.getElementById('phase-text').textContent = status.phase || '-';
        document.getElementById('step-text').textContent =
            status.total_steps > 0
                ? `Step ${{status.step}} / ${{status.total_steps}}`
                : '-';

        // Cost
        document.getElementById('cost-value').textContent = formatCost(status.cost_usd);
        document.getElementById('cost-rate').textContent = `$${{(status.hourly_rate_usd || 0).toFixed(3)}}/hr`;

        // Elapsed time
        document.getElementById('elapsed-value').textContent = formatDuration(status.elapsed_seconds);
        if (status.started_at) {{
            const started = new Date(status.started_at);
            document.getElementById('started-at').textContent = `Started: ${{started.toLocaleTimeString()}}`;
        }} else {{
            document.getElementById('started-at').textContent = 'Not started';
        }}

        // ETA
        document.getElementById('eta-value').textContent = formatDuration(status.eta_seconds);
        if (status.eta_seconds && status.hourly_rate_usd) {{
            const projectedTotal = (status.elapsed_seconds + status.eta_seconds) / 3600 * status.hourly_rate_usd;
            document.getElementById('projected-cost').textContent = `Projected total: $${{projectedTotal.toFixed(2)}}`;
        }} else {{
            document.getElementById('projected-cost').textContent = '-';
        }}

        // VM info
        document.getElementById('vm-state').textContent = status.vm_state || 'unknown';
        document.getElementById('vm-ip').textContent = status.vm_ip || '-';
        document.getElementById('vm-size').textContent = status.vm_size || '-';

        const statusDot = document.getElementById('vm-status-dot');
        statusDot.className = 'status-dot' + (status.vm_state === 'running' ? ' running' : '');

        // VNC button
        const vncBtn = document.getElementById('vnc-button');
        if (status.vnc_url && status.vm_state === 'running') {{
            vncBtn.href = status.vnc_url;
            vncBtn.removeAttribute('disabled');
        }} else {{
            vncBtn.setAttribute('disabled', 'true');
        }}

        // Update VNC embed status based on VM state
        updateVNCFromVMState(status.vm_state, status.vnc_url);

        // Error banner
        const errorBanner = document.getElementById('error-banner');
        const errorMsg = document.getElementById('error-message');
        if (status.error) {{
            errorBanner.classList.add('visible');
            errorMsg.textContent = status.error;
        }} else {{
            errorBanner.classList.remove('visible');
        }}

        // Log viewer
        const logViewer = document.getElementById('log-viewer');
        const logs = status.log_tail || [];

        if (logs.length !== lastLogLength) {{
            lastLogLength = logs.length;
            logViewer.innerHTML = logs.map(line => {{
                const cls = classifyLogLine(line);
                return `<div class="log-line ${{cls}}">${{escapeHtml(line)}}</div>`;
            }}).join('') || '<div class="log-line">No logs yet...</div>';

            if (autoScroll) {{
                logViewer.scrollTop = logViewer.scrollHeight;
            }}
        }}
    }}

    function escapeHtml(text) {{
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }}

    function toggleAutoScroll() {{
        autoScroll = !autoScroll;
        const btn = document.getElementById('auto-scroll-btn');
        btn.classList.toggle('active', autoScroll);
    }}

    function clearLogs() {{
        document.getElementById('log-viewer').innerHTML = '<div class="log-line">Logs cleared...</div>';
        lastLogLength = 0;
    }}

    function copyLogs() {{
        const logViewer = document.getElementById('log-viewer');
        const logLines = logViewer.querySelectorAll('.log-line');
        const text = Array.from(logLines).map(line => line.textContent).join('\\n');

        navigator.clipboard.writeText(text).then(() => {{
            const btn = document.getElementById('copy-logs-btn');
            const originalText = btn.textContent;
            btn.textContent = 'Copied!';
            btn.classList.add('active');
            setTimeout(() => {{
                btn.textContent = originalText;
                btn.classList.remove('active');
            }}, 1500);
        }}).catch(err => {{
            console.error('Failed to copy logs:', err);
            alert('Failed to copy logs to clipboard');
        }});
    }}

    // VNC Embed functionality
    const VNC_URL = 'http://localhost:8006';
    let vncExpanded = false;
    let vncAvailable = false;
    let vncCheckInterval = null;

    function toggleVNC() {{
        vncExpanded = !vncExpanded;
        const container = document.getElementById('vnc-container');
        const toggleIcon = document.getElementById('vnc-toggle-icon');

        if (vncExpanded) {{
            container.classList.add('visible');
            toggleIcon.classList.add('expanded');
            // Check VNC availability and load if available
            checkVNCAndLoad();
        }} else {{
            container.classList.remove('visible');
            toggleIcon.classList.remove('expanded');
        }}
    }}

    function updateVNCSize() {{
        const sizeSelect = document.getElementById('vnc-size');
        const iframe = document.getElementById('vnc-iframe');
        iframe.style.height = sizeSelect.value + 'px';
    }}

    function refreshVNC() {{
        const iframe = document.getElementById('vnc-iframe');
        if (vncAvailable) {{
            // Force reload by resetting src
            const currentSrc = iframe.src;
            iframe.src = '';
            setTimeout(() => {{ iframe.src = currentSrc; }}, 100);
        }} else {{
            checkVNCAndLoad();
        }}
    }}

    function openVNCExternal() {{
        window.open(VNC_URL, '_blank');
    }}

    async function checkVNCAvailability() {{
        try {{
            // Try to fetch from VNC URL (this may fail due to CORS, but that's actually fine)
            // We'll use a different approach - check if the server responds
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);

            // We can't directly check the VNC URL due to CORS, but we can check our backend
            // which should know about tunnel status
            const response = await fetch('/api/tunnels', {{ signal: controller.signal }});
            clearTimeout(timeoutId);

            if (response.ok) {{
                const tunnels = await response.json();
                // Check if VNC tunnel is active
                if (tunnels.vnc && tunnels.vnc.active) {{
                    return true;
                }}
            }}
            // Fallback: try to detect if noVNC is running by checking the iframe
            // This is a bit hacky but works in practice
            return false;
        }} catch (e) {{
            // If /api/tunnels doesn't exist, try direct check
            // Create an img element to test if VNC server responds
            return new Promise((resolve) => {{
                const img = new Image();
                img.onload = () => resolve(true);
                img.onerror = () => {{
                    // Error could mean CORS (server is up) or actually down
                    // Try to load iframe anyway if we get an error (likely CORS)
                    resolve('maybe');
                }};
                img.src = VNC_URL + '/favicon.ico?' + Date.now();
                setTimeout(() => resolve(false), 3000);
            }});
        }}
    }}

    async function checkVNCAndLoad() {{
        const statusEl = document.getElementById('vnc-status');
        const placeholder = document.getElementById('vnc-placeholder');
        const iframe = document.getElementById('vnc-iframe');

        statusEl.textContent = 'Checking...';
        statusEl.className = 'vnc-status';

        const available = await checkVNCAvailability();

        if (available === true || available === 'maybe') {{
            // VNC appears to be available, load the iframe
            vncAvailable = true;
            statusEl.textContent = 'Connected';
            statusEl.className = 'vnc-status connected';
            placeholder.style.display = 'none';
            iframe.style.display = 'block';

            // Only set src if not already set
            if (!iframe.src || iframe.src === window.location.href) {{
                iframe.src = VNC_URL;
            }}

            // Start periodic checks to update status
            startVNCStatusCheck();
        }} else {{
            // VNC not available
            vncAvailable = false;
            statusEl.textContent = 'Disconnected';
            statusEl.className = 'vnc-status disconnected';
            placeholder.style.display = 'flex';
            iframe.style.display = 'none';
            iframe.src = '';

            // Start periodic checks to detect when VNC becomes available
            startVNCStatusCheck();
        }}
    }}

    function startVNCStatusCheck() {{
        // Only run checks if VNC section is expanded
        if (vncCheckInterval) {{
            clearInterval(vncCheckInterval);
        }}
        vncCheckInterval = setInterval(async () => {{
            if (!vncExpanded) {{
                clearInterval(vncCheckInterval);
                vncCheckInterval = null;
                return;
            }}
            await checkVNCAndLoad();
        }}, 10000);  // Check every 10 seconds
    }}

    // Also update VNC status when VM state changes
    function updateVNCFromVMState(vmState, vncUrl) {{
        const statusEl = document.getElementById('vnc-status');

        if (vmState === 'running' && vncUrl) {{
            if (!vncAvailable && vncExpanded) {{
                // VM just came online, check VNC
                checkVNCAndLoad();
            }}
        }} else if (vmState !== 'running') {{
            // VM is not running, mark VNC as disconnected
            vncAvailable = false;
            statusEl.textContent = 'VM Offline';
            statusEl.className = 'vnc-status disconnected';

            if (vncExpanded) {{
                const placeholder = document.getElementById('vnc-placeholder');
                const iframe = document.getElementById('vnc-iframe');
                placeholder.style.display = 'flex';
                iframe.style.display = 'none';
            }}
        }}
    }}

    function updateIndicator(mode, extra) {{
        const indicator = document.getElementById('refresh-indicator');
        if (mode === 'sse') {{
            indicator.textContent = 'Connected via SSE';
            indicator.classList.remove('loading');
        }} else if (mode === 'polling') {{
            indicator.textContent = 'Polling every 2s';
            indicator.classList.remove('loading');
        }} else if (mode === 'connecting') {{
            indicator.textContent = 'Connecting...';
            indicator.classList.add('loading');
        }} else if (mode === 'error') {{
            indicator.textContent = `Error: ${{extra || 'unknown'}}`;
            indicator.classList.remove('loading');
        }}
    }}

    // SSE connection
    function connectSSE() {{
        if (eventSource) {{
            eventSource.close();
        }}

        updateIndicator('connecting');

        eventSource = new EventSource('/api/azure-ops-sse');

        eventSource.addEventListener('connected', (event) => {{
            console.log('SSE connected:', JSON.parse(event.data));
            updateIndicator('sse');
        }});

        eventSource.addEventListener('status', (event) => {{
            const status = JSON.parse(event.data);
            updateUI(status);
        }});

        eventSource.addEventListener('heartbeat', (event) => {{
            // Keep-alive received, connection is healthy
            console.log('SSE heartbeat:', JSON.parse(event.data));
        }});

        eventSource.addEventListener('error', (event) => {{
            if (event.data) {{
                const error = JSON.parse(event.data);
                console.error('SSE error event:', error);
            }}
        }});

        eventSource.onerror = (event) => {{
            console.warn('SSE connection error, falling back to polling');
            eventSource.close();
            eventSource = null;
            useSSE = false;
            startPolling();
        }};
    }}

    // Polling fallback
    async function fetchStatus() {{
        const indicator = document.getElementById('refresh-indicator');
        indicator.textContent = 'Refreshing...';
        indicator.classList.add('loading');

        try {{
            const response = await fetch('/api/azure-ops-status');
            if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
            const status = await response.json();
            updateUI(status);
            updateIndicator('polling');
        }} catch (error) {{
            console.error('Failed to fetch status:', error);
            updateIndicator('error', error.message);
        }}
    }}

    function startPolling() {{
        updateIndicator('polling');
        fetchStatus();  // Initial fetch
        pollInterval = setInterval(fetchStatus, 2000);  // Poll every 2 seconds
    }}

    function stopPolling() {{
        if (pollInterval) {{
            clearInterval(pollInterval);
            pollInterval = null;
        }}
    }}

    function stopSSE() {{
        if (eventSource) {{
            eventSource.close();
            eventSource = null;
        }}
    }}

    // Initialize connection
    function initConnection() {{
        // Server computes elapsed_seconds/cost_usd fresh on each request
        // No client-side timer needed - just connect to data source
        if (useSSE && typeof EventSource !== 'undefined') {{
            connectSSE();
        }} else {{
            startPolling();
        }}
    }}

    // Handle visibility changes
    document.addEventListener('visibilitychange', () => {{
        if (document.hidden) {{
            stopPolling();
            stopSSE();
        }} else {{
            initConnection();
        }}
    }});

    // Initialize
    document.addEventListener('DOMContentLoaded', initConnection);
    </script>
</body>
</html>
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html
