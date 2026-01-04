"""Benchmark viewer HTML generation.

This module generates a standalone HTML viewer for benchmark results,
showing task list with pass/fail status, step-by-step replay of
benchmark executions, screenshots, actions, and reasoning at each step.

Usage:
    from openadapt_ml.benchmarks.viewer import generate_benchmark_viewer

    # Generate viewer from benchmark results directory
    generate_benchmark_viewer(
        benchmark_dir=Path("benchmark_results/waa_eval_20241214"),
        output_path=Path("benchmark_results/waa_eval_20241214/benchmark.html"),
    )

Directory structure expected:
    benchmark_results/{run_name}/
    |-- metadata.json          # Benchmark config, models evaluated
    |-- summary.json           # Aggregate results
    |-- tasks/
    |   |-- task_001/
    |   |   |-- task.json      # Task definition
    |   |   |-- execution.json # Execution trace with steps
    |   |   |-- screenshots/   # Step screenshots
    |   |       |-- step_000.png
    |   |       |-- step_001.png
    |   |       |-- ...
    |   |-- task_002/
    |   |   |-- ...
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from openadapt_ml.training.shared_ui import (
    get_shared_header_css as _get_shared_header_css,
    generate_shared_header_html as _generate_shared_header_html,
)

logger = logging.getLogger(__name__)


def load_benchmark_metadata(benchmark_dir: Path) -> dict[str, Any]:
    """Load benchmark metadata from metadata.json.

    Args:
        benchmark_dir: Path to benchmark run directory.

    Returns:
        Metadata dictionary with benchmark_name, run_name, model_id, etc.
    """
    metadata_path = benchmark_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {
        "benchmark_name": "unknown",
        "run_name": benchmark_dir.name,
        "model_id": "unknown",
        "created_at": None,
    }


def load_benchmark_summary(benchmark_dir: Path) -> dict[str, Any]:
    """Load benchmark summary from summary.json.

    Args:
        benchmark_dir: Path to benchmark run directory.

    Returns:
        Summary dictionary with success_rate, num_tasks, etc.
    """
    summary_path = benchmark_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {
        "num_tasks": 0,
        "num_success": 0,
        "success_rate": 0.0,
        "avg_score": 0.0,
        "avg_steps": 0.0,
        "tasks": [],
    }


def load_task_results(benchmark_dir: Path) -> list[dict[str, Any]]:
    """Load all task results from benchmark run.

    Args:
        benchmark_dir: Path to benchmark run directory.

    Returns:
        List of task dictionaries with task definition, execution trace,
        and screenshot paths.
    """
    tasks_dir = benchmark_dir / "tasks"
    if not tasks_dir.exists():
        return []

    results = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        task_data: dict[str, Any] = {
            "task_dir": str(task_dir),
            "task_id": task_dir.name,
        }

        # Load task definition
        task_json = task_dir / "task.json"
        if task_json.exists():
            with open(task_json) as f:
                task_data["definition"] = json.load(f)
        else:
            task_data["definition"] = {"task_id": task_dir.name, "instruction": ""}

        # Load execution trace
        execution_json = task_dir / "execution.json"
        if execution_json.exists():
            with open(execution_json) as f:
                task_data["execution"] = json.load(f)
        else:
            task_data["execution"] = {"steps": [], "success": False, "num_steps": 0}

        # Load screenshot paths
        screenshots_dir = task_dir / "screenshots"
        if screenshots_dir.exists():
            screenshot_paths = sorted(screenshots_dir.glob("*.png"))
            task_data["screenshots"] = [str(p.relative_to(benchmark_dir)) for p in screenshot_paths]
        else:
            task_data["screenshots"] = []

        results.append(task_data)

    return results


def _encode_image_to_base64(image_path: Path) -> str | None:
    """Encode image to base64 data URL for embedding in HTML.

    Args:
        image_path: Path to PNG image.

    Returns:
        Data URL string or None if image cannot be loaded.
    """
    try:
        if image_path.exists():
            with open(image_path, "rb") as f:
                data = f.read()
            return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {e}")
    return None


def _get_domain_stats(tasks: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Calculate per-domain statistics.

    Args:
        tasks: List of task result dictionaries.

    Returns:
        Dictionary mapping domain name to {total, success, fail} counts.
    """
    domain_stats: dict[str, dict[str, int]] = {}

    for task in tasks:
        domain = task.get("definition", {}).get("domain", "unknown")
        success = task.get("execution", {}).get("success", False)

        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "success": 0, "fail": 0}

        domain_stats[domain]["total"] += 1
        if success:
            domain_stats[domain]["success"] += 1
        else:
            domain_stats[domain]["fail"] += 1

    return domain_stats


def generate_benchmark_viewer(
    benchmark_dir: Path,
    output_path: Path | None = None,
    embed_screenshots: bool = False,
) -> Path:
    """Generate HTML viewer for benchmark results.

    Args:
        benchmark_dir: Path to benchmark run directory containing metadata.json,
            summary.json, and tasks/ subdirectory.
        output_path: Path for output HTML file. Defaults to benchmark_dir/benchmark.html.
        embed_screenshots: If True, embed screenshots as base64 data URLs.
            This creates a larger but fully standalone HTML file.

    Returns:
        Path to generated HTML file.
    """
    benchmark_dir = Path(benchmark_dir)
    if output_path is None:
        output_path = benchmark_dir / "benchmark.html"

    # Load all data
    metadata = load_benchmark_metadata(benchmark_dir)
    summary = load_benchmark_summary(benchmark_dir)
    tasks = load_task_results(benchmark_dir)

    # Calculate domain statistics
    domain_stats = _get_domain_stats(tasks)

    # Generate HTML
    html = _generate_benchmark_viewer_html(
        metadata=metadata,
        summary=summary,
        tasks=tasks,
        domain_stats=domain_stats,
        benchmark_dir=benchmark_dir,
        embed_screenshots=embed_screenshots,
    )

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    logger.info(f"Generated benchmark viewer: {output_path}")
    return output_path


def _generate_benchmark_viewer_html(
    metadata: dict[str, Any],
    summary: dict[str, Any],
    tasks: list[dict[str, Any]],
    domain_stats: dict[str, dict[str, int]],
    benchmark_dir: Path,
    embed_screenshots: bool = False,
) -> str:
    """Generate the HTML content for benchmark viewer.

    Args:
        metadata: Benchmark metadata.
        summary: Summary statistics.
        tasks: List of task result dictionaries.
        domain_stats: Per-domain statistics.
        benchmark_dir: Base directory for resolving relative paths.
        embed_screenshots: If True, embed screenshots as base64.

    Returns:
        HTML string.
    """
    # Get shared header components
    shared_header_css = _get_shared_header_css()
    shared_header_html = _generate_shared_header_html("benchmarks")

    # Serialize data for JavaScript
    metadata_json = json.dumps(metadata)
    summary_json = json.dumps(summary)
    domain_stats_json = json.dumps(domain_stats)

    # Process tasks for JavaScript - include execution steps and screenshot paths
    tasks_for_js = []
    for task in tasks:
        task_js = {
            "task_id": task.get("task_id"),
            "definition": task.get("definition", {}),
            "execution": task.get("execution", {}),
            "screenshots": task.get("screenshots", []),
        }

        # Optionally embed screenshots as base64
        if embed_screenshots:
            embedded_screenshots = []
            for screenshot_rel_path in task.get("screenshots", []):
                screenshot_path = benchmark_dir / screenshot_rel_path
                data_url = _encode_image_to_base64(screenshot_path)
                embedded_screenshots.append(data_url or "")
            task_js["embedded_screenshots"] = embedded_screenshots

        tasks_for_js.append(task_js)

    tasks_json = json.dumps(tasks_for_js)

    # Calculate aggregate metrics
    num_tasks = len(tasks)
    num_success = sum(1 for t in tasks if t.get("execution", {}).get("success", False))
    success_rate = (num_success / num_tasks * 100) if num_tasks > 0 else 0

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Viewer - {metadata.get("run_name", "Unknown")}</title>
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
            max-width: 1600px;
            margin: 0 auto;
            padding: 24px;
        }}
        {shared_header_css}

        /* Summary Panel */
        .summary-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .summary-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .summary-header h2 {{
            font-size: 1rem;
            font-weight: 600;
        }}
        .summary-meta {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: "SF Mono", Monaco, monospace;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }}
        .stat-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 16px;
        }}
        .stat-card .stat-value {{
            font-size: 1.8rem;
            font-weight: 600;
            font-family: "SF Mono", Monaco, monospace;
        }}
        .stat-card .stat-value.success {{ color: var(--success); }}
        .stat-card .stat-value.error {{ color: var(--error); }}
        .stat-card .stat-label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }}

        /* Domain breakdown */
        .domain-breakdown {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .domain-tag {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 0.75rem;
        }}
        .domain-tag .domain-name {{
            color: var(--text-primary);
        }}
        .domain-tag .domain-stats {{
            font-family: "SF Mono", Monaco, monospace;
            color: var(--text-secondary);
        }}

        /* Filters */
        .filter-bar {{
            display: flex;
            gap: 16px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .filter-label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .filter-select {{
            padding: 8px 32px 8px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M3 4.5L6 7.5L9 4.5'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 10px center;
            transition: all 0.2s;
        }}
        .filter-select:hover {{ border-color: var(--accent); }}
        .filter-count {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-left: auto;
        }}

        /* Main Content Layout */
        .main-content {{
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 24px;
        }}
        @media (max-width: 1200px) {{
            .main-content {{ grid-template-columns: 1fr; }}
        }}

        /* Task List */
        .task-list {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            max-height: calc(100vh - 300px);
            overflow-y: auto;
        }}
        .task-list-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            background: var(--bg-secondary);
            z-index: 10;
        }}
        .task-list-header h3 {{
            font-size: 0.9rem;
            font-weight: 600;
        }}
        .task-item {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            transition: background 0.2s;
        }}
        .task-item:hover {{ background: var(--bg-tertiary); }}
        .task-item.active {{
            background: var(--accent-dim);
            border-left: 3px solid var(--accent);
        }}
        .task-item.hidden {{ display: none; }}
        .task-item .task-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }}
        .task-item .task-id {{
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .task-item .task-status {{
            font-size: 0.7rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        .task-item .task-status.success {{
            background: rgba(52, 211, 153, 0.2);
            color: var(--success);
        }}
        .task-item .task-status.fail {{
            background: rgba(255, 95, 95, 0.2);
            color: var(--error);
        }}
        .task-item .task-info {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        .task-item .task-domain {{
            color: var(--accent);
        }}

        /* Task Detail Panel */
        .task-detail {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }}
        .task-detail-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
        }}
        .task-detail-header h2 {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .task-detail-meta {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }}
        .task-detail-instruction {{
            font-style: italic;
            color: var(--text-primary);
            margin-top: 8px;
            padding: 10px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 0.85rem;
        }}

        /* Step Viewer */
        .step-viewer {{
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 16px;
            padding: 16px;
        }}
        @media (max-width: 900px) {{
            .step-viewer {{ grid-template-columns: 1fr; }}
        }}
        .screenshot-container {{
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .screenshot-container img {{
            max-width: 100%;
            max-height: 70vh;
            object-fit: contain;
        }}
        .screenshot-placeholder {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        .click-marker {{
            position: absolute;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            pointer-events: none;
            z-index: 100;
            background: rgba(167, 139, 250, 0.4);
            border: 2px solid #a78bfa;
            color: #a78bfa;
        }}

        /* Step Controls */
        .step-sidebar {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .step-controls {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .step-btn {{
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            min-width: 40px;
            text-align: center;
            transition: all 0.2s;
        }}
        .step-btn:hover {{ border-color: var(--accent); }}
        .step-btn.primary {{ flex: 1; min-width: 60px; }}
        .step-btn.active {{
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }}
        .step-progress {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: "SF Mono", Monaco, monospace;
        }}

        /* Step List */
        .step-list {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
        }}
        .step-list-item {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            transition: background 0.2s;
            font-size: 0.8rem;
        }}
        .step-list-item:hover {{ background: var(--bg-secondary); }}
        .step-list-item.active {{
            background: var(--accent-dim);
            border-left: 2px solid var(--accent);
        }}
        .step-list-item .step-num {{
            font-weight: 600;
            color: var(--accent);
            margin-right: 8px;
        }}
        .step-list-item .step-action {{
            color: var(--text-secondary);
        }}

        /* Action Detail */
        .action-detail {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
        }}
        .action-detail h4 {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .action-content {{
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.8rem;
            color: var(--text-primary);
            word-break: break-word;
        }}
        .reasoning-box {{
            margin-top: 12px;
            padding: 10px;
            background: var(--bg-secondary);
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.6;
            max-height: 200px;
            overflow-y: auto;
        }}
        .reasoning-box h4 {{
            margin-bottom: 8px;
        }}

        /* Speed Control */
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-left: auto;
        }}
        .speed-control label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .speed-control select {{
            padding: 4px 8px;
            border-radius: 4px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            font-size: 0.8rem;
            cursor: pointer;
        }}

        /* Progress Bar */
        .progress-bar {{
            width: 100%;
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
            cursor: pointer;
        }}
        .progress-bar .progress {{
            height: 100%;
            background: var(--accent);
            transition: width 0.1s ease;
        }}

        /* No task selected state */
        .no-task-selected {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
            color: var(--text-muted);
        }}
        .no-task-selected .icon {{
            font-size: 3rem;
            margin-bottom: 16px;
        }}
        .no-task-selected p {{
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    {shared_header_html}

    <div class="container">
        <!-- Summary Panel -->
        <div class="summary-panel">
            <div class="summary-header">
                <h2>Benchmark Results: {metadata.get("run_name", "Unknown")}</h2>
                <div class="summary-meta">
                    <span>Model: {metadata.get("model_id", "unknown")}</span>
                    <span> | </span>
                    <span>Created: {metadata.get("created_at", "N/A")}</span>
                </div>
            </div>
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-value">{num_tasks}</div>
                    <div class="stat-label">Total Tasks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value success">{num_success}</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value error">{num_tasks - num_success}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value {'success' if success_rate >= 50 else 'error'}">{success_rate:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
            <div class="domain-breakdown" id="domain-breakdown"></div>
        </div>

        <!-- Filters -->
        <div class="filter-bar">
            <div class="filter-group">
                <span class="filter-label">Domain:</span>
                <select class="filter-select" id="domain-filter">
                    <option value="all">All Domains</option>
                </select>
            </div>
            <div class="filter-group">
                <span class="filter-label">Status:</span>
                <select class="filter-select" id="status-filter">
                    <option value="all">All</option>
                    <option value="success">Passed</option>
                    <option value="fail">Failed</option>
                </select>
            </div>
            <span class="filter-count" id="filter-count">{num_tasks} tasks</span>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <!-- Task List -->
            <div class="task-list">
                <div class="task-list-header">
                    <h3>Tasks</h3>
                </div>
                <div id="task-list-items"></div>
            </div>

            <!-- Task Detail Panel -->
            <div class="task-detail" id="task-detail">
                <div class="no-task-selected" id="no-task-selected">
                    <div class="icon">+</div>
                    <p>Select a task from the list to view details</p>
                </div>
                <div id="task-detail-content" style="display:none;"></div>
            </div>
        </div>
    </div>

    <script>
    // Data from Python
    const metadata = {metadata_json};
    const summary = {summary_json};
    const domainStats = {domain_stats_json};
    const tasks = {tasks_json};
    const embedScreenshots = {'true' if embed_screenshots else 'false'};

    let currentTaskIndex = -1;
    let currentStepIndex = 0;
    let isPlaying = false;
    let playInterval = null;
    let playSpeed = 1000;

    // Initialize page
    function init() {{
        renderDomainBreakdown();
        populateDomainFilter();
        renderTaskList();
        setupFilters();
    }}

    function renderDomainBreakdown() {{
        const container = document.getElementById('domain-breakdown');
        let html = '';
        for (const [domain, stats] of Object.entries(domainStats)) {{
            const rate = stats.total > 0 ? (stats.success / stats.total * 100).toFixed(0) : 0;
            html += `
                <div class="domain-tag">
                    <span class="domain-name">${{domain}}</span>
                    <span class="domain-stats">${{stats.success}}/${{stats.total}} (${{rate}}%)</span>
                </div>
            `;
        }}
        container.innerHTML = html;
    }}

    function populateDomainFilter() {{
        const select = document.getElementById('domain-filter');
        for (const domain of Object.keys(domainStats).sort()) {{
            const option = document.createElement('option');
            option.value = domain;
            option.textContent = domain;
            select.appendChild(option);
        }}
    }}

    function renderTaskList() {{
        const container = document.getElementById('task-list-items');
        let html = '';
        tasks.forEach((task, idx) => {{
            const def = task.definition || {{}};
            const exec = task.execution || {{}};
            const success = exec.success || false;
            const domain = def.domain || 'unknown';
            const numSteps = exec.num_steps || 0;

            html += `
                <div class="task-item" data-idx="${{idx}}" data-domain="${{domain}}" data-status="${{success ? 'success' : 'fail'}}" onclick="selectTask(${{idx}})">
                    <div class="task-header">
                        <span class="task-id">${{task.task_id}}</span>
                        <span class="task-status ${{success ? 'success' : 'fail'}}">${{success ? 'PASS' : 'FAIL'}}</span>
                    </div>
                    <div class="task-info">
                        <span class="task-domain">${{domain}}</span>
                        <span> | ${{numSteps}} steps</span>
                    </div>
                </div>
            `;
        }});
        container.innerHTML = html;
    }}

    function setupFilters() {{
        document.getElementById('domain-filter').addEventListener('change', filterTasks);
        document.getElementById('status-filter').addEventListener('change', filterTasks);
    }}

    function filterTasks() {{
        const domainFilter = document.getElementById('domain-filter').value;
        const statusFilter = document.getElementById('status-filter').value;

        let visibleCount = 0;
        document.querySelectorAll('.task-item').forEach(item => {{
            const domain = item.dataset.domain;
            const status = item.dataset.status;

            const matchDomain = domainFilter === 'all' || domain === domainFilter;
            const matchStatus = statusFilter === 'all' || status === statusFilter;

            if (matchDomain && matchStatus) {{
                item.classList.remove('hidden');
                visibleCount++;
            }} else {{
                item.classList.add('hidden');
            }}
        }});

        document.getElementById('filter-count').textContent = `${{visibleCount}} tasks`;
    }}

    function selectTask(idx) {{
        currentTaskIndex = idx;
        currentStepIndex = 0;

        // Update active state in list
        document.querySelectorAll('.task-item').forEach((item, i) => {{
            item.classList.toggle('active', parseInt(item.dataset.idx) === idx);
        }});

        // Show task detail
        document.getElementById('no-task-selected').style.display = 'none';
        document.getElementById('task-detail-content').style.display = 'block';

        renderTaskDetail();
    }}

    function renderTaskDetail() {{
        if (currentTaskIndex < 0) return;

        const task = tasks[currentTaskIndex];
        const def = task.definition || {{}};
        const exec = task.execution || {{}};
        const steps = exec.steps || [];
        const success = exec.success || false;

        const container = document.getElementById('task-detail-content');
        container.innerHTML = `
            <div class="task-detail-header">
                <h2>${{task.task_id}} - <span style="color: ${{success ? 'var(--success)' : 'var(--error)'}}">${{success ? 'PASSED' : 'FAILED'}}</span></h2>
                <div class="task-detail-meta">
                    Domain: <strong>${{def.domain || 'unknown'}}</strong> |
                    Steps: <strong>${{exec.num_steps || steps.length}}</strong> |
                    Time: <strong>${{(exec.total_time_seconds || 0).toFixed(1)}}s</strong>
                    ${{exec.error ? `<br>Error: <span style="color:var(--error)">${{exec.error}}</span>` : ''}}
                </div>
                <div class="task-detail-instruction">
                    ${{def.instruction || 'No instruction available'}}
                </div>
            </div>
            <div class="step-viewer">
                <div class="screenshot-container" id="screenshot-container">
                    ${{steps.length > 0 ? '<img id="screenshot-img" src="" alt="Step screenshot">' : '<span class="screenshot-placeholder">No screenshots available</span>'}}
                </div>
                <div class="step-sidebar">
                    <div class="step-controls">
                        <button class="step-btn" onclick="prevStep()">Prev</button>
                        <button class="step-btn primary" id="play-btn" onclick="togglePlay()">Play</button>
                        <button class="step-btn" onclick="nextStep()">Next</button>
                        <span class="step-progress" id="step-progress">0 / ${{steps.length}}</span>
                        <div class="speed-control">
                            <label>Speed</label>
                            <select id="speed-select" onchange="changeSpeed(this.value)">
                                <option value="2000">0.5x</option>
                                <option value="1000" selected>1x</option>
                                <option value="500">2x</option>
                                <option value="250">4x</option>
                            </select>
                        </div>
                    </div>
                    <div class="progress-bar" onclick="seekStep(event)">
                        <div class="progress" id="step-progress-bar" style="width: 0%"></div>
                    </div>
                    <div class="step-list" id="step-list"></div>
                    <div class="action-detail" id="action-detail">
                        <h4>Action</h4>
                        <div class="action-content" id="action-content">-</div>
                    </div>
                    <div class="reasoning-box" id="reasoning-box" style="display:none;">
                        <h4>Reasoning</h4>
                        <div id="reasoning-content"></div>
                    </div>
                </div>
            </div>
        `;

        renderStepList();
        if (steps.length > 0) {{
            updateStep();
        }}
    }}

    function renderStepList() {{
        if (currentTaskIndex < 0) return;

        const task = tasks[currentTaskIndex];
        const steps = task.execution?.steps || [];
        const container = document.getElementById('step-list');

        let html = '';
        steps.forEach((step, idx) => {{
            const action = step.action || {{}};
            const actionType = action.type || 'unknown';
            html += `
                <div class="step-list-item ${{idx === currentStepIndex ? 'active' : ''}}" onclick="goToStep(${{idx}})">
                    <span class="step-num">#${{idx}}</span>
                    <span class="step-action">${{actionType.toUpperCase()}}</span>
                </div>
            `;
        }});
        container.innerHTML = html || '<div style="padding:12px;color:var(--text-muted);">No steps</div>';
    }}

    function updateStep() {{
        if (currentTaskIndex < 0) return;

        const task = tasks[currentTaskIndex];
        const steps = task.execution?.steps || [];
        const screenshots = task.screenshots || [];

        if (steps.length === 0) return;

        const step = steps[currentStepIndex] || {{}};
        const action = step.action || {{}};

        // Update screenshot
        const img = document.getElementById('screenshot-img');
        if (img) {{
            if (embedScreenshots && task.embedded_screenshots && task.embedded_screenshots[currentStepIndex]) {{
                img.src = task.embedded_screenshots[currentStepIndex];
            }} else if (screenshots[currentStepIndex]) {{
                img.src = screenshots[currentStepIndex];
            }} else if (step.screenshot_path) {{
                img.src = step.screenshot_path;
            }} else {{
                img.src = '';
            }}
        }}

        // Update click marker if action has coordinates
        const container = document.getElementById('screenshot-container');
        // Remove existing markers
        container.querySelectorAll('.click-marker').forEach(m => m.remove());

        if (action.x !== null && action.y !== null && action.x !== undefined && action.y !== undefined) {{
            const marker = document.createElement('div');
            marker.className = 'click-marker';
            marker.style.left = `${{action.x * 100}}%`;
            marker.style.top = `${{action.y * 100}}%`;
            marker.textContent = 'AI';
            container.appendChild(marker);
        }}

        // Update progress
        document.getElementById('step-progress').textContent = `${{currentStepIndex + 1}} / ${{steps.length}}`;
        const progressPct = steps.length > 1 ? (currentStepIndex / (steps.length - 1)) * 100 : 0;
        document.getElementById('step-progress-bar').style.width = `${{progressPct}}%`;

        // Update action detail
        const actionContent = document.getElementById('action-content');
        let actionText = action.type ? action.type.toUpperCase() : 'unknown';
        if (action.x !== null && action.y !== null && action.x !== undefined && action.y !== undefined) {{
            actionText += ` (${{(action.x * 100).toFixed(1)}}%, ${{(action.y * 100).toFixed(1)}}%)`;
        }}
        if (action.text) {{
            actionText += ` "${{action.text}}"`;
        }}
        if (action.key) {{
            actionText += ` [${{action.key}}]`;
        }}
        actionContent.textContent = actionText;

        // Update reasoning
        const reasoningBox = document.getElementById('reasoning-box');
        const reasoningContent = document.getElementById('reasoning-content');
        if (step.reasoning) {{
            reasoningBox.style.display = 'block';
            reasoningContent.textContent = step.reasoning;
        }} else {{
            reasoningBox.style.display = 'none';
        }}

        // Update step list active state
        document.querySelectorAll('.step-list-item').forEach((item, idx) => {{
            item.classList.toggle('active', idx === currentStepIndex);
        }});
    }}

    function prevStep() {{
        if (currentStepIndex > 0) {{
            currentStepIndex--;
            updateStep();
        }}
    }}

    function nextStep() {{
        const task = tasks[currentTaskIndex];
        const steps = task?.execution?.steps || [];
        if (currentStepIndex < steps.length - 1) {{
            currentStepIndex++;
            updateStep();
        }} else if (isPlaying) {{
            stopPlay();
        }}
    }}

    function goToStep(idx) {{
        currentStepIndex = idx;
        updateStep();
    }}

    function seekStep(event) {{
        const task = tasks[currentTaskIndex];
        const steps = task?.execution?.steps || [];
        if (steps.length === 0) return;

        const bar = event.currentTarget;
        const rect = bar.getBoundingClientRect();
        const pct = (event.clientX - rect.left) / rect.width;
        currentStepIndex = Math.floor(pct * steps.length);
        currentStepIndex = Math.max(0, Math.min(currentStepIndex, steps.length - 1));
        updateStep();
    }}

    function togglePlay() {{
        if (isPlaying) {{
            stopPlay();
        }} else {{
            startPlay();
        }}
    }}

    function startPlay() {{
        isPlaying = true;
        document.getElementById('play-btn').textContent = 'Pause';
        document.getElementById('play-btn').classList.add('active');
        playInterval = setInterval(nextStep, playSpeed);
    }}

    function stopPlay() {{
        isPlaying = false;
        document.getElementById('play-btn').textContent = 'Play';
        document.getElementById('play-btn').classList.remove('active');
        if (playInterval) {{
            clearInterval(playInterval);
            playInterval = null;
        }}
    }}

    function changeSpeed(value) {{
        playSpeed = parseInt(value);
        if (isPlaying) {{
            stopPlay();
            startPlay();
        }}
    }}

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        switch (e.key) {{
            case ' ':
                e.preventDefault();
                togglePlay();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                prevStep();
                break;
            case 'ArrowRight':
                e.preventDefault();
                nextStep();
                break;
            case 'Home':
                e.preventDefault();
                goToStep(0);
                break;
            case 'End':
                e.preventDefault();
                const task = tasks[currentTaskIndex];
                const steps = task?.execution?.steps || [];
                goToStep(steps.length - 1);
                break;
        }}
    }});

    // Initialize on load
    document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
'''

    return html
