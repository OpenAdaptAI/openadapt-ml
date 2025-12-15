"""Benchmark viewer generation functions.

This module provides functions to generate HTML viewers for benchmark evaluation results.
It is imported and used by trainer.py to maintain consistency with other viewer components.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_benchmark_viewer(
    benchmark_dir: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Generate benchmark viewer HTML from benchmark results directory.

    Args:
        benchmark_dir: Path to benchmark results directory (e.g., benchmark_results/waa_eval_20241214/)
        output_path: Optional path for output benchmark.html (default: benchmark_dir/benchmark.html)

    Returns:
        Path to generated benchmark.html file

    Example:
        from openadapt_ml.training.benchmark_viewer import generate_benchmark_viewer

        viewer_path = generate_benchmark_viewer("benchmark_results/test_run_phase1")
        print(f"Generated: {viewer_path}")
    """
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    if output_path is None:
        output_path = benchmark_dir / "benchmark.html"
    else:
        output_path = Path(output_path)

    # Load metadata
    metadata_path = benchmark_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {benchmark_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load summary
    summary_path = benchmark_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    # Load all task results
    tasks_dir = benchmark_dir / "tasks"
    task_results = []

    if tasks_dir.exists():
        for task_dir in sorted(tasks_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_json = task_dir / "task.json"
            execution_json = task_dir / "execution.json"

            if not task_json.exists() or not execution_json.exists():
                continue

            with open(task_json) as f:
                task_data = json.load(f)

            with open(execution_json) as f:
                execution_data = json.load(f)

            # Combine task and execution data
            task_result = {
                "task_id": task_data["task_id"],
                "instruction": task_data["instruction"],
                "domain": task_data.get("domain", "unknown"),
                "success": execution_data["success"],
                "score": execution_data.get("score", 0.0),
                "num_steps": execution_data["num_steps"],
                "total_time_seconds": execution_data.get("total_time_seconds", 0.0),
                "error": execution_data.get("error"),
                "reason": execution_data.get("reason"),
                "steps": execution_data.get("steps", []),
                "screenshots_dir": str(task_dir / "screenshots"),
            }
            task_results.append(task_result)

    # Import shared header components from trainer
    from openadapt_ml.training.trainer import _get_shared_header_css, _generate_shared_header_html

    # Generate HTML
    html = _generate_benchmark_viewer_html(
        metadata=metadata,
        summary=summary,
        tasks=task_results,
        benchmark_dir=benchmark_dir,
        shared_header_css=_get_shared_header_css(),
        shared_header_html=_generate_shared_header_html("benchmarks"),
    )

    output_path.write_text(html)
    print(f"Generated benchmark viewer: {output_path}")
    return output_path


def generate_empty_benchmark_viewer(output_path: Path | str) -> Path:
    """Generate an empty benchmark viewer with guidance when no real data exists.

    Args:
        output_path: Path to output benchmark.html

    Returns:
        Path to generated file
    """
    output_path = Path(output_path)

    # Import shared header components from trainer
    from openadapt_ml.training.trainer import _get_shared_header_css, _generate_shared_header_html

    shared_header_css = _get_shared_header_css()
    shared_header_html = _generate_shared_header_html("benchmarks")

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Viewer - No Data</title>
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
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }}
        {shared_header_css}
        .empty-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: calc(100vh - 60px);
            padding: 40px;
            text-align: center;
        }}
        .empty-icon {{
            font-size: 64px;
            margin-bottom: 24px;
            opacity: 0.5;
        }}
        .empty-title {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 12px;
        }}
        .empty-description {{
            color: var(--text-secondary);
            margin-bottom: 32px;
            max-width: 500px;
            line-height: 1.6;
        }}
        .guide-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 16px;
            max-width: 600px;
            text-align: left;
        }}
        .guide-card h3 {{
            color: var(--accent);
            margin-bottom: 12px;
            font-size: 16px;
        }}
        .guide-card code {{
            background: var(--bg-tertiary);
            padding: 12px 16px;
            border-radius: 8px;
            display: block;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            color: var(--text-primary);
            white-space: pre-wrap;
            margin-bottom: 12px;
        }}
        .guide-card p {{
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.5;
        }}
        a {{
            color: var(--accent);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    {shared_header_html}

    <div class="empty-state">
        <div class="empty-icon">🚧</div>
        <h1 class="empty-title">Windows Agent Arena Integration</h1>
        <p class="empty-description">
            This tab will display results from <strong>WAA benchmark</strong> evaluations (154 real Windows tasks).<br>
            <span style="color: var(--text-muted);">Status: Work in Progress - requires Windows VM or Azure setup</span>
        </p>

        <div class="guide-card" style="background: var(--bg-tertiary); border-color: var(--accent);">
            <h3 style="color: var(--text-primary);">Looking for synthetic benchmark results?</h3>
            <code>uv run python -m openadapt_ml.scripts.eval_policy \\
  --config configs/qwen3vl_synthetic_som.yaml \\
  --backend qwen3 --dsl-mode som</code>
            <p>The synthetic login benchmark (with SoM mode achieving 100%) uses eval_policy.py, not this viewer.</p>
        </div>

        <div class="guide-card">
            <h3>WAA Local Setup (Windows Required)</h3>
            <code># Clone WAA repository
git clone https://github.com/anthropics/WindowsAgentArena

# Run evaluation
uv run python -m openadapt_ml.benchmarks.cli run-local \\
  --waa-path /path/to/WindowsAgentArena</code>
            <p>Requires Windows environment. See <a href="https://github.com/anthropics/WindowsAgentArena" style="color: var(--accent);">WAA repo</a> for setup.</p>
        </div>

        <div class="guide-card">
            <h3>WAA on Azure (Parallel VMs)</h3>
            <code># Setup Azure resources
python scripts/setup_azure.py

# Run evaluation on Azure VMs
uv run python -m openadapt_ml.benchmarks.cli run-azure --workers 4</code>
            <p>Runs WAA tasks in parallel on Azure Windows VMs. See docs/azure_waa_setup.md</p>
        </div>
    </div>
</body>
</html>'''

    output_path.write_text(html)
    return output_path


def _generate_benchmark_viewer_html(
    metadata: dict,
    summary: dict,
    tasks: list[dict],
    benchmark_dir: Path,
    shared_header_css: str,
    shared_header_html: str,
) -> str:
    """Generate the benchmark viewer HTML content.

    Args:
        metadata: Benchmark metadata (run name, model ID, etc.)
        summary: Summary statistics (success rate, avg steps, etc.)
        tasks: List of task results with execution data
        benchmark_dir: Path to benchmark directory (for relative paths)
        shared_header_css: CSS for shared header
        shared_header_html: HTML for shared header

    Returns:
        Complete HTML string
    """
    # Prepare data as JSON
    tasks_json = json.dumps(tasks)
    summary_json = json.dumps(summary)
    metadata_json = json.dumps(metadata)

    # Calculate unique domains for filter
    domains = sorted(set(task["domain"] for task in tasks))
    domains_json = json.dumps(domains)

    # Generate HTML
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
            --success: #00d4aa;
            --failure: #ff4444;
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
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }}

        {shared_header_css}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .summary-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s;
        }}

        .summary-card:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
        }}

        .summary-card .label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .summary-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .summary-card .subtitle {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        .filters {{
            display: flex;
            gap: 12px;
            padding: 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .filter-select {{
            padding: 8px 32px 8px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            background: rgba(0,0,0,0.4);
            color: var(--text-primary);
            border: 1px solid rgba(255,255,255,0.1);
            cursor: pointer;
            appearance: none;
            background-image: url('data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%2712%27 height=%278%27%3E%3Cpath fill=%27%23888%27 d=%27M0 0l6 8 6-8z%27/%3E%3C/svg%3E');
            background-repeat: no-repeat;
            background-position: right 10px center;
            transition: all 0.2s;
        }}

        .filter-select:hover {{
            border-color: var(--accent);
            background-color: rgba(0,212,170,0.1);
        }}

        .task-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .task-item {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.2s;
        }}

        .task-item:hover {{
            border-color: var(--accent);
        }}

        .task-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            cursor: pointer;
            user-select: none;
        }}

        .task-header:hover {{
            background: var(--bg-tertiary);
        }}

        .task-status {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
            flex-shrink: 0;
        }}

        .task-status.success {{
            background: var(--success);
            color: var(--bg-primary);
        }}

        .task-status.failure {{
            background: var(--failure);
            color: var(--bg-primary);
        }}

        .task-info {{
            flex: 1;
            min-width: 0;
        }}

        .task-id {{
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }}

        .task-instruction {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .task-meta {{
            display: flex;
            gap: 20px;
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: "SF Mono", Monaco, monospace;
        }}

        .task-domain {{
            padding: 4px 10px;
            background: rgba(0,212,170,0.15);
            border-radius: 4px;
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 600;
        }}

        .task-expand-icon {{
            color: var(--text-muted);
            transition: transform 0.2s;
        }}

        .task-item.expanded .task-expand-icon {{
            transform: rotate(90deg);
        }}

        .task-details {{
            display: none;
            padding: 0 20px 20px;
            border-top: 1px solid var(--border-color);
        }}

        .task-item.expanded .task-details {{
            display: block;
        }}

        .steps-list {{
            margin-top: 16px;
        }}

        .step-item {{
            display: flex;
            gap: 16px;
            padding: 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 8px;
        }}

        .step-number {{
            font-weight: 600;
            color: var(--accent);
            min-width: 60px;
        }}

        .step-screenshot {{
            max-width: 200px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }}

        .step-action {{
            flex: 1;
        }}

        .action-type {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            color: var(--accent);
            margin-bottom: 4px;
        }}

        .action-details {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: "SF Mono", Monaco, monospace;
        }}

        .no-tasks {{
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }}

        .no-tasks-icon {{
            font-size: 3rem;
            margin-bottom: 16px;
            opacity: 0.5;
        }}
    </style>
</head>
<body>
    {shared_header_html}

    <div class="container">
        <div class="summary-cards">
            <div class="summary-card">
                <div class="label">Total Tasks</div>
                <div class="value" id="total-tasks">0</div>
            </div>
            <div class="summary-card">
                <div class="label">Success Rate</div>
                <div class="value" id="success-rate">0%</div>
                <div class="subtitle" id="success-count">0 / 0 passed</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Steps</div>
                <div class="value" id="avg-steps">0</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Time</div>
                <div class="value" id="avg-time">0s</div>
            </div>
        </div>

        <div class="filters">
            <span class="filter-label">Status:</span>
            <select class="filter-select" id="filter-status">
                <option value="all">All Tasks</option>
                <option value="success">Success Only</option>
                <option value="failure">Failure Only</option>
            </select>

            <span class="filter-label">Domain:</span>
            <select class="filter-select" id="filter-domain">
                <option value="all">All Domains</option>
            </select>
        </div>

        <div class="task-list" id="task-list"></div>

        <div class="no-tasks" id="no-tasks" style="display: none;">
            <div class="no-tasks-icon">📋</div>
            <div>No tasks match the current filters</div>
        </div>
    </div>

    <script>
        // Data from backend
        const tasks = {tasks_json};
        const summary = {summary_json};
        const metadata = {metadata_json};
        const domains = {domains_json};

        // State
        let currentFilters = {{
            status: 'all',
            domain: 'all'
        }};

        // Initialize
        function init() {{
            updateSummaryCards();
            populateDomainFilter();
            renderTaskList();

            // Event listeners
            document.getElementById('filter-status').addEventListener('change', (e) => {{
                currentFilters.status = e.target.value;
                renderTaskList();
            }});

            document.getElementById('filter-domain').addEventListener('change', (e) => {{
                currentFilters.domain = e.target.value;
                renderTaskList();
            }});
        }}

        function updateSummaryCards() {{
            document.getElementById('total-tasks').textContent = summary.num_tasks || tasks.length;

            const successRate = (summary.success_rate || 0) * 100;
            document.getElementById('success-rate').textContent = successRate.toFixed(1) + '%';
            document.getElementById('success-count').textContent =
                `${{summary.num_success || 0}} / ${{summary.num_tasks || tasks.length}} passed`;

            const avgSteps = summary.avg_steps || 0;
            document.getElementById('avg-steps').textContent = avgSteps.toFixed(1);

            const avgTime = summary.avg_time_seconds || 0;
            document.getElementById('avg-time').textContent = avgTime.toFixed(2) + 's';
        }}

        function populateDomainFilter() {{
            const select = document.getElementById('filter-domain');
            domains.forEach(domain => {{
                const option = document.createElement('option');
                option.value = domain;
                option.textContent = domain.charAt(0).toUpperCase() + domain.slice(1);
                select.appendChild(option);
            }});
        }}

        function filterTasks() {{
            return tasks.filter(task => {{
                if (currentFilters.status !== 'all') {{
                    const isSuccess = task.success;
                    if (currentFilters.status === 'success' && !isSuccess) return false;
                    if (currentFilters.status === 'failure' && isSuccess) return false;
                }}

                if (currentFilters.domain !== 'all' && task.domain !== currentFilters.domain) {{
                    return false;
                }}

                return true;
            }});
        }}

        function renderTaskList() {{
            const filteredTasks = filterTasks();
            const container = document.getElementById('task-list');
            const noTasks = document.getElementById('no-tasks');

            if (filteredTasks.length === 0) {{
                container.innerHTML = '';
                noTasks.style.display = 'block';
                return;
            }}

            noTasks.style.display = 'none';
            container.innerHTML = filteredTasks.map(task => renderTaskItem(task)).join('');

            // Add click handlers
            document.querySelectorAll('.task-header').forEach(header => {{
                header.addEventListener('click', () => {{
                    const item = header.closest('.task-item');
                    item.classList.toggle('expanded');
                }});
            }});
        }}

        function renderTaskItem(task) {{
            const statusClass = task.success ? 'success' : 'failure';
            const statusIcon = task.success ? '✓' : '✗';

            const stepsHtml = task.steps && task.steps.length > 0
                ? task.steps.map(step => renderStep(step, task)).join('')
                : '<div style="padding: 12px; color: var(--text-muted);">No step details available</div>';

            return `
                <div class="task-item" data-task-id="${{task.task_id}}">
                    <div class="task-header">
                        <div class="task-status ${{statusClass}}">${{statusIcon}}</div>
                        <div class="task-info">
                            <div class="task-id">${{task.task_id}}</div>
                            <div class="task-instruction">${{task.instruction}}</div>
                        </div>
                        <div class="task-domain">${{task.domain}}</div>
                        <div class="task-meta">
                            <span>${{task.num_steps}} steps</span>
                            <span>${{task.total_time_seconds.toFixed(2)}}s</span>
                        </div>
                        <div class="task-expand-icon">▶</div>
                    </div>
                    <div class="task-details">
                        <div class="steps-list">
                            ${{stepsHtml}}
                        </div>
                    </div>
                </div>
            `;
        }}

        function renderStep(step, task) {{
            const actionType = step.action.type || 'unknown';
            const actionDetails = formatActionDetails(step.action);

            // Build screenshot path relative to benchmark.html
            const screenshotPath = step.screenshot_path
                ? `tasks/${{task.task_id}}/${{step.screenshot_path}}`
                : '';

            const screenshotHtml = screenshotPath
                ? `<img src="${{screenshotPath}}" class="step-screenshot" alt="Step ${{step.step_idx}}" />`
                : '';

            return `
                <div class="step-item">
                    <div class="step-number">Step ${{step.step_idx}}</div>
                    ${{screenshotHtml}}
                    <div class="step-action">
                        <div class="action-type">${{actionType}}</div>
                        <div class="action-details">${{actionDetails}}</div>
                        ${{step.reasoning ? `<div style="margin-top: 8px; font-style: italic; color: var(--text-secondary);">${{step.reasoning}}</div>` : ''}}
                    </div>
                </div>
            `;
        }}

        function formatActionDetails(action) {{
            const parts = [];

            if (action.x !== null && action.y !== null) {{
                parts.push(`x: ${{action.x.toFixed(3)}}, y: ${{action.y.toFixed(3)}}`);
            }}

            if (action.text) {{
                parts.push(`text: "${{action.text}}"`);
            }}

            if (action.key) {{
                parts.push(`key: ${{action.key}}`);
            }}

            if (action.target_name) {{
                parts.push(`target: ${{action.target_name}}`);
            }}

            return parts.length > 0 ? parts.join(', ') : 'No details';
        }}

        // Initialize on page load
        init();
    </script>
</body>
</html>'''

    return html
