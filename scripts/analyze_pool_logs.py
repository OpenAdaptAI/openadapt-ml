#!/usr/bin/env python3
"""Analyze WAA pool benchmark logs and generate HTML summary.

Usage:
    python scripts/analyze_pool_logs.py benchmark_results/pool_run_20260204/
"""

import re
import sys
from datetime import datetime
from pathlib import Path


def parse_log_file(log_path: Path) -> dict:
    """Parse a WAA benchmark log file."""
    content = log_path.read_text()

    # Extract task completions
    tasks = []
    finished_pattern = r"Finished (\w+)/([a-f0-9-]+-WOS)"
    result_pattern = r"Result: ([\d.]+)"

    # Find all finished tasks with their results
    finished_matches = list(re.finditer(finished_pattern, content))
    result_matches = list(re.finditer(result_pattern, content))

    for match in finished_matches:
        domain = match.group(1)
        task_id = match.group(2)
        # Find the result that precedes this finish
        result = 0.0
        for rm in result_matches:
            if rm.start() < match.start():
                result = float(rm.group(1))
        tasks.append({
            "domain": domain,
            "task_id": task_id,
            "result": result,
            "success": result > 0.0
        })

    # Extract total task count from progress bar
    total_match = re.search(r"Example:\s+\d+%\|.*?\|\s+\d+/(\d+)", content)
    total_tasks = int(total_match.group(1)) if total_match else 0

    return {
        "file": log_path.name,
        "tasks_completed": len(tasks),
        "tasks_total": total_tasks,
        "tasks": tasks,
        "successes": sum(1 for t in tasks if t["success"]),
    }


def generate_html_report(results: list, output_path: Path) -> None:
    """Generate HTML summary report."""
    total_completed = sum(r["tasks_completed"] for r in results)
    total_tasks = sum(r["tasks_total"] for r in results)
    total_success = sum(r["successes"] for r in results)
    success_rate = (total_success / total_completed * 100) if total_completed > 0 else 0

    # Group by domain
    domain_stats = {}
    for r in results:
        for task in r["tasks"]:
            domain = task["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "success": 0}
            domain_stats[domain]["total"] += 1
            if task["success"]:
                domain_stats[domain]["success"] += 1

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>WAA Benchmark Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats {{ display: flex; gap: 30px; margin: 20px 0; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 36px; font-weight: bold; color: #2563eb; }}
        .stat-label {{ color: #666; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .success {{ color: #16a34a; }}
        .fail {{ color: #dc2626; }}
        .worker {{ margin-top: 30px; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .badge-success {{ background: #dcfce7; color: #16a34a; }}
        .badge-fail {{ background: #fee2e2; color: #dc2626; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>WAA Benchmark Results</h1>

        <div class="summary">
            <h2>Summary</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{total_completed}/{total_tasks}</div>
                    <div class="stat-label">Tasks Completed</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{total_success}</div>
                    <div class="stat-label">Successes</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{success_rate:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(results)}</div>
                    <div class="stat-label">Workers</div>
                </div>
            </div>
        </div>

        <div class="summary">
            <h2>By Domain</h2>
            <table>
                <tr><th>Domain</th><th>Completed</th><th>Success</th><th>Rate</th></tr>
"""

    for domain, stats in sorted(domain_stats.items()):
        rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        html += f"""                <tr>
                    <td>{domain}</td>
                    <td>{stats['total']}</td>
                    <td>{stats['success']}</td>
                    <td>{rate:.0f}%</td>
                </tr>
"""

    html += """            </table>
        </div>
"""

    for r in results:
        html += f"""
        <div class="worker">
            <h2>{r['file']}</h2>
            <p>Completed: {r['tasks_completed']}/{r['tasks_total']} tasks</p>
            <table>
                <tr><th>Domain</th><th>Task ID</th><th>Result</th><th>Status</th></tr>
"""
        for task in r["tasks"]:
            status_class = "badge-success" if task["success"] else "badge-fail"
            status_text = "PASS" if task["success"] else "FAIL"
            html += f"""                <tr>
                    <td>{task['domain']}</td>
                    <td><code>{task['task_id'][:20]}...</code></td>
                    <td>{task['result']:.2f}</td>
                    <td><span class="badge {status_class}">{status_text}</span></td>
                </tr>
"""
        html += """            </table>
        </div>
"""

    html += f"""
        <footer style="margin-top: 40px; color: #666; font-size: 12px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </footer>
    </div>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"Generated: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_pool_logs.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    # Find log files
    log_files = list(results_dir.glob("waa-pool-*.log"))
    if not log_files:
        print(f"No log files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(log_files)} log files")

    # Parse logs
    results = []
    for log_file in sorted(log_files):
        print(f"  Parsing {log_file.name}...")
        results.append(parse_log_file(log_file))

    # Generate HTML
    output_path = results_dir / "results.html"
    generate_html_report(results, output_path)

    # Print summary
    total_completed = sum(r["tasks_completed"] for r in results)
    total_success = sum(r["successes"] for r in results)
    print(f"\nSummary: {total_completed} tasks completed, {total_success} successes ({total_success/total_completed*100:.0f}% rate)" if total_completed > 0 else "\nNo tasks completed")


if __name__ == "__main__":
    main()
