"""Local GPU training CLI.

Provides commands equivalent to lambda_labs.py but for local execution
on CUDA or Apple Silicon.

Usage:
    # Train on a capture
    uv run python -m openadapt_ml.cloud.local train --capture ~/captures/my-workflow

    # Check training status
    uv run python -m openadapt_ml.cloud.local status

    # Check training health
    uv run python -m openadapt_ml.cloud.local check

    # Start dashboard server
    uv run python -m openadapt_ml.cloud.local serve --open

    # Regenerate viewer
    uv run python -m openadapt_ml.cloud.local viewer
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

# Training output directory
TRAINING_OUTPUT = Path("training_output")


def get_current_output_dir() -> Path:
    """Get the current job's output directory.

    Returns the 'current' symlink path if it exists, otherwise falls back
    to the base training_output directory for backward compatibility.
    """
    current_link = TRAINING_OUTPUT / "current"
    if current_link.is_symlink() or current_link.exists():
        return current_link
    # Fallback for backward compatibility with old structure
    return TRAINING_OUTPUT


def _regenerate_viewer_if_possible(output_dir: Path) -> bool:
    """Regenerate viewer.html if comparison data exists.

    Returns True if viewer was regenerated, False otherwise.
    """
    from openadapt_ml.training.trainer import generate_unified_viewer_from_output_dir

    try:
        viewer_path = generate_unified_viewer_from_output_dir(output_dir)
        if viewer_path:
            print(f"Regenerated viewer: {viewer_path}")
            return True
        return False
    except Exception as e:
        print(f"Could not regenerate viewer: {e}")
        return False


def _is_mock_benchmark(benchmark_dir: Path) -> bool:
    """Check if a benchmark run is mock/test data (not real evaluation).

    Returns True if the benchmark is mock data that should be filtered out.

    Note: API evaluations using the mock WAA adapter (waa-mock) are considered
    real evaluations and should NOT be filtered out, since they represent actual
    model performance on test tasks.
    """
    # Check summary.json for model_id
    summary_path = benchmark_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            model_id = summary.get("model_id", "").lower()
            # Filter out mock/test/random agent runs (but keep API models like "anthropic-api")
            if any(term in model_id for term in ["random-agent", "scripted-agent"]):
                return True
        except Exception:
            pass

    # Check metadata.json for model_id
    metadata_path = benchmark_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            model_id = metadata.get("model_id", "").lower()
            if any(term in model_id for term in ["random-agent", "scripted-agent"]):
                return True
        except Exception:
            pass

    # Check for test runs (but allow waa-mock evaluations with real API models)
    # Only filter out purely synthetic test data directories
    if any(term in benchmark_dir.name.lower() for term in ["test_run", "test_cli", "quick_demo"]):
        return True

    return False


def _regenerate_benchmark_viewer_if_available(output_dir: Path) -> bool:
    """Regenerate benchmark.html from all real benchmark results.

    Loads all non-mock benchmark runs from benchmark_results/ directory
    and generates a unified benchmark viewer supporting multiple runs.
    If no real benchmark data exists, generates an empty state viewer with guidance.

    Returns True if benchmark viewer was regenerated, False otherwise.
    """
    from openadapt_ml.training.benchmark_viewer import (
        generate_multi_run_benchmark_viewer,
        generate_empty_benchmark_viewer,
    )

    benchmark_results_dir = Path("benchmark_results")

    # Find real (non-mock) benchmark runs
    real_benchmarks = []
    if benchmark_results_dir.exists():
        for d in benchmark_results_dir.iterdir():
            if d.is_dir() and (d / "summary.json").exists():
                if not _is_mock_benchmark(d):
                    real_benchmarks.append(d)

    benchmark_html_path = output_dir / "benchmark.html"

    if not real_benchmarks:
        # No real benchmark data - generate empty state viewer
        try:
            generate_empty_benchmark_viewer(benchmark_html_path)

            # Still create symlink for azure_jobs.json access (even without real benchmarks)
            if benchmark_results_dir.exists():
                benchmark_results_link = output_dir / "benchmark_results"
                if benchmark_results_link.is_symlink():
                    benchmark_results_link.unlink()
                elif benchmark_results_link.exists():
                    shutil.rmtree(benchmark_results_link)
                benchmark_results_link.symlink_to(benchmark_results_dir.absolute())

            print("  Generated benchmark viewer: No real evaluation data yet")
            return True
        except Exception as e:
            print(f"  Could not generate empty benchmark viewer: {e}")
            return False

    # Sort by modification time (most recent first)
    real_benchmarks.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    try:
        # Generate multi-run benchmark.html in the output directory
        generate_multi_run_benchmark_viewer(real_benchmarks, benchmark_html_path)

        # Copy all tasks folders for screenshots (organized by run)
        benchmark_tasks_dir = output_dir / "benchmark_tasks"
        if benchmark_tasks_dir.exists():
            shutil.rmtree(benchmark_tasks_dir)
        benchmark_tasks_dir.mkdir(exist_ok=True)

        for benchmark_dir in real_benchmarks:
            tasks_src = benchmark_dir / "tasks"
            if tasks_src.exists():
                tasks_dst = benchmark_tasks_dir / benchmark_dir.name
                shutil.copytree(tasks_src, tasks_dst)

        # Create symlink for benchmark_results directory (for azure_jobs.json access)
        benchmark_results_link = output_dir / "benchmark_results"
        if benchmark_results_link.is_symlink():
            benchmark_results_link.unlink()
        elif benchmark_results_link.exists():
            shutil.rmtree(benchmark_results_link)
        benchmark_results_link.symlink_to(benchmark_results_dir.absolute())

        print(f"  Regenerated benchmark viewer with {len(real_benchmarks)} run(s)")
        return True
    except Exception as e:
        print(f"  Could not regenerate benchmark viewer: {e}")
        import traceback
        traceback.print_exc()
        return False


def detect_device() -> str:
    """Detect available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return f"cuda ({device_name})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (Apple Silicon)"
        else:
            return "cpu"
    except ImportError:
        return "unknown (torch not installed)"


def get_training_status() -> dict[str, Any]:
    """Get current training status from training_output/current."""
    current_dir = get_current_output_dir()

    status = {
        "running": False,
        "epoch": 0,
        "step": 0,
        "loss": None,
        "device": detect_device(),
        "has_dashboard": False,
        "has_viewer": False,
        "checkpoints": [],
        "job_id": None,
        "output_dir": str(current_dir),
    }

    log_file = current_dir / "training_log.json"
    if log_file.exists():
        try:
            with open(log_file) as f:
                data = json.load(f)
            status["job_id"] = data.get("job_id")
            status["epoch"] = data.get("epoch", 0)
            status["step"] = data.get("step", 0)
            status["loss"] = data.get("loss")
            status["learning_rate"] = data.get("learning_rate")
            status["losses"] = data.get("losses", [])
            status["status"] = data.get("status", "unknown")
            status["running"] = data.get("status") == "training"
        except (json.JSONDecodeError, KeyError):
            pass

    status["has_dashboard"] = (current_dir / "dashboard.html").exists()
    status["has_viewer"] = (current_dir / "viewer.html").exists()

    # Find checkpoints
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        status["checkpoints"] = sorted([
            d.name for d in checkpoints_dir.iterdir()
            if d.is_dir() and (d / "adapter_config.json").exists()
        ])

    return status


def cmd_status(args: argparse.Namespace) -> int:
    """Show local training status."""
    status = get_training_status()
    current_dir = get_current_output_dir()

    print(f"\n{'='*50}")
    print("LOCAL TRAINING STATUS")
    print(f"{'='*50}")
    print(f"Device: {status['device']}")
    print(f"Status: {'RUNNING' if status['running'] else 'IDLE'}")
    if status.get("job_id"):
        print(f"Job ID: {status['job_id']}")
    print(f"Output: {current_dir}")

    if status.get("epoch"):
        print(f"\nProgress:")
        print(f"  Epoch: {status['epoch']}")
        print(f"  Step: {status['step']}")
        if status.get("loss"):
            print(f"  Loss: {status['loss']:.4f}")
        if status.get("learning_rate"):
            print(f"  LR: {status['learning_rate']:.2e}")

    if status["checkpoints"]:
        print(f"\nCheckpoints ({len(status['checkpoints'])}):")
        for cp in status["checkpoints"][-5:]:  # Show last 5
            print(f"  - {cp}")

    print(f"\nDashboard: {'✓' if status['has_dashboard'] else '✗'} {current_dir}/dashboard.html")
    print(f"Viewer: {'✓' if status['has_viewer'] else '✗'} {current_dir}/viewer.html")
    print()

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run training locally."""
    capture_path = Path(args.capture).expanduser().resolve()
    if not capture_path.exists():
        print(f"Error: Capture not found: {capture_path}")
        return 1

    # Determine goal from capture directory name if not provided
    goal = args.goal
    if not goal:
        goal = capture_path.name.replace("-", " ").replace("_", " ").title()

    # Select config based on device
    config = args.config
    if not config:
        device = detect_device()
        if "cuda" in device:
            config = "configs/qwen3vl_capture.yaml"
        else:
            config = "configs/qwen3vl_capture_4bit.yaml"

    config_path = Path(config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1

    print(f"\n{'='*50}")
    print("STARTING LOCAL TRAINING")
    print(f"{'='*50}")
    print(f"Capture: {capture_path}")
    print(f"Goal: {goal}")
    print(f"Config: {config}")
    print(f"Device: {detect_device()}")
    print()

    # Build command
    cmd = [
        sys.executable, "-m", "openadapt_ml.scripts.train",
        "--config", str(config_path),
        "--capture", str(capture_path),
        "--goal", goal,
    ]

    if args.open:
        cmd.append("--open")

    # Run training
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 130


def cmd_check(args: argparse.Namespace) -> int:
    """Check training health and early stopping analysis."""
    status = get_training_status()

    print(f"\n{'='*50}")
    print("TRAINING HEALTH CHECK")
    print(f"{'='*50}")

    raw_losses = status.get("losses", [])
    if not raw_losses:
        print("No training data found.")
        print("Run training first with: uv run python -m openadapt_ml.cloud.local train --capture <path>")
        return 1

    # Extract loss values (handle both dict and float formats)
    losses = []
    for item in raw_losses:
        if isinstance(item, dict):
            losses.append(item.get("loss", 0))
        else:
            losses.append(float(item))

    print(f"Total steps: {len(losses)}")
    print(f"Current epoch: {status.get('epoch', 0)}")

    # Loss analysis
    if len(losses) >= 2:
        first_loss = losses[0]
        last_loss = losses[-1]
        min_loss = min(losses)
        max_loss = max(losses)

        print(f"\nLoss progression:")
        print(f"  First: {first_loss:.4f}")
        print(f"  Last: {last_loss:.4f}")
        print(f"  Min: {min_loss:.4f}")
        print(f"  Max: {max_loss:.4f}")
        print(f"  Reduction: {((first_loss - last_loss) / first_loss * 100):.1f}%")

        # Check for convergence
        if len(losses) >= 10:
            recent = losses[-10:]
            recent_avg = sum(recent) / len(recent)
            recent_std = (sum((x - recent_avg) ** 2 for x in recent) / len(recent)) ** 0.5

            print(f"\nRecent stability (last 10 steps):")
            print(f"  Avg loss: {recent_avg:.4f}")
            print(f"  Std dev: {recent_std:.4f}")

            if recent_std < 0.01:
                print("  Status: ✓ Converged (stable)")
            elif last_loss > first_loss:
                print("  Status: ⚠ Loss increasing - may need lower learning rate")
            else:
                print("  Status: Training in progress")

    print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start local web server for dashboard.

    Automatically regenerates dashboard and viewer before serving to ensure
    the latest code and data are reflected.
    """
    from openadapt_ml.training.trainer import regenerate_local_dashboard

    port = args.port

    # Determine what to serve: benchmark directory or training output
    if hasattr(args, 'benchmark') and args.benchmark:
        serve_dir = Path(args.benchmark).expanduser().resolve()
        if not serve_dir.exists():
            print(f"Error: Benchmark directory not found: {serve_dir}")
            return 1

        # Regenerate benchmark viewer if needed
        if not args.no_regenerate:
            print("Regenerating benchmark viewer...")
            try:
                from openadapt_ml.training.benchmark_viewer import generate_benchmark_viewer
                generate_benchmark_viewer(serve_dir)
            except Exception as e:
                print(f"Warning: Could not regenerate benchmark viewer: {e}")

        start_page = "benchmark.html"
    else:
        serve_dir = get_current_output_dir()

        if not serve_dir.exists():
            print(f"Error: {serve_dir} not found. Run training first.")
            return 1

        # Regenerate dashboard and viewer with latest code before serving
        if not args.no_regenerate:
            print("Regenerating dashboard and viewer...")
            try:
                regenerate_local_dashboard(str(serve_dir))
                # Also regenerate viewer if comparison data exists
                _regenerate_viewer_if_possible(serve_dir)
            except Exception as e:
                print(f"Warning: Could not regenerate: {e}")

            # Also regenerate benchmark viewer from latest benchmark results
            _regenerate_benchmark_viewer_if_available(serve_dir)

        start_page = "dashboard.html"

    # Override start page if specified
    if hasattr(args, 'start_page') and args.start_page:
        start_page = args.start_page

    # Serve from the specified directory
    os.chdir(serve_dir)

    # Custom handler with /api/stop support
    quiet_mode = args.quiet

    class StopHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *log_args):
            if quiet_mode:
                pass  # Suppress request logging
            else:
                super().log_message(format, *log_args)

        def do_POST(self):
            if self.path == '/api/stop':
                # Create stop signal file
                stop_file = serve_dir / "STOP_TRAINING"
                stop_file.touch()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'{"status": "stop_signal_created"}')
                print(f"\n⏹ Stop signal created: {stop_file}")
            elif self.path == '/api/run-benchmark':
                # Parse request body for provider
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8') if content_length else '{}'
                try:
                    params = json.loads(body)
                except json.JSONDecodeError:
                    params = {}

                provider = params.get('provider', 'anthropic')
                tasks = params.get('tasks', 5)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "started", "provider": provider, "tasks": tasks}).encode())

                # Run benchmark in background thread with progress logging
                def run_benchmark():
                    import subprocess
                    from dotenv import load_dotenv

                    # Load .env file for API keys
                    project_root = Path(__file__).parent.parent.parent
                    load_dotenv(project_root / ".env")

                    # Create progress log file (in cwd which is serve_dir)
                    progress_file = Path("benchmark_progress.json")

                    print(f"\n🚀 Starting {provider} benchmark evaluation ({tasks} tasks)...")

                    # Write initial progress
                    progress_file.write_text(json.dumps({
                        "status": "running",
                        "provider": provider,
                        "tasks_total": tasks,
                        "tasks_complete": 0,
                        "message": f"Starting {provider} evaluation..."
                    }))

                    # Copy environment with loaded vars
                    env = os.environ.copy()

                    result = subprocess.run(
                        ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "run-api",
                         "--provider", provider, "--tasks", str(tasks),
                         "--model-id", f"{provider}-api"],
                        capture_output=True, text=True, cwd=str(project_root), env=env
                    )

                    print(f"\n📋 Benchmark output:\n{result.stdout}")
                    if result.stderr:
                        print(f"Stderr: {result.stderr}")

                    if result.returncode == 0:
                        print(f"✅ Benchmark complete. Regenerating viewer...")
                        progress_file.write_text(json.dumps({
                            "status": "complete",
                            "provider": provider,
                            "message": "Evaluation complete! Refreshing results..."
                        }))
                        # Regenerate benchmark viewer
                        _regenerate_benchmark_viewer_if_available(serve_dir)
                    else:
                        print(f"❌ Benchmark failed: {result.stderr}")
                        progress_file.write_text(json.dumps({
                            "status": "error",
                            "provider": provider,
                            "message": f"Evaluation failed: {result.stderr[:200]}"
                        }))

                threading.Thread(target=run_benchmark, daemon=True).start()
            elif self.path == '/api/vms/register':
                # Register a new VM
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8') if content_length else '{}'
                try:
                    vm_data = json.loads(body)
                    result = self._register_vm(vm_data)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_error(404, "Not found")

        def do_GET(self):
            if self.path.startswith('/api/benchmark-progress'):
                # Return benchmark progress
                progress_file = Path("benchmark_progress.json")  # Relative to serve_dir (cwd)
                if progress_file.exists():
                    progress = progress_file.read_text()
                else:
                    progress = json.dumps({"status": "idle"})

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(progress.encode())
            elif self.path.startswith('/api/benchmark-live'):
                # Return live evaluation state
                live_file = Path("benchmark_live.json")  # Relative to serve_dir (cwd)
                if live_file.exists():
                    live_state = live_file.read_text()
                else:
                    live_state = json.dumps({"status": "idle"})

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(live_state.encode())
            elif self.path.startswith('/api/tasks'):
                # Return background task status (VM, Docker, benchmarks)
                try:
                    tasks = self._fetch_background_tasks()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(tasks).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            elif self.path.startswith('/api/azure-jobs'):
                # Return LIVE Azure job status from Azure ML
                # Supports ?force=true parameter for manual refresh (always fetches live)
                try:
                    from urllib.parse import urlparse, parse_qs
                    query = parse_qs(urlparse(self.path).query)
                    force_refresh = query.get('force', ['false'])[0].lower() == 'true'

                    # Always fetch live data (force just indicates manual refresh for logging)
                    if force_refresh:
                        print("Azure Jobs: Manual refresh requested")

                    jobs = self._fetch_live_azure_jobs()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(jobs).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            elif self.path.startswith('/api/benchmark-sse'):
                # Server-Sent Events endpoint for real-time benchmark updates
                try:
                    from urllib.parse import urlparse, parse_qs
                    query = parse_qs(urlparse(self.path).query)
                    interval = int(query.get('interval', [5])[0])

                    # Validate interval (min 1s, max 60s)
                    interval = max(1, min(60, interval))

                    self._stream_benchmark_updates(interval)
                except Exception as e:
                    self.send_error(500, f"SSE error: {e}")
            elif self.path.startswith('/api/vms'):
                # Return VM registry with live status
                try:
                    vms = self._fetch_vm_registry()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(vms).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            elif self.path.startswith('/api/azure-job-logs'):
                # Return live logs for running Azure job
                try:
                    # Parse job_id from query string
                    from urllib.parse import urlparse, parse_qs
                    query = parse_qs(urlparse(self.path).query)
                    job_id = query.get('job_id', [None])[0]

                    logs = self._fetch_azure_job_logs(job_id)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(logs).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            elif self.path.startswith('/api/probe-vm'):
                # Probe the VM to check if WAA server is responding
                try:
                    result = self._probe_vm()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e), "responding": False}).encode())
            elif self.path.startswith('/api/current-run'):
                # Return currently running benchmark info
                try:
                    result = self._get_current_run()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e), "running": False}).encode())
            elif self.path.startswith('/api/background-tasks'):
                # Alias for /api/tasks - background task status
                try:
                    tasks = self._fetch_background_tasks()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(tasks).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                # Default file serving
                super().do_GET()

        def _fetch_live_azure_jobs(self):
            """Fetch live job status from Azure ML."""
            import subprocess
            result = subprocess.run(
                ["az", "ml", "job", "list",
                 "--resource-group", "openadapt-agents",
                 "--workspace-name", "openadapt-ml",
                 "--query", "[].{name:name,display_name:display_name,status:status,creation_context:creation_context.created_at}",
                 "-o", "json"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise Exception(f"Azure CLI error: {result.stderr}")

            jobs = json.loads(result.stdout)
            # Format for frontend
            experiment_id = "ad29082c-0607-4fda-8cc7-38944eb5a518"
            wsid = "/subscriptions/78add6c6-c92a-4a53-b751-eb644ac77e59/resourceGroups/openadapt-agents/providers/Microsoft.MachineLearningServices/workspaces/openadapt-ml"

            formatted = []
            for job in jobs[:10]:  # Limit to 10 most recent
                formatted.append({
                    "job_id": job.get("name", "unknown"),
                    "display_name": job.get("display_name", ""),
                    "status": job.get("status", "unknown").lower(),
                    "started_at": job.get("creation_context", ""),
                    "azure_dashboard_url": f"https://ml.azure.com/experiments/id/{experiment_id}/runs/{job.get('name', '')}?wsid={wsid}",
                    "is_live": True  # Flag to indicate this is live data
                })
            return formatted

        def _fetch_azure_job_logs(self, job_id: str | None):
            """Fetch logs for an Azure ML job (streaming for running jobs)."""
            import subprocess

            if not job_id:
                # Get the most recent running job
                jobs = self._fetch_live_azure_jobs()
                running = [j for j in jobs if j['status'] == 'running']
                if running:
                    job_id = running[0]['job_id']
                else:
                    return {"logs": "No running jobs found", "job_id": None, "status": "idle"}

            # Try to stream logs for running job using az ml job stream
            try:
                result = subprocess.run(
                    ["az", "ml", "job", "stream",
                     "--name", job_id,
                     "--resource-group", "openadapt-agents",
                     "--workspace-name", "openadapt-ml"],
                    capture_output=True, text=True, timeout=3  # Short timeout
                )
                if result.returncode == 0 and result.stdout.strip():
                    return {"logs": result.stdout[-5000:], "job_id": job_id, "status": "streaming"}
            except subprocess.TimeoutExpired:
                pass  # Fall through to job show

            # Get job details instead
            result = subprocess.run(
                ["az", "ml", "job", "show",
                 "--name", job_id,
                 "--resource-group", "openadapt-agents",
                 "--workspace-name", "openadapt-ml",
                 "-o", "json"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                job_info = json.loads(result.stdout)
                return {
                    "logs": f"Job {job_id} is {job_info.get('status', 'unknown')}\\n\\nCommand: {job_info.get('command', 'N/A')}",
                    "job_id": job_id,
                    "status": job_info.get('status', 'unknown').lower(),
                    "command": job_info.get('command', '')
                }

            return {"logs": f"Could not fetch logs: {result.stderr}", "job_id": job_id, "status": "error"}

        def _get_vm_detailed_metadata(self, vm_ip: str, container_name: str, logs: str, phase: str) -> dict:
            """Get detailed VM metadata for the VM Details panel.

            Returns:
                dict with disk_usage_gb, memory_usage_mb, setup_script_phase, probe_response, qmp_connected, dependencies
            """
            import subprocess
            import re

            metadata = {
                "disk_usage_gb": None,
                "memory_usage_mb": None,
                "setup_script_phase": None,
                "probe_response": None,
                "qmp_connected": False,
                "dependencies": []
            }

            # 1. Get disk usage from docker stats
            try:
                disk_result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-i", str(Path.home() / ".ssh" / "id_rsa"),
                     f"azureuser@{vm_ip}",
                     f"docker exec {container_name} df -h /storage 2>/dev/null | tail -1"],
                    capture_output=True, text=True, timeout=10
                )
                if disk_result.returncode == 0 and disk_result.stdout.strip():
                    # Parse: "Filesystem      Size  Used Avail Use% Mounted on"
                    # Example: "/dev/sda1        30G  9.2G   20G  31% /storage"
                    parts = disk_result.stdout.split()
                    if len(parts) >= 3:
                        used_str = parts[2]  # e.g., "9.2G"
                        total_str = parts[1]  # e.g., "30G"
                        # Convert to GB (handle M/G suffixes)
                        def to_gb(s):
                            if s.endswith('G'):
                                return float(s[:-1])
                            elif s.endswith('M'):
                                return float(s[:-1]) / 1024
                            elif s.endswith('K'):
                                return float(s[:-1]) / (1024 * 1024)
                            return 0
                        metadata["disk_usage_gb"] = f"{to_gb(used_str):.1f} GB / {to_gb(total_str):.0f} GB used"
            except Exception:
                pass

            # 2. Get memory usage from docker stats
            try:
                mem_result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-i", str(Path.home() / ".ssh" / "id_rsa"),
                     f"azureuser@{vm_ip}",
                     f"docker stats {container_name} --no-stream --format '{{{{.MemUsage}}}}'"],
                    capture_output=True, text=True, timeout=10
                )
                if mem_result.returncode == 0 and mem_result.stdout.strip():
                    # Example: "1.5GiB / 4GiB"
                    metadata["memory_usage_mb"] = mem_result.stdout.strip()
            except Exception:
                pass

            # 3. Parse setup script phase from logs
            metadata["setup_script_phase"] = self._parse_setup_phase_from_logs(logs, phase)

            # 4. Check /probe endpoint
            try:
                probe_result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-i", str(Path.home() / ".ssh" / "id_rsa"),
                     f"azureuser@{vm_ip}",
                     "curl -s --connect-timeout 2 http://20.20.20.21:5000/probe 2>/dev/null"],
                    capture_output=True, text=True, timeout=10
                )
                if probe_result.returncode == 0 and probe_result.stdout.strip():
                    metadata["probe_response"] = probe_result.stdout.strip()
                else:
                    metadata["probe_response"] = "Not responding"
            except Exception:
                metadata["probe_response"] = "Connection failed"

            # 5. Check QMP connection (port 7200)
            try:
                qmp_result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-i", str(Path.home() / ".ssh" / "id_rsa"),
                     f"azureuser@{vm_ip}",
                     "nc -z -w2 localhost 7200 2>&1"],
                    capture_output=True, text=True, timeout=10
                )
                metadata["qmp_connected"] = qmp_result.returncode == 0
            except Exception:
                pass

            # 6. Parse dependencies from logs
            metadata["dependencies"] = self._parse_dependencies_from_logs(logs, phase)

            return metadata

        def _parse_setup_phase_from_logs(self, logs: str, current_phase: str) -> str:
            """Parse the current setup script phase from logs.

            Looks for patterns indicating which script is running:
            - install.bat
            - setup.ps1
            - on-logon.ps1
            """
            if current_phase == "ready":
                return "Setup complete"
            elif current_phase == "oobe":
                # Check for specific script patterns
                if "on-logon.ps1" in logs.lower():
                    return "Running on-logon.ps1"
                elif "setup.ps1" in logs.lower():
                    return "Running setup.ps1"
                elif "install.bat" in logs.lower():
                    return "Running install.bat"
                else:
                    return "Windows installation in progress"
            elif current_phase == "booting":
                return "Booting Windows"
            elif current_phase in ["downloading", "extracting", "configuring", "building"]:
                return "Preparing Windows VM"
            else:
                return "Initializing..."

        def _parse_dependencies_from_logs(self, logs: str, phase: str) -> list[dict]:
            """Parse dependency installation status from logs.

            Returns list of dependencies with their installation status:
            - Python
            - Chrome
            - LibreOffice
            - VSCode
            - etc.
            """
            dependencies = [
                {"name": "Python", "icon": "🐍", "status": "pending"},
                {"name": "Chrome", "icon": "🌐", "status": "pending"},
                {"name": "LibreOffice", "icon": "📝", "status": "pending"},
                {"name": "VSCode", "icon": "💻", "status": "pending"},
                {"name": "WAA Server", "icon": "🔧", "status": "pending"},
            ]

            if phase not in ["oobe", "ready"]:
                # Not yet at Windows setup phase
                return dependencies

            logs_lower = logs.lower()

            # Check for installation patterns
            if "python" in logs_lower and ("installing python" in logs_lower or "python.exe" in logs_lower):
                dependencies[0]["status"] = "installing"
            elif "python" in logs_lower and "installed" in logs_lower:
                dependencies[0]["status"] = "complete"

            if "chrome" in logs_lower and ("downloading" in logs_lower or "installing" in logs_lower):
                dependencies[1]["status"] = "installing"
            elif "chrome" in logs_lower and "installed" in logs_lower:
                dependencies[1]["status"] = "complete"

            if "libreoffice" in logs_lower and ("downloading" in logs_lower or "installing" in logs_lower):
                dependencies[2]["status"] = "installing"
            elif "libreoffice" in logs_lower and "installed" in logs_lower:
                dependencies[2]["status"] = "complete"

            if "vscode" in logs_lower or "visual studio code" in logs_lower:
                if "installing" in logs_lower:
                    dependencies[3]["status"] = "installing"
                elif "installed" in logs_lower:
                    dependencies[3]["status"] = "complete"

            if "waa" in logs_lower or "flask" in logs_lower:
                if "starting" in logs_lower or "running" in logs_lower:
                    dependencies[4]["status"] = "installing"
                elif phase == "ready":
                    dependencies[4]["status"] = "complete"

            return dependencies

        def _fetch_background_tasks(self):
            """Fetch status of all background tasks: Azure VM, Docker containers, benchmarks."""
            import subprocess
            from datetime import datetime
            import time

            tasks = []

            # Check for VM IP from environment (set by CLI when auto-launching viewer)
            env_vm_ip = os.environ.get("WAA_VM_IP")
            env_internal_ip = os.environ.get("WAA_INTERNAL_IP", "172.30.0.2")

            # 1. Check Azure WAA VM status
            vm_ip = None
            if env_vm_ip:
                # Use environment variable - VM IP was provided directly
                vm_ip = env_vm_ip
                tasks.append({
                    "task_id": "azure-vm-waa",
                    "task_type": "vm_provision",
                    "status": "completed",
                    "title": "Azure VM Host",
                    "description": f"Linux host running at {vm_ip}",
                    "progress_percent": 100.0,
                    "elapsed_seconds": 0,
                    "metadata": {
                        "vm_name": "waa-eval-vm",
                        "ip_address": vm_ip,
                        "internal_ip": env_internal_ip
                    }
                })
            else:
                # Query Azure CLI for VM status
                try:
                    result = subprocess.run(
                        ["az", "vm", "get-instance-view",
                         "--name", "waa-eval-vm",
                         "--resource-group", "openadapt-agents",
                         "--query", "instanceView.statuses",
                         "-o", "json"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        statuses = json.loads(result.stdout)
                        power_state = "unknown"
                        for s in statuses:
                            if s.get("code", "").startswith("PowerState/"):
                                power_state = s["code"].replace("PowerState/", "")

                        # Get VM IP
                        ip_result = subprocess.run(
                            ["az", "vm", "list-ip-addresses",
                             "--name", "waa-eval-vm",
                             "--resource-group", "openadapt-agents",
                             "--query", "[0].virtualMachine.network.publicIpAddresses[0].ipAddress",
                             "-o", "tsv"],
                            capture_output=True, text=True, timeout=10
                        )
                        vm_ip = ip_result.stdout.strip() if ip_result.returncode == 0 else None

                        if power_state == "running":
                            tasks.append({
                                "task_id": "azure-vm-waa",
                                "task_type": "vm_provision",
                                "status": "completed",
                                "title": "Azure VM Host",
                                "description": f"Linux host running at {vm_ip}" if vm_ip else "Linux host running",
                                "progress_percent": 100.0,
                                "elapsed_seconds": 0,
                                "metadata": {
                                    "vm_name": "waa-eval-vm",
                                    "ip_address": vm_ip
                                    # No VNC link - that's for the Windows container
                                }
                            })
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass

            # 2. Check Docker container status on VM (if we have an IP)
            if vm_ip:
                try:
                    docker_result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                         "-i", str(Path.home() / ".ssh" / "id_rsa"),
                         f"azureuser@{vm_ip}",
                         "docker ps --format '{{.Names}}|{{.Status}}|{{.Image}}'"],
                        capture_output=True, text=True, timeout=15
                    )
                    if docker_result.returncode == 0 and docker_result.stdout.strip():
                        for line in docker_result.stdout.strip().split('\n'):
                            parts = line.split('|')
                            if len(parts) >= 3:
                                container_name, status, image = parts[0], parts[1], parts[2]
                                # Parse "Up X minutes" to determine if healthy
                                is_healthy = "Up" in status

                                # Check for Windows VM specifically
                                if "windows" in image.lower() or container_name == "winarena":
                                    # Get detailed progress from docker logs
                                    log_check = subprocess.run(
                                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                                         "-i", str(Path.home() / ".ssh" / "id_rsa"),
                                         f"azureuser@{vm_ip}",
                                         f"docker logs {container_name} 2>&1 | tail -30"],
                                        capture_output=True, text=True, timeout=10
                                    )
                                    logs = log_check.stdout if log_check.returncode == 0 else ""

                                    # Parse progress from logs
                                    phase = "unknown"
                                    progress = 0.0
                                    description = "Starting..."

                                    # Use internal IP from environment if available
                                    probe_ip = env_internal_ip if env_vm_ip else "20.20.20.21"

                                    if "Windows started successfully" in logs:
                                        # Check if WAA server is ready
                                        server_check = subprocess.run(
                                            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                                             "-i", str(Path.home() / ".ssh" / "id_rsa"),
                                             f"azureuser@{vm_ip}",
                                             f"curl -s --connect-timeout 2 http://{probe_ip}:5000/probe 2>/dev/null"],
                                            capture_output=True, text=True, timeout=10
                                        )
                                        if server_check.returncode == 0 and server_check.stdout.strip():
                                            phase = "ready"
                                            progress = 100.0
                                            description = "WAA Server ready - benchmarks can run"
                                        else:
                                            phase = "oobe"
                                            progress = 80.0  # Phase 5/6 - VM install in progress
                                            description = "Phase 5/6: Windows installing (check VNC for %). OEM scripts will run after."
                                    elif "Booting Windows" in logs:
                                        phase = "booting"
                                        progress = 70.0  # Phase 4/6
                                        description = "Phase 4/6: Booting Windows from installer..."
                                    elif "Building Windows" in logs or "Creating a" in logs:
                                        phase = "building"
                                        progress = 60.0  # Phase 3/6
                                        description = "Phase 3/6: Building Windows VM disk..."
                                    elif "Adding" in logs and "image" in logs:
                                        phase = "configuring"
                                        progress = 50.0  # Phase 2/6
                                        description = "Phase 2/6: Configuring Windows image with WAA scripts..."
                                    elif "Extracting" in logs:
                                        phase = "extracting"
                                        progress = 35.0  # Phase 1/6 (after download)
                                        description = "Phase 1/6: Extracting Windows ISO..."
                                    else:
                                        # Check for download progress (e.g., "1234K ........ 45% 80M 30s")
                                        import re
                                        download_match = re.search(r'(\d+)%\s+[\d.]+[KMG]\s+(\d+)s', logs)
                                        if download_match:
                                            phase = "downloading"
                                            dl_pct = float(download_match.group(1))
                                            progress = dl_pct * 0.30  # 0-30% for download phase
                                            eta = download_match.group(2)
                                            description = f"Phase 0/6: Downloading Windows 11... {download_match.group(1)}% ({eta}s left)"

                                    # Improve phase detection - if Windows is booted but WAA not ready,
                                    # it might be at login screen waiting for OEM scripts
                                    if phase == "oobe" and "Boot0004" in logs:
                                        # Windows finished installing, at login/desktop
                                        description = "Phase 5/6: Windows ready. Login via VNC, then run: \\\\host.lan\\Data\\install.bat"
                                        progress = 85.0

                                    # Get detailed metadata for VM Details panel
                                    vm_metadata = self._get_vm_detailed_metadata(vm_ip, container_name, logs, phase)

                                    tasks.append({
                                        "task_id": f"docker-{container_name}",
                                        "task_type": "docker_container",
                                        "status": "completed" if phase == "ready" else "running",
                                        "title": "Windows 11 + WAA Server",
                                        "description": description,
                                        "progress_percent": progress,
                                        "elapsed_seconds": 0,
                                        "phase": phase,
                                        "metadata": {
                                            "container": container_name,
                                            "image": image,
                                            "status": status,
                                            "phase": phase,
                                            "windows_ready": phase in ["oobe", "ready"],
                                            "waa_server_ready": phase == "ready",
                                            "vnc_url": f"http://{vm_ip}:8006",
                                            "windows_username": "Docker",
                                            "windows_password": "(empty - just press Enter)",
                                            "recent_logs": logs[-500:] if logs else "",
                                            # Enhanced VM details
                                            "disk_usage_gb": vm_metadata["disk_usage_gb"],
                                            "memory_usage_mb": vm_metadata["memory_usage_mb"],
                                            "setup_script_phase": vm_metadata["setup_script_phase"],
                                            "probe_response": vm_metadata["probe_response"],
                                            "qmp_connected": vm_metadata["qmp_connected"],
                                            "dependencies": vm_metadata["dependencies"],
                                        }
                                    })
                except Exception as e:
                    # SSH failed, VM might still be starting
                    pass

            # 3. Check local benchmark progress
            progress_file = Path("benchmark_progress.json")
            if progress_file.exists():
                try:
                    progress = json.loads(progress_file.read_text())
                    if progress.get("status") == "running":
                        tasks.append({
                            "task_id": "benchmark-local",
                            "task_type": "benchmark_run",
                            "status": "running",
                            "title": f"{progress.get('provider', 'API').upper()} Benchmark",
                            "description": progress.get("message", "Running benchmark..."),
                            "progress_percent": (progress.get("tasks_complete", 0) / max(progress.get("tasks_total", 1), 1)) * 100,
                            "elapsed_seconds": 0,
                            "metadata": progress
                        })
                except Exception:
                    pass

            return tasks

        def _fetch_vm_registry(self):
            """Fetch VM registry with live status checks."""
            import subprocess
            from datetime import datetime

            # Path to VM registry file (relative to project root)
            project_root = Path(__file__).parent.parent.parent
            registry_file = project_root / "benchmark_results" / "vm_registry.json"

            if not registry_file.exists():
                return []

            try:
                with open(registry_file) as f:
                    vms = json.load(f)
            except Exception as e:
                return {"error": f"Failed to read VM registry: {e}"}

            # Check status for each VM
            for vm in vms:
                vm["status"] = "unknown"
                vm["last_checked"] = datetime.now().isoformat()
                vm["vnc_reachable"] = False
                vm["waa_probe_status"] = "unknown"

                # Check VNC (HTTP HEAD request)
                try:
                    vnc_url = f"http://{vm['ssh_host']}:{vm['vnc_port']}"
                    result = subprocess.run(
                        ["curl", "-I", "-s", "--connect-timeout", "3", vnc_url],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and "200" in result.stdout:
                        vm["vnc_reachable"] = True
                except Exception:
                    pass

                # Check WAA probe via SSH
                try:
                    internal_ip = vm.get("internal_ip", "20.20.20.21")
                    waa_port = vm.get("waa_port", 5000)
                    ssh_cmd = f"curl -s --connect-timeout 2 http://{internal_ip}:{waa_port}/probe 2>/dev/null"

                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=3",
                         "-i", str(Path.home() / ".ssh" / "id_rsa"),
                         f"{vm['ssh_user']}@{vm['ssh_host']}",
                         ssh_cmd],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        vm["waa_probe_status"] = "ready"
                        vm["status"] = "online"
                    else:
                        vm["waa_probe_status"] = "not responding"
                        vm["status"] = "offline"
                except Exception:
                    vm["waa_probe_status"] = "ssh failed"
                    vm["status"] = "offline"

            return vms

        def _probe_vm(self) -> dict:
            """Probe the Azure VM to check if WAA server is responding.

            Returns:
                dict with:
                - responding: bool - whether the WAA server is responding
                - vm_ip: str - the VM's IP address
                - container: str - the container name
                - probe_result: str - the raw probe response or error message
                - last_checked: str - ISO timestamp
            """
            import subprocess
            from datetime import datetime

            result = {
                "responding": False,
                "vm_ip": None,
                "container": None,
                "probe_result": None,
                "last_checked": datetime.now().isoformat(),
            }

            # First get VM IP
            try:
                ip_result = subprocess.run(
                    ["az", "vm", "list-ip-addresses",
                     "--name", "waa-eval-vm",
                     "--resource-group", "openadapt-agents",
                     "--query", "[0].virtualMachine.network.publicIpAddresses[0].ipAddress",
                     "-o", "tsv"],
                    capture_output=True, text=True, timeout=10
                )
                if ip_result.returncode == 0 and ip_result.stdout.strip():
                    vm_ip = ip_result.stdout.strip()
                    result["vm_ip"] = vm_ip

                    # Try to probe WAA server via SSH
                    # Use the correct internal IP for the Windows VM inside Docker
                    probe_result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                         "-i", str(Path.home() / ".ssh" / "id_rsa"),
                         f"azureuser@{vm_ip}",
                         "docker exec waa-container curl -s --connect-timeout 3 http://172.30.0.2:5000/probe 2>/dev/null || echo 'probe_failed'"],
                        capture_output=True, text=True, timeout=15
                    )

                    result["container"] = "waa-container"

                    if probe_result.returncode == 0:
                        probe_output = probe_result.stdout.strip()
                        if probe_output and "probe_failed" not in probe_output:
                            result["responding"] = True
                            result["probe_result"] = probe_output
                        else:
                            result["probe_result"] = "WAA server not responding"
                    else:
                        result["probe_result"] = f"SSH/Docker error: {probe_result.stderr[:200]}"
                else:
                    result["probe_result"] = "Could not get VM IP"
            except subprocess.TimeoutExpired:
                result["probe_result"] = "Connection timeout"
            except Exception as e:
                result["probe_result"] = f"Error: {str(e)}"

            return result

        def _get_current_run(self) -> dict:
            """Get info about any currently running benchmark.

            Checks:
            1. Local benchmark_progress.json for API benchmarks
            2. Azure VM for WAA benchmarks running via SSH

            Returns:
                dict with:
                - running: bool - whether a benchmark is running
                - type: str - 'local' or 'azure_vm'
                - model: str - model being evaluated
                - progress: dict with tasks_completed, total_tasks
                - current_task: str - current task ID
                - started_at: str - ISO timestamp
                - elapsed_minutes: int
            """
            import subprocess
            from datetime import datetime
            import re

            result = {
                "running": False,
                "type": None,
                "model": None,
                "progress": {"tasks_completed": 0, "total_tasks": 0},
                "current_task": None,
                "started_at": None,
                "elapsed_minutes": 0,
            }

            # Check local benchmark progress first
            progress_file = Path("benchmark_progress.json")
            if progress_file.exists():
                try:
                    progress = json.loads(progress_file.read_text())
                    if progress.get("status") == "running":
                        result["running"] = True
                        result["type"] = "local"
                        result["model"] = progress.get("provider", "unknown")
                        result["progress"]["tasks_completed"] = progress.get("tasks_complete", 0)
                        result["progress"]["total_tasks"] = progress.get("tasks_total", 0)
                        return result
                except Exception:
                    pass

            # Check Azure VM for running benchmark
            try:
                # Get VM IP
                ip_result = subprocess.run(
                    ["az", "vm", "list-ip-addresses",
                     "--name", "waa-eval-vm",
                     "--resource-group", "openadapt-agents",
                     "--query", "[0].virtualMachine.network.publicIpAddresses[0].ipAddress",
                     "-o", "tsv"],
                    capture_output=True, text=True, timeout=10
                )

                if ip_result.returncode == 0 and ip_result.stdout.strip():
                    vm_ip = ip_result.stdout.strip()

                    # Check if benchmark process is running
                    process_check = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                         "-i", str(Path.home() / ".ssh" / "id_rsa"),
                         f"azureuser@{vm_ip}",
                         "docker exec waa-container pgrep -f 'python.*run.py' 2>/dev/null && echo 'RUNNING' || echo 'NOT_RUNNING'"],
                        capture_output=True, text=True, timeout=10
                    )

                    if process_check.returncode == 0 and "RUNNING" in process_check.stdout:
                        result["running"] = True
                        result["type"] = "azure_vm"

                        # Get log file for more details
                        log_check = subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                             "-i", str(Path.home() / ".ssh" / "id_rsa"),
                             f"azureuser@{vm_ip}",
                             "tail -100 /tmp/waa_benchmark.log 2>/dev/null || echo ''"],
                            capture_output=True, text=True, timeout=10
                        )

                        if log_check.returncode == 0 and log_check.stdout.strip():
                            logs = log_check.stdout

                            # Parse model from logs
                            model_match = re.search(r'model[=:\s]+([^\s,]+)', logs, re.IGNORECASE)
                            if model_match:
                                result["model"] = model_match.group(1)

                            # Parse progress
                            task_match = re.search(r'Task\s+(\d+)/(\d+)', logs)
                            if task_match:
                                result["progress"]["tasks_completed"] = int(task_match.group(1))
                                result["progress"]["total_tasks"] = int(task_match.group(2))

                            # Parse current task
                            task_id_match = re.search(r'(?:Running|Processing|task)[:\s]+([a-f0-9-]+)', logs, re.IGNORECASE)
                            if task_id_match:
                                result["current_task"] = task_id_match.group(1)

            except Exception:
                pass

            return result

        async def _detect_running_benchmark(self, vm_ip: str, container_name: str = "winarena") -> dict:
            """Detect if a benchmark is running on the VM and extract progress.

            SSH into VM and check:
            1. Process running: docker exec {container} pgrep -f 'python.*run.py'
            2. Log progress: tail /tmp/waa_benchmark.log

            Returns:
                dict with:
                - running: bool
                - current_task: str (task ID or description)
                - progress: dict with tasks_completed, total_tasks, current_step
                - recent_logs: str (last few log lines)
            """
            import subprocess
            import re

            result = {
                "running": False,
                "current_task": None,
                "progress": {
                    "tasks_completed": 0,
                    "total_tasks": 0,
                    "current_step": 0,
                },
                "recent_logs": "",
            }

            try:
                # Check if benchmark process is running
                process_check = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-i", str(Path.home() / ".ssh" / "id_rsa"),
                     f"azureuser@{vm_ip}",
                     f"docker exec {container_name} pgrep -f 'python.*run.py' 2>/dev/null || echo ''"],
                    capture_output=True, text=True, timeout=10
                )

                if process_check.returncode == 0 and process_check.stdout.strip():
                    result["running"] = True

                    # Get benchmark log
                    log_check = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                         "-i", str(Path.home() / ".ssh" / "id_rsa"),
                         f"azureuser@{vm_ip}",
                         "tail -100 /tmp/waa_benchmark.log 2>/dev/null || echo ''"],
                        capture_output=True, text=True, timeout=10
                    )

                    if log_check.returncode == 0 and log_check.stdout.strip():
                        logs = log_check.stdout
                        result["recent_logs"] = logs[-500:]  # Last 500 chars

                        # Parse progress from logs
                        # Look for patterns like "Task 5/30" or "Completed: 5, Remaining: 25"
                        task_match = re.search(r'Task\s+(\d+)/(\d+)', logs)
                        if task_match:
                            result["progress"]["tasks_completed"] = int(task_match.group(1))
                            result["progress"]["total_tasks"] = int(task_match.group(2))

                        # Extract current task ID
                        task_id_match = re.search(r'(?:Running|Processing) task:\s*(\S+)', logs)
                        if task_id_match:
                            result["current_task"] = task_id_match.group(1)

                        # Extract step info
                        step_match = re.search(r'Step\s+(\d+)', logs)
                        if step_match:
                            result["progress"]["current_step"] = int(step_match.group(1))

            except Exception as e:
                # SSH or parsing failed - leave defaults
                pass

            return result

        def _parse_task_result(self, log_lines: list[str], task_id: str) -> dict:
            """Parse task success/failure from log output.

            WAA log patterns:
            - Success: "Task task_001 completed successfully"
            - Success: "Result: PASS"
            - Failure: "Task task_001 failed"
            - Failure: "Result: FAIL"
            - Score: "Score: 0.85"
            """
            import re

            success = None
            score = None

            # Search backwards from most recent
            for line in reversed(log_lines):
                # Check for explicit result
                if 'Result: PASS' in line or 'completed successfully' in line:
                    success = True
                elif 'Result: FAIL' in line or 'failed' in line.lower():
                    success = False

                # Check for score
                score_match = re.search(r'Score:\s*([\d.]+)', line)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                    except ValueError:
                        pass

                # Check for task-specific completion
                if task_id in line:
                    if 'success' in line.lower() or 'pass' in line.lower():
                        success = True
                    elif 'fail' in line.lower() or 'error' in line.lower():
                        success = False

            # Default to True if no explicit failure found (backwards compatible)
            if success is None:
                success = True

            return {"success": success, "score": score}

        def _stream_benchmark_updates(self, interval: int):
            """Stream Server-Sent Events for benchmark status updates.

            Streams events:
            - connected: Initial connection event
            - status: VM status and probe results
            - progress: Benchmark progress (tasks completed, current task)
            - task_complete: When a task finishes
            - heartbeat: Keep-alive signal every 30 seconds
            - error: Error messages

            Uses a generator-based approach to avoid blocking the main thread
            and properly handles client disconnection.
            """
            import time
            import select

            HEARTBEAT_INTERVAL = 30  # seconds

            # Set SSE headers
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Connection', 'keep-alive')
            self.send_header('X-Accel-Buffering', 'no')  # Disable nginx buffering
            self.end_headers()

            # Track connection state
            client_connected = True

            def send_event(event_type: str, data: dict) -> bool:
                """Send an SSE event. Returns False if client disconnected."""
                nonlocal client_connected
                if not client_connected:
                    return False
                try:
                    event_str = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                    self.wfile.write(event_str.encode('utf-8'))
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    # Client disconnected
                    client_connected = False
                    return False
                except Exception as e:
                    # Other error - log and assume disconnected
                    print(f"SSE send error: {e}")
                    client_connected = False
                    return False

            def check_client_connected() -> bool:
                """Check if client is still connected using socket select."""
                nonlocal client_connected
                if not client_connected:
                    return False
                try:
                    # Check if socket has data (would indicate client sent something or closed)
                    # Use non-blocking check with 0 timeout
                    rlist, _, xlist = select.select([self.rfile], [], [self.rfile], 0)
                    if xlist:
                        # Error condition on socket
                        client_connected = False
                        return False
                    if rlist:
                        # Client sent data - for SSE this usually means disconnect
                        # (SSE is server-push only, client doesn't send data)
                        data = self.rfile.read(1)
                        if not data:
                            client_connected = False
                            return False
                    return True
                except Exception:
                    client_connected = False
                    return False

            last_task = None
            last_heartbeat = time.time()
            recent_log_lines = []

            # Send initial connected event
            if not send_event("connected", {
                "timestamp": time.time(),
                "interval": interval,
                "version": "1.0"
            }):
                return

            try:
                iteration_count = 0
                max_iterations = 3600 // interval  # Max 1 hour of streaming

                while client_connected and iteration_count < max_iterations:
                    iteration_count += 1
                    current_time = time.time()

                    # Check client connection before doing work
                    if not check_client_connected():
                        break

                    # Send heartbeat every 30 seconds to prevent proxy/LB timeouts
                    if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
                        if not send_event("heartbeat", {"timestamp": current_time}):
                            break
                        last_heartbeat = current_time

                    # Fetch background tasks (includes VM status)
                    tasks = self._fetch_background_tasks()

                    # Send VM status event
                    vm_task = next((t for t in tasks if t.get("task_type") == "docker_container"), None)
                    if vm_task:
                        vm_data = {
                            "type": "vm_status",
                            "connected": vm_task.get("status") in ["running", "completed"],
                            "phase": vm_task.get("phase", "unknown"),
                            "waa_ready": vm_task.get("metadata", {}).get("waa_server_ready", False),
                            "probe": {
                                "status": vm_task.get("metadata", {}).get("probe_response", "unknown"),
                                "vnc_url": vm_task.get("metadata", {}).get("vnc_url"),
                            }
                        }

                        if not send_event("status", vm_data):
                            break

                        # If VM is ready, check for running benchmark
                        if vm_data["waa_ready"]:
                            # Get VM IP from tasks
                            vm_ip = None
                            azure_vm = next((t for t in tasks if t.get("task_type") == "vm_provision"), None)
                            if azure_vm:
                                vm_ip = azure_vm.get("metadata", {}).get("ip_address")

                            if vm_ip:
                                # Detect running benchmark using sync version
                                benchmark_status = self._detect_running_benchmark_sync(
                                    vm_ip, vm_task.get("metadata", {}).get("container", "winarena")
                                )

                                if benchmark_status["running"]:
                                    # Store log lines for result parsing
                                    if benchmark_status.get("recent_logs"):
                                        recent_log_lines = benchmark_status["recent_logs"].split('\n')

                                    # Send progress event
                                    progress_data = {
                                        "tasks_completed": benchmark_status["progress"]["tasks_completed"],
                                        "total_tasks": benchmark_status["progress"]["total_tasks"],
                                        "current_task": benchmark_status["current_task"],
                                        "current_step": benchmark_status["progress"]["current_step"],
                                    }

                                    if not send_event("progress", progress_data):
                                        break

                                    # Check if task completed
                                    current_task = benchmark_status["current_task"]
                                    if current_task and current_task != last_task:
                                        if last_task is not None:
                                            # Previous task completed - parse result from logs
                                            result = self._parse_task_result(recent_log_lines, last_task)
                                            complete_data = {
                                                "task_id": last_task,
                                                "success": result["success"],
                                                "score": result["score"],
                                            }
                                            if not send_event("task_complete", complete_data):
                                                break

                                        last_task = current_task

                    # Check local benchmark progress file
                    progress_file = Path("benchmark_progress.json")
                    if progress_file.exists():
                        try:
                            progress = json.loads(progress_file.read_text())
                            if progress.get("status") == "running":
                                progress_data = {
                                    "tasks_completed": progress.get("tasks_complete", 0),
                                    "total_tasks": progress.get("tasks_total", 0),
                                    "current_task": progress.get("provider", "unknown"),
                                }
                                if not send_event("progress", progress_data):
                                    break
                        except Exception:
                            pass

                    # Non-blocking sleep using select with timeout
                    # This allows checking for client disconnect during sleep
                    try:
                        select.select([self.rfile], [], [], interval)
                    except Exception:
                        break

            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Client disconnected - this is normal, don't log as error
                pass
            except Exception as e:
                # Send error event if still connected
                send_event("error", {"message": str(e)})
            finally:
                # Cleanup - connection is ending
                client_connected = False

        def _detect_running_benchmark_sync(self, vm_ip: str, container_name: str = "winarena") -> dict:
            """Synchronous version of _detect_running_benchmark.

            Avoids creating a new event loop on each call which causes issues
            when called from a synchronous context.
            """
            import subprocess
            import re

            result = {
                "running": False,
                "current_task": None,
                "progress": {
                    "tasks_completed": 0,
                    "total_tasks": 0,
                    "current_step": 0,
                },
                "recent_logs": "",
            }

            try:
                # Check if benchmark process is running
                process_check = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-i", str(Path.home() / ".ssh" / "id_rsa"),
                     f"azureuser@{vm_ip}",
                     f"docker exec {container_name} pgrep -f 'python.*run.py' 2>/dev/null || echo ''"],
                    capture_output=True, text=True, timeout=10
                )

                if process_check.returncode == 0 and process_check.stdout.strip():
                    result["running"] = True

                    # Get benchmark log
                    log_check = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                         "-i", str(Path.home() / ".ssh" / "id_rsa"),
                         f"azureuser@{vm_ip}",
                         "tail -100 /tmp/waa_benchmark.log 2>/dev/null || echo ''"],
                        capture_output=True, text=True, timeout=10
                    )

                    if log_check.returncode == 0 and log_check.stdout.strip():
                        logs = log_check.stdout
                        result["recent_logs"] = logs[-500:]  # Last 500 chars

                        # Parse progress from logs
                        task_match = re.search(r'Task\s+(\d+)/(\d+)', logs)
                        if task_match:
                            result["progress"]["tasks_completed"] = int(task_match.group(1))
                            result["progress"]["total_tasks"] = int(task_match.group(2))

                        # Extract current task ID
                        task_id_match = re.search(r'(?:Running|Processing) task:\s*(\S+)', logs)
                        if task_id_match:
                            result["current_task"] = task_id_match.group(1)

                        # Extract step info
                        step_match = re.search(r'Step\s+(\d+)', logs)
                        if step_match:
                            result["progress"]["current_step"] = int(step_match.group(1))

            except Exception:
                # SSH or parsing failed - leave defaults
                pass

            return result

        def _register_vm(self, vm_data):
            """Register a new VM in the registry."""
            # Path to VM registry file (relative to project root)
            project_root = Path(__file__).parent.parent.parent
            registry_file = project_root / "benchmark_results" / "vm_registry.json"

            # Load existing registry
            vms = []
            if registry_file.exists():
                try:
                    with open(registry_file) as f:
                        vms = json.load(f)
                except Exception:
                    pass

            # Add new VM
            new_vm = {
                "name": vm_data.get("name", "unnamed-vm"),
                "ssh_host": vm_data.get("ssh_host", ""),
                "ssh_user": vm_data.get("ssh_user", "azureuser"),
                "vnc_port": vm_data.get("vnc_port", 8006),
                "waa_port": vm_data.get("waa_port", 5000),
                "docker_container": vm_data.get("docker_container", "win11-waa"),
                "internal_ip": vm_data.get("internal_ip", "20.20.20.21")
            }

            vms.append(new_vm)

            # Save registry
            try:
                registry_file.parent.mkdir(parents=True, exist_ok=True)
                with open(registry_file, 'w') as f:
                    json.dump(vms, f, indent=2)
                return {"status": "success", "vm": new_vm}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        def do_OPTIONS(self):
            # Handle CORS preflight
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True  # Don't block shutdown

    with ThreadedTCPServer(("", port), StopHandler) as httpd:
        url = f"http://localhost:{port}/{start_page}"
        print(f"\nServing at: {url}")
        print(f"Directory: {serve_dir}")
        print("Press Ctrl+C to stop\n")

        if args.open:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

    return 0


def cmd_viewer(args: argparse.Namespace) -> int:
    """Regenerate viewer from local training output."""
    from openadapt_ml.training.trainer import (
        generate_training_dashboard,
        generate_unified_viewer_from_output_dir,
        TrainingState,
        TrainingConfig,
    )

    current_dir = get_current_output_dir()

    if not current_dir.exists():
        print(f"Error: {current_dir} not found. Run training first.")
        return 1

    print(f"Regenerating viewer from {current_dir}...")

    # Regenerate dashboard
    log_file = current_dir / "training_log.json"
    if log_file.exists():
        with open(log_file) as f:
            data = json.load(f)

        state = TrainingState(job_id=data.get("job_id", ""))
        state.epoch = data.get("epoch", 0)
        state.step = data.get("step", 0)
        state.loss = data.get("loss", 0)
        state.learning_rate = data.get("learning_rate", 0)
        state.losses = data.get("losses", [])
        state.status = data.get("status", "completed")
        state.elapsed_time = data.get("elapsed_time", 0.0)  # Load elapsed time for completed training
        state.goal = data.get("goal", "")
        state.config_path = data.get("config_path", "")
        state.capture_path = data.get("capture_path", "")

        # Load model config from training_log.json or fall back to reading config file
        state.model_name = data.get("model_name", "")
        state.lora_r = data.get("lora_r", 0)
        state.lora_alpha = data.get("lora_alpha", 0)
        state.load_in_4bit = data.get("load_in_4bit", False)

        # If model config not in JSON, try to read from config file
        if not state.model_name and state.config_path:
            try:
                import yaml
                # Try relative to project root first, then as absolute path
                project_root = Path(__file__).parent.parent.parent
                config_file = project_root / state.config_path
                if not config_file.exists():
                    config_file = Path(state.config_path)
                if config_file.exists():
                    with open(config_file) as cf:
                        cfg = yaml.safe_load(cf)
                    if cfg and "model" in cfg:
                        state.model_name = cfg["model"].get("name", "")
                        state.load_in_4bit = cfg["model"].get("load_in_4bit", False)
                    if cfg and "lora" in cfg:
                        state.lora_r = cfg["lora"].get("r", 0)
                        state.lora_alpha = cfg["lora"].get("lora_alpha", 0)
            except Exception as e:
                print(f"  Warning: Could not read config file: {e}")

        config = TrainingConfig(
            num_train_epochs=data.get("total_epochs", 5),
            learning_rate=data.get("learning_rate", 5e-5),
        )

        dashboard_html = generate_training_dashboard(state, config)
        (current_dir / "dashboard.html").write_text(dashboard_html)
        print(f"  Regenerated: dashboard.html")

    # Generate unified viewer using consolidated function
    viewer_path = generate_unified_viewer_from_output_dir(current_dir)
    if viewer_path:
        print(f"\nGenerated: {viewer_path}")
    else:
        print("\nNo comparison data found. Run comparison first or copy from capture directory.")

    # Also regenerate benchmark viewer from latest benchmark results
    _regenerate_benchmark_viewer_if_available(current_dir)

    if args.open:
        webbrowser.open(str(current_dir / "viewer.html"))

    return 0


def cmd_benchmark_viewer(args: argparse.Namespace) -> int:
    """Generate benchmark viewer from benchmark results."""
    from openadapt_ml.training.benchmark_viewer import generate_benchmark_viewer

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        return 1

    print(f"\n{'='*50}")
    print("GENERATING BENCHMARK VIEWER")
    print(f"{'='*50}")
    print(f"Benchmark dir: {benchmark_dir}")
    print()

    try:
        viewer_path = generate_benchmark_viewer(benchmark_dir)
        print(f"\nSuccess! Benchmark viewer generated at: {viewer_path}")

        if args.open:
            webbrowser.open(str(viewer_path))

        return 0
    except Exception as e:
        print(f"Error generating benchmark viewer: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Run human vs AI comparison on local checkpoint."""
    capture_path = Path(args.capture).expanduser().resolve()
    if not capture_path.exists():
        print(f"Error: Capture not found: {capture_path}")
        return 1

    checkpoint = args.checkpoint
    if checkpoint and not Path(checkpoint).exists():
        print(f"Error: Checkpoint not found: {checkpoint}")
        return 1

    print(f"\n{'='*50}")
    print("RUNNING COMPARISON")
    print(f"{'='*50}")
    print(f"Capture: {capture_path}")
    print(f"Checkpoint: {checkpoint or 'None (capture only)'}")
    print()

    cmd = [
        sys.executable, "-m", "openadapt_ml.scripts.compare",
        "--capture", str(capture_path),
    ]

    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])

    if args.open:
        cmd.append("--open")

    result = subprocess.run(cmd, check=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Local GPU training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on a capture (auto-detects CUDA/MPS/CPU)
  uv run python -m openadapt_ml.cloud.local train --capture ~/captures/my-workflow --open

  # Check training status
  uv run python -m openadapt_ml.cloud.local status

  # Check training health (loss progression)
  uv run python -m openadapt_ml.cloud.local check

  # Start dashboard server
  uv run python -m openadapt_ml.cloud.local serve --open

  # Regenerate viewer
  uv run python -m openadapt_ml.cloud.local viewer --open

  # Generate benchmark viewer
  uv run python -m openadapt_ml.cloud.local benchmark-viewer benchmark_results/test_run --open

  # Run comparison
  uv run python -m openadapt_ml.cloud.local compare --capture ~/captures/my-workflow --checkpoint checkpoints/model
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # status
    p_status = subparsers.add_parser("status", help="Show local training status")
    p_status.set_defaults(func=cmd_status)

    # train
    p_train = subparsers.add_parser("train", help="Run training locally")
    p_train.add_argument("--capture", required=True, help="Path to capture directory")
    p_train.add_argument("--goal", help="Task goal (default: derived from capture name)")
    p_train.add_argument("--config", help="Config file (default: auto-select based on device)")
    p_train.add_argument("--open", action="store_true", help="Open dashboard in browser")
    p_train.set_defaults(func=cmd_train)

    # check
    p_check = subparsers.add_parser("check", help="Check training health")
    p_check.set_defaults(func=cmd_check)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start web server for dashboard")
    p_serve.add_argument("--port", type=int, default=8765, help="Port number")
    p_serve.add_argument("--open", action="store_true", help="Open in browser")
    p_serve.add_argument("--quiet", "-q", action="store_true", help="Suppress request logging")
    p_serve.add_argument("--no-regenerate", action="store_true",
                         help="Skip regenerating dashboard/viewer (serve existing files)")
    p_serve.add_argument("--benchmark", help="Serve benchmark results directory instead of training output")
    p_serve.add_argument("--start-page", help="Override default start page (e.g., benchmark.html)")
    p_serve.set_defaults(func=cmd_serve)

    # viewer
    p_viewer = subparsers.add_parser("viewer", help="Regenerate viewer")
    p_viewer.add_argument("--open", action="store_true", help="Open in browser")
    p_viewer.set_defaults(func=cmd_viewer)

    # benchmark_viewer
    p_benchmark = subparsers.add_parser("benchmark-viewer", help="Generate benchmark viewer")
    p_benchmark.add_argument("benchmark_dir", help="Path to benchmark results directory")
    p_benchmark.add_argument("--open", action="store_true", help="Open viewer in browser")
    p_benchmark.set_defaults(func=cmd_benchmark_viewer)

    # compare
    p_compare = subparsers.add_parser("compare", help="Run human vs AI comparison")
    p_compare.add_argument("--capture", required=True, help="Path to capture directory")
    p_compare.add_argument("--checkpoint", help="Path to checkpoint (optional)")
    p_compare.add_argument("--open", action="store_true", help="Open viewer in browser")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
