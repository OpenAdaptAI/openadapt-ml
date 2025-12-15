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
    """
    # Check summary.json for model_id
    summary_path = benchmark_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            model_id = summary.get("model_id", "").lower()
            # Filter out mock/test/random agent runs
            if any(term in model_id for term in ["random", "mock", "test"]):
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
            if any(term in model_id for term in ["random", "mock", "test"]):
                return True
        except Exception:
            pass

    # Check benchmark name for "mock"
    if "mock" in benchmark_dir.name.lower():
        return True

    return False


def _regenerate_benchmark_viewer_if_available(output_dir: Path) -> bool:
    """Regenerate benchmark.html from latest real benchmark results.

    Looks for the most recent non-mock benchmark run in benchmark_results/ directory
    and generates a benchmark viewer in the output directory. If no real benchmark
    data exists, generates an empty state viewer with guidance.

    Returns True if benchmark viewer was regenerated, False otherwise.
    """
    from openadapt_ml.training.benchmark_viewer import (
        generate_benchmark_viewer,
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
            print("  Generated benchmark viewer: No real evaluation data yet")
            return True
        except Exception as e:
            print(f"  Could not generate empty benchmark viewer: {e}")
            return False

    # Sort by modification time to get the latest real benchmark
    latest_benchmark = max(real_benchmarks, key=lambda d: d.stat().st_mtime)

    try:
        # Generate benchmark.html in the output directory
        generate_benchmark_viewer(latest_benchmark, benchmark_html_path)

        # Copy tasks folder for screenshots
        tasks_src = latest_benchmark / "tasks"
        tasks_dst = output_dir / "tasks"
        if tasks_src.exists():
            if tasks_dst.exists():
                shutil.rmtree(tasks_dst)
            shutil.copytree(tasks_src, tasks_dst)

        print(f"  Regenerated benchmark viewer from: {latest_benchmark.name}")
        return True
    except Exception as e:
        print(f"  Could not regenerate benchmark viewer: {e}")
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
            else:
                # Default file serving
                super().do_GET()

        def do_OPTIONS(self):
            # Handle CORS preflight
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

    with socketserver.TCPServer(("", port), StopHandler) as httpd:
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
