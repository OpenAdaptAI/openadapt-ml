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
    """Get current training status from training_output."""
    status = {
        "running": False,
        "epoch": 0,
        "step": 0,
        "loss": None,
        "device": detect_device(),
        "has_dashboard": False,
        "has_viewer": False,
        "checkpoints": [],
    }

    log_file = TRAINING_OUTPUT / "training_log.json"
    if log_file.exists():
        try:
            with open(log_file) as f:
                data = json.load(f)
            status["epoch"] = data.get("epoch", 0)
            status["step"] = data.get("step", 0)
            status["loss"] = data.get("loss")
            status["learning_rate"] = data.get("learning_rate")
            status["losses"] = data.get("losses", [])
            status["status"] = data.get("status", "unknown")
            status["running"] = data.get("status") == "training"
        except (json.JSONDecodeError, KeyError):
            pass

    status["has_dashboard"] = (TRAINING_OUTPUT / "dashboard.html").exists()
    status["has_viewer"] = (TRAINING_OUTPUT / "viewer.html").exists()

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

    print(f"\n{'='*50}")
    print("LOCAL TRAINING STATUS")
    print(f"{'='*50}")
    print(f"Device: {status['device']}")
    print(f"Status: {'RUNNING' if status['running'] else 'IDLE'}")

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

    print(f"\nDashboard: {'✓' if status['has_dashboard'] else '✗'} training_output/dashboard.html")
    print(f"Viewer: {'✓' if status['has_viewer'] else '✗'} training_output/viewer.html")
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
    """Start local web server for dashboard."""
    port = args.port

    if not TRAINING_OUTPUT.exists():
        print(f"Error: {TRAINING_OUTPUT} not found. Run training first.")
        return 1

    os.chdir(TRAINING_OUTPUT)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress request logging

    handler = QuietHandler if args.quiet else http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/dashboard.html"
        print(f"\nServing training output at: {url}")
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
        _enhance_comparison_to_unified_viewer,
        TrainingState,
        TrainingConfig,
    )

    if not TRAINING_OUTPUT.exists():
        print(f"Error: {TRAINING_OUTPUT} not found. Run training first.")
        return 1

    print("Regenerating viewer from training_output...")

    # Regenerate dashboard
    log_file = TRAINING_OUTPUT / "training_log.json"
    if log_file.exists():
        with open(log_file) as f:
            data = json.load(f)

        state = TrainingState()
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
        (TRAINING_OUTPUT / "dashboard.html").write_text(dashboard_html)
        print(f"  Regenerated: dashboard.html")

    # Find comparison HTML to enhance
    comparison_files = list(TRAINING_OUTPUT.glob("comparison_epoch*.html"))
    if comparison_files:
        # Use the latest epoch comparison
        base_file = sorted(comparison_files)[-1]

        # Load all prediction files
        predictions_by_checkpoint = {"None": []}
        for pred_file in TRAINING_OUTPUT.glob("predictions_*.json"):
            checkpoint_name = pred_file.stem.replace("predictions_", "")
            # Map to display name
            if "epoch" in checkpoint_name:
                display_name = checkpoint_name.replace("epoch", "Epoch ").replace("_", " ").title()
            elif checkpoint_name == "preview":
                display_name = "Preview"
            else:
                display_name = checkpoint_name.title()

            try:
                with open(pred_file) as f:
                    predictions_by_checkpoint[display_name] = json.load(f)
                print(f"  Loaded predictions from {pred_file.name}")
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse {pred_file.name}")

        # Get capture info
        capture_id = "capture"
        goal = "Complete the recorded workflow"

        _enhance_comparison_to_unified_viewer(
            base_file,
            predictions_by_checkpoint,
            TRAINING_OUTPUT / "viewer.html",
            capture_id,
            goal,
        )

    print(f"\nGenerated: {TRAINING_OUTPUT / 'viewer.html'}")

    if args.open:
        webbrowser.open(str(TRAINING_OUTPUT / "viewer.html"))

    return 0


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
    p_serve.set_defaults(func=cmd_serve)

    # viewer
    p_viewer = subparsers.add_parser("viewer", help="Regenerate viewer")
    p_viewer.add_argument("--open", action="store_true", help="Open in browser")
    p_viewer.set_defaults(func=cmd_viewer)

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
