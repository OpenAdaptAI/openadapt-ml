"""Modal cloud GPU integration for training.

Modal is a Python-native serverless cloud platform:
- No SSH, no instances to manage
- Decorate functions with @app.function(gpu="A10G") to run remotely
- Per-second billing, $30/month free credits
- Data transfer via modal.Volume

Setup:
    pip install modal
    modal token set  # or set MODAL_TOKEN_ID + MODAL_TOKEN_SECRET

Usage:
    # Train with auto-convert from demos
    python -m openadapt_ml.cloud.modal_cloud train \
        --demo-dir /path/to/demos \
        --captures-dir /path/to/captures

    # Train with pre-built bundle
    python -m openadapt_ml.cloud.modal_cloud train --bundle /path/to/bundle

    # Check training status
    python -m openadapt_ml.cloud.modal_cloud status

    # Download results
    python -m openadapt_ml.cloud.modal_cloud download --output ./results

    # List volumes
    python -m openadapt_ml.cloud.modal_cloud list-volumes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Modal app definition (imported lazily so CLI parsing works without modal)
# ---------------------------------------------------------------------------

MODAL_APP_NAME = "openadapt-training"
VOLUME_NAME = "openadapt-training-data"
VOLUME_MOUNT = "/training"
BUNDLE_REMOTE_PATH = "/training/bundle"
RESULTS_REMOTE_PATH = "/training/results"


def _get_modal():
    """Lazy-import modal, with a helpful error if not installed."""
    try:
        import modal

        return modal
    except ImportError:
        print(
            "Error: modal is not installed.\n"
            "  Install it with: pip install modal\n"
            "  Then authenticate: modal token set"
        )
        sys.exit(1)


def _build_app():
    """Build and return the Modal app, image, and volume.

    Returns:
        (app, training_image, training_volume)
    """
    modal = _get_modal()

    app = modal.App(MODAL_APP_NAME)

    training_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    training_image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "bitsandbytes",
        "accelerate",
        "pyyaml",
        "pillow",
    )

    return app, training_image, training_volume


# We build these at module level so that `modal run` can discover the app,
# but guard behind a function so tests can import the module without modal.
_app = None
_training_image = None
_training_volume = None


def _ensure_app():
    """Ensure the Modal app is initialized (lazy singleton)."""
    global _app, _training_image, _training_volume
    if _app is None:
        _app, _training_image, _training_volume = _build_app()
    return _app, _training_image, _training_volume


def get_app():
    """Public accessor for the Modal app (used by `modal run`)."""
    app, _, _ = _ensure_app()
    return app


def _register_train_function():
    """Register the remote training function on the app.

    Returns the function handle that can be called with .remote().
    """
    app, training_image, training_volume = _ensure_app()

    @app.function(
        gpu="A10G",
        image=training_image,
        volumes={VOLUME_MOUNT: training_volume},
        timeout=3600,
    )
    def train_model(
        config_yaml: str,
        bundle_path: str = BUNDLE_REMOTE_PATH,
    ) -> str:
        """Run SFT training on Modal GPU.

        Args:
            config_yaml: YAML training config as a string.
            bundle_path: Path to the bundle directory inside the volume.

        Returns:
            JSON string with training results summary.
        """
        import json as _json
        import os as _os
        import time

        import yaml

        results_dir = RESULTS_REMOTE_PATH
        os.makedirs(results_dir, exist_ok=True)

        config = yaml.safe_load(config_yaml)

        # Point config at volume paths
        config["dataset_path"] = f"{bundle_path}/training_data.jsonl"
        config["image_dir"] = f"{bundle_path}/images"
        config["output_dir"] = results_dir

        # Write config to disk for the trainer
        config_path = f"{VOLUME_MOUNT}/train_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Log start
        training_log = {
            "status": "running",
            "start_time": time.time(),
            "config": config,
            "losses": [],
        }
        log_path = f"{results_dir}/training_log.json"
        with open(log_path, "w") as f:
            _json.dump(training_log, f, indent=2)

        # Commit volume so logs are visible during training
        training_volume.commit()

        # Run training via subprocess (same pattern as Lambda)
        cmd = [
            sys.executable,
            "-m",
            "openadapt_ml.scripts.train",
            "--config",
            config_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3500,  # Slightly less than function timeout
            )

            training_log["status"] = "completed" if result.returncode == 0 else "failed"
            training_log["returncode"] = result.returncode
            training_log["end_time"] = time.time()
            training_log["elapsed_time"] = (
                training_log["end_time"] - training_log["start_time"]
            )

            if result.stdout:
                training_log["stdout_tail"] = result.stdout[-2000:]
            if result.stderr:
                training_log["stderr_tail"] = result.stderr[-2000:]

        except subprocess.TimeoutExpired:
            training_log["status"] = "timeout"
            training_log["end_time"] = time.time()
            training_log["elapsed_time"] = (
                training_log["end_time"] - training_log["start_time"]
            )
        except Exception as e:
            training_log["status"] = "error"
            training_log["error"] = str(e)
            training_log["end_time"] = time.time()
            training_log["elapsed_time"] = (
                training_log["end_time"] - training_log["start_time"]
            )

        # Read losses from the trainer's own log if it exists
        trainer_log = f"{results_dir}/training_log.json"
        if _os.path.exists(trainer_log):
            try:
                with open(trainer_log) as f:
                    trainer_data = _json.load(f)
                if "losses" in trainer_data:
                    training_log["losses"] = trainer_data["losses"]
                if "epoch" in trainer_data:
                    training_log["epoch"] = trainer_data["epoch"]
            except Exception:
                pass

        # Save final log and commit volume
        with open(log_path, "w") as f:
            _json.dump(training_log, f, indent=2)

        training_volume.commit()

        return _json.dumps(
            {
                "status": training_log["status"],
                "elapsed_time": training_log.get("elapsed_time", 0),
                "results_path": results_dir,
            }
        )

    return train_model


# ---------------------------------------------------------------------------
# Local helpers for CLI commands
# ---------------------------------------------------------------------------


def upload_bundle_to_volume(local_bundle: str | Path) -> None:
    """Upload a local bundle directory to the Modal volume.

    Uses `modal volume put` CLI for simplicity and progress display.

    Args:
        local_bundle: Path to local bundle directory containing
                      training_data.jsonl and images/.
    """
    local_bundle = Path(local_bundle)
    if not local_bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {local_bundle}")

    jsonl = local_bundle / "training_data.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"No training_data.jsonl in bundle: {local_bundle}")

    print(f"Uploading bundle to Modal volume '{VOLUME_NAME}'...")

    cmd = [
        "modal",
        "volume",
        "put",
        VOLUME_NAME,
        str(local_bundle),
        "/bundle",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Volume upload failed: {result.stderr or result.stdout}")
    print("Upload complete.")


def download_results_from_volume(local_output: str | Path) -> None:
    """Download results from the Modal volume.

    Args:
        local_output: Local directory to download results into.
    """
    local_output = Path(local_output)
    local_output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading results from Modal volume '{VOLUME_NAME}'...")

    cmd = [
        "modal",
        "volume",
        "get",
        VOLUME_NAME,
        "/results",
        str(local_output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Volume download failed: {result.stderr or result.stdout}")
    print(f"Results downloaded to: {local_output}")


def list_volumes() -> list[str]:
    """List Modal volumes.

    Returns:
        List of volume names.
    """
    cmd = ["modal", "volume", "list"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error listing volumes: {result.stderr}")
        return []
    print(result.stdout)
    return result.stdout.strip().splitlines()


def check_status() -> dict:
    """Check if a training function is currently running.

    Returns:
        Status dict with 'running' bool and details.
    """
    # Try to read training_log.json from the volume
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "modal",
            "volume",
            "get",
            VOLUME_NAME,
            "/results/training_log.json",
            tmpdir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return {
                "running": False,
                "status": "no_data",
                "message": "No training log found",
            }

        log_path = Path(tmpdir) / "training_log.json"
        if not log_path.exists():
            return {
                "running": False,
                "status": "no_data",
                "message": "No training log found",
            }

        try:
            with open(log_path) as f:
                log_data = json.load(f)
            return {
                "running": log_data.get("status") == "running",
                "status": log_data.get("status", "unknown"),
                "epoch": log_data.get("epoch", 0),
                "elapsed_time": log_data.get("elapsed_time", 0),
                "losses": log_data.get("losses", []),
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "running": False,
                "status": "corrupt_log",
                "message": "Could not parse training log",
            }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for Modal cloud GPU training.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Modal cloud GPU training for OpenAdapt",
        prog="python -m openadapt_ml.cloud.modal_cloud",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Run training on Modal GPU")
    train_parser.add_argument(
        "--bundle",
        "-b",
        help="Local bundle directory (with training_data.jsonl + images/)",
    )
    train_parser.add_argument(
        "--demo-dir",
        help="Directory with annotated demo JSON files (auto-converts to bundle)",
    )
    train_parser.add_argument(
        "--captures-dir",
        help="Parent directory containing capture directories (for screenshot resolution)",
    )
    train_parser.add_argument(
        "--mapping",
        help="Pre-computed screenshot_mapping.json (optional)",
    )
    train_parser.add_argument(
        "--config",
        default="configs/qwen3vl_capture_4bit.yaml",
        help="Training config YAML file (default: configs/qwen3vl_capture_4bit.yaml)",
    )
    train_parser.add_argument(
        "--gpu",
        default="A10G",
        help="GPU type (default: A10G). Options: T4, A10G, A100, H100",
    )
    train_parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Training timeout in seconds (default: 3600)",
    )

    # --- status ---
    subparsers.add_parser("status", help="Check training status")

    # --- download ---
    download_parser = subparsers.add_parser(
        "download", help="Download results from Modal volume"
    )
    download_parser.add_argument(
        "--output",
        "-o",
        default="training_output/modal",
        help="Local output directory (default: training_output/modal)",
    )

    # --- list-volumes ---
    subparsers.add_parser("list-volumes", help="List Modal volumes")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "train":
        return _cmd_train(args)
    elif args.command == "status":
        return _cmd_status(args)
    elif args.command == "download":
        return _cmd_download(args)
    elif args.command == "list-volumes":
        return _cmd_list_volumes(args)
    else:
        parser.print_help()
        return 1


def _cmd_train(args: argparse.Namespace) -> int:
    """Execute the train command.

    Flow:
    1. Auto-convert demos to bundle if --demo-dir provided
    2. Upload bundle to Modal Volume
    3. Run training function remotely
    4. Download results
    """
    modal = _get_modal()

    # --- Step 0: Auto-convert demos if --demo-dir provided ---
    bundle_dir = args.bundle
    if args.demo_dir and not bundle_dir:
        from openadapt_ml.training.convert_demos import prepare_bundle

        if not args.captures_dir:
            print("Error: --captures-dir is required with --demo-dir")
            return 1

        print("=" * 50)
        print("Step 0: Converting demos to training bundle")
        print("=" * 50)
        try:
            bundle_path = prepare_bundle(
                demo_dir=args.demo_dir,
                captures_dir=args.captures_dir,
                mapping_path=args.mapping,
            )
            bundle_dir = str(bundle_path)
            print(f"Bundle ready at: {bundle_dir}\n")
        except Exception as e:
            print(f"Error converting demos: {e}")
            return 1

    if not bundle_dir:
        print("Error: Provide --bundle or --demo-dir (with --captures-dir)")
        return 1

    # --- Step 1: Upload bundle to volume ---
    print("=" * 50)
    print("Step 1: Uploading bundle to Modal volume")
    print("=" * 50)
    try:
        upload_bundle_to_volume(bundle_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    # --- Step 2: Read config ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config_yaml = config_path.read_text()

    # --- Step 3: Run training remotely ---
    print("=" * 50)
    print("Step 2: Running training on Modal GPU")
    print("=" * 50)
    print(f"GPU: {args.gpu}")
    print(f"Config: {args.config}")
    print(f"Timeout: {args.timeout}s")
    print()

    try:
        modal.enable_output()

        train_fn = _register_train_function()

        with get_app().run():
            result_json = train_fn.remote(
                config_yaml=config_yaml,
                bundle_path=BUNDLE_REMOTE_PATH,
            )

        result = json.loads(result_json)
        print()
        print("=" * 50)
        print(f"Training {result.get('status', 'unknown')}")
        print(f"Elapsed: {result.get('elapsed_time', 0):.1f}s")
        print("=" * 50)

    except Exception as e:
        print(f"Training failed: {e}")
        return 1

    # --- Step 4: Download results ---
    print()
    print("Downloading results...")
    output_dir = Path("training_output") / "modal"
    try:
        download_results_from_volume(output_dir)
    except RuntimeError as e:
        print(f"Warning: Could not download results: {e}")
        print(
            "You can download later with: python -m openadapt_ml.cloud.modal_cloud download"
        )

    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Show training status."""
    print("MODAL TRAINING STATUS")
    print("=" * 40)

    status = check_status()

    print(f"Status: {status.get('status', 'unknown')}")
    if status.get("running"):
        print("Training is currently RUNNING")
    if status.get("epoch"):
        print(f"Epoch: {status['epoch']}")
    if status.get("elapsed_time"):
        elapsed = status["elapsed_time"]
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"Elapsed: {mins}m {secs}s")
    if status.get("losses"):
        last_loss = status["losses"][-1]
        if isinstance(last_loss, dict):
            print(f"Last loss: {last_loss.get('loss', 'N/A')}")
        else:
            print(f"Last loss: {last_loss}")
    if status.get("message"):
        print(f"Note: {status['message']}")

    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    """Download results from Modal volume."""
    try:
        download_results_from_volume(args.output)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1


def _cmd_list_volumes(args: argparse.Namespace) -> int:
    """List Modal volumes."""
    list_volumes()
    return 0


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(cli_main())
