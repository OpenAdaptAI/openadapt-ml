"""Lambda Labs cloud GPU integration.

Lambda Labs provides affordable GPU instances for training:
- A100 40GB: ~$1.10/hour
- H100: ~$2.00/hour
- A10: ~$0.60/hour

API docs: https://cloud.lambdalabs.com/api/v1/docs

Usage:
    # Set API key
    export LAMBDA_API_KEY=your_key_here

    # List available instances
    python -m openadapt_ml.cloud.lambda_labs list

    # Launch instance for training
    python -m openadapt_ml.cloud.lambda_labs launch --type gpu_1x_a100

    # Check running instances
    python -m openadapt_ml.cloud.lambda_labs status

    # Terminate instance
    python -m openadapt_ml.cloud.lambda_labs terminate <instance_id>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


API_BASE = "https://cloud.lambdalabs.com/api/v1"

# Default port for HTTP server
DEFAULT_SERVER_PORT = 8765


def start_dashboard_server(output_dir: Path, port: int = DEFAULT_SERVER_PORT) -> tuple[subprocess.Popen, str]:
    """Start a background HTTP server for the dashboard.

    Args:
        output_dir: Directory containing dashboard files
        port: Port to serve on

    Returns:
        (process, url): The server process and the dashboard URL
    """
    import webbrowser
    import threading

    # Start simple HTTP server in background thread
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=str(output_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url = f"http://localhost:{port}/dashboard.html"

    # Give server time to start
    time.sleep(0.5)

    return server_proc, url


def open_dashboard_in_browser(output_dir: Path, port: int = DEFAULT_SERVER_PORT):
    """Start HTTP server and open dashboard in browser.

    Args:
        output_dir: Directory containing dashboard files
        port: Port to serve on

    Returns:
        Server process (caller should call .terminate() when done), or None if failed
    """
    import webbrowser

    try:
        server_proc, url = start_dashboard_server(output_dir, port)
        webbrowser.open(url)
        print(f"Dashboard: {url}")
        print("  Stop Training button enabled in dashboard")
        return server_proc
    except Exception as e:
        print(f"Warning: Could not start dashboard server: {e}")
        return None


def setup_capture_screenshots_symlink(output_dir: Path, capture_path: str | Path) -> bool:
    """Create symlink from output_dir/screenshots to capture's screenshots folder.

    This allows the dashboard to serve screenshots via relative paths.

    Args:
        output_dir: Training output directory (e.g., training_output/job_id/)
        capture_path: Path to capture directory (local)

    Returns:
        True if symlink created successfully
    """
    capture_path = Path(capture_path)
    screenshots_src = capture_path / "screenshots"
    screenshots_dst = output_dir / "screenshots"

    if not screenshots_src.exists():
        return False

    # Remove existing symlink or directory
    if screenshots_dst.is_symlink():
        screenshots_dst.unlink()
    elif screenshots_dst.exists():
        return False  # Don't overwrite real directory

    try:
        screenshots_dst.symlink_to(screenshots_src.resolve())
        return True
    except Exception:
        return False


def rewrite_evaluation_paths(evaluations: list[dict], remote_prefix: str = "/home/ubuntu/capture/") -> list[dict]:
    """Rewrite Lambda paths in evaluations to relative paths.

    Converts: /home/ubuntu/capture/screenshots/foo.png -> screenshots/foo.png

    Args:
        evaluations: List of evaluation dicts with image_path
        remote_prefix: The Lambda path prefix to replace

    Returns:
        Evaluations with rewritten paths
    """
    for ev in evaluations:
        if "image_path" in ev and ev["image_path"].startswith(remote_prefix):
            ev["image_path"] = ev["image_path"].replace(remote_prefix, "")
    return evaluations


def download_checkpoints_from_instance(instance_ip: str, output_dir: Path, ssh_key: str | None = None) -> bool:
    """Download checkpoints from Lambda instance.

    Args:
        instance_ip: IP address of Lambda instance
        output_dir: Local directory to save checkpoints
        ssh_key: Path to SSH key (uses default if not provided)

    Returns:
        True if download succeeded
    """
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ssh_key = ssh_key or str(Path.home() / ".ssh" / "lambda_id_ed25519")
    ssh_opts = f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key}"

    # Download checkpoints from remote
    remote_path = f"ubuntu@{instance_ip}:~/openadapt-ml/checkpoints/"
    local_path = str(checkpoints_dir) + "/"

    cmd = f"rsync -avz --progress -e 'ssh {ssh_opts}' {remote_path} {local_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        return True
    return False


def check_stop_signal(output_dir: Path) -> bool:
    """Check if stop signal file exists.

    The dashboard can create this file to signal training should stop.
    """
    stop_file = output_dir / "STOP_TRAINING"
    return stop_file.exists()


@dataclass
class InstanceType:
    """Lambda Labs instance type."""
    name: str
    price_cents_per_hour: int
    description: str
    gpu_count: int
    gpu_type: str
    vcpus: int
    memory_gb: int
    storage_gb: int
    available_regions: list[str]

    @property
    def price_per_hour(self) -> float:
        return self.price_cents_per_hour / 100

    def __str__(self) -> str:
        regions = ", ".join(self.available_regions[:3])
        if len(self.available_regions) > 3:
            regions += f" (+{len(self.available_regions) - 3} more)"
        return (
            f"{self.name}: ${self.price_per_hour:.2f}/hr | "
            f"{self.gpu_count}x {self.gpu_type} | {self.vcpus} vCPUs | "
            f"{self.memory_gb}GB RAM | {self.storage_gb}GB SSD | "
            f"Regions: {regions}"
        )


@dataclass
class Instance:
    """Running Lambda Labs instance."""
    id: str
    name: str
    instance_type: str
    status: str
    ip: str | None
    region: str
    ssh_key_names: list[str]

    def __str__(self) -> str:
        ip_str = self.ip or "pending"
        return f"{self.id[:8]}... | {self.instance_type} | {self.status} | IP: {ip_str} | {self.region}"


class LambdaLabsClient:
    """Client for Lambda Labs API."""

    def __init__(self, api_key: str | None = None):
        # Try provided key, then settings, then env var
        if not api_key:
            from openadapt_ml.config import settings
            api_key = settings.lambda_api_key or os.environ.get("LAMBDA_API_KEY")

        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "Lambda Labs API key required. Set LAMBDA_API_KEY in .env file "
                "or get one at https://cloud.lambdalabs.com/api-keys"
            )
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _get(self, endpoint: str) -> dict[str, Any]:
        """Make GET request to API."""
        resp = self.session.get(f"{API_BASE}{endpoint}")
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make POST request to API."""
        resp = self.session.post(f"{API_BASE}{endpoint}", json=data)
        if not resp.ok:
            error = resp.json().get("error", {})
            raise RuntimeError(f"API error: {error.get('message', resp.text)}")
        return resp.json()

    def list_instance_types(self) -> list[InstanceType]:
        """List available GPU instance types."""
        data = self._get("/instance-types")
        types = []

        for name, info in data.get("data", {}).items():
            specs = info.get("instance_type", {}).get("specs", {})
            regions = [r["name"] for r in info.get("regions_with_capacity_available", [])]

            types.append(InstanceType(
                name=name,
                price_cents_per_hour=info.get("instance_type", {}).get("price_cents_per_hour", 0),
                description=info.get("instance_type", {}).get("description", ""),
                gpu_count=specs.get("gpus", 0),
                gpu_type=info.get("instance_type", {}).get("gpu_description", ""),
                vcpus=specs.get("vcpus", 0),
                memory_gb=specs.get("memory_gib", 0),
                storage_gb=specs.get("storage_gib", 0),
                available_regions=regions,
            ))

        # Sort by price
        types.sort(key=lambda t: t.price_cents_per_hour)
        return types

    def list_ssh_keys(self) -> list[dict[str, str]]:
        """List registered SSH keys."""
        data = self._get("/ssh-keys")
        return data.get("data", [])

    def add_ssh_key(self, name: str, public_key: str) -> dict[str, str]:
        """Add an SSH key."""
        data = self._post("/ssh-keys", {"name": name, "public_key": public_key})
        return data.get("data", {})

    def list_instances(self) -> list[Instance]:
        """List running instances."""
        data = self._get("/instances")
        instances = []

        for inst in data.get("data", []):
            # ssh_key_names can be list of strings or list of dicts
            ssh_keys = inst.get("ssh_key_names", [])
            if ssh_keys and isinstance(ssh_keys[0], dict):
                ssh_key_names = [k["name"] for k in ssh_keys]
            else:
                ssh_key_names = ssh_keys  # Already list of strings

            instances.append(Instance(
                id=inst["id"],
                name=inst.get("name", ""),
                instance_type=inst.get("instance_type", {}).get("name", "unknown"),
                status=inst.get("status", "unknown"),
                ip=inst.get("ip"),
                region=inst.get("region", {}).get("name", "unknown"),
                ssh_key_names=ssh_key_names,
            ))

        return instances

    def launch_instance(
        self,
        instance_type: str,
        region: str | None = None,
        ssh_key_names: list[str] | None = None,
        name: str | None = None,
    ) -> Instance:
        """Launch a new GPU instance.

        Args:
            instance_type: Instance type name (e.g., 'gpu_1x_a100')
            region: Region name (auto-selects if None)
            ssh_key_names: SSH key names to use
            name: Optional instance name

        Returns:
            Launched instance
        """
        # If no region specified, find one with capacity
        if not region:
            types = self.list_instance_types()
            for t in types:
                if t.name == instance_type and t.available_regions:
                    region = t.available_regions[0]
                    break
            if not region:
                raise RuntimeError(f"No regions available for {instance_type}")

        # If no SSH key specified, use first available
        if not ssh_key_names:
            keys = self.list_ssh_keys()
            if not keys:
                raise RuntimeError(
                    "No SSH keys found. Add one at https://cloud.lambdalabs.com/ssh-keys"
                )
            ssh_key_names = [keys[0]["name"]]

        payload = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": ssh_key_names,
        }
        if name:
            payload["name"] = name

        data = self._post("/instance-operations/launch", payload)
        instance_ids = data.get("data", {}).get("instance_ids", [])

        if not instance_ids:
            raise RuntimeError("Failed to launch instance")

        # Wait for instance to be ready
        print(f"Instance {instance_ids[0]} launched, waiting for IP...")
        instance = None
        for _ in range(60):  # Wait up to 5 minutes for IP
            instances = self.list_instances()
            for inst in instances:
                if inst.id == instance_ids[0] and inst.ip:
                    instance = inst
                    break
            if instance:
                break
            time.sleep(5)

        if not instance:
            raise RuntimeError("Timed out waiting for instance IP")

        # Wait for SSH to be ready - be patient, instances can take a while to boot
        print(f"Instance IP: {instance.ip}, waiting for SSH...")
        for attempt in range(60):  # Wait up to 5 minutes for SSH
            try:
                result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                     f"ubuntu@{instance.ip}", "echo ready"],
                    capture_output=True, text=True, timeout=20
                )
                if result.returncode == 0:
                    print("SSH ready!")
                    return instance
            except subprocess.TimeoutExpired:
                pass
            if attempt % 6 == 5:  # Log progress every 30 seconds
                print(f"  Still waiting for SSH ({(attempt+1)*5}s elapsed)...")
            time.sleep(5)

        print("Warning: SSH may not be ready yet, continuing anyway...")
        return instance

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance."""
        data = self._post("/instance-operations/terminate", {"instance_ids": [instance_id]})
        terminated = data.get("data", {}).get("terminated_instances", [])
        return any(t.get("id") == instance_id for t in terminated)

    def get_ssh_command(self, instance: Instance, user: str = "ubuntu") -> str:
        """Get SSH command for an instance."""
        if not instance.ip:
            return "# Instance IP not yet available"
        return f"ssh {user}@{instance.ip}"

    def ssh_run(self, instance: Instance, command: str, timeout: int | None = None, retries: int = 3) -> subprocess.CompletedProcess:
        """Run a command on an instance via SSH.

        Args:
            instance: Instance to run on
            command: Shell command to run
            timeout: Optional timeout in seconds
            retries: Number of retries on connection failure

        Returns:
            CompletedProcess with stdout/stderr
        """
        if not instance.ip:
            raise RuntimeError("Instance has no IP address")

        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",  # Increased from 10
            "-o", "ServerAliveInterval=60",  # Keep connection alive
            "-o", "ServerAliveCountMax=3",
            f"ubuntu@{instance.ip}",
            command
        ]

        last_error = None
        for attempt in range(retries):
            try:
                return subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as e:
                last_error = e
                if attempt < retries - 1:
                    print(f"  SSH timeout, retrying ({attempt + 1}/{retries})...")
                    time.sleep(5)

        raise last_error if last_error else RuntimeError("SSH failed")

    def setup_instance(self, instance: Instance, repo_url: str = "https://github.com/OpenAdaptAI/openadapt-ml.git", clean_gpu: bool = True) -> bool:
        """Set up training environment on instance.

        Clones repo, installs uv, syncs dependencies.
        Optionally clears GPU memory from previous runs.
        Returns True if successful.
        """
        print(f"Setting up instance {instance.ip}...")

        # Clean GPU memory if requested (don't fail if this doesn't work)
        if clean_gpu:
            print("  Clearing GPU memory...")
            try:
                self.ssh_run(instance, '''
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print('GPU memory cleared')
" 2>/dev/null || true
# Kill any stale python processes using GPU
pkill -f "python.*train" 2>/dev/null || true
''', timeout=60)
            except Exception as e:
                print(f"  GPU cleanup skipped: {e}")

        setup_script = f'''
set -e
cd ~

# Install uv via official installer (most robust)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Clone or update repo
if [ ! -d "openadapt-ml" ]; then
    git clone {repo_url}
else
    cd openadapt-ml && git pull origin main && cd ~
fi

cd openadapt-ml
uv sync
echo "SETUP_COMPLETE"
'''

        try:
            result = self.ssh_run(instance, setup_script, timeout=900)  # 15 min timeout for setup

            if "SETUP_COMPLETE" in result.stdout:
                print("  Environment ready")
                return True
            else:
                stderr_preview = result.stderr[:500] if result.stderr else "(no stderr)"
                print(f"  Setup failed: {stderr_preview}")
                return False
        except subprocess.TimeoutExpired:
            print("  Setup timed out after 15 minutes")
            return False
        except Exception as e:
            print(f"  Setup failed: {e}")
            return False

    def sync_local_code(self, instance: Instance, local_repo_path: str = ".", retries: int = 3) -> bool:
        """Sync local code changes to remote instance.

        Uses rsync to push local code, excluding .venv, .git, etc.
        This ensures the remote has the same code as local.

        Args:
            instance: Instance to sync to
            local_repo_path: Local repository path
            retries: Number of retry attempts

        Returns:
            True if successful
        """
        if not instance.ip:
            raise RuntimeError("Instance has no IP address")

        print(f"Syncing local code to {instance.ip}...")

        # SSH options for more robust connection
        ssh_opts = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=60"

        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "--timeout=120",  # 2 minute timeout per file
            "--exclude", ".venv",
            "--exclude", ".git",
            "--exclude", "__pycache__",
            "--exclude", "*.pyc",
            "--exclude", ".env",
            "--exclude", "training_output",
            "--exclude", "checkpoints",
            "--exclude", "synthetic*",
            "-e", ssh_opts,
            f"{local_repo_path}/",
            f"ubuntu@{instance.ip}:~/openadapt-ml/"
        ]

        for attempt in range(retries):
            result = subprocess.run(rsync_cmd)
            if result.returncode == 0:
                print("  Code synced")
                return True
            if attempt < retries - 1:
                print(f"  Sync failed, retrying ({attempt + 1}/{retries})...")
                time.sleep(5)

        return False

    def upload_capture(self, instance: Instance, local_path: str, remote_path: str = "~/capture", retries: int = 3) -> bool:
        """Upload a capture directory to instance via rsync.

        Args:
            instance: Instance to upload to
            local_path: Local path to capture directory
            remote_path: Remote path (default: ~/capture)
            retries: Number of retry attempts

        Returns:
            True if successful
        """
        if not instance.ip:
            raise RuntimeError("Instance has no IP address")

        print(f"Uploading capture to {instance.ip}:{remote_path}...")

        # SSH options for more robust connection
        ssh_opts = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=60"

        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "--timeout=120",  # 2 minute timeout per file
            "-e", ssh_opts,
            f"{local_path}/",
            f"ubuntu@{instance.ip}:{remote_path}/"
        ]

        for attempt in range(retries):
            result = subprocess.run(rsync_cmd)
            if result.returncode == 0:
                return True
            if attempt < retries - 1:
                print(f"  Upload failed, retrying ({attempt + 1}/{retries})...")
                time.sleep(5)

        return False

    def run_training(
        self,
        instance: Instance,
        config: str = "configs/qwen3vl_capture.yaml",
        capture: str | None = None,
        goal: str | None = None,
        background: bool = True,
    ) -> subprocess.Popen | subprocess.CompletedProcess:
        """Run training on instance.

        Args:
            instance: Instance to train on
            config: Config file path (relative to repo)
            capture: Remote capture path (if uploaded)
            goal: Task goal description
            background: Run in background (returns Popen) or foreground

        Returns:
            Popen if background=True, CompletedProcess if background=False
        """
        if not instance.ip:
            raise RuntimeError("Instance has no IP address")

        # Build training command
        train_cmd = f"uv run python -m openadapt_ml.scripts.train --config {config}"
        if capture:
            train_cmd += f" --capture {capture}"
        if goal:
            train_cmd += f' --goal "{goal}"'

        # Full script with environment setup
        script = f'''
cd ~/openadapt-ml
export PATH="$HOME/.local/bin:$PATH"
{train_cmd}
'''

        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            f"ubuntu@{instance.ip}",
            script
        ]

        print(f"Running training on {instance.ip}...")
        print(f"  Config: {config}")
        if capture:
            print(f"  Capture: {capture}")

        if background:
            # Run in background, return Popen for monitoring
            return subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            # Run in foreground, stream output
            return subprocess.run(ssh_cmd)

    def download_results(
        self,
        instance: Instance,
        remote_path: str = "~/openadapt-ml",
        local_path: str = ".",
        include_checkpoint: bool = True,
        include_logs: bool = True,
    ) -> bool:
        """Download training results from instance.

        Args:
            instance: Instance to download from
            remote_path: Remote openadapt-ml directory
            local_path: Local directory to download to
            include_checkpoint: Download checkpoint weights
            include_logs: Download training logs and dashboard

        Returns:
            True if successful
        """
        if not instance.ip:
            raise RuntimeError("Instance has no IP address")

        print(f"Downloading results from {instance.ip}...")
        success = True

        # Download training output (logs, dashboard)
        if include_logs:
            print("  Downloading training logs...")
            rsync_cmd = [
                "rsync", "-avz",
                "-e", "ssh -o StrictHostKeyChecking=no",
                f"ubuntu@{instance.ip}:{remote_path}/training_output/",
                f"{local_path}/training_output_lambda/"
            ]
            result = subprocess.run(rsync_cmd, capture_output=True)
            if result.returncode == 0:
                print("    Training logs downloaded to training_output_lambda/")
            else:
                print(f"    Warning: Failed to download logs")
                success = False

        # Download checkpoint
        if include_checkpoint:
            print("  Downloading checkpoint...")
            rsync_cmd = [
                "rsync", "-avz",
                "-e", "ssh -o StrictHostKeyChecking=no",
                f"ubuntu@{instance.ip}:{remote_path}/checkpoints/",
                f"{local_path}/checkpoints_lambda/"
            ]
            result = subprocess.run(rsync_cmd, capture_output=True)
            if result.returncode == 0:
                print("    Checkpoint downloaded to checkpoints_lambda/")
            else:
                print(f"    Warning: Failed to download checkpoint (may not exist yet)")

        # Regenerate all dashboards with static navigation and correct status
        if include_logs:
            try:
                from openadapt_ml.training.trainer import regenerate_all_dashboards
                output_dir = Path(local_path) / "training_output_lambda"
                if output_dir.exists():
                    print("  Regenerating dashboards with static navigation...")
                    regenerate_all_dashboards(output_dir)
            except Exception as e:
                print(f"    Warning: Failed to regenerate dashboards: {e}")

        return success

    def get_training_status(self, instance: Instance) -> dict:
        """Check training status by reading training_log.json on instance."""
        result = self.ssh_run(
            instance,
            "cat ~/openadapt-ml/training_output/training_log.json 2>/dev/null || echo '{}'",
            timeout=10,
        )
        try:
            import json
            return json.loads(result.stdout.strip())
        except:
            return {}


def setup_lambda_ssh_key(client: LambdaLabsClient) -> str:
    """Set up SSH key for Lambda Labs if not already done.

    Returns the SSH key name that was added/found.
    """
    # Check if we already have keys
    keys = client.list_ssh_keys()
    if keys:
        print(f"Using existing SSH key: {keys[0]['name']}")
        return keys[0]["name"]

    # Look for local SSH key
    ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
    if not ssh_key_path.exists():
        ssh_key_path = Path.home() / ".ssh" / "id_ed25519.pub"

    if not ssh_key_path.exists():
        raise RuntimeError(
            "No SSH key found at ~/.ssh/id_rsa.pub or ~/.ssh/id_ed25519.pub\n"
            "Generate one with: ssh-keygen -t ed25519"
        )

    public_key = ssh_key_path.read_text().strip()
    key_name = f"openadapt-{os.environ.get('USER', 'user')}"

    print(f"Adding SSH key '{key_name}' to Lambda Labs...")
    client.add_ssh_key(key_name, public_key)
    return key_name


def main():
    """CLI for Lambda Labs."""
    import argparse

    parser = argparse.ArgumentParser(description="Lambda Labs GPU management")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List instances command
    list_parser = subparsers.add_parser("list", help="List available instance types")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show running instances")

    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch a GPU instance")
    launch_parser.add_argument(
        "--type", "-t",
        default="gpu_1x_a100",
        help="Instance type (default: gpu_1x_a100)",
    )
    launch_parser.add_argument("--region", "-r", help="Region (auto-selects if not specified)")
    launch_parser.add_argument("--name", "-n", help="Instance name")

    # Terminate command
    term_parser = subparsers.add_parser("terminate", help="Terminate an instance")
    term_parser.add_argument("instance_id", help="Instance ID to terminate")

    # SSH command - run commands or get interactive shell
    ssh_parser = subparsers.add_parser("ssh", help="SSH into Lambda instance or run command")
    ssh_parser.add_argument("instance_id", nargs="?", help="Instance ID (uses first if not specified)")
    ssh_parser.add_argument("--cmd", "-c", help="Command to run (opens shell if not specified)")
    ssh_parser.add_argument("--timeout", "-t", type=int, default=60, help="Command timeout in seconds")

    # Serve command - start dashboard server with stop button support
    serve_parser = subparsers.add_parser("serve", help="Start dashboard server with stop button support")
    serve_parser.add_argument("--output", "-o", default="training_output", help="Output directory (default: training_output)")
    serve_parser.add_argument("--port", "-p", type=int, default=8765, help="Port (default: 8765)")
    serve_parser.add_argument("--open", action="store_true", help="Open dashboard in browser")

    # Rsync command - copy files to/from Lambda instance
    rsync_parser = subparsers.add_parser("rsync", help="Rsync files to/from Lambda instance")
    rsync_parser.add_argument("source", help="Source path (prefix with 'remote:' for remote paths)")
    rsync_parser.add_argument("dest", help="Destination path (prefix with 'remote:' for remote paths)")
    rsync_parser.add_argument("instance_id", nargs="?", help="Instance ID (uses first if not specified)")
    rsync_parser.add_argument("--delete", action="store_true", help="Delete extraneous files from dest")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up SSH key for Lambda Labs")

    # Train command - full automated training pipeline
    train_parser = subparsers.add_parser("train", help="Run training on Lambda GPU")
    train_parser.add_argument("--capture", "-c", help="Local path to capture directory")
    train_parser.add_argument("--goal", "-g", help="Task goal description")
    train_parser.add_argument("--config", default="configs/qwen3vl_capture_4bit.yaml", help="Config file (default: 4bit for memory efficiency)")
    train_parser.add_argument("--type", "-t", default="gpu_1x_a10", help="Instance type")
    train_parser.add_argument("--instance", "-i", help="Use existing instance ID instead of launching new")
    train_parser.add_argument("--no-terminate", action="store_true", help="Don't terminate instance after training")
    train_parser.add_argument("--max-runtime", type=int, default=60, help="Max runtime in minutes before auto-terminate (default: 60)")
    train_parser.add_argument("--open", action="store_true", help="Open dashboard in browser when training starts")

    # Training status command
    train_status_parser = subparsers.add_parser("train-status", help="Check training status on instance")
    train_status_parser.add_argument("instance_id", nargs="?", help="Instance ID")

    # Monitor command - live dashboard for Lambda training
    monitor_parser = subparsers.add_parser("monitor", help="Monitor Lambda training with live dashboard")
    monitor_parser.add_argument("instance_id", nargs="?", help="Instance ID")
    monitor_parser.add_argument("--open", action="store_true", help="Open dashboard in browser")
    monitor_parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds (default: 5)")
    monitor_parser.add_argument("--capture", type=str, help="Local capture path for screenshot symlink")
    monitor_parser.add_argument("--auto-stop-loss", type=float, default=0.5, help="Auto-terminate when loss drops below this (default: 0.5)")
    monitor_parser.add_argument("--download-checkpoints", action="store_true", default=True, help="Auto-download checkpoints each epoch")
    monitor_parser.add_argument("--no-download-checkpoints", action="store_false", dest="download_checkpoints", help="Disable checkpoint download")
    monitor_parser.add_argument("--stub", action="store_true", help="Use stub training provider (no GPU, instant simulation)")

    # Refresh command - one-shot dashboard update
    refresh_parser = subparsers.add_parser("refresh", help="One-shot refresh of training dashboard")
    refresh_parser.add_argument("instance_id", nargs="?", help="Instance ID")
    refresh_parser.add_argument("--open", action="store_true", help="Open dashboard in browser")
    refresh_parser.add_argument("--capture", type=str, help="Local capture path for screenshot preview")

    # Checkpoints command - list remote checkpoints
    checkpoints_parser = subparsers.add_parser("checkpoints", help="List checkpoints on remote instance")
    checkpoints_parser.add_argument("instance_id", nargs="?", help="Instance ID")

    # Download results command
    download_parser = subparsers.add_parser("download", help="Download training results from instance")
    download_parser.add_argument("instance_id", nargs="?", help="Instance ID")
    download_parser.add_argument("--output", "-o", default=".", help="Local output directory")

    # Check files on instance
    files_parser = subparsers.add_parser("files", help="List training files on instance")
    files_parser.add_argument("instance_id", nargs="?", help="Instance ID")
    files_parser.add_argument("--path", "-p", default="~/openadapt-ml", help="Path to check")

    # Kill command - terminate training processes
    kill_parser = subparsers.add_parser("kill", help="Kill training/inference processes on instance")
    kill_parser.add_argument("instance_id", nargs="?", help="Instance ID")
    kill_parser.add_argument("--local", action="store_true", help="Also kill local Lambda-related processes")
    kill_parser.add_argument("--all", action="store_true", help="Kill all Python processes on instance (careful!)")

    # Check command - analyze training status and early stopping
    check_parser = subparsers.add_parser("check", help="Check training health and early stopping status")
    check_parser.add_argument("instance_id", nargs="?", help="Instance ID")
    check_parser.add_argument("--threshold", "-t", type=float, default=0.01,
                             help="Early stopping threshold (loss improvement over last N steps)")
    check_parser.add_argument("--window", "-w", type=int, default=10,
                             help="Number of recent steps to check for improvement")

    # Compare command - run comparison on Lambda and sync back
    compare_parser = subparsers.add_parser("compare", help="Run human vs AI comparison on Lambda")
    compare_parser.add_argument("instance_id", nargs="?", help="Instance ID")
    compare_parser.add_argument("--checkpoint", "-c", help="Checkpoint to use (default: latest)")
    compare_parser.add_argument("--epoch", "-e", type=int, help="Use checkpoint from specific epoch")
    compare_parser.add_argument("--open", action="store_true", help="Open viewer after generation")

    # Results viewer command - downloads and generates comparison viewer
    results_parser = subparsers.add_parser("results", help="Download results and generate comparison viewer")
    results_parser.add_argument("--capture", "-c", required=True, help="Local capture directory (for comparison)")
    results_parser.add_argument("--goal", "-g", help="Task goal description")
    results_parser.add_argument("--open", action="store_true", help="Open viewer in browser")
    results_parser.add_argument("instance_id", nargs="?", help="Instance ID")

    # Sync command - sync training output and regenerate navigation for file:// protocol
    sync_parser = subparsers.add_parser("sync", help="Sync training output from Lambda and regenerate navigation")
    sync_parser.add_argument("instance_id", nargs="?", help="Instance ID")
    sync_parser.add_argument("--output", "-o", default="training_output", help="Local output directory (default: training_output)")
    sync_parser.add_argument("--open", action="store_true", help="Open dashboard in browser after sync")

    # Viewer command - regenerate local viewer (no Lambda required)
    viewer_parser = subparsers.add_parser("viewer", help="Regenerate local viewer (no Lambda required)")
    viewer_parser.add_argument("--output", "-o", default="training_output", help="Training output directory (default: training_output)")
    viewer_parser.add_argument("--dashboard", "-d", action="store_true", help="Regenerate dashboard instead of viewer")
    viewer_parser.add_argument("--open", action="store_true", help="Open in browser (use 'serve' instead for better experience)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        client = LambdaLabsClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nGet your API key at https://cloud.lambdalabs.com/api-keys")
        print("Then set it: export LAMBDA_API_KEY=your_key_here")
        return

    if args.command == "list":
        print("Available GPU instances:\n")
        types = client.list_instance_types()
        for t in types:
            avail = "available" if t.available_regions else "no capacity"
            print(f"  {t}")
        print(f"\nTotal: {len(types)} instance types")
        print("\nLaunch with: python -m openadapt_ml.cloud.lambda_labs launch --type <name>")

    elif args.command == "status":
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
        else:
            print("Running instances:\n")
            for inst in instances:
                print(f"  {inst}")
            print(f"\nTotal: {len(instances)} instances")

    elif args.command == "launch":
        # Ensure SSH key is set up
        ssh_key = setup_lambda_ssh_key(client)

        print(f"Launching {args.type}...")
        instance = client.launch_instance(
            instance_type=args.type,
            region=args.region,
            ssh_key_names=[ssh_key],
            name=args.name,
        )
        print(f"\nInstance launched!")
        print(f"  ID: {instance.id}")
        print(f"  IP: {instance.ip}")
        print(f"  Type: {instance.instance_type}")
        print(f"  Region: {instance.region}")
        print(f"\nConnect with: ssh ubuntu@{instance.ip}")
        print(f"\nTerminate with: python -m openadapt_ml.cloud.lambda_labs terminate {instance.id}")

    elif args.command == "terminate":
        if client.terminate_instance(args.instance_id):
            print(f"Instance {args.instance_id} terminated.")
        else:
            print(f"Failed to terminate instance {args.instance_id}")

    elif args.command == "ssh":
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        if hasattr(args, 'cmd') and args.cmd:
            # Run single command
            print(f"Running on {instance.ip}: {args.cmd}")
            result = client.ssh_run(instance, args.cmd, timeout=args.timeout)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"[stderr] {result.stderr}", file=sys.stderr)
            if result.returncode != 0:
                sys.exit(result.returncode)
        else:
            # Print SSH command for interactive use
            print(client.get_ssh_command(instance))

    elif args.command == "rsync":
        # Rsync files to/from Lambda instance
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        # Parse source and dest - 'remote:' prefix indicates remote path
        source = args.source
        dest = args.dest

        if source.startswith("remote:"):
            source = f"ubuntu@{instance.ip}:{source[7:]}"
        if dest.startswith("remote:"):
            dest = f"ubuntu@{instance.ip}:{dest[7:]}"

        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "-e", "ssh -o StrictHostKeyChecking=no",
        ]
        if args.delete:
            rsync_cmd.append("--delete")
        rsync_cmd.extend([source, dest])

        print(f"Running: {' '.join(rsync_cmd)}")
        result = subprocess.run(rsync_cmd)
        sys.exit(result.returncode)

    elif args.command == "setup":
        ssh_key = setup_lambda_ssh_key(client)
        print(f"SSH key '{ssh_key}' is configured.")

    elif args.command == "train":
        # Full automated training pipeline
        import time as time_module

        instance = None
        start_time = time_module.time()
        launched_new = False
        training_completed = False  # Track if training actually finished

        # Instance pricing (approximate $/hr)
        INSTANCE_PRICES = {
            "gpu_1x_a10": 0.75,
            "gpu_1x_a100": 1.29,
            "gpu_1x_a100_sxm4": 1.29,
            "gpu_1x_h100_pcie": 2.49,
            "gpu_1x_h100_sxm5": 3.29,
        }

        # Get or launch instance
        if args.instance:
            instances = client.list_instances()
            instance = next((i for i in instances if i.id.startswith(args.instance)), None)
            if not instance:
                print(f"Error: Instance {args.instance} not found")
                return
        else:
            # Check for existing instances
            instances = client.list_instances()
            if instances:
                print(f"Using existing instance: {instances[0].id[:8]}...")
                instance = instances[0]
            else:
                # Launch new instance
                ssh_key = setup_lambda_ssh_key(client)
                print(f"Launching {args.type}...")
                instance = client.launch_instance(
                    instance_type=args.type,
                    ssh_key_names=[ssh_key],
                    name="openadapt-training",
                )
                print(f"Instance launched: {instance.id[:8]}... at {instance.ip}")
                launched_new = True

        price_per_hour = INSTANCE_PRICES.get(instance.instance_type, 1.00)
        print(f"  Instance type: {instance.instance_type} (~${price_per_hour:.2f}/hr)")
        print(f"  Max runtime: {args.max_runtime} minutes")

        # Generate initial dashboard with setup status
        from pathlib import Path
        from openadapt_ml.training.trainer import (
            TrainingState, TrainingConfig, generate_training_dashboard,
            setup_job_directory
        )
        import time as time_module
        job_id = time_module.strftime("%Y%m%d_%H%M%S")
        output_dir = setup_job_directory("training_output", job_id)
        dashboard_path = output_dir / "dashboard.html"
        log_path = output_dir / "training_log.json"

        def update_dashboard(status: str, logs: list, step: int = 0, loss: float = 0.0, epoch: int = 0):
            """Update dashboard with current setup/training status."""
            state = TrainingState(job_id=job_id)
            state.cloud_provider = "lambda"
            state.cloud_dashboard_url = "https://cloud.lambda.ai/instances"
            state.cloud_instance_id = instance.id
            state.instance_ip = instance.ip or ""
            state.instance_type = instance.instance_type
            state.setup_status = status
            state.setup_logs = logs
            state.epoch = epoch
            state.step = step
            state.loss = loss
            state.start_time = start_time
            config = TrainingConfig(num_train_epochs=5, learning_rate=5e-5)
            dashboard_path.write_text(generate_training_dashboard(state, config))
            # Also write log for polling
            log_path.write_text(json.dumps(state.to_dict(), indent=2))

        # Initial dashboard
        setup_logs = [
            f"Lambda Cloud instance: {instance.id[:8]}...",
            f"Instance type: {instance.instance_type} (~${price_per_hour:.2f}/hr)",
            f"IP address: {instance.ip or 'pending...'}",
        ]
        update_dashboard("booting", setup_logs)

        # Open dashboard in browser via HTTP server
        server_proc = None
        if args.open:
            server_proc = open_dashboard_in_browser(output_dir)

        try:
            # Set up environment with retries at the command level
            setup_logs.append("Connecting to instance...")
            update_dashboard("booting", setup_logs)

            setup_success = False
            for setup_attempt in range(3):
                setup_logs.append(f"Setup attempt {setup_attempt + 1}/3...")
                update_dashboard("installing", setup_logs)
                if client.setup_instance(instance):
                    setup_success = True
                    setup_logs.append("Instance setup complete!")
                    update_dashboard("installing", setup_logs)
                    break
                if setup_attempt < 2:
                    setup_logs.append(f"Setup attempt {setup_attempt + 1} failed, retrying in 30s...")
                    update_dashboard("booting", setup_logs)
                    print(f"  Setup attempt {setup_attempt + 1} failed, retrying in 30s...")
                    time_module.sleep(30)

            if not setup_success:
                setup_logs.append("ERROR: Failed to set up instance after 3 attempts")
                update_dashboard("booting", setup_logs)
                print("\nError: Failed to set up instance after 3 attempts")
                print(f"Instance still running: {instance.ip}")
                print("Debug via: ssh ubuntu@" + instance.ip)
                print(f"Terminate with: python -m openadapt_ml.cloud.lambda_labs terminate {instance.id}")
                return  # Don't terminate - let user debug

            # Sync local code to ensure remote has latest changes
            setup_logs.append("Syncing local code to instance...")
            update_dashboard("installing", setup_logs)
            if not client.sync_local_code(instance):
                setup_logs.append("Warning: Failed to sync local code, using remote repo version")
                update_dashboard("installing", setup_logs)
                print("Warning: Failed to sync local code, using remote repo version")
            else:
                setup_logs.append("Code synced successfully")
                update_dashboard("installing", setup_logs)

            # Upload capture if provided
            remote_capture = None
            if args.capture:
                setup_logs.append(f"Uploading capture data...")
                update_dashboard("installing", setup_logs)
                if client.upload_capture(instance, args.capture, "~/capture"):
                    remote_capture = "~/capture"
                    setup_logs.append(f"Capture uploaded to {instance.ip}:~/capture")
                    update_dashboard("installing", setup_logs)
                    print(f"Capture uploaded to {instance.ip}:~/capture")
                else:
                    setup_logs.append("ERROR: Failed to upload capture after retries")
                    update_dashboard("installing", setup_logs)
                    print("\nError: Failed to upload capture after retries")
                    print(f"Instance still running: {instance.ip}")
                    print("Debug via: ssh ubuntu@" + instance.ip)
                    print(f"Terminate with: python -m openadapt_ml.cloud.lambda_labs terminate {instance.id}")
                    return  # Don't terminate - let user debug

            # Run training in background and poll for status
            setup_logs.append("Installing dependencies and starting training...")
            update_dashboard("training", setup_logs)
            print("\n" + "=" * 50)
            print("Starting training...")
            print("=" * 50 + "\n")

            proc = client.run_training(
                instance,
                config=args.config,
                capture=remote_capture,
                goal=args.goal,
                background=True,  # Run in background so we can poll
            )

            # Poll for training status and update dashboard
            poll_interval = 10  # seconds
            last_step = 0
            last_epoch = 0
            print(f"Polling training status every {poll_interval}s (Ctrl+C to stop)...\n")

            while True:
                try:
                    status = client.get_training_status(instance)

                    if status and status.get("step", 0) > 0:
                        step = status.get("step", 0)
                        epoch = status.get("epoch", 0)
                        loss = status.get("loss", 0)
                        elapsed_training = status.get("elapsed_time", 0)
                        total_epochs = status.get("total_epochs", 5)

                        # Print progress when step changes
                        if step > last_step or epoch > last_epoch:
                            print(f"  Epoch {epoch+1}/{total_epochs} | Step {step} | Loss: {loss:.4f} | Elapsed: {elapsed_training:.0f}s")
                            last_step = step
                            last_epoch = epoch

                        # Update local training_log.json (dashboard polls this)
                        status["total_epochs"] = total_epochs
                        if not status.get("instance_ip"):
                            status["instance_ip"] = instance.ip
                        if not status.get("instance_type"):
                            status["instance_type"] = instance.instance_type
                        # Add cloud provider info
                        status["cloud_provider"] = "lambda"
                        status["cloud_dashboard_url"] = "https://cloud.lambda.ai/instances"
                        status["cloud_instance_id"] = instance.id
                        status["setup_status"] = "training"
                        status["setup_logs"] = setup_logs
                        log_path.write_text(json.dumps(status, indent=2))

                        # Regenerate dashboard with updated data
                        state = TrainingState()
                        state.job_id = status.get("job_id", "")
                        state.hostname = status.get("hostname", "lambda")
                        state.instance_ip = instance.ip or ""
                        state.instance_type = instance.instance_type
                        state.epoch = epoch
                        state.step = step
                        state.total_epochs = total_epochs
                        state.loss = loss
                        state.learning_rate = status.get("learning_rate", 5e-5)
                        state.losses = status.get("losses", [])
                        state.evaluations = status.get("evaluations", [])
                        state.start_time = time_module.time() - elapsed_training
                        state.cloud_provider = "lambda"
                        state.cloud_dashboard_url = "https://cloud.lambda.ai/instances"
                        state.cloud_instance_id = instance.id
                        state.setup_status = "training"
                        state.setup_logs = setup_logs

                        config = TrainingConfig(
                            num_train_epochs=total_epochs,
                            learning_rate=status.get("learning_rate", 5e-5)
                        )
                        dashboard_path.write_text(generate_training_dashboard(state, config))

                        # Check if training is complete (all epochs done)
                        if epoch >= total_epochs - 1:
                            # Check if step count stopped increasing
                            time_module.sleep(poll_interval)
                            new_status = client.get_training_status(instance)
                            if new_status and new_status.get("step", 0) == step:
                                print("\n" + "=" * 50)
                                print("Training complete!")
                                print("=" * 50)
                                training_completed = True
                                break
                    else:
                        # Training not started yet, show setup status
                        print("  Waiting for training to start...")

                except Exception as e:
                    print(f"  Poll error: {e}")

                time_module.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        finally:
            # Clean up HTTP server if running
            if server_proc:
                server_proc.terminate()
                print("Dashboard server stopped.")

        # Only auto-terminate if training completed successfully or user requested it
        elapsed = time_module.time() - start_time
        cost = (elapsed / 3600) * price_per_hour

        if training_completed and not args.no_terminate:
            # Download results before terminating
            print("\n" + "=" * 50)
            print("Downloading results...")
            print("=" * 50)
            client.download_results(instance)

            print(f"\nTerminating instance {instance.id[:8]}...")
            client.terminate_instance(instance.id)
            print("Instance terminated.")
            print(f"\nFinal cost: ~${cost:.2f} ({elapsed/60:.1f} minutes)")
        else:
            print(f"\nInstance still running: {instance.ip}")
            print(f"  Current cost: ~${cost:.2f}")
            if not training_completed:
                print(f"  (Not terminating - training did not complete successfully)")
            print(f"Terminate with: python -m openadapt_ml.cloud.lambda_labs terminate {instance.id}")

    elif args.command == "train-status":
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        print(f"Checking training status on {instance.ip}...")
        status = client.get_training_status(instance)

        if status:
            print(f"  Epoch: {status.get('epoch', 'N/A')}")
            print(f"  Step: {status.get('step', 'N/A')}")
            print(f"  Loss: {status.get('loss', 'N/A')}")
            print(f"  Elapsed: {status.get('elapsed_time', 0):.1f}s")
        else:
            print("  No training log found (training may not have started yet)")

    elif args.command == "checkpoints":
        # List checkpoints on remote instance
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        print(f"Checking checkpoints on {instance.ip}...")

        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            f"ubuntu@{instance.ip}",
            "ls -la ~/openadapt-ml/checkpoints/ 2>/dev/null && "
            "du -sh ~/openadapt-ml/checkpoints/ 2>/dev/null || echo 'No checkpoints directory found'"
        ]

        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("No checkpoints found yet")
            if result.stderr:
                print(f"  Error: {result.stderr}")

    elif args.command == "refresh":
        # One-shot dashboard refresh
        import time as time_module
        from pathlib import Path
        from openadapt_ml.training.trainer import TrainingState, TrainingConfig, generate_training_dashboard

        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        # Use current job directory via symlink
        from openadapt_ml.training.trainer import get_current_job_directory, setup_job_directory
        base_dir = Path("training_output")
        base_dir.mkdir(exist_ok=True)

        status = client.get_training_status(instance)

        if status and status.get("step", 0) > 0:
            # Get or create job directory based on remote job_id
            remote_job_id = status.get("job_id", "")
            if remote_job_id:
                output_dir = setup_job_directory(base_dir, remote_job_id)
            else:
                output_dir = get_current_job_directory(base_dir) or base_dir
            dashboard_path = output_dir / "dashboard.html"
            log_path = output_dir / "training_log.json"

            # Setup screenshots symlink if local capture path provided
            local_capture = args.capture if hasattr(args, 'capture') and args.capture else None
            if local_capture:
                setup_capture_screenshots_symlink(output_dir, local_capture)

            # Rewrite evaluation paths from Lambda to relative
            if "evaluations" in status:
                status["evaluations"] = rewrite_evaluation_paths(status["evaluations"])

            # Ensure instance metadata is present
            status["instance_ip"] = instance.ip
            status["instance_type"] = instance.instance_type
            status["total_epochs"] = status.get("total_epochs", 5)

            # Save log
            log_path.write_text(json.dumps(status, indent=2))

            # Generate dashboard
            state = TrainingState(job_id=remote_job_id)
            state.job_id = remote_job_id
            state.hostname = status.get("hostname", "lambda")
            state.instance_ip = instance.ip or ""
            state.instance_type = instance.instance_type
            state.config_path = status.get("config_path", "")
            # Use local capture path for screenshots if provided, else remote path
            state.capture_path = args.capture if args.capture else status.get("capture_path", "")
            state.epoch = status.get("epoch", 0)
            state.step = status.get("step", 0)
            state.loss = status.get("loss", 0)
            state.learning_rate = status.get("learning_rate", 5e-5)
            state.losses = status.get("losses", [])
            state.evaluations = status.get("evaluations", [])
            state.total_epochs = status.get("total_epochs", 5)
            state.start_time = time_module.time() - status.get("elapsed_time", 0)
            # Cloud provider info
            state.cloud_provider = "lambda"
            state.cloud_dashboard_url = "https://cloud.lambda.ai/instances"
            state.cloud_instance_id = instance.id
            state.setup_status = status.get("setup_status", "training")
            state.setup_logs = status.get("setup_logs", [])

            config = TrainingConfig(
                num_train_epochs=status.get("total_epochs", 5),
                learning_rate=status.get("learning_rate", 5e-5)
            )

            dashboard_path.write_text(generate_training_dashboard(state, config))

            # Regenerate navigation for file:// protocol
            try:
                from openadapt_ml.training.trainer import regenerate_all_dashboards
                regenerate_all_dashboards(output_dir)
            except Exception:
                pass  # Silent fail for navigation

            epoch = status.get("epoch", 0)
            step = status.get("step", 0)
            loss = status.get("loss", 0)
            elapsed = status.get("elapsed_time", 0)
            print(f"Epoch {epoch+1}/{state.total_epochs} | Step {step} | Loss: {loss:.4f} | Elapsed: {elapsed:.0f}s")
            print(f"Dashboard: {dashboard_path.absolute()}")

            if args.open:
                import subprocess as sp
                sp.run(["open", str(dashboard_path)], capture_output=True)
        else:
            print("No training data yet")

    elif args.command == "monitor":
        # Live dashboard monitoring for Lambda training
        # Updates training_output/training_log.json so the existing dashboard auto-refreshes
        import time as time_module
        from pathlib import Path

        # Stub mode - simulate training without actual GPU
        if getattr(args, 'stub', False):
            from openadapt_ml.training.stub_provider import StubTrainingProvider
            from openadapt_ml.training.trainer import (
                TrainingState, TrainingConfig, generate_training_dashboard
            )

            print("\n[Stub Mode] Simulating training without GPU...")
            output_dir = Path("training_output")
            output_dir.mkdir(exist_ok=True)

            # Start dashboard server if requested
            server_proc = None
            if args.open:
                server_proc = open_dashboard_in_browser(output_dir)

            # Run stub training
            stub = StubTrainingProvider(
                output_dir=output_dir,
                epochs=5,
                steps_per_epoch=10,
                step_delay=0.3,  # Fast simulation
            )

            def update_dashboard(status):
                """Regenerate dashboard after each step."""
                state = TrainingState()
                state.job_id = status.get("job_id", "")
                state.hostname = status.get("hostname", "stub")
                state.instance_ip = "127.0.0.1"
                state.instance_type = "stub"
                state.epoch = status.get("epoch", 0)
                state.step = status.get("step", 0)
                state.loss = status.get("loss", 0)
                state.learning_rate = status.get("learning_rate", 5e-5)
                state.losses = status.get("losses", [])
                state.evaluations = status.get("evaluations", [])
                state.cloud_provider = "stub"
                state.setup_status = "training"

                config = TrainingConfig(
                    num_train_epochs=status.get("total_epochs", 5),
                    learning_rate=state.learning_rate
                )

                dashboard_path = output_dir / "dashboard.html"
                dashboard_path.write_text(generate_training_dashboard(state, config))

            try:
                stub.run(callback=update_dashboard)
            except KeyboardInterrupt:
                print("\n[Stub] Interrupted by user.")
            finally:
                if server_proc:
                    server_proc.terminate()
                    print("[Stub] Dashboard server stopped.")

            print(f"\n[Stub] Results in: {output_dir}")
            return

        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        if instance.status == "booting" or not instance.ip:
            print(f"Instance {instance.id[:8]} is still booting, waiting for IP...")
            while True:
                time_module.sleep(5)
                instances = client.list_instances()
                instance = next((i for i in instances if i.id == instance.id), None)
                if not instance:
                    print("Instance terminated or not found.")
                    return
                if instance.ip and instance.status == "active":
                    print(f"Instance ready at {instance.ip}")
                    break
                print(f"  Status: {instance.status}...")

        # Use job-scoped directory structure
        from openadapt_ml.training.trainer import (
            TrainingState, TrainingConfig, generate_training_dashboard,
            setup_job_directory, get_current_job_directory
        )
        base_dir = Path("training_output")
        base_dir.mkdir(exist_ok=True)

        # Get current job directory or wait for first status to determine job_id
        output_dir = get_current_job_directory(base_dir) or base_dir
        dashboard_path = output_dir / "dashboard.html"
        log_path = output_dir / "training_log.json"

        # Check for existing log with job_id
        current_job_id = None
        if log_path.exists():
            try:
                existing_log = json.loads(log_path.read_text())
                current_job_id = existing_log.get("job_id")
            except (json.JSONDecodeError, IOError):
                pass

        print(f"\nMonitoring Lambda training on {instance.ip}")
        print(f"Dashboard: {dashboard_path.absolute()}")
        print(f"Polling every {args.interval}s (Ctrl+C to stop)\n")

        # Generate initial dashboard if it doesn't exist
        if not dashboard_path.exists():
            state = TrainingState(job_id=current_job_id or "")
            state.cloud_provider = "lambda"
            state.cloud_dashboard_url = "https://cloud.lambda.ai/instances"
            state.cloud_instance_id = instance.id
            state.instance_ip = instance.ip or ""
            state.instance_type = instance.instance_type
            state.setup_status = "booting"
            state.setup_logs = ["Starting Lambda Cloud instance...", f"Instance ID: {instance.id[:8]}...", f"Instance type: {instance.instance_type}"]
            config = TrainingConfig(num_train_epochs=5, learning_rate=5e-5)
            dashboard_path.write_text(generate_training_dashboard(state, config))

        # Open dashboard if requested via HTTP server
        server_proc = None
        if args.open:
            server_proc = open_dashboard_in_browser(output_dir)

        last_step = 0
        last_epoch = -1
        auto_stop_loss = getattr(args, 'auto_stop_loss', 0.5)
        download_checkpoints = getattr(args, 'download_checkpoints', True)
        step_stall_count = 0  # Track how many times step hasn't increased

        print(f"  Auto-stop loss threshold: {auto_stop_loss}")
        print(f"  Checkpoint download: {'enabled' if download_checkpoints else 'disabled'}")

        try:
            while True:
                # Check for stop signal from dashboard
                if check_stop_signal(output_dir):
                    print("\n  Stop signal received from dashboard!")
                    print("  Downloading final checkpoints...")
                    if download_checkpoints:
                        download_checkpoints_from_instance(instance.ip, output_dir)

                    # Update status with termination info before terminating
                    termination_status = {
                        "termination_status": "user_stop",
                        "termination_message": "Training stopped by user via dashboard"
                    }
                    current_log = log_path.read_text() if log_path.exists() else "{}"
                    import json as json_module
                    current_data = json_module.loads(current_log)
                    current_data.update(termination_status)
                    log_path.write_text(json_module.dumps(current_data, indent=2))

                    print(f"  Terminating instance {instance.id}...")
                    client.terminate_instance(instance.id)
                    # Remove stop signal
                    (output_dir / "STOP_TRAINING").unlink(missing_ok=True)
                    print("  Training stopped by user.")
                    break

                try:
                    # Fetch training log from remote
                    status = client.get_training_status(instance)

                    if status and status.get("step", 0) > 0:
                        step = status.get("step", 0)
                        epoch = status.get("epoch", 0)
                        loss = status.get("loss", 0)
                        elapsed = status.get("elapsed_time", 0)
                        remote_job_id = status.get("job_id")

                        # Detect job_id change - clear old data if new job started
                        if remote_job_id and current_job_id and remote_job_id != current_job_id:
                            print(f"\n  New job detected: {remote_job_id} (was: {current_job_id})")
                            print("  Clearing old job data...")
                            last_step = 0  # Reset step tracking
                            current_job_id = remote_job_id

                        # Update local training log (dashboard polls this file)
                        # Add total_epochs to status for dashboard
                        status["total_epochs"] = status.get("total_epochs", 5)
                        # Ensure instance metadata is present
                        if not status.get("instance_ip"):
                            status["instance_ip"] = instance.ip
                        if not status.get("instance_type"):
                            status["instance_type"] = instance.instance_type
                        # Add cloud provider info
                        status["cloud_provider"] = "lambda"
                        status["cloud_dashboard_url"] = "https://cloud.lambda.ai/instances"
                        status["cloud_instance_id"] = instance.id
                        status["setup_status"] = status.get("setup_status", "training")

                        # Setup screenshots symlink if local capture path provided
                        local_capture = args.capture if hasattr(args, 'capture') and args.capture else None
                        if local_capture:
                            setup_capture_screenshots_symlink(output_dir, local_capture)

                        # Rewrite evaluation paths from Lambda to relative
                        if "evaluations" in status:
                            status["evaluations"] = rewrite_evaluation_paths(status["evaluations"])

                        log_path.write_text(json.dumps(status, indent=2))

                        if step > last_step:
                            print(f"  Epoch {epoch+1} | Step {step} | Loss: {loss:.4f} | Elapsed: {elapsed:.0f}s")
                            last_step = step
                            step_stall_count = 0  # Reset stall counter when step increases
                            if not current_job_id:
                                current_job_id = remote_job_id

                            # Regenerate dashboard with updated data
                            state = TrainingState()
                            state.job_id = status.get("job_id", "")
                            state.hostname = status.get("hostname", "lambda")
                            state.instance_ip = instance.ip or ""
                            state.instance_type = instance.instance_type
                            state.epoch = epoch
                            state.step = step
                            state.loss = loss
                            state.learning_rate = status.get("learning_rate", 5e-5)
                            state.losses = status.get("losses", [])
                            state.evaluations = status.get("evaluations", [])
                            state.start_time = time_module.time() - elapsed
                            # Cloud provider info
                            state.cloud_provider = "lambda"
                            state.cloud_dashboard_url = "https://cloud.lambda.ai/instances"
                            state.cloud_instance_id = instance.id
                            state.setup_status = status.get("setup_status", "training")
                            state.setup_logs = status.get("setup_logs", [])
                            state.termination_status = status.get("termination_status", "")
                            state.termination_message = status.get("termination_message", "")

                            config = TrainingConfig(
                                num_train_epochs=status.get("total_epochs", 5),
                                learning_rate=status.get("learning_rate", 5e-5)
                            )

                            dashboard_path.write_text(generate_training_dashboard(state, config))

                            # Download checkpoints on epoch change
                            if download_checkpoints and epoch > last_epoch:
                                print(f"  Epoch {epoch+1} completed - downloading checkpoints...")
                                if download_checkpoints_from_instance(instance.ip, output_dir):
                                    print(f"  Checkpoints saved to {output_dir}/checkpoints/")
                                else:
                                    print("  Warning: checkpoint download failed")
                                last_epoch = epoch

                            # Auto-terminate when loss is low enough
                            if loss < auto_stop_loss and loss > 0:
                                print(f"\n  Loss {loss:.4f} < threshold {auto_stop_loss}")
                                print("  Downloading final checkpoints...")
                                if download_checkpoints:
                                    download_checkpoints_from_instance(instance.ip, output_dir)

                                # Update status with termination info
                                status["termination_status"] = "auto_low_loss"
                                status["termination_message"] = f"Training auto-stopped: loss {loss:.4f} < threshold {auto_stop_loss}"
                                log_path.write_text(json.dumps(status, indent=2))

                                print(f"  Auto-terminating instance {instance.id}...")
                                client.terminate_instance(instance.id)
                                print("  Training completed (auto-stopped)!")
                                break
                        else:
                            # Step didn't increase - check if training is complete
                            step_stall_count += 1
                            total_epochs = status.get("total_epochs", 5)

                            # If on last epoch and step hasn't increased for 3 polls, training is complete
                            if epoch >= total_epochs - 1 and step_stall_count >= 3:
                                print(f"\n  Training complete (epoch {epoch+1}/{total_epochs}, step stopped increasing)")
                                print("  Downloading final checkpoints...")
                                if download_checkpoints:
                                    download_checkpoints_from_instance(instance.ip, output_dir)

                                # Update status with termination info
                                status["termination_status"] = "auto_complete"
                                status["termination_message"] = f"Training completed successfully ({epoch+1}/{total_epochs} epochs)"
                                log_path.write_text(json.dumps(status, indent=2))

                                print(f"  Terminating instance {instance.id}...")
                                client.terminate_instance(instance.id)
                                print("  Instance terminated.")
                                break

                    else:
                        print("  Waiting for training to start...")

                except Exception as e:
                    print(f"  Poll error: {e}")

                time_module.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            print(f"Dashboard: {dashboard_path.absolute()}")
        finally:
            # Clean up HTTP server if running
            if server_proc:
                server_proc.terminate()
                print("Dashboard server stopped.")

    elif args.command == "files":
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        print(f"Files on {instance.ip} at {args.path}:")
        result = client.ssh_run(instance, f"find {args.path} -type f -name '*.pt' -o -name '*.json' -o -name '*.bin' 2>/dev/null | head -20", timeout=30)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print("  (no checkpoint files found)")

    elif args.command == "kill":
        # Kill training/inference processes
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            if args.local:
                print("\nKilling local Lambda-related processes...")
                subprocess.run(
                    ["pkill", "-f", "ssh.*ubuntu@.*openadapt"],
                    capture_output=True
                )
                subprocess.run(
                    ["pkill", "-f", "lambda_labs"],
                    capture_output=True
                )
                print("Done.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        print(f"Checking processes on {instance.ip}...")

        # List Python processes first
        result = client.ssh_run(
            instance,
            "ps aux | grep python | grep -v grep | grep -v jupyter",
            timeout=30
        )
        if result.stdout.strip():
            print("Found Python processes:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line[:100]}...")
        else:
            print("No training/inference Python processes found.")
            return

        if args.all:
            print("\nKilling ALL Python processes (except jupyter)...")
            cmd = "pkill -f 'python.*train\\|python.*compare\\|python.*openadapt' || true"
        else:
            print("\nKilling training and inference processes...")
            cmd = "pkill -f 'python.*train' ; pkill -f 'python.*compare' || true"

        result = client.ssh_run(instance, cmd, timeout=30)
        print("Remote processes killed.")

        if args.local:
            print("\nKilling local Lambda-related processes...")
            subprocess.run(
                ["pkill", "-f", "ssh.*ubuntu@.*openadapt"],
                capture_output=True
            )
            subprocess.run(
                ["pkill", "-f", "lambda_labs.*train"],
                capture_output=True
            )
            print("Local processes killed.")

        print("\nDone. Current status:")
        result = client.ssh_run(
            instance,
            "ps aux | grep python | grep -v grep | grep -v jupyter | wc -l",
            timeout=30
        )
        count = result.stdout.strip()
        print(f"  {count} Python processes remaining on instance")

    elif args.command == "check":
        # Analyze training status and early stopping
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        print(f"Checking training on {instance.ip}...")

        # Get training log
        result = client.ssh_run(
            instance,
            "cat ~/openadapt-ml/training_output/training_log.json 2>/dev/null",
            timeout=30
        )

        if not result.stdout.strip():
            print("No training log found.")
            return

        try:
            data = json.loads(result.stdout)
            losses = data.get("losses", [])
        except json.JSONDecodeError:
            print("Could not parse training log.")
            return

        if not losses:
            print("No training data yet.")
            return

        total_steps = len(losses)
        epochs = sorted(set(l["epoch"] for l in losses))
        total_epochs = data.get("total_epochs", 5)
        min_loss = min(l["loss"] for l in losses)
        current_loss = losses[-1]["loss"]

        print(f"\n{'='*50}")
        print(f"TRAINING STATUS")
        print(f"{'='*50}")
        print(f"Steps: {total_steps}")
        print(f"Epochs: {max(epochs)+1}/{total_epochs}")
        print(f"Current loss: {current_loss:.4f}")
        print(f"Min loss: {min_loss:.4f}")

        # Check if training is running
        proc_result = client.ssh_run(
            instance,
            "ps aux | grep 'python.*train' | grep -v grep | wc -l",
            timeout=30
        )
        is_running = int(proc_result.stdout.strip()) > 0

        if is_running:
            print(f"Status: RUNNING")
        else:
            print(f"Status: STOPPED")

        # Early stopping analysis
        window = min(args.window, len(losses))
        if window < 2:
            print("\nNot enough data for early stopping analysis.")
        else:
            recent_losses = [l["loss"] for l in losses[-window:]]
            older_losses = [l["loss"] for l in losses[-window*2:-window]] if len(losses) >= window*2 else [l["loss"] for l in losses[:window]]

            recent_avg = sum(recent_losses) / len(recent_losses)
            older_avg = sum(older_losses) / len(older_losses) if older_losses else recent_avg

            improvement = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0
            loss_variance = max(recent_losses) - min(recent_losses)

            print(f"\n{'='*50}")
            print(f"EARLY STOPPING ANALYSIS (window={window})")
            print(f"{'='*50}")
            print(f"Recent avg loss: {recent_avg:.4f}")
            print(f"Prior avg loss: {older_avg:.4f}")
            print(f"Improvement: {improvement*100:.2f}%")
            print(f"Loss variance: {loss_variance:.4f}")

            should_stop = improvement < args.threshold and loss_variance < 0.1
            if should_stop:
                print(f"\n⚠️  EARLY STOPPING RECOMMENDED")
                print(f"   Loss has plateaued (improvement < {args.threshold*100}%)")
                if not is_running:
                    print(f"   (Training already stopped)")
                else:
                    print(f"\n   To stop: uv run python -m openadapt_ml.cloud.lambda_labs kill")
            else:
                print(f"\n✓ Training still improving, continue.")

        # Time estimate
        if is_running and len(losses) >= 2:
            avg_time_per_step = losses[-1].get("time", 0) / len(losses) if losses[-1].get("time") else 50
            steps_per_epoch = len(losses) / (max(epochs) + 1)
            remaining_epochs = total_epochs - max(epochs) - 1
            remaining_steps = remaining_epochs * steps_per_epoch
            eta_seconds = remaining_steps * avg_time_per_step
            eta_mins = eta_seconds / 60

            print(f"\n{'='*50}")
            print(f"TIME ESTIMATE")
            print(f"{'='*50}")
            print(f"Remaining epochs: {remaining_epochs}")
            print(f"Est. remaining steps: {remaining_steps:.0f}")
            print(f"ETA: {eta_mins:.1f} minutes")

    elif args.command == "compare":
        # Run comparison on Lambda and sync back
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        # Determine checkpoint to use
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        elif args.epoch is not None:
            checkpoint_path = f"/home/ubuntu/openadapt-ml/checkpoints/epoch_{args.epoch}"
        else:
            # Use latest (main checkpoint)
            checkpoint_path = "/home/ubuntu/openadapt-ml/checkpoints/qwen3vl2b_capture_lora"

        # Check if checkpoint exists
        result = client.ssh_run(
            instance,
            f"ls {checkpoint_path}/adapter_config.json 2>/dev/null && echo 'exists'",
            timeout=30
        )
        if "exists" not in result.stdout:
            print(f"Checkpoint not found at {checkpoint_path}")
            # List available checkpoints
            result = client.ssh_run(
                instance,
                "ls -la ~/openadapt-ml/checkpoints/",
                timeout=30
            )
            print(f"Available checkpoints:\n{result.stdout}")
            return

        print(f"Running comparison on {instance.ip}...")
        print(f"Using checkpoint: {checkpoint_path}")

        # Run comparison on Lambda
        output_name = f"comparison_{time.strftime('%H%M%S')}.html"
        cmd = f"""cd ~/openadapt-ml && source .venv/bin/activate && \
            python -m openadapt_ml.scripts.compare \
            --capture ~/capture \
            --checkpoint {checkpoint_path} \
            --output training_output/{output_name} 2>&1"""

        print("Generating predictions (this may take a few minutes)...")
        result = client.ssh_run(instance, cmd, timeout=600)

        if result.returncode != 0:
            print(f"Comparison failed:\n{result.stderr}")
            return

        # Check if file was created
        result = client.ssh_run(
            instance,
            f"ls -la ~/openadapt-ml/training_output/{output_name}",
            timeout=30
        )
        if result.returncode != 0:
            print("Comparison file not created.")
            return

        print(f"Comparison generated: {output_name}")

        # Sync back to local
        local_output = Path("training_output") / output_name
        local_output.parent.mkdir(parents=True, exist_ok=True)

        print(f"Syncing to {local_output}...")
        subprocess.run([
            "rsync", "-avz",
            f"ubuntu@{instance.ip}:~/openadapt-ml/training_output/{output_name}",
            str(local_output)
        ], capture_output=True)

        print(f"Done! Comparison saved to: {local_output}")

        if args.open:
            subprocess.run(["open", str(local_output)], capture_output=True)
            print("Opened in browser.")

    elif args.command == "download":
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        client.download_results(instance, local_path=args.output)

    elif args.command == "results":
        # Download results and generate comparison viewer
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        # Download results
        print("Step 1: Downloading training results...")
        client.download_results(instance)

        # Generate comparison viewer
        print("\nStep 2: Generating comparison viewer...")
        checkpoint_path = "checkpoints_lambda/qwen3vl2b_capture_lora"

        import subprocess as sp
        cmd = [
            "uv", "run", "python", "-m", "openadapt_ml.scripts.compare",
            "--capture", args.capture,
            "--checkpoint", checkpoint_path,
        ]
        if args.goal:
            cmd.extend(["--goal", args.goal])
        if args.open:
            cmd.append("--open")

        result = sp.run(cmd)
        if result.returncode == 0:
            print("\nComparison viewer generated!")
            if not args.open:
                print(f"Open with: open {args.capture}/comparison.html")
        else:
            print("Warning: Failed to generate comparison viewer")

    elif args.command == "serve":
        # Start web server for live dashboard with stop button support
        import http.server
        import socketserver
        import threading
        import time as time_module
        from pathlib import Path

        output_dir = Path(args.output) if hasattr(args, 'output') else Path("training_output")
        port = args.port

        if not output_dir.exists():
            print(f"No {output_dir} directory. Run 'refresh' first.")
            return

        # Define handler with /api/stop support
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(output_dir), **kwargs)

            def do_POST(self):
                if self.path == '/api/stop':
                    # Create stop signal file
                    stop_file = output_dir / "STOP_TRAINING"
                    stop_file.touch()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(b'{"status": "stop signal created"}')
                    print(f"  Stop signal created: {stop_file}")
                else:
                    self.send_error(404)

            def do_OPTIONS(self):
                # Handle CORS preflight
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress log messages


        # Start web server
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}/dashboard.html"
            print(f"\nDashboard server started at {url}")
            print("Press Ctrl+C to stop\n")

            if args.open:
                subprocess.run(["open", url], capture_output=True)

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")

    elif args.command == "sync":
        # Sync training output from Lambda and regenerate navigation for file:// protocol
        from pathlib import Path
        from openadapt_ml.training.trainer import (
            TrainingState, TrainingConfig, generate_training_dashboard,
            regenerate_all_dashboards
        )

        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id.startswith(args.instance_id)), None)
            if not instance:
                print(f"Instance {args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        print(f"Syncing training output from {instance.ip}...")

        # Sync all training output files
        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "-e", "ssh -o StrictHostKeyChecking=no",
            f"ubuntu@{instance.ip}:~/openadapt-ml/training_output/",
            str(output_dir) + "/"
        ]
        result = subprocess.run(rsync_cmd, capture_output=False)

        if result.returncode != 0:
            print("Warning: rsync may have had issues")

        # Update dashboard with instance metadata
        log_path = output_dir / "training_log.json"
        dashboard_path = output_dir / "dashboard.html"

        if log_path.exists():
            try:
                import time as time_module
                status = json.loads(log_path.read_text())

                # Update with instance info
                status["instance_ip"] = instance.ip
                status["instance_type"] = instance.instance_type
                status["cloud_provider"] = "lambda"
                status["cloud_dashboard_url"] = "https://cloud.lambda.ai/instances"
                status["cloud_instance_id"] = instance.id

                log_path.write_text(json.dumps(status, indent=2))

                # Generate updated dashboard
                state = TrainingState()
                state.job_id = status.get("job_id", "")
                state.hostname = status.get("hostname", "lambda")
                state.instance_ip = instance.ip or ""
                state.instance_type = instance.instance_type
                state.config_path = status.get("config_path", "")
                state.capture_path = status.get("capture_path", "")
                state.epoch = status.get("epoch", 0)
                state.step = status.get("step", 0)
                state.loss = status.get("loss", 0)
                state.learning_rate = status.get("learning_rate", 5e-5)
                state.losses = status.get("losses", [])
                state.evaluations = status.get("evaluations", [])
                state.total_epochs = status.get("total_epochs", 5)
                state.start_time = time_module.time() - status.get("elapsed_time", 0)
                state.cloud_provider = "lambda"
                state.cloud_dashboard_url = "https://cloud.lambda.ai/instances"
                state.cloud_instance_id = instance.id

                config = TrainingConfig(
                    num_train_epochs=status.get("total_epochs", 5),
                    learning_rate=status.get("learning_rate", 5e-5)
                )

                dashboard_path.write_text(generate_training_dashboard(state, config))
            except Exception as e:
                print(f"Warning: Could not update dashboard: {e}")

        # Regenerate ALL dashboards with static navigation (for file:// protocol)
        print("Regenerating navigation links...")
        try:
            regenerated = regenerate_all_dashboards(output_dir)
            print(f"  Updated {len(regenerated)} files with static navigation")
        except Exception as e:
            print(f"Warning: Navigation regeneration failed: {e}")

        # Summary
        files = list(output_dir.glob("*.html"))
        print(f"\nSynced {len(files)} HTML files to {output_dir}/")
        for f in sorted(files):
            print(f"  - {f.name}")

        print(f"\nDashboard: {dashboard_path.absolute()}")

        if args.open:
            subprocess.run(["open", str(dashboard_path)], capture_output=True)

    elif args.command == "viewer":
        # Regenerate and open local viewer (no Lambda required)
        from pathlib import Path
        from openadapt_ml.training.trainer import regenerate_all_dashboards
        import re

        output_dir = Path(args.output)

        if not output_dir.exists():
            print(f"Error: {output_dir} does not exist")
            print("Run training or sync first to populate the directory.")
            return

        if not (output_dir / "training_log.json").exists():
            print(f"Error: No training_log.json found in {output_dir}")
            print("This directory doesn't contain training results.")
            return

        # Auto-link local screenshots if available
        screenshots_link = output_dir / "screenshots"
        if not screenshots_link.exists():
            # Try to find capture ID from training log or predictions
            try:
                capture_id = None

                # First try training log
                log_data = json.loads((output_dir / "training_log.json").read_text())
                capture_path = log_data.get("capture_path", "")
                capture_match = re.search(r'capture_(\d+)', capture_path)
                if capture_match:
                    capture_id = capture_match.group(1)

                # If not found, try predictions JSON files
                if not capture_id:
                    for pred_file in output_dir.glob("predictions_*.json"):
                        pred_data = json.loads(pred_file.read_text())
                        base_data = pred_data.get("base_data", [])
                        if base_data:
                            image_path = base_data[0].get("image_path", "")
                            capture_match = re.search(r'capture_(\d+)', image_path)
                            if capture_match:
                                capture_id = capture_match.group(1)
                                break

                if capture_id:
                    # Search for local screenshots in openadapt-capture
                    openadapt_capture_dir = Path.home() / "oa" / "src" / "openadapt-capture"
                    if openadapt_capture_dir.exists():
                        for capture_dir in openadapt_capture_dir.iterdir():
                            if capture_dir.is_dir():
                                screenshots_dir = capture_dir / "screenshots"
                                if screenshots_dir.exists():
                                    # Check if this capture has our screenshots
                                    sample_file = list(screenshots_dir.glob(f"capture_{capture_id}_step_*.png"))
                                    if sample_file:
                                        print(f"Found local screenshots in {screenshots_dir}")
                                        screenshots_link.symlink_to(screenshots_dir)
                                        print(f"  Linked: {screenshots_link} -> {screenshots_dir}")
                                        break
            except Exception as e:
                pass  # Silently continue if auto-link fails

        print(f"Regenerating viewer from {output_dir}...")
        regenerated = regenerate_all_dashboards(output_dir)
        print(f"  Updated {len(regenerated)} files")

        # Show path info
        if args.dashboard:
            target = output_dir / "dashboard.html"
        else:
            target = output_dir / "viewer.html"

        print(f"\nGenerated: {target.absolute()}")
        print(f"View with: uv run python -m openadapt_ml.cloud.lambda_labs serve --open")

        if args.open:
            subprocess.run(["open", str(target)], capture_output=True)


if __name__ == "__main__":
    main()
