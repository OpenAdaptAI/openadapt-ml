"""Vast.ai cloud GPU integration.

Vast.ai is a GPU marketplace with competitive pricing:
- A10 24GB: ~$0.17/hour
- RTX 4090 24GB: ~$0.20/hour
- A100 80GB: ~$0.80/hour

API docs: https://docs.vast.ai/api

Usage:
    # Set API key
    export VAST_API_KEY=your_key_here

    # List available GPU offers
    python -m openadapt_ml.cloud.vast_ai list

    # Launch instance
    python -m openadapt_ml.cloud.vast_ai launch --gpu A10 --max-price 0.30

    # Check running instances
    python -m openadapt_ml.cloud.vast_ai status

    # Terminate instance
    python -m openadapt_ml.cloud.vast_ai terminate <instance_id>

    # Full training pipeline
    python -m openadapt_ml.cloud.vast_ai train --bundle /path/to/bundle
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# Vast.ai API base URL
API_BASE = "https://console.vast.ai/api/v0"

# Default Docker image for training instances
DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"

# Default disk size in GB
DEFAULT_DISK_GB = 50


@dataclass
class VastOffer:
    """An available GPU offer on the Vast.ai marketplace."""

    id: int
    gpu_name: str
    num_gpus: int
    gpu_ram_gb: float
    cpu_cores: int
    ram_gb: float
    disk_gb: float
    dph_total: float  # dollars per hour
    dlperf: float  # deep learning performance score
    reliability: float
    inet_down: float  # download bandwidth Mbps
    inet_up: float  # upload bandwidth Mbps
    geolocation: str

    @property
    def price_per_hour(self) -> float:
        return self.dph_total

    def __str__(self) -> str:
        return (
            f"#{self.id}: ${self.dph_total:.3f}/hr | "
            f"{self.num_gpus}x {self.gpu_name} ({self.gpu_ram_gb:.0f}GB) | "
            f"{self.cpu_cores} CPUs | {self.ram_gb:.0f}GB RAM | "
            f"{self.disk_gb:.0f}GB disk | "
            f"DLPerf: {self.dlperf:.1f} | "
            f"Rel: {self.reliability:.2f} | "
            f"{self.geolocation}"
        )


@dataclass
class VastInstance:
    """A running Vast.ai instance."""

    id: int
    status: str
    gpu_name: str
    ssh_host: str
    ssh_port: int
    ip: str | None = None
    actual_status: str | None = None
    image: str | None = None
    num_gpus: int = 1
    gpu_ram_gb: float = 0.0
    dph_total: float = 0.0
    disk_gb: float = 0.0

    def __str__(self) -> str:
        status_str = self.actual_status or self.status
        ip_str = self.ip or self.ssh_host or "pending"
        return (
            f"#{self.id} | {self.num_gpus}x {self.gpu_name} | "
            f"{status_str} | {ip_str}:{self.ssh_port} | "
            f"${self.dph_total:.3f}/hr"
        )


class VastAIClient:
    """Client for the Vast.ai REST API."""

    def __init__(self, api_key: str | None = None):
        # Try provided key, then settings, then env var
        if not api_key:
            from openadapt_ml.config import settings

            api_key = settings.vast_api_key or os.environ.get("VAST_API_KEY")

        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "Vast.ai API key required. Set VAST_API_KEY in .env file "
                "or get one at https://cloud.vast.ai/api/"
            )
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Make GET request to API."""
        resp = self.session.get(f"{API_BASE}{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _put(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """Make PUT request to API."""
        resp = self.session.put(f"{API_BASE}{endpoint}", json=data or {})
        if not resp.ok:
            raise RuntimeError(f"API error ({resp.status_code}): {resp.text}")
        return resp.json()

    def _delete(self, endpoint: str) -> Any:
        """Make DELETE request to API."""
        resp = self.session.delete(f"{API_BASE}{endpoint}")
        if not resp.ok:
            raise RuntimeError(f"API error ({resp.status_code}): {resp.text}")
        # Some DELETE endpoints return empty body
        if resp.text:
            return resp.json()
        return {}

    def list_offers(
        self,
        gpu_type: str | None = None,
        min_vram: float | None = None,
        max_price: float | None = None,
        num_gpus: int = 1,
        min_reliability: float = 0.9,
        order_by: str = "dph_total",
        limit: int = 20,
    ) -> list[VastOffer]:
        """Search available GPU offers on the marketplace.

        Args:
            gpu_type: GPU name filter (e.g., "A10", "RTX 4090", "A100")
            min_vram: Minimum GPU VRAM in GB
            max_price: Maximum price per hour in dollars
            num_gpus: Number of GPUs required
            min_reliability: Minimum reliability score (0-1)
            order_by: Sort field (dph_total, dlperf, etc.)
            limit: Maximum number of offers to return

        Returns:
            List of matching offers, sorted by price
        """
        # Build query for Vast.ai search API
        query: dict[str, Any] = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "num_gpus": {"eq": num_gpus},
            "reliability2": {"gte": min_reliability},
            "type": "on-demand",
        }

        if gpu_type:
            query["gpu_name"] = {"eq": gpu_type}
        if min_vram:
            query["gpu_ram"] = {"gte": min_vram * 1024}  # API uses MB
        if max_price:
            query["dph_total"] = {"lte": max_price}

        payload = {
            "q": query,
            "order": [[order_by, "asc"]],
            "limit": limit,
        }

        data = self._put("/search/asks/", payload)

        offers = []
        for o in data.get("offers", []):
            offers.append(
                VastOffer(
                    id=o.get("id", 0),
                    gpu_name=o.get("gpu_name", "unknown"),
                    num_gpus=o.get("num_gpus", 1),
                    gpu_ram_gb=o.get("gpu_ram", 0) / 1024,
                    cpu_cores=o.get("cpu_cores_effective", 0),
                    ram_gb=o.get("cpu_ram", 0) / 1024,
                    disk_gb=o.get("disk_space", 0),
                    dph_total=o.get("dph_total", 0),
                    dlperf=o.get("dlperf", 0),
                    reliability=o.get("reliability2", 0),
                    inet_down=o.get("inet_down", 0),
                    inet_up=o.get("inet_up", 0),
                    geolocation=o.get("geolocation", "unknown"),
                )
            )

        return offers

    def create_instance(
        self,
        offer_id: int,
        image: str = DEFAULT_IMAGE,
        disk_gb: int = DEFAULT_DISK_GB,
        onstart: str | None = None,
    ) -> VastInstance:
        """Create (rent) an instance from a marketplace offer.

        Args:
            offer_id: The offer ID to rent
            image: Docker image to use
            disk_gb: Disk space in GB
            onstart: Optional startup script

        Returns:
            The created instance
        """
        payload: dict[str, Any] = {
            "image": image,
            "disk": disk_gb,
            "client_id": "me",
        }
        if onstart:
            payload["onstart"] = onstart

        data = self._put(f"/asks/{offer_id}/", payload)

        if not data.get("success"):
            raise RuntimeError(
                f"Failed to create instance: {data.get('msg', 'unknown error')}"
            )

        instance_id = data.get("new_contract")
        if not instance_id:
            raise RuntimeError("No instance ID returned from API")

        print(f"Instance #{instance_id} created, waiting for it to start...")

        # Poll until instance is running and has SSH info
        for attempt in range(120):  # Wait up to 10 minutes
            instances = self.list_instances()
            for inst in instances:
                if inst.id == instance_id:
                    if inst.ssh_host and inst.ssh_port > 0:
                        if inst.actual_status == "running":
                            print(f"Instance ready: {inst.ssh_host}:{inst.ssh_port}")
                            return inst
                        elif attempt % 6 == 5:
                            print(
                                f"  Status: {inst.actual_status or inst.status} "
                                f"({(attempt + 1) * 5}s elapsed)..."
                            )
            time.sleep(5)

        raise RuntimeError("Timed out waiting for instance to start")

    def list_instances(self) -> list[VastInstance]:
        """List all current instances (running, loading, etc.)."""
        data = self._get("/instances/", params={"owner": "me"})

        instances = []
        for inst in data.get("instances", []):
            ssh_host = inst.get("ssh_host", "")
            ssh_port = inst.get("ssh_port", 0)

            # Parse public IP from ssh_host if available
            ip = ssh_host if ssh_host else inst.get("public_ipaddr")

            instances.append(
                VastInstance(
                    id=inst.get("id", 0),
                    status=inst.get("intended_status", "unknown"),
                    gpu_name=inst.get("gpu_name", "unknown"),
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    ip=ip,
                    actual_status=inst.get("actual_status"),
                    image=inst.get("image_uuid"),
                    num_gpus=inst.get("num_gpus", 1),
                    gpu_ram_gb=inst.get("gpu_ram", 0) / 1024,
                    dph_total=inst.get("dph_total", 0),
                    disk_gb=inst.get("disk_space", 0),
                )
            )

        return instances

    def destroy_instance(self, instance_id: int) -> bool:
        """Destroy (terminate) an instance.

        Args:
            instance_id: Instance ID to destroy

        Returns:
            True if successful
        """
        try:
            data = self._delete(f"/instances/{instance_id}/")
            return data.get("success", True)
        except RuntimeError:
            return False

    def _ssh_opts(self, instance: VastInstance) -> list[str]:
        """Build SSH command options for an instance."""
        if not instance.ssh_host or not instance.ssh_port:
            raise RuntimeError(
                f"Instance #{instance.id} has no SSH info "
                f"(host={instance.ssh_host}, port={instance.ssh_port})"
            )
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(instance.ssh_port),
            f"root@{instance.ssh_host}",
        ]

    def ssh_run(
        self,
        instance: VastInstance,
        command: str,
        timeout: int | None = None,
        retries: int = 3,
    ) -> subprocess.CompletedProcess:
        """Run a command on an instance via SSH.

        Args:
            instance: Instance to run on
            command: Shell command to run
            timeout: Optional timeout in seconds
            retries: Number of retries on connection failure

        Returns:
            CompletedProcess with stdout/stderr
        """
        ssh_cmd = self._ssh_opts(instance) + [command]

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

    def wait_for_ssh(self, instance: VastInstance, max_wait: int = 300) -> bool:
        """Wait for SSH to become available on an instance.

        Args:
            instance: Instance to connect to
            max_wait: Maximum wait time in seconds

        Returns:
            True if SSH is ready
        """
        print(f"Waiting for SSH on {instance.ssh_host}:{instance.ssh_port}...")
        deadline = time.time() + max_wait

        while time.time() < deadline:
            try:
                result = subprocess.run(
                    self._ssh_opts(instance) + ["echo ready"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0 and "ready" in result.stdout:
                    print("SSH ready!")
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass

            elapsed = int(time.time() + max_wait - deadline)
            if elapsed % 30 < 5:
                print(f"  Still waiting for SSH ({elapsed}s elapsed)...")
            time.sleep(5)

        print("Warning: SSH may not be ready yet, continuing anyway...")
        return False

    def rsync_to(
        self,
        instance: VastInstance,
        local_path: str,
        remote_path: str,
        retries: int = 3,
    ) -> bool:
        """Upload files to instance via rsync.

        Args:
            instance: Instance to upload to
            local_path: Local source path
            remote_path: Remote destination path
            retries: Number of retry attempts

        Returns:
            True if successful
        """
        ssh_opts = (
            f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-o ConnectTimeout=30 -o ServerAliveInterval=60 "
            f"-p {instance.ssh_port}"
        )

        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "--timeout=120",
            "-e",
            ssh_opts,
            f"{local_path}/",
            f"root@{instance.ssh_host}:{remote_path}/",
        ]

        for attempt in range(retries):
            result = subprocess.run(rsync_cmd)
            if result.returncode == 0:
                return True
            if attempt < retries - 1:
                print(f"  Rsync failed, retrying ({attempt + 1}/{retries})...")
                time.sleep(5)

        return False

    def rsync_from(
        self,
        instance: VastInstance,
        remote_path: str,
        local_path: str,
        retries: int = 3,
    ) -> bool:
        """Download files from instance via rsync.

        Args:
            instance: Instance to download from
            remote_path: Remote source path
            local_path: Local destination path
            retries: Number of retry attempts

        Returns:
            True if successful
        """
        ssh_opts = (
            f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-o ConnectTimeout=30 -o ServerAliveInterval=60 "
            f"-p {instance.ssh_port}"
        )

        Path(local_path).mkdir(parents=True, exist_ok=True)

        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e",
            ssh_opts,
            f"root@{instance.ssh_host}:{remote_path}/",
            f"{local_path}/",
        ]

        for attempt in range(retries):
            result = subprocess.run(rsync_cmd)
            if result.returncode == 0:
                return True
            if attempt < retries - 1:
                print(f"  Rsync failed, retrying ({attempt + 1}/{retries})...")
                time.sleep(5)

        return False

    def setup_instance(
        self,
        instance: VastInstance,
        repo_url: str = "https://github.com/OpenAdaptAI/openadapt-ml.git",
    ) -> bool:
        """Set up training environment on instance.

        Installs uv, clones repo, syncs dependencies.
        Vast.ai instances run as root in Docker containers.

        Returns:
            True if successful
        """
        print(f"Setting up instance #{instance.id}...")

        setup_script = f"""
set -e

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Install git if not present (some Docker images lack it)
if ! command -v git &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq git openssh-client rsync > /dev/null 2>&1
fi

# Clone or update repo
if [ ! -d "/root/openadapt-ml" ]; then
    git clone {repo_url} /root/openadapt-ml
else
    cd /root/openadapt-ml && git pull origin main && cd /root
fi

cd /root/openadapt-ml

# Remove uv.sources (local path deps that don't exist on remote)
sed -i '/\\[tool.uv.sources\\]/,/^$/d' pyproject.toml 2>/dev/null || true

uv sync
echo "SETUP_COMPLETE"
"""

        try:
            result = self.ssh_run(instance, setup_script, timeout=900)

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

    def upload_capture(
        self,
        instance: VastInstance,
        local_path: str,
        remote_path: str = "/root/capture",
    ) -> bool:
        """Upload training data to instance.

        Args:
            instance: Instance to upload to
            local_path: Local path to capture/bundle directory
            remote_path: Remote destination path

        Returns:
            True if successful
        """
        print(f"Uploading {local_path} to #{instance.id}:{remote_path}...")
        return self.rsync_to(instance, local_path, remote_path)

    def run_training(
        self,
        instance: VastInstance,
        config: str = "configs/qwen3vl_capture.yaml",
        capture: str | None = None,
        goal: str | None = None,
        jsonl: str | None = None,
        background: bool = True,
    ) -> subprocess.Popen | subprocess.CompletedProcess:
        """Run training on instance.

        Args:
            instance: Instance to train on
            config: Config file path (relative to repo)
            capture: Remote capture path
            goal: Task goal description
            jsonl: Remote path to JSONL training data
            background: Run in background (returns Popen) or foreground

        Returns:
            Popen if background=True, CompletedProcess if background=False
        """
        # Build training command
        train_cmd = (
            f"uv run python -m openadapt_ml.scripts.train "
            f"--config {shlex.quote(config)}"
        )
        if jsonl:
            train_cmd += f" --jsonl {shlex.quote(jsonl)}"
        elif capture:
            train_cmd += f" --capture {shlex.quote(capture)}"
        if goal:
            train_cmd += f" --goal {shlex.quote(goal)}"

        script = f"""
cd /root/openadapt-ml
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
{train_cmd}
"""

        ssh_cmd = self._ssh_opts(instance) + [script]

        print(f"Running training on #{instance.id}...")
        print(f"  Config: {config}")
        if capture:
            print(f"  Capture: {capture}")
        if jsonl:
            print(f"  JSONL: {jsonl}")

        if background:
            return subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            return subprocess.run(ssh_cmd)

    def get_training_status(self, instance: VastInstance) -> dict:
        """Check training status by reading training_log.json on instance.

        Returns:
            Parsed training log dict, or empty dict if not available
        """
        result = self.ssh_run(
            instance,
            "cat /root/openadapt-ml/training_output/training_log.json 2>/dev/null || echo '{}'",
            timeout=15,
        )
        try:
            return json.loads(result.stdout.strip())
        except Exception:
            return {}

    def download_results(
        self,
        instance: VastInstance,
        remote_path: str = "/root/openadapt-ml",
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
        print(f"Downloading results from #{instance.id}...")
        success = True

        if include_logs:
            print("  Downloading training logs...")
            if self.rsync_from(
                instance,
                f"{remote_path}/training_output",
                f"{local_path}/training_output_vast",
            ):
                print("    Training logs downloaded to training_output_vast/")
            else:
                print("    Warning: Failed to download logs")
                success = False

        if include_checkpoint:
            print("  Downloading checkpoint...")
            if self.rsync_from(
                instance,
                f"{remote_path}/checkpoints",
                f"{local_path}/checkpoints_vast",
            ):
                print("    Checkpoint downloaded to checkpoints_vast/")
            else:
                print("    Warning: Failed to download checkpoint (may not exist yet)")

        return success


def main():
    """CLI for Vast.ai GPU management."""
    import argparse

    parser = argparse.ArgumentParser(description="Vast.ai GPU management")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # --- list command ---
    list_parser = subparsers.add_parser("list", help="List available GPU offers")
    list_parser.add_argument(
        "--gpu", "-g", help="GPU type filter (e.g., A10, 'RTX 4090', A100)"
    )
    list_parser.add_argument(
        "--max-price", type=float, help="Maximum price per hour in dollars"
    )
    list_parser.add_argument("--min-vram", type=float, help="Minimum GPU VRAM in GB")
    list_parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs (default: 1)"
    )
    list_parser.add_argument(
        "--limit", type=int, default=20, help="Max results (default: 20)"
    )

    # --- status command ---
    subparsers.add_parser("status", help="Show running instances")

    # --- launch command ---
    launch_parser = subparsers.add_parser("launch", help="Create a GPU instance")
    launch_parser.add_argument(
        "--gpu", "-g", default="A10", help="GPU type (default: A10)"
    )
    launch_parser.add_argument(
        "--max-price", type=float, default=0.50, help="Max price/hr (default: $0.50)"
    )
    launch_parser.add_argument("--min-vram", type=float, help="Minimum GPU VRAM in GB")
    launch_parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Docker image (default: {DEFAULT_IMAGE})",
    )
    launch_parser.add_argument(
        "--disk",
        type=int,
        default=DEFAULT_DISK_GB,
        help=f"Disk space in GB (default: {DEFAULT_DISK_GB})",
    )
    launch_parser.add_argument(
        "--offer-id",
        type=int,
        help="Specific offer ID (skips search)",
    )

    # --- terminate command ---
    term_parser = subparsers.add_parser("terminate", help="Terminate an instance")
    term_parser.add_argument("instance_id", type=int, help="Instance ID to terminate")

    # --- ssh command ---
    ssh_parser = subparsers.add_parser("ssh", help="SSH into instance or run command")
    ssh_parser.add_argument(
        "instance_id",
        nargs="?",
        type=int,
        help="Instance ID (uses first if not specified)",
    )
    ssh_parser.add_argument(
        "--cmd", "-c", help="Command to run (prints SSH command if not specified)"
    )
    ssh_parser.add_argument(
        "--timeout", "-t", type=int, default=60, help="Command timeout (default: 60s)"
    )

    # --- train command ---
    train_parser = subparsers.add_parser("train", help="Run training on Vast.ai GPU")
    train_parser.add_argument("--capture", "-c", help="Local path to capture directory")
    train_parser.add_argument("--goal", "-g", help="Task goal description")
    train_parser.add_argument(
        "--config",
        default="configs/qwen3vl_capture_4bit.yaml",
        help="Config file (default: 4bit for memory efficiency)",
    )
    train_parser.add_argument(
        "--gpu", default="A10", help="GPU type to search for (default: A10)"
    )
    train_parser.add_argument(
        "--max-price",
        type=float,
        default=0.50,
        help="Max price per hour (default: $0.50)",
    )
    train_parser.add_argument("--min-vram", type=float, help="Minimum GPU VRAM in GB")
    train_parser.add_argument(
        "--instance",
        "-i",
        type=int,
        help="Use existing instance ID instead of launching new",
    )
    train_parser.add_argument(
        "--bundle",
        "-b",
        help="Local bundle directory (with training_data.jsonl + images/) to upload",
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
        help="Pre-computed screenshot_mapping.json (optional, auto-generated if omitted)",
    )
    train_parser.add_argument(
        "--gpu-wait",
        type=int,
        default=0,
        metavar="MINUTES",
        help="Wait up to N minutes for GPU availability (0=fail immediately, default: 0)",
    )
    train_parser.add_argument(
        "--no-terminate",
        action="store_true",
        help="Don't terminate instance after training",
    )
    train_parser.add_argument(
        "--max-runtime",
        type=int,
        default=60,
        help="Max runtime in minutes before auto-terminate (default: 60)",
    )
    train_parser.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard in browser when training starts",
    )

    # --- train-status command ---
    train_status_parser = subparsers.add_parser(
        "train-status", help="Check training status on instance"
    )
    train_status_parser.add_argument(
        "instance_id", nargs="?", type=int, help="Instance ID"
    )

    # --- download command ---
    download_parser = subparsers.add_parser(
        "download", help="Download training results from instance"
    )
    download_parser.add_argument("instance_id", nargs="?", type=int, help="Instance ID")
    download_parser.add_argument(
        "--output", "-o", default=".", help="Local output directory"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        client = VastAIClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nGet your API key at https://cloud.vast.ai/api/")
        print("Then set it: export VAST_API_KEY=your_key_here")
        return

    if args.command == "list":
        print("Searching Vast.ai marketplace...\n")
        offers = client.list_offers(
            gpu_type=args.gpu,
            max_price=args.max_price,
            min_vram=args.min_vram,
            num_gpus=args.num_gpus,
            limit=args.limit,
        )
        if not offers:
            print("No offers found matching criteria.")
            print("Try: --max-price 1.00 or a different --gpu type")
        else:
            for o in offers:
                print(f"  {o}")
            print(f"\nFound {len(offers)} offers")
            print(
                "\nLaunch with: python -m openadapt_ml.cloud.vast_ai launch "
                "--offer-id <ID>"
            )

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
        if args.offer_id:
            offer_id = args.offer_id
        else:
            print(f"Searching for {args.gpu} GPU (max ${args.max_price:.2f}/hr)...")
            offers = client.list_offers(
                gpu_type=args.gpu,
                max_price=args.max_price,
                min_vram=args.min_vram,
            )
            if not offers:
                print("No offers found matching criteria.")
                print("Try: --max-price 1.00 or a different --gpu type")
                return
            offer = offers[0]  # Cheapest
            offer_id = offer.id
            print(f"Selected offer: {offer}")

        print(f"\nCreating instance from offer #{offer_id}...")
        instance = client.create_instance(
            offer_id=offer_id,
            image=args.image,
            disk_gb=args.disk,
        )
        print("\nInstance created!")
        print(f"  ID: {instance.id}")
        print(f"  GPU: {instance.num_gpus}x {instance.gpu_name}")
        print(f"  SSH: ssh -p {instance.ssh_port} root@{instance.ssh_host}")
        print(f"  Price: ${instance.dph_total:.3f}/hr")
        print(
            f"\nTerminate with: python -m openadapt_ml.cloud.vast_ai "
            f"terminate {instance.id}"
        )

    elif args.command == "terminate":
        if client.destroy_instance(args.instance_id):
            print(f"Instance #{args.instance_id} terminated.")
        else:
            print(f"Failed to terminate instance #{args.instance_id}")

    elif args.command == "ssh":
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id == args.instance_id), None)
            if not instance:
                print(f"Instance #{args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        if args.cmd:
            print(f"Running on #{instance.id}: {args.cmd}")
            result = client.ssh_run(instance, args.cmd, timeout=args.timeout)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"[stderr] {result.stderr}", file=sys.stderr)
            if result.returncode != 0:
                sys.exit(result.returncode)
        else:
            # Print SSH command for interactive use
            print(
                f"ssh -p {instance.ssh_port} "
                f"-o StrictHostKeyChecking=no "
                f"root@{instance.ssh_host}"
            )

    elif args.command == "train":
        import time as time_module

        instance = None
        start_time = time_module.time()
        training_completed = False

        # --- Step 0: Auto-convert demos to bundle if --demo-dir provided ---
        if args.demo_dir and not args.bundle:
            from openadapt_ml.training.convert_demos import prepare_bundle

            if not args.captures_dir:
                print("Error: --captures-dir is required with --demo-dir")
                return

            print("=" * 50)
            print("Step 0: Converting demos to training bundle")
            print("=" * 50)
            try:
                bundle_dir = prepare_bundle(
                    demo_dir=args.demo_dir,
                    captures_dir=args.captures_dir,
                    mapping_path=args.mapping,
                )
                args.bundle = str(bundle_dir)
                print(f"Bundle ready at: {bundle_dir}\n")
            except Exception as e:
                print(f"Error converting demos: {e}")
                return

        # Get or launch instance
        if args.instance:
            instances = client.list_instances()
            instance = next((i for i in instances if i.id == args.instance), None)
            if not instance:
                print(f"Error: Instance #{args.instance} not found")
                return
        else:
            # Check for existing instances
            instances = client.list_instances()
            running = [i for i in instances if i.actual_status == "running"]
            if running:
                print(f"Using existing instance: #{running[0].id}")
                instance = running[0]
            else:
                # Search and launch new instance
                gpu_wait = getattr(args, "gpu_wait", 0)
                deadline = time_module.time() + gpu_wait * 60

                while True:
                    print(
                        f"Searching for {args.gpu} GPU "
                        f"(max ${args.max_price:.2f}/hr)..."
                    )
                    offers = client.list_offers(
                        gpu_type=args.gpu,
                        max_price=args.max_price,
                        min_vram=args.min_vram,
                    )

                    if offers:
                        offer = offers[0]
                        print(f"Selected: {offer}")
                        instance = client.create_instance(
                            offer_id=offer.id,
                            image=DEFAULT_IMAGE,
                            disk_gb=DEFAULT_DISK_GB,
                        )
                        print(
                            f"Instance #{instance.id} created at "
                            f"{instance.ssh_host}:{instance.ssh_port}"
                        )
                        break

                    if gpu_wait <= 0:
                        print("No offers found matching criteria.")
                        return

                    remaining = deadline - time_module.time()
                    if remaining <= 0:
                        print(f"Error: No GPU available after {gpu_wait} min wait")
                        return

                    mins_left = int(remaining / 60)
                    print(
                        f"  No offers found, retrying in 60s "
                        f"({mins_left} min remaining)..."
                    )
                    time_module.sleep(60)

        price_per_hour = instance.dph_total or 0.50
        print(f"  GPU: {instance.num_gpus}x {instance.gpu_name}")
        print(f"  Price: ~${price_per_hour:.3f}/hr")
        print(f"  Max runtime: {args.max_runtime} minutes")

        try:
            # Wait for SSH
            client.wait_for_ssh(instance)

            # Set up environment with retries
            setup_success = False
            for setup_attempt in range(3):
                print(f"Setup attempt {setup_attempt + 1}/3...")
                if client.setup_instance(instance):
                    setup_success = True
                    break
                if setup_attempt < 2:
                    print(
                        f"  Setup attempt {setup_attempt + 1} failed, "
                        f"retrying in 30s..."
                    )
                    time_module.sleep(30)

            if not setup_success:
                print("\nError: Failed to set up instance after 3 attempts")
                print(f"Debug via: ssh -p {instance.ssh_port} root@{instance.ssh_host}")
                print(
                    f"Terminate with: python -m openadapt_ml.cloud.vast_ai "
                    f"terminate {instance.id}"
                )
                return

            # Upload capture if provided
            remote_capture = None
            if args.capture:
                if client.upload_capture(instance, args.capture, "/root/capture"):
                    remote_capture = "/root/capture"
                    print(f"Capture uploaded to #{instance.id}:/root/capture")
                else:
                    print("\nError: Failed to upload capture after retries")
                    return

            # Upload bundle if provided
            remote_jsonl = None
            if args.bundle:
                if client.upload_capture(instance, args.bundle, "/root/training_data"):
                    remote_jsonl = "/root/training_data/training_data.jsonl"
                    print(f"Bundle uploaded to #{instance.id}:/root/training_data")
                else:
                    print("\nError: Failed to upload bundle after retries")
                    return

            # Start training
            print("\n" + "=" * 50)
            print("Starting training...")
            print("=" * 50 + "\n")

            client.run_training(
                instance,
                config=args.config,
                capture=remote_capture,
                goal=args.goal,
                jsonl=remote_jsonl,
                background=True,
            )

            # Poll for training status
            poll_interval = 10
            sync_interval = 300
            last_step = 0
            last_epoch = 0
            last_sync_time = time_module.time()
            log_path = Path("training_output_vast") / "training_log.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            print(
                f"Polling training status every {poll_interval}s (Ctrl+C to stop)...\n"
            )

            while True:
                try:
                    elapsed = time_module.time() - start_time
                    if elapsed > args.max_runtime * 60:
                        print(f"\nMax runtime ({args.max_runtime} min) exceeded.")
                        break

                    status = client.get_training_status(instance)

                    if status and status.get("step", 0) > 0:
                        step = status.get("step", 0)
                        epoch = status.get("epoch", 0)
                        loss = status.get("loss", 0)
                        elapsed_training = status.get("elapsed_time", 0)
                        total_epochs = status.get("total_epochs", 5)

                        if step > last_step or epoch > last_epoch:
                            print(
                                f"  Epoch {epoch + 1}/{total_epochs} | "
                                f"Step {step} | "
                                f"Loss: {loss:.4f} | "
                                f"Elapsed: {elapsed_training:.0f}s"
                            )
                            last_step = step
                            last_epoch = epoch

                        # Update local log
                        status["cloud_provider"] = "vast"
                        status["cloud_instance_id"] = instance.id
                        log_path.write_text(json.dumps(status, indent=2))

                        # Check if training is complete
                        if epoch >= total_epochs - 1:
                            time_module.sleep(poll_interval)
                            new_status = client.get_training_status(instance)
                            if new_status and new_status.get("step", 0) == step:
                                print("\n" + "=" * 50)
                                print("Training complete!")
                                print("=" * 50)
                                training_completed = True
                                break
                    else:
                        print("  Waiting for training to start...")

                except Exception as e:
                    print(f"  Poll error: {e}")

                # Periodic sync of training log
                now = time_module.time()
                if now - last_sync_time >= sync_interval:
                    try:
                        ssh_opts = (
                            f"ssh -o StrictHostKeyChecking=no "
                            f"-o UserKnownHostsFile=/dev/null "
                            f"-o ConnectTimeout=10 "
                            f"-p {instance.ssh_port}"
                        )
                        subprocess.run(
                            [
                                "rsync",
                                "-az",
                                "-e",
                                ssh_opts,
                                f"root@{instance.ssh_host}:/root/openadapt-ml/training_output/training_log.json",
                                str(log_path),
                            ],
                            capture_output=True,
                            timeout=30,
                        )
                    except Exception:
                        pass
                    last_sync_time = now

                time_module.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")

        # Wrap up
        elapsed = time_module.time() - start_time
        cost = (elapsed / 3600) * price_per_hour

        if training_completed and not args.no_terminate:
            print("\n" + "=" * 50)
            print("Downloading results...")
            print("=" * 50)
            client.download_results(instance)

            print(f"\nTerminating instance #{instance.id}...")
            client.destroy_instance(instance.id)
            print("Instance terminated.")
            print(f"\nFinal cost: ~${cost:.2f} ({elapsed / 60:.1f} minutes)")
        else:
            print(
                f"\nInstance still running: "
                f"ssh -p {instance.ssh_port} root@{instance.ssh_host}"
            )
            print(f"  Current cost: ~${cost:.2f}")
            if not training_completed:
                print("  (Not terminating - training did not complete successfully)")
            print(
                f"Terminate with: python -m openadapt_ml.cloud.vast_ai "
                f"terminate {instance.id}"
            )

    elif args.command == "train-status":
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id == args.instance_id), None)
            if not instance:
                print(f"Instance #{args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        print(f"Checking training status on #{instance.id}...")
        status = client.get_training_status(instance)

        if status and status.get("step"):
            print(f"  Epoch: {status.get('epoch', 'N/A')}")
            print(f"  Step: {status.get('step', 'N/A')}")
            print(f"  Loss: {status.get('loss', 'N/A')}")
            print(f"  Elapsed: {status.get('elapsed_time', 0):.1f}s")
        else:
            print("  No training log found (training may not have started yet)")

    elif args.command == "download":
        instances = client.list_instances()
        if not instances:
            print("No running instances.")
            return

        if args.instance_id:
            instance = next((i for i in instances if i.id == args.instance_id), None)
            if not instance:
                print(f"Instance #{args.instance_id} not found.")
                return
        else:
            instance = instances[0]

        client.download_results(instance, local_path=args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
