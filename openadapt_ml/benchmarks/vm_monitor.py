"""VM monitoring utilities for WAA benchmark evaluation.

This module provides reusable classes for monitoring Windows VMs running WAA.
Can be used by the viewer, CLI, or as a standalone tool.

Usage:
    # Monitor a single VM
    from openadapt_ml.benchmarks.vm_monitor import VMMonitor, VMConfig

    config = VMConfig(
        name="azure-waa-vm",
        ssh_host="172.171.112.41",
        ssh_user="azureuser",
        docker_container="winarena",
        internal_ip="20.20.20.21",
    )

    monitor = VMMonitor(config)
    status = monitor.check_status()
    print(f"VNC: {status.vnc_reachable}, WAA: {status.waa_ready}")

    # Or run continuous monitoring
    monitor.run_monitor(callback=lambda s: print(s))
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable
import urllib.request
import urllib.error
import socket


@dataclass
class VMConfig:
    """Configuration for a WAA VM."""

    name: str
    ssh_host: str
    ssh_user: str = "azureuser"
    vnc_port: int = 8006
    waa_port: int = 5000
    qmp_port: int = 7200
    docker_container: str = "winarena"
    internal_ip: str = "20.20.20.21"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VMConfig:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VMStatus:
    """Status of a WAA VM at a point in time."""

    config: VMConfig
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    ssh_reachable: bool = False
    vnc_reachable: bool = False
    waa_ready: bool = False
    waa_probe_response: str | None = None
    container_running: bool = False
    container_logs: str | None = None
    disk_usage_gb: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config.to_dict(),
            "timestamp": self.timestamp,
            "ssh_reachable": self.ssh_reachable,
            "vnc_reachable": self.vnc_reachable,
            "waa_ready": self.waa_ready,
            "waa_probe_response": self.waa_probe_response,
            "container_running": self.container_running,
            "container_logs": self.container_logs,
            "disk_usage_gb": self.disk_usage_gb,
            "error": self.error,
        }


class VMMonitor:
    """Monitor a single WAA VM."""

    def __init__(self, config: VMConfig, timeout: int = 5):
        """Initialize monitor.

        Args:
            config: VM configuration.
            timeout: Timeout in seconds for network operations.
        """
        self.config = config
        self.timeout = timeout

    def check_vnc(self) -> bool:
        """Check if VNC port is reachable."""
        try:
            url = f"http://{self.config.ssh_host}:{self.config.vnc_port}/"
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=self.timeout):
                return True
        except (urllib.error.URLError, socket.timeout, Exception):
            return False

    def check_ssh(self) -> bool:
        """Check if SSH is reachable."""
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", f"ConnectTimeout={self.timeout}",
                    "-o", "BatchMode=yes",
                    f"{self.config.ssh_user}@{self.config.ssh_host}",
                    "echo ok",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, Exception):
            return False

    def check_waa_probe(self) -> tuple[bool, str | None]:
        """Check if WAA /probe endpoint responds.

        Returns:
            Tuple of (ready, response_text).
        """
        try:
            cmd = f"curl -s --connect-timeout {self.timeout} http://{self.config.internal_ip}:{self.config.waa_port}/probe"
            result = subprocess.run(
                [
                    "ssh",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", f"ConnectTimeout={self.timeout}",
                    "-o", "BatchMode=yes",
                    f"{self.config.ssh_user}@{self.config.ssh_host}",
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout + 10,
            )
            response = result.stdout.strip()
            if response and "error" not in response.lower():
                return True, response
            return False, response or None
        except (subprocess.TimeoutExpired, Exception) as e:
            return False, str(e)

    def get_container_status(self) -> tuple[bool, str | None]:
        """Check container status and get recent logs.

        Returns:
            Tuple of (running, last_log_lines).
        """
        try:
            cmd = f"docker ps -q -f name={self.config.docker_container}"
            result = subprocess.run(
                [
                    "ssh",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", f"ConnectTimeout={self.timeout}",
                    "-o", "BatchMode=yes",
                    f"{self.config.ssh_user}@{self.config.ssh_host}",
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )
            running = bool(result.stdout.strip())

            if running:
                # Get last few log lines
                log_cmd = f"docker logs {self.config.docker_container} 2>&1 | tail -5"
                log_result = subprocess.run(
                    [
                        "ssh",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", f"ConnectTimeout={self.timeout}",
                        "-o", "BatchMode=yes",
                        f"{self.config.ssh_user}@{self.config.ssh_host}",
                        log_cmd,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 10,
                )
                return True, log_result.stdout.strip()
            return False, None
        except (subprocess.TimeoutExpired, Exception) as e:
            return False, str(e)

    def get_disk_usage(self) -> float | None:
        """Get disk usage of data.img in GB."""
        try:
            # Try common paths
            paths = [
                "/home/azureuser/waa-storage/data.img",
                "/home/ubuntu/waa-storage/data.img",
                "/storage/data.img",
            ]
            for path in paths:
                cmd = f"du -b {path} 2>/dev/null | cut -f1"
                result = subprocess.run(
                    [
                        "ssh",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", f"ConnectTimeout={self.timeout}",
                        "-o", "BatchMode=yes",
                        f"{self.config.ssh_user}@{self.config.ssh_host}",
                        cmd,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        bytes_size = int(result.stdout.strip())
                        return round(bytes_size / (1024 ** 3), 2)
                    except ValueError:
                        continue
            return None
        except (subprocess.TimeoutExpired, Exception):
            return None

    def check_status(self) -> VMStatus:
        """Perform full status check on the VM.

        Returns:
            VMStatus with all checks performed.
        """
        status = VMStatus(config=self.config)

        try:
            # Check VNC first (fastest, no SSH needed)
            status.vnc_reachable = self.check_vnc()

            # Check SSH
            status.ssh_reachable = self.check_ssh()

            if status.ssh_reachable:
                # Check container
                status.container_running, status.container_logs = self.get_container_status()

                # Check WAA probe
                status.waa_ready, status.waa_probe_response = self.check_waa_probe()

                # Get disk usage
                status.disk_usage_gb = self.get_disk_usage()
        except Exception as e:
            status.error = str(e)

        return status

    def run_monitor(
        self,
        callback: Callable[[VMStatus], None] | None = None,
        interval: int = 30,
        stop_on_ready: bool = True,
        output_file: str | Path | None = None,
    ) -> VMStatus:
        """Run continuous monitoring until WAA is ready.

        Args:
            callback: Optional callback function called with each status update.
            interval: Seconds between checks.
            stop_on_ready: Stop monitoring when WAA is ready.
            output_file: Optional file to write status updates (JSON lines).

        Returns:
            Final VMStatus (typically when WAA is ready).
        """
        output_path = Path(output_file) if output_file else None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            status = self.check_status()

            # Call callback if provided
            if callback:
                callback(status)

            # Write to file if provided
            if output_path:
                with open(output_path, "a") as f:
                    f.write(json.dumps(status.to_dict()) + "\n")

            # Check if we should stop
            if stop_on_ready and status.waa_ready:
                return status

            time.sleep(interval)


class VMRegistry:
    """Manage a registry of VMs and their status."""

    def __init__(self, registry_file: str | Path = "benchmark_results/vm_registry.json"):
        """Initialize registry.

        Args:
            registry_file: Path to JSON registry file.
        """
        self.registry_file = Path(registry_file)
        self._vms: list[VMConfig] = []
        self.load()

    def load(self) -> None:
        """Load VMs from registry file."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                self._vms = [VMConfig.from_dict(vm) for vm in data]

    def save(self) -> None:
        """Save VMs to registry file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump([vm.to_dict() for vm in self._vms], f, indent=2)

    def add(self, config: VMConfig) -> None:
        """Add a VM to the registry."""
        # Remove existing VM with same name
        self._vms = [vm for vm in self._vms if vm.name != config.name]
        self._vms.append(config)
        self.save()

    def remove(self, name: str) -> bool:
        """Remove a VM from the registry.

        Returns:
            True if VM was found and removed.
        """
        original_len = len(self._vms)
        self._vms = [vm for vm in self._vms if vm.name != name]
        if len(self._vms) < original_len:
            self.save()
            return True
        return False

    def get(self, name: str) -> VMConfig | None:
        """Get a VM by name."""
        for vm in self._vms:
            if vm.name == name:
                return vm
        return None

    def list(self) -> list[VMConfig]:
        """List all VMs."""
        return list(self._vms)

    def check_all(self, timeout: int = 5) -> list[VMStatus]:
        """Check status of all VMs.

        Args:
            timeout: Timeout per VM check.

        Returns:
            List of VMStatus for each registered VM.
        """
        statuses = []
        for config in self._vms:
            monitor = VMMonitor(config, timeout=timeout)
            statuses.append(monitor.check_status())
        return statuses


def main():
    """CLI entry point for VM monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor WAA VMs")
    parser.add_argument("--host", help="SSH host")
    parser.add_argument("--user", default="azureuser", help="SSH user")
    parser.add_argument("--container", default="winarena", help="Docker container name")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--output", help="Output file for status updates (JSON lines)")
    parser.add_argument("--list", action="store_true", help="List all registered VMs")
    parser.add_argument("--check-all", action="store_true", help="Check all registered VMs")

    args = parser.parse_args()

    if args.list:
        registry = VMRegistry()
        for vm in registry.list():
            print(f"  {vm.name}: {vm.ssh_user}@{vm.ssh_host} (container: {vm.docker_container})")
        return

    if args.check_all:
        registry = VMRegistry()
        for status in registry.check_all():
            print(f"\n{status.config.name}:")
            print(f"  SSH: {'✓' if status.ssh_reachable else '✗'}")
            print(f"  VNC: {'✓' if status.vnc_reachable else '✗'}")
            print(f"  WAA: {'✓ READY' if status.waa_ready else '✗ Not ready'}")
            if status.disk_usage_gb:
                print(f"  Disk: {status.disk_usage_gb} GB")
        return

    if not args.host:
        parser.error("--host is required for monitoring")

    config = VMConfig(
        name="cli-vm",
        ssh_host=args.host,
        ssh_user=args.user,
        docker_container=args.container,
    )

    monitor = VMMonitor(config)

    def print_status(status: VMStatus):
        ts = datetime.now().strftime("%H:%M:%S")
        waa_str = "READY!" if status.waa_ready else "not ready"
        disk_str = f"{status.disk_usage_gb}GB" if status.disk_usage_gb else "?"
        print(f"[{ts}] SSH: {'✓' if status.ssh_reachable else '✗'} | "
              f"VNC: {'✓' if status.vnc_reachable else '✗'} | "
              f"WAA: {waa_str} | Disk: {disk_str}")
        if status.container_logs:
            # Show last log line
            last_line = status.container_logs.split('\n')[-1][:80]
            print(f"         Log: {last_line}")

    print(f"Monitoring {args.host}... (Ctrl+C to stop)")
    try:
        final_status = monitor.run_monitor(
            callback=print_status,
            interval=args.interval,
            output_file=args.output,
        )
        print(f"\n✓ WAA is ready! Probe response: {final_status.waa_probe_response}")
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
