"""Azure VM operations for WAA benchmark evaluation.

Provides a clean Python API for Azure VM lifecycle management.
Uses Azure Python SDK (DefaultAzureCredential) as the primary auth path,
falling back to az CLI when the SDK packages aren't installed.

This is the recommended auth pattern for enterprise environments where
users may not have subscription-level RBAC on their personal identity.
DefaultAzureCredential automatically tries (in order):
    1. Environment variables (AZURE_CLIENT_ID + SECRET + TENANT_ID)
    2. Workload identity (Kubernetes)
    3. Managed identity (Azure VMs)
    4. Azure CLI credential (az login)
    5. Azure PowerShell credential
    6. Interactive browser

Example:
    from openadapt_ml.benchmarks.azure_vm import AzureVMManager

    # Uses DefaultAzureCredential automatically
    vm = AzureVMManager(resource_group="my-rg", subscription_id="sub-123")
    ip = vm.get_vm_ip("waa-eval-vm")

    # Or pass an explicit credential
    from azure.identity import ClientSecretCredential
    cred = ClientSecretCredential(tenant_id=..., client_id=..., client_secret=...)
    vm = AzureVMManager(resource_group="my-rg", subscription_id="sub-123", credential=cred)
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# SSH options used for all VM connections
SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=60",
    "-o",
    "ServerAliveCountMax=10",
]

# VM size constants
VM_SIZE_STANDARD = "Standard_D4ds_v4"
VM_SIZE_FAST = "Standard_D8ds_v5"
VM_SIZE_FAST_FALLBACKS = [
    ("Standard_D8ds_v5", 0.38),
    ("Standard_D8s_v5", 0.36),
    ("Standard_D8ds_v4", 0.38),
    ("Standard_D8as_v5", 0.34),
]
VM_REGIONS = ["centralus", "eastus", "westus2", "eastus2"]

# Ubuntu 22.04 LTS image reference for Azure SDK
_UBUNTU_2204_IMAGE = {
    "publisher": "Canonical",
    "offer": "0001-com-ubuntu-server-jammy",
    "sku": "22_04-lts-gen2",
    "version": "latest",
}


def _default_resource_group() -> str:
    """Get default resource group from config."""
    try:
        from openadapt_ml.config import settings

        return settings.azure_resource_group
    except Exception:
        return "openadapt-agents"


def _default_subscription_id() -> str | None:
    """Get default subscription ID from config."""
    try:
        from openadapt_ml.config import settings

        return settings.azure_subscription_id
    except Exception:
        return None


def _get_credential():
    """Get Azure credential using the same pattern as benchmarks/azure.py.

    Priority:
        1. Service principal (if AZURE_CLIENT_ID + SECRET + TENANT_ID set)
        2. DefaultAzureCredential (CLI login, managed identity, etc.)
    """
    from openadapt_ml.config import settings

    try:
        from azure.identity import (
            ClientSecretCredential,
            DefaultAzureCredential,
        )
    except ImportError:
        return None

    if all(
        [
            settings.azure_client_id,
            settings.azure_client_secret,
            settings.azure_tenant_id,
        ]
    ):
        logger.info("Using service principal authentication for VM operations")
        return ClientSecretCredential(
            tenant_id=settings.azure_tenant_id,
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
        )

    logger.info("Using DefaultAzureCredential for VM operations")
    return DefaultAzureCredential()


def _sdk_available() -> bool:
    """Check if Azure management SDK packages are installed."""
    try:
        import azure.mgmt.compute  # noqa: F401
        import azure.mgmt.network  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class AzureVMManager:
    """Manages Azure VMs for WAA benchmark evaluation.

    Uses Azure Python SDK when available (recommended for enterprise),
    falling back to az CLI subprocess calls.

    Args:
        resource_group: Azure resource group name.
        subscription_id: Azure subscription ID. Required for SDK path.
            Auto-loaded from AZURE_SUBSCRIPTION_ID env var if not provided.
        credential: Optional Azure SDK credential. If None, uses
            DefaultAzureCredential (tries service principal, CLI, managed
            identity automatically).
    """

    resource_group: str = field(default_factory=_default_resource_group)
    subscription_id: str | None = field(default_factory=_default_subscription_id)
    credential: Any = None

    def __post_init__(self) -> None:
        self._compute_client = None
        self._network_client = None
        self._use_sdk = _sdk_available() and self.subscription_id is not None

    def _get_compute_client(self):
        """Lazy-load Azure ComputeManagementClient."""
        if self._compute_client is None:
            from azure.mgmt.compute import ComputeManagementClient

            cred = self.credential or _get_credential()
            self._compute_client = ComputeManagementClient(cred, self.subscription_id)
        return self._compute_client

    def _get_network_client(self):
        """Lazy-load Azure NetworkManagementClient."""
        if self._network_client is None:
            from azure.mgmt.network import NetworkManagementClient

            cred = self.credential or _get_credential()
            self._network_client = NetworkManagementClient(cred, self.subscription_id)
        return self._network_client

    def _az_run(
        self,
        args: list[str],
        capture_output: bool = True,
        text: bool = True,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run az CLI command (fallback when SDK not available)."""
        return subprocess.run(
            ["az", *args],
            capture_output=capture_output,
            text=text,
            **kwargs,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def get_vm_ip(self, name: str = "waa-eval-vm") -> Optional[str]:
        """Get VM public IP address.

        Args:
            name: VM name.

        Returns:
            Public IP string, or None if VM doesn't exist.
        """
        if self._use_sdk:
            return self._sdk_get_vm_ip(name)
        return self._cli_get_vm_ip(name)

    def get_vm_state(self, name: str = "waa-eval-vm") -> Optional[str]:
        """Get VM power state.

        Args:
            name: VM name.

        Returns:
            Power state string (e.g., "VM running"), or None.
        """
        if self._use_sdk:
            return self._sdk_get_vm_state(name)
        return self._cli_get_vm_state(name)

    def create_vm(
        self,
        name: str,
        region: str,
        size: str,
        image: str = "Ubuntu2204",
        admin_username: str = "azureuser",
    ) -> dict[str, Any]:
        """Create an Azure VM.

        Args:
            name: VM name.
            region: Azure region.
            size: VM size (e.g., "Standard_D4ds_v4").
            image: OS image (used by CLI path; SDK uses Ubuntu 22.04 LTS).
            admin_username: Admin user name.

        Returns:
            VM info dict with at least "publicIpAddress" key.

        Raises:
            RuntimeError: If VM creation fails.
        """
        if self._use_sdk:
            return self._sdk_create_vm(name, region, size, admin_username)
        return self._cli_create_vm(name, region, size, image, admin_username)

    def delete_vm(self, name: str) -> bool:
        """Delete VM and associated resources (NIC, public IP, disk).

        Args:
            name: VM name.

        Returns:
            True if deletion succeeded.
        """
        if self._use_sdk:
            return self._sdk_delete_vm(name)
        return self._cli_delete_vm(name)

    def set_auto_shutdown(
        self,
        name: str,
        hours: int = 4,
    ) -> bool:
        """Set Azure auto-shutdown policy on a VM.

        Safety net to prevent orphaned VMs from running indefinitely.
        Uses azure-mgmt-resource SDK (generic resource API) when available,
        falling back to az CLI.

        Args:
            name: VM name.
            hours: Hours from now when VM should auto-shutdown.

        Returns:
            True if auto-shutdown was set successfully.
        """
        if self._use_sdk:
            return self._sdk_set_auto_shutdown(name, hours)
        return self._cli_set_auto_shutdown(name, hours)

    def _sdk_set_auto_shutdown(self, name: str, hours: int) -> bool:
        """Set auto-shutdown via Azure SDK (generic resource API).

        Auto-shutdown is a Microsoft.DevTestLab/schedules resource, not part
        of the compute SDK. We use azure-mgmt-resource's generic resource
        client to create it.
        """
        try:
            from azure.mgmt.resource import ResourceManagementClient
        except ImportError:
            logger.debug("azure-mgmt-resource not available, falling back to CLI")
            return self._cli_set_auto_shutdown(name, hours)

        try:
            cred = self.credential or _get_credential()
            resource_client = ResourceManagementClient(cred, self.subscription_id)

            shutdown_time = datetime.utcnow() + timedelta(hours=hours)
            shutdown_time_str = shutdown_time.strftime("%H%M")

            # Get the VM to find its location
            compute = self._get_compute_client()
            vm = compute.virtual_machines.get(self.resource_group, name)

            schedule_name = f"shutdown-computevm-{name}"
            vm_id = (
                f"/subscriptions/{self.subscription_id}"
                f"/resourceGroups/{self.resource_group}"
                f"/providers/Microsoft.Compute/virtualMachines/{name}"
            )

            resource_client.resources.begin_create_or_update_by_id(
                resource_id=(
                    f"/subscriptions/{self.subscription_id}"
                    f"/resourceGroups/{self.resource_group}"
                    f"/providers/Microsoft.DevTestLab/schedules/{schedule_name}"
                ),
                api_version="2018-09-15",
                parameters={
                    "location": vm.location,
                    "properties": {
                        "status": "Enabled",
                        "taskType": "ComputeVmShutdownTask",
                        "dailyRecurrence": {"time": shutdown_time_str},
                        "timeZoneId": "UTC",
                        "targetResourceId": vm_id,
                    },
                },
            ).result()

            return True
        except Exception as e:
            logger.warning(f"SDK set_auto_shutdown failed for {name}: {e}")
            # Fall back to CLI
            return self._cli_set_auto_shutdown(name, hours)

    def _cli_set_auto_shutdown(self, name: str, hours: int) -> bool:
        """Set auto-shutdown via az CLI."""
        shutdown_time = datetime.utcnow() + timedelta(hours=hours)
        shutdown_time_str = shutdown_time.strftime("%H:%M")

        result = self._az_run(
            [
                "vm",
                "auto-shutdown",
                "-g",
                self.resource_group,
                "-n",
                name,
                "--time",
                shutdown_time_str,
            ]
        )
        return result.returncode == 0

    def find_available_size_and_region(
        self,
        fast: bool = True,
    ) -> tuple[str, str, float]:
        """Find a working VM size and region by creating a test VM.

        Tries size/region combinations until one succeeds, then cleans up
        the test VM.

        Args:
            fast: If True, try D8 sizes first. If False, use standard D4.

        Returns:
            Tuple of (vm_size, region, cost_per_hour).

        Raises:
            RuntimeError: If no available size/region found.
        """
        if fast:
            sizes_to_try = VM_SIZE_FAST_FALLBACKS
        else:
            sizes_to_try = [(VM_SIZE_STANDARD, 0.19)]

        test_vm_to_cleanup = None
        try:
            for vm_size, cost in sizes_to_try:
                for region in VM_REGIONS:
                    test_name = f"waa-pool-test-{int(time.time())}"
                    test_vm_to_cleanup = test_name
                    try:
                        self.create_vm(
                            name=test_name,
                            region=region,
                            size=vm_size,
                        )
                        self.delete_vm(test_name)
                        test_vm_to_cleanup = None
                        return (vm_size, region, cost)
                    except RuntimeError:
                        test_vm_to_cleanup = None  # Creation failed
        finally:
            if test_vm_to_cleanup:
                self.delete_vm(test_vm_to_cleanup)

        raise RuntimeError(
            "No available VM size/region found. "
            "Check quota: uv run python -m openadapt_ml.benchmarks.cli azure-ml-quota"
        )

    # =========================================================================
    # Azure SDK implementations
    # =========================================================================

    def _sdk_get_vm_ip(self, name: str) -> Optional[str]:
        """Get VM public IP via Azure SDK."""
        try:
            compute = self._get_compute_client()
            vm = compute.virtual_machines.get(
                self.resource_group, name, expand="instanceView"
            )
            # Walk NIC -> IP config -> public IP
            for nic_ref in vm.network_profile.network_interfaces or []:
                nic_name = nic_ref.id.split("/")[-1]
                network = self._get_network_client()
                nic = network.network_interfaces.get(self.resource_group, nic_name)
                for ip_config in nic.ip_configurations or []:
                    if ip_config.public_ip_address:
                        pip_name = ip_config.public_ip_address.id.split("/")[-1]
                        pip = network.public_ip_addresses.get(
                            self.resource_group, pip_name
                        )
                        if pip.ip_address:
                            return pip.ip_address
        except Exception as e:
            logger.debug(f"SDK get_vm_ip failed for {name}: {e}")
        return None

    def _sdk_get_vm_state(self, name: str) -> Optional[str]:
        """Get VM power state via Azure SDK."""
        try:
            compute = self._get_compute_client()
            vm = compute.virtual_machines.instance_view(self.resource_group, name)
            for status in vm.statuses or []:
                if status.code and status.code.startswith("PowerState/"):
                    return status.display_status
        except Exception as e:
            logger.debug(f"SDK get_vm_state failed for {name}: {e}")
        return None

    def _sdk_create_vm(
        self,
        name: str,
        region: str,
        size: str,
        admin_username: str = "azureuser",
    ) -> dict[str, Any]:
        """Create VM via Azure SDK."""
        network = self._get_network_client()
        compute = self._get_compute_client()

        # Read local SSH public key
        ssh_pub_key_path = Path.home() / ".ssh" / "id_rsa.pub"
        if not ssh_pub_key_path.exists():
            raise RuntimeError(
                f"SSH public key not found at {ssh_pub_key_path}. "
                "Run: ssh-keygen -t rsa -b 4096"
            )
        ssh_pub_key = ssh_pub_key_path.read_text().strip()

        # Create public IP
        pip_name = f"{name}PublicIP"
        pip_poller = network.public_ip_addresses.begin_create_or_update(
            self.resource_group,
            pip_name,
            {
                "location": region,
                "sku": {"name": "Standard"},
                "public_ip_allocation_method": "Static",
            },
        )
        pip = pip_poller.result()

        # Create NIC
        nic_name = f"{name}VMNic"

        # Get or create a subnet — use default VNet naming
        vnet_name = f"{name}-vnet"
        subnet_name = "default"

        # Create VNet + Subnet
        vnet_poller = network.virtual_networks.begin_create_or_update(
            self.resource_group,
            vnet_name,
            {
                "location": region,
                "address_space": {"address_prefixes": ["10.0.0.0/16"]},
                "subnets": [{"name": subnet_name, "address_prefix": "10.0.0.0/24"}],
            },
        )
        vnet = vnet_poller.result()
        subnet_id = vnet.subnets[0].id

        nic_poller = network.network_interfaces.begin_create_or_update(
            self.resource_group,
            nic_name,
            {
                "location": region,
                "ip_configurations": [
                    {
                        "name": "ipconfig1",
                        "subnet": {"id": subnet_id},
                        "public_ip_address": {"id": pip.id},
                    }
                ],
            },
        )
        nic = nic_poller.result()

        # Create VM
        vm_params = {
            "location": region,
            "hardware_profile": {"vm_size": size},
            "storage_profile": {
                "image_reference": _UBUNTU_2204_IMAGE,
                "os_disk": {
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "Premium_LRS"},
                },
            },
            "os_profile": {
                "computer_name": name,
                "admin_username": admin_username,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": f"/home/{admin_username}/.ssh/authorized_keys",
                                "key_data": ssh_pub_key,
                            }
                        ]
                    },
                },
            },
            "network_profile": {
                "network_interfaces": [{"id": nic.id}],
            },
        }

        try:
            vm_poller = compute.virtual_machines.begin_create_or_update(
                self.resource_group, name, vm_params
            )
            vm_poller.result()
        except Exception as e:
            # Clean up on failure
            try:
                network.network_interfaces.begin_delete(
                    self.resource_group, nic_name
                ).result()
                network.public_ip_addresses.begin_delete(
                    self.resource_group, pip_name
                ).result()
                network.virtual_networks.begin_delete(
                    self.resource_group, vnet_name
                ).result()
            except Exception:
                pass
            raise RuntimeError(f"VM creation failed: {e}") from e

        return {"publicIpAddress": pip.ip_address, "name": name}

    def _sdk_delete_vm(self, name: str) -> bool:
        """Delete VM and associated resources via Azure SDK."""
        compute = self._get_compute_client()
        network = self._get_network_client()

        try:
            # Delete VM (blocking)
            compute.virtual_machines.begin_delete(self.resource_group, name).result()
        except Exception as e:
            logger.debug(f"SDK VM delete failed for {name}: {e}")
            return False

        # Best-effort cleanup of associated resources
        for cleanup_fn, resource_name in [
            (network.network_interfaces.begin_delete, f"{name}VMNic"),
            (network.public_ip_addresses.begin_delete, f"{name}PublicIP"),
            (network.virtual_networks.begin_delete, f"{name}-vnet"),
        ]:
            try:
                cleanup_fn(self.resource_group, resource_name).result()
            except Exception:
                pass

        return True

    # =========================================================================
    # az CLI fallback implementations
    # =========================================================================

    def _cli_get_vm_ip(self, name: str) -> Optional[str]:
        """Get VM public IP via az CLI."""
        result = self._az_run(
            [
                "vm",
                "show",
                "-d",
                "-g",
                self.resource_group,
                "-n",
                name,
                "--query",
                "publicIps",
                "-o",
                "tsv",
            ]
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    def _cli_get_vm_state(self, name: str) -> Optional[str]:
        """Get VM power state via az CLI."""
        result = self._az_run(
            [
                "vm",
                "get-instance-view",
                "-g",
                self.resource_group,
                "-n",
                name,
                "--query",
                "instanceView.statuses[1].displayStatus",
                "-o",
                "tsv",
            ]
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    def _cli_create_vm(
        self,
        name: str,
        region: str,
        size: str,
        image: str,
        admin_username: str,
    ) -> dict[str, Any]:
        """Create VM via az CLI."""
        result = self._az_run(
            [
                "vm",
                "create",
                "--resource-group",
                self.resource_group,
                "--name",
                name,
                "--location",
                region,
                "--image",
                image,
                "--size",
                size,
                "--admin-username",
                admin_username,
                "--generate-ssh-keys",
                "--public-ip-sku",
                "Standard",
            ]
        )

        if result.returncode != 0:
            error_msg = result.stderr or "unknown error"
            try:
                error_json = json.loads(error_msg)
                if "error" in error_json:
                    error_msg = error_json["error"].get("message", error_msg)[:500]
            except json.JSONDecodeError:
                error_msg = error_msg[:500]
            raise RuntimeError(f"VM creation failed: {error_msg}")

        return json.loads(result.stdout)

    def _cli_delete_vm(self, name: str) -> bool:
        """Delete VM and associated resources via az CLI."""
        result = self._az_run(
            [
                "vm",
                "delete",
                "-g",
                self.resource_group,
                "-n",
                name,
                "--yes",
                "--force-deletion",
                "true",
            ]
        )
        self._az_run(
            [
                "network",
                "nic",
                "delete",
                "-g",
                self.resource_group,
                "-n",
                f"{name}VMNic",
            ]
        )
        self._az_run(
            [
                "network",
                "public-ip",
                "delete",
                "-g",
                self.resource_group,
                "-n",
                f"{name}PublicIP",
            ]
        )
        return result.returncode == 0


# =========================================================================
# SSH utilities (not Azure-specific, but always needed for VM access)
# =========================================================================


def ssh_run(
    ip: str,
    cmd: str,
    stream: bool = False,
    step: str = "SSH",
    log_fn: Any = None,
) -> subprocess.CompletedProcess:
    """Run command on VM via SSH.

    When stream=True, runs command with output redirected to a persistent
    log file on the VM, streaming locally in real-time.

    Args:
        ip: VM public IP address.
        cmd: Command to execute on the VM.
        stream: If True, stream output in real-time.
        step: Log prefix for output lines.
        log_fn: Optional logging function (signature: log_fn(step, message)).
            If None, uses print.

    Returns:
        CompletedProcess with return code and output.
    """

    def _log(step: str, message: str, end: str = "\n"):
        if log_fn:
            log_fn(step, message, end=end)
        else:
            print(f"[{step}] {message}", end=end, flush=True)

    if stream:
        remote_log_dir = "/home/azureuser/cli_logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_log = f"{remote_log_dir}/{step.lower()}_{timestamp}.log"

        subprocess.run(
            ["ssh", *SSH_OPTS, f"azureuser@{ip}", f"mkdir -p {remote_log_dir}"],
            capture_output=True,
        )

        _log(step, f"Remote log: {remote_log}")

        wrapped_cmd = f"""
set -o pipefail
{{
  {cmd}
  echo $? > {remote_log}.exit
}} 2>&1 | tee {remote_log}
"""
        full_cmd = ["ssh", *SSH_OPTS, f"azureuser@{ip}", wrapped_cmd]

        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    clean_line = line.rstrip()
                    if "\r" in clean_line:
                        parts = clean_line.split("\r")
                        clean_line = parts[-1].strip()
                    if clean_line:
                        _log(step, clean_line)
            process.wait()
        except KeyboardInterrupt:
            _log(step, "Interrupted - command continues on VM")
            _log(step, f"View full log: ssh azureuser@{ip} 'cat {remote_log}'")
            process.terminate()
            return subprocess.CompletedProcess(cmd, 130, "", "")

        result = subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                f"azureuser@{ip}",
                f"cat {remote_log}.exit 2>/dev/null || echo 1",
            ],
            capture_output=True,
            text=True,
        )
        exit_code = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 1

        if exit_code != 0:
            _log(step, f"Command failed (exit {exit_code})")
            _log(step, f"Full log: ssh azureuser@{ip} 'cat {remote_log}'")

        return subprocess.CompletedProcess(cmd, exit_code, "", "")
    else:
        full_cmd = ["ssh", *SSH_OPTS, f"azureuser@{ip}", cmd]
        return subprocess.run(full_cmd, capture_output=True, text=True)


def wait_for_ssh(ip: str, timeout: int = 120) -> bool:
    """Wait for SSH to become available on a VM.

    Args:
        ip: VM public IP address.
        timeout: Maximum seconds to wait.

    Returns:
        True if SSH is reachable within timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["ssh", *SSH_OPTS, f"azureuser@{ip}", "echo ok"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    return False
