"""Pool management for parallel WAA benchmark evaluation.

Provides a clean Python API for creating and managing pools of Azure VMs,
distributing benchmark tasks across workers, and collecting results.

Extracted from cli.py for reuse as a library.

Example:
    from openadapt_ml.benchmarks.pool import PoolManager

    manager = PoolManager()
    pool = manager.create(workers=4)
    ready = manager.wait()
    result = manager.run(tasks=10)
    manager.cleanup(confirm=False)

    # With custom resource group:
    from openadapt_ml.benchmarks.azure_vm import AzureVMManager
    vm = AzureVMManager(resource_group="my-rg")
    manager = PoolManager(vm_manager=vm)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from openadapt_ml.benchmarks.azure_vm import (
    AzureVMManager,
    ssh_run,
    wait_for_ssh,
)
from openadapt_ml.benchmarks.vm_monitor import (
    VMConfig,
    VMMonitor,
    VMPool,
    VMPoolRegistry,
    PoolWorker,
)

logger = logging.getLogger(__name__)


@dataclass
class PoolRunResult:
    """Result of running benchmark tasks across a pool."""

    total_tasks: int
    completed: int
    failed: int
    elapsed_seconds: float
    worker_results: list[tuple[str, int, int, str | None]]
    """List of (worker_name, completed, failed, error_or_none)."""


# Docker setup script for WAA workers
DOCKER_SETUP_SCRIPT = """
set -e

# Wait for apt lock (unattended upgrades on fresh VMs)
echo "Waiting for apt lock..."
while sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 || sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    sleep 5
done
echo "Apt lock released"

sudo apt-get update -qq
sudo apt-get install -y -qq docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Configure Docker to use /mnt (larger temp disk)
sudo systemctl stop docker
sudo mkdir -p /mnt/docker
sudo bash -c 'echo "{\\"data-root\\": \\"/mnt/docker\\"}" > /etc/docker/daemon.json'
sudo systemctl start docker

# Pull base images (use sudo since usermod hasn't taken effect yet)
sudo docker pull dockurr/windows:latest
sudo docker pull windowsarena/winarena:latest

# Build waa-auto image that has working auto-download
mkdir -p /tmp/waa-build
cat > /tmp/waa-build/Dockerfile << 'DOCKERFILE_EOF'
FROM dockurr/windows:latest
COPY --from=windowsarena/winarena:latest /entry.sh /entry.sh
COPY --from=windowsarena/winarena:latest /entry_setup.sh /entry_setup.sh
COPY --from=windowsarena/winarena:latest /start_client.sh /start_client.sh
COPY --from=windowsarena/winarena:latest /client /client
COPY --from=windowsarena/winarena:latest /models /models
COPY --from=windowsarena/winarena:latest /oem /oem
COPY --from=windowsarena/winarena:latest /usr/local/bin/python* /usr/local/bin/
COPY --from=windowsarena/winarena:latest /usr/local/bin/pip* /usr/local/bin/
COPY --from=windowsarena/winarena:latest /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=windowsarena/winarena:latest /usr/local/lib/libpython3.9.so* /usr/local/lib/
RUN ldconfig && \\
    ln -sf /usr/local/bin/python3.9 /usr/local/bin/python && \\
    ln -sf /usr/local/bin/python3.9 /usr/bin/python && \\
    ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip
RUN sed -i '/^return 0$/i cp -r /oem/* /tmp/smb/ 2>/dev/null || true' /run/samba.sh
RUN printf '#!/bin/bash\\n/usr/bin/tini -s /run/entry.sh\\n' > /start_vm.sh && chmod +x /start_vm.sh
RUN sed -i 's|20.20.20.21|172.30.0.2|g' /entry_setup.sh /entry.sh /start_client.sh
RUN find /client -name "*.py" -exec sed -i 's|20.20.20.21|172.30.0.2|g' {} \\;
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ENV VERSION="11e" RAM_SIZE="8G" CPU_CORES="4"
ENTRYPOINT ["/usr/bin/tini", "-s", "/run/entry.sh"]
DOCKERFILE_EOF

sudo docker build -t waa-auto:latest /tmp/waa-build/
rm -rf /tmp/waa-build
"""

# WAA container start script
WAA_START_SCRIPT = """
# Check if container already running
if docker ps --format '{{.Names}}' | grep -q '^winarena$'; then
    echo "ALREADY_RUNNING"
    exit 0
fi

# Container not running, start it
docker rm -f winarena 2>/dev/null || true
sudo mkdir -p /mnt/waa-storage
sudo chown azureuser:azureuser /mnt/waa-storage
docker run -d --name winarena \\
  --device=/dev/kvm \\
  --cap-add NET_ADMIN \\
  --stop-timeout 120 \\
  -p 5000:5000 \\
  -p 8006:8006 \\
  -p 7200:7200 \\
  -v /mnt/waa-storage:/storage \\
  -e VERSION=11e \\
  -e RAM_SIZE=8G \\
  -e CPU_CORES=4 \\
  -e DISK_SIZE=64G \\
  --entrypoint /bin/bash \\
  windowsarena/winarena:latest \\
  -c './entry.sh --prepare-image false --start-client false'
echo "STARTED"
"""


@dataclass
class PoolManager:
    """Manages a pool of Azure VMs for parallel WAA benchmark evaluation.

    Provides the full pool lifecycle: create, wait, run, cleanup.

    Args:
        vm_manager: AzureVMManager instance (controls resource group, auth).
        registry: VMPoolRegistry for persisting pool state.
        log_fn: Optional logging function with signature log_fn(step, message).
    """

    vm_manager: AzureVMManager = field(default_factory=AzureVMManager)
    registry: VMPoolRegistry = field(default_factory=VMPoolRegistry)
    log_fn: Any = None

    def _log(self, step: str, message: str, end: str = "\n") -> None:
        """Log a message using the configured log function or print."""
        if self.log_fn:
            self.log_fn(step, message, end=end)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{step}] {message}", end=end, flush=True)

    def create(
        self,
        workers: int = 3,
        fast: bool = True,
        auto_shutdown_hours: int = 4,
    ) -> VMPool:
        """Create a pool of VMs for parallel WAA evaluation.

        Creates VMs in parallel, installs Docker, and registers the pool.

        Args:
            workers: Number of worker VMs to create.
            fast: If True, use D8 VM sizes. If False, use standard D4.
            auto_shutdown_hours: Hours until auto-shutdown (safety net).

        Returns:
            Created VMPool.

        Raises:
            RuntimeError: If no VMs could be created.
        """
        self._log("POOL", f"Creating pool with {workers} workers...")

        # Check for existing pool
        if self.registry.get_pool() is not None:
            raise RuntimeError("Pool already exists. Delete it first with: delete-pool")

        # Find available size/region
        self._log("POOL", "Finding available region and VM size...")
        vm_size, region, cost = self.vm_manager.find_available_size_and_region(
            fast=fast,
        )
        self._log("POOL", f"Using {vm_size} (${cost:.2f}/hr) in {region}")

        if auto_shutdown_hours > 0:
            self._log("POOL", f"VMs will auto-shutdown in {auto_shutdown_hours} hours")

        # Create VMs in parallel
        self._log("POOL", f"Creating {workers} VMs in parallel...")
        workers_created: list[tuple[str, str]] = []

        def create_worker(worker_idx: int) -> tuple[str, str | None, str | None]:
            name = f"waa-pool-{worker_idx:02d}"

            # Check if VM already exists
            existing_ip = self.vm_manager.get_vm_ip(name)
            if existing_ip:
                return (name, existing_ip, None)

            try:
                vm_info = self.vm_manager.create_vm(
                    name=name,
                    region=region,
                    size=vm_size,
                )
                ip = vm_info.get("publicIpAddress", "")
                self.vm_manager.set_auto_shutdown(name, auto_shutdown_hours)
                return (name, ip, None)
            except RuntimeError as e:
                return (name, None, str(e))

        with ThreadPoolExecutor(max_workers=min(workers, 5)) as executor:
            futures = {executor.submit(create_worker, i): i for i in range(workers)}
            for future in as_completed(futures):
                name, ip, error = future.result()
                if error:
                    self._log("POOL", f"  {name}: FAILED - {error}")
                else:
                    self._log("POOL", f"  {name}: {ip}")
                    workers_created.append((name, ip))

        if not workers_created:
            raise RuntimeError("No VMs created successfully")

        self._log("POOL", f"\nCreated {len(workers_created)}/{workers} VMs")

        # Wait for SSH
        self._log("POOL", "Waiting for SSH access...")
        workers_ready: list[tuple[str, str]] = []
        for name, ip in workers_created:
            if wait_for_ssh(ip, timeout=120):
                self._log("POOL", f"  {name}: SSH ready")
                workers_ready.append((name, ip))
            else:
                self._log("POOL", f"  {name}: SSH timeout")

        if not workers_ready:
            raise RuntimeError("No VMs have SSH access")

        # Install Docker on all VMs
        self._log("POOL", "Installing Docker on all VMs...")

        def setup_docker(
            name_ip: tuple[str, str],
        ) -> tuple[str, bool, str]:
            name, ip = name_ip
            result = ssh_run(ip, DOCKER_SETUP_SCRIPT, stream=False, step="DOCKER")
            error = result.stderr[:200] if result.stderr else ""
            return (name, result.returncode == 0, error)

        with ThreadPoolExecutor(max_workers=min(len(workers_ready), 5)) as executor:
            futures = {executor.submit(setup_docker, w): w[0] for w in workers_ready}
            workers_docker_ok: list[tuple[str, str]] = []
            for future in as_completed(futures):
                name, success, error = future.result()
                status = "Docker ready" if success else f"Docker FAILED: {error[:100]}"
                self._log("POOL", f"  {name}: {status}")
                if success:
                    workers_docker_ok.append((name, dict(workers_ready)[name]))

        if not workers_docker_ok:
            raise RuntimeError("Docker setup failed on all VMs")

        # Register pool
        pool = self.registry.create_pool(
            workers=workers_docker_ok,
            resource_group=self.vm_manager.resource_group,
            location=region,
            vm_size=vm_size,
        )

        self._log("POOL", "=" * 60)
        self._log("POOL", f"Pool created: {pool.pool_id}")
        self._log("POOL", f"  Workers: {len(workers_docker_ok)}")
        self._log("POOL", f"  Region: {region}")
        self._log("POOL", f"  Size: {vm_size} (${cost:.2f}/hr)")
        self._log(
            "POOL",
            f"  Est. hourly cost: ${cost * len(workers_docker_ok):.2f}/hr",
        )
        self._log("POOL", "")
        self._log("POOL", "Next steps:")
        self._log("POOL", "  1. Wait for WAA ready: pool-wait")
        self._log("POOL", "  2. Run benchmark:      pool-run --tasks 154")
        self._log("POOL", "  3. Delete pool:        delete-pool")
        self._log("POOL", "=" * 60)

        return pool

    def wait(
        self,
        timeout_minutes: int = 30,
        start_containers: bool = True,
    ) -> list[PoolWorker]:
        """Wait for all pool workers to have WAA ready.

        Optionally starts WAA containers on each worker first.

        Args:
            timeout_minutes: Maximum minutes to wait for WAA readiness.
            start_containers: If True, start WAA containers before waiting.

        Returns:
            List of ready PoolWorker instances.

        Raises:
            RuntimeError: If no pool exists.
        """
        pool = self.registry.get_pool()
        if pool is None:
            raise RuntimeError(
                "No active pool. Create one with: pool-create --workers N"
            )

        self._log("POOL-WAIT", f"Pool: {pool.pool_id} ({len(pool.workers)} workers)")

        # Start WAA containers
        if start_containers:
            self._log("POOL-WAIT", "Checking WAA containers on all workers...")

            def start_container(
                worker: PoolWorker,
            ) -> tuple[str, bool, str]:
                result = ssh_run(
                    worker.ip, WAA_START_SCRIPT, stream=False, step="START"
                )
                output = result.stdout.strip() if result.stdout else ""
                return (worker.name, result.returncode == 0, output)

            with ThreadPoolExecutor(max_workers=min(len(pool.workers), 5)) as executor:
                futures = {
                    executor.submit(start_container, w): w.name for w in pool.workers
                }
                for future in as_completed(futures):
                    name, success, output = future.result()
                    if "ALREADY_RUNNING" in output:
                        status = "already running"
                    elif "STARTED" in output:
                        status = "container started"
                    elif success:
                        status = "ok"
                    else:
                        status = "FAILED"
                    self._log("POOL-WAIT", f"  {name}: {status}")

        # Wait for WAA readiness
        self._log(
            "POOL-WAIT",
            f"Waiting for WAA server on all workers (timeout: {timeout_minutes}m)...",
        )
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        workers_pending = {w.name: w for w in pool.workers}
        workers_ready: list[PoolWorker] = []

        while workers_pending and (time.time() - start_time) < timeout_seconds:
            for name, worker in list(workers_pending.items()):
                try:
                    config = VMConfig(
                        name=name,
                        ssh_host=worker.ip,
                        internal_ip="172.30.0.2",
                    )
                    monitor = VMMonitor(config, timeout=5)
                    ready, _response = monitor.check_waa_probe()

                    if ready:
                        self._log("POOL-WAIT", f"  {name}: READY")
                        workers_ready.append(worker)
                        del workers_pending[name]
                        self.registry.update_worker(
                            name, waa_ready=True, status="ready"
                        )
                except Exception:
                    pass

            if workers_pending:
                elapsed = int(time.time() - start_time)
                pending_names = ", ".join(workers_pending.keys())
                print(
                    f"\r  [{elapsed}s] Waiting for: {pending_names}...",
                    end="",
                    flush=True,
                )
                time.sleep(10)

        print()  # New line after progress

        if workers_pending:
            self._log(
                "POOL-WAIT",
                f"TIMEOUT: {len(workers_pending)} workers not ready",
            )
            for name in workers_pending:
                self._log(
                    "POOL-WAIT",
                    f"  {name}: not ready"
                    f" (check with: ssh azureuser@{workers_pending[name].ip})",
                )

        self._log("POOL-WAIT", "=" * 60)
        self._log(
            "POOL-WAIT",
            f"Workers ready: {len(workers_ready)}/{len(pool.workers)}",
        )

        if workers_ready:
            self._log("POOL-WAIT", "")
            self._log("POOL-WAIT", "Ready to run benchmark:")
            self._log("POOL-WAIT", "  pool-run --tasks 154")

        return workers_ready

    def run(
        self,
        tasks: int = 10,
        agent: str = "navi",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        agent_factory: Callable[[], Any] | None = None,
    ) -> PoolRunResult:
        """Run benchmark tasks distributed across pool workers.

        When agent_factory is None, runs WAA's built-in agent inside the
        Docker container via docker exec. When agent_factory is provided,
        the agent runs externally and communicates via the WAA Flask API.

        Args:
            tasks: Number of tasks to run.
            agent: Agent name for WAA's run.py (default: "navi").
            model: Model name for the agent.
            api_key: API key for the agent. Auto-loaded from config if None.
            agent_factory: Optional callable that returns a BenchmarkAgent.
                When provided, overrides agent/model and runs externally.

        Returns:
            PoolRunResult with task counts and timing.

        Raises:
            RuntimeError: If no pool or no ready workers.
        """
        pool = self.registry.get_pool()
        if pool is None:
            raise RuntimeError(
                "No active pool. Create one with: pool-create --workers N"
            )

        # Load API key from config if not provided
        if not api_key:
            try:
                from openadapt_ml.config import settings

                api_key = settings.openai_api_key
            except Exception:
                pass

        if not api_key and agent_factory is None:
            raise RuntimeError(
                "No API key provided. Use api_key param or set OPENAI_API_KEY in .env"
            )

        # Get ready workers
        ready_workers = [w for w in pool.workers if w.waa_ready or w.status == "ready"]
        if not ready_workers:
            raise RuntimeError("No workers ready. Run wait() first.")

        self._log("POOL-RUN", "=" * 60)
        self._log(
            "POOL-RUN",
            f"Running WAA benchmark across {len(ready_workers)} workers",
        )
        self._log("POOL-RUN", f"  Tasks: {tasks}")
        if agent_factory:
            self._log("POOL-RUN", "  Agent: custom (external)")
        else:
            self._log("POOL-RUN", f"  Agent: {agent}")
            self._log("POOL-RUN", f"  Model: {model}")
        self._log("POOL-RUN", "=" * 60)

        # Update registry
        pool.total_tasks = tasks
        self.registry.save()

        # Create experiment name
        exp_name = datetime.now().strftime("pool_%Y%m%d_%H%M%S")
        num_workers = len(ready_workers)

        if agent_factory is not None:
            return self._run_external_agent(
                ready_workers, tasks, agent_factory, exp_name
            )

        # Default path: run WAA's built-in agent via docker exec
        def run_on_worker(
            worker: PoolWorker,
            worker_idx: int,
            total_workers: int,
        ) -> tuple[str, int, int, str | None]:
            run_cmd = f"""
docker exec -e OPENAI_API_KEY='{api_key}' winarena bash -c 'cd /client && python run.py \\
    --agent {agent} \\
    --model {model} \\
    --exp_name {exp_name}_{worker.name} \\
    --worker_id {worker_idx} \\
    --num_workers {total_workers} \\
    --emulator_ip 172.30.0.2 2>&1' | tee /home/azureuser/benchmark.log
"""
            result = ssh_run(
                worker.ip, run_cmd, stream=True, step="RUN", log_fn=self.log_fn
            )
            if result.returncode == 0:
                return (worker.name, 1, 0, None)
            else:
                return (worker.name, 0, 1, "benchmark failed")

        self._log("POOL-RUN", "")
        self._log("POOL-RUN", "Starting benchmark on all workers...")
        start_time = time.time()

        results: list[tuple[str, int, int, str | None]] = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for worker_idx, worker in enumerate(ready_workers):
                future = executor.submit(run_on_worker, worker, worker_idx, num_workers)
                futures[future] = worker.name

            for future in as_completed(futures):
                name, completed, failed, error = future.result()
                if error:
                    self._log("POOL-RUN", f"  {name}: FAILED - {error}")
                else:
                    self._log("POOL-RUN", f"  {name}: completed {completed} tasks")
                results.append((name, completed, failed, error))
                self.registry.update_pool_progress(completed=completed, failed=failed)

        elapsed = time.time() - start_time
        total_completed = sum(r[1] for r in results)
        total_failed = sum(r[2] for r in results)

        self._log("POOL-RUN", "")
        self._log("POOL-RUN", "=" * 60)
        self._log("POOL-RUN", "BENCHMARK COMPLETE")
        self._log("POOL-RUN", f"  Time: {elapsed / 60:.1f} minutes")
        self._log("POOL-RUN", f"  Completed: {total_completed}/{tasks}")
        self._log("POOL-RUN", f"  Failed: {total_failed}")
        self._log("POOL-RUN", "=" * 60)

        return PoolRunResult(
            total_tasks=tasks,
            completed=total_completed,
            failed=total_failed,
            elapsed_seconds=elapsed,
            worker_results=results,
        )

    def _run_external_agent(
        self,
        workers: list[PoolWorker],
        tasks: int,
        agent_factory: Callable[[], Any],
        exp_name: str,
    ) -> PoolRunResult:
        """Run an external agent against pool workers via Flask API.

        This is a placeholder for the full external agent integration.
        The external agent communicates with WAA's Flask server on each
        worker via SSH tunnel.

        Args:
            workers: Ready pool workers.
            tasks: Number of tasks.
            agent_factory: Callable that returns a BenchmarkAgent.
            exp_name: Experiment name for result tracking.

        Returns:
            PoolRunResult.
        """
        self._log(
            "POOL-RUN",
            "External agent support is a preview feature. "
            "Use openadapt-evals CLI for full external agent evaluation.",
        )
        # TODO: Implement SSH tunnel setup + agent loop per worker
        # For now, raise to signal this path isn't fully wired yet
        raise NotImplementedError(
            "External agent_factory support is not yet fully implemented. "
            "Use the openadapt-evals CLI with --agent api-claude for now."
        )

    def status(self) -> VMPool | None:
        """Get current pool status.

        Returns:
            VMPool if a pool exists, None otherwise.
        """
        return self.registry.get_pool()

    def cleanup(
        self,
        confirm: bool = True,
    ) -> bool:
        """Clean up orphaned pool resources (VMs, NICs, IPs, disks).

        Args:
            confirm: If True, prompt for confirmation before deleting.

        Returns:
            True if cleanup succeeded.
        """
        rg = self.vm_manager.resource_group

        self._log("POOL-CLEANUP", "Searching for orphaned pool resources...")

        # Find pool resources
        vms = self._list_pool_resources("vm", "list", rg)
        nics = self._list_pool_resources("network nic", "list", rg)
        ips = self._list_pool_resources("network public-ip", "list", rg)
        disks = self._list_pool_resources("disk", "list", rg)

        total = len(vms) + len(nics) + len(ips) + len(disks)

        if total == 0:
            self._log("POOL-CLEANUP", "No orphaned resources found.")
            return True

        self._log("POOL-CLEANUP", f"Found {total} orphaned resources:")
        if vms:
            self._log("POOL-CLEANUP", f"  VMs: {len(vms)}")
        if nics:
            self._log("POOL-CLEANUP", f"  NICs: {len(nics)}")
        if ips:
            self._log("POOL-CLEANUP", f"  Public IPs: {len(ips)}")
        if disks:
            self._log("POOL-CLEANUP", f"  Disks: {len(disks)}")

        if confirm:
            user_input = input("\nDelete these resources? [y/N]: ")
            if user_input.lower() != "y":
                self._log("POOL-CLEANUP", "Aborted.")
                return False

        # Delete VMs first (releases NICs)
        for vm in vms:
            self._log("POOL-CLEANUP", f"  Deleting VM: {vm}")
            self.vm_manager._az_run(
                [
                    "vm",
                    "delete",
                    "-g",
                    rg,
                    "-n",
                    vm,
                    "--yes",
                    "--force-deletion",
                    "true",
                ]
            )

        for nic in nics:
            self._log("POOL-CLEANUP", f"  Deleting NIC: {nic}")
            self.vm_manager._az_run(
                [
                    "network",
                    "nic",
                    "delete",
                    "-g",
                    rg,
                    "-n",
                    nic,
                ]
            )

        for ip in ips:
            self._log("POOL-CLEANUP", f"  Deleting IP: {ip}")
            self.vm_manager._az_run(
                [
                    "network",
                    "public-ip",
                    "delete",
                    "-g",
                    rg,
                    "-n",
                    ip,
                ]
            )

        for disk in disks:
            self._log("POOL-CLEANUP", f"  Deleting disk: {disk}")
            self.vm_manager._az_run(
                [
                    "disk",
                    "delete",
                    "-g",
                    rg,
                    "-n",
                    disk,
                    "--yes",
                ]
            )

        # Delete registry
        self.registry.delete_pool()

        self._log("POOL-CLEANUP", "Cleanup complete.")
        return True

    def _list_pool_resources(
        self,
        resource_type: str,
        action: str,
        resource_group: str,
    ) -> list[str]:
        """List Azure resources matching 'waa-pool' in the resource group."""
        # Split resource type for compound types like "network nic"
        type_parts = resource_type.split()
        result = self.vm_manager._az_run(
            [
                *type_parts,
                action,
                "-g",
                resource_group,
                "--query",
                "[?contains(name, 'waa-pool')].name",
                "-o",
                "tsv",
            ]
        )
        if result.returncode == 0 and result.stdout.strip():
            return [r.strip() for r in result.stdout.strip().split("\n") if r.strip()]
        return []
