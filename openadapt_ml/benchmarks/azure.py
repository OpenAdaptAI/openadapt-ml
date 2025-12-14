"""Azure deployment automation for WAA benchmark.

This module provides Azure VM orchestration for running Windows Agent Arena
at scale across multiple parallel VMs.

Requirements:
    - azure-ai-ml
    - azure-identity
    - Azure subscription with ML workspace

Example:
    from openadapt_ml.benchmarks.azure import AzureWAAOrchestrator, AzureConfig

    config = AzureConfig(
        subscription_id="your-subscription-id",
        resource_group="agents",
        workspace_name="agents_ml",
    )
    orchestrator = AzureWAAOrchestrator(config, waa_repo_path="/path/to/WAA")

    # Run evaluation on 40 parallel VMs
    results = orchestrator.run_evaluation(
        agent=my_agent,
        num_workers=40,
        task_ids=None,  # All tasks
    )
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openadapt_ml.benchmarks.agent import BenchmarkAgent
from openadapt_ml.benchmarks.base import BenchmarkResult, BenchmarkTask

logger = logging.getLogger(__name__)


@dataclass
class AzureConfig:
    """Azure configuration for WAA deployment.

    Attributes:
        subscription_id: Azure subscription ID.
        resource_group: Resource group containing ML workspace.
        workspace_name: Azure ML workspace name.
        vm_size: VM size for compute instances (must support nested virtualization).
        idle_timeout_minutes: Auto-shutdown after idle (minutes).
        docker_image: Docker image for agent container.
        storage_account: Storage account for results (auto-detected if None).
        use_managed_identity: Whether to use managed identity for auth.
        managed_identity_name: Name of managed identity (if using).
    """

    subscription_id: str
    resource_group: str
    workspace_name: str
    vm_size: str = "Standard_D2_v3"  # 2 vCPUs (fits free trial with existing usage)
    idle_timeout_minutes: int = 60
    docker_image: str = "ghcr.io/microsoft/windowsagentarena:latest"
    storage_account: str | None = None
    use_managed_identity: bool = False
    managed_identity_name: str | None = None

    @classmethod
    def from_env(cls) -> AzureConfig:
        """Create config from environment variables / .env file.

        Uses settings from openadapt_ml.config which loads from:
        1. Environment variables
        2. .env file
        3. Default values

        Required settings:
            AZURE_SUBSCRIPTION_ID
            AZURE_ML_RESOURCE_GROUP
            AZURE_ML_WORKSPACE_NAME

        Optional settings:
            AZURE_VM_SIZE (default: Standard_D4_v3 for free trial compatibility)
            AZURE_DOCKER_IMAGE (default: ghcr.io/microsoft/windowsagentarena:latest)

        Authentication (one of):
            - AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID (service principal)
            - Azure CLI login (`az login`)
            - Managed Identity (when running on Azure)

        Raises:
            ValueError: If required settings are not configured.
        """
        from openadapt_ml.config import settings

        # Validate required settings
        if not settings.azure_subscription_id:
            raise ValueError(
                "AZURE_SUBSCRIPTION_ID not set. "
                "Run 'python scripts/setup_azure.py' to configure Azure credentials."
            )
        if not settings.azure_ml_resource_group:
            raise ValueError(
                "AZURE_ML_RESOURCE_GROUP not set. "
                "Run 'python scripts/setup_azure.py' to configure Azure credentials."
            )
        if not settings.azure_ml_workspace_name:
            raise ValueError(
                "AZURE_ML_WORKSPACE_NAME not set. "
                "Run 'python scripts/setup_azure.py' to configure Azure credentials."
            )

        return cls(
            subscription_id=settings.azure_subscription_id,
            resource_group=settings.azure_ml_resource_group,
            workspace_name=settings.azure_ml_workspace_name,
            vm_size=settings.azure_vm_size,
            docker_image=settings.azure_docker_image,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> AzureConfig:
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class WorkerState:
    """State of a single worker VM."""

    worker_id: int
    compute_name: str
    status: str = "pending"  # pending, running, completed, failed
    assigned_tasks: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    results: list[BenchmarkResult] = field(default_factory=list)
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class EvaluationRun:
    """State of an evaluation run across multiple workers."""

    run_id: str
    experiment_name: str
    num_workers: int
    total_tasks: int
    workers: list[WorkerState] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    start_time: float | None = None
    end_time: float | None = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "num_workers": self.num_workers,
            "total_tasks": self.total_tasks,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "compute_name": w.compute_name,
                    "status": w.status,
                    "assigned_tasks": w.assigned_tasks,
                    "completed_tasks": w.completed_tasks,
                    "error": w.error,
                }
                for w in self.workers
            ],
        }


class AzureMLClient:
    """Wrapper around Azure ML SDK for compute management.

    This provides a simplified interface for creating and managing
    Azure ML compute instances for WAA evaluation.
    """

    def __init__(self, config: AzureConfig):
        self.config = config
        self._client = None
        self._ensure_sdk_available()

    def _ensure_sdk_available(self) -> None:
        """Check that Azure SDK is available."""
        try:
            from azure.ai.ml import MLClient
            from azure.identity import (
                ClientSecretCredential,
                DefaultAzureCredential,
            )

            self._MLClient = MLClient
            self._DefaultAzureCredential = DefaultAzureCredential
            self._ClientSecretCredential = ClientSecretCredential
        except ImportError as e:
            raise ImportError(
                "Azure ML SDK not installed. Install with: "
                "pip install azure-ai-ml azure-identity"
            ) from e

    @property
    def client(self):
        """Lazy-load ML client.

        Uses service principal credentials if configured in .env,
        otherwise falls back to DefaultAzureCredential (CLI login, managed identity, etc.)
        """
        if self._client is None:
            credential = self._get_credential()
            self._client = self._MLClient(
                credential=credential,
                subscription_id=self.config.subscription_id,
                resource_group_name=self.config.resource_group,
                workspace_name=self.config.workspace_name,
            )
            logger.info(f"Connected to Azure ML workspace: {self.config.workspace_name}")
        return self._client

    def _get_credential(self):
        """Get Azure credential, preferring service principal if configured."""
        from openadapt_ml.config import settings

        # Use service principal if credentials are configured
        if all([
            settings.azure_client_id,
            settings.azure_client_secret,
            settings.azure_tenant_id,
        ]):
            logger.info("Using service principal authentication")
            return self._ClientSecretCredential(
                tenant_id=settings.azure_tenant_id,
                client_id=settings.azure_client_id,
                client_secret=settings.azure_client_secret,
            )

        # Fall back to DefaultAzureCredential (CLI login, managed identity, etc.)
        logger.info(
            "Using DefaultAzureCredential (ensure you're logged in with 'az login' "
            "or have service principal credentials in .env)"
        )
        return self._DefaultAzureCredential()

    def create_compute_instance(
        self,
        name: str,
        startup_script: str | None = None,  # noqa: ARG002 - reserved for future use
    ) -> str:
        """Create a compute instance.

        Args:
            name: Compute instance name.
            startup_script: Optional startup script content (not yet implemented).

        Returns:
            Compute instance name.
        """
        # TODO: Add startup_script support when implementing full WAA integration
        _ = startup_script  # Reserved for future use
        from azure.ai.ml.entities import ComputeInstance

        # Check if already exists
        try:
            existing = self.client.compute.get(name)
            if existing:
                logger.info(f"Compute instance {name} already exists")
                return name
        except Exception:
            pass  # Doesn't exist, create it

        compute = ComputeInstance(
            name=name,
            size=self.config.vm_size,
            idle_time_before_shutdown_minutes=self.config.idle_timeout_minutes,
        )

        # Add managed identity if configured
        if self.config.use_managed_identity and self.config.managed_identity_name:
            identity_id = (
                f"/subscriptions/{self.config.subscription_id}"
                f"/resourceGroups/{self.config.resource_group}"
                f"/providers/Microsoft.ManagedIdentity"
                f"/userAssignedIdentities/{self.config.managed_identity_name}"
            )
            compute.identity = {"type": "UserAssigned", "user_assigned_identities": [identity_id]}

        print(f"      Creating VM: {name}...", end="", flush=True)
        self.client.compute.begin_create_or_update(compute).result()
        print(" done")

        return name

    def delete_compute_instance(self, name: str) -> None:
        """Delete a compute instance.

        Args:
            name: Compute instance name.
        """
        try:
            logger.info(f"Deleting compute instance: {name}")
            self.client.compute.begin_delete(name).result()
            logger.info(f"Compute instance {name} deleted")
        except Exception as e:
            logger.warning(f"Failed to delete compute instance {name}: {e}")

    def list_compute_instances(self, prefix: str | None = None) -> list[str]:
        """List compute instances.

        Args:
            prefix: Optional name prefix filter.

        Returns:
            List of compute instance names.
        """
        computes = self.client.compute.list()
        names = [c.name for c in computes if c.type == "ComputeInstance"]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return names

    def get_compute_status(self, name: str) -> str:
        """Get compute instance status.

        Args:
            name: Compute instance name.

        Returns:
            Status string (Running, Stopped, etc.)
        """
        compute = self.client.compute.get(name)
        return compute.state

    def submit_job(
        self,
        compute_name: str,
        command: str,
        environment_variables: dict[str, str] | None = None,
        display_name: str | None = None,
    ) -> str:
        """Submit a job to a compute instance.

        Args:
            compute_name: Target compute instance.
            command: Command to run.
            environment_variables: Environment variables.
            display_name: Job display name.

        Returns:
            Job name/ID.
        """
        from azure.ai.ml import command as ml_command
        from azure.ai.ml.entities import Environment

        # Create environment with Docker image
        env = Environment(
            image=self.config.docker_image,
            name="waa-agent-env",
        )

        job = ml_command(
            command=command,
            environment=env,
            compute=compute_name,
            display_name=display_name or f"waa-job-{compute_name}",
            environment_variables=environment_variables or {},
        )

        submitted = self.client.jobs.create_or_update(job)
        logger.info(f"Job submitted: {submitted.name}")
        return submitted.name

    def wait_for_job(self, job_name: str, timeout_seconds: int = 3600) -> dict:
        """Wait for a job to complete.

        Args:
            job_name: Job name/ID.
            timeout_seconds: Maximum wait time.

        Returns:
            Job result dict.
        """
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            job = self.client.jobs.get(job_name)
            if job.status in ["Completed", "Failed", "Canceled"]:
                return {
                    "status": job.status,
                    "outputs": job.outputs if hasattr(job, "outputs") else {},
                }
            time.sleep(10)

        raise TimeoutError(f"Job {job_name} did not complete within {timeout_seconds}s")


class AzureWAAOrchestrator:
    """Orchestrates WAA evaluation across multiple Azure VMs.

    This class manages the full lifecycle of a distributed WAA evaluation:
    1. Provisions Azure ML compute instances
    2. Distributes tasks across workers
    3. Monitors progress and collects results
    4. Cleans up resources

    Example:
        config = AzureConfig.from_env()
        orchestrator = AzureWAAOrchestrator(config, waa_repo_path="/path/to/WAA")

        results = orchestrator.run_evaluation(
            agent=my_agent,
            num_workers=40,
        )
        print(f"Success rate: {sum(r.success for r in results) / len(results):.1%}")
    """

    def __init__(
        self,
        config: AzureConfig,
        waa_repo_path: str | Path,
        experiment_name: str = "waa-eval",
    ):
        """Initialize orchestrator.

        Args:
            config: Azure configuration.
            waa_repo_path: Path to WAA repository.
            experiment_name: Name prefix for this evaluation.
        """
        self.config = config
        self.waa_repo_path = Path(waa_repo_path)
        self.experiment_name = experiment_name
        self.ml_client = AzureMLClient(config)
        self._current_run: EvaluationRun | None = None

    def run_evaluation(
        self,
        agent: BenchmarkAgent,
        num_workers: int = 10,
        task_ids: list[str] | None = None,
        max_steps_per_task: int = 15,
        on_worker_complete: Callable[[WorkerState], None] | None = None,
        cleanup_on_complete: bool = True,
    ) -> list[BenchmarkResult]:
        """Run evaluation across multiple Azure VMs.

        Args:
            agent: Agent to evaluate (must be serializable or API-based).
            num_workers: Number of parallel VMs.
            task_ids: Specific tasks to run (None = all 154 tasks).
            max_steps_per_task: Maximum steps per task.
            on_worker_complete: Callback when a worker finishes.
            cleanup_on_complete: Whether to delete VMs after completion.

        Returns:
            List of BenchmarkResult for all tasks.
        """
        # Load tasks
        from openadapt_ml.benchmarks.waa import WAAAdapter

        adapter = WAAAdapter(waa_repo_path=self.waa_repo_path)
        if task_ids:
            tasks = [adapter.load_task(tid) for tid in task_ids]
        else:
            tasks = adapter.list_tasks()

        print(f"[1/4] Loaded {len(tasks)} tasks for {num_workers} worker(s)")

        # Create evaluation run
        run_id = f"{self.experiment_name}-{int(time.time())}"
        self._current_run = EvaluationRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            num_workers=num_workers,
            total_tasks=len(tasks),
            status="running",
            start_time=time.time(),
        )

        # Distribute tasks across workers
        task_batches = self._distribute_tasks(tasks, num_workers)

        # Create workers
        # VM names: 3-24 chars, letters/numbers/hyphens, start with letter
        # Cannot end with number after hyphen, so we add 'x' suffix
        workers = []
        short_id = str(int(time.time()))[-4:]  # Last 4 digits of timestamp
        for i, batch in enumerate(task_batches):
            worker = WorkerState(
                worker_id=i,
                compute_name=f"waa{short_id}w{i}",  # e.g., "waa6571w0" (no trailing hyphen-number)
                assigned_tasks=[t.task_id for t in batch],
            )
            workers.append(worker)
        self._current_run.workers = workers

        try:
            # Provision VMs in parallel
            print(f"[2/4] Provisioning {num_workers} Azure VM(s)... (this takes 3-5 minutes)")
            self._provision_workers(workers)
            print(f"      VM(s) ready")

            # Submit jobs to workers
            print(f"[3/4] Submitting evaluation jobs...")
            self._submit_worker_jobs(workers, task_batches, agent, max_steps_per_task)
            print(f"      Jobs submitted")

            # Wait for completion and collect results
            print(f"[4/4] Waiting for workers to complete...")
            results = self._wait_and_collect_results(workers, on_worker_complete)

            self._current_run.status = "completed"
            self._current_run.end_time = time.time()

            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self._current_run.status = "failed"
            raise

        finally:
            if cleanup_on_complete:
                self._cleanup_workers(workers)

    def _distribute_tasks(
        self, tasks: list[BenchmarkTask], num_workers: int
    ) -> list[list[BenchmarkTask]]:
        """Distribute tasks evenly across workers."""
        batches: list[list[BenchmarkTask]] = [[] for _ in range(num_workers)]
        for i, task in enumerate(tasks):
            batches[i % num_workers].append(task)
        return batches

    def _provision_workers(self, workers: list[WorkerState]) -> None:
        """Provision all worker VMs in parallel."""
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = {
                executor.submit(
                    self.ml_client.create_compute_instance,
                    worker.compute_name,
                ): worker
                for worker in workers
            }

            for future in as_completed(futures):
                worker = futures[future]
                try:
                    future.result()
                    worker.status = "provisioned"
                    logger.info(f"Worker {worker.worker_id} provisioned")
                except Exception as e:
                    worker.status = "failed"
                    worker.error = str(e)
                    logger.error(f"Failed to provision worker {worker.worker_id}: {e}")

    def _submit_worker_jobs(
        self,
        workers: list[WorkerState],
        task_batches: list[list[BenchmarkTask]],
        agent: BenchmarkAgent,
        max_steps: int,
    ) -> None:
        """Submit evaluation jobs to workers."""
        for worker, tasks in zip(workers, task_batches):
            if worker.status == "failed":
                continue

            try:
                # Serialize task IDs for this worker
                task_ids = [t.task_id for t in tasks]
                task_ids_json = json.dumps(task_ids)

                # Build command
                command = self._build_worker_command(task_ids_json, max_steps, agent)

                # Submit job
                self.ml_client.submit_job(
                    compute_name=worker.compute_name,
                    command=command,
                    environment_variables={
                        "WAA_TASK_IDS": task_ids_json,
                        "WAA_MAX_STEPS": str(max_steps),
                    },
                    display_name=f"waa-worker-{worker.worker_id}",
                )
                worker.status = "running"
                worker.start_time = time.time()

            except Exception as e:
                worker.status = "failed"
                worker.error = str(e)
                logger.error(f"Failed to submit job for worker {worker.worker_id}: {e}")

    def _build_worker_command(
        self,
        task_ids_json: str,
        max_steps: int,
        agent: BenchmarkAgent,  # noqa: ARG002 - will be used for agent config serialization
    ) -> str:
        """Build the command to run on a worker VM.

        Args:
            task_ids_json: JSON string of task IDs for this worker.
            max_steps: Maximum steps per task.
            agent: Agent to run (TODO: serialize agent config for remote execution).
        """
        # TODO: Serialize agent config and pass to remote worker
        # For now, workers use a default agent configuration
        _ = agent  # Reserved for agent serialization
        return f"""
        cd /workspace/WindowsAgentArena && \
        python -m client.run \
            --task_ids '{task_ids_json}' \
            --max_steps {max_steps} \
            --output_dir /outputs
        """

    def _wait_and_collect_results(
        self,
        workers: list[WorkerState],
        on_worker_complete: Callable[[WorkerState], None] | None,
    ) -> list[BenchmarkResult]:
        """Wait for all workers and collect results."""
        all_results: list[BenchmarkResult] = []

        # Poll workers for completion
        pending_workers = [w for w in workers if w.status == "running"]

        while pending_workers:
            for worker in pending_workers[:]:
                try:
                    status = self.ml_client.get_compute_status(worker.compute_name)

                    # Check if job completed (simplified - real impl would check job status)
                    if status in ["Stopped", "Deallocated"]:
                        worker.status = "completed"
                        worker.end_time = time.time()

                        # Fetch results from blob storage
                        results = self._fetch_worker_results(worker)
                        worker.results = results
                        all_results.extend(results)

                        if on_worker_complete:
                            on_worker_complete(worker)

                        pending_workers.remove(worker)
                        logger.info(
                            f"Worker {worker.worker_id} completed: "
                            f"{len(results)} results"
                        )

                except Exception as e:
                    logger.warning(f"Error checking worker {worker.worker_id}: {e}")

            if pending_workers:
                time.sleep(30)

        return all_results

    def _fetch_worker_results(self, worker: WorkerState) -> list[BenchmarkResult]:
        """Fetch results from a worker's output storage."""
        # In a real implementation, this would download results from blob storage
        # For now, return placeholder results
        results = []
        for task_id in worker.assigned_tasks:
            results.append(
                BenchmarkResult(
                    task_id=task_id,
                    success=False,  # Placeholder
                    score=0.0,
                    num_steps=0,
                )
            )
        return results

    def _cleanup_workers(self, workers: list[WorkerState]) -> None:
        """Delete all worker VMs."""
        logger.info("Cleaning up worker VMs...")
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [
                executor.submit(self.ml_client.delete_compute_instance, w.compute_name)
                for w in workers
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Cleanup error: {e}")

    def get_run_status(self) -> dict | None:
        """Get current run status."""
        if self._current_run is None:
            return None
        return self._current_run.to_dict()

    def cancel_run(self) -> None:
        """Cancel the current run and cleanup resources."""
        if self._current_run is None:
            return

        logger.info("Canceling evaluation run...")
        self._cleanup_workers(self._current_run.workers)
        self._current_run.status = "canceled"
        self._current_run.end_time = time.time()


def estimate_cost(
    num_tasks: int = 154,
    num_workers: int = 1,
    avg_task_duration_minutes: float = 1.0,
    vm_hourly_cost: float = 0.19,  # Standard_D4_v3 in East US (free trial compatible)
) -> dict:
    """Estimate Azure costs for a WAA evaluation run.

    Args:
        num_tasks: Number of tasks to run.
        num_workers: Number of parallel VMs (default: 1 for free trial).
        avg_task_duration_minutes: Average time per task.
        vm_hourly_cost: Hourly cost per VM (D4_v3 = $0.19/hr, D8_v3 = $0.38/hr).

    Returns:
        Dict with cost estimates.
    """
    tasks_per_worker = num_tasks / num_workers
    total_minutes = tasks_per_worker * avg_task_duration_minutes
    total_hours = total_minutes / 60

    # Add overhead for provisioning/cleanup
    overhead_hours = 0.25  # ~15 minutes

    vm_hours = (total_hours + overhead_hours) * num_workers
    total_cost = vm_hours * vm_hourly_cost

    return {
        "num_tasks": num_tasks,
        "num_workers": num_workers,
        "tasks_per_worker": tasks_per_worker,
        "estimated_duration_minutes": total_minutes + (overhead_hours * 60),
        "total_vm_hours": vm_hours,
        "estimated_cost_usd": total_cost,
        "cost_per_task_usd": total_cost / num_tasks,
    }
