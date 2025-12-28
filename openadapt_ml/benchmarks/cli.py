"""CLI for WAA benchmark evaluation.

Usage:
    # ============================================
    # WAA on Dedicated VM (RECOMMENDED for real evaluation)
    # ============================================

    # One-command setup: Creates VM, installs Docker, pulls image, clones WAA repo
    python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key YOUR_OPENAI_KEY

    # Prepare Windows 11 image (one-time, ~20 min)
    python -m openadapt_ml.benchmarks.cli vm prepare-windows

    # Run WAA benchmark
    python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5

    # Check VM status
    python -m openadapt_ml.benchmarks.cli vm status

    # SSH into VM for manual control
    python -m openadapt_ml.benchmarks.cli vm ssh

    # Clean up when done
    python -m openadapt_ml.benchmarks.cli vm delete

    # ============================================
    # Analyze Results
    # ============================================

    # Analyze results on remote VM (fast, no download)
    python -m openadapt_ml.benchmarks.cli analyze --vm-ip <IP> --remote

    # Analyze with verbose output (shows task IDs)
    python -m openadapt_ml.benchmarks.cli analyze --vm-ip <IP> --remote --verbose

    # Save analysis to JSON
    python -m openadapt_ml.benchmarks.cli analyze --vm-ip <IP> --remote --output results.json

    # Analyze local results directory
    python -m openadapt_ml.benchmarks.cli analyze --results-dir /path/to/results

    # ============================================
    # Mock/Testing (no Windows required)
    # ============================================

    # Test with mock adapter
    python -m openadapt_ml.benchmarks.cli test-mock --tasks 20

    # Test data collection (with screenshots and execution traces)
    python -m openadapt_ml.benchmarks.cli test-collection --tasks 5

    # ============================================
    # API-backed evaluation (Claude/GPT baselines)
    # ============================================

    python -m openadapt_ml.benchmarks.cli run-api --provider anthropic --tasks 5
    python -m openadapt_ml.benchmarks.cli run-api --provider openai --tasks 5

    # ============================================
    # Azure ML (Note: doesn't support nested virt)
    # ============================================

    python -m openadapt_ml.benchmarks.cli estimate --workers 40
    python -m openadapt_ml.benchmarks.cli run-azure --config azure_config.json --workers 40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from openadapt_ml.config import settings

logger = logging.getLogger(__name__)

# Pre-configure loggers to be quiet by default (before any Azure imports)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("msrest").setLevel(logging.WARNING)
logging.getLogger("openadapt_ml.benchmarks.azure").setLevel(logging.WARNING)

# Suppress Azure SDK experimental class warnings
import warnings
warnings.filterwarnings("ignore", message=".*experimental class.*")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with appropriate verbosity.

    Args:
        verbose: If True, show all logs. If False, suppress Azure SDK noise.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Suppress noisy Azure SDK logs unless verbose
    if not verbose:
        logging.getLogger("azure").setLevel(logging.WARNING)
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("msrest").setLevel(logging.WARNING)


def bypass_product_key_dialog(ip: str, max_attempts: int = 3) -> bool:
    """Send keyboard commands via QEMU monitor to skip the product key dialog.

    Windows 11 Evaluation ISOs require clicking "I don't have a product key".
    This function sends Tab + Enter keys via QEMU monitor to click that link.

    Args:
        ip: IP address of the Azure VM running the container.
        max_attempts: Number of times to try clicking (in case of timing issues).

    Returns:
        True if commands were sent successfully.
    """
    import subprocess
    import time

    # QEMU sendkey commands to navigate to "I don't have a product key" link
    # The link is at the bottom of the dialog - Tab navigates through UI elements
    # We need to Tab to the link and press Enter
    qemu_commands = '''
# Navigate to "I don't have a product key" link and click it
# Tab through: text field -> Next button -> Back button -> link
sendkey tab
sendkey tab
sendkey tab
sendkey tab
sendkey ret
'''

    for attempt in range(max_attempts):
        try:
            # Send commands via QEMU monitor (port 7100 in container)
            ssh_cmd = f'''
# Use telnet to send QEMU commands
(
echo "sendkey tab"
sleep 0.3
echo "sendkey tab"
sleep 0.3
echo "sendkey tab"
sleep 0.3
echo "sendkey tab"
sleep 0.3
echo "sendkey ret"
sleep 0.5
) | timeout 10 docker exec -i winarena nc localhost 7100 2>/dev/null
'''
            result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                 f"azureuser@{ip}", ssh_cmd],
                capture_output=True, text=True, timeout=30
            )

            if "QEMU" in result.stdout or result.returncode == 0:
                return True

            time.sleep(2)

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    return False


def find_waa_path() -> Path | None:
    """Auto-detect Windows Agent Arena repository path.

    Searches in order:
    1. vendor/WindowsAgentArena (git submodule)
    2. ../WindowsAgentArena (sibling directory)
    3. ~/WindowsAgentArena (home directory)

    Returns:
        Path to WAA repo, or None if not found.
    """
    # Get the project root (where this package is installed)
    project_root = Path(__file__).parent.parent.parent

    candidates = [
        project_root / "vendor" / "WindowsAgentArena",
        project_root.parent / "WindowsAgentArena",
        Path.home() / "WindowsAgentArena",
    ]

    for path in candidates:
        if path.exists() and (path / "src").exists():
            return path

    return None


def get_waa_path(args_path: str | None) -> Path:
    """Get WAA path from args or auto-detect.

    Args:
        args_path: Path from command line args, or None.

    Returns:
        Resolved WAA path.

    Raises:
        SystemExit: If WAA cannot be found.
    """
    if args_path:
        path = Path(args_path)
        if not path.exists():
            print(f"ERROR: WAA path does not exist: {path}")
            sys.exit(1)
        return path

    path = find_waa_path()
    if path:
        print(f"  Using WAA from: {path}")
        return path

    print("ERROR: Windows Agent Arena not found!")
    print("\nTo fix, run:")
    print("  git submodule update --init --recursive")
    print("\nOr specify path manually:")
    print("  --waa-path /path/to/WindowsAgentArena")
    sys.exit(1)


def cmd_estimate(args: argparse.Namespace) -> None:
    """Estimate Azure costs."""
    from openadapt_ml.benchmarks.azure import estimate_cost

    estimate = estimate_cost(
        num_tasks=args.tasks,
        num_workers=args.workers,
        avg_task_duration_minutes=args.duration,
        vm_hourly_cost=args.vm_cost,
    )

    print("\n=== WAA Azure Cost Estimate ===")
    print(f"Tasks:                    {estimate['num_tasks']}")
    print(f"Workers:                  {estimate['num_workers']}")
    print(f"Tasks per worker:         {estimate['tasks_per_worker']:.1f}")
    print(f"Estimated duration:       {estimate['estimated_duration_minutes']:.1f} minutes")
    print(f"Total VM hours:           {estimate['total_vm_hours']:.2f}")
    print(f"Estimated cost:           ${estimate['estimated_cost_usd']:.2f}")
    print(f"Cost per task:            ${estimate['cost_per_task_usd']:.4f}")
    print()


def cmd_az_status(args: argparse.Namespace) -> None:
    """Check Azure resource status for WAA benchmark deployment."""
    import subprocess

    def run_az(cmd: list[str], description: str) -> tuple[bool, str]:
        """Run an az command and return (success, output)."""
        try:
            result = subprocess.run(
                ["az"] + cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
        except FileNotFoundError:
            return False, "Azure CLI not installed"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    print("\n=== Azure WAA Benchmark Status ===\n")

    # Check Azure CLI
    ok, output = run_az(["--version"], "Azure CLI version")
    if ok:
        version = output.split("\n")[0] if output else "unknown"
        print(f"  Azure CLI:        ✓ {version}")
    else:
        print(f"  Azure CLI:        ✗ Not installed")
        print(f"                    Install: brew install azure-cli")
        return

    # Check login
    ok, output = run_az(["account", "show", "--query", "name", "-o", "tsv"], "Azure login")
    if ok:
        print(f"  Logged in:        ✓ {output}")
    else:
        print(f"  Logged in:        ✗ Run: az login")
        return

    # Check resource group
    rg = args.resource_group
    ok, output = run_az(
        ["group", "show", "--name", rg, "--query", "location", "-o", "tsv"],
        "Resource group"
    )
    if ok:
        print(f"  Resource group:   ✓ {rg} ({output})")
    else:
        print(f"  Resource group:   ✗ {rg} not found")
        print(f"                    Run: python scripts/setup_azure.py")
        return

    # Check ML workspace
    ws = args.workspace
    ok, output = run_az(
        ["ml", "workspace", "show", "--name", ws, "--resource-group", rg, "--query", "location", "-o", "tsv"],
        "ML workspace"
    )
    if ok:
        print(f"  ML workspace:     ✓ {ws} ({output})")
    else:
        print(f"  ML workspace:     ✗ {ws} not found")

    # Check ACR
    acr = args.acr_name
    ok, output = run_az(
        ["acr", "show", "--name", acr, "--resource-group", rg, "--query", "loginServer", "-o", "tsv"],
        "Container registry"
    )
    if ok:
        print(f"  Container registry: ✓ {output}")
    else:
        print(f"  Container registry: ✗ {acr} not found")

    # Check WAA Docker image
    ok, output = run_az(
        ["acr", "repository", "show", "--name", acr, "--repository", "winarena", "--query", "imageName", "-o", "tsv"],
        "WAA Docker image"
    )
    if ok:
        print(f"  WAA Docker image:   ✓ winarena")
    else:
        print(f"  WAA Docker image:   ✗ Not imported")
        print(f"                    Run: python scripts/setup_azure.py")

    # Check .env file
    env_path = Path(".env")
    if env_path.exists():
        env_content = env_path.read_text()
        has_azure = "AZURE_SUBSCRIPTION_ID" in env_content
        print(f"  .env file:        ✓ {'Azure credentials found' if has_azure else 'Missing Azure credentials'}")
    else:
        print(f"  .env file:        ✗ Not found")

    # Check WAA submodule
    waa_path = find_waa_path()
    if waa_path:
        # Count tasks
        from openadapt_ml.benchmarks import WAAAdapter
        try:
            adapter = WAAAdapter(waa_repo_path=waa_path)
            task_count = len(adapter.list_tasks())
            print(f"  WAA submodule:    ✓ {task_count} tasks at {waa_path}")
        except Exception as e:
            print(f"  WAA submodule:    ⚠ Found but error: {e}")
    else:
        print(f"  WAA submodule:    ✗ Not found")
        print(f"                    Run: git submodule update --init --recursive")

    print()
    print("Ready for benchmark evaluation!" if ok else "Some resources missing - run setup_azure.py")


def cmd_run_local(args: argparse.Namespace) -> None:
    """Run evaluation locally on Windows."""
    from openadapt_ml.benchmarks import (
        RandomAgent,
        WAAAdapter,
        compute_metrics,
        evaluate_agent_on_benchmark,
    )

    # Check platform
    if sys.platform != "win32" and not args.force:
        print("ERROR: WAA requires Windows. Use --force to override.")
        sys.exit(1)

    # Parse task IDs
    task_ids = None
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]

    # Get WAA path (auto-detect if not specified)
    waa_path = get_waa_path(args.waa_path)

    # Create adapter
    adapter = WAAAdapter(waa_repo_path=waa_path)

    # Create agent (for now, just random - in practice, would load a model)
    if args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    else:
        print(f"ERROR: Unknown agent type: {args.agent}")
        sys.exit(1)

    # Run evaluation
    print(f"\nRunning WAA evaluation...")
    print(f"  WAA path: {waa_path}")
    print(f"  Tasks: {len(task_ids) if task_ids else 'all (154)'}")
    print(f"  Max steps: {args.max_steps}")
    print()

    results = evaluate_agent_on_benchmark(
        agent=agent,
        adapter=adapter,
        task_ids=task_ids,
        max_steps=args.max_steps,
    )

    # Print results
    metrics = compute_metrics(results)
    print("\n=== Results ===")
    print(f"Tasks:        {metrics['num_tasks']}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Avg score:    {metrics['avg_score']:.3f}")
    print(f"Avg steps:    {metrics['avg_steps']:.1f}")
    print()

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "results": [
                        {
                            "task_id": r.task_id,
                            "success": r.success,
                            "score": r.score,
                            "num_steps": r.num_steps,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
            )
        print(f"Results saved to: {output_path}")


def _get_azure_ml_studio_url(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    view: str = "compute",
) -> str:
    """Get Azure ML Studio URL for a workspace.

    Args:
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        workspace_name: ML workspace name
        view: Which view to open - "compute", "jobs", "overview"

    Returns:
        Azure ML Studio URL
    """
    workspace_id = (
        f"/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}"
    )

    # Azure ML Studio URL format
    # The experiments page shows all jobs for this workspace
    # Format: https://ml.azure.com/experiments/id/{experiment_id}?wsid={workspace_id}
    # NOTE: This experiment_id is specific to the openadapt-ml workspace
    # TODO: Retrieve experiment_id dynamically from Azure instead of hardcoding
    experiment_id = "ad29082c-0607-4fda-8cc7-38944eb5a518"
    return f"https://ml.azure.com/experiments/id/{experiment_id}?wsid={workspace_id}"


def _write_azure_job_status(
    output_dir: Path,
    job_id: str,
    status: str,
    workers: int,
    num_tasks: int,
    task_ids: list[str] | None,
    azure_url: str,
    start_time: str | None = None,
    end_time: str | None = None,
    results: dict | None = None,
) -> None:
    """Write Azure job status to a JSON file for the benchmark viewer."""
    import datetime

    jobs_file = output_dir / "azure_jobs.json"

    # Load existing jobs
    jobs = []
    if jobs_file.exists():
        try:
            with open(jobs_file) as f:
                jobs = json.load(f)
        except json.JSONDecodeError:
            jobs = []

    # Find or create this job
    job_entry = None
    for job in jobs:
        if job.get("job_id") == job_id:
            job_entry = job
            break

    if job_entry is None:
        job_entry = {
            "job_id": job_id,
            "started_at": start_time or datetime.datetime.now().isoformat(),
        }
        jobs.insert(0, job_entry)  # Most recent first

    # Update job entry
    job_entry.update({
        "status": status,
        "workers": workers,
        "num_tasks": num_tasks,
        "task_ids": task_ids[:5] if task_ids and len(task_ids) > 5 else task_ids,  # First 5 for display
        "azure_dashboard_url": azure_url,
        "updated_at": datetime.datetime.now().isoformat(),
    })

    if end_time:
        job_entry["ended_at"] = end_time
    if results:
        job_entry["results"] = results

    # Keep only last 10 jobs
    jobs = jobs[:10]

    # Write back
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(jobs_file, "w") as f:
        json.dump(jobs, f, indent=2)


def cmd_run_azure(args: argparse.Namespace) -> None:
    """Run evaluation on Azure."""
    import datetime
    import random
    from openadapt_ml.benchmarks import RandomAgent, WAAAdapter
    from openadapt_ml.benchmarks.azure import AzureConfig, AzureWAAOrchestrator

    # Load config
    if args.config:
        config = AzureConfig.from_json(args.config)
    else:
        config = AzureConfig.from_env()

    # Get WAA path (auto-detect if not specified)
    waa_path = get_waa_path(args.waa_path)

    # Load WAA adapter to get available tasks
    adapter = WAAAdapter(waa_repo_path=waa_path)
    all_tasks = adapter.list_tasks()  # Returns list[BenchmarkTask]
    all_task_ids = [t.task_id for t in all_tasks]  # Extract task_id strings
    print(f"  Available tasks: {len(all_task_ids)}")

    # Determine which tasks to run
    task_ids = None
    if args.task_ids:
        # Specific task IDs provided
        task_ids = [t.strip() for t in args.task_ids.split(",")]
        # Validate task IDs exist
        invalid = [t for t in task_ids if t not in all_task_ids]
        if invalid:
            print(f"ERROR: Invalid task IDs: {invalid[:5]}...")
            print(f"  Available tasks start with: {all_task_ids[:3]}")
            sys.exit(1)
    elif args.num_tasks:
        # Select random subset of tasks
        random.seed(args.seed)
        num_to_select = min(args.num_tasks, len(all_task_ids))
        task_ids = random.sample(all_task_ids, num_to_select)  # Sample from string IDs
        print(f"  Selected {num_to_select} random tasks")

    # Create orchestrator
    orchestrator = AzureWAAOrchestrator(
        config=config,
        waa_repo_path=waa_path,
        experiment_name=args.experiment,
    )

    # Create agent
    if args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    else:
        print(f"ERROR: Unknown agent type: {args.agent}")
        sys.exit(1)

    # Estimate costs first
    from openadapt_ml.benchmarks.azure import estimate_cost

    num_tasks = len(task_ids) if task_ids else len(all_task_ids)
    estimate = estimate_cost(num_tasks=num_tasks, num_workers=args.workers)

    print(f"\n=== Azure WAA Evaluation ===")
    print(f"  Workers:          {args.workers}")
    print(f"  Tasks:            {num_tasks}")
    print(f"  Estimated cost:   ${estimate['estimated_cost_usd']:.2f}")
    print(f"  Estimated time:   {estimate['estimated_duration_minutes']:.1f} minutes")
    print()

    if not args.yes:
        response = input("Proceed? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Generate job ID and Azure dashboard URL
    job_id = f"waa_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    azure_url = _get_azure_ml_studio_url(
        subscription_id=config.subscription_id,
        resource_group=config.resource_group,
        workspace_name=config.workspace_name,
        view="compute",
    )
    output_dir = Path("benchmark_results")
    start_time = datetime.datetime.now().isoformat()

    # Write initial job status
    _write_azure_job_status(
        output_dir=output_dir,
        job_id=job_id,
        status="provisioning",
        workers=args.workers,
        num_tasks=num_tasks,
        task_ids=task_ids,
        azure_url=azure_url,
        start_time=start_time,
    )

    # Run evaluation
    print("\nStarting Azure evaluation...")
    print(f"  Job ID: {job_id}")
    print(f"  Monitor at: {azure_url}")
    print("  (VM provisioning takes 3-5 minutes)")
    print()

    try:
        # Update status to running once provisioning starts
        _write_azure_job_status(
            output_dir=output_dir,
            job_id=job_id,
            status="running",
            workers=args.workers,
            num_tasks=num_tasks,
            task_ids=task_ids,
            azure_url=azure_url,
        )

        results = orchestrator.run_evaluation(
            agent=agent,
            num_workers=args.workers,
            task_ids=task_ids,
            max_steps_per_task=args.max_steps,
            cleanup_on_complete=not args.no_cleanup,
        )

        # Print results
        from openadapt_ml.benchmarks import compute_metrics

        metrics = compute_metrics(results)
        print("\n=== Results ===")
        print(f"Tasks:        {metrics['num_tasks']}")
        print(f"Success rate: {metrics['success_rate']:.1%}")
        print(f"Avg score:    {metrics['avg_score']:.3f}")
        print()

        # Update job status to completed
        _write_azure_job_status(
            output_dir=output_dir,
            job_id=job_id,
            status="completed",
            workers=args.workers,
            num_tasks=num_tasks,
            task_ids=task_ids,
            azure_url=azure_url,
            end_time=datetime.datetime.now().isoformat(),
            results={
                "success_rate": metrics.get("success_rate", 0.0),
                "num_success": metrics.get("success_count", 0),
                "avg_score": metrics.get("avg_score", 0.0),
            },
        )

        # Save results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "metrics": metrics,
                        "run_status": orchestrator.get_run_status(),
                        "results": [
                            {
                                "task_id": r.task_id,
                                "success": r.success,
                                "score": r.score,
                                "num_steps": r.num_steps,
                            }
                            for r in results
                        ],
                    },
                    f,
                    indent=2,
                )
            print(f"Results saved to: {output_path}")

    except Exception as e:
        # Update job status to failed
        _write_azure_job_status(
            output_dir=output_dir,
            job_id=job_id,
            status="failed",
            workers=args.workers,
            num_tasks=num_tasks,
            task_ids=task_ids,
            azure_url=azure_url,
            end_time=datetime.datetime.now().isoformat(),
            results={"error": str(e)},
        )
        raise


def cmd_test_mock(args: argparse.Namespace) -> None:
    """Test with mock adapter (no Windows required)."""
    from openadapt_ml.benchmarks import (
        RandomAgent,
        WAAMockAdapter,
        compute_domain_metrics,
        compute_metrics,
        evaluate_agent_on_benchmark,
    )

    print(f"\n=== Testing with Mock Adapter ===")
    print(f"  Tasks:     {args.tasks}")
    print(f"  Max steps: {args.max_steps}")
    print()

    # Create mock adapter
    adapter = WAAMockAdapter(num_tasks=args.tasks)
    agent = RandomAgent(seed=args.seed)

    # Run evaluation
    results = evaluate_agent_on_benchmark(
        agent=agent,
        adapter=adapter,
        max_steps=args.max_steps,
    )

    # Print results
    metrics = compute_metrics(results)
    print("=== Results ===")
    print(f"Tasks:        {metrics['num_tasks']}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Successes:    {metrics['success_count']}")
    print(f"Failures:     {metrics['fail_count']}")
    print(f"Avg steps:    {metrics['avg_steps']:.1f}")
    print()

    # Domain breakdown
    tasks = adapter.list_tasks()
    domain_metrics = compute_domain_metrics(results, tasks)
    if domain_metrics:
        print("=== By Domain ===")
        for domain, dm in domain_metrics.items():
            print(f"  {domain}: {dm['success_rate']:.1%} ({dm['success_count']}/{dm['num_tasks']})")
    print()


def cmd_test_collection(args: argparse.Namespace) -> None:
    """Test benchmark data collection with mock adapter.

    This command runs a benchmark evaluation with data collection enabled,
    creating a full directory structure with screenshots, execution traces,
    and metadata suitable for the benchmark viewer.
    """
    import json
    from pathlib import Path

    from openadapt_ml.benchmarks import RandomAgent, WAAMockAdapter
    from openadapt_ml.benchmarks.runner import EvaluationConfig, evaluate_agent_on_benchmark

    print(f"\n=== Testing Benchmark Data Collection ===")
    print(f"  Tasks:       {args.tasks}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Output dir:  {args.output}")
    print(f"  Run name:    {args.run_name or '(auto-generated)'}")
    print()

    # Create mock adapter
    adapter = WAAMockAdapter(num_tasks=args.tasks, domains=["browser", "office"])
    agent = RandomAgent(action_types=["click", "type", "scroll", "done"], seed=args.seed)

    # Configure evaluation with data collection
    config = EvaluationConfig(
        max_steps=args.max_steps,
        parallel=1,
        save_trajectories=True,
        save_execution_traces=True,
        model_id=args.model_id,
        output_dir=args.output,
        run_name=args.run_name,
        verbose=True,
    )

    # Run evaluation
    results = evaluate_agent_on_benchmark(
        agent=agent,
        adapter=adapter,
        config=config,
    )

    # Print results
    success_count = sum(1 for r in results if r.success)
    success_rate = success_count / len(results) if results else 0.0
    avg_steps = sum(r.num_steps for r in results) / len(results) if results else 0.0

    print(f"\n=== Results ===")
    print(f"Total tasks:  {len(results)}")
    print(f"Success:      {success_count} ({success_rate:.1%})")
    print(f"Failure:      {len(results) - success_count}")
    print(f"Avg steps:    {avg_steps:.1f}")

    # Find the actual output directory by reading metadata
    output_dir = Path(args.output)
    run_dirs = sorted(output_dir.glob("*/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_dirs:
        run_dir = run_dirs[0].parent
        with open(run_dirs[0]) as f:
            metadata = json.load(f)
        run_name = metadata.get("run_name", run_dir.name)
    else:
        run_dir = output_dir
        run_name = "unknown"

    print(f"\n=== Output Directory ===")
    print(f"Location:     {run_dir.absolute()}")
    print(f"\nDirectory structure:")
    print(f"  {run_dir.name}/")
    print(f"  ├── metadata.json")
    print(f"  ├── summary.json")
    print(f"  └── tasks/")
    print(f"      ├── task_001/")
    print(f"      │   ├── task.json")
    print(f"      │   ├── execution.json")
    print(f"      │   └── screenshots/")
    print(f"      │       ├── step_000.png")
    print(f"      │       ├── step_001.png")
    print(f"      │       └── ...")
    print(f"      └── ...")
    print(f"\nYou can inspect the results at: {run_dir.absolute()}")
    print()


def cmd_run_api(args: argparse.Namespace) -> None:
    """Run evaluation using API-backed VLM (Claude/GPT-5.1).

    This provides baselines for comparing against fine-tuned models.
    """
    from openadapt_ml.benchmarks import (
        APIBenchmarkAgent,
        WAAMockAdapter,
        compute_domain_metrics,
        compute_metrics,
    )
    from openadapt_ml.benchmarks.runner import EvaluationConfig, evaluate_agent_on_benchmark

    provider_names = {
        "anthropic": "Claude",
        "openai": "GPT-5.1",
    }

    print(f"\n=== API-Backed Benchmark Evaluation ===")
    print(f"  Provider:    {args.provider} ({provider_names.get(args.provider, 'Unknown')})")
    print(f"  Tasks:       {args.tasks}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Output dir:  {args.output}")

    # Check for API key
    import os
    key_name = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
    if not os.getenv(key_name):
        print(f"WARNING: {key_name} environment variable not set!")
        print(f"  Set it in your .env file or export it before running.")
        print()

    # Determine which adapter to use
    task_ids = None
    if args.mock:
        # User explicitly requested mock adapter
        print(f"  Adapter:     Mock (forced by --mock flag)")
        print()
        adapter = WAAMockAdapter(num_tasks=args.tasks, domains=["browser", "office"])
    else:
        # Auto-detect WAA or use explicit path
        waa_path = None
        if args.waa_path:
            # Explicit path provided
            waa_path = Path(args.waa_path)
            if not waa_path.exists():
                print(f"ERROR: WAA path does not exist: {waa_path}")
                sys.exit(1)
        else:
            # Try to auto-detect
            waa_path = find_waa_path()

        if waa_path:
            # Real WAA available
            if sys.platform != "win32" and not args.force:
                print(f"  Adapter:     WAA (detected at {waa_path})")
                print("ERROR: WAA requires Windows. Use --mock to use mock adapter instead.")
                sys.exit(1)

            from openadapt_ml.benchmarks import WAAAdapter
            print(f"  Adapter:     WAA (real, from {waa_path})")
            print()
            adapter = WAAAdapter(waa_repo_path=waa_path)
            if args.task_ids:
                task_ids = [t.strip() for t in args.task_ids.split(",")]
        else:
            # WAA not found, fall back to mock
            print("  Adapter:     Mock (WAA not found)")
            print("  Note:        To use real WAA, run: git submodule update --init --recursive")
            print("               Or specify with: --waa-path /path/to/WindowsAgentArena")
            print()
            adapter = WAAMockAdapter(num_tasks=args.tasks, domains=["browser", "office"])

    # Create API-backed agent
    agent = APIBenchmarkAgent(
        provider=args.provider,
        max_tokens=args.max_tokens,
        use_accessibility_tree=not args.no_a11y,
        use_history=not args.no_history,
    )

    # Configure evaluation
    model_id = args.model_id if args.model_id else f"{args.provider}-api"
    config = EvaluationConfig(
        max_steps=args.max_steps,
        parallel=1,  # API calls should be sequential to avoid rate limits
        save_trajectories=True,
        save_execution_traces=True,
        model_id=model_id,
        output_dir=args.output,
        run_name=args.run_name,
        verbose=args.verbose,
    )

    # Run evaluation
    print("Starting evaluation...")
    print("  (Each step calls the API - this may take a while)")
    print()

    try:
        results = evaluate_agent_on_benchmark(
            agent=agent,
            adapter=adapter,
            task_ids=task_ids,
            config=config,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        if "API key" in str(e) or "api_key" in str(e).lower():
            print(f"\nMake sure {key_name} is set in your environment.")
        sys.exit(1)

    # Print results
    metrics = compute_metrics(results)
    print("\n=== Results ===")
    print(f"Tasks:        {metrics['num_tasks']}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Successes:    {metrics['success_count']}")
    print(f"Failures:     {metrics['fail_count']}")
    print(f"Avg score:    {metrics['avg_score']:.3f}")
    print(f"Avg steps:    {metrics['avg_steps']:.1f}")
    print()

    # Domain breakdown
    tasks = adapter.list_tasks()
    domain_metrics = compute_domain_metrics(results, tasks)
    if domain_metrics:
        print("=== By Domain ===")
        for domain, dm in domain_metrics.items():
            print(f"  {domain}: {dm['success_rate']:.1%} ({dm['success_count']}/{dm['num_tasks']})")
    print()

    # Find output directory
    output_dir = Path(args.output)
    run_dirs = sorted(output_dir.glob("*/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_dirs:
        run_dir = run_dirs[0].parent
        print(f"Results saved to: {run_dir.absolute()}")
        print(f"View with: uv run python -m openadapt_ml.cloud.local serve --open")
    print()


def cmd_create_config(args: argparse.Namespace) -> None:
    """Create a sample Azure config file."""
    from openadapt_ml.benchmarks.azure import AzureConfig

    config = AzureConfig(
        subscription_id="<your-subscription-id>",
        resource_group="agents",
        workspace_name="agents_ml",
        vm_size="Standard_D4_v3",
    )

    output_path = Path(args.output)
    config.to_json(output_path)
    print(f"Sample config saved to: {output_path}")
    print("\nEdit this file with your Azure credentials before using.")


def cmd_status(args: argparse.Namespace) -> None:
    """Check Azure workspace and compute status."""
    setup_logging(args.verbose)

    # Import after logging setup to suppress Azure SDK noise
    from openadapt_ml.benchmarks.azure import AzureConfig, AzureMLClient  # noqa: E402

    print("\n=== Azure WAA Status ===\n")

    # Check config
    try:
        config = AzureConfig.from_env()
        print(f"Subscription:    {config.subscription_id[:8]}...")
        print(f"Resource Group:  {config.resource_group}")
        print(f"Workspace:       {config.workspace_name}")
        print(f"VM Size:         {config.vm_size}")
    except ValueError as e:
        print(f"Config Error: {e}")
        print("\nRun 'python scripts/setup_azure.py' to configure.")
        return

    # Check WAA
    waa_path = find_waa_path()
    if waa_path:
        print(f"WAA Path:        {waa_path}")
    else:
        print("WAA Path:        NOT FOUND")
        print("  Run: git submodule update --init --recursive")

    # Check Azure connection
    print("\nConnecting to Azure...")
    try:
        client = AzureMLClient(config)
        computes = client.list_compute_instances(prefix="w")
        print(f"Connection:      OK")

        if computes:
            print(f"\nActive Compute Instances ({len(computes)}):")
            for name in computes:
                try:
                    status = client.get_compute_status(name)
                    print(f"  - {name}: {status}")
                except Exception:
                    print(f"  - {name}: (status unknown)")
        else:
            print("\nNo active compute instances.")

    except Exception as e:
        print(f"Connection:      FAILED")
        print(f"  Error: {e}")

    print()


def cmd_cleanup(args: argparse.Namespace) -> None:
    """Clean up all Azure compute resources."""
    setup_logging(args.verbose)

    from openadapt_ml.benchmarks.azure import AzureConfig, AzureMLClient

    print("\n=== Azure WAA Cleanup ===\n")

    try:
        config = AzureConfig.from_env()
    except ValueError as e:
        print(f"Config Error: {e}")
        return

    print(f"Workspace: {config.workspace_name}")
    print(f"Resource Group: {config.resource_group}")
    print()

    client = AzureMLClient(config)

    # List ALL compute instances (no prefix filter)
    print("Finding all compute instances...")
    computes = client.list_compute_instances()  # No prefix = get all

    if not computes:
        print("  No compute instances found")
    else:
        print(f"  Found {len(computes)} compute instance(s):")
        for name in computes:
            try:
                status = client.get_compute_status(name)
            except Exception:
                status = "unknown"
            print(f"    - {name} ({status})")

        print()
        for name in computes:
            if not args.yes:
                confirm = input(f"  Delete '{name}'? [y/N]: ").strip().lower()
                if confirm != "y":
                    print(f"    Skipped {name}")
                    continue
            print(f"    Deleting {name}...", end="", flush=True)
            try:
                client.delete_compute_instance(name)
                print(" done")
            except Exception as e:
                print(f" FAILED: {e}")

    print("\nCleanup complete.")
    print("Note: Resource deletion may take a few minutes to free quota.")
    print()


def cmd_cleanup_vms(args: argparse.Namespace) -> None:
    """Clean up Azure compute instances to free quota."""
    import subprocess

    print("\n=== Cleaning up Azure Compute Instances ===\n")

    # List current VMs
    result = subprocess.run(
        [
            "az", "ml", "compute", "list",
            "--resource-group", args.resource_group,
            "--workspace-name", args.workspace,
            "--query", "[].name",
            "-o", "tsv",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error listing VMs: {result.stderr}")
        sys.exit(1)

    vms = [v.strip() for v in result.stdout.strip().split("\n") if v.strip()]

    if not vms:
        print("No compute instances found.")
        return

    print(f"Found {len(vms)} compute instance(s):")
    for vm in vms:
        print(f"  - {vm}")
    print()

    if not args.yes:
        response = input(f"Delete all {len(vms)} VM(s)? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    for vm in vms:
        print(f"Deleting {vm}...", end=" ", flush=True)
        del_result = subprocess.run(
            [
                "az", "ml", "compute", "delete",
                "--name", vm,
                "--resource-group", args.resource_group,
                "--workspace-name", args.workspace,
                "--yes",
            ],
            capture_output=True,
            text=True,
        )
        if del_result.returncode == 0:
            print("done")
        else:
            print(f"failed: {del_result.stderr[:100]}")

    print("\nCleanup complete. Quota should be freed within a few minutes.")


def cmd_list_jobs(args: argparse.Namespace) -> None:
    """List recent Azure ML jobs."""
    import subprocess

    print("\n=== Recent Azure ML Jobs ===\n")

    result = subprocess.run(
        [
            "az", "ml", "job", "list",
            "--resource-group", args.resource_group,
            "--workspace-name", args.workspace,
            "-o", "table",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    # Filter out experimental warnings
    lines = [l for l in result.stdout.split("\n") if "experimental" not in l.lower()]
    print("\n".join(lines[:args.limit + 3]))  # +3 for header rows


def cmd_job_logs(args: argparse.Namespace) -> None:
    """Download and display logs for an Azure ML job."""
    import subprocess
    import tempfile

    print(f"\n=== Fetching logs for job: {args.job_name} ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "az", "ml", "job", "download",
                "--name", args.job_name,
                "--resource-group", args.resource_group,
                "--workspace-name", args.workspace,
                "--download-path", tmpdir,
                "--all",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)

        # Find and display logs
        log_files = [
            f"{tmpdir}/artifacts/user_logs/std_log.txt",
            f"{tmpdir}/artifacts/system_logs/lifecycler/execution-wrapper.log",
        ]

        for log_file in log_files:
            if Path(log_file).exists():
                print(f"=== {Path(log_file).name} ===")
                with open(log_file) as f:
                    content = f.read()
                    if content.strip():
                        print(content[:5000])  # Limit output
                        if len(content) > 5000:
                            print(f"\n... (truncated, full log at {log_file})")
                    else:
                        print("(empty)")
                print()


def get_vm_ip(resource_group: str, vm_name: str) -> str | None:
    """Get the public IP address of an Azure VM.

    Args:
        resource_group: Azure resource group name
        vm_name: Name of the VM

    Returns:
        Public IP address or None if VM not found/running
    """
    import subprocess

    result = subprocess.run(
        ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
         "--query", "publicIps", "-o", "tsv"],
        capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def capture_vm_screenshot(ip: str, output_path: Path | str = None) -> Path | None:
    """Capture a screenshot from the Windows VM via QEMU monitor.

    Args:
        ip: Public IP address of the Azure VM
        output_path: Path to save the screenshot. Defaults to training_output/current/vm_screenshot.png

    Returns:
        Path to the saved screenshot, or None on failure
    """
    import subprocess
    import tempfile

    if output_path is None:
        output_path = Path("training_output/current/vm_screenshot.png")
    output_path = Path(output_path)

    try:
        # Take screenshot via QEMU monitor and convert to PNG on VM
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             f"azureuser@{ip}",
             '(echo "screendump /tmp/screen.ppm"; sleep 1) | docker exec -i winarena nc localhost 7100 2>/dev/null; '
             'docker cp winarena:/tmp/screen.ppm /tmp/screen.ppm 2>/dev/null && '
             'convert /tmp/screen.ppm /tmp/screen.png && '
             'cat /tmp/screen.png | base64'],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0 and result.stdout.strip():
            # Decode base64 and save
            import base64
            png_data = base64.b64decode(result.stdout.strip())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(png_data)
            return output_path
        else:
            logger.warning(f"Screenshot capture failed: {result.stderr[:200] if result.stderr else 'No output'}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("Screenshot capture timed out")
        return None
    except Exception as e:
        logger.warning(f"Screenshot capture error: {e}")
        return None


def check_waa_probe(ip: str, timeout: int = 5, internal_ip: str = "172.30.0.2") -> tuple[bool, str | None]:
    """Check if the WAA /probe endpoint is responding.

    Args:
        ip: Public IP address of the Azure VM
        timeout: Connection timeout in seconds
        internal_ip: Internal IP of the Windows VM inside QEMU.
                     172.30.0.2 for dockurr/windows:latest
                     20.20.20.21 for official windowsarena/winarena

    Returns:
        Tuple of (is_ready, response_text)
    """
    import subprocess

    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             f"azureuser@{ip}",
             f"curl -s --connect-timeout {timeout} http://{internal_ip}:5000/probe 2>/dev/null"],
            capture_output=True, text=True, timeout=30
        )
        response = result.stdout.strip() if result.stdout else None
        return bool(response), response
    except subprocess.TimeoutExpired:
        return False, None
    except Exception:
        return False, None


def poll_waa_probe(ip: str, max_attempts: int = 30, interval: int = 20, internal_ip: str = "172.30.0.2") -> bool:
    """Poll the WAA /probe endpoint until it responds or timeout.

    Args:
        ip: Public IP address of the Azure VM
        max_attempts: Maximum number of polling attempts
        interval: Seconds between attempts
        internal_ip: Internal IP of the Windows VM inside QEMU

    Returns:
        True if probe responded, False if timeout
    """
    import time

    print(f"  Polling /probe endpoint at {internal_ip}:5000 (max {max_attempts * interval}s)...")
    print(f"  Monitor Windows at: http://{ip}:8006 (VNC)")
    print()

    for attempt in range(1, max_attempts + 1):
        is_ready, response = check_waa_probe(ip, timeout=5, internal_ip=internal_ip)
        if is_ready:
            print(f"\n  ✓ WAA server is READY after {attempt * interval}s")
            print(f"  Response: {response[:100] if response else '(empty)'}")
            return True
        print(f"  [{attempt}/{max_attempts}] Not ready yet... waiting {interval}s")
        time.sleep(interval)

    print(f"\n  ✗ Timeout after {max_attempts * interval}s")
    return False


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze WAA benchmark results and generate summary statistics.

    Can analyze results from:
    1. Local directory (--results-dir)
    2. Remote Azure VM via SSH (--vm-ip --remote) - faster, no download
    3. Remote Azure VM with download (--vm-ip) - downloads files first

    Outputs per-domain success rates and overall metrics.
    """
    import subprocess
    import tempfile
    from datetime import datetime

    results_dir = args.results_dir
    vm_ip = args.vm_ip
    remote = getattr(args, 'remote', False)
    verbose = getattr(args, 'verbose', False)

    # If --remote flag, run analysis via SSH on the VM
    if vm_ip and remote:
        print(f"Analyzing results on VM at {vm_ip} via SSH...")
        remote_path = "/mnt/WindowsAgentArena/src/win-arena-container/client/results/pyautogui/a11y_tree"

        # Build SSH command to analyze results on VM
        analysis_script = '''
import os
import json
from pathlib import Path

results_path = Path("{remote_path}")
model_dirs = list(results_path.glob("*/0"))

total_tasks = 0
total_success = 0
total_fail = 0
total_incomplete = 0
domain_stats = {{}}
successful_tasks = []
failed_tasks = []

for model_dir in model_dirs:
    model_name = model_dir.parent.name
    for domain_dir in sorted(model_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        tasks = [t for t in domain_dir.iterdir() if t.is_dir()]
        success = fail = incomplete = 0
        for task_dir in tasks:
            result_file = task_dir / "result.txt"
            if result_file.exists():
                result = result_file.read_text().strip()
                if result == "1.0":
                    success += 1
                    successful_tasks.append(f"{{domain}}/{{task_dir.name}}")
                else:
                    fail += 1
                    failed_tasks.append(f"{{domain}}/{{task_dir.name}}")
            else:
                incomplete += 1
        total_tasks += len(tasks)
        total_success += success
        total_fail += fail
        total_incomplete += incomplete
        domain_stats[domain] = {{"total": len(tasks), "success": success, "fail": fail, "incomplete": incomplete}}

result = {{
    "model": model_name if model_dirs else "unknown",
    "total_tasks": total_tasks,
    "evaluated": total_success + total_fail,
    "success": total_success,
    "fail": total_fail,
    "incomplete": total_incomplete,
    "success_rate": total_success / (total_success + total_fail) * 100 if (total_success + total_fail) > 0 else 0,
    "domains": domain_stats,
    "successful_tasks": successful_tasks,
    "failed_tasks": failed_tasks
}}
print(json.dumps(result))
'''.format(remote_path=remote_path)

        try:
            result = subprocess.run([
                "ssh", "-o", "StrictHostKeyChecking=no",
                f"azureuser@{vm_ip}",
                f"python3 -c '{analysis_script}'"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"SSH analysis failed: {result.stderr}")
                return

            data = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            print("SSH timeout")
            return
        except json.JSONDecodeError as e:
            print(f"Failed to parse results: {e}")
            print(f"Output: {result.stdout[:500]}")
            return

        # Display results
        print("\n" + "=" * 60)
        print("WAA BENCHMARK RESULTS ANALYSIS")
        print("=" * 60)
        print(f"\nModel: {data['model']}")
        print("-" * 40)

        for domain, stats in sorted(data['domains'].items()):
            status = "✓" if stats['success'] > 0 else "○"
            rate = f"{stats['success']}/{stats['total']}"
            print(f"  {status} {domain:20s} {rate:8s} ({stats['fail']} fail, {stats['incomplete']} incomplete)")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total tasks:        {data['total_tasks']}")
        print(f"Evaluated:          {data['evaluated']}")
        print(f"Incomplete:         {data['incomplete']}")
        print(f"Successful:         {data['success']}")
        print(f"Failed:             {data['fail']}")
        print(f"Success rate:       {data['success_rate']:.1f}% (of evaluated)")
        if data['total_tasks'] > 0:
            print(f"Completion rate:    {data['evaluated'] / data['total_tasks'] * 100:.1f}%")

        if verbose:
            print("\n" + "-" * 40)
            print("SUCCESSFUL TASKS:")
            for task in data['successful_tasks']:
                print(f"  ✓ {task}")
            print("\nFAILED TASKS:")
            for task in data['failed_tasks']:
                print(f"  ✗ {task}")

        if args.output:
            data['date'] = datetime.now().isoformat()
            Path(args.output).write_text(json.dumps(data, indent=2))
            print(f"\nSummary saved to: {args.output}")

        return

    # If VM IP provided without --remote, fetch results from remote
    if vm_ip and not results_dir:
        print(f"Fetching results from VM at {vm_ip}...")
        print("(Use --remote for faster analysis without downloading)")
        remote_path = "/mnt/WindowsAgentArena/src/win-arena-container/client/results"

        # Create temp directory for results
        results_dir = tempfile.mkdtemp(prefix="waa_results_")
        print(f"Downloading to {results_dir}...")

        try:
            subprocess.run([
                "scp", "-r", "-o", "StrictHostKeyChecking=no",
                f"azureuser@{vm_ip}:{remote_path}/pyautogui",
                results_dir
            ], check=True, capture_output=True)
            results_dir = Path(results_dir) / "pyautogui"
        except subprocess.CalledProcessError as e:
            print(f"Failed to fetch results: {e}")
            return

    if not results_dir:
        print("Error: Provide --results-dir or --vm-ip")
        return

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        return

    # Find the model results directory
    # Structure: pyautogui/a11y_tree/{model}/0/{domain}/{task_id}/
    model_dirs = list(results_path.glob("a11y_tree/*/0"))
    if not model_dirs:
        # Try direct path
        model_dirs = list(results_path.glob("*/0"))
    if not model_dirs:
        print(f"No model results found in {results_path}")
        return

    print("\n" + "=" * 60)
    print("WAA BENCHMARK RESULTS ANALYSIS")
    print("=" * 60)

    total_tasks = 0
    total_success = 0
    total_fail = 0
    total_incomplete = 0
    domain_stats = {}
    successful_tasks = []
    failed_tasks = []

    for model_dir in model_dirs:
        model_name = model_dir.parent.name
        print(f"\nModel: {model_name}")
        print("-" * 40)

        # Iterate through domains
        for domain_dir in sorted(model_dir.iterdir()):
            if not domain_dir.is_dir():
                continue

            domain = domain_dir.name
            tasks = list(domain_dir.iterdir())
            task_count = len([t for t in tasks if t.is_dir()])

            success = 0
            fail = 0
            incomplete = 0

            for task_dir in tasks:
                if not task_dir.is_dir():
                    continue

                result_file = task_dir / "result.txt"
                if result_file.exists():
                    result = result_file.read_text().strip()
                    if result == "1.0":
                        success += 1
                        successful_tasks.append(f"{domain}/{task_dir.name}")
                    else:
                        fail += 1
                        failed_tasks.append(f"{domain}/{task_dir.name}")
                else:
                    incomplete += 1

            total_tasks += task_count
            total_success += success
            total_fail += fail
            total_incomplete += incomplete

            domain_stats[domain] = {
                "total": task_count,
                "success": success,
                "fail": fail,
                "incomplete": incomplete
            }

            # Format output
            status = "✓" if success > 0 else "○"
            rate = f"{success}/{task_count}" if task_count > 0 else "0/0"
            print(f"  {status} {domain:20s} {rate:8s} ({fail} fail, {incomplete} incomplete)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    evaluated = total_success + total_fail
    print(f"Total tasks:        {total_tasks}")
    print(f"Evaluated:          {evaluated}")
    print(f"Incomplete:         {total_incomplete}")
    print(f"Successful:         {total_success}")
    print(f"Failed:             {total_fail}")
    if evaluated > 0:
        print(f"Success rate:       {total_success / evaluated * 100:.1f}% (of evaluated)")
    if total_tasks > 0:
        print(f"Completion rate:    {evaluated / total_tasks * 100:.1f}%")

    if verbose:
        print("\n" + "-" * 40)
        print("SUCCESSFUL TASKS:")
        for task in successful_tasks:
            print(f"  ✓ {task}")
        print("\nFAILED TASKS:")
        for task in failed_tasks:
            print(f"  ✗ {task}")

    # Save summary JSON if requested
    if args.output:
        summary = {
            "date": datetime.now().isoformat(),
            "model": model_name if model_dirs else "unknown",
            "total_tasks": total_tasks,
            "evaluated": evaluated,
            "success": total_success,
            "fail": total_fail,
            "incomplete": total_incomplete,
            "success_rate": total_success / evaluated * 100 if evaluated > 0 else 0,
            "domains": domain_stats,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary saved to: {output_path}")


def cmd_vm(args: argparse.Namespace) -> None:
    """Manage dedicated WAA eval VM with nested virtualization support.

    This creates a standalone Azure VM (not Azure ML compute) that supports
    nested virtualization, which is required for running WAA's Windows VM
    inside Docker/QEMU.
    """
    import subprocess

    vm_name = args.name
    resource_group = args.resource_group
    vm_size = args.size
    location = args.location

    if args.action == "list-sizes":
        print(f"\n=== Available VM Sizes with Nested Virtualization in {location} ===\n")
        print("Checking available D-series sizes (support nested virt)...")

        # Get available sizes
        result = subprocess.run(
            ["az", "vm", "list-skus", "--location", location,
             "--size", "Standard_D", "--all", "--output", "table",
             "--query", "[?restrictions[?reasonCode=='NotAvailableForSubscription']==`[]`].{Name:name, vCPUs:capabilities[?name=='vCPUs'].value|[0], Memory:capabilities[?name=='MemoryGB'].value|[0]}"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)

        print(result.stdout)
        print("\nRecommended sizes for WAA (support nested virt):")
        print("  - Standard_D4s_v3  (4 vCPU, 16GB) ~$0.19/hr")
        print("  - Standard_D8s_v3  (8 vCPU, 32GB) ~$0.38/hr")
        print("  - Standard_D4ds_v5 (4 vCPU, 16GB) ~$0.19/hr")
        print("  - Standard_D8ds_v5 (8 vCPU, 32GB) ~$0.38/hr")
        print(f"\nTry different locations if sizes are unavailable: westus2, centralus, westeurope")
        return

    elif args.action == "create":
        print(f"\n=== Creating WAA Eval VM: {vm_name} ===\n")
        print(f"  Resource Group: {resource_group}")
        print(f"  Location: {location}")
        print(f"  VM Size: {vm_size} (supports nested virtualization)")
        print(f"  OS: Ubuntu 22.04 LTS")
        print()

        # Check if VM already exists
        check = subprocess.run(
            ["az", "vm", "show", "-g", resource_group, "-n", vm_name, "-o", "json"],
            capture_output=True, text=True
        )
        if check.returncode == 0:
            print(f"✗ VM '{vm_name}' already exists. Use 'vm status' to check it or 'vm delete' first.")
            sys.exit(1)

        print("Creating VM (this takes 2-3 minutes)...")
        result = subprocess.run(
            [
                "az", "vm", "create",
                "--resource-group", resource_group,
                "--name", vm_name,
                "--location", location,
                "--image", "Ubuntu2204",
                "--size", vm_size,
                "--admin-username", "azureuser",
                "--generate-ssh-keys",
                "--public-ip-sku", "Standard",
            ],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"✗ Error creating VM: {result.stderr}")
            sys.exit(1)

        # Parse output to get IP
        import json
        vm_info = json.loads(result.stdout)
        public_ip = vm_info.get("publicIpAddress", "unknown")

        print(f"\n✓ VM created successfully!")
        print(f"\n  Public IP: {public_ip}")
        print(f"  SSH command: ssh azureuser@{public_ip}")
        print(f"\n  Next steps:")
        print(f"    1. SSH into the VM: uv run python -m openadapt_ml.benchmarks.cli vm ssh")
        print(f"    2. Verify nested virt: egrep -c '(vmx|svm)' /proc/cpuinfo")
        print(f"    3. Install Docker and run WAA")

    elif args.action == "status":
        print(f"\n=== WAA Eval VM Status: {vm_name} ===\n")

        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "{name:name,powerState:powerState,publicIps:publicIps,size:hardwareProfile.vmSize}",
             "-o", "json"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"✗ VM '{vm_name}' not found in resource group '{resource_group}'")
            print(f"  Create it with: uv run python -m openadapt_ml.benchmarks.cli vm create")
            sys.exit(1)

        import json
        info = json.loads(result.stdout)
        print(f"  Name: {info.get('name')}")
        print(f"  State: {info.get('powerState')}")
        print(f"  Size: {info.get('size')}")
        print(f"  Public IP: {info.get('publicIps')}")

        if info.get('publicIps'):
            print(f"\n  SSH command: ssh azureuser@{info.get('publicIps')}")

    elif args.action == "ssh":
        # Get IP and SSH
        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "publicIps", "-o", "tsv"],
            capture_output=True, text=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            print(f"✗ Could not get IP for VM '{vm_name}'. Is it running?")
            sys.exit(1)

        ip = result.stdout.strip()
        print(f"Connecting to {vm_name} at {ip}...")
        import os
        os.execvp("ssh", ["ssh", f"azureuser@{ip}"])

    elif args.action == "delete":
        print(f"\n=== Deleting WAA Eval VM: {vm_name} ===\n")

        confirm = input(f"Are you sure you want to delete VM '{vm_name}'? (y/N): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

        print("Deleting VM and associated resources...")
        result = subprocess.run(
            ["az", "vm", "delete", "-g", resource_group, "-n", vm_name, "--yes", "--force-deletion"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"✗ Error deleting VM: {result.stderr}")
            sys.exit(1)

        print(f"✓ VM '{vm_name}' deleted")

    elif args.action == "setup":
        print(f"\n=== Setting up WAA Eval VM: {vm_name} ===\n")

        # Get VM IP
        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "publicIps", "-o", "tsv"],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f"✗ Could not get IP for VM '{vm_name}'. Create it first with 'vm create'")
            sys.exit(1)

        ip = result.stdout.strip()
        print(f"  VM IP: {ip}")
        print("\n[1/3] Installing Docker...")

        # Install Docker
        docker_cmd = (
            "sudo apt-get update -qq && "
            "sudo apt-get install -y -qq docker.io && "
            "sudo systemctl start docker && "
            "sudo systemctl enable docker && "
            "sudo usermod -aG docker $USER"
        )
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", docker_cmd],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"✗ Error installing Docker: {result.stderr}")
            sys.exit(1)
        print("  ✓ Docker installed")

        print("\n[2/3] Verifying nested virtualization...")
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
             "egrep -c '(vmx|svm)' /proc/cpuinfo"],
            capture_output=True, text=True
        )
        cpu_count = result.stdout.strip()
        if cpu_count and int(cpu_count) > 0:
            print(f"  ✓ Nested virt supported ({cpu_count} CPUs with vmx/svm)")
        else:
            print(f"  ⚠ Nested virt may not be supported")

        print("\n[3/3] Setup complete!")
        print(f"\n  Next: Pull WAA image with 'vm pull-image'")
        print(f"  Or SSH in: uv run python -m openadapt_ml.benchmarks.cli vm ssh")

    elif args.action == "pull-image":
        print(f"\n=== Pulling WAA Docker Image to VM: {vm_name} ===\n")

        acr_name = args.acr
        acr_url = f"{acr_name}.azurecr.io"
        image = f"{acr_url}/winarena:latest"

        # Get VM IP
        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "publicIps", "-o", "tsv"],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f"✗ Could not get IP for VM '{vm_name}'")
            sys.exit(1)

        ip = result.stdout.strip()
        print(f"  VM IP: {ip}")
        print(f"  Image: {image}")

        print("\n[1/2] Getting ACR access token...")
        result = subprocess.run(
            ["az", "acr", "login", "--name", acr_name, "--expose-token",
             "--query", "accessToken", "-o", "tsv"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"✗ Error getting ACR token: {result.stderr}")
            sys.exit(1)

        token = result.stdout.strip()
        print("  ✓ Got ACR token")

        print("\n[2/2] Logging into ACR and pulling image (this takes 5-10 minutes)...")
        # Login to ACR on VM and pull
        pull_cmd = f"sudo docker login {acr_url} -u 00000000-0000-0000-0000-000000000000 -p '{token}' && sudo docker pull {image}"
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", pull_cmd],
            capture_output=False  # Show output live
        )
        if result.returncode != 0:
            print(f"\n✗ Error pulling image")
            sys.exit(1)

        print(f"\n✓ WAA image pulled successfully!")
        print(f"\n  Image ready: {image}")
        print(f"  Run WAA with: uv run python -m openadapt_ml.benchmarks.cli vm ssh")

    elif args.action == "setup-waa":
        # Comprehensive one-command WAA setup
        print(f"\n{'='*60}")
        print("  WAA Benchmark Setup - Full Automation")
        print(f"{'='*60}\n")
        print("This will set up everything needed to run WAA benchmarks:")
        print("  1. Create Azure VM with nested virtualization")
        print("  2. Install Docker with proper disk configuration")
        print("  3. Pull WAA Docker image from ACR")
        print("  4. Clone WindowsAgentArena repository")
        print("  5. Prepare Windows 11 VM image (~20 min download)")
        print()

        # Get VM IP or create VM
        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "publicIps", "-o", "tsv"],
            capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            ip = result.stdout.strip()
            print(f"[✓] VM already exists: {ip}")
        else:
            print("[1/6] Creating Azure VM with nested virtualization...")
            # Try multiple locations if needed
            locations_to_try = [location, "westus2", "centralus", "eastus2"]
            vm_created = False
            for loc in locations_to_try:
                result = subprocess.run(
                    ["az", "vm", "create",
                     "--resource-group", resource_group,
                     "--name", vm_name,
                     "--location", loc,
                     "--image", "Ubuntu2204",
                     "--size", "Standard_D4ds_v5",  # v5 series supports nested virt
                     "--admin-username", "azureuser",
                     "--generate-ssh-keys",
                     "--public-ip-sku", "Standard"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    import json as json_mod
                    vm_info = json_mod.loads(result.stdout)
                    ip = vm_info.get("publicIpAddress", "")
                    print(f"  ✓ VM created in {loc}: {ip}")
                    vm_created = True
                    break
                else:
                    print(f"  • {loc}: unavailable, trying next...")

            if not vm_created:
                print("✗ Could not create VM in any region")
                sys.exit(1)

        print(f"\n[2/6] Installing Docker with /mnt storage (147GB)...")
        docker_cmds = [
            "sudo apt-get update -qq",
            "sudo apt-get install -y -qq docker.io",
            "sudo systemctl start docker",
            "sudo systemctl enable docker",
            "sudo usermod -aG docker $USER",
            # Configure Docker to use larger /mnt disk
            "sudo systemctl stop docker",
            "sudo mkdir -p /mnt/docker",
            "echo '{\"data-root\": \"/mnt/docker\"}' | sudo tee /etc/docker/daemon.json",
            "sudo systemctl start docker",
        ]
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
             f"azureuser@{ip}", " && ".join(docker_cmds)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠ Docker setup warning: {result.stderr[:200] if result.stderr else 'unknown'}")
        else:
            print("  ✓ Docker installed with /mnt storage")

        print(f"\n[3/6] Verifying nested virtualization...")
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
             "egrep -c '(vmx|svm)' /proc/cpuinfo"],
            capture_output=True, text=True
        )
        cpu_count = result.stdout.strip()
        if cpu_count and int(cpu_count) > 0:
            print(f"  ✓ Nested virt supported ({cpu_count} CPUs with vmx/svm)")
        else:
            print("  ✗ Nested virtualization not supported - WAA won't work")
            sys.exit(1)

        print(f"\n[4/6] Pulling dockurr/windows image (for Windows VM)...")
        # Use dockurr/windows directly - the ACR winarena image has broken dockur
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
             "sudo docker pull dockurr/windows:latest 2>&1 | tail -5"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  ⚠ Image pull warning: {result.stderr[:100] if result.stderr else ''}")
        print("  ✓ Windows image pulled")

        print(f"\n[5/6] Cloning WindowsAgentArena repository...")
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
             "cd ~ && git clone --depth 1 https://github.com/microsoft/WindowsAgentArena.git 2>/dev/null || echo 'Already cloned'"],
            capture_output=True, text=True
        )
        print("  ✓ WAA repo cloned")

        print(f"\n[6/6] Creating WAA config file...")
        api_key = args.api_key or settings.openai_api_key or ""
        if not api_key:
            print("  ⚠ No API key provided. Set with --api-key, OPENAI_API_KEY env var, or in .env file")
            api_key = "placeholder-set-your-key"

        config_cmd = f'''cat > ~/WindowsAgentArena/config.json << 'EOF'
{{
    "OPENAI_API_KEY": "{api_key}",
    "AZURE_API_KEY": "",
    "AZURE_ENDPOINT": ""
}}
EOF'''
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", config_cmd],
            capture_output=True, text=True
        )
        print("  ✓ Config created")

        print(f"\n{'='*60}")
        print("  WAA Setup Complete!")
        print(f"{'='*60}")
        print(f"\n  VM IP: {ip}")
        print(f"\n  Next step: Prepare Windows image (one-time, ~20 min):")
        print(f"    uv run python -m openadapt_ml.benchmarks.cli vm prepare-windows")
        print(f"\n  Or run WAA directly (will auto-prepare on first run):")
        print(f"    uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5")

    elif args.action == "prepare-windows":
        print(f"\n=== Preparing Windows 11 VM for WAA (Fully Automated) ===\n")
        print("This builds a custom WAA container with automatic setup scripts.")
        print("First run downloads Windows 11 (~7GB). Setup is fully automatic - no VNC needed.\n")

        # Get VM IP
        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "publicIps", "-o", "tsv"],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f"✗ VM '{vm_name}' not found. Run 'vm setup-waa' first.")
            sys.exit(1)
        ip = result.stdout.strip()

        print(f"  VM IP: {ip}")
        print(f"  Monitor progress: http://{ip}:8006 (VNC) or via viewer")
        print()

        # Step 1: Build automated WAA image with custom unattend.xml
        print("[1/4] Building automated WAA image (with custom unattend.xml)...")

        # Find the Dockerfile in our repo
        dockerfile_path = Path(__file__).parent / "waa" / "Dockerfile"
        if not dockerfile_path.exists():
            print(f"  ✗ Dockerfile not found at: {dockerfile_path}")
            sys.exit(1)

        # Sync Dockerfile to VM and build
        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", str(dockerfile_path),
             f"azureuser@{ip}:~/build-waa/Dockerfile"],
            capture_output=True, text=True
        )

        build_cmd = '''
mkdir -p ~/build-waa
cp -r ~/WindowsAgentArena/src/win-arena-container/vm ~/build-waa/
cd ~/build-waa && docker build -t waa-auto:latest . 2>&1 | tail -10
'''
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", build_cmd],
            capture_output=True, text=True, timeout=300
        )
        if "Successfully" not in result.stdout and result.returncode != 0:
            print(f"  ✗ Failed to build image: {result.stderr}")
            print(f"  Output: {result.stdout}")
            sys.exit(1)
        print("  ✓ WAA image built (waa-auto:latest)")

        # Step 2: Stop existing container and clean up for fresh install
        # Use /mnt/waa-storage for temp disk (115GB) instead of ~/waa-storage (root, <10GB)
        print("\n[2/4] Cleaning up for fresh Windows installation...")
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
             "docker stop winarena 2>/dev/null; docker rm -f winarena 2>/dev/null; " +
             "rm -f /mnt/waa-storage/data.img /mnt/waa-storage/windows.* 2>/dev/null; " +
             "sudo mkdir -p /mnt/waa-storage /mnt/waa-results; " +
             "sudo chown azureuser:azureuser /mnt/waa-storage /mnt/waa-results; " +
             "# Migrate old storage if exists\n" +
             "[ -d ~/waa-storage ] && mv ~/waa-storage/* /mnt/waa-storage/ 2>/dev/null; " +
             "rm -rf ~/waa-storage 2>/dev/null"],
            capture_output=True, text=True
        )
        print("  ✓ Cleanup complete (using /mnt for 115GB temp disk)")

        # Step 3: Start automated WAA container
        # Use VERSION=11e for Windows 11 Enterprise (accepts GVLK keys, no product key dialog)
        # Note: VERSION=11 would download Pro, which also works but is less suitable for benchmarks
        print("\n[3/4] Starting automated WAA container...")
        docker_cmd = '''docker run -d \
  --name winarena \
  --device=/dev/kvm \
  --cap-add NET_ADMIN \
  -p 8006:8006 \
  -p 5000:5000 \
  -p 7100:7100 \
  -p 7200:7200 \
  -v /mnt/waa-storage:/storage \
  -e VERSION=11e \
  -e RAM_SIZE=12G \
  -e CPU_CORES=4 \
  -e DISK_SIZE=64G \
  waa-auto:latest'''

        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", docker_cmd],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  ✗ Failed to start container: {result.stderr}")
            sys.exit(1)
        print("  ✓ WAA container started")

        # Step 4: Wait for Windows to boot (Enterprise edition with GVLK should skip product key)
        print("\n[4/5] Waiting for Windows to boot...")
        print("      Using Windows 11 Enterprise with GVLK key (should skip product key dialog)")

        import time as time_module

        # Wait for Windows installer to start (boot + load installer UI)
        # This typically takes 60-90 seconds
        for i in range(12):  # Wait up to 2 minutes
            time_module.sleep(10)
            # Check docker logs for progress
            log_result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
                 "docker logs winarena 2>&1 | tail -1"],
                capture_output=True, text=True, timeout=30
            )
            last_log = log_result.stdout.strip()[:60] if log_result.stdout else "Starting..."
            print(f"      [{(i+1)*10}s] {last_log}...")

            # If Windows has started, the log will show "Windows started successfully"
            if "Windows started" in log_result.stdout:
                print("      Windows installer UI ready")
                break
        else:
            print("      Continuing (Windows may still be loading)...")

        # Send keystrokes to bypass product key dialog
        # We need to try multiple times as the dialog timing can vary
        print("      Sending Tab+Enter to click 'I don't have a product key'...")
        for attempt in range(5):
            time_module.sleep(5)  # Give dialog time to appear
            bypass_product_key_dialog(ip)

        print("  ✓ Product key dialog bypass attempted")
        print(f"      If stuck at product key, VNC to http://{ip}:8006 and:")
        print("        1. Click 'I don't have a product key' link")
        print("        2. Select 'Windows 11 Enterprise Evaluation'")

        # Step 5: Poll /probe endpoint until WAA server is ready
        print("\n[5/5] Waiting for Windows install + WAA server (fully automatic)...")
        print(f"      VNC: http://{ip}:8006")
        print("      Expected time: ~10-15 minutes\n")

        import time
        for i in range(90):  # Wait up to 15 minutes
            time.sleep(10)

            # Check if WAA server /probe endpoint responds
            try:
                probe_result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     f"azureuser@{ip}",
                     "curl -s --connect-timeout 3 http://20.20.20.21:5000/probe 2>/dev/null"],
                    capture_output=True, text=True, timeout=30
                )
            except subprocess.TimeoutExpired:
                probe_result = None

            if probe_result and probe_result.stdout.strip():
                print(f"\n✓ WAA Server ready!")
                print(f"\n  Windows VNC: http://{ip}:8006")
                print(f"  WAA Server: http://{ip}:5000 (internal: 20.20.20.21:5000)")
                print(f"  QMP Port: {ip}:7200")
                print(f"\n  To run WAA benchmarks:")
                print(f"    uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5")
                break

            # Show progress from docker logs
            log_result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
                 "docker logs winarena 2>&1 | tail -2"],
                capture_output=True, text=True
            )
            last_log = log_result.stdout.strip().split('\n')[-1][:70] if log_result.stdout else "Starting..."
            print(f"  [{(i+1)*10}s] {last_log}...")
        else:
            print(f"\n⚠ Timeout waiting for WAA server. Check: http://{ip}:8006")
            print("  The Windows VM may still be installing. Try again later.")
            print("  Note: First-time Windows setup can take 15-20 minutes.")

    elif args.action == "run-waa":
        print(f"\n=== Running WAA Benchmark ===\n")

        # Get VM IP
        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "publicIps", "-o", "tsv"],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f"✗ VM '{vm_name}' not found. Run 'vm setup-waa' first.")
            sys.exit(1)
        ip = result.stdout.strip()

        num_tasks = args.num_tasks
        model = getattr(args, 'model', 'gpt-4o')
        agent = getattr(args, 'agent', 'navi')

        print(f"  VM IP: {ip}")
        print(f"  Model: {model}")
        print(f"  Agent: {agent}")
        print(f"  Tasks: {num_tasks}")
        print(f"\n  Monitor Windows at: http://{ip}:8006")
        print()

        # Get API key from args, settings, or environment (in priority order)
        api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else (settings.openai_api_key or "")
        if not api_key:
            print("✗ No API key provided. Set with --api-key, OPENAI_API_KEY env var, or in .env file")
            sys.exit(1)

        # Stop any existing container
        print("[1/2] Stopping any existing WAA container...")
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
             "docker stop winarena 2>/dev/null; docker rm -f winarena 2>/dev/null"],
            capture_output=True, text=True
        )

        # Start WAA container with full benchmark run
        print("[2/2] Starting WAA benchmark (this will take a while)...")
        print(f"      Agent will run {num_tasks} tasks using {model}")
        print()

        # Use official WAA container with start-client true
        # Storage must be on /mnt (bigger disk) not root
        docker_cmd = f'''docker run --rm \
  --name winarena \
  --device=/dev/kvm \
  --cap-add NET_ADMIN \
  -p 8006:8006 \
  -p 5000:5000 \
  -p 7200:7200 \
  -v /mnt/docker/storage:/storage \
  -v ~/waa-results:/results \
  -e OPENAI_API_KEY="{api_key}" \
  windowsarena/winarena:latest \
  "/entry.sh --start-client true --model {model} --agent {agent} --result-dir /results"'''

        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=60",
             f"azureuser@{ip}", f"mkdir -p ~/waa-results && {docker_cmd}"],
            timeout=7200  # 2 hour timeout for full benchmark
        )

        if result.returncode == 0:
            print(f"\n✓ WAA evaluation complete!")
            print(f"\n  Results saved to: ~/waa-results on the VM")
            print(f"  To download: scp azureuser@{ip}:~/waa-results/* ./benchmark_results/")
        else:
            print(f"\n⚠ WAA run finished with issues (exit code: {result.returncode})")

    elif args.action == "fix-storage":
        print(f"\n=== Fix WAA Storage (Move to /mnt for More Space) ===\n")
        print("Moves WAA storage from root disk (~10GB free) to /mnt temp disk (~115GB free).\n")

        # Get VM IP
        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "publicIps", "-o", "tsv"],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f"✗ VM '{vm_name}' not found. Run 'vm setup-waa' first.")
            sys.exit(1)
        ip = result.stdout.strip()

        print(f"  VM IP: {ip}")
        print()

        # Step 1: Check current storage
        print("[1/4] Checking current storage situation...")
        check_cmd = """
df -h / /mnt 2>/dev/null | grep -E 'Filesystem|/dev'
echo '---'
docker inspect winarena --format='Storage: {{range .Mounts}}{{.Source}}{{end}}' 2>/dev/null || echo 'No container running'
"""
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", check_cmd],
            capture_output=True, text=True
        )
        print(result.stdout)

        # Step 2: Stop container
        print("[2/4] Stopping WAA container...")
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
             "docker stop winarena 2>/dev/null; docker rm winarena 2>/dev/null"],
            capture_output=True, text=True
        )
        print("  ✓ Container stopped")

        # Step 3: Move storage to /mnt
        print("\n[3/4] Moving storage to /mnt (preserves Windows image)...")
        move_cmd = """
sudo mkdir -p /mnt/waa-storage
sudo chown azureuser:azureuser /mnt/waa-storage
# Move existing storage if any
if [ -d ~/waa-storage ]; then
    mv ~/waa-storage/* /mnt/waa-storage/ 2>/dev/null
    rm -rf ~/waa-storage
    echo "Moved from ~/waa-storage"
fi
# Also check /home/azureuser/waa-storage explicitly
if [ -d /home/azureuser/waa-storage ]; then
    mv /home/azureuser/waa-storage/* /mnt/waa-storage/ 2>/dev/null
    rm -rf /home/azureuser/waa-storage
    echo "Moved from /home/azureuser/waa-storage"
fi
ls -lh /mnt/waa-storage/
"""
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", move_cmd],
            capture_output=True, text=True
        )
        print(result.stdout)
        print("  ✓ Storage moved to /mnt/waa-storage")

        # Step 4: Restart container with new mount
        print("\n[4/4] Restarting WAA container with /mnt storage...")
        docker_cmd = '''docker run -d \
  --name winarena \
  --device=/dev/kvm \
  --cap-add NET_ADMIN \
  -p 8006:8006 \
  -p 5000:5000 \
  -p 7200:7200 \
  -v /mnt/waa-storage:/storage \
  -e RAM_SIZE=12G \
  -e CPU_CORES=4 \
  -e DISK_SIZE=64G \
  waa-auto:latest'''

        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", docker_cmd],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  ✗ Failed to start container: {result.stderr}")
            sys.exit(1)
        print("  ✓ WAA container restarted with /mnt storage")

        print(f"\n{'='*60}")
        print("  Storage Fixed!")
        print(f"{'='*60}")
        print(f"\n  Storage now on /mnt: ~115GB available")
        print(f"  VNC: http://{ip}:8006")
        print(f"\n  If Windows was installing, it will resume automatically.")
        print(f"  Monitor: uv run python -m openadapt_ml.benchmarks.cli vm status")

    elif args.action == "reset-windows":
        print(f"\n=== Reset Windows (Clean Install) ===\n")
        print("Deletes existing Windows disk image and does a fresh install.\n")

        # Get VM IP
        result = subprocess.run(
            ["az", "vm", "show", "-d", "-g", resource_group, "-n", vm_name,
             "--query", "publicIps", "-o", "tsv"],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f"✗ VM '{vm_name}' not found. Run 'vm setup-waa' first.")
            sys.exit(1)
        ip = result.stdout.strip()

        print(f"  VM IP: {ip}")
        print()

        # Step 1: Stop container
        print("[1/3] Stopping WAA container...")
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
             "docker stop winarena 2>/dev/null; docker rm winarena 2>/dev/null"],
            capture_output=True, text=True
        )
        print("  ✓ Container stopped")

        # Step 2: Delete Windows disk image (keep ISO for faster reinstall)
        print("\n[2/3] Deleting corrupted disk image (keeping ISO cache)...")
        cleanup_cmd = """
# Ensure storage is on /mnt
sudo mkdir -p /mnt/waa-storage
sudo chown azureuser:azureuser /mnt/waa-storage
# Move from home if needed
[ -d ~/waa-storage ] && mv ~/waa-storage/* /mnt/waa-storage/ 2>/dev/null && rm -rf ~/waa-storage
# Delete disk image but keep ISO cache
rm -f /mnt/waa-storage/data.img /mnt/waa-storage/windows.mac /mnt/waa-storage/windows.rom /mnt/waa-storage/windows.vars
ls -lh /mnt/waa-storage/
"""
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", cleanup_cmd],
            capture_output=True, text=True
        )
        print(result.stdout)
        print("  ✓ Disk image deleted (ISO cache preserved for faster reinstall)")

        # Step 3: Restart with fresh install
        print("\n[3/3] Starting fresh Windows installation...")
        docker_cmd = '''docker run -d \
  --name winarena \
  --device=/dev/kvm \
  --cap-add NET_ADMIN \
  -p 8006:8006 \
  -p 5000:5000 \
  -p 7200:7200 \
  -v /mnt/waa-storage:/storage \
  -e RAM_SIZE=12G \
  -e CPU_CORES=4 \
  -e DISK_SIZE=64G \
  waa-auto:latest'''

        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}", docker_cmd],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  ✗ Failed to start container: {result.stderr}")
            sys.exit(1)
        print("  ✓ Fresh Windows installation started")

        # Wait and monitor
        print(f"\n  VNC: http://{ip}:8006")
        print("  Windows will install automatically (~10-15 min)...")
        print("  WAA server will start on port 5000 when ready.\n")

        import time
        for i in range(45):  # Wait up to 15 minutes
            time.sleep(20)

            # Check if WAA server /probe endpoint responds
            try:
                probe_result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     f"azureuser@{ip}",
                     "curl -s --connect-timeout 3 http://20.20.20.21:5000/probe 2>/dev/null"],
                    capture_output=True, text=True, timeout=30
                )
            except subprocess.TimeoutExpired:
                probe_result = None

            if probe_result and probe_result.stdout.strip():
                print(f"\n✓ WAA Server ready!")
                print(f"  Probe response: {probe_result.stdout.strip()[:100]}")
                print(f"\n  To run benchmarks:")
                print(f"    uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5")
                break

            # Show progress from docker logs
            log_result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", f"azureuser@{ip}",
                 "docker logs winarena 2>&1 | tail -2"],
                capture_output=True, text=True
            )
            last_log = log_result.stdout.strip().split('\n')[-1][:70] if log_result.stdout else "Starting..."
            print(f"  [{(i+1)*20}s] {last_log}...")
        else:
            print(f"\n⚠ Timeout waiting for WAA. Check: http://{ip}:8006")
            print("  Windows installation may still be in progress.")

    elif args.action == "screenshot":
        print(f"\n=== Capturing VM Screenshot ===\n")

        ip = get_vm_ip(resource_group, vm_name)
        if not ip:
            print(f"✗ VM '{vm_name}' not found. Run 'vm setup-waa' first.")
            sys.exit(1)

        print(f"  VM IP: {ip}")
        print("  Capturing screenshot via QEMU monitor...")

        output_path = Path("training_output/current/vm_screenshot.png")
        result_path = capture_vm_screenshot(ip, output_path)

        if result_path:
            print(f"  ✓ Screenshot saved to: {result_path}")
            print(f"\n  View at: http://localhost:8080/vm_screenshot.png (if server running)")
        else:
            print("  ✗ Failed to capture screenshot")
            print("  Make sure the winarena container is running and QEMU monitor is accessible.")
            sys.exit(1)

    elif args.action == "probe":
        print(f"\n=== Checking WAA /probe Endpoint ===\n")

        ip = get_vm_ip(resource_group, vm_name)
        if not ip:
            print(f"✗ VM '{vm_name}' not found. Run 'vm setup-waa' first.")
            sys.exit(1)

        print(f"  VM IP: {ip}")

        # Use 172.30.0.2 for our custom waa-auto image (dockurr/windows base)
        # Use 20.20.20.21 for official windowsarena/winarena image
        internal_ip = getattr(args, 'internal_ip', '172.30.0.2')

        if getattr(args, 'wait', False):
            # Polling mode - keep checking until ready
            max_attempts = getattr(args, 'max_attempts', 30)
            interval = getattr(args, 'interval', 20)
            if poll_waa_probe(ip, max_attempts=max_attempts, interval=interval, internal_ip=internal_ip):
                print(f"\n  Ready to run benchmarks:")
                print(f"    uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5")
            else:
                print(f"\n  Check Windows at: http://{ip}:8006 (VNC)")
                sys.exit(1)
        else:
            # Single check mode
            print("  Checking /probe endpoint...")

            is_ready, response = check_waa_probe(ip, internal_ip=internal_ip)

            if is_ready:
                print(f"  ✓ WAA server is READY")
                print(f"  Response: {response[:100] if response else '(empty)'}")
                print(f"\n  Ready to run benchmarks:")
                print(f"    uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 5")
            else:
                print("  ✗ WAA server NOT responding")
                print(f"\n  To poll until ready, use: vm probe --wait")
                print(f"  Check Windows at: http://{ip}:8006 (VNC)")


def cmd_setup(args: argparse.Namespace) -> None:
    """Run full setup (Azure + WAA submodule)."""
    import subprocess

    print("\n=== OpenAdapt-ML WAA Setup ===\n")

    # Step 1: Git submodule
    print("[1/2] Checking WAA submodule...")
    waa_path = find_waa_path()
    if waa_path:
        print(f"  WAA already available at: {waa_path}")
    else:
        print("  Initializing WAA submodule...")
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                check=True,
                capture_output=not args.verbose,
            )
            print("  WAA submodule initialized")
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e}")
            if not args.force:
                sys.exit(1)

    # Step 2: Azure setup
    print("\n[2/2] Azure setup...")
    setup_script = Path(__file__).parent.parent.parent / "scripts" / "setup_azure.py"
    if setup_script.exists():
        cmd = ["python", str(setup_script)]
        if args.yes:
            cmd.append("--yes")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("  Azure setup failed or was cancelled")
            if not args.force:
                sys.exit(1)
    else:
        print(f"  Setup script not found: {setup_script}")
        print("  Run manually: python scripts/setup_azure.py")

    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("  1. Check status:  python -m openadapt_ml.benchmarks.cli status")
    print("  2. Test locally:  python -m openadapt_ml.benchmarks.cli test-mock")
    print("  3. Run on Azure:  python -m openadapt_ml.benchmarks.cli run-azure")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WAA Benchmark CLI - Windows Agent Arena evaluation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Start:
    # First time setup (Azure + WAA submodule)
    python -m openadapt_ml.benchmarks.cli setup

    # Check everything is configured
    python -m openadapt_ml.benchmarks.cli status

    # Test locally with mock adapter
    python -m openadapt_ml.benchmarks.cli test-mock

    # Run on Azure
    python -m openadapt_ml.benchmarks.cli run-azure
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup (new!)
    p_setup = subparsers.add_parser("setup", help="One-command setup (Azure + WAA)")
    p_setup.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    p_setup.add_argument("--force", action="store_true", help="Continue on errors")
    p_setup.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Status
    p_status = subparsers.add_parser("status", help="Check Azure and WAA status")
    p_status.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Az-status (lightweight, no Azure SDK)
    p_az_status = subparsers.add_parser("az-status", help="Check Azure resource status (uses az CLI)")
    p_az_status.add_argument("--resource-group", default="openadapt-agents", help="Azure resource group name")
    p_az_status.add_argument("--workspace", default="openadapt-ml", help="Azure ML workspace name")
    p_az_status.add_argument("--acr-name", default="openadaptacr", help="Azure Container Registry name")

    # Cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Delete all Azure compute instances")
    p_cleanup.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    p_cleanup.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Estimate costs
    p_estimate = subparsers.add_parser("estimate", help="Estimate Azure costs")
    p_estimate.add_argument("--tasks", type=int, default=154, help="Number of tasks")
    p_estimate.add_argument("--workers", type=int, default=1, help="Number of workers (default: 1 for free trial)")
    p_estimate.add_argument("--duration", type=float, default=1.0, help="Avg task duration (minutes)")
    p_estimate.add_argument("--vm-cost", type=float, default=0.19, help="VM hourly cost ($ for D4_v3)")

    # Run local
    p_local = subparsers.add_parser("run-local", help="Run evaluation locally (Windows)")
    p_local.add_argument("--waa-path", help="Path to WAA repository (auto-detected if not specified)")
    p_local.add_argument("--tasks", help="Comma-separated task IDs (default: all)")
    p_local.add_argument("--max-steps", type=int, default=15, help="Max steps per task")
    p_local.add_argument("--agent", default="random", help="Agent type")
    p_local.add_argument("--seed", type=int, default=42, help="Random seed")
    p_local.add_argument("--output", help="Output JSON path")
    p_local.add_argument("--force", action="store_true", help="Force run on non-Windows")
    p_local.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Run Azure
    p_azure = subparsers.add_parser("run-azure", help="Run evaluation on Azure")
    p_azure.add_argument("--config", help="Azure config JSON path")
    p_azure.add_argument("--waa-path", help="Path to WAA repository (auto-detected if not specified)")
    p_azure.add_argument("--workers", type=int, default=1, help="Number of workers (default: 1 for free trial)")
    p_azure.add_argument("--num-tasks", type=int, help="Number of random tasks to run (default: all)")
    p_azure.add_argument("--task-ids", help="Comma-separated specific task IDs to run")
    p_azure.add_argument("--max-steps", type=int, default=15, help="Max steps per task")
    p_azure.add_argument("--agent", default="random", help="Agent type")
    p_azure.add_argument("--seed", type=int, default=42, help="Random seed")
    p_azure.add_argument("--experiment", default="waa-eval", help="Experiment name")
    p_azure.add_argument("--output", help="Output JSON path")
    p_azure.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    p_azure.add_argument("--no-cleanup", action="store_true", help="Don't delete VMs after")
    p_azure.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Test mock
    p_mock = subparsers.add_parser("test-mock", help="Test with mock adapter")
    p_mock.add_argument("--tasks", type=int, default=20, help="Number of mock tasks")
    p_mock.add_argument("--max-steps", type=int, default=10, help="Max steps per task")
    p_mock.add_argument("--seed", type=int, default=42, help="Random seed")

    # Test collection
    p_collection = subparsers.add_parser("test-collection", help="Test benchmark data collection")
    p_collection.add_argument("--tasks", type=int, default=5, help="Number of mock tasks (default: 5)")
    p_collection.add_argument("--max-steps", type=int, default=10, help="Max steps per task (default: 10)")
    p_collection.add_argument("--seed", type=int, default=42, help="Random seed")
    p_collection.add_argument("--model-id", default="random-agent-test", help="Model identifier")
    p_collection.add_argument("--output", default="benchmark_results", help="Output directory")
    p_collection.add_argument("--run-name", help="Run name (default: auto-generated)")

    # Run API-backed evaluation
    p_api = subparsers.add_parser("run-api", help="Run evaluation with API-backed VLM (Claude/GPT-5.1)")
    p_api.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                       help="API provider (anthropic=Claude, openai=GPT-5.1)")
    p_api.add_argument("--tasks", type=int, default=5, help="Number of mock tasks (default: 5)")
    p_api.add_argument("--max-steps", type=int, default=10, help="Max steps per task (default: 10)")
    p_api.add_argument("--max-tokens", type=int, default=512, help="Max tokens for API response")
    p_api.add_argument("--no-a11y", action="store_true", help="Disable accessibility tree in prompt")
    p_api.add_argument("--no-history", action="store_true", help="Disable action history in prompt")
    p_api.add_argument("--output", default="benchmark_results", help="Output directory")
    p_api.add_argument("--run-name", help="Run name (default: auto-generated)")
    p_api.add_argument("--model-id", help="Model identifier (default: {provider}-api)")
    p_api.add_argument("--mock", action="store_true", help="Force use of mock adapter (even if WAA is available)")
    p_api.add_argument("--waa-path", help="Path to WAA repository (auto-detected if not specified)")
    p_api.add_argument("--task-ids", help="Comma-separated task IDs for real WAA")
    p_api.add_argument("--force", action="store_true", help="Force run on non-Windows")
    p_api.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Create config
    p_config = subparsers.add_parser("create-config", help="Create sample Azure config")
    p_config.add_argument("--output", default="azure_config.json", help="Output path")

    # Cleanup VMs (frees quota)
    p_cleanup_vms = subparsers.add_parser("cleanup-vms", help="Clean up Azure compute instances to free quota")
    p_cleanup_vms.add_argument("--resource-group", default="openadapt-agents", help="Azure resource group")
    p_cleanup_vms.add_argument("--workspace", default="openadapt-ml", help="Azure ML workspace name")
    p_cleanup_vms.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # List jobs
    p_list_jobs = subparsers.add_parser("list-jobs", help="List recent Azure ML jobs")
    p_list_jobs.add_argument("--resource-group", default="openadapt-agents", help="Azure resource group")
    p_list_jobs.add_argument("--workspace", default="openadapt-ml", help="Azure ML workspace name")
    p_list_jobs.add_argument("--limit", type=int, default=20, help="Max number of jobs to show")

    # Job logs
    p_job_logs = subparsers.add_parser("job-logs", help="Download and display logs for an Azure ML job")
    p_job_logs.add_argument("job_name", help="Job name (from list-jobs output)")
    p_job_logs.add_argument("--resource-group", default="openadapt-agents", help="Azure resource group")
    p_job_logs.add_argument("--workspace", default="openadapt-ml", help="Azure ML workspace name")

    # Analyze WAA results
    p_analyze = subparsers.add_parser("analyze", help="Analyze WAA benchmark results")
    p_analyze.add_argument("--results-dir", help="Path to results directory (local)")
    p_analyze.add_argument("--vm-ip", help="IP of Azure VM to analyze results on")
    p_analyze.add_argument("--remote", action="store_true", help="Run analysis on VM via SSH (faster, no download)")
    p_analyze.add_argument("--output", help="Output JSON path for summary")
    p_analyze.add_argument("--verbose", "-v", action="store_true", help="Show detailed task-level results")

    # WAA eval VM management
    p_vm = subparsers.add_parser("vm", help="Manage dedicated WAA eval VM (with nested virtualization)")
    p_vm.add_argument("action", choices=["create", "status", "ssh", "delete", "list-sizes", "setup", "pull-image", "setup-waa", "run-waa", "prepare-windows", "fix-storage", "reset-windows", "screenshot", "probe"], help="Action to perform")
    p_vm.add_argument("--resource-group", default="openadapt-agents", help="Azure resource group")
    p_vm.add_argument("--name", default="waa-eval-vm", help="VM name")
    p_vm.add_argument("--size", default="Standard_D4s_v3", help="VM size (must support nested virt)")
    p_vm.add_argument("--location", default="eastus", help="Azure region")
    p_vm.add_argument("--acr", default="openadaptacr", help="Azure Container Registry name")
    p_vm.add_argument("--api-key", help="OpenAI API key for WAA agent (or set OPENAI_API_KEY env var)")
    p_vm.add_argument("--tasks", help="Comma-separated task IDs to run (e.g., notepad_1,notepad_2)")
    p_vm.add_argument("--num-tasks", type=int, default=5, help="Number of tasks to run (for run-waa)")
    p_vm.add_argument("--model", default="gpt-4o", help="Model to use (gpt-4o, gpt-5.2, etc.)")
    p_vm.add_argument("--agent", default="navi", help="Agent type (navi is the default WAA agent)")
    # Probe options
    p_vm.add_argument("--wait", action="store_true", help="For probe: Poll until server is ready")
    p_vm.add_argument("--interval", type=int, default=20, help="For probe: Seconds between poll attempts")
    p_vm.add_argument("--max-attempts", type=int, default=30, help="For probe: Max poll attempts (default 30 = 10min)")
    p_vm.add_argument("--internal-ip", default="172.30.0.2", help="Internal IP of Windows VM (172.30.0.2 for waa-auto, 20.20.20.21 for official)")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "az-status":
        cmd_az_status(args)
    elif args.command == "cleanup":
        cmd_cleanup(args)
    elif args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "run-local":
        setup_logging(getattr(args, 'verbose', False))
        cmd_run_local(args)
    elif args.command == "run-azure":
        setup_logging(getattr(args, 'verbose', False))
        cmd_run_azure(args)
    elif args.command == "test-mock":
        cmd_test_mock(args)
    elif args.command == "test-collection":
        cmd_test_collection(args)
    elif args.command == "run-api":
        cmd_run_api(args)
    elif args.command == "create-config":
        cmd_create_config(args)
    elif args.command == "cleanup-vms":
        cmd_cleanup_vms(args)
    elif args.command == "list-jobs":
        cmd_list_jobs(args)
    elif args.command == "job-logs":
        cmd_job_logs(args)
    elif args.command == "vm":
        cmd_vm(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
