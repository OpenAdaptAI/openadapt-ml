"""CLI for WAA benchmark evaluation.

Usage:
    # Estimate costs
    python -m openadapt_ml.benchmarks.cli estimate --workers 40

    # Run local evaluation (Windows only)
    python -m openadapt_ml.benchmarks.cli run-local --waa-path /path/to/WAA --tasks notepad_1,notepad_2

    # Run Azure evaluation
    python -m openadapt_ml.benchmarks.cli run-azure --config azure_config.json --workers 40

    # Test with mock adapter
    python -m openadapt_ml.benchmarks.cli test-mock --tasks 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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


def cmd_run_azure(args: argparse.Namespace) -> None:
    """Run evaluation on Azure."""
    from openadapt_ml.benchmarks import RandomAgent
    from openadapt_ml.benchmarks.azure import AzureConfig, AzureWAAOrchestrator

    # Load config
    if args.config:
        config = AzureConfig.from_json(args.config)
    else:
        config = AzureConfig.from_env()

    # Get WAA path (auto-detect if not specified)
    waa_path = get_waa_path(args.waa_path)

    # Parse task IDs
    task_ids = None
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]

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

    num_tasks = len(task_ids) if task_ids else 154
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

    # Run evaluation
    print("\nStarting Azure evaluation...")
    print("  (VM provisioning takes 3-5 minutes - monitor at https://ml.azure.com)")
    print()
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
    p_azure.add_argument("--tasks", help="Comma-separated task IDs (default: all)")
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

    # Create config
    p_config = subparsers.add_parser("create-config", help="Create sample Azure config")
    p_config.add_argument("--output", default="azure_config.json", help="Output path")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "status":
        cmd_status(args)
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
    elif args.command == "create-config":
        cmd_create_config(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
