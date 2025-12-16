#!/usr/bin/env python3
"""Fix ACR authentication for existing Azure ML workspace.

This script fixes the ACR authentication issue for workspaces that were created
before the ACR authentication steps were added to setup_azure.py.

It performs these steps:
1. Attaches ACR to ML workspace
2. Grants AcrPull role to workspace managed identity
3. Syncs workspace keys

Usage:
    python scripts/fix_acr_auth.py

    # With custom names
    python scripts/fix_acr_auth.py --resource-group my-group --workspace my-workspace --acr-name myacr
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_cmd(cmd: list[str], check: bool = True) -> str:
    """Run a command and return output."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=error_msg,
        )
    return result.stdout.strip()


def get_subscription_id() -> str:
    """Get current subscription ID."""
    return run_cmd([
        "az", "account", "show",
        "--query", "id",
        "-o", "tsv",
    ])


def attach_acr_to_workspace(acr_name: str, resource_group: str, workspace_name: str, subscription_id: str) -> None:
    """Attach ACR to ML workspace."""
    acr_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.ContainerRegistry/registries/{acr_name}"

    print(f"  [1/3] Attaching ACR to workspace...")
    try:
        run_cmd([
            "az", "ml", "workspace", "update",
            "--name", workspace_name,
            "--resource-group", resource_group,
            "--container-registry", acr_id,
            "-u",
        ])
        print(f"        ✓ ACR attached to workspace")
    except subprocess.CalledProcessError as e:
        if "already" in str(e.stderr).lower():
            print(f"        ✓ ACR already attached")
        else:
            print(f"        ✗ Could not attach ACR: {e.stderr}")
            raise


def grant_acr_pull_role(acr_name: str, resource_group: str, workspace_name: str, subscription_id: str) -> None:
    """Grant AcrPull role to workspace managed identity."""
    print(f"  [2/3] Granting AcrPull role to workspace managed identity...")

    try:
        # Get workspace managed identity principal ID
        output = run_cmd([
            "az", "ml", "workspace", "show",
            "--name", workspace_name,
            "--resource-group", resource_group,
            "--query", "identity.principal_id",
            "-o", "tsv",
        ])
        principal_id = output.strip()

        if not principal_id or principal_id == "None":
            print(f"        ✗ Workspace does not have a managed identity")
            print(f"        Note: Managed identity is automatically created when workspace is used")
            return

        print(f"        Workspace principal ID: {principal_id}")

        # Build ACR resource ID
        acr_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.ContainerRegistry/registries/{acr_name}"

        # Assign AcrPull role
        run_cmd([
            "az", "role", "assignment", "create",
            "--assignee", principal_id,
            "--role", "AcrPull",
            "--scope", acr_id,
        ])
        print(f"        ✓ AcrPull role granted successfully")

    except subprocess.CalledProcessError as e:
        error_msg = str(e.stderr) if e.stderr else str(e)
        if "already exists" in error_msg.lower() or "conflict" in error_msg.lower():
            print(f"        ✓ AcrPull role already assigned")
        else:
            print(f"        ✗ Could not grant AcrPull role: {error_msg}")
            raise


def sync_workspace_keys(workspace_name: str, resource_group: str) -> None:
    """Sync workspace keys."""
    print(f"  [3/3] Syncing workspace keys...")
    try:
        run_cmd([
            "az", "ml", "workspace", "sync-keys",
            "--name", workspace_name,
            "--resource-group", resource_group,
        ])
        print(f"        ✓ Workspace keys synced")
    except subprocess.CalledProcessError as e:
        print(f"        ✗ Could not sync workspace keys: {e.stderr}")
        print(f"        Note: This is usually not critical")


def main():
    parser = argparse.ArgumentParser(
        description="Fix ACR authentication for existing Azure ML workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resource-group", "-g",
        default="openadapt-agents",
        help="Resource group name (default: openadapt-agents)",
    )
    parser.add_argument(
        "--workspace", "-w",
        default="openadapt-ml",
        help="ML workspace name (default: openadapt-ml)",
    )
    parser.add_argument(
        "--acr-name",
        default="openadaptacr",
        help="Azure Container Registry name (default: openadaptacr)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Fixing ACR Authentication")
    print("=" * 60)

    try:
        # Get subscription ID
        subscription_id = get_subscription_id()
        print(f"\nUsing subscription: {subscription_id}")
        print(f"Resource group: {args.resource_group}")
        print(f"Workspace: {args.workspace}")
        print(f"ACR: {args.acr_name}")
        print()

        # Perform fixes
        attach_acr_to_workspace(args.acr_name, args.resource_group, args.workspace, subscription_id)
        grant_acr_pull_role(args.acr_name, args.resource_group, args.workspace, subscription_id)
        sync_workspace_keys(args.workspace, args.resource_group)

        print("\n" + "=" * 60)
        print("ACR Authentication Fixed!")
        print("=" * 60)
        print("""
Your Azure ML workspace can now pull Docker images from ACR.

Test the fix:
  1. Create a compute instance:
     az ml compute create -n test-vm -t ComputeInstance \\
       --size Standard_D2_v3 -w {workspace} -g {resource_group}

  2. Submit a job using the ACR image:
     python -m openadapt_ml.benchmarks.cli run-azure --workers 1

If you still encounter issues, check:
  - ACR admin user is enabled: az acr show -n {acr} --query adminUserEnabled
  - Workspace has managed identity: az ml workspace show -n {workspace} -g {resource_group} --query identity
  - Role assignment exists: az role assignment list --assignee <principal-id> --scope <acr-id>
""".format(
            workspace=args.workspace,
            resource_group=args.resource_group,
            acr=args.acr_name,
        ))

    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("Fix Failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  - You are logged in: az login")
        print("  - Resource group exists: az group show -n", args.resource_group)
        print("  - Workspace exists: az ml workspace show -n", args.workspace, "-g", args.resource_group)
        print("  - ACR exists: az acr show -n", args.acr_name, "-g", args.resource_group)
        sys.exit(1)


if __name__ == "__main__":
    main()
