#!/usr/bin/env python3
"""Azure setup script for WAA benchmark.

This script automates Azure resource creation for running WAA benchmarks:
1. Checks Azure CLI installation
2. Prompts for `az login` if needed
3. Selects subscription
4. Registers required resource providers (Compute, ML, Storage, ContainerRegistry)
5. Creates resource group
6. Creates service principal with Contributor role
7. Creates ML workspace
8. Creates Azure Container Registry (ACR)
9. Imports WAA Docker image to ACR
10. Requests GPU quota (NCv3/V100) - may auto-approve for small requests

Usage:
    python scripts/setup_azure.py

    # With custom names
    python scripts/setup_azure.py --resource-group my-agents --workspace my-workspace

    # Skip interactive prompts (use defaults)
    python scripts/setup_azure.py --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], capture: bool = True, check: bool = True) -> str:
    """Run a command and return output."""
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,  # Don't raise immediately
    )
    if check and result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=error_msg,
        )
    return result.stdout.strip() if capture else ""


def run_cmd_json(cmd: list[str]) -> dict:
    """Run a command and parse JSON output."""
    output = run_cmd(cmd)
    return json.loads(output) if output else {}


def check_az_cli() -> bool:
    """Check if Azure CLI is installed."""
    return shutil.which("az") is not None


def check_az_logged_in() -> bool:
    """Check if user is logged into Azure CLI with valid token.

    We verify by actually making an API call, not just checking cached credentials.
    """
    try:
        # Try to list resource groups - this validates the token is still valid
        run_cmd(["az", "group", "list", "--query", "[]", "-o", "json"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        # Check if it's a token expiry issue
        if "AADSTS700082" in str(e.stderr) or "refresh token has expired" in str(e.stderr).lower():
            return False
        # Check if it's an auth issue
        if "az login" in str(e.stderr).lower():
            return False
        # Other errors might be permission issues, but user is logged in
        return True


def get_subscriptions() -> list[dict]:
    """Get list of Azure subscriptions."""
    return run_cmd_json(["az", "account", "list", "-o", "json"])


def get_current_subscription() -> dict:
    """Get current subscription."""
    return run_cmd_json(["az", "account", "show", "-o", "json"])


def set_subscription(subscription_id: str) -> None:
    """Set the active subscription."""
    run_cmd(["az", "account", "set", "--subscription", subscription_id])


def create_resource_group(name: str, location: str) -> None:
    """Create a resource group if it doesn't exist."""
    try:
        run_cmd(["az", "group", "show", "--name", name])
        print(f"  Resource group '{name}' already exists")
    except subprocess.CalledProcessError:
        print(f"  Creating resource group '{name}' in {location}...")
        try:
            run_cmd(["az", "group", "create", "--name", name, "--location", location])
            print(f"  Resource group '{name}' created")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to create resource group")
            print(f"  {e.stderr}")
            raise


def create_service_principal(name: str, subscription_id: str) -> dict:
    """Create a service principal and return credentials."""
    print(f"  Creating service principal '{name}'...")
    output = run_cmd([
        "az", "ad", "sp", "create-for-rbac",
        "--name", name,
        "--role", "Contributor",
        "--scopes", f"/subscriptions/{subscription_id}",
        "--sdk-auth",
    ])
    creds = json.loads(output)
    print(f"  Service principal created")
    return creds


def check_ml_extension() -> bool:
    """Check if Azure ML CLI extension is installed."""
    try:
        output = run_cmd(["az", "extension", "list", "-o", "json"])
        extensions = json.loads(output)
        return any(ext.get("name") == "ml" for ext in extensions)
    except Exception:
        return False


def install_ml_extension() -> None:
    """Install Azure ML CLI extension."""
    print("  Installing Azure ML CLI extension...")
    run_cmd(["az", "extension", "add", "--name", "ml", "--yes"])
    print("  Extension installed")


def register_resource_providers() -> None:
    """Register required Azure resource providers.

    These providers must be registered before you can create VMs, ML workspaces,
    or view quotas. Registration is idempotent - already registered providers
    are skipped quickly.
    """
    providers = [
        "Microsoft.Compute",  # For VMs and quota visibility
        "Microsoft.MachineLearningServices",  # For ML workspace
        "Microsoft.Storage",  # For storage accounts (required by ML)
        "Microsoft.ContainerRegistry",  # For container images
    ]

    for provider in providers:
        try:
            # Check current state
            output = run_cmd([
                "az", "provider", "show",
                "--namespace", provider,
                "--query", "registrationState",
                "-o", "tsv",
            ])
            state = output.strip().strip('"')

            if state == "Registered":
                print(f"  {provider}: already registered")
            elif state == "Registering":
                print(f"  {provider}: registration in progress...")
            else:
                print(f"  {provider}: registering...")
                run_cmd(["az", "provider", "register", "--namespace", provider])
                print(f"  {provider}: registration started")
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Could not register {provider}: {e.stderr}")

    # Wait for critical providers to complete registration
    critical_providers = ["Microsoft.Compute", "Microsoft.MachineLearningServices"]
    max_wait = 120  # seconds
    wait_interval = 5

    for provider in critical_providers:
        waited = 0
        while waited < max_wait:
            try:
                output = run_cmd([
                    "az", "provider", "show",
                    "--namespace", provider,
                    "--query", "registrationState",
                    "-o", "tsv",
                ])
                state = output.strip().strip('"')
                if state == "Registered":
                    break
                print(f"  Waiting for {provider} registration ({state})...")
                time.sleep(wait_interval)
                waited += wait_interval
            except subprocess.CalledProcessError:
                break

        if waited >= max_wait:
            print(f"  WARNING: {provider} registration taking longer than expected")
            print("  Continuing anyway - registration will complete in background")


def check_gpu_quota(location: str) -> dict[str, int]:
    """Check current GPU quota for common GPU VM families.

    Returns dict of {family_name: current_limit}.
    """
    gpu_families = [
        "standardNCSv3Family",      # V100 GPUs - good for training
        "standardNCFamily",          # K80 GPUs - older but often available
        "standardNDSv2Family",       # V100 8-GPU - high end
        "standardNCADSA100v4Family", # A100 GPUs - newest
    ]

    quotas = {}
    try:
        output = run_cmd([
            "az", "vm", "list-usage",
            "--location", location,
            "-o", "json",
        ])
        usages = json.loads(output)

        for usage in usages:
            name = usage.get("name", {}).get("value", "")
            if name in gpu_families:
                limit = usage.get("limit", 0)
                current = usage.get("currentValue", 0)
                quotas[name] = {"limit": limit, "current": current}
    except subprocess.CalledProcessError:
        pass

    return quotas


def request_gpu_quota(subscription_id: str, location: str, family: str = "standardNCSv3Family", requested_vcpus: int = 8) -> bool:
    """Request GPU quota increase.

    Small requests (0->8 vCPUs) sometimes auto-approve.
    Returns True if request was submitted successfully.
    """
    # Check if quota extension is installed
    try:
        output = run_cmd(["az", "extension", "list", "-o", "json"])
        extensions = json.loads(output)
        has_quota = any(ext.get("name") == "quota" for ext in extensions)
        if not has_quota:
            print("  Installing Azure quota extension...")
            run_cmd(["az", "extension", "add", "--name", "quota", "--yes"])
    except subprocess.CalledProcessError:
        pass

    # Submit quota request
    scope = f"/subscriptions/{subscription_id}/providers/Microsoft.Compute/locations/{location}"

    try:
        print(f"  Submitting quota request for {family} ({requested_vcpus} vCPUs)...")
        run_cmd([
            "az", "quota", "create",
            "--resource-name", family,
            "--scope", scope,
            "--limit-object", f"value={requested_vcpus}",
            "--resource-type", "dedicated",
        ])
        print(f"  Quota request submitted successfully")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = str(e.stderr) if e.stderr else str(e)
        if "already" in error_msg.lower() or "limit" in error_msg.lower():
            print(f"  Quota already at or above requested level")
            return True
        print(f"  WARNING: Quota request failed: {error_msg}")
        print("  You may need to request manually at:")
        print("  https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade")
        return False


def create_ml_workspace(name: str, resource_group: str, location: str) -> None:
    """Create an ML workspace if it doesn't exist."""
    try:
        run_cmd([
            "az", "ml", "workspace", "show",
            "--name", name,
            "--resource-group", resource_group,
        ])
        print(f"  ML workspace '{name}' already exists")
    except subprocess.CalledProcessError:
        print(f"  Creating ML workspace '{name}'...")
        run_cmd([
            "az", "ml", "workspace", "create",
            "--name", name,
            "--resource-group", resource_group,
            "--location", location,
        ])
        print(f"  ML workspace '{name}' created")


def create_container_registry(name: str, resource_group: str) -> str | None:
    """Create Azure Container Registry if it doesn't exist.

    Returns the login server (e.g., 'myacr.azurecr.io') or None if creation failed.
    """
    # Check if ACR already exists
    try:
        output = run_cmd([
            "az", "acr", "show",
            "--name", name,
            "--resource-group", resource_group,
            "--query", "loginServer",
            "-o", "tsv",
        ])
        login_server = output.strip()
        print(f"  Container registry '{name}' already exists")
        return login_server
    except subprocess.CalledProcessError:
        pass

    # Create ACR
    print(f"  Creating container registry '{name}'...")
    try:
        output = run_cmd([
            "az", "acr", "create",
            "--name", name,
            "--resource-group", resource_group,
            "--sku", "Basic",
            "--admin-enabled", "true",
            "--query", "loginServer",
            "-o", "tsv",
        ])
        login_server = output.strip()
        print(f"  Container registry created: {login_server}")
        return login_server
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Could not create container registry: {e.stderr}")
        return None


def attach_acr_to_workspace(acr_name: str, resource_group: str, workspace_name: str, subscription_id: str) -> None:
    """Attach ACR to ML workspace so compute can pull images."""
    acr_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.ContainerRegistry/registries/{acr_name}"

    print(f"  Attaching ACR to workspace...")
    try:
        run_cmd([
            "az", "ml", "workspace", "update",
            "--name", workspace_name,
            "--resource-group", resource_group,
            "--container-registry", acr_id,
            "-u",  # Update dependent resources
        ])
        print(f"  ACR attached to workspace")
    except subprocess.CalledProcessError as e:
        # Might already be attached
        if "already" in str(e.stderr).lower():
            print(f"  ACR already attached")
        else:
            print(f"  WARNING: Could not attach ACR: {e.stderr}")


def create_storage_account(name: str, resource_group: str, location: str) -> str | None:
    """Create Azure Storage account for checkpoints and comparisons.

    Returns connection string or None if creation failed.
    """
    # Check if already exists
    try:
        output = run_cmd([
            "az", "storage", "account", "show",
            "--name", name,
            "--resource-group", resource_group,
            "--query", "name",
            "-o", "tsv",
        ])
        if output.strip():
            print(f"  Storage account '{name}' already exists")
            # Get connection string
            conn_str = run_cmd([
                "az", "storage", "account", "show-connection-string",
                "--name", name,
                "--resource-group", resource_group,
                "--query", "connectionString",
                "-o", "tsv",
            ]).strip()
            return conn_str
    except subprocess.CalledProcessError:
        pass

    # Create storage account
    print(f"  Creating storage account '{name}'...")
    try:
        run_cmd([
            "az", "storage", "account", "create",
            "--name", name,
            "--resource-group", resource_group,
            "--location", location,
            "--sku", "Standard_LRS",  # Locally redundant, cheapest option
        ])
        # Get connection string
        conn_str = run_cmd([
            "az", "storage", "account", "show-connection-string",
            "--name", name,
            "--resource-group", resource_group,
            "--query", "connectionString",
            "-o", "tsv",
        ]).strip()
        print(f"  Storage account created")
        return conn_str
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Could not create storage account: {e.stderr}")
        return None


def create_queue(connection_string: str, queue_name: str) -> bool:
    """Create Azure Queue for inference jobs.

    Returns True if successful.
    """
    try:
        # Check if queue exists
        result = run_cmd([
            "az", "storage", "queue", "exists",
            "--name", queue_name,
            "--connection-string", connection_string,
            "-o", "json",
        ], check=False)

        if result:
            data = json.loads(result)
            if data.get("exists"):
                print(f"  Queue '{queue_name}' already exists")
                return True

        # Create queue
        run_cmd([
            "az", "storage", "queue", "create",
            "--name", queue_name,
            "--connection-string", connection_string,
        ])
        print(f"  Queue '{queue_name}' created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Could not create queue: {e.stderr}")
        return False


def create_blob_containers(connection_string: str, containers: list[str]) -> bool:
    """Create blob containers for checkpoints and comparisons.

    Returns True if all containers created successfully.
    """
    success = True
    for container in containers:
        try:
            # Check if container exists
            result = run_cmd([
                "az", "storage", "container", "exists",
                "--name", container,
                "--connection-string", connection_string,
                "-o", "json",
            ], check=False)

            if result:
                data = json.loads(result)
                if data.get("exists"):
                    print(f"  Container '{container}' already exists")
                    continue

            # Create container
            run_cmd([
                "az", "storage", "container", "create",
                "--name", container,
                "--connection-string", connection_string,
            ])
            print(f"  Container '{container}' created")
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Could not create container '{container}': {e.stderr}")
            success = False

    return success


def import_waa_image(acr_name: str, source_image: str = "docker.io/windowsarena/winarena:latest") -> str | None:
    """Import WAA image from Docker Hub to ACR.

    Returns the full ACR image path (e.g., 'myacr.azurecr.io/winarena:latest') or None.
    """
    target_image = "winarena:latest"

    # Check if image already exists
    try:
        output = run_cmd([
            "az", "acr", "repository", "show",
            "--name", acr_name,
            "--repository", "winarena",
        ])
        acr_login_server = run_cmd([
            "az", "acr", "show",
            "--name", acr_name,
            "--query", "loginServer",
            "-o", "tsv",
        ]).strip()
        print(f"  Image 'winarena' already exists in {acr_name}")
        return f"{acr_login_server}/{target_image}"
    except subprocess.CalledProcessError:
        pass

    # Import the image
    print(f"  Importing WAA image from Docker Hub (this may take a few minutes)...")
    try:
        run_cmd([
            "az", "acr", "import",
            "--name", acr_name,
            "--source", source_image,
            "--image", target_image,
        ])
        acr_login_server = run_cmd([
            "az", "acr", "show",
            "--name", acr_name,
            "--query", "loginServer",
            "-o", "tsv",
        ]).strip()
        full_image = f"{acr_login_server}/{target_image}"
        print(f"  Image imported: {full_image}")
        return full_image
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Could not import image: {e.stderr}")
        print("  You may need to import manually or use Docker Hub directly.")
        return None


def delete_service_principal(name: str) -> bool:
    """Delete a service principal by name."""
    try:
        # Get the app ID first
        output = run_cmd([
            "az", "ad", "sp", "list",
            "--display-name", name,
            "--query", "[0].appId",
            "-o", "tsv",
        ])
        app_id = output.strip()
        if not app_id:
            print(f"  Service principal '{name}' not found")
            return False

        print(f"  Deleting service principal '{name}' (appId: {app_id})...")
        run_cmd(["az", "ad", "sp", "delete", "--id", app_id])
        print(f"  Service principal deleted")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Could not delete service principal: {e.stderr}")
        return False


def delete_resource_group(name: str) -> bool:
    """Delete a resource group and all its resources."""
    try:
        run_cmd(["az", "group", "show", "--name", name])
    except subprocess.CalledProcessError:
        print(f"  Resource group '{name}' not found")
        return False

    print(f"  Deleting resource group '{name}' (this may take a few minutes)...")
    try:
        run_cmd(["az", "group", "delete", "--name", name, "--yes", "--no-wait"])
        print(f"  Resource group deletion started (running in background)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Could not delete resource group: {e.stderr}")
        return False


def remove_azure_from_env(env_path: Path) -> None:
    """Remove Azure credentials from .env file."""
    if not env_path.exists():
        print(f"  {env_path} not found")
        return

    content = env_path.read_text()
    if "AZURE_SUBSCRIPTION_ID=" not in content:
        print(f"  No Azure credentials found in {env_path}")
        return

    lines = content.split("\n")
    new_lines = []
    skip_until_blank = False
    azure_vars = [
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID",
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_ML_RESOURCE_GROUP",
        "AZURE_ML_WORKSPACE_NAME",
    ]

    for line in lines:
        # Skip Azure section header
        if "Azure Credentials (auto-generated" in line:
            skip_until_blank = True
            continue
        if skip_until_blank:
            if line.strip() == "" or line.startswith("#"):
                if line.strip() == "":
                    skip_until_blank = False
                continue

        # Skip Azure variable lines
        is_azure_var = any(line.startswith(f"{var}=") for var in azure_vars)
        if not is_azure_var:
            new_lines.append(line)

    # Remove trailing empty lines
    while new_lines and new_lines[-1].strip() == "":
        new_lines.pop()

    env_path.write_text("\n".join(new_lines) + "\n")
    print(f"  Azure credentials removed from {env_path}")


def write_env_file(
    env_path: Path,
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    client_id: str,
    client_secret: str,
    tenant_id: str,
    acr_image: str | None = None,
    storage_connection_string: str | None = None,
    inference_queue_name: str = "inference-jobs",
    checkpoints_container: str = "checkpoints",
    comparisons_container: str = "comparisons",
) -> None:
    """Write or update .env file with Azure credentials."""
    # Read existing content
    existing_content = ""
    if env_path.exists():
        existing_content = env_path.read_text()

    # Check if Azure section already exists
    if "AZURE_SUBSCRIPTION_ID=" in existing_content:
        print(f"\n  WARNING: Azure credentials already exist in {env_path}")
        print("  Updating existing values...")

        # Update existing values
        lines = existing_content.split("\n")
        new_lines = []
        azure_vars = {
            "AZURE_CLIENT_ID": client_id,
            "AZURE_CLIENT_SECRET": client_secret,
            "AZURE_TENANT_ID": tenant_id,
            "AZURE_SUBSCRIPTION_ID": subscription_id,
            "AZURE_ML_RESOURCE_GROUP": resource_group,
            "AZURE_ML_WORKSPACE_NAME": workspace_name,
            "AZURE_DOCKER_IMAGE": acr_image or "",
            "AZURE_STORAGE_CONNECTION_STRING": storage_connection_string or "",
            "AZURE_INFERENCE_QUEUE_NAME": inference_queue_name,
            "AZURE_CHECKPOINTS_CONTAINER": checkpoints_container,
            "AZURE_COMPARISONS_CONTAINER": comparisons_container,
        }

        for line in lines:
            updated = False
            for var, value in azure_vars.items():
                if line.startswith(f"{var}="):
                    new_lines.append(f"{var}={value}")
                    updated = True
                    break
            if not updated:
                new_lines.append(line)

        env_path.write_text("\n".join(new_lines))
    else:
        # Append Azure section
        acr_line = f"AZURE_DOCKER_IMAGE={acr_image}" if acr_image else "# AZURE_DOCKER_IMAGE= (not configured)"
        storage_line = f"AZURE_STORAGE_CONNECTION_STRING={storage_connection_string}" if storage_connection_string else "# AZURE_STORAGE_CONNECTION_STRING= (not configured)"
        azure_section = f"""
# =============================================================================
# Azure Credentials (auto-generated by setup_azure.py)
# =============================================================================
AZURE_CLIENT_ID={client_id}
AZURE_CLIENT_SECRET={client_secret}
AZURE_TENANT_ID={tenant_id}

AZURE_SUBSCRIPTION_ID={subscription_id}
AZURE_ML_RESOURCE_GROUP={resource_group}
AZURE_ML_WORKSPACE_NAME={workspace_name}
{acr_line}

# Azure Storage for async inference queue (Phase 2)
{storage_line}
AZURE_INFERENCE_QUEUE_NAME={inference_queue_name}
AZURE_CHECKPOINTS_CONTAINER={checkpoints_container}
AZURE_COMPARISONS_CONTAINER={comparisons_container}
"""
        with open(env_path, "a") as f:
            f.write(azure_section)

    print(f"  Credentials written to {env_path}")


def run_cleanup(args: argparse.Namespace) -> None:
    """Clean up Azure resources created by setup."""
    print("\n" + "=" * 60)
    print("Azure Cleanup for WAA Benchmark")
    print("=" * 60)

    # Step 1: Check Azure CLI
    print("\n[1/4] Checking Azure CLI...")
    if not check_az_cli():
        print("  ERROR: Azure CLI not found!")
        sys.exit(1)
    print("  Azure CLI is installed")

    # Step 2: Login
    print("\n[2/4] Logging into Azure...")
    if not check_az_logged_in():
        print("  Running 'az login'...")
        run_cmd(["az", "login"], capture=False, check=True)
    else:
        print("  Already logged in")

    # Confirm before proceeding
    print(f"\n  Resources to delete:")
    print(f"    - Resource group: {args.resource_group}")
    print(f"    - Service principal: {args.sp_name}")
    print(f"    - Credentials in: {args.env_file}")

    if not args.yes:
        confirm = input("\n  This will delete ALL resources. Continue? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            sys.exit(0)

    # Step 3: Delete resources
    print("\n[3/4] Deleting Azure resources...")
    delete_resource_group(args.resource_group)
    delete_service_principal(args.sp_name)

    # Step 4: Remove from .env
    print("\n[4/4] Cleaning up local files...")
    env_path = Path(args.env_file)
    remove_azure_from_env(env_path)

    print("\n" + "=" * 60)
    print("Cleanup Complete!")
    print("=" * 60)
    print("""
Note: Resource group deletion runs in the background and may take
a few minutes to complete. You can check status in Azure Portal:
  https://portal.azure.com/#view/HubsExtension/BrowseResourceGroups
""")


def main():
    parser = argparse.ArgumentParser(
        description="Set up or tear down Azure resources for WAA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Setup
    python scripts/setup_azure.py
    python scripts/setup_azure.py --resource-group my-agents --workspace my-ml
    python scripts/setup_azure.py --yes --location westus2

    # Cleanup (delete all resources)
    python scripts/setup_azure.py --cleanup
    python scripts/setup_azure.py --cleanup --yes
        """,
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
        "--location", "-l",
        default="eastus",
        help="Azure region (default: eastus)",
    )
    parser.add_argument(
        "--sp-name",
        default="openadapt-ml-waa",
        help="Service principal name (default: openadapt-ml-waa)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--subscription", "-s",
        help="Subscription ID or name to use (skips selection prompt)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete all Azure resources instead of creating them",
    )
    parser.add_argument(
        "--acr-name",
        default="openadaptacr",
        help="Azure Container Registry name (default: openadaptacr)",
    )
    args = parser.parse_args()

    # Run cleanup if requested
    if args.cleanup:
        run_cleanup(args)
        return

    print("\n" + "=" * 60)
    print("Azure Setup for WAA Benchmark")
    print("=" * 60)

    # Step 1: Check Azure CLI
    print("\n[1/12] Checking Azure CLI...")
    if not check_az_cli():
        print("  ERROR: Azure CLI not found!")
        print("\n  Install Azure CLI:")
        print("    macOS:   brew install azure-cli")
        print("    Windows: winget install Microsoft.AzureCLI")
        print("    Linux:   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash")
        print("\n  Then run this script again.")
        sys.exit(1)
    print("  Azure CLI is installed")

    # Step 2: Login and select subscription
    print("\n[2/12] Logging into Azure...")
    print("  Running 'az login'...")
    print("  (A browser window will open - select the account with access to your target subscription)")
    run_cmd(["az", "login"], capture=False, check=True)

    # Show which account was used
    current = get_current_subscription()
    user_info = current.get("user", {})
    user_name = user_info.get("name", "unknown")
    print(f"\n  Logged in as: {user_name}")

    if not args.yes:
        confirm = input("  Is this the correct account? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("\n  Please run 'az logout' and try again with the correct account.")
            print("  Or run: az login --use-device-code")
            sys.exit(0)

    print("  Login confirmed")

    # Step 3: Select subscription
    print("\n[3/12] Selecting subscription...")
    subscriptions = get_subscriptions()

    if len(subscriptions) == 0:
        print("  ERROR: No subscriptions found!")
        print("  Make sure you logged in with an account that has Azure subscriptions.")
        sys.exit(1)

    subscription = None

    # If --subscription is specified, find it
    if args.subscription:
        for sub in subscriptions:
            if args.subscription in (sub["id"], sub["name"]):
                subscription = sub
                break
        if not subscription:
            print(f"  ERROR: Subscription '{args.subscription}' not found!")
            print("  Available subscriptions:")
            for sub in subscriptions:
                print(f"    - {sub['name']} ({sub['id']})")
            print("\n  If you need a different subscription, run 'az login' with a different account.")
            sys.exit(1)
        print(f"  Using specified subscription: {subscription['name']}")
    elif len(subscriptions) == 1:
        subscription = subscriptions[0]
        print(f"  Using subscription: {subscription['name']}")
    else:
        current = get_current_subscription()
        print(f"  Current subscription: {current['name']}")
        print("\n  Available subscriptions:")
        for i, sub in enumerate(subscriptions):
            marker = " *" if sub["id"] == current["id"] else ""
            print(f"    [{i + 1}] {sub['name']} ({sub['id'][:8]}...){marker}")

        if not args.yes:
            choice = input("\n  Select subscription [1]: ").strip() or "1"
            subscription = subscriptions[int(choice) - 1]
        else:
            subscription = current

    set_subscription(subscription["id"])

    subscription_id = subscription["id"]
    print(f"  Subscription ID: {subscription_id}")

    # Step 4: Register resource providers
    print("\n[4/12] Registering resource providers...")
    register_resource_providers()

    # Step 5: Create resource group
    print(f"\n[5/12] Creating resource group '{args.resource_group}'...")
    create_resource_group(args.resource_group, args.location)

    # Step 6: Create service principal
    print("\n[6/12] Creating service principal...")
    creds = create_service_principal(args.sp_name, subscription_id)

    # Step 7: Install ML extension and create workspace
    print(f"\n[7/12] Creating ML workspace '{args.workspace}'...")
    if not check_ml_extension():
        install_ml_extension()
    create_ml_workspace(args.workspace, args.resource_group, args.location)

    # Step 8: Create container registry
    print(f"\n[8/12] Creating container registry '{args.acr_name}'...")
    acr_login_server = create_container_registry(args.acr_name, args.resource_group)

    # Step 9: Import WAA image
    acr_image = None
    if acr_login_server:
        print(f"\n[9/12] Importing WAA Docker image...")
        acr_image = import_waa_image(args.acr_name)
    else:
        print(f"\n[9/12] Skipping WAA image import (no ACR)...")

    # Step 10: Request GPU quota
    print(f"\n[10/12] Requesting GPU quota...")
    print("  Checking current GPU quotas...")
    quotas = check_gpu_quota(args.location)

    if quotas:
        print("  Current GPU quotas:")
        for family, info in quotas.items():
            print(f"    {family}: {info['current']}/{info['limit']} vCPUs")

    # Request quota for NCv3 (V100 GPUs) - good balance of availability and performance
    # 6 vCPUs = 1 NC6s_v3 VM (1 V100 GPU)
    gpu_quota_requested = request_gpu_quota(
        subscription_id=subscription_id,
        location=args.location,
        family="standardNCSv3Family",
        requested_vcpus=6,  # 1 V100 GPU
    )

    # Step 11: Create storage account for async inference
    print(f"\n[11/12] Creating storage account for inference queue...")
    storage_account_name = "openadaptmlstorage"  # Must be unique, lowercase, no hyphens
    storage_connection_string = create_storage_account(
        storage_account_name,
        args.resource_group,
        args.location
    )

    # Step 12: Create queue and blob containers
    inference_queue_name = "inference-jobs"
    checkpoints_container = "checkpoints"
    comparisons_container = "comparisons"

    if storage_connection_string:
        print(f"\n[12/12] Creating inference queue and blob containers...")
        create_queue(storage_connection_string, inference_queue_name)
        create_blob_containers(
            storage_connection_string,
            [checkpoints_container, comparisons_container]
        )
    else:
        print(f"\n[12/12] Skipping queue/container creation (no storage account)...")

    # Write to .env
    print("\n[✓] Writing credentials to .env...")
    env_path = Path(args.env_file)
    write_env_file(
        env_path=env_path,
        subscription_id=subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace,
        client_id=creds["clientId"],
        client_secret=creds["clientSecret"],
        tenant_id=creds["tenantId"],
        acr_image=acr_image,
        storage_connection_string=storage_connection_string,
        inference_queue_name=inference_queue_name,
        checkpoints_container=checkpoints_container,
        comparisons_container=comparisons_container,
    )

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)

    gpu_status = "✓ Quota request submitted (may take hours to approve)" if gpu_quota_requested else "⚠ Manual request needed"

    print(f"""
Resources created:
  - Resource group: {args.resource_group}
  - ML workspace: {args.workspace}
  - Service principal: {args.sp_name}
  - Container registry: {args.acr_name}
  - GPU quota: {gpu_status}
  - Credentials: {env_path}

Next steps:
  1. Check GPU quota status (auto-approval can take minutes to hours):
     az vm list-usage --location {args.location} -o table | grep -i nc

  2. If quota wasn't auto-approved, request manually:
     https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade
     Select: Standard NCSv3 Family → Request increase to 6+ vCPUs

  3. Test the setup:
     python -m openadapt_ml.benchmarks.cli estimate --workers 1

  4. Run GPU training (once quota approved):
     python -m openadapt_ml.scripts.train --config configs/qwen3vl_capture.yaml \\
       --capture /path/to/capture --azure

  5. Run WAA evaluation:
     python -m openadapt_ml.benchmarks.cli run-azure --workers 1
""")


if __name__ == "__main__":
    main()
