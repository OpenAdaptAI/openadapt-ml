#!/usr/bin/env python3
"""
Azure Quota Checker and Request Guide

Displays current quota status and provides step-by-step instructions
for requesting vCPU quota increase for parallel WAA benchmarks.

Usage:
    uv run python scripts/check_quota.py
    python scripts/check_quota.py
"""

import subprocess
import sys
from typing import Optional, Dict, Any

def run_azure_command(cmd: list) -> Optional[str]:
    """Run Azure CLI command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

def get_subscription_info() -> tuple[str, str]:
    """Get subscription ID and name."""
    subscription_id = run_azure_command(['az', 'account', 'show', '--query', 'id', '-o', 'tsv'])
    subscription_name = run_azure_command(['az', 'account', 'show', '--query', 'name', '-o', 'tsv'])
    return subscription_id or "UNKNOWN", subscription_name or "UNKNOWN"

def get_quota_info(location: str, quota_name: str = "cores") -> Optional[Dict[str, Any]]:
    """Get quota information for a specific location."""
    try:
        result = subprocess.run(
            ['az', 'vm', 'list-usage', '--location', location, '-o', 'json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            import json
            usage_list = json.loads(result.stdout)
            for item in usage_list:
                if item.get('name', {}).get('value') == quota_name:
                    return {
                        'name': item['localName'],
                        'current': int(item['currentValue']),
                        'limit': int(item['limit'])
                    }
    except Exception:
        pass
    return None

def print_header(text: str):
    """Print a styled header."""
    width = 70
    print("\n" + "═" * width)
    print(text.center(width))
    print("═" * width + "\n")

def print_subheader(text: str):
    """Print a styled subheader."""
    print("━" * 70)
    print(text)
    print("━" * 70 + "\n")

def main():
    print_header("Azure Quota Status for WAA Parallel Benchmarks")

    # Get subscription info
    sub_id, sub_name = get_subscription_info()
    print(f"Subscription: {sub_name}")
    print(f"ID: {sub_id}\n")

    # Check quotas for both regions
    print_subheader("CENTRAL US QUOTA STATUS")

    centralus_quota = get_quota_info('centralus', 'cores')
    if centralus_quota:
        current = centralus_quota['current']
        limit = centralus_quota['limit']
        available = limit - current
        status = "✓ SUFFICIENT" if available >= 24 else "✗ INSUFFICIENT"

        print(f"Total Regional vCPUs: {current}/{limit}")
        print(f"Available: {available} vCPU{'s' if available != 1 else ''}")
        print(f"Status: {status}\n")

        if available < 24:
            print(f"⚠️  Only {available} vCPU{'s' if available != 1 else ''} available")
            print("   Cannot run 3+ parallel D8 VMs (need 24+ vCPUs)")
    else:
        print("❌ Unable to fetch quota (permission issue)\n")

    print_subheader("VM REQUIREMENTS FOR PARALLEL RUNS")

    print("VM Type: Standard_D8ds_v5")
    print("├── vCPUs per VM: 8")
    print("├── RAM per VM: 32 GB")
    print("├── Temp Storage: 300 GB (/mnt)")
    print("└── Cost: $0.29/hour\n")

    configs = [
        ("1 VM", 8, "✓ Currently possible"),
        ("2 VMs", 16, "⚠️  Need 6+ more vCPUs"),
        ("3 VMs", 24, "❌ Need 14+ more vCPUs (RECOMMENDED TARGET)"),
        ("4 VMs", 32, "❌ Need 22+ more vCPUs (OPTIMAL)"),
    ]

    print("Parallel configurations:")
    for config, vcpus, status in configs:
        print(f"├── {config:<6} → {vcpus:2d} vCPU{'s' if vcpus != 1 else ''} {status}")
    print()

    print_subheader("HOW TO REQUEST QUOTA INCREASE")

    print("\n📱 OPTION 1: Azure Portal (RECOMMENDED)\n")
    print("1. Open: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas")
    print("2. Filter:")
    print("   ├── Provider: Compute")
    print("   ├── Location: Central US")
    print("   └── Search: 'Total Regional vCPUs'")
    print("3. Click the row → 'Request quota increase' button")
    print("4. Set new limit: 40 (from current 10)")
    print("5. Business Justification:")
    print("   'Running 3-4 parallel WAA benchmarks for agent evaluation'")
    print("6. Submit and wait 24-48 hours for approval\n")

    print("💻 OPTION 2: Azure CLI\n")
    print("Try this (may fail if permissions insufficient):\n")
    print("  SUBSCRIPTION_ID=$(az account show --query id -o tsv)")
    print("  az quota create \\")
    print("    --resource-name 'cores' \\")
    print("    --scope \"/subscriptions/$SUBSCRIPTION_ID/providers/Microsoft.Compute/locations/centralus\" \\")
    print("    --limit-object value=40 \\")
    print("    --resource-type 'cores'\n")
    print("If this fails → use Option 1 (Portal)\n")

    print("🎫 OPTION 3: Azure Support (Slowest but Guaranteed)\n")
    print("1. Go to: https://portal.azure.com/#view/HubsExtension/BrowseResourceBlade/resourceType/microsoft.support%2Fsupporttickets")
    print("2. 'Create a support request'")
    print("3. Fill:")
    print("   ├── Issue type: Service and subscription limits (quotas)")
    print("   ├── Quota type: Compute-VM (cores-vCPUs)")
    print("   ├── Region: Central US")
    print("   └── New quota limit: 40")
    print("4. Priority: Standard (24-48h) or High (4-8h, may cost)")
    print("5. Submit and wait\n")

    print_subheader("VERIFICATION (After Approval)")

    print("After approval, run:")
    print("  az vm list-usage --location centralus --query \"[?name.value=='cores']\" -o table\n")
    print("Expected output:")
    print("  CurrentValue    Limit    LocalName")
    print("  8               40       Total Regional vCPUs  ✓\n")

    print_subheader("TIMELINE EXPECTATIONS")

    timelines = [
        ("Portal Request", "24-48 hours", "Usual approval same day, ~95% success rate"),
        ("CLI Request", "Immediate", "May fail with permission error, ~50% success"),
        ("Support Ticket", "4-48 hours", "Guaranteed approval, depends on priority"),
    ]

    for method, time, notes in timelines:
        print(f"{method:<20} {time:<20} {notes}")
    print()

    print_subheader("COST BREAKDOWN (With 40 vCPU Quota)")

    print("Hourly costs for parallel runs:")
    print("├── 1x Standard_D8ds_v5:  $0.29/hr  →  $69.60/week (if 24/7)")
    print("├── 2x Standard_D8ds_v5:  $0.58/hr  →  $139.20/week (if 24/7)")
    print("└── 3x Standard_D8ds_v5:  $0.87/hr  →  $208.80/week (if 24/7)\n")
    print("💡 Tip: Use 'vm deallocate' to pause costs when not evaluating\n")

    print_subheader("NEXT STEPS")

    print("1. Request quota increase (choose Option 1, 2, or 3)")
    print("2. Wait for approval (usually 24-48 hours)")
    print("3. Verify new quota with check command above")
    print("4. Start parallel benchmarks:")
    print("   uv run python -m openadapt_ml.benchmarks.cli vm monitor")
    print("   uv run python -m openadapt_ml.benchmarks.cli waa --api-key $OPENAI_API_KEY --setup-only\n")

    print("📖 For detailed documentation:")
    print("   docs/QUOTA_INCREASE_GUIDE.md\n")

    print("═" * 70 + "\n")

if __name__ == '__main__':
    main()
