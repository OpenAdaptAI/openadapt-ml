# Azure Setup for WAA Benchmark

This guide covers setting up Azure resources to run Windows Agent Arena (WAA) evaluations.

## Quick Start: Single VM (Recommended)

The simplest way to run WAA - one command sets up everything:

```bash
# 1. Setup VM with Docker, WAA repo, and API key (takes ~5 min)
uv run python -m openadapt_ml.benchmarks.cli vm setup-waa --api-key $OPENAI_API_KEY

# 2. Prepare Windows golden image (takes ~25 min, fully automated)
uv run python -m openadapt_ml.benchmarks.cli vm prepare-windows

# 3. Run benchmark
uv run python -m openadapt_ml.benchmarks.cli vm run-waa --num-tasks 30

# 4. Delete VM when done (IMPORTANT: stops billing!)
uv run python -m openadapt_ml.benchmarks.cli vm delete
```

**Cost**: ~$0.50/hour for Standard_D4ds_v5 VM. Delete when done!

**What this does**:
- Creates Azure VM with nested virtualization support
- Builds custom `waa-auto` Docker image (fixes OEM folder issues)
- Installs Windows 11 automatically via QEMU
- Runs WAA benchmark tasks with your chosen model

### Other VM Commands

```bash
# Check VM status
uv run python -m openadapt_ml.benchmarks.cli vm status

# SSH into VM for debugging
uv run python -m openadapt_ml.benchmarks.cli vm ssh

# Check if WAA server is ready
uv run python -m openadapt_ml.benchmarks.cli vm probe

# Reset Windows (if stuck)
uv run python -m openadapt_ml.benchmarks.cli vm reset-windows
```

---

## Alternative: Azure ML Parallel Workers

For running 40+ tasks in parallel across multiple VMs:

```bash
# 1. Run automated setup (creates resources, writes credentials to .env)
python scripts/setup_azure.py

# 2. Estimate costs
python -m openadapt_ml.benchmarks.cli estimate

# 3. Run evaluation (requires WAA repo)
python -m openadapt_ml.benchmarks.cli run-azure --waa-path /path/to/WAA
```

## Prerequisites

1. **Azure CLI** - Install if not present:
   ```bash
   # macOS
   brew install azure-cli

   # Windows
   winget install Microsoft.AzureCLI

   # Linux
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

2. **Azure Account** - Free trial includes:
   - $200 credit for 30 days
   - 4 vCPU quota limit (1 worker with D4_v3 VM)
   - Upgrade to pay-as-you-go for more workers (keeps the $200 credit)

3. **Python packages**:
   ```bash
   pip install azure-ai-ml azure-identity
   # Or with uv:
   uv add --optional azure azure-ai-ml azure-identity
   ```

## Automated Setup

The setup script handles most configuration automatically:

```bash
python scripts/setup_azure.py
```

This will:
1. Open browser for Azure login
2. Let you select a subscription
3. Register required resource providers (Compute, ML, Storage, ContainerRegistry)
4. Create resource group (`openadapt-agents`)
5. Create service principal with Contributor role
6. Create Azure ML workspace (`openadapt-ml`)
7. Write credentials to `.env`

**Note**: Resource provider registration enables quota viewing and VM provisioning. This is done automatically - no manual portal steps required.

### Options

```bash
# Use specific subscription
python scripts/setup_azure.py --subscription "My Subscription Name"

# Custom resource names
python scripts/setup_azure.py \
  --resource-group my-agents \
  --workspace my-ml-workspace \
  --location westus2

# Skip confirmation prompts
python scripts/setup_azure.py --yes
```

## Free Trial Limitations

Azure free trials have a **4 vCPU quota limit** and cannot request quota increases. The defaults are configured to work within this limit:

- **Default VM**: Standard_D4_v3 (4 vCPUs, 16GB RAM)
- **Default workers**: 1 (uses all 4 vCPUs)
- **Estimated cost**: ~$0.50 per full evaluation (154 tasks, ~2.5 hours)

### Upgrade for More Workers

To run more workers in parallel, upgrade to pay-as-you-go:

1. Go to [Azure Subscriptions](https://portal.azure.com/#view/Microsoft_Azure_Billing/SubscriptionsBlade)
2. Click on your subscription
3. Click "Upgrade" to convert to pay-as-you-go
4. Your $200 credit is preserved
5. Request quota increase (see below)

## Request vCPU Quota Increase

After upgrading from free trial, you can request more vCPUs. For 40 workers using Standard_D4_v3 VMs (4 vCPUs each), you need 160 vCPUs.

### Steps to Request Quota Increase

1. **Go to Azure Portal Quotas page**:

   https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas

2. **Filter quotas**:
   - Click "Compute" in the left sidebar
   - Select your subscription
   - Search for "Standard Dv3 Family"

3. **Request increase**:
   - Click on "Standard Dv3 Family Cluster Dedicated vCPUs"
   - Click "Request quota increase" (pencil icon)
   - Enter new limit: `320` (or more)
   - Add justification: "Running Windows Agent Arena benchmark for GUI agent evaluation"
   - Submit request

4. **Wait for approval**:
   - Usually approved within minutes to hours
   - Check email for confirmation

### Alternative: Use Fewer Workers

If quota increase isn't approved, use fewer workers:

```bash
# 10 workers = 80 vCPUs (often within default quota)
python -m openadapt_ml.benchmarks.cli run-azure --waa-path /path/to/WAA --workers 10
```

| Workers | vCPUs Needed | Duration | Cost | Notes |
|---------|--------------|----------|------|-------|
| 1       | 4            | ~2.5 hours | ~$0.50 | Free trial compatible |
| 10      | 40           | ~20 min  | ~$0.60 | Requires upgrade |
| 40      | 160          | ~8 min   | ~$1.00 | Requires upgrade + quota |

## Manual Setup (Alternative)

If you prefer manual control:

### 1. Login to Azure

```bash
az login
```

### 2. Get Subscription ID

```bash
az account show --query id -o tsv
```

### 3. Create Service Principal

```bash
az ad sp create-for-rbac \
  --name "openadapt-ml-waa" \
  --role "Contributor" \
  --scopes "/subscriptions/<your-subscription-id>" \
  --sdk-auth
```

Save the output - you'll need `clientId`, `clientSecret`, `tenantId`.

### 4. Create Resource Group

```bash
az group create --name openadapt-agents --location eastus
```

### 5. Install ML Extension and Create Workspace

```bash
az extension add --name ml
az ml workspace create \
  --name openadapt-ml \
  --resource-group openadapt-agents \
  --location eastus
```

### 6. Configure .env

Add to your `.env` file:

```bash
AZURE_CLIENT_ID=<clientId from step 3>
AZURE_CLIENT_SECRET=<clientSecret from step 3>
AZURE_TENANT_ID=<tenantId from step 3>
AZURE_SUBSCRIPTION_ID=<your subscription id>
AZURE_ML_RESOURCE_GROUP=openadapt-agents
AZURE_ML_WORKSPACE_NAME=openadapt-ml
```

## Cost Management

### Estimate Before Running

```bash
python -m openadapt_ml.benchmarks.cli estimate --workers 40
```

### Monitor Usage

Check your Azure spending:
- https://portal.azure.com/#view/Microsoft_Azure_CostManagement/Menu/~/costanalysis

### Set Budget Alerts

1. Go to Cost Management in Azure Portal
2. Click "Budgets" → "Add"
3. Set monthly budget (e.g., $50)
4. Add alert at 80% threshold

## Troubleshooting

### "Token expired" Error

```bash
az login
# Or clear and re-login:
az account clear
az login
```

### "Subscription not found"

Make sure you logged in with the correct account:
```bash
az account list --output table
az account set --subscription "Your Subscription Name"
```

### "Quota exceeded"

Request quota increase (see above) or use fewer workers.

### "Resource group already exists"

This is fine - the script will reuse existing resources.

## Cleanup

To delete all created resources, use the same setup script with `--cleanup`:

```bash
# Interactive cleanup (will prompt for confirmation)
python scripts/setup_azure.py --cleanup

# Non-interactive cleanup
python scripts/setup_azure.py --cleanup --yes

# Cleanup with custom resource names (if you used custom names during setup)
python scripts/setup_azure.py --cleanup \
  --resource-group my-agents \
  --sp-name my-waa-sp
```

This will:
1. Delete the resource group (and all resources inside it)
2. Delete the service principal
3. Remove Azure credentials from `.env`

### Manual Cleanup (Alternative)

If you prefer manual control:

```bash
# Delete resource group (includes workspace and all resources)
az group delete --name openadapt-agents --yes

# Delete service principal
az ad sp delete --id <client-id>
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Azure ML Workspace                    │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Worker 0   │  │  Worker 1   │  │  Worker N   │     │
│  │ (D8_v3 VM)  │  │ (D8_v3 VM)  │  │ (D8_v3 VM)  │ ... │
│  │             │  │             │  │             │     │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │     │
│  │ │ Windows │ │  │ │ Windows │ │  │ │ Windows │ │     │
│  │ │   VM    │ │  │ │   VM    │ │  │ │   VM    │ │     │
│  │ │ (WAA)   │ │  │ │ (WAA)   │ │  │ │ (WAA)   │ │     │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                          │
│  Tasks distributed:  [1-4]         [5-8]       [N-154]  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Blob Storage│
                    │  (Results)  │
                    └─────────────┘
```

## References

- [Windows Agent Arena](https://github.com/microsoft/WindowsAgentArena)
- [Azure ML Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [Azure Free Tier](https://azure.microsoft.com/en-us/pricing/free-services)
- [Azure Quotas](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/azure-subscription-service-limits)
