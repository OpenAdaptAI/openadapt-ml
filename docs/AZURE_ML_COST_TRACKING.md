# Azure ML Cost Tracking and Teardown

## Overview

This document describes how to track costs and perform full teardown of Azure ML resources used for WAA benchmark evaluation.

## Cost Components

### 1. Compute Instances

Azure ML Compute Instances are the primary cost driver.

| VM Size | vCPU | Memory | Price/hour |
|---------|------|--------|------------|
| Standard_D4_v3 | 4 | 16 GB | $0.19/hr |
| Standard_D8_v3 | 8 | 32 GB | $0.38/hr |
| Standard_D4s_v3 | 4 | 16 GB | $0.19/hr |
| Standard_D8s_v3 | 8 | 32 GB | $0.38/hr |
| Standard_D4ds_v5 | 4 | 16 GB | $0.42/hr |
| Standard_D8ds_v5 | 8 | 32 GB | $0.38/hr |

**Note**: Compute instances continue billing while running, even if idle. Always delete after use.

### 2. Blob Storage

Used for the golden Windows image and benchmark results.

| Component | Size | Price |
|-----------|------|-------|
| Golden image (data.img) | ~25 GB | ~$0.45/month |
| OVMF firmware | ~8 MB | negligible |
| Benchmark results | Variable | ~$0.018/GB/month |

**Blob Container**: `azureml-blobstore-84e6d3a8-2f98-4ff9-9ee1-b2f84ebc0d90`

### 3. File Share

Startup scripts and code mounted to compute instances.

| Component | Size | Price |
|-----------|------|-------|
| Startup script | ~1 KB | negligible |
| Mounted code | Variable | ~$0.06/GB/month |

**File Share**: `code-391ff5ac-6576-460f-ba4d-7e03433c68b6`

### 4. API Calls (External)

OpenAI/Anthropic API costs depend on model and task count.

| Model | Input | Output | Est. per task |
|-------|-------|--------|---------------|
| GPT-4o | $2.50/1M | $10/1M | ~$0.05 |
| GPT-4o-mini | $0.15/1M | $0.60/1M | ~$0.003 |
| Claude Sonnet 4.5 | $3/1M | $15/1M | ~$0.06 |

## CLI Commands

### Cost Summary

Show current Azure ML resource costs and usage.

```bash
# Show cost summary
uv run python -m openadapt_ml.benchmarks.cli run-azure-ml --cost-summary
```

Output:
```
=== Azure ML Cost Summary ===

Compute Instances:
  w0Expe02021336  STANDARD_D4_V3  Running   2.5 hrs  $0.48
  w0Expe02020148  STANDARD_D4_V3  Failed    -        -

  Total compute: 2.5 hrs, ~$0.48

Blob Storage:
  storage/data.img       24.8 GB
  storage/OVMF_CODE...   3.5 MB
  storage/OVMF_VARS...   3.5 MB

  Total storage: 24.81 GB, ~$0.45/month

Experiments:
  openadapt-ml: 3 runs (last 7 days)

Estimated Monthly Cost: ~$0.50 (storage only, compute billed hourly)
```

### Resource Listing

List all Azure ML resources that can be torn down.

```bash
# List all resources
uv run python -m openadapt_ml.benchmarks.cli run-azure-ml --list-resources
```

Output:
```
=== Azure ML Resources ===

Compute Instances (2):
  - w0Expe02021336  STANDARD_D4_V3  Running
  - w0Expe02020148  STANDARD_D4_V3  CreateFailed

Blob Storage (storage/ prefix):
  - data.img (24.8 GB)
  - OVMF_CODE_4M.ms.fd (3.5 MB)
  - OVMF_VARS_4M.ms.fd (3.5 MB)

File Share (Users/openadapt/):
  - compute-instance-startup.sh (1 KB)

Azure ML Jobs (last 7 days): 3
```

### Full Teardown

Delete all Azure ML resources to stop all costs. This is the recommended way to clean up after benchmarking.

```bash
# Teardown (dry run - shows what would be deleted)
uv run python -m openadapt_ml.benchmarks.cli run-azure-ml --teardown

# Teardown with confirmation
uv run python -m openadapt_ml.benchmarks.cli run-azure-ml --teardown --confirm
```

**What gets deleted:**
- All compute instances (stops compute billing)
- Golden image in blob storage (optional, use --keep-image to preserve)
- Benchmark result files in blob storage
- Startup script in file share

**What is NOT deleted:**
- Azure ML Workspace (shared resource)
- Resource Group (shared resource)
- Storage Account (just the data inside)
- Azure ML Experiments (metadata only, no cost)

Output:
```
=== Azure ML Teardown ===

Will delete:
  Compute Instances:
    - w0Expe02021336 (Running, ~$0.48 billed)
    - w0Expe02020148 (Failed)

  Blob Storage:
    - storage/data.img (24.8 GB)
    - storage/OVMF_CODE_4M.ms.fd
    - storage/OVMF_VARS_4M.ms.fd

  File Share:
    - Users/openadapt/compute-instance-startup.sh

Use --confirm to proceed with deletion.
Use --keep-image to preserve the golden image for future runs.
```

### Keeping the Golden Image

If you plan to run more benchmarks later, you can preserve the golden image to avoid re-uploading 30GB.

```bash
# Teardown but keep the golden image
uv run python -m openadapt_ml.benchmarks.cli run-azure-ml --teardown --confirm --keep-image
```

## Cost Tracking Implementation

### Approach 1: Azure CLI Queries (Implemented)

Uses `az ml compute list` and `az storage blob list` to enumerate resources and estimate costs based on known pricing.

**Pros:**
- No additional Azure permissions needed
- Real-time resource visibility
- Works with existing service principal

**Cons:**
- Estimates only (not actual billed amounts)
- Manual pricing table maintenance

### Approach 2: Azure Cost Management API (Future)

Uses `az consumption usage list` to get actual billed amounts.

```bash
# Get actual usage (requires Cost Management Reader role)
az consumption usage list \
  --start-date 2026-01-01 \
  --end-date 2026-02-02 \
  --query "[?contains(resourceGroup, 'openadapt')].{Resource:resourceId, Cost:cost, Date:usageEnd}"
```

**Pros:**
- Actual billed amounts
- Historical cost tracking

**Cons:**
- Requires additional Azure role assignment
- 24-48 hour data delay

## Resource Hierarchy

```
Azure Subscription: 78add6c6-c92a-4a53-b751-eb644ac77e59
└── Resource Group: openadapt-agents
    ├── Azure ML Workspace: openadapt-ml
    │   ├── Compute Instances (BILLABLE - delete after use)
    │   │   └── w0Exp* (created by run_azure.py)
    │   └── Experiments (metadata, no cost)
    │       └── openadapt-ml
    │
    └── Storage Account: openadapstoraged655a89ec
        ├── Blob Container: azureml-blobstore-*
        │   └── storage/ (golden image, ~$0.45/month)
        │
        └── File Share: code-*
            └── Users/openadapt/ (startup script, negligible)
```

## Best Practices

### During Development

1. **Use single worker**: `--workers 1` to minimize compute costs
2. **Use gpt-4o-mini**: Cheapest OpenAI model for testing
3. **Delete immediately**: Run `--teardown --confirm` after each session

### For Full Evaluation

1. **Estimate costs first**: `--cost-summary` before running
2. **Use appropriate workers**: Match to your task count and budget
3. **Keep golden image**: `--keep-image` if running multiple evaluations
4. **Monitor progress**: Check Azure portal for stuck instances

### Cost Optimization

| Scenario | Est. Cost | Recommendation |
|----------|-----------|----------------|
| Single task test | ~$0.10 | 1 worker, gpt-4o-mini |
| 10 tasks | ~$1.50 | 1 worker, gpt-4o-mini |
| Full 154 tasks | ~$20-50 | 4-8 workers, depends on model |

## Troubleshooting

### "Permission denied" on cost queries

The service principal may not have Cost Management Reader role. Use the resource listing approach instead.

### Compute instances stuck in "Creating"

Delete manually: `az ml compute delete --name <name> --workspace-name openadapt-ml --resource-group openadapt-agents --yes`

### Storage costs continue after teardown

Check if you used `--keep-image`. Golden image alone costs ~$0.45/month.

## References

- [Azure ML Pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/)
- [Azure Blob Storage Pricing](https://azure.microsoft.com/en-us/pricing/details/storage/blobs/)
- [WAA Azure Setup](./VANILLA_WAA_AZURE_AUTOMATION.md)
