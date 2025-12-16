# Azure ACR Authentication for ML Workspace

## Overview

Azure ML compute instances need proper authentication to pull Docker images from Azure Container Registry (ACR). This document explains the authentication mechanism and how it's configured.

## The Problem

When you attach an ACR to an Azure ML workspace, the workspace still cannot pull images unless:

1. The ACR is attached to the workspace
2. The workspace's managed identity has `AcrPull` role on the ACR
3. The workspace keys are synced to propagate the credentials

Simply attaching the ACR (step 1) is not sufficient - the managed identity permissions (step 2) are critical.

## Authentication Flow

```
Azure ML Compute Instance
    ↓
Uses Workspace Managed Identity
    ↓
Needs AcrPull role on ACR
    ↓
Can pull Docker images
```

## Automatic Configuration (New Installations)

The `scripts/setup_azure.py` script now automatically configures ACR authentication:

```bash
python scripts/setup_azure.py
```

This performs:
- **Step 10**: Attach ACR to workspace
- **Step 11**: Grant AcrPull role to workspace managed identity
- **Step 12**: Sync workspace keys

## Manual Fix (Existing Installations)

If you ran `setup_azure.py` before these steps were added, use the fix script:

```bash
python scripts/fix_acr_auth.py
```

Or manually:

```bash
# Get workspace managed identity principal ID
PRINCIPAL_ID=$(az ml workspace show \
  -n openadapt-ml \
  -g openadapt-agents \
  --query identity.principal_id \
  -o tsv)

# Get ACR resource ID
ACR_ID="/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/openadapt-agents/providers/Microsoft.ContainerRegistry/registries/openadaptacr"

# Grant AcrPull role
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role AcrPull \
  --scope $ACR_ID

# Sync workspace keys
az ml workspace sync-keys \
  -n openadapt-ml \
  -g openadapt-agents
```

## Verification

Check if the role assignment exists:

```bash
# Get principal ID
PRINCIPAL_ID=$(az ml workspace show -n openadapt-ml -g openadapt-agents --query identity.principal_id -o tsv)

# List role assignments for the principal
az role assignment list --assignee $PRINCIPAL_ID --all
```

You should see an entry with:
- `roleDefinitionName: AcrPull`
- `scope: /subscriptions/.../registries/openadaptacr`

## Troubleshooting

### Error: "Workspace does not have a managed identity"

The managed identity is created automatically when the workspace is first used. If you see this error:

1. Wait a few minutes after workspace creation
2. Try accessing the workspace through Azure Portal
3. Re-run the fix script

### Error: "Failed to pull Docker image"

Check these in order:

1. **ACR exists and has the image**:
   ```bash
   az acr repository list -n openadaptacr -o table
   ```

2. **ACR is attached to workspace**:
   ```bash
   az ml workspace show -n openadapt-ml -g openadapt-agents --query containerRegistry
   ```

3. **Managed identity exists**:
   ```bash
   az ml workspace show -n openadapt-ml -g openadapt-agents --query identity
   ```

4. **AcrPull role is assigned**:
   ```bash
   PRINCIPAL_ID=$(az ml workspace show -n openadapt-ml -g openadapt-agents --query identity.principal_id -o tsv)
   az role assignment list --assignee $PRINCIPAL_ID --all
   ```

### Error: "Image not found in ACR"

Import the WAA image:

```bash
az acr import \
  --name openadaptacr \
  --source docker.io/windowsarena/winarena:latest \
  --image winarena:latest
```

## Alternative: ACR Admin Credentials

If managed identity authentication doesn't work, you can enable ACR admin credentials:

```bash
# Enable admin user
az acr update -n openadaptacr --admin-enabled true

# Get credentials
az acr credential show -n openadaptacr
```

Then configure the environment to use these credentials. However, **managed identity is preferred** because:
- No credential management required
- Automatic credential rotation
- Better security (no shared secrets)
- Follows Azure best practices

## References

- [Azure ML Managed Identity Authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication)
- [ACR Authentication with Managed Identity](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication-managed-identity)
- [Azure RBAC Built-in Roles](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#acrpull)
