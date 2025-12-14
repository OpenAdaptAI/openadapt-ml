# Cloud GPU Training

This guide covers using cloud GPUs for training OpenAdapt-ML models. Two providers are supported:

| Provider | Setup Time | Cost | Best For |
|----------|-----------|------|----------|
| Lambda Labs | ~1 minute | $0.60-2.00/hr | Quick experiments, no approval needed |
| Azure | Hours-days | $0.90-3.00/hr | Free credits ($200), enterprise use |

## Lambda Labs (Recommended)

Lambda Labs provides affordable GPU instances with no quota approval process.

### Setup

1. Create account at [cloud.lambdalabs.com](https://cloud.lambdalabs.com)
2. Get API key at [cloud.lambdalabs.com/api-keys](https://cloud.lambdalabs.com/api-keys)
3. Set environment variable:
   ```bash
   export LAMBDA_API_KEY=your_key_here
   ```

### Usage

```bash
# List available GPU instances and prices
uv run python -m openadapt_ml.cloud.lambda_labs list

# Launch an A100 instance (~$1.10/hr)
uv run python -m openadapt_ml.cloud.lambda_labs launch --type gpu_1x_a100

# Check running instances
uv run python -m openadapt_ml.cloud.lambda_labs status

# Get SSH command for your instance
uv run python -m openadapt_ml.cloud.lambda_labs ssh

# IMPORTANT: Terminate when done (you're billed by the hour!)
uv run python -m openadapt_ml.cloud.lambda_labs terminate <instance_id>
```

### Instance Types

| Type | GPU | VRAM | Price/hr | Use Case |
|------|-----|------|----------|----------|
| `gpu_1x_a10` | 1x A10 | 24GB | ~$0.60 | 2B-8B models, fine-tuning |
| `gpu_1x_a100` | 1x A100 | 40GB | ~$1.10 | Best value, larger models |
| `gpu_1x_h100` | 1x H100 | 80GB | ~$2.00 | Fastest, very large models |

### Training on Lambda Labs

Once your instance is running:

```bash
# SSH into the instance
ssh ubuntu@<instance_ip>

# Clone and set up the repo
git clone https://github.com/OpenAdaptAI/openadapt-ml.git
cd openadapt-ml
pip install uv
uv sync

# Run training
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_capture.yaml \
  --capture /path/to/capture
```

## Azure

Azure offers $200 free credits for new accounts, but requires GPU quota approval.

### Setup

```bash
# Full automated setup (includes GPU quota request)
python scripts/setup_azure.py

# Check GPU quota status (may take hours to approve)
az vm list-usage --location eastus -o table | grep -i nc
```

The setup script:
1. Creates resource group and ML workspace
2. Sets up service principal
3. Creates container registry
4. **Automatically requests GPU quota** (NCv3/V100)

### GPU Quota

Azure requires quota approval for GPU VMs. Small requests (6-8 vCPUs) sometimes auto-approve within minutes. If not approved automatically:

1. Go to [Azure Quota Portal](https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade)
2. Select your subscription
3. Search for "Standard NCSv3 Family"
4. Request increase to 6+ vCPUs

### Cost Comparison

| VM Size | GPU | Price/hr | Notes |
|---------|-----|----------|-------|
| NC6s_v3 | 1x V100 | ~$0.90 | Good for training |
| NC12s_v3 | 2x V100 | ~$1.80 | Faster training |
| ND96asr_v4 | 8x A100 | ~$27.00 | Large scale only |

## Recommendations

1. **For quick experiments**: Use Lambda Labs - no approval wait, simple pricing
2. **For sustained training**: Azure with free credits if you can wait for quota
3. **For production**: Either works, Lambda is simpler, Azure has more enterprise features

## Troubleshooting

### Lambda Labs: "No regions available"
GPU capacity fluctuates. Try:
- Different instance type
- Wait and retry
- Check [status page](https://lambdalabs.com/service/gpu-cloud/status)

### Azure: Quota not approved
- Small requests (6 vCPUs) usually auto-approve
- Larger requests need business justification
- Try a different region (westus2, eastus2)
