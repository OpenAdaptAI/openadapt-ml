# Live Inference During Training - Design Options

This document explores approaches to show model predictions during training, not just after completion.

## Current Flow

1. Training runs on Lambda GPU instance (or Azure GPU)
2. Checkpoints saved after each epoch
3. User manually runs comparison after training completes
4. No visibility into model behavior during training

## Problem

User wants to see model predictions during training to:
- Verify the model is learning correctly
- Catch issues early (wrong predictions, poor learning)
- Monitor qualitative progress, not just loss metrics

## Cloud Platform: Azure

We use **Azure** as the primary cloud platform due to:
- Wide enterprise adoption
- MLDSAI signed up for Azure free tier
- Existing infrastructure in `scripts/setup_azure.py` and `openadapt_ml/benchmarks/azure.py`

### Azure Services Mapping (vs AWS)

| AWS Service | Azure Equivalent | Use Case |
|-------------|------------------|----------|
| S3 | **Azure Blob Storage** | Checkpoint storage, comparison results |
| Lambda | **Azure Functions** | Serverless inference workers |
| SQS | **Azure Queue Storage** | Job queue for inference tasks |
| EC2 | **Azure VMs / ML Compute** | GPU training instances |

## Design Options

### Option A: Periodic Inference on Training Instance

**How it works:**
- After each epoch, pause training
- Run inference on N sample steps from the capture
- Save predictions to comparison_epoch{N}.html
- Resume training

**Pros:**
- Simple to implement
- Uses existing infrastructure
- No extra cost

**Cons:**
- Adds ~5-10 minutes per epoch (inference time)
- Increases total training time
- GPU memory might be tight if loading inference separately

**Implementation:**
```python
# In trainer.py, after each epoch:
if config.eval_every_epoch:
    # Save checkpoint first
    adapter.save_lora_weights(checkpoint_path)
    # Run inference on sample steps
    generate_epoch_comparison(episode, adapter, epoch)
```

### Option B: Separate Inference Instance (Azure)

**How it works:**
- Training runs on Instance A (Lambda Labs GPU or Azure ML Compute)
- Instance B (Azure ML Compute) watches for new checkpoints
- When checkpoint appears, Instance B downloads and runs inference
- Results synced back to user via Azure Blob Storage

**Pros:**
- Training unaffected
- Can run inference continuously
- Parallel execution

**Cons:**
- 2x GPU cost (~$1.50/hr total)
- Complex orchestration
- Need checkpoint sync between instances

**Implementation:**
```bash
# Training instance uploads checkpoints to Azure Blob
az storage blob upload-batch \
    --account-name openadaptml \
    --destination checkpoints \
    --source /checkpoints

# Inference instance watches for new checkpoints
uv run python -m openadapt_ml.cloud.azure inference-watcher \
    --storage-account openadaptml \
    --container checkpoints
```

### Option C: Local Inference (No GPU)

**How it works:**
- Download checkpoint after each epoch from Azure Blob Storage
- Run inference locally on CPU with 4-bit quantization
- Generate comparison viewer

**Pros:**
- No interference with training
- No extra cloud cost
- Can run anytime

**Cons:**
- Very slow on CPU (~30s per step)
- Need to download checkpoint (100-200MB)
- May not match training quality

**Implementation:**
```bash
# Download checkpoint from Azure Blob
az storage blob download-batch \
    --account-name openadaptml \
    --source checkpoints/epoch_1 \
    --destination checkpoints_local/

# Run local inference
uv run python -m openadapt_ml.cloud.lambda_labs compare-local \
    --checkpoint checkpoints_local/epoch_1 \
    --capture /path/to/capture
```

### Option D: Async Inference Queue (Azure - Recommended for Scale)

**How it works:**
1. Training instance saves checkpoints to **Azure Blob Storage**
2. Triggers **Azure Queue Storage** message with checkpoint info
3. **Azure Functions** (serverless) or separate inference worker polls queue
4. Worker downloads checkpoint, runs inference, uploads results to Blob
5. Dashboard polls Blob Storage for new comparison files

**Architecture:**
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Lambda/Azure    │     │ Azure Blob       │     │ Azure Queue     │
│ Training GPU    │────▶│ Storage          │────▶│ Storage         │
│                 │     │ (checkpoints)    │     │ (job messages)  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Local Dashboard │◀────│ Azure Blob       │◀────│ Azure Functions │
│ (browser)       │     │ (comparisons)    │     │ or GPU Worker   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**Pros:**
- Fully decoupled - training never waits
- Scalable - can add more inference workers
- Cost-efficient - serverless for light workloads
- Works with existing Azure setup (`setup_azure.py`)

**Cons:**
- More complex initial setup
- Requires Azure Blob Storage and Queue Storage
- Need to handle authentication across services

**Implementation:**
```python
# openadapt_ml/cloud/azure_inference.py

from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueClient

class AzureInferenceQueue:
    """Manages async inference jobs via Azure Queue Storage."""

    def __init__(self, storage_account: str, queue_name: str = "inference-jobs"):
        self.blob_client = BlobServiceClient.from_connection_string(conn_str)
        self.queue_client = QueueClient.from_connection_string(conn_str, queue_name)

    def submit_checkpoint(self, checkpoint_path: str, capture_path: str):
        """Upload checkpoint and queue inference job."""
        # Upload checkpoint to blob
        blob_name = f"checkpoints/{Path(checkpoint_path).name}"
        self.blob_client.upload_blob(blob_name, open(checkpoint_path, 'rb'))

        # Queue inference job
        job = {"checkpoint": blob_name, "capture": capture_path}
        self.queue_client.send_message(json.dumps(job))

    def poll_and_process(self, adapter, output_container: str = "comparisons"):
        """Worker: poll queue and run inference."""
        while True:
            messages = self.queue_client.receive_messages(max_messages=1)
            for msg in messages:
                job = json.loads(msg.content)
                # Download checkpoint, run inference, upload result
                self._process_job(job, adapter, output_container)
                self.queue_client.delete_message(msg)
```

### Option E: Stream Inference During Training

**How it works:**
- Run inference after every N steps (e.g., every 5 steps)
- Only infer on current step, not full capture
- Overlay prediction on dashboard in real-time

**Pros:**
- Immediate feedback
- Minimal overhead per inference
- Shows learning progression

**Cons:**
- Slows training slightly
- Complex implementation
- May not represent full episode performance

## Recommended Approach

### Phase 1: Option A (Periodic Inference) - Immediate
- Simple to implement
- Provides epoch-by-epoch comparison
- Acceptable training overhead
- **Status: Can implement now**

### Phase 2: Option D (Azure Async Queue) - Next Priority
- Build on existing Azure infrastructure (`setup_azure.py`)
- Fully decoupled from training
- Scales to longer training runs
- **Leverages:**
  - Azure Blob Storage (already used for WAA benchmarks)
  - Azure Queue Storage (simple message queue)
  - Existing `AzureConfig` and authentication from `openadapt_ml/benchmarks/azure.py`

### Phase 3: Option C (Local Inference)
- Allow users with patience to run local inference
- No cloud cost impact
- Works offline

## Implementation Plan

### Phase 1: Add eval_every_epoch to trainer

1. Add `eval_every_epoch` config option (already exists)
2. After saving epoch checkpoint, run inference on sample steps
3. Generate comparison_epoch{N}.html
4. Dashboard auto-discovers new comparison files

### Phase 2: Azure Async Inference

#### Step 1: Extend setup_azure.py
Add storage account and queue creation:
```python
# In scripts/setup_azure.py
def create_storage_account(name: str, resource_group: str) -> str:
    """Create Azure Storage account for checkpoints and comparisons."""
    run_cmd([
        "az", "storage", "account", "create",
        "--name", name,
        "--resource-group", resource_group,
        "--sku", "Standard_LRS",
    ])
    return run_cmd([
        "az", "storage", "account", "show-connection-string",
        "--name", name,
        "--query", "connectionString",
        "-o", "tsv",
    ])

def create_queue(connection_string: str, queue_name: str = "inference-jobs"):
    """Create Azure Queue for inference jobs."""
    run_cmd([
        "az", "storage", "queue", "create",
        "--name", queue_name,
        "--connection-string", connection_string,
    ])
```

#### Step 2: Create azure_inference.py module
New module at `openadapt_ml/cloud/azure_inference.py`:
- `AzureInferenceQueue` class for managing jobs
- `submit_checkpoint()` - called from trainer after each epoch
- `poll_and_process()` - worker loop for inference instance

#### Step 3: Add CLI commands
```bash
# Submit checkpoint for async inference (called by trainer)
uv run python -m openadapt_ml.cloud.azure inference-submit \
    --checkpoint checkpoints/epoch_1

# Start inference worker (runs on separate instance)
uv run python -m openadapt_ml.cloud.azure inference-worker \
    --model Qwen/Qwen2.5-VL-3B

# Watch for new comparison results
uv run python -m openadapt_ml.cloud.azure inference-watch
```

#### Step 4: Integrate with dashboard
- Dashboard polls Azure Blob for new comparison_epoch_N.html files
- Auto-discovers and displays new comparisons
- Shows inference status in real-time

### Phase 3: Add local inference CLI

1. `compare-local` command that uses CPU inference
2. Downloads checkpoint if needed
3. Runs inference with progress bar
4. Generates comparison viewer

## Questions to Answer

1. How many sample steps to evaluate per epoch? (Current: 3)
2. Should we infer on random steps or fixed steps?
3. How to handle GPU memory for inference during training?
4. Should local inference use GGUF/quantized models for speed?
5. Azure Function vs. dedicated GPU instance for inference workers?

## Cost Estimates (Azure)

| Component | Cost | Notes |
|-----------|------|-------|
| Azure Blob Storage | ~$0.02/GB/month | Checkpoint storage |
| Azure Queue Storage | ~$0.0004/10k messages | Job queue |
| Azure Functions | Free tier: 1M requests/month | Serverless option |
| Azure ML Compute (GPU) | ~$0.90/hr (NC6s_v3) | Inference worker |

For typical training (5 epochs, 10 sample steps):
- Blob: ~1GB = $0.02/month
- Queue: ~50 messages = negligible
- Functions: Well within free tier
- **Total overhead: < $0.10/training run**

## Environment Variables

Add to `.env` (via `setup_azure.py`):
```bash
# Azure Storage for inference queue
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_INFERENCE_QUEUE_NAME=inference-jobs
AZURE_CHECKPOINTS_CONTAINER=checkpoints
AZURE_COMPARISONS_CONTAINER=comparisons
```
