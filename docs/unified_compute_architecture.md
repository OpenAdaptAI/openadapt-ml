# Unified Compute Architecture

## Problem Statement

OpenAdapt-ML needs to run training and inference seamlessly across:
- **Local** (CUDA, Apple Silicon MPS, CPU)
- **Lambda Labs** (GPU rental, current implementation)
- **Azure** (ML workspace, WAA benchmarks)
- **AWS EC2** (SoM deployment pattern from microsoft/SoM)

Currently we have fragmented CLIs:
- `openadapt_ml.cloud.local` - Local training
- `openadapt_ml.cloud.lambda_labs` - Lambda Labs cloud
- `openadapt_ml.benchmarks.azure` - Azure WAA (blocked)

## Vision

A single unified interface that abstracts compute backend:

```bash
# Train - same command, different backends
uv run python -m openadapt_ml train --capture ~/captures/workflow --backend local
uv run python -m openadapt_ml train --capture ~/captures/workflow --backend lambda
uv run python -m openadapt_ml train --capture ~/captures/workflow --backend azure

# Inference/Benchmark - same pattern
uv run python -m openadapt_ml benchmark --suite waa --backend azure
uv run python -m openadapt_ml benchmark --suite synthetic --backend local

# Status - unified view
uv run python -m openadapt_ml status  # Shows all active jobs across backends
```

## Current State

### Local (`cloud/local.py`)
- ✅ `status` - Check device and training status
- ✅ `train` - Run training locally
- ✅ `check` - Training health analysis
- ✅ `serve` - Dashboard web server
- ✅ `viewer` - Regenerate viewer
- ✅ `compare` - Human vs AI comparison

### Lambda Labs (`cloud/lambda_labs.py`)
- ✅ `list` - Available instance types
- ✅ `launch` - Start GPU instance
- ✅ `train` - Full training pipeline
- ✅ `train-status` - Check remote training
- ✅ `check` - Training health
- ✅ `download` - Get results
- ✅ `terminate` - Stop instance

### Azure (`benchmarks/azure.py`)
- ⚠️ `AzureWAAOrchestrator` - Blocked by ACR auth issue
- 🔲 Training on Azure ML not implemented

### AWS EC2 (from microsoft/SoM `deploy.py`)
- Pattern: GitHub Actions triggered deployment
- Creates EC2 instance with GPU
- Builds Docker container on instance
- Runs Gradio server for inference
- Commands: `start`, `pause`, `stop`, `status`

## Proposed Architecture

### 1. Backend Abstraction Layer

```python
# openadapt_ml/compute/base.py
class ComputeBackend(ABC):
    """Abstract base class for compute backends."""

    @abstractmethod
    def launch(self, config: JobConfig) -> str:
        """Launch compute resources. Returns job_id."""
        pass

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Get job status."""
        pass

    @abstractmethod
    def run(self, job_id: str, command: str) -> None:
        """Run command on compute."""
        pass

    @abstractmethod
    def download(self, job_id: str, remote_path: str, local_path: str) -> None:
        """Download files from compute."""
        pass

    @abstractmethod
    def terminate(self, job_id: str) -> None:
        """Terminate compute resources."""
        pass
```

### 2. Backend Implementations

```
openadapt_ml/compute/
├── base.py           # ComputeBackend ABC
├── local.py          # LocalBackend (current local.py)
├── lambda_labs.py    # LambdaBackend (current lambda_labs.py)
├── azure.py          # AzureBackend (Azure ML)
├── aws_ec2.py        # AWSEC2Backend (from SoM deploy.py pattern)
└── __init__.py       # Backend registry
```

### 3. Job Types

```python
class JobType(Enum):
    TRAIN = "train"           # Fine-tuning
    INFERENCE = "inference"   # Single prediction
    BENCHMARK = "benchmark"   # Benchmark suite
    SERVE = "serve"           # Inference server (Gradio/API)
```

### 4. Unified CLI

```python
# openadapt_ml/__main__.py
@cli.command()
def train(
    capture: str,
    backend: str = "local",  # local, lambda, azure, aws
    config: str = None,      # Auto-select based on backend
    goal: str = None,
):
    """Train on any backend."""
    backend_impl = get_backend(backend)
    job_id = backend_impl.launch(TrainJobConfig(capture=capture, ...))
    # ... monitor and report
```

## Integration Points

### SoM (Set-of-Marks)

SoM is critical for achieving high accuracy. Integration options:

1. **Local SoM** - Run SoM model locally for element detection
2. **SoM Server** - Deploy SoM to cloud (AWS EC2 pattern from microsoft/SoM)
3. **API SoM** - Use hosted SoM service

For real captures to work with SoM:
1. Run element detection on screenshots
2. Generate numbered overlays
3. Train/infer with `CLICK([N])` instead of coordinates

### WAA Benchmarks

WAA requires Windows VMs. Options:
- **Azure** - Native Windows VMs (current approach, blocked by ACR)
- **AWS** - Windows AMIs
- **Local** - Windows VM via Parallels/VMware (limited)

### Grounding Module

`openadapt_ml/grounding/` provides element location:
- `GeminiGrounder` - Google Gemini API
- `OracleGrounder` - Ground truth (for eval)
- Could add `SoMGrounder` - SoM-based detection

## Migration Path

### Phase 1: Consolidate (Current)
- [x] `cloud/local.py` - Local training CLI
- [x] `cloud/lambda_labs.py` - Lambda Labs CLI
- [ ] Unify common patterns

### Phase 2: Abstract
- [ ] Create `ComputeBackend` ABC
- [ ] Refactor local.py to `LocalBackend`
- [ ] Refactor lambda_labs.py to `LambdaBackend`

### Phase 3: Extend
- [ ] Add `AzureBackend` (fix ACR issue first)
- [ ] Add `AWSEC2Backend` (port SoM deploy.py pattern)
- [ ] Unified CLI entry point

### Phase 4: SoM Integration
- [ ] SoM element detection pipeline
- [ ] SoM overlay generation for real captures
- [ ] SoM-based training on real data

## Priority Evaluation

Given current state and goals:

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **1** | Fix model output format (training issue) | Medium | High - demos work |
| **2** | SoM on synthetic (already 100%) | Low | High - proven path |
| **3** | SoM element detection for real captures | High | High - real data works |
| **4** | Unified compute abstraction | Medium | Medium - cleaner code |
| **5** | WAA benchmarks (fix Azure ACR) | Medium | High - publishable results |
| **6** | PyPI publishing | Low | Medium - distribution |

## Recommendation

**Immediate focus**: Get a working demo by either:
1. Using synthetic SoM (100% accuracy, already works)
2. Fixing the training/inference prompt mismatch for real captures

**Next**: WAA benchmarks on Azure (need to fix ACR auth)

**Then**: Unified compute architecture for maintainability

## Related Files

- `openadapt_ml/cloud/local.py` - Local training CLI
- `openadapt_ml/cloud/lambda_labs.py` - Lambda Labs CLI
- `openadapt_ml/benchmarks/azure.py` - Azure WAA orchestration
- `microsoft/SoM/deploy.py` - AWS EC2 deployment pattern
- `docs/cloud_gpu_training.md` - Lambda Labs documentation
- `docs/azure_waa_setup.md` - Azure setup guide

## References

- [microsoft/SoM deploy.py](https://github.com/microsoft/SoM/blob/main/deploy.py) - AWS EC2 deployment pattern
- [Lambda Labs API](https://cloud.lambdalabs.com/api/v1/docs) - GPU rental
- [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/) - Azure ML workspace
