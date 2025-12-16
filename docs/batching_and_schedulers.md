# Batching and Learning Rate Schedulers Guide

This guide explains how to use the new batching and learning rate scheduler features added in Priority 1.

## Table of Contents
- [Batching](#batching)
- [Learning Rate Schedulers](#learning-rate-schedulers)
- [Run Directory Logging](#run-directory-logging)
- [Configuration Examples](#configuration-examples)
- [Best Practices](#best-practices)

## Batching

### Overview
Previously, the Qwen-VL adapter only supported `batch_size=1`, which limited GPU throughput. The new implementation supports arbitrary batch sizes.

### How It Works
1. Multiple samples are processed simultaneously in `QwenVLAdapter.prepare_inputs()`
2. The processor applies padding and truncation to create uniform-length sequences
3. Per-sample label masking ensures only assistant tokens are supervised
4. Padding tokens are masked with `-100` to exclude from loss computation

### Configuration
Set `per_device_train_batch_size` in your training config:

```yaml
training:
  per_device_train_batch_size: 4  # Batch 4 samples together
  gradient_accumulation_steps: 1
```

### Memory Considerations
Larger batch sizes require more GPU memory:
- `batch_size=1`: ~6GB VRAM (Qwen3-VL-2B with 4-bit quantization)
- `batch_size=2`: ~8GB VRAM
- `batch_size=4`: ~12GB VRAM
- `batch_size=8`: ~20GB VRAM

Use gradient accumulation to simulate larger batches if memory is limited:

```yaml
training:
  per_device_train_batch_size: 2  # Actual batch size
  gradient_accumulation_steps: 4  # Effective batch size = 2 * 4 = 8
```

### Performance Impact
- **Training speed**: 2-4x faster with `batch_size=4` vs `batch_size=1`
- **Loss stability**: Improved with larger batches (smoother gradient estimates)
- **Convergence**: Often faster with larger batches

## Learning Rate Schedulers

### Overview
Learning rate schedulers adjust the learning rate during training to improve convergence and final performance.

### Supported Schedulers

#### 1. Linear (Default)
Linear warmup followed by linear decay to zero.

```yaml
training:
  lr_scheduler_type: linear
  learning_rate: 1.0e-4
  warmup_ratio: 0.1  # 10% of steps for warmup
```

**When to use:**
- General-purpose default
- Works well for most tasks
- Recommended for fine-tuning

#### 2. Cosine
Linear warmup followed by cosine decay.

```yaml
training:
  lr_scheduler_type: cosine
  learning_rate: 2.0e-4
  warmup_ratio: 0.05
```

**When to use:**
- Longer training runs
- When you want smoother learning rate changes
- Often better for training from scratch

#### 3. Constant (No Scheduling)
Learning rate stays constant throughout training.

```yaml
training:
  lr_scheduler_type: constant
  learning_rate: 5.0e-5
```

**When to use:**
- Very short training runs
- When warmup/decay causes instability
- Debugging purposes

#### 4. None (Disable Scheduler)
Equivalent to `constant`.

```yaml
training:
  lr_scheduler_type: none
```

### Warmup Ratio
The `warmup_ratio` controls what fraction of training steps are used for warmup:

- `warmup_ratio: 0.0` → No warmup
- `warmup_ratio: 0.05` → Warmup for 5% of steps
- `warmup_ratio: 0.1` → Warmup for 10% of steps
- `warmup_ratio: 0.2` → Warmup for 20% of steps

**Recommended values:**
- Short training (1-2 epochs): `0.05` - `0.1`
- Medium training (3-5 epochs): `0.1` - `0.15`
- Long training (10+ epochs): `0.15` - `0.2`

### Learning Rate Logging
The current learning rate is logged at each step in `training_log.json`:

```json
{
  "epoch": 0,
  "step": 100,
  "loss": 0.234,
  "lr": 0.000095,  // Current learning rate
  "time": 45.2
}
```

You can visualize the learning rate schedule in the training dashboard.

## Run Directory Logging

### Directory Structure
Each training run creates a self-contained directory:

```
training_output/
├── 20251215_143022/           # Timestamp-based job ID
│   ├── config.json            # Training config snapshot
│   ├── training_log.json      # Step-wise metrics
│   ├── training.log           # Terminal output with timestamps
│   ├── dashboard.html         # Training visualization
│   └── checkpoints/
│       ├── epoch_0/
│       ├── epoch_1/
│       └── ...
└── current -> 20251215_143022/  # Symlink to latest run
```

### Config Snapshot
The `config.json` file contains all training configuration:

```json
{
  "num_train_epochs": 5,
  "per_device_train_batch_size": 4,
  "learning_rate": 0.0001,
  "lr_scheduler_type": "linear",
  "warmup_ratio": 0.1,
  ...
}
```

This enables:
- **Reproducibility**: Exact config used for each run
- **Debugging**: Compare configs across runs
- **Experiment tracking**: Know what settings produced what results

### Accessing Logs
```bash
# View latest config
cat training_output/current/config.json

# View training log
cat training_output/current/training_log.json

# View terminal output
cat training_output/current/training.log

# Open dashboard
open training_output/current/dashboard.html
```

## Configuration Examples

### Example 1: Fast Development Training
Small batch, linear scheduler, minimal epochs:

```yaml
model:
  name: Qwen/Qwen3-VL-2B-Instruct
  load_in_4bit: true
  max_pixels: 262144  # 512x512 for speed

lora:
  r: 8
  lora_alpha: 16
  target_modules: [q_proj, v_proj]

training:
  num_train_epochs: 2
  per_device_train_batch_size: 2
  learning_rate: 1.0e-4
  warmup_ratio: 0.05
  lr_scheduler_type: linear
  logging_steps: 1
```

### Example 2: Production Fine-Tuning
Larger batch, cosine scheduler, more epochs:

```yaml
model:
  name: Qwen/Qwen3-VL-2B-Instruct
  load_in_4bit: true
  max_pixels: 262144

lora:
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj]

training:
  num_train_epochs: 5
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2  # Effective batch = 8
  learning_rate: 5.0e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  lr_scheduler_type: cosine
  logging_steps: 10
  early_stop_loss: 0.01
  early_stop_patience: 20
```

### Example 3: Memory-Constrained Setup
Small batch with gradient accumulation:

```yaml
training:
  num_train_epochs: 5
  per_device_train_batch_size: 1  # Low memory
  gradient_accumulation_steps: 8  # Simulate batch=8
  learning_rate: 5.0e-5
  warmup_ratio: 0.1
  lr_scheduler_type: linear
```

### Example 4: Debugging (No Scheduling)
Constant learning rate for stability:

```yaml
training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  learning_rate: 1.0e-4
  lr_scheduler_type: none  # Constant LR
  logging_steps: 1
```

## Best Practices

### Batching
1. **Start small**: Try `batch_size=2` first, then increase if memory allows
2. **Monitor memory**: Use `nvidia-smi` or similar to track VRAM usage
3. **Use gradient accumulation**: Simulate larger batches if memory is limited
4. **Profile performance**: Measure training speed with different batch sizes

### Learning Rate Schedulers
1. **Linear for fine-tuning**: Good default for transfer learning
2. **Cosine for longer runs**: Better for training from scratch or many epochs
3. **Warmup is important**: Use 5-10% warmup ratio to avoid instability
4. **Tune learning rate**: Scheduler doesn't fix a bad base learning rate
5. **Monitor the curve**: Check that LR decreases smoothly in logs

### General
1. **Save configs**: Config snapshots enable reproducibility
2. **Track experiments**: Use different `output_dir` for each experiment
3. **Compare runs**: Use `training_log.json` to compare metrics across runs
4. **Start with defaults**: Only tune if you see issues

### Hyperparameter Tuning Order
1. **Learning rate** (most important)
2. **Batch size** (for speed and memory)
3. **Warmup ratio** (affects early training stability)
4. **Scheduler type** (linear vs cosine)
5. **Weight decay** (regularization)

## Troubleshooting

### Loss is NaN or Inf
- **Reduce learning rate**: Try 10x smaller (e.g., `1e-5` instead of `1e-4`)
- **Increase warmup**: Try `warmup_ratio: 0.2`
- **Check batch size**: Very small batches can be unstable

### Loss not decreasing
- **Increase learning rate**: Try 2-3x larger
- **Check scheduler**: Make sure `lr_scheduler_type` is not `none`
- **Verify data**: Ensure labels are correct

### Out of memory
- **Reduce batch size**: Try `per_device_train_batch_size: 1`
- **Use gradient accumulation**: Set `gradient_accumulation_steps: 4`
- **Reduce image resolution**: Lower `max_pixels`
- **Enable 4-bit quantization**: `load_in_4bit: true`

### Training too slow
- **Increase batch size**: Try doubling until OOM
- **Reduce image resolution**: Smaller images train faster
- **Use fewer LoRA layers**: Reduce `target_modules`

## Command Line Usage

```bash
# Basic training with new config
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_capture_batched.yaml \
  --capture /path/to/capture

# Check config snapshot
cat training_output/current/config.json | jq .

# Monitor training log
tail -f training_output/current/training.log

# Open dashboard
uv run python -m openadapt_ml.cloud.local serve --open
```

## Migration Guide

If you have existing configs, add these fields:

```yaml
training:
  # Add this line:
  lr_scheduler_type: linear  # or cosine, constant, none

  # Optionally increase batch size:
  per_device_train_batch_size: 4  # was 1
```

All other fields remain compatible.
