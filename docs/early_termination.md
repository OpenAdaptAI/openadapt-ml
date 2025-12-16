# Early Termination Controls for Lambda Labs Training

This document describes the complete early termination implementation for Lambda Labs cloud training.

## Features Implemented

### 1. Auto-Termination on Low Loss
Training automatically stops and terminates the Lambda instance when loss drops below a configurable threshold.

**Configuration:**
- Flag: `--auto-stop-loss` (default: 0.5)
- Example: `uv run python -m openadapt_ml.cloud.lambda_labs monitor --auto-stop-loss 0.3`

**Behavior:**
- Monitor loop checks loss after each step
- When `loss < threshold`, triggers termination sequence:
  1. Downloads final checkpoints via rsync
  2. Updates training log with termination status
  3. Terminates Lambda instance via API
  4. Exits monitor loop

**Implementation:**
- Location: `openadapt_ml/cloud/lambda_labs.py`, lines 1751-1766
- Sets `termination_status: "auto_low_loss"`
- Sets `termination_message` with loss value and threshold

### 2. Auto-Termination on Training Completion
Training automatically stops when it reaches the final epoch and steps stop increasing.

**Detection Logic:**
- Tracks step count across polling intervals
- On final epoch, if step doesn't increase for 3 consecutive polls (step_stall_count >= 3)
- Triggers same termination sequence as low loss

**Implementation:**
- Location: `openadapt_ml/cloud/lambda_labs.py`, lines 1767-1787
- Sets `termination_status: "auto_complete"`
- Sets `termination_message` with epoch count

### 3. User-Initiated Stop via Dashboard
User can click "Stop Training" button in dashboard to gracefully stop training and terminate instance.

**Mechanism:**
- Dashboard sends POST to `/api/stop` endpoint (if server running)
- OR: Displays manual command: `touch training_output/STOP_TRAINING`
- Training loop checks for `STOP_TRAINING` file every step
- Monitor loop checks for file every polling interval

**Training Loop Check:**
- Location: `openadapt_ml/training/trainer.py`, lines 3151-3157
- Checks for file at start of each batch
- Removes file and breaks training loop if found

**Monitor Loop Check:**
- Location: `openadapt_ml/cloud/lambda_labs.py`, lines 1640-1663
- Checks file on Lambda instance before each poll
- Downloads checkpoints and terminates instance if found
- Sets `termination_status: "user_stop"`

### 4. Checkpoint Download Before Termination
All termination paths automatically download checkpoints before destroying the instance.

**Download Logic:**
- Uses `download_checkpoints_from_instance()` helper
- Rsync from remote `~/openadapt-ml/checkpoints/` to local `training_output/checkpoints/`
- Runs before terminate API call in all three termination paths

**Configuration:**
- Controlled by `--download-checkpoints` flag (default: True)
- Can disable with `--no-download-checkpoints` for testing

### 5. Dashboard Termination Status Display
Dashboard shows termination reason with appropriate styling when training stops.

**Status Types:**
- `auto_complete` - Green checkmark, "Training Complete"
- `auto_low_loss` - Green checkmark, "Auto-Stopped (Low Loss)"
- `user_stop` - Orange square, "Stopped by User"

**Implementation:**
- Location: `openadapt_ml/training/trainer.py`, `_generate_termination_status_html()`
- Reads `termination_status` and `termination_message` from TrainingState
- Renders with appropriate icon and color
- Shows detailed message below status label

## Data Flow

### TrainingState Schema Extensions
Two new fields added to track termination:

```python
@dataclass
class TrainingState:
    # ... existing fields ...
    termination_status: str = ""  # "auto_low_loss" | "auto_complete" | "user_stop" | ""
    termination_message: str = ""  # Human-readable reason
```

### JSON Serialization
Fields included in `to_dict()` method for persistence:
- Written to `training_output/training_log.json`
- Synced between Lambda instance and local machine
- Read by dashboard for display

## Testing

### Unit Tests
Run `test_termination.py` to verify:
1. Stop signal file creation/deletion
2. TrainingState termination fields
3. Dashboard HTML generation with termination status
4. JSON serialization of termination fields
5. Different termination status rendering

### Integration Test with Stub Mode
```bash
# Test locally without GPU
uv run python -m openadapt_ml.cloud.lambda_labs monitor --stub --auto-stop-loss 0.3 --open

# Expected: Training simulates, dashboard updates, completes when loss < 0.3
```

### Live Test on Lambda
```bash
# Launch instance and train with auto-stop
uv run python -m openadapt_ml.cloud.lambda_labs train \
  --capture /path/to/capture \
  --auto-stop-loss 0.4 \
  --open

# Expected:
# 1. Instance launches, training starts
# 2. Dashboard opens in browser
# 3. When loss < 0.4: downloads checkpoints, terminates instance
# 4. Dashboard shows "Auto-Stopped (Low Loss)" with green checkmark
```

## Error Handling

### Checkpoint Download Failures
- If rsync fails, prints warning but continues with termination
- Instance still terminates to avoid wasted GPU credits
- User can manually rsync later: `rsync -avz ubuntu@<ip>:~/openadapt-ml/checkpoints/ ./training_output/checkpoints/`

### Instance Termination Failures
- If Lambda API call fails, prints error
- Monitor loop exits anyway
- User must manually terminate: `uv run python -m openadapt_ml.cloud.lambda_labs terminate <id>`

### File Permission Issues
- STOP_TRAINING file created with default permissions
- Should work in all environments (local, Lambda, Docker)

## Cost Savings

With auto-termination:
- **A10 GPU** ($0.75/hr): Saves ~$0.12-0.25 per run if training completes 10-20 minutes early
- **A100 GPU** ($1.29/hr): Saves ~$0.22-0.43 per run

Over 100 training runs:
- A10: Saves $12-25
- A100: Saves $22-43

## Configuration Examples

### Conservative (wait for very low loss)
```bash
monitor --auto-stop-loss 0.1
```

### Aggressive (stop as soon as reasonable)
```bash
monitor --auto-stop-loss 0.8
```

### Disable auto-stop (manual control only)
```bash
monitor --auto-stop-loss -1  # Negative value disables
```

### Download checkpoints after every epoch
```bash
monitor --download-checkpoints  # This is default behavior
```

## Related Files

- `openadapt_ml/cloud/lambda_labs.py` - Monitor loop, termination logic
- `openadapt_ml/training/trainer.py` - Training loop, stop signal check, dashboard rendering
- `openadapt_ml/training/stub_provider.py` - Stub mode for testing
- `training_output/training_log.json` - Persisted training state with termination fields
- `training_output/dashboard.html` - Live dashboard showing termination status

## Future Enhancements

### Potential Improvements
1. **SMS/Email notifications** - Alert user when training completes
2. **Automatic model upload** - Push checkpoints to S3/HuggingFace Hub before terminating
3. **Smart threshold adjustment** - Lower auto-stop-loss as training progresses
4. **Cost-based termination** - Auto-stop when reaching budget limit
5. **Multi-instance coordination** - Terminate all instances when best model found

### Known Limitations
1. Step stall detection requires 3 polls (30 seconds default) to confirm completion
2. Checkpoint download can be slow on large models (several GB)
3. No automatic retry if termination fails
4. Dashboard must be manually refreshed to see termination status in real-time

## Troubleshooting

### Training doesn't stop when loss is low
- Check `--auto-stop-loss` value in monitor command
- Verify loss is actually below threshold in dashboard
- Check for poll errors in monitor output

### "Stop Training" button doesn't work
- Ensure dashboard HTTP server is running (use `--open` flag)
- Check browser console for fetch errors
- Fallback: Manually create file: `touch training_output/STOP_TRAINING`

### Checkpoints not downloaded
- Check SSH connectivity: `ssh ubuntu@<instance-ip>`
- Verify checkpoints exist on instance: `ssh ubuntu@<instance-ip> ls ~/openadapt-ml/checkpoints/`
- Check rsync output for permission errors
- Manually download: `rsync -avz ubuntu@<ip>:~/openadapt-ml/checkpoints/ ./`

### Dashboard shows wrong termination status
- Refresh browser (Ctrl+R or Cmd+R)
- Check `training_output/training_log.json` for `termination_status` field
- Verify monitor loop set status before terminating
