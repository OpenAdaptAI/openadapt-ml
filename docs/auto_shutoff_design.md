# Auto-Shutoff Design

## Problem Statement

Cloud GPU training costs $0.75-$3.29/hr. Training should automatically stop when:
1. Model has learned sufficiently (loss plateau)
2. Maximum budget/time is reached
3. Training is diverging (loss exploding)

Without auto-shutoff, training can run indefinitely wasting cloud credits.

## Current Implementation

### 1. Config-Based Early Stopping

Located in `configs/qwen3vl_capture*.yaml`:

```yaml
training:
  # INVARIANT: Training stops when loss <= 1.0
  early_stop_loss: 1.0
  early_stop_patience: 5  # consecutive steps below threshold
```

**Where enforced**: `openadapt_ml/training/trainer.py` in training loop.

### 2. Dashboard Auto-Stop (NEW)

Located in `trainer.py` dashboard JavaScript:

```javascript
const AUTO_STOP_LOSS_THRESHOLD = 1.0;

// When loss <= threshold, automatically call /api/stop
if (!autoStopTriggered && !isTrainingComplete && data.loss <= AUTO_STOP_LOSS_THRESHOLD) {
    fetch('/api/stop', { method: 'POST' });
}
```

**Why both?** Redundancy - if the training loop doesn't catch it, the dashboard will.

## Design Principles

### 1. Defense in Depth
Multiple layers check for stop conditions:
- Training loop (primary)
- Dashboard monitor (secondary)
- Max runtime limit (failsafe)

### 2. Fail-Safe Defaults
- `early_stop_loss: 1.0` - Conservative threshold that catches most convergence
- `max_runtime: 60` minutes - Prevents runaway training
- Instance auto-terminate on training completion

### 3. Observable
- Dashboard shows current loss vs threshold
- Notification when auto-stop triggers
- Terminal logs stop reason

## Stop Conditions

| Condition | Threshold | Where Checked | Priority |
|-----------|-----------|---------------|----------|
| Loss convergence | loss <= 1.0 | Training loop, Dashboard | Primary |
| Max runtime | 60 minutes | Lambda CLI | Failsafe |
| User stop | Button click | Dashboard /api/stop | Manual |
| STOP_TRAINING file | File exists | Training loop | Remote trigger |

## Future Enhancements

### Phase 1: Configurable Thresholds (TODO)
Add UI controls in dashboard:
```html
<input type="number" id="loss-threshold" value="1.0" />
<button onclick="updateThreshold()">Update</button>
```

Store in `training_config.json` alongside `training_log.json`.

### Phase 2: Cost-Based Stopping (TODO)
Stop when estimated cost exceeds budget:
```javascript
const MAX_COST = 5.00;  // $5 budget
if (currentCost >= MAX_COST) triggerStop('budget_exceeded');
```

### Phase 3: Divergence Detection (TODO)
Stop if loss is increasing consistently:
```javascript
const recentLosses = data.losses.slice(-10);
const trend = calculateTrend(recentLosses);
if (trend > 0.1) triggerStop('diverging');  // Loss increasing
```

### Phase 4: Smart Convergence (TODO)
Use statistical methods to detect true convergence:
- Moving average plateau detection
- Gradient of loss curve approaching zero
- Validation loss not improving

## Implementation Checklist

- [x] Config-based early_stop_loss
- [x] Dashboard auto-stop when loss <= threshold
- [x] Stop notification in UI
- [x] All capture configs updated to loss <= 1.0
- [ ] Configurable threshold in UI
- [ ] Cost-based stopping
- [ ] Divergence detection
- [ ] Cumulative cost tracking across runs
- [ ] SQLite persistence for training history

## Testing

To verify auto-stop works:

```bash
# Run stub training (fast, no GPU)
uv run python -m openadapt_ml.cloud.local serve --port 8080 --stub --open

# Watch dashboard - should auto-stop when loss drops below 1.0
```

## Related Files

- `configs/qwen3vl_capture.yaml` - Early stop config
- `openadapt_ml/training/trainer.py` - Dashboard with auto-stop JS
- `openadapt_ml/training/stub_provider.py` - Early stop logic
- `openadapt_ml/cloud/lambda_labs.py` - Instance termination
