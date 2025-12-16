# Weights & Biases Integration Options

This document explores options for integrating [Weights & Biases (W&B)](https://wandb.ai) with the openadapt-ml viewer and training pipeline.

## Current State

Our viewer provides:
- Real-time training dashboard with loss curves
- Terminal output streaming
- Evaluation gallery with human vs. predicted action overlays
- Step-by-step capture replay with model predictions
- Cost tracking for cloud training

**Question**: Should we integrate W&B, and if so, how deeply?

---

## Integration Options

### Option 1: Minimal - Metrics Only

**What**: Log scalar metrics (loss, accuracy, learning rate) to W&B alongside our existing dashboard.

```python
import wandb

wandb.init(project="openadapt-ml", config=config)

# In training loop
wandb.log({
    "loss": loss,
    "learning_rate": lr,
    "epoch": epoch,
    "step": step,
})
```

**Pros**:
- Simple to implement (~20 lines of code)
- W&B handles metric aggregation, comparisons across runs
- Free tier is generous for individuals
- No changes to our viewer

**Cons**:
- Duplicate dashboards (ours + W&B)
- Users must create W&B account
- No image/evaluation data in W&B

**Effort**: 1-2 hours

---

### Option 2: Medium - Metrics + Evaluation Images

**What**: Log metrics plus evaluation screenshots with action overlays as W&B Tables.

```python
import wandb

# Log evaluation samples with overlays
eval_table = wandb.Table(columns=["step", "image", "human_action", "predicted_action", "correct"])

for eval in evaluations:
    # Create image with markers
    img = wandb.Image(
        eval.screenshot_path,
        boxes={
            "human": {
                "box_data": [{"position": {"middle": [hx, hy], "width": 20, "height": 20}, "class_id": 0}],
                "class_labels": {0: "human"}
            },
            "predicted": {
                "box_data": [{"position": {"middle": [px, py], "width": 20, "height": 20}, "class_id": 1}],
                "class_labels": {1: "predicted"}
            }
        }
    )
    eval_table.add_data(eval.step, img, str(eval.human_action), str(eval.predicted_action), eval.correct)

wandb.log({"evaluations": eval_table})
```

**Pros**:
- W&B's table filtering is powerful (filter by correct/incorrect, group by action type)
- Images with overlays can be toggled/compared in W&B UI
- Shareable reports for stakeholders
- Model comparison across runs

**Cons**:
- More complex integration
- Storage limits on free tier (100GB)
- Potential duplication with our viewer
- W&B Tables limited to 200k rows

**Effort**: 4-8 hours

---

### Option 3: Full - Replace Our Dashboard

**What**: Migrate entirely to W&B for visualization, deprecate our custom dashboard.

**Pros**:
- No dashboard code to maintain
- Industry-standard tooling
- Collaboration features built-in
- Sweep/hyperparameter tuning included

**Cons**:
- **Loss of our custom features**: Step-by-step replay, audio sync, transcript integration
- Vendor lock-in
- Free tier limits may be hit
- Less control over UX
- Our viewer has domain-specific features W&B lacks

**Effort**: 20-40 hours (plus feature loss)

**Recommendation**: NOT recommended - our viewer has unique value

---

### Option 4: Hybrid - W&B for Runs, Our Viewer for Details

**What**: Use W&B for run management and comparison, our viewer for detailed step-by-step analysis.

```
W&B Dashboard                          Our Viewer
─────────────────────────────          ─────────────────────────────
• All runs in one place                • Single run deep-dive
• Loss curves comparison               • Step-by-step replay
• Hyperparameter sweeps                • Audio/transcript sync
• Model registry                       • Click marker overlays
• Team collaboration                   • Capture playback
                    ↓
         Link from W&B run to our viewer
                    ↓
         wandb.log({"viewer_url": "http://..."})
```

**Implementation**:
```python
# In training
wandb.init(project="openadapt-ml", config=config)
wandb.log({
    "loss": loss,
    "viewer_url": f"http://localhost:8080/viewer.html?run={run_id}",
    "dashboard_url": f"http://localhost:8080/dashboard.html?run={run_id}",
})

# Log summary evaluation metrics
wandb.log({
    "eval/accuracy": accuracy,
    "eval/avg_distance": avg_distance,
    "eval/correct_count": correct_count,
})
```

**Pros**:
- Best of both worlds
- W&B for what it's good at (run comparison, sweeps, collaboration)
- Our viewer for what we're good at (domain-specific visualization)
- Clear separation of concerns

**Cons**:
- Two systems to understand
- URL linking requires serving our viewer

**Effort**: 4-6 hours

---

## W&B-Specific Features Worth Considering

### 1. Hyperparameter Sweeps
W&B Sweeps automate hyperparameter search:
```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [1, 2, 4]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="openadapt-ml")
wandb.agent(sweep_id, train_function)
```

### 2. Model Registry
Version and stage models:
```python
# Log model artifact
artifact = wandb.Artifact("qwen-lora", type="model")
artifact.add_dir("checkpoints/")
wandb.log_artifact(artifact)

# Link to registry
artifact.link("openadapt-ml/qwen-lora-production")
```

### 3. Reports for Stakeholders
W&B Reports can be embedded via iframe:
```html
<iframe src="https://wandb.ai/team/project/reports/Report-Name--VmlldzoxMjM0NTY3"
        style="border:none;height:1024px;width:100%">
</iframe>
```

### 4. Embedding Visualization
W&B can project high-dimensional embeddings (PCA, UMAP, t-SNE) for analyzing model representations.

---

## Pricing Considerations

| Tier | Cost | Limits |
|------|------|--------|
| Free | $0 | 100GB storage, unlimited personal projects |
| Teams | ~$50/user/mo | Collaboration, higher limits |
| Enterprise | Custom | Self-hosted, SSO, audit logs |

For openadapt-ml as an open-source project:
- Individual users: Free tier is sufficient
- Enterprise deployments: They'd need their own W&B account

---

## Alternatives to W&B

| Tool | License | Hosting | Best For |
|------|---------|---------|----------|
| **TensorBoard** | Apache 2.0 | Self/Cloud | TensorFlow users, free |
| **MLflow** | Apache 2.0 | Self/Databricks | Full ML lifecycle, self-hosted |
| **Neptune.ai** | Proprietary | Cloud | Similar to W&B, different UX |
| **Aim** | Apache 2.0 | Self | Open-source W&B alternative |
| **ClearML** | Apache 2.0 | Self/Cloud | MLOps pipelines |

**If we want fully open-source**: MLflow or Aim are the best alternatives.

---

## Recommendation

**Start with Option 4 (Hybrid)**, implemented in phases:

### Phase 1: Optional Metrics Logging (This Week)
```python
# In trainer.py
if config.wandb_enabled:
    import wandb
    wandb.init(project=config.wandb_project or "openadapt-ml")

# In training loop
if config.wandb_enabled:
    wandb.log({"loss": loss, "lr": lr, "epoch": epoch})
```

- Add `--wandb` flag to training CLI
- Log basic metrics only
- No breaking changes

### Phase 2: Evaluation Summary (Next Sprint)
- Log evaluation accuracy/distance to W&B
- Add `viewer_url` to W&B run for deep-dive link

### Phase 3: Sweeps (When Needed)
- Add sweep config generation
- Document hyperparameter search workflow

### Do NOT Do:
- Replace our viewer (unique domain value)
- Log every screenshot to W&B (storage limits, overkill)
- Make W&B required (keep it optional)

---

## Implementation Sketch

```python
# openadapt_ml/training/wandb_logger.py

from typing import Optional
import wandb

class WandbLogger:
    """Optional W&B integration for training runs."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "openadapt-ml",
        config: Optional[dict] = None,
    ):
        self.enabled = enabled
        if enabled:
            wandb.init(project=project, config=config)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_evaluation_summary(self, accuracy: float, avg_distance: float):
        if self.enabled:
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/avg_distance": avg_distance,
            })

    def set_viewer_url(self, url: str):
        if self.enabled:
            wandb.run.summary["viewer_url"] = url

    def finish(self):
        if self.enabled:
            wandb.finish()
```

---

## Sources

- [W&B Logging Documentation](https://docs.wandb.ai/guides/track/log/)
- [W&B Tables Tutorial](https://docs.wandb.ai/tutorials/tables/)
- [W&B Custom Charts](https://docs.wandb.ai/guides/app/features/custom-charts/)
- [W&B Embed Reports](https://docs.wandb.ai/guides/reports/embed-reports/)
- [W&B BoundingBoxes2D](https://docs.wandb.ai/ref/python/data-types/boundingboxes2d/)
- [Gradio W&B Integration](https://www.gradio.app/guides/Gradio-and-Wandb-Integration)
