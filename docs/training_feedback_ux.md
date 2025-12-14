# Training Feedback UX: Critical Path Analysis

## The Core Problem

**User trains a model → sees loss curves → has no idea if it actually learned anything useful**

Loss going down is necessary but not sufficient:
- Model could be overfitting
- Model could be learning the wrong objective
- Coordinate predictions could be "close" but miss the target element
- No way to compare "before training" vs "after training"

## What Would Give Users Confidence?

### Level 1: Quantitative Signals (Current)
- Loss curve going down ✓
- "Accuracy" percentage ✗ (broken - coordinate threshold meaningless)

### Level 2: Qualitative Examples (Needed)
- Side-by-side: human action vs model prediction
- Visual overlay: click positions on screenshot
- Model's reasoning text (Thought: ... Action: ...)
- Progression: same screenshot across epochs

### Level 3: Meaningful Metrics (Future)
- Element-level accuracy: did it click the RIGHT element?
- Region IoU: how much does predicted region overlap with target?
- ScreenSpot-Pro style metrics (GUI-Actor approach)

### Level 4: Agent Conversation (Future)
- "Why did the model click there?"
- "Is it learning the right thing?"
- "What should I do to improve?"

## Current State vs Ideal State

### Dashboard (training progress)

| Aspect | Current | Ideal |
|--------|---------|-------|
| Loss curve | ✓ Works | ✓ |
| Epoch/step progress | ✓ Works | ✓ |
| Eval samples | Shows coords, "correct" | Show screenshot + visual overlay |
| Model thinking | Shows truncated text | Full reasoning, expandable |
| Accuracy metric | Broken (coordinate threshold) | Element-level or region IoU |

### Viewer (human vs model comparison)

| Aspect | Current | Ideal |
|--------|---------|-------|
| Human actions | ✓ Shows timeline | ✓ |
| Video playback | ✓ Works | ✓ |
| Model predictions | ✗ "No model loaded" | Per-step predictions |
| Visual comparison | ✗ Missing | Screenshot + click overlays |
| Checkpoint selector | ✓ Dropdown exists | ✓ + show progression |
| Screenshots | ✗ 404 (on Lambda) | Synced locally |

## The Critical UX Flow

```
1. User records capture
   └── openadapt-capture creates screenshots + events

2. User starts training
   └── Lambda GPU begins training

3. Dashboard shows training progress
   └── Loss curves, epoch progress, setup logs

4. At each epoch evaluation:
   └── Model predicts on sample screenshots
   └── Predictions saved to training_output/
   └── Dashboard updates eval samples section

5. Viewer shows human vs model:
   └── User selects epoch/checkpoint
   └── Sees side-by-side: human clicked HERE, model clicked THERE
   └── Visual overlay on screenshot
   └── Model's reasoning: "I clicked there because..."

6. User understands progression:
   └── "Epoch 1: model clicks random places"
   └── "Epoch 3: model clicks near target"
   └── "Epoch 5: model clicks correct element"

7. User has confidence: "It worked!"
```

## What's Blocking This Today

### 1. Screenshots Not Synced
- Eval samples reference `/home/ubuntu/capture/screenshots/...`
- These exist on Lambda, not locally
- Viewer shows 404

**Fix:** Sync screenshots from Lambda during training, or embed base64 in predictions

### 2. Evaluation Metrics Are Meaningless
- `distance < 50` threshold where distance is 0-1 normalized
- Everything marked "correct" even when 45% off
- No element-level ground truth

**Fix:** Move to GUI-Actor style region proposals, or at minimum use element bboxes

### 3. Viewer Doesn't Show Model Predictions During Training
- Checkpoint dropdown exists but predictions aren't generated mid-training
- User has to wait until training completes

**Fix:** Generate predictions at each epoch checkpoint, save to training_output/

### 4. No Visual Comparison
- Viewer shows video and events
- Doesn't overlay predicted vs actual click positions
- No screenshot-level comparison

**Fix:** Add visual overlay layer to viewer

## Proposed Architecture

### Data Flow

```
Lambda Training
    │
    ├── training_log.json (loss, epoch, step)
    │
    ├── predictions_epoch{N}.json
    │   └── [{step, image_path, human_action, predicted_action, raw_output}]
    │
    ├── screenshots/ (synced from capture)
    │   └── capture_XXXX_step_N.png
    │
    └── checkpoints/
        └── epoch{N}/
```

### Viewer Architecture

```
┌─────────────────────────────────────────────────┐
│ Capture Viewer                                  │
├─────────────────────────────────────────────────┤
│ [Checkpoint: Epoch 3 ▼]  [Step: 7/21 ◀ ▶]      │
├─────────────────────────────────────────────────┤
│ ┌─────────────────────┐  ┌───────────────────┐ │
│ │   Screenshot        │  │ Action Details    │ │
│ │   ┌───┐             │  │                   │ │
│ │   │ H │ ← Human     │  │ Human: CLICK      │ │
│ │   └───┘   click     │  │ (0.65, 0.65)      │ │
│ │       ┌───┐         │  │ Element: "General"│ │
│ │       │ M │ ← Model │  │                   │ │
│ │       └───┘   pred  │  │ Model: CLICK      │ │
│ │                     │  │ (0.74, 0.21)      │ │
│ └─────────────────────┘  │ Element: ???      │ │
│                          │                   │ │
│                          │ Distance: 45%    │ │
│                          │ Correct: ✗        │ │
│                          └───────────────────┘ │
├─────────────────────────────────────────────────┤
│ Model Reasoning:                                │
│ "The next action is to click the General       │
│ section in System Settings because..."          │
└─────────────────────────────────────────────────┘
```

### Progression View

```
┌─────────────────────────────────────────────────┐
│ Training Progression (Step 7)                   │
├─────────────────────────────────────────────────┤
│ Epoch 1        Epoch 3        Epoch 5          │
│ ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│ │  H   M  │   │  H  M   │   │  HM     │       │
│ │  ↓   ↓  │   │  ↓  ↓   │   │  ↓      │       │
│ └─────────┘   └─────────┘   └─────────┘       │
│ Dist: 45%     Dist: 12%     Dist: 3%          │
│ ✗ Wrong       ✓ Close       ✓ Correct         │
└─────────────────────────────────────────────────┘
```

## Priority Matrix

| Task | Impact | Effort | Priority |
|------|--------|--------|----------|
| Sync screenshots from Lambda | High | Low | **P0** |
| Show predictions in viewer | High | Medium | **P0** |
| Visual overlay on screenshot | High | Medium | **P1** |
| Fix accuracy metrics | High | High | **P1** |
| Progression view | Medium | Medium | **P2** |
| Agent conversation | Low | High | **P3** |

## Immediate Next Steps

### P0: Get screenshots and predictions visible (1-2 hours)

1. **Sync screenshots during training**
   - In lambda_labs.py monitor loop, rsync screenshots folder
   - Or: embed screenshots as base64 in predictions JSON

2. **Generate predictions at each epoch**
   - Already happening? Check if predictions_epoch{N}.json exists
   - If not, add evaluation step after each epoch

3. **Wire viewer to show predictions**
   - Load predictions JSON for selected checkpoint
   - Display alongside human actions

### P1: Make comparison meaningful (2-4 hours)

4. **Visual overlay**
   - Add canvas layer over screenshot
   - Draw circles/markers at human and predicted positions
   - Color code: green=human, purple=predicted

5. **Better metrics**
   - Short term: use percentage (0-100%) not misleading "correct"
   - Medium term: element-level accuracy
   - Long term: GUI-Actor region proposals

## Questions for User

1. Do you want screenshots synced locally or embedded as base64?
2. Is element-level accuracy sufficient or do we need full GUI-Actor?
3. Should we add agent conversation capability now or defer?
