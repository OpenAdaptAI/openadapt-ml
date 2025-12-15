# Viewer Enhancements: Timeline, Evaluation & Transcript Integration

## Timeline Visualizer / Scrubber

### Concept
A horizontal timeline bar that visually represents the entire capture duration with:
- **Transcript segments** as colored regions (synced with audio)
- **Action markers** at specific timestamps (clicks, types, etc.)
- **Current position indicator** (playhead)
- **Segment boundaries** showing where transcript segments start/end

### Visual Design
```
┌─────────────────────────────────────────────────────────────────┐
│ ▼                    ▼              ▼         ▼                 │
│ [  Segment 1  ][   Segment 2   ][  Seg 3  ][    Segment 4    ] │
│     ●    ●         ●      ●                    ●     ●   ●     │
│     ↑playhead                                                   │
└─────────────────────────────────────────────────────────────────┘
  ▼ = segment boundaries
  ● = action markers (clicks/types)
```

### Interactions
- **Click anywhere** → seek to that time (both steps and audio)
- **Hover segment** → show transcript text tooltip
- **Click segment** → highlight corresponding transcript text
- **Action markers** → different colors by action type (click=red, type=green, scroll=purple)

### Data Sources
- `transcript.json` → segment boundaries and text
- `baseData` → action timestamps and types
- `audio.mp3` duration → timeline scale

### Implementation
```javascript
function renderTimeline() {
  const totalDuration = audioElement.duration || baseData[baseData.length-1].time;

  // Render transcript segments as background regions
  transcriptSegments.forEach(seg => {
    const left = (seg.start / totalDuration) * 100;
    const width = ((seg.end - seg.start) / totalDuration) * 100;
    // Create segment div with tooltip
  });

  // Render action markers
  baseData.forEach(step => {
    const left = (step.time / totalDuration) * 100;
    // Create marker div with action type color
  });
}
```

---

## Current State

### Training Dashboard
- **Evaluation Samples gallery**: Grid of screenshots with H (human) and AI (predicted) click markers
- **Filters**: Epoch dropdown, correctness filter (All/Correct/Incorrect)
- **Per-sample info**: Distance metric, coordinates, raw model output
- **Timing**: Samples evaluated at end of each epoch during training

### Viewer Tab
- **Full step playback**: All capture steps in sequence
- **Checkpoint selector**: Switch between prediction sets (None, Epoch 1, 2, 3...)
- **Per-step comparison**: Human vs AI action boxes, match indicator
- **Click overlays**: H/AI markers on screenshot (toggleable)

## Gap Analysis

The training tab shows a **subset** of steps (those evaluated during training), while the viewer shows **all** steps. Users can't easily:
1. See which steps were evaluated during training
2. Jump to evaluated steps in the viewer
3. Understand per-step accuracy over training epochs

## Integration Options

### Option A: Evaluation Badges in Step List
Add visual badges to the viewer's step list indicating:
- Whether the step was evaluated
- Correctness status (green checkmark / red X)
- Which epochs it was evaluated at

**Pros**: Non-intrusive, works with existing UI
**Cons**: Doesn't show evaluation progression over epochs

### Option B: Evaluation Filter Mode
Add a filter toggle to show only evaluated steps:
- "Show All" / "Show Evaluated Only" toggle
- When filtered, step list only shows evaluated steps
- Step numbers preserved for context

**Pros**: Focuses attention on evaluated steps
**Cons**: Loses context of surrounding steps

### Option C: Epoch Comparison View (Recommended)
Extend the checkpoint dropdown to show per-step accuracy:
- When checkpoint selected, show accuracy badge next to each step
- Details panel shows prediction progression: Epoch 1 → 2 → 3
- Can see how model improved on specific steps over training

**Implementation:**
```
Step 7 [click] ✓ E1 ✗ E2 ✓ E3   <- badges showing correctness at each epoch
```

### Option D: Side-by-Side Epoch Comparison
New view mode showing same step across multiple epochs:
- Split view: Epoch 1 | Epoch 2 | Epoch 3
- See prediction drift/improvement visually
- Useful for debugging model behavior

## Data Requirements

The viewer already has access to `predictionsByCheckpoint` which contains predictions organized by epoch. To show evaluation status, we need:

1. **evaluations** from training_log.json (already available)
2. **Mapping** from evaluation sample_idx to step index
3. **Per-epoch correctness** status

## Recommended Implementation

**Phase 1: Evaluation badges**
- Add `eval-badge` class to step items that were evaluated
- Show ✓/✗ based on correctness
- Tooltip shows distance and epoch

**Phase 2: Details panel enhancement**
- When step was evaluated, show evaluation history
- "Evaluated at: Epoch 1 (✗ 12.3px), Epoch 3 (✓ 4.1px)"
- Show improvement trend

**Phase 3: Gallery view toggle**
- Button to switch between "Playback" and "Evaluation Gallery" views
- Gallery view shows only evaluated steps in grid layout
- Matches training dashboard eval panel visual style

## Files to Modify

- `openadapt_ml/training/trainer.py`: `_generate_unified_viewer_from_extracted_data()`
  - Add evaluation data to JS
  - Add badges to step list HTML
  - Enhance details panel

- `openadapt_ml/cloud/local.py`: `regenerate_viewer()`
  - Pass evaluation data from training_log.json to viewer generation

---

## Implementation Priority

1. **Timeline Visualizer** (High) - Core navigation improvement
   - Transcript segments as colored regions
   - Action markers by type
   - Click-to-seek with audio sync

2. **Evaluation Badges** (Medium) - Training/viewer connection
   - ✓/✗ badges on evaluated steps
   - Tooltip with distance metric

3. **Details Panel Enhancement** (Medium) - Deeper insights
   - Evaluation history across epochs
   - Improvement trend visualization

4. **Gallery View Toggle** (Low) - Alternative view mode
   - Switch between playback and grid views
   - Matches training dashboard style
