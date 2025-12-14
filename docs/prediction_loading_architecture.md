# Prediction Loading Architecture

## Current Issues

### 1. "No Prediction" Displayed Despite Predictions Existing

**Symptom**: Viewer shows "No prediction" even though predictions exist in `predictions_epoch3.json`.

**Root Cause Analysis**:

The viewer HTML has multiple script blocks with overlapping responsibilities:

1. **Script 1 (lines ~869-1542)**: Base openadapt-capture viewer
   - Contains `const frames = [...]` with base64 encoded images
   - Has its own `init()` function that initializes the viewer
   - Defines `let currentIndex = 0`
   - Uses `frames` data, not `comparisonData`

2. **Script 2 (lines ~1544-1737)**: Comparison overlay script
   - Contains `const comparisonData = [...]` WITH predictions
   - Defines `function updateComparison(index)`
   - Has its own `computeMetrics()`, `updateClickOverlays()` functions
   - No explicit initialization call

3. **Script 3 (lines ~1739-1807)**: Checkpoint dropdown script
   - Contains `predictionsByCheckpoint` with all checkpoints
   - `initCheckpointDropdown()` called via setTimeout
   - Calls `applyCheckpointPredictions()` which modifies `comparisonData`

**The Problem**:
- When Script 3 calls `applyCheckpointPredictions()`, it updates `comparisonData`
- Then tries to call `updateComparison(currentIndex)` to refresh display
- But `currentIndex` and `updateComparison` may have scoping issues between scripts
- Also, the base viewer in Script 1 uses `frames` data independently of `comparisonData`

### 2. Model Output Format Issue

The model outputs narrative text instead of structured `CLICK(x=..., y=...)` format:

```json
{
  "predicted_action": {
    "type": "predicted",
    "raw_output": "Based on the current state of the terminal..."
  },
  "match": false
}
```

Expected format with coordinates:
```json
{
  "predicted_action": {
    "type": "click",
    "x": 0.42,
    "y": 0.73
  },
  "match": true
}
```

**Cause**: The fine-tuned model isn't outputting the expected `Action: CLICK(x=..., y=...)` format. This could be:
- Insufficient training epochs
- Training data format mismatch
- Model too small to learn the format (current: 2B/8B, recommended: 27B+)

## Proposed Solutions

### Short-term Fix: Script Consolidation

1. **Consolidate into single script**: Move all JavaScript into one `<script>` block to eliminate scoping issues

2. **Explicit initialization**: Call `updateComparison(0)` after `comparisonData` is populated

3. **Global variables**: Use `window.` prefix for variables that need cross-script access:
   ```javascript
   window.comparisonData = [...];
   window.currentIndex = 0;
   window.updateComparison = function(index) { ... };
   ```

### Long-term Fix: Clean Architecture

1. **Single source of truth**: One `viewer_data.js` file containing:
   ```javascript
   const ViewerData = {
     frames: [...],
     comparisonData: [...],
     predictionsByCheckpoint: {...}
   };
   ```

2. **Modular viewer**: Separate concerns:
   - `viewer_core.js` - Base playback/timeline functionality
   - `viewer_comparison.js` - Prediction overlay and comparison
   - `viewer_checkpoints.js` - Checkpoint dropdown logic

3. **Event-driven updates**: Use custom events:
   ```javascript
   document.dispatchEvent(new CustomEvent('checkpointChanged', {
     detail: { checkpoint: 'Epoch 3' }
   }));
   ```

### Model Output Fix

1. **Improve parsing**: Add more flexible regex patterns:
   ```python
   # Current patterns
   CLICK(x=0.42, y=0.31)
   click at (0.42, 0.31)

   # Add patterns for:
   "click the button at position (0.42, 0.31)"
   "coordinates: 0.42, 0.31"
   ```

2. **Post-processing with VLM**: If no coordinates found, use a second VLM call to extract:
   ```python
   if not parsed_coords:
       coords = vlm.extract_coords(raw_output, screenshot)
   ```

3. **Training data audit**: Ensure training targets exactly match expected format

## Implementation Plan

### Phase 1: Fix Display (Quick)
1. Move checkpoint script logic into the main comparison script
2. Add explicit `updateComparison(0)` call after data loading
3. Ensure `comparisonData` modifications propagate to display

### Phase 2: Improve Parsing (Medium)
1. Add more flexible coordinate extraction patterns
2. Show raw model output when no coords available (already partly implemented)
3. Add "parse error" indicator instead of "no prediction"

### Phase 3: Architecture Cleanup (Future)
1. Refactor to modular JavaScript architecture
2. Create reusable viewer components
3. Add unit tests for viewer JavaScript

## Debugging Steps

1. **Check browser console** for JavaScript errors when loading viewer.html
   - Open http://localhost:8765/viewer.html
   - Press F12 → Console tab
   - Look for red error messages

2. **Verify data is correct** in browser console:
   ```javascript
   console.log(comparisonData[0]);  // Should show predicted_action and match
   console.log(predictionsByCheckpoint);  // Should show all checkpoints
   ```

3. **Test updateComparison manually**:
   ```javascript
   updateComparison(0);  // Should update display
   ```

## Known Issues

### Preview Checkpoint Shows "No Prediction"
The "Preview" checkpoint contains placeholder predictions with coordinates but `match: null`. This is because preview predictions are generated before actual model inference runs. Select "Epoch 3" (or other trained checkpoints) to see actual model predictions.

### Missing `recording.end` Event
The events list shows `recording.start` but no corresponding `recording.end`. This should be added during capture completion in the openadapt-capture module.

## Files to Modify

- `openadapt_ml/training/trainer.py` - Fix `_enhance_comparison_to_unified_viewer()`
- `openadapt_ml/scripts/compare.py` - Improve coordinate parsing in `predict_action()`
- `docs/viewer_layout_redesign.md` - Cross-reference with layout changes
