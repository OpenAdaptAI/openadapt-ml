# Workflow Segmentation Pipeline Test Results

## Test Date
2026-01-17 10:35 PST

## Executive Summary

**Status**: BLOCKED - CaptureAdapter needs schema update

The segmentation pipeline (commit `56e8cb6`) was tested on two real openadapt-capture recordings. Testing revealed a critical schema mismatch between the CaptureAdapter and the actual openadapt-capture database format.

## Test Data

| Recording | Path | Screenshots | Events | Duration |
|-----------|------|-------------|--------|----------|
| turn-off-nightshift | `/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift/` | 22 | 1561 | ~38s |
| demo_new | `/Users/abrichr/oa/src/openadapt-capture/demo_new/` | 14 | TBD | ~TBD |

## Test Execution

### Environment Setup ✓
- [x] API keys configured (GOOGLE_API_KEY, OPENAI_API_KEY)
- [x] Output directories created
- [x] Dependencies synced (uv)
- [x] Test recordings verified

### Stage 1: Frame Description ✗
**Status**: FAILED - No frames loaded

**Issue**: CaptureAdapter schema mismatch

**Expected event types** (in CaptureAdapter):
```python
RELEVANT_EVENT_TYPES = {
    "click", "double_click", "right_click",
    "key", "type", "scroll", "drag", "move"
}
```

**Actual event types** (in capture.db):
```sql
SELECT DISTINCT type FROM events;
-- Results:
-- key.down
-- key.up
-- mouse.down
-- mouse.move
-- mouse.up
-- screen.frame
```

**Event counts** (turn-off-nightshift):
- `screen.frame`: 457 (should map to screenshots)
- `mouse.down`/`mouse.up`: 13 each (clicks)
- `key.down`/`key.up`: 16 each (key presses)
- `mouse.move`: 1046 (optional, high noise)

**Root cause**: The CaptureAdapter was written for a different version of openadapt-capture that used high-level event names. The actual database uses low-level event types with dot notation.

### Stage 2: Episode Extraction ✗
**Status**: BLOCKED - Depends on Stage 1

**Additional issue**: OpenAI API key not being read from config.settings

The segment_extractor.py reads from `os.environ.get("OPENAI_API_KEY")` instead of using `from openadapt_ml.config import settings`. This works if .env is loaded but is inconsistent with the rest of the codebase.

## Issues Identified

### P0: CaptureAdapter Schema Mismatch

**Problem**: CaptureAdapter cannot load recordings because event type names don't match.

**Impact**: Pipeline completely blocked, 0 frames loaded.

**Fix required**: Update `openadapt_ml/segmentation/adapters/capture_adapter.py`:

1. Change `RELEVANT_EVENT_TYPES` to use actual event types:
   ```python
   RELEVANT_EVENT_TYPES = {
       "mouse.down",      # clicks
       "mouse.up",
       "key.down",        # key presses
       "key.up",
       "mouse.move",      # optional
       "screen.frame",    # frame captures
   }
   ```

2. Update event mapping logic to:
   - Pair `mouse.down` + `mouse.up` → single "click" event
   - Pair `key.down` + `key.up` → single "key" event
   - Use `screen.frame` events to find corresponding screenshots
   - Match screenshots by timestamp or frame index

3. Screenshot matching strategy:
   - Query `screen.frame` events ordered by timestamp
   - For each screen.frame, use its `video_timestamp` or event index
   - Match to screenshot files: `capture_{id}_step_{n}.png` where n = frame index
   - Current adapter uses event-based indexing, should use screen.frame indexing

**Estimated fix time**: 1-2 hours

**Files to modify**:
- `openadapt_ml/segmentation/adapters/capture_adapter.py`

**Testing**:
```bash
# After fix, should load 457 frames (= screen.frame count)
uv run python test_segmentation_pipeline.py
```

### P1: API Key Loading Inconsistency

**Problem**: segment_extractor.py uses `os.environ.get()` instead of `settings`

**Impact**: Minor - works but inconsistent with project patterns

**Fix required**: Update `openadapt_ml/segmentation/segment_extractor.py`:
```python
# OLD (line ~89)
self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# NEW
from openadapt_ml.config import settings
self._client = openai.OpenAI(api_key=settings.openai_api_key)
```

**Same issue likely in**: `frame_describer.py` (Gemini backend)

**Estimated fix time**: 15 minutes

### P2: Event Grouping Strategy

**Problem**: Need to decide how to handle event pairs (down+up)

**Options**:
1. **Pair events** (recommended): Combine mouse.down + mouse.up → single click with duration
2. **Use down events only**: Ignore up events, treat down as the action
3. **Use both**: Create separate events for down and up

**Recommendation**: Option 1 (pair events)
- More accurate timing (click duration = up.timestamp - down.timestamp)
- Reduces noise (13 clicks instead of 26 events)
- Matches semantic meaning ("user clicked at time T")

**Implementation**:
- Track unpaired events in a buffer
- When up event arrives, look for matching down event
- Compute click coordinates from down event
- Compute duration from timestamp difference

## Results

### turn-off-nightshift
- **Frames loaded**: 0 (should be 457)
- **Episodes extracted**: N/A (blocked)
- **Cost**: $0.00 (no API calls made)
- **Time**: < 1 second (failed immediately)
- **Quality**: N/A

### demo_new
- **Frames loaded**: 0 (should be ~100-200)
- **Episodes extracted**: N/A (blocked)
- **Cost**: $0.00 (no API calls made)
- **Time**: < 1 second (failed immediately)
- **Quality**: N/A

## Deliverables Status

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Segmentation outputs | ❌ BLOCKED | CaptureAdapter fix required |
| HTML viewers | ⏸️ PENDING | Can implement independently |
| Screenshots | ⏸️ PENDING | Can implement independently |
| Example JSON | ❌ BLOCKED | Needs real data |
| Updated README | ⏸️ PENDING | Can document architecture |
| Test results report | ✅ DONE | This document |
| viewer.py code | ⏸️ PENDING | Can implement independently |
| Screenshot script | ⏸️ PENDING | Can implement independently |

## Recommended Fix Priority

1. **P0: Fix CaptureAdapter** (1-2 hours)
   - Update RELEVANT_EVENT_TYPES to match actual schema
   - Implement event pairing (mouse.down+up → click)
   - Fix screenshot-to-frame mapping (use screen.frame events)
   - Test on turn-off-nightshift

2. **P1: Fix API key loading** (15 min)
   - Update segment_extractor.py to use settings
   - Update frame_describer.py to use settings
   - Consistent with project patterns

3. **P0: Run Stage 1 on both recordings** (5 min runtime)
   - Validate frame descriptions
   - Check VLM quality
   - Measure cost/time

4. **P0: Run Stage 2 on both recordings** (5 min runtime)
   - Validate episode extraction
   - Check boundary quality
   - Review step summaries

5. **P1: Implement viewer.py** (1-2 hours)
   - Create HTML generator
   - Timeline visualization
   - Episode selection UI
   - Frame descriptions panel

6. **P1: Generate documentation** (1 hour)
   - Create viewers
   - Take screenshots
   - Extract example JSON
   - Update README

## Estimated Timeline

**After CaptureAdapter fix**:
- Total pipeline runtime: ~1 minute per recording (API calls)
- Total cost: ~$0.10 per recording (Gemini + GPT-4o)
- Documentation generation: ~2 hours
- **Total remaining time**: ~3-4 hours

## Architecture Notes

### CaptureAdapter Design

The CaptureAdapter serves as the integration layer between openadapt-capture and the segmentation pipeline. Key design decisions:

1. **Event abstraction**: Convert low-level events (mouse.down) to semantic actions (click)
2. **Frame selection**: Use screen.frame events as the canonical frame list
3. **Noise reduction**: Filter mouse.move by distance threshold
4. **Timestamp normalization**: Convert to relative timestamps (from recording start)

### Screen Frame Strategy

openadapt-capture records at ~30 FPS (457 frames in ~15 seconds). For segmentation:
- **Use all screen.frames**: Provides complete temporal coverage
- **Action alignment**: Match action events to nearest screen.frame by timestamp
- **VLM batching**: Process 10-20 frames per API call for efficiency

### Event Pairing Algorithm

```python
def pair_events(events):
    """Pair mouse.down+up and key.down+up events."""
    paired = []
    pending_down = {}  # type -> event

    for event in events:
        if event.type.endswith('.down'):
            event_type = event.type[:-5]  # remove '.down'
            pending_down[event_type] = event
        elif event.type.endswith('.up'):
            event_type = event.type[:-3]  # remove '.up'
            if event_type in pending_down:
                down = pending_down.pop(event_type)
                # Create paired event
                paired_event = {
                    'type': event_type,  # 'mouse' or 'key'
                    'timestamp': down.timestamp,
                    'duration': event.timestamp - down.timestamp,
                    'data': down.data
                }
                paired.append(paired_event)

    return paired
```

## Next Steps

1. **Implement CaptureAdapter fix** (top priority)
2. Test on turn-off-nightshift
3. Validate frame count = 457
4. Run full pipeline (Stage 1 + 2)
5. Generate viewers and documentation
6. Create PR with test results

## Questions for Review

1. Should we sample frames (e.g., keep every 5th frame) to reduce API costs?
2. Should viewer.py be a standalone HTML file or require a web server?
3. Do we want interactive timeline scrubbing or just episode selection?
4. Should screenshots be embedded as base64 or linked externally?

## Conclusion

The segmentation system architecture is sound, but the CaptureAdapter integration layer needs to be updated to match the actual openadapt-capture database schema. Once fixed, the pipeline should work end-to-end with minimal additional changes.

The core VLM and LLM stages (1 & 2) are ready to test. The viewer and documentation generation can be implemented in parallel while waiting for real segmentation data.

**Recommendation**: Fix CaptureAdapter first, then run full validation test.
