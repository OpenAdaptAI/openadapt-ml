# Workflow Segmentation Pipeline Test Plan

## Test Date
2026-01-17

## Objective
Validate the workflow segmentation pipeline (commit `56e8cb6`) on real captures:
1. Run Stages 1-2 on real captures
2. Generate HTML viewers to visualize results
3. Create screenshots for documentation
4. Produce example outputs for README

## Test Data

### Turn Off Night Shift
- **Path**: `/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift/`
- **Screenshots**: 22 frames
- **Task**: Configure Night Shift settings in macOS System Preferences
- **Database**: capture.db (SQLite format)

### Demo New
- **Path**: `/Users/abrichr/oa/src/openadapt-capture/demo_new/`
- **Screenshots**: 14 frames
- **Task**: Unknown demo task
- **Database**: capture.db (SQLite format)

## Test Stages

### Stage 0: Environment Setup ✓
- [x] Verify API keys are set (GEMINI_API_KEY, OPENAI_API_KEY)
- [x] Create output directories
- [x] Verify CaptureAdapter exists
- [x] Check uv sync completed

### Stage 1: Frame Description (VLM)
**Command**:
```bash
cd /Users/abrichr/oa/src/openadapt-ml
uv run python -m openadapt_ml.segmentation.cli describe \
  --recording /Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift \
  --model gemini-2.0-flash \
  --format json \
  --output segmentation_output/turn-off-nightshift_transcript.json \
  --verbose 2>&1 | tee segmentation_output/stage1_nightshift.log
```

**Expected Output**:
- `segmentation_output/turn-off-nightshift_transcript.json`
- JSON containing ActionTranscript with 22 FrameDescription objects
- Each frame should have: timestamp, visible_application, visible_elements, action_type, apparent_intent

**Success Criteria**:
- [ ] CLI completes without errors
- [ ] JSON file created and valid
- [ ] All 22 frames have descriptions
- [ ] VLM correctly identifies macOS System Preferences
- [ ] Actions are properly labeled (click, type, etc.)

**Cost Estimate**: $0.01 - $0.05 (Gemini 2.0 Flash)
**Time Estimate**: 20-30 seconds

### Stage 2: Episode Extraction (LLM)
**Command**:
```bash
cd /Users/abrichr/oa/src/openadapt-ml
uv run python -m openadapt_ml.segmentation.cli extract \
  --transcript segmentation_output/turn-off-nightshift_transcript.json \
  --model gpt-4o \
  --output segmentation_output/turn-off-nightshift_episodes.json \
  --verbose 2>&1 | tee segmentation_output/stage2_nightshift.log
```

**Expected Output**:
- `segmentation_output/turn-off-nightshift_episodes.json`
- JSON containing EpisodeExtractionResult with 1-3 Episode objects
- Episodes should have: name, description, start_time, end_time, step_summaries, boundary_confidence

**Success Criteria**:
- [ ] CLI completes without errors
- [ ] JSON file created and valid
- [ ] At least 1 episode extracted
- [ ] Episode boundaries are reasonable (not too short/long)
- [ ] Steps are coherent and match the task

**Cost Estimate**: $0.01 - $0.02 (GPT-4o)
**Time Estimate**: 5-15 seconds

### Stage 3: Repeat for demo_new
**Commands**:
```bash
# Stage 1
uv run python -m openadapt_ml.segmentation.cli describe \
  --recording /Users/abrichr/oa/src/openadapt-capture/demo_new \
  --model gemini-2.0-flash \
  --format json \
  --output segmentation_output/demo_new_transcript.json \
  --verbose 2>&1 | tee segmentation_output/stage1_demo.log

# Stage 2
uv run python -m openadapt_ml.segmentation.cli extract \
  --transcript segmentation_output/demo_new_transcript.json \
  --model gpt-4o \
  --output segmentation_output/demo_new_episodes.json \
  --verbose 2>&1 | tee segmentation_output/stage2_demo.log
```

**Success Criteria**:
- [ ] Both stages complete successfully
- [ ] 14 frames described
- [ ] At least 1 episode extracted

### Stage 4: Generate HTML Viewers
Create `openadapt_ml/segmentation/viewer.py` to generate interactive viewers.

**Features**:
- Timeline showing all frames
- Episode boundaries highlighted
- Click episodes to see details
- Display frame descriptions
- Show extracted steps

**Output Files**:
- `segmentation_output/turn-off-nightshift_viewer.html`
- `segmentation_output/demo_new_viewer.html`

**Success Criteria**:
- [ ] viewer.py created and functional
- [ ] HTML files generated
- [ ] Viewers open in browser
- [ ] Episode selection works
- [ ] All data displayed correctly

### Stage 5: Generate Screenshots
Create `scripts/generate_segmentation_screenshots.py` using Playwright.

**Screenshots to Capture**:
1. Full timeline view showing all frames and episodes
2. Episode detail view (first episode selected)
3. Frame description panel

**Output Directory**: `docs/images/segmentation/`

**Success Criteria**:
- [ ] Screenshot script created
- [ ] 3 screenshots generated per viewer (6 total)
- [ ] Screenshots are high quality (1200x800)
- [ ] All UI elements visible

### Stage 6: Extract Example JSON
Create example outputs for documentation.

**Commands**:
```bash
# Example episode
cd /Users/abrichr/oa/src/openadapt-ml
cat segmentation_output/turn-off-nightshift_episodes.json | \
  jq '.episodes[0]' > docs/examples/segmentation_example_episode.json

# Example frames
cat segmentation_output/turn-off-nightshift_transcript.json | \
  jq '.frames[0:3]' > docs/examples/segmentation_example_frames.json
```

**Success Criteria**:
- [ ] Example files created
- [ ] JSON is properly formatted
- [ ] Contains representative data

### Stage 7: Update README
Add to `openadapt_ml/segmentation/README.md`:

**Sections to Add**:
1. "Example Results" section with screenshots
2. Real-world test data section
3. Cost and time benchmarks
4. Links to example JSON files

**Success Criteria**:
- [ ] README updated with examples
- [ ] Screenshots embedded
- [ ] Example JSON snippets included
- [ ] Test results documented

### Stage 8: Create Test Results Report
Create `SEGMENTATION_TEST_RESULTS.md` with:

**Content**:
- Test summary
- Per-recording results (frames, episodes, cost, time)
- Quality assessment
- Issues encountered
- Fixes applied
- Next steps

**Success Criteria**:
- [ ] Report created
- [ ] All metrics documented
- [ ] Issues and fixes listed
- [ ] Recommendations included

## Expected Deliverables

1. **Segmentation Outputs**:
   - `segmentation_output/turn-off-nightshift_transcript.json`
   - `segmentation_output/turn-off-nightshift_episodes.json`
   - `segmentation_output/demo_new_transcript.json`
   - `segmentation_output/demo_new_episodes.json`

2. **HTML Viewers**:
   - `segmentation_output/turn-off-nightshift_viewer.html`
   - `segmentation_output/demo_new_viewer.html`

3. **Screenshots**:
   - `docs/images/segmentation/nightshift_timeline.png`
   - `docs/images/segmentation/nightshift_episode_detail.png`
   - `docs/images/segmentation/nightshift_frames.png`
   - (same 3 for demo_new)

4. **Example JSON**:
   - `docs/examples/segmentation_example_episode.json`
   - `docs/examples/segmentation_example_frames.json`

5. **Documentation**:
   - `openadapt_ml/segmentation/README.md` (updated)
   - `SEGMENTATION_TEST_RESULTS.md` (new)

6. **Code**:
   - `openadapt_ml/segmentation/viewer.py` (new)
   - `scripts/generate_segmentation_screenshots.py` (new)

## Success Metrics

### Functional
- [ ] Pipeline runs end-to-end without errors
- [ ] CaptureAdapter correctly loads from capture.db
- [ ] Frame descriptions are accurate and meaningful
- [ ] Episodes are correctly segmented
- [ ] Viewers are interactive and display all data

### Quality
- [ ] Episode boundaries make semantic sense
- [ ] Step summaries match actual actions
- [ ] Confidence scores are reasonable (>0.7)
- [ ] No duplicate or overlapping episodes

### Performance
- [ ] Total processing time < 60 seconds per recording
- [ ] Total cost < $0.10 per recording
- [ ] Memory usage reasonable (< 2GB)

### Documentation
- [ ] Screenshots clearly show functionality
- [ ] Examples are self-explanatory
- [ ] README is accurate and complete
- [ ] Test report is thorough

## Failure Scenarios & Mitigations

### CaptureAdapter Issues
**Symptom**: Cannot load capture.db
**Mitigation**: Check database schema, verify files exist, add error logging

### API Errors
**Symptom**: VLM/LLM API calls fail
**Mitigation**: Check API keys, verify rate limits, add retry logic

### Poor Segmentation Quality
**Symptom**: Episodes don't make sense
**Mitigation**: Review prompts, adjust thresholds, try different models

### Viewer Generation Fails
**Symptom**: HTML not generated or broken
**Mitigation**: Simplify viewer first, add features incrementally

## Timeline

Total estimated time: 2-3 hours

- Environment setup: 10 min
- Stage 1+2 (nightshift): 20 min
- Stage 1+2 (demo_new): 20 min
- Create viewer.py: 30 min
- Generate viewers: 10 min
- Screenshot script: 20 min
- Generate screenshots: 10 min
- Extract examples: 5 min
- Update README: 15 min
- Test results report: 10 min
- Validation & fixes: 30 min

## Notes

- Use Gemini 2.0 Flash for Stage 1 (cheap, fast)
- Use GPT-4o for Stage 2 (better segmentation)
- Skip Stage 3 (deduplication) for now - only have 2 recordings
- Skip Stage 4 (annotation) for now - focus on basic pipeline
- Generate standalone HTML viewers (no external dependencies)
- Use dark theme for viewers (consistent with existing dashboards)
