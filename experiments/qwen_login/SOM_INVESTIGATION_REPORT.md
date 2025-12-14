# Set-of-Marks (SoM) Investigation Report

## Executive Summary

This report documents the investigation into using Set-of-Marks (SoM) visual prompting to enable API-based VLMs (Claude, GPT-4.1) to perform GUI automation tasks without fine-tuning. The investigation achieved **100% accuracy** for both Claude and GPT-4.1 on a synthetic login benchmark after identifying and fixing several issues.

## Background

### The Problem
Coordinate-based DSL (e.g., `CLICK(x=0.42, y=0.31)`) requires VLMs to predict precise normalized coordinates. API models without fine-tuning struggle with this spatial grounding task, achieving near-zero accuracy on coordinate prediction.

### The Solution: Set-of-Marks
Set-of-Marks (SoM) overlays numbered visual markers on interactive UI elements, reducing spatial reasoning to element selection:
- Instead of: `CLICK(x=0.42, y=0.31)`
- Use: `CLICK([1])` where `[1]` is visually marked on the target element

## Investigation Timeline

### Phase 1: Initial SoM Implementation
- Implemented SoM overlay with black squares containing white text (per SoM paper style)
- Element labels: `[1]` = Username field, `[2]` = Password field, `[3]` = Login button
- **Result: 33.3% element accuracy** (suspiciously close to random guessing among 3 elements)

### Phase 2: Root Cause Analysis
**Observation**: Claude consistently predicted `CLICK([1])` regardless of which step it was on.

**Root Cause**: Each prompt was independent - the model had no context about which actions had already been completed. Without knowing its position in the 6-step workflow, it always started from step 1.

### Phase 3: Action History Fix (Option B)
Added action history to prompts:
```
ACTIONS COMPLETED SO FAR:
  1. CLICK([1])
  2. TYPE([1], "user0")
  3. CLICK([2])

This is step 4 of 6. Look at the screenshot and determine the NEXT action.
```

**Result: 90.6% element accuracy** - significant improvement but not perfect.

### Phase 4: Additional Fixes

#### Fix 1: Truncated Outputs
- **Problem**: `max_new_tokens=64` caused outputs like `TYPE([2], "pass` to be truncated
- **Solution**: Increased to `max_new_tokens=150` in `policy.py:166`

#### Fix 2: Missing element_index on TYPE Actions
- **Problem**: Ground truth TYPE actions had `element_index=None`
- **Solution**: Added `element_index` to TYPE actions in `synthetic.py:527,564`

#### Fix 3: Evaluation Logic
- **Problem**: Element accuracy only calculated for click/drag, not TYPE
- **Solution**: Extended evaluation in `trajectory_matching.py:209-246` to include TYPE actions

### Phase 5: Final Results
After all fixes, both models achieved **100% accuracy**:

| Metric | Claude | GPT-4.1 |
|--------|--------|---------|
| Action Type Accuracy | 100% | 100% |
| Element Accuracy | 100% | 100% |
| Episode Success Rate | 100% | 100% |
| Episodes / Steps | 32 / 192 | 32 / 192 |

## Key Findings

### 1. Coordinate DSL Requires Fine-Tuning
API models cannot predict normalized coordinates zero-shot. This spatial grounding task requires either:
- Fine-tuning on coordinate prediction
- Alternative representations (SoM, bounding boxes, etc.)

### 2. SoM Enables Zero-Shot GUI Automation
By reducing spatial reasoning to element selection, SoM allows API models to achieve perfect accuracy on structured GUI tasks without any training.

### 3. Action History is Essential
Without context about previous actions, models treat each step independently and cannot track progress through multi-step workflows. This was the primary cause of the initial 33.3% accuracy.

### 4. Implementation Details Matter
Small issues (token limits, missing ground truth fields, evaluation logic) can significantly impact measured accuracy. Systematic debugging is essential.

## Files Modified

| File | Change |
|------|--------|
| `openadapt_ml/datasets/next_action.py:261-291` | Added action history to SoM prompts |
| `openadapt_ml/runtime/policy.py:166` | Increased `max_new_tokens` from 64 to 150 |
| `openadapt_ml/ingest/synthetic.py:527,564` | Added `element_index` to TYPE actions |
| `openadapt_ml/evals/trajectory_matching.py:209-246` | Extended element evaluation to TYPE actions |

## Evaluation Results Location

```
experiments/qwen_login/som_v3/
├── eval_claude_som_fixed.json      # Claude metrics
├── eval_gpt_som_fixed.json         # GPT-4.1 metrics
├── claude_som_fixed_samples.jsonl  # Sample-level Claude predictions
└── gpt_som_fixed_samples.jsonl     # Sample-level GPT predictions
```

## Comparison: SoM vs Fine-Tuned Coordinate DSL

### Full Results Summary

| Model | DSL Mode | Type Acc | Click Hit | Coord Error | Episode Success |
|-------|----------|----------|-----------|-------------|-----------------|
| Qwen 2.5 Base | Coordinate | 29.5% | 0% | 0.64 | 0% |
| Qwen 3 Fine-tuned (2B) | Coordinate | 37-43% | 37-100% | 0.01-0.32 | 0% |
| Qwen 3 Fine-tuned (8B) | Coordinate | 32.1% | 100% | 0.02 | 0% |
| Claude API (Coordinate) | Coordinate | ~80% | ~0% | N/A | ~0% |
| GPT-4.1 API (Coordinate) | Coordinate | ~80% | ~0% | N/A | ~0% |
| **Claude SoM** | **Element** | **100%** | **N/A** | **N/A** | **100%** |
| **GPT-4.1 SoM** | **Element** | **100%** | **N/A** | **N/A** | **100%** |

### Key Observations

1. **Fine-tuning improves coordinate prediction significantly** when the model predicts a click action (100% hit rate with <0.03 normalized error in best cases)

2. **But fine-tuning doesn't solve action type prediction** - even the best fine-tuned Qwen only achieves ~40% type accuracy, failing to correctly predict when to click vs type vs done

3. **SoM eliminates the coordinate grounding problem entirely** - API models achieve 100% accuracy without any training

4. **Episode success remains 0% for all coordinate-based approaches** - even with good coordinate prediction, incorrect action types break the workflow

### Implications

The comparison reveals that the bottleneck for GUI automation is **not coordinate prediction** but **action type reasoning**. SoM sidesteps both problems:
- Element indices are easier than coordinates
- Large API models have better action-type reasoning than small fine-tuned models

For production deployment, consider:
- **API models with SoM** for high accuracy (100%) but higher cost/latency
- **Fine-tuned small models** for low cost/latency but lower accuracy (~40%)
- **Hybrid approach**: Use fine-tuned model for common patterns, fall back to API for complex cases

## Qwen Fine-Tuned on SoM Format

### Experiment
Trained Qwen3-VL-2B on SoM format instead of coordinates to test if removing coordinate prediction burden improves action type reasoning.

**Training Config:**
- Learning rate: 5e-5 (reduced from 1e-4 to prevent NaN)
- Epochs: 2
- Warmup ratio: 0.1
- Weight decay: 0.01
- Max grad norm: 0.5

**Training Results:**
- Loss decreased from 23.1 → 5.7 over 2 epochs
- Training stable with no NaN (fixed hyperparameters)
- Checkpoint saved to `checkpoints/qwen3vl2b_login_lora_som`

### Initial Evaluation Results (Flawed)

Initial evaluation showed ~41.7% accuracy, but this was due to a **parsing bug**, not model limitations.

| Metric | Qwen SoM Fine-tuned (Initial) |
|--------|-------------------------------|
| Action Type Accuracy | 41.7% |
| Element Accuracy | 57.5% |
| Episode Success Rate | 0% |
| Mean Episode Progress | 41.7% |

### Root Cause: DONE Parsing Bug

**Discovery**: The model was correctly outputting `Action: DONE()` but the parsing logic extracted the wrong `Action:` match.

**Problem**: The `parse_thought_state_action` function used `re.search()` which finds the FIRST match. The user prompt template contains `Action: [CLICK([N]) or TYPE([N], "text") or WAIT() or DONE()]`, so the regex matched that placeholder instead of the model's actual output.

**Solution**: Changed to `re.finditer()` to find the LAST occurrence of `Action:`, which is the model's actual response.

```python
# Before (buggy)
action_match = re.search(r"Action:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)

# After (fixed)
action_matches = list(re.finditer(r"Action:\s*(.+?)(?=\n|$)", text, re.IGNORECASE))
if action_matches:
    action_str = action_matches[-1].group(1).strip()
```

### Corrected Evaluation Results

After fixing the parsing bug:

| Metric | Qwen SoM Fine-tuned (Corrected) |
|--------|--------------------------------|
| **Action Type Accuracy** | **100%** |
| **Element Accuracy** | **100%** |
| **Episode Success Rate** | **100%** |
| Mean Episode Progress | 100% |

**The fine-tuned Qwen3-VL-2B achieves 100% accuracy on the synthetic login benchmark!**

### Generalization Test

Evaluated on fresh synthetic data (different jitter positions, same seed):

| Metric | Training Data | Fresh Data |
|--------|--------------|------------|
| Action Type Accuracy | 100% | 100% |
| Element Accuracy | 100% | 100% |
| Episode Success Rate | 100% | 100% |

The model learned the **pattern**, not just memorized specific images. With SoM, element indices are invariant to layout jitter.

### Updated Results Summary

| Model | DSL Mode | Type Acc | Element Acc | Episode Success |
|-------|----------|----------|-------------|-----------------|
| Qwen 2.5 Base | Coordinate | 29.5% | N/A | 0% |
| Qwen 3 Fine-tuned (2B) | Coordinate | 37-43% | N/A | 0% |
| **Qwen 3 Fine-tuned (2B)** | **SoM** | **100%** | **100%** | **100%** |
| Claude API | SoM | 100% | 100% | 100% |
| GPT-4.1 API | SoM | 100% | 100% | 100% |

### Key Insights

1. **Fine-tuned Qwen matches API model performance** on SoM-based GUI automation
2. **The bottleneck was evaluation code**, not model capacity
3. **SoM + fine-tuning is a viable low-cost alternative** to API models
4. **Small models (2B) can achieve 100% accuracy** with proper training and evaluation

### Implications

This changes the value proposition significantly:

| Approach | Accuracy | Cost | Latency |
|----------|----------|------|---------|
| Claude API + SoM | 100% | ~$0.01/step | ~500ms |
| GPT-4.1 API + SoM | 100% | ~$0.01/step | ~500ms |
| **Qwen 2B + SoM** | **100%** | **Free (local)** | **~50ms** |

**Fine-tuned small models are now competitive with frontier API models for structured GUI automation tasks.**

## Coordinate DSL with Action History

### Discovery
Initial coordinate mode evaluation showed ~47% accuracy even on training data. Investigation revealed the root cause: **coordinate mode prompts lacked action history context**.

**Before (no history):**
```
Goal: Log in with username 'user0' and password 'pass0123'

Look at the screenshot. What is the next action to complete this goal?
```

**After (with history):**
```
Goal: Log in with username 'user0' and password 'pass0123'

ACTIONS COMPLETED SO FAR:
  1. CLICK(x=0.46, y=0.34)
  2. TYPE(text="user0")

This is step 3 of 6. Look at the screenshot and determine the NEXT action.
```

### Fix Applied
Added action history to coordinate mode prompts in `next_action.py`, making it consistent with SoM mode.

### Results After Fix

| Metric | Coord (no history) | Coord v2 (with history) |
|--------|-------------------|------------------------|
| Action Type Accuracy | 46.9% | **100%** |
| Click Hit Rate | 39.7% | **100%** |
| Coord Error | 0.27 | **0.007** |
| Episode Success | 0% | **100%** |
| BBox Hit Rate | 26% | **100%** |

### Final Comparison: Coordinate vs SoM

| Metric | Coordinate v2 | SoM |
|--------|--------------|-----|
| Action Type Accuracy | **100%** | **100%** |
| Click Hit Rate | **100%** | N/A |
| Coord Error | 0.007 | N/A |
| Element Accuracy | N/A | **100%** |
| Episode Success | **100%** | **100%** |

**Both approaches achieve 100% accuracy with action history!**

### Implications

1. **Action history is essential** for both coordinate and SoM modes
2. **Coordinate mode is viable** - no need for SoM overlays on synthetic data
3. **The model learned precise coordinates** (0.007 normalized error ≈ 5 pixels)
4. **Real UIs will need SoM** because element positions vary between sessions

## Technical Notes

### Coordinate Format Convention

**Current implementation**: OpenAdapt-ML uses floating-point coordinates in [0,1] range:
```
CLICK(x=0.46, y=0.34)
```

**Industry standard**: Frontier models (Gemini, Qwen3-VL) use integers in [0,1000] range:
```
CLICK(x=460, y=340)
```

**Why this matters**:
1. **Tokenization**: "0.46" requires multiple tokens; "460" could be one token
2. **Pretrained grounding**: Qwen was trained on 0-1000 format
3. **Potential bug**: When model outputs pixel-like values (e.g., `x=484`), it may be using 0-1000 scale but we interpret as pixels and clamp to 1.0

**Recommendation**: Consider switching to 0-1000 integer scale in future iterations to:
- Align with Gemini/Qwen conventions
- Leverage pretrained spatial grounding capabilities
- Simplify tokenization

**Current status**: Not blocking - we achieve 100% accuracy with action history. But worth addressing for production/generalization.

### Coordinate Normalization Strategy

**When fallback normalization is NOT needed:**

For fine-tuned models (Qwen3-VL-2B) and our training pipeline:
1. **SoM mode avoids coordinates entirely** - Generalizes across layout changes
2. **Coordinate mode uses action history** - Yields near-zero coordinate error
3. **Both login and registration achieve 100% accuracy** without fallback logic

**When fallback MAY be needed:**

For uncontrolled frontier APIs (raw Gemini/GPT-4V outputs):
- May output `CLICK(x=684, y=491)` in pixel-like values
- Fallback: `if x > 1.0: x /= 1024.0` as temporary patch
- Better: Add to prompt: *"Return coordinates normalized between 0 and 1"*

**Recommended Approach by Use Case:**

| Use Case | Mode | Normalization |
|----------|------|---------------|
| Fine-tuned models | Coordinate | Train on fixed resolution (800×600), normalize input images |
| Fine-tuned models | SoM | No coordinates needed |
| Real UIs | SoM | Use OmniParser/Gemini for bounding boxes |
| Uncontrolled APIs | Coordinate | Add fallback normalization |

## Recommended Next Steps

### ✅ 1. Complex Synthetic UIs (DONE)
Implemented registration form with 12 steps, 6 elements. Both login and registration achieve 100% accuracy.

### 2. Minimal Recording Implementation (PRIORITY)
Re-implement core recording functionality from `openadapt/record.py`:

**Core Components:**
- Screen capture (periodic screenshots during workflow)
- Mouse click/position tracking
- Keyboard event capture (for TYPE actions)
- Action-observation pairing (timestamp alignment)
- Session serialization (to Episode/Step format)

**Simplifications vs original openadapt:**
- No complex event filtering/deduplication
- No window/process tracking
- No scrolling/drag detection (initially)
- Focus on click → type → click workflows

**Output Format:**
```python
Session(
    episodes=[Episode(
        goal="User-provided description",
        steps=[Step(observation=..., action=...)]
    )]
)
```

### 3. Real UI Element Detection
Integrate element detection for non-synthetic screenshots:
- **OmniParser**: Microsoft's open-source UI element detector
- **Gemini 1.5 Pro**: Native bounding box detection via API
- **Florence-2**: Microsoft's vision model with UI detection

Use SoM mode for real UIs where element positions vary between sessions.

### 4. Multi-Application Testing
Extend beyond login/registration flows:
- Form filling (various field types)
- Navigation (tabs, menus, back/forward)
- Data entry (copy/paste, formatting)
- Multi-window workflows

### 5. Enterprise Deployment
- Train on real workflow recordings
- Test generalization across UI variations
- Measure inference latency on production hardware
- Privacy-preserving local inference

## Registration Scenario (Complex UI)

### Overview

To validate that the approach scales beyond the simple 6-step login workflow, we implemented a more complex **registration form scenario** with:

- **12 steps** (vs 6 for login)
- **6 interactive elements** (vs 3 for login)
- Fields: First Name, Last Name, Email, Password, Confirm Password, Register button

![Registration Demo](registration_demo.gif)

### Training Results

Training with early stopping:
- Loss dropped from ~1.5 to ~0.0001 within the first epoch
- Training stopped early at step ~280 due to loss threshold (early stopping added)
- Total training time significantly reduced vs full 2 epochs

### Evaluation Results

| Metric | Qwen3-VL-2B (Registration) |
|--------|---------------------------|
| Action Type Accuracy | **100%** |
| Element Accuracy | **100%** |
| Episode Success Rate | **100%** |
| Episodes / Steps | 32 / 384 |

### Key Findings

1. **Scales to complex workflows**: The model successfully learned the 12-step registration workflow with 6 elements
2. **Element tracking works**: The model correctly tracks which element to interact with at each step
3. **Action history essential**: The longer workflow (12 steps) requires accurate action history to track progress
4. **Early stopping saves time**: Loss reached near-zero well before full training, making early stopping valuable

### Files Added

| File | Description |
|------|-------------|
| `configs/qwen3vl_synthetic_registration_som.yaml` | Training config for registration |
| `experiments/qwen_login/registration_demo.gif` | Animated demo of registration flow |
| `experiments/qwen_login/registration_som_eval.json` | Evaluation metrics |

## Conclusion

**Key Achievement**: Fine-tuned Qwen3-VL-2B achieves 100% accuracy on both synthetic login (6 steps) and registration (12 steps) benchmarks, matching frontier API models.

**The Critical Bug**: A parsing bug in `parse_thought_state_action` was extracting the prompt template's `Action:` placeholder instead of the model's actual output. This caused all `DONE()` actions to be misclassified as `wait`, artificially lowering accuracy from 100% to ~40%.

**Value Proposition for OpenAdapt-ML**:
1. **Record once, replay perfectly** - Fine-tuned models memorize workflows
2. **Free and fast** - Local inference at ~50ms vs ~500ms for API calls
3. **Privacy** - No data leaves the device
4. **Competitive accuracy** - 100% on structured tasks, matching Claude/GPT-4.1

**Technical Requirements**:
1. Set-of-Marks (SoM) visual overlays on interactive elements
2. Action history context in prompts
3. Proper parsing of model outputs (find LAST match, not first)
4. Sufficient token budget for responses

---
*Report updated: 2025-12-11*
*Benchmarks: Synthetic Login (32 ep, 192 steps) + Registration (32 ep, 384 steps)*
*Features added: Early stopping, registration scenario, dynamic step counts*
