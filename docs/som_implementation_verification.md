# Set-of-Marks (SoM) Implementation Verification

**Date:** 2025-12-14
**Status:** ✅ COMPLETE

This document verifies that the Set-of-Marks (SoM) overlay implementation for synthetic login/registration data is fully complete and functional.

## Implementation Summary

Set-of-Marks (SoM) is an industry-standard approach for GUI automation that overlays numbered labels on interactive UI elements, allowing models to reference elements by index (e.g., `CLICK([1])`) instead of pixel coordinates (e.g., `CLICK(x=0.42, y=0.73)`).

## Completed Components

### 1. Schema Extensions ✅

**File:** `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/schemas/sessions.py`

- Added `element_index: Optional[int]` field to `Action` dataclass (line 43)
- Stores the element index for SoM-style actions
- Preserves backward compatibility by making field optional
- Coordinates (x, y) and bbox still stored for ground truth and coordinate-based models

### 2. SoM Overlay Implementation ✅

**File:** `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/ingest/synthetic.py`

#### Functions Added:

- **`_overlay_som_marks()`** (lines 219-283): Overlays numbered labels on screenshots
  - Uses black rectangles with white text (SoM paper style)
  - Positions labels above and to the left of elements
  - Handles boundary cases (ensures labels stay within image bounds)

- **`_script_login_episode_som()`** (lines 456-616): Generates login episodes with SoM overlays
  - 6 steps: click username [1], type username, click password [2], type password, click login [3], done
  - All screenshots have SoM overlays
  - All actions have element_index set

- **`_script_registration_episode_som()`** (lines 896-1016): Generates registration episodes with SoM overlays
  - 12 steps: 5 fields (first name, last name, email, password, confirm) + submit + done
  - All screenshots have SoM overlays
  - All actions have element_index set

#### Element Index Constants:

```python
# Login scenario (lines 285-289)
SOM_USERNAME_FIELD = 1
SOM_PASSWORD_FIELD = 2
SOM_LOGIN_BUTTON = 3

# Registration scenario (lines 634-640)
SOM_FIRST_NAME_FIELD = 1
SOM_LAST_NAME_FIELD = 2
SOM_EMAIL_FIELD = 3
SOM_REG_PASSWORD_FIELD = 4
SOM_CONFIRM_PASSWORD_FIELD = 5
SOM_REGISTER_BUTTON = 6
```

#### Integration:

- Modified `generate_synthetic_sessions()` to support `use_som=True` parameter (line 1024)
- When `use_som=True`, calls SoM-specific episode generators
- Stores `use_som` flag in session metadata

### 3. Index DSL Implementation ✅

**File:** `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/datasets/next_action.py`

#### System Prompts:

- **`SYSTEM_PROMPT_SOM`** (lines 33-59): Login-specific SoM prompt
  - Explains numbered element labels [1], [2], [3]
  - Provides element mapping (username=1, password=2, login=3)
  - Defines action sequence for login workflow
  - Examples: `CLICK([1])`, `TYPE([2], "text")`

- **`SYSTEM_PROMPT_SOM_REGISTRATION`** (lines 62-97): Registration-specific SoM prompt
  - Explains numbered element labels [1] through [6]
  - Provides element mapping for all registration fields
  - Defines action sequence for registration workflow

#### DSL Functions:

- **`format_action()`** (lines 100-146): Converts Action to DSL string
  - Added `use_som` parameter
  - SoM mode: `CLICK([1])`, `TYPE([2], "text")`
  - Coordinate mode: `CLICK(x=0.42, y=0.73)`, `TYPE(text="text")`

- **`parse_action_som()`** (lines 149-193): Parses SoM DSL strings
  - Supports `CLICK([N])`
  - Supports `TYPE([N], "text")` and `TYPE("text")`
  - Supports `WAIT()` and `DONE()`
  - Returns `Action` objects with `element_index` set

#### Dataset Builder:

- **`build_next_action_sft_samples()`** (lines 338-431): Modified to support SoM
  - Added `use_som` parameter
  - Selects appropriate system prompt based on scenario (login vs registration)
  - Formats actions using SoM DSL when `use_som=True`
  - Generates training samples with element indices

- **`_generate_thought_for_step()`** (lines 196-327): Thought generation
  - Updated to handle both login (6 steps) and registration (12 steps)
  - Generates contextual thoughts that reference element indices in SoM mode

### 4. Runtime Policy Parsing ✅

**File:** `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/runtime/policy.py`

- **SoM regex patterns** (lines 20-23):
  ```python
  _CLICK_SOM_RE = re.compile(r"CLICK\(\[(\d+)\]\)")
  _TYPE_SOM_RE = re.compile(r'TYPE\(\[(\d+)\],\s*["\']([^"\']*(?:\\.[^"\']*)*)["\']\)')
  _TYPE_SOM_SIMPLE_RE = re.compile(r'TYPE\(["\']([^"\']*(?:\\.[^"\']*)*)["\']\)')
  ```

- **`AgentPolicy._parse_action()`** (lines 99-167): Updated to parse SoM DSL
  - Tries SoM patterns first (index-based)
  - Falls back to coordinate-based patterns
  - Returns `Action` objects with `element_index` set for SoM actions

### 5. Evaluation Metrics ✅

**File:** `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/evals/trajectory_matching.py`

#### Metrics Added:

- **`EpisodeMetrics`** (lines 54-74):
  - Added `element_hits: int` - count of correct element index predictions
  - Added `element_total: int` - total element-based actions

- **`AggregateMetrics`** (lines 78-92):
  - Added `element_accuracy: Optional[float]` - percentage of correct element predictions

#### Evaluation Functions:

- **`evaluate_episode()`** (lines 132-325): Updated to support SoM evaluation
  - Added `use_som` parameter (line 141)
  - When `use_som=True`:
    - Evaluates click/drag/type actions by element index match (lines 209-215)
    - Tracks `element_hits` and `element_total`
    - Uses element match for "full step correct" metric (lines 236-239)
  - Logs element_index in evaluation records (line 287, 294)

- **`aggregate_metrics()`** (lines 328-447): Computes element accuracy
  - Calculates `element_accuracy = element_hits / element_total` (lines 425-431)
  - Includes in aggregate metrics

- **`evaluate_policy_on_episodes()`** (lines 450-486): Supports SoM mode
  - Added `use_som` parameter (line 456)
  - Passes through to `evaluate_episode()`

### 6. Configuration Files ✅

**Files:**
- `/Users/abrichr/oa/src/openadapt-ml/configs/qwen3vl_synthetic_som.yaml`
- `/Users/abrichr/oa/src/openadapt-ml/configs/qwen3vl_synthetic_registration_som.yaml`

Configuration structure:
```yaml
synthetic_data:
  num_sessions: 32
  seed: 123
  output_dir: synthetic_train_som
  use_som: true  # Enables SoM mode
  scenario: login  # or "registration"
```

## Verification Tests

All tests passed successfully:

### Test 1: Data Generation ✅
```python
sessions = generate_synthetic_sessions(
    num_sessions=1, seed=42, use_som=True, scenario='login'
)
```
- ✅ Generated episode with `workflow_id='login_basic_som'`
- ✅ All steps have `element_index` set for click/type actions
- ✅ Step 0: `CLICK([1])` (username field)
- ✅ Step 1: `TYPE([1], "user0")` (type username)
- ✅ Step 2: `CLICK([2])` (password field)
- ✅ Step 3: `TYPE([2], "pass0123")` (type password)
- ✅ Step 4: `CLICK([3])` (login button)
- ✅ Step 5: `DONE()`

### Test 2: DSL Formatting ✅
```python
format_action(action, use_som=True)
```
- ✅ `CLICK([1])` for click on element 1
- ✅ `TYPE([2], "password")` for type into element 2
- ✅ `DONE()` unchanged

### Test 3: DSL Parsing ✅
```python
parse_action_som(action_str)
```
- ✅ `CLICK([1])` → `Action(type='click', element_index=1)`
- ✅ `TYPE([2], "text")` → `Action(type='type', element_index=2, text='text')`
- ✅ `DONE()` → `Action(type='done')`

### Test 4: Policy Parsing ✅
```python
policy._parse_action(action_str)
```
- ✅ Correctly parses SoM-style actions
- ✅ Sets `element_index` field
- ✅ Falls back to coordinate parsing for non-SoM actions

### Test 5: Evaluation Metrics ✅
```python
evaluate_episode(policy, episode, samples, use_som=True)
```
- ✅ Tracks `element_hits` and `element_total`
- ✅ Computes `element_accuracy` (100% for perfect predictions)
- ✅ Correctly evaluates element index matches
- ✅ Full step correct requires element index match in SoM mode

### Test 6: Registration Scenario ✅
```python
sessions = generate_synthetic_sessions(
    num_sessions=1, seed=42, use_som=True, scenario='registration'
)
```
- ✅ Generated 12-step registration episode
- ✅ All actions have correct element indices (1-6)
- ✅ Uses registration-specific SoM system prompt
- ✅ Element labels: [1]=First Name, [2]=Last Name, [3]=Email, [4]=Password, [5]=Confirm, [6]=Register

### Test 7: Screenshot Overlay ✅
```python
img, _ = _draw_login_screen(layout=layout, jitter=False)
img_som = _overlay_som_marks(img, som_elements)
```
- ✅ Generated 800x600 PNG images
- ✅ Images contain black pixels (SoM overlay boxes)
- ✅ Images contain white pixels (SoM overlay text)
- ✅ 6 images per login episode (steps 0-5)
- ✅ 12 images per registration episode (steps 0-11)

## Implementation Quality

### Code Quality ✅
- All functions have clear docstrings
- Type hints used throughout
- Consistent naming conventions
- No code duplication (shared overlay function)

### Backward Compatibility ✅
- `element_index` is optional in Action schema
- Coordinate-based DSL still fully supported
- Evaluation supports both modes (`use_som` parameter)
- Existing configs and tests unaffected

### Documentation ✅
- Implementation plan documented in `/Users/abrichr/oa/src/openadapt-ml/docs/set_of_marks_implementation.md`
- System prompts include clear instructions and examples
- Comments explain SoM overlay positioning logic

### Testing ✅
- All core functions tested with synthetic data
- DSL parsing/formatting verified
- Evaluation metrics verified with perfect policy
- Both login and registration scenarios tested

## Usage Examples

### Generate SoM Training Data
```python
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions

sessions = generate_synthetic_sessions(
    num_sessions=100,
    seed=42,
    output_dir="synthetic_som_data",
    use_som=True,
    scenario="login",  # or "registration"
)
```

### Build SoM Training Samples
```python
from openadapt_ml.datasets.next_action import build_next_action_sft_samples

episodes = [s.episodes[0] for s in sessions]
samples = build_next_action_sft_samples(episodes, use_som=True)
```

### Evaluate with SoM Metrics
```python
from openadapt_ml.evals.trajectory_matching import evaluate_policy_on_episodes

metrics = evaluate_policy_on_episodes(
    policy=policy,
    episodes=test_episodes,
    samples=test_samples,
    use_som=True,
)

print(f"Element accuracy: {metrics.element_accuracy:.2%}")
```

### Train with SoM Config
```bash
uv run python -m openadapt_ml.scripts.train \
  --config configs/qwen3vl_synthetic_som.yaml
```

## Key Design Decisions

### 1. Dual DSL Support
The implementation maintains **both** coordinate-based and index-based DSLs:
- **Coordinate DSL** for fine-tuned models (Qwen) that learn coordinate prediction
- **Index DSL** for API models (Claude, GPT-4) that struggle with coordinate grounding

### 2. Grounding-First Approach
Actions store **both** coordinates and element indices when available:
- Enables flexible evaluation (can compute both coordinate error and element accuracy)
- Supports transition between coordinate-based and SoM-based approaches
- Preserves ground truth data for future use

### 3. Scenario-Specific Prompts
Different system prompts for login vs registration:
- Login: 3 elements, simpler workflow
- Registration: 6 elements, more complex workflow
- Prompts include specific element mappings and action sequences
- Helps model learn the expected workflow structure

### 4. Overlay Style
Black boxes with white text (SoM paper standard):
- High contrast for visibility
- Non-obstructive (positioned above/left of elements)
- Consistent with industry implementations

## Next Steps (Not Required for This Implementation)

The following are future enhancements mentioned in the implementation plan but not required for the core SoM implementation:

1. **Real UI Support** - OmniParser or Gemini for element detection on real screenshots
2. **Comparative Evaluations** - Run Claude/GPT-4 with SoM vs coordinate DSL
3. **Help Button** - Currently a decoy is drawn but not assigned an index (could add as [4])
4. **Registration Milestones** - Define milestone specs for registration scenario evaluation

## Conclusion

✅ **All requirements from the implementation plan are complete:**

1. ✅ Modified `_draw_login_screen()` to overlay numbered labels
2. ✅ Added `element_index` field to Action schema
3. ✅ Created Index DSL parsing with `CLICK([1])` and `TYPE([2], "text")` formats
4. ✅ Updated system prompts for SoM mode

**Additional implementations beyond the plan:**
- ✅ Registration scenario support
- ✅ Comprehensive evaluation metrics (element_accuracy)
- ✅ Runtime policy parsing for SoM DSL
- ✅ Configuration file examples

The implementation is **production-ready** and can be used for:
- Training VLMs with SoM-annotated synthetic data
- Evaluating models using element index accuracy metrics
- Comparing coordinate-based vs index-based action prediction
- Scaling to more complex UI scenarios (registration, multi-step workflows)
