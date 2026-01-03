# Capture Format Decision: Native vs Conversion Layer

**Date**: January 2026
**Status**: Decision Needed
**Author**: OpenAdapt Team

## Context

We have two projects that need to share data:

1. **openadapt-capture**: Records user interactions (screenshots, clicks, keystrokes)
2. **openadapt-ml**: Trains models and runs experiments on interaction data

Currently, openadapt-capture has its own output format, and openadapt-ml has the unified `Episode` schema. We need to decide how these should interoperate.

## Decision Framework

### Key Questions

1. **Who owns the schema?** Where is the source of truth?
2. **What's the interface contract?** What format do consumers expect?
3. **Where does conversion happen?** In the producer, consumer, or middle layer?
4. **What about versioning?** How do we handle schema evolution?

---

## Options

### Option A: openadapt-capture Outputs Native Episode Format

**Description**: Modify openadapt-capture to write recordings directly in the `Episode` Pydantic schema format.

**Pros**:
- No conversion needed in openadapt-ml
- Single source of truth (openadapt-ml owns schema)
- Recordings are immediately usable
- Validates at capture time (Pydantic catches errors)

**Cons**:
- Creates dependency: openadapt-capture depends on openadapt-ml
- Schema changes require updating both projects
- May lose capture-specific metadata that doesn't fit Episode
- Heavier dependency for capture tool

**Implementation**:
```python
# In openadapt-capture
from openadapt_ml.schema import Episode, Step, Action, Observation

class CaptureRecorder:
    def save(self, path: str) -> None:
        episode = Episode(
            episode_id=self.session_id,
            instruction=self.goal,
            steps=[
                Step(
                    step_index=i,
                    observation=Observation(...),
                    action=Action(...)
                )
                for i, event in enumerate(self.events)
            ]
        )
        path.write_text(episode.model_dump_json())
```

### Option B: Conversion Layer in openadapt-ml (Current Approach)

**Description**: openadapt-capture writes its native format; openadapt-ml has converters.

**Pros**:
- Loose coupling between projects
- Capture format can evolve independently
- Can support multiple capture sources
- Conversion can handle versioning/migrations

**Cons**:
- Duplicate effort maintaining converters
- Potential data loss in conversion
- Extra step before using recordings
- Two schemas to understand

**Implementation** (current):
```python
# In openadapt-ml
from openadapt_ml.ingest.capture import capture_to_episode

episode = capture_to_episode("/path/to/capture")
```

### Option C: Shared Schema Package (Third Project)

**Description**: Extract `Episode` schema to a separate package that both depend on.

**Pros**:
- Clear ownership of schema
- Both projects stay decoupled
- Versioned schema can be pinned
- Other tools can also produce Episodes

**Cons**:
- Third package to maintain
- Coordination overhead
- Potential version conflicts

**Implementation**:
```
openadapt-schema/       # New package
  pyproject.toml
  openadapt_schema/
    episode.py         # Episode, Step, Action, Observation

openadapt-capture/
  pyproject.toml       # depends on openadapt-schema

openadapt-ml/
  pyproject.toml       # depends on openadapt-schema
```

### Option D: Dual Output in openadapt-capture

**Description**: openadapt-capture writes both native format AND Episode format.

**Pros**:
- Backward compatibility
- Choice of which format to use
- Gradual migration path

**Cons**:
- Duplicate data on disk
- Must keep both formats in sync
- Complexity in capture tool

---

## Recommended Approach

**Recommendation: Option B (Conversion Layer) with Documentation**

### Rationale

1. **Proven pattern**: The existing `capture_to_episode()` works and is tested
2. **Separation of concerns**: Capture is about recording; ML is about training
3. **Flexibility**: Can add support for other capture formats later
4. **Lower risk**: No changes required to openadapt-capture
5. **Schema evolution**: openadapt-ml can evolve schema without breaking capture

### Guidelines for Conversion Layer

1. **Converters are the ONLY legacy boundary**
   - All conversion happens in `openadapt_ml/ingest/`
   - Rest of codebase uses Episode schema only

2. **Lossless where possible**
   - Store original capture data in `raw_observation`, `raw_action` fields
   - Convert what we can, preserve what we can't

3. **Validate on conversion**
   - Pydantic validation catches issues early
   - Clear error messages for malformed data

4. **Version handling**
   - Converters should handle multiple capture format versions
   - Add version detection if format changes

### Future Consideration: Option A

If we find that:
- Most users only use openadapt-capture as the source
- Conversion overhead becomes significant
- Schema has stabilized

Then we can revisit Option A (native output). The conversion layer makes this migration easier since we understand the mapping well.

---

## Text Demo Format (WAA Experiment)

For the WAA demo experiment, we have a **third format**: text-based demos that are included in LLM prompts.

```
DEMONSTRATION:
Goal: Turn off Night Shift

Step 1:
  [Screen: System Settings app open]
  [Action: CLICK(Night Shift button)]
  [Result: Night Shift panel opens]
```

### Conversion Chain for WAA

```
openadapt-capture recording
    ↓ (capture_to_episode)
Episode object
    ↓ (episode_to_text_demo)
Text demo for LLM prompt
```

### episode_to_text_demo() Implementation

This converter should:
1. Extract instruction from Episode
2. Format each step with screen description, action, result
3. Use screenshot descriptions from accessibility tree or window title
4. Format actions in our DSL syntax

---

## Action Items

1. **Document conversion layer** in `openadapt_ml/ingest/` README
2. **Add episode_to_text_demo()** for WAA experiment
3. **Test with real captures** once Windows recordings arrive
4. **Monitor pain points** to inform future architecture decisions

---

## Appendix: Format Comparison

| Field | openadapt-capture | Episode Schema | Text Demo |
|-------|-------------------|----------------|-----------|
| Task description | `goal` | `instruction` | `Goal:` line |
| Step index | implicit (array order) | `step_index` | `Step N:` |
| Screenshot | `screenshots/frame_*.png` | `observation.screenshot_path` | `[Screen: ...]` |
| Action type | `action_type` (string) | `action.type` (ActionType enum) | `[Action: TYPE(...)]` |
| Coordinates | `x`, `y` (pixels) | `normalized_coordinates` (0-1) | Included in action |
| Metadata | Various fields | `metadata` dict + `raw_*` | Not included |
