# Schema Consolidation Plan

## Schema Ownership Rule

> `openadapt_ml/schema/episode.py` is the only source of truth. Any feature that cannot be represented here is not allowed elsewhere.

## Overview

This document outlines the plan to consolidate from two schema modules to one:

- **DELETE**: `openadapt_ml/schemas/` (dataclass-based, legacy)
- **KEEP**: `openadapt_ml/schema/` (Pydantic-based, canonical)

## Current State

### Two Schema Modules

| Module | Implementation | Field Names | Features |
|--------|---------------|-------------|----------|
| `schemas.sessions` | dataclass | `id`, `goal`, `t`, `image_path`, `x`/`y` | None |
| `schema.episode` | Pydantic | `episode_id`, `instruction`, `step_index`, `screenshot_path`, `coordinates`/`normalized_coordinates` | Validation, JSON Schema |

### Dependency Analysis

**22 files** import from `openadapt_ml.schemas`:

#### Core Modules (16 files)

| File | Imports | Usage |
|------|---------|-------|
| `openadapt_ml/ingest/loader.py` | Episode, Action, Observation, Step, validate_episodes, summarize_episodes | Load episodes from JSON |
| `openadapt_ml/ingest/capture.py` | Episode, Action, Observation, Session, Step | Convert captures to episodes |
| `openadapt_ml/ingest/synthetic.py` | Episode, Action, Observation, Session, Step | Generate synthetic episodes |
| `openadapt_ml/training/trainer.py` | Episode | Training loop |
| `openadapt_ml/export/parquet.py` | Episode, Action, Observation, Step | Export to Parquet |
| `openadapt_ml/retrieval/retriever.py` | Episode | Demo retrieval |
| `openadapt_ml/retrieval/index.py` | Episode | Demo indexing |
| `openadapt_ml/datasets/next_action.py` | Episode, Action, Step | Dataset building |
| `openadapt_ml/evals/grounding.py` | Episode | Grounding evaluation |
| `openadapt_ml/evals/trajectory_matching.py` | Episode, Action | Trajectory comparison |
| `openadapt_ml/runtime/policy.py` | Action | Runtime policy |
| `openadapt_ml/benchmarks/agent.py` | Action | Benchmark agent |
| `openadapt_ml/scripts/compare.py` | Episode, Step | Comparison script |
| `openadapt_ml/experiments/demo_prompt/format_demo.py` | Episode, Action, Step | Demo formatting |
| `openadapt_ml/schemas/__init__.py` | (re-exports) | Module interface |
| `openadapt_ml/schemas/validation.py` | Episode, Action, Observation, Session, Step | Validation logic |

#### Tests (4 files)

| File | Imports |
|------|---------|
| `tests/test_action_parsing.py` | Action |
| `tests/test_parquet_export.py` | Episode, Action, Observation, Step |
| `tests/test_retrieval.py` | Episode, Action, Observation, Step |
| `test_retrieval.py` (root) | Episode, Action, Observation, Step |

#### Examples (2 files)

| File | Imports |
|------|---------|
| `examples/demo_retrieval_example.py` | Episode, Action, Observation, Step |
| `examples/train_from_json.py` | validate_episodes, summarize_episodes |

#### Documentation (4 files with code examples)

- `RETRIEVAL_QUICKSTART.md`
- `examples/README.md`
- `docs/demo_retrieval_design.md`
- `docs/enterprise_integration.md`

## Field Mapping

| Old (`schemas.sessions`) | New (`schema.episode`) | Notes |
|-------------------------|------------------------|-------|
| `Episode.id` | `Episode.episode_id` | Rename |
| `Episode.goal` | `Episode.instruction` | Rename |
| `Episode.steps` | `Episode.steps` | Same |
| `Episode.summary` | `Episode.metadata["summary"]` | Move to metadata |
| `Episode.success` | `Episode.success` | Same |
| `Episode.workflow_id` | `Episode.metadata["workflow_id"]` | Move to metadata |
| `Step.t` | `Step.step_index` + `Step.timestamp` | Split: index (int) + timestamp (float) |
| `Step.observation` | `Step.observation` | Same (nested structure differs) |
| `Step.action` | `Step.action` | Same (nested structure differs) |
| `Step.thought` | `Step.reasoning` | Rename |
| `Observation.image_path` | `Observation.screenshot_path` | Rename |
| `Observation.meta` | `Observation.raw` | Rename |
| `Observation.accessibility_tree` | `Observation.a11y_tree` | Rename |
| `Observation.dom_html` | `Observation.dom` | Rename |
| `Action.x`, `Action.y` | `Action.normalized_coordinates` | Combine into tuple |
| `Action.end_x`, `Action.end_y` | `Action.normalized_end` | Combine into tuple |
| `Action.type` (str) | `Action.type` (ActionType enum) | Type change |
| `Action.element_index` | `Action.element.element_id` (str) | Nested + type change |
| `Action.bbox` | `Action.element.bounds` | Nested |
| `Action.raw` | `Action.raw` | Same |
| `Session` | (removed) | Container not needed |

## Converters as the ONLY Legacy Boundary

- `openadapt_ml/schema/converters.py` is the ONLY place for legacy format handling
- All legacy JSON to canonical Episode conversion happens there
- Everywhere else assumes canonical Episode only
- **Rule**: If a function accepts Episode, it never branches on "old vs new"

## Temporal Truth Rule

- `step_index`: required, contiguous, zero-based, validated
- `timestamp`: optional, float, monotonic if present, never used for ordering

## Migration Strategy

### Phase 1: Prepare New Schema (DONE)

- [x] Create `openadapt_ml/schema/episode.py` with Pydantic models
- [x] Create `openadapt_ml/schema/converters.py` with format converters
- [x] Export JSON Schema to `docs/schema/episode.schema.json`
- [x] Add documentation in `docs/schema/README.md`

### Phase 2: Update All Modules, Tests, and Docs (Single PR)

This phase consolidates all migration work into a single focused PR.

#### Add Validation to New Schema

Port validation logic from `schemas/validation.py` to Pydantic validators:

```python
# schema/episode.py
class Episode(BaseModel):
    @field_validator("steps")
    @classmethod
    def validate_steps_not_empty(cls, v):
        if not v:
            raise ValueError("Episode must have at least one step")
        return v

    @model_validator(mode="after")
    def validate_step_indices(self):
        for i, step in enumerate(self.steps):
            if step.step_index != i:
                raise ValueError(f"Step index mismatch: expected {i}, got {step.step_index}")
        return self
```

#### Update Each Module

Update imports and field access in order of dependency:

**Core Data Structures (no dependencies)**

1. **`openadapt_ml/schema/episode.py`** - Already done

**Ingest Layer (depends on schema)**

2. **`openadapt_ml/ingest/loader.py`**
   ```python
   # Before
   from openadapt_ml.schemas.sessions import Episode, Action, Observation, Step

   # After
   from openadapt_ml.schema import Episode, Step, Action, Observation
   ```
   - Update JSON parsing to use Pydantic `.model_validate()`
   - Update field access: `ep.id` → `ep.episode_id`, `ep.goal` → `ep.instruction`

3. **`openadapt_ml/ingest/capture.py`** - Update Episode/Step construction

4. **`openadapt_ml/ingest/synthetic.py`** - Update Episode/Step construction

**Processing Layer (depends on ingest)**

5. **`openadapt_ml/datasets/next_action.py`** - Update field access

6. **`openadapt_ml/export/parquet.py`** - Update column names

7. **`openadapt_ml/retrieval/retriever.py`** - Update Episode usage

8. **`openadapt_ml/retrieval/index.py`** - Update Episode usage

**Training Layer (depends on ingest + processing)**

9. **`openadapt_ml/training/trainer.py`** - Update Episode usage

10. **`openadapt_ml/scripts/compare.py`** - Update field access

**Evaluation Layer (depends on all above)**

11. **`openadapt_ml/evals/grounding.py`** - Update Episode usage

12. **`openadapt_ml/evals/trajectory_matching.py`** - Update field access

**Runtime (standalone)**

13. **`openadapt_ml/runtime/policy.py`** - Update Action usage

14. **`openadapt_ml/benchmarks/agent.py`** - Update Action usage

**Experiments**

15. **`openadapt_ml/experiments/demo_prompt/format_demo.py`** - Update field access

#### Update Tests

16. **`tests/test_action_parsing.py`** - Update Action construction
17. **`tests/test_parquet_export.py`** - Update Episode construction
18. **`tests/test_retrieval.py`** - Update Episode construction
19. **`test_retrieval.py`** (root) - DELETE (duplicate of tests/test_retrieval.py)

Add new tests:
- `tests/test_schema_validation.py` - Test Pydantic validation
- `tests/test_schema_serialization.py` - Test JSON round-trip

#### Update Documentation

20. **`RETRIEVAL_QUICKSTART.md`** - Update import paths
21. **`examples/README.md`** - Update examples
22. **`examples/demo_retrieval_example.py`** - Update imports
23. **`examples/train_from_json.py`** - Update imports
24. **`docs/demo_retrieval_design.md`** - Update code examples
25. **`docs/enterprise_integration.md`** - Update code examples
26. **`docs/schema/README.md`** - Remove "internal format" section

### Phase 3: Delete Old Schema

27. Delete `openadapt_ml/schemas/sessions.py`
28. Delete `openadapt_ml/schemas/validation.py`
29. Delete `openadapt_ml/schemas/__init__.py`
30. Remove `openadapt_ml/schemas/` directory

No backward compatibility shim. Let imports break loudly.

## Testing Strategy

### Unit Tests

For each module update, verify:
1. Module imports successfully
2. Existing tests pass with updated imports
3. Field access works correctly

### Integration Tests

1. **End-to-end ingest**: Load JSON → Episode objects → validate
2. **Training pipeline**: Episodes → Dataset → Training loop
3. **Export pipeline**: Episodes → Parquet → reload

### Regression Tests

1. Load existing episode JSON files (both old and new format)
2. Verify all fields are correctly parsed
3. Verify validation catches expected errors

## Rollback Plan

If issues are discovered:

1. Git revert the consolidation commits
2. Re-add `openadapt_ml/schemas/` from git history
3. Document the issue for next attempt

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: Prepare New Schema | DONE |
| Phase 2: Single PR (modules, tests, docs) | 4-5 hours |
| Phase 3: Delete Old Schema | 10 minutes |

**Total: ~4-5 hours** of focused work (single PR approach)

## Open Questions

1. **Session class**: The old schema has `Session` (container for episodes). Do we need this in the new schema?
   - Recommendation: No, just use `list[Episode]`

2. **ActionType enum**: The old schema uses string literals. The new uses an enum. Should we keep string compatibility?
   - Recommendation: Yes, Pydantic handles this with `use_enum_values=True`

3. **Normalized vs pixel coordinates**: The old schema only has normalized. The new has both. Which is default?
   - Recommendation: Support both, prefer normalized for training (resolution-independent)

4. **Validation strictness**: Should validation be strict (raise errors) or lenient (warnings)?
   - **Decision**: Default to strict. No lenient mode exposed.
