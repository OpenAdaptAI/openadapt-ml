# Parquet Export Design

**Status**: Design
**Date**: December 2025

## Overview

This document describes the design for Parquet export support in OpenAdapt-ML. Parquet is supported as a **derived format** for analytics and governance, not as a primary schema.

---

## Design Principles

### Episode JSON Remains Canonical

| Format | Primary Use | When to Use |
|--------|-------------|-------------|
| **Episode JSON** | Interchange, validation, retrieval | Always (source of truth) |
| Parquet | Analytics, filtering, governance | Downstream analysis |

Parquet is a **projection** of Episode data, not a replacement.

### Why Not Make Parquet Primary?

1. **Trajectories are not tabular**: Nested variable-length step sequences don't map cleanly to rows
2. **Image handling ambiguity**: Paths vs. embedded bytes become unclear
3. **Step ordering becomes implicit**: Parquet doesn't enforce sequence semantics
4. **Episode boundaries weaken**: The strong grouping of steps within episodes is lost

---

## Use Cases for Parquet Export

### 1. Dataset Governance & Slicing

```sql
-- Include only verified segments
SELECT * FROM episodes
WHERE metadata->>'include' = 'true'
  AND metadata->>'failure_modes' IS NULL

-- Filter by workflow
SELECT * FROM episodes
WHERE workflow_id = 'expense_reporting_v2'
```

### 2. Analytics & Reporting

```sql
-- Action type distribution
SELECT action_type, COUNT(*)
FROM steps
GROUP BY action_type

-- Average trajectory length by goal
SELECT goal, AVG(step_count) as avg_steps
FROM episodes
GROUP BY goal
```

### 3. Pre-Training Filtering

Narrow down to clean data before materializing WebDataset shards:

```python
import duckdb

# Filter in DuckDB, then convert filtered results back to Episodes
filtered = duckdb.query("""
    SELECT * FROM 'episodes.parquet'
    WHERE quality_label = 'include'
      AND step_count > 3
""").df()
```

### 4. Enterprise Data Team Integration

Many enterprises require Parquet for:
- Data lineage tracking
- Governance compliance
- Lakehouse ingestion (Databricks, Snowflake)

---

## Parquet Schema

### Flattened Steps (Recommended)

One row per step, with episode-level fields repeated:

| Column | Type | Source |
|--------|------|--------|
| `episode_id` | STRING | Episode.id |
| `goal` | STRING | Episode.goal |
| `workflow_id` | STRING | Episode.workflow_id (nullable) |
| `step_index` | INT32 | Position in episode (0-indexed) |
| `timestamp` | FLOAT64 | Step.t |
| `action_type` | STRING | Step.action.type |
| `x` | FLOAT64 | Step.action.x (nullable) |
| `y` | FLOAT64 | Step.action.y (nullable) |
| `end_x` | FLOAT64 | Step.action.end_x (nullable) |
| `end_y` | FLOAT64 | Step.action.end_y (nullable) |
| `text` | STRING | Step.action.text (nullable) |
| `key` | STRING | Step.action.key (nullable) |
| `modifiers` | LIST[STRING] | Step.action.modifiers (nullable) |
| `scroll_direction` | STRING | Step.action.scroll_direction (nullable) |
| `image_path` | STRING | Step.observation.image_path |
| `window_title` | STRING | Step.observation.window_title (nullable) |
| `app_name` | STRING | Step.observation.app_name (nullable) |
| `url` | STRING | Step.observation.url (nullable) |
| `thought` | STRING | Step.thought (nullable, may be scrubbed via [openadapt-privacy](https://github.com/OpenAdaptAI/openadapt-privacy)) |
| `episode_metadata` | STRING (JSON) | Episode.metadata (serialized) |
| `step_metadata` | STRING (JSON) | Step.metadata (serialized) |

### Episode-Level Summary Table (Optional)

For queries that don't need step details:

| Column | Type | Source |
|--------|------|--------|
| `episode_id` | STRING | Episode.id |
| `goal` | STRING | Episode.goal |
| `workflow_id` | STRING | Episode.workflow_id |
| `step_count` | INT32 | len(Episode.steps) |
| `duration` | FLOAT64 | last_step.t - first_step.t |
| `success` | BOOLEAN | Episode.success |
| `first_action_type` | STRING | Episode.steps[0].action.type |
| `last_action_type` | STRING | Episode.steps[-1].action.type |
| `metadata` | STRING (JSON) | Episode.metadata |

---

## API Design

### Export Function

```python
from openadapt_ml.export import to_parquet

# Load episodes
episodes = load_episodes("workflow_exports/")

# Export to Parquet
to_parquet(
    episodes,
    output_path="episodes.parquet",
    flatten_steps=True,     # One row per step (default)
    include_summary=True,   # Also generate episodes_summary.parquet
)
```

### Import Function (Reconstruction)

```python
from openadapt_ml.export import from_parquet

# Load back to Episodes (lossy if metadata was complex)
episodes = from_parquet("episodes.parquet")
```

### CLI

```bash
# Export
uv run python -m openadapt_ml.export parquet \
  --input workflow_exports/ \
  --output episodes.parquet

# With summary table
uv run python -m openadapt_ml.export parquet \
  --input workflow_exports/ \
  --output episodes.parquet \
  --include-summary
```

---

## Implementation Notes

### Dependencies

```toml
# pyproject.toml - optional dependency group
[project.optional-dependencies]
parquet = ["pyarrow>=14.0.0"]
```

Users install with: `pip install openadapt-ml[parquet]`

### Metadata Handling

Episode and Step `metadata` fields are serialized as JSON strings:

```python
import json

row["episode_metadata"] = json.dumps(episode.metadata) if episode.metadata else None
row["step_metadata"] = json.dumps(step.metadata) if step.metadata else None
```

This preserves arbitrary metadata without requiring schema changes.

### Image Paths vs. Embedding

Parquet export stores **paths**, not image bytes:
- Keeps file size manageable
- Maintains compatibility with existing image directories
- Users who need embedded images can use WebDataset instead

### Roundtrip Fidelity

`from_parquet()` can reconstruct Episodes, but with caveats:
- Metadata is deserialized from JSON (objects become dicts)
- Step ordering is recovered from `step_index`
- Episode boundaries are recovered from `episode_id` grouping

Reconstruction is intended for inspection and debugging only, not as a replacement for Episode JSON as an interchange format.

For full fidelity, always keep Episode JSON as the source of truth.

---

## Documentation Updates

### Add to enterprise_integration.md

```markdown
## Parquet Export (Analytics)

For dataset governance and analytics, export Episodes to Parquet:

\`\`\`python
from openadapt_ml.export import to_parquet

episodes = load_episodes("workflow_exports/")
to_parquet(episodes, "episodes.parquet")
\`\`\`

Parquet exports are provided for analytics, slicing, and governance.
Episode JSON remains the canonical representation for training, retrieval, and replay.
```

---

## Not In Scope

The following are explicitly **not** supported via Parquet:

1. **Demo retrieval**: Use DemoIndex with Episode objects
2. **Training data loading**: Use Episode JSON or WebDataset
3. **Schema validation**: Use Episode JSON + Pydantic
4. **Real-time ingestion**: Use Episode JSON streaming

Parquet is for **batch analytics**, not runtime agent operations.

---

## Future Enhancements

These are not in v1 scope but may be added based on usage:

1. **Partition support**: `partition_by=["workflow_id"]` for faster queries on large datasets
2. **Selective export**: `filters={"include": True}` to export only matching episodes
3. **DuckDB integration**: `episodes.query("SELECT * WHERE ...")` for in-memory analytics

---

## File Locations

| File | Purpose |
|------|---------|
| `openadapt_ml/export/__init__.py` | Public API |
| `openadapt_ml/export/parquet.py` | Implementation |
| `tests/test_parquet_export.py` | Tests |
| `docs/parquet_export_design.md` | This document |
