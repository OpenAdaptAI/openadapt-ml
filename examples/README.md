# Examples

Example scripts demonstrating openadapt-ml usage.

## Training from JSON Data

The standard workflow for using openadapt-ml:

```python
from openadapt_ml.ingest import load_episodes
from openadapt_ml.schemas import validate_episodes, summarize_episodes

# 1. Load your exported data
episodes = load_episodes("your_data/")

# 2. Validate against schema
warnings = validate_episodes(episodes, check_images=True)

# 3. Check summary
print(summarize_episodes(episodes))
```

### Full Training Example

```bash
# Validate data only
python examples/train_from_json.py --data your_data/ --validate-only

# Train a model
python examples/train_from_json.py \
  --data your_data/ \
  --output results/ \
  --config configs/qwen3vl_capture.yaml
```

## Data Format

Episodes should be JSON files following the openadapt-ml schema.
See `sample_data.json` for the expected format.

### Required Fields

**Episode:**
- `id` (string): Unique identifier
- `goal` (string): Task description
- `steps` (array): List of Step objects

**Step:**
- `t` (float): Timestamp in seconds
- `observation` (object): GUI state
- `action` (object): Action taken

**Observation:**
- `image_path` (string): Path to screenshot

**Action:**
- `type` (string): One of: click, double_click, right_click, drag, scroll, type, key, done
- `x`, `y` (float, 0-1): Normalized coordinates (for click/drag)
- `text` (string): Text content (for type action)
- `key` (string): Key name (for key action)

### Optional Fields

See `openadapt_ml/schemas/sessions.py` for full schema with all optional fields.
