# Workflow Segmentation System

The workflow segmentation system automatically extracts, deduplicates, and annotates reusable workflow episodes from GUI recordings. This creates a library of canonical workflows that can be used for:

- **Training data curation**: Identify high-quality demonstration episodes for fine-tuning
- **Demo retrieval**: Build libraries of workflows for demo-conditioned prompting
- **Workflow documentation**: Automatically generate step-by-step workflow guides
- **Deduplication**: Find similar workflows across recordings to build canonical definitions

## Architecture

The system uses a **4-stage pipeline**:

### Stage 1: Frame Description (VLM)
Converts screenshots + actions into semantic descriptions using Vision-Language Models.

**Input**: Recording directory with screenshots and action events
**Output**: `ActionTranscript` with frame-by-frame descriptions

**Example**:
```python
from openadapt_ml.segmentation import FrameDescriber

describer = FrameDescriber(model="gemini-2.0-flash")
transcript = describer.describe_recording("/path/to/recording")

# View as plain text
print(transcript.to_transcript_text())
# [00:00.0] User opens System Preferences from Apple menu
# [00:02.5] User clicks Display settings icon
# [00:05.1] User navigates to Night Shift tab
# ...
```

**Supported VLMs**:
- Gemini 2.0 Flash / Pro (recommended for speed)
- Claude Sonnet 4 / Haiku
- GPT-4o / GPT-4o-mini

**Features**:
- Automatic caching to avoid reprocessing frames
- Batch processing for API efficiency
- Extracts: application name, visible elements, screen context, action target, user intent

---

### Stage 2: Episode Extraction (LLM)
Identifies coherent workflow boundaries and extracts episodes using Large Language Models.

**Input**: `ActionTranscript` from Stage 1
**Output**: `EpisodeExtractionResult` with identified episodes

**Example**:
```python
from openadapt_ml.segmentation import SegmentExtractor

extractor = SegmentExtractor(
    model="gpt-4o",
    use_few_shot=True,  # Include examples in prompts
    min_segment_duration=2.0,  # Minimum episode length
    max_segment_duration=300.0  # Maximum episode length
)

result = extractor.extract_segments(transcript)

for episode in result.episodes:
    print(f"{episode.name}: {episode.start_time_formatted} - {episode.end_time_formatted}")
    print(f"  Steps: {', '.join(episode.step_summaries)}")
    print(f"  Confidence: {episode.boundary_confidence:.2f}")
```

**Output**:
- Episode name and description
- Precise start/end timestamps
- Step-by-step breakdown
- Prerequisites and outcomes
- Boundary confidence scores

**Supported LLMs**:
- GPT-4o / GPT-4o-mini (recommended)
- Claude Sonnet 4 / Haiku
- Gemini 2.0 Pro / Flash

**Features**:
- Few-shot prompting for better segmentation quality
- Hierarchical extraction (nested subtasks)
- Confidence-based filtering
- Manual boundary adjustment helpers

---

### Stage 3: Deduplication (Embeddings)
Finds and merges similar workflows across recordings using embedding similarity.

**Input**: List of `EpisodeExtractionResult` from multiple recordings
**Output**: `EpisodeLibrary` with canonical workflows

**Example**:
```python
from openadapt_ml.segmentation import WorkflowDeduplicator

dedup = WorkflowDeduplicator(
    threshold=0.85,  # Cosine similarity threshold (0.80-0.90 recommended)
    embedding_model="text-embedding-3-large",
    merge_strategy="centroid"  # or "longest", "first"
)

library = dedup.deduplicate(extraction_results)

print(f"Total episodes: {library.total_episodes_extracted}")
print(f"Unique workflows: {library.unique_episode_count}")
print(f"Deduplication ratio: {library.deduplication_ratio:.1%}")
```

**Features**:
- Semantic similarity using text embeddings (OpenAI API or local HuggingFace models)
- Agglomerative clustering with cosine similarity
- Multiple merge strategies (centroid, longest, first)
- Incremental library updates (add new recordings to existing libraries)

**Merge Strategies**:
- `centroid`: Use episode closest to cluster centroid (most representative)
- `longest`: Use episode with longest/most detailed description
- `first`: Use first encountered episode

---

### Stage 4: Annotation (VLM Quality Assessment)
Automatically annotates episodes for training data quality control.

**Input**: `EpisodeExtractionResult` + recording path
**Output**: `AnnotatedEpisodeLibrary` with gold/exclusion labels

**Example**:
```python
from openadapt_ml.segmentation import EpisodeAnnotator

annotator = EpisodeAnnotator(
    model="gemini-2.0-flash",
    lookahead_frames=10  # Analyze frames after episode to detect failures
)

library = annotator.annotate_extraction_result(
    extraction_result=result,
    recording_path="/path/to/recording"
)

print(f"Total episodes: {library.total_episodes}")
print(f"Recommended as gold: {library.gold_count}")
print(f"Pending human review: {library.total_episodes - library.verified_count}")

# Get gold episodes for export
gold_episodes = library.get_verified_gold_episodes()
```

**What it checks**:
- Boundary accuracy (are start/end frames correct?)
- Workflow completeness (did all steps execute successfully?)
- Failure detection:
  - Error dialogs or messages
  - Undo actions (Ctrl+Z, etc.)
  - Repeated attempts at same action
  - User navigating back or canceling
- Post-episode analysis (examines frames *after* episode ends for delayed failures)

**Output**:
- `is_gold`: Boolean recommendation for training data inclusion
- `confidence`: VLM confidence in assessment (0-1)
- `failure_signals`: List of detected issues
- `exclusion_reason`: Explanation if not gold
- `start_frame` / `end_frame`: Refined boundaries

**Human-in-the-loop review**:
```python
from openadapt_ml.segmentation import verify_annotation

# After reviewing an annotation
verified = verify_annotation(
    annotation=ann,
    is_gold=True,  # Human decision
    notes="Verified - workflow completed successfully",
    verified_by="reviewer_name"
)
```

---

## Complete Pipeline

Run all 4 stages together:

```python
from openadapt_ml.segmentation import SegmentationPipeline, PipelineConfig

config = PipelineConfig(
    vlm_model="gemini-2.0-flash",  # Stage 1
    llm_model="gpt-4o",  # Stage 2
    similarity_threshold=0.85,  # Stage 3
    use_local_embeddings=False,  # Use OpenAI embeddings
    cache_enabled=True
)

pipeline = SegmentationPipeline(config)

result = pipeline.run(
    recordings=[
        "/path/to/recording1",
        "/path/to/recording2"
    ],
    output_dir="segmentation_output",
    progress_callback=lambda stage, cur, tot: print(f"[{stage}] {cur}/{tot}")
)

print(f"Recordings processed: {result.recordings_processed}")
print(f"Total episodes: {result.total_episodes_extracted}")
print(f"Unique workflows: {result.unique_episodes}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

The pipeline automatically saves intermediate results:
- `{recording_id}_transcript.json` - Stage 1 output
- `{recording_id}_episodes.json` - Stage 2 output
- `episode_library.json` - Stage 3 output (final deduplicated library)

---

## CLI Usage

All stages have CLI commands:

### Describe (Stage 1)
```bash
# Generate frame descriptions
python -m openadapt_ml.segmentation.cli describe \
  --recording /path/to/recording \
  --model gemini-2.0-flash \
  --output transcript.json

# View as plain text
python -m openadapt_ml.segmentation.cli describe \
  --recording /path/to/recording \
  --format text
```

### Extract (Stage 2)
```bash
# Extract episodes from a recording
python -m openadapt_ml.segmentation.cli extract \
  --recording /path/to/recording \
  --model gpt-4o \
  --output episodes.json

# Or from existing transcript
python -m openadapt_ml.segmentation.cli extract \
  --transcript transcript.json \
  --model gpt-4o \
  --output episodes.json
```

### Deduplicate (Stage 3)
```bash
# Deduplicate across multiple recordings
python -m openadapt_ml.segmentation.cli deduplicate \
  recording1_episodes.json recording2_episodes.json \
  --threshold 0.85 \
  --output library.json

# Or from a directory
python -m openadapt_ml.segmentation.cli deduplicate \
  --input-dir segmentation_output/ \
  --threshold 0.85 \
  --output library.json
```

### Annotate (Stage 4)
```bash
# Auto-annotate episodes for quality control
python -m openadapt_ml.segmentation.cli annotate \
  --episodes recording1_episodes.json \
  --recording /path/to/recording1 \
  --model gemini-2.0-flash \
  --output annotated_library.json

# Review annotations interactively
python -m openadapt_ml.segmentation.cli review \
  --library annotated_library.json \
  --recording /path/to/recording1 \
  --reviewer your_name \
  --auto-approve-high-confidence  # Auto-approve confidence > 0.9

# Export gold episodes for fine-tuning
python -m openadapt_ml.segmentation.cli export-gold \
  annotated_library.json \
  --format jsonl \
  --output gold_episodes.jsonl \
  --include-screenshots
```

### Complete Pipeline (all stages)
```bash
python -m openadapt_ml.segmentation.cli pipeline \
  /path/to/recording1 /path/to/recording2 /path/to/recording3 \
  --vlm-model gemini-2.0-flash \
  --llm-model gpt-4o \
  --threshold 0.85 \
  --output segmentation_output/ \
  --save-intermediate \
  --verbose
```

### List Library Contents
```bash
python -m openadapt_ml.segmentation.cli list \
  --library library.json \
  --details
```

### Export Library
```bash
# Export as CSV, JSONL, or HTML
python -m openadapt_ml.segmentation.cli export \
  library.json \
  --format html \
  --output workflows.html
```

---

## Data Schemas

All schemas are defined using Pydantic in `openadapt_ml/segmentation/schemas.py`:

### `FrameDescription` (Stage 1 output)
```python
{
    "timestamp": 2.5,
    "formatted_time": "00:02.5",
    "visible_application": "System Preferences",
    "visible_elements": ["Night Shift toggle", "Schedule slider"],
    "screen_context": "Display settings panel with Night Shift tab active",
    "action_type": "click",
    "action_target": "Night Shift toggle",
    "action_value": None,
    "apparent_intent": "Enable Night Shift automatic scheduling",
    "confidence": 0.95,
    "frame_index": 5,
    "vlm_model": "gemini-2.0-flash"
}
```

### `Episode` (Stage 2 output)
```python
{
    "episode_id": "uuid-here",
    "name": "Configure Night Shift Schedule",
    "start_time": 0.0,
    "end_time": 12.5,
    "start_time_formatted": "00:00.0",
    "end_time_formatted": "00:12.5",
    "description": "Enable and configure Night Shift automatic scheduling...",
    "step_summaries": [
        "Open System Preferences",
        "Navigate to Display > Night Shift",
        "Enable Night Shift",
        "Set schedule 9 PM - 7 AM"
    ],
    "application": "System Preferences",
    "prerequisites": ["System Preferences must be accessible"],
    "outcomes": ["Night Shift enabled with custom schedule"],
    "boundary_confidence": 0.95,
    "coherence_score": 0.90,
    "recording_id": "recording1",
    "frame_indices": [0, 1, 2, 3, 4, 5]
}
```

### `CanonicalEpisode` (Stage 3 output)
```python
{
    "canonical_id": "uuid-here",
    "canonical_name": "Configure Night Shift Schedule",
    "canonical_description": "Enable and configure Night Shift...",
    "canonical_steps": ["Open System Preferences", "Navigate to Display > Night Shift", ...],
    "variant_names": ["Adjust Night Shift Settings", "Set up Night Shift"],
    "variant_descriptions": ["...", "..."],
    "source_recordings": ["recording1", "recording2"],
    "source_episode_ids": ["uuid1", "uuid2"],
    "occurrence_count": 3,
    "embedding": [0.123, -0.456, ...],
    "cluster_id": 0,
    "internal_similarity": 0.92
}
```

### `EpisodeAnnotation` (Stage 4 output)
```python
{
    "annotation_id": "uuid-here",
    "episode_id": "uuid-of-episode",
    "start_frame": 0,
    "end_frame": 5,
    "is_gold": True,
    "exclusion_reason": None,
    "confidence": 0.95,
    "human_verified": False,
    "notes": None,
    "failure_signals": [],
    "created_at": "2026-01-17T10:00:00",
    "verified_at": None,
    "verified_by": None
}
```

---

## Configuration

### API Keys

Set environment variables for VLM/LLM providers:

```bash
export GOOGLE_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-claude-key"
export OPENAI_API_KEY="your-openai-key"
```

### Caching

Frame descriptions are automatically cached to avoid reprocessing:

```python
# Cache location: ~/.openadapt/cache/descriptions/

# Clear cache for a specific recording
describer.clear_cache(recording_id="recording1")

# Disable caching
describer = FrameDescriber(cache_enabled=False)
```

### Local Embeddings (No API required)

Use local HuggingFace models instead of OpenAI embeddings:

```python
dedup = WorkflowDeduplicator(
    use_local_embeddings=True  # Uses intfloat/e5-large-v2
)

# Requires: pip install transformers torch
```

---

## Use Cases

### 1. Training Data Curation

Extract and filter high-quality episodes for fine-tuning:

```python
# Extract episodes from all recordings
results = []
for recording in recordings:
    transcript = describer.describe_recording(recording)
    result = extractor.extract_segments(transcript)
    results.append(result)

# Deduplicate to find unique workflows
library = dedup.deduplicate(results)

# Annotate for quality
annotator = EpisodeAnnotator()
for recording, result in zip(recordings, results):
    annotated = annotator.annotate_extraction_result(result, recording)

    # Human review
    for episode, annotation in annotated.get_pending_review():
        # Present to human for verification
        verified = verify_annotation(annotation, is_gold=True, verified_by="human")

# Export gold episodes
from openadapt_ml.segmentation import export_gold_episodes
export_gold_episodes(
    library=annotated_library,
    output_path="training_data.jsonl",
    format="jsonl"
)
```

### 2. Demo Retrieval Library

Build a searchable library of workflow demonstrations:

```python
# Build library from multiple recordings
library = pipeline.run(recordings, output_dir="demo_library").library

# Find similar workflows for retrieval
target_episode = Episode(...)  # Current task
similar = dedup.find_similar(target_episode, library, top_k=5)

for canonical, similarity in similar:
    print(f"{canonical.canonical_name}: {similarity:.2f}")
    print(f"  Found in: {canonical.source_recordings}")
    print(f"  Steps: {canonical.canonical_steps}")
```

### 3. Workflow Documentation

Generate documentation from recordings:

```python
result = pipeline.run(recordings, output_dir="docs")

# Export as HTML
from openadapt_ml.segmentation.cli import export
export(
    library=result.library,
    format="html",
    output="workflow_guide.html"
)
```

---

## Advanced Features

### Hierarchical Segmentation

Extract nested task/subtask structures:

```python
extractor = SegmentExtractor(hierarchical=True)
result = extractor.extract_segments(transcript)

for episode in result.episodes:
    if episode.child_episode_ids:
        print(f"{episode.name} contains {len(episode.child_episode_ids)} subtasks")
```

### Boundary Refinement

Manually adjust or automatically refine boundaries:

```python
# Automatic refinement
refined = extractor.refine_segment(segment, transcript)

# Manual adjustment
adjusted = extractor.adjust_boundary(
    segment,
    new_start=2.5,  # New start time
    new_end=15.0,   # New end time
    transcript=transcript
)
```

### Segment Merging

Merge adjacent segments that belong together:

```python
merged = extractor.merge_segments(
    segments=episodes,
    max_gap=2.0  # Max seconds between segments to merge
)
```

### Incremental Library Updates

Add new recordings to an existing library:

```python
# Load existing library
import json
library_data = json.loads(Path("library.json").read_text())
existing_library = EpisodeLibrary.model_validate(library_data)

# Add new recording
new_result = pipeline.run(
    ["new_recording"],
    existing_library=existing_library
)

# Library now contains both old and new workflows
```

---

## Integration with openadapt-capture

**Status**: Integration layer needed

The segmentation system currently expects recordings in one of these formats:

1. **openadapt-capture format** (preferred):
   - Directory with `metadata.json` and `events.json`
   - `screenshots/` subdirectory with numbered PNGs

2. **JSON format**:
   - Single JSON file with base64-encoded screenshots

3. **Directory format**:
   - Directory with numbered PNG files
   - Creates synthetic event data

**Required**: Create adapter to load from `capture.db` (SQLite format used by openadapt-capture).

See [Integration Requirements](#integration-requirements) section below for details.

---

## Next Steps & Recommendations

### P0 (High Priority)

1. **Create openadapt-capture adapter**
   - Read events from `capture.db` SQLite database
   - Convert to format expected by FrameDescriber
   - Location: `openadapt_ml/segmentation/adapters/capture_adapter.py`

2. **Add visualization generator**
   - Create annotated screenshots showing segment boundaries
   - Highlight key actions within segments
   - Generate comparison views (before/after deduplication)

3. **Integration tests**
   - Test full pipeline on real openadapt-capture recordings
   - Validate output quality
   - Benchmark performance (time, API costs)

### P1 (Medium Priority)

4. **Improve prompt engineering**
   - Refine few-shot examples based on real data
   - Add domain-specific examples (web, desktop, mobile)
   - Experiment with structured output formats (JSON schema)

5. **Cost optimization**
   - Implement frame sampling strategies (skip similar frames)
   - Add batch processing limits to control API costs
   - Support vision-only models (no text description needed)

6. **Quality metrics**
   - Add inter-annotator agreement metrics
   - Track segmentation quality over time
   - Benchmark against human annotations

### P2 (Nice to Have)

7. **Active learning**
   - Suggest most valuable recordings to annotate next
   - Identify edge cases that need human review
   - Adapt prompts based on human feedback

8. **Multi-modal features**
   - Incorporate audio transcripts (already captured)
   - Use OCR for better text extraction
   - Analyze cursor movement patterns

9. **Export formats**
   - HuggingFace datasets format
   - Parquet for large-scale storage
   - Demo-conditioning format for retrieval

---

## Integration Requirements

### openadapt-capture Adapter

The current recordings use `capture.db` (SQLite) but the segmentation system expects `events.json`. Create an adapter:

```python
# openadapt_ml/segmentation/adapters/capture_adapter.py

import sqlite3
import json
from pathlib import Path
from PIL import Image

class CaptureAdapter:
    """Adapter for openadapt-capture SQLite format."""

    def load_recording(self, capture_path: Path) -> tuple[list[Image.Image], list[dict]]:
        """Load recording from capture.db format.

        Args:
            capture_path: Path to recording directory with capture.db

        Returns:
            Tuple of (images, action_events)
        """
        db_path = capture_path / "capture.db"
        screenshots_dir = capture_path / "screenshots"

        # Connect to SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query events
        cursor.execute("""
            SELECT timestamp, type, data
            FROM events
            WHERE type IN ('click', 'type', 'scroll', 'key', 'move')
            ORDER BY timestamp
        """)

        images = []
        events = []

        for i, (timestamp, event_type, data_json) in enumerate(cursor.fetchall()):
            data = json.loads(data_json)

            # Find corresponding screenshot
            screenshot_path = self._find_screenshot(screenshots_dir, i)
            if screenshot_path:
                images.append(Image.open(screenshot_path))

                # Convert to expected format
                event = {
                    "timestamp": timestamp,
                    "frame_index": i,
                    "name": event_type,
                    "mouse_x": data.get("x"),
                    "mouse_y": data.get("y"),
                    "text": data.get("text"),
                }
                events.append(event)

        conn.close()
        return images, events

    def _find_screenshot(self, screenshots_dir: Path, frame_index: int) -> Path | None:
        """Find screenshot file for frame index."""
        # openadapt-capture uses format: capture_{id}_step_{n}.png
        matches = list(screenshots_dir.glob(f"*_step_{frame_index}.png"))
        return matches[0] if matches else None
```

**Integration**:

Update `FrameDescriber._load_recording()` to use the adapter:

```python
# In frame_describer.py

def _load_recording(self, recording_path: Path):
    # Check for capture.db
    if (recording_path / "capture.db").exists():
        from openadapt_ml.segmentation.adapters import CaptureAdapter
        adapter = CaptureAdapter()
        return adapter.load_recording(recording_path)

    # ... existing code for other formats
```

---

## Cost Estimates

Approximate API costs for a 30-second recording (~20 frames):

### Stage 1 (Frame Description)
- **Gemini 2.0 Flash**: $0.01 - $0.05 per recording
- **Claude Haiku**: $0.10 - $0.30 per recording
- **GPT-4o-mini**: $0.05 - $0.15 per recording

### Stage 2 (Episode Extraction)
- **GPT-4o**: $0.01 - $0.02 per recording
- **Claude Sonnet 4**: $0.02 - $0.05 per recording

### Stage 3 (Deduplication)
- **OpenAI text-embedding-3-large**: $0.001 per recording
- **Local embeddings**: Free (requires GPU for speed)

### Stage 4 (Annotation)
- **Gemini 2.0 Flash**: $0.02 - $0.10 per episode
- **GPT-4o-mini**: $0.05 - $0.15 per episode

**Total per recording**: ~$0.05 - $0.50 depending on model choices

**Recommendation**: Use Gemini 2.0 Flash for Stages 1 & 4, GPT-4o for Stage 2, local embeddings for Stage 3.

---

## Performance

Approximate processing times for a 30-second recording (~20 frames):

- **Stage 1 (Description)**: 10-30 seconds (with batching)
- **Stage 2 (Extraction)**: 5-15 seconds
- **Stage 3 (Deduplication)**: 1-5 seconds (per 100 episodes)
- **Stage 4 (Annotation)**: 10-20 seconds per episode

**Bottleneck**: VLM API calls (Stages 1 & 4). Use caching and batching to optimize.

---

## Troubleshooting

### "GOOGLE_API_KEY not set"
Set the API key: `export GOOGLE_API_KEY="your-key"`

### "Failed to load recording"
Check that the recording directory has the expected format (screenshots/ and events.json or capture.db)

### "No episodes extracted"
- Lower `min_segment_duration` if recordings are short
- Check `confidence_threshold` (try 0.5 instead of 0.7)
- Review Stage 1 transcript to ensure VLM descriptions are accurate

### "Deduplication not working"
- Lower `threshold` (try 0.75 instead of 0.85)
- Check that episode descriptions are sufficiently detailed
- Verify embeddings are being generated correctly

### "High API costs"
- Enable caching: `cache_enabled=True`
- Use faster/cheaper models (Gemini Flash, GPT-4o-mini)
- Reduce batch size to process fewer frames per call
- Use local embeddings for Stage 3

---

## References

- **Schemas**: `openadapt_ml/segmentation/schemas.py`
- **Frame Describer**: `openadapt_ml/segmentation/frame_describer.py`
- **Segment Extractor**: `openadapt_ml/segmentation/segment_extractor.py`
- **Deduplicator**: `openadapt_ml/segmentation/deduplicator.py`
- **Annotator**: `openadapt_ml/segmentation/annotator.py`
- **Pipeline**: `openadapt_ml/segmentation/pipeline.py`
- **CLI**: `openadapt_ml/segmentation/cli.py`

---

## Example: Complete Workflow

```python
from openadapt_ml.segmentation import (
    SegmentationPipeline,
    PipelineConfig,
    EpisodeAnnotator,
    export_gold_episodes
)

# Configure pipeline
config = PipelineConfig(
    vlm_model="gemini-2.0-flash",  # Fast and cheap for Stage 1
    llm_model="gpt-4o",  # Best quality for Stage 2
    similarity_threshold=0.85,
    use_local_embeddings=True,  # No API cost for Stage 3
    cache_enabled=True
)

# Run segmentation on multiple recordings
pipeline = SegmentationPipeline(config)
result = pipeline.run(
    recordings=[
        "/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift",
        "/Users/abrichr/oa/src/openadapt-capture/demo_new"
    ],
    output_dir="workflow_library",
    progress_callback=lambda stage, cur, tot: print(f"[{stage}] {cur}/{tot}")
)

print(f"\nExtraction complete!")
print(f"  Unique workflows: {result.unique_episodes}")
print(f"  Deduplication: {result.library.deduplication_ratio:.1%}")

# Annotate for quality (Stage 4)
annotator = EpisodeAnnotator(model="gemini-2.0-flash")

for recording, extraction in zip(
    ["/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift"],
    [result.extractions["turn-off-nightshift"]]
):
    annotated = annotator.annotate_extraction_result(extraction, recording)
    print(f"\nAnnotation: {annotated.gold_count}/{annotated.total_episodes} gold episodes")

# Export gold episodes for training
export_gold_episodes(
    library=annotated,
    output_path="gold_episodes.jsonl",
    format="jsonl"
)

print(f"\nWorkflow library saved to: workflow_library/episode_library.json")
```

---

## Contributing

To add support for new VLM/LLM providers:

1. Create a new backend class in `frame_describer.py` or `segment_extractor.py`
2. Implement the required methods (`describe_frame`, `describe_batch`, etc.)
3. Update `_create_backend()` to detect and instantiate your backend
4. Add to `SUPPORTED_MODELS` list

Example:

```python
class CustomVLMBackend(VLMBackend):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def describe_frame(self, image, action_context, system_prompt, user_prompt):
        # Your implementation here
        pass

    def describe_batch(self, images, action_contexts, system_prompt, user_prompt):
        # Your implementation here
        pass
```
