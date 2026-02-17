# VLM Annotation Pipeline Design

## Problem

Recorded demos produce raw coordinates: `CLICK(0.26, 0.96)`. This is useless for demo conditioning because:
- Coordinates don't transfer across screen states
- No description of what was clicked
- The agent can't match its current screen to the demo

The hand-written demos in `demos.py` work well but are fabricated — they describe UI that may not match reality.

We need a pipeline that takes recorded captures and produces structured text traces like the hand-written demos, grounded in actual screenshots.

## Literature-Informed Decision

Based on review of ShowUI-Aloha (60.1% OSWorld), LearnAct (+168% Gemini), Instruction Agent (Microsoft), and EchoTrail-GUI: **text-only demos at inference outperform or match multimodal demos**, with dramatically lower token cost and better retrieval compatibility.

The winning pattern: VLM pre-processes multimodal recordings into structured text traces offline. At inference, only the text trace + current live screenshot are passed to the agent.

## Design

### Input

A capture directory (e.g., `~/.openadapt/waa_demos/37e10fc4-.../`) containing `recording.db` + video.

### Output

A JSON file (`demo_trace.json`) with this structure:

```json
{
  "schema_version": "0.1",
  "task_id": "37e10fc4-...-wos",
  "instruction": "Turn off all notifications on Windows",
  "source": "recorded",
  "annotator": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929"
  },
  "recording_meta": {
    "platform": "win32",
    "screen_px": [1920, 1080],
    "capture_id": "abc123",
    "raw_action_count": 597,
    "coalesced_step_count": 42
  },
  "steps": [
    {
      "step_index": 0,
      "timestamp_ms": 1234,
      "observation": "Windows desktop with taskbar at bottom. Start button and search icon visible in center of taskbar. System tray with clock and notification icons on right.",
      "intent": "Open Windows Settings to access notification settings",
      "action": "CLICK(Start button in center of taskbar)",
      "action_raw": "CLICK(0.500, 0.979)",
      "action_px": [960, 1057],
      "result_observation": "Start menu opened showing pinned apps and search bar at top.",
      "expected_result": "Start menu opens showing pinned apps and search bar"
    }
  ]
}
```

This format is based on ShowUI-Aloha (Observation/Think/Action/Expectation) with grounded `result_observation` from the next screenshot.

### Why This Format

| Field | Purpose | Based On |
|-------|---------|----------|
| `observation` | Describe screen state so agent can match its current screen | ShowUI-Aloha |
| `intent` | Why this action (reasoning chain) | ShowUI-Aloha "Think" |
| `action` | Semantic action description (human-readable) | Hand-written demos, LearnAct |
| `action_raw` | Original normalized coordinates (for debugging) | Internal |
| `action_px` | Pixel coordinates (unambiguous with `screen_px`) | GPT review |
| `result_observation` | **Grounded** description of what actually changed (from next screenshot) | GPT review / Instruction Agent |
| `expected_result` | Forward-looking intent (what *should* happen) | ShowUI-Aloha "Expectation" |

`result_observation` is grounded in the actual next frame. `expected_result` is the VLM's prediction of what should happen. Having both lets us detect when the demo went off-script.

### Pipeline Architecture

```
capture dir
    │
    ▼
capture_to_episode()          # existing (ingest/capture.py)
    │
    ▼
Episode (steps with screenshots + raw coords)
    │
    ▼
coalesce_steps()              # NEW — filter/merge redundant steps
    │
    ▼
Episode (30-80 meaningful steps)
    │
    ▼
annotate_episode()            # NEW — calls VLM per step
    │                           sends: screenshot_before (with click marker)
    │                                  screenshot_after (next frame)
    │                                  raw action + previous annotation
    ▼
AnnotatedDemo (JSON)
    │
    ▼
validate_annotations()        # NEW — JSON parse + field checks + sanity
    │
    ▼
format_annotated_demo()       # NEW — format for prompt injection
    │
    ▼
Text demo string              # same shape as hand-written demos
```

### Step Coalescing (Pre-Annotation)

Raw recordings have ~500 actions for a 40s task. Most are noise. Before annotation:

```python
def coalesce_steps(episode: Episode) -> Episode:
    """Filter and merge steps to produce meaningful demo steps.

    Rules:
    1. Drop mouse moves (already excluded by capture.actions())
    2. Merge consecutive TYPE actions into one (same window)
    3. Merge consecutive SCROLLs into one with direction + count
    4. Drop key.up events (only keep key.down for hotkeys)
    5. Drop "focus clicks" — a CLICK immediately followed by TYPE
       in the same location gets merged into one step
    6. Drop stop-sequence keypresses (ctrl×3)
    """
```

Expected reduction: ~500 raw → ~30-80 coalesced steps.

### Click Marker Rendering

Before sending a screenshot to the VLM, render a visual marker at the click point:

```python
def render_click_marker(
    image: Image,
    x_norm: float,
    y_norm: float,
    radius: int = 15,
    color: str = "red",
    width: int = 3,
) -> Image:
    """Draw a crosshair + circle at the click point on a screenshot.

    This helps the VLM identify which element was clicked, especially
    when multiple targets are near the coordinates.
    """
```

The marker is only on the image sent to the annotation VLM — not stored permanently. This is the single highest-leverage change for annotation quality.

### New File: `openadapt_ml/experiments/demo_prompt/annotate.py`

```python
"""VLM annotation of recorded demos into structured text traces."""

def annotate_episode(
    episode: Episode,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    max_steps: int = 50,
    output_path: str | Path | None = None,
) -> AnnotatedDemo:
    """Annotate an Episode by calling a VLM for each step.

    For each step:
    1. Load screenshot (before) and next screenshot (after)
    2. Render click marker on before-screenshot at action coordinates
    3. Send both images + action info + previous step context to VLM
    4. Get back: observation, intent, semantic action, result, expected_result

    Args:
        episode: Episode from capture_to_episode(), ideally coalesced.
        provider: "anthropic" or "openai".
        model: Model override (default: provider's default).
        api_key: API key override.
        max_steps: Maximum steps to annotate.
        output_path: If set, save JSON to this path.

    Returns:
        AnnotatedDemo with structured text per step.
    """
```

### Annotation Prompt (Per Step)

The VLM receives:
1. **Screenshot BEFORE action** — with click marker drawn at action coordinates
2. **Screenshot AFTER action** — the next step's screenshot (when available)
3. The raw action string
4. Context: task instruction, step number, total steps
5. **Previous step's annotation** (for coherence)

```
You are annotating a human GUI demonstration for a task automation system.

Task: {instruction}
Step {n} of {total}.

The user performed: {raw_action}
A red marker on the BEFORE image shows where the user clicked/interacted.

{previous_context}

Describe this step. Be specific about UI element names, labels, and visual landmarks.

For the OBSERVATION, describe:
- The application/window name
- The current panel or page
- 3-6 key visible UI elements with their relative positions

For the ACTION, describe which element was interacted with (reference the red marker). Name the element by its visible label/text, not by coordinates.

For the RESULT, describe what actually changed between the BEFORE and AFTER images.

Respond in this exact JSON format:
{
  "observation": "...",
  "intent": "...",
  "action": "...",
  "result_observation": "...",
  "expected_result": "..."
}
```

When a previous step annotation exists, `{previous_context}` is:
```
Previous step: {prev.action}
Previous result: {prev.result_observation}
```

### Data Model

```python
@dataclass
class AnnotatedStep:
    step_index: int
    timestamp_ms: int | None
    observation: str           # VLM description of screen state (before action)
    intent: str                # Why this action
    action: str                # Semantic action (human-readable)
    action_raw: str            # Original raw action string
    action_px: list[int] | None  # Pixel coordinates [x, y]
    result_observation: str    # What actually changed (from after-screenshot)
    expected_result: str       # What should happen next

@dataclass
class AnnotatedDemo:
    schema_version: str        # "0.1"
    task_id: str | None
    instruction: str
    source: str                # "recorded"
    annotator: dict            # {"provider": ..., "model": ...}
    recording_meta: dict       # platform, screen_px, capture_id, counts
    steps: list[AnnotatedStep]

    def to_json(self) -> str: ...
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "AnnotatedDemo": ...
```

### Validation Pass

After annotation, run lightweight checks:

```python
def validate_annotations(demo: AnnotatedDemo) -> list[str]:
    """Check annotation quality. Returns list of warnings.

    Checks:
    - All fields non-empty
    - JSON parseable (already guaranteed by dataclass)
    - observation mentions an app/window name
    - action doesn't contain raw coordinates (should be semantic)
    - result_observation differs from observation (something changed)
    - No obvious platform mismatch (e.g., macOS terms for Windows recording)
    """
```

### Formatting for Prompt Injection

New function `format_annotated_demo()` produces text matching the hand-written demo format:

```python
def format_annotated_demo(demo: AnnotatedDemo, compact: bool = True) -> str:
    """Format AnnotatedDemo as text for prompt injection.

    If compact=True (default for injection), uses only observation (first sentence),
    action, and result:

        DEMONSTRATION:
        Goal: Turn off all notifications

        Step 1:
          [Screen: Windows Settings app, System page.]
          [Action: CLICK(Notifications option in left sidebar)]
          [Result: Notifications settings panel opens on right side.]

    If compact=False, includes full observation and intent (for debugging).
    """
```

The compact format keeps the demo short for the agent's context window. Full observation and intent are stored in JSON for retrieval/debugging.

### Integration with DemoConditionedAgent

`_build_task_demo_map()` currently loads from `demos.py`. We add a priority fallback:

```python
def _build_task_demo_map(self):
    # 1. Try annotated demos (from recorded captures) — highest quality
    for trace_path in ANNOTATED_DEMO_DIR.glob("*.json"):
        demo = AnnotatedDemo.load(trace_path)
        text = format_annotated_demo(demo)
        if demo.task_id:
            self._task_demo_map[demo.task_id] = text

    # 2. Fall back to hand-written demos (fabricated but usable)
    for task_num, task in TASKS.items():
        if task.task_id not in self._task_demo_map:
            demo = get_demo(task_num)
            if demo and "[PLACEHOLDER" not in demo:
                self._task_demo_map[task_num] = demo
                self._task_demo_map[task.task_id] = demo
```

DemoConditionedAgent's `act()` and `_build_sample()` need zero changes — they already inject demo text strings. We just produce better text.

### CLI

```bash
# Annotate a single capture
uv run python -m openadapt_ml.experiments.demo_prompt.annotate \
    /path/to/capture \
    --provider anthropic \
    --output demos/annotated/37e10fc4.json

# Annotate all captures in a directory
uv run python -m openadapt_ml.experiments.demo_prompt.annotate \
    /path/to/captures/ \
    --all \
    --output demos/annotated/

# Preview formatted demo text from an annotation JSON
uv run python -m openadapt_ml.experiments.demo_prompt.annotate \
    --preview demos/annotated/37e10fc4.json
```

### Cost Estimate

Per step: ~2 screenshots (before+after, ~765 tokens each at low detail) + ~300 text tokens = ~1830 input tokens. Output: ~200 tokens.

After coalescing, ~30-80 steps per recording instead of ~500.

| Recording | Raw Actions | Est. Coalesced Steps | Input Tokens | Output Tokens | Cost (Sonnet) |
|-----------|-------------|---------------------|--------------|---------------|---------------|
| notifications | 597 | ~40 | ~73K | ~8K | ~$0.25 |
| archive | 510 | ~35 | ~64K | ~7K | ~$0.22 |
| notepad | 473 | ~30 | ~55K | ~6K | ~$0.19 |
| **Total** | | **~105** | **~192K** | **~21K** | **~$0.66** |

Very cheap. Could use Opus for higher quality at ~10x cost (~$6.60 total), still negligible.

## Files to Create/Modify

| File | Change |
|------|--------|
| `openadapt_ml/experiments/demo_prompt/annotate.py` | **NEW** — Core annotation pipeline + CLI |
| `openadapt_ml/experiments/demo_prompt/format_demo.py` | Add `format_annotated_demo()` |
| `openadapt_ml/experiments/waa_demo/runner.py` | Update `_build_task_demo_map()` to prefer annotated demos |
| `openadapt_ml/experiments/waa_demo/demos.py` | Mark fabricated demos clearly |

## Non-Goals (for now)

- **Retrieval/embedding** — 3 demos, 10 tasks. Direct task_id mapping is fine.
- **Async/parallel VLM calls** — ~105 sequential calls ≈ 2-3 minutes. Not worth async complexity.
- **Batch API** — Not worth it for 3 recordings.
- **Image hash caching** — One-time annotation, not iterating prompt versions yet.
- **Confidence scores / bbox detection** — Add when we need to filter low-quality steps.
- **Fine-tuning on traces** — In-context conditioning first.
- **Accessibility tree** — Recordings don't have a11y data. If we re-record with RECORD_WINDOW_DATA enabled, the annotation prompt can incorporate it.

## Key Design Decisions

1. **Click markers on screenshots** — Highest-leverage change. A red circle at the click point removes ambiguity about which element was clicked, especially near dense UIs (toolbars, menus, system trays).

2. **Before+after frame pairs** — Grounds `result_observation` in what actually happened, not what the VLM guesses. Captures exactly what changed.

3. **Sequential context** — Passing previous step's annotation to the next step's prompt. Cheap, improves coherence, catches inconsistencies.

4. **Coalescing before annotation** — 500→40 steps means we annotate meaningful actions, not noise. Also dramatically reduces cost and improves trace quality.

5. **Compact vs full format** — Full annotations stored in JSON; compact subset injected into agent prompt. Full data available for debugging and future retrieval.

6. **Coordinates are full-screen normalized** — Our recordings normalize to monitor dimensions (1920×1080), not active window. `action_px` + `recording_meta.screen_px` make this unambiguous.

## Verification

1. Annotate all 3 recorded demos
2. Validate: all fields populated, no platform mismatches, actions are semantic
3. Compare annotated demo text quality vs hand-written demos
4. Run DemoConditionedAgent with annotated demos on WAA tasks #7 and #9
5. Compare: zero-shot vs hand-written demo vs annotated demo
