"""VLM annotation of recorded demos into structured text traces.

Takes a recorded capture, converts to Episode, coalesces steps, then calls a VLM
to annotate each step with semantic descriptions grounded in screenshots.

Usage:
    # Annotate a single capture
    uv run python -m openadapt_ml.experiments.demo_prompt.annotate \\
        annotate /path/to/capture --output traces/37e10fc4.json

    # Annotate all captures in a directory
    uv run python -m openadapt_ml.experiments.demo_prompt.annotate \\
        annotate-all /path/to/captures/ --output traces/

    # Preview formatted demo text from an annotation JSON
    uv run python -m openadapt_ml.experiments.demo_prompt.annotate \\
        preview traces/37e10fc4.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from openadapt_ml.experiments.demo_prompt.format_demo import format_action
from openadapt_ml.schema import ActionType, Episode, Step

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "0.1"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedStep:
    """A single annotated step in a demo trace."""

    step_index: int
    timestamp_ms: int | None
    observation: str
    intent: str
    action: str
    action_raw: str
    action_px: list[int] | None
    result_observation: str
    expected_result: str


@dataclass
class AnnotatedDemo:
    """A fully annotated demo trace, produced by VLM annotation."""

    schema_version: str
    task_id: str | None
    instruction: str
    source: str  # "recorded"
    annotator: dict[str, str]
    recording_meta: dict[str, Any]
    steps: list[AnnotatedStep]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info(f"Saved annotated demo to {path}")

    @classmethod
    def load(cls, path: str | Path) -> AnnotatedDemo:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        steps = [AnnotatedStep(**s) for s in data.pop("steps")]
        return cls(**data, steps=steps)


# ---------------------------------------------------------------------------
# Step coalescing
# ---------------------------------------------------------------------------

# Keys that are part of the stop sequence (ctrl×3)
_STOP_KEYS = {"ctrl", "ctrl_l", "ctrl_r", "Control_L", "Control_R"}


def coalesce_steps(episode: Episode) -> Episode:
    """Filter and merge Episode steps to produce meaningful demo steps.

    Rules:
    1. Drop DONE/FAIL terminal actions (re-added at end if needed)
    2. Drop KEY actions that are part of the stop sequence (ctrl×3)
    3. Merge consecutive TYPE actions into one combined text
    4. Merge consecutive SCROLL actions with same direction into one
    5. Drop "focus clicks" — CLICK immediately followed by TYPE at similar
       position gets merged (click absorbed into TYPE step)

    Returns:
        New Episode with coalesced steps, re-indexed.
    """
    steps = [s for s in episode.steps if s.action.type not in (ActionType.DONE, ActionType.FAIL)]

    # Filter stop-sequence keypresses
    steps = _filter_stop_sequence_keys(steps)

    # Merge consecutive TYPE actions
    steps = _merge_consecutive_types(steps)

    # Merge consecutive SCROLL actions
    steps = _merge_consecutive_scrolls(steps)

    # Drop focus clicks before TYPE
    steps = _drop_focus_clicks(steps)

    # Re-index
    reindexed = []
    for i, step in enumerate(steps):
        reindexed.append(step.model_copy(update={"step_index": i}))

    return episode.model_copy(update={"steps": reindexed})


def _filter_stop_sequence_keys(steps: list[Step]) -> list[Step]:
    """Drop KEY actions that look like stop-sequence presses (ctrl×3)."""
    result = []
    for step in steps:
        if step.action.type == ActionType.KEY:
            key = (step.action.key or "").lower()
            # Also check raw keys
            raw_keys = (step.action.raw or {}).get("keys", [])
            raw_key_names = {k.lower() for k in raw_keys} if raw_keys else set()
            if key in _STOP_KEYS or raw_key_names & _STOP_KEYS:
                continue
        result.append(step)
    return result


def _merge_consecutive_types(steps: list[Step]) -> list[Step]:
    """Merge consecutive TYPE actions into one with combined text."""
    if not steps:
        return steps

    result: list[Step] = []
    i = 0
    while i < len(steps):
        step = steps[i]
        if step.action.type != ActionType.TYPE:
            result.append(step)
            i += 1
            continue

        # Accumulate consecutive TYPE actions
        combined_text = step.action.text or ""
        j = i + 1
        while j < len(steps) and steps[j].action.type == ActionType.TYPE:
            combined_text += steps[j].action.text or ""
            j += 1

        if j > i + 1:
            # Create merged step using first step's observation, combined text
            merged_action = step.action.model_copy(update={"text": combined_text})
            merged_step = step.model_copy(update={"action": merged_action})
            result.append(merged_step)
        else:
            result.append(step)
        i = j

    return result


def _merge_consecutive_scrolls(steps: list[Step]) -> list[Step]:
    """Merge consecutive SCROLL actions with same direction."""
    if not steps:
        return steps

    result: list[Step] = []
    i = 0
    while i < len(steps):
        step = steps[i]
        if step.action.type != ActionType.SCROLL:
            result.append(step)
            i += 1
            continue

        direction = step.action.scroll_direction
        count = 1
        j = i + 1
        while (
            j < len(steps)
            and steps[j].action.type == ActionType.SCROLL
            and steps[j].action.scroll_direction == direction
        ):
            count += 1
            j += 1

        if count > 1:
            # Store count in raw for the annotation prompt
            raw = dict(step.action.raw or {})
            raw["scroll_count"] = count
            merged_action = step.action.model_copy(update={"raw": raw})
            merged_step = step.model_copy(update={"action": merged_action})
            result.append(merged_step)
        else:
            result.append(step)
        i = j

    return result


def _drop_focus_clicks(steps: list[Step]) -> list[Step]:
    """Drop CLICK immediately before TYPE at a similar position.

    A click followed by typing in the same area is a "focus click" — the user
    clicked to focus an input field, then typed. We merge by keeping only the
    TYPE step (which uses the CLICK's screenshot as its observation).
    """
    if len(steps) < 2:
        return steps

    result: list[Step] = []
    skip_next = False

    for i in range(len(steps) - 1):
        if skip_next:
            skip_next = False
            continue

        current = steps[i]
        nxt = steps[i + 1]

        if (
            current.action.type == ActionType.CLICK
            and nxt.action.type == ActionType.TYPE
        ):
            # Use current step's observation (screenshot before click) with next step's action
            merged = nxt.model_copy(
                update={"observation": current.observation}
            )
            result.append(merged)
            skip_next = True
        else:
            result.append(current)

    # Handle last step
    if not skip_next:
        result.append(steps[-1])

    return result


# ---------------------------------------------------------------------------
# Click marker rendering
# ---------------------------------------------------------------------------


def render_click_marker(
    image: Image.Image,
    x_norm: float,
    y_norm: float,
    radius: int = 20,
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
) -> Image.Image:
    """Draw a crosshair + circle at the click point on a screenshot.

    Args:
        image: PIL Image (not modified in place).
        x_norm: Normalized X coordinate (0.0-1.0).
        y_norm: Normalized Y coordinate (0.0-1.0).
        radius: Radius of the circle in pixels.
        color: RGB color tuple.
        width: Line width.

    Returns:
        New PIL Image with marker drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = int(x_norm * w)
    cy = int(y_norm * h)

    # Circle
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=color,
        width=width,
    )
    # Crosshair lines (extending beyond circle)
    arm = radius + 8
    draw.line([cx - arm, cy, cx + arm, cy], fill=color, width=width)
    draw.line([cx, cy - arm, cx, cy + arm], fill=color, width=width)

    return img


# ---------------------------------------------------------------------------
# Core annotation
# ---------------------------------------------------------------------------

ANNOTATION_SYSTEM_PROMPT = """\
You are annotating a human GUI demonstration for a task automation system.
Your annotations will be used to guide an AI agent performing the same task on a different screen.
Be precise about UI element names, labels, and visual landmarks.
Always respond with valid JSON only — no markdown, no extra text."""

ANNOTATION_STEP_PROMPT = """\
Task: {instruction}
Step {step_num} of {total_steps}.

The user performed: {action_raw}
A red marker on the BEFORE image shows where the user clicked/interacted.

{previous_context}
Describe this step:

- OBSERVATION: Describe what is visible on the BEFORE image. Include:
  - Application/window name
  - Current panel, page, or dialog
  - 3-6 key visible UI elements with relative positions

- INTENT: Why is the user performing this action? (1 sentence)

- ACTION: Describe which element was interacted with. Name the element by its visible label/text, not by coordinates. Reference the red marker to identify the target.

- RESULT: Describe what actually changed between the BEFORE and AFTER images.{no_after_note}

- EXPECTED_RESULT: What should the screen look like after this action?

Respond with valid JSON only:
{{"observation": "...", "intent": "...", "action": "...", "result_observation": "...", "expected_result": "..."}}"""


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
    4. Parse JSON response into AnnotatedStep

    Args:
        episode: Episode from capture_to_episode(), ideally already coalesced.
        provider: "anthropic" or "openai".
        model: Model override (default: provider's default).
        api_key: API key override.
        max_steps: Maximum steps to annotate.
        output_path: If set, save JSON to this path.

    Returns:
        AnnotatedDemo with structured text per step.
    """
    from openadapt_ml.models.providers import get_provider

    prov = get_provider(provider)
    if api_key is None:
        api_key = prov.get_api_key()
    client = prov.create_client(api_key)
    model = model or prov.default_model

    steps_to_annotate = episode.steps[:max_steps]
    total = len(steps_to_annotate)
    annotated_steps: list[AnnotatedStep] = []
    prev_annotation: AnnotatedStep | None = None

    for i, step in enumerate(steps_to_annotate):
        logger.info(f"Annotating step {i + 1}/{total}")
        action_raw = format_action(step.action)

        # Load before screenshot
        img_before = _load_step_screenshot(step)
        if img_before is None:
            logger.warning(f"Step {i}: no screenshot, skipping")
            continue

        # Render click marker for mouse actions
        if step.action.normalized_coordinates:
            x_n, y_n = step.action.normalized_coordinates
            img_before_marked = render_click_marker(img_before, x_n, y_n)
        else:
            img_before_marked = img_before

        # Load after screenshot (next step's observation)
        img_after = None
        if i + 1 < len(steps_to_annotate):
            img_after = _load_step_screenshot(steps_to_annotate[i + 1])

        # Build previous context
        previous_context = ""
        if prev_annotation is not None:
            previous_context = (
                f"Previous step: {prev_annotation.action}\n"
                f"Previous result: {prev_annotation.result_observation}\n"
            )

        no_after_note = ""
        if img_after is None:
            no_after_note = " (No AFTER image available — describe expected result only.)"

        # Build prompt
        prompt_text = ANNOTATION_STEP_PROMPT.format(
            instruction=episode.instruction,
            step_num=i + 1,
            total_steps=total,
            action_raw=action_raw,
            previous_context=previous_context,
            no_after_note=no_after_note,
        )

        # Build content blocks
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": "BEFORE image (with red marker):"},
            prov.encode_image(img_before_marked),
        ]
        if img_after is not None:
            content.append({"type": "text", "text": "AFTER image:"})
            content.append(prov.encode_image(img_after))

        # Call VLM
        response = prov.send_message(
            client,
            model=model,
            system=ANNOTATION_SYSTEM_PROMPT,
            content=content,
            max_tokens=512,
            temperature=0.1,
        )

        # Parse response
        parsed = _parse_annotation_response(response)

        # Compute pixel coordinates
        action_px = None
        if step.action.normalized_coordinates:
            screen = episode.steps[0].observation.screen_size
            if screen:
                sw, sh = screen
                x_n, y_n = step.action.normalized_coordinates
                action_px = [int(x_n * sw), int(y_n * sh)]

        timestamp_ms = None
        if step.timestamp is not None:
            timestamp_ms = int(step.timestamp * 1000)

        annotated = AnnotatedStep(
            step_index=i,
            timestamp_ms=timestamp_ms,
            observation=parsed.get("observation", ""),
            intent=parsed.get("intent", ""),
            action=parsed.get("action", action_raw),
            action_raw=action_raw,
            action_px=action_px,
            result_observation=parsed.get("result_observation", ""),
            expected_result=parsed.get("expected_result", ""),
        )
        annotated_steps.append(annotated)
        prev_annotation = annotated

    # Build recording metadata
    screen_px = None
    if episode.steps and episode.steps[0].observation.screen_size:
        screen_px = list(episode.steps[0].observation.screen_size)

    demo = AnnotatedDemo(
        schema_version=SCHEMA_VERSION,
        task_id=episode.task_id,
        instruction=episode.instruction,
        source="recorded",
        annotator={"provider": provider, "model": model},
        recording_meta={
            "platform": episode.environment or "",
            "screen_px": screen_px,
            "capture_id": episode.episode_id,
            "raw_step_count": len(episode.steps),
            "annotated_step_count": len(annotated_steps),
        },
        steps=annotated_steps,
    )

    if output_path:
        demo.save(output_path)

    return demo


def _load_step_screenshot(step: Step) -> Image.Image | None:
    """Load screenshot from a Step's observation."""
    if step.observation and step.observation.screenshot_path:
        path = Path(step.observation.screenshot_path)
        if path.exists():
            return Image.open(path)
    return None


def _parse_annotation_response(response: str) -> dict[str, str]:
    """Parse VLM JSON response, tolerant of minor formatting issues."""
    text = response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    logger.warning(f"Failed to parse VLM response as JSON: {text[:200]}")
    return {
        "observation": text[:200] if text else "",
        "intent": "",
        "action": "",
        "result_observation": "",
        "expected_result": "",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_annotations(demo: AnnotatedDemo) -> list[str]:
    """Check annotation quality. Returns list of warnings.

    Checks:
    - All key fields non-empty
    - Action doesn't contain raw coordinates (should be semantic)
    - Result_observation differs from observation
    - No obvious platform mismatch
    """
    warnings: list[str] = []

    for step in demo.steps:
        prefix = f"Step {step.step_index}"

        if not step.observation:
            warnings.append(f"{prefix}: empty observation")
        if not step.intent:
            warnings.append(f"{prefix}: empty intent")
        if not step.action:
            warnings.append(f"{prefix}: empty action")
        if not step.result_observation and not step.expected_result:
            warnings.append(f"{prefix}: no result_observation or expected_result")

        # Check if action still contains raw coordinates
        import re
        if re.search(r"CLICK\(\s*0\.\d+\s*,\s*0\.\d+\s*\)", step.action):
            warnings.append(f"{prefix}: action contains raw coordinates: {step.action}")

        # Check for platform mismatches (Windows recording described with macOS terms)
        platform = (demo.recording_meta or {}).get("platform", "")
        if "win" in platform.lower():
            mac_terms = ["finder", "dock", "spotlight", "cmd+", "command+"]
            action_lower = step.action.lower() + " " + step.observation.lower()
            for term in mac_terms:
                if term in action_lower:
                    warnings.append(f"{prefix}: macOS term '{term}' in Windows recording")

    return warnings


# ---------------------------------------------------------------------------
# Formatting for prompt injection
# ---------------------------------------------------------------------------


def format_annotated_demo(demo: AnnotatedDemo, compact: bool = True) -> str:
    """Format AnnotatedDemo as text for prompt injection.

    If compact=True (default), uses brief observation, action, and result:

        DEMONSTRATION:
        Goal: Turn off all notifications

        Step 1:
          [Screen: Windows Settings app, Notifications page.]
          [Action: CLICK(Notifications toggle to turn off)]
          [Result: Notifications toggle switched to Off.]

    If compact=False, includes full observation and intent.

    Args:
        demo: AnnotatedDemo to format.
        compact: If True, use compact format for agent prompt.

    Returns:
        Formatted demo string.
    """
    lines = [
        "DEMONSTRATION:",
        f"Goal: {demo.instruction}",
        "",
    ]

    for step in demo.steps:
        lines.append(f"Step {step.step_index + 1}:")

        if compact:
            # Use first sentence of observation for compactness
            obs = _first_sentence(step.observation)
            lines.append(f"  [Screen: {obs}]")
        else:
            lines.append(f"  [Screen: {step.observation}]")
            lines.append(f"  [Intent: {step.intent}]")

        lines.append(f"  [Action: {step.action}]")

        # Prefer result_observation (grounded), fall back to expected_result
        result = step.result_observation or step.expected_result
        if result:
            lines.append(f"  [Result: {result}]")

        lines.append("")

    return "\n".join(lines)


def _first_sentence(text: str) -> str:
    """Extract first sentence from text."""
    if not text:
        return ""
    # Split on period followed by space or end
    for i, ch in enumerate(text):
        if ch == "." and (i + 1 >= len(text) or text[i + 1] == " "):
            return text[: i + 1]
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_annotate(
    capture_path: str,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    output: str | None = None,
    max_steps: int = 50,
    instruction: str | None = None,
    task_id: str | None = None,
) -> None:
    """Annotate a single capture directory.

    Args:
        capture_path: Path to capture directory with recording.db.
        provider: VLM provider ("anthropic" or "openai").
        model: Model override.
        api_key: API key override.
        output: Output JSON path. Default: <capture_path>/demo_trace.json.
        max_steps: Maximum steps to annotate.
        instruction: Override task instruction.
        task_id: Override task ID. Default: directory name of capture_path.
    """
    from openadapt_ml.ingest.capture import capture_to_episode

    capture_path = Path(capture_path)
    if output is None:
        output = str(capture_path / "demo_trace.json")

    print(f"Loading capture from {capture_path}...")
    episode = capture_to_episode(capture_path, instruction=instruction)
    print(f"  Raw steps: {len(episode.steps)}")

    # Use task_id from: explicit param > episode metadata > directory name
    resolved_task_id = task_id or episode.task_id or capture_path.name
    if episode.task_id != resolved_task_id:
        episode = episode.model_copy(update={"task_id": resolved_task_id})
    print(f"  task_id: {resolved_task_id}")

    print("Coalescing steps...")
    episode = coalesce_steps(episode)
    print(f"  Coalesced steps: {len(episode.steps)}")

    print(f"Annotating with {provider} (max {max_steps} steps)...")
    demo = annotate_episode(
        episode,
        provider=provider,
        model=model,
        api_key=api_key,
        max_steps=max_steps,
        output_path=output,
    )

    # Validate
    warnings = validate_annotations(demo)
    if warnings:
        print(f"\nValidation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nValidation: all checks passed")

    print(f"\nAnnotated {len(demo.steps)} steps -> {output}")

    # Show compact preview
    print("\n" + "=" * 60)
    print("PREVIEW (compact format for agent prompt):")
    print("=" * 60)
    print(format_annotated_demo(demo, compact=True))


def cmd_annotate_all(
    captures_dir: str,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    output: str | None = None,
    max_steps: int = 50,
) -> None:
    """Annotate all captures in a directory.

    Looks for subdirectories containing recording.db.

    Args:
        captures_dir: Directory containing capture subdirectories.
        provider: VLM provider.
        model: Model override.
        api_key: API key override.
        output: Output directory for JSON files. Default: captures_dir.
        max_steps: Maximum steps per capture.
    """
    captures_dir = Path(captures_dir)
    output_dir = Path(output) if output else captures_dir

    # Find all recording.db files
    db_files = list(captures_dir.glob("**/recording.db"))
    if not db_files:
        print(f"No captures found in {captures_dir}")
        return

    print(f"Found {len(db_files)} captures")

    for db_path in db_files:
        capture_path = db_path.parent
        out_path = output_dir / f"{capture_path.name}.json"
        print(f"\n{'='*60}")
        print(f"Annotating: {capture_path.name}")
        print(f"{'='*60}")

        try:
            cmd_annotate(
                str(capture_path),
                provider=provider,
                model=model,
                api_key=api_key,
                output=str(out_path),
                max_steps=max_steps,
            )
        except Exception as e:
            print(f"  ERROR: {e}")


def cmd_preview(path: str, compact: bool = True) -> None:
    """Preview formatted demo text from an annotation JSON.

    Args:
        path: Path to annotated demo JSON file.
        compact: Use compact format (default: True).
    """
    demo = AnnotatedDemo.load(path)
    print(format_annotated_demo(demo, compact=compact))

    warnings = validate_annotations(demo)
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")


def main() -> None:
    """CLI entry point."""
    import fire

    fire.Fire({
        "annotate": cmd_annotate,
        "annotate-all": cmd_annotate_all,
        "preview": cmd_preview,
    })


if __name__ == "__main__":
    main()
