"""Convert annotated demo JSON files to SFT training format.

Reads annotated demos produced by the VLM annotation pipeline and converts
them into JSONL training data compatible with the internal SFT format
(matching build_next_action_sft_samples() output) for VLM fine-tuning.

Input format (annotated demo JSON):
    {
      "instruction": "Turn off notifications...",
      "steps": [{
        "step_index": 0,
        "observation": "The desktop shows...",
        "intent": "The user is trying to...",
        "action_raw": "CLICK(0.294, 0.532)",
        ...
      }]
    }

Output format (internal SFT JSONL):
    {
      "images": ["/path/to/screenshot.png"],
      "messages": [
        {"role": "system", "content": "You are a GUI agent..."},
        {"role": "user", "content": "<image>\\nInstruction: ...\\n...\\nOutput exactly one action."},
        {"role": "assistant", "content": "<think>...</think>\\nclick(x=294, y=532)"}
      ]
    }

The training conversation format is aligned with the inference prompt in
qwen3vl_agent.py. If you change the prompt format here, update the agent
to match (and vice versa).

Coordinates are converted from [0, 1] to [0, 1000] (Qwen3-VL format).

Usage:
    python -m openadapt_ml.training.convert_demos \\
        --demo-dir /path/to/annotated_demos \\
        --mapping /path/to/screenshot_mapping.json \\
        --output /path/to/output.jsonl

    # Create self-contained bundle for upload:
    python -m openadapt_ml.training.convert_demos \\
        --demo-dir /path/to/annotated_demos \\
        --mapping /path/to/screenshot_mapping.json \\
        --bundle /path/to/bundle_dir
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# System prompt for SFT training data
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a GUI agent. You observe screenshots of a desktop and output "
    "exactly one action per step. Use the following action format:\n"
    "click(x=<int>, y=<int>)\n"
    "double_click(x=<int>, y=<int>)\n"
    "right_click(x=<int>, y=<int>)\n"
    'type(text="<string>")\n'
    'press(keys=["<key1>", ...])\n'
    'scroll(direction="<up|down|left|right>", amount=<int>)\n'
    "drag(from_coord=[<x1>, <y1>], to_coord=[<x2>, <y2>])\n"
    "wait()\n"
    "finished()\n\n"
    "Coordinates are in [0, 1000] range where (0,0) is top-left and "
    "(1000,1000) is bottom-right."
)

# Required fields in annotated demo JSON
_REQUIRED_DEMO_FIELDS = {"instruction", "steps"}
_REQUIRED_STEP_FIELDS = {"step_index", "action_raw"}


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def _validate_coord(val: float, name: str, action_raw: str) -> float:
    """Validate that a coordinate is in [0, 1] range.

    Args:
        val: Coordinate value.
        name: Coordinate name for error message.
        action_raw: Original action string for context.

    Returns:
        The value, unchanged.
    """
    if not (0.0 <= val <= 1.0):
        warnings.warn(
            f"Coordinate {name}={val} outside [0, 1] in '{action_raw}'. "
            f"Expected normalized coordinates from format_action(). "
            f"If these are pixel coordinates, the output will be wrong.",
            stacklevel=3,
        )
    return val


def _parse_action_raw(action_raw: str) -> tuple[str, dict[str, Any]]:
    """Parse action_raw string into (action_type, params).

    Args:
        action_raw: Raw action string like "CLICK(0.294, 0.532)" or "TYPE(\"text\")".

    Returns:
        Tuple of (action_type, params_dict).
    """
    # DOUBLE_CLICK(x, y)
    m = re.match(
        r"DOUBLE_CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_raw, re.IGNORECASE
    )
    if m:
        x = _validate_coord(float(m.group(1)), "x", action_raw)
        y = _validate_coord(float(m.group(2)), "y", action_raw)
        return "double_click", {"x": x, "y": y}

    # RIGHT_CLICK(x, y)
    m = re.match(
        r"RIGHT_CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_raw, re.IGNORECASE
    )
    if m:
        x = _validate_coord(float(m.group(1)), "x", action_raw)
        y = _validate_coord(float(m.group(2)), "y", action_raw)
        return "right_click", {"x": x, "y": y}

    # CLICK(x, y)
    m = re.match(
        r"CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_raw, re.IGNORECASE
    )
    if m:
        x = _validate_coord(float(m.group(1)), "x", action_raw)
        y = _validate_coord(float(m.group(2)), "y", action_raw)
        return "click", {"x": x, "y": y}

    # TYPE("text") or TYPE(text)
    m = re.match(r"TYPE\s*\(\s*[\"']?(.*?)[\"']?\s*\)", action_raw, re.IGNORECASE)
    if m:
        return "type", {"text": m.group(1)}

    # KEY(key) or PRESS(key)
    m = re.match(r"(?:KEY|PRESS)\s*\(\s*(.+?)\s*\)", action_raw, re.IGNORECASE)
    if m:
        return "press", {"keys": m.group(1)}

    # SCROLL(direction)
    m = re.match(r"SCROLL\s*\(\s*(\w+)\s*\)", action_raw, re.IGNORECASE)
    if m:
        return "scroll", {"direction": m.group(1)}

    # DRAG(x1, y1, x2, y2)
    m = re.match(
        r"DRAG\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
        action_raw,
        re.IGNORECASE,
    )
    if m:
        fx = _validate_coord(float(m.group(1)), "from_x", action_raw)
        fy = _validate_coord(float(m.group(2)), "from_y", action_raw)
        tx = _validate_coord(float(m.group(3)), "to_x", action_raw)
        ty = _validate_coord(float(m.group(4)), "to_y", action_raw)
        return "drag", {"from_x": fx, "from_y": fy, "to_x": tx, "to_y": ty}

    # DONE / FINISHED / WAIT
    if re.match(r"(DONE|FINISHED|WAIT)\s*\(\s*\)", action_raw, re.IGNORECASE):
        return action_raw.split("(")[0].strip().lower(), {}

    return "unknown", {"raw": action_raw}


def _coord_01_to_1000(val: float) -> int:
    """Convert a [0, 1] coordinate to [0, 1000] (Qwen3-VL format).

    Args:
        val: Coordinate in [0, 1] range.

    Returns:
        Coordinate in [0, 1000] range.
    """
    return int(round(val * 1000))


def _format_action_qwen(action_type: str, params: dict[str, Any]) -> str:
    """Format action as Qwen3-VL action string with [0, 1000] coordinates.

    Args:
        action_type: Parsed action type.
        params: Action parameters.

    Returns:
        Formatted action string.
    """
    if action_type == "click":
        x = _coord_01_to_1000(params["x"])
        y = _coord_01_to_1000(params["y"])
        return f"click(x={x}, y={y})"

    if action_type == "double_click":
        x = _coord_01_to_1000(params["x"])
        y = _coord_01_to_1000(params["y"])
        return f"double_click(x={x}, y={y})"

    if action_type == "right_click":
        x = _coord_01_to_1000(params["x"])
        y = _coord_01_to_1000(params["y"])
        return f"right_click(x={x}, y={y})"

    if action_type == "type":
        text = params.get("text", "")
        return f'type(text="{text}")'

    if action_type == "press":
        keys = params.get("keys", "")
        if isinstance(keys, str):
            keys_list = [k.strip() for k in keys.split("+")]
        else:
            keys_list = keys
        keys_formatted = ", ".join(f'"{k}"' for k in keys_list)
        return f"press(keys=[{keys_formatted}])"

    if action_type == "scroll":
        direction = params.get("direction", "down")
        return f'scroll(direction="{direction}", amount=3)'

    if action_type == "drag":
        fx = _coord_01_to_1000(params["from_x"])
        fy = _coord_01_to_1000(params["from_y"])
        tx = _coord_01_to_1000(params["to_x"])
        ty = _coord_01_to_1000(params["to_y"])
        return f"drag(from_coord=[{fx}, {fy}], to_coord=[{tx}, {ty}])"

    if action_type == "wait":
        return "wait()"

    if action_type in ("done", "finished"):
        return "finished()"

    return f"# unknown: {params.get('raw', action_type)}"


def _build_think_block(step: dict[str, Any]) -> str:
    """Build a <think> block from observation + intent fields.

    Args:
        step: A step dict from the annotated demo.

    Returns:
        Think block string, or empty string if no useful content.
    """
    parts = []
    if step.get("observation"):
        parts.append(step["observation"])
    if step.get("intent"):
        parts.append(step["intent"])
    if not parts:
        return ""
    return "<think>\n" + " ".join(parts) + "\n</think>\n"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_demo(demo: dict[str, Any], path: Path) -> list[str]:
    """Validate annotated demo JSON structure.

    Args:
        demo: Parsed demo dict.
        path: Path to the demo file (for error messages).

    Returns:
        List of warning messages (empty if valid).
    """
    issues: list[str] = []

    for field in _REQUIRED_DEMO_FIELDS:
        if field not in demo:
            issues.append(f"{path.name}: missing required field '{field}'")

    if not demo.get("instruction"):
        issues.append(f"{path.name}: 'instruction' is empty")

    steps = demo.get("steps", [])
    if not steps:
        issues.append(f"{path.name}: 'steps' is empty")

    for i, step in enumerate(steps):
        for field in _REQUIRED_STEP_FIELDS:
            if field not in step:
                issues.append(f"{path.name}: step {i} missing required field '{field}'")
        if not step.get("action_raw"):
            issues.append(f"{path.name}: step {i} has empty 'action_raw'")

    return issues


# ---------------------------------------------------------------------------
# Step conversion
# ---------------------------------------------------------------------------


def convert_step(
    step: dict[str, Any],
    instruction: str,
    previous_actions: list[str],
    screenshot_path: str | None = None,
    include_thinking: bool = True,
) -> dict[str, Any] | None:
    """Convert a single annotated step to ms-swift SFT format.

    The user message format is aligned with qwen3vl_agent._build_prompt()
    so the model sees the same structure at training and inference time.

    Args:
        step: Step dict from annotated demo.
        instruction: Task instruction.
        previous_actions: List of previous action strings.
        screenshot_path: Path to the screenshot image for this step.
        include_thinking: Whether to include <think> blocks.

    Returns:
        ms-swift training sample dict, or None if step cannot be converted.
    """
    action_raw = step.get("action_raw", "")
    if not action_raw:
        return None

    action_type, params = _parse_action_raw(action_raw)

    # Build user content — aligned with qwen3vl_agent._build_prompt()
    user_parts = ["<image>"]
    user_parts.append(f"Instruction: {instruction}")

    if previous_actions:
        user_parts.append("")
        user_parts.append("Previous actions:")
        for i, act in enumerate(previous_actions):
            user_parts.append(f"  Step {i}: {act}")

    user_parts.append("")
    if include_thinking:
        user_parts.append(
            "First reason about what you see in <think>...</think> "
            "tags, then output exactly one action."
        )
    else:
        user_parts.append("Output exactly one action.")

    user_content = "\n".join(user_parts)

    # Build assistant content
    assistant_parts = []
    if include_thinking:
        think = _build_think_block(step)
        if think:
            assistant_parts.append(think)

    action_str = _format_action_qwen(action_type, params)
    assistant_parts.append(action_str)
    assistant_content = "".join(assistant_parts)

    sample: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }

    if screenshot_path:
        sample["images"] = [screenshot_path]

    return sample


# ---------------------------------------------------------------------------
# Screenshot resolution
# ---------------------------------------------------------------------------


def _resolve_screenshots_from_capture(
    task_id: str,
    captures_dir: Path,
    num_steps: int,
) -> dict[int, str]:
    """Derive screenshot paths by re-running coalesce on the capture.

    The annotated demo step indices correspond to coalesced episode steps,
    not raw capture steps. This function re-derives the mapping using the
    capture API, which is the only programmatic method that produces correct
    results.

    Args:
        task_id: Task ID (matches capture directory name).
        captures_dir: Parent directory containing capture directories.
        num_steps: Number of annotated steps expected.

    Returns:
        Dict mapping annotated step_index -> absolute screenshot path.
    """
    capture_dir = captures_dir / task_id
    if not capture_dir.is_dir():
        # Try case-insensitive match
        for d in captures_dir.iterdir():
            if d.is_dir() and d.name.lower() == task_id.lower():
                capture_dir = d
                break
        else:
            return {}

    screenshots_dir = capture_dir / "screenshots"
    if not screenshots_dir.is_dir():
        return {}

    try:
        from openadapt_ml.ingest.capture import capture_to_episode
        from openadapt_ml.experiments.demo_prompt.annotate import coalesce_steps

        episode = capture_to_episode(
            str(capture_dir),
            output_dir=str(screenshots_dir),
        )
        coalesced = coalesce_steps(episode)

        mapping = {}
        for step in coalesced.steps[:num_steps]:
            if step.observation and step.observation.screenshot_path:
                mapping[step.step_index] = step.observation.screenshot_path
        return mapping
    except Exception as e:
        print(
            f"  Warning: capture API unavailable for {task_id}: {e}",
            file=sys.stderr,
        )
        print(
            "  Use --mapping with a pre-computed screenshot_mapping.json instead.",
            file=sys.stderr,
        )
        return {}


# ---------------------------------------------------------------------------
# Demo conversion
# ---------------------------------------------------------------------------


def convert_demo(
    demo_path: Path,
    captures_dir: Path | None = None,
    screenshot_mapping: dict[str, dict[str, str]] | None = None,
    include_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Convert an annotated demo JSON file to ms-swift SFT samples.

    Screenshot resolution priority:
        1. Explicit --mapping JSON (most reliable, works everywhere)
        2. capture_to_episode + coalesce_steps (requires openadapt-capture)

    Args:
        demo_path: Path to annotated demo JSON file.
        captures_dir: Parent directory containing capture dirs.
        screenshot_mapping: Pre-computed mapping of task_id -> {step_index -> path}.
        include_thinking: Whether to include <think> blocks.

    Returns:
        List of ms-swift training sample dicts.
    """
    with open(demo_path) as f:
        demo = json.load(f)

    # Validate input
    issues = _validate_demo(demo, demo_path)
    if issues:
        for issue in issues:
            print(f"  Warning: {issue}", file=sys.stderr)

    instruction = demo.get("instruction", "")
    steps = demo.get("steps", [])
    task_id = demo.get("task_id", demo_path.stem)

    # Resolve screenshot mapping — priority: explicit mapping > capture API
    screenshot_map: dict[int, str] = {}
    if screenshot_mapping and task_id in screenshot_mapping:
        screenshot_map = {int(k): v for k, v in screenshot_mapping[task_id].items()}
    elif captures_dir:
        screenshot_map = _resolve_screenshots_from_capture(
            task_id,
            captures_dir,
            len(steps),
        )

    if not screenshot_map:
        print(
            f"  Warning: no screenshots resolved for {task_id}. "
            f"Training samples will lack images.",
            file=sys.stderr,
        )

    samples = []
    previous_actions: list[str] = []

    for step in steps:
        step_idx = step.get("step_index", len(previous_actions))
        screenshot_path = screenshot_map.get(step_idx)

        sample = convert_step(
            step=step,
            instruction=instruction,
            previous_actions=previous_actions,
            screenshot_path=screenshot_path,
            include_thinking=include_thinking,
        )

        if sample is not None:
            samples.append(sample)
            # Track action for history
            action_raw = step.get("action_raw", "")
            action_type, params = _parse_action_raw(action_raw)
            previous_actions.append(_format_action_qwen(action_type, params))

    return samples


def convert_all_demos(
    demo_dir: Path,
    captures_dir: Path | None = None,
    screenshot_mapping: dict[str, dict[str, str]] | None = None,
    output_path: Path | None = None,
    include_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Convert all annotated demo JSON files in a directory to SFT format.

    Args:
        demo_dir: Directory containing annotated demo JSON files.
        captures_dir: Parent directory containing capture dirs.
        screenshot_mapping: Pre-computed mapping JSON (task_id -> {step -> path}).
        output_path: Optional path to write JSONL output.
        include_thinking: Whether to include <think> blocks.

    Returns:
        List of all training samples.
    """
    all_samples = []

    demo_files = sorted(demo_dir.glob("*.json"))
    if not demo_files:
        print(f"No JSON files found in {demo_dir}", file=sys.stderr)
        return []

    for demo_path in demo_files:
        print(f"Converting {demo_path.name}...")
        samples = convert_demo(
            demo_path=demo_path,
            captures_dir=captures_dir,
            screenshot_mapping=screenshot_mapping,
            include_thinking=include_thinking,
        )
        n_with_img = sum(1 for s in samples if "images" in s)
        print(f"  -> {len(samples)} training samples ({n_with_img} with screenshots)")
        all_samples.extend(samples)

    n_total_img = sum(1 for s in all_samples if "images" in s)
    print(
        f"\nTotal: {len(all_samples)} training samples ({n_total_img} with screenshots) from {len(demo_files)} demos"
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Written to {output_path}")

    return all_samples


def create_bundle(
    samples: list[dict[str, Any]],
    bundle_dir: Path,
) -> Path:
    """Create a self-contained training bundle directory.

    Copies referenced screenshots into bundle_dir/images/ and rewrites
    image paths to relative so the bundle can be uploaded anywhere.

    Args:
        samples: Training samples (with absolute image paths).
        bundle_dir: Output directory for the bundle.

    Returns:
        Path to the written JSONL file inside the bundle.
    """
    images_dir = bundle_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    bundled_samples = []
    for sample in samples:
        sample = dict(sample)  # shallow copy
        if "images" in sample:
            new_paths = []
            for img_path in sample["images"]:
                src = Path(img_path)
                if src.exists():
                    dst = images_dir / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    new_paths.append(f"images/{src.name}")
                else:
                    print(
                        f"  Warning: image not found, skipping: {src}", file=sys.stderr
                    )
            sample["images"] = new_paths
        bundled_samples.append(sample)

    jsonl_path = bundle_dir / "training_data.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in bundled_samples:
            f.write(json.dumps(sample) + "\n")

    n_images = len(list(images_dir.iterdir()))
    print(f"Bundle created: {bundle_dir}")
    print(f"  {len(bundled_samples)} samples in training_data.jsonl")
    print(f"  {n_images} images in images/")
    return jsonl_path


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert annotated demo JSON files to ms-swift SFT format",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        required=True,
        help="Directory containing annotated demo JSON files",
    )
    parser.add_argument(
        "--captures-dir",
        type=str,
        help="Parent directory containing capture directories (requires openadapt-capture)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.jsonl",
        help="Output JSONL file path (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        help=(
            "Pre-computed screenshot mapping JSON file "
            "(task_id -> {step_index -> path}). "
            "This is the recommended method for screenshot resolution."
        ),
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Exclude <think> blocks from training data",
    )
    parser.add_argument(
        "--bundle",
        type=str,
        help=(
            "Create a self-contained bundle directory with images/ and "
            "training_data.jsonl (rewrites paths to relative). "
            "Overrides --output."
        ),
    )

    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    if not demo_dir.is_dir():
        print(f"ERROR: {demo_dir} is not a directory", file=sys.stderr)
        return 1

    captures_dir = Path(args.captures_dir) if args.captures_dir else None

    screenshot_mapping = None
    if args.mapping:
        mapping_path = Path(args.mapping)
        if not mapping_path.exists():
            print(f"ERROR: {mapping_path} not found", file=sys.stderr)
            return 1
        with open(mapping_path) as f:
            screenshot_mapping = json.load(f)
        print(f"Loaded screenshot mapping: {len(screenshot_mapping)} tasks")

    if args.bundle:
        # Bundle mode: convert then create self-contained directory
        samples = convert_all_demos(
            demo_dir=demo_dir,
            captures_dir=captures_dir,
            screenshot_mapping=screenshot_mapping,
            output_path=None,  # don't write yet
            include_thinking=not args.no_thinking,
        )
        create_bundle(samples, Path(args.bundle))
    else:
        convert_all_demos(
            demo_dir=demo_dir,
            captures_dir=captures_dir,
            screenshot_mapping=screenshot_mapping,
            output_path=Path(args.output),
            include_thinking=not args.no_thinking,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
