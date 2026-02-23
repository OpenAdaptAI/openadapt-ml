"""Convert annotated demo JSON files to ms-swift SFT training format.

Reads annotated demos produced by the VLM annotation pipeline and converts
them into JSONL training data compatible with ms-swift for Qwen3-VL
fine-tuning.

Input format (annotated demo JSON):
    {
      "steps": [{
        "step_index": 0,
        "observation": "The desktop shows...",
        "intent": "The user is trying to...",
        "action_raw": "CLICK(0.294, 0.532)",
        ...
      }]
    }

Output format (ms-swift SFT JSONL):
    {
      "image": "step_000_screenshot.png",
      "conversations": [
        {"role": "user", "content": "<image>\\nInstruction: ...\\nPrevious actions:\\n..."},
        {"role": "assistant", "content": "<think>...</think>\\nclick(x=294, y=532)"}
      ]
    }

Coordinates are converted from [0, 1] to [0, 1000] (Qwen3-VL format).

Usage:
    python -m openadapt_ml.training.convert_demos \\
        --demo-dir /path/to/annotated_demos \\
        --screenshot-dir /path/to/screenshots \\
        --output /path/to/output.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _parse_action_raw(action_raw: str) -> tuple[str, dict[str, Any]]:
    """Parse action_raw string into (action_type, params).

    Args:
        action_raw: Raw action string like "CLICK(0.294, 0.532)" or "TYPE(\"text\")".

    Returns:
        Tuple of (action_type, params_dict).
    """
    # DOUBLE_CLICK(x, y)
    m = re.match(r"DOUBLE_CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_raw, re.IGNORECASE)
    if m:
        return "double_click", {"x": float(m.group(1)), "y": float(m.group(2))}

    # RIGHT_CLICK(x, y)
    m = re.match(r"RIGHT_CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_raw, re.IGNORECASE)
    if m:
        return "right_click", {"x": float(m.group(1)), "y": float(m.group(2))}

    # CLICK(x, y)
    m = re.match(r"CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_raw, re.IGNORECASE)
    if m:
        return "click", {"x": float(m.group(1)), "y": float(m.group(2))}

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
        return "drag", {
            "from_x": float(m.group(1)),
            "from_y": float(m.group(2)),
            "to_x": float(m.group(3)),
            "to_y": float(m.group(4)),
        }

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

    if action_type in ("wait", "done", "finished"):
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


def convert_step(
    step: dict[str, Any],
    instruction: str,
    previous_actions: list[str],
    screenshot_path: str | None = None,
    include_thinking: bool = True,
) -> dict[str, Any] | None:
    """Convert a single annotated step to ms-swift SFT format.

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

    # Build user content
    user_parts = ["<image>"]
    user_parts.append(f"Instruction: {instruction}")
    if previous_actions:
        user_parts.append("")
        user_parts.append("Previous actions:")
        for i, act in enumerate(previous_actions):
            user_parts.append(f"  Step {i}: {act}")

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
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }

    if screenshot_path:
        sample["image"] = screenshot_path

    return sample


def convert_demo(
    demo_path: Path,
    screenshot_dir: Path | None = None,
    include_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Convert an annotated demo JSON file to ms-swift SFT samples.

    Args:
        demo_path: Path to annotated demo JSON file.
        screenshot_dir: Directory containing screenshot PNGs.
        include_thinking: Whether to include <think> blocks.

    Returns:
        List of ms-swift training sample dicts.
    """
    with open(demo_path) as f:
        demo = json.load(f)

    instruction = demo.get("instruction", "")
    steps = demo.get("steps", [])
    task_id = demo.get("task_id", demo_path.stem)

    samples = []
    previous_actions: list[str] = []

    for step in steps:
        step_idx = step.get("step_index", len(previous_actions))

        # Find screenshot
        screenshot_path = None
        if screenshot_dir:
            # Try common naming patterns
            for pattern in [
                f"step_{step_idx:03d}_screenshot.png",
                f"step_{step_idx:03d}.png",
                f"screenshot_{step_idx:03d}.png",
                f"{task_id}_step_{step_idx:03d}.png",
            ]:
                candidate = screenshot_dir / pattern
                if candidate.exists():
                    screenshot_path = str(candidate)
                    break

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
    screenshot_dir: Path | None = None,
    output_path: Path | None = None,
    include_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Convert all annotated demo JSON files in a directory to SFT format.

    Args:
        demo_dir: Directory containing annotated demo JSON files.
        screenshot_dir: Directory containing screenshot PNGs.
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
            screenshot_dir=screenshot_dir,
            include_thinking=include_thinking,
        )
        print(f"  -> {len(samples)} training samples")
        all_samples.extend(samples)

    print(f"\nTotal: {len(all_samples)} training samples from {len(demo_files)} demos")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Written to {output_path}")

    return all_samples


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
        "--screenshot-dir",
        type=str,
        help="Directory containing screenshot PNGs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.jsonl",
        help="Output JSONL file path (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Exclude <think> blocks from training data",
    )

    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    if not demo_dir.is_dir():
        print(f"ERROR: {demo_dir} is not a directory", file=sys.stderr)
        return 1

    screenshot_dir = Path(args.screenshot_dir) if args.screenshot_dir else None

    convert_all_demos(
        demo_dir=demo_dir,
        screenshot_dir=screenshot_dir,
        output_path=Path(args.output),
        include_thinking=not args.no_thinking,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
