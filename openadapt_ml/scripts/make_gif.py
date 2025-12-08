from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:  # type: ignore[name-defined]
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT = _load_font(16)


def _load_frames(frames_dir: Path, pattern: str) -> List[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(frames_dir / pattern)))
    if not paths:
        raise ValueError(f"No frames matched pattern '{pattern}' under {frames_dir}")
    return paths


def _default_login_caption(filename: str, index: int) -> str:
    # Heuristic captions for the synthetic login script based on step index
    # and the conventional *_step_{k}.png naming.
    name = os.path.basename(filename)
    # Try to extract step index from name if present.
    step_idx = index
    for part in name.split("_"):
        if part.startswith("step"):
            try:
                step_idx = int(part.replace("step", "").replace(".png", ""))
            except ValueError:
                pass
    if step_idx == 0:
        return "Step 0: Initial login screen (WAIT)"
    if step_idx == 1:
        return "Step 1: CLICK username field"
    if step_idx == 2:
        return "Step 2: TYPE username"
    if step_idx == 3:
        return "Step 3: CLICK password field"
    if step_idx == 4:
        return "Step 4: TYPE password (masked)"
    if step_idx == 5:
        return "Step 5: CLICK Login button"
    if step_idx == 6:
        return "Step 6: DONE (logged in)"
    return f"Step {step_idx}: synthetic step"


def _draw_caption(image: Image.Image, text: str) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Draw a semi-transparent rectangle at the bottom for text background
    padding = 8
    text_width, text_height = draw.textbbox((0, 0), text, font=FONT)[2:4]  # type: ignore[assignment]
    rect_height = text_height + 2 * padding
    y0 = height - rect_height
    draw.rectangle([(0, y0), (width, height)], fill=(0, 0, 0, 180))
    x_text = max(padding, (width - text_width) // 2)
    y_text = y0 + padding
    draw.text((x_text, y_text), text, font=FONT, fill=(255, 255, 255))
    return img


def make_gif(
    frames_dir: Path,
    pattern: str,
    output: Path,
    duration_ms: int = 1000,
    scenario: Optional[str] = None,
) -> None:
    frame_paths = _load_frames(frames_dir, pattern)

    frames: List[Image.Image] = []
    for idx, frame_path in enumerate(frame_paths):
        img = Image.open(frame_path)
        if scenario == "login":
            caption = _default_login_caption(frame_path.name, idx)
            img = _draw_caption(img, caption)
        frames.append(img)

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an animated GIF from a sequence of PNG frames.",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing frame PNGs (e.g. synthetic_demo/session_0000)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*step_*.png",
        help="Glob pattern for frame filenames inside frames-dir (default: *step_*.png)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output GIF path",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=1000,
        help="Frame duration in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["login", None],  # type: ignore[list-item]
        help="Optional built-in captioning scenario (e.g. 'login')",
    )

    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output = Path(args.output)

    make_gif(
        frames_dir=frames_dir,
        pattern=args.pattern,
        output=output,
        duration_ms=args.duration_ms,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
