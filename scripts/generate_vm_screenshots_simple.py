#!/usr/bin/env python3
"""Generate VM monitor screenshots for documentation (Simple Python-only version).

This script generates terminal screenshots using pure Python without external dependencies.
It captures the terminal output and renders it as an image using PIL.

Prerequisites:
    pip install pillow  (should already be installed for openadapt-ml)

Usage:
    python scripts/generate_vm_screenshots_simple.py

Output:
    docs/screenshots/vm_monitor_*.png

Features:
    - Pure Python solution (no external tools required)
    - Uses --mock flag to avoid Azure VM costs
    - Generates PNG images with monospace font
    - Works on any platform
"""

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def capture_terminal_output(command: list[str]) -> str:
    """Run command and capture terminal output.

    Args:
        command: Command to run (list of strings)

    Returns:
        Terminal output as string
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out"
    except Exception as e:
        return f"ERROR: {e}"


def render_terminal_output(
    output: str,
    output_path: Path,
    font_size: int = 14,
    padding: int = 20,
    line_height: float = 1.4,
):
    """Render terminal output as PNG image.

    Args:
        output: Terminal output text
        output_path: Path to save PNG
        font_size: Font size in pixels
        padding: Padding around text in pixels
        line_height: Line height multiplier
    """
    # Terminal color scheme (dark theme)
    bg_color = (30, 30, 30)  # Dark gray background
    text_color = (220, 220, 220)  # Light gray text
    accent_color = (100, 200, 100)  # Green for success

    # Try to load a monospace font
    try:
        # Try Monaco (macOS)
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", font_size)
    except:
        try:
            # Try Courier New (cross-platform)
            font = ImageFont.truetype("Courier New", font_size)
        except:
            # Fallback to default
            font = ImageFont.load_default()

    # Split output into lines
    lines = output.split("\n")

    # Calculate image dimensions
    # Use a rough estimate for monospace font width
    char_width = font_size * 0.6
    char_height = int(font_size * line_height)

    max_line_len = max(len(line) for line in lines) if lines else 80
    width = int(max_line_len * char_width) + 2 * padding
    height = len(lines) * char_height + 2 * padding

    # Create image
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text line by line
    y = padding
    for line in lines:
        # Remove ANSI color codes (simple regex would be better but keeping it simple)
        clean_line = line.replace("\033[0m", "").replace("\033[1m", "")

        # Draw the line
        draw.text((padding, y), clean_line, fill=text_color, font=font)
        y += char_height

    # Save image
    img.save(output_path)
    print(f"✅ Saved: {output_path.relative_to(PROJECT_ROOT)}")


def generate_screenshot(
    command: list[str],
    output_name: str,
    title: str | None = None,
):
    """Generate a screenshot by running command and rendering output.

    Args:
        command: Command to run
        output_name: Output filename (without extension)
        title: Optional title for logging
    """
    title_str = f" ({title})" if title else ""
    print(f"\n📹 Capturing{title_str}: {' '.join(command)}")

    # Capture output
    output = capture_terminal_output(command)

    # Render to PNG
    png_file = SCREENSHOTS_DIR / f"{output_name}.png"
    render_terminal_output(output, png_file)

    return True


def main():
    """Generate all VM monitor screenshots."""

    print("=" * 70)
    print(" VM Monitor Screenshot Generator (Simple) ".center(70))
    print("=" * 70)

    print(f"\n📁 Output directory: {SCREENSHOTS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"🎯 Generating screenshots with mock data (no VM required)")

    # Screenshot 1: Full monitor dashboard (default)
    generate_screenshot(
        ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "vm", "monitor", "--mock"],
        "vm_monitor_dashboard_full",
        title="Full Dashboard",
    )

    # Screenshot 2: Monitor with --details flag
    generate_screenshot(
        ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "vm", "monitor", "--mock", "--details"],
        "vm_monitor_details",
        title="With Details",
    )

    print("\n" + "=" * 70)
    print(" ✅ All screenshots generated successfully! ".center(70))
    print("=" * 70)
    print(f"\n📂 Location: {SCREENSHOTS_DIR}")
    print("\nGenerated files:")
    for png_file in sorted(SCREENSHOTS_DIR.glob("vm_monitor_*.png")):
        size_kb = png_file.stat().st_size / 1024
        print(f"  • {png_file.name} ({size_kb:.1f} KB)")

    print("\n📝 Next steps:")
    print("  1. Review screenshots in docs/screenshots/")
    print("  2. Update README.md to include screenshots")
    print("  3. Commit changes to git")

    return 0


if __name__ == "__main__":
    exit(main())
