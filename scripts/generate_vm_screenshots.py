#!/usr/bin/env python3
"""Generate VM monitor screenshots for documentation.

This script generates terminal screenshots using asciinema and agg (asciinema-to-gif).

Prerequisites:
    brew install asciinema
    brew install agg

Usage:
    python scripts/generate_vm_screenshots.py

    # Or run directly with permissions:
    chmod +x scripts/generate_vm_screenshots.py
    ./scripts/generate_vm_screenshots.py

Output:
    docs/screenshots/vm_monitor_*.png

Features:
    - Uses --mock flag to avoid Azure VM costs
    - Generates multiple variants (full, details, idle)
    - High quality PNG output with Monaco font
    - Automatic cleanup of intermediate .cast files
"""

import os
import subprocess
import time
from pathlib import Path

# Project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def check_prerequisites():
    """Check if asciinema and agg are installed."""
    try:
        subprocess.run(["asciinema", "--version"], capture_output=True, check=True)
        subprocess.run(["agg", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("❌ ERROR: Required tools not found!")
        print("\nPlease install prerequisites:")
        print("  brew install asciinema")
        print("  brew install agg")
        print(f"\nError: {e}")
        return False


def record_and_convert(
    command: list[str],
    output_name: str,
    width: int = 120,
    height: int = 50,
    title: str | None = None,
):
    """Record terminal command and convert to PNG.

    Args:
        command: Command to run (list of strings)
        output_name: Output filename (without extension)
        width: Terminal width in columns
        height: Terminal height in rows
        title: Optional title for the recording
    """
    cast_file = SCREENSHOTS_DIR / f"{output_name}.cast"
    png_file = SCREENSHOTS_DIR / f"{output_name}.png"

    title_str = f" ({title})" if title else ""
    print(f"\n📹 Recording{title_str}: {' '.join(command)}")

    # Set up environment with terminal dimensions
    env = os.environ.copy()
    env["COLUMNS"] = str(width)
    env["LINES"] = str(height)

    # Record with asciinema
    try:
        subprocess.run(
            [
                "asciinema",
                "rec",
                str(cast_file),
                "--overwrite",
                "--command",
                " ".join(command),
            ],
            env=env,
            check=True,
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Recording failed: {e}")
        return False

    print(f"🎨 Converting to PNG: {png_file.name}")

    # Convert to PNG with agg
    try:
        subprocess.run(
            [
                "agg",
                str(cast_file),
                str(png_file),
                "--font-family",
                "Monaco",
                "--font-size",
                "14",
                "--line-height",
                "1.4",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        return False

    print(f"✅ Saved: {png_file.relative_to(PROJECT_ROOT)}")

    # Clean up cast file
    cast_file.unlink()
    return True


def main():
    """Generate all VM monitor screenshots."""

    print("=" * 70)
    print(" VM Monitor Screenshot Generator ".center(70))
    print("=" * 70)

    # Check prerequisites
    if not check_prerequisites():
        return 1

    print(f"\n📁 Output directory: {SCREENSHOTS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"🎯 Generating screenshots with mock data (no VM required)")

    # Screenshot 1: Full monitor dashboard (default)
    success = record_and_convert(
        ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "vm", "monitor", "--mock"],
        "vm_monitor_dashboard_full",
        width=120,
        height=50,
        title="Full Dashboard",
    )
    if not success:
        print("❌ Failed to generate vm_monitor_dashboard_full.png")
        return 1

    time.sleep(1)  # Small delay between recordings

    # Screenshot 2: Monitor with --details flag
    success = record_and_convert(
        ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "vm", "monitor", "--mock", "--details"],
        "vm_monitor_details",
        width=120,
        height=55,
        title="With Details",
    )
    if not success:
        print("❌ Failed to generate vm_monitor_details.png")
        return 1

    time.sleep(1)

    # Screenshot 3: VM status (quick check) - Note: status doesn't have --mock yet
    # Skip this one for now since status command doesn't support mock mode
    # Can add later if needed

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
