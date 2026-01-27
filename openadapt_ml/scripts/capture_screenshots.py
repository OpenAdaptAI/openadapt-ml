#!/usr/bin/env python3
"""Capture screenshots of dashboards and VMs for documentation.

This script captures screenshots from:
1. Azure ops dashboard (http://localhost:8765/azure_ops.html)
2. VNC viewer (http://localhost:8006) - Windows VM
3. Terminal output from VM monitor (uses PIL rendering)
4. Training dashboard (http://localhost:8080/dashboard.html)

Prerequisites:
    - PIL (Pillow) - for terminal screenshots and image manipulation
    - Optional: playwright - for web page screenshots (better quality)
    - Optional: macOS screencapture - fallback for web pages

Usage:
    # Capture all available dashboards
    uv run python -m openadapt_ml.scripts.capture_screenshots

    # Capture specific targets
    uv run python -m openadapt_ml.scripts.capture_screenshots --target azure-ops
    uv run python -m openadapt_ml.scripts.capture_screenshots --target vnc
    uv run python -m openadapt_ml.scripts.capture_screenshots --target terminal
    uv run python -m openadapt_ml.scripts.capture_screenshots --target training

    # Capture with custom output directory
    uv run python -m openadapt_ml.scripts.capture_screenshots --output /path/to/screenshots

    # List available targets
    uv run python -m openadapt_ml.scripts.capture_screenshots --list

Output:
    docs/screenshots/{target}_{timestamp}.png
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "screenshots"


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def check_url_available(url: str, timeout: int = 5) -> bool:
    """Check if a URL is accessible."""
    import urllib.request
    import urllib.error

    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def capture_web_page_playwright(url: str, output_path: Path) -> bool:
    """Capture web page screenshot using playwright.

    Args:
        url: URL to capture
        output_path: Path to save screenshot

    Returns:
        True if successful, False otherwise
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(url, wait_until="networkidle")
            # Wait a bit for any dynamic content
            page.wait_for_timeout(2000)
            page.screenshot(path=str(output_path), full_page=False)
            browser.close()
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  Playwright error: {e}")
        return False


def capture_web_page_selenium(url: str, output_path: Path) -> bool:
    """Capture web page screenshot using selenium.

    Args:
        url: URL to capture
        output_path: Path to save screenshot

    Returns:
        True if successful, False otherwise
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--window-size=1280,900")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        import time

        time.sleep(2)  # Wait for dynamic content
        driver.save_screenshot(str(output_path))
        driver.quit()
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  Selenium error: {e}")
        return False


def capture_web_page_macos(url: str, output_path: Path) -> bool:
    """Capture web page by opening in browser and using macOS screencapture.

    This is a fallback method that requires manual interaction.

    Args:
        url: URL to capture
        output_path: Path to save screenshot

    Returns:
        True if user completed capture, False otherwise
    """
    if sys.platform != "darwin":
        return False

    print(f"  Opening {url} in browser...")
    subprocess.run(["open", url], check=True)

    print("  Press Enter when ready to capture (or 'q' to skip)...")
    response = input().strip().lower()
    if response == "q":
        return False

    # Use screencapture with interactive mode (-i) for user to select window
    print("  Click on the window to capture...")
    result = subprocess.run(
        ["screencapture", "-i", "-W", str(output_path)], capture_output=True
    )
    return result.returncode == 0 and output_path.exists()


def capture_web_page(url: str, output_path: Path, interactive: bool = False) -> bool:
    """Capture web page screenshot using best available method.

    Args:
        url: URL to capture
        output_path: Path to save screenshot
        interactive: If True, allow interactive capture methods

    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try playwright first (best quality)
    if capture_web_page_playwright(url, output_path):
        return True

    # Try selenium as fallback
    if capture_web_page_selenium(url, output_path):
        return True

    # On macOS, offer interactive capture
    if interactive and capture_web_page_macos(url, output_path):
        return True

    return False


def capture_terminal_output(command: list[str], output_path: Path) -> bool:
    """Capture terminal command output as image using PIL.

    Args:
        command: Command to run
        output_path: Path to save screenshot

    Returns:
        True if successful, False otherwise
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=60,
        )
        output = result.stdout or result.stderr
    except subprocess.TimeoutExpired:
        output = "ERROR: Command timed out"
    except Exception as e:
        output = f"ERROR: {e}"

    # Strip ANSI codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    output = ansi_escape.sub("", output)

    # Terminal color scheme
    bg_color = (30, 30, 30)
    text_color = (220, 220, 220)

    # Load font
    font_size = 14
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("Courier New", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Calculate dimensions
    lines = output.split("\n")
    line_height = int(font_size * 1.4)
    char_width = font_size * 0.6
    padding = 20

    max_line_len = max((len(line) for line in lines), default=80)
    width = int(max_line_len * char_width) + 2 * padding
    height = len(lines) * line_height + 2 * padding

    # Create image
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=text_color, font=font)
        y += line_height

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return True


def capture_vnc_screenshot(output_path: Path) -> bool:
    """Capture VNC viewer screenshot.

    The VNC viewer at localhost:8006 is a noVNC HTML5 client.
    We capture it as a web page.

    Args:
        output_path: Path to save screenshot

    Returns:
        True if successful, False otherwise
    """
    url = "http://localhost:8006"
    if not check_url_available(url):
        print(f"  VNC not available at {url}")
        return False

    return capture_web_page(url, output_path, interactive=True)


def capture_azure_ops_dashboard(output_path: Path) -> bool:
    """Capture Azure ops dashboard screenshot.

    Args:
        output_path: Path to save screenshot

    Returns:
        True if successful, False otherwise
    """
    url = "http://localhost:8765/azure_ops.html"
    if not check_url_available(url):
        print(f"  Azure ops dashboard not available at {url}")
        return False

    return capture_web_page(url, output_path, interactive=True)


def capture_training_dashboard(output_path: Path) -> bool:
    """Capture training dashboard screenshot.

    Args:
        output_path: Path to save screenshot

    Returns:
        True if successful, False otherwise
    """
    url = "http://localhost:8080/dashboard.html"
    if not check_url_available(url):
        print(f"  Training dashboard not available at {url}")
        return False

    return capture_web_page(url, output_path, interactive=True)


def capture_vm_monitor(output_path: Path, mock: bool = True) -> bool:
    """Capture VM monitor terminal output.

    Args:
        output_path: Path to save screenshot
        mock: If True, use --mock flag to avoid Azure costs

    Returns:
        True if successful, False otherwise
    """
    cmd = ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "vm", "monitor"]
    if mock:
        cmd.append("--mock")

    return capture_terminal_output(cmd, output_path)


def capture_vm_screenshot_from_vm(output_path: Path) -> bool:
    """Capture VM screen directly via QEMU monitor.

    This uses the existing vm screenshot command in the CLI.

    Args:
        output_path: Path to save screenshot

    Returns:
        True if successful, False otherwise
    """
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "openadapt_ml.benchmarks.cli",
            "vm",
            "screenshot",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(
            f"  VM screenshot failed: {result.stderr[:200] if result.stderr else 'Unknown error'}"
        )
        return False

    # The CLI saves to training_output/current/vm_screenshot.png
    src_path = PROJECT_ROOT / "training_output" / "current" / "vm_screenshot.png"
    if src_path.exists():
        import shutil

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, output_path)
        return True

    return False


TARGETS = {
    "azure-ops": {
        "description": "Azure ops dashboard (localhost:8765)",
        "capture_fn": capture_azure_ops_dashboard,
        "filename": "azure_ops_dashboard",
    },
    "vnc": {
        "description": "VNC viewer (localhost:8006) - Windows VM",
        "capture_fn": capture_vnc_screenshot,
        "filename": "vnc_viewer",
    },
    "terminal": {
        "description": "VM monitor terminal output",
        "capture_fn": lambda p: capture_vm_monitor(p, mock=True),
        "filename": "vm_monitor_terminal",
    },
    "terminal-live": {
        "description": "VM monitor terminal output (live, no mock)",
        "capture_fn": lambda p: capture_vm_monitor(p, mock=False),
        "filename": "vm_monitor_terminal_live",
    },
    "training": {
        "description": "Training dashboard (localhost:8080)",
        "capture_fn": capture_training_dashboard,
        "filename": "training_dashboard",
    },
    "vm-screen": {
        "description": "Windows VM screen (via QEMU)",
        "capture_fn": capture_vm_screenshot_from_vm,
        "filename": "vm_screen",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Capture screenshots of dashboards and VMs for documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Capture all available targets
    uv run python -m openadapt_ml.scripts.capture_screenshots

    # Capture specific target
    uv run python -m openadapt_ml.scripts.capture_screenshots --target azure-ops

    # Capture multiple targets
    uv run python -m openadapt_ml.scripts.capture_screenshots --target azure-ops --target vnc

    # List available targets
    uv run python -m openadapt_ml.scripts.capture_screenshots --list
""",
    )
    parser.add_argument(
        "--target",
        "-t",
        action="append",
        choices=list(TARGETS.keys()),
        help="Target to capture (can specify multiple)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for screenshots",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available targets",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't add timestamp to filenames",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Allow interactive capture methods (e.g., macOS screencapture)",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable screenshot targets:\n")
        for name, info in TARGETS.items():
            print(f"  {name:15} - {info['description']}")
        print()
        return 0

    # Determine targets to capture
    targets = args.target or list(TARGETS.keys())

    # Create output directory
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Screenshot Capture Tool ".center(60))
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Targets: {', '.join(targets)}\n")

    timestamp = get_timestamp() if not args.no_timestamp else ""
    results = {}

    for target in targets:
        info = TARGETS[target]
        print(f"\n[{target}] {info['description']}")

        filename = info["filename"]
        if timestamp:
            filename = f"{filename}_{timestamp}"
        output_path = output_dir / f"{filename}.png"

        try:
            success = info["capture_fn"](output_path)
            if success:
                size_kb = output_path.stat().st_size / 1024
                print(f"  OK: {output_path.name} ({size_kb:.1f} KB)")
                results[target] = str(output_path)
            else:
                print("  SKIP: Not available or capture failed")
                results[target] = None
        except Exception as e:
            print(f"  ERROR: {e}")
            results[target] = None

    # Summary
    print("\n" + "=" * 60)
    print(" Summary ".center(60))
    print("=" * 60)

    successful = [t for t, p in results.items() if p]
    failed = [t for t, p in results.items() if not p]

    if successful:
        print(f"\nCaptured ({len(successful)}):")
        for target in successful:
            print(f"  - {results[target]}")

    if failed:
        print(f"\nSkipped/Failed ({len(failed)}):")
        for target in failed:
            print(f"  - {target}")

    print()
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
