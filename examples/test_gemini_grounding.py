#!/usr/bin/env python3
"""Example script demonstrating GeminiGrounder for UI element detection.

This script shows how to use the Gemini-based grounding module to:
1. Extract all interactive UI elements from a screenshot
2. Overlay Set-of-Marks (SoM) numbered labels on elements
3. Use GeminiGrounder to locate specific elements by description

Requirements:
    - GOOGLE_API_KEY environment variable set (from .env file)
    - google-generativeai package installed

Usage:
    python examples/test_gemini_grounding.py <screenshot_path>

Example:
    python examples/test_gemini_grounding.py screenshots/login.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from openadapt_ml.grounding import (
    GeminiGrounder,
    extract_ui_elements,
    overlay_element_marks,
)


def main() -> None:
    """Test Gemini grounding on a screenshot."""
    if len(sys.argv) < 2:
        print("Usage: python test_gemini_grounding.py <screenshot_path>")
        print("\nExample:")
        print("  python test_gemini_grounding.py screenshots/login.png")
        sys.exit(1)

    screenshot_path = Path(sys.argv[1])
    if not screenshot_path.exists():
        print(f"Error: Screenshot not found: {screenshot_path}")
        sys.exit(1)

    print(f"Loading screenshot: {screenshot_path}")
    screenshot = Image.open(screenshot_path)

    # Step 1: Extract all UI elements
    print("\n=== Step 1: Extracting UI elements ===")
    elements = extract_ui_elements(screenshot)

    if not elements:
        print("No UI elements detected!")
        return

    print(f"Found {len(elements)} interactive elements:")
    for elem in elements:
        print(
            f"  [{elem['id']}] {elem['label']} ({elem['type']}) "
            f"at {elem['bbox']}"
        )

    # Step 2: Overlay SoM marks
    print("\n=== Step 2: Creating Set-of-Marks overlay ===")
    marked_screenshot = overlay_element_marks(screenshot, elements, style="compact")

    # Save marked screenshot
    output_path = screenshot_path.parent / f"{screenshot_path.stem}_marked.png"
    marked_screenshot.save(output_path)
    print(f"Saved marked screenshot to: {output_path}")

    # Step 3: Test grounding a specific element
    print("\n=== Step 3: Testing GeminiGrounder ===")
    grounder = GeminiGrounder()

    # Try to find a button
    print("Searching for 'submit button'...")
    candidates = grounder.ground(screenshot, "submit button", k=3)

    if candidates:
        print(f"Found {len(candidates)} candidates:")
        for i, candidate in enumerate(candidates, 1):
            print(
                f"  Candidate {i}: bbox={candidate.bbox}, "
                f"confidence={candidate.confidence:.2f}, "
                f"label={candidate.element_label}"
            )
    else:
        print("No candidates found for 'submit button'")

    # Try to find a text field
    print("\nSearching for 'username field'...")
    candidates = grounder.ground(screenshot, "username field", k=3)

    if candidates:
        print(f"Found {len(candidates)} candidates:")
        for i, candidate in enumerate(candidates, 1):
            print(
                f"  Candidate {i}: bbox={candidate.bbox}, "
                f"confidence={candidate.confidence:.2f}, "
                f"label={candidate.element_label}"
            )
    else:
        print("No candidates found for 'username field'")

    print("\n=== Done! ===")
    print(f"Check the marked screenshot at: {output_path}")


if __name__ == "__main__":
    main()
