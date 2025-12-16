"""Detector-based grounding using external vision APIs.

This module provides grounding implementations that use external element detection
services (Gemini, OmniParser) to locate UI elements on screenshots.

Functions:
    extract_ui_elements: Extract all interactive UI elements from a screenshot
    overlay_element_marks: Overlay numbered labels (Set-of-Marks) on elements
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import TYPE_CHECKING

from openadapt_ml.config import settings
from openadapt_ml.grounding.base import GroundingModule, RegionCandidate

if TYPE_CHECKING:
    from PIL import Image, ImageDraw, ImageFont


class GeminiGrounder(GroundingModule):
    """Grounding using Google Gemini's vision capabilities.

    Uses Gemini to identify UI elements matching a description and return
    their bounding boxes.

    Requires:
        - GOOGLE_API_KEY environment variable
        - google-generativeai package: pip install google-generativeai

    Example:
        grounder = GeminiGrounder()
        candidates = grounder.ground(screenshot, "the login button")
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        """Initialize Gemini grounder.

        Args:
            model: Gemini model to use. Options include:
                - "gemini-2.5-flash" (fast, good for grounding)
                - "gemini-2.5-pro" (higher quality)
                - "gemini-3-pro-preview" (most capable)
            api_key: Google API key. If None, uses GOOGLE_API_KEY from settings.
        """
        self._model_name = model
        self._api_key = api_key or settings.google_api_key
        self._model = None

    def _get_model(self):
        """Lazy-load the Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError(
                    "google-generativeai is required. Install with: "
                    "pip install google-generativeai"
                ) from e

            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set. "
                    "Get an API key from https://makersuite.google.com/app/apikey"
                )

            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)

        return self._model

    def _image_to_base64(self, image: "Image") -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_bbox_response(
        self,
        response_text: str,
        image_width: int,
        image_height: int,
    ) -> list[RegionCandidate]:
        """Parse Gemini's bbox response into RegionCandidates.

        Args:
            response_text: Raw text response from Gemini.
            image_width: Image width for normalization.
            image_height: Image height for normalization.

        Returns:
            List of RegionCandidate objects.
        """
        candidates = []

        # Try to parse JSON from the response
        # Look for JSON array or object in the response
        json_match = re.search(r'\[[\s\S]*\]|\{[\s\S]*\}', response_text)
        if not json_match:
            return candidates

        try:
            data = json.loads(json_match.group())

            # Handle both single object and array
            if isinstance(data, dict):
                data = [data]

            for item in data:
                # Extract bbox - handle various formats
                bbox = item.get("bbox") or item.get("bounding_box") or item.get("box")
                if not bbox:
                    # Try to get individual coordinates
                    if all(k in item for k in ["x1", "y1", "x2", "y2"]):
                        bbox = [item["x1"], item["y1"], item["x2"], item["y2"]]
                    elif all(k in item for k in ["x", "y", "width", "height"]):
                        x, y, w, h = item["x"], item["y"], item["width"], item["height"]
                        bbox = [x, y, x + w, y + h]
                    else:
                        continue

                # Normalize coordinates
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox

                    # Check if already normalized (all values <= 1)
                    if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                        norm_bbox = (x1, y1, x2, y2)
                    else:
                        # Normalize pixel coordinates
                        norm_bbox = (
                            x1 / image_width,
                            y1 / image_height,
                            x2 / image_width,
                            y2 / image_height,
                        )

                    # Clamp to valid range
                    norm_bbox = tuple(max(0, min(1, v)) for v in norm_bbox)

                    # Compute centroid
                    cx = (norm_bbox[0] + norm_bbox[2]) / 2
                    cy = (norm_bbox[1] + norm_bbox[3]) / 2

                    # Get confidence (default to 0.8 if not provided)
                    confidence = item.get("confidence", 0.8)

                    candidates.append(
                        RegionCandidate(
                            bbox=norm_bbox,
                            centroid=(cx, cy),
                            confidence=confidence,
                            element_label=item.get("label") or item.get("type"),
                            text_content=item.get("text"),
                            metadata={"raw": item},
                        )
                    )

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return candidates

    def ground(
        self,
        image: "Image",
        target_description: str,
        k: int = 1,
    ) -> list[RegionCandidate]:
        """Locate regions matching the target description using Gemini.

        Args:
            image: PIL Image of the screenshot.
            target_description: Natural language description of the target.
            k: Maximum number of candidates to return.

        Returns:
            List of RegionCandidate objects sorted by confidence.
        """
        model = self._get_model()

        # Include image dimensions in prompt for accurate coordinate detection
        prompt = f"""Analyze this screenshot and find the UI element matching this description: "{target_description}"

The image is {image.width} pixels wide and {image.height} pixels tall.

Return a JSON array with the bounding box(es) of matching elements. Each element should have:
- "bbox": [x1, y1, x2, y2] in pixel coordinates (top-left to bottom-right)
- "confidence": float between 0 and 1
- "label": element type (button, input, link, etc.)
- "text": visible text content if any

IMPORTANT: Use exact pixel coordinates based on the image dimensions provided above.

Return up to {k} best matches. If no match found, return an empty array [].

Example response format:
[{{"bbox": [100, 200, 250, 240], "confidence": 0.95, "label": "button", "text": "Submit"}}]

Return ONLY the JSON array, no other text."""

        try:
            # Create content with image
            import google.generativeai as genai

            response = model.generate_content(
                [prompt, image],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )

            candidates = self._parse_bbox_response(
                response.text,
                image.width,
                image.height,
            )

            # Sort by confidence and limit to k
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            return candidates[:k]

        except Exception as e:
            # Log error but don't crash
            print(f"Gemini grounding error: {e}")
            return []

    @property
    def supports_batch(self) -> bool:
        """Gemini doesn't have optimized batch processing."""
        return False


def extract_ui_elements(
    screenshot: "Image",
    model_name: str = "gemini-2.0-flash",
    api_key: str | None = None,
) -> list[dict]:
    """Extract all interactive UI elements from a screenshot using Gemini.

    This function uses Gemini's vision capabilities to detect and extract
    all interactive UI elements (buttons, text fields, links, etc.) with
    their bounding boxes. Useful for Set-of-Marks (SoM) processing.

    Args:
        screenshot: PIL Image of the screenshot to analyze.
        model_name: Gemini model to use (default: "gemini-2.0-flash").
        api_key: Google API key. If None, uses GOOGLE_API_KEY from settings.

    Returns:
        List of element dictionaries with format:
        {
            "id": int,              # Sequential ID starting at 1
            "label": str,           # Descriptive name (e.g., "Login button")
            "bbox": [x1,y1,x2,y2], # Normalized coordinates [0,1]
            "type": str,           # Element type (button, text_field, etc.)
            "text": str,           # Visible text content (optional)
        }

    Example:
        >>> from PIL import Image
        >>> img = Image.open("login.png")
        >>> elements = extract_ui_elements(img)
        >>> print(elements[0])
        {
            "id": 1,
            "label": "Username text field",
            "bbox": [0.25, 0.30, 0.75, 0.38],
            "type": "text_field",
            "text": ""
        }

    Raises:
        ImportError: If google-generativeai package not installed.
        ValueError: If GOOGLE_API_KEY not set.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "google-generativeai is required. Install with: "
            "pip install google-generativeai"
        ) from e

    api_key = api_key or settings.google_api_key
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Get an API key from https://makersuite.google.com/app/apikey"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = f"""Analyze this screenshot and identify ALL interactive UI elements.

The image is {screenshot.width} pixels wide and {screenshot.height} pixels tall.

For each interactive element (buttons, text fields, links, checkboxes, dropdowns, icons, tabs, menu items), output a JSON object with:

- "id": Sequential integer starting at 1
- "label": Descriptive name (e.g., "Login button", "Username text field", "Submit icon")
- "bbox": Bounding box as [x1, y1, x2, y2] in pixel coordinates (top-left to bottom-right)
- "type": One of: "button", "text_field", "checkbox", "link", "icon", "dropdown", "tab", "menu_item", "other"
- "text": Visible text content if any (empty string if no text)

IMPORTANT:
1. Use exact pixel coordinates based on the image dimensions provided above
2. Include ALL interactive elements you can see, even if they're small
3. Order elements from top-to-bottom, left-to-right
4. Return ONLY a valid JSON array, no markdown formatting, no explanation

Example output format:
[
  {{"id": 1, "label": "Username text field", "bbox": [100, 150, 400, 185], "type": "text_field", "text": ""}},
  {{"id": 2, "label": "Password text field", "bbox": [100, 200, 400, 235], "type": "text_field", "text": ""}},
  {{"id": 3, "label": "Login button", "bbox": [200, 260, 300, 295], "type": "button", "text": "Login"}}
]"""

    try:
        response = model.generate_content(
            [prompt, screenshot],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4096,  # More tokens for many elements
            ),
        )

        # Parse JSON response
        response_text = response.text

        # Try to extract JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if not json_match:
            # Maybe it's just a plain array
            if response_text.strip().startswith('['):
                json_match = re.match(r'.*', response_text)
            else:
                return []

        elements = json.loads(json_match.group())

        # Normalize coordinates to [0, 1]
        normalized_elements = []
        for elem in elements:
            bbox = elem.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # Check if already normalized
                if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                    norm_bbox = [x1, y1, x2, y2]
                else:
                    # Normalize pixel coordinates
                    norm_bbox = [
                        max(0, min(1, x1 / screenshot.width)),
                        max(0, min(1, y1 / screenshot.height)),
                        max(0, min(1, x2 / screenshot.width)),
                        max(0, min(1, y2 / screenshot.height)),
                    ]

                normalized_elements.append({
                    "id": elem.get("id", len(normalized_elements) + 1),
                    "label": elem.get("label", f"Element {elem.get('id', len(normalized_elements) + 1)}"),
                    "bbox": norm_bbox,
                    "type": elem.get("type", "other"),
                    "text": elem.get("text", ""),
                })

        return normalized_elements

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Failed to parse Gemini response: {e}")
        return []
    except Exception as e:
        print(f"Error extracting UI elements: {e}")
        return []


def overlay_element_marks(
    screenshot: "Image",
    elements: list[dict],
    style: str = "compact",
) -> "Image":
    """Overlay numbered labels (Set-of-Marks) on UI elements.

    Creates a new image with numbered markers ([1], [2], [3], etc.) overlaid
    on each UI element. This enables element-based interaction using indices
    instead of coordinates (e.g., CLICK([1]) instead of CLICK(x=0.42, y=0.31)).

    Args:
        screenshot: PIL Image to annotate.
        elements: List of element dicts from extract_ui_elements().
            Each element must have "id" and "bbox" keys.
        style: Label style - "compact" (small circles) or "full" (larger boxes).

    Returns:
        New PIL Image with numbered labels overlaid.

    Example:
        >>> elements = extract_ui_elements(screenshot)
        >>> marked_img = overlay_element_marks(screenshot, elements)
        >>> marked_img.save("screenshot_with_marks.png")
    """
    from PIL import ImageDraw, ImageFont

    img = screenshot.copy()
    draw = ImageDraw.Draw(img)

    width, height = img.size

    # Try to load a good font
    try:
        # Try common font paths
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 14)
                break
            except OSError:
                continue

        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for elem in elements:
        elem_id = elem.get("id", 0)
        bbox = elem.get("bbox", [])

        if len(bbox) != 4:
            continue

        # Convert normalized coords to pixels
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        label = f"[{elem_id}]"

        if style == "compact":
            # Small circle with number at top-left corner
            circle_radius = 12
            circle_x = x1 + circle_radius
            circle_y = y1 + circle_radius

            # Ensure circle is within image bounds
            circle_x = max(circle_radius, min(width - circle_radius, circle_x))
            circle_y = max(circle_radius, min(height - circle_radius, circle_y))

            # Draw red circle background
            draw.ellipse(
                [
                    circle_x - circle_radius,
                    circle_y - circle_radius,
                    circle_x + circle_radius,
                    circle_y + circle_radius,
                ],
                fill="red",
                outline="white",
                width=1,
            )

            # Draw white text centered in circle
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = circle_x - text_width // 2
            text_y = circle_y - text_height // 2

            draw.text((text_x, text_y), label, fill="white", font=font)

        else:  # "full" style
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Draw label box at top-right corner
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0] + 8
            text_height = text_bbox[3] - text_bbox[1] + 4

            label_x = x2 - text_width
            label_y = y1 - text_height

            # Ensure label is within image bounds
            label_x = max(0, min(width - text_width, label_x))
            label_y = max(0, min(height - text_height, label_y))

            # Draw label background
            draw.rectangle(
                [label_x, label_y, label_x + text_width, label_y + text_height],
                fill="red",
                outline="white",
                width=1,
            )

            # Draw label text
            draw.text(
                (label_x + 4, label_y + 2),
                label,
                fill="white",
                font=font,
            )

    return img


class DetectorGrounder(GroundingModule):
    """Generic detector-based grounding with fallback support.

    Wraps multiple detection backends and provides fallback if one fails.

    Example:
        grounder = DetectorGrounder()  # Uses Gemini by default
        grounder = DetectorGrounder(backend="omniparser")  # Use OmniParser
    """

    def __init__(
        self,
        backend: str = "gemini",
        **kwargs,
    ) -> None:
        """Initialize detector grounder.

        Args:
            backend: Detection backend ("gemini", "omniparser").
            **kwargs: Backend-specific arguments.
        """
        self._backend_name = backend

        if backend == "gemini":
            self._backend = GeminiGrounder(**kwargs)
        elif backend == "omniparser":
            raise NotImplementedError(
                "OmniParser backend not yet implemented. "
                "Use backend='gemini' for now."
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def ground(
        self,
        image: "Image",
        target_description: str,
        k: int = 1,
    ) -> list[RegionCandidate]:
        """Delegate to backend grounder."""
        return self._backend.ground(image, target_description, k=k)

    @property
    def name(self) -> str:
        """Return name including backend."""
        return f"DetectorGrounder({self._backend_name})"
