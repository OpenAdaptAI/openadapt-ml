"""Grounding modules for visual element localization.

This package provides strategies for grounding natural language target descriptions
to specific regions on screenshots. The grounding/policy separation enables:

- Training policy and grounding separately or jointly
- Swapping grounding strategies without retraining policy
- Evaluating each layer independently
- Composing different grounding modules per platform

Available grounding strategies:

- OracleGrounder: Uses ground-truth bboxes (for evaluation)
- GeminiGrounder: Uses Google Gemini vision API for element detection
- DetectorGrounder: Generic detector wrapper with backend selection
- SoMGrounder: Element index selection from Set-of-Marks overlay (coming soon)
- AttentionGrounder: GUI-Actor style attention-based selection (coming soon)

Functions:

- extract_ui_elements: Extract all interactive UI elements from a screenshot
- overlay_element_marks: Overlay numbered labels (Set-of-Marks) on elements
"""

from openadapt_ml.grounding.base import (
    GroundingModule,
    OracleGrounder,
    RegionCandidate,
)
from openadapt_ml.grounding.detector import (
    DetectorGrounder,
    GeminiGrounder,
    extract_ui_elements,
    overlay_element_marks,
)

__all__ = [
    "GroundingModule",
    "OracleGrounder",
    "RegionCandidate",
    "DetectorGrounder",
    "GeminiGrounder",
    "extract_ui_elements",
    "overlay_element_marks",
]
