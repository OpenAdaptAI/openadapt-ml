"""Base interface for grounding modules.

Grounding is the process of converting a natural language target description
(e.g., "the login button") into executable coordinates on a screenshot.

This module defines the abstract interface that all grounding strategies must implement,
enabling policy/grounding separation as described in the architecture document.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class RegionCandidate:
    """A candidate region for action execution.

    Represents a potential target location on the screen, with confidence
    score and optional metadata.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in normalized [0,1] coordinates.
        centroid: Click point as (x, y) in normalized coordinates.
        confidence: Confidence score in [0, 1], higher is better.
        element_label: Optional label describing the element (e.g., "button", "input").
        text_content: Optional text content of the element.
        metadata: Additional grounding-specific data.
    """

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    centroid: tuple[float, float]  # click point (x, y)
    confidence: float
    element_label: str | None = None
    text_content: str | None = None
    metadata: dict | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate coordinates are in [0, 1] range."""
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.centroid

        for val in [x1, y1, x2, y2, cx, cy]:
            if not 0 <= val <= 1:
                raise ValueError(f"Coordinates must be in [0, 1] range, got {val}")

        if x1 > x2 or y1 > y2:
            raise ValueError(f"Invalid bbox: x1 > x2 or y1 > y2: {self.bbox}")

        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @property
    def area(self) -> float:
        """Compute normalized area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def iou(self, other: "RegionCandidate") -> float:
        """Compute Intersection over Union with another region.

        Args:
            other: Another RegionCandidate.

        Returns:
            IoU score in [0, 1].
        """
        x1, y1, x2, y2 = self.bbox
        ox1, oy1, ox2, oy2 = other.bbox

        # Intersection
        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the bounding box.

        Args:
            x: X coordinate in normalized [0, 1].
            y: Y coordinate in normalized [0, 1].

        Returns:
            True if point is inside bbox.
        """
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class GroundingModule(ABC):
    """Abstract base class for grounding strategies.

    A grounding module takes a screenshot and a natural language description
    of the target element, and returns candidate regions where the target
    might be located.

    Implementations include:
    - SoMGrounder: Uses pre-labeled element indices (for synthetic/SoM mode)
    - CoordinateGrounder: Fine-tuned VLM regression
    - DetectorGrounder: External detection (OmniParser, Gemini bbox API)
    - AttentionGrounder: GUI-Actor style attention-based region selection

    Example:
        grounder = DetectorGrounder()
        candidates = grounder.ground(screenshot, "the submit button", k=3)
        best = candidates[0]  # Highest confidence
        click(best.centroid[0], best.centroid[1])
    """

    @abstractmethod
    def ground(
        self,
        image: "Image",
        target_description: str,
        k: int = 1,
    ) -> list[RegionCandidate]:
        """Locate regions matching the target description.

        Args:
            image: PIL Image of the screenshot to search.
            target_description: Natural language description of the target
                element (e.g., "login button", "username field", "the red X").
            k: Maximum number of candidates to return.

        Returns:
            List of candidate regions, sorted by confidence descending.
            Returns empty list if no candidates found.
        """
        pass

    def ground_batch(
        self,
        images: list["Image"],
        target_descriptions: list[str],
        k: int = 1,
    ) -> list[list[RegionCandidate]]:
        """Batch grounding for multiple images/targets.

        Default implementation calls ground() for each pair.
        Subclasses can override for more efficient batching.

        Args:
            images: List of PIL Images.
            target_descriptions: List of target descriptions (same length as images).
            k: Maximum candidates per image.

        Returns:
            List of candidate lists, one per input image.
        """
        if len(images) != len(target_descriptions):
            raise ValueError("images and target_descriptions must have same length")

        return [
            self.ground(img, desc, k=k)
            for img, desc in zip(images, target_descriptions)
        ]

    @property
    def name(self) -> str:
        """Return the name of this grounding module."""
        return self.__class__.__name__

    @property
    def supports_batch(self) -> bool:
        """Whether this module has optimized batch processing."""
        return False


class OracleGrounder(GroundingModule):
    """Oracle grounding using ground-truth bounding boxes.

    Used for evaluation to measure policy performance independent of grounding.
    Returns the ground-truth bbox as the only candidate with confidence 1.0.
    """

    def __init__(self) -> None:
        """Initialize oracle grounder."""
        self._ground_truth: dict[str, RegionCandidate] = {}

    def set_ground_truth(
        self,
        target_description: str,
        bbox: tuple[float, float, float, float],
        centroid: tuple[float, float] | None = None,
    ) -> None:
        """Set ground truth for a target description.

        Args:
            target_description: The target to set ground truth for.
            bbox: Ground truth bounding box (x1, y1, x2, y2).
            centroid: Optional click point. If None, uses bbox center.
        """
        if centroid is None:
            x1, y1, x2, y2 = bbox
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

        self._ground_truth[target_description] = RegionCandidate(
            bbox=bbox,
            centroid=centroid,
            confidence=1.0,
            element_label="ground_truth",
        )

    def ground(
        self,
        image: "Image",
        target_description: str,
        k: int = 1,
    ) -> list[RegionCandidate]:
        """Return ground truth if available.

        Args:
            image: Screenshot (ignored, we use ground truth).
            target_description: Target to look up.
            k: Ignored (always returns 0 or 1 candidate).

        Returns:
            List containing ground truth candidate, or empty list.
        """
        if target_description in self._ground_truth:
            return [self._ground_truth[target_description]]
        return []
