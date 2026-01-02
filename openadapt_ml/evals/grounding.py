"""Grounding-specific evaluation metrics.

This module provides metrics for evaluating grounding accuracy independent
of policy performance, as described in the architecture document.

Metrics:
    - bbox_iou: Intersection over Union with ground-truth element bbox
    - centroid_hit_rate: Whether click point lands inside correct element
    - oracle_hit_rate@k: Any of top-k candidates correct
    - grounding_latency: Time per grounding call
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from openadapt_ml.grounding.base import GroundingModule, RegionCandidate


@dataclass
class GroundingResult:
    """Result of a single grounding evaluation."""

    target_description: str
    ground_truth_bbox: tuple[float, float, float, float] | None
    predicted_candidates: list["RegionCandidate"]
    latency_ms: float

    # Computed metrics
    best_iou: float = 0.0
    centroid_hit: bool = False
    oracle_hit_at_k: dict[int, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute metrics from predictions and ground truth."""
        if not self.ground_truth_bbox or not self.predicted_candidates:
            return

        gt_x1, gt_y1, gt_x2, gt_y2 = self.ground_truth_bbox

        for k, candidate in enumerate(self.predicted_candidates, start=1):
            # IoU
            iou = self._compute_iou(candidate.bbox, self.ground_truth_bbox)
            if iou > self.best_iou:
                self.best_iou = iou

            # Centroid hit
            cx, cy = candidate.centroid
            if gt_x1 <= cx <= gt_x2 and gt_y1 <= cy <= gt_y2:
                if not self.centroid_hit:
                    self.centroid_hit = True

            # Oracle hit at k (if any candidate up to k is a hit)
            hit = iou > 0.5 or (gt_x1 <= cx <= gt_x2 and gt_y1 <= cy <= gt_y2)
            if hit:
                # Mark all k >= current k as hits
                for check_k in range(k, max(len(self.predicted_candidates) + 1, 6)):
                    self.oracle_hit_at_k[check_k] = True

    def _compute_iou(
        self,
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two bboxes."""
        x1, y1, x2, y2 = bbox1
        ox1, oy1, ox2, oy2 = bbox2

        # Intersection
        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (ox2 - ox1) * (oy2 - oy1)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class GroundingMetrics:
    """Aggregated grounding metrics across multiple evaluations."""

    results: list[GroundingResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of evaluated samples."""
        return len(self.results)

    @property
    def mean_iou(self) -> float:
        """Mean IoU across all samples."""
        if not self.results:
            return 0.0
        return sum(r.best_iou for r in self.results) / len(self.results)

    @property
    def centroid_hit_rate(self) -> float:
        """Fraction of samples where centroid hit ground truth."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.centroid_hit) / len(self.results)

    def oracle_hit_rate(self, k: int = 1) -> float:
        """Fraction of samples where any of top-k candidates hit.

        Args:
            k: Number of candidates to consider.

        Returns:
            Hit rate in [0, 1].
        """
        if not self.results:
            return 0.0
        hits = sum(1 for r in self.results if r.oracle_hit_at_k.get(k, False))
        return hits / len(self.results)

    @property
    def mean_latency_ms(self) -> float:
        """Mean grounding latency in milliseconds."""
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def summary(self) -> dict:
        """Return summary dict of all metrics."""
        return {
            "count": self.count,
            "mean_iou": self.mean_iou,
            "centroid_hit_rate": self.centroid_hit_rate,
            "oracle_hit_rate@1": self.oracle_hit_rate(1),
            "oracle_hit_rate@3": self.oracle_hit_rate(3),
            "oracle_hit_rate@5": self.oracle_hit_rate(5),
            "mean_latency_ms": self.mean_latency_ms,
        }

    def __str__(self) -> str:
        """Pretty-print metrics summary."""
        s = self.summary()
        return (
            f"Grounding Metrics (n={s['count']}):\n"
            f"  Mean IoU:           {s['mean_iou']:.3f}\n"
            f"  Centroid Hit Rate:  {s['centroid_hit_rate']:.3f}\n"
            f"  Oracle Hit @1:      {s['oracle_hit_rate@1']:.3f}\n"
            f"  Oracle Hit @3:      {s['oracle_hit_rate@3']:.3f}\n"
            f"  Oracle Hit @5:      {s['oracle_hit_rate@5']:.3f}\n"
            f"  Mean Latency:       {s['mean_latency_ms']:.1f}ms"
        )


def evaluate_grounder(
    grounder: "GroundingModule",
    test_cases: list[tuple["Image", str, tuple[float, float, float, float]]],
    k: int = 5,
) -> GroundingMetrics:
    """Evaluate a grounding module on test cases.

    Args:
        grounder: GroundingModule to evaluate.
        test_cases: List of (image, target_description, ground_truth_bbox) tuples.
        k: Number of candidates to request from grounder.

    Returns:
        GroundingMetrics with aggregated results.
    """
    metrics = GroundingMetrics()

    for image, target_desc, gt_bbox in test_cases:
        start = time.perf_counter()
        candidates = grounder.ground(image, target_desc, k=k)
        latency_ms = (time.perf_counter() - start) * 1000

        result = GroundingResult(
            target_description=target_desc,
            ground_truth_bbox=gt_bbox,
            predicted_candidates=candidates,
            latency_ms=latency_ms,
        )
        metrics.results.append(result)

    return metrics


def evaluate_grounder_on_episode(
    grounder: "GroundingModule",
    episode: "Episode",
    k: int = 5,
) -> GroundingMetrics:
    """Evaluate a grounding module on an Episode's click actions.

    Only evaluates steps with click actions that have ground-truth bboxes.

    Args:
        grounder: GroundingModule to evaluate.
        episode: Episode with Steps containing Actions with bboxes.
        k: Number of candidates to request.

    Returns:
        GroundingMetrics for click actions with bboxes.
    """
    from PIL import Image

    from openadapt_ml.schema import Episode, ActionType

    test_cases = []

    for step in episode.steps:
        action = step.action

        # Get action type as string for comparison
        action_type_str = action.type.value if isinstance(action.type, ActionType) else action.type

        # Only evaluate clicks with bboxes
        if action_type_str not in ("click", "double_click"):
            continue

        # Check for bbox - in new schema, bbox is in element.bounds or raw
        bbox = None
        if action.element and action.element.bounds:
            b = action.element.bounds
            bbox = (b.x, b.y, b.x + b.width, b.y + b.height)
        elif action.raw and "bbox" in action.raw:
            bbox = action.raw["bbox"]

        if bbox is None:
            continue
        if step.observation.screenshot_path is None:
            continue

        # Load image
        try:
            image = Image.open(step.observation.screenshot_path)
        except Exception:
            continue

        # Create target description from reasoning or action coordinates
        coords_x, coords_y = None, None
        if action.normalized_coordinates:
            coords_x, coords_y = action.normalized_coordinates
        if coords_x is not None and coords_y is not None:
            target_desc = step.reasoning or f"element at ({coords_x:.2f}, {coords_y:.2f})"
        else:
            target_desc = step.reasoning or "target element"

        test_cases.append((image, target_desc, bbox))

    return evaluate_grounder(grounder, test_cases, k=k)
