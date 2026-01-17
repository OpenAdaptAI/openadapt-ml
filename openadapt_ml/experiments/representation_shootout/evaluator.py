"""Evaluation under drift conditions for the Representation Shootout.

This module implements:
1. Drift transformations (resolution, translation, theme, scroll)
2. Metrics computation (click-hit rate, grounding accuracy, etc.)
3. Decision rule for recommendation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from openadapt_ml.experiments.representation_shootout.conditions import (
    ConditionBase,
    Observation,
    ParsedAction,
    UIElement,
    UIElementGraph,
)
from openadapt_ml.experiments.representation_shootout.config import (
    ConditionName,
    DriftConfig,
    DriftType,
    MetricName,
    ResolutionDriftParams,
    ScrollDriftParams,
    ThemeDriftParams,
    TranslationDriftParams,
)

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """A single evaluation sample.

    Attributes:
        sample_id: Unique identifier for this sample.
        observation: Observation data (screenshot, UI elements).
        goal: Task instruction.
        ground_truth: Ground truth action dict.
        drift_config: Applied drift configuration.
    """

    sample_id: str
    observation: Observation
    goal: str
    ground_truth: dict[str, Any]
    drift_config: DriftConfig | None = None


@dataclass
class SampleResult:
    """Result of evaluating a single sample.

    Attributes:
        sample_id: Sample identifier.
        condition: Condition that was evaluated.
        drift: Drift configuration applied.
        prediction: Parsed prediction from model.
        ground_truth: Ground truth action.
        metrics: Computed metrics for this sample.
    """

    sample_id: str
    condition: ConditionName
    drift: str
    prediction: ParsedAction
    ground_truth: dict[str, Any]
    metrics: dict[str, float]


@dataclass
class EvaluationResult:
    """Aggregated evaluation results for a condition under a drift.

    Attributes:
        condition: Condition evaluated.
        drift: Drift configuration.
        num_samples: Number of samples evaluated.
        metrics: Aggregated metrics (averages).
        sample_results: Individual sample results.
    """

    condition: ConditionName
    drift: str
    num_samples: int
    metrics: dict[str, float]
    sample_results: list[SampleResult] = field(default_factory=list)


@dataclass
class Recommendation:
    """Final recommendation from the experiment.

    Attributes:
        recommended: Recommended approach ("COORDINATES" or "MARKS").
        reason: Explanation for the recommendation.
        coords_cues_avg: Average performance of Coords+Cues across drifts.
        marks_avg: Average performance of Marks across drifts.
        tolerance: Tolerance threshold used for decision.
        detailed_comparison: Per-drift comparison data.
    """

    recommended: str  # "COORDINATES" or "MARKS"
    reason: str
    coords_cues_avg: float
    marks_avg: float
    tolerance: float
    detailed_comparison: dict[str, dict[str, float]] = field(default_factory=dict)


class DriftTransformer:
    """Applies drift transformations to samples."""

    @staticmethod
    def apply_drift(
        observation: Observation,
        ground_truth: dict[str, Any],
        drift_config: DriftConfig,
    ) -> tuple[Observation, dict[str, Any]]:
        """Apply drift transformation to observation and ground truth.

        Args:
            observation: Original observation.
            ground_truth: Original ground truth action.
            drift_config: Drift to apply.

        Returns:
            Tuple of (transformed_observation, transformed_ground_truth).
        """
        if drift_config.is_canonical:
            return observation, ground_truth

        if drift_config.drift_type == DriftType.RESOLUTION:
            return DriftTransformer._apply_resolution_drift(
                observation,
                ground_truth,
                drift_config.params,  # type: ignore
            )
        elif drift_config.drift_type == DriftType.TRANSLATION:
            return DriftTransformer._apply_translation_drift(
                observation,
                ground_truth,
                drift_config.params,  # type: ignore
            )
        elif drift_config.drift_type == DriftType.THEME:
            return DriftTransformer._apply_theme_drift(
                observation,
                ground_truth,
                drift_config.params,  # type: ignore
            )
        elif drift_config.drift_type == DriftType.SCROLL:
            return DriftTransformer._apply_scroll_drift(
                observation,
                ground_truth,
                drift_config.params,  # type: ignore
            )
        else:
            logger.warning(f"Unknown drift type: {drift_config.drift_type}")
            return observation, ground_truth

    @staticmethod
    def _apply_resolution_drift(
        observation: Observation,
        ground_truth: dict[str, Any],
        params: ResolutionDriftParams,
    ) -> tuple[Observation, dict[str, Any]]:
        """Apply resolution scaling.

        For normalized coordinates, no transformation is needed (they scale automatically).
        For pixel coordinates, scale by the factor.
        For UI elements, scale bounding boxes.
        """
        scale = params.scale

        # Create new observation with scaled screen size
        new_screen_size = None
        if observation.screen_size:
            w, h = observation.screen_size
            new_screen_size = (int(w * scale), int(h * scale))

        # Scale UI element bboxes if they are in pixels
        new_ui_elements = None
        if observation.ui_elements:
            new_elements = []
            for el in observation.ui_elements.elements:
                # Assuming bboxes are normalized (0-1), no scaling needed
                # If they were pixels, we would scale them here
                new_elements.append(el)
            new_ui_elements = UIElementGraph(elements=new_elements)

        new_observation = Observation(
            screenshot_path=observation.screenshot_path,  # Would need actual resize
            screenshot_bytes=observation.screenshot_bytes,
            screen_size=new_screen_size,
            ui_elements=new_ui_elements,
            window_title=observation.window_title,
            url=observation.url,
        )

        # Ground truth coordinates are normalized, so no change needed
        # If they were pixels, we would scale them
        new_ground_truth = ground_truth.copy()

        logger.debug(f"Applied resolution drift {scale}x: {new_screen_size}")
        return new_observation, new_ground_truth

    @staticmethod
    def _apply_translation_drift(
        observation: Observation,
        ground_truth: dict[str, Any],
        params: TranslationDriftParams,
    ) -> tuple[Observation, dict[str, Any]]:
        """Apply window translation.

        This shifts the window position while keeping the UI elements
        in their relative positions within the window.
        """
        offset_x = params.offset_x
        offset_y = params.offset_y

        # For normalized coordinates within the window, no change is needed
        # The translation affects where the window is on screen, but not
        # the relative positions within the window

        # However, if coordinates are screen-absolute, we need to adjust
        # For this experiment, we assume window-relative normalized coords

        new_ground_truth = ground_truth.copy()

        # If ground truth has screen-absolute coordinates, adjust them
        if "screen_x" in ground_truth and "screen_y" in ground_truth:
            # Convert pixel offset to normalized offset
            if observation.screen_size:
                w, h = observation.screen_size
                norm_offset_x = offset_x / w
                norm_offset_y = offset_y / h
                new_ground_truth["screen_x"] = ground_truth["screen_x"] + norm_offset_x
                new_ground_truth["screen_y"] = ground_truth["screen_y"] + norm_offset_y

        logger.debug(f"Applied translation drift: ({offset_x}, {offset_y})")
        return observation, new_ground_truth

    @staticmethod
    def _apply_theme_drift(
        observation: Observation,
        ground_truth: dict[str, Any],
        params: ThemeDriftParams,
    ) -> tuple[Observation, dict[str, Any]]:
        """Apply theme change.

        Theme changes affect visual appearance but not coordinates.
        Full implementation would load theme-variant screenshots.
        """
        theme = params.theme

        # For scaffolding, we don't transform the screenshot
        # Full implementation would:
        # 1. Load a pre-recorded screenshot in the target theme, OR
        # 2. Apply synthetic color transformations

        logger.debug(f"Applied theme drift: {theme}")
        return observation, ground_truth

    @staticmethod
    def _apply_scroll_drift(
        observation: Observation,
        ground_truth: dict[str, Any],
        params: ScrollDriftParams,
    ) -> tuple[Observation, dict[str, Any]]:
        """Apply scroll offset.

        Scroll changes the visible portion of the page, affecting
        which elements are visible and their y-coordinates.
        """
        offset_y = params.offset_y

        # Adjust UI element bboxes for scroll
        new_ui_elements = None
        if observation.ui_elements and observation.screen_size:
            _, screen_h = observation.screen_size
            norm_offset = offset_y / screen_h

            new_elements = []
            for el in observation.ui_elements.elements:
                x1, y1, x2, y2 = el.bbox
                # Shift y coordinates up by scroll amount
                new_y1 = y1 - norm_offset
                new_y2 = y2 - norm_offset

                # Only include elements still visible on screen
                if new_y2 > 0 and new_y1 < 1:
                    new_elements.append(
                        UIElement(
                            element_id=el.element_id,
                            role=el.role,
                            name=el.name,
                            bbox=(x1, max(0, new_y1), x2, min(1, new_y2)),
                        )
                    )

            new_ui_elements = UIElementGraph(elements=new_elements)

        new_observation = Observation(
            screenshot_path=observation.screenshot_path,  # Would need scroll-shifted image
            screenshot_bytes=observation.screenshot_bytes,
            screen_size=observation.screen_size,
            ui_elements=new_ui_elements,
            window_title=observation.window_title,
            url=observation.url,
        )

        # Adjust ground truth coordinates
        new_ground_truth = ground_truth.copy()
        if "y" in ground_truth and observation.screen_size:
            _, screen_h = observation.screen_size
            norm_offset = offset_y / screen_h
            new_ground_truth["y"] = ground_truth["y"] - norm_offset

        logger.debug(f"Applied scroll drift: {offset_y}px")
        return new_observation, new_ground_truth


def compute_metrics(
    prediction: ParsedAction,
    ground_truth: dict[str, Any],
    ui_elements: UIElementGraph | None = None,
) -> dict[str, float]:
    """Compute all metrics for a single prediction.

    Args:
        prediction: Parsed prediction from model.
        ground_truth: Ground truth action dict with coordinates/element_id.
        ui_elements: UI elements (needed for click-hit computation).

    Returns:
        Dict of metric name to value.
    """
    metrics: dict[str, float] = {}

    # Click-Hit Rate: Is predicted coordinate within target element bbox?
    if prediction.type == "click":
        hit = 0.0

        if prediction.x is not None and prediction.y is not None:
            # Coordinate-based prediction
            target_bbox = ground_truth.get("target_bbox")
            if target_bbox:
                x1, y1, x2, y2 = target_bbox
                if x1 <= prediction.x <= x2 and y1 <= prediction.y <= y2:
                    hit = 1.0

            # Also check if coordinates are within the target element from ui_elements
            elif ui_elements and ground_truth.get("element_id"):
                target_el = ui_elements.get_element(ground_truth["element_id"])
                if target_el and target_el.contains_point(prediction.x, prediction.y):
                    hit = 1.0

        elif prediction.element_id is not None and ui_elements:
            # Element-based prediction - find element and check if it matches target
            pred_el = ui_elements.get_element(prediction.element_id)
            gt_el_id = ground_truth.get("element_id")
            if pred_el and gt_el_id:
                # Normalize IDs for comparison
                pred_id = prediction.element_id.lower().replace("e", "")
                gt_id = str(gt_el_id).lower().replace("e", "")
                if pred_id == gt_id:
                    hit = 1.0

        metrics[MetricName.CLICK_HIT_RATE.value] = hit

    # Grounding Top-1 Accuracy: Is predicted element ID correct?
    if prediction.element_id is not None:
        gt_el_id = ground_truth.get("element_id")
        if gt_el_id:
            pred_id = prediction.element_id.lower().replace("e", "")
            gt_id = str(gt_el_id).lower().replace("e", "")
            metrics[MetricName.GROUNDING_TOP1_ACCURACY.value] = (
                1.0 if pred_id == gt_id else 0.0
            )
        else:
            metrics[MetricName.GROUNDING_TOP1_ACCURACY.value] = 0.0

    # Coordinate Distance: L2 distance to target (normalized)
    gt_x = ground_truth.get("x")
    gt_y = ground_truth.get("y")

    if gt_x is not None and gt_y is not None:
        if prediction.x is not None and prediction.y is not None:
            distance = math.sqrt(
                (prediction.x - gt_x) ** 2 + (prediction.y - gt_y) ** 2
            )
        else:
            # If prediction failed or is element-based, compute distance from element center
            if prediction.element_id and ui_elements:
                pred_el = ui_elements.get_element(prediction.element_id)
                if pred_el:
                    cx, cy = pred_el.center
                    distance = math.sqrt((cx - gt_x) ** 2 + (cy - gt_y) ** 2)
                else:
                    distance = math.sqrt(2)  # Max normalized distance
            else:
                distance = math.sqrt(2)  # Max normalized distance

        metrics[MetricName.COORD_DISTANCE.value] = distance

    return metrics


def aggregate_metrics(sample_results: list[SampleResult]) -> dict[str, float]:
    """Aggregate metrics across multiple samples.

    Args:
        sample_results: List of individual sample results.

    Returns:
        Dict of metric name to averaged value.
    """
    if not sample_results:
        return {}

    # Collect all metrics
    all_metrics: dict[str, list[float]] = {}
    for result in sample_results:
        for metric_name, value in result.metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Compute averages
    aggregated = {}
    for metric_name, values in all_metrics.items():
        aggregated[metric_name] = sum(values) / len(values)

    return aggregated


class DriftEvaluator:
    """Evaluates conditions under drift conditions.

    This class orchestrates the evaluation process:
    1. Apply drift transformations to samples
    2. Generate predictions using conditions
    3. Compute metrics
    4. Aggregate results
    """

    def __init__(
        self,
        conditions: dict[ConditionName, ConditionBase],
        drift_configs: list[DriftConfig],
    ):
        """Initialize evaluator.

        Args:
            conditions: Map of condition name to condition instance.
            drift_configs: List of drift configurations to test.
        """
        self.conditions = conditions
        self.drift_configs = drift_configs
        self._canonical_results: dict[ConditionName, dict[str, float]] = {}

    def evaluate_sample(
        self,
        condition: ConditionBase,
        sample: Sample,
        drift_config: DriftConfig,
        model_output: str,
    ) -> SampleResult:
        """Evaluate a single sample under a drift condition.

        Args:
            condition: Condition to use for evaluation.
            sample: Sample to evaluate.
            drift_config: Drift to apply.
            model_output: Raw model output to parse.

        Returns:
            SampleResult with metrics.
        """
        # Apply drift
        transformed_obs, transformed_gt = DriftTransformer.apply_drift(
            sample.observation, sample.ground_truth, drift_config
        )

        # Parse model output
        prediction = condition.parse_output(model_output)

        # Compute metrics
        metrics = compute_metrics(
            prediction, transformed_gt, transformed_obs.ui_elements
        )

        return SampleResult(
            sample_id=sample.sample_id,
            condition=condition.name,
            drift=drift_config.name,
            prediction=prediction,
            ground_truth=transformed_gt,
            metrics=metrics,
        )

    def evaluate_condition_under_drift(
        self,
        condition: ConditionBase,
        samples: list[Sample],
        drift_config: DriftConfig,
        model_outputs: list[str],
    ) -> EvaluationResult:
        """Evaluate a condition on all samples under a drift.

        Args:
            condition: Condition to evaluate.
            samples: Samples to evaluate.
            drift_config: Drift to apply.
            model_outputs: Model outputs corresponding to samples.

        Returns:
            EvaluationResult with aggregated metrics.
        """
        sample_results = []
        for sample, output in zip(samples, model_outputs):
            result = self.evaluate_sample(condition, sample, drift_config, output)
            sample_results.append(result)

        aggregated = aggregate_metrics(sample_results)

        return EvaluationResult(
            condition=condition.name,
            drift=drift_config.name,
            num_samples=len(samples),
            metrics=aggregated,
            sample_results=sample_results,
        )

    def compute_robustness_scores(
        self,
        results: list[EvaluationResult],
        primary_metric: str = MetricName.CLICK_HIT_RATE.value,
    ) -> dict[ConditionName, dict[str, float]]:
        """Compute robustness scores relative to canonical baseline.

        Args:
            results: Evaluation results across conditions and drifts.
            primary_metric: Metric to use for robustness computation.

        Returns:
            Dict mapping condition to dict of drift to robustness score.
        """
        # Group results by condition
        by_condition: dict[ConditionName, list[EvaluationResult]] = {}
        for r in results:
            if r.condition not in by_condition:
                by_condition[r.condition] = []
            by_condition[r.condition].append(r)

        robustness_scores: dict[ConditionName, dict[str, float]] = {}

        for condition, cond_results in by_condition.items():
            # Find canonical result
            canonical_result = next(
                (r for r in cond_results if r.drift == "canonical"), None
            )
            if not canonical_result:
                logger.warning(f"No canonical result for condition {condition}")
                continue

            canonical_value = canonical_result.metrics.get(primary_metric, 0)
            if canonical_value == 0:
                canonical_value = 1e-6  # Avoid division by zero

            robustness_scores[condition] = {}
            for r in cond_results:
                if r.drift == "canonical":
                    robustness_scores[condition][r.drift] = 1.0
                else:
                    drift_value = r.metrics.get(primary_metric, 0)
                    robustness_scores[condition][r.drift] = (
                        drift_value / canonical_value
                    )

        return robustness_scores


def make_recommendation(
    results: list[EvaluationResult],
    tolerance: float = 0.05,
    primary_metric: str = MetricName.CLICK_HIT_RATE.value,
) -> Recommendation:
    """Make recommendation based on evaluation results.

    Decision rule (from design doc):
    - If Coords+Cues within 5% of Marks under drift -> choose Coordinates
    - Otherwise -> choose Marks

    Args:
        results: Evaluation results across all conditions and drifts.
        tolerance: Tolerance threshold for decision (default 5%).
        primary_metric: Metric to use for comparison.

    Returns:
        Recommendation with explanation.
    """
    # Group results by condition and compute averages across drifts
    by_condition: dict[ConditionName, list[float]] = {}
    detailed_comparison: dict[str, dict[str, float]] = {}

    for r in results:
        if r.condition not in by_condition:
            by_condition[r.condition] = []

        metric_value = r.metrics.get(primary_metric, 0)
        by_condition[r.condition].append(metric_value)

        # Track detailed comparison
        drift_key = r.drift
        if drift_key not in detailed_comparison:
            detailed_comparison[drift_key] = {}
        detailed_comparison[drift_key][r.condition.value] = metric_value

    # Compute averages
    condition_averages: dict[ConditionName, float] = {}
    for condition, values in by_condition.items():
        condition_averages[condition] = sum(values) / len(values) if values else 0

    # Get averages for decision
    coords_cues_avg = condition_averages.get(ConditionName.COORDS_CUES, 0)
    marks_avg = condition_averages.get(ConditionName.MARKS, 0)

    # Apply decision rule
    if coords_cues_avg >= marks_avg - tolerance:
        recommended = "COORDINATES"
        reason = (
            f"Coords+Cues ({coords_cues_avg:.1%}) is within {tolerance * 100}% of "
            f"Marks ({marks_avg:.1%}) under drift. Coordinates approach is simpler "
            "and doesn't require element detection pipeline."
        )
    else:
        recommended = "MARKS"
        gap = marks_avg - coords_cues_avg
        reason = (
            f"Marks ({marks_avg:.1%}) outperforms Coords+Cues ({coords_cues_avg:.1%}) "
            f"by {gap:.1%} (>{tolerance * 100}%) under drift. Element-based approach "
            "provides better robustness to UI changes."
        )

    return Recommendation(
        recommended=recommended,
        reason=reason,
        coords_cues_avg=coords_cues_avg,
        marks_avg=marks_avg,
        tolerance=tolerance,
        detailed_comparison=detailed_comparison,
    )
