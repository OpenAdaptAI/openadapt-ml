"""Experimental conditions for the Representation Shootout.

Defines the three conditions:
- Condition A: Raw Coordinates - Direct coordinate regression
- Condition B: Coordinates + Visual Cues - Enhanced with markers and zoom
- Condition C: Marks (Element IDs) - Element classification using SoM

Each condition implements:
1. Input preparation (screenshot augmentation, prompt construction)
2. Output parsing (model response to action dict)
3. Loss computation (for training)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openadapt_ml.experiments.representation_shootout.config import (
    ConditionConfig,
    ConditionName,
    MarksConfig,
    OutputFormat,
    VisualCuesConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """UI element with ID and bounding box for marks condition."""

    element_id: str
    role: str
    name: str | None
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized or pixels

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of element."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within this element's bbox."""
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class UIElementGraph:
    """Collection of UI elements for marks condition."""

    elements: list[UIElement]

    def get_element(self, element_id: str) -> UIElement | None:
        """Get element by ID."""
        for el in self.elements:
            if el.element_id == element_id:
                return el
        return None

    def find_element_at(self, x: float, y: float) -> UIElement | None:
        """Find element containing the given point."""
        for el in self.elements:
            if el.contains_point(x, y):
                return el
        return None

    def to_prompt_text(self, max_elements: int | None = None) -> str:
        """Format elements for text prompt.

        Returns:
            Formatted string like:
            [e1] button "Submit" at (0.4, 0.8)-(0.6, 0.85)
            [e17] textfield "Username" at (0.3, 0.4)-(0.7, 0.45)
        """
        lines = []
        elements_to_show = (
            self.elements[:max_elements] if max_elements else self.elements
        )
        for el in elements_to_show:
            name_part = f' "{el.name}"' if el.name else ""
            x1, y1, x2, y2 = el.bbox
            lines.append(
                f"[{el.element_id}] {el.role}{name_part} at ({x1:.2f}, {y1:.2f})-({x2:.2f}, {y2:.2f})"
            )
        return "\n".join(lines)


@dataclass
class Observation:
    """Observation data for input preparation.

    This is a simplified observation structure for the experiment.
    In production, use openadapt_ml.benchmarks.base.BenchmarkObservation.
    """

    screenshot_path: str | None = None
    screenshot_bytes: bytes | None = None
    screen_size: tuple[int, int] | None = None  # (width, height)
    ui_elements: UIElementGraph | None = None
    window_title: str | None = None
    url: str | None = None


@dataclass
class ActionHistory:
    """History of previous actions."""

    actions: list[dict[str, Any]]  # List of action dicts

    def to_prompt_text(self, max_steps: int = 5) -> str:
        """Format history for text prompt."""
        lines = []
        for i, action in enumerate(self.actions[-max_steps:], 1):
            action_type = action.get("type", "unknown").upper()
            if action_type == "CLICK":
                if "element_id" in action:
                    lines.append(f"{i}. CLICK([{action['element_id']}])")
                elif "x" in action and "y" in action:
                    lines.append(f"{i}. CLICK({action['x']:.3f}, {action['y']:.3f})")
                else:
                    lines.append(f"{i}. CLICK()")
            elif action_type == "TYPE":
                text = action.get("text", "")
                lines.append(f'{i}. TYPE("{text}")')
            elif action_type == "KEY":
                key = action.get("key", "")
                lines.append(f"{i}. KEY({key})")
            elif action_type == "SCROLL":
                direction = action.get("direction", "down")
                lines.append(f"{i}. SCROLL({direction})")
            else:
                lines.append(f"{i}. {action_type}()")
        return "\n".join(lines)


@dataclass
class PreparedInput:
    """Prepared input for the model.

    Attributes:
        screenshot_path: Path to (possibly augmented) screenshot.
        prompt: Text prompt for the model.
        metadata: Additional metadata for debugging/analysis.
    """

    screenshot_path: str | None
    prompt: str
    metadata: dict[str, Any] | None = None


@dataclass
class ParsedAction:
    """Parsed action from model output.

    Attributes:
        type: Action type (e.g., "click", "type", "scroll", "done").
        x: X coordinate (for coordinate-based outputs).
        y: Y coordinate (for coordinate-based outputs).
        element_id: Element ID (for marks-based outputs).
        text: Text to type (for type actions).
        raw_output: Original model output string.
        parse_error: Error message if parsing failed.
    """

    type: str
    x: float | None = None
    y: float | None = None
    element_id: str | None = None
    text: str | None = None
    raw_output: str | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to action dictionary."""
        result: dict[str, Any] = {"type": self.type}
        if self.x is not None:
            result["x"] = self.x
        if self.y is not None:
            result["y"] = self.y
        if self.element_id is not None:
            result["element_id"] = self.element_id
        if self.text is not None:
            result["text"] = self.text
        return result


class ConditionBase(ABC):
    """Abstract base class for experimental conditions.

    Each condition defines how to:
    1. Prepare input from observations (possibly augmenting screenshots)
    2. Parse model output into structured actions
    3. Compute training loss
    """

    def __init__(self, config: ConditionConfig):
        """Initialize condition.

        Args:
            config: Condition-specific configuration.
        """
        self.config = config

    @property
    def name(self) -> ConditionName:
        """Condition name."""
        return self.config.name

    @property
    def output_format(self) -> OutputFormat:
        """Expected output format."""
        return self.config.output_format

    @abstractmethod
    def prepare_input(
        self,
        observation: Observation,
        goal: str,
        history: ActionHistory | None = None,
    ) -> PreparedInput:
        """Prepare model input from observation.

        Args:
            observation: Current observation with screenshot and UI elements.
            goal: Task goal/instruction.
            history: Optional history of previous actions.

        Returns:
            PreparedInput with (possibly augmented) screenshot and prompt.
        """
        pass

    @abstractmethod
    def parse_output(self, model_output: str) -> ParsedAction:
        """Parse model output to structured action.

        Args:
            model_output: Raw string output from model.

        Returns:
            ParsedAction with extracted action information.
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        prediction: ParsedAction,
        ground_truth: dict[str, Any],
    ) -> float:
        """Compute training loss for a single sample.

        Args:
            prediction: Parsed prediction from model.
            ground_truth: Ground truth action dict with coordinates/element_id.

        Returns:
            Loss value (lower is better).
        """
        pass

    def _build_base_prompt(
        self,
        goal: str,
        history: ActionHistory | None = None,
    ) -> str:
        """Build the base prompt text (shared across conditions)."""
        parts = [f"GOAL: {goal}"]

        if self.config.include_history and history and history.actions:
            history_text = history.to_prompt_text(self.config.max_history_steps)
            parts.append(f"\nPREVIOUS ACTIONS:\n{history_text}")

        return "\n".join(parts)


class RawCoordsCondition(ConditionBase):
    """Condition A: Raw Coordinates.

    Input: Screenshot (unmodified) + goal + history
    Output: {"type": "CLICK", "x": float, "y": float}
    Training: Coordinate regression (MSE loss)
    """

    def prepare_input(
        self,
        observation: Observation,
        goal: str,
        history: ActionHistory | None = None,
    ) -> PreparedInput:
        """Prepare input without any screenshot augmentation."""
        prompt = self._build_base_prompt(goal, history)
        prompt += "\n\nAnalyze the screenshot and provide the next action."
        prompt += "\nRespond with: ACTION: CLICK(x, y) where x and y are normalized coordinates (0.0-1.0)"

        return PreparedInput(
            screenshot_path=observation.screenshot_path,
            prompt=prompt,
            metadata={"condition": "raw_coords"},
        )

    def parse_output(self, model_output: str) -> ParsedAction:
        """Parse coordinate output from model."""
        import re

        # Look for ACTION: CLICK(x, y) pattern
        action_match = re.search(
            r"ACTION:\s*CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
            model_output,
            re.IGNORECASE,
        )
        if action_match:
            try:
                x = float(action_match.group(1))
                y = float(action_match.group(2))
                return ParsedAction(type="click", x=x, y=y, raw_output=model_output)
            except ValueError as e:
                return ParsedAction(
                    type="click",
                    raw_output=model_output,
                    parse_error=f"Invalid coordinates: {e}",
                )

        # Try looser patterns
        coord_match = re.search(
            r"CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
            model_output,
            re.IGNORECASE,
        )
        if coord_match:
            try:
                x = float(coord_match.group(1))
                y = float(coord_match.group(2))
                return ParsedAction(type="click", x=x, y=y, raw_output=model_output)
            except ValueError:
                pass

        # Check for TYPE action
        type_match = re.search(
            r'TYPE\s*\(\s*["\'](.+?)["\']\s*\)', model_output, re.IGNORECASE
        )
        if type_match:
            return ParsedAction(
                type="type", text=type_match.group(1), raw_output=model_output
            )

        # Check for DONE action
        if re.search(r"DONE\s*\(\s*\)", model_output, re.IGNORECASE):
            return ParsedAction(type="done", raw_output=model_output)

        return ParsedAction(
            type="unknown",
            raw_output=model_output,
            parse_error="No action pattern found",
        )

    def compute_loss(
        self,
        prediction: ParsedAction,
        ground_truth: dict[str, Any],
    ) -> float:
        """Compute MSE loss between predicted and ground truth coordinates."""
        gt_x = ground_truth.get("x")
        gt_y = ground_truth.get("y")

        if gt_x is None or gt_y is None:
            logger.warning("Ground truth missing coordinates, returning max loss")
            return 1.0

        if prediction.x is None or prediction.y is None:
            # Prediction failed, return max loss
            return 1.0

        # MSE in normalized coordinate space
        mse = (prediction.x - gt_x) ** 2 + (prediction.y - gt_y) ** 2
        return mse


class CoordsCuesCondition(ConditionBase):
    """Condition B: Coordinates + Visual Cues.

    Input: Screenshot with red marker at click target + zoomed patch + goal
    Output: {"type": "CLICK", "x": float, "y": float}
    Training: Enhanced coordinate regression (MSE loss)

    Note: Visual cues are only added during training. At test time,
    the model must predict without the cues.
    """

    def __init__(self, config: ConditionConfig):
        super().__init__(config)
        self.visual_cues = config.visual_cues or VisualCuesConfig()

    def prepare_input(
        self,
        observation: Observation,
        goal: str,
        history: ActionHistory | None = None,
        target_coords: tuple[float, float] | None = None,
        is_training: bool = False,
    ) -> PreparedInput:
        """Prepare input with visual cues during training.

        Args:
            observation: Current observation.
            goal: Task goal.
            history: Action history.
            target_coords: Target (x, y) for training augmentation.
            is_training: Whether this is for training (add cues) or eval (no cues).
        """
        prompt = self._build_base_prompt(goal, history)

        augmented_path = observation.screenshot_path
        metadata: dict[str, Any] = {
            "condition": "coords_cues",
            "is_training": is_training,
        }

        if is_training and target_coords:
            # Add visual cues for training
            prompt += (
                "\n\nThe red marker and zoomed inset show the target click location."
            )
            prompt += "\nLearn to identify this location based on the UI context."

            # Augment screenshot (placeholder - actual implementation would use PIL/cv2)
            augmented_path = self._augment_screenshot(
                observation.screenshot_path,
                target_coords,
                observation.screen_size,
            )
            metadata["target_coords"] = target_coords
            metadata["augmented"] = True
        else:
            prompt += "\n\nAnalyze the screenshot and provide the next action."

        prompt += "\nRespond with: ACTION: CLICK(x, y) where x and y are normalized coordinates (0.0-1.0)"

        return PreparedInput(
            screenshot_path=augmented_path,
            prompt=prompt,
            metadata=metadata,
        )

    def _augment_screenshot(
        self,
        screenshot_path: str | None,
        target_coords: tuple[float, float],
        screen_size: tuple[int, int] | None,
    ) -> str | None:
        """Add visual cues to screenshot.

        This is a scaffolding implementation. Full implementation would:
        1. Draw red marker at target location
        2. Extract and magnify patch around target
        3. Overlay patch in corner opposite to target

        Args:
            screenshot_path: Path to original screenshot.
            target_coords: Normalized (x, y) target coordinates.
            screen_size: Screen dimensions for pixel conversion.

        Returns:
            Path to augmented screenshot.
        """
        if not screenshot_path:
            return None

        # Placeholder: In full implementation, would use PIL/cv2 to:
        # 1. Load image
        # 2. Draw red circle at target_coords
        # 3. Extract zoom patch
        # 4. Place patch in corner
        # 5. Save augmented image

        logger.debug(
            f"Would augment screenshot {screenshot_path} with marker at {target_coords}"
        )

        # For scaffolding, return original path
        # TODO: Implement actual augmentation
        return screenshot_path

    def parse_output(self, model_output: str) -> ParsedAction:
        """Parse coordinate output (same as RawCoords)."""
        # Reuse RawCoords parsing logic
        return RawCoordsCondition(self.config).parse_output(model_output)

    def compute_loss(
        self,
        prediction: ParsedAction,
        ground_truth: dict[str, Any],
    ) -> float:
        """Compute MSE loss (same as RawCoords)."""
        return RawCoordsCondition(self.config).compute_loss(prediction, ground_truth)


class MarksCondition(ConditionBase):
    """Condition C: Marks (Element IDs).

    Input: Screenshot with SoM overlay + UIElementGraph + goal
    Output: {"type": "CLICK", "element_id": "e17"}
    Training: Element classification (cross-entropy loss)
    """

    def __init__(self, config: ConditionConfig):
        super().__init__(config)
        self.marks_config = config.marks or MarksConfig()

    def prepare_input(
        self,
        observation: Observation,
        goal: str,
        history: ActionHistory | None = None,
    ) -> PreparedInput:
        """Prepare input with element marks overlay and UIElementGraph."""
        prompt = self._build_base_prompt(goal, history)

        # Add UIElementGraph text representation
        if observation.ui_elements:
            elements_text = observation.ui_elements.to_prompt_text(
                self.marks_config.max_elements
            )
            prompt += f"\n\nUI ELEMENTS:\n{elements_text}"
        else:
            prompt += "\n\nNo UI elements detected."

        prompt += "\n\nWhich element should be clicked?"
        prompt += (
            "\nRespond with: ACTION: CLICK([element_id]) e.g., ACTION: CLICK([e17])"
        )

        # Augment screenshot with marks overlay
        augmented_path = self._add_marks_overlay(
            observation.screenshot_path,
            observation.ui_elements,
            observation.screen_size,
        )

        return PreparedInput(
            screenshot_path=augmented_path,
            prompt=prompt,
            metadata={
                "condition": "marks",
                "num_elements": len(observation.ui_elements.elements)
                if observation.ui_elements
                else 0,
            },
        )

    def _add_marks_overlay(
        self,
        screenshot_path: str | None,
        ui_elements: UIElementGraph | None,
        screen_size: tuple[int, int] | None,
    ) -> str | None:
        """Add SoM-style marks overlay to screenshot.

        This is a scaffolding implementation. Full implementation would:
        1. Draw numbered labels on each element's bounding box
        2. Use consistent styling (font, colors)

        Args:
            screenshot_path: Path to original screenshot.
            ui_elements: UI elements to mark.
            screen_size: Screen dimensions.

        Returns:
            Path to screenshot with marks overlay.
        """
        if not screenshot_path:
            return None

        if not ui_elements:
            return screenshot_path

        # Placeholder: In full implementation, would use PIL to:
        # 1. Load image
        # 2. Draw colored box around each element
        # 3. Add element ID label
        # 4. Save marked image

        logger.debug(
            f"Would add marks overlay to {screenshot_path} "
            f"with {len(ui_elements.elements)} elements"
        )

        # For scaffolding, return original path
        # TODO: Implement actual overlay
        return screenshot_path

    def parse_output(self, model_output: str) -> ParsedAction:
        """Parse element ID output from model."""
        import re

        # Look for ACTION: CLICK([element_id]) pattern
        action_match = re.search(
            r"ACTION:\s*CLICK\s*\(\s*\[?\s*([a-zA-Z]?\d+)\s*\]?\s*\)",
            model_output,
            re.IGNORECASE,
        )
        if action_match:
            element_id = action_match.group(1)
            # Normalize element ID format
            if not element_id.startswith("e"):
                element_id = f"e{element_id}"
            return ParsedAction(
                type="click", element_id=element_id, raw_output=model_output
            )

        # Try looser patterns
        element_match = re.search(
            r"CLICK\s*\(\s*\[?\s*([a-zA-Z]?\d+)\s*\]?\s*\)",
            model_output,
            re.IGNORECASE,
        )
        if element_match:
            element_id = element_match.group(1)
            if not element_id.startswith("e"):
                element_id = f"e{element_id}"
            return ParsedAction(
                type="click", element_id=element_id, raw_output=model_output
            )

        # Check for element mentioned in text (e.g., "click element e17")
        text_match = re.search(r"\b[eE](\d+)\b", model_output)
        if text_match:
            return ParsedAction(
                type="click",
                element_id=f"e{text_match.group(1)}",
                raw_output=model_output,
            )

        # Check for TYPE action
        type_match = re.search(
            r'TYPE\s*\(\s*["\'](.+?)["\']\s*\)', model_output, re.IGNORECASE
        )
        if type_match:
            return ParsedAction(
                type="type", text=type_match.group(1), raw_output=model_output
            )

        # Check for DONE action
        if re.search(r"DONE\s*\(\s*\)", model_output, re.IGNORECASE):
            return ParsedAction(type="done", raw_output=model_output)

        return ParsedAction(
            type="unknown",
            raw_output=model_output,
            parse_error="No element ID pattern found",
        )

    def compute_loss(
        self,
        prediction: ParsedAction,
        ground_truth: dict[str, Any],
    ) -> float:
        """Compute classification loss.

        For scaffolding, this returns 0 for correct, 1 for incorrect.
        Full implementation would return proper cross-entropy loss.
        """
        gt_element_id = ground_truth.get("element_id")

        if gt_element_id is None:
            logger.warning("Ground truth missing element_id, returning max loss")
            return 1.0

        if prediction.element_id is None:
            # Prediction failed
            return 1.0

        # Normalize both IDs for comparison
        pred_id = prediction.element_id.lower().replace("e", "")
        gt_id = str(gt_element_id).lower().replace("e", "")

        return 0.0 if pred_id == gt_id else 1.0


def create_condition(config: ConditionConfig) -> ConditionBase:
    """Factory function to create condition from config.

    Args:
        config: Condition configuration.

    Returns:
        Appropriate ConditionBase subclass instance.

    Raises:
        ValueError: If condition name is unknown.
    """
    condition_map = {
        ConditionName.RAW_COORDS: RawCoordsCondition,
        ConditionName.COORDS_CUES: CoordsCuesCondition,
        ConditionName.MARKS: MarksCondition,
    }

    condition_cls = condition_map.get(config.name)
    if condition_cls is None:
        raise ValueError(f"Unknown condition name: {config.name}")

    return condition_cls(config)
