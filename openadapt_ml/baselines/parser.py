"""Response parsing for baseline adapters.

Extracts structured actions from VLM responses with support for:
- JSON format extraction
- Function-call syntax (CLICK(x, y))
- PyAutoGUI format (OSWorld compatible)
- UFO format (Observation/Thought/ControlLabel)
- Element ID to coordinate normalization
- Robust fallback parsing

Based on patterns from:
- Claude Computer Use
- OSWorld benchmark
- Microsoft UFO
- Agent-S
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openadapt_ml.baselines.config import ScreenConfig

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """UI element with bounding box for coordinate conversion.

    Used to convert element_id actions to coordinate actions.
    """

    element_id: int | str
    role: str = ""
    name: str = ""
    bbox: tuple[float, float, float, float] | None = None  # (x1, y1, x2, y2)

    @property
    def center(self) -> tuple[float, float] | None:
        """Get center point of element."""
        if self.bbox is None:
            return None
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class ElementRegistry:
    """Registry of UI elements for element_id to coordinate conversion.

    Used by the parser to convert Track C (SoM) element IDs to
    Track A coordinates when needed.
    """

    elements: dict[int, UIElement] = field(default_factory=dict)
    screen_width: int = 1920
    screen_height: int = 1080
    is_normalized: bool = True  # Whether bbox coordinates are normalized (0-1)

    def add_element(
        self,
        element_id: int | str,
        bbox: tuple[float, float, float, float],
        role: str = "",
        name: str = "",
    ) -> None:
        """Add an element to the registry."""
        eid = int(element_id) if isinstance(element_id, str) else element_id
        self.elements[eid] = UIElement(element_id=eid, role=role, name=name, bbox=bbox)

    def get_element(self, element_id: int) -> UIElement | None:
        """Get element by ID."""
        return self.elements.get(element_id)

    def get_center_coords(
        self, element_id: int, normalize: bool = True
    ) -> tuple[float, float] | None:
        """Get center coordinates for an element.

        Args:
            element_id: Element ID to look up.
            normalize: Whether to return normalized (0-1) coordinates.

        Returns:
            (x, y) center coordinates, or None if element not found.
        """
        element = self.get_element(element_id)
        if element is None or element.bbox is None:
            return None

        center = element.center
        if center is None:
            return None

        x, y = center

        # Handle normalization
        if self.is_normalized and not normalize:
            # Convert from normalized to pixels
            x = x * self.screen_width
            y = y * self.screen_height
        elif not self.is_normalized and normalize:
            # Convert from pixels to normalized
            x = x / self.screen_width
            y = y / self.screen_height

        return (x, y)

    @classmethod
    def from_a11y_tree(
        cls,
        tree: dict[str, Any] | list[dict[str, Any]],
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> "ElementRegistry":
        """Build registry from accessibility tree.

        Args:
            tree: Accessibility tree as dict or list of element dicts.
            screen_width: Screen width for coordinate conversion.
            screen_height: Screen height for coordinate conversion.

        Returns:
            ElementRegistry with all elements from tree.
        """
        registry = cls(screen_width=screen_width, screen_height=screen_height)

        def process_node(node: dict[str, Any]) -> None:
            node_id = node.get("id", node.get("node_id", node.get("element_id")))
            if node_id is not None:
                try:
                    eid = int(
                        str(node_id).replace("e", "").replace("[", "").replace("]", "")
                    )
                    bbox = node.get("bbox", node.get("bounds"))
                    if bbox and len(bbox) >= 4:
                        registry.add_element(
                            element_id=eid,
                            bbox=tuple(bbox[:4]),
                            role=node.get("role", ""),
                            name=node.get("name", ""),
                        )
                except (ValueError, TypeError):
                    pass

            # Process children
            for child in node.get("children", []):
                if isinstance(child, dict):
                    process_node(child)

        if isinstance(tree, dict):
            process_node(tree)
        elif isinstance(tree, list):
            for node in tree:
                if isinstance(node, dict):
                    process_node(node)

        return registry


@dataclass
class ParsedAction:
    """Parsed action from model response.

    Attributes:
        action_type: Action type (click, type, key, scroll, done, wait, fail, unknown).
        x: X coordinate (normalized 0-1) for coordinate actions.
        y: Y coordinate (normalized 0-1) for coordinate actions.
        element_id: Element ID for SoM actions.
        text: Text content for type actions.
        key: Key name for key actions.
        modifiers: Key modifiers (ctrl, shift, alt) for key/hotkey actions.
        direction: Scroll direction for scroll actions.
        amount: Scroll amount for scroll actions.
        observation: Observed state description (ReAct/UFO format).
        thought: Reasoning text (ReAct/UFO format).
        plan: Multi-step plan (UFO format).
        status: Execution status (UFO format: CONTINUE, FINISH, ERROR).
        raw_response: Original model response.
        parse_error: Error message if parsing failed.
        confidence: Parser confidence score (0-1).
        metadata: Additional parsed data.
    """

    action_type: str
    x: float | None = None
    y: float | None = None
    element_id: int | None = None
    text: str | None = None
    key: str | None = None
    modifiers: list[str] | None = None
    direction: str | None = None
    amount: int | None = None
    observation: str | None = None
    thought: str | None = None
    plan: list[str] | None = None
    status: str | None = None
    raw_response: str | None = None
    parse_error: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the action was successfully parsed."""
        return self.parse_error is None and self.action_type != "unknown"

    @property
    def is_terminal(self) -> bool:
        """Check if this action terminates the episode."""
        return self.action_type in ("done", "fail")

    @property
    def has_coordinates(self) -> bool:
        """Check if action has coordinate data."""
        return self.x is not None and self.y is not None

    @property
    def has_element_id(self) -> bool:
        """Check if action has element ID."""
        return self.element_id is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to action dictionary for benchmark integration."""
        result: dict[str, Any] = {"type": self.action_type}

        if self.x is not None:
            result["x"] = self.x
        if self.y is not None:
            result["y"] = self.y
        if self.element_id is not None:
            result["element_id"] = self.element_id
        if self.text is not None:
            result["text"] = self.text
        if self.key is not None:
            result["key"] = self.key
        if self.modifiers:
            result["modifiers"] = self.modifiers
        if self.direction is not None:
            result["direction"] = self.direction
        if self.amount is not None:
            result["amount"] = self.amount
        if self.observation is not None:
            result["observation"] = self.observation
        if self.thought is not None:
            result["thought"] = self.thought
        if self.plan:
            result["plan"] = self.plan
        if self.status is not None:
            result["status"] = self.status

        return result

    def to_pyautogui(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> str:
        """Convert to PyAutoGUI code string.

        Args:
            screen_width: Screen width for coordinate conversion.
            screen_height: Screen height for coordinate conversion.

        Returns:
            PyAutoGUI code string.
        """
        if self.action_type == "click":
            if self.x is not None and self.y is not None:
                px = int(self.x * screen_width)
                py = int(self.y * screen_height)
                return f"pyautogui.click({px}, {py})"
            elif self.element_id is not None:
                return (
                    f"# CLICK element {self.element_id} (needs coordinate conversion)"
                )
        elif self.action_type == "type":
            text = self.text or ""
            return f"pyautogui.write('{text}')"
        elif self.action_type == "key":
            key = self.key or ""
            if self.modifiers:
                keys = ", ".join([f"'{k}'" for k in self.modifiers + [key]])
                return f"pyautogui.hotkey({keys})"
            return f"pyautogui.press('{key}')"
        elif self.action_type == "scroll":
            direction = self.direction or "down"
            amount = self.amount or 3
            clicks = -amount if direction == "down" else amount
            return f"pyautogui.scroll({clicks})"
        elif self.action_type == "done":
            return "DONE"
        elif self.action_type == "wait":
            return "WAIT"
        elif self.action_type == "fail":
            return "FAIL"

        return f"# Unknown action: {self.action_type}"

    def with_coordinates(
        self,
        x: float,
        y: float,
        source: str = "conversion",
    ) -> "ParsedAction":
        """Create a copy with coordinates added.

        Useful for converting element_id actions to coordinate actions.

        Args:
            x: X coordinate (normalized 0-1).
            y: Y coordinate (normalized 0-1).
            source: Source of coordinates for metadata.

        Returns:
            New ParsedAction with coordinates.
        """
        return ParsedAction(
            action_type=self.action_type,
            x=x,
            y=y,
            element_id=self.element_id,
            text=self.text,
            key=self.key,
            modifiers=self.modifiers,
            direction=self.direction,
            amount=self.amount,
            observation=self.observation,
            thought=self.thought,
            plan=self.plan,
            status=self.status,
            raw_response=self.raw_response,
            parse_error=self.parse_error,
            confidence=self.confidence,
            metadata={**self.metadata, "coord_source": source},
        )


class UnifiedResponseParser:
    """Parser for VLM responses across all tracks and formats.

    Supports:
    - JSON format: {"action": "CLICK", "x": 0.5, "y": 0.3}
    - Function format: CLICK(0.5, 0.3) or CLICK([17])
    - PyAutoGUI format: pyautogui.click(960, 540)
    - UFO format: {"Observation": ..., "Thought": ..., "ControlLabel": 17}
    - Mixed format: ReAct-style with thought + action

    Example:
        parser = UnifiedResponseParser()
        action = parser.parse('{"action": "CLICK", "x": 0.5, "y": 0.3}')
        print(action.x, action.y)  # 0.5, 0.3

        # With element registry for ID->coordinate conversion
        registry = ElementRegistry.from_a11y_tree(tree)
        parser = UnifiedResponseParser(element_registry=registry)
        action = parser.parse('{"action": "CLICK", "element_id": 17}')
        action = parser.resolve_element_id(action)
        print(action.x, action.y)  # Converted coordinates
    """

    def __init__(
        self,
        element_registry: ElementRegistry | None = None,
        screen_config: "ScreenConfig | None" = None,
        normalize_coordinates: bool = True,
    ):
        """Initialize parser.

        Args:
            element_registry: Optional registry for element_id conversion.
            screen_config: Optional screen configuration for coordinate handling.
            normalize_coordinates: Whether to normalize coordinates to 0-1.
        """
        self.element_registry = element_registry
        self.screen_config = screen_config
        self.normalize_coordinates = normalize_coordinates

        # Default screen dimensions
        self._screen_width = screen_config.width if screen_config else 1920
        self._screen_height = screen_config.height if screen_config else 1080

    def parse(self, response: str) -> ParsedAction:
        """Parse model response into structured action.

        Tries multiple parsing strategies in order:
        1. JSON extraction (most reliable)
        2. PyAutoGUI code patterns
        3. Function-style patterns (CLICK, TYPE, etc.)
        4. Special keywords (DONE, WAIT, FAIL)

        Args:
            response: Raw model response string.

        Returns:
            ParsedAction with extracted fields.
        """
        if not response:
            return ParsedAction(
                action_type="unknown",
                raw_response=response,
                parse_error="Empty response",
            )

        response = response.strip()

        # Try JSON first (most structured)
        action = self._try_json_parse(response)
        if action.is_valid:
            action.raw_response = response
            return action

        # Try PyAutoGUI format
        action = self._try_pyautogui_parse(response)
        if action.is_valid:
            action.raw_response = response
            return action

        # Try function-call patterns
        action = self._try_regex_parse(response)
        if action.is_valid:
            action.raw_response = response
            return action

        # Try special keywords
        action = self._try_keyword_parse(response)
        if action.is_valid:
            action.raw_response = response
            return action

        # Return unknown action with error
        return ParsedAction(
            action_type="unknown",
            raw_response=response,
            parse_error="No action pattern found in response",
            confidence=0.0,
        )

    def _try_json_parse(self, response: str) -> ParsedAction:
        """Try to extract and parse JSON from response."""
        # Try to find JSON object in response
        json_patterns = [
            r"```json\s*(\{[^`]*\})\s*```",  # Markdown code block
            r"```\s*(\{[^`]*\})\s*```",  # Plain code block
            r"(\{[^{}]*\})",  # Simple JSON object
            r"(\{[^{}]*\{[^{}]*\}[^{}]*\})",  # Nested JSON (max 1 level)
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    action = self._dict_to_action(data)
                    if action.is_valid:
                        return action
                except json.JSONDecodeError:
                    continue

        return ParsedAction(action_type="unknown", parse_error="No valid JSON found")

    def _dict_to_action(self, data: dict[str, Any]) -> ParsedAction:
        """Convert parsed dict to ParsedAction.

        Handles multiple formats:
        - Standard: {"action": "CLICK", "x": 0.5, "y": 0.3}
        - UFO: {"Observation": ..., "Thought": ..., "ControlLabel": 17}
        - ReAct: {"observation": ..., "thought": ..., "action": "CLICK"}
        """
        # Extract ReAct/UFO fields first
        observation = data.get("observation", data.get("Observation"))
        thought = data.get("thought", data.get("Thought"))
        plan = data.get("plan", data.get("Plan"))
        status = data.get("status", data.get("Status"))

        # Get action type (handle various key names)
        action_type = (
            data.get("action", "")
            or data.get("type", "")
            or data.get("Function", "")  # UFO format
        ).lower()

        # Handle UFO ControlLabel as element click
        control_label = data.get("ControlLabel", data.get("control_label"))
        if control_label is not None and not action_type:
            action_type = "click"

        if action_type == "click":
            # Check for element_id first (SoM/UFO)
            element_id = data.get("element_id", data.get("ControlLabel"))
            if element_id is not None:
                return ParsedAction(
                    action_type="click",
                    element_id=self._normalize_element_id(element_id),
                    observation=observation,
                    thought=thought,
                    plan=plan,
                    status=status,
                )

            # Then check for coordinates
            if "x" in data and "y" in data:
                x, y = self._normalize_coords(float(data["x"]), float(data["y"]))
                return ParsedAction(
                    action_type="click",
                    x=x,
                    y=y,
                    observation=observation,
                    thought=thought,
                    plan=plan,
                    status=status,
                )

            # Check coordinate array format
            if "coordinate" in data:
                coords = data["coordinate"]
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    x, y = self._normalize_coords(float(coords[0]), float(coords[1]))
                    return ParsedAction(
                        action_type="click",
                        x=x,
                        y=y,
                        observation=observation,
                        thought=thought,
                    )

            return ParsedAction(
                action_type="click",
                parse_error="CLICK missing coordinates or element_id",
                observation=observation,
                thought=thought,
            )

        elif action_type in ("type", "input_text", "write"):
            text = data.get("text", "")
            # Handle UFO Args format
            args = data.get("Args", data.get("args", []))
            if not text and args:
                text = args[0] if args else ""
            return ParsedAction(
                action_type="type",
                text=text,
                observation=observation,
                thought=thought,
            )

        elif action_type in ("key", "press", "hotkey"):
            key = data.get("key", "")
            modifiers = data.get("modifiers", [])

            # Handle UFO Args format for hotkey
            args = data.get("Args", data.get("args", []))
            if args and not key:
                if len(args) == 1:
                    key = args[0]
                else:
                    modifiers = args[:-1]
                    key = args[-1]

            return ParsedAction(
                action_type="key",
                key=key,
                modifiers=modifiers if modifiers else None,
                observation=observation,
                thought=thought,
            )

        elif action_type == "scroll":
            direction = data.get("direction", data.get("scroll_direction", "down"))
            amount = data.get("amount", data.get("scroll_amount", 3))

            # Handle UFO Args format
            args = data.get("Args", data.get("args", []))
            if args and not direction:
                direction = args[0] if args else "down"

            return ParsedAction(
                action_type="scroll",
                direction=direction,
                amount=amount,
                observation=observation,
                thought=thought,
            )

        elif action_type in ("done", "finish", "complete"):
            return ParsedAction(
                action_type="done",
                status="FINISH",
                observation=observation,
                thought=thought,
            )

        elif action_type in ("wait", "pause"):
            return ParsedAction(
                action_type="wait",
                observation=observation,
                thought=thought,
            )

        elif action_type in ("fail", "error", "impossible"):
            return ParsedAction(
                action_type="fail",
                status="ERROR",
                observation=observation,
                thought=thought,
            )

        else:
            return ParsedAction(
                action_type="unknown",
                parse_error=f"Unknown action type: {action_type}",
                observation=observation,
                thought=thought,
            )

    def _try_pyautogui_parse(self, response: str) -> ParsedAction:
        """Try to parse PyAutoGUI-style code."""
        # pyautogui.click(x, y)
        click_match = re.search(
            r"pyautogui\.click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
            response,
            re.IGNORECASE,
        )
        if click_match:
            x = int(click_match.group(1))
            y = int(click_match.group(2))
            x, y = self._normalize_coords(x, y)
            return ParsedAction(action_type="click", x=x, y=y)

        # pyautogui.doubleClick(x, y)
        dclick_match = re.search(
            r"pyautogui\.doubleClick\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
            response,
            re.IGNORECASE,
        )
        if dclick_match:
            x = int(dclick_match.group(1))
            y = int(dclick_match.group(2))
            x, y = self._normalize_coords(x, y)
            return ParsedAction(
                action_type="click",
                x=x,
                y=y,
                metadata={"double_click": True},
            )

        # pyautogui.write('text')
        write_match = re.search(
            r'pyautogui\.write\s*\(\s*[\'"](.+?)[\'"]\s*\)',
            response,
            re.IGNORECASE,
        )
        if write_match:
            return ParsedAction(action_type="type", text=write_match.group(1))

        # pyautogui.press('key')
        press_match = re.search(
            r'pyautogui\.press\s*\(\s*[\'"](.+?)[\'"]\s*\)',
            response,
            re.IGNORECASE,
        )
        if press_match:
            return ParsedAction(action_type="key", key=press_match.group(1))

        # pyautogui.hotkey('key1', 'key2')
        hotkey_match = re.search(
            r"pyautogui\.hotkey\s*\(\s*(.+?)\s*\)",
            response,
            re.IGNORECASE,
        )
        if hotkey_match:
            keys_str = hotkey_match.group(1)
            # Extract keys from quotes
            keys = re.findall(r'[\'"]([^\'"]+)[\'"]', keys_str)
            if keys:
                modifiers = keys[:-1] if len(keys) > 1 else None
                key = keys[-1]
                return ParsedAction(
                    action_type="key",
                    key=key,
                    modifiers=modifiers,
                )

        # pyautogui.scroll(amount)
        scroll_match = re.search(
            r"pyautogui\.scroll\s*\(\s*(-?\d+)\s*\)",
            response,
            re.IGNORECASE,
        )
        if scroll_match:
            clicks = int(scroll_match.group(1))
            direction = "up" if clicks > 0 else "down"
            return ParsedAction(
                action_type="scroll",
                direction=direction,
                amount=abs(clicks),
            )

        return ParsedAction(
            action_type="unknown", parse_error="No PyAutoGUI pattern matched"
        )

    def _try_regex_parse(self, response: str) -> ParsedAction:
        """Try regex patterns for function-style actions."""
        # CLICK(x, y) - normalized coordinates
        click_norm = re.search(
            r"CLICK\s*\(\s*(0?\.\d+)\s*,\s*(0?\.\d+)\s*\)",
            response,
            re.IGNORECASE,
        )
        if click_norm:
            return ParsedAction(
                action_type="click",
                x=float(click_norm.group(1)),
                y=float(click_norm.group(2)),
            )

        # CLICK(x, y) - larger numbers (pixels)
        click_pixel = re.search(
            r"CLICK\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)",
            response,
            re.IGNORECASE,
        )
        if click_pixel:
            x = float(click_pixel.group(1))
            y = float(click_pixel.group(2))
            x, y = self._normalize_coords(x, y)
            return ParsedAction(action_type="click", x=x, y=y)

        # CLICK([id]) - element ID
        click_element = re.search(
            r"CLICK\s*\(\s*\[\s*(\d+)\s*\]\s*\)",
            response,
            re.IGNORECASE,
        )
        if click_element:
            return ParsedAction(
                action_type="click",
                element_id=int(click_element.group(1)),
            )

        # CLICK(id) without brackets
        click_id = re.search(
            r"CLICK\s*\(\s*(\d+)\s*\)",
            response,
            re.IGNORECASE,
        )
        if click_id:
            # Check if it's likely an element ID (small number) vs coordinate
            val = int(click_id.group(1))
            if val < 1000:  # Likely element ID
                return ParsedAction(action_type="click", element_id=val)

        # TYPE("text") or TYPE('text')
        type_match = re.search(
            r'TYPE\s*\(\s*["\'](.+?)["\']\s*\)',
            response,
            re.IGNORECASE,
        )
        if type_match:
            return ParsedAction(action_type="type", text=type_match.group(1))

        # KEY(key) or KEY(mod+key)
        key_match = re.search(
            r"KEY\s*\(\s*([a-zA-Z0-9_+]+)\s*\)",
            response,
            re.IGNORECASE,
        )
        if key_match:
            key_str = key_match.group(1).lower()
            if "+" in key_str:
                parts = key_str.split("+")
                modifiers = parts[:-1]
                key = parts[-1]
                return ParsedAction(action_type="key", key=key, modifiers=modifiers)
            return ParsedAction(action_type="key", key=key_str)

        # SCROLL(direction) or SCROLL(direction, amount)
        scroll_match = re.search(
            r"SCROLL\s*\(\s*([a-zA-Z]+)(?:\s*,\s*(\d+))?\s*\)",
            response,
            re.IGNORECASE,
        )
        if scroll_match:
            direction = scroll_match.group(1).lower()
            amount = int(scroll_match.group(2)) if scroll_match.group(2) else 3
            return ParsedAction(
                action_type="scroll", direction=direction, amount=amount
            )

        return ParsedAction(
            action_type="unknown", parse_error="No regex pattern matched"
        )

    def _try_keyword_parse(self, response: str) -> ParsedAction:
        """Try special keywords."""
        response_upper = response.upper().strip()

        # DONE() or just DONE
        if (
            re.search(r"\bDONE\s*\(\s*\)\s*$", response, re.IGNORECASE)
            or response_upper == "DONE"
        ):
            return ParsedAction(action_type="done")

        # WAIT() or WAIT
        if (
            re.search(r"\bWAIT\s*\(\s*\)\s*$", response, re.IGNORECASE)
            or response_upper == "WAIT"
        ):
            return ParsedAction(action_type="wait")

        # FAIL() or FAIL
        if (
            re.search(r"\bFAIL\s*\(\s*\)\s*$", response, re.IGNORECASE)
            or response_upper == "FAIL"
        ):
            return ParsedAction(action_type="fail")

        # Look for "task is complete" or similar phrases
        if re.search(
            r"task\s+(?:is\s+)?(?:complete|done|finished)", response, re.IGNORECASE
        ):
            return ParsedAction(
                action_type="done",
                confidence=0.7,
                metadata={"inferred": True},
            )

        return ParsedAction(action_type="unknown", parse_error="No keyword matched")

    def _normalize_coords(self, x: float, y: float) -> tuple[float, float]:
        """Normalize coordinates to 0-1 range if needed."""
        if not self.normalize_coordinates:
            return (x, y)

        # If coordinates are large, assume they're pixels
        if x > 1.5 or y > 1.5:
            x = x / self._screen_width
            y = y / self._screen_height

        # Clamp to valid range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        return (x, y)

    def _normalize_element_id(self, element_id: Any) -> int | None:
        """Normalize element_id to integer format."""
        if element_id is None:
            return None

        if isinstance(element_id, int):
            return element_id

        if isinstance(element_id, str):
            # Extract number from "e17", "[17]", "element_17" etc.
            match = re.search(r"\d+", element_id)
            if match:
                return int(match.group())

        try:
            return int(element_id)
        except (ValueError, TypeError):
            return None

    def resolve_element_id(
        self,
        action: ParsedAction,
        registry: ElementRegistry | None = None,
    ) -> ParsedAction:
        """Convert element_id to coordinates if registry available.

        Args:
            action: ParsedAction with element_id.
            registry: Element registry (uses self.element_registry if None).

        Returns:
            ParsedAction with coordinates added if conversion succeeded,
            original action otherwise.
        """
        if not action.has_element_id or action.has_coordinates:
            return action

        reg = registry or self.element_registry
        if reg is None:
            return action

        coords = reg.get_center_coords(action.element_id, normalize=True)
        if coords is not None:
            return action.with_coordinates(
                x=coords[0],
                y=coords[1],
                source=f"element_{action.element_id}",
            )

        return action

    def parse_and_resolve(
        self,
        response: str,
        registry: ElementRegistry | None = None,
    ) -> ParsedAction:
        """Parse response and resolve element_id to coordinates.

        Convenience method that combines parse() and resolve_element_id().

        Args:
            response: Raw model response.
            registry: Optional element registry for ID conversion.

        Returns:
            ParsedAction with coordinates if available.
        """
        action = self.parse(response)
        return self.resolve_element_id(action, registry)
