"""Response parsing for baseline adapters.

Extracts structured actions from VLM responses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedAction:
    """Parsed action from model response.

    Attributes:
        action_type: Action type (click, type, key, scroll, done, unknown).
        x: X coordinate (normalized 0-1) for coordinate actions.
        y: Y coordinate (normalized 0-1) for coordinate actions.
        element_id: Element ID for SoM actions.
        text: Text content for type actions.
        key: Key name for key actions.
        direction: Scroll direction for scroll actions.
        thought: Reasoning text (for ReAct track).
        raw_response: Original model response.
        parse_error: Error message if parsing failed.
        metadata: Additional parsed data.
    """

    action_type: str
    x: float | None = None
    y: float | None = None
    element_id: int | None = None
    text: str | None = None
    key: str | None = None
    direction: str | None = None
    thought: str | None = None
    raw_response: str | None = None
    parse_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the action was successfully parsed."""
        return self.parse_error is None and self.action_type != "unknown"

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
        if self.direction is not None:
            result["direction"] = self.direction
        if self.thought is not None:
            result["thought"] = self.thought

        return result


class UnifiedResponseParser:
    """Parser for VLM responses across all tracks.

    Supports:
    - JSON format: {"action": "CLICK", "x": 0.5, "y": 0.3}
    - Function format: CLICK(0.5, 0.3) or CLICK([17])
    - Mixed format: Thought + action

    Example:
        parser = UnifiedResponseParser()
        action = parser.parse("{"action": "CLICK", "x": 0.5, "y": 0.3}")
        print(action.x, action.y)  # 0.5, 0.3
    """

    def parse(self, response: str) -> ParsedAction:
        """Parse model response into structured action.

        Tries multiple parsing strategies:
        1. JSON extraction
        2. Regex patterns for function-style actions
        3. Fallback text patterns

        Args:
            response: Raw model response string.

        Returns:
            ParsedAction with extracted fields.
        """
        response = response.strip()

        # Try JSON first
        action = self._try_json_parse(response)
        if action.is_valid:
            action.raw_response = response
            return action

        # Try regex patterns
        action = self._try_regex_parse(response)
        if action.is_valid:
            action.raw_response = response
            return action

        # Return unknown action with error
        return ParsedAction(
            action_type="unknown",
            raw_response=response,
            parse_error="No action pattern found in response",
        )

    def _try_json_parse(self, response: str) -> ParsedAction:
        """Try to extract and parse JSON from response."""
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response)
        if not json_match:
            return ParsedAction(action_type="unknown", parse_error="No JSON found")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return ParsedAction(action_type="unknown", parse_error=f"JSON parse error: {e}")

        return self._dict_to_action(data)

    def _dict_to_action(self, data: dict[str, Any]) -> ParsedAction:
        """Convert parsed dict to ParsedAction."""
        action_type = data.get("action", data.get("type", "")).lower()
        thought = data.get("thought")

        if action_type == "click":
            # Check for element_id (SoM) vs coordinates
            if "element_id" in data:
                element_id = data["element_id"]
                if isinstance(element_id, str):
                    # Extract number from "e17" or "[17]" format
                    match = re.search(r'\d+', element_id)
                    element_id = int(match.group()) if match else None
                return ParsedAction(
                    action_type="click",
                    element_id=element_id,
                    thought=thought,
                )
            elif "x" in data and "y" in data:
                return ParsedAction(
                    action_type="click",
                    x=float(data["x"]),
                    y=float(data["y"]),
                    thought=thought,
                )
            else:
                return ParsedAction(
                    action_type="click",
                    parse_error="CLICK missing coordinates or element_id",
                )

        elif action_type == "type":
            return ParsedAction(
                action_type="type",
                text=data.get("text", ""),
                thought=thought,
            )

        elif action_type == "key":
            return ParsedAction(
                action_type="key",
                key=data.get("key", ""),
                thought=thought,
            )

        elif action_type == "scroll":
            return ParsedAction(
                action_type="scroll",
                direction=data.get("direction", "down"),
                thought=thought,
            )

        elif action_type == "done":
            return ParsedAction(action_type="done", thought=thought)

        else:
            return ParsedAction(
                action_type="unknown",
                parse_error=f"Unknown action type: {action_type}",
            )

    def _try_regex_parse(self, response: str) -> ParsedAction:
        """Try regex patterns for function-style actions."""
        # CLICK(x, y) pattern
        click_coords = re.search(
            r'CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',
            response,
            re.IGNORECASE,
        )
        if click_coords:
            try:
                return ParsedAction(
                    action_type="click",
                    x=float(click_coords.group(1)),
                    y=float(click_coords.group(2)),
                )
            except ValueError:
                pass

        # CLICK([id]) pattern for SoM
        click_element = re.search(
            r'CLICK\s*\(\s*\[?\s*(\d+)\s*\]?\s*\)',
            response,
            re.IGNORECASE,
        )
        if click_element:
            return ParsedAction(
                action_type="click",
                element_id=int(click_element.group(1)),
            )

        # TYPE("text") pattern
        type_match = re.search(
            r'TYPE\s*\(\s*["\'](.+?)["\']\s*\)',
            response,
            re.IGNORECASE,
        )
        if type_match:
            return ParsedAction(
                action_type="type",
                text=type_match.group(1),
            )

        # KEY(key) pattern
        key_match = re.search(
            r'KEY\s*\(\s*([a-zA-Z_]+)\s*\)',
            response,
            re.IGNORECASE,
        )
        if key_match:
            return ParsedAction(
                action_type="key",
                key=key_match.group(1).lower(),
            )

        # SCROLL(direction) pattern
        scroll_match = re.search(
            r'SCROLL\s*\(\s*([a-zA-Z]+)\s*\)',
            response,
            re.IGNORECASE,
        )
        if scroll_match:
            return ParsedAction(
                action_type="scroll",
                direction=scroll_match.group(1).lower(),
            )

        # DONE() pattern
        if re.search(r'DONE\s*\(\s*\)', response, re.IGNORECASE):
            return ParsedAction(action_type="done")

        return ParsedAction(action_type="unknown", parse_error="No regex pattern matched")

    def normalize_element_id(self, action: ParsedAction) -> ParsedAction:
        """Normalize element_id to integer format.

        Args:
            action: Action with possibly string element_id.

        Returns:
            Action with normalized integer element_id.
        """
        if action.element_id is not None and isinstance(action.element_id, str):
            match = re.search(r'\d+', str(action.element_id))
            if match:
                action.element_id = int(match.group())
        return action
