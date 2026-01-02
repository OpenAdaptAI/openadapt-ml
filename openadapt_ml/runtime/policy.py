from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from openadapt_ml.models.base_adapter import BaseVLMAdapter
from openadapt_ml.schema import Action, ActionType, UIElement


# Coordinate-based DSL patterns
_CLICK_RE = re.compile(r"CLICK\(x=(-?[0-9]*\.?[0-9]+),\s*y=(-?[0-9]*\.?[0-9]+)\)")
_TYPE_RE = re.compile(r'TYPE\(text="([^"\\]*(?:\\.[^"\\]*)*)"\)')
_WAIT_RE = re.compile(r"\bWAIT\s*\(\s*\)")
_DONE_RE = re.compile(r"\bDONE\s*\(\s*\)")

# SoM (Set-of-Marks) index-based DSL patterns
_CLICK_SOM_RE = re.compile(r"CLICK\(\[(\d+)\]\)")
_TYPE_SOM_RE = re.compile(r'TYPE\(\[(\d+)\],\s*["\']([^"\']*(?:\\.[^"\']*)*)["\']\)')
_TYPE_SOM_SIMPLE_RE = re.compile(r'TYPE\(["\']([^"\']*(?:\\.[^"\']*)*)["\']\)')


@dataclass
class PolicyOutput:
    """Result of a single policy step."""
    action: Action
    thought: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    raw_text: str = ""


def parse_thought_state_action(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """Parse Thought / State / Action blocks from model output.

    Expected format:
        Thought: [reasoning]
        State: {"success": false, "progress": 0.5, ...}
        Action: CLICK(x=0.42, y=0.31)

    Returns:
        (thought, state, action_str):
        - thought: Content after 'Thought:' up to 'State:' or 'Action:'
        - state: Parsed JSON dict from 'State:' line, or None
        - action_str: Content after 'Action:', or whole text if missing

    Note: We look for the LAST occurrence of 'Action:' to handle cases where
    the user prompt template also contains 'Action:' placeholders.
    """
    thought: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    action_str: str = text.strip()

    # Extract Thought - find the LAST occurrence (model's response, not template)
    thought_matches = list(re.finditer(r"Thought:\s*(.+?)(?=State:|Action:|$)", text, re.DOTALL | re.IGNORECASE))
    if thought_matches:
        thought = thought_matches[-1].group(1).strip()

    # Extract State (JSON on same line or next line) - last occurrence
    state_matches = list(re.finditer(r"State:\s*(\{.*?\})", text, re.DOTALL | re.IGNORECASE))
    if state_matches:
        try:
            state = json.loads(state_matches[-1].group(1))
        except json.JSONDecodeError:
            state = None

    # Extract Action - find the LAST occurrence to get the model's actual action
    # not the placeholder in the prompt template
    action_matches = list(re.finditer(r"Action:\s*(.+?)(?=\n|$)", text, re.IGNORECASE))
    if action_matches:
        action_str = action_matches[-1].group(1).strip()

    return thought, state, action_str


class AgentPolicy:
    """Runtime policy wrapper around a trained VLM adapter.

    Formats goal-conditioned inputs and parses textual actions into
    structured `Action` objects.
    """

    def __init__(self, adapter: BaseVLMAdapter) -> None:
        self.adapter = adapter

    def _build_sample(self, image: Image.Image, goal: str) -> Dict[str, Any]:
        # For runtime we keep the same structure as SFT samples but use
        # an in-memory image. The adapter's generate method currently expects
        # paths, so we require the caller to supply a path-based sample. For
        # now, we save responsibility for image loading to the caller; this
        # method is kept for future extensibility.
        raise NotImplementedError(
            "AgentPolicy._build_sample is not used directly; pass a sample dict "
            "compatible with the adapter's `generate` method."
        )

    def _parse_action(self, text: str) -> Action:
        """Parse a DSL action string into an Action object.

        Supported formats (coordinate-based):
        - CLICK(x=<float>, y=<float>)
        - TYPE(text="...")

        Supported formats (SoM index-based):
        - CLICK([N])
        - TYPE([N], "text")
        - TYPE("text")

        Common formats:
        - WAIT()
        - DONE()

        Returns Action(type="failed") if no valid action is found.
        """
        # Try SoM patterns first (index-based)
        # CLICK([N])
        m = _CLICK_SOM_RE.search(text)
        if m:
            idx = int(m.group(1))
            return Action(type=ActionType.CLICK, element=UIElement(element_id=str(idx)))

        # TYPE([N], "text")
        m = _TYPE_SOM_RE.search(text)
        if m:
            idx = int(m.group(1))
            raw_text = m.group(2)
            unescaped = raw_text.replace('\\"', '"').replace("\\\\", "\\")
            return Action(type=ActionType.TYPE, text=unescaped, element=UIElement(element_id=str(idx)))

        # TYPE("text") - SoM style without index
        m = _TYPE_SOM_SIMPLE_RE.search(text)
        if m:
            raw_text = m.group(1)
            unescaped = raw_text.replace('\\"', '"').replace("\\\\", "\\")
            return Action(type=ActionType.TYPE, text=unescaped)

        # Coordinate-based patterns
        # CLICK(x=..., y=...)
        m = _CLICK_RE.search(text)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            # Clamp to [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return Action(type=ActionType.CLICK, normalized_coordinates=(x, y))

        # TYPE(text="...")
        m = _TYPE_RE.search(text)
        if m:
            # Unescape the text content
            raw_text = m.group(1)
            unescaped = raw_text.replace('\\"', '"').replace("\\\\", "\\")
            return Action(type=ActionType.TYPE, text=unescaped)

        # WAIT()
        if _WAIT_RE.search(text):
            return Action(type=ActionType.WAIT)

        # DONE()
        if _DONE_RE.search(text):
            return Action(type=ActionType.DONE)

        # Fallback
        return Action(type=ActionType.FAIL, raw={"text": text})

    def predict_action_from_sample(
        self, sample: Dict[str, Any], max_new_tokens: int = 150
    ) -> Tuple[Action, Optional[str], Optional[Dict[str, Any]], str]:
        """Run the adapter on a pre-built SFT-style sample and parse the result.

        Returns (Action, thought, state, raw_text) where:
        - thought: Reasoning text from 'Thought:' block
        - state: Parsed JSON dict from 'State:' block (may contain 'success' bool)
        - raw_text: The raw model output text for debugging
        """
        text = self.adapter.generate(sample, max_new_tokens=max_new_tokens)
        thought, state, action_str = parse_thought_state_action(text)
        action = self._parse_action(action_str)
        return action, thought, state, text
