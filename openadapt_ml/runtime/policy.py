from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from openadapt_ml.models.base_adapter import BaseVLMAdapter
from openadapt_ml.schemas.sessions import Action


_CLICK_RE = re.compile(r"CLICK\(x=([0-9]*\.?[0-9]+),\s*y=([0-9]*\.?[0-9]+)\)")
_DONE_RE = re.compile(r"\bDONE\s*\(\s*\)")


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
        # Prefer explicit CLICK(...) if present
        m = _CLICK_RE.search(text)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            # Clamp to [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return Action(type="click", x=x, y=y)

        # DONE() indicates termination
        if _DONE_RE.search(text):
            return Action(type="done")

        # Fallback
        return Action(type="failed", raw={"text": text})

    def predict_action_from_sample(self, sample: Dict[str, Any], max_new_tokens: int = 64) -> Tuple[Action, Optional[str]]:
        """Run the adapter on a pre-built SFT-style sample and parse the result.

        Returns (Action, thought). For now, thought is always None; in future
        we may parse a `Thought: ...` prefix.
        """

        text = self.adapter.generate(sample, max_new_tokens=max_new_tokens)
        action = self._parse_action(text)
        return action, None
