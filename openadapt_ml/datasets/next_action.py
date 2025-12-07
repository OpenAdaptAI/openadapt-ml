from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from openadapt_ml.schemas.sessions import Action, Episode


SYSTEM_PROMPT = (
    "You are a precise GUI automation agent. Given a screenshot and a user goal, "
    "you must predict exactly one next action in a strict DSL. Allowed actions are: "
    "CLICK(x=<float in [0,1]>, y=<float in [0,1]>), TYPE(text=\"...\"), WAIT(), and DONE(). "
    "Respond with a single line containing only the action, with no extra text or explanation. "
    "Do not include any reasoning, comments, or additional text."
)


def format_action(action: Action) -> str:
    """Serialize an Action into a simple textual command.

    For v1 we support a small subset:
    - click: CLICK(x=0.42, y=0.73)
    - type:  TYPE(text="hello")
    - wait:  WAIT()
    - done:  DONE()
    Other types fall back to a generic representation.
    """

    t = action.type
    if t == "click" and action.x is not None and action.y is not None:
        return f"CLICK(x={action.x:.4f}, y={action.y:.4f})"
    if t == "type" and action.text is not None:
        escaped = action.text.replace("\\", "\\\\").replace("\"", "\\\"")
        return f"TYPE(text=\"{escaped}\")"
    if t == "wait":
        return "WAIT()"
    if t == "done":
        return "DONE()"
    # Fallback
    return f"ACTION(type={t})"


def build_next_action_sft_samples(episodes: List[Episode]) -> List[Dict[str, Any]]:
    """Convert Episodes into goal-conditioned next-action SFT samples.

    One sample per step (including terminal DONE), with structure:
    {
        "images": [image_path],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": action_text},
        ],
    }
    """

    samples: List[Dict[str, Any]] = []

    for episode in episodes:
        goal = episode.goal
        for step in episode.steps:
            image_path = step.observation.image_path
            if not image_path:
                # Skip steps without an associated image
                continue

            user_content = (
                f"Goal: {goal}\n"
                "You are controlling the computer using only the current screenshot.\n"
                "Choose exactly one next action that moves toward completing the goal.\n\n"
                "Respond with ONE line containing only a valid action in this format:\n"
                "CLICK(x=<float in [0,1]>, y=<float in [0,1]>) or TYPE(text=\"...\") or WAIT() or DONE().\n"
                "Do not output explanations, reasoning, or any extra text.\n\n"
                "Now output only the single next action."
            )
            assistant_content = format_action(step.action)

            sample = {
                "images": [image_path],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
            }
            samples.append(sample)

    return samples


@dataclass
class NextActionSample:
    images: List[str]
    messages: List[Dict[str, str]]


class NextActionDataset(Dataset):
    """Thin PyTorch Dataset wrapper around pre-built SFT samples."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self._samples = samples

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        return self._samples[idx]
