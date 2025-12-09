from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from openadapt_ml.schemas.sessions import Action, Episode, Step


SYSTEM_PROMPT = (
    "You are a GUI automation agent. Given a screenshot and a user goal, "
    "you must predict the next action in a strict DSL. Allowed actions are:\n"
    "- CLICK(x=<float in [0,1]>, y=<float in [0,1]>)  # click at normalized coordinates\n"
    "- TYPE(text=\"...\")                         # type text into the focused field\n"
    "- WAIT()                                       # wait for the UI to update\n"
    "- DONE()                                       # task is complete\n\n"
    "Your reply MUST follow this exact format:\n\n"
    "Thought: [Your reasoning about what you see and what to do next]\n"
    "Action: [Exactly one action from the list above]\n\n"
    "The Thought can be multiple sentences, but the Action line must contain only one valid DSL action."
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


def _generate_thought_for_step(step_index: int, step: Step, goal: str) -> str:
    """Generate a simple but semantically meaningful Thought for a login step.

    This is specific to the synthetic login workflow, which always follows the
    same 7-step pattern. The goal text is included where helpful so the model
    can learn to connect actions back to the stated objective.
    """

    action = step.action
    t = action.type

    # Step 0: initial blank login screen
    if step_index == 0 and t == "wait":
        return (
            "I see a login screen with empty username and password fields and a Login button. "
            "I will briefly wait on this initial screen before taking the first action."
        )

    # Step 1: click username field
    if step_index == 1 and t == "click":
        return (
            "To start logging in, I need to focus the username field so I can type the username "
            f"specified in the goal ({goal}). I will click on the username input box."
        )

    # Step 2: type username
    if step_index == 2 and t == "type":
        return (
            "The username field is focused. To move toward the login goal, I should type the "
            "username into this field."
        )

    # Step 3: click password field
    if step_index == 3 and t == "click":
        return (
            "The username has been entered. Next, I need to focus the password field so that I can "
            "enter the password for this login. I will click on the password input box."
        )

    # Step 4: type password
    if step_index == 4 and t == "type":
        return (
            "The password field is focused. To continue the login process, I should type the "
            "password (which will appear as masked characters on the screen)."
        )

    # Step 5: click Login button
    if step_index == 5 and t == "click":
        return (
            "Both the username and password have been entered. To submit the form and attempt the "
            "login, I should click the Login button."
        )

    # Step 6: DONE on logged-in screen
    if step_index == 6 and t == "done":
        return (
            "I now see a logged-in confirmation screen indicating the goal has been satisfied. "
            "The task is complete, so I should emit DONE()."
        )

    # Fallback for any unexpected cases
    return (
        "Based on the current screen and the login goal, I will take the next action that moves "
        "the workflow forward."
    )


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
        for step_index, step in enumerate(episode.steps):
            image_path = step.observation.image_path
            if not image_path:
                # Skip steps without an associated image
                continue

            user_content = (
                f"Goal: {goal}\n\n"
                "You are controlling the computer using only the current screenshot.\n"
                "Think step-by-step before choosing an action:\n"
                "1. What UI elements do you see that are relevant to the goal?\n"
                "2. What is the next logical step toward completing the goal?\n"
                "3. What single action should you take now?\n\n"
                "Respond in the required format:\n\n"
                "Thought: [your reasoning]\n"
                "Action: [one of CLICK(...), TYPE(...), WAIT(), DONE()]\n"
            )

            # Provide a deterministic, semantically meaningful Thought while supervising
            # the exact DSL Action.
            action_text = format_action(step.action)
            thought_text = _generate_thought_for_step(step_index, step, goal)
            assistant_content = f"Thought: {thought_text}\nAction: {action_text}"

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
