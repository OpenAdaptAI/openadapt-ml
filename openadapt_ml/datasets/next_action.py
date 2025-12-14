from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from openadapt_ml.schemas.sessions import Action, Episode, Step


# Coordinate-based DSL system prompt (original)
SYSTEM_PROMPT = (
    "You are a GUI automation agent. Given a screenshot and a user goal, "
    "predict the single next action.\n\n"
    "COORDINATE SYSTEM:\n"
    "- x=0.0 is the LEFT edge, x=1.0 is the RIGHT edge\n"
    "- y=0.0 is the TOP edge, y=1.0 is the BOTTOM edge\n"
    "- To click the CENTER of an element, estimate its center position as a fraction of screen width/height\n"
    "- Example: An element in the middle of the screen would be approximately x=0.5, y=0.5\n\n"
    "ALLOWED ACTIONS (use exactly this format):\n"
    "- CLICK(x=0.XX, y=0.XX)  → click at normalized coordinates\n"
    "- TYPE(text=\"...\")     → type text into the currently focused field\n"
    "- WAIT()                 → wait for UI to update\n"
    "- DONE()                 → task is complete\n\n"
    "RESPONSE FORMAT (required):\n"
    "Thought: [Brief reasoning: what element to interact with and why]\n"
    "Action: [Exactly one action, e.g., CLICK(x=0.35, y=0.42)]\n\n"
    "IMPORTANT: Output coordinates with 2 decimal places. Estimate the center of target elements."
)

# Set-of-Marks (SoM) DSL system prompt - uses element indices instead of coordinates
SYSTEM_PROMPT_SOM = (
    "You are a GUI automation agent. Given a screenshot and a user goal, "
    "predict the single next action.\n\n"
    "INTERACTIVE ELEMENTS:\n"
    "The screenshot shows numbered labels [1], [2], [3], etc. on interactive UI elements.\n"
    "These labels indicate clickable elements like buttons, text fields, links, etc.\n\n"
    "ELEMENT LABELS ON THIS LOGIN SCREEN:\n"
    "[1] = Username text field\n"
    "[2] = Password text field\n"
    "[3] = Login button\n\n"
    "ALLOWED ACTIONS (use exactly this format):\n"
    "- CLICK([N])            → click element with number N to focus/activate it\n"
    "- TYPE([N], \"text\")   → type text into element N (e.g., TYPE([2], \"hello\"))\n"
    "- WAIT()                → wait for UI to update\n"
    "- DONE()                → task is complete\n\n"
    "ACTION SEQUENCE FOR LOGIN:\n"
    "1. CLICK([1]) to focus username field\n"
    "2. TYPE([1], \"username\") to enter username\n"
    "3. CLICK([2]) to focus password field\n"
    "4. TYPE([2], \"password\") to enter password\n"
    "5. CLICK([3]) to submit login\n"
    "6. DONE() when login is complete\n\n"
    "RESPONSE FORMAT (required):\n"
    "Thought: [Brief reasoning: which numbered element to interact with and why]\n"
    "Action: [Exactly one action from the sequence above]\n\n"
    "IMPORTANT: Follow the action sequence step by step. Each step must be done separately."
)

# SoM prompt for registration scenario
SYSTEM_PROMPT_SOM_REGISTRATION = (
    "You are a GUI automation agent. Given a screenshot and a user goal, "
    "predict the single next action.\n\n"
    "INTERACTIVE ELEMENTS:\n"
    "The screenshot shows numbered labels [1], [2], [3], etc. on interactive UI elements.\n"
    "These labels indicate clickable elements like buttons, text fields, links, etc.\n\n"
    "ELEMENT LABELS ON THIS REGISTRATION SCREEN:\n"
    "[1] = First Name text field\n"
    "[2] = Last Name text field\n"
    "[3] = Email text field\n"
    "[4] = Password text field\n"
    "[5] = Confirm Password text field\n"
    "[6] = Register button\n\n"
    "ALLOWED ACTIONS (use exactly this format):\n"
    "- CLICK([N])            → click element with number N to focus/activate it\n"
    "- TYPE([N], \"text\")   → type text into element N (e.g., TYPE([2], \"hello\"))\n"
    "- WAIT()                → wait for UI to update\n"
    "- DONE()                → task is complete\n\n"
    "ACTION SEQUENCE FOR REGISTRATION:\n"
    "1. CLICK([1]) to focus first name field\n"
    "2. TYPE([1], \"name\") to enter first name\n"
    "3. CLICK([2]) to focus last name field\n"
    "4. TYPE([2], \"name\") to enter last name\n"
    "5. CLICK([3]) to focus email field\n"
    "6. TYPE([3], \"email\") to enter email\n"
    "7. CLICK([4]) to focus password field\n"
    "8. TYPE([4], \"pass\") to enter password\n"
    "9. CLICK([5]) to focus confirm password field\n"
    "10. TYPE([5], \"pass\") to enter confirmation\n"
    "11. CLICK([6]) to submit registration\n"
    "12. DONE() when registration is complete\n\n"
    "RESPONSE FORMAT (required):\n"
    "Thought: [Brief reasoning: which numbered element to interact with and why]\n"
    "Action: [Exactly one action from the sequence above]\n\n"
    "IMPORTANT: Follow the action sequence step by step. Each step must be done separately."
)


def format_action(action: Action, use_som: bool = False) -> str:
    """Serialize an Action into a simple textual command.

    For v1 we support a small subset:
    - click: CLICK(x=0.42, y=0.73) or CLICK([1]) in SoM mode
    - type:  TYPE(text="hello") or TYPE([1], "hello") in SoM mode
    - wait:  WAIT()
    - done:  DONE()
    Other types fall back to a generic representation.

    Args:
        action: The action to format.
        use_som: If True, use Set-of-Marks (SoM) index-based format instead of
                 coordinate-based format. Requires element_index to be set.
    """

    t = action.type
    if use_som:
        # SoM mode: use element indices instead of coordinates
        if t == "click" and action.element_index is not None:
            return f"CLICK([{action.element_index}])"
        if t == "type" and action.text is not None:
            escaped = action.text.replace("\\", "\\\\").replace("\"", "\\\"")
            if action.element_index is not None:
                return f"TYPE([{action.element_index}], \"{escaped}\")"
            else:
                # Fallback: TYPE without element reference (for focused field)
                return f"TYPE(\"{escaped}\")"
        if t == "wait":
            return "WAIT()"
        if t == "done":
            return "DONE()"
        # Fallback
        return f"ACTION(type={t})"
    else:
        # Coordinate mode (original)
        if t == "click" and action.x is not None and action.y is not None:
            return f"CLICK(x={action.x:.2f}, y={action.y:.2f})"
        if t == "type" and action.text is not None:
            escaped = action.text.replace("\\", "\\\\").replace("\"", "\\\"")
            return f"TYPE(text=\"{escaped}\")"
        if t == "wait":
            return "WAIT()"
        if t == "done":
            return "DONE()"
        # Fallback
        return f"ACTION(type={t})"


def parse_action_som(text: str) -> Action:
    """Parse a SoM-style action string into an Action object.

    Supported formats:
    - CLICK([N])          → click element N
    - TYPE([N], "text")   → type text into element N
    - TYPE("text")        → type text into focused field
    - WAIT()              → wait
    - DONE()              → done

    Returns Action with element_index set for click/type actions.
    """
    import re

    text = text.strip()

    # CLICK([N])
    match = re.match(r"CLICK\(\[(\d+)\]\)", text)
    if match:
        idx = int(match.group(1))
        return Action(type="click", element_index=idx)

    # TYPE([N], "text") or TYPE([N], 'text')
    match = re.match(r'TYPE\(\[(\d+)\],\s*["\'](.*)["\']\)', text, re.DOTALL)
    if match:
        idx = int(match.group(1))
        content = match.group(2).replace("\\\"", "\"").replace("\\\\", "\\")
        return Action(type="type", text=content, element_index=idx)

    # TYPE("text") - no element index
    match = re.match(r'TYPE\(["\'](.*)["\']\)', text, re.DOTALL)
    if match:
        content = match.group(1).replace("\\\"", "\"").replace("\\\\", "\\")
        return Action(type="type", text=content)

    # WAIT()
    if text.upper().startswith("WAIT"):
        return Action(type="wait")

    # DONE()
    if text.upper().startswith("DONE"):
        return Action(type="done")

    # Failed to parse
    return Action(type="failed", raw={"text": text})


def _generate_thought_for_step(
    step_index: int,
    step: Step,
    goal: str,
    scenario: str = "login",
    total_steps: int = 6,
) -> str:
    """Generate a simple but semantically meaningful Thought for a step.

    This handles both login (6 steps) and registration (12 steps) workflows.
    The goal text is included where helpful so the model can learn to connect
    actions back to the stated objective.
    """

    action = step.action
    t = action.type

    if scenario == "registration":
        return _generate_registration_thought(step_index, step, goal, total_steps)
    else:
        return _generate_login_thought(step_index, step, goal, total_steps)


def _generate_login_thought(step_index: int, step: Step, goal: str, total_steps: int) -> str:
    """Generate thought for login scenario (6 steps)."""
    action = step.action
    t = action.type

    # Step 0: click username field
    if step_index == 0 and t == "click":
        return (
            "I see a login screen with empty username and password fields and a Login button. "
            f"To start logging in, I need to click on the username field to focus it ({goal})."
        )

    # Step 1: type username
    if step_index == 1 and t == "type":
        return (
            "The username field is focused. To move toward the login goal, I should type the "
            "username into this field."
        )

    # Step 2: click password field
    if step_index == 2 and t == "click":
        return (
            "The username has been entered. Next, I need to focus the password field so that I can "
            "enter the password for this login. I will click on the password input box."
        )

    # Step 3: type password
    if step_index == 3 and t == "type":
        return (
            "The password field is focused. To continue the login process, I should type the "
            "password (which will appear as masked characters on the screen)."
        )

    # Step 4: click Login button
    if step_index == 4 and t == "click":
        return (
            "Both the username and password have been entered. To submit the form and attempt the "
            "login, I should click the Login button."
        )

    # Step 5: DONE on logged-in screen
    if step_index == 5 and t == "done":
        return (
            "I now see a logged-in confirmation screen indicating the goal has been satisfied. "
            "The task is complete, so I should emit DONE()."
        )

    # Fallback for any unexpected cases
    return (
        "Based on the current screen and the login goal, I will take the next action that moves "
        "the workflow forward."
    )


def _generate_registration_thought(step_index: int, step: Step, goal: str, total_steps: int) -> str:
    """Generate thought for registration scenario (12 steps)."""
    action = step.action
    t = action.type

    # Registration step mapping (pairs of click + type for 5 fields, then submit + done)
    thoughts = {
        (0, "click"): (
            "I see a registration form with empty fields for name, email, and password. "
            f"To start registration, I need to click on the First Name field ({goal})."
        ),
        (1, "type"): (
            "The First Name field is focused. I should type the first name."
        ),
        (2, "click"): (
            "First name entered. Now I need to focus the Last Name field to enter it."
        ),
        (3, "type"): (
            "The Last Name field is focused. I should type the last name."
        ),
        (4, "click"): (
            "Last name entered. Now I need to focus the Email field to enter the email address."
        ),
        (5, "type"): (
            "The Email field is focused. I should type the email address."
        ),
        (6, "click"): (
            "Email entered. Now I need to focus the Password field to create a password."
        ),
        (7, "type"): (
            "The Password field is focused. I should type the password."
        ),
        (8, "click"): (
            "Password entered. Now I need to focus the Confirm Password field to verify the password."
        ),
        (9, "type"): (
            "The Confirm Password field is focused. I should type the same password again."
        ),
        (10, "click"): (
            "All form fields are filled. I should click the Register button to submit the form."
        ),
        (11, "done"): (
            "Registration is complete - I see a success screen. The task is finished."
        ),
    }

    key = (step_index, t)
    if key in thoughts:
        return thoughts[key]

    # Fallback
    return (
        "Based on the current screen and the registration goal, I will take the next action "
        "that moves the workflow forward."
    )


def _detect_scenario(episode: Episode) -> str:
    """Detect scenario from episode workflow_id."""
    workflow_id = episode.workflow_id or ""
    if "registration" in workflow_id.lower():
        return "registration"
    return "login"


def build_next_action_sft_samples(
    episodes: List[Episode],
    use_som: bool = False,
) -> List[Dict[str, Any]]:
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

    Args:
        episodes: List of episodes to convert.
        use_som: If True, use Set-of-Marks (SoM) DSL with element indices
                 instead of coordinate-based DSL.
    """

    samples: List[Dict[str, Any]] = []

    for episode in episodes:
        goal = episode.goal
        total_steps = len(episode.steps)
        scenario = _detect_scenario(episode)

        # Select appropriate system prompt based on mode and scenario
        if use_som:
            if scenario == "registration":
                system_prompt = SYSTEM_PROMPT_SOM_REGISTRATION
            else:
                system_prompt = SYSTEM_PROMPT_SOM
        else:
            system_prompt = SYSTEM_PROMPT

        for step_index, step in enumerate(episode.steps):
            image_path = step.observation.image_path
            if not image_path:
                # Skip steps without an associated image
                continue

            # Build action history from previous steps
            action_history = []
            for prev_idx in range(step_index):
                prev_step = episode.steps[prev_idx]
                prev_action_text = format_action(prev_step.action, use_som=use_som)
                action_history.append(prev_action_text)

            # Build history section for both modes - use actual step count
            if action_history:
                history_text = "ACTIONS COMPLETED SO FAR:\n"
                for i, action_text in enumerate(action_history, 1):
                    history_text += f"  {i}. {action_text}\n"
                history_text += f"\nThis is step {step_index + 1} of {total_steps}. "
            else:
                history_text = f"This is step 1 of {total_steps} (no actions completed yet). "

            if use_som:
                user_content = (
                    f"Goal: {goal}\n\n"
                    f"{history_text}"
                    "Look at the screenshot and determine the NEXT action.\n\n"
                    "Thought: [which numbered element to interact with and why]\n"
                    "Action: [CLICK([N]) or TYPE([N], \"text\") or WAIT() or DONE()]"
                )
            else:
                user_content = (
                    f"Goal: {goal}\n\n"
                    f"{history_text}"
                    "Look at the screenshot and determine the NEXT action.\n\n"
                    "Thought: [what element to interact with and why]\n"
                    "Action: [CLICK(x=..., y=...) or TYPE(text=\"...\") or WAIT() or DONE()]"
                )

            # Provide a deterministic, semantically meaningful Thought while supervising
            # the exact DSL Action.
            action_text = format_action(step.action, use_som=use_som)
            thought_text = _generate_thought_for_step(step_index, step, goal, scenario, total_steps)
            assistant_content = f"Thought: {thought_text}\nAction: {action_text}"

            sample = {
                "images": [image_path],
                "messages": [
                    {"role": "system", "content": system_prompt},
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
