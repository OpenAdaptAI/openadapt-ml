"""Chain-of-thought warm-up for GRPO training.

Provides utilities to annotate successful demonstration episodes with
chain-of-thought reasoning, then convert them to SFT training format.
This CoT SFT warm-up initializes the policy before GRPO online RL,
giving the model a better starting point for action generation.

The two-step process:
    1. generate_cot_annotations(): Use a capable model to add reasoning
       to each step of successful demonstrations.
    2. build_cot_sft_samples(): Convert annotated episodes to the
       TRL SFT format used by trl_trainer.py.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_cot_annotations(
    episodes: list[Any],
    annotator_model: str = "gpt-4o",
) -> list[Any]:
    """Add chain-of-thought reasoning to successful demonstrations.

    For each step in a successful episode, uses the specified model to
    generate reasoning that explains the action choice given the
    screenshot context. This produces <think>...</think> blocks that
    teach the model to reason before acting.

    Args:
        episodes: List of Episode objects from openadapt_ml.schema.
            Only successful episodes (episode.success == True) are
            annotated; others are returned unchanged.
        annotator_model: Model identifier for generating CoT annotations.
            Must support vision (image inputs).

    Returns:
        List of episodes with reasoning fields populated on each step.
        Episodes that were already annotated or unsuccessful are
        returned unchanged.
    """
    # Deferred import for optional dependency
    try:
        from openadapt_ml.models.api_adapter import get_api_adapter
    except ImportError:
        logger.error(
            "openadapt_ml.models.api_adapter not available. "
            "Cannot generate CoT annotations."
        )
        return episodes

    annotated = []

    for episode in episodes:
        if not getattr(episode, "success", False):
            annotated.append(episode)
            continue

        instruction = getattr(episode, "instruction", "")
        steps = getattr(episode, "steps", [])

        for step_idx, step in enumerate(steps):
            # Skip if already annotated
            if getattr(step, "reasoning", None):
                continue

            screenshot_path = getattr(
                getattr(step, "observation", None),
                "screenshot_path",
                None,
            )
            action = getattr(step, "action", None)

            if not screenshot_path or not action:
                continue

            prompt = (
                f"You are analyzing step {step_idx + 1} of {len(steps)} "
                f"in a GUI automation task.\n\n"
                f"Task instruction: {instruction}\n\n"
                f"The action taken at this step was: {action}\n\n"
                "Explain in 1-2 sentences WHY this action was taken. "
                "Focus on what the agent sees on screen and how the "
                "action moves toward completing the task. "
                "Be concise and specific."
            )

            try:
                adapter = get_api_adapter(annotator_model)
                reasoning = adapter.generate(
                    {
                        "images": [screenshot_path],
                        "messages": [
                            {"role": "user", "content": prompt},
                        ],
                    },
                    max_new_tokens=150,
                )
                step.reasoning = reasoning.strip()
                logger.debug(
                    "Annotated step %d: %s",
                    step_idx,
                    step.reasoning[:80],
                )
            except Exception as e:
                logger.warning("Failed to annotate step %d: %s", step_idx, e)

        annotated.append(episode)

    logger.info("CoT annotation complete: %d episodes processed", len(annotated))
    return annotated


def build_cot_sft_samples(annotated_episodes: list[Any]) -> list[dict]:
    """Convert CoT-annotated episodes to TRL SFT format.

    Produces training samples where the assistant response includes a
    <think> block before the action, teaching the model to reason
    step-by-step during inference.

    Format:
        User: <image>
              Instruction: Open Notepad and type Hello
              Previous actions: CLICK(x=0.05, y=0.95)

        Assistant: <think>I see the Start menu is open. The task requires
                   opening Notepad, so I need to search for it.</think>
                   TYPE(text="notepad")

    Args:
        annotated_episodes: Episodes with reasoning fields on steps,
            typically from generate_cot_annotations().

    Returns:
        List of SFT sample dicts compatible with trl_trainer.py:
        {
            "images": [path],
            "messages": [system, user, assistant],
        }
    """
    from openadapt_ml.datasets.next_action import (
        SYSTEM_PROMPT,
        format_action,
    )

    samples: list[dict] = []

    for episode in annotated_episodes:
        if not getattr(episode, "success", False):
            continue

        instruction = getattr(episode, "instruction", "")
        steps = getattr(episode, "steps", [])

        for step in steps:
            screenshot_path = getattr(
                getattr(step, "observation", None),
                "screenshot_path",
                None,
            )
            action = getattr(step, "action", None)
            reasoning = getattr(step, "reasoning", None)

            if not screenshot_path or not action:
                continue

            # Build action history
            step_index = getattr(step, "step_index", 0)
            prev_actions = []
            for prev_step in steps:
                prev_idx = getattr(prev_step, "step_index", 0)
                if prev_idx < step_index:
                    prev_actions.append(format_action(prev_step.action))

            # Build user content
            parts = [f"Instruction: {instruction}"]
            if prev_actions:
                parts.append("Previous actions: " + " -> ".join(prev_actions))
            user_content = "\n".join(parts)

            # Build assistant content with CoT
            action_text = format_action(action)
            if reasoning:
                assistant_content = f"<think>{reasoning}</think>\n{action_text}"
            else:
                assistant_content = action_text

            sample = {
                "images": [screenshot_path],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
            }
            samples.append(sample)

    logger.info("Built %d CoT SFT samples", len(samples))
    return samples
