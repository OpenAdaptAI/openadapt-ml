"""Demo formatting utilities for few-shot prompting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openadapt_ml.schema import Action, ActionType, Episode, Step


def format_action(action: "Action") -> str:
    """Format an Action as a string for the prompt.

    Args:
        action: Action to format.

    Returns:
        String representation like "CLICK(0.5, 0.3)" or "TYPE('hello')".
    """
    # Get action type value (handle both enum and string)
    action_type = action.type.value if hasattr(action.type, 'value') else action.type

    if action_type == "click":
        if action.normalized_coordinates is not None:
            x, y = action.normalized_coordinates
            return f"CLICK({x:.3f}, {y:.3f})"
        return "CLICK()"

    elif action_type == "double_click":
        if action.normalized_coordinates is not None:
            x, y = action.normalized_coordinates
            return f"DOUBLE_CLICK({x:.3f}, {y:.3f})"
        return "DOUBLE_CLICK()"

    elif action_type == "type":
        text = action.text or ""
        # Escape quotes and truncate if very long
        text = text.replace('"', '\\"')
        if len(text) > 50:
            text = text[:47] + "..."
        return f'TYPE("{text}")'

    elif action_type == "key":
        key = action.key or "unknown"
        if action.modifiers:
            mods = "+".join(action.modifiers)
            return f"KEY({mods}+{key})"
        return f"KEY({key})"

    elif action_type == "scroll":
        direction = action.scroll_direction or "down"
        return f"SCROLL({direction})"

    elif action_type == "drag":
        if action.normalized_coordinates is not None and action.normalized_end is not None:
            x, y = action.normalized_coordinates
            end_x, end_y = action.normalized_end
            return f"DRAG({x:.3f}, {y:.3f}, {end_x:.3f}, {end_y:.3f})"
        return "DRAG()"

    else:
        return f"{action_type.upper()}()"


def format_step(step: "Step", step_num: int) -> str:
    """Format a single step for the demo.

    Args:
        step: Step to format.
        step_num: Step number (1-indexed).

    Returns:
        Formatted step string.
    """
    lines = [f"Step {step_num}:"]

    # Add window context if available
    if step.observation and step.observation.window_title:
        lines.append(f"  Window: {step.observation.window_title}")

    # Add action
    if step.action:
        action_str = format_action(step.action)
        lines.append(f"  Action: {action_str}")

    return "\n".join(lines)


def format_episode_as_demo(
    episode: "Episode",
    max_steps: int = 10,
    include_screenshots: bool = False,
) -> str:
    """Convert an Episode to a few-shot demo format.

    Args:
        episode: Episode containing the demonstration.
        max_steps: Maximum number of steps to include.
        include_screenshots: Whether to include screenshot paths (for multi-image).

    Returns:
        Formatted demo string for prompt injection.
    """
    lines = [
        "DEMONSTRATION:",
        f"Task: {episode.instruction}",
        "",
    ]

    for i, step in enumerate(episode.steps[:max_steps], 1):
        lines.append(format_step(step, i))

        # Optionally include screenshot reference
        if include_screenshots and step.observation and step.observation.screenshot_path:
            lines.append(f"  [Screenshot: {step.observation.screenshot_path}]")

        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def format_episode_verbose(
    episode: "Episode",
    max_steps: int = 10,
) -> str:
    """Format episode with more context per step.

    Includes:
    - Screen summary
    - User intent (inferred)
    - Action taken
    - Observed result

    Args:
        episode: Episode to format.
        max_steps: Maximum steps to include.

    Returns:
        Verbose demo string.
    """
    lines = [
        "DEMONSTRATION:",
        f"Goal: {episode.instruction}",
        "",
        "The following shows the step-by-step procedure:",
        "",
    ]

    for i, step in enumerate(episode.steps[:max_steps], 1):
        lines.append(f"Step {i}:")

        # Screen summary
        if step.observation:
            if step.observation.window_title:
                lines.append(f"  [Screen: {step.observation.window_title}]")

        # Action taken
        if step.action:
            action_str = format_action(step.action)
            lines.append(f"  [Action: {action_str}]")

        # Observed result (inferred from next step's observation)
        if i < len(episode.steps):
            next_step = episode.steps[i]
            if next_step.observation and next_step.observation.window_title:
                if (
                    not step.observation
                    or next_step.observation.window_title != step.observation.window_title
                ):
                    lines.append(f"  [Result: Window changed to {next_step.observation.window_title}]")

        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def get_demo_screenshot_paths(
    episode: "Episode",
    max_steps: int = 10,
) -> list[str]:
    """Get screenshot paths from episode for multi-image prompting.

    Args:
        episode: Episode to extract screenshots from.
        max_steps: Maximum steps to include.

    Returns:
        List of screenshot paths.
    """
    paths = []
    for step in episode.steps[:max_steps]:
        if step.observation and step.observation.screenshot_path:
            path = step.observation.screenshot_path
            if Path(path).exists():
                paths.append(path)
    return paths


def generate_length_matched_control(demo: str) -> str:
    """Generate a control prompt with the same token count but no trajectory info.

    Used to control for prompt length effects.

    Args:
        demo: The demo string to match length of.

    Returns:
        Control string of similar length with irrelevant content.
    """
    # Use generic placeholder text
    placeholder = (
        "This is placeholder text that serves as a control condition. "
        "It contains no relevant information about the task or demonstration. "
        "The purpose is to match the token count of the demonstration prompt. "
    )

    # Repeat to match approximate length
    target_len = len(demo)
    control = ""
    while len(control) < target_len:
        control += placeholder

    return control[:target_len]
