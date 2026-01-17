"""Prompt templates for baseline adapters.

Provides track-specific system prompts and user content builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openadapt_ml.baselines.config import TrackConfig, TrackType

if TYPE_CHECKING:
    from PIL import Image


# System prompts for each track
SYSTEM_PROMPTS = {
    TrackType.TRACK_A: """You are a GUI automation agent. Your task is to interact with graphical user interfaces by analyzing screenshots and determining the next action.

CAPABILITIES:
- CLICK(x, y): Click at normalized coordinates (0.0-1.0) where (0,0) is top-left
- TYPE("text"): Type the specified text
- KEY(key): Press a key (e.g., KEY(enter), KEY(escape))
- SCROLL(direction): Scroll up or down
- DONE(): Mark task as complete

RULES:
1. Analyze the screenshot carefully before acting
2. Use normalized coordinates (0.0-1.0) for all clicks
3. Return ONLY the action in JSON format
4. If the goal is achieved, use DONE()

OUTPUT FORMAT:
{"action": "CLICK", "x": 0.5, "y": 0.3}
or
{"action": "TYPE", "text": "hello world"}
or
{"action": "KEY", "key": "enter"}
or
{"action": "SCROLL", "direction": "down"}
or
{"action": "DONE"}""",
    TrackType.TRACK_B: """You are a GUI automation agent using ReAct (Reason + Act) to interact with interfaces.

CAPABILITIES:
- CLICK(x, y): Click at normalized coordinates (0.0-1.0) where (0,0) is top-left
- TYPE("text"): Type the specified text
- KEY(key): Press a key (e.g., KEY(enter), KEY(escape))
- SCROLL(direction): Scroll up or down
- DONE(): Mark task as complete

PROCESS:
1. OBSERVE: Describe what you see in the screenshot
2. THINK: Reason about what action to take and why
3. ACT: Execute the chosen action

RULES:
1. Always explain your reasoning before acting
2. Use normalized coordinates (0.0-1.0) for all clicks
3. Return both thought and action in JSON format
4. If the goal is achieved, use DONE()

OUTPUT FORMAT:
{"thought": "I can see a login form. The username field is at the top. I should click it first to enter credentials.", "action": "CLICK", "x": 0.5, "y": 0.3}
or
{"thought": "The username field is now focused. I should type the username.", "action": "TYPE", "text": "user@example.com"}
or
{"thought": "The task is complete - I can see the success message.", "action": "DONE"}""",
    TrackType.TRACK_C: """You are a GUI automation agent. UI elements are marked with numbered labels [1], [2], etc.

CAPABILITIES:
- CLICK([id]): Click the element with the given ID
- TYPE("text"): Type the specified text
- KEY(key): Press a key (e.g., KEY(enter), KEY(escape))
- SCROLL(direction): Scroll up or down
- DONE(): Mark task as complete

RULES:
1. Use element IDs from the labels, NOT coordinates
2. Each element has a number in brackets like [1], [17], [42]
3. Return ONLY the action in JSON format
4. If the goal is achieved, use DONE()

OUTPUT FORMAT:
{"action": "CLICK", "element_id": 17}
or
{"action": "TYPE", "text": "hello world"}
or
{"action": "KEY", "key": "enter"}
or
{"action": "SCROLL", "direction": "down"}
or
{"action": "DONE"}""",
}


class PromptBuilder:
    """Builds prompts for baseline API calls.

    Constructs system prompts and user content based on track configuration.

    Example:
        builder = PromptBuilder(track_config)
        system = builder.get_system_prompt()
        content = builder.build_user_content(
            goal="Log into the application",
            screenshot=img,
            a11y_tree=tree,
            history=history,
        )
    """

    def __init__(self, track: TrackConfig):
        """Initialize prompt builder.

        Args:
            track: Track configuration.
        """
        self.track = track

    def get_system_prompt(self, demo: str | None = None) -> str:
        """Get the system prompt for this track.

        Args:
            demo: Optional demo text to include.

        Returns:
            System prompt string.
        """
        base_prompt = SYSTEM_PROMPTS[self.track.track_type]

        if demo:
            base_prompt += f"\n\nEXAMPLE DEMONSTRATION:\n{demo}"

        return base_prompt

    def build_user_content(
        self,
        goal: str,
        screenshot: "Image" | None = None,
        a11y_tree: str | dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        encode_image_fn: Any = None,
    ) -> list[dict[str, Any]]:
        """Build user message content for API call.

        Args:
            goal: Task goal/instruction.
            screenshot: Screenshot image (PIL Image).
            a11y_tree: Accessibility tree (string or dict).
            history: List of previous actions.
            encode_image_fn: Function to encode image for API.

        Returns:
            List of content blocks for API message.
        """
        content: list[dict[str, Any]] = []

        # Build text prompt
        text_parts = [f"GOAL: {goal}"]

        # Add accessibility tree if configured
        if self.track.use_a11y_tree and a11y_tree:
            tree_text = self._format_a11y_tree(a11y_tree)
            if tree_text:
                text_parts.append(f"\nACCESSIBILITY TREE:\n{tree_text}")

        # Add action history if configured
        if self.track.include_history and history:
            history_text = self._format_history(history)
            if history_text:
                text_parts.append(f"\nPREVIOUS ACTIONS:\n{history_text}")

        # Add instruction
        text_parts.append("\nAnalyze the screenshot and provide the next action.")
        text_parts.append(f"OUTPUT FORMAT: {self.track.output_format}")

        # Add text content
        content.append({"type": "text", "text": "\n".join(text_parts)})

        # Add screenshot if provided
        if screenshot is not None and encode_image_fn is not None:
            content.append(encode_image_fn(screenshot))

        return content

    def _format_a11y_tree(self, tree: str | dict[str, Any]) -> str:
        """Format accessibility tree for prompt.

        Args:
            tree: Accessibility tree as string or dict.

        Returns:
            Formatted string (possibly truncated).
        """
        if isinstance(tree, str):
            text = tree
        elif isinstance(tree, dict):
            text = self._dict_to_tree_string(tree)
        else:
            return ""

        # Truncate if needed
        max_lines = self.track.max_a11y_elements
        lines = text.split("\n")
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append(f"... (truncated, {len(lines)} of {max_lines} elements)")

        return "\n".join(lines)

    def _dict_to_tree_string(
        self, tree: dict[str, Any], indent: int = 0, max_depth: int = 5
    ) -> str:
        """Convert dict tree to formatted string.

        Args:
            tree: Dictionary representing accessibility tree.
            indent: Current indentation level.
            max_depth: Maximum recursion depth.

        Returns:
            Formatted tree string.
        """
        if indent > max_depth:
            return ""

        lines = []
        prefix = "  " * indent

        role = tree.get("role", "unknown")
        name = tree.get("name", "")
        node_id = tree.get("id", "")

        # Format node
        if node_id:
            line = f"{prefix}[{node_id}] {role}"
        else:
            line = f"{prefix}{role}"

        if name:
            line += f': "{name}"'

        lines.append(line)

        # Process children
        children = tree.get("children", [])
        for child in children:
            if isinstance(child, dict):
                lines.append(self._dict_to_tree_string(child, indent + 1, max_depth))

        return "\n".join(lines)

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        """Format action history for prompt.

        Args:
            history: List of action dictionaries.

        Returns:
            Formatted history string.
        """
        if not history:
            return ""

        lines = []
        max_steps = self.track.max_history_steps
        recent = history[-max_steps:] if len(history) > max_steps else history

        for i, action in enumerate(recent, 1):
            action_type = action.get("type", action.get("action", "unknown")).upper()

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
            elif action_type == "DONE":
                lines.append(f"{i}. DONE()")
            else:
                lines.append(f"{i}. {action_type}()")

        return "\n".join(lines)
