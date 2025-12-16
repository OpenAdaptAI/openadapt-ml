"""Agent interface for benchmark evaluation.

This module provides the BenchmarkAgent interface that agents must implement
to be evaluated on benchmarks, plus adapters to wrap existing openadapt-ml
components.

Example:
    from openadapt_ml.benchmarks import PolicyAgent
    from openadapt_ml.runtime.policy import AgentPolicy

    policy = AgentPolicy(adapter)
    agent = PolicyAgent(policy)
    results = evaluate_agent_on_benchmark(agent, benchmark_adapter)

    # API-backed agents (GPT-5.1, Claude)
    from openadapt_ml.benchmarks import APIBenchmarkAgent

    agent = APIBenchmarkAgent(provider="anthropic")  # Uses Claude
    agent = APIBenchmarkAgent(provider="openai")     # Uses GPT-5.1
    results = evaluate_agent_on_benchmark(agent, benchmark_adapter)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from openadapt_ml.benchmarks.base import (
    BenchmarkAction,
    BenchmarkObservation,
    BenchmarkTask,
)

if TYPE_CHECKING:
    from openadapt_ml.models.api_adapter import ApiVLMAdapter
    from openadapt_ml.runtime.policy import AgentPolicy
    from openadapt_ml.schemas.sessions import Action


class BenchmarkAgent(ABC):
    """Abstract interface for agents evaluated on benchmarks.

    Agents must implement the `act` method to receive observations
    and return actions. The agent can maintain internal state across
    steps within an episode.
    """

    @abstractmethod
    def act(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None = None,
    ) -> BenchmarkAction:
        """Given observation and task, return next action.

        Args:
            observation: Current observation from the environment.
            task: Task being performed.
            history: Optional list of previous (observation, action) pairs.

        Returns:
            Action to execute.
        """
        pass

    def reset(self) -> None:
        """Reset agent state between episodes.

        Called before starting a new task. Override to clear any
        internal state.
        """
        pass


class PolicyAgent(BenchmarkAgent):
    """Wraps openadapt-ml AgentPolicy for benchmark evaluation.

    Converts between BenchmarkObservation/BenchmarkAction and the
    SFT sample format expected by AgentPolicy.

    Args:
        policy: AgentPolicy instance to wrap.
        use_accessibility_tree: Whether to include accessibility tree in prompt.
        use_history: Whether to include action history in prompt.
    """

    def __init__(
        self,
        policy: AgentPolicy,
        use_accessibility_tree: bool = True,
        use_history: bool = True,
    ):
        self.policy = policy
        self.use_accessibility_tree = use_accessibility_tree
        self.use_history = use_history

    def act(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None = None,
    ) -> BenchmarkAction:
        """Convert observation to SFT sample and get action from policy.

        Args:
            observation: Benchmark observation.
            task: Benchmark task.
            history: Previous observations and actions.

        Returns:
            BenchmarkAction from policy.
        """
        # Build SFT-style sample
        sample = self._build_sample(observation, task, history)

        # Get action from policy
        action, thought = self.policy.predict(sample)

        # Convert to BenchmarkAction
        return self._to_benchmark_action(action, thought)

    def _build_sample(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None,
    ) -> dict:
        """Build SFT-style sample from benchmark observation.

        Args:
            observation: Current observation.
            task: Current task.
            history: Action history.

        Returns:
            Sample dict with 'images' and 'messages'.
        """
        # Build user message content
        content_parts = [f"Goal: {task.instruction}"]

        # Add accessibility tree if available and enabled
        if self.use_accessibility_tree and observation.accessibility_tree:
            tree_str = self._format_accessibility_tree(observation.accessibility_tree)
            content_parts.append(f"UI Elements:\n{tree_str}")

        # Add context
        if observation.url:
            content_parts.append(f"URL: {observation.url}")
        if observation.window_title:
            content_parts.append(f"Window: {observation.window_title}")

        # Add history if enabled
        if self.use_history and history:
            history_str = self._format_history(history)
            content_parts.append(f"Previous actions:\n{history_str}")

        content_parts.append("What action should be taken next?")

        # Build sample
        sample = {
            "messages": [
                {"role": "user", "content": "\n\n".join(content_parts)},
            ],
        }

        # Add image if available
        if observation.screenshot_path:
            sample["images"] = [observation.screenshot_path]

        return sample

    def _format_accessibility_tree(self, tree: dict, indent: int = 0) -> str:
        """Format accessibility tree for prompt.

        Args:
            tree: Accessibility tree dict.
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        # Simple formatting - can be overridden for platform-specific formatting
        lines = []
        prefix = "  " * indent

        role = tree.get("role", "unknown")
        name = tree.get("name", "")
        node_id = tree.get("id", tree.get("node_id", ""))

        line = f"{prefix}[{node_id}] {role}"
        if name:
            line += f": {name}"
        lines.append(line)

        for child in tree.get("children", []):
            lines.append(self._format_accessibility_tree(child, indent + 1))

        return "\n".join(lines)

    def _format_history(
        self, history: list[tuple[BenchmarkObservation, BenchmarkAction]]
    ) -> str:
        """Format action history for prompt.

        Args:
            history: List of (observation, action) pairs.

        Returns:
            Formatted string.
        """
        lines = []
        for i, (obs, action) in enumerate(history[-5:], 1):  # Last 5 actions
            action_str = self._action_to_string(action)
            lines.append(f"{i}. {action_str}")
        return "\n".join(lines)

    def _action_to_string(self, action: BenchmarkAction) -> str:
        """Convert BenchmarkAction to string representation.

        Args:
            action: Action to convert.

        Returns:
            String representation.
        """
        if action.type == "click":
            if action.target_name:
                return f"CLICK({action.target_name})"
            return f"CLICK(x={action.x:.3f}, y={action.y:.3f})"
        elif action.type == "type":
            return f"TYPE({action.text!r})"
        elif action.type == "key":
            mods = "+".join(action.modifiers or [])
            key = action.key
            if mods:
                return f"KEY({mods}+{key})"
            return f"KEY({key})"
        elif action.type == "scroll":
            return f"SCROLL({action.scroll_direction})"
        elif action.type == "done":
            return "DONE()"
        elif action.type == "answer":
            return f"ANSWER({action.answer!r})"
        else:
            return f"{action.type.upper()}()"

    def _to_benchmark_action(
        self, action: Action, thought: str | None
    ) -> BenchmarkAction:
        """Convert openadapt-ml Action to BenchmarkAction.

        Args:
            action: Action from policy.
            thought: Optional thought/reasoning.

        Returns:
            BenchmarkAction.
        """
        return BenchmarkAction(
            type=action.type,
            x=action.x,
            y=action.y,
            text=action.text,
            target_bbox=action.bbox,
            # Map additional fields if present
            target_node_id=getattr(action, "target_node_id", None),
            target_role=getattr(action, "target_role", None),
            target_name=getattr(action, "target_name", None),
            key=getattr(action, "key", None),
            modifiers=getattr(action, "modifiers", None),
            scroll_direction=getattr(action, "scroll_direction", None),
            scroll_amount=getattr(action, "scroll_amount", None),
            end_x=getattr(action, "end_x", None),
            end_y=getattr(action, "end_y", None),
            answer=getattr(action, "answer", None),
            raw_action={"thought": thought} if thought else None,
        )

    def reset(self) -> None:
        """Reset agent state."""
        # PolicyAgent is stateless, nothing to reset
        pass


class ScriptedAgent(BenchmarkAgent):
    """Agent that follows a predefined script of actions.

    Useful for testing benchmark adapters or replaying trajectories.

    Args:
        actions: List of actions to execute in order.
    """

    def __init__(self, actions: list[BenchmarkAction]):
        self.actions = actions
        self._step = 0

    def act(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None = None,
    ) -> BenchmarkAction:
        """Return the next scripted action.

        Args:
            observation: Ignored.
            task: Ignored.
            history: Ignored.

        Returns:
            Next action from script, or DONE if script exhausted.
        """
        if self._step < len(self.actions):
            action = self.actions[self._step]
            self._step += 1
            return action
        return BenchmarkAction(type="done")

    def reset(self) -> None:
        """Reset step counter."""
        self._step = 0


class RandomAgent(BenchmarkAgent):
    """Agent that takes random actions.

    Useful for baseline comparisons.

    Args:
        action_types: List of action types to randomly select from.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        action_types: list[str] | None = None,
        seed: int | None = None,
    ):
        import random

        self.action_types = action_types or ["click", "type", "scroll", "done"]
        self.rng = random.Random(seed)

    def act(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None = None,
    ) -> BenchmarkAction:
        """Return a random action.

        Args:
            observation: Used to get viewport bounds.
            task: Ignored.
            history: Used to decide when to stop.

        Returns:
            Random action.
        """
        # Stop after many actions
        if history and len(history) > 20:
            return BenchmarkAction(type="done")

        action_type = self.rng.choice(self.action_types)

        if action_type == "click":
            return BenchmarkAction(
                type="click",
                x=self.rng.random(),
                y=self.rng.random(),
            )
        elif action_type == "type":
            return BenchmarkAction(
                type="type",
                text="test",
            )
        elif action_type == "scroll":
            return BenchmarkAction(
                type="scroll",
                scroll_direction=self.rng.choice(["up", "down"]),
            )
        else:
            return BenchmarkAction(type="done")

    def reset(self) -> None:
        """Nothing to reset."""
        pass


class APIBenchmarkAgent(BenchmarkAgent):
    """Agent that uses hosted VLM APIs (Claude, GPT-5.1) for benchmark evaluation.

    This agent wraps ApiVLMAdapter to provide Claude or GPT-5.1 baselines
    for benchmark evaluation. It converts BenchmarkObservation to the
    API format and parses VLM responses into BenchmarkActions.

    Args:
        provider: API provider - "anthropic" (Claude) or "openai" (GPT-5.1).
        api_key: Optional API key override. If not provided, uses env vars.
        model: Optional model name override. Defaults to provider's best VLM.
        max_tokens: Maximum tokens for VLM response.
        use_accessibility_tree: Whether to include accessibility tree in prompt.
        use_history: Whether to include action history in prompt.

    Example:
        # Claude baseline
        agent = APIBenchmarkAgent(provider="anthropic")
        results = evaluate_agent_on_benchmark(agent, waa_adapter)

        # GPT-5.1 baseline
        agent = APIBenchmarkAgent(provider="openai")
        results = evaluate_agent_on_benchmark(agent, waa_adapter)
    """

    # System prompt for GUI automation
    SYSTEM_PROMPT = """You are a GUI automation agent. Given a screenshot and task instruction, determine the next action to take.

Available actions:
- CLICK(x, y) - Click at coordinates (can be pixel values or normalized 0.0-1.0)
- CLICK([id]) - Click element with given ID from accessibility tree
- TYPE("text") - Type the given text
- KEY(key) - Press a key (e.g., Enter, Tab, Escape)
- KEY(modifier+key) - Press key combination (e.g., Ctrl+c, Alt+Tab)
- SCROLL(direction) - Scroll up or down
- DRAG(x1, y1, x2, y2) - Drag from (x1,y1) to (x2,y2) (pixel or normalized)
- DONE() - Task is complete
- ANSWER("response") - For QA tasks, provide the answer

Respond with exactly ONE action in the format shown above.
If the task appears complete, use DONE().

Think step by step:
1. What is the current state of the UI?
2. What is the goal?
3. What is the next logical action?

Then output the action on a new line starting with "ACTION:"
"""

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
        use_accessibility_tree: bool = True,
        use_history: bool = True,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.use_accessibility_tree = use_accessibility_tree
        self.use_history = use_history
        self._adapter: ApiVLMAdapter | None = None

    def _get_adapter(self) -> "ApiVLMAdapter":
        """Lazily initialize the API adapter."""
        if self._adapter is None:
            from openadapt_ml.models.api_adapter import ApiVLMAdapter

            self._adapter = ApiVLMAdapter(
                provider=self.provider,
                api_key=self.api_key,
            )
        return self._adapter

    def act(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None = None,
    ) -> BenchmarkAction:
        """Use VLM API to determine next action.

        Args:
            observation: Current observation with screenshot.
            task: Task being performed.
            history: Previous observations and actions.

        Returns:
            BenchmarkAction parsed from VLM response.
        """
        adapter = self._get_adapter()

        # Build the sample for the API
        sample = self._build_sample(observation, task, history)

        # Call the VLM API
        try:
            response = adapter.generate(sample, max_new_tokens=self.max_tokens)
        except Exception as e:
            # On API error, return done to avoid infinite loops
            return BenchmarkAction(
                type="done",
                raw_action={"error": str(e)},
            )

        # Parse the response into a BenchmarkAction
        return self._parse_response(response, observation)

    def _build_sample(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None,
    ) -> dict[str, Any]:
        """Build API sample from benchmark observation.

        Args:
            observation: Current observation.
            task: Current task.
            history: Action history.

        Returns:
            Sample dict with 'images' and 'messages'.
        """
        # Build user message content
        content_parts = [f"GOAL: {task.instruction}"]

        # Add context
        if observation.url:
            content_parts.append(f"URL: {observation.url}")
        if observation.window_title:
            content_parts.append(f"Window: {observation.window_title}")

        # Add accessibility tree if available and enabled
        if self.use_accessibility_tree and observation.accessibility_tree:
            tree_str = self._format_accessibility_tree(observation.accessibility_tree)
            # Truncate if too long
            if len(tree_str) > 4000:
                tree_str = tree_str[:4000] + "\n... (truncated)"
            content_parts.append(f"UI Elements:\n{tree_str}")

        # Add history if enabled
        if self.use_history and history:
            history_str = self._format_history(history)
            content_parts.append(f"Previous actions:\n{history_str}")

        content_parts.append("\nWhat is the next action?")

        # Build sample
        sample: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(content_parts)},
            ],
        }

        # Add image if available
        if observation.screenshot_path:
            sample["images"] = [observation.screenshot_path]

        return sample

    def _format_accessibility_tree(self, tree: dict, indent: int = 0) -> str:
        """Format accessibility tree for prompt.

        Args:
            tree: Accessibility tree dict.
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        lines = []
        prefix = "  " * indent

        role = tree.get("role", "unknown")
        name = tree.get("name", "")
        node_id = tree.get("id", tree.get("node_id", ""))

        line = f"{prefix}[{node_id}] {role}"
        if name:
            line += f": {name}"
        lines.append(line)

        for child in tree.get("children", []):
            lines.append(self._format_accessibility_tree(child, indent + 1))

        return "\n".join(lines)

    def _format_history(
        self, history: list[tuple[BenchmarkObservation, BenchmarkAction]]
    ) -> str:
        """Format action history for prompt.

        Args:
            history: List of (observation, action) pairs.

        Returns:
            Formatted string.
        """
        lines = []
        for i, (obs, action) in enumerate(history[-5:], 1):  # Last 5 actions
            action_str = self._action_to_string(action)
            lines.append(f"{i}. {action_str}")
        return "\n".join(lines)

    def _action_to_string(self, action: BenchmarkAction) -> str:
        """Convert BenchmarkAction to string representation.

        Args:
            action: Action to convert.

        Returns:
            String representation.
        """
        if action.type == "click":
            if action.target_node_id:
                return f"CLICK([{action.target_node_id}])"
            if action.target_name:
                return f"CLICK({action.target_name})"
            return f"CLICK({action.x:.3f}, {action.y:.3f})"
        elif action.type == "type":
            return f"TYPE({action.text!r})"
        elif action.type == "key":
            mods = "+".join(action.modifiers or [])
            key = action.key
            if mods:
                return f"KEY({mods}+{key})"
            return f"KEY({key})"
        elif action.type == "scroll":
            return f"SCROLL({action.scroll_direction})"
        elif action.type == "drag":
            return f"DRAG({action.x:.3f}, {action.y:.3f}, {action.end_x:.3f}, {action.end_y:.3f})"
        elif action.type == "done":
            return "DONE()"
        elif action.type == "answer":
            return f"ANSWER({action.answer!r})"
        else:
            return f"{action.type.upper()}()"

    def _parse_response(
        self, response: str, observation: BenchmarkObservation | None = None
    ) -> BenchmarkAction:
        """Parse VLM response into BenchmarkAction.

        Handles various response formats:
        - ACTION: CLICK(0.5, 0.3)
        - CLICK(0.5, 0.3)
        - I'll click at coordinates (0.5, 0.3) -> CLICK(0.5, 0.3)

        Args:
            response: Raw VLM response text.
            observation: Current observation (used for coordinate normalization).

        Returns:
            Parsed BenchmarkAction.
        """
        # Store raw response for debugging
        raw_action = {"response": response}

        # Extract action line (look for ACTION: prefix or action pattern)
        action_line = None

        # Try to find ACTION: prefix
        action_match = re.search(r"ACTION:\s*(.+)", response, re.IGNORECASE)
        if action_match:
            action_line = action_match.group(1).strip()
        else:
            # Look for action pattern anywhere in response
            patterns = [
                r"(CLICK\s*\([^)]+\))",
                r"(TYPE\s*\([^)]+\))",
                r"(KEY\s*\([^)]+\))",
                r"(SCROLL\s*\([^)]+\))",
                r"(DRAG\s*\([^)]+\))",
                r"(DONE\s*\(\s*\))",
                r"(ANSWER\s*\([^)]+\))",
            ]
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    action_line = match.group(1).strip()
                    break

        if not action_line:
            # Could not parse action, return done
            raw_action["parse_error"] = "No action pattern found"
            return BenchmarkAction(type="done", raw_action=raw_action)

        # Parse CLICK action
        click_match = re.match(
            r"CLICK\s*\(\s*\[?(\d+)\]?\s*\)", action_line, re.IGNORECASE
        )
        if click_match:
            # CLICK([id]) - element ID
            node_id = click_match.group(1)
            return BenchmarkAction(
                type="click",
                target_node_id=node_id,
                raw_action=raw_action,
            )

        click_coords = re.match(
            r"CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_line, re.IGNORECASE
        )
        if click_coords:
            # CLICK(x, y) - coordinates
            x = float(click_coords.group(1))
            y = float(click_coords.group(2))

            # Normalize coordinates if they appear to be pixel values
            # If x or y > 1.0, assume pixel coordinates and normalize using viewport
            if observation and observation.viewport and (x > 1.0 or y > 1.0):
                width, height = observation.viewport
                x_norm = x / width
                y_norm = y / height
                raw_action["original_coords"] = {"x": x, "y": y}
                raw_action["normalized"] = True
                x = x_norm
                y = y_norm

            return BenchmarkAction(
                type="click",
                x=x,
                y=y,
                raw_action=raw_action,
            )

        # Parse TYPE action
        type_match = re.match(
            r"TYPE\s*\(\s*[\"'](.+?)[\"']\s*\)", action_line, re.IGNORECASE
        )
        if type_match:
            text = type_match.group(1)
            return BenchmarkAction(
                type="type",
                text=text,
                raw_action=raw_action,
            )

        # Parse KEY action
        key_match = re.match(r"KEY\s*\(\s*(.+?)\s*\)", action_line, re.IGNORECASE)
        if key_match:
            key_str = key_match.group(1)
            # Handle modifier+key format
            if "+" in key_str:
                parts = key_str.split("+")
                key = parts[-1]
                modifiers = parts[:-1]
                return BenchmarkAction(
                    type="key",
                    key=key,
                    modifiers=modifiers,
                    raw_action=raw_action,
                )
            return BenchmarkAction(
                type="key",
                key=key_str,
                raw_action=raw_action,
            )

        # Parse SCROLL action
        scroll_match = re.match(
            r"SCROLL\s*\(\s*(up|down)\s*\)", action_line, re.IGNORECASE
        )
        if scroll_match:
            direction = scroll_match.group(1).lower()
            return BenchmarkAction(
                type="scroll",
                scroll_direction=direction,
                raw_action=raw_action,
            )

        # Parse DRAG action
        drag_match = re.match(
            r"DRAG\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
            action_line,
            re.IGNORECASE,
        )
        if drag_match:
            x = float(drag_match.group(1))
            y = float(drag_match.group(2))
            end_x = float(drag_match.group(3))
            end_y = float(drag_match.group(4))

            # Normalize coordinates if they appear to be pixel values
            if observation and observation.viewport and (x > 1.0 or y > 1.0 or end_x > 1.0 or end_y > 1.0):
                width, height = observation.viewport
                raw_action["original_coords"] = {"x": x, "y": y, "end_x": end_x, "end_y": end_y}
                raw_action["normalized"] = True
                x = x / width
                y = y / height
                end_x = end_x / width
                end_y = end_y / height

            return BenchmarkAction(
                type="drag",
                x=x,
                y=y,
                end_x=end_x,
                end_y=end_y,
                raw_action=raw_action,
            )

        # Parse DONE action
        if re.match(r"DONE\s*\(\s*\)", action_line, re.IGNORECASE):
            return BenchmarkAction(type="done", raw_action=raw_action)

        # Parse ANSWER action
        answer_match = re.match(
            r"ANSWER\s*\(\s*[\"'](.+?)[\"']\s*\)", action_line, re.IGNORECASE
        )
        if answer_match:
            answer = answer_match.group(1)
            return BenchmarkAction(
                type="answer",
                answer=answer,
                raw_action=raw_action,
            )

        # Unknown action format
        raw_action["parse_error"] = f"Unknown action format: {action_line}"
        return BenchmarkAction(type="done", raw_action=raw_action)

    def reset(self) -> None:
        """Reset agent state."""
        # APIBenchmarkAgent is stateless, nothing to reset
        pass
