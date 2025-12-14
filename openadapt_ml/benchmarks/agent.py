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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from openadapt_ml.benchmarks.base import (
    BenchmarkAction,
    BenchmarkObservation,
    BenchmarkTask,
)

if TYPE_CHECKING:
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
