"""ML-specific agents for benchmark evaluation.

This module provides agents that wrap openadapt-ml components (VLM adapters,
policies, baselines) for benchmark evaluation.

For standalone agents without ML dependencies, use openadapt_evals:
    from openadapt_evals import ApiAgent, ScriptedAgent, RandomAgent

ML-specific agents in this module:
    - PolicyAgent: Wraps openadapt_ml.runtime.policy.AgentPolicy
    - APIBenchmarkAgent: Uses openadapt_ml.models.api_adapter.ApiVLMAdapter
    - UnifiedBaselineAgent: Uses openadapt_ml.baselines adapters

Example:
    from openadapt_ml.benchmarks import PolicyAgent
    from openadapt_ml.runtime.policy import AgentPolicy

    policy = AgentPolicy(adapter)
    agent = PolicyAgent(policy)
    results = evaluate_agent_on_benchmark(agent, benchmark_adapter)

    # API-backed agents (GPT-5.1, Claude) using openadapt-ml adapters
    from openadapt_ml.benchmarks import APIBenchmarkAgent

    agent = APIBenchmarkAgent(provider="anthropic")  # Uses Claude
    agent = APIBenchmarkAgent(provider="openai")     # Uses GPT-5.1
    results = evaluate_agent_on_benchmark(agent, benchmark_adapter)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

# Import base classes from openadapt-evals (canonical location)
from openadapt_evals import (
    BenchmarkAction,
    BenchmarkAgent,
    BenchmarkObservation,
    BenchmarkTask,
)

if TYPE_CHECKING:
    from openadapt_ml.models.api_adapter import ApiVLMAdapter
    from openadapt_ml.runtime.policy import AgentPolicy
    from openadapt_ml.schema import Action


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
        """Build SFT-style sample from benchmark observation."""
        content_parts = [f"Goal: {task.instruction}"]

        if self.use_accessibility_tree and observation.accessibility_tree:
            tree_str = self._format_accessibility_tree(observation.accessibility_tree)
            content_parts.append(f"UI Elements:\n{tree_str}")

        if observation.url:
            content_parts.append(f"URL: {observation.url}")
        if observation.window_title:
            content_parts.append(f"Window: {observation.window_title}")

        if self.use_history and history:
            history_str = self._format_history(history)
            content_parts.append(f"Previous actions:\n{history_str}")

        content_parts.append("What action should be taken next?")

        sample = {
            "messages": [
                {"role": "user", "content": "\n\n".join(content_parts)},
            ],
        }

        if observation.screenshot_path:
            sample["images"] = [observation.screenshot_path]

        return sample

    def _format_accessibility_tree(self, tree: dict, indent: int = 0) -> str:
        """Format accessibility tree for prompt."""
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
        """Format action history for prompt."""
        lines = []
        for i, (obs, action) in enumerate(history[-5:], 1):
            action_str = self._action_to_string(action)
            lines.append(f"{i}. {action_str}")
        return "\n".join(lines)

    def _action_to_string(self, action: BenchmarkAction) -> str:
        """Convert BenchmarkAction to string representation."""
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
        """Convert openadapt-ml Action to BenchmarkAction."""
        x, y = None, None
        if action.normalized_coordinates is not None:
            x, y = action.normalized_coordinates

        end_x, end_y = None, None
        if action.normalized_end is not None:
            end_x, end_y = action.normalized_end

        action_type = (
            action.type.value if hasattr(action.type, "value") else action.type
        )

        target_node_id = None
        target_role = None
        target_name = None
        target_bbox = None
        if action.element is not None:
            target_node_id = action.element.element_id
            target_role = action.element.role
            target_name = action.element.name
            if action.element.bounds is not None:
                target_bbox = (
                    action.element.bounds.x,
                    action.element.bounds.y,
                    action.element.bounds.x + action.element.bounds.width,
                    action.element.bounds.y + action.element.bounds.height,
                )

        return BenchmarkAction(
            type=action_type,
            x=x,
            y=y,
            text=action.text,
            target_bbox=target_bbox,
            target_node_id=target_node_id,
            target_role=target_role,
            target_name=target_name,
            key=getattr(action, "key", None),
            modifiers=getattr(action, "modifiers", None),
            scroll_direction=getattr(action, "scroll_direction", None),
            scroll_amount=getattr(action, "scroll_amount", None),
            end_x=end_x,
            end_y=end_y,
            answer=getattr(action, "answer", None),
            raw_action={"thought": thought} if thought else None,
        )

    def reset(self) -> None:
        """Reset agent state."""
        pass


class APIBenchmarkAgent(BenchmarkAgent):
    """Agent that uses hosted VLM APIs via openadapt-ml ApiVLMAdapter.

    This agent wraps ApiVLMAdapter to provide Claude or GPT-5.1 baselines
    for benchmark evaluation. It converts BenchmarkObservation to the
    API format and parses VLM responses into BenchmarkActions.

    Note: For standalone API evaluation without openadapt-ml, use
    openadapt_evals.ApiAgent instead (has P0 demo persistence fix).

    Args:
        provider: API provider - "anthropic" (Claude) or "openai" (GPT-5.1).
        api_key: Optional API key override. If not provided, uses env vars.
        model: Optional model name override.
        max_tokens: Maximum tokens for VLM response.
        use_accessibility_tree: Whether to include accessibility tree in prompt.
        use_history: Whether to include action history in prompt.
    """

    SYSTEM_PROMPT = """You are a GUI automation agent. Given a screenshot and task instruction, determine the next action to take.

Available actions:
- CLICK(x, y) - Click at coordinates (can be pixel values or normalized 0.0-1.0)
- CLICK([id]) - Click element with given ID from accessibility tree
- TYPE("text") - Type the given text
- KEY(key) - Press a key (e.g., Enter, Tab, Escape)
- KEY(modifier+key) - Press key combination (e.g., Ctrl+c, Alt+Tab)
- SCROLL(direction) - Scroll up or down
- DRAG(x1, y1, x2, y2) - Drag from (x1,y1) to (x2,y2)
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
        """Use VLM API to determine next action."""
        adapter = self._get_adapter()
        sample = self._build_sample(observation, task, history)

        try:
            response = adapter.generate(sample, max_new_tokens=self.max_tokens)
        except Exception as e:
            return BenchmarkAction(type="done", raw_action={"error": str(e)})

        return self._parse_response(response, observation)

    def _build_sample(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None,
    ) -> dict[str, Any]:
        """Build API sample from benchmark observation."""
        content_parts = [f"GOAL: {task.instruction}"]

        if observation.url:
            content_parts.append(f"URL: {observation.url}")
        if observation.window_title:
            content_parts.append(f"Window: {observation.window_title}")

        if self.use_accessibility_tree and observation.accessibility_tree:
            tree_str = self._format_accessibility_tree(observation.accessibility_tree)
            if len(tree_str) > 4000:
                tree_str = tree_str[:4000] + "\n... (truncated)"
            content_parts.append(f"UI Elements:\n{tree_str}")

        if self.use_history and history:
            history_str = self._format_history(history)
            content_parts.append(f"Previous actions:\n{history_str}")

        content_parts.append("\nWhat is the next action?")

        sample: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(content_parts)},
            ],
        }

        if observation.screenshot_path:
            sample["images"] = [observation.screenshot_path]

        return sample

    def _format_accessibility_tree(self, tree: dict, indent: int = 0) -> str:
        """Format accessibility tree for prompt."""
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
        """Format action history for prompt."""
        lines = []
        for i, (obs, action) in enumerate(history[-5:], 1):
            action_str = self._action_to_string(action)
            lines.append(f"{i}. {action_str}")
        return "\n".join(lines)

    def _action_to_string(self, action: BenchmarkAction) -> str:
        """Convert BenchmarkAction to string representation."""
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
        """Parse VLM response into BenchmarkAction."""
        raw_action = {"response": response}

        action_line = None
        action_match = re.search(r"ACTION:\s*(.+)", response, re.IGNORECASE)
        if action_match:
            action_line = action_match.group(1).strip()
        else:
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
            raw_action["parse_error"] = "No action pattern found"
            return BenchmarkAction(type="done", raw_action=raw_action)

        # Parse CLICK([id])
        click_match = re.match(
            r"CLICK\s*\(\s*\[?(\d+)\]?\s*\)", action_line, re.IGNORECASE
        )
        if click_match:
            node_id = click_match.group(1)
            return BenchmarkAction(
                type="click", target_node_id=node_id, raw_action=raw_action
            )

        # Parse CLICK(x, y)
        click_coords = re.match(
            r"CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_line, re.IGNORECASE
        )
        if click_coords:
            x = float(click_coords.group(1))
            y = float(click_coords.group(2))
            if observation and observation.viewport and (x > 1.0 or y > 1.0):
                width, height = observation.viewport
                raw_action["original_coords"] = {"x": x, "y": y}
                raw_action["normalized"] = True
                x, y = x / width, y / height
            return BenchmarkAction(type="click", x=x, y=y, raw_action=raw_action)

        # Parse TYPE
        type_match = re.match(
            r"TYPE\s*\(\s*[\"'](.+?)[\"']\s*\)", action_line, re.IGNORECASE
        )
        if type_match:
            return BenchmarkAction(
                type="type", text=type_match.group(1), raw_action=raw_action
            )

        # Parse KEY
        key_match = re.match(r"KEY\s*\(\s*(.+?)\s*\)", action_line, re.IGNORECASE)
        if key_match:
            key_str = key_match.group(1)
            if "+" in key_str:
                parts = key_str.split("+")
                return BenchmarkAction(
                    type="key",
                    key=parts[-1],
                    modifiers=parts[:-1],
                    raw_action=raw_action,
                )
            return BenchmarkAction(type="key", key=key_str, raw_action=raw_action)

        # Parse SCROLL
        scroll_match = re.match(
            r"SCROLL\s*\(\s*(up|down)\s*\)", action_line, re.IGNORECASE
        )
        if scroll_match:
            return BenchmarkAction(
                type="scroll",
                scroll_direction=scroll_match.group(1).lower(),
                raw_action=raw_action,
            )

        # Parse DRAG
        drag_match = re.match(
            r"DRAG\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
            action_line,
            re.IGNORECASE,
        )
        if drag_match:
            x, y = float(drag_match.group(1)), float(drag_match.group(2))
            end_x, end_y = float(drag_match.group(3)), float(drag_match.group(4))
            if (
                observation
                and observation.viewport
                and (x > 1.0 or y > 1.0 or end_x > 1.0 or end_y > 1.0)
            ):
                width, height = observation.viewport
                raw_action["original_coords"] = {
                    "x": x,
                    "y": y,
                    "end_x": end_x,
                    "end_y": end_y,
                }
                raw_action["normalized"] = True
                x, y, end_x, end_y = (
                    x / width,
                    y / height,
                    end_x / width,
                    end_y / height,
                )
            return BenchmarkAction(
                type="drag", x=x, y=y, end_x=end_x, end_y=end_y, raw_action=raw_action
            )

        # Parse DONE
        if re.match(r"DONE\s*\(\s*\)", action_line, re.IGNORECASE):
            return BenchmarkAction(type="done", raw_action=raw_action)

        # Parse ANSWER
        answer_match = re.match(
            r"ANSWER\s*\(\s*[\"'](.+?)[\"']\s*\)", action_line, re.IGNORECASE
        )
        if answer_match:
            return BenchmarkAction(
                type="answer", answer=answer_match.group(1), raw_action=raw_action
            )

        raw_action["parse_error"] = f"Unknown action format: {action_line}"
        return BenchmarkAction(type="done", raw_action=raw_action)

    def reset(self) -> None:
        """Reset agent state."""
        pass


class UnifiedBaselineAgent(BenchmarkAgent):
    """Agent that uses UnifiedBaselineAdapter for benchmark evaluation.

    Provides unified interface for Claude, GPT, and Gemini baselines
    across multiple tracks (A: coordinates, B: ReAct, C: SoM).

    Args:
        model_alias: Model alias (e.g., 'claude-opus-4.5', 'gpt-5.2').
        track: Track type ('A', 'B', or 'C'). Defaults to 'A'.
        api_key: Optional API key override.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens for response.
        demo: Optional demo text for prompts.
        verbose: Whether to print debug output.
    """

    def __init__(
        self,
        model_alias: str = "claude-opus-4.5",
        track: str = "A",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        demo: str | None = None,
        verbose: bool = False,
    ):
        self.model_alias = model_alias
        self.track = track.upper()
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.demo = demo
        self.verbose = verbose
        self._adapter = None

    def _get_adapter(self):
        """Lazily initialize the UnifiedBaselineAdapter."""
        if self._adapter is None:
            from openadapt_ml.baselines import TrackConfig, UnifiedBaselineAdapter

            track_configs = {
                "A": TrackConfig.track_a(),
                "B": TrackConfig.track_b(),
                "C": TrackConfig.track_c(),
            }
            track_config = track_configs.get(self.track, TrackConfig.track_a())

            self._adapter = UnifiedBaselineAdapter.from_alias(
                self.model_alias,
                track=track_config,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                demo=self.demo,
                verbose=self.verbose,
            )
        return self._adapter

    def act(
        self,
        observation: BenchmarkObservation,
        task: BenchmarkTask,
        history: list[tuple[BenchmarkObservation, BenchmarkAction]] | None = None,
    ) -> BenchmarkAction:
        """Use UnifiedBaselineAdapter to determine next action."""
        from PIL import Image

        adapter = self._get_adapter()

        screenshot = None
        if observation.screenshot_path:
            try:
                screenshot = Image.open(observation.screenshot_path)
            except Exception as e:
                if self.verbose:
                    print(f"[UnifiedBaselineAgent] Failed to load screenshot: {e}")

        a11y_tree = (
            observation.accessibility_tree if observation.accessibility_tree else None
        )

        adapter_history = None
        if history:
            adapter_history = [
                self._benchmark_action_to_dict(a) for _, a in history[-5:]
            ]

        try:
            parsed_action = adapter.predict(
                screenshot=screenshot,
                goal=task.instruction,
                a11y_tree=a11y_tree,
                history=adapter_history,
            )
        except Exception as e:
            if self.verbose:
                print(f"[UnifiedBaselineAgent] Adapter error: {e}")
            return BenchmarkAction(type="done", raw_action={"error": str(e)})

        return self._parsed_to_benchmark_action(parsed_action, observation)

    def _benchmark_action_to_dict(self, action: BenchmarkAction) -> dict[str, Any]:
        """Convert BenchmarkAction to dict for history."""
        result = {"type": action.type}
        if action.x is not None:
            result["x"] = action.x
        if action.y is not None:
            result["y"] = action.y
        if action.text:
            result["text"] = action.text
        if action.key:
            result["key"] = action.key
        if action.target_node_id:
            result["element_id"] = action.target_node_id
        if action.scroll_direction:
            result["direction"] = action.scroll_direction
        return result

    def _parsed_to_benchmark_action(
        self, parsed_action, observation: BenchmarkObservation | None = None
    ) -> BenchmarkAction:
        """Convert ParsedAction to BenchmarkAction."""
        raw_action = {
            "raw_response": parsed_action.raw_response,
            "thought": parsed_action.thought,
        }

        if not parsed_action.is_valid:
            raw_action["parse_error"] = parsed_action.parse_error
            return BenchmarkAction(type="done", raw_action=raw_action)

        action_type = parsed_action.action_type

        if action_type == "click":
            if parsed_action.element_id is not None:
                return BenchmarkAction(
                    type="click",
                    target_node_id=str(parsed_action.element_id),
                    raw_action=raw_action,
                )
            elif parsed_action.x is not None and parsed_action.y is not None:
                x, y = parsed_action.x, parsed_action.y
                if observation and observation.viewport and (x > 1.0 or y > 1.0):
                    width, height = observation.viewport
                    raw_action["original_coords"] = {"x": x, "y": y}
                    x, y = x / width, y / height
                return BenchmarkAction(type="click", x=x, y=y, raw_action=raw_action)

        elif action_type == "type":
            return BenchmarkAction(
                type="type", text=parsed_action.text, raw_action=raw_action
            )

        elif action_type == "key":
            return BenchmarkAction(
                type="key", key=parsed_action.key, raw_action=raw_action
            )

        elif action_type == "scroll":
            return BenchmarkAction(
                type="scroll",
                scroll_direction=parsed_action.direction,
                raw_action=raw_action,
            )

        elif action_type == "done":
            return BenchmarkAction(type="done", raw_action=raw_action)

        elif action_type == "drag":
            return BenchmarkAction(
                type="drag",
                x=parsed_action.x,
                y=parsed_action.y,
                end_x=getattr(parsed_action, "end_x", None),
                end_y=getattr(parsed_action, "end_y", None),
                raw_action=raw_action,
            )

        raw_action["unknown_action"] = action_type
        return BenchmarkAction(type="done", raw_action=raw_action)

    def reset(self) -> None:
        """Reset agent state."""
        pass

    def __repr__(self) -> str:
        return f"UnifiedBaselineAgent(model={self.model_alias}, track={self.track})"
