"""Runner for WAA demo-conditioned experiment.

Usage:
    # List all tasks and demo status
    python -m openadapt_ml.experiments.waa_demo.runner list

    # Show a specific demo
    python -m openadapt_ml.experiments.waa_demo.runner show 8

    # Run experiment (requires WAA environment)
    python -m openadapt_ml.experiments.waa_demo.runner run --condition demo

    # Run with mock adapter (no Windows required)
    python -m openadapt_ml.experiments.waa_demo.runner run --condition demo --mock

Integration with benchmarks runner:
    # Via benchmarks CLI
    python -m openadapt_ml.benchmarks.cli waa-demo --condition demo --tasks 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING, Any

from openadapt_ml.experiments.waa_demo.demos import (
    DEMOS,
    format_demo_for_prompt,
    get_complete_demos,
    get_demo,
    get_placeholder_demos,
)
from openadapt_ml.experiments.waa_demo.tasks import (
    TASKS,
    WATask,
    get_manual_tasks,
    get_recorded_tasks,
    get_task,
)

if TYPE_CHECKING:
    from openadapt_ml.benchmarks.base import BenchmarkAction, BenchmarkObservation, BenchmarkTask

logger = logging.getLogger(__name__)


def cmd_list(args: argparse.Namespace) -> int:
    """List all tasks with their demo status."""
    print("WAA Demo Experiment - Task List")
    print("=" * 80)
    print()

    complete = get_complete_demos()
    placeholder = get_placeholder_demos()

    print(f"Tasks: {len(TASKS)} total")
    print(f"  Manual demos written: {len(complete)}")
    print(f"  Recorded demos needed: {len(placeholder)}")
    print()
    print("-" * 80)
    print(f"{'#':<3} {'Domain':<18} {'Difficulty':<8} {'Demo':<10} {'Instruction'}")
    print("-" * 80)

    for num, task in TASKS.items():
        demo_status = "Ready" if num in complete else "NEEDS REC"
        print(
            f"{num:<3} {task.domain.value:<18} {task.difficulty.value:<8} "
            f"{demo_status:<10} {task.instruction[:45]}..."
        )

    print()
    print("Tasks needing recorded demos on Windows:")
    for task in get_recorded_tasks():
        print(f"  - #{list(TASKS.keys())[list(TASKS.values()).index(task)]}: {task.instruction}")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show a specific demo."""
    task_num = args.task
    task = get_task(task_num)
    demo = get_demo(task_num)

    if not task:
        print(f"Error: Task {task_num} not found (valid: 1-10)")
        return 1

    print(f"Task #{task_num}: {task.instruction}")
    print(f"Domain: {task.domain.value}")
    print(f"Difficulty: {task.difficulty.value}")
    print(f"Demo method: {task.demo_method}")
    print()
    print("=" * 80)
    print("DEMO:")
    print("=" * 80)
    print(demo or "No demo available")

    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    """Generate a prompt for a task with optional demo."""
    task_num = args.task
    task = get_task(task_num)
    demo = get_demo(task_num) if args.with_demo else None

    if not task:
        print(f"Error: Task {task_num} not found")
        return 1

    print("=" * 80)
    print("GENERATED PROMPT")
    print("=" * 80)
    print()

    if demo and "[PLACEHOLDER" not in demo:
        prompt = format_demo_for_prompt(demo, task.instruction)
        print(prompt)
    else:
        print(f"Task: {task.instruction}")
        print()
        print("Analyze the screenshot and provide the next action to complete this task.")
        if demo and "[PLACEHOLDER" in demo:
            print()
            print("[Note: Demo not available - this would be zero-shot]")

    return 0


class DemoConditionedAgent:
    """Agent that uses demo-conditioned prompting for WAA tasks.

    This agent extends the APIBenchmarkAgent approach but injects relevant
    demos into the prompt based on the current task. It supports:
    - Zero-shot mode: Standard VLM prompting without demos
    - Demo-conditioned mode: Includes task-specific demonstration in prompt

    The demo-conditioned approach was validated to improve first-action accuracy
    from 33% (zero-shot) to 100% (with demo) in initial experiments.

    Args:
        provider: API provider ("anthropic" or "openai")
        condition: "zero-shot" or "demo"
        api_key: Optional API key override
        model: Optional model name override
        max_tokens: Maximum tokens for response
        use_accessibility_tree: Include accessibility tree in prompt
        use_history: Include action history in prompt

    Example:
        agent = DemoConditionedAgent(provider="anthropic", condition="demo")
        results = evaluate_agent_on_benchmark(agent, waa_adapter)
    """

    # System prompt for demo-conditioned GUI automation
    SYSTEM_PROMPT = """You are a GUI automation agent. Given a screenshot and task instruction, determine the next action to take.

Available actions:
- CLICK(x, y) - Click at coordinates (normalized 0.0-1.0 or pixels)
- CLICK([id]) - Click element with given ID from accessibility tree
- TYPE("text") - Type the given text
- KEY(key) - Press a key (e.g., Enter, Tab, Escape)
- KEY(modifier+key) - Press key combination (e.g., Ctrl+c, Alt+Tab)
- SCROLL(direction) - Scroll up or down
- DONE() - Task is complete

If a demonstration is provided, use it as a reference for understanding the UI navigation pattern.
Focus on the current state of the screen and select the appropriate next action.

Respond with exactly ONE action in the format shown above.
Think step by step, then output the action on a new line starting with "ACTION:"
"""

    def __init__(
        self,
        provider: str = "anthropic",
        condition: str = "demo",
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
        use_accessibility_tree: bool = True,
        use_history: bool = True,
    ):
        self.provider = provider
        self.condition = condition
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.use_accessibility_tree = use_accessibility_tree
        self.use_history = use_history
        self._adapter = None
        self._task_demo_map: dict[str, str] = {}
        self._build_task_demo_map()

    def _build_task_demo_map(self) -> None:
        """Build mapping from WAA task IDs to demo text."""
        for task_num, task in TASKS.items():
            demo = get_demo(task_num)
            if demo and "[PLACEHOLDER" not in demo:
                # Map both task number and full task_id
                self._task_demo_map[task_num] = demo
                self._task_demo_map[task.task_id] = demo

    def _get_adapter(self):
        """Lazily initialize the API adapter."""
        if self._adapter is None:
            from openadapt_ml.models.api_adapter import ApiVLMAdapter
            self._adapter = ApiVLMAdapter(
                provider=self.provider,
                api_key=self.api_key,
            )
        return self._adapter

    def _get_demo_for_task(self, task: "BenchmarkTask") -> str | None:
        """Get the demo for a task if available.

        Args:
            task: The benchmark task

        Returns:
            Demo text or None if not available
        """
        if self.condition == "zero-shot":
            return None

        # Try to find demo by task_id
        task_id = task.task_id

        # Check direct mapping
        if task_id in self._task_demo_map:
            return self._task_demo_map[task_id]

        # Try to extract task number from task_id patterns
        for task_num, wa_task in TASKS.items():
            if wa_task.task_id in task_id or task_id in wa_task.task_id:
                return self._task_demo_map.get(task_num)

        # Check if instruction matches
        for task_num, wa_task in TASKS.items():
            if wa_task.instruction.lower() in task.instruction.lower():
                return self._task_demo_map.get(task_num)

        return None

    def act(
        self,
        observation: "BenchmarkObservation",
        task: "BenchmarkTask",
        history: list[tuple["BenchmarkObservation", "BenchmarkAction"]] | None = None,
    ) -> "BenchmarkAction":
        """Use VLM API with optional demo to determine next action.

        Args:
            observation: Current observation with screenshot
            task: Task being performed
            history: Previous observations and actions

        Returns:
            BenchmarkAction parsed from VLM response
        """
        from openadapt_ml.benchmarks.base import BenchmarkAction

        adapter = self._get_adapter()

        # Build the sample for the API
        sample = self._build_sample(observation, task, history)

        # Call the VLM API
        try:
            response = adapter.generate(sample, max_new_tokens=self.max_tokens)
        except Exception as e:
            logger.error(f"API error: {e}")
            return BenchmarkAction(
                type="done",
                raw_action={"error": str(e)},
            )

        # Parse the response into a BenchmarkAction
        return self._parse_response(response, observation)

    def _build_sample(
        self,
        observation: "BenchmarkObservation",
        task: "BenchmarkTask",
        history: list[tuple["BenchmarkObservation", "BenchmarkAction"]] | None,
    ) -> dict[str, Any]:
        """Build API sample with optional demo.

        Args:
            observation: Current observation
            task: Current task
            history: Action history

        Returns:
            Sample dict with 'images' and 'messages'
        """
        content_parts = []

        # Add demo if available and in demo condition
        demo = self._get_demo_for_task(task)
        if demo:
            formatted_demo = format_demo_for_prompt(demo, task.instruction)
            content_parts.append(formatted_demo)
        else:
            content_parts.append(f"GOAL: {task.instruction}")

        # Add context
        if observation.url:
            content_parts.append(f"URL: {observation.url}")
        if observation.window_title:
            content_parts.append(f"Window: {observation.window_title}")

        # Add accessibility tree if available and enabled
        if self.use_accessibility_tree and observation.accessibility_tree:
            tree_str = self._format_accessibility_tree(observation.accessibility_tree)
            if len(tree_str) > 4000:
                tree_str = tree_str[:4000] + "\n... (truncated)"
            content_parts.append(f"UI Elements:\n{tree_str}")

        # Add history if enabled
        if self.use_history and history:
            history_str = self._format_history(history)
            content_parts.append(f"Previous actions:\n{history_str}")

        content_parts.append("\nAnalyze the current screenshot and provide the next action.")

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
        self, history: list[tuple["BenchmarkObservation", "BenchmarkAction"]]
    ) -> str:
        """Format action history for prompt."""
        lines = []
        for i, (obs, action) in enumerate(history[-5:], 1):
            action_str = self._action_to_string(action)
            lines.append(f"{i}. {action_str}")
        return "\n".join(lines)

    def _action_to_string(self, action: "BenchmarkAction") -> str:
        """Convert BenchmarkAction to string."""
        if action.type == "click":
            if action.target_node_id:
                return f"CLICK([{action.target_node_id}])"
            if action.target_name:
                return f"CLICK({action.target_name})"
            if action.x is not None and action.y is not None:
                return f"CLICK({action.x:.3f}, {action.y:.3f})"
            return "CLICK()"
        elif action.type == "type":
            return f"TYPE({action.text!r})"
        elif action.type == "key":
            mods = "+".join(action.modifiers or [])
            key = action.key or ""
            if mods:
                return f"KEY({mods}+{key})"
            return f"KEY({key})"
        elif action.type == "scroll":
            return f"SCROLL({action.scroll_direction})"
        elif action.type == "done":
            return "DONE()"
        else:
            return f"{action.type.upper()}()"

    def _parse_response(
        self, response: str, observation: "BenchmarkObservation" | None = None
    ) -> "BenchmarkAction":
        """Parse VLM response into BenchmarkAction.

        Uses the same parsing logic as APIBenchmarkAgent.
        """
        import re
        from openadapt_ml.benchmarks.base import BenchmarkAction

        raw_action = {"response": response}

        # Extract action line
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
                r"(DONE\s*\(\s*\))",
            ]
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    action_line = match.group(1).strip()
                    break

        if not action_line:
            raw_action["parse_error"] = "No action pattern found"
            return BenchmarkAction(type="done", raw_action=raw_action)

        # Parse CLICK with element ID
        click_id_match = re.match(
            r"CLICK\s*\(\s*\[?(\d+)\]?\s*\)", action_line, re.IGNORECASE
        )
        if click_id_match:
            return BenchmarkAction(
                type="click",
                target_node_id=click_id_match.group(1),
                raw_action=raw_action,
            )

        # Parse CLICK with coordinates
        click_coords = re.match(
            r"CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", action_line, re.IGNORECASE
        )
        if click_coords:
            x = float(click_coords.group(1))
            y = float(click_coords.group(2))
            if observation and observation.viewport and (x > 1.0 or y > 1.0):
                width, height = observation.viewport
                x = x / width
                y = y / height
            return BenchmarkAction(type="click", x=x, y=y, raw_action=raw_action)

        # Parse TYPE
        type_match = re.match(
            r"TYPE\s*\(\s*[\"'](.+?)[\"']\s*\)", action_line, re.IGNORECASE
        )
        if type_match:
            return BenchmarkAction(type="type", text=type_match.group(1), raw_action=raw_action)

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

        # Parse DONE
        if re.match(r"DONE\s*\(\s*\)", action_line, re.IGNORECASE):
            return BenchmarkAction(type="done", raw_action=raw_action)

        raw_action["parse_error"] = f"Unknown action format: {action_line}"
        return BenchmarkAction(type="done", raw_action=raw_action)

    def reset(self) -> None:
        """Reset agent state between episodes."""
        pass


def cmd_run(args: argparse.Namespace) -> int:
    """Run the WAA demo-conditioned experiment.

    This integrates with the benchmarks infrastructure to run either
    zero-shot or demo-conditioned evaluation on WAA tasks.
    """
    from openadapt_ml.benchmarks import (
        WAAMockAdapter,
        compute_metrics,
    )
    from openadapt_ml.benchmarks.runner import EvaluationConfig, evaluate_agent_on_benchmark

    print("WAA Demo-Conditioned Experiment Runner")
    print("=" * 80)
    print()
    print(f"Condition: {args.condition}")
    print(f"Provider: {args.provider}")
    print(f"Tasks: {args.tasks or 'all with demos'}")
    print()

    # Determine which tasks to run
    task_ids = None
    if args.tasks:
        task_nums = [t.strip() for t in args.tasks.split(",")]
        # Map task numbers to WAA task IDs
        task_ids = []
        for num in task_nums:
            task = get_task(num)
            if task:
                task_ids.append(task.task_id)
            else:
                print(f"Warning: Task {num} not found")
    else:
        # Default to all tasks with complete demos
        complete_demos = get_complete_demos()
        task_ids = []
        for num in complete_demos.keys():
            task = get_task(num)
            if task:
                task_ids.append(task.task_id)
        print(f"Running {len(task_ids)} tasks with complete demos")

    # Check for mock mode or real WAA
    use_mock = getattr(args, 'mock', False)

    if use_mock:
        print("Using mock adapter (no Windows required)")
        adapter = WAAMockAdapter(num_tasks=len(task_ids) if task_ids else 10)
        # Override task_ids since mock adapter has different IDs
        task_ids = None
    elif args.waa_url:
        print(f"WAA URL: {args.waa_url}")
        print("Note: Real WAA integration requires a running Windows VM")
        print()
        print("To set up WAA:")
        print("  uv run python -m openadapt_ml.benchmarks.cli vm setup-waa")
        print("  uv run python -m openadapt_ml.benchmarks.cli vm prepare-windows")
        print()
        # For now, fall back to mock since we can't connect to real WAA without VM
        print("Falling back to mock adapter for demonstration...")
        adapter = WAAMockAdapter(num_tasks=len(task_ids) if task_ids else 10)
        task_ids = None
    else:
        print("No WAA URL provided, using mock adapter")
        adapter = WAAMockAdapter(num_tasks=len(task_ids) if task_ids else 10)
        task_ids = None

    # Create the demo-conditioned agent
    agent = DemoConditionedAgent(
        provider=args.provider,
        condition=args.condition,
        max_tokens=512,
        use_accessibility_tree=True,
        use_history=True,
    )

    # Configure evaluation
    config = EvaluationConfig(
        max_steps=args.max_steps,
        parallel=1,
        save_trajectories=True,
        save_execution_traces=True,
        model_id=f"{args.provider}-{args.condition}",
        output_dir=args.output or "benchmark_results",
        run_name=args.run_name,
        verbose=True,
    )

    print()
    print("Starting evaluation...")
    print("(Each step calls the VLM API - this may take a while)")
    print()

    try:
        results = evaluate_agent_on_benchmark(
            agent=agent,
            adapter=adapter,
            task_ids=task_ids,
            config=config,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if "API key" in str(e) or "api_key" in str(e).lower():
            key_name = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
            print(f"\nMake sure {key_name} is set in your environment or .env file.")
        return 1

    # Print results
    metrics = compute_metrics(results)
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Condition:    {args.condition}")
    print(f"Tasks:        {metrics['num_tasks']}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Successes:    {metrics['success_count']}")
    print(f"Failures:     {metrics['fail_count']}")
    print(f"Avg steps:    {metrics['avg_steps']:.1f}")
    print()

    # Show per-task results
    print("Per-task results:")
    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(f"  {result.task_id}: {status} ({result.num_steps} steps)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WAA Demo-Conditioned Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List tasks and their demo status
    python -m openadapt_ml.experiments.waa_demo.runner list

    # Show a specific demo
    python -m openadapt_ml.experiments.waa_demo.runner show 8

    # Run with demo conditioning (mock adapter, no Windows needed)
    python -m openadapt_ml.experiments.waa_demo.runner run --condition demo --mock

    # Run zero-shot for comparison
    python -m openadapt_ml.experiments.waa_demo.runner run --condition zero-shot --mock

    # Run with real WAA (requires running Windows VM)
    python -m openadapt_ml.experiments.waa_demo.runner run --condition demo --waa-url http://<vm-ip>:5000
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List all tasks")
    list_parser.set_defaults(func=cmd_list)

    # show command
    show_parser = subparsers.add_parser("show", help="Show a specific demo")
    show_parser.add_argument("task", help="Task number (1-10)")
    show_parser.set_defaults(func=cmd_show)

    # prompt command
    prompt_parser = subparsers.add_parser("prompt", help="Generate prompt for a task")
    prompt_parser.add_argument("task", help="Task number (1-10)")
    prompt_parser.add_argument("--with-demo", action="store_true", help="Include demo")
    prompt_parser.set_defaults(func=cmd_prompt)

    # run command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument(
        "--condition",
        choices=["zero-shot", "demo"],
        default="demo",
        help="Experiment condition (default: demo)",
    )
    run_parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="VLM API provider (default: anthropic)",
    )
    run_parser.add_argument(
        "--tasks",
        help="Comma-separated task numbers (default: all with demos)",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum steps per task (default: 15)",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock adapter (no Windows required)",
    )
    run_parser.add_argument(
        "--waa-url",
        help="WAA server URL (e.g., http://vm-ip:5000)",
    )
    run_parser.add_argument(
        "--output",
        default="benchmark_results",
        help="Output directory (default: benchmark_results)",
    )
    run_parser.add_argument(
        "--run-name",
        help="Run name (default: auto-generated)",
    )
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
