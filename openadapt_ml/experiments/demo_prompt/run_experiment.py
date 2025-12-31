"""Demo-conditioned prompt experiment runner.

Tests whether including a human demonstration improves VLM performance.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openadapt_ml.experiments.demo_prompt.format_demo import (
    format_episode_as_demo,
    format_episode_verbose,
    generate_length_matched_control,
    get_demo_screenshot_paths,
)


# System prompt for GUI automation
SYSTEM_PROMPT = """You are a GUI automation agent. Given a screenshot and task instruction, determine the next action to take.

Available actions:
- CLICK(x, y) - Click at normalized coordinates (0.0-1.0)
- TYPE("text") - Type the given text
- KEY(key) - Press a key (e.g., Enter, Tab, Escape)
- KEY(modifier+key) - Press key combination (e.g., Cmd+c, Ctrl+v)
- SCROLL(direction) - Scroll up or down
- DONE() - Task is complete

Respond with exactly ONE action.
Think step by step, then output the action on a new line starting with "ACTION:"
"""


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    task: str
    condition: str  # "zero_shot", "with_demo", "control"
    response: str
    action_parsed: str | None
    success: bool | None  # None if not evaluated
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TaskPair:
    """A demo-test task pair."""

    demo_task: str
    test_task: str
    similarity: str  # "near", "medium", "far"
    demo_capture_path: str | None = None
    test_screenshot_path: str | None = None


class DemoPromptExperiment:
    """Run demo-conditioned prompt experiments."""

    def __init__(
        self,
        provider: str = "anthropic",
        max_tokens: int = 512,
        verbose: bool = True,
    ):
        """Initialize experiment.

        Args:
            provider: API provider ("anthropic" or "openai").
            max_tokens: Maximum tokens for response.
            verbose: Whether to print progress.
        """
        self.provider = provider
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._client = None

    def _get_client(self) -> Any:
        """Lazily initialize API client."""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            from anthropic import Anthropic
            from openadapt_ml.config import settings
            import os

            key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = Anthropic(api_key=key)

        elif self.provider == "openai":
            from openai import OpenAI
            from openadapt_ml.config import settings
            import os

            key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=key)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def _call_api(
        self,
        user_content: str,
        image_paths: list[str] | None = None,
    ) -> str:
        """Call the API with text and optional images.

        Args:
            user_content: User message text.
            image_paths: Optional list of image paths to include.

        Returns:
            Model response text.
        """
        client = self._get_client()

        if self.provider == "anthropic":
            content: list[dict[str, Any]] = []

            # Add images first
            if image_paths:
                for path in image_paths[:5]:  # Limit to 5 images
                    if Path(path).exists():
                        with open(path, "rb") as f:
                            image_b64 = base64.b64encode(f.read()).decode("utf-8")
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        })

            # Add text
            content.append({"type": "text", "text": user_content})

            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )

            parts = getattr(response, "content", [])
            texts = [getattr(p, "text", "") for p in parts if getattr(p, "type", "") == "text"]
            return "\n".join([t for t in texts if t]).strip()

        elif self.provider == "openai":
            user_content_parts: list[dict[str, Any]] = []

            # Add images first
            if image_paths:
                for path in image_paths[:5]:
                    if Path(path).exists():
                        with open(path, "rb") as f:
                            image_b64 = base64.b64encode(f.read()).decode("utf-8")
                        user_content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        })

            # Add text
            user_content_parts.append({"type": "text", "text": user_content})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content_parts},
                ],
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content or ""

        raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_action(self, response: str) -> str | None:
        """Extract action from response.

        Args:
            response: Model response text.

        Returns:
            Extracted action string or None.
        """
        import re

        # Look for ACTION: prefix
        match = re.search(r"ACTION:\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Look for action patterns
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
                return match.group(1).strip()

        return None

    def run_zero_shot(
        self,
        task: str,
        screenshot_path: str,
    ) -> ExperimentResult:
        """Run zero-shot (no demo) condition.

        Args:
            task: Task instruction.
            screenshot_path: Path to current screenshot.

        Returns:
            ExperimentResult.
        """
        if self.verbose:
            print(f"  Running zero-shot: {task[:50]}...")

        prompt = f"Goal: {task}\n\nWhat is the next action?"

        try:
            response = self._call_api(prompt, [screenshot_path])
            action = self._parse_action(response)
            return ExperimentResult(
                task=task,
                condition="zero_shot",
                response=response,
                action_parsed=action,
                success=None,  # Manual evaluation needed
            )
        except Exception as e:
            return ExperimentResult(
                task=task,
                condition="zero_shot",
                response="",
                action_parsed=None,
                success=False,
                error=str(e),
            )

    def run_with_demo(
        self,
        task: str,
        screenshot_path: str,
        demo_text: str,
        demo_screenshots: list[str] | None = None,
    ) -> ExperimentResult:
        """Run with-demo condition.

        Args:
            task: Task instruction.
            screenshot_path: Path to current screenshot.
            demo_text: Formatted demo text.
            demo_screenshots: Optional demo screenshot paths.

        Returns:
            ExperimentResult.
        """
        if self.verbose:
            print(f"  Running with-demo: {task[:50]}...")

        prompt = f"{demo_text}\n\nNOW PERFORM THIS TASK:\nGoal: {task}\n\nWhat is the next action?"

        # Combine demo screenshots with current screenshot
        all_images = (demo_screenshots or []) + [screenshot_path]

        try:
            response = self._call_api(prompt, all_images)
            action = self._parse_action(response)
            return ExperimentResult(
                task=task,
                condition="with_demo",
                response=response,
                action_parsed=action,
                success=None,
            )
        except Exception as e:
            return ExperimentResult(
                task=task,
                condition="with_demo",
                response="",
                action_parsed=None,
                success=False,
                error=str(e),
            )

    def run_control(
        self,
        task: str,
        screenshot_path: str,
        control_text: str,
    ) -> ExperimentResult:
        """Run length-matched control condition.

        Args:
            task: Task instruction.
            screenshot_path: Path to current screenshot.
            control_text: Length-matched control text.

        Returns:
            ExperimentResult.
        """
        if self.verbose:
            print(f"  Running control: {task[:50]}...")

        prompt = f"{control_text}\n\nGoal: {task}\n\nWhat is the next action?"

        try:
            response = self._call_api(prompt, [screenshot_path])
            action = self._parse_action(response)
            return ExperimentResult(
                task=task,
                condition="control",
                response=response,
                action_parsed=action,
                success=None,
            )
        except Exception as e:
            return ExperimentResult(
                task=task,
                condition="control",
                response="",
                action_parsed=None,
                success=False,
                error=str(e),
            )

    def run_task_pair(
        self,
        demo_episode: Any,  # Episode
        test_task: str,
        test_screenshot: str,
        include_demo_images: bool = False,
    ) -> dict[str, ExperimentResult]:
        """Run all conditions for a task pair.

        Args:
            demo_episode: Episode containing the demonstration.
            test_task: Test task instruction.
            test_screenshot: Path to test screenshot.
            include_demo_images: Whether to include demo screenshots.

        Returns:
            Dict mapping condition name to result.
        """
        # Format demo
        demo_text = format_episode_verbose(demo_episode, max_steps=10)

        # Get demo screenshots if requested
        demo_screenshots = None
        if include_demo_images:
            demo_screenshots = get_demo_screenshot_paths(demo_episode, max_steps=5)

        # Generate control
        control_text = generate_length_matched_control(demo_text)

        results = {}

        # Run all conditions
        results["zero_shot"] = self.run_zero_shot(test_task, test_screenshot)
        results["with_demo"] = self.run_with_demo(
            test_task, test_screenshot, demo_text, demo_screenshots
        )
        results["control"] = self.run_control(test_task, test_screenshot, control_text)

        return results


def run_experiment(
    demo_capture_path: str,
    test_task: str,
    test_screenshot: str,
    provider: str = "anthropic",
    output_dir: str | None = None,
    include_demo_images: bool = False,
    goal: str | None = None,
) -> dict[str, Any]:
    """Run the full experiment.

    Args:
        demo_capture_path: Path to demo capture directory.
        test_task: Test task instruction.
        test_screenshot: Path to test screenshot.
        provider: API provider.
        output_dir: Optional output directory for results.
        include_demo_images: Whether to include demo screenshots.
        goal: Optional goal for demo episode (overrides capture's).

    Returns:
        Dict with results.
    """
    from openadapt_ml.ingest.capture import capture_to_episode

    print(f"Loading demo from: {demo_capture_path}")
    episode = capture_to_episode(demo_capture_path, goal=goal)
    print(f"  Loaded {len(episode.steps)} steps, goal: {episode.goal}")

    print(f"\nTest task: {test_task}")
    print(f"Test screenshot: {test_screenshot}")

    experiment = DemoPromptExperiment(provider=provider)
    results = experiment.run_task_pair(
        demo_episode=episode,
        test_task=test_task,
        test_screenshot=test_screenshot,
        include_demo_images=include_demo_images,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for condition, result in results.items():
        print(f"\n{condition.upper()}:")
        print(f"  Action: {result.action_parsed}")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  Response preview: {result.response[:200]}...")

    # Save results if output dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "demo_capture": demo_capture_path,
                    "test_task": test_task,
                    "test_screenshot": test_screenshot,
                    "provider": provider,
                    "results": {
                        k: {
                            "task": v.task,
                            "condition": v.condition,
                            "action_parsed": v.action_parsed,
                            "response": v.response,
                            "error": v.error,
                            "timestamp": v.timestamp,
                        }
                        for k, v in results.items()
                    },
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {results_file}")

    return {"results": results, "episode": episode}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run demo-conditioned prompt experiment"
    )
    parser.add_argument(
        "--demo-capture",
        required=True,
        help="Path to demo capture directory",
    )
    parser.add_argument(
        "--test-task",
        required=True,
        help="Test task instruction",
    )
    parser.add_argument(
        "--test-screenshot",
        required=True,
        help="Path to test screenshot",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="API provider (default: anthropic)",
    )
    parser.add_argument(
        "--output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--include-demo-images",
        action="store_true",
        help="Include demo screenshots in prompt",
    )
    parser.add_argument(
        "--goal",
        help="Override goal for demo episode",
    )

    args = parser.parse_args()

    run_experiment(
        demo_capture_path=args.demo_capture,
        test_task=args.test_task,
        test_screenshot=args.test_screenshot,
        provider=args.provider,
        output_dir=args.output,
        include_demo_images=args.include_demo_images,
        goal=args.goal,
    )


if __name__ == "__main__":
    main()
