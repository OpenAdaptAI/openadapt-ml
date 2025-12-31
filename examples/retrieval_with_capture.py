#!/usr/bin/env python3
"""Example: Using retrieval with real capture data.

This demonstrates loading episodes from a capture directory and using
retrieval to find similar demonstrations.
"""

from __future__ import annotations

import sys
from pathlib import Path

from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo
from openadapt_ml.ingest.capture import capture_to_episode
from openadapt_ml.retrieval import DemoIndex, DemoRetriever


def main() -> None:
    """Run retrieval on real capture data."""
    # Check if capture path provided
    if len(sys.argv) < 2:
        print("Usage: python retrieval_with_capture.py <capture_path> [task]")
        print("\nExample:")
        print("  python retrieval_with_capture.py /path/to/capture 'Turn off Night Shift'")
        print("\nIf task is not provided, will use the episode's goal.")
        sys.exit(1)

    capture_path = sys.argv[1]
    if not Path(capture_path).exists():
        print(f"Error: Capture path does not exist: {capture_path}")
        sys.exit(1)

    # Load episode from capture
    print(f"Loading episode from: {capture_path}")
    try:
        episode = capture_to_episode(capture_path, include_moves=False)
    except Exception as e:
        print(f"Error loading capture: {e}")
        sys.exit(1)

    # For demo purposes, create a small library with the same episode
    # In real usage, you would have multiple different captures
    episodes = [episode]

    print(f"Loaded episode: {episode.goal}")
    print(f"  Steps: {len(episode.steps)}")

    # Build index
    print("\nBuilding retrieval index...")
    index = DemoIndex()
    index.add_many(episodes)
    index.build()

    print(f"Index: {index}")
    print(f"Apps: {', '.join(index.get_apps()) or 'None'}")
    print(f"Domains: {', '.join(index.get_domains()) or 'None'}")

    # Create retriever
    retriever = DemoRetriever(index, domain_bonus=0.3)

    # Get task from command line or use episode goal as example
    if len(sys.argv) >= 3:
        task = " ".join(sys.argv[2:])
    else:
        # Use episode goal with slight modification for demo
        task = episode.goal.replace("Turn off", "Disable").replace("Open", "Launch")
        print(f"\nNo task provided, using modified goal for demo: '{task}'")

    # Retrieve similar demos
    print("\n" + "=" * 80)
    print("RETRIEVAL")
    print("=" * 80)
    print(f"\nTask: {task}")

    # Extract app context from episode if available
    app_context = None
    if episode.steps:
        first_obs = episode.steps[0].observation
        if first_obs:
            app_context = first_obs.app_name or None

    if app_context:
        print(f"App context: {app_context}")

    # Retrieve with scores
    print(f"\nRetrieving top-3 demonstrations...")
    results = retriever.retrieve_with_scores(task, app_context, top_k=3)

    print(f"\nFound {len(results)} similar demo(s):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Text similarity: {result.text_score:.3f}")
        print(f"   Domain bonus: {result.domain_bonus:.3f}")
        print(f"   Goal: {result.demo.episode.goal}")
        print(f"   Steps: {len(result.demo.episode.steps)}")
        if result.demo.app_name:
            print(f"   App: {result.demo.app_name}")
        if result.demo.domain:
            print(f"   Domain: {result.demo.domain}")

    # Format best demo
    if results:
        print("\n" + "=" * 80)
        print("FORMATTED DEMONSTRATION (Best Match)")
        print("=" * 80)
        best_demo = results[0].demo.episode
        formatted = format_episode_as_demo(best_demo, max_steps=10)
        print(formatted)

        # Show example prompt
        print("\n" + "=" * 80)
        print("EXAMPLE FEW-SHOT PROMPT")
        print("=" * 80)

        prompt = f"""You are a GUI automation agent. Below is a demonstration of a similar task.

{formatted}

Now, perform this new task following a similar approach:
Task: {task}
{f'App: {app_context}' if app_context else ''}

Provide your first action."""

        print(prompt)

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
