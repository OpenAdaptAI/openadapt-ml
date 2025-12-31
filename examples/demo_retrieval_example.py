#!/usr/bin/env python3
"""Example of using demo retrieval with prompt formatting.

This demonstrates the full pipeline:
1. Create/load demonstrations
2. Build retrieval index
3. Retrieve relevant demos for a new task
4. Format demos for few-shot prompting
"""

from __future__ import annotations

from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo
from openadapt_ml.retrieval import DemoIndex, DemoRetriever
from openadapt_ml.schemas.sessions import Action, Episode, Observation, Step


def create_demo_episode(
    episode_id: str,
    goal: str,
    steps_data: list[tuple[str, str, tuple[float, float]]],
    app_name: str | None = None,
    url: str | None = None,
) -> Episode:
    """Create a demo episode with multiple steps.

    Args:
        episode_id: Episode ID.
        goal: Task goal.
        steps_data: List of (window_title, action_type, (x, y)) tuples.
        app_name: Optional app name.
        url: Optional URL.

    Returns:
        Episode object.
    """
    steps = []
    for i, (window_title, action_type, coords) in enumerate(steps_data):
        obs = Observation(
            app_name=app_name,
            window_title=window_title,
            url=url,
        )
        action = Action(type=action_type, x=coords[0], y=coords[1])
        step = Step(t=float(i), observation=obs, action=action)
        steps.append(step)

    return Episode(
        id=episode_id,
        goal=goal,
        steps=steps,
    )


def main() -> None:
    """Run the demo retrieval example."""
    print("Demo Retrieval + Prompt Formatting Example")
    print("=" * 80)

    # Create a library of demo episodes
    demos = [
        create_demo_episode(
            "demo_nightshift",
            "Turn off Night Shift in System Settings",
            [
                ("Finder", "click", (0.5, 0.1)),  # Click Apple menu
                ("System Settings", "click", (0.3, 0.4)),  # Click System Settings
                ("System Settings - Displays", "click", (0.2, 0.5)),  # Click Displays
                ("System Settings - Displays", "click", (0.7, 0.6)),  # Toggle Night Shift
            ],
            app_name="System Settings",
        ),
        create_demo_episode(
            "demo_github_search",
            "Search for machine learning repositories on GitHub",
            [
                ("GitHub", "click", (0.3, 0.1)),  # Click search box
                ("GitHub", "type", (0.3, 0.1)),  # Type query
                ("GitHub - Search", "click", (0.5, 0.3)),  # Click result
            ],
            app_name="Chrome",
            url="https://github.com/search?q=machine+learning",
        ),
        create_demo_episode(
            "demo_calculator",
            "Calculate 25 * 16 using Calculator",
            [
                ("Calculator", "click", (0.3, 0.5)),  # Click 2
                ("Calculator", "click", (0.6, 0.5)),  # Click 5
                ("Calculator", "click", (0.8, 0.3)),  # Click *
                ("Calculator", "click", (0.3, 0.6)),  # Click 1
                ("Calculator", "click", (0.6, 0.5)),  # Click 6
                ("Calculator", "click", (0.8, 0.7)),  # Click =
            ],
            app_name="Calculator",
        ),
    ]

    print(f"\nCreated {len(demos)} demonstration episodes")
    for demo in demos:
        print(f"  - {demo.goal} ({len(demo.steps)} steps)")

    # Build the retrieval index
    print("\nBuilding retrieval index...")
    index = DemoIndex()
    index.add_many(demos)
    index.build()
    print(f"Index: {index}")
    print(f"Apps in index: {', '.join(index.get_apps())}")
    print(f"Domains in index: {', '.join(index.get_domains())}")

    # Create retriever
    retriever = DemoRetriever(index, domain_bonus=0.3)

    # Simulate a new task
    print("\n" + "=" * 80)
    print("NEW TASK")
    print("=" * 80)
    new_task = "Disable dark mode in macOS settings"
    app_context = "System Settings"

    print(f"\nTask: {new_task}")
    print(f"App context: {app_context}")

    # Retrieve relevant demos
    print(f"\nRetrieving top-3 similar demonstrations...")
    results = retriever.retrieve_with_scores(new_task, app_context, top_k=3)

    print(f"\nFound {len(results)} similar demos:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.demo.episode.goal}")
        print(f"   Score: {result.score:.3f} (text: {result.text_score:.3f}, domain bonus: {result.domain_bonus:.3f})")

    # Format the best demo for prompting
    if results:
        best_demo = results[0].demo.episode
        print("\n" + "=" * 80)
        print("FORMATTED DEMO FOR PROMPT")
        print("=" * 80)
        formatted_demo = format_episode_as_demo(best_demo, max_steps=10)
        print(formatted_demo)

        # Show how this would be used in a prompt
        print("\n" + "=" * 80)
        print("FULL PROMPT EXAMPLE")
        print("=" * 80)

        full_prompt = f"""You are a GUI automation agent. I will show you a demonstration of a similar task, then ask you to perform a new task.

{formatted_demo}

Now, please perform the following task:
Task: {new_task}
App: {app_context}

What is your first action?"""

        print(full_prompt)

    print("\n" + "=" * 80)
    print("Example completed!")
    print("\nNext steps:")
    print("- Load real episodes from captures using openadapt_ml.ingest.capture")
    print("- Integrate with VLM prompting pipeline")
    print("- Experiment with different retrieval parameters")
    print("- Add more sophisticated embedding models (sentence-transformers)")


if __name__ == "__main__":
    main()
