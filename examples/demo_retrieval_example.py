#!/usr/bin/env python3
"""Example of using demo retrieval with prompt formatting.

This demonstrates the full pipeline:
1. Create/load demonstrations
2. Build retrieval index
3. Retrieve relevant demos for a new task
4. Format demos for few-shot prompting

Two APIs are shown:
- New API (DemoRetriever): Simpler, supports multiple embedding methods
- Legacy API (DemoIndex + LegacyDemoRetriever): For backward compatibility
"""

from __future__ import annotations

from openadapt_ml.retrieval import DemoRetriever
from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step


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
        # Create action with appropriate parameters based on type
        action_kwargs = {
            "type": ActionType(action_type),
            "normalized_coordinates": (coords[0], coords[1]),
        }
        if action_type == "type":
            action_kwargs["text"] = "example text"
        action = Action(**action_kwargs)
        step = Step(step_index=i, observation=obs, action=action)
        steps.append(step)

    return Episode(
        episode_id=episode_id,
        instruction=goal,
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
        create_demo_episode(
            "demo_dark_mode",
            "Enable dark mode in system appearance settings",
            [
                ("Finder", "click", (0.5, 0.1)),  # Click Apple menu
                ("System Settings", "click", (0.3, 0.4)),  # Click System Settings
                ("System Settings - Appearance", "click", (0.2, 0.3)),  # Click Appearance
                ("System Settings - Appearance", "click", (0.4, 0.5)),  # Click Dark mode
            ],
            app_name="System Settings",
        ),
    ]

    print(f"\nCreated {len(demos)} demonstration episodes")
    for demo in demos:
        print(f"  - {demo.instruction} ({len(demo.steps)} steps)")

    # =========================================================================
    # New API: DemoRetriever (Recommended)
    # =========================================================================
    print("\n" + "=" * 80)
    print("USING NEW API: DemoRetriever")
    print("=" * 80)

    # Create retriever with TF-IDF embeddings (default, no dependencies)
    retriever = DemoRetriever(
        embedding_method="tfidf",  # Options: "tfidf", "sentence_transformers", "openai"
        domain_bonus=0.2,
        app_bonus=0.15,
    )

    # Add demos
    for demo in demos:
        retriever.add_demo(demo)

    # Build the index
    retriever.build_index()
    print(f"\nRetriever: {retriever}")
    print(f"Apps in index: {', '.join(retriever.get_apps())}")
    print(f"Domains in index: {', '.join(retriever.get_domains())}")

    # Simulate a new task
    print("\n" + "-" * 40)
    print("NEW TASK")
    print("-" * 40)
    new_task = "Disable dark mode in macOS settings"
    app_context = "System Settings"

    print(f"\nTask: {new_task}")
    print(f"App context: {app_context}")

    # Retrieve relevant demos
    print(f"\nRetrieving top-3 similar demonstrations...")
    results = retriever.retrieve(new_task, top_k=3, app_context=app_context)

    print(f"\nFound {len(results)} similar demos:")
    for result in results:
        print(f"\n{result.rank}. {result.demo.goal}")
        print(f"   Score: {result.score:.3f} (text: {result.text_score:.3f}, bonus: {result.domain_bonus:.3f})")
        print(f"   App: {result.demo.app_name}, Platform: {result.demo.platform}")

    # Format for prompt using built-in method
    if results:
        print("\n" + "-" * 40)
        print("FORMATTED DEMO FOR PROMPT")
        print("-" * 40)
        formatted_demo = retriever.format_for_prompt(
            results[:1],  # Just the top result
            max_steps_per_demo=10,
            include_scores=True,
        )
        print(formatted_demo)

    # Show how this would be used in a prompt
    print("\n" + "-" * 40)
    print("FULL PROMPT EXAMPLE")
    print("-" * 40)

    full_prompt = f"""You are a GUI automation agent. I will show you a demonstration of a similar task, then ask you to perform a new task.

{formatted_demo}

Now, please perform the following task:
Task: {new_task}
App: {app_context}

What is your first action?"""

    print(full_prompt)

    # =========================================================================
    # Alternative: Using sentence-transformers for better semantic matching
    # =========================================================================
    print("\n" + "=" * 80)
    print("USING SENTENCE-TRANSFORMERS (if installed)")
    print("=" * 80)

    try:
        retriever_st = DemoRetriever(
            embedding_method="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",  # Fast, 22MB
        )
        for demo in demos:
            retriever_st.add_demo(demo)
        retriever_st.build_index()

        # Now semantic similarity works!
        # "Disable dark mode" should match "Enable dark mode" despite different wording
        results_st = retriever_st.retrieve("Disable blue light filter", top_k=2)
        print("\nQuery: 'Disable blue light filter' (semantic)")
        for result in results_st:
            print(f"  {result.rank}. {result.demo.goal} (score: {result.score:.3f})")

    except ImportError:
        print("\nsentence-transformers not installed. Install with:")
        print("  pip install sentence-transformers")

    print("\n" + "=" * 80)
    print("Example completed!")
    print("\nNext steps:")
    print("- Load real episodes from captures using openadapt_ml.ingest.capture")
    print("- Use sentence-transformers for semantic matching")
    print("- Save/load index for persistence")
    print("- Filter by platform, tags, or app")


if __name__ == "__main__":
    main()
