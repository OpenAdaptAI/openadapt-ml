"""CLI for baseline adapter operations.

Provides commands for comparing VLMs across tracks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from openadapt_ml.baselines.config import MODELS


@click.group()
def baselines():
    """Baseline adapter commands for VLM comparison."""
    pass


@baselines.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(list(MODELS.keys())),
    help="Model alias to use",
)
@click.option(
    "--track",
    "-t",
    type=click.Choice(["A", "B", "C"]),
    default="A",
    help="Evaluation track (A=coords, B=ReAct, C=SoM)",
)
@click.option(
    "--image",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Screenshot image path",
)
@click.option(
    "--goal",
    "-g",
    required=True,
    help="Task goal/instruction",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file path",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def run(
    model: str,
    track: str,
    image: str,
    goal: str,
    output: str | None,
    verbose: bool,
):
    """Run a single baseline prediction.

    Example:
        uv run python -m openadapt_ml.baselines.cli run \\
            --model claude-opus-4.5 \\
            --track A \\
            --image screenshot.png \\
            --goal "Click the submit button"
    """
    from PIL import Image

    from openadapt_ml.baselines import UnifiedBaselineAdapter, TrackConfig

    # Select track config
    track_configs = {
        "A": TrackConfig.track_a(),
        "B": TrackConfig.track_b(),
        "C": TrackConfig.track_c(),
    }
    track_config = track_configs[track]

    click.echo(f"Model: {model}")
    click.echo(f"Track: {track} ({track_config.track_type.value})")
    click.echo(f"Image: {image}")
    click.echo(f"Goal: {goal}")
    click.echo()

    # Load image
    screenshot = Image.open(image)

    # Create adapter
    adapter = UnifiedBaselineAdapter.from_alias(
        model,
        track=track_config,
        verbose=verbose,
    )

    # Run prediction
    click.echo("Running prediction...")
    action = adapter.predict(screenshot, goal)

    # Display result
    click.echo()
    click.echo("=" * 50)
    click.echo("RESULT")
    click.echo("=" * 50)

    if action.is_valid:
        click.echo(f"Action: {action.action_type.upper()}")
        if action.x is not None and action.y is not None:
            click.echo(f"Coordinates: ({action.x:.4f}, {action.y:.4f})")
        if action.element_id is not None:
            click.echo(f"Element ID: {action.element_id}")
        if action.text is not None:
            click.echo(f"Text: {action.text}")
        if action.thought is not None:
            click.echo(f"Thought: {action.thought}")
    else:
        click.echo(f"Parse Error: {action.parse_error}")
        click.echo(
            f"Raw Response: {action.raw_response[:200] if action.raw_response else 'None'}..."
        )

    # Save output if requested
    if output:
        result = {
            "model": model,
            "track": track,
            "goal": goal,
            "action": action.to_dict(),
            "raw_response": action.raw_response,
            "parse_error": action.parse_error,
        }
        Path(output).write_text(json.dumps(result, indent=2))
        click.echo(f"\nSaved to: {output}")


@baselines.command()
@click.option(
    "--models",
    "-m",
    required=True,
    help="Comma-separated model aliases",
)
@click.option(
    "--track",
    "-t",
    type=click.Choice(["A", "B", "C"]),
    default="A",
    help="Evaluation track",
)
@click.option(
    "--image",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Screenshot image path",
)
@click.option(
    "--goal",
    "-g",
    required=True,
    help="Task goal/instruction",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file path",
)
def compare(
    models: str,
    track: str,
    image: str,
    goal: str,
    output: str | None,
):
    """Compare multiple models on the same task.

    Example:
        uv run python -m openadapt_ml.baselines.cli compare \\
            --models claude-opus-4.5,gpt-5.2,gemini-3-pro \\
            --track C \\
            --image screenshot.png \\
            --goal "Click the login button"
    """
    from PIL import Image

    from openadapt_ml.baselines import UnifiedBaselineAdapter, TrackConfig

    model_list = [m.strip() for m in models.split(",")]

    # Validate models
    for m in model_list:
        if m not in MODELS:
            click.echo(f"Error: Unknown model '{m}'", err=True)
            click.echo(f"Available: {', '.join(MODELS.keys())}", err=True)
            sys.exit(1)

    # Select track config
    track_configs = {
        "A": TrackConfig.track_a(),
        "B": TrackConfig.track_b(),
        "C": TrackConfig.track_c(),
    }
    track_config = track_configs[track]

    click.echo(f"Comparing {len(model_list)} models on Track {track}")
    click.echo(f"Image: {image}")
    click.echo(f"Goal: {goal}")
    click.echo()

    # Load image
    screenshot = Image.open(image)

    results = []

    # Run each model
    for model in model_list:
        click.echo(f"Running {model}...")

        try:
            adapter = UnifiedBaselineAdapter.from_alias(model, track=track_config)
            action = adapter.predict(screenshot, goal)

            results.append(
                {
                    "model": model,
                    "success": action.is_valid,
                    "action": action.to_dict(),
                    "error": action.parse_error,
                }
            )

            status = "OK" if action.is_valid else "FAILED"
            click.echo(f"  {status}: {action.action_type}")

        except Exception as e:
            results.append(
                {
                    "model": model,
                    "success": False,
                    "action": None,
                    "error": str(e),
                }
            )
            click.echo(f"  ERROR: {e}")

    # Summary table
    click.echo()
    click.echo("=" * 60)
    click.echo("COMPARISON SUMMARY")
    click.echo("=" * 60)
    click.echo(f"{'Model':<25} {'Status':<10} {'Action':<25}")
    click.echo("-" * 60)

    for r in results:
        model = r["model"]
        status = "OK" if r["success"] else "FAILED"
        action = r["action"]
        if action:
            if action.get("x") is not None:
                action_str = f"CLICK({action['x']:.3f}, {action['y']:.3f})"
            elif action.get("element_id") is not None:
                action_str = f"CLICK([{action['element_id']}])"
            else:
                action_str = action.get("type", "unknown").upper()
        else:
            action_str = r.get("error", "Unknown error")[:25]

        click.echo(f"{model:<25} {status:<10} {action_str:<25}")

    # Save output if requested
    if output:
        full_results = {
            "models": model_list,
            "track": track,
            "goal": goal,
            "results": results,
        }
        Path(output).write_text(json.dumps(full_results, indent=2))
        click.echo(f"\nSaved to: {output}")


@baselines.command()
def list_models():
    """List available models and their providers."""
    click.echo("Available models:")
    click.echo()
    click.echo(f"{'Alias':<20} {'Provider':<12} {'Model ID':<35} {'Default'}")
    click.echo("-" * 75)

    for alias, spec in MODELS.items():
        default = "*" if spec.is_default else ""
        click.echo(f"{alias:<20} {spec.provider:<12} {spec.model_id:<35} {default}")


# Entry point for direct execution
def main():
    baselines()


if __name__ == "__main__":
    main()
