"""CLI commands for workflow segmentation.

This module provides command-line interface for the segmentation pipeline.
"""

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group()
def segment():
    """Workflow segmentation commands."""
    pass


@segment.command("describe")
@click.option(
    "--recording", "-r", required=True, multiple=True, help="Recording to describe"
)
@click.option("--model", "-m", default="gemini-2.0-flash", help="VLM model")
@click.option("--batch-size", "-b", default=10, help="Frames per API call")
@click.option("--output", "-o", help="Output file for transcript")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option("--no-cache", is_flag=True, help="Disable caching")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
def describe(recording, model, batch_size, output, format, no_cache, verbose):
    """Generate frame descriptions for a recording (Stage 1)."""
    from openadapt_ml.segmentation.frame_describer import FrameDescriber

    if verbose:
        logging.basicConfig(level=logging.INFO)

    describer = FrameDescriber(
        model=model,
        batch_size=batch_size,
        cache_enabled=not no_cache,
    )

    for rec_path in recording:
        click.echo(f"Processing: {rec_path}")
        transcript = describer.describe_recording(rec_path)

        if output:
            output_path = Path(output)
            if len(recording) > 1:
                output_path = (
                    output_path.parent / f"{Path(rec_path).stem}_{output_path.name}"
                )

            if format == "json":
                output_path.write_text(transcript.model_dump_json(indent=2))
            else:
                output_path.write_text(transcript.to_transcript_text())
            click.echo(f"  Saved to: {output_path}")
        else:
            if format == "json":
                click.echo(transcript.model_dump_json(indent=2))
            else:
                click.echo(transcript.to_transcript_text())


@segment.command("extract")
@click.option("--recording", "-r", help="Recording to segment")
@click.option("--transcript", "-t", help="Existing transcript file")
@click.option("--model", "-m", default="gpt-4o", help="LLM model")
@click.option("--hierarchical", "-h", is_flag=True, help="Extract nested segments")
@click.option("--no-few-shot", is_flag=True, help="Disable few-shot examples")
@click.option("--min-duration", default=2.0, help="Minimum segment length (seconds)")
@click.option("--max-duration", default=300.0, help="Maximum segment length (seconds)")
@click.option("--output", "-o", help="Output file for segments")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
def extract(
    recording,
    transcript,
    model,
    hierarchical,
    no_few_shot,
    min_duration,
    max_duration,
    output,
    verbose,
):
    """Extract workflow segments from a recording (Stage 2)."""
    from openadapt_ml.segmentation.frame_describer import FrameDescriber
    from openadapt_ml.segmentation.segment_extractor import SegmentExtractor
    from openadapt_ml.segmentation.schemas import ActionTranscript

    if verbose:
        logging.basicConfig(level=logging.INFO)

    if not recording and not transcript:
        raise click.UsageError("Specify either --recording or --transcript")

    # Load or generate transcript
    if transcript:
        data = json.loads(Path(transcript).read_text())
        action_transcript = ActionTranscript.model_validate(data)
    else:
        describer = FrameDescriber()
        action_transcript = describer.describe_recording(recording)

    # Extract segments
    extractor = SegmentExtractor(
        model=model,
        use_few_shot=not no_few_shot,
        hierarchical=hierarchical,
        min_segment_duration=min_duration,
        max_segment_duration=max_duration,
    )

    result = extractor.extract_segments(action_transcript)

    # Output
    if output:
        Path(output).write_text(result.model_dump_json(indent=2))
        click.echo(f"Saved to: {output}")
    else:
        click.echo(f"\nFound {len(result.episodes)} episodes:")
        for ep in result.episodes:
            click.echo(
                f"  - {ep.name} ({ep.start_time_formatted} - {ep.end_time_formatted})"
            )
            click.echo(f"    {ep.description[:80]}...")


@segment.command("deduplicate")
@click.argument("segments", nargs=-1)
@click.option("--input-dir", "-i", help="Directory with segment files")
@click.option("--threshold", "-t", default=0.85, help="Similarity threshold (0-1)")
@click.option(
    "--embedding-model", default="text-embedding-3-large", help="Embedding model"
)
@click.option(
    "--merge-strategy",
    type=click.Choice(["centroid", "longest", "first"]),
    default="centroid",
    help="Merge strategy",
)
@click.option("--existing", "-e", help="Existing library to merge with")
@click.option("--output", "-o", required=True, help="Output library file")
@click.option(
    "--local-embeddings", is_flag=True, help="Use local HuggingFace embeddings"
)
@click.option("--verbose", "-v", is_flag=True, help="Show clustering details")
def deduplicate(
    segments,
    input_dir,
    threshold,
    embedding_model,
    merge_strategy,
    existing,
    output,
    local_embeddings,
    verbose,
):
    """Deduplicate segments across recordings (Stage 3)."""
    from openadapt_ml.segmentation.deduplicator import WorkflowDeduplicator
    from openadapt_ml.segmentation.schemas import (
        EpisodeExtractionResult,
        EpisodeLibrary,
    )

    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Collect segment files
    segment_files = list(segments)
    if input_dir:
        segment_files.extend(Path(input_dir).glob("*_episodes.json"))

    if not segment_files:
        raise click.UsageError("No segment files specified")

    # Load extraction results
    extraction_results = []
    for seg_file in segment_files:
        data = json.loads(Path(seg_file).read_text())
        result = EpisodeExtractionResult.model_validate(data)
        extraction_results.append(result)
        click.echo(f"Loaded: {seg_file} ({len(result.episodes)} episodes)")

    # Load existing library
    existing_library = None
    if existing:
        data = json.loads(Path(existing).read_text())
        existing_library = EpisodeLibrary.model_validate(data)
        click.echo(
            f"Merging with existing library ({existing_library.unique_episode_count} workflows)"
        )

    # Deduplicate
    dedup = WorkflowDeduplicator(
        threshold=threshold,
        embedding_model=embedding_model,
        merge_strategy=merge_strategy,
        use_local_embeddings=local_embeddings,
    )

    library = dedup.deduplicate(extraction_results, existing_library)

    # Save
    Path(output).write_text(library.model_dump_json(indent=2))

    click.echo("\nResults:")
    click.echo(f"  Total episodes: {library.total_episodes_extracted}")
    click.echo(f"  Unique workflows: {library.unique_episode_count}")
    click.echo(f"  Deduplication ratio: {library.deduplication_ratio:.1%}")
    click.echo(f"\nSaved to: {output}")


@segment.command("pipeline")
@click.argument("recordings", nargs=-1)
@click.option("--vlm-model", default="gemini-2.0-flash", help="VLM for Stage 1")
@click.option("--llm-model", default="gpt-4o", help="LLM for Stage 2")
@click.option("--threshold", default=0.85, help="Dedup threshold for Stage 3")
@click.option("--output", "-o", required=True, help="Output directory or library file")
@click.option("--save-intermediate", is_flag=True, help="Save Stage 1/2 outputs")
@click.option("--resume", help="Resume from checkpoint directory")
@click.option("--existing", "-e", help="Existing library to merge with")
@click.option("--local-embeddings", is_flag=True, help="Use local embeddings")
@click.option("--verbose", "-v", is_flag=True, help="Detailed progress")
def pipeline(
    recordings,
    vlm_model,
    llm_model,
    threshold,
    output,
    save_intermediate,
    resume,
    existing,
    local_embeddings,
    verbose,
):
    """Run complete segmentation pipeline."""
    from openadapt_ml.segmentation.pipeline import SegmentationPipeline, PipelineConfig
    from openadapt_ml.segmentation.schemas import EpisodeLibrary

    if verbose:
        logging.basicConfig(level=logging.INFO)

    config = PipelineConfig(
        vlm_model=vlm_model,
        llm_model=llm_model,
        similarity_threshold=threshold,
        use_local_embeddings=local_embeddings,
        verbose=verbose,
    )

    pipeline = SegmentationPipeline(config)

    # Determine output directory
    output_path = Path(output)
    if output_path.suffix == ".json":
        output_dir = output_path.parent if save_intermediate else None
        library_path = output_path
    else:
        output_dir = output_path
        library_path = output_path / "episode_library.json"

    # Load existing library
    existing_library = None
    if existing:
        data = json.loads(Path(existing).read_text())
        existing_library = EpisodeLibrary.model_validate(data)

    # Run or resume
    if resume:
        result = pipeline.resume(resume, list(recordings) if recordings else None)
    else:
        if not recordings:
            raise click.UsageError("Specify recordings to process")
        result = pipeline.run(
            list(recordings),
            output_dir=output_dir,
            existing_library=existing_library,
            progress_callback=lambda stage, cur, tot: click.echo(
                f"  [{stage}] {cur}/{tot}"
            )
            if verbose
            else None,
        )

    # Save final library if not already saved
    if not save_intermediate and result.library:
        library_path.parent.mkdir(parents=True, exist_ok=True)
        library_path.write_text(result.library.model_dump_json(indent=2))

    click.echo("\nPipeline complete:")
    click.echo(f"  Recordings processed: {result.recordings_processed}")
    click.echo(f"  Total episodes: {result.total_episodes_extracted}")
    click.echo(f"  Unique workflows: {result.unique_episodes}")
    click.echo(f"  Processing time: {result.processing_time_seconds:.1f}s")
    click.echo(f"\nLibrary saved to: {library_path}")


@segment.command("list")
@click.option("--library", "-l", required=True, help="Library file to inspect")
@click.option("--details", "-d", is_flag=True, help="Show segment details")
@click.option("--app", "-a", help="Filter by application")
def list_segments(library, details, app):
    """List existing segments and libraries."""
    from openadapt_ml.segmentation.schemas import EpisodeLibrary

    data = json.loads(Path(library).read_text())
    lib = EpisodeLibrary.model_validate(data)

    click.echo(f"Episode Library: {library}")
    click.echo(f"  Created: {lib.created_at}")
    click.echo(f"  Recordings: {lib.total_recordings_processed}")
    click.echo(f"  Total episodes: {lib.total_episodes_extracted}")
    click.echo(f"  Unique workflows: {lib.unique_episode_count}")
    click.echo(f"  Dedup ratio: {lib.deduplication_ratio:.1%}")

    click.echo("\nWorkflows:")
    for ep in lib.episodes:
        # Filter by app if specified
        # Note: CanonicalEpisode doesn't have application field directly
        # Would need to track this from source episodes

        click.echo(f"\n  {ep.canonical_name}")
        click.echo(f"    Occurrences: {ep.occurrence_count}")
        click.echo(
            f"    Recordings: {', '.join(ep.source_recordings[:3])}{'...' if len(ep.source_recordings) > 3 else ''}"
        )

        if details:
            click.echo(f"    Description: {ep.canonical_description[:100]}...")
            click.echo(
                f"    Steps: {', '.join(ep.canonical_steps[:3])}{'...' if len(ep.canonical_steps) > 3 else ''}"
            )


@segment.command("annotate")
@click.option("--episodes", "-e", required=True, help="Episodes JSON file from extract")
@click.option("--recording", "-r", required=True, help="Recording directory path")
@click.option(
    "--model", "-m", default="gemini-2.0-flash", help="VLM model for annotation"
)
@click.option("--lookahead", default=10, help="Frames to analyze after episode end")
@click.option("--output", "-o", required=True, help="Output annotated library file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
def annotate(episodes, recording, model, lookahead, output, verbose):
    """Annotate extracted episodes with VLM analysis.

    This command analyzes episodes to determine if they are suitable
    for training (gold) by examining the episode frames and frames
    after the episode ends to detect failures.
    """
    from openadapt_ml.segmentation.annotator import EpisodeAnnotator
    from openadapt_ml.segmentation.schemas import EpisodeExtractionResult

    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Load episodes
    data = json.loads(Path(episodes).read_text())
    extraction_result = EpisodeExtractionResult.model_validate(data)

    click.echo(f"Loaded {len(extraction_result.episodes)} episodes from {episodes}")
    click.echo(f"Using VLM: {model}")

    # Create annotator
    annotator = EpisodeAnnotator(
        model=model,
        lookahead_frames=lookahead,
    )

    # Annotate
    def progress(current, total):
        if verbose:
            click.echo(f"  Progress: {current}/{total}")

    library = annotator.annotate_extraction_result(
        extraction_result=extraction_result,
        recording_path=recording,
        progress_callback=progress,
    )

    # Save
    Path(output).write_text(library.model_dump_json(indent=2))

    click.echo("\nAnnotation complete:")
    click.echo(f"  Total episodes: {library.total_episodes}")
    click.echo(f"  Recommended as gold: {library.gold_count}")
    click.echo(
        f"  Pending human review: {library.total_episodes - library.verified_count}"
    )
    click.echo(f"\nSaved to: {output}")
    click.echo("\nNext step: Run 'segment review' to verify annotations")


@segment.command("review")
@click.option("--library", "-l", required=True, help="Annotated library file")
@click.option("--recording", "-r", help="Recording directory (for viewing frames)")
@click.option("--reviewer", default="human", help="Reviewer name/ID")
@click.option(
    "--auto-approve-high-confidence", is_flag=True, help="Auto-approve confidence > 0.9"
)
@click.option("--output", "-o", help="Output file (defaults to overwriting input)")
def review(library, recording, reviewer, auto_approve_high_confidence, output):
    """Interactive review of annotated episodes.

    This command presents each annotation for human verification.
    Reviewers can approve, reject, or edit each annotation.
    """
    from openadapt_ml.segmentation.schemas import AnnotatedEpisodeLibrary
    from openadapt_ml.segmentation.annotator import verify_annotation

    # Load library
    data = json.loads(Path(library).read_text())
    lib = AnnotatedEpisodeLibrary.model_validate(data)

    click.echo(f"Loaded annotated library: {library}")
    click.echo(f"  Total episodes: {lib.total_episodes}")
    click.echo(f"  Already verified: {lib.verified_count}")
    click.echo(f"  Pending review: {lib.total_episodes - lib.verified_count}")

    # Auto-approve high confidence if requested
    if auto_approve_high_confidence:
        auto_approved = 0
        new_annotations = []
        for ann in lib.annotations:
            if not ann.human_verified and ann.confidence > 0.9 and ann.is_gold:
                new_ann = verify_annotation(
                    ann,
                    is_gold=True,
                    notes="Auto-approved (confidence > 0.9)",
                    verified_by=f"{reviewer}_auto",
                )
                new_annotations.append(new_ann)
                auto_approved += 1
            else:
                new_annotations.append(ann)
        lib.annotations = new_annotations
        click.echo(f"\nAuto-approved {auto_approved} high-confidence gold episodes")

    # Get pending reviews
    pending = lib.get_pending_review()

    if not pending:
        click.echo("\nNo episodes pending review!")
        if output:
            Path(output).write_text(lib.model_dump_json(indent=2))
            click.echo(f"Saved to: {output}")
        return

    click.echo(f"\n{len(pending)} episodes to review:")
    click.echo("Commands: [a]pprove, [r]eject, [s]kip, [n]otes, [q]uit\n")

    # Interactive review
    reviewed = 0
    new_annotations = []
    annotation_map = {a.annotation_id: a for a in lib.annotations}

    for episode, annotation in pending:
        click.echo("-" * 60)
        click.echo(f"Episode: {episode.name}")
        click.echo(f"Description: {episode.description}")
        click.echo(
            f"Time: {episode.start_time_formatted} - {episode.end_time_formatted}"
        )
        click.echo(f"Application: {episode.application}")
        click.echo(f"Steps: {', '.join(episode.step_summaries[:5])}")
        click.echo()
        click.echo("VLM Assessment:")
        click.echo(f"  Is Gold: {annotation.is_gold}")
        click.echo(f"  Confidence: {annotation.confidence:.2f}")
        if annotation.failure_signals:
            click.echo(f"  Failure Signals: {', '.join(annotation.failure_signals)}")
        if annotation.exclusion_reason:
            click.echo(f"  Exclusion Reason: {annotation.exclusion_reason}")
        click.echo()

        while True:
            choice = click.prompt(
                "Action [a/r/s/n/q]",
                type=click.Choice(["a", "r", "s", "n", "q"]),
                default="s",
            )

            if choice == "a":
                notes = click.prompt("Notes (optional)", default="", show_default=False)
                new_ann = verify_annotation(
                    annotation,
                    is_gold=True,
                    notes=notes if notes else None,
                    verified_by=reviewer,
                )
                annotation_map[annotation.annotation_id] = new_ann
                click.echo("  Approved as gold")
                reviewed += 1
                break

            elif choice == "r":
                reason = click.prompt("Rejection reason", default="Manual rejection")
                new_ann = verify_annotation(
                    annotation,
                    is_gold=False,
                    notes=reason,
                    verified_by=reviewer,
                )
                annotation_map[annotation.annotation_id] = new_ann
                click.echo("  Rejected")
                reviewed += 1
                break

            elif choice == "s":
                click.echo("  Skipped")
                break

            elif choice == "n":
                notes = click.prompt("Add notes")
                annotation.notes = notes
                annotation_map[annotation.annotation_id] = annotation
                click.echo(f"  Notes added: {notes}")
                # Continue to ask for a/r/s

            elif choice == "q":
                click.echo("\nQuitting review...")
                break

        if choice == "q":
            break

    # Update library with new annotations
    lib.annotations = list(annotation_map.values())

    # Save
    output_path = Path(output) if output else Path(library)
    output_path.write_text(lib.model_dump_json(indent=2))

    click.echo("\nReview session complete:")
    click.echo(f"  Reviewed: {reviewed}")
    click.echo(f"  Total verified: {lib.verified_count}")
    click.echo(f"  Gold episodes: {lib.gold_count}")
    click.echo(f"  Export-ready: {lib.export_ready_count}")
    click.echo(f"\nSaved to: {output_path}")


@segment.command("export-gold")
@click.argument("library")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "jsonl", "hf"]),
    default="jsonl",
    help="Export format",
)
@click.option("--output", "-o", required=True, help="Output file/directory")
@click.option("--recording", "-r", help="Recording directory (for screenshots)")
@click.option(
    "--include-screenshots", is_flag=True, help="Include screenshots in export"
)
def export_gold(library, format, output, recording, include_screenshots):
    """Export verified gold episodes for fine-tuning.

    Only exports episodes where is_gold=True AND human_verified=True.
    """
    from openadapt_ml.segmentation.schemas import AnnotatedEpisodeLibrary
    from openadapt_ml.segmentation.annotator import export_gold_episodes

    # Load library
    data = json.loads(Path(library).read_text())
    lib = AnnotatedEpisodeLibrary.model_validate(data)

    click.echo(f"Loaded library: {library}")
    click.echo(f"  Export-ready episodes: {lib.export_ready_count}")

    if lib.export_ready_count == 0:
        click.echo("\nNo episodes ready for export!")
        click.echo("Run 'segment review' first to verify annotations.")
        return

    # Export
    count = export_gold_episodes(
        library=lib,
        output_path=output,
        recording_path=recording,
        format=format,
        include_screenshots=include_screenshots,
    )

    click.echo(f"\nExported {count} gold episodes to: {output}")


@segment.command("export")
@click.argument("library")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "jsonl", "html"]),
    default="jsonl",
    help="Export format",
)
@click.option("--output", "-o", required=True, help="Output file")
@click.option("--workflow", "-w", help="Export specific workflow")
def export(library, format, output, workflow):
    """Export segments to various formats."""
    import csv
    from openadapt_ml.segmentation.schemas import EpisodeLibrary

    data = json.loads(Path(library).read_text())
    lib = EpisodeLibrary.model_validate(data)

    # Filter if specified
    episodes = lib.episodes
    if workflow:
        episodes = [e for e in episodes if workflow.lower() in e.canonical_name.lower()]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["name", "description", "steps", "occurrences", "recordings"]
            )
            for ep in episodes:
                writer.writerow(
                    [
                        ep.canonical_name,
                        ep.canonical_description,
                        "; ".join(ep.canonical_steps),
                        ep.occurrence_count,
                        ", ".join(ep.source_recordings),
                    ]
                )

    elif format == "jsonl":
        with open(output_path, "w") as f:
            for ep in episodes:
                f.write(ep.model_dump_json() + "\n")

    elif format == "html":
        html = ["<html><head><style>"]
        html.append("body { font-family: sans-serif; margin: 2em; }")
        html.append(
            ".workflow { border: 1px solid #ccc; padding: 1em; margin: 1em 0; }"
        )
        html.append(".steps { margin-left: 2em; }")
        html.append("</style></head><body>")
        html.append("<h1>Episode Library</h1>")
        html.append(f"<p>{len(episodes)} workflows</p>")

        for ep in episodes:
            html.append('<div class="workflow">')
            html.append(f"<h2>{ep.canonical_name}</h2>")
            html.append(f"<p>{ep.canonical_description}</p>")
            html.append(f"<p><strong>Occurrences:</strong> {ep.occurrence_count}</p>")
            html.append('<div class="steps"><strong>Steps:</strong><ol>')
            for step in ep.canonical_steps:
                html.append(f"<li>{step}</li>")
            html.append("</ol></div></div>")

        html.append("</body></html>")
        output_path.write_text("\n".join(html))

    click.echo(f"Exported {len(episodes)} workflows to: {output_path}")


if __name__ == "__main__":
    segment()
