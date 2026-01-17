#!/usr/bin/env python
"""Test script for workflow segmentation pipeline.

This script validates the segmentation system on real captures and generates
documentation artifacts (viewers, screenshots, examples).
"""

import json
import logging
import sys
from pathlib import Path

from openadapt_ml.config import settings

# Create output directory first
Path("segmentation_output").mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("segmentation_output/test_run.log")
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check that environment is properly configured."""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT CHECK")
    logger.info("=" * 60)

    issues = []

    # Check API keys
    if not settings.google_api_key:
        issues.append("GOOGLE_API_KEY not set (needed for Stage 1 - Frame Description)")
    else:
        logger.info("✓ GOOGLE_API_KEY is set")

    if not settings.openai_api_key:
        issues.append("OPENAI_API_KEY not set (needed for Stage 2 - Episode Extraction)")
    else:
        logger.info("✓ OPENAI_API_KEY is set")

    # Check directories exist
    captures_dir = Path("/Users/abrichr/oa/src/openadapt-capture")
    if not captures_dir.exists():
        issues.append(f"Captures directory not found: {captures_dir}")
    else:
        logger.info(f"✓ Captures directory exists: {captures_dir}")

    # Check test recordings
    nightshift = captures_dir / "turn-off-nightshift"
    demo_new = captures_dir / "demo_new"

    if not nightshift.exists():
        issues.append(f"Test recording not found: {nightshift}")
    else:
        db = nightshift / "capture.db"
        screenshots = nightshift / "screenshots"
        if not db.exists():
            issues.append(f"capture.db not found in {nightshift}")
        if not screenshots.exists():
            issues.append(f"screenshots/ not found in {nightshift}")
        else:
            num_screenshots = len(list(screenshots.glob("*.png")))
            logger.info(f"✓ turn-off-nightshift: {num_screenshots} screenshots")

    if not demo_new.exists():
        issues.append(f"Test recording not found: {demo_new}")
    else:
        db = demo_new / "capture.db"
        screenshots = demo_new / "screenshots"
        if not db.exists():
            issues.append(f"capture.db not found in {demo_new}")
        if not screenshots.exists():
            issues.append(f"screenshots/ not found in {demo_new}")
        else:
            num_screenshots = len(list(screenshots.glob("*.png")))
            logger.info(f"✓ demo_new: {num_screenshots} screenshots")

    # Create output directories
    output_dir = Path("segmentation_output")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"✓ Output directory: {output_dir}")

    docs_images = Path("docs/images/segmentation")
    docs_images.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Docs images directory: {docs_images}")

    docs_examples = Path("docs/examples")
    docs_examples.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Docs examples directory: {docs_examples}")

    if issues:
        logger.error("\nISSUES FOUND:")
        for issue in issues:
            logger.error(f"  ✗ {issue}")
        return False

    logger.info("\n✓ All environment checks passed!")
    return True


def run_stage1(recording_path: Path, output_path: Path):
    """Run Stage 1: Frame Description using VLM."""
    from openadapt_ml.segmentation.frame_describer import FrameDescriber

    logger.info("=" * 60)
    logger.info(f"STAGE 1: Frame Description - {recording_path.name}")
    logger.info("=" * 60)

    logger.info(f"Recording: {recording_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: gemini-2.0-flash")

    describer = FrameDescriber(
        model="gemini-2.0-flash",
        batch_size=10,
        cache_enabled=True,
    )

    logger.info("Describing frames...")
    transcript = describer.describe_recording(str(recording_path))

    # Save as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcript.model_dump_json(indent=2))

    logger.info(f"✓ Transcript saved: {output_path}")
    logger.info(f"  Frames: {len(transcript.frames)}")
    logger.info(f"  Duration: {transcript.total_duration:.1f}s")
    logger.info(f"  Recording ID: {transcript.recording_id}")

    return transcript


def run_stage2(transcript_path: Path, output_path: Path):
    """Run Stage 2: Episode Extraction using LLM."""
    from openadapt_ml.segmentation.segment_extractor import SegmentExtractor
    from openadapt_ml.segmentation.schemas import ActionTranscript

    logger.info("=" * 60)
    logger.info(f"STAGE 2: Episode Extraction - {transcript_path.stem}")
    logger.info("=" * 60)

    logger.info(f"Transcript: {transcript_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: gpt-4o")

    # Load transcript
    data = json.loads(transcript_path.read_text())
    transcript = ActionTranscript.model_validate(data)

    logger.info(f"Loaded transcript with {len(transcript.frames)} frames")

    # Extract episodes
    extractor = SegmentExtractor(
        model="gpt-4o",
        use_few_shot=True,
        min_segment_duration=2.0,
        max_segment_duration=300.0,
    )

    logger.info("Extracting episodes...")
    result = extractor.extract_segments(transcript)

    # Save as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2))

    logger.info(f"✓ Episodes saved: {output_path}")
    logger.info(f"  Episodes: {len(result.episodes)}")

    for i, ep in enumerate(result.episodes, 1):
        logger.info(f"\n  Episode {i}: {ep.name}")
        logger.info(f"    Time: {ep.start_time_formatted} - {ep.end_time_formatted}")
        logger.info(f"    Duration: {ep.duration:.1f}s")
        logger.info(f"    Steps: {len(ep.step_summaries)}")
        logger.info(f"    Confidence: {ep.boundary_confidence:.2f}")
        logger.info(f"    Description: {ep.description[:100]}...")

    return result


def main():
    """Run full test pipeline."""
    logger.info("=" * 60)
    logger.info("WORKFLOW SEGMENTATION PIPELINE TEST")
    logger.info("=" * 60)
    logger.info("Test Date: 2026-01-17")
    logger.info("Commit: 56e8cb6")
    logger.info("")

    # Check environment
    if not check_environment():
        logger.error("\nEnvironment check failed. Please fix issues and retry.")
        sys.exit(1)

    logger.info("")

    # Define test recordings
    base_path = Path("/Users/abrichr/oa/src/openadapt-capture")
    recordings = [
        ("turn-off-nightshift", base_path / "turn-off-nightshift"),
        ("demo_new", base_path / "demo_new"),
    ]

    results = {}

    # Process each recording
    for name, recording_path in recordings:
        logger.info("\n" + "=" * 60)
        logger.info(f"PROCESSING: {name}")
        logger.info("=" * 60)

        try:
            # Stage 1: Frame Description
            transcript_path = Path(f"segmentation_output/{name}_transcript.json")
            if transcript_path.exists():
                logger.info(f"✓ Transcript already exists: {transcript_path}")
                logger.info("  Skipping Stage 1 (delete file to regenerate)")
                transcript_data = json.loads(transcript_path.read_text())
                from openadapt_ml.segmentation.schemas import ActionTranscript
                transcript = ActionTranscript.model_validate(transcript_data)
            else:
                transcript = run_stage1(recording_path, transcript_path)

            # Stage 2: Episode Extraction
            episodes_path = Path(f"segmentation_output/{name}_episodes.json")
            if episodes_path.exists():
                logger.info(f"\n✓ Episodes already exist: {episodes_path}")
                logger.info("  Skipping Stage 2 (delete file to regenerate)")
                episodes_data = json.loads(episodes_path.read_text())
                from openadapt_ml.segmentation.schemas import EpisodeExtractionResult
                result = EpisodeExtractionResult.model_validate(episodes_data)
            else:
                result = run_stage2(transcript_path, episodes_path)

            results[name] = {
                "recording_path": str(recording_path),
                "frames": len(transcript.frames),
                "duration": transcript.total_duration,
                "episodes": len(result.episodes),
                "transcript_path": str(transcript_path),
                "episodes_path": str(episodes_path),
            }

            logger.info(f"\n✓ {name} completed successfully!")

        except Exception as e:
            logger.error(f"\n✗ {name} FAILED: {e}", exc_info=True)
            results[name] = {"error": str(e)}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for name, result in results.items():
        logger.info(f"\n{name}:")
        if "error" in result:
            logger.info(f"  ✗ FAILED: {result['error']}")
        else:
            logger.info(f"  ✓ Frames: {result['frames']}")
            logger.info(f"  ✓ Duration: {result['duration']:.1f}s")
            logger.info(f"  ✓ Episodes: {result['episodes']}")
            logger.info(f"  ✓ Transcript: {result['transcript_path']}")
            logger.info(f"  ✓ Episodes: {result['episodes_path']}")

    # Save results
    results_path = Path("segmentation_output/test_results.json")
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\n✓ Results saved: {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info("1. Create HTML viewer (openadapt_ml/segmentation/viewer.py)")
    logger.info("2. Generate viewers for both recordings")
    logger.info("3. Create screenshot capture script (Playwright)")
    logger.info("4. Generate documentation screenshots")
    logger.info("5. Extract example JSON for README")
    logger.info("6. Update README with results")
    logger.info("7. Create final test results report")


if __name__ == "__main__":
    main()
