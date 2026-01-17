"""End-to-end segmentation pipeline.

This module provides a unified interface for running the complete
three-stage segmentation pipeline for episode extraction.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from openadapt_ml.segmentation.schemas import (
    ActionTranscript,
    EpisodeExtractionResult,
    EpisodeLibrary,
)
from openadapt_ml.segmentation.frame_describer import FrameDescriber
from openadapt_ml.segmentation.segment_extractor import SegmentExtractor
from openadapt_ml.segmentation.deduplicator import WorkflowDeduplicator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the segmentation pipeline."""

    # Stage 1: Frame description
    vlm_model: str = "gemini-2.0-flash"
    vlm_batch_size: int = 10

    # Stage 2: Episode extraction
    llm_model: str = "gpt-4o"
    use_few_shot: bool = True
    hierarchical: bool = False
    min_segment_duration: float = 2.0
    max_segment_duration: float = 300.0

    # Stage 3: Deduplication
    similarity_threshold: float = 0.85
    embedding_model: str = "text-embedding-3-large"
    merge_strategy: str = "centroid"
    use_local_embeddings: bool = False

    # General
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    verbose: bool = False


@dataclass
class PipelineResult:
    """Result of running the segmentation pipeline."""

    # Per-recording outputs
    transcripts: dict[str, ActionTranscript] = field(default_factory=dict)
    extractions: dict[str, EpisodeExtractionResult] = field(default_factory=dict)

    # Combined output
    library: Optional[EpisodeLibrary] = None

    # Metadata
    config: Optional[PipelineConfig] = None
    recordings_processed: int = 0
    total_episodes_extracted: int = 0
    unique_episodes: int = 0
    processing_time_seconds: float = 0.0


class SegmentationPipeline:
    """Complete workflow segmentation pipeline.

    Orchestrates all three stages to process recordings into
    a deduplicated episode library.

    Example:
        >>> pipeline = SegmentationPipeline()
        >>> result = pipeline.run(
        ...     recordings=["recording1/", "recording2/"],
        ...     output_dir="segments/",
        ... )
        >>> print(f"Extracted {result.unique_episodes} unique workflows")
        >>> result.library.to_dict()
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not specified.
        """
        self.config = config or PipelineConfig()
        self._describer: Optional[FrameDescriber] = None
        self._extractor: Optional[SegmentExtractor] = None
        self._deduplicator: Optional[WorkflowDeduplicator] = None

    @property
    def describer(self) -> FrameDescriber:
        """Lazy-load frame describer."""
        if self._describer is None:
            self._describer = FrameDescriber(
                model=self.config.vlm_model,
                batch_size=self.config.vlm_batch_size,
                cache_enabled=self.config.cache_enabled,
                cache_dir=self.config.cache_dir,
            )
        return self._describer

    @property
    def extractor(self) -> SegmentExtractor:
        """Lazy-load segment extractor."""
        if self._extractor is None:
            self._extractor = SegmentExtractor(
                model=self.config.llm_model,
                use_few_shot=self.config.use_few_shot,
                hierarchical=self.config.hierarchical,
                min_segment_duration=self.config.min_segment_duration,
                max_segment_duration=self.config.max_segment_duration,
            )
        return self._extractor

    @property
    def deduplicator(self) -> WorkflowDeduplicator:
        """Lazy-load deduplicator."""
        if self._deduplicator is None:
            self._deduplicator = WorkflowDeduplicator(
                threshold=self.config.similarity_threshold,
                embedding_model=self.config.embedding_model,
                merge_strategy=self.config.merge_strategy,
                use_local_embeddings=self.config.use_local_embeddings,
            )
        return self._deduplicator

    def run(
        self,
        recordings: list[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        existing_library: Optional[EpisodeLibrary] = None,
        progress_callback: Optional[callable] = None,
    ) -> PipelineResult:
        """Run the complete pipeline on a set of recordings.

        Args:
            recordings: List of recording paths to process.
            output_dir: Directory to save intermediate and final outputs.
            existing_library: Existing library to merge with.
            progress_callback: Optional callback(stage, current, total).

        Returns:
            PipelineResult with all outputs.
        """
        start_time = time.time()
        result = PipelineResult(config=self.config)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Generate descriptions for each recording
        logger.info(f"Stage 1: Processing {len(recordings)} recordings")
        for i, recording_path in enumerate(recordings):
            recording_path = Path(recording_path)
            recording_id = recording_path.name

            if progress_callback:
                progress_callback("describe", i + 1, len(recordings))

            logger.info(f"  Describing: {recording_id}")
            transcript = self.run_stage1(recording_path)
            result.transcripts[recording_id] = transcript

            # Save intermediate result
            if output_dir:
                transcript_path = output_dir / f"{recording_id}_transcript.json"
                transcript_path.write_text(transcript.model_dump_json(indent=2))

        # Stage 2: Extract episodes from each transcript
        logger.info("Stage 2: Extracting episodes")
        extraction_results = []
        for i, (recording_id, transcript) in enumerate(result.transcripts.items()):
            if progress_callback:
                progress_callback("extract", i + 1, len(result.transcripts))

            logger.info(f"  Extracting: {recording_id}")
            extraction = self.run_stage2(transcript)
            result.extractions[recording_id] = extraction
            extraction_results.append(extraction)

            # Save intermediate result
            if output_dir:
                extraction_path = output_dir / f"{recording_id}_episodes.json"
                extraction_path.write_text(extraction.model_dump_json(indent=2))

        # Stage 3: Deduplicate across all recordings
        logger.info("Stage 3: Deduplicating episodes")
        if progress_callback:
            progress_callback("deduplicate", 1, 1)

        result.library = self.run_stage3(extraction_results, existing_library)

        # Save final result
        if output_dir:
            library_path = output_dir / "episode_library.json"
            library_path.write_text(result.library.model_dump_json(indent=2))

        # Calculate statistics
        result.recordings_processed = len(recordings)
        result.total_episodes_extracted = sum(
            len(ext.episodes) for ext in extraction_results
        )
        result.unique_episodes = result.library.unique_episode_count
        result.processing_time_seconds = time.time() - start_time

        logger.info(
            f"Pipeline complete: {result.unique_episodes} unique episodes "
            f"from {result.total_episodes_extracted} total "
            f"({result.library.deduplication_ratio:.1%} duplicates)"
        )

        return result

    def run_stage1(
        self,
        recording: Union[str, Path],
    ) -> ActionTranscript:
        """Run only Stage 1 (frame description).

        Useful for inspecting intermediate outputs or debugging.

        Args:
            recording: Recording path.

        Returns:
            ActionTranscript for this recording.
        """
        return self.describer.describe_recording(recording)

    def run_stage2(
        self,
        transcript: ActionTranscript,
    ) -> EpisodeExtractionResult:
        """Run only Stage 2 (episode extraction).

        Args:
            transcript: ActionTranscript from Stage 1.

        Returns:
            EpisodeExtractionResult for this transcript.
        """
        return self.extractor.extract_segments(transcript)

    def run_stage3(
        self,
        extractions: list[EpisodeExtractionResult],
        existing_library: Optional[EpisodeLibrary] = None,
    ) -> EpisodeLibrary:
        """Run only Stage 3 (deduplication).

        Args:
            extractions: List of extraction results from Stage 2.
            existing_library: Existing library to merge with.

        Returns:
            Deduplicated EpisodeLibrary.
        """
        return self.deduplicator.deduplicate(extractions, existing_library)

    def resume(
        self,
        output_dir: Union[str, Path],
        recordings: Optional[list[Union[str, Path]]] = None,
    ) -> PipelineResult:
        """Resume a previously interrupted pipeline run.

        Loads cached intermediate results and continues from where it stopped.

        Args:
            output_dir: Directory with previous run's outputs.
            recordings: Additional recordings to process (optional).

        Returns:
            PipelineResult with combined outputs.
        """
        import json

        output_dir = Path(output_dir)
        result = PipelineResult(config=self.config)

        # Load existing transcripts
        for transcript_file in output_dir.glob("*_transcript.json"):
            data = json.loads(transcript_file.read_text())
            transcript = ActionTranscript.model_validate(data)
            result.transcripts[transcript.recording_id] = transcript

        # Load existing extractions
        for extraction_file in output_dir.glob("*_episodes.json"):
            data = json.loads(extraction_file.read_text())
            extraction = EpisodeExtractionResult.model_validate(data)
            result.extractions[extraction.recording_id] = extraction

        # Load existing library if present
        library_path = output_dir / "episode_library.json"
        existing_library = None
        if library_path.exists():
            data = json.loads(library_path.read_text())
            existing_library = EpisodeLibrary.model_validate(data)

        # Process new recordings if provided
        if recordings:
            new_recordings = [
                r for r in recordings if Path(r).name not in result.transcripts
            ]
            if new_recordings:
                new_result = self.run(
                    new_recordings,
                    output_dir=output_dir,
                    existing_library=existing_library,
                )
                # Merge results
                result.transcripts.update(new_result.transcripts)
                result.extractions.update(new_result.extractions)
                result.library = new_result.library

        # If no new recordings, just re-run deduplication
        if not recordings and result.extractions:
            extraction_results = list(result.extractions.values())
            result.library = self.run_stage3(extraction_results, existing_library)

        result.recordings_processed = len(result.transcripts)
        result.total_episodes_extracted = sum(
            len(ext.episodes) for ext in result.extractions.values()
        )
        if result.library:
            result.unique_episodes = result.library.unique_episode_count

        return result
