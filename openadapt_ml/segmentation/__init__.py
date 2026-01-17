"""Workflow segmentation module for OpenAdapt.

This module provides a three-stage pipeline for extracting and deduplicating
workflow episodes from GUI recordings:

1. **Stage 1 - Frame Description (VLM)**: Generate semantic descriptions
   of each frame + action pair using Vision-Language Models

2. **Stage 2 - Episode Extraction (LLM)**: Identify coherent workflow
   boundaries and extract episodes using Large Language Models

3. **Stage 3 - Deduplication (Embeddings)**: Find and merge similar
   episodes across recordings using embedding similarity

Example usage:
    >>> from openadapt_ml.segmentation import SegmentationPipeline
    >>> pipeline = SegmentationPipeline()
    >>> result = pipeline.run(
    ...     recordings=["recording1/", "recording2/"],
    ...     output_dir="segments/",
    ... )
    >>> print(f"Found {result.unique_episodes} unique workflows")
"""

from openadapt_ml.segmentation.schemas import (
    ActionTranscript,
    ActionType,
    AnnotatedEpisodeLibrary,
    CanonicalEpisode,
    Episode,
    EpisodeAnnotation,
    EpisodeBoundary,
    EpisodeExtractionResult,
    EpisodeLibrary,
    EpisodeStep,
    FrameDescription,
)
from openadapt_ml.segmentation.frame_describer import (
    FrameDescriber,
    VLMBackend,
    GeminiBackend,
    ClaudeBackend,
    OpenAIBackend,
)
from openadapt_ml.segmentation.segment_extractor import SegmentExtractor
from openadapt_ml.segmentation.deduplicator import (
    WorkflowDeduplicator,
    OpenAIEmbedder,
    LocalEmbedder,
    episode_to_text,
)
from openadapt_ml.segmentation.pipeline import (
    SegmentationPipeline,
    PipelineConfig,
    PipelineResult,
)
from openadapt_ml.segmentation.annotator import (
    EpisodeAnnotator,
    verify_annotation,
    export_gold_episodes,
)

__all__ = [
    # Schemas
    "ActionTranscript",
    "ActionType",
    "AnnotatedEpisodeLibrary",
    "CanonicalEpisode",
    "Episode",
    "EpisodeAnnotation",
    "EpisodeBoundary",
    "EpisodeExtractionResult",
    "EpisodeLibrary",
    "EpisodeStep",
    "FrameDescription",
    # Frame Describer (Stage 1)
    "FrameDescriber",
    "VLMBackend",
    "GeminiBackend",
    "ClaudeBackend",
    "OpenAIBackend",
    # Segment Extractor (Stage 2)
    "SegmentExtractor",
    # Deduplicator (Stage 3)
    "WorkflowDeduplicator",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "episode_to_text",
    # Pipeline
    "SegmentationPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Annotation (Stage 4)
    "EpisodeAnnotator",
    "verify_annotation",
    "export_gold_episodes",
]
