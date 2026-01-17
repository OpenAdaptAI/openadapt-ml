"""Data schemas for workflow segmentation.

This module defines the Pydantic models used throughout the
segmentation pipeline, ensuring type safety and validation.

In OpenAdapt terminology:
- "Episode" = A coherent workflow segment
- "Trajectory" = Sequence of observation-action pairs (full recording)
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class ActionType(str, Enum):
    """Types of user actions that can be captured."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    SCROLL = "scroll"
    DRAG = "drag"
    HOTKEY = "hotkey"
    MOVE = "move"


class FrameDescription(BaseModel):
    """Description of a single frame + action pair from VLM analysis.

    This is the output of Stage 1 for each frame in the recording.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Timing
    timestamp: float = Field(description="Timestamp in seconds from recording start")
    formatted_time: str = Field(description="Human-readable time format (MM:SS.m)")

    # Screen context
    visible_application: str = Field(
        description="Primary application visible on screen"
    )
    visible_elements: list[str] = Field(
        default_factory=list,
        description="Notable UI elements visible in the frame",
    )
    screen_context: str = Field(description="Brief description of overall screen state")

    # Action details
    action_type: ActionType = Field(description="Type of action performed")
    action_target: Optional[str] = Field(
        default=None,
        description="UI element that was the target of the action",
    )
    action_value: Optional[str] = Field(
        default=None,
        description="Value associated with action (e.g., typed text)",
    )

    # Semantic interpretation
    apparent_intent: str = Field(
        description="What the user appears to be trying to accomplish"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="VLM confidence in this description"
    )

    # Metadata
    frame_index: int = Field(description="Index of this frame in the recording")
    vlm_model: str = Field(description="Model used for description generation")

    def to_transcript_line(self) -> str:
        """Format as a single transcript line."""
        return f"[{self.formatted_time}] {self.apparent_intent}"


class ActionTranscript(BaseModel):
    """Complete transcript of a recording from VLM analysis.

    This is the full output of Stage 1.
    """

    recording_id: str = Field(description="Unique identifier for the source recording")
    recording_name: str = Field(description="Human-readable recording name")
    task_description: Optional[str] = Field(
        default=None,
        description="User-provided task description (if available)",
    )

    # Frame descriptions
    frames: list[FrameDescription] = Field(
        default_factory=list,
        description="Ordered list of frame descriptions",
    )

    # Processing metadata
    total_duration: float = Field(description="Total recording duration in seconds")
    frame_count: int = Field(description="Total number of frames processed")
    vlm_model: str = Field(description="Primary VLM model used")
    processing_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this transcript was generated",
    )

    def to_transcript_text(self) -> str:
        """Format as plain text transcript."""
        lines = []
        for frame in self.frames:
            lines.append(frame.to_transcript_line())
        return "\n".join(lines)

    @property
    def duration_formatted(self) -> str:
        """Return duration as MM:SS format."""
        minutes = int(self.total_duration // 60)
        seconds = self.total_duration % 60
        return f"{minutes:02d}:{seconds:05.2f}"


class EpisodeStep(BaseModel):
    """A single step within an episode (workflow segment)."""

    description: str = Field(description="What this step accomplishes")
    start_timestamp: float = Field(description="Start time in seconds")
    end_timestamp: float = Field(description="End time in seconds")
    frame_indices: list[int] = Field(
        default_factory=list,
        description="Indices of frames belonging to this step",
    )


class EpisodeBoundary(BaseModel):
    """Represents a boundary between episodes with confidence."""

    timestamp: float = Field(description="Time of the boundary")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this is a true episode boundary",
    )
    reason: str = Field(description="Explanation for why this is a boundary")


class Episode(BaseModel):
    """A coherent workflow segment (episode) extracted from a recording.

    This is the output of Stage 2 for each identified workflow.

    In OpenAdapt, an Episode represents a self-contained unit of work
    that can be used for:
    - Training data for fine-tuning
    - Demo conditioning context
    - Workflow library building
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identification
    episode_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this episode",
    )
    name: str = Field(
        description="Concise name for this workflow (e.g., 'Adjust Night Shift Settings')"
    )

    # Timing
    start_time: float = Field(description="Start timestamp in seconds")
    end_time: float = Field(description="End timestamp in seconds")
    start_time_formatted: str = Field(description="Formatted start time (MM:SS.m)")
    end_time_formatted: str = Field(description="Formatted end time (MM:SS.m)")

    # Content
    description: str = Field(
        description="Detailed description of what this workflow accomplishes"
    )
    steps: list[EpisodeStep] = Field(
        default_factory=list,
        description="Ordered list of steps in this workflow",
    )
    step_summaries: list[str] = Field(
        default_factory=list,
        description="Simple list of step descriptions for quick reference",
    )

    # Context
    application: str = Field(description="Primary application used in this workflow")
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Conditions that must be true before starting",
    )
    outcomes: list[str] = Field(
        default_factory=list,
        description="Expected state changes after completion",
    )

    # Hierarchy
    parent_episode_id: Optional[UUID] = Field(
        default=None,
        description="Parent episode if this is a subtask",
    )
    child_episode_ids: list[UUID] = Field(
        default_factory=list,
        description="Child episodes if this contains subtasks",
    )

    # Quality metrics
    boundary_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in episode boundaries",
    )
    coherence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How coherent/self-contained this episode is",
    )

    # Source
    recording_id: str = Field(description="Source recording identifier")
    frame_indices: list[int] = Field(
        default_factory=list,
        description="Indices of frames in this episode",
    )

    @property
    def duration(self) -> float:
        """Episode duration in seconds."""
        return self.end_time - self.start_time

    @property
    def step_count(self) -> int:
        """Number of steps in this episode."""
        return len(self.steps)


class EpisodeExtractionResult(BaseModel):
    """Complete extraction result for a single recording.

    This is the full output of Stage 2.
    """

    recording_id: str = Field(description="Source recording identifier")
    recording_name: str = Field(description="Human-readable recording name")

    # Extracted episodes
    episodes: list[Episode] = Field(
        default_factory=list,
        description="Extracted workflow episodes",
    )

    # Boundaries
    boundaries: list[EpisodeBoundary] = Field(
        default_factory=list,
        description="All identified episode boundaries",
    )

    # Processing metadata
    llm_model: str = Field(description="LLM model used for extraction")
    processing_timestamp: datetime = Field(default_factory=datetime.now)

    # Quality metrics
    coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of recording covered by episodes",
    )
    avg_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Average boundary confidence",
    )


class CanonicalEpisode(BaseModel):
    """A deduplicated, canonical episode definition.

    This represents a workflow type that may appear across multiple recordings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identification
    canonical_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this canonical episode",
    )
    canonical_name: str = Field(description="Standardized name for this workflow")

    # Variants
    variant_names: list[str] = Field(
        default_factory=list,
        description="Alternative names from merged episodes",
    )
    variant_descriptions: list[str] = Field(
        default_factory=list,
        description="Alternative descriptions from merged episodes",
    )

    # Source tracking
    source_recordings: list[str] = Field(
        default_factory=list,
        description="Recording IDs containing this workflow",
    )
    source_episode_ids: list[UUID] = Field(
        default_factory=list,
        description="Original episode IDs that were merged",
    )
    occurrence_count: int = Field(
        ge=1,
        description="Number of times this workflow appears",
    )

    # Canonical definition
    canonical_description: str = Field(
        description="Best/merged description of this workflow"
    )
    canonical_steps: list[str] = Field(
        default_factory=list,
        description="Standardized step list",
    )

    # Embedding
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding for similarity search",
    )

    # Clustering metadata
    cluster_id: int = Field(default=0, description="Cluster ID from deduplication")
    cluster_centroid_distance: float = Field(
        default=0.0,
        ge=0.0,
        description="Distance from cluster centroid",
    )

    # Quality
    internal_similarity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Average similarity between merged variants",
    )


class EpisodeAnnotation(BaseModel):
    """Annotation for an episode indicating its quality for training.

    This model is used to mark episodes as "gold" (suitable for training)
    or exclude them with reasons. VLM-based auto-annotation can populate
    initial values, which humans can then verify.

    Attributes:
        annotation_id: Unique identifier for this annotation
        episode_id: ID of the Episode being annotated
        start_frame: Exact start frame index (refined from Episode)
        end_frame: Exact end frame index (refined from Episode)
        is_gold: Whether this episode should be included in training export
        exclusion_reason: Why this episode was excluded (if not gold)
        confidence: VLM confidence in the annotation (0-1)
        human_verified: Whether a human has confirmed this annotation
        notes: Optional human notes about the episode
        failure_signals: Detected failure signals from post-episode analysis
        created_at: When this annotation was created
        verified_at: When a human verified this annotation
        verified_by: Who verified this annotation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identification
    annotation_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this annotation",
    )
    episode_id: UUID = Field(
        description="ID of the Episode being annotated",
    )

    # Refined boundaries
    start_frame: int = Field(
        ge=0,
        description="Exact start frame index",
    )
    end_frame: int = Field(
        ge=0,
        description="Exact end frame index",
    )

    # Quality assessment
    is_gold: bool = Field(
        default=False,
        description="Should this episode be included in training export?",
    )
    exclusion_reason: Optional[str] = Field(
        default=None,
        description="Why this episode was excluded (e.g., 'task failed', 'incomplete', 'error visible')",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="VLM confidence in the annotation",
    )

    # Human verification
    human_verified: bool = Field(
        default=False,
        description="Has a human confirmed this annotation?",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional human notes about the episode",
    )

    # Failure detection
    failure_signals: list[str] = Field(
        default_factory=list,
        description="Detected failure signals from post-episode analysis",
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this annotation was created",
    )
    verified_at: Optional[datetime] = Field(
        default=None,
        description="When a human verified this annotation",
    )
    verified_by: Optional[str] = Field(
        default=None,
        description="Who verified this annotation",
    )


class AnnotatedEpisodeLibrary(BaseModel):
    """Collection of episodes with their annotations.

    This is used for reviewing, exporting, and managing annotated episodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identification
    library_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this library",
    )
    created_at: datetime = Field(default_factory=datetime.now)

    # Content
    episodes: list[Episode] = Field(
        default_factory=list,
        description="All episodes in this library",
    )
    annotations: list[EpisodeAnnotation] = Field(
        default_factory=list,
        description="Annotations for episodes",
    )

    # Source tracking
    source_recordings: list[str] = Field(
        default_factory=list,
        description="Recording IDs that were processed",
    )

    # Statistics
    @property
    def total_episodes(self) -> int:
        """Total number of episodes."""
        return len(self.episodes)

    @property
    def annotated_count(self) -> int:
        """Number of episodes with annotations."""
        annotated_ids = {a.episode_id for a in self.annotations}
        return len(annotated_ids)

    @property
    def gold_count(self) -> int:
        """Number of gold episodes."""
        return sum(1 for a in self.annotations if a.is_gold)

    @property
    def verified_count(self) -> int:
        """Number of human-verified annotations."""
        return sum(1 for a in self.annotations if a.human_verified)

    @property
    def export_ready_count(self) -> int:
        """Number of episodes ready for export (gold AND verified)."""
        return sum(1 for a in self.annotations if a.is_gold and a.human_verified)

    def get_annotation(self, episode_id: UUID) -> Optional[EpisodeAnnotation]:
        """Get annotation for a specific episode."""
        for annotation in self.annotations:
            if annotation.episode_id == episode_id:
                return annotation
        return None

    def get_episode(self, episode_id: UUID) -> Optional[Episode]:
        """Get episode by ID."""
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return episode
        return None

    def get_gold_episodes(self) -> list[tuple[Episode, EpisodeAnnotation]]:
        """Get all gold episodes with their annotations."""
        result = []
        for annotation in self.annotations:
            if annotation.is_gold:
                episode = self.get_episode(annotation.episode_id)
                if episode:
                    result.append((episode, annotation))
        return result

    def get_verified_gold_episodes(self) -> list[tuple[Episode, EpisodeAnnotation]]:
        """Get episodes that are both gold AND human-verified."""
        result = []
        for annotation in self.annotations:
            if annotation.is_gold and annotation.human_verified:
                episode = self.get_episode(annotation.episode_id)
                if episode:
                    result.append((episode, annotation))
        return result

    def get_pending_review(self) -> list[tuple[Episode, EpisodeAnnotation]]:
        """Get episodes that have annotations but need human verification."""
        result = []
        for annotation in self.annotations:
            if not annotation.human_verified:
                episode = self.get_episode(annotation.episode_id)
                if episode:
                    result.append((episode, annotation))
        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotatedEpisodeLibrary":
        """Create from dictionary."""
        return cls.model_validate(data)


class EpisodeLibrary(BaseModel):
    """Complete deduplicated episode library.

    This is the final output of Stage 3 - a library of canonical
    workflow episodes that can be used for training data curation,
    demo conditioning, and workflow retrieval.
    """

    # Library metadata
    library_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this library version",
    )
    created_at: datetime = Field(default_factory=datetime.now)

    # Workflows
    episodes: list[CanonicalEpisode] = Field(
        default_factory=list,
        description="All canonical episodes",
    )

    # Statistics
    total_recordings_processed: int = Field(
        ge=0,
        description="Number of recordings analyzed",
    )
    total_episodes_extracted: int = Field(
        ge=0,
        description="Total episodes before deduplication",
    )
    unique_episode_count: int = Field(
        ge=0,
        description="Number of unique episodes after deduplication",
    )
    deduplication_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of episodes that were duplicates",
    )

    # Processing parameters
    similarity_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Threshold used for clustering",
    )
    embedding_model: str = Field(description="Model used for embeddings")

    def get_episode_by_name(self, name: str) -> Optional[CanonicalEpisode]:
        """Find episode by canonical name."""
        for episode in self.episodes:
            if episode.canonical_name.lower() == name.lower():
                return episode
            if name.lower() in [v.lower() for v in episode.variant_names]:
                return episode
        return None

    def get_episodes_for_recording(self, recording_id: str) -> list[CanonicalEpisode]:
        """Get all episodes that appear in a specific recording."""
        return [e for e in self.episodes if recording_id in e.source_recordings]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeLibrary":
        """Create from dictionary."""
        return cls.model_validate(data)
