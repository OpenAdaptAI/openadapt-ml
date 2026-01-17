"""VLM-based episode annotation for training data quality control.

This module provides automatic annotation of extracted episodes using
Vision-Language Models to determine which episodes are suitable for
training ("gold") and which should be excluded.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from openadapt_ml.segmentation.schemas import (
    AnnotatedEpisodeLibrary,
    Episode,
    EpisodeAnnotation,
    EpisodeExtractionResult,
)

logger = logging.getLogger(__name__)


class EpisodeAnnotator:
    """Annotates episodes using VLM analysis for training data quality.

    This class examines episode frames and post-episode frames to:
    1. Identify precise episode boundaries
    2. Detect failure signals (errors, undos, repeated attempts)
    3. Assess whether the workflow completed successfully
    4. Generate is_gold recommendations

    Example:
        >>> annotator = EpisodeAnnotator(model="gemini-2.0-flash")
        >>> library = annotator.annotate_episodes(
        ...     episodes=extraction_result.episodes,
        ...     recording_path="/path/to/recording",
        ... )
        >>> print(f"Found {library.gold_count} gold episodes")

    Attributes:
        model: VLM model identifier
        lookback_frames: Number of frames to analyze before episode
        lookahead_frames: Number of frames to analyze after episode
        confidence_threshold: Minimum confidence to mark as gold
    """

    SUPPORTED_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-pro",
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    # Common failure signals to detect
    FAILURE_INDICATORS = [
        "error",
        "failed",
        "undo",
        "cancel",
        "retry",
        "oops",
        "wrong",
        "delete",
        "remove",
        "revert",
        "back",
        "ctrl+z",
        "cmd+z",
    ]

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        lookback_frames: int = 3,
        lookahead_frames: int = 10,
        confidence_threshold: float = 0.7,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the episode annotator.

        Args:
            model: VLM model to use for analysis.
            lookback_frames: Number of frames to check before episode start.
            lookahead_frames: Number of frames to check after episode end.
            confidence_threshold: Minimum confidence to recommend as gold.
            api_key: API key for VLM provider (uses env var if not provided).
        """
        self.model = model
        self.lookback_frames = lookback_frames
        self.lookahead_frames = lookahead_frames
        self.confidence_threshold = confidence_threshold
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        """Get or create VLM client."""
        if self._client is not None:
            return self._client

        from openadapt_ml.config import settings

        if "gemini" in self.model.lower():
            import google.generativeai as genai

            api_key = self._api_key or settings.google_api_key
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
        elif "claude" in self.model.lower():
            import anthropic

            api_key = self._api_key or settings.anthropic_api_key
            self._client = anthropic.Anthropic(api_key=api_key)
        elif "gpt" in self.model.lower():
            import openai

            api_key = self._api_key or settings.openai_api_key
            self._client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return self._client

    def _encode_image(self, image: Image.Image) -> dict:
        """Encode image for API calls."""
        import base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        return b64

    def _get_annotation_prompt(
        self,
        episode: Episode,
        has_post_frames: bool,
    ) -> str:
        """Generate prompt for episode annotation."""
        return f"""You are analyzing a GUI workflow episode to determine if it should be included in a training dataset.

## Episode Information
- **Name**: {episode.name}
- **Description**: {episode.description}
- **Duration**: {episode.start_time_formatted} - {episode.end_time_formatted}
- **Steps**: {", ".join(episode.step_summaries)}
- **Application**: {episode.application}

## Analysis Task

Examine the provided screenshots and determine:

1. **Boundary Accuracy**: Are the episode boundaries (start/end frames) correct?
   - Does the first frame show the actual start of the workflow?
   - Does the last frame show the actual completion?

2. **Workflow Completeness**: Did the workflow complete successfully?
   - Were all steps executed?
   - Is there a clear completion state visible?

3. **Failure Detection**: Look for any signs of failure:
   - Error dialogs or messages
   - User performing undo actions (Ctrl+Z, etc.)
   - Repeated attempts at the same action
   - User navigating back or canceling
   - Signs of frustration (rapid clicking, erratic movements)

{"4. **Post-Episode Analysis**: Examine frames AFTER the episode ended:" if has_post_frames else ""}
{"   - Are there error dialogs appearing after completion?" if has_post_frames else ""}
{"   - Does the user immediately undo or retry the task?" if has_post_frames else ""}
{"   - Is there evidence the workflow actually failed?" if has_post_frames else ""}

## Response Format

Respond with JSON:
```json
{{
  "is_gold": true/false,
  "confidence": 0.0-1.0,
  "start_frame_correct": true/false,
  "end_frame_correct": true/false,
  "suggested_start_offset": 0,
  "suggested_end_offset": 0,
  "workflow_complete": true/false,
  "failure_signals": ["list of detected issues"],
  "exclusion_reason": "reason if not gold, null if gold",
  "analysis_notes": "brief explanation of assessment"
}}
```

**Guidelines for is_gold**:
- TRUE if: Workflow completed successfully, no errors visible, episode is coherent and self-contained
- FALSE if: Any errors detected, incomplete workflow, user had to retry, or evidence of failure in post-frames
"""

    def _call_vlm(
        self,
        prompt: str,
        images: list[Image.Image],
    ) -> dict:
        """Call VLM with images and return parsed response."""
        client = self._get_client()

        if "gemini" in self.model.lower():
            content = [prompt] + images
            response = client.generate_content(content)
            text = response.text
        elif "claude" in self.model.lower():
            content = []
            for img in images:
                b64 = self._encode_image(img)
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    }
                )
            content.append({"type": "text", "text": prompt})
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": content}],
            )
            text = response.content[0].text
        elif "gpt" in self.model.lower():
            content = []
            for img in images:
                b64 = self._encode_image(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            content.append({"type": "text", "text": prompt})
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": content}],
            )
            text = response.choices[0].message.content
        else:
            raise ValueError(f"Unknown model: {self.model}")

        # Parse JSON from response
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        # Return default if parsing failed
        return {
            "is_gold": False,
            "confidence": 0.3,
            "failure_signals": ["Failed to parse VLM response"],
            "exclusion_reason": "VLM response parsing failed",
            "analysis_notes": text[:200],
        }

    def _load_frames(
        self,
        recording_path: Path,
        frame_indices: list[int],
    ) -> list[Image.Image]:
        """Load frames from recording directory."""
        images = []
        screenshots_dir = recording_path / "screenshots"

        if not screenshots_dir.exists():
            # Try direct directory with numbered PNGs
            png_files = sorted(recording_path.glob("*.png"))
            if png_files:
                for idx in frame_indices:
                    if 0 <= idx < len(png_files):
                        try:
                            images.append(Image.open(png_files[idx]))
                        except Exception as e:
                            logger.warning(f"Failed to load frame {idx}: {e}")
            return images

        # Load from screenshots directory
        for idx in frame_indices:
            path = screenshots_dir / f"{idx:06d}.png"
            if path.exists():
                try:
                    images.append(Image.open(path))
                except Exception as e:
                    logger.warning(f"Failed to load frame {idx}: {e}")

        return images

    def annotate_episode(
        self,
        episode: Episode,
        recording_path: Union[str, Path],
        total_frames: int,
    ) -> EpisodeAnnotation:
        """Annotate a single episode.

        Args:
            episode: Episode to annotate.
            recording_path: Path to the recording directory.
            total_frames: Total number of frames in the recording.

        Returns:
            EpisodeAnnotation with VLM-generated assessment.
        """
        recording_path = Path(recording_path)

        # Determine frame ranges to analyze
        start_frame = min(episode.frame_indices) if episode.frame_indices else 0
        end_frame = max(episode.frame_indices) if episode.frame_indices else 0

        # Get episode frames (sample if too many)
        episode_frames = episode.frame_indices
        if len(episode_frames) > 10:
            # Sample: first 3, middle 4, last 3
            sampled = (
                episode_frames[:3]
                + episode_frames[
                    len(episode_frames) // 2 - 2 : len(episode_frames) // 2 + 2
                ]
                + episode_frames[-3:]
            )
            episode_frames = sorted(set(sampled))

        # Get post-episode frames
        post_start = end_frame + 1
        post_end = min(end_frame + self.lookahead_frames + 1, total_frames)
        post_frames = list(range(post_start, post_end))

        # Load images
        all_frames = episode_frames + post_frames
        images = self._load_frames(recording_path, all_frames)

        if not images:
            logger.warning(f"No frames loaded for episode {episode.episode_id}")
            return EpisodeAnnotation(
                episode_id=episode.episode_id,
                start_frame=start_frame,
                end_frame=end_frame,
                is_gold=False,
                exclusion_reason="Failed to load episode frames",
                confidence=0.0,
                failure_signals=["No frames available for analysis"],
            )

        # Generate annotation
        prompt = self._get_annotation_prompt(
            episode=episode,
            has_post_frames=len(post_frames) > 0,
        )

        result = self._call_vlm(prompt, images)

        # Apply boundary adjustments
        adjusted_start = start_frame + result.get("suggested_start_offset", 0)
        adjusted_end = end_frame + result.get("suggested_end_offset", 0)

        return EpisodeAnnotation(
            episode_id=episode.episode_id,
            start_frame=max(0, adjusted_start),
            end_frame=min(total_frames - 1, adjusted_end),
            is_gold=result.get("is_gold", False)
            and result.get("confidence", 0) >= self.confidence_threshold,
            exclusion_reason=result.get("exclusion_reason"),
            confidence=result.get("confidence", 0.5),
            failure_signals=result.get("failure_signals", []),
        )

    def annotate_episodes(
        self,
        episodes: list[Episode],
        recording_path: Union[str, Path],
        total_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> AnnotatedEpisodeLibrary:
        """Annotate multiple episodes from a recording.

        Args:
            episodes: List of episodes to annotate.
            recording_path: Path to the recording directory.
            total_frames: Total number of frames (auto-detected if not provided).
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            AnnotatedEpisodeLibrary with all episodes and annotations.
        """
        recording_path = Path(recording_path)

        # Auto-detect total frames if not provided
        if total_frames is None:
            screenshots_dir = recording_path / "screenshots"
            if screenshots_dir.exists():
                total_frames = len(list(screenshots_dir.glob("*.png")))
            else:
                total_frames = len(list(recording_path.glob("*.png")))

        annotations = []
        for i, episode in enumerate(episodes):
            logger.info(f"Annotating episode {i + 1}/{len(episodes)}: {episode.name}")

            annotation = self.annotate_episode(
                episode=episode,
                recording_path=recording_path,
                total_frames=total_frames,
            )
            annotations.append(annotation)

            if progress_callback:
                progress_callback(i + 1, len(episodes))

        # Build library
        recording_ids = list(set(e.recording_id for e in episodes))

        return AnnotatedEpisodeLibrary(
            episodes=episodes,
            annotations=annotations,
            source_recordings=recording_ids,
        )

    def annotate_extraction_result(
        self,
        extraction_result: EpisodeExtractionResult,
        recording_path: Union[str, Path],
        total_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> AnnotatedEpisodeLibrary:
        """Annotate all episodes from an extraction result.

        Args:
            extraction_result: Output from SegmentExtractor.
            recording_path: Path to the recording directory.
            total_frames: Total number of frames.
            progress_callback: Optional callback for progress.

        Returns:
            AnnotatedEpisodeLibrary with annotations.
        """
        return self.annotate_episodes(
            episodes=extraction_result.episodes,
            recording_path=recording_path,
            total_frames=total_frames,
            progress_callback=progress_callback,
        )


def verify_annotation(
    annotation: EpisodeAnnotation,
    is_gold: bool,
    notes: Optional[str] = None,
    verified_by: Optional[str] = None,
) -> EpisodeAnnotation:
    """Update an annotation with human verification.

    Args:
        annotation: The annotation to verify.
        is_gold: Human decision on gold status.
        notes: Optional notes from the reviewer.
        verified_by: Name/ID of the person verifying.

    Returns:
        Updated EpisodeAnnotation with human_verified=True.
    """
    return EpisodeAnnotation(
        annotation_id=annotation.annotation_id,
        episode_id=annotation.episode_id,
        start_frame=annotation.start_frame,
        end_frame=annotation.end_frame,
        is_gold=is_gold,
        exclusion_reason=annotation.exclusion_reason if not is_gold else None,
        confidence=annotation.confidence,
        human_verified=True,
        notes=notes or annotation.notes,
        failure_signals=annotation.failure_signals,
        created_at=annotation.created_at,
        verified_at=datetime.now(),
        verified_by=verified_by,
    )


def export_gold_episodes(
    library: AnnotatedEpisodeLibrary,
    output_path: Union[str, Path],
    recording_path: Optional[Union[str, Path]] = None,
    format: str = "jsonl",
    include_screenshots: bool = False,
) -> int:
    """Export gold episodes for fine-tuning.

    Only exports episodes where is_gold=True AND human_verified=True.

    Args:
        library: AnnotatedEpisodeLibrary to export from.
        output_path: Path to output file/directory.
        recording_path: Path to recording (needed if include_screenshots=True).
        format: Export format ("jsonl", "json", or "hf" for HuggingFace).
        include_screenshots: Whether to include screenshots in export.

    Returns:
        Number of episodes exported.
    """
    output_path = Path(output_path)

    # Get verified gold episodes
    gold_episodes = library.get_verified_gold_episodes()

    if not gold_episodes:
        logger.warning("No verified gold episodes to export")
        return 0

    if format == "jsonl":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for episode, annotation in gold_episodes:
                record = {
                    "episode_id": str(episode.episode_id),
                    "name": episode.name,
                    "description": episode.description,
                    "application": episode.application,
                    "steps": episode.step_summaries,
                    "start_frame": annotation.start_frame,
                    "end_frame": annotation.end_frame,
                    "recording_id": episode.recording_id,
                    "annotation_confidence": annotation.confidence,
                    "verified_by": annotation.verified_by,
                    "notes": annotation.notes,
                }
                f.write(json.dumps(record) + "\n")

    elif format == "json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for episode, annotation in gold_episodes:
            records.append(
                {
                    "episode_id": str(episode.episode_id),
                    "name": episode.name,
                    "description": episode.description,
                    "application": episode.application,
                    "steps": episode.step_summaries,
                    "start_frame": annotation.start_frame,
                    "end_frame": annotation.end_frame,
                    "start_time": episode.start_time,
                    "end_time": episode.end_time,
                    "recording_id": episode.recording_id,
                    "annotation_confidence": annotation.confidence,
                    "verified_by": annotation.verified_by,
                    "notes": annotation.notes,
                }
            )
        output_path.write_text(json.dumps(records, indent=2))

    elif format == "hf":
        # Export in HuggingFace datasets format
        output_path.mkdir(parents=True, exist_ok=True)
        records = []
        for episode, annotation in gold_episodes:
            record = {
                "episode_id": str(episode.episode_id),
                "task_name": episode.name,
                "task_description": episode.description,
                "application": episode.application,
                "steps": episode.step_summaries,
                "frame_indices": list(
                    range(annotation.start_frame, annotation.end_frame + 1)
                ),
                "recording_id": episode.recording_id,
            }

            if include_screenshots and recording_path:
                # Load and save screenshots
                episode_dir = output_path / str(episode.episode_id)
                episode_dir.mkdir(parents=True, exist_ok=True)

                screenshots_src = Path(recording_path) / "screenshots"
                screenshot_paths = []
                for idx in range(annotation.start_frame, annotation.end_frame + 1):
                    src = screenshots_src / f"{idx:06d}.png"
                    if src.exists():
                        dst = episode_dir / f"frame_{idx:06d}.png"
                        import shutil

                        shutil.copy(src, dst)
                        screenshot_paths.append(str(dst))
                record["screenshot_paths"] = screenshot_paths

            records.append(record)

        # Save metadata
        (output_path / "metadata.json").write_text(json.dumps(records, indent=2))

    else:
        raise ValueError(f"Unknown export format: {format}")

    logger.info(f"Exported {len(gold_episodes)} gold episodes to {output_path}")
    return len(gold_episodes)
