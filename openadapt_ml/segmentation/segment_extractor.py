"""Workflow segment extraction using Large Language Models.

This module analyzes action transcripts to identify coherent
workflow segments (episodes) with clear boundaries (Stage 2 of pipeline).
"""

import json
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from openadapt_ml.segmentation.schemas import (
    ActionTranscript,
    Episode,
    EpisodeBoundary,
    EpisodeExtractionResult,
)

logger = logging.getLogger(__name__)


class SegmentExtractor:
    """Extracts workflow segments (episodes) from action transcripts using LLMs.

    This class implements Stage 2 of the segmentation pipeline, identifying
    coherent workflow boundaries within recorded sessions.

    Example:
        >>> extractor = SegmentExtractor(model="gpt-4o")
        >>> result = extractor.extract_segments(transcript)
        >>> for episode in result.episodes:
        ...     print(f"{episode.name}: {episode.start_time_formatted} - {episode.end_time_formatted}")
        Adjust Night Shift Settings: 00:00.0 - 00:12.5
        Change Display Resolution: 00:15.3 - 00:28.1

    Attributes:
        model: LLM model identifier
        use_few_shot: Whether to include few-shot examples
        hierarchical: Whether to extract hierarchical segments
    """

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "gemini-2.0-pro",
        "gemini-2.0-flash",
    ]

    def __init__(
        self,
        model: str = "gpt-4o",
        use_few_shot: bool = True,
        hierarchical: bool = False,
        min_segment_duration: float = 2.0,
        max_segment_duration: float = 300.0,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize the segment extractor.

        Args:
            model: LLM model to use.
            use_few_shot: Include few-shot examples in prompts.
            hierarchical: Extract nested task/subtask structure.
            min_segment_duration: Minimum segment length in seconds.
            max_segment_duration: Maximum segment length in seconds.
            confidence_threshold: Minimum boundary confidence to accept.
        """
        self.model = model
        self.use_few_shot = use_few_shot
        self.hierarchical = hierarchical
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.confidence_threshold = confidence_threshold
        self._client = None

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client

        if "gpt" in self.model.lower():
            import openai
            from openadapt_ml.config import settings

            self._client = openai.OpenAI(api_key=settings.openai_api_key)
        elif "claude" in self.model.lower():
            import anthropic
            from openadapt_ml.config import settings

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        elif "gemini" in self.model.lower():
            import google.generativeai as genai
            from openadapt_ml.config import settings

            genai.configure(api_key=settings.google_api_key)
            self._client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return self._client

    def _get_system_prompt(self) -> str:
        """Return system prompt for LLM."""
        return """You are an expert at analyzing user workflows in GUI applications. Your task is to identify distinct workflow segments (episodes) within a transcript of user actions.

A workflow segment is:
- A coherent sequence of actions with a clear goal
- Self-contained (could be taught/explained as a single procedure)
- Has a clear beginning and end

Guidelines for identifying segments:
1. **Goal boundaries**: When the user's apparent goal changes, that's a new segment
2. **Application switches**: Major application changes often indicate segment boundaries
3. **Task completion**: Successful completion of a task (clicking Save, Submit, etc.) often ends a segment
4. **Natural pauses**: Significant time gaps may indicate segment boundaries
5. **Hierarchical tasks**: Large tasks may contain sub-segments (e.g., "Create document" contains "Add title", "Add body", "Save")

Avoid:
- Creating segments that are too granular (single actions)
- Creating segments that are too broad (entire session as one segment)
- Missing obvious task boundaries"""

    def _get_few_shot_examples(self) -> str:
        """Return few-shot examples for better extraction."""
        return """Here are examples of correctly segmented transcripts:

## Example 1: System Settings Workflow
**Transcript**:
```
[00:00.0] User opens System Preferences from Apple menu
[00:02.5] User clicks Display settings
[00:05.1] User navigates to Night Shift tab
[00:07.3] User enables Night Shift toggle
[00:09.8] User adjusts schedule slider to 9 PM - 7 AM
[00:12.5] User closes System Preferences
[00:15.0] User opens Notes application
[00:17.2] User creates a new note
[00:20.5] User types "Meeting notes for tomorrow"
```

**Expected segments**:
```json
{
  "segments": [
    {
      "name": "Configure Night Shift Schedule",
      "start_time": 0.0,
      "end_time": 12.5,
      "description": "Enable and configure Night Shift automatic scheduling in Display settings",
      "step_summaries": [
        "Open System Preferences",
        "Navigate to Display > Night Shift",
        "Enable Night Shift",
        "Set schedule 9 PM - 7 AM"
      ],
      "application": "System Preferences",
      "boundary_confidence": 0.95
    },
    {
      "name": "Create Meeting Notes",
      "start_time": 15.0,
      "end_time": 20.5,
      "description": "Start a new note for meeting notes in the Notes application",
      "step_summaries": [
        "Open Notes application",
        "Create new note",
        "Add title"
      ],
      "application": "Notes",
      "boundary_confidence": 0.85
    }
  ]
}
```

## Example 2: Web Browser Workflow
**Transcript**:
```
[00:00.0] User opens Chrome browser
[00:02.1] User clicks URL bar
[00:03.5] User types "github.com"
[00:05.2] User presses Enter to navigate
[00:08.4] User clicks "Sign in" button
[00:10.1] User types email address
[00:12.8] User types password
[00:15.3] User clicks "Sign in" button
[00:18.5] User clicks "New repository" button
[00:21.2] User types "my-project" as repository name
[00:24.8] User selects "Private" radio button
[00:27.1] User clicks "Create repository" button
```

**Expected segments**:
```json
{
  "segments": [
    {
      "name": "Sign In to GitHub",
      "start_time": 0.0,
      "end_time": 15.3,
      "description": "Navigate to GitHub and authenticate with email and password",
      "step_summaries": [
        "Open browser and navigate to github.com",
        "Click Sign in",
        "Enter credentials",
        "Submit login form"
      ],
      "application": "Chrome - GitHub",
      "boundary_confidence": 0.95
    },
    {
      "name": "Create Private Repository",
      "start_time": 18.5,
      "end_time": 27.1,
      "description": "Create a new private repository named my-project",
      "step_summaries": [
        "Click New repository",
        "Enter repository name",
        "Select Private visibility",
        "Create repository"
      ],
      "application": "Chrome - GitHub",
      "boundary_confidence": 0.9
    }
  ]
}
```

---

"""

    def _build_user_prompt(
        self, transcript: ActionTranscript, context: Optional[str]
    ) -> str:
        """Build user prompt for segment extraction."""
        lines = []

        if self.use_few_shot:
            lines.append(self._get_few_shot_examples())

        lines.append("Now analyze this transcript:\n")
        lines.append("## Recording Information")
        lines.append(f"- Recording ID: {transcript.recording_id}")
        lines.append(f"- Total Duration: {transcript.duration_formatted}")
        if transcript.task_description:
            lines.append(f"- Task Description: {transcript.task_description}")
        if context:
            lines.append(f"- Additional Context: {context}")

        lines.append("\n## Action Transcript")
        lines.append("```")
        lines.append(transcript.to_transcript_text())
        lines.append("```")

        lines.append("""
Identify all workflow segments in this transcript. For each segment, provide:
1. A concise name (e.g., "Adjust Night Shift Settings")
2. Start and end timestamps
3. A description of what the workflow accomplishes
4. A list of high-level steps
5. Confidence in the segment boundaries (0-1)

Respond with JSON in this format:
```json
{
  "segments": [
    {
      "name": "Segment Name",
      "start_time": 0.0,
      "end_time": 12.5,
      "start_time_formatted": "00:00.0",
      "end_time_formatted": "00:12.5",
      "description": "What this workflow accomplishes",
      "step_summaries": ["Step 1", "Step 2", "Step 3"],
      "application": "Primary application",
      "boundary_confidence": 0.9,
      "coherence_score": 0.85
    }
  ],
  "boundaries": [
    {
      "timestamp": 12.5,
      "confidence": 0.9,
      "reason": "Task completed - settings saved"
    }
  ]
}
```""")
        return "\n".join(lines)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return response text."""
        client = self._get_client()

        if "gpt" in self.model.lower():
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        elif "claude" in self.model.lower():
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        elif "gemini" in self.model.lower():
            response = client.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return response.text

        raise ValueError(f"Unknown model: {self.model}")

    def _parse_response(
        self, text: str, transcript: ActionTranscript
    ) -> tuple[list[Episode], list[EpisodeBoundary]]:
        """Parse LLM response into Episode and EpisodeBoundary objects."""
        episodes = []
        boundaries = []

        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])

                # Parse segments
                for seg_data in data.get("segments", []):
                    # Find frame indices for this segment
                    start_time = seg_data.get("start_time", 0)
                    end_time = seg_data.get("end_time", 0)
                    frame_indices = [
                        f.frame_index
                        for f in transcript.frames
                        if start_time <= f.timestamp <= end_time
                    ]

                    episode = Episode(
                        episode_id=uuid4(),
                        name=seg_data.get("name", "Unknown"),
                        start_time=start_time,
                        end_time=end_time,
                        start_time_formatted=seg_data.get(
                            "start_time_formatted",
                            f"{int(start_time // 60):02d}:{start_time % 60:04.1f}",
                        ),
                        end_time_formatted=seg_data.get(
                            "end_time_formatted",
                            f"{int(end_time // 60):02d}:{end_time % 60:04.1f}",
                        ),
                        description=seg_data.get("description", ""),
                        step_summaries=seg_data.get("step_summaries", []),
                        application=seg_data.get("application", "Unknown"),
                        boundary_confidence=seg_data.get("boundary_confidence", 0.5),
                        coherence_score=seg_data.get("coherence_score", 0.5),
                        recording_id=transcript.recording_id,
                        frame_indices=frame_indices,
                    )
                    episodes.append(episode)

                # Parse boundaries
                for bnd_data in data.get("boundaries", []):
                    boundary = EpisodeBoundary(
                        timestamp=bnd_data.get("timestamp", 0),
                        confidence=bnd_data.get("confidence", 0.5),
                        reason=bnd_data.get("reason", ""),
                    )
                    boundaries.append(boundary)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Create a single episode covering the entire transcript
            episode = Episode(
                episode_id=uuid4(),
                name="Full Recording",
                start_time=0,
                end_time=transcript.total_duration,
                start_time_formatted="00:00.0",
                end_time_formatted=transcript.duration_formatted,
                description="Complete recording (automatic segmentation failed)",
                step_summaries=[f.apparent_intent for f in transcript.frames[:5]],
                application="Unknown",
                boundary_confidence=0.1,
                coherence_score=0.1,
                recording_id=transcript.recording_id,
                frame_indices=[f.frame_index for f in transcript.frames],
            )
            episodes.append(episode)

        return episodes, boundaries

    def extract_segments(
        self,
        transcript: ActionTranscript,
        context: Optional[str] = None,
    ) -> EpisodeExtractionResult:
        """Extract workflow segments from a transcript.

        Args:
            transcript: ActionTranscript from Stage 1.
            context: Additional context (e.g., user-provided task description).

        Returns:
            EpisodeExtractionResult with identified episodes.
        """
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(transcript, context)

        response_text = self._call_llm(system_prompt, user_prompt)
        episodes, boundaries = self._parse_response(response_text, transcript)

        # Filter by duration
        filtered_episodes = [
            e
            for e in episodes
            if self.min_segment_duration <= e.duration <= self.max_segment_duration
        ]

        # Filter by confidence
        filtered_episodes = [
            e
            for e in filtered_episodes
            if e.boundary_confidence >= self.confidence_threshold
        ]

        # Calculate coverage
        total_covered = sum(e.duration for e in filtered_episodes)
        coverage = (
            total_covered / transcript.total_duration
            if transcript.total_duration > 0
            else 0
        )

        # Calculate average confidence
        avg_confidence = (
            sum(e.boundary_confidence for e in filtered_episodes)
            / len(filtered_episodes)
            if filtered_episodes
            else 0
        )

        return EpisodeExtractionResult(
            recording_id=transcript.recording_id,
            recording_name=transcript.recording_name,
            episodes=filtered_episodes,
            boundaries=boundaries,
            llm_model=self.model,
            processing_timestamp=datetime.now(),
            coverage=min(coverage, 1.0),
            avg_confidence=avg_confidence,
        )

    def identify_boundaries(
        self,
        transcript: ActionTranscript,
    ) -> list[EpisodeBoundary]:
        """Identify potential segment boundaries in a transcript.

        This is a lighter-weight method that just finds boundaries
        without full episode extraction.

        Args:
            transcript: ActionTranscript from Stage 1.

        Returns:
            List of potential boundaries with confidence scores.
        """
        result = self.extract_segments(transcript)
        return result.boundaries

    def refine_segment(
        self,
        segment: Episode,
        transcript: ActionTranscript,
    ) -> Episode:
        """Refine a segment's boundaries and description.

        Use this to improve segment quality after initial extraction.

        Args:
            segment: Segment to refine.
            transcript: Full transcript for context.

        Returns:
            Refined Episode.
        """
        # Get frames around the segment boundaries
        context_frames = [
            f
            for f in transcript.frames
            if segment.start_time - 5 <= f.timestamp <= segment.end_time + 5
        ]

        context = f"Refining segment '{segment.name}' with original boundaries {segment.start_time_formatted} - {segment.end_time_formatted}"

        # Create mini-transcript
        mini_transcript = ActionTranscript(
            recording_id=transcript.recording_id,
            recording_name=transcript.recording_name,
            frames=context_frames,
            total_duration=context_frames[-1].timestamp - context_frames[0].timestamp
            if context_frames
            else 0,
            frame_count=len(context_frames),
            vlm_model=transcript.vlm_model,
        )

        result = self.extract_segments(mini_transcript, context)
        if result.episodes:
            return result.episodes[0]
        return segment

    def merge_segments(
        self,
        segments: list[Episode],
        max_gap: float = 2.0,
    ) -> list[Episode]:
        """Merge adjacent segments that appear to be part of the same workflow.

        Args:
            segments: List of segments to potentially merge.
            max_gap: Maximum gap (seconds) between segments to consider merging.

        Returns:
            List of merged segments.
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)

        merged = [sorted_segments[0]]
        for segment in sorted_segments[1:]:
            last = merged[-1]
            gap = segment.start_time - last.end_time

            # Check if should merge
            if gap <= max_gap and segment.application == last.application:
                # Merge segments
                merged_segment = Episode(
                    episode_id=uuid4(),
                    name=f"{last.name} + {segment.name}",
                    start_time=last.start_time,
                    end_time=segment.end_time,
                    start_time_formatted=last.start_time_formatted,
                    end_time_formatted=segment.end_time_formatted,
                    description=f"{last.description}. Then, {segment.description}",
                    step_summaries=last.step_summaries + segment.step_summaries,
                    application=last.application,
                    boundary_confidence=min(
                        last.boundary_confidence, segment.boundary_confidence
                    ),
                    coherence_score=(last.coherence_score + segment.coherence_score)
                    / 2,
                    recording_id=last.recording_id,
                    frame_indices=last.frame_indices + segment.frame_indices,
                )
                merged[-1] = merged_segment
            else:
                merged.append(segment)

        return merged

    def adjust_boundary(
        self,
        segment: Episode,
        new_start: Optional[float] = None,
        new_end: Optional[float] = None,
        transcript: Optional[ActionTranscript] = None,
    ) -> Episode:
        """Manually adjust segment boundaries.

        For human-in-the-loop refinement.

        Args:
            segment: Segment to adjust.
            new_start: New start time (or None to keep existing).
            new_end: New end time (or None to keep existing).
            transcript: Transcript to re-extract step info from new boundaries.

        Returns:
            Adjusted Episode.
        """
        start_time = new_start if new_start is not None else segment.start_time
        end_time = new_end if new_end is not None else segment.end_time

        # Update frame indices if transcript provided
        frame_indices = segment.frame_indices
        if transcript:
            frame_indices = [
                f.frame_index
                for f in transcript.frames
                if start_time <= f.timestamp <= end_time
            ]

        return Episode(
            episode_id=segment.episode_id,
            name=segment.name,
            start_time=start_time,
            end_time=end_time,
            start_time_formatted=f"{int(start_time // 60):02d}:{start_time % 60:04.1f}",
            end_time_formatted=f"{int(end_time // 60):02d}:{end_time % 60:04.1f}",
            description=segment.description,
            steps=segment.steps,
            step_summaries=segment.step_summaries,
            application=segment.application,
            prerequisites=segment.prerequisites,
            outcomes=segment.outcomes,
            parent_episode_id=segment.parent_episode_id,
            child_episode_ids=segment.child_episode_ids,
            boundary_confidence=segment.boundary_confidence
            * 0.9,  # Reduce confidence for manual adjustment
            coherence_score=segment.coherence_score,
            recording_id=segment.recording_id,
            frame_indices=frame_indices,
        )

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported LLM models."""
        return self.SUPPORTED_MODELS
