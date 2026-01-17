"""Frame-level description using Vision-Language Models.

This module processes recording frames with their associated actions
to generate semantic descriptions of user behavior (Stage 1 of pipeline).
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from openadapt_ml.segmentation.schemas import (
    ActionTranscript,
    ActionType,
    FrameDescription,
)

logger = logging.getLogger(__name__)


class VLMBackend(ABC):
    """Abstract base class for VLM backend implementations."""

    @abstractmethod
    def describe_frame(
        self,
        image: Image.Image,
        action_context: dict,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """Generate description for a single frame."""
        pass

    @abstractmethod
    def describe_batch(
        self,
        images: list[Image.Image],
        action_contexts: list[dict],
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict]:
        """Generate descriptions for multiple frames (more efficient)."""
        pass


class GeminiBackend(VLMBackend):
    """Google Gemini VLM backend."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            from openadapt_ml.config import settings

            api_key = self._api_key or settings.google_api_key
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def describe_frame(
        self,
        image: Image.Image,
        action_context: dict,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        client = self._get_client()
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = client.generate_content([full_prompt, image])
        return self._parse_response(response.text)

    def describe_batch(
        self,
        images: list[Image.Image],
        action_contexts: list[dict],
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict]:
        # Gemini can handle multiple images in one call
        client = self._get_client()
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        content = [full_prompt] + images
        response = client.generate_content(content)
        return self._parse_batch_response(response.text, len(images))

    def _parse_response(self, text: str) -> dict:
        """Parse JSON from response text."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {"apparent_intent": text, "confidence": 0.5}

    def _parse_batch_response(self, text: str, count: int) -> list[dict]:
        """Parse batch JSON response."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                if "frames" in data:
                    return data["frames"]
        except json.JSONDecodeError:
            pass
        return [
            {"apparent_intent": f"Frame {i}", "confidence": 0.5} for i in range(count)
        ]


class ClaudeBackend(VLMBackend):
    """Anthropic Claude VLM backend."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            from openadapt_ml.config import settings

            api_key = self._api_key or settings.anthropic_api_key
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _encode_image(self, image: Image.Image) -> dict:
        """Encode image for Claude API."""
        import base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        }

    def describe_frame(
        self,
        image: Image.Image,
        action_context: dict,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        self._encode_image(image),
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        return self._parse_response(response.content[0].text)

    def describe_batch(
        self,
        images: list[Image.Image],
        action_contexts: list[dict],
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict]:
        client = self._get_client()
        content = []
        for img in images:
            content.append(self._encode_image(img))
        content.append({"type": "text", "text": user_prompt})

        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return self._parse_batch_response(response.content[0].text, len(images))

    def _parse_response(self, text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {"apparent_intent": text, "confidence": 0.5}

    def _parse_batch_response(self, text: str, count: int) -> list[dict]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                if "frames" in data:
                    return data["frames"]
        except json.JSONDecodeError:
            pass
        return [
            {"apparent_intent": f"Frame {i}", "confidence": 0.5} for i in range(count)
        ]


class OpenAIBackend(VLMBackend):
    """OpenAI GPT-4V backend."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            from openadapt_ml.config import settings

            api_key = self._api_key or settings.openai_api_key
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def _encode_image(self, image: Image.Image) -> dict:
        """Encode image for OpenAI API."""
        import base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }

    def describe_frame(
        self,
        image: Image.Image,
        action_context: dict,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        self._encode_image(image),
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        )
        return self._parse_response(response.choices[0].message.content)

    def describe_batch(
        self,
        images: list[Image.Image],
        action_contexts: list[dict],
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict]:
        client = self._get_client()
        content = [self._encode_image(img) for img in images]
        content.append({"type": "text", "text": user_prompt})

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return self._parse_batch_response(
            response.choices[0].message.content, len(images)
        )

    def _parse_response(self, text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {"apparent_intent": text, "confidence": 0.5}

    def _parse_batch_response(self, text: str, count: int) -> list[dict]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                if "frames" in data:
                    return data["frames"]
        except json.JSONDecodeError:
            pass
        return [
            {"apparent_intent": f"Frame {i}", "confidence": 0.5} for i in range(count)
        ]


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.m"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"


def _get_action_type(action_name: str) -> ActionType:
    """Convert action name to ActionType enum."""
    name_lower = action_name.lower()
    if "double" in name_lower:
        return ActionType.DOUBLE_CLICK
    elif "right" in name_lower:
        return ActionType.RIGHT_CLICK
    elif "click" in name_lower:
        return ActionType.CLICK
    elif "type" in name_lower or "key" in name_lower:
        return ActionType.TYPE
    elif "scroll" in name_lower:
        return ActionType.SCROLL
    elif "drag" in name_lower:
        return ActionType.DRAG
    elif "hotkey" in name_lower:
        return ActionType.HOTKEY
    elif "move" in name_lower:
        return ActionType.MOVE
    return ActionType.CLICK


class FrameDescriber:
    """Generates semantic descriptions of recording frames using VLMs.

    This class implements Stage 1 of the segmentation pipeline, converting
    raw screenshots and action data into human-readable descriptions.

    Example:
        >>> describer = FrameDescriber(model="gemini-2.0-flash")
        >>> transcript = describer.describe_recording(recording)
        >>> print(transcript.to_transcript_text())
        [00:00.0] User opens System Preferences from Apple menu
        [00:02.5] User clicks Display settings icon
        ...

    Attributes:
        model: VLM model identifier
        batch_size: Number of frames to process per API call
        cache_enabled: Whether to cache descriptions
    """

    SUPPORTED_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-pro",
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        batch_size: int = 10,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
        backend: Optional[VLMBackend] = None,
    ) -> None:
        """Initialize the frame describer.

        Args:
            model: VLM model to use.
            batch_size: Number of frames per API call.
            cache_enabled: Cache descriptions to avoid reprocessing.
            cache_dir: Directory for cached descriptions.
            backend: Custom VLM backend (for testing or custom models).
        """
        self.model = model
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.cache_dir = (
            cache_dir or Path.home() / ".openadapt" / "cache" / "descriptions"
        )
        self._backend = backend or self._create_backend(model)

    def _create_backend(self, model: str) -> VLMBackend:
        """Create appropriate backend for the specified model."""
        if "gemini" in model.lower():
            return GeminiBackend(model=model)
        elif "claude" in model.lower():
            return ClaudeBackend(model=model)
        elif "gpt" in model.lower():
            return OpenAIBackend(model=model)
        else:
            raise ValueError(
                f"Unknown model: {model}. Supported: {self.SUPPORTED_MODELS}"
            )

    def _get_system_prompt(self) -> str:
        """Return system prompt for VLM."""
        return """You are an expert at analyzing GUI screenshots and user actions. Your task is to describe what the user is doing in each screenshot, focusing on:

1. **Context**: What application is open? What screen/view is visible?
2. **Action**: What specific action did the user take? (click, type, scroll, etc.)
3. **Intent**: What is the user trying to accomplish with this action?

Provide descriptions that would help someone understand and reproduce the workflow.

Guidelines:
- Be specific about UI elements (e.g., "Night Shift toggle" not "a button")
- Include relevant text visible on screen when it clarifies intent
- Note any state changes visible in the screenshot
- Keep descriptions concise but complete (1-2 sentences typically)"""

    def _get_user_prompt(self, frames_data: list[dict]) -> str:
        """Build user prompt for batch of frames."""
        lines = ["Analyze the following screenshot(s) and action(s):\n"]

        for i, frame in enumerate(frames_data, 1):
            lines.append(f"## Frame {i} ({frame['timestamp_formatted']})")
            lines.append("**Action performed**:")
            lines.append(f"- Type: {frame['action']['name']}")
            if frame["action"].get("mouse_x") is not None:
                lines.append(
                    f"- Location: ({int(frame['action']['mouse_x'])}, {int(frame['action']['mouse_y'])})"
                )
            if frame["action"].get("text"):
                lines.append(f'- Text typed: "{frame["action"]["text"]}"')
            lines.append("")

        lines.append("""For each frame, provide a JSON response with this structure:
```json
{
  "frames": [
    {
      "frame_index": 1,
      "visible_application": "Application name",
      "visible_elements": ["element1", "element2"],
      "screen_context": "Brief description of the overall screen state",
      "action_target": "Specific UI element targeted",
      "apparent_intent": "What the user is trying to accomplish",
      "confidence": 0.95
    }
  ]
}
```""")
        return "\n".join(lines)

    def _cache_key(self, image: Image.Image, action: dict) -> str:
        """Generate cache key for a frame."""
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_hash = hashlib.md5(buffer.getvalue()).hexdigest()[:12]
        action_str = json.dumps(action, sort_keys=True)
        action_hash = hashlib.md5(action_str.encode()).hexdigest()[:8]
        return f"{img_hash}_{action_hash}"

    def _load_cached(self, cache_key: str) -> Optional[dict]:
        """Load cached description."""
        if not self.cache_enabled:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass
        return None

    def _save_cached(self, cache_key: str, description: dict) -> None:
        """Save description to cache."""
        if not self.cache_enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            cache_file.write_text(json.dumps(description))
        except Exception as e:
            logger.warning(f"Failed to cache description: {e}")

    def describe_recording(
        self,
        recording_path: Union[str, Path],
        progress_callback: Optional[callable] = None,
    ) -> ActionTranscript:
        """Generate descriptions for all frames in a recording.

        Args:
            recording_path: Path to recording directory or file.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            ActionTranscript with descriptions for all frames.
        """
        recording_path = Path(recording_path)
        if not recording_path.exists():
            raise FileNotFoundError(f"Recording not found: {recording_path}")

        # Load recording data
        images, action_events = self._load_recording(recording_path)
        recording_id = recording_path.name
        recording_name = recording_path.stem

        # Process in batches
        frame_descriptions = []
        total_frames = len(images)

        for batch_start in range(0, total_frames, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_frames)
            batch_images = images[batch_start:batch_end]
            batch_actions = action_events[batch_start:batch_end]

            # Check cache first
            batch_results = []
            uncached_indices = []

            for i, (img, action) in enumerate(zip(batch_images, batch_actions)):
                cache_key = self._cache_key(img, action)
                cached = self._load_cached(cache_key)
                if cached:
                    batch_results.append((i, cached))
                else:
                    uncached_indices.append(i)

            # Process uncached frames
            if uncached_indices:
                uncached_images = [batch_images[i] for i in uncached_indices]
                uncached_actions = [batch_actions[i] for i in uncached_indices]

                frames_data = [
                    {
                        "timestamp_formatted": _format_timestamp(a.get("timestamp", 0)),
                        "action": a,
                    }
                    for a in uncached_actions
                ]

                descriptions = self._backend.describe_batch(
                    uncached_images,
                    uncached_actions,
                    self._get_system_prompt(),
                    self._get_user_prompt(frames_data),
                )

                for i, desc in zip(uncached_indices, descriptions):
                    batch_results.append((i, desc))
                    cache_key = self._cache_key(batch_images[i], batch_actions[i])
                    self._save_cached(cache_key, desc)

            # Sort by index and create FrameDescriptions
            batch_results.sort(key=lambda x: x[0])
            for i, (idx, desc) in enumerate(batch_results):
                frame_idx = batch_start + idx
                action = batch_actions[idx]
                timestamp = action.get("timestamp", 0)

                frame_desc = FrameDescription(
                    timestamp=timestamp,
                    formatted_time=_format_timestamp(timestamp),
                    visible_application=desc.get("visible_application", "Unknown"),
                    visible_elements=desc.get("visible_elements", []),
                    screen_context=desc.get("screen_context", ""),
                    action_type=_get_action_type(action.get("name", "click")),
                    action_target=desc.get("action_target"),
                    action_value=action.get("text"),
                    apparent_intent=desc.get("apparent_intent", "Unknown action"),
                    confidence=desc.get("confidence", 0.5),
                    frame_index=frame_idx,
                    vlm_model=self.model,
                )
                frame_descriptions.append(frame_desc)

            if progress_callback:
                progress_callback(batch_end, total_frames)

        # Calculate total duration
        total_duration = 0
        if frame_descriptions:
            total_duration = max(f.timestamp for f in frame_descriptions)

        return ActionTranscript(
            recording_id=recording_id,
            recording_name=recording_name,
            frames=frame_descriptions,
            total_duration=total_duration,
            frame_count=len(frame_descriptions),
            vlm_model=self.model,
            processing_timestamp=datetime.now(),
        )

    def _load_recording(
        self, recording_path: Path
    ) -> tuple[list[Image.Image], list[dict]]:
        """Load recording data from various formats."""
        # Try to load from openadapt-capture SQLite format
        if (recording_path / "capture.db").exists():
            try:
                from openadapt_ml.segmentation.adapters import CaptureAdapter

                adapter = CaptureAdapter()
                return adapter.load_recording(recording_path)
            except Exception as e:
                logger.warning(f"Failed to load via CaptureAdapter: {e}")
                # Fall through to other formats

        # Try to load from openadapt-capture format (events.json)
        metadata_file = recording_path / "metadata.json"
        if metadata_file.exists():
            return self._load_capture_format(recording_path)

        # Try loading from a single JSON file
        if recording_path.suffix == ".json":
            return self._load_json_format(recording_path)

        # Try loading from directory with screenshots
        return self._load_directory_format(recording_path)

    def _load_capture_format(
        self, recording_path: Path
    ) -> tuple[list[Image.Image], list[dict]]:
        """Load from openadapt-capture format."""
        _metadata = json.loads((recording_path / "metadata.json").read_text())
        # Note: _metadata contains recording_id, goal, timestamps but we load
        # these at the transcript level, not per-frame
        images = []
        actions = []

        screenshots_dir = recording_path / "screenshots"
        events_file = recording_path / "events.json"

        if events_file.exists():
            events = json.loads(events_file.read_text())
            for event in events:
                screenshot_path = screenshots_dir / f"{event['frame_index']:06d}.png"
                if screenshot_path.exists():
                    images.append(Image.open(screenshot_path))
                    actions.append(event)

        return images, actions

    def _load_json_format(
        self, json_path: Path
    ) -> tuple[list[Image.Image], list[dict]]:
        """Load from JSON file with base64 screenshots."""
        import base64
        import io

        data = json.loads(json_path.read_text())
        images = []
        actions = []

        for frame in data.get("frames", []):
            if "screenshot_base64" in frame:
                img_data = base64.b64decode(frame["screenshot_base64"])
                images.append(Image.open(io.BytesIO(img_data)))
                actions.append(frame.get("action", {}))

        return images, actions

    def _load_directory_format(
        self, dir_path: Path
    ) -> tuple[list[Image.Image], list[dict]]:
        """Load from directory with numbered screenshots."""
        images = []
        actions = []

        # Find all PNG files
        png_files = sorted(dir_path.glob("*.png"))
        for i, png_file in enumerate(png_files):
            images.append(Image.open(png_file))
            # Create synthetic action event
            actions.append(
                {
                    "name": "unknown",
                    "timestamp": i * 1.0,  # Assume 1 second between frames
                    "frame_index": i,
                }
            )

        return images, actions

    def describe_frame(
        self,
        image: Image.Image,
        action_event: dict,
        previous_context: Optional[str] = None,
    ) -> FrameDescription:
        """Generate description for a single frame."""
        frames_data = [
            {
                "timestamp_formatted": _format_timestamp(
                    action_event.get("timestamp", 0)
                ),
                "action": action_event,
            }
        ]

        descriptions = self._backend.describe_batch(
            [image],
            [action_event],
            self._get_system_prompt(),
            self._get_user_prompt(frames_data),
        )

        desc = descriptions[0] if descriptions else {}
        timestamp = action_event.get("timestamp", 0)

        return FrameDescription(
            timestamp=timestamp,
            formatted_time=_format_timestamp(timestamp),
            visible_application=desc.get("visible_application", "Unknown"),
            visible_elements=desc.get("visible_elements", []),
            screen_context=desc.get("screen_context", ""),
            action_type=_get_action_type(action_event.get("name", "click")),
            action_target=desc.get("action_target"),
            action_value=action_event.get("text"),
            apparent_intent=desc.get("apparent_intent", "Unknown action"),
            confidence=desc.get("confidence", 0.5),
            frame_index=action_event.get("frame_index", 0),
            vlm_model=self.model,
        )

    def clear_cache(self, recording_id: Optional[str] = None) -> int:
        """Clear cached descriptions.

        Args:
            recording_id: If specified, only clear cache for this recording.

        Returns:
            Number of cached items cleared.
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if recording_id is None or recording_id in cache_file.name:
                cache_file.unlink()
                count += 1
        return count

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported VLM models."""
        return self.SUPPORTED_MODELS
