# openadapt-ml Recording Implementation Plan

This document analyzes the original `openadapt/record.py` and proposes a minimal reimplementation for `openadapt-ml`.

## Original openadapt.record Architecture

The original `openadapt/record.py` (1654 lines) is a comprehensive recording system.

### Event Types
- `screen` - Periodic screenshots
- `action` - Mouse/keyboard events
- `window` - Active window metadata
- `browser` - Browser DOM events (via WebSocket)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Process                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Keyboard    │  │ Mouse       │  │ Screen      │              │
│  │ Listener    │  │ Listener    │  │ Reader      │              │
│  │ (pynput)    │  │ (pynput)    │  │ (mss)       │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│                    ┌───────────┐                                 │
│                    │ event_q   │                                 │
│                    └─────┬─────┘                                 │
│                          ▼                                       │
│                ┌─────────────────┐                               │
│                │ process_events  │                               │
│                │ (correlates     │                               │
│                │  action+screen) │                               │
│                └────────┬────────┘                               │
│                         │                                        │
│    ┌────────────────────┼────────────────────┐                   │
│    ▼                    ▼                    ▼                   │
│ ┌──────────┐      ┌──────────┐         ┌──────────┐             │
│ │ action_  │      │ screen_  │         │ window_  │             │
│ │ write_q  │      │ write_q  │         │ write_q  │             │
│ └────┬─────┘      └────┬─────┘         └────┬─────┘             │
└──────┼─────────────────┼───────────────────┼────────────────────┘
       │                 │                   │
       ▼                 ▼                   ▼
  ┌─────────┐       ┌─────────┐        ┌─────────┐
  │ Writer  │       │ Writer  │        │ Writer  │
  │ Process │       │ Process │        │ Process │
  │ (DB)    │       │ (DB)    │        │ (DB)    │
  └─────────┘       └─────────┘        └─────────┘
```

### Key Data Models (from openadapt/models.py)

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `Recording` | Session metadata | monitor_width, monitor_height, platform, task_description |
| `ActionEvent` | Mouse/keyboard action | timestamp, name, mouse_x, mouse_y, key_char, element_state |
| `Screenshot` | Screen capture | timestamp, png_data |
| `WindowEvent` | Active window | title, left, top, width, height, state |

### Key Functions in record.py

```python
# Event handlers
on_move(event_q, x, y)           # Mouse move (filtered in openadapt-ml)
on_click(event_q, x, y, button)  # Mouse click → CLICK action
on_scroll(event_q, x, y, dx, dy) # Scroll (not needed initially)
on_press(event_q, key)           # Key press → TYPE action
on_release(event_q, key)         # Key release

# Event processing
process_events(event_q, ...)     # Correlates actions with screenshots
read_screen_events(event_q, ...) # Captures screenshots in loop
read_keyboard_events(...)        # Starts pynput keyboard listener
read_mouse_events(...)           # Starts pynput mouse listener

# Database writes
write_action_event(db, recording, event)
write_screen_event(db, recording, event)
write_window_event(db, recording, event)
```

## Comparison: Original vs Minimal

| Feature | Original openadapt | Minimal openadapt-ml |
|---------|-------------------|----------------------|
| Lines of code | ~1654 | ~200 (target) |
| Dependencies | pynput, mss, av, whisper, sqlalchemy | pynput, mss only |
| Storage | SQLite database | JSON + PNG files |
| Screenshot format | PNG in DB or video | PNG files on disk |
| Action types | move, click, scroll, press, release | click, type, done |
| Window tracking | Full accessibility API | None (initially) |
| Browser events | WebSocket DOM capture | None |
| Audio | Whisper transcription | None |
| Video | AV container encoding | None |
| Multi-process | Yes (writers in separate processes) | No (single thread) |

## Proposed Minimal Implementation

### Target: `openadapt_ml/record.py` (~200 lines)

```python
"""Minimal recording for openadapt-ml training data generation."""

from dataclasses import dataclass
from typing import List, Optional
import threading
import time
import json
from pathlib import Path

from pynput import keyboard, mouse
import mss
from PIL import Image

from openadapt_ml.common import Action, Observation, Step, Episode, Session


@dataclass
class RecordedEvent:
    """Raw recorded event before processing."""
    timestamp: float
    event_type: str  # "click", "key_press", "key_release"
    data: dict


class MinimalRecorder:
    """Record user interactions for training data generation."""

    def __init__(self, output_dir: str, screenshot_interval: float = 0.1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_interval = screenshot_interval

        self.events: List[RecordedEvent] = []
        self.screenshots: List[tuple[float, str]] = []  # (timestamp, path)
        self.running = False
        self.current_text = ""  # Accumulate typed text

    def start(self, goal: str) -> None:
        """Start recording."""
        self.goal = goal
        self.running = True
        self.start_time = time.time()

        # Start screenshot thread
        self.screenshot_thread = threading.Thread(target=self._capture_screenshots)
        self.screenshot_thread.start()

        # Start input listeners
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.mouse_listener.start()
        self.keyboard_listener.start()

    def stop(self) -> Episode:
        """Stop recording and return Episode."""
        self.running = False
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        self.screenshot_thread.join()

        return self._build_episode()

    def _capture_screenshots(self) -> None:
        """Continuously capture screenshots."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            while self.running:
                timestamp = time.time()
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                # Save screenshot
                filename = f"screenshot_{timestamp:.3f}.png"
                path = self.output_dir / filename
                img.save(path)
                self.screenshots.append((timestamp, str(path)))

                time.sleep(self.screenshot_interval)

    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        """Handle mouse click."""
        if not pressed:  # Only record on click release
            return

        # Flush any accumulated text as TYPE action first
        if self.current_text:
            self.events.append(RecordedEvent(
                timestamp=time.time(),
                event_type="type",
                data={"text": self.current_text}
            ))
            self.current_text = ""

        # Record click
        self.events.append(RecordedEvent(
            timestamp=time.time(),
            event_type="click",
            data={"x": x, "y": y, "button": button.name}
        ))

    def _on_press(self, key) -> None:
        """Handle key press - accumulate for TYPE actions."""
        # Check for stop sequence (Ctrl+Q)
        # ... (handle stop)

        try:
            char = key.char
            if char:
                self.current_text += char
        except AttributeError:
            # Special key
            if key == keyboard.Key.space:
                self.current_text += " "
            elif key == keyboard.Key.enter:
                self.current_text += "\n"
            # ... handle other special keys

    def _on_release(self, key) -> None:
        """Handle key release."""
        pass

    def _build_episode(self) -> Episode:
        """Convert raw events to Episode format."""
        steps = []

        for event in self.events:
            # Find screenshot closest to (but before) this event
            screenshot_path = self._find_closest_screenshot(event.timestamp)

            observation = Observation(image_path=screenshot_path)

            if event.event_type == "click":
                # Normalize coordinates
                x_norm = event.data["x"] / self.monitor_width
                y_norm = event.data["y"] / self.monitor_height
                action = Action(type="click", raw={"x": x_norm, "y": y_norm})
            elif event.event_type == "type":
                action = Action(type="type", raw={"text": event.data["text"]})
            else:
                continue

            steps.append(Step(observation=observation, action=action))

        # Add DONE step
        if self.screenshots:
            final_screenshot = self.screenshots[-1][1]
            steps.append(Step(
                observation=Observation(image_path=final_screenshot),
                action=Action(type="done", raw={})
            ))

        return Episode(goal=self.goal, steps=steps)

    def _find_closest_screenshot(self, timestamp: float) -> str:
        """Find screenshot taken closest to (but before) timestamp."""
        closest = None
        for ts, path in self.screenshots:
            if ts <= timestamp:
                closest = path
            else:
                break
        return closest or self.screenshots[0][1] if self.screenshots else ""


def record_episode(goal: str, output_dir: str) -> Episode:
    """Convenience function to record an episode.

    Usage:
        episode = record_episode("Login to the website", "recordings/session1")
    """
    recorder = MinimalRecorder(output_dir)
    print(f"Recording started. Goal: {goal}")
    print("Press Ctrl+Q to stop recording.")

    recorder.start(goal)

    # Wait for stop signal (Ctrl+Q handled in keyboard listener)
    while recorder.running:
        time.sleep(0.1)

    episode = recorder.stop()

    # Save episode
    episode_path = Path(output_dir) / "episode.json"
    with open(episode_path, "w") as f:
        json.dump(episode.to_dict(), f, indent=2)

    print(f"Recording saved to {episode_path}")
    return episode
```

### Usage Example

```python
from openadapt_ml.record import record_episode

# Record a demonstration
episode = record_episode(
    goal="Login to example.com with username 'admin'",
    output_dir="recordings/login_demo"
)

# Episode is automatically saved to recordings/login_demo/episode.json
# Screenshots are saved to recordings/login_demo/screenshot_*.png
```

## Implementation Phases

### Phase 1: Basic Recorder
- [x] Define target API (above)
- [ ] Implement `MinimalRecorder` class
- [ ] Screenshot capture thread with mss
- [ ] Mouse click listener with pynput
- [ ] Keyboard listener with text aggregation
- [ ] JSON serialization of Episode

### Phase 2: Element Detection Integration
- [ ] Integrate OmniParser for UI element detection
- [ ] Generate bounding boxes for interactive elements
- [ ] Convert click coordinates to SoM element indices
- [ ] Save SoM-annotated screenshots

### Phase 3: Training Pipeline Integration
- [ ] Load recorded episodes into training format
- [ ] Generate SFT samples with SoM prompts
- [ ] Fine-tune on real (non-synthetic) workflows
- [ ] Evaluate on held-out test episodes

## Dependencies

Minimal dependencies for Phase 1:
```toml
[project.optional-dependencies]
record = [
    "pynput>=1.7.6",  # Already used by openadapt
    "mss>=9.0.0",     # Already used by openadapt
    "pillow>=10.0.0", # Already a dependency
]
```

## Notes

### Coordinate Normalization

For fine-tuned models:
- Train on fixed resolution (e.g., 800×600)
- Normalize coordinates to 0-1 range at record time
- Resize screenshots to training resolution at inference time

For SoM mode:
- Coordinates not needed (element indices instead)
- OmniParser/Gemini generates bounding boxes
- Model outputs `CLICK(element_id=N)` instead of coordinates

### Stop Sequence

Original openadapt uses configurable stop sequences. For minimal implementation:
- Default: Ctrl+Q to stop recording
- Alternative: GUI button (future)

### Limitations of Minimal Implementation

1. No window context - can't crop to active window
2. No accessibility data - can't get element names
3. No browser DOM - can't get web element structure
4. No video - larger storage footprint
5. Single-threaded - may miss events under heavy load

These limitations are acceptable for initial training data generation. Features can be added incrementally as needed.
