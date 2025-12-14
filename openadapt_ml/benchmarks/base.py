"""Base classes for benchmark integration.

This module provides the core abstractions for integrating GUI agent benchmarks
into openadapt-ml. It supports both interactive environments (WAA, OSWorld) and
static trajectory datasets (Mind2Web).

Example:
    from openadapt_ml.benchmarks import WAAAdapter, evaluate_agent_on_benchmark

    adapter = WAAAdapter(waa_repo_path="/path/to/WAA")
    results = evaluate_agent_on_benchmark(agent, adapter, max_steps=50)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    pass


@dataclass
class BenchmarkTask:
    """Canonical task representation.

    Attributes:
        task_id: Unique identifier for the task.
        instruction: Natural language task instruction.
        domain: Task domain ("web", "desktop", "mobile").
        initial_state_ref: Reference to initial state (VM snapshot, URL, etc.).
        time_limit_steps: Maximum steps allowed for the task.
        raw_config: Original benchmark config (lossless preservation).
        evaluation_spec: Benchmark-native evaluation specification.
    """

    task_id: str
    instruction: str
    domain: str  # "web", "desktop", "mobile"

    # Environment setup
    initial_state_ref: str | None = None  # VM snapshot, storage_state, start URL
    time_limit_steps: int | None = None

    # Preserve original config losslessly
    raw_config: dict[str, Any] = field(default_factory=dict)

    # Evaluation spec (benchmark-native)
    evaluation_spec: dict[str, Any] | None = None


@dataclass
class BenchmarkObservation:
    """Canonical observation at each step.

    Supports multiple observation modalities:
    - Visual: screenshots with viewport info
    - Structured UI: accessibility tree (UIA/AXTree/DOM)
    - Context: URL, window title, focused element

    Attributes:
        screenshot: PNG image bytes.
        screenshot_path: Path to saved screenshot.
        viewport: (width, height) of the viewport.
        accessibility_tree: Platform-specific UI tree (UIA/AXTree/DOM).
        dom_html: Raw HTML for web tasks.
        url: Current URL for web tasks.
        window_title: Active window title for desktop tasks.
        focused_element: Currently focused UI element.
        raw_observation: Original benchmark observation (lossless).
    """

    # Visual
    screenshot: bytes | None = None  # PNG image bytes
    screenshot_path: str | None = None
    viewport: tuple[int, int] | None = None  # (width, height)

    # Structured UI (format varies by platform)
    accessibility_tree: dict | None = None  # UIA (Windows), AXTree (macOS), DOM (web)
    dom_html: str | None = None  # Raw HTML for web

    # Context
    url: str | None = None  # For web tasks
    window_title: str | None = None  # For desktop tasks
    app_name: str | None = None  # Active application
    focused_element: dict | None = None  # {node_id, bbox, text}

    # Raw benchmark-specific data (lossless)
    raw_observation: dict[str, Any] | None = None


@dataclass
class BenchmarkAction:
    """Canonical action representation.

    Supports multiple action types with both coordinate-based and element-based
    grounding. The "grounding-first" approach stores both when available.

    Attributes:
        type: Action type ("click", "type", "scroll", "key", "drag", "answer", "done").
        x: X coordinate (normalized [0,1] or pixels).
        y: Y coordinate (normalized [0,1] or pixels).
        target_node_id: Element ID from accessibility tree.
        target_bbox: Element bounding box.
        target_role: Element role (button, textfield, etc.).
        target_name: Element accessible name.
        text: Text to type (for "type" action).
        key: Single key (for "key" action, e.g., "Enter", "Tab").
        modifiers: Key modifiers (["ctrl", "shift", "alt"]).
        scroll_direction: Scroll direction ("up", "down", "left", "right").
        scroll_amount: Scroll amount (pixels or normalized).
        end_x: Drag end X coordinate.
        end_y: Drag end Y coordinate.
        answer: Answer string (for benchmarks that score by answer).
        raw_action: Original benchmark action (lossless).
    """

    type: str  # "click", "type", "scroll", "key", "drag", "answer", "done"

    # Pointer actions - coordinates
    x: float | None = None  # Normalized [0,1] or pixel
    y: float | None = None

    # Element grounding (when available)
    target_node_id: str | None = None  # DOM/AX/UIA node ID
    target_bbox: tuple[float, float, float, float] | None = None
    target_role: str | None = None  # "button", "textfield", etc.
    target_name: str | None = None  # Accessible name

    # Keyboard actions
    text: str | None = None  # For "type" action - text to type
    key: str | None = None  # For "key" action - single key
    modifiers: list[str] | None = None  # ["ctrl", "shift", "alt"]

    # Scroll actions
    scroll_direction: str | None = None  # "up", "down", "left", "right"
    scroll_amount: float | None = None  # Pixels or normalized

    # Drag actions
    end_x: float | None = None
    end_y: float | None = None

    # Answer action (some benchmarks score by final answer)
    answer: str | None = None

    # Raw benchmark-specific format (lossless)
    raw_action: dict[str, Any] | None = None


@dataclass
class BenchmarkResult:
    """Result of a single task evaluation.

    Attributes:
        task_id: ID of the evaluated task.
        success: Whether the task was completed successfully.
        score: Score between 0.0 and 1.0.
        steps: List of (observation, action) pairs from the trajectory.
        num_steps: Number of steps taken.
        error: Error message if task failed due to error.
        reason: Explanation of success/failure.
        total_time_seconds: Total time taken for the task.
    """

    task_id: str
    success: bool
    score: float  # 0.0 to 1.0

    # Trajectory
    steps: list[tuple[BenchmarkObservation, BenchmarkAction]] = field(
        default_factory=list
    )
    num_steps: int = 0

    # Diagnostics
    error: str | None = None
    reason: str | None = None  # Why success/fail

    # Timing
    total_time_seconds: float = 0.0


@dataclass
class UIElement:
    """Normalized UI element for cross-platform use.

    Provides a common representation for UI elements across platforms
    (Windows UIA, macOS AXTree, web DOM).

    Attributes:
        node_id: Unique identifier for the element.
        role: Element role (button, textfield, link, etc.).
        name: Accessible name/label.
        bbox: Bounding box (normalized [0,1] or pixels).
        text: Text content.
        value: Current value (for inputs).
        children: Child elements.
        attributes: Additional platform-specific attributes.
    """

    node_id: str
    role: str  # "button", "textfield", "link", etc.
    name: str | None = None  # Accessible name/label
    bbox: tuple[float, float, float, float] | None = None  # (x1, y1, x2, y2)
    text: str | None = None  # Text content
    value: str | None = None  # Current value (for inputs)
    children: list[UIElement] | None = None
    attributes: dict[str, Any] | None = None  # Platform-specific


class BenchmarkAdapter(ABC):
    """Abstract interface for benchmark integration.

    Subclasses implement this interface to integrate specific benchmarks
    (WAA, OSWorld, WebArena, etc.) with openadapt-ml.

    Two types of adapters:
    - Interactive: Run environment, step through tasks (WAA, OSWorld)
    - Static: Load trajectories for offline training/eval (Mind2Web)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name (e.g., 'waa', 'osworld', 'webarena')."""
        pass

    @property
    @abstractmethod
    def benchmark_type(self) -> str:
        """Benchmark type: 'interactive' or 'static'."""
        pass

    @property
    def supports_parallel(self) -> bool:
        """Whether the adapter supports parallel task execution."""
        return False

    @abstractmethod
    def list_tasks(self, domain: str | None = None) -> list[BenchmarkTask]:
        """List available tasks, optionally filtered by domain.

        Args:
            domain: Optional domain filter (e.g., "browser", "office").

        Returns:
            List of BenchmarkTask objects.
        """
        pass

    @abstractmethod
    def load_task(self, task_id: str) -> BenchmarkTask:
        """Load a specific task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            BenchmarkTask object.

        Raises:
            KeyError: If task_id not found.
        """
        pass

    @abstractmethod
    def reset(self, task: BenchmarkTask) -> BenchmarkObservation:
        """Reset environment to task's initial state.

        Args:
            task: Task to initialize.

        Returns:
            Initial observation.
        """
        pass

    @abstractmethod
    def step(
        self, action: BenchmarkAction
    ) -> tuple[BenchmarkObservation, bool, dict[str, Any]]:
        """Execute action and return new observation.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (observation, done, info).
        """
        pass

    @abstractmethod
    def evaluate(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run benchmark's native evaluation on current state.

        Args:
            task: Task to evaluate.

        Returns:
            BenchmarkResult with success/score.
        """
        pass

    def close(self) -> None:
        """Clean up resources (VMs, browser, etc.)."""
        pass

    def __enter__(self) -> BenchmarkAdapter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class StaticDatasetAdapter(BenchmarkAdapter):
    """Base for static trajectory datasets (Mind2Web, demos).

    Static adapters load pre-recorded trajectories for offline training
    or evaluation, rather than running an interactive environment.
    """

    @property
    def benchmark_type(self) -> str:
        """Static datasets are not interactive."""
        return "static"

    @abstractmethod
    def load_trajectories(
        self, split: str = "test"
    ) -> Iterator[tuple[BenchmarkTask, list[tuple[BenchmarkObservation, BenchmarkAction]]]]:
        """Iterate over expert trajectories.

        Args:
            split: Dataset split ("train", "val", "test").

        Yields:
            Tuples of (task, trajectory) where trajectory is a list of
            (observation, action) pairs.
        """
        pass

    def reset(self, task: BenchmarkTask) -> BenchmarkObservation:
        """Not supported for static datasets."""
        raise NotImplementedError(
            "Static datasets don't support interactive reset. "
            "Use load_trajectories() instead."
        )

    def step(
        self, action: BenchmarkAction
    ) -> tuple[BenchmarkObservation, bool, dict[str, Any]]:
        """Not supported for static datasets."""
        raise NotImplementedError(
            "Static datasets don't support interactive stepping. "
            "Use load_trajectories() instead."
        )

    def evaluate(self, task: BenchmarkTask) -> BenchmarkResult:
        """Not supported for static datasets."""
        raise NotImplementedError(
            "Static datasets don't support execution-based evaluation. "
            "Use offline metrics instead."
        )
