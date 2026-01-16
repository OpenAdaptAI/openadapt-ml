"""
Integration Bridge between openadapt-grounding and openadapt-ml

This module provides the UIElementGraph class which wraps parsed UI elements
from openadapt-grounding parsers and converts them to the openadapt-ml schema.

Types imported from openadapt-grounding:
- Parser: Protocol for UI element parsers (OmniParser, UITars, etc.)
- Element: A detected UI element with normalized bounds
- LocatorResult: Result of attempting to locate an element
- RegistryEntry: A stable element that survived temporal filtering
"""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from openadapt_ml.schema.episode import BoundingBox, UIElement

# Lazy import for openadapt-grounding to make it an optional dependency
_grounding_available: Optional[bool] = None


def _check_grounding_available() -> bool:
    """Check if openadapt-grounding is installed."""
    global _grounding_available
    if _grounding_available is None:
        try:
            import openadapt_grounding  # noqa: F401

            _grounding_available = True
        except ImportError:
            _grounding_available = False
    return _grounding_available


def _get_grounding_types():
    """Import and return openadapt-grounding types.

    Raises:
        ImportError: If openadapt-grounding is not installed
    """
    if not _check_grounding_available():
        raise ImportError(
            "openadapt-grounding is required for perception integration. "
            "Install it with: pip install openadapt-grounding"
        )
    from openadapt_grounding import Element, LocatorResult, Parser, RegistryEntry

    return Element, LocatorResult, Parser, RegistryEntry


# Source types for UI element graphs
SourceType = Literal["omniparser", "uitars", "ax", "dom", "mixed"]


def element_to_ui_element(
    element: Any,  # openadapt_grounding.Element
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    element_index: Optional[int] = None,
    source: Optional[str] = None,
) -> UIElement:
    """Convert an openadapt-grounding Element to a schema UIElement.

    Args:
        element: Element from openadapt-grounding with normalized bounds
        image_width: Image width in pixels (for coordinate conversion)
        image_height: Image height in pixels (for coordinate conversion)
        element_index: Optional index to use as element_id
        source: Source parser name (stored in automation_id for reference)

    Returns:
        UIElement with converted bounds and properties
    """
    Element, _, _, _ = _get_grounding_types()

    if not isinstance(element, Element):
        raise TypeError(f"Expected Element, got {type(element)}")

    # Extract normalized bounds (x, y, width, height)
    norm_x, norm_y, norm_w, norm_h = element.bounds

    # Convert to pixel coordinates if dimensions provided
    if image_width is not None and image_height is not None:
        bounds = BoundingBox(
            x=int(norm_x * image_width),
            y=int(norm_y * image_height),
            width=int(norm_w * image_width),
            height=int(norm_h * image_height),
        )
    else:
        # Store normalized as integers (multiply by 10000 for precision)
        # This allows using BoundingBox which requires int values
        bounds = BoundingBox(
            x=int(norm_x * 10000),
            y=int(norm_y * 10000),
            width=int(norm_w * 10000),
            height=int(norm_h * 10000),
        )

    # Generate element_id from index if provided
    element_id = str(element_index) if element_index is not None else None

    return UIElement(
        role=element.element_type,
        name=element.text,
        bounds=bounds,
        element_id=element_id,
        automation_id=source,  # Store source in automation_id for reference
    )


def ui_element_to_element(
    ui_element: UIElement,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Any:  # Returns openadapt_grounding.Element
    """Convert a schema UIElement back to an openadapt-grounding Element.

    Useful for evaluation or when passing elements back to grounding functions.

    Args:
        ui_element: UIElement from openadapt-ml schema
        image_width: Image width in pixels (for coordinate normalization)
        image_height: Image height in pixels (for coordinate normalization)

    Returns:
        Element with normalized bounds
    """
    Element, _, _, _ = _get_grounding_types()

    if ui_element.bounds is None:
        # Return element with zero bounds if no bounds available
        return Element(
            bounds=(0.0, 0.0, 0.0, 0.0),
            text=ui_element.name,
            element_type=ui_element.role or "unknown",
            confidence=1.0,
        )

    # Convert pixel coordinates to normalized
    if image_width is not None and image_height is not None:
        norm_x = ui_element.bounds.x / image_width
        norm_y = ui_element.bounds.y / image_height
        norm_w = ui_element.bounds.width / image_width
        norm_h = ui_element.bounds.height / image_height
    else:
        # Assume bounds are already in 10000-scale normalized format
        norm_x = ui_element.bounds.x / 10000
        norm_y = ui_element.bounds.y / 10000
        norm_w = ui_element.bounds.width / 10000
        norm_h = ui_element.bounds.height / 10000

    return Element(
        bounds=(norm_x, norm_y, norm_w, norm_h),
        text=ui_element.name,
        element_type=ui_element.role or "unknown",
        confidence=1.0,
    )


class UIElementGraph(BaseModel):
    """A graph of UI elements parsed from a screenshot.

    This class wraps a list of UIElement objects and tracks their source
    (which parser produced them). It provides a bridge between openadapt-grounding
    parsers and the openadapt-ml schema system.

    Attributes:
        graph_id: Unique identifier for this graph (UUID)
        elements: List of UIElement objects
        source: Primary source parser ("omniparser", "uitars", "ax", "dom", "mixed")
        source_summary: Count of elements by source (e.g., {"omniparser": 15, "ax": 8})
        timestamp_ms: Optional timestamp when the screenshot was captured
        image_width: Original image width (for coordinate reference)
        image_height: Original image height (for coordinate reference)

    Example:
        >>> from openadapt_grounding import OmniParserClient
        >>> from openadapt_ml.perception import UIElementGraph
        >>>
        >>> parser = OmniParserClient(endpoint="http://localhost:8000")
        >>> elements = parser.parse(image)
        >>> graph = UIElementGraph.from_parser_output(elements, "omniparser")
        >>> print(f"Found {len(graph.elements)} elements")
    """

    graph_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this graph",
    )
    elements: list[UIElement] = Field(
        default_factory=list,
        description="List of UI elements in the graph",
    )
    source: SourceType = Field(
        default="mixed",
        description="Primary source parser for the elements",
    )
    source_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of elements by source",
    )
    timestamp_ms: Optional[int] = Field(
        None,
        description="Timestamp when the screenshot was captured (milliseconds)",
    )
    image_width: Optional[int] = Field(
        None,
        description="Original image width in pixels",
    )
    image_height: Optional[int] = Field(
        None,
        description="Original image height in pixels",
    )

    @classmethod
    def from_parser_output(
        cls,
        elements: list[Any],  # list[openadapt_grounding.Element]
        source: Union[SourceType, str],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
    ) -> "UIElementGraph":
        """Create a UIElementGraph from parser output.

        Args:
            elements: List of Element objects from openadapt-grounding parser
            source: Parser source name ("omniparser", "uitars", "ax", "dom")
            image_width: Image width in pixels (for coordinate conversion)
            image_height: Image height in pixels (for coordinate conversion)
            timestamp_ms: Optional timestamp when screenshot was captured

        Returns:
            UIElementGraph with converted UIElement objects
        """
        # Validate source type
        valid_sources = {"omniparser", "uitars", "ax", "dom", "mixed"}
        if source not in valid_sources:
            # Allow custom sources but warn
            pass

        # Convert elements
        ui_elements = [
            element_to_ui_element(
                element=el,
                image_width=image_width,
                image_height=image_height,
                element_index=i,
                source=source,
            )
            for i, el in enumerate(elements)
        ]

        # Build source summary
        source_summary = {source: len(elements)}

        return cls(
            elements=ui_elements,
            source=source if source in valid_sources else "mixed",
            source_summary=source_summary,
            timestamp_ms=timestamp_ms,
            image_width=image_width,
            image_height=image_height,
        )

    @classmethod
    def merge(
        cls,
        graphs: list["UIElementGraph"],
        deduplicate_iou_threshold: Optional[float] = None,
    ) -> "UIElementGraph":
        """Merge multiple UIElementGraphs into one.

        Args:
            graphs: List of UIElementGraph objects to merge
            deduplicate_iou_threshold: If provided, remove duplicate elements
                with IoU greater than this threshold (0.0-1.0)

        Returns:
            New UIElementGraph with combined elements
        """
        if not graphs:
            return cls()

        # Combine all elements
        all_elements: list[UIElement] = []
        source_counts: Counter[str] = Counter()

        for graph in graphs:
            all_elements.extend(graph.elements)
            for src, count in graph.source_summary.items():
                source_counts[src] += count

        # Get image dimensions from first graph that has them
        image_width = None
        image_height = None
        for graph in graphs:
            if graph.image_width is not None:
                image_width = graph.image_width
                image_height = graph.image_height
                break

        # TODO: Implement deduplication by IoU if threshold provided
        # For now, just combine all elements

        return cls(
            elements=all_elements,
            source="mixed" if len(source_counts) > 1 else list(source_counts.keys())[0],
            source_summary=dict(source_counts),
            timestamp_ms=graphs[0].timestamp_ms if graphs else None,
            image_width=image_width,
            image_height=image_height,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UIElementGraph":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            UIElementGraph instance
        """
        return cls.model_validate(data)

    def get_element_by_id(self, element_id: str) -> Optional[UIElement]:
        """Find element by ID.

        Args:
            element_id: Element ID to search for

        Returns:
            UIElement if found, None otherwise
        """
        for element in self.elements:
            if element.element_id == element_id:
                return element
        return None

    def get_elements_by_role(self, role: str) -> list[UIElement]:
        """Find elements by role.

        Args:
            role: Role to filter by (e.g., "button", "textbox")

        Returns:
            List of matching UIElements
        """
        return [el for el in self.elements if el.role == role]

    def get_elements_by_text(
        self,
        text: str,
        exact: bool = False,
    ) -> list[UIElement]:
        """Find elements by text content.

        Args:
            text: Text to search for
            exact: If True, require exact match; if False, use substring match

        Returns:
            List of matching UIElements
        """
        results = []
        for el in self.elements:
            if el.name is None:
                continue
            if exact:
                if el.name == text:
                    results.append(el)
            else:
                if text.lower() in el.name.lower():
                    results.append(el)
        return results

    def __len__(self) -> int:
        """Return number of elements in the graph."""
        return len(self.elements)

    def __iter__(self):
        """Iterate over elements."""
        return iter(self.elements)
