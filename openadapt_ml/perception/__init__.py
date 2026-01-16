"""
Perception Integration Module

Bridges openadapt-grounding (UI element detection) with openadapt-ml (ML schema).

This module provides:
- UIElementGraph: A wrapper class for parsed UI elements
- Conversion utilities between grounding types and ML schema types

Usage:
    from openadapt_ml.perception import UIElementGraph

    # From parser output
    graph = UIElementGraph.from_parser_output(elements, source="omniparser")

    # Access elements
    for element in graph.elements:
        print(f"{element.role}: {element.name}")

Requires:
    pip install openadapt-grounding
    # or: uv add openadapt-grounding
"""

from openadapt_ml.perception.integration import (
    UIElementGraph,
    element_to_ui_element,
    ui_element_to_element,
)

__all__ = [
    "UIElementGraph",
    "element_to_ui_element",
    "ui_element_to_element",
]
