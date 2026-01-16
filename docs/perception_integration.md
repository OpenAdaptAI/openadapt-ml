# Perception Integration: openadapt-grounding to openadapt-ml

## Overview

This document describes the integration layer between `openadapt-grounding` (UI element detection) and `openadapt-ml` (ML training and inference). The integration enables parsed UI elements from various grounding backends to be used within the openadapt-ml schema system.

## Architecture

```
openadapt-grounding                    openadapt-ml
+------------------+                   +------------------+
|                  |                   |                  |
|  Parser Protocol |                   |  UIElementGraph  |
|  - OmniParser    | ---- Element ---> |  - graph_id      |
|  - UITars        |     List[Element] |  - elements      |
|  - AX Parser     |                   |  - source        |
|  - DOM Parser    |                   |  - timestamp_ms  |
|                  |                   |                  |
|  Element:        |                   |  UIElement:      |
|  - bounds (norm) |    conversion     |  - bounds (px)   |
|  - text          | ----------------> |  - name          |
|  - element_type  |                   |  - role          |
|  - confidence    |                   |  - element_id    |
|                  |                   |                  |
+------------------+                   +------------------+
```

## Design Decisions

### 1. Wrapper Pattern (Not Inheritance)

The `UIElementGraph` class wraps a list of `UIElement` objects rather than directly using `openadapt_grounding.Element`. This allows:

- **Schema stability**: The ML schema remains independent of grounding implementation details
- **Coordinate normalization**: Elements from grounding use normalized (0-1) coordinates; the wrapper can convert to pixel coordinates when needed
- **Source tracking**: Each graph knows which parser produced it (omniparser, uitars, ax, dom)
- **Aggregation**: Multiple sources can be combined into a single graph with source attribution

### 2. Factory Method Pattern

`UIElementGraph.from_parser_output(elements, source)` provides a clean conversion from grounding outputs:

```python
from openadapt_grounding import OmniParserClient, Element
from openadapt_ml.perception import UIElementGraph

# Parse screenshot
parser = OmniParserClient(endpoint="http://localhost:8000")
elements: list[Element] = parser.parse(screenshot_image)

# Convert to ML schema
graph = UIElementGraph.from_parser_output(elements, source="omniparser")
```

### 3. Bidirectional Type Mapping

| openadapt_grounding.Element | openadapt_ml.UIElement |
|----------------------------|------------------------|
| `bounds: Tuple[float, float, float, float]` | `bounds: BoundingBox` |
| `text: Optional[str]` | `name: Optional[str]` |
| `element_type: str` | `role: Optional[str]` |
| `confidence: float` | (stored in raw dict) |

### 4. Source Tracking

Each `UIElementGraph` tracks its source(s) to enable:

- **Debugging**: Know which parser detected each element
- **Fusion**: Combine elements from multiple sources with attribution
- **Metrics**: Compare detection rates across parsers

```python
graph.source  # "omniparser"
graph.source_summary  # {"omniparser": 15, "ax": 8}
```

## Implementation Files

### `openadapt_ml/perception/__init__.py`

Package initialization that exports the public API:

```python
from openadapt_ml.perception.integration import UIElementGraph

__all__ = ["UIElementGraph"]
```

### `openadapt_ml/perception/integration.py`

Core integration module containing:

- `UIElementGraph`: Pydantic model wrapping parsed UI elements
- `element_to_ui_element()`: Convert grounding Element to schema UIElement
- `ui_element_to_element()`: Reverse conversion (for evaluation)

## Usage Examples

### Basic Parsing Integration

```python
from PIL import Image
from openadapt_grounding import OmniParserClient
from openadapt_ml.perception import UIElementGraph

# Load screenshot
image = Image.open("screenshot.png")

# Parse with OmniParser
parser = OmniParserClient(endpoint="http://localhost:8000")
elements = parser.parse(image)

# Convert to ML schema
graph = UIElementGraph.from_parser_output(
    elements=elements,
    source="omniparser",
    image_width=image.width,
    image_height=image.height,
)

# Use in ML pipeline
for ui_element in graph.elements:
    print(f"{ui_element.role}: {ui_element.name} at {ui_element.bounds}")
```

### Multi-Source Fusion

```python
from openadapt_grounding import OmniParserClient, UITarsClient
from openadapt_ml.perception import UIElementGraph

# Parse with multiple backends
omni_elements = omni_parser.parse(image)
uitars_elements = uitars_parser.parse(image)

# Create individual graphs
omni_graph = UIElementGraph.from_parser_output(omni_elements, "omniparser")
uitars_graph = UIElementGraph.from_parser_output(uitars_elements, "uitars")

# Merge (future: with deduplication)
combined = UIElementGraph.merge([omni_graph, uitars_graph])
```

### With Episode Schema

```python
from openadapt_ml.schema import Episode, Step, Observation
from openadapt_ml.perception import UIElementGraph

# Include parsed elements in observation
observation = Observation(
    screenshot_path="step_0.png",
    a11y_tree=graph.to_dict(),  # Serialized element graph
)
```

## Dependencies

The integration requires `openadapt-grounding` as an optional dependency:

```toml
[project.optional-dependencies]
grounding = [
    "openadapt-grounding>=0.1.0",
]
```

If `openadapt-grounding` is not installed, the perception module will raise an `ImportError` with installation instructions.

## Future Enhancements

1. **Element deduplication**: When merging from multiple sources, detect overlapping elements by IoU
2. **Confidence-weighted fusion**: Prefer high-confidence detections when elements conflict
3. **Temporal tracking**: Link elements across frames using the grounding registry
4. **Lazy evaluation**: Parse elements on-demand rather than upfront
