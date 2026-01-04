# Semantic Element Capture Design

**Goal**: Enable SoM (Set-of-Marks) mode in demos by capturing semantic element references during recording.

**Status**: Design document.

---

## 1. Problem Statement

For demo-conditioned agents to use element-based actions (`CLICK([5])` instead of `CLICK(500, 300)`), demos must include element information.

Current openadapt-capture records:
- Screenshots
- Mouse/keyboard events with coordinates

This limits agents to coordinate-based actions, which are:
- Brittle across UI changes
- Hard to transfer across similar tasks
- Less explainable

---

## 2. Design Principles

### 2.1 Capture Semantics, Not Executors

**DO**: Record semantic element references per action
```json
{
  "role": "button",
  "name": "Submit",
  "bbox": [100, 200, 180, 240]
}
```

**DON'T**: Record full accessibility trees
```json
// TOO MUCH - huge, platform-specific, brittle
{"AT": {"id": "root", "children": [...]}}  // 100KB+ per frame
```

### 2.2 Resolver at Runtime, Not Capture Time

Demos store **intent**, runtimes resolve to **executor IDs**.

```
Demo: "click the Submit button at approx (140, 220)"
         ↓
Runtime resolver (a11y / DOM / vision)
         ↓
Executor: computer.mouse.move_id('btn_submit')
```

### 2.3 Graceful Degradation

| Environment | Capture Source | Replay Resolver |
|-------------|----------------|-----------------|
| Native Windows | UIA | UIA / OmniParser |
| Native macOS | AX API | AX / Vision |
| Browser | DOM | DOM / Vision |
| Citrix / VDI | Vision only | Vision only |

Vision fallback ensures Citrix/VDI support.

---

## 3. Schema: SemanticElementRef

```python
from pydantic import BaseModel
from typing import Optional

class SemanticElementRef(BaseModel):
    """Lightweight semantic reference to a UI element.

    Captured per action, not per frame. Portable across platforms.
    """

    # Core identity (at least one required)
    role: Optional[str] = None       # "button", "textbox", "menu", "link"
    name: Optional[str] = None       # Visible label or accessible name

    # Spatial hint (for visual matching)
    bbox: Optional[list[int]] = None  # [left, top, right, bottom] screen coords

    # Hierarchy hint (optional, for disambiguation)
    container_path: Optional[list[str]] = None  # ["LoginForm", "ButtonGroup"]

    # State hints (optional)
    enabled: Optional[bool] = None
    focused: Optional[bool] = None

    # Platform-specific ID (optional, not for cross-platform use)
    automation_id: Optional[str] = None  # Windows UIA
    accessibility_id: Optional[str] = None  # macOS AX

    # Confidence (for vision-based capture)
    confidence: Optional[float] = None  # 0.0-1.0 if from OmniParser


class ActionWithElement(BaseModel):
    """Action annotated with semantic element reference."""

    type: str  # "click", "type", "key", etc.

    # Coordinates (always captured for fallback)
    x: Optional[float] = None
    y: Optional[float] = None

    # Semantic element (when available)
    element: Optional[SemanticElementRef] = None

    # Action-specific fields
    text: Optional[str] = None  # for "type"
    key: Optional[str] = None   # for "key"
```

---

## 4. Capture Flow

### 4.1 Per-Action Capture (NOT Per-Frame)

```python
def on_click(x: int, y: int):
    """Called when user clicks during recording."""

    # 1. Always capture coordinates
    action = {"type": "click", "x": x, "y": y}

    # 2. Try to get semantic element (best-effort)
    element = get_element_at_point(x, y)  # Platform-specific
    if element:
        action["element"] = {
            "role": element.role,
            "name": element.name,
            "bbox": element.bbox,
        }

    # 3. Store action
    record_action(action)
```

### 4.2 Platform-Specific Element Capture

```python
# Windows (UIA)
def get_element_at_point_windows(x: int, y: int) -> SemanticElementRef:
    import uiautomation
    element = uiautomation.ControlFromPoint(x, y)
    return SemanticElementRef(
        role=element.ControlTypeName,
        name=element.Name,
        bbox=element.BoundingRectangle,
        automation_id=element.AutomationId,
    )

# macOS (AX API)
def get_element_at_point_macos(x: int, y: int) -> SemanticElementRef:
    import ApplicationServices
    element = AXUIElementCopyElementAtPosition(...)
    return SemanticElementRef(
        role=element.AXRole,
        name=element.AXTitle or element.AXDescription,
        bbox=element.AXFrame,
    )

# Vision fallback (OmniParser)
def get_element_at_point_vision(x: int, y: int, screenshot: bytes) -> SemanticElementRef:
    elements = omniparser.detect(screenshot)
    for elem in elements:
        if point_in_bbox((x, y), elem.bbox):
            return SemanticElementRef(
                role=elem.predicted_role,
                name=elem.ocr_text,
                bbox=elem.bbox,
                confidence=elem.confidence,
            )
    return None
```

---

## 5. Runtime Resolution

### 5.1 Resolver Priority

```python
def resolve_element(
    element_ref: SemanticElementRef,
    observation: BenchmarkObservation,
) -> str:
    """Resolve semantic ref to executor element ID.

    Priority: a11y > DOM > vision
    """

    # 1. Try accessibility tree (fastest, most accurate)
    if observation.accessibility_tree:
        match = find_in_a11y_tree(
            observation.accessibility_tree,
            role=element_ref.role,
            name=element_ref.name,
            bbox_hint=element_ref.bbox,
        )
        if match:
            return match.id

    # 2. Try DOM (for browser environments)
    if observation.dom_html:
        match = find_in_dom(
            observation.dom_html,
            role=element_ref.role,
            name=element_ref.name,
        )
        if match:
            return match.xpath

    # 3. Fall back to vision (OmniParser)
    if observation.screenshot:
        match = find_via_vision(
            observation.screenshot,
            role=element_ref.role,
            name=element_ref.name,
            bbox_hint=element_ref.bbox,
        )
        if match:
            return match.id

    # 4. Last resort: use bbox center
    if element_ref.bbox:
        return f"coords:{element_ref.bbox}"

    raise ResolutionError(f"Cannot resolve element: {element_ref}")
```

### 5.2 Matching Strategy

```python
def find_in_a11y_tree(
    tree: dict,
    role: str,
    name: str,
    bbox_hint: list[int],
) -> Optional[dict]:
    """Find element in a11y tree by semantic properties."""

    candidates = []

    def visit(node):
        score = 0

        # Role match (strong signal)
        if node.get("role") == role:
            score += 10

        # Name match (strong signal)
        if node.get("name") == name:
            score += 10
        elif name and name.lower() in (node.get("name") or "").lower():
            score += 5

        # Bbox proximity (weak signal, for disambiguation)
        if bbox_hint and node.get("bbox"):
            distance = bbox_distance(bbox_hint, node["bbox"])
            if distance < 50:
                score += 3

        if score > 0:
            candidates.append((score, node))

        for child in node.get("children", []):
            visit(child)

    visit(tree)

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    return None
```

---

## 6. Why Not Vision-Only?

Vision (OmniParser) is necessary for Citrix/VDI but is strictly worse when a11y is available:

| Failure Mode | Vision | A11y |
|--------------|--------|------|
| Duplicate "OK" buttons | Can't disambiguate | Tree position resolves |
| Invisible but interactive | Misses | Knows it exists |
| Disabled elements | Misfires | Knows state |
| Small UI shifts | Bbox drift | Stable identity |
| Cost per action | VLM call (~100ms) | Local lookup (~1ms) |
| Determinism | Probabilistic | Ground truth |

**Bottom line**: Vision is a semantic bootstrapper. A11y is a semantic oracle.

---

## 7. Integration with openadapt-capture

### 7.1 Changes Required

1. **Add element capture hook**: On each click, query OS accessibility for target element
2. **Store SemanticElementRef**: In action events, alongside coordinates
3. **Make it optional**: Gracefully handle environments without a11y

### 7.2 Minimal Implementation

```python
# In openadapt-capture event handler
def record_click(x: int, y: int):
    event = {
        "type": "click",
        "x": x,
        "y": y,
        "timestamp": time.time(),
    }

    # Best-effort element capture
    try:
        element = get_element_at_point(x, y)
        if element:
            event["element"] = element.model_dump()
    except Exception:
        pass  # Continue without element data

    self.events.append(event)
```

### 7.3 Backwards Compatibility

Existing recordings without `element` field:
- Still work (coordinate-based fallback)
- Can be post-processed with OmniParser to add element annotations
- Acceptable for action sequencing, less ideal for SoM generalization

---

## 8. Browser Capture: Chrome Extension

### 8.1 Existing OpenAdapt Chrome Extension

OpenAdapt already has a Chrome extension ([source](https://github.com/OpenAdaptAI/OpenAdapt/tree/main/chrome_extension)) that captures:

| Data | Captured | Notes |
|------|----------|-------|
| Click events | ✅ | clientX/Y, screenX/Y, button |
| Element bounding box | ✅ | top, left, bottom, right |
| Element ID | ✅ | DOM element identifier |
| Keyboard events | ✅ | key, code |
| Visible HTML | ✅ | Serialized DOM snapshot |
| Scroll events | ✅ | Displacement from last position |
| DOM mutations | ✅ | Via MutationObserver |

**Architecture:**
```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  content.js     │ ←──────────────→   │  OpenAdapt      │
│  (per page)     │   ws://localhost   │  Backend        │
└────────┬────────┘       :8765        └─────────────────┘
         │
         │ chrome.runtime.sendMessage
         ▼
┌─────────────────┐
│  background.js  │
│  (service worker)│
└─────────────────┘
```

### 8.2 Integration with openadapt-capture

**Option A: Use existing extension directly**
- openadapt-capture starts WebSocket server on port 8765
- Extension sends DOM events + visible HTML
- Map DOM element IDs to SemanticElementRef

**Option B: Fork/enhance extension for SoM**
- Add explicit `role` extraction (button, link, input, etc.)
- Add `name` extraction (aria-label, innerText, placeholder)
- Add `xpath` or `css_selector` for stable identification
- Match SemanticElementRef schema

### 8.3 Enhanced content.js Capture

```javascript
// Proposed enhancement for SoM-compatible capture
function getSemanticElementRef(element) {
    return {
        // Core identity
        role: element.getAttribute('role') || element.tagName.toLowerCase(),
        name: element.getAttribute('aria-label')
            || element.innerText?.slice(0, 50)
            || element.getAttribute('placeholder')
            || element.getAttribute('title'),

        // Spatial hint
        bbox: (() => {
            const rect = element.getBoundingClientRect();
            return [rect.left, rect.top, rect.right, rect.bottom];
        })(),

        // Stable selectors
        css_selector: getCssSelector(element),
        xpath: getXPath(element),

        // State
        enabled: !element.disabled,
        focused: document.activeElement === element,
        visible: isElementVisible(element),

        // DOM-specific
        tag: element.tagName.toLowerCase(),
        id: element.id || null,
        classes: Array.from(element.classList),
    };
}

function getCssSelector(element) {
    // Generate minimal unique CSS selector
    if (element.id) return `#${element.id}`;
    // ... fallback to class/nth-child path
}

function getXPath(element) {
    // Generate XPath for element
    // ... implementation
}
```

### 8.4 Mapping to SemanticElementRef

```python
def dom_event_to_semantic_ref(event: dict) -> SemanticElementRef:
    """Convert Chrome extension event to SemanticElementRef."""
    return SemanticElementRef(
        role=event.get("role") or event.get("tag"),
        name=event.get("name") or event.get("innerText"),
        bbox=event.get("bbox"),
        container_path=event.get("xpath", "").split("/"),
        # Browser-specific
        css_selector=event.get("css_selector"),
        xpath=event.get("xpath"),
    )
```

### 8.5 Runtime Resolution for Browser

```python
def resolve_element_browser(
    element_ref: SemanticElementRef,
    dom_html: str,
) -> str:
    """Resolve semantic ref to DOM selector."""

    # Priority 1: Stable selectors (if captured)
    if element_ref.css_selector:
        return f"css:{element_ref.css_selector}"
    if element_ref.xpath:
        return f"xpath:{element_ref.xpath}"

    # Priority 2: Semantic matching
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(dom_html, 'html.parser')

    # Find by role + name
    candidates = soup.find_all(
        attrs={"role": element_ref.role}
    )
    for elem in candidates:
        if element_ref.name in (elem.get_text() or ""):
            return f"xpath:{get_xpath(elem)}"

    # Priority 3: Bbox-based (fallback)
    return f"coords:{element_ref.bbox}"
```

---

## 9. Unified Capture Architecture

### 9.1 Platform Capture Matrix

| Platform | Capture Source | Data Available | Integration |
|----------|----------------|----------------|-------------|
| **Windows** | UIA (UI Automation) | role, name, bbox, automation_id, state | Python `uiautomation` package |
| **macOS** | AX API (Accessibility) | role, name, bbox, description | PyObjC / `ApplicationServices` |
| **Browser** | DOM + Chrome Extension | role, name, bbox, xpath, css_selector, visible HTML | WebSocket to port 8765 |
| **Citrix/VDI** | Vision (OmniParser) | predicted_role, ocr_text, bbox, confidence | Screenshot processing |

### 9.2 openadapt-capture Integration

```
┌─────────────────────────────────────────────────────────────────┐
│  openadapt-capture                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Event Loop                                                │  │
│  │   on_click(x, y) ────┬───→ get_element_at_point()         │  │
│  │   on_key(key)        │           │                        │  │
│  │   on_scroll(dx, dy)  │           ▼                        │  │
│  │                      │    ┌─────────────────────┐         │  │
│  │                      │    │ Platform Selector   │         │  │
│  │                      │    │  ├─ Windows: UIA    │         │  │
│  │                      │    │  ├─ macOS: AX       │         │  │
│  │                      │    │  ├─ Browser: DOM    │         │  │
│  │                      │    │  └─ Fallback: None  │         │  │
│  │                      │    └─────────────────────┘         │  │
│  │                      │           │                        │  │
│  │                      │           ▼                        │  │
│  │                      │    SemanticElementRef              │  │
│  │                      │           │                        │  │
│  │                      └───────────┼───────────────────────→│  │
│  │                                  ▼                        │  │
│  │                         ActionWithElement                 │  │
│  │                              │                            │  │
│  └──────────────────────────────┼────────────────────────────┘  │
│                                 ▼                               │
│                          Recording Store                        │
│                          (JSON / Parquet)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 WebSocket Server for Browser Capture

```python
# openadapt-capture/browser_bridge.py
import asyncio
import websockets
import json

class BrowserCaptureBridge:
    """WebSocket server that receives events from Chrome extension."""

    def __init__(self, port: int = 8765):
        self.port = port
        self.events: list[dict] = []

    async def handle_connection(self, websocket):
        async for message in websocket:
            event = json.loads(message)

            if event.get("type") == "click":
                # Extract semantic element ref from DOM event
                element_ref = self._extract_semantic_ref(event)
                self.events.append({
                    "type": "click",
                    "x": event["clientX"],
                    "y": event["clientY"],
                    "element": element_ref,
                    "timestamp": event["timestamp"],
                    "url": event["url"],
                    "visible_html": event.get("visibleHtml"),
                })

    def _extract_semantic_ref(self, event: dict) -> dict:
        return {
            "role": event.get("role") or event.get("tag", "unknown"),
            "name": event.get("name") or event.get("innerText", "")[:50],
            "bbox": event.get("bbox"),
            "css_selector": event.get("css_selector"),
            "xpath": event.get("xpath"),
        }

    async def start(self):
        async with websockets.serve(self.handle_connection, "localhost", self.port):
            await asyncio.Future()  # Run forever
```

### 9.4 Recommended Implementation Order

1. **Browser (Chrome Extension)** - Easiest, extension already exists
   - Add WebSocket server to openadapt-capture
   - Enhance content.js to capture SemanticElementRef fields
   - Test with web-based WAA tasks

2. **Windows (UIA)** - Critical for WAA
   - Add `uiautomation` dependency
   - Hook into click events
   - Query element at click point

3. **macOS (AX API)** - For parity
   - Add PyObjC bindings
   - Mirror Windows approach

4. **Vision fallback** - For Citrix
   - Integrate OmniParser
   - Use when a11y unavailable

---

## 10. Summary

| Layer | What to Capture | What to Store |
|-------|-----------------|---------------|
| **Capture** | Element at click point | SemanticElementRef per action |
| **Demo** | Intent | role + name + bbox hint |
| **Runtime** | Resolution | a11y → DOM → vision |
| **Execution** | Grounding | Platform-specific ID |

**Platform coverage:**

| Platform | Capture | Replay | Notes |
|----------|---------|--------|-------|
| Windows | UIA | UIA / OmniParser | Primary WAA target |
| macOS | AX API | AX / Vision | Desktop parity |
| Browser | Chrome Extension | DOM / Vision | Web benchmarks |
| Citrix/VDI | Vision only | Vision only | Full support via fallback |

This design:
- Supports SoM mode in demos
- Works across Windows/macOS/browser
- Degrades gracefully to vision for Citrix/VDI
- Keeps demos portable and small
- Enables hybrid resolver architecture
- Leverages existing OpenAdapt Chrome extension
