# Chrome Extension Design for openadapt-capture

## Overview

This document describes the design for re-implementing the OpenAdapt Chrome extension within the openadapt-capture project. The extension enables DOM-level event capture for browser interactions, providing rich semantic element references that complement native OS-level input capture.

**Status**: Design Document
**Target Repository**: openadapt-capture
**Reference Implementation**: https://github.com/OpenAdaptAI/OpenAdapt/tree/main/chrome_extension

---

## 1. Why Re-implement

### 1.1 Legacy Repository Limitations

The existing Chrome extension in OpenAdaptAI/OpenAdapt has several constraints:

| Issue | Impact |
|-------|--------|
| Tightly coupled to legacy OpenAdapt | Difficult to use standalone or with openadapt-capture |
| No pip-installable package | Requires full OpenAdapt repo checkout |
| Manifest V2 patterns | Browser APIs evolving toward Manifest V3 |
| Missing semantic element capture | Only captures coordinates, not element identifiers |
| Coordinate transformation complexity | Uses linear regression to map client-to-screen coordinates |

### 1.2 Benefits of Integration with openadapt-capture

| Benefit | Description |
|---------|-------------|
| **Unified event stream** | Browser events merge seamlessly with OS-level capture |
| **SemanticElementRef support** | Rich element identification for SoM mode |
| **Lightweight dependency** | `uv add openadapt-capture` includes browser bridge |
| **Single recording session** | No separate browser recording process |
| **Cross-platform consistency** | Same WebSocket protocol on macOS, Windows, Linux |
| **Modern architecture** | Manifest V3, asyncio WebSocket server, Pydantic schemas |

### 1.3 Use Cases Enabled

1. **Training data for browser automation** - Capture demonstrations with element-level grounding
2. **Cross-application workflows** - Seamlessly record browser + native app interactions
3. **SoM (Set-of-Marks) mode support** - Element IDs enable `CLICK([1])` instead of `CLICK(x, y)`
4. **Replay verification** - XPath and CSS selectors enable element re-identification

---

## 2. Architecture Overview

```
+------------------+      WebSocket       +----------------------+
|  Chrome Browser  | <------------------> |  openadapt-capture   |
+------------------+      (ws://8765)     +----------------------+
        |                                          |
        v                                          v
+------------------+                      +----------------------+
|  content.js      |                      |  browser_bridge.py   |
|  (DOM capture)   |                      |  (WebSocket server)  |
+------------------+                      +----------------------+
        |                                          |
        v                                          v
+------------------+                      +----------------------+
|  background.js   |                      |  CaptureStorage      |
|  (WS bridge)     |                      |  (SQLite + events)   |
+------------------+                      +----------------------+
```

### 2.1 Chrome Extension (Manifest V3)

The extension consists of three main components:

1. **manifest.json** - Extension configuration and permissions
2. **content.js** - Injected into web pages to capture DOM events
3. **background.js** - Service worker managing WebSocket connection

### 2.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `content.js` | Capture DOM events, extract element refs, send to background |
| `background.js` | Maintain WebSocket connection, relay messages, sync mode state |
| `browser_bridge.py` | WebSocket server, event storage, mode control |

---

## 3. SemanticElementRef Capture

### 3.1 Data Structure

For each user action involving a DOM element, we capture a `SemanticElementRef`:

```typescript
interface SemanticElementRef {
  // Identity
  role: string;           // ARIA role or inferred from tag
  name: string;           // Accessible name (aria-label, innerText, etc.)

  // Location
  bbox: {
    x: number;            // Left edge in viewport pixels
    y: number;            // Top edge in viewport pixels
    width: number;        // Element width
    height: number;       // Element height
  };

  // Selectors (for replay)
  xpath: string;          // Absolute XPath
  css_selector: string;   // Minimal unique CSS selector

  // State
  state: {
    enabled: boolean;     // Not disabled
    focused: boolean;     // Has focus
    visible: boolean;     // Computed visibility
    checked?: boolean;    // For checkboxes/radios
    selected?: boolean;   // For options
    expanded?: boolean;   // For expandable elements
    value?: string;       // Current value (inputs)
  };

  // Optional
  tag_name: string;       // HTML tag name
  id?: string;            // Element ID if present
  class_list?: string[];  // CSS classes
}
```

### 3.2 Role Extraction

Priority order for determining element role:

1. Explicit `role` attribute
2. Implicit role from tag name (semantic HTML):
   - `<button>` -> "button"
   - `<a>` -> "link"
   - `<input type="text">` -> "textbox"
   - `<input type="checkbox">` -> "checkbox"
   - `<select>` -> "combobox"
   - `<textarea>` -> "textbox"
   - `<h1>...<h6>` -> "heading"
   - `<img>` -> "image"
   - `<nav>` -> "navigation"
   - `<main>` -> "main"
   - `<aside>` -> "complementary"
3. Generic role based on interactivity:
   - Clickable `<div>` with handler -> "button"
   - Focusable element -> "generic"

### 3.3 Name Extraction

Priority order for accessible name:

1. `aria-label` attribute
2. `aria-labelledby` (resolve to referenced element's text)
3. `<label>` element (for form controls)
4. `alt` attribute (for images)
5. `title` attribute
6. `placeholder` attribute (for inputs)
7. `innerText` (truncated to 100 chars)
8. `value` attribute (for buttons)

### 3.4 Bounding Box Calculation

```javascript
function getBbox(element) {
  const rect = element.getBoundingClientRect();
  return {
    x: rect.left + window.scrollX,
    y: rect.top + window.scrollY,
    width: rect.width,
    height: rect.height
  };
}
```

Note: Coordinates are in page space (includes scroll offset).

### 3.5 XPath Generation

Generate absolute XPath for stable element identification:

```javascript
function getXPath(element) {
  if (element.id) {
    return `//*[@id="${element.id}"]`;
  }

  const parts = [];
  let current = element;

  while (current && current.nodeType === Node.ELEMENT_NODE) {
    let index = 1;
    let sibling = current.previousElementSibling;

    while (sibling) {
      if (sibling.tagName === current.tagName) {
        index++;
      }
      sibling = sibling.previousElementSibling;
    }

    const tagName = current.tagName.toLowerCase();
    const position = `[${index}]`;
    parts.unshift(`${tagName}${position}`);
    current = current.parentElement;
  }

  return '/' + parts.join('/');
}
```

### 3.6 CSS Selector Generation

Generate minimal unique CSS selector:

```javascript
function getMinimalSelector(element) {
  // Try ID first
  if (element.id) {
    return `#${CSS.escape(element.id)}`;
  }

  // Try unique class
  const classes = Array.from(element.classList);
  for (const cls of classes) {
    const selector = `.${CSS.escape(cls)}`;
    if (document.querySelectorAll(selector).length === 1) {
      return selector;
    }
  }

  // Try tag + nth-of-type
  const parent = element.parentElement;
  if (parent) {
    const siblings = Array.from(parent.children).filter(
      c => c.tagName === element.tagName
    );
    const index = siblings.indexOf(element) + 1;
    const parentSelector = getMinimalSelector(parent);
    return `${parentSelector} > ${element.tagName.toLowerCase()}:nth-of-type(${index})`;
  }

  return element.tagName.toLowerCase();
}
```

### 3.7 State Extraction

```javascript
function getElementState(element) {
  const computedStyle = window.getComputedStyle(element);

  return {
    enabled: !element.disabled && !element.getAttribute('aria-disabled'),
    focused: document.activeElement === element,
    visible: computedStyle.display !== 'none' &&
             computedStyle.visibility !== 'hidden' &&
             computedStyle.opacity !== '0',
    checked: element.checked,
    selected: element.selected,
    expanded: element.getAttribute('aria-expanded') === 'true',
    value: element.value
  };
}
```

---

## 4. Event Types to Capture

### 4.1 Click Events

```typescript
interface BrowserClickEvent {
  type: "browser.click";
  timestamp: number;           // Unix timestamp (ms)
  url: string;                 // Current page URL

  // Coordinates
  clientX: number;             // Viewport X
  clientY: number;             // Viewport Y
  pageX: number;               // Page X (with scroll)
  pageY: number;               // Page Y (with scroll)

  // Click details
  button: 0 | 1 | 2;           // left, middle, right
  clickCount: 1 | 2;           // single or double

  // Target element
  element: SemanticElementRef;
}
```

### 4.2 Keyboard Events

```typescript
interface BrowserKeyEvent {
  type: "browser.keydown" | "browser.keyup";
  timestamp: number;
  url: string;

  // Key identification
  key: string;                 // Logical key value ("a", "Enter", "Shift")
  code: string;                // Physical key code ("KeyA", "Enter", "ShiftLeft")
  keyCode: number;             // Legacy key code

  // Modifiers
  shiftKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  metaKey: boolean;

  // Target element (if focused)
  element?: SemanticElementRef;
}
```

### 4.3 Scroll Events

```typescript
interface BrowserScrollEvent {
  type: "browser.scroll";
  timestamp: number;
  url: string;

  // Scroll position
  scrollX: number;             // Horizontal scroll offset
  scrollY: number;             // Vertical scroll offset

  // Scroll delta
  deltaX: number;              // Horizontal scroll change
  deltaY: number;              // Vertical scroll change

  // Target (window or element)
  target: "window" | SemanticElementRef;
}
```

### 4.4 Form Input Events

```typescript
interface BrowserInputEvent {
  type: "browser.input";
  timestamp: number;
  url: string;

  // Input details
  inputType: string;           // "insertText", "deleteContentBackward", etc.
  data: string | null;         // Inserted text
  value: string;               // Current field value

  // Target element
  element: SemanticElementRef;
}
```

### 4.5 Navigation Events

```typescript
interface BrowserNavigationEvent {
  type: "browser.navigate";
  timestamp: number;

  // Navigation details
  url: string;                 // New URL
  previousUrl: string;         // Previous URL
  navigationType: "link" | "typed" | "reload" | "back_forward" | "form_submit";
}
```

### 4.6 Focus Events

```typescript
interface BrowserFocusEvent {
  type: "browser.focus" | "browser.blur";
  timestamp: number;
  url: string;

  // Target element
  element: SemanticElementRef;
}
```

---

## 5. WebSocket Protocol

### 5.1 Connection

- **Server**: `ws://localhost:8765`
- **Reconnection**: Automatic with 1-second interval
- **Protocol**: JSON messages

### 5.2 Message Format

All messages follow this envelope:

```typescript
interface WebSocketMessage {
  type: string;                // Message type
  timestamp: number;           // Unix timestamp (ms)
  tabId?: number;              // Chrome tab ID
  payload: object;             // Message-specific data
}
```

### 5.3 Server -> Extension Messages

#### SET_MODE

Control recording/replay/idle mode:

```json
{
  "type": "SET_MODE",
  "timestamp": 1704067200000,
  "payload": {
    "mode": "record"
  }
}
```

Valid modes: `"idle"`, `"record"`, `"replay"`

#### PING

Keep-alive check:

```json
{
  "type": "PING",
  "timestamp": 1704067200000,
  "payload": {}
}
```

#### EXECUTE_ACTION (Replay Mode)

Execute an action in the browser:

```json
{
  "type": "EXECUTE_ACTION",
  "timestamp": 1704067200000,
  "payload": {
    "action": {
      "type": "click",
      "xpath": "/html/body/div[1]/button[2]",
      "css_selector": "#submit-btn"
    }
  }
}
```

### 5.4 Extension -> Server Messages

#### DOM_EVENT

Captured DOM event:

```json
{
  "type": "DOM_EVENT",
  "timestamp": 1704067200000,
  "tabId": 123,
  "payload": {
    "eventType": "browser.click",
    "url": "https://example.com/page",
    "clientX": 150,
    "clientY": 200,
    "pageX": 150,
    "pageY": 450,
    "button": 0,
    "clickCount": 1,
    "element": {
      "role": "button",
      "name": "Submit",
      "bbox": {"x": 100, "y": 180, "width": 100, "height": 40},
      "xpath": "/html/body/div[1]/button[2]",
      "css_selector": "#submit-btn",
      "state": {"enabled": true, "focused": false, "visible": true},
      "tag_name": "button",
      "id": "submit-btn"
    }
  }
}
```

#### DOM_SNAPSHOT

Full DOM snapshot (on navigation or periodically):

```json
{
  "type": "DOM_SNAPSHOT",
  "timestamp": 1704067200000,
  "tabId": 123,
  "payload": {
    "url": "https://example.com/page",
    "title": "Example Page",
    "html": "<!DOCTYPE html>...",
    "visibleElements": [
      {"role": "button", "name": "Submit", "bbox": {...}, ...},
      {"role": "link", "name": "Home", "bbox": {...}, ...}
    ]
  }
}
```

#### PONG

Response to PING:

```json
{
  "type": "PONG",
  "timestamp": 1704067200000,
  "payload": {}
}
```

#### ERROR

Error notification:

```json
{
  "type": "ERROR",
  "timestamp": 1704067200000,
  "tabId": 123,
  "payload": {
    "code": "ELEMENT_NOT_FOUND",
    "message": "Could not locate element with xpath: /html/body/div[1]/button[99]"
  }
}
```

---

## 6. Python WebSocket Server

### 6.1 Implementation

```python
# openadapt_capture/browser_bridge.py

import asyncio
import json
from enum import Enum
from typing import Callable, Optional
from dataclasses import dataclass, field

import websockets
from websockets.server import WebSocketServerProtocol

from openadapt_capture.events import BaseEvent
from openadapt_capture.storage import CaptureStorage


class BrowserMode(str, Enum):
    IDLE = "idle"
    RECORD = "record"
    REPLAY = "replay"


@dataclass
class BrowserEvent(BaseEvent):
    """Browser event from Chrome extension."""

    type: str  # e.g., "browser.click", "browser.keydown"
    url: str
    tab_id: int
    payload: dict = field(default_factory=dict)


class BrowserBridge:
    """WebSocket server for Chrome extension communication."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        storage: Optional[CaptureStorage] = None,
        on_event: Optional[Callable[[BrowserEvent], None]] = None,
    ):
        self.host = host
        self.port = port
        self.storage = storage
        self.on_event = on_event

        self._mode = BrowserMode.IDLE
        self._clients: set[WebSocketServerProtocol] = set()
        self._server = None
        self._running = False

    @property
    def mode(self) -> BrowserMode:
        return self._mode

    async def set_mode(self, mode: BrowserMode) -> None:
        """Set mode and broadcast to all connected clients."""
        self._mode = mode
        message = json.dumps({
            "type": "SET_MODE",
            "timestamp": time.time() * 1000,
            "payload": {"mode": mode.value}
        })
        await self._broadcast(message)

    async def _broadcast(self, message: str) -> None:
        """Send message to all connected clients."""
        if self._clients:
            await asyncio.gather(
                *[client.send(message) for client in self._clients],
                return_exceptions=True
            )

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a connected Chrome extension."""
        self._clients.add(websocket)

        try:
            # Send current mode on connect
            await websocket.send(json.dumps({
                "type": "SET_MODE",
                "timestamp": time.time() * 1000,
                "payload": {"mode": self._mode.value}
            }))

            # Process incoming messages
            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)

    async def _handle_message(
        self,
        websocket: WebSocketServerProtocol,
        message: str
    ) -> None:
        """Process message from extension."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "DOM_EVENT":
                await self._handle_dom_event(data)
            elif msg_type == "DOM_SNAPSHOT":
                await self._handle_dom_snapshot(data)
            elif msg_type == "PONG":
                pass  # Keep-alive response
            elif msg_type == "ERROR":
                print(f"Browser error: {data.get('payload', {}).get('message')}")

        except json.JSONDecodeError:
            print(f"Invalid JSON from extension: {message[:100]}")

    async def _handle_dom_event(self, data: dict) -> None:
        """Process DOM event from extension."""
        if self._mode != BrowserMode.RECORD:
            return

        payload = data.get("payload", {})
        event = BrowserEvent(
            timestamp=data.get("timestamp", 0) / 1000,  # Convert to seconds
            type=payload.get("eventType", "browser.unknown"),
            url=payload.get("url", ""),
            tab_id=data.get("tabId", 0),
            payload=payload
        )

        # Store event
        if self.storage:
            self.storage.add_event(event)

        # Notify callback
        if self.on_event:
            self.on_event(event)

    async def _handle_dom_snapshot(self, data: dict) -> None:
        """Process DOM snapshot from extension."""
        if self._mode != BrowserMode.RECORD:
            return

        # Store snapshot for SoM generation
        payload = data.get("payload", {})
        # Implementation depends on storage format

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        print(f"Browser bridge listening on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def run_forever(self) -> None:
        """Run server until stopped."""
        await self.start()
        try:
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            await self.stop()
```

### 6.2 Integration with Capture Loop

```python
# openadapt_capture/recorder.py (additions)

class Recorder:
    def __init__(self, ...):
        # ... existing init ...
        self._browser_bridge: Optional[BrowserBridge] = None

    async def start_with_browser(self) -> None:
        """Start recording with browser event capture."""
        # Start browser bridge
        self._browser_bridge = BrowserBridge(
            storage=self._storage,
            on_event=self._on_browser_event
        )
        await self._browser_bridge.start()
        await self._browser_bridge.set_mode(BrowserMode.RECORD)

        # Start regular capture
        await self.start()

    def _on_browser_event(self, event: BrowserEvent) -> None:
        """Handle incoming browser event."""
        # Merge with OS-level events by timestamp
        # Browser events include element refs that OS events lack
        pass

    async def stop(self) -> None:
        """Stop recording."""
        if self._browser_bridge:
            await self._browser_bridge.set_mode(BrowserMode.IDLE)
            await self._browser_bridge.stop()

        await super().stop()
```

### 6.3 Event Storage Format

Browser events are stored in the same SQLite database as other events:

```sql
-- events table (existing)
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL  -- JSON
);

-- Example browser event data:
-- {
--   "type": "browser.click",
--   "url": "https://example.com",
--   "tab_id": 123,
--   "clientX": 150,
--   "clientY": 200,
--   "element": {
--     "role": "button",
--     "name": "Submit",
--     ...
--   }
-- }
```

---

## 7. Installation & Usage

### 7.1 Installing the Extension

1. **Build the extension** (included in openadapt-capture):
   ```bash
   # Extension files are in: openadapt_capture/chrome_extension/
   ```

2. **Load in Chrome**:
   - Navigate to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select `path/to/openadapt-capture/chrome_extension/`

3. **Verify installation**:
   - Extension icon appears in toolbar
   - Background script shows "Connected" status

### 7.2 Starting the Capture Server

```bash
# Start capture with browser bridge
capture record --name my_session --browser

# Or programmatically
python -c "
import asyncio
from openadapt_capture.recorder import Recorder

async def main():
    recorder = Recorder(capture_dir='./my_session')
    await recorder.start_with_browser()
    # Recording runs until Ctrl+C
    await asyncio.Event().wait()

asyncio.run(main())
"
```

### 7.3 Developer Workflow

1. **Start capture**:
   ```bash
   capture record --name demo --browser --task "Book a flight"
   ```

2. **Perform task in browser**:
   - Extension captures all interactions with element refs
   - Native capture records screenshots

3. **Stop recording** (Ctrl+C or GUI):
   - Events merged and processed
   - Screenshots aligned to events

4. **Review capture**:
   ```bash
   capture view ./demo
   # Shows timeline with browser events and element refs
   ```

5. **Export for training**:
   ```python
   from openadapt_capture import CaptureSession

   session = CaptureSession.load("./demo")
   for action in session.actions():
       print(f"{action.type}: {action.element.name if action.element else 'N/A'}")
   ```

---

## 8. Testing Strategy

### 8.1 Unit Tests for content.js

```javascript
// tests/content.test.js

describe('SemanticElementRef extraction', () => {
  test('extracts role from button element', () => {
    document.body.innerHTML = '<button id="test">Click me</button>';
    const element = document.getElementById('test');
    const ref = extractSemanticElementRef(element);
    expect(ref.role).toBe('button');
    expect(ref.name).toBe('Click me');
  });

  test('extracts role from ARIA attribute', () => {
    document.body.innerHTML = '<div id="test" role="checkbox">Option</div>';
    const element = document.getElementById('test');
    const ref = extractSemanticElementRef(element);
    expect(ref.role).toBe('checkbox');
  });

  test('generates correct XPath', () => {
    document.body.innerHTML = `
      <div>
        <span>First</span>
        <span id="target">Second</span>
      </div>
    `;
    const element = document.getElementById('target');
    const xpath = getXPath(element);
    expect(document.evaluate(xpath, document, null,
      XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue)
      .toBe(element);
  });

  test('generates minimal CSS selector', () => {
    document.body.innerHTML = '<button id="unique-btn">Submit</button>';
    const element = document.getElementById('unique-btn');
    const selector = getMinimalSelector(element);
    expect(selector).toBe('#unique-btn');
  });

  test('captures click event with element ref', async () => {
    const events = [];
    mockChromeRuntime({ sendMessage: (msg) => events.push(msg) });

    document.body.innerHTML = '<button id="test">Click</button>';
    const element = document.getElementById('test');

    element.click();
    await flushPromises();

    expect(events).toHaveLength(1);
    expect(events[0].payload.eventType).toBe('browser.click');
    expect(events[0].payload.element.role).toBe('button');
  });
});
```

### 8.2 Integration Tests with Python Server

```python
# tests/test_browser_bridge.py

import asyncio
import pytest
import websockets
import json

from openadapt_capture.browser_bridge import BrowserBridge, BrowserMode


@pytest.fixture
async def bridge():
    bridge = BrowserBridge(port=8766)
    await bridge.start()
    yield bridge
    await bridge.stop()


@pytest.mark.asyncio
async def test_client_connection(bridge):
    """Test extension can connect and receive mode."""
    async with websockets.connect("ws://localhost:8766") as ws:
        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
        data = json.loads(message)
        assert data["type"] == "SET_MODE"
        assert data["payload"]["mode"] == "idle"


@pytest.mark.asyncio
async def test_mode_broadcast(bridge):
    """Test mode changes broadcast to all clients."""
    async with websockets.connect("ws://localhost:8766") as ws1:
        async with websockets.connect("ws://localhost:8766") as ws2:
            # Drain initial mode messages
            await ws1.recv()
            await ws2.recv()

            # Change mode
            await bridge.set_mode(BrowserMode.RECORD)

            # Both clients should receive
            for ws in [ws1, ws2]:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(message)
                assert data["payload"]["mode"] == "record"


@pytest.mark.asyncio
async def test_dom_event_storage(bridge, tmp_path):
    """Test DOM events are stored correctly."""
    from openadapt_capture.storage import CaptureStorage

    storage = CaptureStorage(tmp_path / "capture.db")
    bridge.storage = storage
    await bridge.set_mode(BrowserMode.RECORD)

    async with websockets.connect("ws://localhost:8766") as ws:
        await ws.recv()  # Drain mode message

        # Send DOM event
        event = {
            "type": "DOM_EVENT",
            "timestamp": 1704067200000,
            "tabId": 123,
            "payload": {
                "eventType": "browser.click",
                "url": "https://example.com",
                "clientX": 100,
                "clientY": 200,
                "element": {
                    "role": "button",
                    "name": "Submit"
                }
            }
        }
        await ws.send(json.dumps(event))

        # Allow processing
        await asyncio.sleep(0.1)

        # Verify storage
        events = storage.get_events(event_types=["browser.click"])
        assert len(events) == 1
        assert events[0].payload["element"]["role"] == "button"
```

### 8.3 Manual Testing Checklist

#### Extension Installation
- [ ] Extension loads without errors in Chrome
- [ ] Icon appears in toolbar
- [ ] No console errors on installation

#### Recording Mode
- [ ] Mode changes propagate from server to extension
- [ ] Click events captured with element refs
- [ ] Keyboard events captured correctly
- [ ] Scroll events captured
- [ ] Form input events captured
- [ ] Navigation events captured
- [ ] Events have correct timestamps

#### Element Identification
- [ ] Role extracted from semantic HTML
- [ ] Role extracted from ARIA attributes
- [ ] Accessible name extracted correctly
- [ ] XPath resolves to correct element
- [ ] CSS selector resolves to correct element
- [ ] Bounding boxes are accurate

#### WebSocket Communication
- [ ] Extension reconnects after server restart
- [ ] Multiple tabs handled correctly
- [ ] Large DOM snapshots transfer successfully
- [ ] Error handling works (invalid messages, disconnects)

#### Integration with Capture
- [ ] Browser events merge with OS events
- [ ] Timestamps align correctly
- [ ] Capture session includes browser events
- [ ] Replay can use element refs

---

## 9. File Structure

```
openadapt-capture/
├── chrome_extension/
│   ├── manifest.json          # Extension configuration (Manifest V3)
│   ├── content.js             # DOM event capture, element ref extraction
│   ├── background.js          # Service worker, WebSocket bridge
│   ├── popup.html             # Extension popup UI (optional)
│   ├── popup.js               # Popup logic (optional)
│   └── icons/
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
│
├── openadapt_capture/
│   ├── __init__.py
│   ├── browser_bridge.py      # WebSocket server
│   ├── browser_events.py      # Browser event schemas
│   ├── capture.py             # Updated with browser support
│   ├── events.py              # Core event schemas
│   ├── recorder.py            # Updated with browser bridge
│   └── storage.py             # Event storage
│
├── tests/
│   ├── test_browser_bridge.py # Python WebSocket tests
│   ├── test_browser_events.py # Event schema tests
│   └── js/
│       ├── content.test.js    # content.js unit tests
│       └── setup.js           # Test setup/mocks
│
└── docs/
    └── chrome_extension_design.md  # This document
```

---

## 10. Migration from Legacy

### 10.1 What to Copy

| File | Action | Notes |
|------|--------|-------|
| `manifest.json` | Adapt | Update to Manifest V3, adjust permissions |
| `background.js` | Rewrite | Keep WebSocket concept, modernize for service workers |
| `content.js` | Enhance | Add SemanticElementRef extraction |
| `icons/` | Copy | Reuse existing icon assets |

### 10.2 What to Rewrite

| Component | Changes |
|-----------|---------|
| **WebSocket handling** | Use native WebSocket in service worker (MV3) |
| **Mode synchronization** | Simplify tab message passing |
| **Coordinate mapping** | Remove linear regression, use page coordinates directly |
| **DOM capture** | Add full SemanticElementRef extraction |
| **Event format** | Align with openadapt-capture schemas |

### 10.3 Legacy Code Reference

From the existing `content.js`:
- **Keep**: Event listener structure, DOM visibility checks
- **Remove**: Complex coordinate transformation via linear regression
- **Enhance**: Add element ref extraction on every action

From the existing `background.js`:
- **Keep**: WebSocket connection pattern, mode state management
- **Remove**: Time offset synchronization (use monotonic timestamps)
- **Enhance**: Add DOM snapshot handling

### 10.4 Enhancements for SoM Mode

The re-implementation adds these capabilities for Set-of-Marks mode:

1. **Element ID generation**: Assign stable IDs to visible interactive elements
2. **Visible element enumeration**: On DOM changes, enumerate all actionable elements
3. **SoM overlay support**: Enable visual overlays showing element IDs
4. **ID-based action mapping**: Map `CLICK([1])` to specific elements

```javascript
// SoM mode enhancement in content.js
function generateSoMData() {
  const elements = getInteractiveElements();
  return elements.map((el, index) => ({
    id: index + 1,  // 1-indexed for user display
    element: extractSemanticElementRef(el),
    center: {
      x: el.getBoundingClientRect().left + el.offsetWidth / 2,
      y: el.getBoundingClientRect().top + el.offsetHeight / 2
    }
  }));
}

function getInteractiveElements() {
  const selectors = [
    'button', 'a', 'input', 'select', 'textarea',
    '[role="button"]', '[role="link"]', '[role="checkbox"]',
    '[onclick]', '[tabindex]:not([tabindex="-1"])'
  ];

  return Array.from(document.querySelectorAll(selectors.join(',')))
    .filter(isVisible)
    .filter(isInViewport);
}
```

---

## Appendix A: Message Type Reference

| Type | Direction | Description |
|------|-----------|-------------|
| `SET_MODE` | Server -> Extension | Set recording/replay/idle mode |
| `PING` | Server -> Extension | Keep-alive check |
| `PONG` | Extension -> Server | Keep-alive response |
| `EXECUTE_ACTION` | Server -> Extension | Execute action during replay |
| `DOM_EVENT` | Extension -> Server | Captured DOM event |
| `DOM_SNAPSHOT` | Extension -> Server | Full DOM snapshot |
| `ERROR` | Extension -> Server | Error notification |

## Appendix B: Event Type Reference

| Event Type | Description |
|------------|-------------|
| `browser.click` | Mouse click on element |
| `browser.keydown` | Key press |
| `browser.keyup` | Key release |
| `browser.scroll` | Scroll event |
| `browser.input` | Form input change |
| `browser.navigate` | Page navigation |
| `browser.focus` | Element focus |
| `browser.blur` | Element blur |

## Appendix C: Role Mapping Table

| HTML Element | Implicit Role |
|--------------|---------------|
| `<a href>` | link |
| `<button>` | button |
| `<input type="text">` | textbox |
| `<input type="checkbox">` | checkbox |
| `<input type="radio">` | radio |
| `<input type="submit">` | button |
| `<select>` | combobox |
| `<textarea>` | textbox |
| `<img>` | image |
| `<h1>` - `<h6>` | heading |
| `<nav>` | navigation |
| `<main>` | main |
| `<aside>` | complementary |
| `<footer>` | contentinfo |
| `<header>` | banner |
| `<article>` | article |
| `<section>` | region |
| `<form>` | form |
| `<table>` | table |
| `<ul>`, `<ol>` | list |
| `<li>` | listitem |
