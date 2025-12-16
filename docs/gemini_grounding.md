# Gemini Grounding Module

The Gemini grounding module uses Google's Gemini vision API to detect and locate UI elements on screenshots. It provides two main capabilities:

1. **Element Detection**: Extract all interactive UI elements with bounding boxes
2. **Element Grounding**: Find specific elements by natural language description

## Features

- **Zero-shot detection**: No training required, works on any UI
- **Multi-element extraction**: Detect all interactive elements in a single API call
- **Set-of-Marks (SoM) support**: Generate numbered labels for element-based actions
- **Natural language grounding**: Find elements by description (e.g., "login button")
- **Confidence scores**: Each detection includes a confidence estimate

## Setup

### 1. Install Dependencies

```bash
pip install google-generativeai
# or
uv pip install google-generativeai
```

### 2. Get API Key

Get a Gemini API key from one of these sources:

**Option A: Google AI Studio (Easiest)**
1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Select or create a Google Cloud project
4. Copy the key

**Option B: Google Cloud Console**
1. Go to https://console.cloud.google.com/apis/credentials
2. Select your project
3. Click "Create Credentials" → "API Key"
4. Enable "Generative Language API" at:
   https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com

### 3. Configure Environment

Add to your `.env` file:

```bash
GOOGLE_API_KEY=your-api-key-here
```

## Usage

### Extract All UI Elements

```python
from PIL import Image
from openadapt_ml.grounding import extract_ui_elements

# Load screenshot
screenshot = Image.open("login.png")

# Extract all interactive elements
elements = extract_ui_elements(screenshot)

# Print results
for elem in elements:
    print(f"[{elem['id']}] {elem['label']} ({elem['type']})")
    print(f"  bbox: {elem['bbox']}")
    print(f"  text: {elem['text']}")
```

Output:
```
[1] Username text field (text_field)
  bbox: [0.25, 0.30, 0.75, 0.38]
  text:
[2] Password text field (text_field)
  bbox: [0.25, 0.42, 0.75, 0.50]
  text:
[3] Login button (button)
  bbox: [0.35, 0.55, 0.65, 0.62]
  text: Login
```

### Overlay Set-of-Marks

```python
from openadapt_ml.grounding import overlay_element_marks

# Create marked screenshot with numbered labels
marked_img = overlay_element_marks(screenshot, elements, style="compact")
marked_img.save("login_marked.png")
```

This creates a screenshot with red circles containing [1], [2], [3], etc. overlaid on each element.

### Ground Specific Elements

```python
from openadapt_ml.grounding import GeminiGrounder

grounder = GeminiGrounder()

# Find the login button
candidates = grounder.ground(screenshot, "login button", k=3)

if candidates:
    best = candidates[0]
    print(f"Found at: {best.centroid}")
    print(f"Confidence: {best.confidence}")

    # Use the centroid for clicking
    click_x, click_y = best.centroid
```

## API Reference

### `extract_ui_elements(screenshot, model_name="gemini-2.0-flash", api_key=None)`

Extract all interactive UI elements from a screenshot.

**Parameters:**
- `screenshot` (PIL.Image): Screenshot to analyze
- `model_name` (str): Gemini model to use
  - `"gemini-2.0-flash"` - Fast, good for grounding (default)
  - `"gemini-2.5-flash"` - Faster, newer model
  - `"gemini-2.5-pro"` - Higher quality
- `api_key` (str, optional): Google API key (uses GOOGLE_API_KEY env var if None)

**Returns:**
List of element dictionaries:
```python
{
    "id": int,              # Sequential ID starting at 1
    "label": str,           # Descriptive name
    "bbox": [x1,y1,x2,y2], # Normalized coordinates [0,1]
    "type": str,           # Element type
    "text": str,           # Visible text content
}
```

**Element Types:**
- `"button"` - Clickable buttons
- `"text_field"` - Text input fields
- `"checkbox"` - Checkboxes and toggles
- `"link"` - Hyperlinks
- `"icon"` - Clickable icons
- `"dropdown"` - Dropdown menus
- `"tab"` - Tab controls
- `"menu_item"` - Menu items
- `"other"` - Other interactive elements

### `overlay_element_marks(screenshot, elements, style="compact")`

Overlay numbered labels (Set-of-Marks) on UI elements.

**Parameters:**
- `screenshot` (PIL.Image): Screenshot to annotate
- `elements` (list): Element list from `extract_ui_elements()`
- `style` (str): Label style
  - `"compact"` - Small red circles with numbers (default)
  - `"full"` - Bounding boxes with label boxes

**Returns:**
New PIL.Image with numbered labels overlaid.

### `GeminiGrounder(model="gemini-2.5-flash", api_key=None)`

Grounding module for finding specific elements by description.

**Methods:**

#### `ground(image, target_description, k=1)`

Locate regions matching a target description.

**Parameters:**
- `image` (PIL.Image): Screenshot to search
- `target_description` (str): Natural language description (e.g., "login button")
- `k` (int): Maximum number of candidates to return

**Returns:**
List of `RegionCandidate` objects sorted by confidence:
```python
RegionCandidate(
    bbox=(x1, y1, x2, y2),     # Normalized [0,1]
    centroid=(cx, cy),          # Click point
    confidence=0.95,            # Score [0,1]
    element_label="button",     # Element type
    text_content="Login",       # Text if any
    metadata={...}              # Additional data
)
```

## Integration with Policy

The grounding module integrates with the policy/grounding separation architecture:

```python
# Policy generates action intent
action_intent = policy(screenshot, goal, history)
# → {"action_type": "click", "target": "login button"}

# Grounding converts to executable action
grounder = GeminiGrounder()
candidates = grounder.ground(screenshot, action_intent["target"])
best = candidates[0]

# Execute the action
execute_action("click", x=best.centroid[0], y=best.centroid[1])
```

## Performance

- **Latency**: ~2-5 seconds per screenshot (API call)
- **Cost**: ~$0.01-0.05 per image (varies by model and image size)
- **Accuracy**: ~80-90% element detection rate on typical UIs
- **Scalability**: Limited by API rate limits (60 requests/minute for free tier)

## Comparison with Other Grounders

| Module | Strategy | Latency | Cost | Accuracy | Best For |
|--------|----------|---------|------|----------|----------|
| GeminiGrounder | Gemini API | ~2-5s | $0.01-0.05 | High | Real UIs, prototyping |
| OmniParser | Local detection | ~100ms | Free | Medium-High | Production, privacy |
| SoMGrounder | Pre-labeled indices | ~0ms | Free | Perfect | Synthetic, controlled UIs |
| AttentionGrounder | VLM attention | ~50ms | Free | Medium | Fine-tuned models |

## Limitations

1. **Internet required**: Must send screenshots to Google API
2. **Privacy considerations**: Screenshots sent to Google servers
3. **Cost**: Not free, charges per API call
4. **Latency**: Slower than local detection models
5. **Rate limits**: Free tier has 60 requests/minute limit
6. **Detection gaps**: May miss very small or obscured elements

## Example: Full Workflow

```python
from PIL import Image
from openadapt_ml.grounding import (
    GeminiGrounder,
    extract_ui_elements,
    overlay_element_marks,
)

# 1. Load screenshot
screenshot = Image.open("app.png")

# 2. Extract all elements for SoM
elements = extract_ui_elements(screenshot)
print(f"Found {len(elements)} elements")

# 3. Create marked screenshot
marked = overlay_element_marks(screenshot, elements)
marked.save("app_marked.png")

# 4. Ground specific target
grounder = GeminiGrounder()
candidates = grounder.ground(screenshot, "submit button")

if candidates:
    best = candidates[0]
    print(f"Click at: {best.centroid}")
    print(f"Confidence: {best.confidence}")

    # Element ID from SoM
    for elem in elements:
        if grounder.ground(screenshot, elem['label'])[0].iou(best) > 0.5:
            print(f"This is element [{elem['id']}] in SoM")
            break
```

## Troubleshooting

### "ImportError: google-generativeai required"

Install the package:
```bash
pip install google-generativeai
```

### "ValueError: GOOGLE_API_KEY not set"

Add to `.env`:
```bash
GOOGLE_API_KEY=your-key-here
```

### "No elements detected"

Possible causes:
1. Screenshot quality too low
2. Elements too small or obscured
3. API rate limit exceeded
4. Model failed to parse UI

Try:
- Use higher resolution screenshot
- Use `model="gemini-2.5-pro"` for better quality
- Wait and retry if rate limited

### "Failed to parse Gemini response"

The model may have returned malformed JSON. This is usually transient. Retry the call.

## References

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Set-of-Marks Paper](https://arxiv.org/abs/2310.11441)
- [GUI Grounding Architecture](docs/NEXT_STEPS_GROUNDING_ARCHITECTURE.md)
- [Example Script](examples/test_gemini_grounding.py)
