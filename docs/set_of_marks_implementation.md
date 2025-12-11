# Set-of-Marks (SoM) Implementation Plan

## Context and Diagnosis

We've confirmed that Claude's 0% bbox_hit_rate is **not an evaluation bug**—it's a fundamental **coordinate grounding failure**. The literature confirms this is expected behavior:

> "Continuous coordinate prediction can be challenging for LLMs due to their discrete token nature and lack of explicit spatial supervision... early baseline agents did attempt to generate raw coordinates, but this proved brittle: agents often mispredicted click locations and made repeated misclicks."
> — OSWorld paper

VideoAgentTrek needed **26 billion tokens** and **1.52 million labeled steps** to train a 7B model to reliably output pixel coordinates. We're asking Claude to do this zero-shot from a prompt description alone—that's why it fails.

The mean_coord_error of 0.29 (29% of screen) is only marginally better than random clicking (~0.38 expected error for uniform random). Claude can identify *what action type* to take (~52% accuracy) but cannot ground *where* to perform spatial actions.

## Current Results (Coordinate DSL)

| Model | action_type_accuracy | click_hit_rate | bbox_hit_rate | episode_success |
|-------|---------------------|----------------|---------------|-----------------|
| Claude | 51.6% | 0% | 0% | 0% |
| GPT-4.1 | 56.7% | 71.7% | TBD | 0% |
| Qwen2.5-VL Base | 29.5% | 0% | N/A | 0% |
| Qwen3 Fine-tuned | 37.1% | 37.3% | N/A | 0% |

## The Solution: Set-of-Marks (SoM)

The industry-standard solution, used by OSWorld, Claude Computer Use, and OpenAI Operator, is **Set-of-Marks**:

1. **Overlay numbered labels** on interactive UI elements in the screenshot (e.g., `[1]` on username field, `[2]` on password field, `[3]` on login button)
2. **Change the DSL** from coordinate-based to index-based:
   - Old: `CLICK(x=0.487, y=0.328)`
   - New: `CLICK([1])`
3. **Environment maps** the index back to coordinates for execution

This "dramatically simplifies the prediction problem—the model just chooses an element ID—and prevents small coordinate errors from causing big misses."

## Implementation Plan

### Phase 1: Synthetic Benchmark (Immediate Priority)

For our synthetic login benchmark, we already have element positions. Implement SoM by:

#### 1. Modify `_draw_login_screen()` in `openadapt_ml/ingest/synthetic.py`

After rendering the login screen, overlay numbered labels on each interactive element:
- Use a visible but non-obstructive style (e.g., small red circle with white number, positioned at top-left corner of each element's bbox)
- Elements to label:
  - `[1]` → Username text field
  - `[2]` → Password text field
  - `[3]` → Login button
  - Optionally: `[4]` → "Remember Me" checkbox, `[5]` → "Forgot Password?" link

#### 2. Update the Action schema in `openadapt_ml/schemas/sessions.py`

Add `element_index: Optional[int]` field to Action:
- Keep `x, y, bbox` for ground truth storage and Qwen fine-tuning (which learns coordinates)

#### 3. Create a new DSL variant for index-based actions

Update `format_action()` and `parse_action()` in `openadapt_ml/datasets/next_action.py`:
- `CLICK([1])` — click element with index 1
- `TYPE([2], "password123")` — type into element 2
- `DONE()` — unchanged

#### 4. Update the system prompt for SoM

```
The screenshot shows numbered labels [1], [2], [3], etc. on interactive elements.
To click an element, use CLICK([N]) where N is the element's number.
To type into a field, first reference it by number: TYPE([N], "text")
```

#### 5. Run comparative evals

- Claude with SoM DSL (expect dramatic improvement)
- GPT-4.1 with SoM DSL
- Keep Qwen fine-tuned on coordinate DSL (it learned the mapping from training data)

### Phase 2: Real UI Support (Future)

For real-world UIs where we don't have ground-truth element positions, there are two approaches:

#### Option A: Gemini for Element Extraction (Recommended for Prototyping)

Google's Gemini 2.0+ and Gemini 3 can directly output bounding boxes from screenshots. This is simpler than integrating a separate detection model.

```python
# openadapt_ml/grounding/gemini_parser.py

import google.generativeai as genai
from PIL import Image
import json
from typing import List, Dict

def extract_ui_elements(screenshot: Image.Image, model_name: str = "gemini-2.0-flash") -> List[Dict]:
    """
    Use Gemini to extract interactive UI elements with bounding boxes.

    Returns list of:
    {
        "id": 1,
        "label": "Username text field",
        "bbox": [x_min, y_min, x_max, y_max],  # normalized [0,1]
        "type": "text_field"
    }
    """
    model = genai.GenerativeModel(model_name)

    prompt = """Analyze this screenshot and list ALL interactive UI elements.

For each element, output a JSON object with:
- id: sequential integer starting at 1
- label: descriptive name (e.g., "Login button", "Username text field")
- bbox: bounding box as [x_min, y_min, x_max, y_max] in normalized coordinates where (0,0) is top-left and (1,1) is bottom-right
- type: one of "button", "text_field", "checkbox", "link", "icon", "dropdown", "tab", "menu_item"

Output ONLY a valid JSON array, no markdown formatting, no explanation.

Example output format:
[
  {"id": 1, "label": "Username text field", "bbox": [0.25, 0.30, 0.75, 0.38], "type": "text_field"},
  {"id": 2, "label": "Password text field", "bbox": [0.25, 0.42, 0.75, 0.50], "type": "text_field"}
]"""

    response = model.generate_content([screenshot, prompt])

    # Parse JSON response
    try:
        elements = json.loads(response.text)
        return elements
    except json.JSONDecodeError:
        # Try to extract JSON from response if it has extra text
        import re
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Failed to parse Gemini response: {response.text}")


def overlay_element_marks(screenshot: Image.Image, elements: List[Dict]) -> Image.Image:
    """
    Overlay numbered labels on screenshot at each element's position.
    Returns a new image with [1], [2], [3] markers.
    """
    from PIL import ImageDraw, ImageFont

    img = screenshot.copy()
    draw = ImageDraw.Draw(img)

    width, height = img.size

    for elem in elements:
        bbox = elem["bbox"]
        idx = elem["id"]

        # Convert normalized coords to pixels
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)

        # Draw label at top-left of element
        label = f"[{idx}]"

        # Red circle background
        circle_radius = 12
        circle_center = (x_min + circle_radius, y_min + circle_radius)
        draw.ellipse(
            [circle_center[0] - circle_radius, circle_center[1] - circle_radius,
             circle_center[0] + circle_radius, circle_center[1] + circle_radius],
            fill="red"
        )

        # White text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        # Center text in circle
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = circle_center[0] - text_width // 2
        text_y = circle_center[1] - text_height // 2

        draw.text((text_x, text_y), label, fill="white", font=font)

    return img
```

**Tradeoffs of Gemini approach:**
- Pros: No model installation, just API key; Single API for detection AND action selection; Gemini 3 is SOTA for "screen understanding"
- Cons: Cost (~$0.01-0.05 per image); Latency (~2-5 seconds per image); Requires internet, sends screenshots to Google

#### Option B: OmniParser (Better for Production)

Microsoft's OmniParser is the open-source SOTA for vision-only UI element detection:
- YOLOv8-Nano backbone (~5MB)
- PaddleOCR for text regions
- Florence VLM for icon captioning
- ~100ms inference on GPU
- ~90% detection rate on typical UIs
- GitHub: `microsoft/OmniParser`

```python
# openadapt_ml/grounding/omniparser.py

from omniparser import OmniParser

parser = OmniParser()

def extract_ui_elements(screenshot_path: str) -> List[Dict]:
    """Extract UI elements using OmniParser."""
    elements = parser.parse(screenshot_path)
    # Returns: [{bbox: [x1,y1,x2,y2], type: "icon"|"text", interactivity: bool, content: str}]

    # Convert to our format with sequential IDs
    result = []
    for i, elem in enumerate(elements, start=1):
        if elem.get("interactivity", True):  # Only interactive elements
            result.append({
                "id": i,
                "label": elem.get("content", f"Element {i}"),
                "bbox": elem["bbox"],  # Already normalized
                "type": elem.get("type", "unknown")
            })
    return result
```

**Tradeoffs of OmniParser:**
- Pros: Free, runs locally; Fast (~100ms); Privacy-preserving
- Cons: Requires installing model weights; Detection only, no reasoning

## Updated Evaluation Strategy

### Dual DSL Support

Maintain **two DSL modes** in the codebase:

1. **Coordinate DSL** (for fine-tuned models like Qwen):
   ```
   CLICK(x=0.487, y=0.328)
   TYPE(text="alice")
   DONE()
   ```

2. **Index DSL** (for API models like Claude, GPT-4, Gemini):
   ```
   CLICK([1])
   TYPE([2], "alice")
   DONE()
   ```

The evaluation harness should support both:
- For coordinate DSL: use existing bbox_hit_rate, coord_error metrics
- For index DSL: check if predicted index matches ground truth index (simpler, more reliable)

### New Metrics for Index DSL

```python
@dataclass
class IndexDSLMetrics:
    element_accuracy: float  # % of steps where predicted index == GT index
    action_type_accuracy: float  # unchanged
    episode_success_rate: float  # all indices correct in sequence
```

### Comparative Eval Matrix

Run the following experiments:

| Model | DSL | Expected Performance |
|-------|-----|---------------------|
| Claude 3.5 Sonnet | Coordinate | ~0% bbox hit (baseline, already measured) |
| Claude 3.5 Sonnet | Index (SoM) | **60-80%+ element accuracy** (expected) |
| GPT-4.1 | Coordinate | Measure bbox_hit_rate |
| GPT-4.1 | Index (SoM) | **60-80%+ element accuracy** (expected) |
| Qwen3 Fine-tuned | Coordinate | 37% click_hit_rate (already measured) |
| Qwen3 Fine-tuned | Index (SoM) | Not applicable (trained on coordinates) |

This will establish:
1. Whether SoM fixes API model performance (expected: yes)
2. The upper bound achievable with API models on this benchmark
3. Whether fine-tuning Qwen on coordinates is actually competitive with SoM + API models

## Concrete Next Steps (Priority Order)

1. **Implement SoM overlay for synthetic login screens** (~30 min)
   - Modify `_draw_login_screen()` to add numbered labels
   - Store element-to-index mapping in episode metadata

2. **Add Index DSL parsing** (~20 min)
   - Update `format_action()` and `parse_action()`
   - Add `element_index` field to Action schema

3. **Update system prompt for SoM** (~10 min)
   - Explain the numbered labels
   - Give examples of `CLICK([1])`, `TYPE([2], "text")`

4. **Run Claude eval with SoM** (~5 min)
   - Expect dramatic improvement over 0% bbox hit rate
   - Target: >50% element accuracy

5. **Run GPT-4.1 eval with SoM** for comparison

6. **Document findings** and update the results table

## Key Literature References

For context on why this approach is standard:

- **OSWorld** uses tag-based actions: `Click [tag_2]` instead of coordinates
- **VideoAgentTrek** acknowledges coordinate prediction requires massive training data (26B tokens)
- **OmniParser** + GPT-4V is Microsoft's recommended approach for reliable GUI agents
- **GUI-Actor (NeurIPS 2025)** explores attention-based grounding to avoid coordinate outputs entirely

The conclusion from the literature: **raw coordinate prediction without fine-tuning is fundamentally unreliable for current LLMs**. Set-of-Marks is the proven solution.

## Files to Modify

| File | Changes |
|------|---------|
| `openadapt_ml/ingest/synthetic.py` | Add SoM overlay to `_draw_login_screen()` |
| `openadapt_ml/schemas/sessions.py` | Add `element_index` field to Action |
| `openadapt_ml/datasets/next_action.py` | Add Index DSL format/parse functions |
| `openadapt_ml/evals/trajectory_matching.py` | Add index-based metrics |
| `openadapt_ml/scripts/eval_policy.py` | Support `--dsl-mode` flag |
