# Gemini Grounding Quick Start

**5-minute guide to using GeminiGrounder for UI element detection**

## 1. Setup (One-Time)

### Install Dependencies
```bash
uv pip install google-generativeai
```

### Get API Key
1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Copy the key

### Configure
Add to `.env`:
```bash
GOOGLE_API_KEY=your-key-here
```

## 2. Three Ways to Use It

### A. Extract All Elements (Set-of-Marks)

```python
from PIL import Image
from openadapt_ml.grounding import extract_ui_elements, overlay_element_marks

# Load screenshot
screenshot = Image.open("app.png")

# Get all interactive elements
elements = extract_ui_elements(screenshot)
# Returns: [
#   {"id": 1, "label": "Username field", "bbox": [0.2, 0.3, 0.8, 0.4], ...},
#   {"id": 2, "label": "Password field", "bbox": [0.2, 0.5, 0.8, 0.6], ...},
#   ...
# ]

# Overlay numbered labels
marked = overlay_element_marks(screenshot, elements)
marked.save("app_marked.png")
```

**Use case**: Generate Set-of-Marks overlays for element-based actions.

### B. Find Specific Element

```python
from openadapt_ml.grounding import GeminiGrounder

grounder = GeminiGrounder()

# Find a specific element
candidates = grounder.ground(screenshot, "login button", k=3)

# Use the best match
best = candidates[0]
click_x, click_y = best.centroid
print(f"Click at ({click_x}, {click_y}) with {best.confidence:.0%} confidence")
```

**Use case**: Ground natural language target descriptions to coordinates.

### C. Complete Workflow (Extract + Ground)

```python
# 1. Extract all elements
elements = extract_ui_elements(screenshot)

# 2. Create SoM overlay
marked = overlay_element_marks(screenshot, elements)

# 3. Ground a specific target
grounder = GeminiGrounder()
candidates = grounder.ground(screenshot, "submit button")

# 4. Match to SoM element ID
if candidates:
    best = candidates[0]
    for elem in elements:
        # Simple centroid-based matching
        elem_center_x = (elem['bbox'][0] + elem['bbox'][2]) / 2
        elem_center_y = (elem['bbox'][1] + elem['bbox'][3]) / 2
        if abs(elem_center_x - best.centroid[0]) < 0.01 and \
           abs(elem_center_y - best.centroid[1]) < 0.01:
            print(f"Target is element [{elem['id']}]")
            break
```

**Use case**: Full policy/grounding separation with SoM support.

## 3. Run the Example

```bash
# Test on your own screenshot
python examples/test_gemini_grounding.py screenshots/login.png

# Output:
# Found 3 interactive elements:
#   [1] Username text field (text_field) at [0.25, 0.30, 0.75, 0.38]
#   [2] Password text field (text_field) at [0.25, 0.42, 0.75, 0.50]
#   [3] Login button (button) at [0.35, 0.55, 0.65, 0.62]
# Saved marked screenshot to: screenshots/login_marked.png
```

## 4. Integration Patterns

### With Policy Module

```python
from openadapt_ml.runtime.policy import AgentPolicy
from openadapt_ml.grounding import GeminiGrounder

# Policy decides what to do
action_intent = policy.predict(screenshot, goal, history)
# → {"action_type": "click", "target": "login button"}

# Grounding finds where to do it
grounder = GeminiGrounder()
candidates = grounder.ground(screenshot, action_intent["target"])
best = candidates[0]

# Execute
execute_click(best.centroid[0], best.centroid[1])
```

### With Synthetic Episodes

```python
from openadapt_ml.grounding import extract_ui_elements, overlay_element_marks

# Add SoM overlay to synthetic screenshots
for episode in synthetic_episodes:
    for step in episode.steps:
        elements = extract_ui_elements(step.screenshot)
        step.screenshot = overlay_element_marks(step.screenshot, elements)
        step.metadata["som_elements"] = elements
```

### With Real Workflows

```python
from openadapt_ml.grounding import GeminiGrounder

# Evaluate grounding on real recordings
grounder = GeminiGrounder()

for step in real_workflow.steps:
    candidates = grounder.ground(
        step.screenshot,
        step.action.target_description
    )

    # Compare with ground truth
    if step.action.bbox:
        predicted_iou = candidates[0].iou_with_bbox(step.action.bbox)
        print(f"Step {step.index}: IoU = {predicted_iou:.2f}")
```

## 5. Common Patterns

### Check if element exists
```python
candidates = grounder.ground(screenshot, "logout button")
if not candidates:
    print("Logout button not found!")
```

### Get multiple candidates
```python
candidates = grounder.ground(screenshot, "text field", k=5)
for i, c in enumerate(candidates, 1):
    print(f"Candidate {i}: {c.element_label} at {c.centroid}")
```

### Custom model selection
```python
# Use faster model
grounder = GeminiGrounder(model="gemini-2.5-flash")

# Use higher quality model
grounder = GeminiGrounder(model="gemini-2.5-pro")
```

### Element filtering
```python
elements = extract_ui_elements(screenshot)

# Filter by type
buttons = [e for e in elements if e['type'] == 'button']
text_fields = [e for e in elements if e['type'] == 'text_field']

# Filter by size (large elements only)
large_elements = [e for e in elements
                  if (e['bbox'][2] - e['bbox'][0]) > 0.1]
```

## 6. Troubleshooting

### No elements detected
```python
elements = extract_ui_elements(screenshot)
if not elements:
    # Try higher quality model
    elements = extract_ui_elements(screenshot, model_name="gemini-2.5-pro")
```

### Rate limit exceeded
```python
import time

for screenshot in screenshots:
    try:
        elements = extract_ui_elements(screenshot)
    except Exception as e:
        if "quota" in str(e).lower():
            time.sleep(60)  # Wait 1 minute
            elements = extract_ui_elements(screenshot)
```

### Low confidence results
```python
candidates = grounder.ground(screenshot, "submit button", k=5)

# Check top-3 confidence
for c in candidates[:3]:
    if c.confidence > 0.8:
        print(f"High confidence match: {c.centroid}")
    else:
        print(f"Low confidence: {c.confidence:.0%}")
```

## 7. Performance Tips

1. **Batch similar screenshots**: Process multiple screenshots in parallel (API has rate limits)
2. **Cache results**: Store extracted elements to avoid re-detection
3. **Use faster model**: `gemini-2.5-flash` for speed, `gemini-2.5-pro` for quality
4. **Reduce image size**: Resize large screenshots before sending (but not too small!)

## 8. Next Steps

- **Full Documentation**: See `docs/gemini_grounding.md`
- **Architecture**: See `docs/NEXT_STEPS_GROUNDING_ARCHITECTURE.md`
- **Example Script**: Run `examples/test_gemini_grounding.py`
- **API Reference**: Check docstrings in `openadapt_ml/grounding/detector.py`

## 9. Cost & Performance

| Model | Latency | Cost/Image | Quality | Best For |
|-------|---------|------------|---------|----------|
| gemini-2.0-flash | ~2s | ~$0.01 | Good | General use |
| gemini-2.5-flash | ~1.5s | ~$0.01 | Good | Speed priority |
| gemini-2.5-pro | ~4s | ~$0.05 | Best | Quality priority |

**Free tier**: 60 requests/minute, 1500 requests/day

## 10. Common Issues

| Issue | Solution |
|-------|----------|
| API key not found | Add `GOOGLE_API_KEY` to `.env` file |
| Import error | Run `uv pip install google-generativeai` |
| No elements found | Screenshot may be too blurry or small |
| Wrong elements detected | Try different model or provide better screenshot |
| Rate limit hit | Wait 60 seconds or use batch processing |

---

**That's it!** You now have zero-shot UI element detection working.

For questions or issues, see the full documentation in `docs/gemini_grounding.md`.
