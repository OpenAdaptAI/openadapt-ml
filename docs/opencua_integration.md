# OpenCUA and VideoAgentTrek-CUA Integration

## Overview

This document outlines the potential integration of xlangai's OpenCUA models into openadapt-ml as alternative VLM adapters for GUI automation tasks.

## Model Variants

### 1. OpenCUA-7B
- **HuggingFace**: `xlangai/OpenCUA-7B`
- **Base Architecture**: Qwen2.5-VL-7B-Instruction
- **Parameters**: 7 billion
- **Training Data**: AgentNet - 22.6K human-annotated computer-use tasks
- **Key Strength**: Clean, high-quality human annotations from AgentNet dataset
- **Multi-OS Support**: Ubuntu, Windows, macOS

### 2. OpenCUA-32B
- **HuggingFace**: `xlangai/OpenCUA-32B`
- **Parameters**: 32 billion
- **Performance**: SOTA among open-source models (34.8% on OSWorld-Verified @ 100 steps)
- **Note**: Likely too large for most training scenarios, but useful for baseline evaluation

### 3. VideoAgentTrek-CUA-7B
- **HuggingFace**: `xlangai/VideoAgentTrek-CUA-7B`
- **Base**: OpenCUA-7B with additional pretraining
- **Parameters**: 8 billion (listed as 8B on HF, likely same architecture)
- **Training Addition**: Pretrained on video-mined trajectories
- **License**: Apache 2.0
- **Key Tradeoff**: More computer-use signal from video mining, but noisier labels than pure AgentNet

**Note**: VideoAgentTrek-CUA-7B's model card is currently empty on HuggingFace, so details about the video pretraining methodology, dataset size, and performance comparisons are not yet publicly documented.

## Key Differences

### OpenCUA-7B
- **Data Quality**: Highest quality - human-annotated via AgentNetTool
- **Data Source**: AgentNet dataset with verified action sequences
- **Action Processing**: Sophisticated action reduction pipeline
  - Merges low-level mouse moves into semantic clicks
  - Coalesces scroll events
  - Groups keypresses into semantic text input
- **CoT Generation**: Reflective Chain-of-Thought with:
  - Reflection on previous action
  - Explanation of action choice
  - Alternative actions considered
  - Forecast of expected next state

### VideoAgentTrek-CUA-7B
- **Data Quality**: Lower quality - video-mined trajectories (noisier labels)
- **Data Source**: OpenCUA-7B + additional video-mined computer-use data
- **Hypothesis**: More exposure to computer-use patterns, but less precise annotations
- **Use Case**: Test if additional noisy pretraining helps downstream task performance

## AgentNet Dataset Details

The AgentNet dataset represents the first large-scale desktop agent trajectory dataset with human annotations:

### Scale
- **22.6K tasks** across Windows, macOS, Ubuntu
- Multi-OS coverage ensures diverse UI patterns
- Human-annotated for quality and correctness

### Collection Pipeline

#### 1. AgentNetTool (Annotation Interface)
Cross-platform GUI recorder that captures:
- **Screen video** with synchronized timestamps
- **Mouse and keyboard events** at raw level
- **Accessibility trees** for semantic grounding
- **In-browser review interface** for trim and submission
- Available for Windows, macOS, Ubuntu

#### 2. DataProcessor (Action Reduction)
Converts raw event streams into semantic actions:
```
Raw Events → Action Reduction → Semantic PyAutoGUI actions
           → State-Action Matching → Aligned state-action pairs
```

**Action Reduction Logic**:
- Mouse moves followed by clicks → single click action
- Multiple scroll events → single scroll action with combined delta
- Keypresses → text input strings

**State-Action Matching**:
- Aligns actions with the last visually distinct frame before action
- Ensures training samples have clear cause-effect relationships

#### 3. CoTGenerator (Reflective CoT Synthesis)
Generates step-level reasoning containing:
1. **Reflection on previous action**: What was accomplished?
2. **Explanation of current action**: Why this action given observation/history?
3. **Alternative actions considered**: What else could be done?
4. **Expected next state forecast**: What should happen after this action?

This rich reasoning signal can improve model planning and error recovery.

## Performance Benchmarks

### OSWorld-Verified (Online Agent Evaluation)
End-to-end agent performance on multi-step desktop tasks:

| Model | 15 Steps | 50 Steps | 100 Steps |
|-------|----------|----------|-----------|
| OpenCUA-7B | 24.3% | 27.9% | 26.6% |
| OpenCUA-32B | 29.7% | 34.1% | **34.8%** |
| GPT-4o | 16.8% | 20.5% | 22.5% |
| Claude 3.7 Sonnet | 27.1% | 35.8% | 35.9% |

**Key Insight**: OpenCUA-32B achieves SOTA among open-source models, outperforming GPT-4o and approaching Claude 3.7 Sonnet performance.

### GUI Grounding Performance
Element localization and spatial reasoning:

| Model | OSWorld-G | ScreenSpot-V2 | ScreenSpot-Pro |
|-------|-----------|---------------|----------------|
| OpenCUA-7B | 55.3% | 92.3% | 50.0% |
| OpenCUA-32B | 59.6% | 93.4% | 55.3% |

### AgentNetBench (Offline Evaluation)
Action prediction accuracy on held-out AgentNet trajectories:

| Model | Coordinate | Content | Function | Average |
|-------|-----------|---------|----------|---------|
| OpenCUA-7B | 79.0% | 62.0% | 44.3% | **75.2%** |
| OpenCUA-32B | 81.9% | 66.1% | 55.7% | 79.1% |

**Note**: VideoAgentTrek-CUA-7B benchmarks not yet available.

## Action Format / DSL

OpenCUA models output executable PyAutoGUI commands as text:

### Supported Actions
```python
# Click actions
pyautogui.click(x=960, y=324)
pyautogui.doubleClick(x=500, y=200)
pyautogui.rightClick(x=300, y=400)

# Mouse movement
pyautogui.moveTo(x=500, y=200)

# Keyboard actions
pyautogui.write("text input")
pyautogui.press('enter')
pyautogui.hotkey('ctrl', 'c')

# Scrolling
pyautogui.scroll(-5)  # negative = scroll down
pyautogui.scroll(3)   # positive = scroll up
```

### Coordinate System

**Important**: OpenCUA uses absolute coordinates post-resize, not normalized [0,1] coordinates.

#### Qwen2.5-based Models (OpenCUA-7B, OpenCUA-32B)
- Output **absolute pixel coordinates** after smart resize
- Require conversion to original image dimensions for execution

```python
def qwen25_smart_resize_to_absolute(model_x, model_y, original_width, original_height):
    """Convert OpenCUA model coordinates to original image coordinates.

    Args:
        model_x, model_y: Coordinates from model output
        original_width, original_height: Original screenshot dimensions

    Returns:
        Tuple of (absolute_x, absolute_y) for original image
    """
    # Calculate smart-resized dimensions
    resized_height, resized_width = smart_resize(
        original_height, original_width,
        factor=28,
        min_pixels=3136,
        max_pixels=12845056
    )

    # Convert to relative coordinates
    rel_x = model_x / resized_width
    rel_y = model_y / resized_height

    # Convert to absolute coordinates on original image
    abs_x = int(rel_x * original_width)
    abs_y = int(rel_y * original_height)
    return abs_x, abs_y
```

#### Smart Resize Algorithm
Qwen2.5-VL uses dynamic image resizing with constraints:
- **Factor**: 28 (divisibility requirement for vision encoder)
- **Min pixels**: 3136 (56x56 minimum)
- **Max pixels**: 12845056 (limits memory usage)

This differs from openadapt-ml's normalized [0,1] coordinate system.

### Comparison to OpenAdapt-ML Actions

Current openadapt-ml action format:
```python
Action(
    type="click",
    x=0.42,  # normalized [0, 1]
    y=0.73,  # normalized [0, 1]
    text=None,
    raw=None
)
```

OpenCUA output format:
```python
"pyautogui.click(x=960, y=324)"  # absolute pixels post-resize
```

**Integration Requirement**: Parse PyAutoGUI commands and convert coordinates to normalized [0,1] format for openadapt-ml schema.

## Proposed Integration

### Architecture Changes

Add `OpenCUAAdapter` class similar to `QwenVLAdapter`:

```python
# openadapt_ml/models/opencua.py

from openadapt_ml.models.base_adapter import BaseVLMAdapter
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torch
from typing import Dict, Any, List

class OpenCUAAdapter(BaseVLMAdapter):
    """Adapter for OpenCUA models (OpenCUA-7B, VideoAgentTrek-CUA-7B).

    OpenCUA models are based on Qwen2.5-VL but use modified architecture:
    - 1D RoPE instead of Multimodal RoPE
    - Kimi-VL tokenizer and chat template
    - PyAutoGUI-compatible action output format
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        lora_config: Optional[LoraConfig | Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> "OpenCUAAdapter":
        """Load OpenCUA model with custom loading logic.

        Important: OpenCUA models cannot use default transformers classes
        due to modified M-RoPE and tokenizer. Must use trust_remote_code=True.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Apply LoRA if specified
        if lora_config is not None:
            # Similar to QwenVLAdapter LoRA logic
            pass

        return cls(model=model, processor=(tokenizer, image_processor), device=device)

    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert SFT samples to OpenCUA input format.

        OpenCUA expects system prompt:
        "You are a GUI agent. You are given a task and a screenshot of the screen.
         You need to perform a series of pyautogui actions to complete the task."
        """
        # Build messages with OpenCUA system prompt
        # Process images with image_processor
        # Return tokenized inputs with labels
        pass

    def generate(self, sample: Dict[str, Any], max_new_tokens: int = 64) -> str:
        """Generate PyAutoGUI action string."""
        # Use OpenCUA chat template
        # Generate with model
        # Return raw PyAutoGUI command string
        pass

    def _parse_action(self, output_text: str, original_width: int, original_height: int) -> Action:
        """Parse PyAutoGUI command into openadapt-ml Action.

        Args:
            output_text: Raw model output (e.g., "pyautogui.click(x=960, y=324)")
            original_width: Original screenshot width in pixels
            original_height: Original screenshot height in pixels

        Returns:
            Action with normalized [0,1] coordinates
        """
        import re

        # Parse different PyAutoGUI command patterns
        click_pattern = r'pyautogui\.(click|doubleClick|rightClick)\(x=(\d+),\s*y=(\d+)\)'
        write_pattern = r'pyautogui\.write\(["\'](.+?)["\']\)'
        press_pattern = r'pyautogui\.press\(["\'](.+?)["\']\)'
        scroll_pattern = r'pyautogui\.scroll\((-?\d+)\)'

        if match := re.search(click_pattern, output_text):
            action_type, x_abs, y_abs = match.groups()

            # Convert absolute coordinates to normalized [0,1]
            x_norm, y_norm = self._convert_opencua_coordinates(
                int(x_abs), int(y_abs),
                original_width, original_height
            )

            type_map = {
                "click": "click",
                "doubleClick": "double_click",
                "rightClick": "right_click"
            }

            return Action(
                type=type_map[action_type],
                x=x_norm,
                y=y_norm,
                text=None,
                raw={"original_output": output_text}
            )

        elif match := re.search(write_pattern, output_text):
            text = match.group(1)
            return Action(
                type="type",
                x=None,
                y=None,
                text=text,
                raw={"original_output": output_text}
            )

        elif match := re.search(press_pattern, output_text):
            key = match.group(1)
            return Action(
                type="key_press",
                x=None,
                y=None,
                text=key,
                raw={"original_output": output_text}
            )

        elif match := re.search(scroll_pattern, output_text):
            amount = int(match.group(1))
            return Action(
                type="scroll",
                x=None,
                y=None,
                text=None,
                raw={
                    "original_output": output_text,
                    "scroll_amount": amount,
                    "scroll_direction": "down" if amount < 0 else "up"
                }
            )

        else:
            # Fallback for unparseable output
            return Action(
                type="failed",
                x=None,
                y=None,
                text=None,
                raw={"original_output": output_text, "error": "Failed to parse"}
            )

    def _convert_opencua_coordinates(
        self,
        model_x: int,
        model_y: int,
        original_width: int,
        original_height: int
    ) -> tuple[float, float]:
        """Convert OpenCUA absolute coordinates to normalized [0,1].

        OpenCUA outputs absolute pixel coordinates after smart resize.
        We need to:
        1. Calculate smart-resized dimensions
        2. Convert to relative coordinates on resized image
        3. Apply same relative coordinates to original image
        4. Normalize to [0,1]
        """
        # Calculate smart-resized dimensions
        resized_height, resized_width = self._smart_resize(
            original_height, original_width,
            factor=28,
            min_pixels=3136,
            max_pixels=12845056
        )

        # Convert to relative coordinates
        rel_x = model_x / resized_width
        rel_y = model_y / resized_height

        # Clamp to [0, 1] to handle edge cases
        norm_x = max(0.0, min(1.0, rel_x))
        norm_y = max(0.0, min(1.0, rel_y))

        return norm_x, norm_y

    def _smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 3136,
        max_pixels: int = 12845056
    ) -> tuple[int, int]:
        """Replicate Qwen2.5-VL smart resize logic.

        This must match the resize used during model inference to correctly
        convert coordinates.
        """
        # Implementation of Qwen2.5 smart resize algorithm
        # See: https://github.com/QwenLM/Qwen2-VL/blob/main/qwen_vl_utils/vision_process.py
        pass
```

### Training Configuration

Add OpenCUA configs alongside existing Qwen configs:

```yaml
# configs/opencua_7b_baseline.yaml
model:
  name: xlangai/OpenCUA-7B
  adapter_type: opencua
  lora:
    r: 16
    lora_alpha: 32
    target_modules:
      - q_proj
      - v_proj
      - k_proj
      - o_proj
    lora_dropout: 0.05

training:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 3
  gradient_accumulation_steps: 4
  max_pixels: 262144  # 512x512 for faster training

# configs/videoagenttrek_7b_ablation.yaml
model:
  name: xlangai/VideoAgentTrek-CUA-7B
  adapter_type: opencua
  # Same LoRA/training config as baseline
```

### Experiment Plan

#### Baseline: OpenCUA-7B
**Hypothesis**: Clean AgentNet data should provide strong foundation for GUI automation.

**Test on**:
1. Synthetic semantic UIs (login, settings)
2. Real captures from openadapt-capture
3. WAA benchmark (if time permits)

**Metrics**:
- Action prediction accuracy (coordinate, content, function)
- Success rate on multi-step tasks
- Grounding accuracy (if applicable)

#### Ablation: VideoAgentTrek-CUA-7B
**Hypothesis**: Additional video pretraining might help OR might hurt due to noisy labels.

**Test on**: Same benchmarks as baseline

**Compare**:
- Does video pretraining improve generalization?
- Are there specific task types where VideoAgentTrek excels?
- Does noise in video-mined data hurt fine-tuning stability?

#### Control: Qwen2.5-VL-7B (existing)
Keep existing Qwen baseline for comparison.

### Implementation Checklist

- [ ] Create `openadapt_ml/models/opencua.py` with `OpenCUAAdapter`
- [ ] Implement coordinate conversion logic (`_convert_opencua_coordinates`)
- [ ] Implement smart resize algorithm matching Qwen2.5-VL
- [ ] Implement PyAutoGUI command parser (`_parse_action`)
- [ ] Add OpenCUA configs to `configs/`
- [ ] Update `openadapt_ml/training/trainer.py` to support opencua adapter_type
- [ ] Test on synthetic data
- [ ] Test on real captures
- [ ] Document results in `docs/opencua_experiments.md`

## Known Limitations

### vLLM Support
OpenCUA models do not yet support vLLM for fast inference. Must use `transformers` library directly.

### Modified Architecture
- OpenCUA uses 1D RoPE instead of Multimodal RoPE
- Uses Kimi-VL tokenizer instead of default Qwen tokenizer
- **Cannot use default transformers classes** - must set `trust_remote_code=True`

### Coordinate System Complexity
Converting between OpenCUA's absolute post-resize coordinates and openadapt-ml's normalized [0,1] coordinates requires careful implementation of smart resize logic.

### VideoAgentTrek Documentation Gap
VideoAgentTrek-CUA-7B lacks detailed documentation:
- No published benchmarks
- No details on video-mining methodology
- No information on dataset size or composition
- Empty model card on HuggingFace

This means the ablation study will be exploratory rather than based on published claims.

## Future Work

### Multi-Image History
OpenCUA supports up to 3 screenshot history for context. This could improve action prediction by showing UI state transitions.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "screenshot_t-2.png"},
            {"type": "image", "image": "screenshot_t-1.png"},
            {"type": "image", "image": "screenshot_t.png"},
            {"type": "text", "text": "Goal: Turn off Night Shift"}
        ]
    }
]
```

### Reflective CoT Integration
AgentNet includes rich CoT annotations. Could enhance training by:
1. Training on CoT-augmented samples
2. Generating intermediate reasoning steps
3. Using reasoning for error recovery

### OpenCUA-32B Evaluation
Use 32B model as strong baseline (SOTA open-source performance) for comparison, even if too large for training.

### Element-Based Actions
OpenCUA can output element IDs from accessibility trees, similar to SoM mode:
```python
"pyautogui.click(element_id='submit-button')"
```

This could bridge OpenCUA's PyAutoGUI format with openadapt-ml's SoM mode.

## Resources

- **Website**: https://opencua.xlang.ai/
- **Paper**: https://arxiv.org/abs/2508.09123
- **Code**: https://github.com/xlang-ai/OpenCUA
- **Dataset**: https://huggingface.co/datasets/xlangai/AgentNet
- **Models**:
  - OpenCUA-7B: https://huggingface.co/xlangai/OpenCUA-7B
  - OpenCUA-32B: https://huggingface.co/xlangai/OpenCUA-32B
  - VideoAgentTrek-CUA-7B: https://huggingface.co/xlangai/VideoAgentTrek-CUA-7B

## License

- **OpenCUA Models**: MIT License
- **AgentNet Dataset**: MIT License
- **VideoAgentTrek-CUA**: Apache 2.0 License

All compatible with openadapt-ml's MIT license.
