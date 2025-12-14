# GUI-Actor Integration Plan

## Overview

**GUI-Actor** (Microsoft Research, June 2025) introduces coordinate-free visual grounding for GUI agents.
Instead of regressing `(x,y)` coordinates, it learns attention maps over ViT patch tokens.

- Paper: [arXiv:2506.03143](https://arxiv.org/abs/2506.03143)
- Code: [github.com/microsoft/GUI-Actor](https://github.com/microsoft/GUI-Actor)
- Models: [HuggingFace microsoft/GUI-Actor-7B-Qwen2.5-VL](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL)

## Why This Matters

### Problems with Coordinate Regression

1. **Mismatch with visual representation**: VLMs use discrete patch tokens, but coordinate regression requires continuous output
2. **Fuzzy supervision**: Many GUI labels have multiple valid click points; coordinate loss punishes valid alternatives
3. **Resolution sensitivity**: Coordinates don't generalize across screen sizes without normalization hacks

### GUI-Actor Approach

1. **`<ACTOR>` token**: Dedicated token that attends over visual patch tokens
2. **Action head**: Small head (~100M params) outputs attention map over patches
3. **Region proposals**: One forward pass → multiple candidate regions with confidence scores
4. **Verifier reranking**: Optional scorer picks best region among candidates

## Benchmark Performance

| Model | Params | ScreenSpot-Pro |
|-------|--------|----------------|
| UI-TARS-72B | 72B | 38.1 |
| **GUI-Actor-7B** | 7B | **44.6** |

GUI-Actor achieves SOTA with 10x fewer parameters.

## Integration Plan for OpenAdapt-ML

### A) Schema/API Change (Highest ROI)

Add coordinate-free target representation:

```python
@dataclass
class RegionProposal:
    """A candidate action region."""
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    confidence: float
    patch_indices: list[int] | None = None
    mask: np.ndarray | None = None

@dataclass
class ActionTarget:
    """Target for an action - coordinate-free."""
    proposals: list[RegionProposal]
    selected_idx: int = 0  # Index of chosen proposal

    @property
    def execution_point(self) -> tuple[float, float]:
        """Convert selected region to click point (centroid)."""
        bbox = self.proposals[self.selected_idx].bbox
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
```

### B) Model Adapter

```python
class GUIActorGrounder(BaseGrounder):
    """Grounding using GUI-Actor attention-based regions."""

    def __init__(self, model_name: str = "microsoft/GUI-Actor-7B-Qwen2.5-VL"):
        self.model = AutoModel.from_pretrained(model_name)

    def ground(
        self,
        image: Image,
        instruction: str,
        top_k: int = 5
    ) -> list[RegionProposal]:
        """Get top-k region proposals for instruction."""
        # Returns ranked regions, not coordinates
        pass

    def verify_and_rerank(
        self,
        image: Image,
        instruction: str,
        proposals: list[RegionProposal]
    ) -> list[RegionProposal]:
        """Rerank proposals using verifier."""
        pass
```

### C) Benchmark Support

Add ScreenSpot-Pro as evaluation target:
- High-resolution professional desktop GUIs
- Out-of-distribution from typical training data
- Direct measure of grounding generalization

## Engineering Leverage

1. **Freeze backbone, fine-tune head**: Only ~100M params to adapt per domain
2. **Verifier as reusable reranker**: Can rerank regions from ANY grounder (accessibility tree, DOM, etc.)
3. **Multiple candidates enable search**: Try top-k, verify, then execute

## Evaluation Implications

Current approach (coordinate distance threshold) is problematic:
- Normalized distance < 0.15 is arbitrary
- Doesn't account for element size (clicking 15% away from a large button might still hit it)
- Doesn't match how GUI-Actor or benchmarks measure success

Better evaluation:
- **Region IoU**: Does predicted region overlap with ground truth element?
- **Element hit rate**: Does click point land inside the target element?
- **ScreenSpot-Pro metrics**: Standard benchmark for comparison

## Next Steps

1. [ ] Add `RegionProposal` and `ActionTarget` to schemas
2. [ ] Implement `GUIActorGrounder` adapter using HF checkpoint
3. [ ] Add ScreenSpot-Pro benchmark loader
4. [ ] Update evaluation to use region-based metrics
5. [ ] Fine-tune action head on OpenAdapt captures

## References

- [GUI-Actor Paper](https://arxiv.org/abs/2506.03143)
- [GUI-Actor GitHub](https://github.com/microsoft/GUI-Actor)
- [ScreenSpot-Pro Benchmark](https://arxiv.org/abs/2504.07981)
- [GUI-Actor HuggingFace](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL)
