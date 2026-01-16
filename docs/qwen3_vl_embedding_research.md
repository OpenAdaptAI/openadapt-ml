# Qwen3-VL-Embedding Research for Multimodal Demo Retrieval

**Date**: January 2026
**Status**: Research Summary
**Authors**: OpenAdapt ML Team

## Executive Summary

Qwen3-VL-Embedding is a new multimodal embedding model from Alibaba that maps text, images, screenshots, and video into a unified vector space. This research evaluates its potential for improving demo retrieval in OpenAdapt by enabling **visual + semantic similarity search** rather than text-only matching.

**Key Finding**: Qwen3-VL-Embedding achieves state-of-the-art multimodal retrieval (77.8 on MMEB-V2), supports flexible embedding dimensions via MRL, and has reasonable hardware requirements (2B model fits on consumer GPUs). This could significantly improve demo retrieval for GUI automation by matching screenshots alongside task descriptions.

---

## 1. What is Qwen3-VL-Embedding?

### 1.1 Overview

Qwen3-VL-Embedding is a multimodal embedding model built on the Qwen3-VL foundation. It maps diverse inputs (text, images, document screenshots, videos) into a unified high-dimensional semantic vector space.

**Key Specifications**:

| Model | Parameters | Layers | Context | Embedding Dim | Quantization | MRL |
|-------|-----------|--------|---------|---------------|--------------|-----|
| Qwen3-VL-Embedding-2B | 2B | 28 | 32K | 2048 (64-2048) | Yes | Yes |
| Qwen3-VL-Embedding-8B | 8B | 36 | 32K | 4096 (64-4096) | Yes | Yes |

**Supported Modalities**:
- Text (instructions, task descriptions)
- Images (screenshots, photos)
- Document images (PDFs, scanned documents)
- Videos (screen recordings, demos)
- Arbitrary multimodal combinations (text + image, text + video)

### 1.2 Architecture

The model uses a **dual-tower design** that extracts the hidden state corresponding to the [EOS] token as the final representation. Key features:

- **Base Model**: Qwen3-VL-2B-Instruct or Qwen3-VL-8B-Instruct
- **Training**: Multi-stage paradigm (contrastive pre-training -> reranker distillation)
- **Matryoshka Representation Learning (MRL)**: Supports flexible embedding dimensions
- **Instruction-Aware**: Task-specific instructions provide 1-5% improvement

### 1.3 Performance Benchmarks

**MMEB-V2 Results** (Multimodal Embedding Benchmark):

| Model | Overall | Image | Video | VisDoc |
|-------|---------|-------|-------|--------|
| Qwen3-VL-Embedding-8B | **77.8** | 80.1 | 67.1 | 82.4 |
| Qwen3-VL-Embedding-2B | 73.2 | 75.0 | 61.9 | 79.2 |
| Previous SOTA (open) | 71.1 | - | - | - |

**MMTEB Results** (Multilingual Text Embedding Benchmark):

| Model | Mean (Task) | Retrieval | Classification |
|-------|-------------|-----------|----------------|
| Qwen3-VL-Embedding-8B | 67.88 | 81.08 | 71.95 |
| Qwen3-VL-Embedding-2B | 63.87 | 78.50 | - |

---

## 2. Qwen3-VL-Reranker: Two-Stage Retrieval

### 2.1 What is the Reranker?

Qwen3-VL-Reranker is a complementary cross-encoder model that takes (query, document) pairs and outputs precise relevance scores. It uses cross-attention mechanisms for fine-grained relevance estimation.

**Models Available**:
- Qwen3-VL-Reranker-2B
- Qwen3-VL-Reranker-8B

### 2.2 Two-Stage Pipeline

```
Query + Screenshot
       |
       v
[Stage 1: Embedding Model]  <- Fast, approximate
       |
       v
   Top 100 candidates
       |
       v
[Stage 2: Reranker Model]   <- Slow, precise
       |
       v
   Top K final results
```

**Benefits**:
1. **Embedding**: Fast initial recall (dot product similarity)
2. **Reranker**: High-precision final ranking (cross-attention)

### 2.3 Reranker Performance

| Model | MMEB-v2 Avg | Image | Video | VisDoc |
|-------|-------------|-------|-------|--------|
| Qwen3-VL-Reranker-8B | **79.2** | 80.7 | 55.8 | 86.3 |
| Qwen3-VL-Reranker-2B | 75.1 | 73.8 | 52.1 | 83.4 |

---

## 3. How Could This Improve Demo Retrieval?

### 3.1 Current Approach in openadapt-ml

Currently, `openadapt_ml/retrieval/` uses **text-only embeddings**:

```python
# Current: Text-only retrieval
from openadapt_ml.retrieval import DemoRetriever

retriever = DemoRetriever(embedding_method="sentence_transformers")
retriever.add_demo(episode)  # Only embeds episode.instruction (text)
results = retriever.retrieve("Turn off Night Shift")  # Text similarity only
```

**Supported Backends**:
- `tfidf`: Simple baseline (no dependencies)
- `sentence_transformers`: all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d)
- `openai`: text-embedding-3-small (1536d)

### 3.2 Proposed Multimodal Approach

With Qwen3-VL-Embedding, we could embed **screenshot + task description**:

```python
# Proposed: Multimodal retrieval
from openadapt_ml.retrieval import MultimodalDemoRetriever

retriever = MultimodalDemoRetriever(
    embedding_model="Qwen/Qwen3-VL-Embedding-2B"
)

# Index demos with screenshots
for episode in episodes:
    retriever.add_demo(
        episode,
        screenshot=episode.steps[0].observation.screenshot_path  # Include visual context
    )

# Retrieve by task + current screenshot
results = retriever.retrieve(
    task="Turn off Night Shift",
    screenshot=current_screenshot  # Match similar UI layouts
)
```

### 3.3 Use Cases for Visual Retrieval

| Use Case | Current (Text-Only) | With Qwen3-VL-Embedding |
|----------|---------------------|-------------------------|
| Similar task names | Works well | Works well |
| Similar UI layouts | Cannot match | Can match by screenshot similarity |
| Different apps, same flow | Misses | Can find similar workflows |
| Form filling patterns | Task name only | Matches form layouts |
| Navigation patterns | Limited | Matches navigation UI patterns |

**Example**: Finding demos for "submit a support ticket"

- **Text-only**: Matches demos with similar task descriptions
- **Multimodal**: Also matches demos with similar form layouts (text fields, submit buttons), even if the task name is different

---

## 4. Integration Approach

### 4.1 Hardware Requirements

**Qwen3-VL-Embedding-2B** (Recommended for local inference):
- VRAM: ~6-8 GB (FP16), ~4-5 GB (INT8 quantized)
- Supported GPUs: RTX 3060+, RTX 4060+, M1/M2/M3 Mac (MPS)
- Inference: ~50-200ms per embedding

**Qwen3-VL-Embedding-8B** (Higher quality):
- VRAM: ~18-20 GB (FP16), ~10-12 GB (INT8)
- Supported GPUs: RTX 4090, A100, H100
- Inference: ~150-400ms per embedding

### 4.2 Installation

```bash
# Core dependencies
uv add transformers>=4.57.0 qwen-vl-utils>=0.0.14 torch>=2.0

# For faster inference (optional)
uv add flash-attn>=2.0

# For vLLM-based inference (optional, for batched processing)
uv add vllm
```

### 4.3 Proposed Integration Architecture

```
openadapt_ml/retrieval/
├── embeddings.py           # Existing: TF-IDF, SentenceTransformer, OpenAI
├── multimodal_embeddings.py  # NEW: Qwen3VLEmbedder
├── demo_retriever.py       # Existing: DemoRetriever
├── multimodal_retriever.py   # NEW: MultimodalDemoRetriever
└── reranker.py              # NEW: Qwen3VLReranker
```

### 4.4 Example Implementation

**New Embedder Class** (`multimodal_embeddings.py`):

```python
import torch
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from pathlib import Path

class Qwen3VLEmbedder:
    """Multimodal embedder using Qwen3-VL-Embedding."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        use_flash_attention: bool = True,
        embedding_dim: Optional[int] = None,  # MRL: use smaller dims if needed
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.use_flash_attention = use_flash_attention
        self.embedding_dim = embedding_dim
        self._model = None

    def _load_model(self):
        """Lazy-load the model."""
        if self._model is not None:
            return

        # Import from Qwen3-VL-Embedding repo
        # See: https://github.com/QwenLM/Qwen3-VL-Embedding
        from scripts.qwen3_vl_embedding import Qwen3VLEmbedder as _Qwen3VLEmbedder

        self._model = _Qwen3VLEmbedder(
            model_name_or_path=self.model_name,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2" if self.use_flash_attention else None,
        )

    def embed(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        instruction: Optional[str] = None,
    ) -> Any:
        """Embed text, image, or both.

        Args:
            text: Text to embed.
            image: Image path, URL, or PIL Image.
            instruction: Task-specific instruction for better retrieval.

        Returns:
            Embedding vector (numpy array).
        """
        self._load_model()

        input_dict: Dict[str, Any] = {}

        if text:
            input_dict["text"] = text
        if image:
            input_dict["image"] = str(image) if isinstance(image, Path) else image
        if instruction:
            input_dict["instruction"] = instruction

        embeddings = self._model.process([input_dict])

        # Apply MRL dimension reduction if specified
        if self.embedding_dim:
            embeddings = embeddings[:, :self.embedding_dim]

        return embeddings[0]

    def embed_batch(
        self,
        inputs: List[Dict[str, Any]],
    ) -> Any:
        """Embed multiple inputs.

        Args:
            inputs: List of dicts with 'text', 'image', and/or 'instruction' keys.

        Returns:
            Embeddings matrix (numpy array).
        """
        self._load_model()
        embeddings = self._model.process(inputs)

        if self.embedding_dim:
            embeddings = embeddings[:, :self.embedding_dim]

        return embeddings
```

**Multimodal Retriever** (`multimodal_retriever.py`):

```python
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image

from openadapt_ml.schema import Episode
from openadapt_ml.retrieval.multimodal_embeddings import Qwen3VLEmbedder

class MultimodalDemoRetriever:
    """Demo retriever with multimodal (text + image) support."""

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B",
        embedding_dim: Optional[int] = None,
        use_reranker: bool = False,
        reranker_model: str = "Qwen/Qwen3-VL-Reranker-2B",
    ):
        self.embedder = Qwen3VLEmbedder(
            model_name=embedding_model,
            embedding_dim=embedding_dim,
        )
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self._demos = []
        self._embeddings = None

    def add_demo(
        self,
        episode: Episode,
        screenshot: Optional[Union[str, Path, Image.Image]] = None,
    ):
        """Add a demo with optional screenshot."""
        # Extract screenshot from first step if not provided
        if screenshot is None and episode.steps:
            first_obs = episode.steps[0].observation
            if first_obs and first_obs.screenshot_path:
                screenshot = first_obs.screenshot_path

        # Create embedding input
        embed_input = {
            "text": episode.instruction,
            "instruction": "Retrieve demonstrations for GUI automation tasks.",
        }
        if screenshot:
            embed_input["image"] = str(screenshot) if isinstance(screenshot, Path) else screenshot

        embedding = self.embedder.embed(**embed_input)

        self._demos.append({
            "episode": episode,
            "screenshot": screenshot,
            "embedding": embedding,
        })

    def retrieve(
        self,
        task: str,
        screenshot: Optional[Union[str, Path, Image.Image]] = None,
        top_k: int = 3,
        rerank_top_n: int = 20,
    ) -> List[Episode]:
        """Retrieve similar demos.

        Args:
            task: Task description.
            screenshot: Current screenshot for visual matching.
            top_k: Number of results to return.
            rerank_top_n: Number of candidates for reranking (if enabled).

        Returns:
            List of Episode objects.
        """
        import numpy as np

        # Create query embedding
        query_input = {
            "text": task,
            "instruction": "Retrieve demonstrations for GUI automation tasks.",
        }
        if screenshot:
            query_input["image"] = str(screenshot) if isinstance(screenshot, Path) else screenshot

        query_embedding = self.embedder.embed(**query_input)

        # Compute similarities
        scores = []
        for demo in self._demos:
            sim = np.dot(query_embedding, demo["embedding"])
            sim /= (np.linalg.norm(query_embedding) * np.linalg.norm(demo["embedding"]) + 1e-9)
            scores.append(sim)

        # Get top candidates
        n_candidates = rerank_top_n if self.use_reranker else top_k
        top_indices = np.argsort(scores)[::-1][:n_candidates]

        # Rerank if enabled
        if self.use_reranker and len(top_indices) > top_k:
            top_indices = self._rerank(task, screenshot, top_indices)[:top_k]

        return [self._demos[i]["episode"] for i in top_indices]

    def _rerank(self, task, screenshot, candidate_indices):
        """Rerank candidates using Qwen3-VL-Reranker."""
        # TODO: Implement reranker integration
        # from scripts.qwen3_vl_reranker import Qwen3VLReranker
        # reranker = Qwen3VLReranker(model_name_or_path=self.reranker_model)
        # ...
        return candidate_indices
```

### 4.5 Usage Example

```python
from openadapt_ml.retrieval import MultimodalDemoRetriever
from openadapt_ml.ingest.capture import capture_to_episode

# Load demos from captures
capture_dirs = [
    "/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift",
    "/Users/abrichr/oa/src/openadapt-capture/search-github",
    "/Users/abrichr/oa/src/openadapt-capture/open-calculator",
]

# Initialize retriever
retriever = MultimodalDemoRetriever(
    embedding_model="Qwen/Qwen3-VL-Embedding-2B",
    embedding_dim=512,  # Use MRL for smaller storage
)

# Index demos with screenshots
for capture_dir in capture_dirs:
    episode = capture_to_episode(capture_dir)
    retriever.add_demo(episode)

# At runtime: retrieve by task + current screenshot
current_screenshot = "/path/to/current_screen.png"
similar_demos = retriever.retrieve(
    task="Disable Night Shift on macOS",
    screenshot=current_screenshot,
    top_k=3,
)

# Use retrieved demos in prompt
for demo in similar_demos:
    print(f"Found similar demo: {demo.instruction}")
```

---

## 5. Pros and Cons

### 5.1 Advantages

| Aspect | Benefit |
|--------|---------|
| **Visual Matching** | Find demos with similar UI layouts, not just similar task names |
| **Multimodal Fusion** | Combines text and visual signals for better retrieval |
| **State-of-the-Art** | 77.8 on MMEB-V2, beating all other open-source models |
| **Flexible Dimensions** | MRL allows 64-4096 dim embeddings for storage/speed tradeoffs |
| **Multilingual** | 30+ languages out of the box |
| **Instruction-Aware** | Task-specific instructions improve retrieval 1-5% |
| **Two-Stage Pipeline** | Embedding for recall, reranker for precision |

### 5.2 Disadvantages

| Aspect | Challenge | Mitigation |
|--------|-----------|------------|
| **GPU Required** | 2B model needs ~6GB VRAM | Use cloud GPU, quantization, or MPS |
| **Slower than Text** | ~100ms vs ~10ms per embedding | Batch processing, pre-compute embeddings |
| **New Dependency** | Requires transformers>=4.57.0 | Pin version, optional feature |
| **Model Size** | 2B model is ~4GB download | One-time download, cache locally |
| **Early Adoption** | Released Jan 2026, may have bugs | Monitor issues, fallback to text |

### 5.3 Recommendation

**Start with Qwen3-VL-Embedding-2B** for local development:
- Fits on consumer GPUs (RTX 3060+)
- Good balance of quality and speed
- Use MRL (512d) for faster similarity search

**Consider 8B model** for:
- Production deployments with better GPUs
- Critical retrieval quality requirements
- Server-side batch processing

---

## 6. Implementation Roadmap

### Phase 1: Proof of Concept (1 week)
- [ ] Add `multimodal_embeddings.py` with Qwen3VLEmbedder
- [ ] Create minimal `MultimodalDemoRetriever`
- [ ] Test on existing captures (turn-off-nightshift, etc.)
- [ ] Benchmark: multimodal vs text-only retrieval quality

### Phase 2: Integration (1 week)
- [ ] Add to `DemoRetriever` as optional backend
- [ ] Update CLI: `--embedding-method qwen3vl`
- [ ] Add batch processing for index building
- [ ] Persist embeddings to disk

### Phase 3: Optimization (1 week)
- [ ] Implement MRL dimension selection
- [ ] Add reranker support
- [ ] Quantized inference (INT8)
- [ ] vLLM batched inference

### Phase 4: Production (ongoing)
- [ ] Cloud GPU inference API
- [ ] Caching and incremental updates
- [ ] A/B testing vs text-only
- [ ] Documentation and examples

---

## 7. References

- [Qwen3-VL-Embedding HuggingFace Collection](https://huggingface.co/collections/Qwen/qwen3-vl-embedding)
- [Qwen3-VL-Embedding-2B Model Card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [Qwen3-VL-Embedding-8B Model Card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
- [GitHub Repository](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Qwen Blog Announcement](https://qwen.ai/blog?id=qwen3-vl-embedding)
- [arXiv Paper: Qwen3-VL-Embedding and Qwen3-VL-Reranker](https://arxiv.org/abs/2601.04720)
- [GPU Requirements Guide for Qwen Models](https://apxml.com/posts/gpu-system-requirements-qwen-models)

---

## Appendix A: Existing Retrieval Code Structure

Current implementation in `openadapt_ml/retrieval/`:

```
retrieval/
├── __init__.py              # Exports: DemoIndex, DemoRetriever, create_embedder
├── embeddings.py            # BaseEmbedder, TFIDFEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder
├── index.py                 # DemoIndex (simple wrapper around demos)
├── demo_retriever.py        # DemoRetriever (main retrieval class)
└── retriever.py             # Additional retriever utilities
```

Key classes:
- `BaseEmbedder`: Abstract base with `embed()`, `embed_batch()`, `cosine_similarity()`
- `DemoRetriever`: Main class with `add_demo()`, `build_index()`, `retrieve()`
- `DemoMetadata`: Stores episode, app_name, domain, embedding, metadata

The multimodal embedder would inherit from `BaseEmbedder` and be integrated as a new backend option.

---

## Appendix B: Full Benchmark Comparison

**Qwen3-VL-Embedding vs Alternatives**:

| Model | Type | MMEB-V2 | MMTEB | Size | Notes |
|-------|------|---------|-------|------|-------|
| Qwen3-VL-Embedding-8B | Multimodal | **77.8** | 67.88 | 8B | SOTA |
| Qwen3-VL-Embedding-2B | Multimodal | 73.2 | 63.87 | 2B | Good for local |
| VLM2Vec | Multimodal | 71.1 | - | - | Previous SOTA |
| CLIP ViT-L/14 | Image+Text | ~55 | - | 400M | Classic baseline |
| all-mpnet-base-v2 | Text-only | - | 57.8 | 110M | Current in openadapt-ml |
| text-embedding-3-small | Text-only | - | ~62 | API | OpenAI |

Qwen3-VL-Embedding is **6.7% better** than the previous best open-source multimodal model on MMEB-V2.
