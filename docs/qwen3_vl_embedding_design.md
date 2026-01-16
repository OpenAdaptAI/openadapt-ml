# Qwen3-VL-Embedding Integration Design

**Document Version**: 1.0
**Date**: January 2026
**Status**: Design Proposal
**Authors**: OpenAdapt ML Team

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [Architecture](#2-architecture)
3. [Data Flow](#3-data-flow)
4. [API Design](#4-api-design)
5. [Storage and Persistence](#5-storage-and-persistence)
6. [Performance Considerations](#6-performance-considerations)
7. [Hardware Requirements](#7-hardware-requirements)
8. [Testing Strategy](#8-testing-strategy)
9. [Migration Path](#9-migration-path)
10. [Implementation Plan](#10-implementation-plan)

---

## 1. Goals and Non-Goals

### 1.1 Goals

**Primary Goal**: Enable multimodal (visual + semantic) demo retrieval for GUI automation by integrating Qwen3-VL-Embedding into the existing retrieval module.

**Specific Objectives**:

1. **Visual Context Matching**: Retrieve demos based on UI screenshot similarity, not just task description text similarity.

2. **Unified Embedding Space**: Map screenshots, task descriptions, and their combinations into a shared vector space for cross-modal retrieval.

3. **Improved Demo Selection**: Find relevant demos even when:
   - Task descriptions differ but UI layouts are similar
   - Users work with similar form layouts across different apps
   - Navigation patterns match across different contexts

4. **Backward Compatibility**: Preserve existing text-only retrieval as a fallback and default option.

5. **Production Readiness**: Support batch processing, incremental indexing, persistence, and reasonable hardware requirements.

### 1.2 Non-Goals (v1)

The following are explicitly **out of scope** for the initial implementation:

| Non-Goal | Rationale |
|----------|-----------|
| Video embedding | Complexity; screenshots are sufficient for demo retrieval |
| Real-time reranking during inference | Focus on embedding-based retrieval first |
| Multi-GPU inference | Single-GPU is sufficient for batch sizes we expect |
| Quantization-aware training | Use pre-trained quantized models from HuggingFace |
| Dynamic embedding dimensions | Fix dimension at initialization; MRL can be added later |
| Cloud-hosted embedding API | Focus on local inference first |
| Automatic model selection | User specifies model explicitly |
| OCR text extraction from screenshots | Rely on VLM's native visual understanding |
| Integration with external vector databases (Pinecone, Weaviate) | Use FAISS/numpy for v1 |

### 1.3 Success Criteria

1. **Functional**: `MultimodalDemoRetriever` can index and retrieve demos using text+image inputs
2. **Quality**: Multimodal retrieval achieves higher precision@k than text-only on manual evaluation
3. **Performance**: Index 100 demos in < 5 minutes on RTX 4090; single query < 500ms
4. **Integration**: Existing `DemoRetriever` tests continue to pass unchanged

---

## 2. Architecture

### 2.1 Module Structure

The new multimodal components will be added alongside existing retrieval code:

```
openadapt_ml/retrieval/
├── __init__.py                    # Updated exports
├── embeddings.py                  # Existing: BaseEmbedder, TFIDFEmbedder, etc.
├── multimodal_embeddings.py       # NEW: Qwen3VLEmbedder
├── index.py                       # Existing: DemoIndex (legacy)
├── demo_retriever.py              # Existing: DemoRetriever (text-only)
├── multimodal_retriever.py        # NEW: MultimodalDemoRetriever
├── reranker.py                    # NEW: Qwen3VLReranker (optional)
├── retriever.py                   # Existing: LegacyDemoRetriever
└── storage.py                     # NEW: EmbeddingStorage (FAISS + numpy persistence)
```

### 2.2 Class Hierarchy

```
                    ┌─────────────────┐
                    │   BaseEmbedder  │  (Abstract)
                    │   (embeddings.py)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│ TFIDFEmbedder │  │SentenceTransformer   │  │ OpenAIEmbedder   │
│               │  │Embedder              │  │                  │
└───────────────┘  └──────────────────────┘  └──────────────────┘

                    ┌─────────────────────┐
                    │ MultimodalEmbedder  │  (Abstract, NEW)
                    │                     │
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  Qwen3VLEmbedder    │  (NEW)
                    │                     │
                    └─────────────────────┘


                    ┌─────────────────────┐
                    │    DemoRetriever    │  (Existing, text-only)
                    │                     │
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │ MultimodalDemo      │  (NEW)
                    │ Retriever           │
                    │                     │
                    │ - uses Qwen3VL      │
                    │   Embedder          │
                    │ - optional reranker │
                    └─────────────────────┘
```

### 2.3 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `MultimodalEmbedder` (ABC) | Abstract interface for multimodal embedding models |
| `Qwen3VLEmbedder` | Concrete implementation using Qwen3-VL-Embedding |
| `MultimodalDemoRetriever` | Orchestrates multimodal indexing and retrieval |
| `Qwen3VLReranker` | Optional cross-encoder for precision improvement |
| `EmbeddingStorage` | FAISS index management and numpy array persistence |

### 2.4 Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MultimodalDemoRetriever                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐   │
│  │    Episode   │───▶│  Qwen3VLEmbedder│───▶│   EmbeddingStorage   │   │
│  │  + Screenshot│    │                 │    │   (FAISS + numpy)    │   │
│  └──────────────┘    └─────────────────┘    └──────────────────────┘   │
│                              │                        │                 │
│                              │                        │                 │
│                              ▼                        ▼                 │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐   │
│  │    Query     │───▶│  Query Embedding │───▶│   Similarity Search  │   │
│  │  + Screenshot│    │                 │    │   (top-K candidates) │   │
│  └──────────────┘    └─────────────────┘    └──────────────────────┘   │
│                                                       │                 │
│                                                       ▼                 │
│                                             ┌──────────────────────┐   │
│                                             │  Qwen3VLReranker     │   │
│                                             │  (optional)          │   │
│                                             └──────────────────────┘   │
│                                                       │                 │
│                                                       ▼                 │
│                                             ┌──────────────────────┐   │
│                                             │  RetrievalResult[]   │   │
│                                             └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Indexing Flow (Demo Ingestion)

When a demo is added to the index:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INDEXING FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  Episode
     │
     ▼
┌─────────────────┐
│ Extract data:   │
│ - instruction   │
│ - screenshot    │
│   (first step)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Qwen3VLEmbedder.embed()                    │
│                                                         │
│  Input:                                                 │
│  {                                                      │
│    "text": "Turn off Night Shift",                      │
│    "image": "/path/to/screenshot.png",                  │
│    "instruction": "Retrieve demonstrations for GUI..."  │
│  }                                                      │
│                                                         │
│  Processing:                                            │
│  1. Load/resize image (max 1280px)                      │
│  2. Tokenize text with Qwen tokenizer                   │
│  3. Forward pass through Qwen3-VL-Embedding-2B          │
│  4. Extract [EOS] token hidden state                    │
│  5. Apply MRL truncation (if embedding_dim < 2048)      │
│  6. L2 normalize                                        │
│                                                         │
│  Output: float32[embedding_dim] vector                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    EmbeddingStorage                     │
│                                                         │
│  1. Append embedding to numpy matrix                    │
│  2. Add to FAISS IndexFlatIP                            │
│  3. Store demo metadata mapping                         │
│  4. Update index state                                  │
└─────────────────────────────────────────────────────────┘
```

**Key Decisions**:
- Use the **first step's screenshot** as the representative image for the demo
- Apply **task-specific instruction** ("Retrieve demonstrations for GUI automation tasks") for better retrieval quality (1-5% improvement per Qwen research)
- **Lazy load** the model only when first embedding is requested

### 3.2 Query Flow (Retrieval)

When a query is executed:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               QUERY FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  Query: "Disable Night Shift"
  Current Screenshot: /path/to/current_screen.png
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│              Qwen3VLEmbedder.embed()                    │
│                                                         │
│  Input:                                                 │
│  {                                                      │
│    "text": "Disable Night Shift",                       │
│    "image": "/path/to/current_screen.png",              │
│    "instruction": "Retrieve demonstrations for GUI..."  │
│  }                                                      │
│                                                         │
│  Output: query_embedding float32[embedding_dim]         │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                   FAISS Similarity Search               │
│                                                         │
│  1. Compute dot product: query @ embeddings_matrix.T    │
│  2. Sort by score descending                            │
│  3. Return top-N indices (N = rerank_top_n or top_k)    │
│                                                         │
│  Complexity: O(n) for IndexFlatIP, O(log n) for IVF     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                  Optional: Reranking                    │
│                 (if use_reranker=True)                  │
│                                                         │
│  For each candidate in top-N:                           │
│    1. Create (query, candidate) pair                    │
│    2. Forward through Qwen3-VL-Reranker                 │
│    3. Get cross-attention relevance score               │
│                                                         │
│  Sort by reranker scores, return top-K                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│              Return RetrievalResult[]                   │
│                                                         │
│  For each result:                                       │
│  - demo: DemoMetadata                                   │
│  - score: combined similarity score                     │
│  - text_score: embedding similarity component           │
│  - rank: position in result list                        │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Embedding Input Modes

The `Qwen3VLEmbedder` supports three input modes:

| Mode | Input | Use Case |
|------|-------|----------|
| **Text-only** | `{"text": "..."}` | Query without screenshot, legacy fallback |
| **Image-only** | `{"image": "..."}` | Find demos by visual similarity only |
| **Text+Image** | `{"text": "...", "image": "..."}` | Full multimodal retrieval (recommended) |

The embedding dimension is the same regardless of input mode, enabling cross-modal retrieval.

---

## 4. API Design

### 4.1 MultimodalEmbedder Abstract Base Class

```python
# openadapt_ml/retrieval/multimodal_embeddings.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class MultimodalEmbedder(ABC):
    """Abstract base class for multimodal embedding models.

    Multimodal embedders map text, images, or combinations into a unified
    vector space. All inputs produce vectors of the same dimension.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @abstractmethod
    def embed(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        instruction: Optional[str] = None,
    ) -> NDArray[np.float32]:
        """Embed a single input (text, image, or both).

        Args:
            text: Text to embed. Can be None for image-only.
            image: Image path, URL, or PIL Image. Can be None for text-only.
            instruction: Task-specific instruction for better retrieval.

        Returns:
            Embedding vector as float32 numpy array of shape (embedding_dim,).

        Raises:
            ValueError: If both text and image are None.
        """
        pass

    @abstractmethod
    def embed_batch(
        self,
        inputs: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Embed multiple inputs.

        Args:
            inputs: List of dicts with 'text', 'image', and/or 'instruction' keys.
            show_progress: Whether to show progress bar.

        Returns:
            Embeddings matrix of shape (n_inputs, embedding_dim).
        """
        pass

    def cosine_similarity(
        self,
        vec1: NDArray[np.float32],
        vec2: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
```

### 4.2 Qwen3VLEmbedder Implementation

```python
# openadapt_ml/retrieval/multimodal_embeddings.py (continued)

import logging
from dataclasses import dataclass
from typing import Literal

import torch

logger = logging.getLogger(__name__)


@dataclass
class Qwen3VLEmbedderConfig:
    """Configuration for Qwen3VLEmbedder."""

    model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    """HuggingFace model name or local path."""

    embedding_dim: Optional[int] = None
    """Target embedding dimension. If None, uses model's full dimension.
    Supported values: 64, 128, 256, 512, 1024, 2048 (2B) or 4096 (8B).
    Uses Matryoshka Representation Learning (MRL) for dimension reduction."""

    device: Optional[str] = None
    """Device to use: "cuda", "cpu", "mps", or specific like "cuda:0".
    If None, auto-detects best available device."""

    torch_dtype: torch.dtype = torch.float16
    """Model dtype. Use float16 for GPU, float32 for CPU."""

    use_flash_attention: bool = True
    """Whether to use Flash Attention 2 (requires flash-attn package)."""

    max_image_size: int = 1280
    """Maximum image dimension. Images larger than this are resized."""

    normalize_embeddings: bool = True
    """Whether to L2-normalize embeddings (required for cosine similarity)."""

    cache_dir: Optional[Path] = None
    """Directory for caching downloaded models."""

    default_instruction: str = "Retrieve demonstrations for GUI automation tasks."
    """Default instruction prefix for retrieval queries."""


class Qwen3VLEmbedder(MultimodalEmbedder):
    """Multimodal embedder using Qwen3-VL-Embedding.

    This embedder maps text, images, and text+image combinations into
    a unified semantic vector space using Alibaba's Qwen3-VL-Embedding model.

    Features:
    - Supports text-only, image-only, and multimodal inputs
    - Matryoshka Representation Learning (MRL) for flexible dimensions
    - Lazy model loading to minimize startup time
    - GPU/CPU/MPS support with automatic device detection
    - Flash Attention 2 for faster inference

    Example:
        >>> embedder = Qwen3VLEmbedder()
        >>>
        >>> # Text-only embedding
        >>> text_emb = embedder.embed(text="Turn off Night Shift")
        >>>
        >>> # Image-only embedding
        >>> img_emb = embedder.embed(image="/path/to/screenshot.png")
        >>>
        >>> # Multimodal embedding (recommended)
        >>> mm_emb = embedder.embed(
        ...     text="Turn off Night Shift",
        ...     image="/path/to/screenshot.png",
        ... )
        >>>
        >>> # Batch embedding
        >>> embeddings = embedder.embed_batch([
        ...     {"text": "Task 1", "image": "img1.png"},
        ...     {"text": "Task 2", "image": "img2.png"},
        ... ])

    Hardware Requirements:
        - 2B model: ~6-8 GB VRAM (FP16), ~4-5 GB (INT8)
        - 8B model: ~18-20 GB VRAM (FP16), ~10-12 GB (INT8)
        - Inference: ~50-200ms per embedding (2B), ~150-400ms (8B)
    """

    # Model-specific max dimensions
    _MAX_DIM = {
        "Qwen/Qwen3-VL-Embedding-2B": 2048,
        "Qwen/Qwen3-VL-Embedding-8B": 4096,
    }

    def __init__(
        self,
        config: Optional[Qwen3VLEmbedderConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Qwen3-VL-Embedding embedder.

        Args:
            config: Configuration object. If None, uses defaults.
            **kwargs: Override config fields (convenience).
        """
        if config is None:
            config = Qwen3VLEmbedderConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self._model = None
        self._processor = None
        self._device = None

        # Validate embedding_dim
        max_dim = self._MAX_DIM.get(config.model_name, 2048)
        if config.embedding_dim is not None:
            if config.embedding_dim > max_dim:
                raise ValueError(
                    f"embedding_dim {config.embedding_dim} exceeds model maximum {max_dim}"
                )
            if config.embedding_dim < 64:
                raise ValueError("embedding_dim must be at least 64")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if self.config.embedding_dim is not None:
            return self.config.embedding_dim
        return self._MAX_DIM.get(self.config.model_name, 2048)

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.config.model_name

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self._device is not None:
            return self._device

        if self.config.device is not None:
            self._device = torch.device(self.config.device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info(f"Using device: {self._device}")
        return self._device

    def _load_model(self) -> None:
        """Lazy-load the model and processor."""
        if self._model is not None:
            return

        logger.info(f"Loading Qwen3-VL-Embedding model: {self.config.model_name}")

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers>=4.57.0 is required for Qwen3VLEmbedder. "
                "Install with: uv add transformers>=4.57.0"
            )

        device = self._get_device()

        # Determine attention implementation
        attn_impl = None
        if self.config.use_flash_attention and device.type == "cuda":
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.warning(
                    "flash-attn not installed, using default attention. "
                    "Install with: uv add flash-attn"
                )

        # Load model
        model_kwargs = {
            "torch_dtype": self.config.torch_dtype,
            "device_map": "auto" if device.type == "cuda" else None,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        if self.config.cache_dir:
            model_kwargs["cache_dir"] = str(self.config.cache_dir)

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        if device.type != "cuda":
            self._model = self._model.to(device)

        self._model.eval()

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
        )

        logger.info(f"Model loaded successfully on {device}")

    def _prepare_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Image.Image:
        """Load and preprocess image."""
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
            elif str(image).startswith(("http://", "https://")):
                import requests
                response = requests.get(str(image), timeout=30)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                raise FileNotFoundError(f"Image not found: {image}")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Resize if too large
        max_size = self.config.max_image_size
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def embed(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        instruction: Optional[str] = None,
    ) -> NDArray[np.float32]:
        """Embed a single input."""
        if text is None and image is None:
            raise ValueError("At least one of text or image must be provided")

        inputs = [{"text": text, "image": image, "instruction": instruction}]
        embeddings = self.embed_batch(inputs, show_progress=False)
        return embeddings[0]

    def embed_batch(
        self,
        inputs: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Embed multiple inputs."""
        if not inputs:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        self._load_model()

        # Use default instruction if not provided
        default_instr = self.config.default_instruction

        embeddings = []

        # Process inputs (can batch in future for efficiency)
        iterator = inputs
        if show_progress and len(inputs) > 1:
            try:
                from tqdm import tqdm
                iterator = tqdm(inputs, desc="Embedding")
            except ImportError:
                pass

        for inp in iterator:
            text = inp.get("text")
            image = inp.get("image")
            instruction = inp.get("instruction", default_instr)

            if text is None and image is None:
                raise ValueError("Each input must have at least text or image")

            # Build messages for Qwen format
            messages = []
            content = []

            if instruction:
                content.append({"type": "text", "text": instruction})

            if image is not None:
                img = self._prepare_image(image)
                content.append({"type": "image", "image": img})

            if text:
                content.append({"type": "text", "text": text})

            messages.append({"role": "user", "content": content})

            # Process through model
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            model_inputs = self._processor(
                text=[prompt],
                images=[img] if image is not None else None,
                return_tensors="pt",
            )

            model_inputs = {
                k: v.to(self._model.device) for k, v in model_inputs.items()
            }

            with torch.no_grad():
                outputs = self._model(**model_inputs, output_hidden_states=True)

                # Get last hidden state at EOS position
                hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)

                # Find EOS token position (last non-padding token)
                attention_mask = model_inputs.get("attention_mask")
                if attention_mask is not None:
                    seq_lens = attention_mask.sum(dim=1)
                    eos_positions = seq_lens - 1
                else:
                    eos_positions = torch.tensor(
                        [hidden_states.size(1) - 1],
                        device=hidden_states.device,
                    )

                # Extract EOS embeddings
                batch_indices = torch.arange(
                    hidden_states.size(0),
                    device=hidden_states.device,
                )
                embedding = hidden_states[batch_indices, eos_positions]

                # Apply MRL dimension reduction if needed
                if self.config.embedding_dim is not None:
                    embedding = embedding[:, :self.config.embedding_dim]

                # Normalize
                if self.config.normalize_embeddings:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

                embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings).astype(np.float32)

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")
```

### 4.3 MultimodalDemoRetriever

```python
# openadapt_ml/retrieval/multimodal_retriever.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from openadapt_ml.schema import Episode
from openadapt_ml.retrieval.multimodal_embeddings import (
    Qwen3VLEmbedder,
    Qwen3VLEmbedderConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class MultimodalDemoMetadata:
    """Metadata for a multimodal demonstration.

    Extends DemoMetadata with screenshot information.
    """

    demo_id: str
    """Unique identifier for the demo."""

    episode: Episode
    """The full Episode object."""

    goal: str
    """Task description/instruction."""

    screenshot_path: Optional[str] = None
    """Path to the representative screenshot."""

    app_name: Optional[str] = None
    """Application name (e.g., 'System Settings')."""

    domain: Optional[str] = None
    """Domain (e.g., 'github.com')."""

    platform: Optional[str] = None
    """Operating system platform."""

    tags: List[str] = field(default_factory=list)
    """User-provided tags."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional custom metadata."""


@dataclass
class MultimodalRetrievalResult:
    """A single multimodal retrieval result with score breakdown."""

    demo: MultimodalDemoMetadata
    """The demo metadata."""

    score: float
    """Combined retrieval score (higher is better)."""

    embedding_score: float
    """Embedding similarity component."""

    rerank_score: Optional[float] = None
    """Reranker score (if reranking was used)."""

    rank: int = 0
    """Rank in the result list (1-indexed)."""


@dataclass
class MultimodalRetrieverConfig:
    """Configuration for MultimodalDemoRetriever."""

    # Embedding model settings
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    """HuggingFace model name for embedding."""

    embedding_dim: Optional[int] = 512
    """Target embedding dimension (uses MRL). None for full dimension."""

    device: Optional[str] = None
    """Device for inference. Auto-detected if None."""

    # Reranker settings
    use_reranker: bool = False
    """Whether to use reranker for precision improvement."""

    reranker_model: str = "Qwen/Qwen3-VL-Reranker-2B"
    """HuggingFace model name for reranking."""

    rerank_top_n: int = 20
    """Number of candidates to rerank."""

    # Storage settings
    cache_dir: Optional[Path] = None
    """Directory for model and embedding cache."""

    index_path: Optional[Path] = None
    """Path for persisting the index."""

    # Retrieval settings
    use_faiss: bool = True
    """Whether to use FAISS for similarity search."""

    domain_bonus: float = 0.1
    """Bonus score for matching domain."""

    app_bonus: float = 0.1
    """Bonus score for matching app name."""


class MultimodalDemoRetriever:
    """Demo retriever with multimodal (text + image) support.

    This retriever uses Qwen3-VL-Embedding to create joint text+image
    embeddings for demos and queries, enabling visual similarity matching
    in addition to semantic text matching.

    Features:
    - Multimodal embedding: combines task description + screenshot
    - FAISS-accelerated similarity search
    - Optional reranking with Qwen3-VL-Reranker
    - Persistence to disk (embeddings + metadata)
    - Incremental index updates

    Example:
        >>> from openadapt_ml.retrieval import MultimodalDemoRetriever
        >>> from openadapt_ml.ingest.capture import capture_to_episode
        >>>
        >>> # Initialize retriever
        >>> retriever = MultimodalDemoRetriever(
        ...     embedding_model="Qwen/Qwen3-VL-Embedding-2B",
        ...     embedding_dim=512,  # Use MRL for smaller storage
        ... )
        >>>
        >>> # Index demos
        >>> for capture_dir in capture_dirs:
        ...     episode = capture_to_episode(capture_dir)
        ...     retriever.add_demo(episode)
        >>>
        >>> retriever.build_index()
        >>>
        >>> # Retrieve by task + current screenshot
        >>> results = retriever.retrieve(
        ...     task="Disable Night Shift on macOS",
        ...     screenshot="/path/to/current_screen.png",
        ...     top_k=3,
        ... )
        >>>
        >>> for result in results:
        ...     print(f"Found: {result.demo.goal} (score: {result.score:.3f})")
    """

    def __init__(
        self,
        config: Optional[MultimodalRetrieverConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the multimodal demo retriever.

        Args:
            config: Configuration object. If None, uses defaults.
            **kwargs: Override config fields (convenience).
        """
        if config is None:
            config = MultimodalRetrieverConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Set up cache directory
        if config.cache_dir is None:
            config.cache_dir = Path.home() / ".cache" / "openadapt_ml" / "multimodal"
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedder (lazy loaded)
        self._embedder: Optional[Qwen3VLEmbedder] = None
        self._reranker = None

        # Index state
        self._demos: List[MultimodalDemoMetadata] = []
        self._embeddings: Optional[NDArray[np.float32]] = None
        self._faiss_index = None
        self._is_indexed = False

    @property
    def embedder(self) -> Qwen3VLEmbedder:
        """Get or create the embedder (lazy initialization)."""
        if self._embedder is None:
            embedder_config = Qwen3VLEmbedderConfig(
                model_name=self.config.embedding_model,
                embedding_dim=self.config.embedding_dim,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
            )
            self._embedder = Qwen3VLEmbedder(config=embedder_config)
        return self._embedder

    # =========================================================================
    # Demo Management
    # =========================================================================

    def add_demo(
        self,
        episode: Episode,
        screenshot: Optional[Union[str, Path, Image.Image]] = None,
        demo_id: Optional[str] = None,
        app_name: Optional[str] = None,
        domain: Optional[str] = None,
        platform: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalDemoMetadata:
        """Add a demonstration episode to the library.

        Args:
            episode: The Episode to add.
            screenshot: Representative screenshot. If None, extracts from first step.
            demo_id: Unique ID. Auto-generated from episode_id if not provided.
            app_name: Application name. Auto-extracted if not provided.
            domain: Domain. Auto-extracted from URLs if not provided.
            platform: Platform. Auto-detected if not provided.
            tags: User-provided tags for categorization.
            metadata: Additional custom metadata.

        Returns:
            MultimodalDemoMetadata object for the added demo.
        """
        # Auto-generate demo_id
        if demo_id is None:
            demo_id = episode.episode_id

        # Auto-extract screenshot from first step if not provided
        screenshot_path = None
        if screenshot is not None:
            if isinstance(screenshot, (str, Path)):
                screenshot_path = str(screenshot)
        elif episode.steps:
            first_obs = episode.steps[0].observation
            if first_obs and first_obs.screenshot_path:
                screenshot_path = first_obs.screenshot_path

        # Auto-extract app_name
        if app_name is None:
            app_name = self._extract_app_name(episode)

        # Auto-extract domain
        if domain is None:
            domain = self._extract_domain(episode)

        # Auto-detect platform
        if platform is None:
            platform = self._detect_platform(episode, app_name, domain)

        demo_meta = MultimodalDemoMetadata(
            demo_id=demo_id,
            episode=episode,
            goal=episode.instruction,
            screenshot_path=screenshot_path,
            app_name=app_name,
            domain=domain,
            platform=platform,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._demos.append(demo_meta)
        self._is_indexed = False  # Need to rebuild index

        return demo_meta

    def add_demos(
        self,
        episodes: List[Episode],
        **kwargs: Any,
    ) -> List[MultimodalDemoMetadata]:
        """Add multiple demonstration episodes."""
        return [self.add_demo(ep, **kwargs) for ep in episodes]

    def get_demo_count(self) -> int:
        """Get the number of demos in the library."""
        return len(self._demos)

    def get_all_demos(self) -> List[MultimodalDemoMetadata]:
        """Get all demo metadata objects."""
        return list(self._demos)

    def clear(self) -> None:
        """Clear all demos and reset the index."""
        self._demos = []
        self._embeddings = None
        self._faiss_index = None
        self._is_indexed = False

    # =========================================================================
    # Indexing
    # =========================================================================

    def build_index(self, force: bool = False) -> None:
        """Build the search index from all added demos.

        This computes multimodal embeddings for all demos and builds
        the FAISS index for similarity search.

        Args:
            force: If True, rebuild even if already indexed.

        Raises:
            ValueError: If no demos have been added.
        """
        if self._is_indexed and not force:
            logger.debug("Index already built, skipping (use force=True to rebuild)")
            return

        if not self._demos:
            raise ValueError("Cannot build index: no demos added. Use add_demo() first.")

        logger.info(f"Building multimodal index for {len(self._demos)} demos...")

        # Prepare batch inputs
        inputs = []
        for demo in self._demos:
            inp = {
                "text": demo.goal,
            }
            if demo.screenshot_path:
                inp["image"] = demo.screenshot_path
            inputs.append(inp)

        # Compute embeddings
        self._embeddings = self.embedder.embed_batch(inputs)

        # Build FAISS index
        if self.config.use_faiss:
            self._build_faiss_index()

        self._is_indexed = True
        logger.info(f"Index built successfully with {len(self._demos)} demos")

    def _build_faiss_index(self) -> None:
        """Build FAISS index from embeddings."""
        try:
            import faiss

            embeddings = self._embeddings.astype(np.float32)
            dim = embeddings.shape[1]

            # Use IndexFlatIP for cosine similarity (assumes normalized embeddings)
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(embeddings)

            logger.debug(f"Built FAISS index with {len(embeddings)} vectors, dim={dim}")
        except ImportError:
            logger.debug("FAISS not available, using brute-force search")
            self._faiss_index = None

    # =========================================================================
    # Retrieval
    # =========================================================================

    def retrieve(
        self,
        task: str,
        screenshot: Optional[Union[str, Path, Image.Image]] = None,
        top_k: int = 3,
        app_context: Optional[str] = None,
        domain_context: Optional[str] = None,
    ) -> List[MultimodalRetrievalResult]:
        """Retrieve top-K most similar demos for a query.

        Args:
            task: Task description to find demos for.
            screenshot: Optional current screenshot for visual matching.
            top_k: Number of demos to retrieve.
            app_context: Optional app context for bonus scoring.
            domain_context: Optional domain context for bonus scoring.

        Returns:
            List of MultimodalRetrievalResult objects, ordered by relevance.

        Raises:
            ValueError: If index has not been built.
        """
        if not self._is_indexed:
            raise ValueError("Index not built. Call build_index() first.")

        if not self._demos:
            return []

        # Create query embedding
        query_input = {"text": task}
        if screenshot is not None:
            query_input["image"] = screenshot

        query_embedding = self.embedder.embed(**query_input)

        # Get candidates
        n_candidates = (
            self.config.rerank_top_n if self.config.use_reranker else top_k
        )
        n_candidates = min(n_candidates, len(self._demos))

        if self._faiss_index is not None:
            # FAISS search
            scores, indices = self._faiss_index.search(
                query_embedding.reshape(1, -1),
                n_candidates,
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Brute-force search
            scores = self._embeddings @ query_embedding
            indices = np.argsort(scores)[::-1][:n_candidates]
            scores = scores[indices]

        # Build results with context bonuses
        results = []
        for idx, score in zip(indices, scores):
            demo = self._demos[idx]
            bonus = self._compute_context_bonus(demo, app_context, domain_context)

            results.append(MultimodalRetrievalResult(
                demo=demo,
                score=float(score) + bonus,
                embedding_score=float(score),
            ))

        # Rerank if enabled
        if self.config.use_reranker and len(results) > top_k:
            results = self._rerank(task, screenshot, results)

        # Sort by final score and assign ranks
        results.sort(key=lambda r: r.score, reverse=True)
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1

        return results[:top_k]

    def retrieve_episodes(
        self,
        task: str,
        screenshot: Optional[Union[str, Path, Image.Image]] = None,
        top_k: int = 3,
        **kwargs: Any,
    ) -> List[Episode]:
        """Retrieve top-K episodes (convenience method)."""
        results = self.retrieve(task, screenshot=screenshot, top_k=top_k, **kwargs)
        return [r.demo.episode for r in results]

    def _compute_context_bonus(
        self,
        demo: MultimodalDemoMetadata,
        app_context: Optional[str],
        domain_context: Optional[str],
    ) -> float:
        """Compute context bonus for app/domain matching."""
        bonus = 0.0

        if app_context and demo.app_name:
            if app_context.lower() in demo.app_name.lower():
                bonus += self.config.app_bonus

        if domain_context and demo.domain:
            if domain_context.lower() in demo.domain.lower():
                bonus += self.config.domain_bonus

        return bonus

    def _rerank(
        self,
        task: str,
        screenshot: Optional[Union[str, Path, Image.Image]],
        candidates: List[MultimodalRetrievalResult],
    ) -> List[MultimodalRetrievalResult]:
        """Rerank candidates using Qwen3-VL-Reranker."""
        # TODO: Implement reranker integration
        logger.warning("Reranker not yet implemented, returning embedding scores")
        return candidates

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the index to disk.

        Args:
            path: Directory to save index files. Uses config.index_path if None.
        """
        if path is None:
            path = self.config.index_path
        if path is None:
            raise ValueError("No save path specified and config.index_path is None")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = []
        for demo in self._demos:
            meta = {
                "demo_id": demo.demo_id,
                "goal": demo.goal,
                "screenshot_path": demo.screenshot_path,
                "app_name": demo.app_name,
                "domain": demo.domain,
                "platform": demo.platform,
                "tags": demo.tags,
                "metadata": demo.metadata,
                "episode_file": f"{demo.demo_id}.json",
            }
            metadata.append(meta)

            # Save episode
            episode_path = path / f"{demo.demo_id}.json"
            with open(episode_path, "w") as f:
                f.write(demo.episode.to_json())

        # Save index config
        config_data = {
            "embedding_model": self.config.embedding_model,
            "embedding_dim": self.config.embedding_dim,
            "demos": metadata,
            "is_indexed": self._is_indexed,
        }

        with open(path / "index.json", "w") as f:
            json.dump(config_data, f, indent=2)

        # Save embeddings
        if self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)

        logger.info(f"Index saved to {path} with {len(self._demos)} demos")

    def load(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Load index from disk.

        Args:
            path: Directory containing index files. Uses config.index_path if None.
        """
        if path is None:
            path = self.config.index_path
        if path is None:
            raise ValueError("No load path specified and config.index_path is None")

        path = Path(path)

        with open(path / "index.json") as f:
            config_data = json.load(f)

        # Validate model compatibility
        if config_data.get("embedding_model") != self.config.embedding_model:
            logger.warning(
                f"Model mismatch: index uses {config_data.get('embedding_model')}, "
                f"current config uses {self.config.embedding_model}"
            )

        # Load demos
        self._demos = []
        for meta in config_data.get("demos", []):
            # Load episode
            episode_path = path / meta["episode_file"]
            if episode_path.exists():
                episode = Episode.from_json(episode_path.read_text())
            else:
                logger.warning(f"Episode file not found: {episode_path}")
                continue

            demo = MultimodalDemoMetadata(
                demo_id=meta["demo_id"],
                episode=episode,
                goal=meta["goal"],
                screenshot_path=meta.get("screenshot_path"),
                app_name=meta.get("app_name"),
                domain=meta.get("domain"),
                platform=meta.get("platform"),
                tags=meta.get("tags", []),
                metadata=meta.get("metadata", {}),
            )
            self._demos.append(demo)

        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            self._embeddings = np.load(embeddings_path)

            if self.config.use_faiss:
                self._build_faiss_index()

            self._is_indexed = config_data.get("is_indexed", False)

        logger.info(f"Index loaded from {path} with {len(self._demos)} demos")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_app_name(self, episode: Episode) -> Optional[str]:
        """Extract app name from episode observations."""
        for step in episode.steps:
            if step.observation and step.observation.app_name:
                return step.observation.app_name
        return None

    def _extract_domain(self, episode: Episode) -> Optional[str]:
        """Extract domain from episode URLs."""
        for step in episode.steps:
            if step.observation and step.observation.url:
                url = step.observation.url
                if "://" in url:
                    domain = url.split("://")[1].split("/")[0]
                    if domain.startswith("www."):
                        domain = domain[4:]
                    return domain
        return None

    def _detect_platform(
        self,
        episode: Episode,
        app_name: Optional[str],
        domain: Optional[str],
    ) -> Optional[str]:
        """Detect platform from episode context."""
        if domain:
            return "web"

        macos_apps = {"System Settings", "Finder", "Safari", "Preview", "TextEdit"}
        if app_name and app_name in macos_apps:
            return "macos"

        windows_apps = {"Settings", "File Explorer", "Microsoft Edge", "Notepad"}
        if app_name and app_name in windows_apps:
            return "windows"

        return None

    def __len__(self) -> int:
        """Return number of demos in the library."""
        return len(self._demos)

    def __repr__(self) -> str:
        """String representation."""
        status = "indexed" if self._is_indexed else "not indexed"
        return f"MultimodalDemoRetriever({len(self._demos)} demos, {self.config.embedding_model}, {status})"
```

### 4.4 Configuration Options Summary

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embedding_model` | str | `"Qwen/Qwen3-VL-Embedding-2B"` | HuggingFace model name |
| `embedding_dim` | int | `512` | Target dimension (uses MRL) |
| `device` | str | `None` (auto) | "cuda", "cpu", "mps" |
| `use_reranker` | bool | `False` | Enable two-stage retrieval |
| `reranker_model` | str | `"Qwen/Qwen3-VL-Reranker-2B"` | Reranker model name |
| `rerank_top_n` | int | `20` | Candidates to rerank |
| `use_faiss` | bool | `True` | Use FAISS for search |
| `domain_bonus` | float | `0.1` | Bonus for domain match |
| `app_bonus` | float | `0.1` | Bonus for app match |
| `cache_dir` | Path | `~/.cache/openadapt_ml/multimodal` | Cache directory |
| `index_path` | Path | `None` | Persistence path |

### 4.5 Example Usage

```python
# Basic usage
from openadapt_ml.retrieval import MultimodalDemoRetriever
from openadapt_ml.ingest.capture import capture_to_episode

# Initialize with defaults
retriever = MultimodalDemoRetriever()

# Or with custom config
retriever = MultimodalDemoRetriever(
    embedding_model="Qwen/Qwen3-VL-Embedding-2B",
    embedding_dim=512,      # Smaller for faster search
    use_reranker=False,     # Start without reranker
    device="cuda",
)

# Index demos
capture_dirs = [
    "/path/to/captures/turn-off-nightshift",
    "/path/to/captures/search-github",
]

for capture_dir in capture_dirs:
    episode = capture_to_episode(capture_dir)
    retriever.add_demo(episode)

retriever.build_index()

# Save for later use
retriever.save("/path/to/demo_index")

# Query with text only
results = retriever.retrieve(
    task="Disable Night Shift",
    top_k=3,
)

# Query with text + screenshot (recommended)
results = retriever.retrieve(
    task="Disable Night Shift",
    screenshot="/path/to/current_screen.png",
    top_k=3,
    app_context="System Settings",
)

# Format for prompt
for i, result in enumerate(results, 1):
    print(f"{i}. {result.demo.goal} (score: {result.score:.3f})")
    print(f"   App: {result.demo.app_name}")
    print(f"   Screenshot: {result.demo.screenshot_path}")
```

---

## 5. Storage and Persistence

### 5.1 Storage Architecture

```
index_directory/
├── index.json              # Index metadata and config
├── embeddings.npy          # Embedding matrix (float32)
├── faiss.index             # FAISS index (optional, for large indices)
├── episode_001.json        # Episode 1 JSON
├── episode_002.json        # Episode 2 JSON
└── ...
```

### 5.2 index.json Schema

```json
{
  "schema_version": "1.0.0",
  "created_at": "2026-01-16T10:30:00Z",
  "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
  "embedding_dim": 512,
  "is_indexed": true,
  "demo_count": 50,
  "demos": [
    {
      "demo_id": "turn-off-nightshift",
      "goal": "Turn off Night Shift in System Settings",
      "screenshot_path": "/path/to/screenshot.png",
      "app_name": "System Settings",
      "domain": null,
      "platform": "macos",
      "tags": ["settings", "display"],
      "metadata": {},
      "episode_file": "turn-off-nightshift.json"
    }
  ]
}
```

### 5.3 Incremental Updates

The retriever supports incremental index updates:

```python
# Load existing index
retriever.load("/path/to/demo_index")

# Add new demos
retriever.add_demo(new_episode)

# Rebuild only the new embeddings
retriever.build_index(incremental=True)  # Future feature

# For v1: rebuild full index after adding
retriever.build_index(force=True)

# Save updated index
retriever.save()
```

### 5.4 FAISS Index Types

| Index Type | Use Case | Storage | Build Time | Query Time |
|------------|----------|---------|------------|------------|
| `IndexFlatIP` | < 10K vectors | O(n*d) | O(n) | O(n) |
| `IndexIVFFlat` | 10K-1M vectors | O(n*d) | O(n*k) | O(n/k) |
| `IndexHNSW` | > 100K vectors | O(n*d*M) | O(n*log(n)) | O(log(n)) |

For v1, we use `IndexFlatIP` (exact search) since demo libraries are typically < 1000 demos.

### 5.5 Embedding Serialization

Embeddings are stored as numpy arrays for efficiency:

```python
# Save embeddings
np.save(path / "embeddings.npy", embeddings)

# Load embeddings
embeddings = np.load(path / "embeddings.npy")

# Memory-mapped loading for large indices (future)
embeddings = np.load(path / "embeddings.npy", mmap_mode="r")
```

---

## 6. Performance Considerations

### 6.1 Batch Processing for Index Building

Building embeddings for many demos should use batching:

```python
def build_index(self, batch_size: int = 8) -> None:
    """Build index with batched embedding computation."""
    inputs = [...]  # Prepare all inputs

    embeddings = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_embeddings = self.embedder.embed_batch(batch)
        embeddings.append(batch_embeddings)

    self._embeddings = np.vstack(embeddings)
```

**Recommended batch sizes by hardware**:
| GPU | VRAM | Batch Size (2B) | Batch Size (8B) |
|-----|------|-----------------|-----------------|
| RTX 3060 | 12 GB | 4-8 | 1-2 |
| RTX 4090 | 24 GB | 16-32 | 4-8 |
| A100 | 40 GB | 32-64 | 8-16 |

### 6.2 Lazy Model Loading

The model is only loaded when first needed:

```python
class Qwen3VLEmbedder:
    def __init__(self, ...):
        self._model = None  # Not loaded yet

    def embed(self, ...):
        self._load_model()  # Load on first use
        ...

    def unload_model(self):
        """Free GPU memory when not needed."""
        del self._model
        self._model = None
        torch.cuda.empty_cache()
```

### 6.3 Caching Strategies

**Model caching**: Models are cached in `~/.cache/huggingface/hub/` by transformers.

**Embedding caching**: For repeated queries, cache embeddings in memory:

```python
class Qwen3VLEmbedder:
    def __init__(self, ...):
        self._cache = {}  # text -> embedding
        self._image_cache = {}  # image_path -> embedding

    def embed(self, text=None, image=None, ...):
        cache_key = (text, str(image) if image else None)
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self._compute_embedding(text, image)
        self._cache[cache_key] = embedding
        return embedding
```

### 6.4 Memory Management

**Embedding storage**: With 512-dim embeddings, memory usage is:
- 1000 demos: 1000 * 512 * 4 bytes = 2 MB
- 10000 demos: 20 MB
- 100000 demos: 200 MB

**Model memory**: Keep only one model loaded at a time:

```python
def retrieve_with_rerank(self, ...):
    # Use embedding model
    query_emb = self.embedder.embed(...)
    candidates = self._search(query_emb)

    # Unload embedder, load reranker
    self.embedder.unload_model()
    reranked = self._rerank(candidates)

    return reranked
```

### 6.5 Performance Benchmarks (Expected)

| Operation | Demo Count | Time (RTX 4090) | Time (CPU) |
|-----------|------------|-----------------|------------|
| Index 1 demo | 1 | ~200ms | ~2s |
| Index 100 demos | 100 | ~15s | ~3min |
| Index 1000 demos | 1000 | ~2.5min | ~30min |
| Query (text only) | any | ~100ms | ~1s |
| Query (text+image) | any | ~150ms | ~2s |
| Query with rerank | 20 candidates | ~500ms | ~10s |

---

## 7. Hardware Requirements

### 7.1 Minimum Requirements

**Qwen3-VL-Embedding-2B** (Recommended for development):

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (12GB) | RTX 4090 (24GB) |
| VRAM | 6 GB (FP16) | 8 GB |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB (model cache) | 20 GB |

**Qwen3-VL-Embedding-8B** (Higher quality):

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 4090 (24GB) | A100 (40GB) |
| VRAM | 18 GB (FP16) | 24 GB |
| RAM | 32 GB | 64 GB |
| Storage | 20 GB (model cache) | 30 GB |

### 7.2 CPU-Only Mode

For machines without GPU:

```python
retriever = MultimodalDemoRetriever(
    device="cpu",
    embedding_dim=256,  # Smaller dimension for speed
)
```

Performance impact:
- ~10x slower embedding computation
- No batching benefits
- Still usable for small demo libraries (< 100)

### 7.3 Apple Silicon (MPS)

Supported on M1/M2/M3 Macs:

```python
retriever = MultimodalDemoRetriever(
    device="mps",
)
```

Performance:
- M1 Pro: ~500ms per embedding (2B model)
- M2 Max: ~300ms per embedding
- M3 Max: ~200ms per embedding

### 7.4 Quantization Options

For reduced memory usage:

```python
# INT8 quantization (requires bitsandbytes)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Apply in model loading (future enhancement)
embedder = Qwen3VLEmbedder(
    quantization_config=quantization_config,  # Not yet implemented
)
```

Memory savings with quantization:
- 2B FP16: ~6 GB -> INT8: ~4 GB
- 8B FP16: ~18 GB -> INT8: ~10 GB

### 7.5 Fallback Strategy

When hardware requirements aren't met:

```python
def create_retriever(prefer_multimodal: bool = True) -> Union[MultimodalDemoRetriever, DemoRetriever]:
    """Create appropriate retriever based on hardware."""

    if prefer_multimodal and torch.cuda.is_available():
        try:
            # Try to load multimodal embedder
            retriever = MultimodalDemoRetriever()
            retriever.embedder._load_model()  # Test loading
            return retriever
        except (RuntimeError, OutOfMemoryError):
            logger.warning("Insufficient GPU memory, falling back to text-only")

    # Fallback to text-only retriever
    return DemoRetriever(embedding_method="sentence_transformers")
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Test file**: `tests/retrieval/test_multimodal_embeddings.py`

```python
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from openadapt_ml.retrieval.multimodal_embeddings import (
    Qwen3VLEmbedder,
    Qwen3VLEmbedderConfig,
)


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image for testing."""
    img = Image.new("RGB", (256, 256), color="red")
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


class TestQwen3VLEmbedder:
    """Unit tests for Qwen3VLEmbedder."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_embed_text_only(self):
        """Test text-only embedding."""
        embedder = Qwen3VLEmbedder(embedding_dim=512)

        embedding = embedder.embed(text="Turn off Night Shift")

        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_embed_image_only(self, sample_image):
        """Test image-only embedding."""
        embedder = Qwen3VLEmbedder(embedding_dim=512)

        embedding = embedder.embed(image=sample_image)

        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_embed_multimodal(self, sample_image):
        """Test text+image embedding."""
        embedder = Qwen3VLEmbedder(embedding_dim=512)

        embedding = embedder.embed(
            text="Turn off Night Shift",
            image=sample_image,
        )

        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_embed_batch(self, sample_image):
        """Test batch embedding."""
        embedder = Qwen3VLEmbedder(embedding_dim=512)

        inputs = [
            {"text": "Task 1"},
            {"text": "Task 2", "image": sample_image},
            {"image": sample_image},
        ]

        embeddings = embedder.embed_batch(inputs)

        assert embeddings.shape == (3, 512)

    def test_invalid_input(self):
        """Test that empty input raises error."""
        embedder = Qwen3VLEmbedder(embedding_dim=512)

        with pytest.raises(ValueError, match="at least one of text or image"):
            embedder.embed()

    def test_config_validation(self):
        """Test config validation."""
        # Invalid embedding_dim
        with pytest.raises(ValueError, match="exceeds model maximum"):
            Qwen3VLEmbedder(embedding_dim=5000)

        with pytest.raises(ValueError, match="must be at least 64"):
            Qwen3VLEmbedder(embedding_dim=32)
```

### 8.2 Integration Tests

**Test file**: `tests/retrieval/test_multimodal_retriever.py`

```python
import pytest
import numpy as np
from pathlib import Path

from openadapt_ml.schema import Episode, Step, Action, Observation, ActionType
from openadapt_ml.retrieval import MultimodalDemoRetriever


@pytest.fixture
def sample_episodes(tmp_path):
    """Create sample episodes with screenshots."""
    episodes = []

    for i, (task, app) in enumerate([
        ("Turn off Night Shift", "System Settings"),
        ("Search for Python on GitHub", "Chrome"),
        ("Open Calculator app", "Finder"),
    ]):
        # Create dummy screenshot
        from PIL import Image
        img = Image.new("RGB", (256, 256), color=f"#{i*50:02x}0000")
        screenshot_path = tmp_path / f"screenshot_{i}.png"
        img.save(screenshot_path)

        episode = Episode(
            episode_id=f"demo_{i}",
            instruction=task,
            steps=[
                Step(
                    step_index=0,
                    observation=Observation(
                        screenshot_path=str(screenshot_path),
                        app_name=app,
                    ),
                    action=Action(type=ActionType.CLICK),
                )
            ],
        )
        episodes.append(episode)

    return episodes


class TestMultimodalDemoRetriever:
    """Integration tests for MultimodalDemoRetriever."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_add_and_retrieve(self, sample_episodes):
        """Test basic add and retrieve workflow."""
        retriever = MultimodalDemoRetriever(embedding_dim=512)

        for episode in sample_episodes:
            retriever.add_demo(episode)

        retriever.build_index()

        results = retriever.retrieve(
            task="Disable Night Shift",
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2
        # Night Shift demo should be top result
        assert "Night Shift" in results[0].demo.goal

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_multimodal_query(self, sample_episodes, tmp_path):
        """Test query with screenshot."""
        retriever = MultimodalDemoRetriever(embedding_dim=512)

        for episode in sample_episodes:
            retriever.add_demo(episode)

        retriever.build_index()

        # Create query screenshot
        from PIL import Image
        query_img = Image.new("RGB", (256, 256), color="red")
        query_path = tmp_path / "query.png"
        query_img.save(query_path)

        results = retriever.retrieve(
            task="Turn off display setting",
            screenshot=query_path,
            top_k=2,
        )

        assert len(results) == 2
        assert all(r.embedding_score >= 0 for r in results)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_save_and_load(self, sample_episodes, tmp_path):
        """Test persistence."""
        # Build and save
        retriever = MultimodalDemoRetriever(
            embedding_dim=512,
            index_path=tmp_path / "test_index",
        )

        for episode in sample_episodes:
            retriever.add_demo(episode)

        retriever.build_index()
        retriever.save()

        # Load in new retriever
        retriever2 = MultimodalDemoRetriever(
            embedding_dim=512,
            index_path=tmp_path / "test_index",
        )
        retriever2.load()

        assert len(retriever2) == len(sample_episodes)

        # Verify retrieval works
        results = retriever2.retrieve("Night Shift", top_k=1)
        assert len(results) == 1
```

### 8.3 Mock Tests (No GPU Required)

For CI/CD without GPU:

```python
# tests/retrieval/test_multimodal_mock.py

import pytest
from unittest.mock import Mock, patch
import numpy as np

from openadapt_ml.retrieval import MultimodalDemoRetriever


class TestMultimodalRetrieverMock:
    """Tests using mocked embedder."""

    def test_retriever_with_mock_embedder(self, sample_episodes):
        """Test retriever logic with mocked embeddings."""
        # Create mock embedder
        mock_embedder = Mock()
        mock_embedder.embedding_dim = 512
        mock_embedder.embed.return_value = np.random.randn(512).astype(np.float32)
        mock_embedder.embed_batch.return_value = np.random.randn(3, 512).astype(np.float32)

        with patch.object(MultimodalDemoRetriever, 'embedder', mock_embedder):
            retriever = MultimodalDemoRetriever()

            for episode in sample_episodes:
                retriever.add_demo(episode)

            retriever.build_index()

            results = retriever.retrieve("test query", top_k=2)

            assert len(results) == 2
            mock_embedder.embed.assert_called()
```

### 8.4 Benchmark Tests

**Test file**: `tests/retrieval/test_multimodal_benchmark.py`

```python
import pytest
import time
import numpy as np

from openadapt_ml.retrieval import MultimodalDemoRetriever


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.gpu
class TestMultimodalBenchmarks:
    """Performance benchmarks for multimodal retrieval."""

    def test_embedding_latency(self, sample_image):
        """Benchmark single embedding latency."""
        embedder = Qwen3VLEmbedder(embedding_dim=512)

        # Warm up
        embedder.embed(text="warmup")

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            embedder.embed(text="Turn off Night Shift", image=sample_image)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"Embedding latency: {avg_time*1000:.1f} +/- {std_time*1000:.1f} ms")

        # Assert reasonable latency
        assert avg_time < 1.0, "Embedding should take < 1 second"

    def test_retrieval_throughput(self, sample_episodes):
        """Benchmark retrieval throughput."""
        retriever = MultimodalDemoRetriever(embedding_dim=512)

        for episode in sample_episodes * 10:  # 30 demos
            retriever.add_demo(episode)

        retriever.build_index()

        # Measure
        times = []
        for _ in range(20):
            start = time.perf_counter()
            retriever.retrieve("test query", top_k=5)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        qps = 1.0 / avg_time

        print(f"Retrieval: {avg_time*1000:.1f} ms ({qps:.1f} QPS)")

        assert avg_time < 0.5, "Retrieval should take < 500ms"

    def test_compare_text_vs_multimodal(self, sample_episodes, sample_image):
        """Compare text-only vs multimodal retrieval quality."""
        from openadapt_ml.retrieval import DemoRetriever

        # Text-only retriever
        text_retriever = DemoRetriever(embedding_method="sentence_transformers")
        for episode in sample_episodes:
            text_retriever.add_demo(episode)
        text_retriever.build_index()

        # Multimodal retriever
        mm_retriever = MultimodalDemoRetriever(embedding_dim=512)
        for episode in sample_episodes:
            mm_retriever.add_demo(episode)
        mm_retriever.build_index()

        # Query
        query = "Disable display brightness setting"

        text_results = text_retriever.retrieve(query, top_k=3)
        mm_results = mm_retriever.retrieve(query, screenshot=sample_image, top_k=3)

        print(f"Text-only results: {[r.demo.goal for r in text_results]}")
        print(f"Multimodal results: {[r.demo.goal for r in mm_results]}")

        # Both should return results
        assert len(text_results) > 0
        assert len(mm_results) > 0
```

---

## 9. Migration Path

### 9.1 Backward Compatibility

The existing `DemoRetriever` class remains unchanged and is the default:

```python
# Existing code continues to work
from openadapt_ml.retrieval import DemoRetriever

retriever = DemoRetriever(embedding_method="sentence_transformers")
retriever.add_demo(episode)
retriever.build_index()
results = retriever.retrieve("query")
```

### 9.2 Feature Flag / Configuration

Add multimodal as an option in `DemoRetriever`:

```python
# Option 1: New class (recommended for v1)
from openadapt_ml.retrieval import MultimodalDemoRetriever
retriever = MultimodalDemoRetriever()

# Option 2: Unified interface (future enhancement)
from openadapt_ml.retrieval import DemoRetriever

# Text-only (default)
retriever = DemoRetriever(embedding_method="sentence_transformers")

# Multimodal
retriever = DemoRetriever(embedding_method="qwen3vl")
```

### 9.3 CLI Flag

Update CLI to support multimodal:

```bash
# Current (text-only)
python -m openadapt_ml.retrieval.cli index --method sentence_transformers

# New (multimodal)
python -m openadapt_ml.retrieval.cli index --method qwen3vl --embedding-dim 512
```

### 9.4 Gradual Adoption Path

1. **Phase 1**: Add `MultimodalDemoRetriever` as separate class (no breaking changes)
2. **Phase 2**: Add `qwen3vl` as embedding method option in `DemoRetriever`
3. **Phase 3**: Make multimodal the default when GPU is available

### 9.5 Index Migration

Existing text-only indices can be upgraded:

```python
def migrate_text_index_to_multimodal(
    text_index_path: Path,
    multimodal_index_path: Path,
    screenshot_resolver: Callable[[str], str],  # demo_id -> screenshot_path
) -> None:
    """Migrate text-only index to multimodal.

    Args:
        text_index_path: Path to existing text-only index.
        multimodal_index_path: Path for new multimodal index.
        screenshot_resolver: Function to find screenshot for each demo.
    """
    from openadapt_ml.retrieval import DemoRetriever, MultimodalDemoRetriever

    # Load text index
    text_retriever = DemoRetriever()
    text_retriever.load_index(text_index_path)

    # Create multimodal index
    mm_retriever = MultimodalDemoRetriever(index_path=multimodal_index_path)

    for demo in text_retriever.get_all_demos():
        screenshot_path = screenshot_resolver(demo.demo_id)
        mm_retriever.add_demo(
            demo.episode,
            screenshot=screenshot_path,
            app_name=demo.app_name,
            domain=demo.domain,
            tags=demo.tags,
        )

    mm_retriever.build_index()
    mm_retriever.save()

    logger.info(f"Migrated {len(mm_retriever)} demos to multimodal index")
```

---

## 10. Implementation Plan

### 10.1 Phase Overview

| Phase | Duration | Focus | Dependencies |
|-------|----------|-------|--------------|
| **1: Foundation** | 1 week | Core classes, basic embedding | None |
| **2: Integration** | 1 week | Retriever, persistence, CLI | Phase 1 |
| **3: Optimization** | 1 week | Batch processing, caching, MRL | Phase 2 |
| **4: Production** | Ongoing | Reranker, quantization, docs | Phase 3 |

### 10.2 Phase 1: Foundation (Week 1)

**Goal**: Working `Qwen3VLEmbedder` class with basic embedding functionality.

**Tasks**:

- [ ] **P1-1**: Create `multimodal_embeddings.py` with `MultimodalEmbedder` ABC
- [ ] **P1-2**: Implement `Qwen3VLEmbedder` class
  - [ ] Model loading with device selection
  - [ ] Image preprocessing
  - [ ] Text+image embedding
  - [ ] L2 normalization
- [ ] **P1-3**: Unit tests for embedder (text-only, image-only, multimodal)
- [ ] **P1-4**: Basic documentation

**Deliverables**:
- `openadapt_ml/retrieval/multimodal_embeddings.py`
- `tests/retrieval/test_multimodal_embeddings.py`

**Dependencies**: None (can start immediately)

### 10.3 Phase 2: Integration (Week 2)

**Goal**: Working `MultimodalDemoRetriever` with persistence.

**Tasks**:

- [ ] **P2-1**: Create `multimodal_retriever.py` with `MultimodalDemoRetriever`
  - [ ] `add_demo()` with screenshot extraction
  - [ ] `build_index()` with FAISS
  - [ ] `retrieve()` with optional screenshot query
  - [ ] `save()` and `load()` for persistence
- [ ] **P2-2**: Update `__init__.py` exports
- [ ] **P2-3**: Integration tests
- [ ] **P2-4**: CLI support for multimodal indexing

**Deliverables**:
- `openadapt_ml/retrieval/multimodal_retriever.py`
- Updated `openadapt_ml/retrieval/__init__.py`
- `tests/retrieval/test_multimodal_retriever.py`

**Dependencies**: Phase 1 complete

### 10.4 Phase 3: Optimization (Week 3)

**Goal**: Production-quality performance and memory management.

**Tasks**:

- [ ] **P3-1**: Batch embedding with configurable batch size
- [ ] **P3-2**: Embedding caching (in-memory and disk)
- [ ] **P3-3**: MRL dimension selection (64, 128, 256, 512, 1024, 2048)
- [ ] **P3-4**: Lazy model loading and unloading
- [ ] **P3-5**: Memory profiling and optimization
- [ ] **P3-6**: Performance benchmarks

**Deliverables**:
- Optimized `Qwen3VLEmbedder` with batching
- `tests/retrieval/test_multimodal_benchmark.py`
- Performance documentation

**Dependencies**: Phase 2 complete

### 10.5 Phase 4: Production (Ongoing)

**Goal**: Full-featured production system.

**Tasks**:

- [ ] **P4-1**: Qwen3-VL-Reranker integration
- [ ] **P4-2**: INT8 quantization support
- [ ] **P4-3**: Incremental index updates
- [ ] **P4-4**: A/B testing framework (text vs multimodal)
- [ ] **P4-5**: Comprehensive documentation
- [ ] **P4-6**: Example notebooks

**Deliverables**:
- `openadapt_ml/retrieval/reranker.py`
- User guide and API reference
- Example notebooks

**Dependencies**: Phase 3 complete

### 10.6 Parallelization Opportunities

Tasks that can run in parallel:

```
Phase 1:
  P1-1 (ABC) ──┬── P1-2 (Embedder impl)
               │
  P1-3 (Tests) ─┘

Phase 2:
  P2-1 (Retriever) ──┬── P2-3 (Tests)
                     │
  P2-2 (Exports) ────┴── P2-4 (CLI)

Phase 3:
  P3-1 (Batching) ───┬── P3-5 (Memory)
  P3-2 (Caching) ────┤
  P3-3 (MRL) ────────┤
  P3-4 (Lazy load) ──┴── P3-6 (Benchmarks)
```

### 10.7 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model API changes | Pin transformers version, track upstream releases |
| GPU memory issues | Test on various hardware, implement graceful degradation |
| Slow inference | Profile early, optimize hot paths, consider vLLM |
| Index corruption | Atomic writes, checksum validation |
| Breaking changes | Comprehensive test suite, semantic versioning |

---

## Appendix A: References

- [Qwen3-VL-Embedding HuggingFace](https://huggingface.co/collections/Qwen/qwen3-vl-embedding)
- [Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [arXiv Paper](https://arxiv.org/abs/2601.04720)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **MRL** | Matryoshka Representation Learning - technique for flexible embedding dimensions |
| **EOS** | End-of-Sequence token - used to extract final embedding from transformer |
| **FAISS** | Facebook AI Similarity Search - library for efficient vector similarity search |
| **IndexFlatIP** | FAISS index type using inner product (dot product) for similarity |
| **Cross-encoder** | Reranker architecture that processes query-document pairs jointly |

---

## Appendix C: Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-16 | Initial design document |
