"""Embedding functions for demo retrieval.

Supports multiple embedding backends:
- TF-IDF: Simple baseline, no external dependencies
- Sentence Transformers: Local embedding models (recommended)
- OpenAI: API-based embeddings

All embedders implement the same interface:
- embed(text: str) -> numpy.ndarray
- embed_batch(texts: List[str]) -> numpy.ndarray

Example:
    from openadapt_ml.retrieval.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    embeddings = embedder.embed_batch(["Turn off Night Shift", "Search GitHub"])
    print(embeddings.shape)  # (2, 384)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from math import log, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, text: str) -> Any:
        """Embed a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector (numpy array or dict for sparse).
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> Any:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Embeddings matrix (numpy array of shape [n_texts, embedding_dim]).
        """
        pass

    def cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        import numpy as np

        vec1 = np.asarray(vec1, dtype=np.float32).flatten()
        vec2 = np.asarray(vec2, dtype=np.float32).flatten()

        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


# =============================================================================
# TF-IDF Embedder (Baseline)
# =============================================================================


class TFIDFEmbedder(BaseEmbedder):
    """Simple TF-IDF based text embedder.

    This is a minimal implementation for baseline/testing that doesn't require
    any external ML libraries. Uses sparse representations internally but
    converts to dense for compatibility.

    Note: Must call fit() before embed() to build vocabulary.
    """

    def __init__(self, max_features: int = 1000) -> None:
        """Initialize the TF-IDF embedder.

        Args:
            max_features: Maximum vocabulary size.
        """
        self.max_features = max_features
        self.documents: List[str] = []
        self.idf: Dict[str, float] = {}
        self.vocab: List[str] = []
        self.vocab_to_idx: Dict[str, int] = {}
        self._is_fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency for a document.

        Args:
            tokens: List of tokens.

        Returns:
            Dictionary mapping term to frequency.
        """
        counter = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        return {term: count / total for term, count in counter.items()}

    def fit(self, documents: List[str]) -> "TFIDFEmbedder":
        """Fit the IDF on a corpus of documents.

        Args:
            documents: List of text documents.

        Returns:
            self for chaining.
        """
        self.documents = documents

        # Count document frequency for each term
        doc_freq: Dict[str, int] = {}
        all_terms: Counter[str] = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            all_terms.update(tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Select top features by frequency
        top_terms = [term for term, _ in all_terms.most_common(self.max_features)]
        self.vocab = top_terms
        self.vocab_to_idx = {term: idx for idx, term in enumerate(top_terms)}

        # Compute IDF: log(N / df) + 1
        n_docs = max(len(documents), 1)
        self.idf = {
            term: log(n_docs / doc_freq.get(term, 1)) + 1
            for term in self.vocab
        }

        self._is_fitted = True
        return self

    def embed(self, text: str) -> Any:
        """Convert text to TF-IDF vector.

        Args:
            text: Input text.

        Returns:
            Dense embedding vector (numpy array).
        """
        import numpy as np

        if not self._is_fitted:
            # Fit on single document for compatibility
            self.fit([text])

        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)

        # Create dense vector
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for term, tf_val in tf.items():
            if term in self.vocab_to_idx:
                idx = self.vocab_to_idx[term]
                vec[idx] = tf_val * self.idf.get(term, 1.0)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def embed_batch(self, texts: List[str]) -> Any:
        """Embed multiple texts.

        If not fitted, fits on the input texts first.

        Args:
            texts: List of texts to embed.

        Returns:
            Embeddings matrix (numpy array).
        """
        import numpy as np

        if not self._is_fitted:
            self.fit(texts)

        embeddings = np.array([self.embed(text) for text in texts], dtype=np.float32)
        return embeddings


# Alias for backward compatibility
TextEmbedder = TFIDFEmbedder


# =============================================================================
# Sentence Transformers Embedder
# =============================================================================


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding using sentence-transformers library.

    Recommended models:
    - "all-MiniLM-L6-v2": Fast, 22MB, 384 dims (default)
    - "all-mpnet-base-v2": Better quality, 420MB, 768 dims
    - "BAAI/bge-small-en-v1.5": Good balance, 130MB, 384 dims
    - "BAAI/bge-base-en-v1.5": Best quality, 440MB, 768 dims

    Requires: pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        """Initialize the Sentence Transformer embedder.

        Args:
            model_name: Name of the sentence-transformers model.
            cache_dir: Directory for caching model and embeddings.
            device: Device to run on ("cpu", "cuda", "mps"). Auto-detected if None.
            normalize: Whether to L2-normalize embeddings (for cosine similarity).
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.normalize = normalize
        self._model = None
        self._embedding_cache: Dict[str, Any] = {}

    def _load_model(self) -> None:
        """Lazy-load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading sentence-transformers model: {self.model_name}")
        self._model = SentenceTransformer(
            self.model_name,
            cache_folder=str(self.cache_dir) if self.cache_dir else None,
            device=self.device,
        )

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

    def embed(self, text: str) -> Any:
        """Embed a single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector (numpy array).
        """
        import numpy as np

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        self._load_model()

        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        embedding = np.asarray(embedding, dtype=np.float32)

        # Cache result
        self._embedding_cache[cache_key] = embedding

        return embedding

    def embed_batch(self, texts: List[str]) -> Any:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts.

        Returns:
            Embeddings matrix (numpy array of shape [n_texts, dim]).
        """
        import numpy as np

        self._load_model()

        # Check which texts are cached
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                cached_embeddings[i] = self._embedding_cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self._model.encode(
                uncached_texts,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=len(uncached_texts) > 10,
            )

            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text)
                self._embedding_cache[cache_key] = embedding

        # Reassemble in original order
        dim = self._model.get_sentence_embedding_dimension()
        result = np.zeros((len(texts), dim), dtype=np.float32)

        for i, emb in cached_embeddings.items():
            result[i] = emb

        for i, idx in enumerate(uncached_indices):
            result[idx] = new_embeddings[i]

        return result

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache = {}


# =============================================================================
# OpenAI Embedder
# =============================================================================


class OpenAIEmbedder(BaseEmbedder):
    """Embedding using OpenAI's text-embedding API.

    Models:
    - "text-embedding-3-small": Cheap, fast, 1536 dims ($0.00002/1K tokens)
    - "text-embedding-3-large": Best quality, 3072 dims ($0.00013/1K tokens)
    - "text-embedding-ada-002": Legacy, 1536 dims

    Requires: pip install openai
    Environment: OPENAI_API_KEY must be set
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        cache_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 100,
    ) -> None:
        """Initialize the OpenAI embedder.

        Args:
            model_name: OpenAI embedding model name.
            cache_dir: Directory for caching embeddings to disk.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            normalize: Whether to L2-normalize embeddings.
            batch_size: Maximum texts per API call.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.api_key = api_key
        self.normalize = normalize
        self.batch_size = batch_size
        self._client = None
        self._embedding_cache: Dict[str, Any] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        """Load cache from disk."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "embeddings_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                    # Convert lists back to arrays
                    import numpy as np
                    for key, val in cached.items():
                        self._embedding_cache[key] = np.array(val, dtype=np.float32)
                logger.debug(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_disk_cache(self) -> None:
        """Save cache to disk."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "embeddings_cache.json"
        try:
            # Convert arrays to lists for JSON
            cache_data = {
                key: val.tolist()
                for key, val in self._embedding_cache.items()
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIEmbedder. "
                "Install with: pip install openai"
            )

        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

    def embed(self, text: str) -> Any:
        """Embed a single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector (numpy array).
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> Any:
        """Embed multiple texts.

        Args:
            texts: List of texts.

        Returns:
            Embeddings matrix (numpy array).
        """
        import numpy as np

        # Check cache first
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                cached_embeddings[i] = self._embedding_cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Fetch uncached embeddings from API
        new_embeddings = {}
        if uncached_texts:
            client = self._get_client()

            # Process in batches
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch_texts = uncached_texts[batch_start:batch_start + self.batch_size]

                try:
                    response = client.embeddings.create(
                        model=self.model_name,
                        input=batch_texts,
                    )

                    for j, item in enumerate(response.data):
                        idx = uncached_indices[batch_start + j]
                        embedding = np.array(item.embedding, dtype=np.float32)

                        if self.normalize:
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm

                        new_embeddings[idx] = embedding

                        # Cache the result
                        cache_key = self._get_cache_key(batch_texts[j])
                        self._embedding_cache[cache_key] = embedding

                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
                    raise

            # Save to disk cache periodically
            self._save_disk_cache()

        # Determine embedding dimension
        if cached_embeddings:
            dim = next(iter(cached_embeddings.values())).shape[0]
        elif new_embeddings:
            dim = next(iter(new_embeddings.values())).shape[0]
        else:
            # Default dimensions by model
            dim = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(
                self.model_name, 1536
            )

        # Assemble result
        result = np.zeros((len(texts), dim), dtype=np.float32)

        for i, emb in cached_embeddings.items():
            result[i] = emb

        for i, emb in new_embeddings.items():
            result[i] = emb

        return result

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache = {}
        if self.cache_dir:
            cache_file = self.cache_dir / "embeddings_cache.json"
            if cache_file.exists():
                cache_file.unlink()


# =============================================================================
# Factory Function
# =============================================================================


def create_embedder(
    method: str = "tfidf",
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    **kwargs: Any,
) -> BaseEmbedder:
    """Factory function to create an embedder.

    Args:
        method: Embedding method ("tfidf", "sentence_transformers", "openai").
        model_name: Model name (method-specific defaults if None).
        cache_dir: Cache directory for embeddings.
        **kwargs: Additional arguments passed to embedder.

    Returns:
        Embedder instance.
    """
    if method == "tfidf":
        return TFIDFEmbedder(**kwargs)

    elif method == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=model_name or "all-MiniLM-L6-v2",
            cache_dir=cache_dir,
            **kwargs,
        )

    elif method == "openai":
        return OpenAIEmbedder(
            model_name=model_name or "text-embedding-3-small",
            cache_dir=cache_dir,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown embedding method: {method}")
