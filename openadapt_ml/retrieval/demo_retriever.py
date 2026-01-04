"""Main Demo Retriever class for finding similar demonstrations.

This module provides the DemoRetriever class that indexes demos by their task
descriptions using embeddings and retrieves the most similar demo(s) from a library.

Key features:
- Supports both local embeddings (sentence-transformers) and API embeddings (OpenAI)
- Uses FAISS or simple cosine similarity for vector search
- Caches embeddings to avoid recomputing
- Returns top-k most similar demos with formatting for prompts

Example usage:
    from openadapt_ml.retrieval import DemoRetriever
    from openadapt_ml.schema import Episode

    # Create retriever with local embeddings
    retriever = DemoRetriever(embedding_method="sentence_transformers")

    # Add demos
    retriever.add_demo(episode1)
    retriever.add_demo(episode2, app_name="Chrome", domain="github.com")

    # Build index
    retriever.build_index()

    # Retrieve similar demos
    results = retriever.retrieve("Turn off Night Shift", top_k=3)

    # Format for prompt
    prompt_text = retriever.format_for_prompt(results)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from openadapt_ml.schema import Episode

logger = logging.getLogger(__name__)


@dataclass
class DemoMetadata:
    """Metadata for a single demonstration.

    Stores both the episode and computed features for retrieval.

    Attributes:
        demo_id: Unique identifier for the demo.
        episode: The full Episode object.
        goal: Task description/instruction.
        app_name: Optional application name (e.g., "System Settings").
        domain: Optional domain (e.g., "github.com").
        platform: Operating system platform ("macos", "windows", "web").
        action_types: List of action types used in the demo.
        key_elements: Important UI elements touched.
        step_count: Number of steps in the demo.
        tags: User-provided tags for categorization.
        file_path: Path to the source file (if loaded from disk).
        embedding: Computed embedding vector (numpy array).
        metadata: Additional custom metadata.
    """

    demo_id: str
    episode: Episode
    goal: str
    app_name: Optional[str] = None
    domain: Optional[str] = None
    platform: Optional[str] = None
    action_types: List[str] = field(default_factory=list)
    key_elements: List[str] = field(default_factory=list)
    step_count: int = 0
    tags: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    embedding: Optional[Any] = None  # numpy array when computed
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single retrieval result with score breakdown.

    Attributes:
        demo: The demo metadata.
        score: Combined retrieval score (higher is better).
        text_score: Text/embedding similarity component.
        domain_bonus: Domain match bonus applied.
        rank: Rank in the result list (1-indexed).
    """

    demo: DemoMetadata
    score: float
    text_score: float
    domain_bonus: float
    rank: int = 0


class DemoRetriever:
    """Retrieves similar demonstrations from a library using embeddings.

    Supports multiple embedding backends:
    - "tfidf": Simple TF-IDF (no external dependencies, baseline)
    - "sentence_transformers": Local embedding model (recommended)
    - "openai": OpenAI text-embedding API

    The retriever uses FAISS for efficient similarity search when available,
    falling back to brute-force cosine similarity for small indices.

    Example:
        >>> retriever = DemoRetriever(embedding_method="sentence_transformers")
        >>> retriever.add_demo(episode, app_name="Chrome")
        >>> retriever.build_index()
        >>> results = retriever.retrieve("Search on GitHub", top_k=3)
        >>> print(results[0].demo.goal)
    """

    def __init__(
        self,
        embedding_method: str = "tfidf",
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        domain_bonus: float = 0.2,
        app_bonus: float = 0.15,
        use_faiss: bool = True,
    ) -> None:
        """Initialize the DemoRetriever.

        Args:
            embedding_method: Embedding backend ("tfidf", "sentence_transformers", "openai").
            embedding_model: Model name for sentence_transformers or OpenAI.
                - For sentence_transformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", etc.
                - For OpenAI: "text-embedding-3-small", "text-embedding-3-large", etc.
            cache_dir: Directory for caching embeddings. If None, uses ~/.cache/openadapt_ml/embeddings.
            domain_bonus: Bonus score for matching domain (default: 0.2).
            app_bonus: Bonus score for matching app name (default: 0.15).
            use_faiss: Whether to use FAISS for vector search (default: True).
        """
        self.embedding_method = embedding_method
        self.embedding_model = embedding_model
        self.domain_bonus = domain_bonus
        self.app_bonus = app_bonus
        self.use_faiss = use_faiss

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "openadapt_ml" / "embeddings"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._demos: List[DemoMetadata] = []
        self._embedder: Optional[Any] = None
        self._faiss_index: Optional[Any] = None
        self._is_indexed = False
        self._embeddings_matrix: Optional[Any] = None

    # =========================================================================
    # Demo Management
    # =========================================================================

    def add_demo(
        self,
        episode: Episode,
        demo_id: Optional[str] = None,
        app_name: Optional[str] = None,
        domain: Optional[str] = None,
        platform: Optional[str] = None,
        tags: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DemoMetadata:
        """Add a demonstration episode to the library.

        Args:
            episode: The Episode to add.
            demo_id: Unique ID (auto-generated from episode_id if not provided).
            app_name: Application name (auto-extracted if not provided).
            domain: Domain (auto-extracted from URLs if not provided).
            platform: Platform ("macos", "windows", "web"). Auto-detected if not provided.
            tags: User-provided tags for categorization.
            file_path: Path to the source file.
            metadata: Additional custom metadata.

        Returns:
            DemoMetadata object for the added demo.
        """
        # Auto-generate demo_id
        if demo_id is None:
            demo_id = episode.episode_id

        # Auto-extract app_name
        if app_name is None:
            app_name = self._extract_app_name(episode)

        # Auto-extract domain
        if domain is None:
            domain = self._extract_domain(episode)

        # Auto-detect platform
        if platform is None:
            platform = self._detect_platform(episode, app_name, domain)

        # Extract action types
        action_types = list(set(
            step.action.type.value if hasattr(step.action.type, 'value') else str(step.action.type)
            for step in episode.steps
            if step.action
        ))

        # Extract key elements
        key_elements = self._extract_key_elements(episode)

        demo_meta = DemoMetadata(
            demo_id=demo_id,
            episode=episode,
            goal=episode.instruction,
            app_name=app_name,
            domain=domain,
            platform=platform,
            action_types=action_types,
            key_elements=key_elements,
            step_count=len(episode.steps),
            tags=tags or [],
            file_path=file_path,
            metadata=metadata or {},
        )

        self._demos.append(demo_meta)
        self._is_indexed = False  # Need to rebuild index

        return demo_meta

    def add_demos(
        self,
        episodes: List[Episode],
        **kwargs: Any,
    ) -> List[DemoMetadata]:
        """Add multiple demonstration episodes.

        Args:
            episodes: List of Episodes to add.
            **kwargs: Additional arguments passed to add_demo.

        Returns:
            List of DemoMetadata objects.
        """
        return [self.add_demo(ep, **kwargs) for ep in episodes]

    def get_demo_count(self) -> int:
        """Get the number of demos in the library."""
        return len(self._demos)

    def get_all_demos(self) -> List[DemoMetadata]:
        """Get all demo metadata objects."""
        return list(self._demos)

    def get_apps(self) -> List[str]:
        """Get unique app names in the library."""
        apps = {d.app_name for d in self._demos if d.app_name}
        return sorted(apps)

    def get_domains(self) -> List[str]:
        """Get unique domains in the library."""
        domains = {d.domain for d in self._demos if d.domain}
        return sorted(domains)

    def clear(self) -> None:
        """Clear all demos and reset the index."""
        self._demos = []
        self._faiss_index = None
        self._embeddings_matrix = None
        self._is_indexed = False

    # =========================================================================
    # Indexing
    # =========================================================================

    def build_index(self, force: bool = False) -> None:
        """Build the search index from all added demos.

        This computes embeddings for all demos and builds the FAISS index.
        Must be called before retrieve().

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

        logger.info(f"Building index for {len(self._demos)} demos using {self.embedding_method}...")

        # Initialize embedder if needed
        if self._embedder is None:
            self._init_embedder()

        # Compute embeddings for all demos
        texts = self._get_indexable_texts()
        embeddings = self._compute_embeddings(texts)

        # Store embeddings in demo metadata
        for demo, emb in zip(self._demos, embeddings):
            demo.embedding = emb

        # Build FAISS index if available
        self._embeddings_matrix = embeddings
        if self.use_faiss:
            self._build_faiss_index(embeddings)

        self._is_indexed = True
        logger.info(f"Index built successfully with {len(self._demos)} demos")

    def save_index(self, path: Union[str, Path]) -> None:
        """Save the index and embeddings to disk.

        Args:
            path: Directory to save index files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save demo metadata (without embeddings - too large)
        metadata = []
        for demo in self._demos:
            meta = {
                "demo_id": demo.demo_id,
                "goal": demo.goal,
                "app_name": demo.app_name,
                "domain": demo.domain,
                "platform": demo.platform,
                "action_types": demo.action_types,
                "key_elements": demo.key_elements,
                "step_count": demo.step_count,
                "tags": demo.tags,
                "file_path": demo.file_path,
                "metadata": demo.metadata,
            }
            metadata.append(meta)

        # Prepare embedder state for TF-IDF (needed to recreate same embedding dimension)
        embedder_state = {}
        if self.embedding_method == "tfidf" and self._embedder is not None:
            embedder_state = {
                "vocab": self._embedder.vocab,
                "vocab_to_idx": self._embedder.vocab_to_idx,
                "idf": self._embedder.idf,
            }

        with open(path / "index.json", "w") as f:
            json.dump({
                "embedding_method": self.embedding_method,
                "embedding_model": self.embedding_model,
                "demos": metadata,
                "embedder_state": embedder_state,
            }, f, indent=2)

        # Save embeddings as numpy array
        try:
            import numpy as np
            if self._embeddings_matrix is not None:
                np.save(path / "embeddings.npy", self._embeddings_matrix)
        except ImportError:
            logger.warning("numpy not available, embeddings not saved")

        logger.info(f"Index saved to {path}")

    def load_index(self, path: Union[str, Path], episode_loader: Optional[Callable[[str], Episode]] = None) -> None:
        """Load index from disk.

        Args:
            path: Directory containing index files.
            episode_loader: Optional function to load episodes from file_path.
                If not provided, episodes will be None.
        """
        path = Path(path)

        with open(path / "index.json") as f:
            data = json.load(f)

        self.embedding_method = data.get("embedding_method", self.embedding_method)
        self.embedding_model = data.get("embedding_model", self.embedding_model)

        # Load embeddings
        embeddings = None
        try:
            import numpy as np
            embeddings_path = path / "embeddings.npy"
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)
        except ImportError:
            pass

        # Reconstruct demos
        self._demos = []
        for i, meta in enumerate(data.get("demos", [])):
            episode = None
            if episode_loader and meta.get("file_path"):
                try:
                    episode = episode_loader(meta["file_path"])
                except Exception as e:
                    logger.warning(f"Failed to load episode from {meta['file_path']}: {e}")

            # Create placeholder episode if not loaded
            if episode is None:
                from openadapt_ml.schema import Action, ActionType, Observation, Step
                episode = Episode(
                    episode_id=meta["demo_id"],
                    instruction=meta["goal"],
                    steps=[
                        Step(
                            step_index=0,
                            observation=Observation(),
                            action=Action(type=ActionType.DONE),
                        )
                    ],
                )

            demo = DemoMetadata(
                demo_id=meta["demo_id"],
                episode=episode,
                goal=meta["goal"],
                app_name=meta.get("app_name"),
                domain=meta.get("domain"),
                platform=meta.get("platform"),
                action_types=meta.get("action_types", []),
                key_elements=meta.get("key_elements", []),
                step_count=meta.get("step_count", 0),
                tags=meta.get("tags", []),
                file_path=meta.get("file_path"),
                metadata=meta.get("metadata", {}),
                embedding=embeddings[i] if embeddings is not None else None,
            )
            self._demos.append(demo)

        # Rebuild FAISS index if we have embeddings
        if embeddings is not None:
            self._embeddings_matrix = embeddings
            if self.use_faiss:
                self._build_faiss_index(embeddings)
            self._is_indexed = True

        # Restore embedder state for TF-IDF (needed for query embedding)
        embedder_state = data.get("embedder_state", {})
        if embedder_state and self.embedding_method == "tfidf":
            from openadapt_ml.retrieval.embeddings import TFIDFEmbedder
            self._embedder = TFIDFEmbedder()
            self._embedder.vocab = embedder_state.get("vocab", [])
            self._embedder.vocab_to_idx = embedder_state.get("vocab_to_idx", {})
            self._embedder.idf = embedder_state.get("idf", {})
            self._embedder._is_fitted = True

        logger.info(f"Index loaded from {path} with {len(self._demos)} demos")

    # =========================================================================
    # Retrieval
    # =========================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        app_context: Optional[str] = None,
        domain_context: Optional[str] = None,
        filter_platform: Optional[str] = None,
        filter_tags: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve top-K most similar demos for a query.

        Args:
            query: Task description to find demos for.
            top_k: Number of demos to retrieve.
            app_context: Optional app context for bonus scoring (e.g., "Chrome").
            domain_context: Optional domain context for bonus scoring (e.g., "github.com").
            filter_platform: Only return demos from this platform.
            filter_tags: Only return demos with all these tags.

        Returns:
            List of RetrievalResult objects, ordered by relevance (best first).

        Raises:
            ValueError: If index has not been built.
        """
        if not self._is_indexed:
            raise ValueError("Index not built. Call build_index() first.")

        if not self._demos:
            return []

        # Get query embedding
        query_embedding = self._compute_embeddings([query])[0]

        # Get candidates (optionally filtered)
        candidates = self._get_candidates(filter_platform, filter_tags)
        if not candidates:
            return []

        # Compute scores
        results = []
        for demo in candidates:
            text_score = self._compute_similarity(query_embedding, demo.embedding)
            bonus = self._compute_context_bonus(demo, app_context, domain_context)
            total_score = text_score + bonus

            results.append(RetrievalResult(
                demo=demo,
                score=total_score,
                text_score=text_score,
                domain_bonus=bonus,
            ))

        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)

        # Add ranks
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1

        return results[:top_k]

    def retrieve_episodes(
        self,
        query: str,
        top_k: int = 3,
        **kwargs: Any,
    ) -> List[Episode]:
        """Retrieve top-K episodes (convenience method).

        Args:
            query: Task description to find demos for.
            top_k: Number of demos to retrieve.
            **kwargs: Additional arguments passed to retrieve().

        Returns:
            List of Episode objects.
        """
        results = self.retrieve(query, top_k=top_k, **kwargs)
        return [r.demo.episode for r in results]

    # =========================================================================
    # Prompt Formatting
    # =========================================================================

    def format_for_prompt(
        self,
        results: List[RetrievalResult],
        max_steps_per_demo: int = 10,
        include_scores: bool = False,
        format_style: str = "concise",
    ) -> str:
        """Format retrieved demos for inclusion in a prompt.

        Args:
            results: Retrieval results from retrieve().
            max_steps_per_demo: Maximum steps to include per demo.
            include_scores: Whether to include relevance scores.
            format_style: Formatting style ("concise", "verbose", "minimal").

        Returns:
            Formatted string for prompt injection.
        """
        if not results:
            return ""

        from openadapt_ml.experiments.demo_prompt.format_demo import (
            format_episode_as_demo,
            format_episode_verbose,
        )

        lines = []

        if len(results) == 1:
            lines.append("Here is a relevant demonstration:")
        else:
            lines.append(f"Here are {len(results)} relevant demonstrations:")
        lines.append("")

        for i, result in enumerate(results, 1):
            if include_scores:
                lines.append(f"Demo {i} (relevance: {result.score:.2f}):")
            elif len(results) > 1:
                lines.append(f"Demo {i}:")

            if format_style == "verbose":
                demo_text = format_episode_verbose(
                    result.demo.episode,
                    max_steps=max_steps_per_demo,
                )
            elif format_style == "minimal":
                # Just goal and action sequence
                steps_text = " -> ".join(
                    self._format_action_minimal(step.action)
                    for step in result.demo.episode.steps[:max_steps_per_demo]
                    if step.action
                )
                demo_text = f"Task: {result.demo.goal}\nSteps: {steps_text}"
            else:  # concise (default)
                demo_text = format_episode_as_demo(
                    result.demo.episode,
                    max_steps=max_steps_per_demo,
                )

            lines.append(demo_text)
            lines.append("")

        return "\n".join(lines)

    def _format_action_minimal(self, action: Any) -> str:
        """Format action as minimal string."""
        from openadapt_ml.experiments.demo_prompt.format_demo import format_action
        return format_action(action)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _init_embedder(self) -> None:
        """Initialize the embedding backend."""
        if self.embedding_method == "tfidf":
            from openadapt_ml.retrieval.embeddings import TFIDFEmbedder
            self._embedder = TFIDFEmbedder()

        elif self.embedding_method == "sentence_transformers":
            from openadapt_ml.retrieval.embeddings import SentenceTransformerEmbedder
            self._embedder = SentenceTransformerEmbedder(
                model_name=self.embedding_model,
                cache_dir=self.cache_dir / "st_cache",
            )

        elif self.embedding_method == "openai":
            from openadapt_ml.retrieval.embeddings import OpenAIEmbedder
            self._embedder = OpenAIEmbedder(
                model_name=self.embedding_model,
                cache_dir=self.cache_dir / "openai_cache",
            )

        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")

    def _get_indexable_texts(self) -> List[str]:
        """Get text representations for indexing."""
        texts = []
        for demo in self._demos:
            # Combine goal with context
            parts = [demo.goal]
            if demo.app_name:
                parts.append(f"[APP:{demo.app_name}]")
            if demo.domain:
                parts.append(f"[DOMAIN:{demo.domain}]")
            texts.append(" ".join(parts))
        return texts

    def _compute_embeddings(self, texts: List[str]) -> Any:
        """Compute embeddings for texts."""
        import numpy as np

        if self._embedder is None:
            self._init_embedder()

        embeddings = self._embedder.embed_batch(texts)

        # Ensure numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def _build_faiss_index(self, embeddings: Any) -> None:
        """Build FAISS index from embeddings."""
        try:
            import faiss
            import numpy as np

            embeddings = np.asarray(embeddings, dtype=np.float32)
            dim = embeddings.shape[1]

            # Use IndexFlatIP for cosine similarity (assumes normalized embeddings)
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(embeddings)

            logger.debug(f"Built FAISS index with {len(embeddings)} vectors, dim={dim}")
        except ImportError:
            logger.debug("FAISS not available, using brute-force search")
            self._faiss_index = None

    def _compute_similarity(self, query_embedding: Any, doc_embedding: Any) -> float:
        """Compute similarity between query and document embeddings."""
        import numpy as np

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        doc_embedding = np.asarray(doc_embedding, dtype=np.float32)

        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        doc_norm = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-9)

        return float(np.dot(query_norm, doc_norm))

    def _compute_context_bonus(
        self,
        demo: DemoMetadata,
        app_context: Optional[str],
        domain_context: Optional[str],
    ) -> float:
        """Compute context bonus for app/domain matching."""
        bonus = 0.0

        if app_context and demo.app_name:
            if app_context.lower() in demo.app_name.lower():
                bonus += self.app_bonus

        if domain_context and demo.domain:
            if domain_context.lower() in demo.domain.lower():
                bonus += self.domain_bonus

        return bonus

    def _get_candidates(
        self,
        filter_platform: Optional[str],
        filter_tags: Optional[List[str]],
    ) -> List[DemoMetadata]:
        """Get candidate demos after filtering."""
        candidates = self._demos

        if filter_platform:
            candidates = [d for d in candidates if d.platform == filter_platform]

        if filter_tags:
            filter_tags_set = set(filter_tags)
            candidates = [
                d for d in candidates
                if filter_tags_set.issubset(set(d.tags))
            ]

        return candidates

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
        # Check for web indicators
        if domain:
            return "web"

        # Check for macOS app names
        macos_apps = {"System Settings", "Finder", "Safari", "Preview", "TextEdit"}
        if app_name and app_name in macos_apps:
            return "macos"

        # Check for Windows app names
        windows_apps = {"Settings", "File Explorer", "Microsoft Edge", "Notepad"}
        if app_name and app_name in windows_apps:
            return "windows"

        # Check episode metadata
        if episode.environment:
            env_lower = episode.environment.lower()
            if "macos" in env_lower or "darwin" in env_lower:
                return "macos"
            if "windows" in env_lower:
                return "windows"

        return None

    def _extract_key_elements(self, episode: Episode) -> List[str]:
        """Extract key UI elements from episode."""
        elements = []
        for step in episode.steps:
            if step.action and step.action.element:
                elem = step.action.element
                if elem.role and elem.name:
                    elements.append(f"{elem.role}:{elem.name}")
                elif elem.name:
                    elements.append(elem.name)
        return list(set(elements))

    def __len__(self) -> int:
        """Return number of demos in the library."""
        return len(self._demos)

    def __repr__(self) -> str:
        """String representation."""
        status = "indexed" if self._is_indexed else "not indexed"
        return f"DemoRetriever({len(self._demos)} demos, {self.embedding_method}, {status})"
