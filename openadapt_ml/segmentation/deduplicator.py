"""Workflow deduplication using embeddings and clustering.

This module identifies and merges similar workflows across
multiple recordings to create a canonical episode library (Stage 3).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from openadapt_ml.segmentation.schemas import (
    CanonicalEpisode,
    Episode,
    EpisodeExtractionResult,
    EpisodeLibrary,
)

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """OpenAI text embeddings."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            from openadapt_ml.config import settings

            api_key = self._api_key or settings.openai_api_key
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for texts."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
        )
        embeddings = [r.embedding for r in response.data]
        return np.array(embeddings, dtype=np.float32)


class LocalEmbedder:
    """Local HuggingFace embeddings (no API required)."""

    def __init__(
        self,
        model: str = "intfloat/e5-large-v2",
        device: str = "auto",
    ):
        self.model_name = model
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)

                if self.device == "auto":
                    if torch.cuda.is_available():
                        self._model = self._model.cuda()
                    elif (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        self._model = self._model.to("mps")
                elif self.device != "cpu":
                    self._model = self._model.to(self.device)

                self._model.eval()
            except ImportError:
                raise ImportError(
                    "LocalEmbedder requires transformers and torch. "
                    "Install with: pip install transformers torch"
                )

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for texts."""
        import torch

        self._load_model()

        # Add prefix for e5 models
        if "e5" in self.model_name.lower():
            texts = [f"query: {t}" for t in texts]

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if next(self._model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return embeddings.cpu().numpy().astype(np.float32)


def episode_to_text(episode: Episode) -> str:
    """Convert an episode to text for embedding.

    Combines multiple fields for rich semantic representation.
    """
    parts = [
        f"Workflow: {episode.name}",
        f"Description: {episode.description}",
        f"Application: {episode.application}",
        f"Steps: {', '.join(episode.step_summaries)}",
    ]

    if episode.prerequisites:
        parts.append(f"Prerequisites: {', '.join(episode.prerequisites)}")

    if episode.outcomes:
        parts.append(f"Outcomes: {', '.join(episode.outcomes)}")

    return "\n".join(parts)


class WorkflowDeduplicator:
    """Deduplicates workflow episodes using embedding similarity.

    This class implements Stage 3 of the segmentation pipeline, identifying
    similar workflows across recordings and merging them into canonical
    definitions.

    Example:
        >>> dedup = WorkflowDeduplicator(threshold=0.85)
        >>> library = dedup.deduplicate(extraction_results)
        >>> print(f"Found {library.unique_episode_count} unique workflows")
        >>> print(f"Deduplication ratio: {library.deduplication_ratio:.1%}")
        Found 15 unique workflows
        Deduplication ratio: 34.2%

    Attributes:
        threshold: Similarity threshold for clustering
        embedding_model: Model used for text embeddings
        merge_strategy: How to merge similar episodes
    """

    def __init__(
        self,
        threshold: float = 0.85,
        embedding_model: str = "text-embedding-3-large",
        embedding_dim: int = 3072,
        merge_strategy: str = "centroid",
        min_cluster_size: int = 1,
        use_local_embeddings: bool = False,
    ) -> None:
        """Initialize the deduplicator.

        Args:
            threshold: Cosine similarity threshold for clustering.
                Higher = stricter matching, fewer merges.
                Recommended: 0.80-0.90
            embedding_model: Text embedding model.
            embedding_dim: Embedding dimension (model-specific).
            merge_strategy: How to create canonical definition:
                - "centroid": Use episode closest to cluster centroid
                - "longest": Use longest description
                - "first": Use first encountered
            min_cluster_size: Minimum episodes to form a cluster.
            use_local_embeddings: Use local HuggingFace model instead of API.
        """
        self.threshold = threshold
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.merge_strategy = merge_strategy
        self.min_cluster_size = min_cluster_size
        self.use_local_embeddings = use_local_embeddings

        if use_local_embeddings:
            self._embedder = LocalEmbedder(model="intfloat/e5-large-v2")
        else:
            self._embedder = OpenAIEmbedder(model=embedding_model)

    def deduplicate(
        self,
        extraction_results: list[EpisodeExtractionResult],
        existing_library: Optional[EpisodeLibrary] = None,
    ) -> EpisodeLibrary:
        """Deduplicate episodes across multiple extraction results.

        Args:
            extraction_results: List of extraction results from Stage 2.
            existing_library: Optional existing library to merge with.

        Returns:
            EpisodeLibrary with deduplicated canonical episodes.
        """
        # Collect all episodes
        all_episodes = []
        for result in extraction_results:
            all_episodes.extend(result.episodes)

        # Add episodes from existing library
        existing_episodes = []
        if existing_library:
            for canonical in existing_library.episodes:
                # Create synthetic Episode from CanonicalEpisode
                for i, (rec_id, seg_id) in enumerate(
                    zip(canonical.source_recordings, canonical.source_episode_ids)
                ):
                    synthetic = Episode(
                        episode_id=seg_id,
                        name=canonical.variant_names[i]
                        if i < len(canonical.variant_names)
                        else canonical.canonical_name,
                        start_time=0,
                        end_time=0,
                        start_time_formatted="00:00.0",
                        end_time_formatted="00:00.0",
                        description=canonical.variant_descriptions[i]
                        if i < len(canonical.variant_descriptions)
                        else canonical.canonical_description,
                        step_summaries=canonical.canonical_steps,
                        application="Unknown",
                        boundary_confidence=1.0,
                        coherence_score=1.0,
                        recording_id=rec_id,
                    )
                    existing_episodes.append(synthetic)
            all_episodes.extend(existing_episodes)

        if not all_episodes:
            return EpisodeLibrary(
                episodes=[],
                total_recordings_processed=len(extraction_results),
                total_episodes_extracted=0,
                unique_episode_count=0,
                deduplication_ratio=0.0,
                similarity_threshold=self.threshold,
                embedding_model=self.embedding_model,
            )

        # Generate embeddings
        embeddings = self.embed_episodes(all_episodes)

        # Cluster similar episodes
        clusters = self.cluster_episodes(embeddings, all_episodes)

        # Merge clusters into canonical episodes
        canonical_episodes = []
        for cluster_id, indices in enumerate(clusters):
            cluster_episodes = [all_episodes[i] for i in indices]
            cluster_embeddings = embeddings[indices]

            canonical = self.merge_cluster(
                cluster_episodes, cluster_embeddings, cluster_id
            )
            canonical_episodes.append(canonical)

        # Calculate statistics
        total_extracted = len(all_episodes)
        unique_count = len(canonical_episodes)
        dedup_ratio = 1 - (unique_count / total_extracted) if total_extracted > 0 else 0

        return EpisodeLibrary(
            episodes=canonical_episodes,
            total_recordings_processed=len(extraction_results),
            total_episodes_extracted=total_extracted,
            unique_episode_count=unique_count,
            deduplication_ratio=dedup_ratio,
            similarity_threshold=self.threshold,
            embedding_model=self.embedding_model,
        )

    def embed_episode(self, episode: Episode) -> NDArray[np.float32]:
        """Generate embedding for a single workflow episode."""
        text = episode_to_text(episode)
        embeddings = self._embedder.embed([text])
        return embeddings[0]

    def embed_episodes(
        self,
        episodes: list[Episode],
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Generate embeddings for multiple episodes.

        Args:
            episodes: List of episodes to embed.
            show_progress: Show progress bar.

        Returns:
            Embedding matrix of shape (n_episodes, embedding_dim).
        """
        texts = [episode_to_text(ep) for ep in episodes]

        # Process in batches to avoid API limits
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embedder.embed(batch)
            all_embeddings.append(batch_embeddings)

            if show_progress:
                logger.info(
                    f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} episodes"
                )

        return np.vstack(all_embeddings)

    def compute_similarity_matrix(
        self,
        embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Embedding matrix of shape (n, embedding_dim).

        Returns:
            Similarity matrix of shape (n, n) with values in [-1, 1].
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-9)

        # Compute cosine similarity
        similarity = normalized @ normalized.T
        return similarity

    def cluster_episodes(
        self,
        embeddings: NDArray[np.float32],
        episodes: list[Episode],
    ) -> list[list[int]]:
        """Cluster similar episodes using agglomerative clustering.

        Args:
            embeddings: Embedding matrix.
            episodes: Original episodes (for metadata).

        Returns:
            List of clusters, each containing episode indices.
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            logger.warning("sklearn not available, using simple clustering")
            return self._simple_cluster(embeddings)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-9)

        # Compute cosine distances
        distances = 1 - (normalized @ normalized.T)

        # Cluster
        distance_threshold = 1 - self.threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distances)

        # Group indices by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        return list(clusters.values())

    def _simple_cluster(self, embeddings: NDArray[np.float32]) -> list[list[int]]:
        """Simple greedy clustering when sklearn not available."""
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-9)

        n = len(embeddings)
        assigned = [False] * n
        clusters = []

        for i in range(n):
            if assigned[i]:
                continue

            # Start new cluster
            cluster = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue

                # Check similarity
                sim = np.dot(normalized[i], normalized[j])
                if sim >= self.threshold:
                    cluster.append(j)
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def merge_cluster(
        self,
        episodes: list[Episode],
        embeddings: NDArray[np.float32],
        cluster_id: int,
    ) -> CanonicalEpisode:
        """Merge a cluster of similar episodes into a canonical episode.

        Args:
            episodes: Episodes in this cluster.
            embeddings: Embeddings for these episodes.
            cluster_id: ID for this cluster.

        Returns:
            CanonicalEpisode representing the merged cluster.
        """
        if self.merge_strategy == "centroid":
            # Find episode closest to cluster centroid
            centroid = embeddings.mean(axis=0)
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            canonical_idx = int(np.argmin(distances))

        elif self.merge_strategy == "longest":
            # Use episode with longest description
            lengths = [len(ep.description) for ep in episodes]
            canonical_idx = int(np.argmax(lengths))

        elif self.merge_strategy == "first":
            # Use first encountered
            canonical_idx = 0

        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

        canonical_episode = episodes[canonical_idx]

        # Compute internal similarity
        if len(embeddings) > 1:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / np.maximum(norms, 1e-9)
            sim_matrix = normalized @ normalized.T
            # Average of upper triangle (excluding diagonal)
            internal_sim = np.mean(sim_matrix[np.triu_indices(len(sim_matrix), k=1)])
        else:
            internal_sim = 1.0

        return CanonicalEpisode(
            canonical_id=uuid4(),
            canonical_name=canonical_episode.name,
            canonical_description=canonical_episode.description,
            canonical_steps=canonical_episode.step_summaries,
            variant_names=[ep.name for ep in episodes if ep != canonical_episode],
            variant_descriptions=[
                ep.description for ep in episodes if ep != canonical_episode
            ],
            source_recordings=list(set(ep.recording_id for ep in episodes)),
            source_episode_ids=[ep.episode_id for ep in episodes],
            occurrence_count=len(episodes),
            embedding=embeddings[canonical_idx].tolist(),
            cluster_id=cluster_id,
            cluster_centroid_distance=float(
                np.linalg.norm(embeddings[canonical_idx] - embeddings.mean(axis=0))
            ),
            internal_similarity=float(internal_sim),
        )

    def find_similar(
        self,
        episode: Episode,
        library: EpisodeLibrary,
        top_k: int = 5,
    ) -> list[tuple[CanonicalEpisode, float]]:
        """Find similar workflows in an existing library.

        Args:
            episode: Episode to find matches for.
            library: Existing workflow library.
            top_k: Number of results to return.

        Returns:
            List of (canonical_episode, similarity_score) tuples.
        """
        if not library.episodes:
            return []

        # Get embedding for query episode
        query_embedding = self.embed_episode(episode)
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Get embeddings for library
        results = []
        for canonical in library.episodes:
            if canonical.embedding:
                lib_embedding = np.array(canonical.embedding, dtype=np.float32)
                lib_norm = lib_embedding / np.linalg.norm(lib_embedding)
                similarity = float(np.dot(query_norm, lib_norm))
                results.append((canonical, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def add_to_library(
        self,
        episode: Episode,
        library: EpisodeLibrary,
    ) -> tuple[EpisodeLibrary, Optional[CanonicalEpisode]]:
        """Add an episode to an existing library.

        Either merges with existing workflow or creates new one.

        Args:
            episode: New episode to add.
            library: Existing library.

        Returns:
            Tuple of (updated_library, matched_canonical or None if new).
        """
        similar = self.find_similar(episode, library, top_k=1)

        if similar and similar[0][1] >= self.threshold:
            # Merge with existing
            matched_canonical = similar[0][0]

            # Update the canonical episode
            for can in library.episodes:
                if can.canonical_id == matched_canonical.canonical_id:
                    can.variant_names.append(episode.name)
                    can.variant_descriptions.append(episode.description)
                    can.source_recordings.append(episode.recording_id)
                    can.source_episode_ids.append(episode.episode_id)
                    can.occurrence_count += 1
                    break

            library.total_episodes_extracted += 1
            library.deduplication_ratio = 1 - (
                library.unique_episode_count / library.total_episodes_extracted
            )

            return library, matched_canonical

        else:
            # Create new canonical episode
            embedding = self.embed_episode(episode)
            new_canonical = CanonicalEpisode(
                canonical_id=uuid4(),
                canonical_name=episode.name,
                canonical_description=episode.description,
                canonical_steps=episode.step_summaries,
                variant_names=[],
                variant_descriptions=[],
                source_recordings=[episode.recording_id],
                source_episode_ids=[episode.episode_id],
                occurrence_count=1,
                embedding=embedding.tolist(),
                cluster_id=len(library.episodes),
                cluster_centroid_distance=0.0,
                internal_similarity=1.0,
            )

            library.episodes.append(new_canonical)
            library.total_episodes_extracted += 1
            library.unique_episode_count += 1
            library.deduplication_ratio = 1 - (
                library.unique_episode_count / library.total_episodes_extracted
            )

            return library, None

    def save_embeddings(
        self,
        path: Union[str, Path],
        embeddings: NDArray[np.float32],
        episodes: list[Episode],
    ) -> None:
        """Save embeddings and metadata for later reuse.

        Args:
            path: Output file path (will create .npy and .json).
            embeddings: Embedding matrix.
            episodes: Original episodes for metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(str(path.with_suffix(".npy")), embeddings)

        # Save metadata
        metadata = [
            {
                "episode_id": str(ep.episode_id),
                "name": ep.name,
                "recording_id": ep.recording_id,
            }
            for ep in episodes
        ]
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    def load_embeddings(
        self,
        path: Union[str, Path],
    ) -> tuple[NDArray[np.float32], list[dict]]:
        """Load previously saved embeddings.

        Args:
            path: Path to saved embeddings.

        Returns:
            Tuple of (embeddings, episode_metadata).
        """
        path = Path(path)
        embeddings = np.load(str(path.with_suffix(".npy")))
        metadata = json.loads(path.with_suffix(".json").read_text())
        return embeddings, metadata
