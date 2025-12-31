"""Demo retriever for finding similar demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from openadapt_ml.retrieval.index import DemoIndex, DemoMetadata
from openadapt_ml.schemas.sessions import Episode


@dataclass
class RetrievalResult:
    """A single retrieval result with score.

    Attributes:
        demo: The demo metadata.
        score: Retrieval score (higher is better).
        text_score: Text similarity component.
        domain_bonus: Domain match bonus applied.
    """

    demo: DemoMetadata
    score: float
    text_score: float
    domain_bonus: float


class DemoRetriever:
    """Retrieves top-K similar demonstrations from an index.

    Uses text similarity (TF-IDF cosine) with optional domain match bonus.
    """

    def __init__(
        self,
        index: DemoIndex,
        domain_bonus: float = 0.2,
    ) -> None:
        """Initialize the retriever.

        Args:
            index: DemoIndex to retrieve from.
            domain_bonus: Bonus score for domain match (default: 0.2).

        Raises:
            ValueError: If index is empty or not fitted.
        """
        if index.is_empty():
            raise ValueError("Cannot create retriever from empty index")
        if not index.is_fitted():
            raise ValueError("Index must be built before retrieval (call index.build())")

        self.index = index
        self.domain_bonus = domain_bonus

    def _compute_score(
        self,
        task: str,
        demo: DemoMetadata,
        app_context: Optional[str] = None,
    ) -> RetrievalResult:
        """Compute retrieval score for a demo.

        Args:
            task: Task description to match against.
            demo: Demo metadata to score.
            app_context: Optional app/domain context for bonus.

        Returns:
            RetrievalResult with computed scores.
        """
        # Text similarity using TF-IDF
        query_embedding = self.index.embedder.embed(task)
        text_score = self.index.embedder.cosine_similarity(
            query_embedding,
            demo.text_embedding,
        )

        # Domain match bonus
        bonus = 0.0
        if app_context is not None:
            # Check if app_context matches app_name or domain
            app_match = demo.app_name and app_context.lower() in demo.app_name.lower()
            domain_match = demo.domain and app_context.lower() in demo.domain.lower()

            if app_match or domain_match:
                bonus = self.domain_bonus

        # Final score is text similarity + bonus
        total_score = text_score + bonus

        return RetrievalResult(
            demo=demo,
            score=total_score,
            text_score=text_score,
            domain_bonus=bonus,
        )

    def retrieve(
        self,
        task: str,
        app_context: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Episode]:
        """Retrieve top-K most similar demos.

        Args:
            task: Task description to find demos for.
            app_context: Optional app/domain context (e.g., "Chrome", "github.com").
            top_k: Number of demos to retrieve.

        Returns:
            List of Episode objects, ordered by relevance (most similar first).
        """
        if self.index.is_empty():
            return []

        # Score all demos
        results = [
            self._compute_score(task, demo, app_context)
            for demo in self.index.get_all_demos()
        ]

        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)

        # Return top-K episodes
        top_results = results[:top_k]
        return [r.demo.episode for r in top_results]

    def retrieve_with_scores(
        self,
        task: str,
        app_context: Optional[str] = None,
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        """Retrieve top-K demos with their scores.

        Args:
            task: Task description to find demos for.
            app_context: Optional app/domain context.
            top_k: Number of demos to retrieve.

        Returns:
            List of RetrievalResult objects with scores.
        """
        if self.index.is_empty():
            return []

        # Score all demos
        results = [
            self._compute_score(task, demo, app_context)
            for demo in self.index.get_all_demos()
        ]

        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]
