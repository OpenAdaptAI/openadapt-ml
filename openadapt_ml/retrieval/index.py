"""Demo index for storing and retrieving demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openadapt_ml.retrieval.embeddings import TextEmbedder
from openadapt_ml.schema import Episode


@dataclass
class DemoMetadata:
    """Metadata for a single demonstration.

    Stores both the episode and computed features for retrieval.
    """

    episode: Episode
    app_name: Optional[str] = None
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed at index time
    text_embedding: Dict[str, float] = field(default_factory=dict)


class DemoIndex:
    """Index for demonstrations.

    Stores episodes with their metadata and embeddings for efficient retrieval.
    """

    def __init__(self) -> None:
        """Initialize the demo index."""
        self.demos: List[DemoMetadata] = []
        self.embedder = TextEmbedder()
        self._is_fitted = False

    def _extract_app_name(self, episode: Episode) -> Optional[str]:
        """Extract app name from episode steps.

        Args:
            episode: Episode to extract from.

        Returns:
            App name if found, None otherwise.
        """
        # Look through observations to find app_name
        for step in episode.steps:
            if step.observation and step.observation.app_name:
                return step.observation.app_name
        return None

    def _extract_domain(self, episode: Episode) -> Optional[str]:
        """Extract domain from episode metadata or URL.

        Args:
            episode: Episode to extract from.

        Returns:
            Domain if found, None otherwise.
        """
        # Try to extract from URL in observations
        for step in episode.steps:
            if step.observation and step.observation.url:
                url = step.observation.url
                # Simple domain extraction (e.g., "github.com" from "https://github.com/...")
                if "://" in url:
                    domain = url.split("://")[1].split("/")[0]
                    # Remove www. prefix
                    if domain.startswith("www."):
                        domain = domain[4:]
                    return domain

        return None

    def add(
        self,
        episode: Episode,
        app_name: Optional[str] = None,
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an episode to the index.

        Args:
            episode: Episode to add.
            app_name: Optional app name (auto-extracted if not provided).
            domain: Optional domain (auto-extracted if not provided).
            metadata: Additional metadata for the episode.
        """
        # Auto-extract app_name and domain if not provided
        if app_name is None:
            app_name = self._extract_app_name(episode)
        if domain is None:
            domain = self._extract_domain(episode)

        demo_meta = DemoMetadata(
            episode=episode,
            app_name=app_name,
            domain=domain,
            metadata=metadata or {},
        )

        self.demos.append(demo_meta)
        # Mark as not fitted since we added new data
        self._is_fitted = False

    def add_many(self, episodes: List[Episode]) -> None:
        """Add multiple episodes to the index.

        Args:
            episodes: List of episodes to add.
        """
        for episode in episodes:
            self.add(episode)

    def build(self) -> None:
        """Build the index by computing embeddings.

        This must be called after adding all demos and before retrieval.
        """
        if not self.demos:
            return

        # Fit embedder on all instruction texts
        instruction_texts = [demo.episode.instruction for demo in self.demos]
        self.embedder.fit(instruction_texts)

        # Compute embeddings for each demo
        for demo in self.demos:
            demo.text_embedding = self.embedder.embed(demo.episode.instruction)

        self._is_fitted = True

    def is_empty(self) -> bool:
        """Check if the index is empty.

        Returns:
            True if no demos have been added.
        """
        return len(self.demos) == 0

    def is_fitted(self) -> bool:
        """Check if the index has been built.

        Returns:
            True if build() has been called.
        """
        return self._is_fitted

    def get_all_demos(self) -> List[DemoMetadata]:
        """Get all demos in the index.

        Returns:
            List of all DemoMetadata objects.
        """
        return self.demos

    def get_apps(self) -> List[str]:
        """Get list of unique app names in the index.

        Returns:
            List of app names (excluding None).
        """
        apps = {demo.app_name for demo in self.demos if demo.app_name is not None}
        return sorted(apps)

    def get_domains(self) -> List[str]:
        """Get list of unique domains in the index.

        Returns:
            List of domains (excluding None).
        """
        domains = {demo.domain for demo in self.demos if demo.domain is not None}
        return sorted(domains)

    def __len__(self) -> int:
        """Return number of demos in the index.

        Returns:
            Number of demos.
        """
        return len(self.demos)

    def __repr__(self) -> str:
        """String representation of the index.

        Returns:
            String representation.
        """
        status = "fitted" if self._is_fitted else "not fitted"
        return f"DemoIndex({len(self.demos)} demos, {status})"
