"""Unit tests for demo retrieval module."""

from __future__ import annotations

import pytest

from openadapt_ml.retrieval import DemoIndex, DemoRetriever
from openadapt_ml.retrieval.embeddings import TextEmbedder
from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step


@pytest.fixture
def sample_episodes() -> list[Episode]:
    """Create sample episodes for testing."""

    def create_episode(
        episode_id: str,
        instruction: str,
        app_name: str | None = None,
        url: str | None = None,
    ) -> Episode:
        obs = Observation(app_name=app_name, url=url)
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        return Episode(episode_id=episode_id, instruction=instruction, steps=[step])

    return [
        create_episode("ep1", "Turn off Night Shift", app_name="System Settings"),
        create_episode("ep2", "Search GitHub", app_name="Chrome", url="https://github.com"),
        create_episode("ep3", "Open calculator", app_name="Calculator"),
    ]


class TestTextEmbedder:
    """Test TextEmbedder class."""

    def test_tokenize(self) -> None:
        """Test tokenization."""
        embedder = TextEmbedder()
        tokens = embedder._tokenize("Hello World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_fit_and_embed(self) -> None:
        """Test fitting and embedding."""
        embedder = TextEmbedder()
        docs = ["hello world", "world of python", "python programming"]
        embedder.fit(docs)

        # Embed a document
        vec = embedder.embed("hello python")
        assert isinstance(vec, dict)
        assert "hello" in vec
        assert "python" in vec

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity."""
        embedder = TextEmbedder()
        docs = ["machine learning", "deep learning", "cooking recipes"]
        embedder.fit(docs)

        vec1 = embedder.embed("machine learning")
        vec2 = embedder.embed("deep learning")
        vec3 = embedder.embed("cooking recipes")

        # Similar documents should have higher similarity
        sim_12 = embedder.cosine_similarity(vec1, vec2)
        sim_13 = embedder.cosine_similarity(vec1, vec3)

        assert sim_12 > sim_13
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1

    def test_empty_corpus(self) -> None:
        """Test with empty corpus."""
        embedder = TextEmbedder()
        embedder.fit([])
        vec = embedder.embed("test")
        assert vec == {}


class TestDemoIndex:
    """Test DemoIndex class."""

    def test_add_and_len(self, sample_episodes: list[Episode]) -> None:
        """Test adding episodes."""
        index = DemoIndex()
        assert len(index) == 0

        index.add(sample_episodes[0])
        assert len(index) == 1

        index.add_many(sample_episodes[1:])
        assert len(index) == 3

    def test_build(self, sample_episodes: list[Episode]) -> None:
        """Test building the index."""
        index = DemoIndex()
        index.add_many(sample_episodes)

        assert not index.is_fitted()
        index.build()
        assert index.is_fitted()

        # Check embeddings were computed
        for demo in index.get_all_demos():
            assert len(demo.text_embedding) > 0

    def test_extract_app_name(self, sample_episodes: list[Episode]) -> None:
        """Test app name extraction."""
        index = DemoIndex()
        index.add_many(sample_episodes)

        apps = index.get_apps()
        assert "System Settings" in apps
        assert "Chrome" in apps
        assert "Calculator" in apps

    def test_extract_domain(self) -> None:
        """Test domain extraction from URL."""
        index = DemoIndex()
        obs = Observation(url="https://github.com/user/repo")
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction="test", steps=[step])

        index.add(episode)
        domains = index.get_domains()
        assert "github.com" in domains

    def test_empty_index(self) -> None:
        """Test empty index."""
        index = DemoIndex()
        assert index.is_empty()
        assert len(index) == 0
        assert index.get_apps() == []
        assert index.get_domains() == []


class TestDemoRetriever:
    """Test DemoRetriever class."""

    @pytest.fixture
    def built_index(self, sample_episodes: list[Episode]) -> DemoIndex:
        """Create a built index."""
        index = DemoIndex()
        index.add_many(sample_episodes)
        index.build()
        return index

    def test_create_retriever(self, built_index: DemoIndex) -> None:
        """Test creating a retriever."""
        retriever = DemoRetriever(built_index)
        assert retriever.index == built_index

    def test_create_from_empty_index(self) -> None:
        """Test that empty index raises error."""
        index = DemoIndex()
        with pytest.raises(ValueError, match="empty index"):
            DemoRetriever(index)

    def test_create_from_unfitted_index(self, sample_episodes: list[Episode]) -> None:
        """Test that unfitted index raises error."""
        index = DemoIndex()
        index.add_many(sample_episodes)
        with pytest.raises(ValueError, match="must be built"):
            DemoRetriever(index)

    def test_retrieve(self, built_index: DemoIndex) -> None:
        """Test basic retrieval."""
        retriever = DemoRetriever(built_index)
        results = retriever.retrieve("Disable Night Shift", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(ep, Episode) for ep in results)

    def test_retrieve_with_scores(self, built_index: DemoIndex) -> None:
        """Test retrieval with scores."""
        retriever = DemoRetriever(built_index)
        results = retriever.retrieve_with_scores("Turn off Night Shift", top_k=3)

        assert len(results) <= 3
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_domain_bonus(self, built_index: DemoIndex) -> None:
        """Test that domain bonus is applied."""
        retriever = DemoRetriever(built_index, domain_bonus=0.5)

        # Query with matching app context
        results = retriever.retrieve_with_scores(
            "Search code",
            app_context="github.com",
            top_k=3,
        )

        # GitHub demo should get bonus
        github_result = next(r for r in results if "GitHub" in r.demo.episode.instruction)
        assert github_result.domain_bonus == 0.5

    def test_top_k_limit(self, built_index: DemoIndex) -> None:
        """Test top_k limit."""
        retriever = DemoRetriever(built_index)

        results = retriever.retrieve("test", top_k=1)
        assert len(results) == 1

        results = retriever.retrieve("test", top_k=10)
        assert len(results) <= 3  # Only 3 demos in index


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_workflow(self, sample_episodes: list[Episode]) -> None:
        """Test complete workflow."""
        # Build index
        index = DemoIndex()
        index.add_many(sample_episodes)
        index.build()

        # Create retriever
        retriever = DemoRetriever(index, domain_bonus=0.2)

        # Retrieve
        task = "Turn off dark mode"
        results = retriever.retrieve(task, top_k=2)

        assert len(results) > 0
        assert all(isinstance(ep, Episode) for ep in results)

        # Most similar should be Night Shift demo
        assert "Night Shift" in results[0].instruction
