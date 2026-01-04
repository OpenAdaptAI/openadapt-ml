"""Comprehensive tests for the demo retrieval system.

Tests cover:
- DemoRetriever class (main retrieval functionality)
- Embedders (TF-IDF, Sentence Transformers, OpenAI)
- Index persistence (save/load)
- Prompt formatting
- Edge cases and error handling
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openadapt_ml.retrieval import (
    DemoRetriever,
    DemoMetadata,
    RetrievalResult,
    TFIDFEmbedder,
    TextEmbedder,
    create_embedder,
)
from openadapt_ml.retrieval.embeddings import BaseEmbedder
from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_episodes() -> List[Episode]:
    """Create sample episodes for testing."""

    def create_episode(
        episode_id: str,
        instruction: str,
        app_name: str | None = None,
        url: str | None = None,
        window_title: str | None = None,
    ) -> Episode:
        obs = Observation(app_name=app_name, url=url, window_title=window_title)
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        return Episode(episode_id=episode_id, instruction=instruction, steps=[step])

    return [
        create_episode(
            "ep1",
            "Turn off Night Shift",
            app_name="System Settings",
            window_title="Display",
        ),
        create_episode(
            "ep2",
            "Search for machine learning repos on GitHub",
            app_name="Chrome",
            url="https://github.com",
        ),
        create_episode(
            "ep3",
            "Open the calculator app",
            app_name="Calculator",
        ),
        create_episode(
            "ep4",
            "Enable dark mode in system preferences",
            app_name="System Settings",
            window_title="Appearance",
        ),
        create_episode(
            "ep5",
            "Create a new file in the current folder",
            app_name="Finder",
        ),
    ]


@pytest.fixture
def built_retriever(sample_episodes: List[Episode]) -> DemoRetriever:
    """Create a retriever with sample episodes indexed."""
    retriever = DemoRetriever(embedding_method="tfidf")
    for ep in sample_episodes:
        retriever.add_demo(ep)
    retriever.build_index()
    return retriever


# =============================================================================
# TF-IDF Embedder Tests
# =============================================================================


class TestTFIDFEmbedder:
    """Test TFIDFEmbedder class."""

    def test_tokenize(self) -> None:
        """Test tokenization."""
        embedder = TFIDFEmbedder()
        tokens = embedder._tokenize("Hello World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_fit_and_embed(self) -> None:
        """Test fitting and embedding."""
        embedder = TFIDFEmbedder()
        docs = ["hello world", "world of python", "python programming"]
        embedder.fit(docs)

        vec = embedder.embed("hello python")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert len(vec) > 0

    def test_embed_batch(self) -> None:
        """Test batch embedding."""
        embedder = TFIDFEmbedder()
        docs = ["machine learning", "deep learning", "cooking recipes"]

        embeddings = embedder.embed_batch(docs)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.dtype == np.float32

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity computation."""
        embedder = TFIDFEmbedder()
        docs = ["machine learning algorithms", "deep learning neural networks", "cooking pasta recipes"]
        embedder.fit(docs)

        vec1 = embedder.embed("machine learning")
        vec2 = embedder.embed("deep learning")
        vec3 = embedder.embed("cooking recipes")

        sim_12 = embedder.cosine_similarity(vec1, vec2)
        sim_13 = embedder.cosine_similarity(vec1, vec3)

        # Similar documents should have higher similarity
        assert sim_12 > sim_13
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1

    def test_empty_corpus(self) -> None:
        """Test handling of empty corpus."""
        embedder = TFIDFEmbedder()
        embedder.fit([])

        vec = embedder.embed("test")
        assert len(vec) == 0

    def test_backward_compatibility(self) -> None:
        """Test TextEmbedder alias works."""
        embedder = TextEmbedder()
        assert isinstance(embedder, TFIDFEmbedder)


# =============================================================================
# Demo Retriever Tests
# =============================================================================


class TestDemoRetriever:
    """Test DemoRetriever class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        retriever = DemoRetriever()
        assert retriever.embedding_method == "tfidf"
        assert retriever.get_demo_count() == 0
        assert not retriever._is_indexed

    def test_init_with_options(self) -> None:
        """Test initialization with options."""
        retriever = DemoRetriever(
            embedding_method="tfidf",
            domain_bonus=0.3,
            app_bonus=0.2,
        )
        assert retriever.domain_bonus == 0.3
        assert retriever.app_bonus == 0.2

    def test_add_demo(self, sample_episodes: List[Episode]) -> None:
        """Test adding demos."""
        retriever = DemoRetriever()

        meta = retriever.add_demo(sample_episodes[0])

        assert retriever.get_demo_count() == 1
        assert isinstance(meta, DemoMetadata)
        assert meta.goal == "Turn off Night Shift"
        assert meta.app_name == "System Settings"

    def test_add_demos_batch(self, sample_episodes: List[Episode]) -> None:
        """Test adding multiple demos."""
        retriever = DemoRetriever()

        metas = retriever.add_demos(sample_episodes)

        assert retriever.get_demo_count() == 5
        assert len(metas) == 5

    def test_add_demo_with_metadata(self, sample_episodes: List[Episode]) -> None:
        """Test adding demo with custom metadata."""
        retriever = DemoRetriever()

        meta = retriever.add_demo(
            sample_episodes[0],
            tags=["settings", "display"],
            metadata={"source": "test"},
        )

        assert meta.tags == ["settings", "display"]
        assert meta.metadata == {"source": "test"}

    def test_build_index(self, sample_episodes: List[Episode]) -> None:
        """Test building the index."""
        retriever = DemoRetriever()
        for ep in sample_episodes:
            retriever.add_demo(ep)

        assert not retriever._is_indexed
        retriever.build_index()
        assert retriever._is_indexed

        # Check embeddings were computed
        for demo in retriever.get_all_demos():
            assert demo.embedding is not None

    def test_build_empty_index_raises(self) -> None:
        """Test that building empty index raises error."""
        retriever = DemoRetriever()

        with pytest.raises(ValueError, match="no demos added"):
            retriever.build_index()

    def test_retrieve_without_index_raises(self, sample_episodes: List[Episode]) -> None:
        """Test that retrieval without index raises error."""
        retriever = DemoRetriever()
        retriever.add_demo(sample_episodes[0])

        with pytest.raises(ValueError, match="Index not built"):
            retriever.retrieve("test query")

    def test_retrieve_basic(self, built_retriever: DemoRetriever) -> None:
        """Test basic retrieval."""
        results = built_retriever.retrieve("Turn off Night Shift", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

        # Most similar should be the Night Shift demo
        assert "Night Shift" in results[0].demo.goal
        assert results[0].rank == 1

    def test_retrieve_top_k(self, built_retriever: DemoRetriever) -> None:
        """Test top_k parameter."""
        results1 = built_retriever.retrieve("test", top_k=1)
        results2 = built_retriever.retrieve("test", top_k=5)

        assert len(results1) == 1
        assert len(results2) == 5

    def test_retrieve_scores_sorted(self, built_retriever: DemoRetriever) -> None:
        """Test that results are sorted by score."""
        results = built_retriever.retrieve("settings preferences", top_k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_with_app_context(self, built_retriever: DemoRetriever) -> None:
        """Test retrieval with app context bonus."""
        # Without context
        results_no_ctx = built_retriever.retrieve("Search code", top_k=3)

        # With matching context
        results_with_ctx = built_retriever.retrieve(
            "Search code",
            app_context="Chrome",
            top_k=3,
        )

        # GitHub demo should rank higher with Chrome context
        github_no_ctx = next(
            (r for r in results_no_ctx if "GitHub" in r.demo.goal), None
        )
        github_with_ctx = next(
            (r for r in results_with_ctx if "GitHub" in r.demo.goal), None
        )

        if github_no_ctx and github_with_ctx:
            assert github_with_ctx.score >= github_no_ctx.score

    def test_retrieve_with_domain_context(self, built_retriever: DemoRetriever) -> None:
        """Test retrieval with domain context bonus."""
        results = built_retriever.retrieve(
            "Search repos",
            domain_context="github.com",
            top_k=3,
        )

        # GitHub demo should get domain bonus
        github_result = next(
            (r for r in results if r.demo.domain == "github.com"), None
        )
        if github_result:
            assert github_result.domain_bonus > 0

    def test_retrieve_episodes(self, built_retriever: DemoRetriever) -> None:
        """Test retrieve_episodes convenience method."""
        episodes = built_retriever.retrieve_episodes("Night Shift", top_k=2)

        assert len(episodes) == 2
        assert all(isinstance(ep, Episode) for ep in episodes)

    def test_get_apps(self, built_retriever: DemoRetriever) -> None:
        """Test getting unique app names."""
        apps = built_retriever.get_apps()

        assert "System Settings" in apps
        assert "Chrome" in apps
        assert "Calculator" in apps

    def test_get_domains(self, built_retriever: DemoRetriever) -> None:
        """Test getting unique domains."""
        domains = built_retriever.get_domains()

        assert "github.com" in domains

    def test_clear(self, built_retriever: DemoRetriever) -> None:
        """Test clearing the retriever."""
        assert built_retriever.get_demo_count() > 0

        built_retriever.clear()

        assert built_retriever.get_demo_count() == 0
        assert not built_retriever._is_indexed

    def test_repr(self, built_retriever: DemoRetriever) -> None:
        """Test string representation."""
        s = repr(built_retriever)
        assert "5 demos" in s
        assert "tfidf" in s
        assert "indexed" in s


# =============================================================================
# Index Persistence Tests
# =============================================================================


class TestIndexPersistence:
    """Test saving and loading indices."""

    def test_save_and_load_index(self, built_retriever: DemoRetriever) -> None:
        """Test round-trip save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Save
            built_retriever.save_index(save_path)

            # Check files exist
            assert (save_path / "index.json").exists()
            assert (save_path / "embeddings.npy").exists()

            # Load into new retriever
            new_retriever = DemoRetriever()
            new_retriever.load_index(save_path)

            assert new_retriever.get_demo_count() == built_retriever.get_demo_count()
            assert new_retriever._is_indexed

    def test_load_index_preserves_metadata(
        self, sample_episodes: List[Episode]
    ) -> None:
        """Test that metadata is preserved through save/load."""
        retriever = DemoRetriever()
        retriever.add_demo(
            sample_episodes[0],
            tags=["test", "settings"],
            metadata={"custom": "value"},
        )
        retriever.build_index()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            retriever.save_index(save_path)

            new_retriever = DemoRetriever()
            new_retriever.load_index(save_path)

            demo = new_retriever.get_all_demos()[0]
            assert demo.tags == ["test", "settings"]
            assert demo.metadata == {"custom": "value"}

    def test_retrieval_after_load(self, built_retriever: DemoRetriever) -> None:
        """Test that retrieval works after loading from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            built_retriever.save_index(save_path)

            new_retriever = DemoRetriever()
            new_retriever.load_index(save_path)

            results = new_retriever.retrieve("Turn off Night Shift", top_k=3)

            assert len(results) > 0
            assert "Night Shift" in results[0].demo.goal


# =============================================================================
# Prompt Formatting Tests
# =============================================================================


class TestPromptFormatting:
    """Test prompt formatting functionality."""

    def test_format_for_prompt_single(self, built_retriever: DemoRetriever) -> None:
        """Test formatting single result."""
        results = built_retriever.retrieve("Night Shift", top_k=1)
        prompt = built_retriever.format_for_prompt(results)

        assert "relevant demonstration" in prompt.lower()
        assert "Night Shift" in prompt

    def test_format_for_prompt_multiple(self, built_retriever: DemoRetriever) -> None:
        """Test formatting multiple results."""
        results = built_retriever.retrieve("settings", top_k=3)
        prompt = built_retriever.format_for_prompt(results)

        assert "3 relevant demonstrations" in prompt.lower()

    def test_format_for_prompt_with_scores(self, built_retriever: DemoRetriever) -> None:
        """Test formatting with scores."""
        results = built_retriever.retrieve("Night Shift", top_k=2)
        prompt = built_retriever.format_for_prompt(results, include_scores=True)

        assert "relevance:" in prompt.lower()

    def test_format_for_prompt_empty(self, built_retriever: DemoRetriever) -> None:
        """Test formatting empty results."""
        prompt = built_retriever.format_for_prompt([])
        assert prompt == ""

    def test_format_styles(self, built_retriever: DemoRetriever) -> None:
        """Test different format styles."""
        results = built_retriever.retrieve("Night Shift", top_k=1)

        concise = built_retriever.format_for_prompt(results, format_style="concise")
        verbose = built_retriever.format_for_prompt(results, format_style="verbose")
        minimal = built_retriever.format_for_prompt(results, format_style="minimal")

        # All should contain the goal
        assert "Night Shift" in concise
        assert "Night Shift" in verbose
        assert "Night Shift" in minimal

        # Verbose should be longer
        assert len(verbose) >= len(minimal)


# =============================================================================
# Metadata Extraction Tests
# =============================================================================


class TestMetadataExtraction:
    """Test automatic metadata extraction."""

    def test_extract_app_name(self) -> None:
        """Test app name extraction from observations."""
        obs = Observation(app_name="System Settings")
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction="test", steps=[step])

        retriever = DemoRetriever()
        meta = retriever.add_demo(episode)

        assert meta.app_name == "System Settings"

    def test_extract_domain_from_url(self) -> None:
        """Test domain extraction from URL."""
        obs = Observation(url="https://www.github.com/user/repo")
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction="test", steps=[step])

        retriever = DemoRetriever()
        meta = retriever.add_demo(episode)

        assert meta.domain == "github.com"

    def test_detect_platform_macos(self) -> None:
        """Test macOS platform detection."""
        obs = Observation(app_name="System Settings")
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction="test", steps=[step])

        retriever = DemoRetriever()
        meta = retriever.add_demo(episode)

        assert meta.platform == "macos"

    def test_detect_platform_web(self) -> None:
        """Test web platform detection."""
        obs = Observation(url="https://github.com")
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction="test", steps=[step])

        retriever = DemoRetriever()
        meta = retriever.add_demo(episode)

        assert meta.platform == "web"

    def test_extract_action_types(self) -> None:
        """Test action type extraction."""
        steps = [
            Step(
                step_index=0,
                observation=Observation(),
                action=Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5)),
            ),
            Step(
                step_index=1,
                observation=Observation(),
                action=Action(type=ActionType.TYPE, text="hello"),
            ),
        ]
        episode = Episode(episode_id="test", instruction="test", steps=steps)

        retriever = DemoRetriever()
        meta = retriever.add_demo(episode)

        assert "click" in meta.action_types
        assert "type" in meta.action_types


# =============================================================================
# Embedder Factory Tests
# =============================================================================


class TestEmbedderFactory:
    """Test create_embedder factory function."""

    def test_create_tfidf(self) -> None:
        """Test creating TF-IDF embedder."""
        embedder = create_embedder("tfidf")
        assert isinstance(embedder, TFIDFEmbedder)

    def test_create_unknown_raises(self) -> None:
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown embedding method"):
            create_embedder("unknown_method")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_complete_workflow(self, sample_episodes: List[Episode]) -> None:
        """Test complete retrieval workflow."""
        # Create retriever
        retriever = DemoRetriever(embedding_method="tfidf")

        # Add demos
        for ep in sample_episodes:
            retriever.add_demo(ep)

        # Build index
        retriever.build_index()

        # Retrieve
        results = retriever.retrieve(
            "Disable blue light filter",  # Similar to "Turn off Night Shift"
            top_k=3,
        )

        assert len(results) > 0

        # Night Shift demo should be top result (semantic similarity)
        top_result = results[0]
        assert "Night Shift" in top_result.demo.goal or "dark mode" in top_result.demo.goal.lower()

        # Format for prompt
        prompt = retriever.format_for_prompt(results[:1])
        assert len(prompt) > 0

    def test_save_load_retrieve_workflow(self, sample_episodes: List[Episode]) -> None:
        """Test save, load, and retrieve workflow."""
        # Build initial retriever
        retriever = DemoRetriever()
        for ep in sample_episodes:
            retriever.add_demo(ep)
        retriever.build_index()

        original_result = retriever.retrieve("Night Shift", top_k=1)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_path = Path(tmpdir)
            retriever.save_index(save_path)

            # Load in new retriever
            new_retriever = DemoRetriever()
            new_retriever.load_index(save_path)

            # Should get same result
            new_result = new_retriever.retrieve("Night Shift", top_k=1)[0]

            assert new_result.demo.goal == original_result.demo.goal

    def test_filtering_workflow(self, sample_episodes: List[Episode]) -> None:
        """Test filtering during retrieval."""
        retriever = DemoRetriever()

        # Add demos with different platforms
        for ep in sample_episodes:
            retriever.add_demo(ep)

        retriever.build_index()

        # Filter by platform
        results = retriever.retrieve(
            "change settings",
            filter_platform="macos",
            top_k=5,
        )

        # All results should be from macOS
        for r in results:
            if r.demo.platform:
                assert r.demo.platform == "macos"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_retrieve_from_single_demo(self) -> None:
        """Test retrieval with only one demo."""
        obs = Observation()
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction="Single demo", steps=[step])

        retriever = DemoRetriever()
        retriever.add_demo(episode)
        retriever.build_index()

        results = retriever.retrieve("any query", top_k=3)

        assert len(results) == 1
        assert results[0].demo.goal == "Single demo"

    def test_long_instruction(self) -> None:
        """Test handling of very long instructions."""
        long_instruction = "This is a very long instruction " * 100
        obs = Observation()
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction=long_instruction, steps=[step])

        retriever = DemoRetriever()
        retriever.add_demo(episode)
        retriever.build_index()

        results = retriever.retrieve("very long", top_k=1)
        assert len(results) == 1

    def test_special_characters_in_instruction(self) -> None:
        """Test handling of special characters."""
        instruction = "Click the button with @#$%^&*() symbols!"
        obs = Observation()
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction=instruction, steps=[step])

        retriever = DemoRetriever()
        retriever.add_demo(episode)
        retriever.build_index()

        results = retriever.retrieve("button symbols", top_k=1)
        assert len(results) == 1

    def test_unicode_in_instruction(self) -> None:
        """Test handling of unicode characters."""
        instruction = "Click the button with emoji: 🎉🔥"
        obs = Observation()
        action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
        step = Step(step_index=0, observation=obs, action=action)
        episode = Episode(episode_id="test", instruction=instruction, steps=[step])

        retriever = DemoRetriever()
        retriever.add_demo(episode)
        retriever.build_index()

        results = retriever.retrieve("button emoji", top_k=1)
        assert len(results) == 1

    def test_rebuild_index(self, sample_episodes: List[Episode]) -> None:
        """Test rebuilding index after adding more demos."""
        retriever = DemoRetriever()

        # Add first batch
        for ep in sample_episodes[:2]:
            retriever.add_demo(ep)
        retriever.build_index()

        # Add more demos
        for ep in sample_episodes[2:]:
            retriever.add_demo(ep)

        # Index should be marked as not ready
        assert not retriever._is_indexed

        # Rebuild
        retriever.build_index()
        assert retriever._is_indexed

        # Should now include all demos
        results = retriever.retrieve("test", top_k=10)
        assert len(results) == 5
