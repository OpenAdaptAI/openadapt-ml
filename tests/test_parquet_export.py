"""Tests for Parquet export functionality."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing."""
    from openadapt_ml.schemas.sessions import Action, Episode, Observation, Step

    return [
        Episode(
            id="test-001",
            goal="Test click and type",
            steps=[
                Step(
                    t=0.0,
                    observation=Observation(
                        image_path="screenshots/step_0.png",
                        window_title="Test App",
                        app_name="TestApp",
                    ),
                    action=Action(type="click", x=0.5, y=0.5),
                ),
                Step(
                    t=1.5,
                    observation=Observation(
                        image_path="screenshots/step_1.png",
                    ),
                    action=Action(type="type", text="hello world"),
                    thought="Typing greeting",
                ),
            ],
            workflow_id="test-workflow",
        ),
        Episode(
            id="test-002",
            goal="Test scroll",
            steps=[
                Step(
                    t=0.0,
                    observation=Observation(image_path="screenshots/scroll_0.png"),
                    action=Action(type="scroll", scroll_direction="down"),
                ),
            ],
        ),
    ]


@pytest.mark.skipif(
    not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    reason="pyarrow required",
)
class TestParquetExport:
    """Tests for Parquet export."""

    def test_to_parquet_basic(self, sample_episodes, tmp_path):
        """Test basic Parquet export."""
        from openadapt_ml.export import to_parquet

        output_path = tmp_path / "episodes.parquet"
        to_parquet(sample_episodes, str(output_path))

        assert output_path.exists()

        # Verify with pyarrow
        import pyarrow.parquet as pq

        table = pq.read_table(output_path)

        # Should have 3 rows (2 steps + 1 step)
        assert len(table) == 3

        # Check expected columns exist
        expected_columns = [
            "episode_id",
            "goal",
            "step_index",
            "timestamp",
            "action_type",
            "x",
            "y",
            "image_path",
        ]
        for col in expected_columns:
            assert col in table.column_names, f"Missing column: {col}"

    def test_to_parquet_with_summary(self, sample_episodes, tmp_path):
        """Test Parquet export with summary table."""
        from openadapt_ml.export import to_parquet

        output_path = tmp_path / "episodes.parquet"
        to_parquet(sample_episodes, str(output_path), include_summary=True)

        assert output_path.exists()

        summary_path = tmp_path / "episodes_summary.parquet"
        assert summary_path.exists()

        import pyarrow.parquet as pq

        summary_table = pq.read_table(summary_path)

        # Should have 2 rows (2 episodes)
        assert len(summary_table) == 2

        # Check summary columns
        assert "episode_id" in summary_table.column_names
        assert "step_count" in summary_table.column_names
        assert "goal" in summary_table.column_names

    def test_from_parquet_roundtrip(self, sample_episodes, tmp_path):
        """Test Parquet roundtrip (lossy but reconstructable)."""
        from openadapt_ml.export import from_parquet, to_parquet

        output_path = tmp_path / "episodes.parquet"
        to_parquet(sample_episodes, str(output_path))

        # Reconstruct
        reconstructed = from_parquet(str(output_path))

        assert len(reconstructed) == 2

        # Check first episode
        ep1 = next(ep for ep in reconstructed if ep.id == "test-001")
        assert ep1.goal == "Test click and type"
        assert len(ep1.steps) == 2
        assert ep1.steps[0].action.type == "click"
        assert ep1.steps[1].action.type == "type"

    def test_empty_episodes(self, tmp_path):
        """Test exporting empty episode list."""
        from openadapt_ml.export import to_parquet
        from openadapt_ml.schemas.sessions import Episode

        output_path = tmp_path / "empty.parquet"
        to_parquet([], str(output_path))

        import pyarrow.parquet as pq

        table = pq.read_table(output_path)
        assert len(table) == 0

    def test_episode_with_metadata(self, tmp_path):
        """Test exporting episodes with metadata."""
        from openadapt_ml.export import to_parquet
        from openadapt_ml.schemas.sessions import Action, Episode, Observation, Step

        episodes = [
            Episode(
                id="meta-test",
                goal="Test metadata",
                steps=[
                    Step(
                        t=0.0,
                        observation=Observation(image_path="test.png"),
                        action=Action(type="click", x=0.1, y=0.2),
                    )
                ],
                metadata={"domain": "testing", "quality": "high"},
            )
        ]

        output_path = tmp_path / "meta.parquet"
        to_parquet(episodes, str(output_path))

        import pyarrow.parquet as pq

        table = pq.read_table(output_path)
        df = table.to_pandas()

        assert "episode_metadata" in df.columns
        assert '{"domain": "testing"' in df.iloc[0]["episode_metadata"]
