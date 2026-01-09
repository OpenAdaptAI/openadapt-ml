"""Tests for TRL Trainer module.

These tests validate the TRL trainer code structure without requiring GPU or actual model weights.
They focus on:
1. Module imports work correctly
2. Sample conversion to TRL format works
3. TRLTrainingConfig can be created and customized
4. Dry-run logic validation (mocking model loading)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step, UIElement


# Check if optional dependencies are available
def _has_datasets() -> bool:
    """Check if datasets library is installed."""
    try:
        import datasets
        return True
    except ImportError:
        return False


HAS_DATASETS = _has_datasets()
DATASETS_SKIP_REASON = "datasets library not installed"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_episodes() -> List[Episode]:
    """Create minimal sample episodes for testing conversion."""
    return [
        Episode(
            episode_id="test-login-001",
            instruction="Log in with username 'testuser' and password 'testpass'",
            steps=[
                Step(
                    step_index=0,
                    observation=Observation(screenshot_path="/tmp/test_step_0.png"),
                    action=Action(
                        type=ActionType.CLICK,
                        normalized_coordinates=(0.5, 0.3),
                        element=UIElement(element_id="1"),
                    ),
                    reasoning="Click username field",
                ),
                Step(
                    step_index=1,
                    observation=Observation(screenshot_path="/tmp/test_step_1.png"),
                    action=Action(
                        type=ActionType.TYPE,
                        text="testuser",
                        element=UIElement(element_id="1"),
                    ),
                    reasoning="Type username",
                ),
                Step(
                    step_index=2,
                    observation=Observation(screenshot_path="/tmp/test_step_2.png"),
                    action=Action(type=ActionType.DONE),
                    reasoning="Task complete",
                ),
            ],
            success=True,
        )
    ]


@pytest.fixture
def sample_sft_samples() -> List[Dict[str, Any]]:
    """Create minimal SFT samples for testing TRL conversion."""
    return [
        {
            "images": ["/tmp/test_step_0.png"],
            "messages": [
                {"role": "system", "content": "You are a GUI automation agent."},
                {"role": "user", "content": "Goal: Log in\n\nLook at the screenshot."},
                {"role": "assistant", "content": "Thought: Focus username\nAction: CLICK([1])"},
            ],
        },
        {
            "images": ["/tmp/test_step_1.png"],
            "messages": [
                {"role": "system", "content": "You are a GUI automation agent."},
                {"role": "user", "content": "Goal: Log in\n\nLook at the screenshot."},
                {"role": "assistant", "content": "Thought: Type username\nAction: TYPE([1], \"testuser\")"},
            ],
        },
    ]


@pytest.fixture
def temp_images(tmp_path: Path) -> List[Path]:
    """Create temporary test images."""
    from PIL import Image

    images = []
    for i in range(3):
        img_path = tmp_path / f"test_step_{i}.png"
        # Create a simple colored image
        img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
        img.save(img_path)
        images.append(img_path)
    return images


# -----------------------------------------------------------------------------
# Test Module Imports
# -----------------------------------------------------------------------------


class TestTRLTrainerImports:
    """Test that the TRL trainer module can be imported without errors."""

    def test_import_trl_training_config(self) -> None:
        """Test TRLTrainingConfig can be imported."""
        from openadapt_ml.training.trl_trainer import TRLTrainingConfig

        assert TRLTrainingConfig is not None

    def test_import_convert_samples_function(self) -> None:
        """Test _convert_samples_to_trl_format can be imported."""
        from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format

        assert callable(_convert_samples_to_trl_format)

    def test_import_train_with_trl(self) -> None:
        """Test train_with_trl can be imported."""
        from openadapt_ml.training.trl_trainer import train_with_trl

        assert callable(train_with_trl)

    def test_import_train_from_parquet(self) -> None:
        """Test train_from_parquet can be imported."""
        from openadapt_ml.training.trl_trainer import train_from_parquet

        assert callable(train_from_parquet)


# -----------------------------------------------------------------------------
# Test TRLTrainingConfig
# -----------------------------------------------------------------------------


class TestTRLTrainingConfig:
    """Test TRLTrainingConfig dataclass functionality."""

    def test_default_config_creation(self) -> None:
        """Test creating config with default values."""
        from openadapt_ml.training.trl_trainer import TRLTrainingConfig

        config = TRLTrainingConfig()

        # Check default values
        assert config.model_name == "unsloth/Qwen2.5-VL-7B-Instruct"
        assert config.load_in_4bit is True
        assert config.max_seq_length == 4096
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.0
        assert config.finetune_vision_layers is False
        assert config.num_epochs == 3
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 2e-4
        assert config.warmup_ratio == 0.03
        assert config.output_dir == "checkpoints"
        assert config.logging_steps == 10
        assert config.save_strategy == "epoch"

    def test_custom_config_creation(self) -> None:
        """Test creating config with custom values."""
        from openadapt_ml.training.trl_trainer import TRLTrainingConfig

        config = TRLTrainingConfig(
            model_name="my-custom-model",
            load_in_4bit=False,
            max_seq_length=2048,
            lora_r=32,
            lora_alpha=64,
            num_epochs=5,
            batch_size=2,
            learning_rate=1e-4,
            output_dir="/tmp/my_checkpoints",
            finetune_vision_layers=True,
        )

        assert config.model_name == "my-custom-model"
        assert config.load_in_4bit is False
        assert config.max_seq_length == 2048
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.num_epochs == 5
        assert config.batch_size == 2
        assert config.learning_rate == 1e-4
        assert config.output_dir == "/tmp/my_checkpoints"
        assert config.finetune_vision_layers is True

    def test_config_is_dataclass(self) -> None:
        """Test that TRLTrainingConfig is a proper dataclass."""
        from dataclasses import is_dataclass

        from openadapt_ml.training.trl_trainer import TRLTrainingConfig

        assert is_dataclass(TRLTrainingConfig)


# -----------------------------------------------------------------------------
# Test Sample Conversion to TRL Format
# -----------------------------------------------------------------------------


class TestConvertSamplesToTRLFormat:
    """Test conversion of SFT samples to TRL format."""

    def test_convert_samples_empty_list(self) -> None:
        """Test conversion with empty sample list."""
        from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format

        result = _convert_samples_to_trl_format([])
        assert result == []

    def test_convert_samples_missing_images(self) -> None:
        """Test conversion skips samples with missing image files."""
        from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format

        samples = [
            {
                "images": ["/nonexistent/path/image.png"],
                "messages": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"},
                ],
            }
        ]

        result = _convert_samples_to_trl_format(samples)
        # Should skip samples with missing images
        assert len(result) == 0

    def test_convert_samples_with_valid_images(self, temp_images: List[Path]) -> None:
        """Test conversion with valid images loads them as PIL Images."""
        from PIL import Image

        from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format

        samples = [
            {
                "images": [str(temp_images[0])],
                "messages": [
                    {"role": "user", "content": "Test prompt"},
                    {"role": "assistant", "content": "Test response"},
                ],
            },
            {
                "images": [str(temp_images[1])],
                "messages": [
                    {"role": "user", "content": "Another prompt"},
                    {"role": "assistant", "content": "Another response"},
                ],
            },
        ]

        result = _convert_samples_to_trl_format(samples)

        assert len(result) == 2

        # Check that images are now PIL Images
        for sample in result:
            assert "images" in sample
            assert len(sample["images"]) == 1
            assert isinstance(sample["images"][0], Image.Image)

            # Check messages are preserved
            assert "messages" in sample
            assert len(sample["messages"]) >= 2

    def test_convert_samples_with_base_path(self, temp_images: List[Path], tmp_path: Path) -> None:
        """Test conversion with relative paths and base_path."""
        from PIL import Image

        from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format

        # Use relative path
        samples = [
            {
                "images": [temp_images[0].name],  # Just filename, not full path
                "messages": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"},
                ],
            }
        ]

        result = _convert_samples_to_trl_format(samples, base_path=tmp_path)

        assert len(result) == 1
        assert isinstance(result[0]["images"][0], Image.Image)

    def test_convert_samples_preserves_messages(self, temp_images: List[Path]) -> None:
        """Test that messages structure is preserved during conversion."""
        from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format

        original_messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User input"},
            {"role": "assistant", "content": "Assistant output"},
        ]

        samples = [
            {
                "images": [str(temp_images[0])],
                "messages": original_messages,
            }
        ]

        result = _convert_samples_to_trl_format(samples)

        assert len(result) == 1
        assert result[0]["messages"] == original_messages


# -----------------------------------------------------------------------------
# Test Integration with build_next_action_sft_samples
# -----------------------------------------------------------------------------


class TestSFTSampleBuilding:
    """Test building SFT samples from episodes."""

    def test_build_samples_from_episodes(self, sample_episodes: List[Episode]) -> None:
        """Test building SFT samples from Episode objects."""
        from openadapt_ml.datasets.next_action import build_next_action_sft_samples

        samples = build_next_action_sft_samples(sample_episodes, use_som=True)

        # Should create one sample per step with an image
        # All 3 steps have screenshot_path, so we expect 3 samples
        assert len(samples) == 3

        for sample in samples:
            assert "images" in sample
            assert "messages" in sample
            assert len(sample["messages"]) == 3  # system, user, assistant
            assert sample["messages"][0]["role"] == "system"
            assert sample["messages"][1]["role"] == "user"
            assert sample["messages"][2]["role"] == "assistant"

    def test_build_samples_with_som_format(self, sample_episodes: List[Episode]) -> None:
        """Test that SoM format uses element indices in actions."""
        from openadapt_ml.datasets.next_action import build_next_action_sft_samples

        samples = build_next_action_sft_samples(sample_episodes, use_som=True)

        # First action should be CLICK([1])
        first_assistant_content = samples[0]["messages"][2]["content"]
        assert "CLICK([1])" in first_assistant_content

        # Second action should be TYPE([1], "testuser")
        second_assistant_content = samples[1]["messages"][2]["content"]
        assert 'TYPE([1], "testuser")' in second_assistant_content

    def test_build_samples_with_coordinate_format(self, sample_episodes: List[Episode]) -> None:
        """Test that coordinate format uses normalized coordinates."""
        from openadapt_ml.datasets.next_action import build_next_action_sft_samples

        samples = build_next_action_sft_samples(sample_episodes, use_som=False)

        # First action should use coordinates
        first_assistant_content = samples[0]["messages"][2]["content"]
        assert "CLICK(x=0.50, y=0.30)" in first_assistant_content


# -----------------------------------------------------------------------------
# Test Dry-Run / Mock Model Loading
# -----------------------------------------------------------------------------


class TestDryRunModelLoading:
    """Test dry-run functionality by mocking model loading."""

    def test_load_unsloth_model_not_installed(self) -> None:
        """Test fallback when unsloth is not installed."""
        from openadapt_ml.training.trl_trainer import TRLTrainingConfig, _load_unsloth_model

        config = TRLTrainingConfig()

        # Mock unsloth import to raise ImportError
        with patch.dict("sys.modules", {"unsloth": None}):
            with patch(
                "openadapt_ml.training.trl_trainer._load_standard_model"
            ) as mock_standard:
                mock_standard.return_value = (MagicMock(), MagicMock(), False)

                model, tokenizer, is_unsloth = _load_unsloth_model(config)

                # Should fall back to standard model loading
                mock_standard.assert_called_once_with(config)
                assert is_unsloth is False

    @pytest.mark.skipif(not HAS_DATASETS, reason=DATASETS_SKIP_REASON)
    def test_train_with_trl_dry_run(self, temp_images: List[Path]) -> None:
        """Test train_with_trl can be called with mocked dependencies.

        This test validates that the training function properly:
        1. Builds SFT samples from episodes
        2. Converts samples to TRL format (loading images as PIL)
        3. Creates a HuggingFace Dataset
        4. Attempts to load the model

        It uses mocking to avoid actual model downloads.
        """
        from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step
        from openadapt_ml.training.trl_trainer import TRLTrainingConfig, train_with_trl

        # Create episodes with real temp images
        episodes = [
            Episode(
                episode_id="test-001",
                instruction="Test task",
                steps=[
                    Step(
                        step_index=0,
                        observation=Observation(screenshot_path=str(temp_images[0])),
                        action=Action(
                            type=ActionType.CLICK,
                            normalized_coordinates=(0.5, 0.5),
                        ),
                    ),
                    Step(
                        step_index=1,
                        observation=Observation(screenshot_path=str(temp_images[1])),
                        action=Action(type=ActionType.DONE),
                    ),
                ],
            )
        ]

        config = TRLTrainingConfig(num_epochs=1)

        # Mock the model loading to avoid actual model download
        with patch("openadapt_ml.training.trl_trainer._load_unsloth_model") as mock_load_model:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_load_model.return_value = (mock_model, mock_tokenizer, True)

            # This should fail when trying to import trl/SFTTrainer
            # but that's expected - we're just testing up to model loading
            with pytest.raises(ImportError):
                train_with_trl(episodes, config=config)

            # Verify model loading was attempted
            mock_load_model.assert_called_once_with(config)


# -----------------------------------------------------------------------------
# Test Using Synthetic Data
# -----------------------------------------------------------------------------


class TestWithSyntheticData:
    """Test TRL trainer components with synthetic data generation."""

    def test_synthetic_episodes_conversion(self, tmp_path: Path) -> None:
        """Test that synthetic episodes can be converted to SFT samples."""
        from openadapt_ml.datasets.next_action import build_next_action_sft_samples
        from openadapt_ml.ingest.synthetic import generate_synthetic_episodes

        # Generate synthetic episodes (this creates actual images)
        episodes = generate_synthetic_episodes(
            num_episodes=1,
            seed=42,
            output_dir=str(tmp_path / "synthetic"),
            jitter=False,
            use_som=True,
            scenario="login",
        )

        assert len(episodes) == 1
        assert len(episodes[0].steps) == 6  # Login has 6 steps

        # Convert to SFT samples
        samples = build_next_action_sft_samples(episodes, use_som=True)

        assert len(samples) == 6  # One sample per step

    def test_synthetic_to_trl_format(self, tmp_path: Path) -> None:
        """Test full pipeline: synthetic -> SFT samples -> TRL format."""
        from openadapt_ml.datasets.next_action import build_next_action_sft_samples
        from openadapt_ml.ingest.synthetic import generate_synthetic_episodes
        from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format

        # Generate synthetic data
        synthetic_dir = tmp_path / "synthetic"
        episodes = generate_synthetic_episodes(
            num_episodes=1,
            seed=42,
            output_dir=str(synthetic_dir),
            jitter=False,
            use_som=False,
            scenario="login",
        )

        # Convert to SFT samples
        sft_samples = build_next_action_sft_samples(episodes, use_som=False)

        # Convert to TRL format (should load images as PIL)
        trl_samples = _convert_samples_to_trl_format(sft_samples)

        assert len(trl_samples) == 6

        from PIL import Image

        for sample in trl_samples:
            # Images should be loaded as PIL Images
            assert len(sample["images"]) == 1
            assert isinstance(sample["images"][0], Image.Image)

            # Messages should be preserved
            assert "messages" in sample
            assert len(sample["messages"]) == 3

    def test_registration_scenario(self, tmp_path: Path) -> None:
        """Test with registration scenario (more steps)."""
        from openadapt_ml.datasets.next_action import build_next_action_sft_samples
        from openadapt_ml.ingest.synthetic import generate_synthetic_episodes

        # Generate registration episodes
        episodes = generate_synthetic_episodes(
            num_episodes=1,
            seed=42,
            output_dir=str(tmp_path / "synthetic_reg"),
            jitter=False,
            use_som=True,
            scenario="registration",
        )

        assert len(episodes) == 1
        assert len(episodes[0].steps) == 12  # Registration has 12 steps

        # Convert to SFT samples
        samples = build_next_action_sft_samples(episodes, use_som=True)

        assert len(samples) == 12


# -----------------------------------------------------------------------------
# Test Error Handling
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling in TRL trainer."""

    @pytest.mark.skipif(not HAS_DATASETS, reason=DATASETS_SKIP_REASON)
    def test_empty_samples_raises_error(self) -> None:
        """Test that training with no valid samples raises an error."""
        from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step
        from openadapt_ml.training.trl_trainer import train_with_trl

        # Create episode with non-existent image paths
        episodes = [
            Episode(
                episode_id="test-001",
                instruction="Test",
                steps=[
                    Step(
                        step_index=0,
                        observation=Observation(screenshot_path="/nonexistent/image.png"),
                        action=Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5)),
                    ),
                ],
            )
        ]

        # Should raise ValueError because no valid samples after image loading
        with pytest.raises(ValueError, match="No valid training samples"):
            train_with_trl(episodes)


# -----------------------------------------------------------------------------
# Test CLI Interface
# -----------------------------------------------------------------------------


class TestCLIInterface:
    """Test the CLI interface of trl_trainer module."""

    def test_argparse_setup(self) -> None:
        """Test that the module has proper CLI argument parsing."""
        # We can verify the module is set up for CLI by checking it imports
        import openadapt_ml.training.trl_trainer as module

        # The module should have a __name__ == "__main__" block
        # We can verify the argparse setup by checking the source
        import inspect

        source = inspect.getsource(module)
        assert "argparse.ArgumentParser" in source
        assert "--parquet" in source
        assert "--output" in source
        assert "--model" in source
        assert "--epochs" in source
        assert "--use-som" in source
