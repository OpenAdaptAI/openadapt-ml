"""Tests for the GRPO training module.

All tests must pass without a GPU or live WAA server. GPU-dependent
and network-dependent tests are skipped with appropriate markers.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_import_grpo_config():
    """GRPOConfig is importable and has correct defaults."""
    from openadapt_ml.training.grpo import GRPOConfig

    config = GRPOConfig()
    assert config.num_rollouts_per_step == 8
    assert config.max_steps_per_episode == 15
    assert config.temperature == 0.7
    assert config.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert config.load_in_4bit is True
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert config.learning_rate == 5e-6
    assert config.num_training_steps == 1000
    assert config.save_every_steps == 50
    assert config.output_dir == "checkpoints/grpo"
    assert config.stuck_window == 3
    assert config.task_ids == []
    assert config.server_url == "http://localhost:5001"
    assert config.screen_size == (1920, 1080)


def test_grpo_config_custom_values():
    """GRPOConfig accepts custom values."""
    from openadapt_ml.training.grpo import GRPOConfig

    config = GRPOConfig(
        model_name="custom/model",
        num_rollouts_per_step=4,
        task_ids=["task_a", "task_b"],
        learning_rate=1e-5,
        output_dir="/tmp/grpo_test",
    )
    assert config.model_name == "custom/model"
    assert config.num_rollouts_per_step == 4
    assert config.task_ids == ["task_a", "task_b"]
    assert config.learning_rate == 1e-5
    assert config.output_dir == "/tmp/grpo_test"


def test_grpo_config_task_ids_independent():
    """Each GRPOConfig instance has its own task_ids list."""
    from openadapt_ml.training.grpo import GRPOConfig

    config1 = GRPOConfig()
    config2 = GRPOConfig()
    config1.task_ids.append("task_1")
    assert config2.task_ids == []


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------


def test_binary_reward_success():
    """binary_task_success returns 1.0 for scores >= threshold."""
    from openadapt_ml.training.grpo.reward import binary_task_success

    assert binary_task_success(1.0) == 1.0
    assert binary_task_success(0.5) == 1.0
    assert binary_task_success(0.75) == 1.0


def test_binary_reward_failure():
    """binary_task_success returns 0.0 for scores < threshold."""
    from openadapt_ml.training.grpo.reward import binary_task_success

    assert binary_task_success(0.0) == 0.0
    assert binary_task_success(0.49) == 0.0
    assert binary_task_success(0.1) == 0.0


def test_binary_reward_custom_threshold():
    """binary_task_success respects custom threshold."""
    from openadapt_ml.training.grpo.reward import binary_task_success

    assert binary_task_success(0.3, threshold=0.3) == 1.0
    assert binary_task_success(0.29, threshold=0.3) == 0.0


def test_group_advantages_mixed():
    """Mixed rewards produce positive/negative advantages."""
    from openadapt_ml.training.grpo.reward import compute_group_advantages

    advantages = compute_group_advantages([1.0, 0.0, 1.0, 0.0])
    # Successes should get positive advantage
    assert advantages[0] > 0
    assert advantages[2] > 0
    # Failures should get negative advantage
    assert advantages[1] < 0
    assert advantages[3] < 0


def test_group_advantages_all_zero():
    """All-zero rewards produce all-zero advantages."""
    from openadapt_ml.training.grpo.reward import compute_group_advantages

    advantages = compute_group_advantages([0.0, 0.0, 0.0, 0.0])
    assert all(a == 0.0 for a in advantages)


def test_group_advantages_all_one():
    """All-success rewards produce all-zero advantages (no variance)."""
    from openadapt_ml.training.grpo.reward import compute_group_advantages

    advantages = compute_group_advantages([1.0, 1.0, 1.0, 1.0])
    assert all(a == 0.0 for a in advantages)


def test_group_advantages_empty():
    """Empty rewards list produces empty advantages."""
    from openadapt_ml.training.grpo.reward import compute_group_advantages

    assert compute_group_advantages([]) == []


def test_group_advantages_single():
    """Single reward produces zero advantage (no variance)."""
    from openadapt_ml.training.grpo.reward import compute_group_advantages

    assert compute_group_advantages([1.0]) == [0.0]


def test_group_advantages_normalized():
    """Advantages are approximately mean-zero with unit variance."""
    from openadapt_ml.training.grpo.reward import compute_group_advantages

    rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    advantages = compute_group_advantages(rewards)
    mean_adv = sum(advantages) / len(advantages)
    assert abs(mean_adv) < 1e-6  # Mean should be ~0


# ---------------------------------------------------------------------------
# Rollout dataclass tests
# ---------------------------------------------------------------------------


def test_rollout_dataclass():
    """Rollout dataclass initializes correctly."""
    from openadapt_ml.training.grpo.rollout_collector import Rollout

    rollout = Rollout(
        task_id="test_task",
        steps=[{"obs": "mock"}],
        reward=1.0,
        num_steps=5,
    )
    assert rollout.task_id == "test_task"
    assert rollout.reward == 1.0
    assert rollout.num_steps == 5
    assert len(rollout.steps) == 1


def test_rollout_defaults():
    """Rollout has sensible defaults."""
    from openadapt_ml.training.grpo.rollout_collector import Rollout

    rollout = Rollout(task_id="t1")
    assert rollout.steps == []
    assert rollout.reward == 0.0
    assert rollout.num_steps == 0


# ---------------------------------------------------------------------------
# Rollout collector tests (no live VM)
# ---------------------------------------------------------------------------


def test_rollout_collector_requires_evals():
    """GRPORolloutCollector raises ImportError without openadapt-evals."""
    from openadapt_ml.training.grpo.rollout_collector import (
        GRPORolloutCollector,
        RLEnvironment,
    )
    from openadapt_ml.training.grpo import GRPOConfig

    # If openadapt-evals is not installed, this should raise ImportError.
    # If it IS installed, the constructor will try to connect (which we skip).
    config = GRPOConfig(server_url="http://localhost:99999")
    if RLEnvironment is None:
        with pytest.raises(ImportError, match="openadapt-evals"):
            GRPORolloutCollector(config)


# ---------------------------------------------------------------------------
# CoT warmup tests
# ---------------------------------------------------------------------------


def test_cot_sample_format():
    """build_cot_sft_samples produces correctly structured samples."""
    from openadapt_ml.training.grpo.cot_warmup import build_cot_sft_samples
    from openadapt_ml.schema import (
        Action,
        ActionType,
        Episode,
        Observation,
        Step,
    )

    episode = Episode(
        episode_id="test_ep",
        instruction="Click the button",
        steps=[
            Step(
                step_index=0,
                observation=Observation(screenshot_path="/tmp/test.png"),
                action=Action(
                    type=ActionType.CLICK,
                    normalized_coordinates=(0.5, 0.5),
                ),
                reasoning="I see a button in the center of the screen.",
            ),
            Step(
                step_index=1,
                observation=Observation(screenshot_path="/tmp/test2.png"),
                action=Action(type=ActionType.DONE),
                reasoning="The task is complete.",
            ),
        ],
        success=True,
    )

    samples = build_cot_sft_samples([episode])

    assert len(samples) == 2

    # Check structure of first sample
    sample = samples[0]
    assert "images" in sample
    assert "messages" in sample
    assert len(sample["messages"]) == 3  # system, user, assistant

    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant"]

    # Assistant content should have <think> block
    assistant_msg = sample["messages"][2]["content"]
    assert "<think>" in assistant_msg
    assert "</think>" in assistant_msg
    assert "CLICK" in assistant_msg


def test_cot_skips_unsuccessful_episodes():
    """build_cot_sft_samples skips episodes where success=False."""
    from openadapt_ml.training.grpo.cot_warmup import build_cot_sft_samples
    from openadapt_ml.schema import (
        Action,
        ActionType,
        Episode,
        Observation,
        Step,
    )

    episode = Episode(
        episode_id="failed_ep",
        instruction="Do something",
        steps=[
            Step(
                step_index=0,
                observation=Observation(screenshot_path="/tmp/x.png"),
                action=Action(type=ActionType.CLICK, normalized_coordinates=(0.1, 0.1)),
            ),
        ],
        success=False,
    )

    samples = build_cot_sft_samples([episode])
    assert samples == []


def test_cot_no_reasoning_still_works():
    """Samples without reasoning omit <think> block."""
    from openadapt_ml.training.grpo.cot_warmup import build_cot_sft_samples
    from openadapt_ml.schema import (
        Action,
        ActionType,
        Episode,
        Observation,
        Step,
    )

    episode = Episode(
        episode_id="no_cot",
        instruction="Click button",
        steps=[
            Step(
                step_index=0,
                observation=Observation(screenshot_path="/tmp/y.png"),
                action=Action(
                    type=ActionType.CLICK,
                    normalized_coordinates=(0.5, 0.5),
                ),
                # No reasoning field set
            ),
        ],
        success=True,
    )

    samples = build_cot_sft_samples([episode])
    assert len(samples) == 1

    assistant_msg = samples[0]["messages"][2]["content"]
    assert "<think>" not in assistant_msg
    assert "CLICK" in assistant_msg


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_package_exports():
    """All expected names are exported from the grpo package."""
    from openadapt_ml.training.grpo import (
        GRPOConfig,
        binary_task_success,
        compute_group_advantages,
    )

    # Just verify they are importable and are the right types
    assert callable(binary_task_success)
    assert callable(compute_group_advantages)
    assert isinstance(GRPOConfig(), GRPOConfig)


# ---------------------------------------------------------------------------
# Trainer tests (no GPU required)
# ---------------------------------------------------------------------------


def test_trainer_init():
    """GRPOTrainer can be instantiated with a config."""
    from openadapt_ml.training.grpo import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        num_training_steps=0,
        task_ids=["test_task"],
    )
    trainer = GRPOTrainer(config)
    assert trainer._config is config
    assert trainer._step == 0
    assert trainer._model is None


def test_trainer_empty_task_ids():
    """GRPOTrainer.train() raises ValueError if task_ids is empty."""
    from openadapt_ml.training.grpo import GRPOConfig, GRPOTrainer

    config = GRPOConfig(num_training_steps=1, task_ids=[])
    trainer = GRPOTrainer(config)
    with pytest.raises(ValueError, match="task_ids must be non-empty"):
        trainer.train()


# ---------------------------------------------------------------------------
# Action parsing tests
# ---------------------------------------------------------------------------


def test_parse_click_action():
    """_parse_vlm_output_to_action correctly parses CLICK actions."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("CLICK(x=0.5, y=0.25)")
    assert action.type == "click"
    assert action.x == int(0.5 * 1920)
    assert action.y == int(0.25 * 1080)


def test_parse_click_custom_screen_size():
    """_parse_vlm_output_to_action uses custom screen_size."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("CLICK(x=0.5, y=0.5)", screen_size=(1000, 800))
    assert action.x == 500
    assert action.y == 400


def test_parse_click_clamps_negative():
    """_parse_vlm_output_to_action clamps negative coordinates to 0."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("CLICK(x=-0.1, y=-0.2)")
    assert action.x == 0
    assert action.y == 0


def test_parse_click_clamps_above_one():
    """_parse_vlm_output_to_action clamps coordinates above 1.0."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("CLICK(x=1.5, y=2.0)")
    assert action.x == 1920
    assert action.y == 1080


def test_parse_type_action():
    """_parse_vlm_output_to_action correctly parses TYPE actions."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action('TYPE(text="hello world")')
    assert action.type == "type"
    assert action.text == "hello world"


def test_parse_type_action_single_quotes():
    """_parse_vlm_output_to_action parses TYPE with single quotes."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("TYPE(text='hello world')")
    assert action.type == "type"
    assert action.text == "hello world"


def test_parse_wait_action():
    """_parse_vlm_output_to_action correctly parses WAIT actions."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("WAIT()")
    assert action.type == "wait"


def test_parse_done_action():
    """_parse_vlm_output_to_action correctly parses DONE actions."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("DONE()")
    assert action.type == "done"


def test_parse_fallback_to_done():
    """_parse_vlm_output_to_action falls back to DONE for unparseable output."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("I don't know what to do")
    assert action.type == "done"


# ---------------------------------------------------------------------------
# Action formatting tests (_format_action_as_text)
# ---------------------------------------------------------------------------


def test_format_click_action():
    """_format_action_as_text formats CLICK with normalized coordinates."""
    from dataclasses import dataclass

    from openadapt_ml.training.grpo.trainer import _format_action_as_text

    @dataclass
    class FakeAction:
        type: str = "click"
        x: float = 960
        y: float = 600
        text: str = None
        key: str = None

    result = _format_action_as_text(FakeAction(), screen_size=(1920, 1200))
    assert result == "CLICK(x=0.50, y=0.50)"


def test_format_click_custom_screen_size():
    """_format_action_as_text normalizes coordinates to screen size."""
    from dataclasses import dataclass

    from openadapt_ml.training.grpo.trainer import _format_action_as_text

    @dataclass
    class FakeAction:
        type: str = "click"
        x: float = 500
        y: float = 200
        text: str = None
        key: str = None

    result = _format_action_as_text(FakeAction(), screen_size=(1000, 800))
    assert result == "CLICK(x=0.50, y=0.25)"


def test_format_type_action():
    """_format_action_as_text formats TYPE with quoted text."""
    from dataclasses import dataclass

    from openadapt_ml.training.grpo.trainer import _format_action_as_text

    @dataclass
    class FakeAction:
        type: str = "type"
        x: float = None
        y: float = None
        text: str = "hello world"
        key: str = None

    result = _format_action_as_text(FakeAction())
    assert result == 'TYPE(text="hello world")'


def test_format_type_escapes_quotes():
    """_format_action_as_text escapes double quotes in TYPE text."""
    from dataclasses import dataclass

    from openadapt_ml.training.grpo.trainer import _format_action_as_text

    @dataclass
    class FakeAction:
        type: str = "type"
        x: float = None
        y: float = None
        text: str = 'say "hi"'
        key: str = None

    result = _format_action_as_text(FakeAction())
    assert result == 'TYPE(text="say \\"hi\\"")'


def test_format_wait_action():
    """_format_action_as_text formats WAIT."""
    from dataclasses import dataclass

    from openadapt_ml.training.grpo.trainer import _format_action_as_text

    @dataclass
    class FakeAction:
        type: str = "wait"
        x: float = None
        y: float = None
        text: str = None
        key: str = None

    assert _format_action_as_text(FakeAction()) == "WAIT()"


def test_format_done_action():
    """_format_action_as_text formats DONE."""
    from dataclasses import dataclass

    from openadapt_ml.training.grpo.trainer import _format_action_as_text

    @dataclass
    class FakeAction:
        type: str = "done"
        x: float = None
        y: float = None
        text: str = None
        key: str = None

    assert _format_action_as_text(FakeAction()) == "DONE()"


def test_format_unknown_defaults_to_done():
    """_format_action_as_text defaults to DONE for unknown types."""
    from dataclasses import dataclass

    from openadapt_ml.training.grpo.trainer import _format_action_as_text

    @dataclass
    class FakeAction:
        type: str = "scroll"
        x: float = None
        y: float = None
        text: str = None
        key: str = None

    assert _format_action_as_text(FakeAction()) == "DONE()"


def test_format_roundtrip_click():
    """parse -> format -> parse roundtrip preserves CLICK semantics."""
    from openadapt_ml.training.grpo.trainer import (
        _format_action_as_text,
        _parse_vlm_output_to_action,
    )

    original = "CLICK(x=0.50, y=0.25)"
    action = _parse_vlm_output_to_action(original, screen_size=(1920, 1200))
    reconstructed = _format_action_as_text(action, screen_size=(1920, 1200))
    assert reconstructed == original


def test_format_roundtrip_type():
    """parse -> format -> parse roundtrip preserves TYPE semantics."""
    from openadapt_ml.training.grpo.trainer import (
        _format_action_as_text,
        _parse_vlm_output_to_action,
    )

    original = 'TYPE(text="hello world")'
    action = _parse_vlm_output_to_action(original)
    reconstructed = _format_action_as_text(action)
    assert reconstructed == original


# ---------------------------------------------------------------------------
# Case-insensitive parsing tests (M-07)
# ---------------------------------------------------------------------------


def test_parse_click_case_insensitive():
    """_parse_vlm_output_to_action parses lowercase click."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action("click(x=0.5, y=0.25)")
    assert action.type == "click"
    assert action.x == int(0.5 * 1920)
    assert action.y == int(0.25 * 1080)


def test_parse_type_case_insensitive():
    """_parse_vlm_output_to_action parses lowercase type."""
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    action = _parse_vlm_output_to_action('type(text="hello")')
    assert action.type == "type"
    assert action.text == "hello"


# ---------------------------------------------------------------------------
# Unescape order test (I-04)
# ---------------------------------------------------------------------------


def test_parse_type_unescape_order():
    r"""Unescape order: backslash first, then quotes.

    Input contains literal \\\" which should become \" (not \").
    """
    from openadapt_ml.training.grpo.trainer import _parse_vlm_output_to_action

    # Escaped: \\\" means literal backslash + escaped quote
    action = _parse_vlm_output_to_action(r'TYPE(text="hello\\")')
    assert action.type == "type"
    assert action.text == "hello\\"


# ---------------------------------------------------------------------------
# DEFAULT_SCREEN_SIZE constant test (L-01)
# ---------------------------------------------------------------------------


def test_default_screen_size_constant():
    """DEFAULT_SCREEN_SIZE is exported from trainer module."""
    from openadapt_ml.training.grpo.trainer import DEFAULT_SCREEN_SIZE

    assert DEFAULT_SCREEN_SIZE == (1920, 1080)


# ---------------------------------------------------------------------------
# grpo_loss tests
# ---------------------------------------------------------------------------


def test_policy_gradient_loss_positive_advantage():
    """Positive advantage produces gradient that increases log-prob."""
    import torch

    from openadapt_ml.training.grpo.trainer import policy_gradient_loss

    logps = torch.tensor([-1.0], requires_grad=True)
    advantages = torch.tensor([1.0])
    loss = policy_gradient_loss(logps, logps.detach(), advantages)
    loss.backward()
    # Gradient should be negative (minimizing loss = increasing log-prob)
    assert logps.grad.item() < 0
    assert loss.item() < 0


def test_policy_gradient_loss_negative_advantage():
    """Negative advantage produces gradient that decreases log-prob."""
    import torch

    from openadapt_ml.training.grpo.trainer import policy_gradient_loss

    logps = torch.tensor([-1.0], requires_grad=True)
    advantages = torch.tensor([-1.0])
    loss = policy_gradient_loss(logps, logps.detach(), advantages)
    loss.backward()
    # Gradient should be positive (minimizing loss = decreasing log-prob)
    assert logps.grad.item() > 0
    assert loss.item() > 0


def test_policy_gradient_loss_zero_advantage():
    """Zero advantage produces zero gradient."""
    import torch

    from openadapt_ml.training.grpo.trainer import policy_gradient_loss

    logps = torch.tensor([-1.0], requires_grad=True)
    advantages = torch.tensor([0.0])
    loss = policy_gradient_loss(logps, logps.detach(), advantages)
    loss.backward()
    assert abs(logps.grad.item()) < 1e-6
    assert abs(loss.item()) < 1e-6


def test_policy_gradient_loss_clipping():
    """Clipping activates when ratio diverges from 1.0 (off-policy)."""
    import torch

    from openadapt_ml.training.grpo.trainer import policy_gradient_loss

    # Large ratio (current much higher than old)
    current = torch.tensor([-0.5])
    old = torch.tensor([-2.0])
    advantages = torch.tensor([1.0])

    loss = policy_gradient_loss(current, old, advantages, epsilon=0.2)
    # Ratio = exp(-0.5 - (-2.0)) = exp(1.5) ~ 4.48, clipped to 1.2
    # Loss should be -min(4.48 * 1.0, 1.2 * 1.0) = -1.2
    assert abs(loss.item() - (-1.2)) < 1e-4


def test_policy_gradient_loss_importable():
    """policy_gradient_loss and grpo_loss alias are importable."""
    from openadapt_ml.training.grpo.trainer import grpo_loss, policy_gradient_loss

    assert callable(policy_gradient_loss)
    assert grpo_loss is policy_gradient_loss


def test_public_api_exports():
    """Public API exports include parse/format functions and loss."""
    from openadapt_ml.training.grpo import (
        format_action_as_text,
        grpo_loss,
        parse_vlm_output_to_action,
        policy_gradient_loss,
    )

    assert callable(parse_vlm_output_to_action)
    assert callable(format_action_as_text)
    assert callable(policy_gradient_loss)
    assert grpo_loss is policy_gradient_loss


# ---------------------------------------------------------------------------
# Trainer internals: no legacy attributes
# ---------------------------------------------------------------------------


def test_trainer_uses_processor_not_tokenizer():
    """New trainer uses _processor (HF AutoProcessor), not _tokenizer."""
    from openadapt_ml.training.grpo import GRPOConfig, GRPOTrainer

    trainer = GRPOTrainer(GRPOConfig(task_ids=["t"]))
    assert hasattr(trainer, "_processor")
    assert not hasattr(trainer, "_tokenizer")
    assert not hasattr(trainer, "_is_unsloth")
    assert not hasattr(trainer, "_ref_lora_state")


# ---------------------------------------------------------------------------
# get_api_adapter tests (M-03)
# ---------------------------------------------------------------------------


def test_get_api_adapter_exists():
    """get_api_adapter is importable from api_adapter module."""
    from openadapt_ml.models.api_adapter import get_api_adapter

    assert callable(get_api_adapter)
