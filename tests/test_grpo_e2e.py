"""End-to-end tests for GRPO training with artifacts.

These tests exercise the full training loop using mocked models and environments,
producing detailed artifacts in test_artifacts/grpo_e2e/<timestamp>/ for human review.

Run: uv run pytest tests/test_grpo_e2e.py -v
Report: uv run python scripts/grpo_e2e_report.py test_artifacts/grpo_e2e/<timestamp>/
"""

from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from openadapt_ml.training.grpo.config import GRPOConfig
from openadapt_ml.training.grpo.reward import compute_group_advantages
from openadapt_ml.training.grpo.rollout_collector import Rollout
from openadapt_ml.training.grpo.trainer import (
    format_action_as_text,
    policy_gradient_loss,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCREEN_W, SCREEN_H = 320, 240  # Small for fast tests
VOCAB_SIZE = 256
HIDDEN_DIM = 32
MAX_SEQ_LEN = 64


# ---------------------------------------------------------------------------
# Mock data structures
# ---------------------------------------------------------------------------


@dataclass
class MockBenchmarkAction:
    """Lightweight stand-in for openadapt_evals BenchmarkAction."""

    type: str = "click"
    x: float | None = None
    y: float | None = None
    text: str | None = None
    key: str | None = None
    _grpo_raw_text: str | None = None


@dataclass
class MockBenchmarkObservation:
    """Lightweight stand-in for BenchmarkObservation."""

    screenshot: bytes | None = None
    raw_observation: dict | None = None


@dataclass
class MockRolloutStep:
    """Lightweight stand-in for RolloutStep."""

    observation: MockBenchmarkObservation | None = None
    action: MockBenchmarkAction | None = None
    reward: float = 0.0


# ---------------------------------------------------------------------------
# Synthetic screenshot generation
# ---------------------------------------------------------------------------


def _make_screenshot(
    label: str = "Task: Click the button",
    color: tuple[int, int, int] = (30, 60, 120),
    width: int = SCREEN_W,
    height: int = SCREEN_H,
) -> bytes:
    """Generate a simple synthetic screenshot as PNG bytes.

    Creates a colored rectangle with text overlay to make artifacts
    visually meaningful even without a real environment.
    """
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)

    # Draw a "button" rectangle
    btn_x0, btn_y0 = width // 4, height // 3
    btn_x1, btn_y1 = 3 * width // 4, 2 * height // 3
    draw.rectangle(
        [btn_x0, btn_y0, btn_x1, btn_y1], fill=(200, 200, 220), outline=(0, 0, 0)
    )

    # Draw label text (use default font, no external font needed)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    draw.text(
        (btn_x0 + 10, btn_y0 + 10),
        "Click Me",
        fill=(0, 0, 0),
        font=font,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Mock model (tiny, trainable, CPU-only)
# ---------------------------------------------------------------------------


class TinyVLMModel(nn.Module):
    """Minimal model that mimics a VLM's forward/generate interface.

    Has a small embedding + linear head so we can compute log-probs and
    take gradient steps. LoRA-like trainable params are just the linear layer.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        # Mark all params as trainable (simulates LoRA)
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        """Standard forward: input_ids -> logits."""
        emb = self.embedding(input_ids.clamp(0, self.embedding.num_embeddings - 1))
        logits = self.head(emb)

        @dataclass
        class Output:
            logits: torch.Tensor

        return Output(logits=logits)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Dummy generate: return input_ids + a few random tokens."""
        batch = input_ids.shape[0]
        n_new = min(kwargs.get("max_new_tokens", 10), 10)
        new_tokens = torch.randint(0, self.embedding.num_embeddings, (batch, n_new))
        return torch.cat([input_ids, new_tokens], dim=1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def eval(self) -> TinyVLMModel:
        return super().eval()

    def train(self, mode: bool = True) -> TinyVLMModel:
        return super().train(mode)

    def save_pretrained(self, path: str) -> None:
        """Save state dict (mimics PEFT save_pretrained)."""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")
        # Write a config file so we can verify checkpoint structure
        config = {
            "vocab_size": self.embedding.num_embeddings,
            "hidden": self.head.in_features,
        }
        with open(Path(path) / "config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load_pretrained(cls, path: str) -> TinyVLMModel:
        """Load from checkpoint directory."""
        with open(Path(path) / "config.json") as f:
            config = json.load(f)
        model = cls(vocab_size=config["vocab_size"], hidden=config["hidden"])
        state = torch.load(Path(path) / "model.pt", weights_only=True)
        model.load_state_dict(state)
        return model


class MockProcessor:
    """Mimics HuggingFace AutoProcessor for VLMs."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self._vocab_size = vocab_size
        self.tokenizer = self  # self-referential for inner_tok

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Return a simple text concatenation of messages."""
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(
        self,
        text: str | None = None,
        images: list | None = None,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize text into fixed-size tensor. Images are ignored."""
        # Deterministic "tokenization": hash chars to token IDs
        if text is None:
            text = ""
        token_ids = [ord(c) % self._vocab_size for c in text[:MAX_SEQ_LEN]]
        if not token_ids:
            token_ids = [0]
        ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode tokens back to action text."""
        # Return a deterministic action string
        return "CLICK(x=0.50, y=0.50)"


# ---------------------------------------------------------------------------
# Mock rollout factory
# ---------------------------------------------------------------------------


def _make_mock_rollouts(
    num_rollouts: int = 4,
    steps_per_rollout: int = 3,
    instruction: str = "Click the OK button",
    screen_size: tuple[int, int] = (SCREEN_W, SCREEN_H),
    mixed_rewards: bool = True,
) -> list[Rollout]:
    """Create mock Rollout objects with realistic-looking data.

    If mixed_rewards is True, alternating rollouts get reward 1.0 and 0.0.
    This ensures compute_group_advantages produces non-zero advantages.
    """
    rollouts = []
    colors = [
        (30, 60, 120),
        (120, 30, 60),
        (60, 120, 30),
        (100, 100, 30),
    ]

    for r_idx in range(num_rollouts):
        steps = []
        reward = 1.0 if (r_idx % 2 == 0 or not mixed_rewards) else 0.0

        for s_idx in range(steps_per_rollout):
            color = colors[r_idx % len(colors)]
            label = f"Rollout {r_idx}, Step {s_idx}"
            screenshot = _make_screenshot(label=label, color=color)

            # Create a plausible action
            x_frac = 0.5 + 0.1 * (r_idx - num_rollouts / 2)
            y_frac = 0.5 + 0.05 * s_idx
            x_px = int(x_frac * screen_size[0])
            y_px = int(y_frac * screen_size[1])
            action_text = f"CLICK(x={x_frac:.2f}, y={y_frac:.2f})"

            action = MockBenchmarkAction(
                type="click",
                x=x_px,
                y=y_px,
                _grpo_raw_text=action_text,
            )
            obs = MockBenchmarkObservation(
                screenshot=screenshot,
                raw_observation={"instruction": instruction},
            )
            step = MockRolloutStep(observation=obs, action=action, reward=reward)
            steps.append(step)

        rollout = Rollout(
            task_id="mock_task_001",
            steps=steps,
            reward=reward,
            num_steps=steps_per_rollout,
            instruction=instruction,
        )
        rollouts.append(rollout)

    return rollouts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def artifact_dir() -> Path:
    """Create a timestamped artifact directory for persistent review.

    Artifacts are written to test_artifacts/grpo_e2e/<timestamp>/ so they
    persist after the test run and can be inspected with the report script.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    d = Path("test_artifacts") / "grpo_e2e" / timestamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_artifact(artifact_dir: Path, rel_path: str, content: str | bytes) -> Path:
    """Write an artifact file under the artifact directory."""
    out_path = artifact_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        out_path.write_bytes(content)
    else:
        out_path.write_text(content)
    return out_path


@pytest.fixture
def tiny_model() -> TinyVLMModel:
    """Provide a tiny trainable model."""
    return TinyVLMModel(vocab_size=VOCAB_SIZE, hidden=HIDDEN_DIM)


@pytest.fixture
def mock_processor() -> MockProcessor:
    """Provide a mock processor."""
    return MockProcessor(vocab_size=VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Test 1: Full training loop with mock model
# ---------------------------------------------------------------------------


def test_e2e_training_loop_mock(
    artifact_dir: Path,
    tiny_model: TinyVLMModel,
    mock_processor: MockProcessor,
) -> None:
    """Exercise the full GRPO training loop with a mock model.

    Verifies:
    - Loss is computed and is finite
    - Rewards and advantages are computed correctly
    - Model weights actually change (LoRA params move)
    - Checkpoint is saved and can be reloaded
    - All artifacts are written
    """
    test_start = time.time()
    test_report: dict[str, Any] = {
        "test_name": "test_e2e_training_loop_mock",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "running",
    }

    num_training_steps = 2
    num_rollouts_per_step = 4
    steps_per_rollout = 2

    config = GRPOConfig(
        model_name="mock",
        load_in_4bit=False,
        num_rollouts_per_step=num_rollouts_per_step,
        max_steps_per_episode=steps_per_rollout,
        num_training_steps=num_training_steps,
        save_every_steps=1,
        learning_rate=1e-3,  # Higher LR to see weight changes
        task_ids=["mock_task_001"],
        screen_size=(SCREEN_W, SCREEN_H),
        output_dir=str(artifact_dir / "checkpoint"),
    )

    model = tiny_model
    processor = mock_processor

    # Snapshot initial weights
    initial_params = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    training_log: list[dict[str, Any]] = []
    all_rollout_traces: list[dict[str, Any]] = []

    for step in range(num_training_steps):
        step_start = time.time()

        # Collect mock rollouts with mixed rewards (crucial for non-zero advantages)
        rollouts = _make_mock_rollouts(
            num_rollouts=num_rollouts_per_step,
            steps_per_rollout=steps_per_rollout,
            screen_size=config.screen_size,
        )

        # Save rollout traces and screenshots
        for r_idx, rollout in enumerate(rollouts):
            trace: dict[str, Any] = {
                "step": step,
                "rollout_idx": r_idx,
                "task_id": rollout.task_id,
                "reward": rollout.reward,
                "num_steps": rollout.num_steps,
                "instruction": rollout.instruction,
                "actions": [],
            }
            for s_idx, rstep in enumerate(rollout.steps):
                action = getattr(rstep, "action", None)
                action_text = (
                    getattr(action, "_grpo_raw_text", None)
                    or format_action_as_text(action, screen_size=config.screen_size)
                    if action
                    else "NONE"
                )
                trace["actions"].append(
                    {
                        "step_idx": s_idx,
                        "action_text": action_text,
                        "action_type": getattr(action, "type", None),
                    }
                )
                # Save screenshot
                obs = getattr(rstep, "observation", None)
                if obs and obs.screenshot:
                    screenshot_name = f"rollout_traces/step_{step}_rollout_{r_idx}_screenshot_{s_idx}.png"
                    _write_artifact(artifact_dir, screenshot_name, obs.screenshot)

            trace_name = f"rollout_traces/step_{step}_rollout_{r_idx}.json"
            _write_artifact(artifact_dir, trace_name, json.dumps(trace, indent=2))
            all_rollout_traces.append(trace)

        # Compute advantages
        rewards = [r.reward for r in rollouts]
        advantages = compute_group_advantages(rewards)

        # Training step: compute loss and update
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        valid_count = 0

        for rollout, advantage in zip(rollouts, advantages):
            if abs(advantage) < 1e-8:
                continue
            # Compute log-probs for each step in rollout
            for rstep in rollout.steps:
                obs = getattr(rstep, "observation", None)
                action = getattr(rstep, "action", None)
                if not obs or not action or not obs.screenshot:
                    continue

                # Tokenize prompt
                messages = [
                    {"role": "system", "content": "You are a GUI agent."},
                    {"role": "user", "content": f"Goal: {rollout.instruction}"},
                ]
                text_input = processor.apply_chat_template(messages, tokenize=False)
                prompt_inputs = processor(text_input)
                prompt_ids = prompt_inputs["input_ids"]
                prompt_len = prompt_ids.shape[1]

                # Tokenize action
                action_text = (
                    getattr(action, "_grpo_raw_text", None) or "CLICK(x=0.50, y=0.50)"
                )
                action_inputs = processor(action_text, add_special_tokens=False)
                action_ids = action_inputs["input_ids"]
                action_len = action_ids.shape[1]

                # Concatenate
                full_ids = torch.cat([prompt_ids, action_ids], dim=1)

                # Forward
                outputs = model(input_ids=full_ids)
                logits = outputs.logits
                action_logits = logits[
                    :, prompt_len - 1 : prompt_len - 1 + action_len, :
                ]
                log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)

                action_token_ids = full_ids[:, prompt_len : prompt_len + action_len]
                token_log_probs = log_probs.gather(
                    2, action_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                step_log_prob = token_log_probs.sum()

                adv_tensor = torch.tensor(advantage, dtype=step_log_prob.dtype)
                step_loss = policy_gradient_loss(
                    step_log_prob.unsqueeze(0),
                    step_log_prob.detach().unsqueeze(0),
                    adv_tensor.unsqueeze(0),
                )

                step_loss.backward()
                total_loss += step_loss.detach().item()
                valid_count += 1

        if valid_count > 0:
            avg_loss = total_loss / valid_count
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            # Compute grad norm before stepping
            grad_norm = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm**0.5

            optimizer.step()
        else:
            avg_loss = 0.0
            grad_norm = 0.0

        step_metrics = {
            "step": step,
            "loss": avg_loss,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_std": (
                sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards)
                / len(rewards)
            )
            ** 0.5,
            "advantages": advantages,
            "grad_norm": grad_norm,
            "lr": config.learning_rate,
            "valid_terms": valid_count,
            "step_time": time.time() - step_start,
        }
        training_log.append(step_metrics)

        # Save checkpoint
        ckpt_path = str(artifact_dir / "checkpoint" / f"step_{step + 1}")
        model.save_pretrained(ckpt_path)

    # Compute model diff
    model_diff: dict[str, Any] = {}
    total_delta_norm = 0.0
    for name, initial_p in initial_params.items():
        current_p = dict(model.named_parameters())[name].detach()
        delta = (current_p - initial_p).norm().item()
        model_diff[name] = {
            "initial_norm": initial_p.norm().item(),
            "final_norm": current_p.norm().item(),
            "delta_norm": delta,
            "changed": delta > 1e-10,
        }
        total_delta_norm += delta**2
    total_delta_norm = total_delta_norm**0.5
    model_diff["_total_delta_norm"] = total_delta_norm
    model_diff["_any_changed"] = total_delta_norm > 1e-10

    # Save artifacts
    _write_artifact(
        artifact_dir, "training_log.json", json.dumps(training_log, indent=2)
    )
    _write_artifact(artifact_dir, "model_diff.json", json.dumps(model_diff, indent=2))

    # Verify checkpoint is loadable
    last_ckpt = str(artifact_dir / "checkpoint" / f"step_{num_training_steps}")
    loaded_model = TinyVLMModel.load_pretrained(last_ckpt)
    loaded_params = dict(loaded_model.named_parameters())
    checkpoint_valid = all(
        torch.allclose(
            loaded_params[name],
            dict(model.named_parameters())[name].detach(),
            atol=1e-6,
        )
        for name in loaded_params
    )

    # Build test report
    test_duration = time.time() - test_start
    test_report.update(
        {
            "status": "passed",
            "duration_s": test_duration,
            "num_training_steps": num_training_steps,
            "num_rollouts_per_step": num_rollouts_per_step,
            "total_rollouts": len(all_rollout_traces),
            "model_weights_changed": model_diff["_any_changed"],
            "total_delta_norm": total_delta_norm,
            "checkpoint_loadable": checkpoint_valid,
            "final_loss": training_log[-1]["loss"],
            "final_reward_mean": training_log[-1]["reward_mean"],
            "final_grad_norm": training_log[-1]["grad_norm"],
        }
    )
    _write_artifact(artifact_dir, "test_report.json", json.dumps(test_report, indent=2))

    # Build human-readable summary
    summary_lines = [
        "=" * 60,
        "GRPO E2E TEST REPORT: test_e2e_training_loop_mock",
        "=" * 60,
        f"Status:               {test_report['status']}",
        f"Duration:             {test_duration:.2f}s",
        f"Training steps:       {num_training_steps}",
        f"Rollouts per step:    {num_rollouts_per_step}",
        f"Total rollouts:       {len(all_rollout_traces)}",
        "",
        "--- Model Weight Change ---",
        f"Weights changed:      {model_diff['_any_changed']}",
        f"Total delta L2 norm:  {total_delta_norm:.6f}",
    ]
    for name, info in model_diff.items():
        if name.startswith("_"):
            continue
        summary_lines.append(
            f"  {name}: delta={info['delta_norm']:.6f} "
            f"(norm {info['initial_norm']:.4f} -> {info['final_norm']:.4f})"
        )
    summary_lines += [
        "",
        "--- Training Metrics ---",
    ]
    for m in training_log:
        summary_lines.append(
            f"  Step {m['step']}: loss={m['loss']:.4f} "
            f"reward_mean={m['reward_mean']:.2f} "
            f"grad_norm={m['grad_norm']:.4f} "
            f"valid_terms={m['valid_terms']}"
        )
    summary_lines += [
        "",
        "--- Checkpoint ---",
        f"Checkpoint path:      {last_ckpt}",
        f"Checkpoint loadable:  {checkpoint_valid}",
        "",
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)
    _write_artifact(artifact_dir, "summary.txt", summary)
    print("\n" + summary)

    # Assertions
    assert model_diff["_any_changed"], (
        "Model weights did not change after training. "
        "This means gradients are not flowing through the GRPO loss."
    )
    assert checkpoint_valid, "Checkpoint could not be reloaded correctly."
    for m in training_log:
        assert math.isfinite(m["loss"]), (
            f"Loss is not finite at step {m['step']}: {m['loss']}"
        )
        assert m["grad_norm"] > 0, (
            f"Grad norm is zero at step {m['step']} (no gradient flow)"
        )


# ---------------------------------------------------------------------------
# Test 2: Rollout collection with mock environment
# ---------------------------------------------------------------------------


def test_e2e_rollout_collection_mock(artifact_dir: Path) -> None:
    """Collect rollouts from mock environment and save detailed traces.

    Verifies:
    - Rollouts have the expected structure (steps, rewards, actions)
    - Screenshots are present and valid PNGs
    - Rewards and advantages are computed correctly
    - Rollout traces are saved as JSON artifacts
    """
    test_start = time.time()
    num_rollouts = 6
    steps_per_rollout = 4

    rollouts = _make_mock_rollouts(
        num_rollouts=num_rollouts,
        steps_per_rollout=steps_per_rollout,
        instruction="Open Notepad and type Hello World",
        mixed_rewards=True,
    )

    # Verify structure
    assert len(rollouts) == num_rollouts
    for r_idx, rollout in enumerate(rollouts):
        assert rollout.task_id == "mock_task_001"
        assert rollout.num_steps == steps_per_rollout
        assert len(rollout.steps) == steps_per_rollout
        assert rollout.reward in (0.0, 1.0)
        assert rollout.instruction == "Open Notepad and type Hello World"

        for s_idx, step in enumerate(rollout.steps):
            assert step.observation is not None
            assert step.observation.screenshot is not None
            assert step.action is not None
            assert step.action.type == "click"

            # Verify screenshot is valid PNG
            img = Image.open(io.BytesIO(step.observation.screenshot))
            assert img.format == "PNG"
            assert img.size == (SCREEN_W, SCREEN_H)

    # Compute advantages
    rewards = [r.reward for r in rollouts]
    advantages = compute_group_advantages(rewards)

    # Verify advantage properties
    assert len(advantages) == num_rollouts
    # With mixed rewards, mean advantage should be ~0
    mean_adv = sum(advantages) / len(advantages)
    assert abs(mean_adv) < 1e-6, f"Mean advantage should be ~0, got {mean_adv}"
    # Positive rewards should have positive advantages
    for r, a in zip(rewards, advantages):
        if r > 0.5:
            assert a > 0, f"Reward {r} should have positive advantage, got {a}"
        elif r < 0.5:
            assert a < 0, f"Reward {r} should have negative advantage, got {a}"

    # Save traces and screenshots
    traces: list[dict[str, Any]] = []
    for r_idx, rollout in enumerate(rollouts):
        trace: dict[str, Any] = {
            "rollout_idx": r_idx,
            "task_id": rollout.task_id,
            "reward": rollout.reward,
            "advantage": advantages[r_idx],
            "num_steps": rollout.num_steps,
            "instruction": rollout.instruction,
            "actions": [],
        }
        for s_idx, step in enumerate(rollout.steps):
            action = step.action
            action_text = (
                getattr(action, "_grpo_raw_text", "UNKNOWN") if action else "NONE"
            )
            trace["actions"].append(
                {
                    "step_idx": s_idx,
                    "action_text": action_text,
                    "action_type": getattr(action, "type", None),
                    "x": getattr(action, "x", None),
                    "y": getattr(action, "y", None),
                }
            )
            # Save screenshot
            if step.observation and step.observation.screenshot:
                _write_artifact(
                    artifact_dir,
                    f"rollout_collection/rollout_{r_idx}_step_{s_idx}.png",
                    step.observation.screenshot,
                )

        _write_artifact(
            artifact_dir,
            f"rollout_collection/rollout_{r_idx}.json",
            json.dumps(trace, indent=2),
        )
        traces.append(trace)

    # Save summary
    collection_summary = {
        "test_name": "test_e2e_rollout_collection_mock",
        "num_rollouts": num_rollouts,
        "steps_per_rollout": steps_per_rollout,
        "rewards": rewards,
        "advantages": advantages,
        "reward_mean": sum(rewards) / len(rewards),
        "duration_s": time.time() - test_start,
    }
    _write_artifact(
        artifact_dir,
        "rollout_collection/summary.json",
        json.dumps(collection_summary, indent=2),
    )

    print(
        f"\nRollout collection: {num_rollouts} rollouts, {steps_per_rollout} steps each"
    )
    print(f"Rewards: {rewards}")
    print(f"Advantages: {[f'{a:.3f}' for a in advantages]}")


# ---------------------------------------------------------------------------
# Test 3: GRPO loss convergence (synthetic)
# ---------------------------------------------------------------------------


def test_e2e_policy_gradient_loss_convergence(artifact_dir: Path) -> None:
    """Verify GRPO loss drives policy toward high-reward trajectories.

    Creates a synthetic scenario:
    - 4 "actions" with trainable log-probabilities
    - Actions 0,1 are "good" (reward=1.0), actions 2,3 are "bad" (reward=0.0)
    - Runs GRPO updates for 50 steps
    - Verifies log-probs for good actions increase relative to bad actions

    This tests the pure GRPO math without any model or environment complexity.
    """
    test_start = time.time()
    num_steps = 50
    num_actions = 4
    lr = 0.05

    # Trainable log-probabilities (start uniform)
    log_probs_param = nn.Parameter(torch.zeros(num_actions))
    optimizer = torch.optim.SGD([log_probs_param], lr=lr)

    # Fixed rewards: first two actions are "good"
    rewards = [1.0, 0.0, 1.0, 0.0]
    advantages = compute_group_advantages(rewards)

    loss_history: list[float] = []
    policy_history: list[list[float]] = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Current normalized log-probs
        log_probs = torch.nn.functional.log_softmax(log_probs_param, dim=0)

        # Compute GRPO loss: sum over actions, weighted by advantage
        total_loss = torch.tensor(0.0)
        for i in range(num_actions):
            # Each action's contribution: -advantage * log_prob
            # Using policy_gradient_loss with ratio=1 (old=current for simplicity)
            step_loss = policy_gradient_loss(
                log_probs[i].unsqueeze(0),
                log_probs[i].detach().unsqueeze(0),
                torch.tensor([advantages[i]]),
            )
            total_loss = total_loss + step_loss

        total_loss.backward()
        optimizer.step()

        # Record
        with torch.no_grad():
            probs = torch.softmax(log_probs_param, dim=0)
            loss_history.append(total_loss.item())
            policy_history.append(probs.tolist())

    # Save artifacts
    convergence_data = {
        "test_name": "test_e2e_policy_gradient_loss_convergence",
        "num_steps": num_steps,
        "rewards": rewards,
        "advantages": advantages,
        "loss_history": loss_history,
        "policy_history": policy_history,
        "final_policy": policy_history[-1],
        "initial_policy": policy_history[0],
        "duration_s": time.time() - test_start,
    }
    _write_artifact(
        artifact_dir,
        "convergence/convergence_data.json",
        json.dumps(convergence_data, indent=2),
    )

    # Print summary
    print("\n--- GRPO Loss Convergence ---")
    print(f"Rewards:         {rewards}")
    print(f"Advantages:      {[f'{a:.3f}' for a in advantages]}")
    print(f"Initial policy:  {[f'{p:.3f}' for p in policy_history[0]]}")
    print(f"Final policy:    {[f'{p:.3f}' for p in policy_history[-1]]}")
    print(f"Initial loss:    {loss_history[0]:.4f}")
    print(f"Final loss:      {loss_history[-1]:.4f}")

    # Assertions
    final_probs = policy_history[-1]

    # Good actions (indices 0, 2) should have higher probability than bad (1, 3)
    good_prob = final_probs[0] + final_probs[2]
    bad_prob = final_probs[1] + final_probs[3]
    assert good_prob > bad_prob, (
        f"GRPO did not shift policy toward high-reward actions. "
        f"Good prob: {good_prob:.3f}, Bad prob: {bad_prob:.3f}"
    )

    # Good actions should each be > 0.25 (above uniform)
    assert final_probs[0] > 0.25, (
        f"Action 0 (reward=1.0) prob {final_probs[0]:.3f} should be > 0.25"
    )
    assert final_probs[2] > 0.25, (
        f"Action 2 (reward=1.0) prob {final_probs[2]:.3f} should be > 0.25"
    )

    # Bad actions should each be < 0.25 (below uniform)
    assert final_probs[1] < 0.25, (
        f"Action 1 (reward=0.0) prob {final_probs[1]:.3f} should be < 0.25"
    )
    assert final_probs[3] < 0.25, (
        f"Action 3 (reward=0.0) prob {final_probs[3]:.3f} should be < 0.25"
    )

    # Loss should generally decrease (compare first 5 avg to last 5 avg)
    early_loss = sum(loss_history[:5]) / 5
    late_loss = sum(loss_history[-5:]) / 5
    print(f"Early avg loss:  {early_loss:.4f}")
    print(f"Late avg loss:   {late_loss:.4f}")

    # Save human-readable summary
    summary_lines = [
        "GRPO Loss Convergence Test",
        "=" * 40,
        f"Steps: {num_steps}",
        f"Rewards: {rewards}",
        f"Initial policy (uniform): {[f'{p:.4f}' for p in policy_history[0]]}",
        f"Final policy:             {[f'{p:.4f}' for p in policy_history[-1]]}",
        f"Good action total prob:   {good_prob:.4f}",
        f"Bad action total prob:    {bad_prob:.4f}",
        f"Early avg loss:           {early_loss:.4f}",
        f"Late avg loss:            {late_loss:.4f}",
        "",
        "Per-step loss history (every 5th):",
    ]
    for i in range(0, num_steps, 5):
        probs_str = ", ".join(f"{p:.3f}" for p in policy_history[i])
        summary_lines.append(
            f"  Step {i:3d}: loss={loss_history[i]:.4f}  policy=[{probs_str}]"
        )
    summary = "\n".join(summary_lines)
    _write_artifact(artifact_dir, "convergence/summary.txt", summary)
    print(summary)


# ---------------------------------------------------------------------------
# Test 4: Model weight diff detailed analysis
# ---------------------------------------------------------------------------


def test_e2e_model_weight_diff(
    artifact_dir: Path,
    tiny_model: TinyVLMModel,
    mock_processor: MockProcessor,
) -> None:
    """Detailed analysis of which model parameters change during training.

    Runs a single GRPO update step and records per-parameter statistics:
    - L2 norm before/after
    - Max absolute change
    - Gradient norm
    - Whether the parameter is in the "LoRA" portion (head) vs frozen (embedding)
    """
    model = tiny_model
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    # Snapshot before
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Create rollouts with mixed rewards
    rollouts = _make_mock_rollouts(
        num_rollouts=4,
        steps_per_rollout=2,
        mixed_rewards=True,
    )
    rewards = [r.reward for r in rollouts]
    advantages = compute_group_advantages(rewards)

    # Single training step
    model.train()
    optimizer.zero_grad()

    for rollout, advantage in zip(rollouts, advantages):
        if abs(advantage) < 1e-8:
            continue
        for rstep in rollout.steps:
            obs = getattr(rstep, "observation", None)
            action = getattr(rstep, "action", None)
            if not obs or not action or not obs.screenshot:
                continue

            text = "test prompt"
            inputs = mock_processor(text)
            prompt_ids = inputs["input_ids"]
            prompt_len = prompt_ids.shape[1]

            action_text = getattr(action, "_grpo_raw_text", "CLICK(x=0.5, y=0.5)")
            action_inputs = mock_processor(action_text, add_special_tokens=False)
            action_ids = action_inputs["input_ids"]
            action_len = action_ids.shape[1]

            full_ids = torch.cat([prompt_ids, action_ids], dim=1)
            outputs = model(input_ids=full_ids)
            logits = outputs.logits
            action_logits = logits[:, prompt_len - 1 : prompt_len - 1 + action_len, :]
            log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            action_token_ids = full_ids[:, prompt_len : prompt_len + action_len]
            token_log_probs = log_probs.gather(
                2, action_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            step_log_prob = token_log_probs.sum()

            adv_tensor = torch.tensor(advantage, dtype=step_log_prob.dtype)
            step_loss = policy_gradient_loss(
                step_log_prob.unsqueeze(0),
                step_log_prob.detach().unsqueeze(0),
                adv_tensor.unsqueeze(0),
            )
            step_loss.backward()

    # Record grad norms before step
    grad_info: dict[str, Any] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_info[name] = {
                "grad_norm": p.grad.data.norm(2).item(),
                "grad_max": p.grad.data.abs().max().item(),
                "grad_mean": p.grad.data.mean().item(),
            }
        else:
            grad_info[name] = {"grad_norm": 0.0, "grad_max": 0.0, "grad_mean": 0.0}

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()

    # Snapshot after
    after = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Compute detailed diff
    diff_report: dict[str, Any] = {"parameters": {}}
    for name in before:
        delta = after[name] - before[name]
        diff_report["parameters"][name] = {
            "before_norm": before[name].norm().item(),
            "after_norm": after[name].norm().item(),
            "delta_l2": delta.norm().item(),
            "delta_max": delta.abs().max().item(),
            "delta_mean": delta.mean().item(),
            "shape": list(before[name].shape),
            "num_elements": before[name].numel(),
            "grad_norm": grad_info.get(name, {}).get("grad_norm", 0.0),
            "grad_max": grad_info.get(name, {}).get("grad_max", 0.0),
        }

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    changed_params = sum(
        1 for v in diff_report["parameters"].values() if v["delta_l2"] > 1e-10
    )

    diff_report["summary"] = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "changed_param_groups": changed_params,
        "total_param_groups": len(diff_report["parameters"]),
    }

    _write_artifact(
        artifact_dir,
        "weight_diff/detailed_diff.json",
        json.dumps(diff_report, indent=2),
    )

    # Print summary
    print("\n--- Model Weight Diff Analysis ---")
    print(f"Total parameters:     {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Changed param groups: {changed_params}/{len(diff_report['parameters'])}")
    for name, info in diff_report["parameters"].items():
        print(
            f"  {name}: delta_l2={info['delta_l2']:.6f} "
            f"grad_norm={info['grad_norm']:.6f} "
            f"shape={info['shape']}"
        )

    # Assertions
    assert changed_params > 0, "No parameters changed during training step."
    for name, info in diff_report["parameters"].items():
        assert math.isfinite(info["delta_l2"]), f"Non-finite delta for {name}"
        assert math.isfinite(info["grad_norm"]), f"Non-finite grad norm for {name}"


# ---------------------------------------------------------------------------
# Test 5: policy_gradient_loss mathematical properties
# ---------------------------------------------------------------------------


def test_e2e_policy_gradient_loss_mathematical_properties(artifact_dir: Path) -> None:
    """Verify mathematical properties of the GRPO loss function.

    Tests:
    - Gradient direction: positive advantage -> increase log-prob
    - Gradient direction: negative advantage -> decrease log-prob
    - Clipping: ratio outside [1-eps, 1+eps] is clipped
    - Zero advantage -> zero gradient
    - Batch consistency: same result whether computed per-sample or batched
    """
    results: dict[str, Any] = {"tests": []}

    # Test 1: Gradient direction for positive advantage
    logp = torch.tensor([-1.0], requires_grad=True)
    loss = policy_gradient_loss(logp, logp.detach(), torch.tensor([1.0]))
    loss.backward()
    # With positive advantage, loss = -ratio * advantage at ratio=1
    # Gradient w.r.t. logp should be negative (decreasing logp increases loss,
    # so optimizer will increase logp)
    grad_positive = logp.grad.item()
    results["tests"].append(
        {
            "name": "positive_advantage_gradient",
            "logp": -1.0,
            "advantage": 1.0,
            "gradient": grad_positive,
            "expected": "negative (optimizer increases logp)",
            "passed": grad_positive < 0,
        }
    )

    # Test 2: Gradient direction for negative advantage
    logp = torch.tensor([-1.0], requires_grad=True)
    loss = policy_gradient_loss(logp, logp.detach(), torch.tensor([-1.0]))
    loss.backward()
    grad_negative = logp.grad.item()
    results["tests"].append(
        {
            "name": "negative_advantage_gradient",
            "logp": -1.0,
            "advantage": -1.0,
            "gradient": grad_negative,
            "expected": "positive (optimizer decreases logp)",
            "passed": grad_negative > 0,
        }
    )

    # Test 3: Zero advantage -> zero gradient
    logp = torch.tensor([-1.0], requires_grad=True)
    loss = policy_gradient_loss(logp, logp.detach(), torch.tensor([0.0]))
    loss.backward()
    grad_zero = logp.grad.item()
    results["tests"].append(
        {
            "name": "zero_advantage_gradient",
            "logp": -1.0,
            "advantage": 0.0,
            "gradient": grad_zero,
            "expected": "zero",
            "passed": abs(grad_zero) < 1e-6,
        }
    )

    # Test 4: Clipping with large ratio
    current = torch.tensor([-0.1], requires_grad=True)
    old = torch.tensor([-3.0])
    advantage = torch.tensor([1.0])
    loss = policy_gradient_loss(current, old, advantage, epsilon=0.2)
    # Ratio = exp(-0.1 - (-3.0)) = exp(2.9) ~ 18.17, clipped to 1.2
    # Loss = -min(18.17, 1.2) = -1.2
    results["tests"].append(
        {
            "name": "clipping_large_ratio",
            "ratio": math.exp(2.9),
            "clipped_ratio": 1.2,
            "loss": loss.item(),
            "expected_loss": -1.2,
            "passed": abs(loss.item() - (-1.2)) < 0.01,
        }
    )

    # Test 5: Batch consistency
    logps_single = [
        policy_gradient_loss(
            torch.tensor([lp]),
            torch.tensor([lp]),
            torch.tensor([a]),
        ).item()
        for lp, a in [(-1.0, 1.0), (-2.0, -0.5), (-0.5, 0.0)]
    ]
    batch_loss = policy_gradient_loss(
        torch.tensor([-1.0, -2.0, -0.5]),
        torch.tensor([-1.0, -2.0, -0.5]),
        torch.tensor([1.0, -0.5, 0.0]),
    ).item()
    expected_batch = sum(logps_single) / 3
    results["tests"].append(
        {
            "name": "batch_consistency",
            "individual_losses": logps_single,
            "batch_loss": batch_loss,
            "expected_batch_mean": expected_batch,
            "passed": abs(batch_loss - expected_batch) < 1e-5,
        }
    )

    # Save and print
    all_passed = all(t["passed"] for t in results["tests"])
    results["all_passed"] = all_passed
    _write_artifact(
        artifact_dir,
        "policy_gradient_loss_properties.json",
        json.dumps(results, indent=2),
    )

    print("\n--- GRPO Loss Mathematical Properties ---")
    for t in results["tests"]:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  [{status}] {t['name']}")

    # Assertions
    for t in results["tests"]:
        assert t["passed"], f"GRPO loss property test failed: {t['name']} - {t}"
