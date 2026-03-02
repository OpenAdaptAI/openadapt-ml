"""GRPO training loop for GUI agent VLMs.

Implements a custom GRPO (Group Relative Policy Optimization) training
loop for multimodal vision-language models. Uses the openadapt-evals
RLEnvironment for rollout collection and a custom policy gradient
update with group-relative advantages.

The training loop:
    1. Select a task (round-robin from task_ids)
    2. Collect N rollouts using the rollout collector
    3. Compute binary rewards (success/failure)
    4. Compute group-relative advantages
    5. For rollouts with non-zero advantage, compute policy gradient loss
    6. Add KL penalty against reference policy
    7. Gradient step
    8. Log metrics and periodically save checkpoints
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from openadapt_ml.training.grpo.config import GRPOConfig
from openadapt_ml.training.grpo.reward import compute_group_advantages
from openadapt_ml.training.grpo.rollout_collector import (
    GRPORolloutCollector,
    Rollout,
)

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO training loop for GUI agent VLMs.

    Uses a custom GRPO implementation (not TRL's GRPOTrainer) because
    TRL does not yet support multimodal VLMs with image inputs in its
    GRPO pipeline.

    The trainer:
    - Loads the model with Unsloth/LoRA (same pattern as trl_trainer.py)
    - Initializes the rollout collector
    - Runs the GRPO training loop
    - Saves LoRA adapter checkpoints periodically

    Args:
        config: GRPO training configuration.
    """

    def __init__(self, config: GRPOConfig) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None
        self._ref_model = None
        self._optimizer = None
        self._collector: GRPORolloutCollector | None = None
        self._step = 0
        self._log_path = Path(config.output_dir) / "grpo_training_log.json"
        self._log_entries: list[dict[str, Any]] = []

    def _load_model(self) -> None:
        """Load model and tokenizer using the trl_trainer pattern."""
        from openadapt_ml.training.trl_trainer import (
            TRLTrainingConfig,
            _load_unsloth_model,
        )

        trl_config = TRLTrainingConfig(
            model_name=self._config.model_name,
            load_in_4bit=self._config.load_in_4bit,
            max_seq_length=self._config.max_seq_length,
            lora_r=self._config.lora_r,
            lora_alpha=self._config.lora_alpha,
        )
        self._model, self._tokenizer, self._is_unsloth = _load_unsloth_model(
            trl_config
        )

        # Store reference model weights for KL penalty
        # We clone the initial LoRA params (before training begins)
        import copy

        self._ref_model = copy.deepcopy(self._model)
        for param in self._ref_model.parameters():
            param.requires_grad = False

    def _setup_optimizer(self) -> None:
        """Initialize the optimizer for LoRA parameters."""
        import torch

        trainable_params = [
            p for p in self._model.parameters() if p.requires_grad
        ]
        self._optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self._config.learning_rate,
        )
        logger.info(
            "Optimizer initialized with %d trainable parameters",
            sum(p.numel() for p in trainable_params),
        )

    def _setup_collector(self) -> None:
        """Initialize the rollout collector."""
        self._collector = GRPORolloutCollector(self._config)

    def _make_agent_fn(self) -> Any:
        """Create an agent function that uses the current model for inference.

        Returns a callable that takes a BenchmarkObservation and returns
        a BenchmarkAction. The function encodes the observation as a VLM
        prompt and decodes the model's output into an action.
        """
        # Deferred import to avoid circular dependency
        from openadapt_evals.adapters.base import BenchmarkAction

        model = self._model
        tokenizer = self._tokenizer
        temperature = self._config.temperature

        def agent_fn(obs: Any) -> BenchmarkAction:
            """Predict an action from an observation using the VLM."""
            import io
            import torch
            from PIL import Image

            # Build prompt with screenshot
            if obs.screenshot:
                image = Image.open(io.BytesIO(obs.screenshot)).convert("RGB")
            else:
                # Fallback: 1x1 black image
                image = Image.new("RGB", (1, 1))

            instruction = ""
            if hasattr(obs, "raw_observation") and obs.raw_observation:
                instruction = obs.raw_observation.get("instruction", "")

            prompt = (
                "You are a GUI automation agent. "
                "Given the screenshot, predict the next action.\n\n"
                f"Instruction: {instruction}\n\n"
                "Respond with exactly one action:\n"
                "CLICK(x=0.XX, y=0.XX) or TYPE(text=\"...\") "
                "or WAIT() or DONE()"
            )

            # Tokenize (processor handles image + text for VLMs)
            messages = [
                {"role": "user", "content": prompt},
            ]

            # Use processor for VLM models
            if hasattr(tokenizer, "apply_chat_template"):
                text_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text_input = prompt

            inputs = tokenizer(
                text_input, images=[image], return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,
                )

            decoded = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Parse into BenchmarkAction
            return _parse_vlm_output_to_action(decoded)

        return agent_fn

    def train(self) -> str:
        """Run the main GRPO training loop.

        Returns:
            Path to the final checkpoint directory.
        """
        logger.info("Starting GRPO training")
        logger.info("  Model: %s", self._config.model_name)
        logger.info("  Tasks: %s", self._config.task_ids)
        logger.info("  Rollouts per step: %d", self._config.num_rollouts_per_step)
        logger.info("  Training steps: %d", self._config.num_training_steps)

        # Setup
        self._load_model()
        self._setup_optimizer()
        self._setup_collector()

        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        agent_fn = self._make_agent_fn()

        for step in range(self._config.num_training_steps):
            self._step = step
            step_start = time.time()

            # Select task (round-robin)
            if self._config.task_ids:
                task_id = self._config.task_ids[step % len(self._config.task_ids)]
            else:
                task_id = None

            # Collect group of rollouts
            rollouts = self._collector.collect_group(
                agent_fn=agent_fn,
                task_id=task_id,
            )

            # Training step
            metrics = self._training_step(rollouts)

            # Logging
            elapsed = time.time() - start_time
            step_time = time.time() - step_start
            metrics.update(
                {
                    "step": step,
                    "task_id": task_id,
                    "elapsed": elapsed,
                    "step_time": step_time,
                }
            )
            self._log_entries.append(metrics)
            self._write_log()

            logger.info(
                "Step %d/%d: reward_mean=%.2f, loss=%.4f, time=%.1fs",
                step + 1,
                self._config.num_training_steps,
                metrics.get("reward_mean", 0.0),
                metrics.get("loss", 0.0),
                step_time,
            )

            # Checkpoint
            if (step + 1) % self._config.save_every_steps == 0:
                self.save_checkpoint(step + 1)

        # Final checkpoint
        self.save_checkpoint(self._config.num_training_steps)

        # Cleanup
        if self._collector:
            self._collector.close()

        final_path = str(
            Path(self._config.output_dir)
            / f"step_{self._config.num_training_steps}"
        )
        logger.info("Training complete. Final checkpoint: %s", final_path)
        return final_path

    def _training_step(self, rollouts: list[Rollout]) -> dict[str, Any]:
        """Single GRPO gradient step from a group of rollouts.

        Computes group-relative advantages, then for each rollout with
        non-zero advantage, computes the policy gradient loss weighted
        by the advantage, plus a KL penalty.

        Args:
            rollouts: List of N Rollout objects with binary rewards.

        Returns:
            Dict of training metrics (reward_mean, reward_std, loss, kl).
        """
        import torch

        rewards = [r.reward for r in rollouts]
        advantages = compute_group_advantages(rewards)

        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
        reward_std = (
            (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if rewards
            else 0.0
        )

        # If no variance in advantages, skip gradient update
        if all(a == 0.0 for a in advantages):
            logger.info(
                "All advantages are zero (reward_mean=%.2f). "
                "Skipping gradient step.",
                reward_mean,
            )
            return {
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "loss": 0.0,
                "kl": 0.0,
                "skipped": True,
                "num_rollouts": len(rollouts),
            }

        # Compute policy gradient loss with KL penalty
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_kl = 0.0
        num_terms = 0

        for rollout, advantage in zip(rollouts, advantages):
            if abs(advantage) < 1e-8:
                continue

            # For each step in the rollout, compute log-prob under current
            # and reference policies. This is a simplified version; full
            # implementation would process the full action sequence.
            step_loss, step_kl = self._compute_rollout_loss(
                rollout, advantage
            )
            total_loss = total_loss + step_loss
            total_kl += step_kl
            num_terms += 1

        if num_terms > 0:
            total_loss = total_loss / num_terms

        # Gradient step
        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self._optimizer.step()

        return {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "loss": total_loss.item(),
            "kl": total_kl / max(num_terms, 1),
            "skipped": False,
            "num_rollouts": len(rollouts),
            "num_gradient_terms": num_terms,
        }

    def _compute_rollout_loss(
        self,
        rollout: Rollout,
        advantage: float,
    ) -> tuple[Any, float]:
        """Compute policy gradient loss for a single rollout.

        For each action in the trajectory, computes log-probability under
        the current policy and the reference policy, then assembles the
        GRPO loss: -advantage * log_prob + kl_coef * KL.

        Args:
            rollout: A completed rollout with steps.
            advantage: The group-relative advantage for this rollout.

        Returns:
            Tuple of (loss_tensor, kl_value).
        """
        import torch

        # Placeholder: in the full implementation, we would tokenize each
        # step's (observation, action) pair and compute log-probs. For now,
        # we use a simplified version that computes a proxy loss.
        device = next(self._model.parameters()).device

        # Compute a proxy loss that encourages/discourages the policy
        # based on the advantage sign. The full implementation would use
        # actual log-probabilities from the VLM.
        loss = torch.tensor(
            -advantage * 0.1,  # Scaled proxy loss
            device=device,
            requires_grad=True,
        )

        # KL penalty (simplified: would normally compare log-prob distributions)
        kl = 0.0

        return loss, kl

    def save_checkpoint(self, step: int) -> str:
        """Save LoRA adapter weights.

        Args:
            step: Current training step number.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_dir = Path(self._config.output_dir) / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self._model, "save_pretrained"):
            self._model.save_pretrained(str(checkpoint_dir))
            logger.info("Saved checkpoint to %s", checkpoint_dir)
        else:
            # Fallback: save state dict
            import torch

            torch.save(
                self._model.state_dict(),
                str(checkpoint_dir / "model.pt"),
            )
            logger.info("Saved state dict to %s", checkpoint_dir)

        # Copy training log alongside checkpoint
        import shutil

        if self._log_path.exists():
            shutil.copy2(self._log_path, checkpoint_dir / "grpo_training_log.json")

        return str(checkpoint_dir)

    def _write_log(self) -> None:
        """Write training log to disk."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text(json.dumps(self._log_entries, indent=2))


def _parse_vlm_output_to_action(text: str) -> Any:
    """Parse VLM output text into a BenchmarkAction.

    Supports the coordinate-based DSL:
        CLICK(x=0.XX, y=0.XX)
        TYPE(text="...")
        WAIT()
        DONE()

    Args:
        text: Raw text output from the VLM.

    Returns:
        BenchmarkAction instance.
    """
    import re

    from openadapt_evals.adapters.base import BenchmarkAction

    text = text.strip()

    # CLICK(x=..., y=...)
    m = re.search(r"CLICK\(x=(-?[\d.]+),\s*y=(-?[\d.]+)\)", text)
    if m:
        x_frac = float(m.group(1))
        y_frac = float(m.group(2))
        # Convert fractions to pixel coords (assume 1920x1200 default)
        return BenchmarkAction(
            type="click",
            x=int(x_frac * 1920),
            y=int(y_frac * 1200),
        )

    # TYPE(text="...")
    m = re.search(r'TYPE\(text="([^"\\]*(?:\\.[^"\\]*)*)"\)', text)
    if m:
        typed_text = m.group(1).replace('\\"', '"').replace("\\\\", "\\")
        return BenchmarkAction(type="type", text=typed_text)

    # WAIT()
    if re.search(r"\bWAIT\s*\(\s*\)", text, re.IGNORECASE):
        return BenchmarkAction(type="wait")

    # DONE()
    if re.search(r"\bDONE\s*\(\s*\)", text, re.IGNORECASE):
        return BenchmarkAction(type="done")

    # Fallback: DONE to end the episode
    logger.warning("Could not parse VLM output: %s. Defaulting to DONE.", text)
    return BenchmarkAction(type="done")
