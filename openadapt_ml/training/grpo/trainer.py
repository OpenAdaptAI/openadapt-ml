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

from openadapt_ml.datasets.next_action import SYSTEM_PROMPT
from openadapt_ml.training.grpo.config import GRPOConfig
from openadapt_ml.training.grpo.reward import compute_group_advantages
from openadapt_ml.training.grpo.rollout_collector import (
    GRPORolloutCollector,
    Rollout,
)

logger = logging.getLogger(__name__)

DEFAULT_SCREEN_SIZE: tuple[int, int] = (1920, 1080)


def _build_agent_messages(instruction: str) -> list[dict[str, str]]:
    """Build the chat messages for the GRPO agent.

    Uses the same SYSTEM_PROMPT as SFT training (from next_action.py)
    so that the GRPO policy operates in the same prompt distribution
    the model was warm-started on.

    This function is the **single source of truth** for prompt
    construction during both rollout collection and loss computation.
    Any change here is automatically reflected in both paths.

    Args:
        instruction: The task instruction from the environment.

    Returns:
        List of message dicts with ``role`` and ``content`` keys,
        ready for ``apply_chat_template``.
    """
    user_content = (
        f"Goal: {instruction}\n\n"
        "Look at the screenshot and determine the NEXT action.\n\n"
        'Action: [CLICK(x=..., y=...) or TYPE(text="...") or WAIT() or DONE()]'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


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
        self._is_unsloth: bool = False
        self._ref_lora_state: dict[str, Any] = {}
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
        self._model, self._tokenizer, self._is_unsloth = _load_unsloth_model(trl_config)

        # Store reference LoRA weights for KL penalty.
        # Instead of deep-copying the entire model (which would OOM for
        # quantized VLMs), we snapshot the initial LoRA adapter state dict.
        # During KL computation, we can swap adapter weights or use
        # disable_adapter_layers() to get base model log-probs.
        self._ref_lora_state = {
            k: v.detach().clone()
            for k, v in self._model.state_dict().items()
            if "lora" in k.lower()
        }
        logger.info(
            "Saved reference LoRA weights: %d tensors",
            len(self._ref_lora_state),
        )

    def _setup_optimizer(self) -> None:
        """Initialize the optimizer for LoRA parameters."""
        import torch

        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
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

        The closure captures the model by reference, so weight updates
        during training are automatically reflected in subsequent rollouts.
        """
        # Deferred import to avoid circular dependency
        from openadapt_evals.adapters.base import BenchmarkAction

        model = self._model
        tokenizer = self._tokenizer
        temperature = self._config.temperature
        collector = self._collector
        config_screen_size = self._config.screen_size

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

            # CR-01: Get instruction from the environment's current task.
            # WAALiveAdapter._get_observation() does NOT populate
            # raw_observation, so we read instruction directly from
            # the task object set during reset().
            instruction = ""
            if collector and hasattr(collector, "env"):
                task = getattr(collector.env, "_current_task", None)
                if task is not None:
                    instruction = getattr(task, "instruction", "") or ""
            # Fallback: try obs.raw_observation (for future compatibility)
            if not instruction:
                raw_obs = getattr(obs, "raw_observation", None)
                if raw_obs and isinstance(raw_obs, dict):
                    instruction = raw_obs.get("instruction", "")

            # Use shared prompt builder (single source of truth)
            messages = _build_agent_messages(instruction)

            # Use processor for VLM models
            if hasattr(tokenizer, "apply_chat_template"):
                text_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text_input = messages[-1]["content"]

            inputs = tokenizer(text_input, images=[image], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,
                )

            decoded = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # IM-01: Use config screen_size, override from live env if available
            screen_size = config_screen_size
            if (
                collector
                and hasattr(collector, "env")
                and hasattr(collector.env, "screen_size")
            ):
                try:
                    screen_size = collector.env.screen_size
                except Exception:
                    pass
            action = _parse_vlm_output_to_action(decoded, screen_size=screen_size)

            # C-01: Store raw model output for accurate loss computation.
            # _compute_rollout_loss uses this instead of reconstructing DSL text.
            try:
                action._grpo_raw_text = decoded
            except AttributeError:
                pass  # __slots__ dataclass; loss will reconstruct from DSL
            return action

        return agent_fn

    def train(self) -> str:
        """Run the main GRPO training loop.

        Returns:
            Path to the final checkpoint directory.
        """
        if not self._config.task_ids:
            raise ValueError(
                "config.task_ids must be non-empty. Provide at least one "
                "WAA task ID to train on."
            )

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

            # Select task (round-robin); task_ids is validated non-empty above
            task_id = self._config.task_ids[step % len(self._config.task_ids)]

            # Collect group of rollouts (inference mode)
            self._model.eval()
            rollouts = self._collector.collect_group(
                agent_fn=agent_fn,
                task_id=task_id,
            )

            # Training step (training mode)
            self._model.train()
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
            Path(self._config.output_dir) / f"step_{self._config.num_training_steps}"
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
                "All advantages are zero (reward_mean=%.2f). Skipping gradient step.",
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

        # Compute policy gradient loss with KL penalty.
        # I-03: Per-step gradient accumulation. Zero gradients once,
        # accumulate through all rollouts+steps, then clip and step.
        # This prevents OOM from building a computation graph over all
        # steps in all rollouts before calling backward().
        valid_pairs = [(r, a) for r, a in zip(rollouts, advantages) if abs(a) >= 1e-8]
        num_terms = len(valid_pairs)

        if num_terms == 0:
            return {
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "loss": 0.0,
                "kl": 0.0,
                "skipped": True,
                "num_rollouts": len(rollouts),
            }

        self._optimizer.zero_grad()

        total_loss_value = 0.0
        total_kl = 0.0

        for rollout, advantage in valid_pairs:
            loss_val, kl_val = self._compute_rollout_loss(
                rollout, advantage, loss_scale=1.0 / num_terms
            )
            total_loss_value += loss_val
            total_kl += kl_val

        # Clip gradients and step
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self._optimizer.step()

        return {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "loss": total_loss_value / max(num_terms, 1),
            "kl": total_kl / max(num_terms, 1),
            "skipped": False,
            "num_rollouts": len(rollouts),
            "num_gradient_terms": num_terms,
        }

    def _compute_rollout_loss(
        self,
        rollout: Rollout,
        advantage: float,
        loss_scale: float = 1.0,
    ) -> tuple[float, float]:
        """Compute GRPO policy gradient loss for a single rollout.

        For each step in the rollout:
        1. Reconstruct the VLM prompt from the observation screenshot
        2. Use raw model output text if available (C-01), else reconstruct DSL
        3. Tokenize prompt and action *separately* then concatenate (C-02)
        4. Compute log-probability of action tokens under current policy
        5. Compute log-probability under reference policy (weight swap)
        6. Backward immediately per-step to avoid OOM (I-03)

        Args:
            rollout: Rollout with steps containing observations and actions.
            advantage: Group-relative advantage for this rollout.
            loss_scale: Multiplier for gradient scaling (1/num_rollouts).

        Returns:
            Tuple of (mean_loss_scalar, mean_kl_scalar) for logging only.
            Gradients are accumulated via per-step backward() calls.
        """
        import io

        import torch
        from PIL import Image

        device = next(self._model.parameters()).device
        total_loss_value = 0.0
        total_kl = 0.0

        # IM-01: Use config screen_size, override from live env if available
        screen_size = self._config.screen_size
        if self._collector and hasattr(self._collector, "env"):
            try:
                screen_size = self._collector.env.screen_size
            except Exception:
                pass

        # First pass: collect valid steps
        valid_steps = []
        for step in rollout.steps:
            obs = getattr(step, "observation", None)
            action = getattr(step, "action", None)
            if obs is None or action is None:
                continue
            screenshot = getattr(obs, "screenshot", None)
            if not screenshot:
                continue
            valid_steps.append((obs, action, screenshot))

        num_steps = len(valid_steps)
        if num_steps == 0:
            return 0.0, 0.0

        # CR-01: Use rollout.instruction (populated by collector from
        # env._current_task.instruction) instead of trying to extract
        # from each observation's raw_observation (which is never set
        # by WAALiveAdapter).
        instruction = getattr(rollout, "instruction", "") or ""

        for obs, action, screenshot in valid_steps:
            try:
                image = Image.open(io.BytesIO(screenshot)).convert("RGB")
            except Exception:
                continue

            # Use shared prompt builder (must match _make_agent_fn exactly)
            messages = _build_agent_messages(instruction)

            # C-01: Use raw model output if available, else reconstruct DSL
            raw_text = getattr(action, "_grpo_raw_text", None)
            action_text = (
                raw_text
                if raw_text
                else _format_action_as_text(action, screen_size=screen_size)
            )

            # Tokenize prompt (with image)
            if hasattr(self._tokenizer, "apply_chat_template"):
                text_input = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text_input = messages[-1]["content"]

            prompt_inputs = self._tokenizer(
                text_input, images=[image], return_tensors="pt"
            )
            prompt_ids = prompt_inputs["input_ids"]
            prompt_len = prompt_ids.shape[1]

            # C-02: Tokenize action text separately and concatenate.
            # This guarantees correct token boundary alignment regardless
            # of BPE merges that differ when text is tokenized jointly.
            inner_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
            action_ids = inner_tok(
                action_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            action_len = action_ids.shape[1]
            if action_len <= 0:
                continue

            # Build full input by concatenating prompt + action token IDs
            full_ids = torch.cat([prompt_ids, action_ids.to(prompt_ids.device)], dim=1)
            full_inputs = dict(prompt_inputs)
            full_inputs["input_ids"] = full_ids
            full_inputs["attention_mask"] = torch.ones_like(full_ids)
            full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

            action_token_ids = full_ids[:, prompt_len : prompt_len + action_len]

            # --- Current policy log-probs (with gradient) ---
            outputs = self._model(**full_inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

            # Autoregressive: logits[:, t, :] predicts token at position t+1
            action_logits = logits[:, prompt_len - 1 : prompt_len - 1 + action_len, :]

            log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, action_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            step_log_prob = token_log_probs.sum()

            # --- Reference policy log-probs (no gradient) ---
            with torch.no_grad():
                ref_step_log_prob = self._compute_ref_log_probs(
                    full_inputs, prompt_len, action_len, action_token_ids
                )

            # --- Per-step loss + immediate backward (I-03) ---
            step_kl = (step_log_prob - ref_step_log_prob).detach().item()
            total_kl += step_kl

            # Loss: -advantage * log π_θ + β * (log π_θ - log π_ref)
            kl_penalty = step_log_prob - ref_step_log_prob.detach()
            step_loss = -advantage * step_log_prob + self._config.kl_coef * kl_penalty

            # Scale and backward immediately to free the computation graph
            scaled_loss = step_loss * loss_scale / num_steps
            scaled_loss.backward()

            total_loss_value += step_loss.detach().item()

        if num_steps == 0:
            return 0.0, 0.0

        return total_loss_value / num_steps, total_kl / num_steps

    def _compute_ref_log_probs(
        self,
        full_inputs: dict[str, Any],
        prompt_len: int,
        action_len: int,
        action_token_ids: Any,
    ) -> Any:
        """Compute log-probabilities under the reference policy.

        I-01: Prefers weight swapping (captures initial LoRA after SFT
        warm-start). ``disable_adapter()`` gives base model log-probs (no
        LoRA at all), which is wrong after SFT warm-up because the
        reference should be the initial LoRA weights, not the base model.

        Must be called inside ``torch.no_grad()``.

        Args:
            full_inputs: Tokenized full sequence (prompt + action).
            prompt_len: Number of tokens in the prompt.
            action_len: Number of action tokens.
            action_token_ids: Token IDs of the action portion.

        Returns:
            Scalar tensor with sum of reference log-probs for action tokens.
        """
        import torch

        # Primary: swap LoRA weights to reference snapshot.
        # This captures the initial LoRA state after SFT warm-start.
        if self._ref_lora_state:
            saved_state: dict[str, Any] = {}
            for name, param in self._model.named_parameters():
                if name in self._ref_lora_state:
                    saved_state[name] = param.data.clone()
                    param.data.copy_(self._ref_lora_state[name])

            # IM-02: try/finally ensures weights are restored even if
            # the forward pass raises (e.g., OOM). Without this, the
            # model would be permanently left in the reference state.
            try:
                ref_outputs = self._model(**full_inputs)
            finally:
                for name, param in self._model.named_parameters():
                    if name in saved_state:
                        param.data.copy_(saved_state[name])
                del saved_state
        elif hasattr(self._model, "disable_adapter"):
            # Fallback: disable adapters (gives base model, only correct
            # before any SFT warm-start has been applied).
            with self._model.disable_adapter():
                ref_outputs = self._model(**full_inputs)
        else:
            # No reference available; use current model (KL = 0)
            ref_outputs = self._model(**full_inputs)

        ref_logits = ref_outputs.logits
        ref_action_logits = ref_logits[
            :, prompt_len - 1 : prompt_len - 1 + action_len, :
        ]
        ref_log_probs = torch.nn.functional.log_softmax(ref_action_logits, dim=-1)
        ref_token_log_probs = ref_log_probs.gather(
            2, action_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        return ref_token_log_probs.sum()

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
            # Fallback: save only LoRA adapter weights (not the full model)
            import torch

            lora_state = {
                k: v for k, v in self._model.state_dict().items() if "lora" in k.lower()
            }
            torch.save(lora_state, str(checkpoint_dir / "lora_weights.pt"))
            logger.info("Saved %d LoRA tensors to %s", len(lora_state), checkpoint_dir)

        # Copy training log alongside checkpoint
        import shutil

        if self._log_path.exists():
            shutil.copy2(self._log_path, checkpoint_dir / "grpo_training_log.json")

        return str(checkpoint_dir)

    def _write_log(self) -> None:
        """Write training log to disk."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text(json.dumps(self._log_entries, indent=2))


def _format_action_as_text(
    action: Any,
    screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
) -> str:
    """Convert a BenchmarkAction back to DSL text for log-prob computation.

    Reconstructs the action DSL string that the VLM would have generated.
    This is used by _compute_rollout_loss to compute log-probabilities of
    the actions taken during rollout collection.

    Args:
        action: BenchmarkAction (or compatible dataclass) with type, x, y,
            text, key fields.
        screen_size: (width, height) used to convert absolute pixel
            coordinates back to normalized fractions (0.0-1.0).

    Returns:
        DSL text string, e.g. ``CLICK(x=0.50, y=0.25)`` or ``TYPE(text="hello")``.
    """
    action_type = getattr(action, "type", "done")
    width, height = screen_size

    if action_type == "click":
        x_px = getattr(action, "x", 0) or 0
        y_px = getattr(action, "y", 0) or 0
        x_frac = x_px / width if width > 0 else 0.0
        y_frac = y_px / height if height > 0 else 0.0
        return f"CLICK(x={x_frac:.2f}, y={y_frac:.2f})"

    if action_type == "type":
        text = getattr(action, "text", "") or ""
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'TYPE(text="{escaped}")'

    if action_type == "wait":
        return "WAIT()"

    # Default: DONE
    return "DONE()"


def _parse_vlm_output_to_action(
    text: str,
    screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
) -> Any:
    """Parse VLM output text into a BenchmarkAction.

    Supports the coordinate-based DSL:
        CLICK(x=0.XX, y=0.XX)
        TYPE(text="...")
        WAIT()
        DONE()

    Args:
        text: Raw text output from the VLM.
        screen_size: (width, height) for converting normalized fractions
            to absolute pixels.

    Returns:
        BenchmarkAction instance.
    """
    import re

    try:
        from openadapt_evals.adapters.base import BenchmarkAction
    except ImportError:
        # Fallback when openadapt-evals is not installed (e.g. in tests)
        from dataclasses import dataclass as _dc

        @_dc
        class BenchmarkAction:  # type: ignore[no-redef]
            type: str = "done"
            x: float | None = None
            y: float | None = None
            text: str | None = None
            key: str | None = None

    text = text.strip()
    width, height = screen_size

    # CLICK(x=..., y=...) — M-07: case-insensitive
    m = re.search(r"CLICK\(x=(-?[\d.]+),\s*y=(-?[\d.]+)\)", text, re.IGNORECASE)
    if m:
        x_frac = max(0.0, min(1.0, float(m.group(1))))
        y_frac = max(0.0, min(1.0, float(m.group(2))))
        return BenchmarkAction(
            type="click",
            x=int(x_frac * width),
            y=int(y_frac * height),
        )

    # TYPE(text="...") or TYPE(text='...') — M-07: case-insensitive
    m = re.search(
        r"""TYPE\(text=["']([^"'\\]*(?:\\.[^"'\\]*)*)["']\)""",
        text,
        re.IGNORECASE,
    )
    if m:
        # I-04: Unescape backslash first, then quotes
        typed_text = (
            m.group(1).replace("\\\\", "\\").replace('\\"', '"').replace("\\'", "'")
        )
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
