"""Minimal GRPO trainer bridging TRL/HuggingFace and openadapt-evals RLEnvironment.

Note: This is the "standalone" backend. For the verl-agent backend (recommended
for production training with GiGPO and multi-GPU support), see verl_backend.py
or use the VAGEN training config in openadapt-evals/configs/train_waa_vagen.yaml.

Uses REINFORCE with group-relative advantages (equivalent to single-epoch GRPO).
The policy_gradient_loss function includes PPO-style clipping for future multi-epoch
support, but with the current single-epoch design (old_logps == current_logps),
the ratio is always 1.0 and clipping does not activate.

Why not TRL's GRPOTrainer directly: TRL v0.29.0 recomputes log-probs during
training via a forward pass that needs image pixel_values, but only stores token
IDs from rollout_func. Until TRL resolves this for VLM multi-turn trajectories,
we use standalone loss math while reusing HF/PEFT for everything else.

When to switch to full TRL GRPOTrainer:
    - TRL supports passing pixel_values through rollout_func -> training
    - Or TRL supports environment_factory with VLM image inputs
    Track: https://github.com/huggingface/trl/issues (VLM multi-turn GRPO)

Key design decisions:
    - beta=0.0 (no KL penalty) per DAPO/Open-Reasoner-Zero. Simpler, saves
      memory (no reference model needed).
    - Per-step backward to avoid OOM on long trajectories.
    - Standard HF model loading: AutoModelForImageTextToText + AutoProcessor + PEFT.
    - Standard PEFT checkpointing: model.save_pretrained().
"""

from __future__ import annotations

import io
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from PIL import Image

from openadapt_ml.datasets.next_action import SYSTEM_PROMPT
from openadapt_ml.training.grpo.config import GRPOConfig
from openadapt_ml.training.grpo.reward import (
    compute_group_advantages,
    evaluate_milestones_screenshot,
)
from openadapt_ml.training.grpo.rollout_collector import (
    GRPORolloutCollector,
    Rollout,
)

# Optional import for TaskConfig (openadapt-evals may not be installed)
try:
    from openadapt_evals.task_config import TaskConfig

    _HAS_TASK_CONFIG = True
except ImportError:
    TaskConfig = None  # type: ignore[assignment, misc]
    _HAS_TASK_CONFIG = False

logger = logging.getLogger(__name__)

DEFAULT_SCREEN_SIZE: tuple[int, int] = (1920, 1080)


# ---------------------------------------------------------------------------
# Policy gradient loss
# ---------------------------------------------------------------------------


def policy_gradient_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Policy gradient loss with optional PPO-style clipping.

    When old_logps == current_logps (single-epoch, on-policy), this reduces
    to REINFORCE: loss = -advantage * log_prob. The clipping only activates
    when old_logps differ from current_logps (multi-epoch off-policy updates).

    Args:
        current_logps: Log-probs under current policy.
        old_logps: Log-probs under rollout policy (detached). Pass
            current_logps.detach() for single-epoch (REINFORCE).
        advantages: Group-relative advantages.
        epsilon: Clipping range for importance-sampling ratio.

    Returns:
        Scalar loss (to minimize).
    """
    ratio = torch.exp(current_logps - old_logps)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


# Backwards-compatible alias
grpo_loss = policy_gradient_loss


# ---------------------------------------------------------------------------
# Prompt construction (single source of truth)
# ---------------------------------------------------------------------------


def _build_agent_messages(
    instruction: str, *, include_image: bool = False
) -> list[dict]:
    """Build chat messages for the GRPO agent.

    Uses the same SYSTEM_PROMPT as SFT training so GRPO operates in
    the same prompt distribution the model was warm-started on.

    This is the **single source of truth** for prompt construction
    during both rollout collection and loss computation.

    Args:
        instruction: Task instruction text.
        include_image: If True, include an image placeholder in the user
            message so ``apply_chat_template`` inserts ``<|image_pad|>``
            tokens required by Qwen2.5-VL and similar VLMs.
    """
    text_content = (
        f"Goal: {instruction}\n\n"
        "Look at the screenshot and determine the NEXT action.\n\n"
        'Action: [CLICK(x=..., y=...) or TYPE(text="...") or WAIT() or DONE()]'
    )
    if include_image:
        user_content = [
            {"type": "image"},
            {"type": "text", "text": text_content},
        ]
    else:
        user_content = text_content
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# DSL parsing / formatting
# ---------------------------------------------------------------------------


def _parse_vlm_output_to_action(
    text: str,
    screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
) -> Any:
    """Parse VLM output text into a BenchmarkAction.

    Supports: CLICK(x=0.XX, y=0.XX), TYPE(text="..."), WAIT(), DONE().
    """
    try:
        from openadapt_evals.adapters.base import BenchmarkAction
    except ImportError:
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

    # CLICK(x=..., y=...)
    m = re.search(r"CLICK\(x=(-?[\d.]+),\s*y=(-?[\d.]+)\)", text, re.IGNORECASE)
    if m:
        x_frac = max(0.0, min(1.0, float(m.group(1))))
        y_frac = max(0.0, min(1.0, float(m.group(2))))
        return BenchmarkAction(
            type="click", x=int(x_frac * width), y=int(y_frac * height)
        )

    # TYPE(text="..." or '...')
    m = re.search(
        r"""TYPE\(text=["']([^"'\\]*(?:\\.[^"'\\]*)*)["']\)""", text, re.IGNORECASE
    )
    if m:
        typed_text = (
            m.group(1).replace("\\\\", "\\").replace('\\"', '"').replace("\\'", "'")
        )
        return BenchmarkAction(type="type", text=typed_text)

    if re.search(r"\bWAIT\s*\(\s*\)", text, re.IGNORECASE):
        return BenchmarkAction(type="wait")

    if re.search(r"\bDONE\s*\(\s*\)", text, re.IGNORECASE):
        return BenchmarkAction(type="done")

    logger.warning("Could not parse VLM output: %s. Defaulting to DONE.", text)
    return BenchmarkAction(type="done")


def _format_action_as_text(
    action: Any,
    screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
) -> str:
    """Convert a BenchmarkAction back to DSL text for log-prob computation."""
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

    return "DONE()"


# Public aliases (the underscore-prefixed versions are kept for backwards compat)
parse_vlm_output_to_action = _parse_vlm_output_to_action
format_action_as_text = _format_action_as_text


# ---------------------------------------------------------------------------
# Model loading (standard HuggingFace + PEFT)
# ---------------------------------------------------------------------------


def _load_model_and_processor(config: GRPOConfig) -> tuple[Any, Any]:
    """Load a VLM with LoRA using standard HuggingFace + PEFT.

    If config.lora_checkpoint is set, loads an existing LoRA adapter via
    PeftModel.from_pretrained() instead of creating a fresh one. This
    enables GRPO fine-tuning on top of an SFT-trained LoRA.

    Returns:
        (model, processor) tuple. Model has LoRA adapters attached.
    """
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    processor = AutoProcessor.from_pretrained(config.model_name)

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoVLM.from_pretrained(config.model_name, **load_kwargs)

    if config.lora_checkpoint:
        logger.info("Loading existing LoRA from %s", config.lora_checkpoint)
        model = PeftModel.from_pretrained(
            model,
            config.lora_checkpoint,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# GRPOTrainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """Minimal GRPO trainer: collect rollouts, compute advantages, train.

    Uses openadapt-evals RLEnvironment for rollout collection and
    standalone GRPO math for the policy gradient update.
    """

    def __init__(self, config: GRPOConfig) -> None:
        self._config = config
        self._model: Any = None
        self._processor: Any = None
        self._optimizer: Any = None
        self._collector: GRPORolloutCollector | None = None
        self._step: int = 0
        self._task_configs: dict[str, Any] = {}

        # Load task configs from --task-dir if specified
        if config.task_dir:
            self._load_task_configs(config.task_dir)

    def _load_task_configs(self, task_dir: str) -> None:
        """Load TaskConfig YAMLs from a directory.

        Populates ``self._task_configs`` (keyed by task ID) and auto-fills
        ``config.task_ids`` if it was left empty.

        Args:
            task_dir: Path to directory containing YAML/JSON task configs.

        Raises:
            ImportError: If openadapt-evals is not installed.
            FileNotFoundError: If the directory does not exist.
        """
        if not _HAS_TASK_CONFIG:
            raise ImportError(
                "openadapt-evals is required for --task-dir support. "
                "Install with: pip install openadapt-evals"
            )

        task_dir_path = Path(task_dir)
        if not task_dir_path.is_dir():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        configs = TaskConfig.from_dir(str(task_dir_path))
        if not configs:
            raise ValueError(f"No task configs found in {task_dir}")

        for tc in configs:
            self._task_configs[tc.id] = tc
            logger.info(
                "Loaded task config: %s (%s) — %d milestones",
                tc.id,
                tc.name[:50],
                len(tc.milestones),
            )

        # Auto-populate task_ids if empty
        if not self._config.task_ids:
            self._config.task_ids = list(self._task_configs.keys())
            logger.info(
                "Auto-populated task_ids from task_dir: %s",
                self._config.task_ids,
            )

    def _compute_milestone_reward(
        self,
        task_id: str,
        screenshot_bytes: bytes,
    ) -> float:
        """Compute milestone-based reward for a task using VLM judge.

        Evaluates screenshot-type milestones locally without needing the
        WAA /evaluate endpoint.  Falls back to 0.0 if the task has no
        milestones or the task_id is not found in loaded configs.

        Args:
            task_id: The task ID to look up in loaded configs.
            screenshot_bytes: PNG screenshot bytes to evaluate.

        Returns:
            Fraction of screenshot milestones passed (0.0 to 1.0).
        """
        task_config = self._task_configs.get(task_id)
        if task_config is None:
            return 0.0
        return evaluate_milestones_screenshot(task_config, screenshot_bytes)

    def _compute_milestone_reward_from_rollout(
        self,
        rollout: Rollout,
    ) -> float | None:
        """Extract the last screenshot from a rollout and compute milestone reward.

        Returns None if no task config or no screenshot is available,
        signalling the caller to keep the existing reward.
        """
        task_config = self._task_configs.get(rollout.task_id)
        if task_config is None or not getattr(task_config, "milestones", None):
            return None

        # Find the last step with a screenshot
        screenshot_bytes: bytes | None = None
        for step in reversed(rollout.steps):
            obs = getattr(step, "observation", None)
            if obs is not None:
                ss = getattr(obs, "screenshot", None)
                if ss:
                    screenshot_bytes = ss
                    break

        if not screenshot_bytes:
            return None

        return evaluate_milestones_screenshot(task_config, screenshot_bytes)

    def _make_agent_fn(self) -> Callable:
        """Create agent closure: observation -> BenchmarkAction.

        Captures model/processor by reference so weight updates during
        training are reflected in subsequent rollouts.
        """
        from openadapt_evals.adapters.base import BenchmarkAction

        model = self._model
        processor = self._processor
        temperature = self._config.temperature
        collector = self._collector
        config_screen_size = self._config.screen_size

        def agent_fn(obs: Any) -> BenchmarkAction:
            if obs.screenshot:
                image = Image.open(io.BytesIO(obs.screenshot)).convert("RGB")
            else:
                image = Image.new("RGB", (1, 1))

            # Get instruction from environment task
            instruction = ""
            if collector and hasattr(collector, "env"):
                task = getattr(collector.env, "_current_task", None)
                if task is not None:
                    instruction = getattr(task, "instruction", "") or ""
            if not instruction:
                raw_obs = getattr(obs, "raw_observation", None)
                if raw_obs and isinstance(raw_obs, dict):
                    instruction = raw_obs.get("instruction", "")

            messages = _build_agent_messages(instruction, include_image=True)

            if hasattr(processor, "apply_chat_template"):
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text_input = messages[-1]["content"]

            inputs = processor(text=[text_input], images=[image], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,
                )

            decoded = processor.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            screen_size = config_screen_size
            if collector and hasattr(collector, "env"):
                try:
                    screen_size = collector.env.screen_size
                except Exception:
                    pass

            action = _parse_vlm_output_to_action(decoded, screen_size=screen_size)

            # Store raw VLM output for accurate loss computation
            try:
                action._grpo_raw_text = decoded
            except AttributeError:
                pass
            return action

        return agent_fn

    def train(self) -> str:
        """Run GRPO training loop. Returns path to final checkpoint."""
        if not self._config.task_ids:
            raise ValueError(
                "config.task_ids must be non-empty. Provide at least one "
                "WAA task ID to train on, or use --task-dir to load from "
                "YAML files."
            )

        logger.info("Starting GRPO training")
        logger.info("  Model: %s", self._config.model_name)
        logger.info("  Tasks: %s", self._config.task_ids)
        logger.info("  Task dir: %s", self._config.task_dir or "(none)")
        logger.info("  Task configs loaded: %d", len(self._task_configs))
        logger.info("  Rollouts/step: %d", self._config.num_rollouts_per_step)
        logger.info("  Training steps: %d", self._config.num_training_steps)

        # Setup
        self._model, self._processor = _load_model_and_processor(self._config)
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(trainable, lr=self._config.learning_rate)
        self._collector = GRPORolloutCollector(
            self._config,
            task_configs=self._task_configs if self._task_configs else None,
        )

        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)
        agent_fn = self._make_agent_fn()
        start_time = time.time()

        for step in range(self._config.num_training_steps):
            self._step = step
            step_start = time.time()
            task_id = self._config.task_ids[step % len(self._config.task_ids)]

            # Collect rollouts (inference)
            self._model.eval()
            rollouts = self._collector.collect_group(agent_fn=agent_fn, task_id=task_id)

            # If task configs with milestones are loaded, override the
            # binary rewards with milestone-based dense rewards.
            if self._task_configs:
                for rollout in rollouts:
                    milestone_reward = self._compute_milestone_reward_from_rollout(
                        rollout
                    )
                    if milestone_reward is not None:
                        rollout.reward = max(rollout.reward, milestone_reward)

            # Train (gradient update)
            self._model.train()
            metrics = self._training_step(rollouts)

            metrics.update(
                {
                    "step": step,
                    "task_id": task_id,
                    "elapsed": time.time() - start_time,
                    "step_time": time.time() - step_start,
                }
            )
            logger.info(
                "Step %d/%d: reward=%.2f loss=%.4f time=%.1fs",
                step + 1,
                self._config.num_training_steps,
                metrics.get("reward_mean", 0.0),
                metrics.get("loss", 0.0),
                metrics["step_time"],
            )

            if (step + 1) % self._config.save_every_steps == 0:
                self._save_checkpoint(step + 1)

        self._save_checkpoint(self._config.num_training_steps)

        if self._collector:
            self._collector.close()

        final_path = str(
            Path(self._config.output_dir) / f"step_{self._config.num_training_steps}"
        )
        logger.info("Training complete. Final checkpoint: %s", final_path)
        return final_path

    def _training_step(self, rollouts: list[Rollout]) -> dict[str, float]:
        """Single GRPO gradient step from a group of rollouts."""
        rewards = [r.reward for r in rollouts]
        advantages = compute_group_advantages(rewards)
        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0

        # Skip if no variance (no gradient signal)
        valid = [(r, a) for r, a in zip(rollouts, advantages) if abs(a) >= 1e-8]
        if not valid:
            return {"reward_mean": reward_mean, "loss": 0.0, "skipped": True}

        self._optimizer.zero_grad()
        total_loss = 0.0
        n = len(valid)

        for rollout, advantage in valid:
            loss_val = self._compute_rollout_loss(
                rollout, advantage, loss_scale=1.0 / n
            )
            total_loss += loss_val

        torch.nn.utils.clip_grad_norm_(
            [p for p in self._model.parameters() if p.requires_grad], max_norm=1.0
        )
        self._optimizer.step()

        return {
            "reward_mean": reward_mean,
            "loss": total_loss / max(n, 1),
            "skipped": False,
            "num_rollouts": len(rollouts),
            "num_gradient_terms": n,
        }

    def _compute_rollout_loss(
        self,
        rollout: Rollout,
        advantage: float,
        loss_scale: float = 1.0,
    ) -> float:
        """Compute GRPO loss for one rollout. Backward per-step to avoid OOM."""
        device = next(self._model.parameters()).device
        total_loss_value = 0.0

        screen_size = self._config.screen_size
        if self._collector and hasattr(self._collector, "env"):
            try:
                screen_size = self._collector.env.screen_size
            except Exception:
                pass

        instruction = getattr(rollout, "instruction", "") or ""

        # Gather valid steps
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

        if not valid_steps:
            return 0.0

        num_steps = len(valid_steps)

        for _obs, action, screenshot in valid_steps:
            try:
                image = Image.open(io.BytesIO(screenshot)).convert("RGB")
            except Exception:
                continue

            messages = _build_agent_messages(instruction, include_image=True)

            # Raw text from rollout or reconstruct from DSL
            raw_text = getattr(action, "_grpo_raw_text", None)
            action_text = (
                raw_text
                if raw_text
                else _format_action_as_text(action, screen_size=screen_size)
            )

            # Tokenize prompt with image
            if hasattr(self._processor, "apply_chat_template"):
                text_input = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text_input = messages[-1]["content"]

            prompt_inputs = self._processor(
                text=[text_input], images=[image], return_tensors="pt"
            )
            prompt_ids = prompt_inputs["input_ids"]
            prompt_len = prompt_ids.shape[1]

            # Tokenize action separately, concatenate
            inner_tok = getattr(self._processor, "tokenizer", self._processor)
            action_ids = inner_tok(
                action_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            action_len = action_ids.shape[1]
            if action_len <= 0:
                continue

            full_ids = torch.cat([prompt_ids, action_ids.to(prompt_ids.device)], dim=1)
            full_inputs = dict(prompt_inputs)
            full_inputs["input_ids"] = full_ids
            full_inputs["attention_mask"] = torch.ones_like(full_ids)
            full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

            action_token_ids = full_ids[:, prompt_len : prompt_len + action_len]

            # Forward pass -> log-probs for action tokens
            outputs = self._model(**full_inputs)
            logits = outputs.logits
            action_logits = logits[:, prompt_len - 1 : prompt_len - 1 + action_len, :]
            log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, action_token_ids.unsqueeze(-1).to(device)
            ).squeeze(-1)
            step_log_prob = token_log_probs.sum()

            # Single-epoch policy gradient: old_logps == current_logps,
            # so ratio=1.0 and this is REINFORCE with group-relative
            # advantages. Clipping activates if we add multi-epoch support.
            adv_tensor = torch.tensor(
                advantage, device=device, dtype=step_log_prob.dtype
            )
            step_loss = policy_gradient_loss(
                step_log_prob.unsqueeze(0),
                step_log_prob.detach().unsqueeze(0),
                adv_tensor.unsqueeze(0),
            )

            # Per-step backward to avoid OOM
            scaled = step_loss * loss_scale / num_steps
            scaled.backward()
            total_loss_value += step_loss.detach().item()

        return total_loss_value / max(num_steps, 1)

    def _save_checkpoint(self, step: int) -> str:
        """Save LoRA adapter via standard PEFT save_pretrained."""
        ckpt_dir = Path(self._config.output_dir) / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(ckpt_dir))
        logger.info("Saved checkpoint to %s", ckpt_dir)
        return str(ckpt_dir)

    def save_checkpoint(self, step: int) -> str:
        """Public interface for checkpoint saving."""
        return self._save_checkpoint(step)
