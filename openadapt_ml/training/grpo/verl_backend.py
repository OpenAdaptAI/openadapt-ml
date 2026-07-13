"""verl-agent / VAGEN backend for GRPO training.

This module provides the integration point for training via verl-agent
(https://github.com/VAGEN), which offers:
    - GiGPO (Generalized Group Relative Policy Optimization)
    - Multi-GPU distributed training via veRL
    - Desktop environment integration via WAADesktopEnv

The actual training loop is managed by verl-agent's own training script,
not by our GRPOTrainer. This module builds the VAGEN-compatible config
from our GRPOConfig and documents how to run training.

Usage:
    To train with the verl backend, set backend="verl" in GRPOConfig.
    The train_with_verl() function will print instructions and raise
    NotImplementedError until full integration is wired up.

    For now, training with verl-agent should be done via:
        1. Generate a VAGEN config: train_with_verl(config)
        2. Run verl-agent's training script with that config

See also:
    - openadapt-evals/configs/train_waa_vagen.yaml
    - docs/verl_agent_decision.md (if available)
"""

from __future__ import annotations

import logging
from typing import Any

from openadapt_ml.training.grpo.config import GRPOConfig

logger = logging.getLogger(__name__)

def _load_waa_desktop_env() -> Any | None:
    """Lazily import the concrete WAADesktopEnv from openadapt-evals.

    Kept out of module scope so openadapt-ml has no module-level
    openadapt-evals import (ml stays a dependency leaf). Returns the class,
    or ``None`` if openadapt-evals is not installed.
    """
    try:
        from openadapt_evals.adapters.verl_env import WAADesktopEnv
    except ImportError:
        return None
    return WAADesktopEnv


def build_vagen_config(config: GRPOConfig) -> dict[str, Any]:
    """Build a VAGEN-compatible config dict from GRPOConfig.

    Maps our config fields to the structure expected by verl-agent's
    training script. This dict can be serialized to YAML for use with
    VAGEN's CLI.

    Args:
        config: Our GRPO training configuration.

    Returns:
        Dict matching VAGEN's expected config structure.
    """
    return {
        "model": {
            "name": config.model_name,
            "load_in_4bit": config.load_in_4bit,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        },
        "training": {
            "learning_rate": config.learning_rate,
            "num_training_steps": config.num_training_steps,
            "save_every_steps": config.save_every_steps,
            "output_dir": config.output_dir,
            "num_rollouts_per_step": config.num_rollouts_per_step,
            "temperature": config.temperature,
        },
        "environment": {
            "type": "waa_desktop",
            "server_url": config.server_url,
            "task_ids": config.task_ids,
            "max_steps_per_episode": config.max_steps_per_episode,
            "screen_size": list(config.screen_size),
            "stuck_window": config.stuck_window,
        },
    }


def train_with_verl(config: GRPOConfig) -> None:
    """Entry point for verl-agent backend training.

    Currently a placeholder that documents the integration point.
    The actual training happens via verl-agent's own CLI/training script,
    not through this function.

    Args:
        config: GRPO training configuration with backend="verl".

    Raises:
        NotImplementedError: Always, until full verl-agent integration
            is wired up. The error message includes instructions for
            running training via verl-agent directly.
    """
    vagen_config = build_vagen_config(config)

    if _load_waa_desktop_env() is not None:
        logger.info(
            "WAADesktopEnv is available. verl-agent can use it for "
            "desktop environment interaction."
        )
    else:
        logger.warning(
            "WAADesktopEnv not found. Install openadapt-evals to enable "
            "desktop environment support: uv add openadapt-evals"
        )

    logger.info("VAGEN config built from GRPOConfig:")
    logger.info("  Model: %s", vagen_config["model"]["name"])
    logger.info("  Tasks: %s", vagen_config["environment"]["task_ids"])
    logger.info("  Steps: %d", vagen_config["training"]["num_training_steps"])
    logger.info("")
    logger.info(
        "To train with verl-agent, use the VAGEN training script with "
        "a config derived from the above. Example:"
    )
    logger.info("  python -m vagen.train --config configs/train_waa_vagen.yaml")

    raise NotImplementedError(
        "verl-agent training runs out-of-process via VAGEN's training script, "
        "not through this function. Use the E2E orchestration script:\n"
        "\n"
        "  python openadapt-evals/scripts/train_verl_e2e.py \\\n"
        "    --server-url http://localhost:5000 \\\n"
        "    --task-ids <TASK_ID> \\\n"
        "    --model Qwen/Qwen2.5-VL-7B-Instruct\n"
        "\n"
        "Or build a VAGEN config from GRPOConfig:\n"
        "  config_dict = build_vagen_config(config)\n"
        "\n"
        "See also:\n"
        "  - openadapt-evals/scripts/train_verl_e2e.py (573-line E2E script)\n"
        "  - openadapt-evals/configs/train_waa_vagen.yaml (Hydra config)\n"
        "  - openadapt-evals/scripts/setup_gpu_training.sh (GPU VM setup)\n"
        "  - docs/verl_agent_decision.md (architecture rationale)"
    )
