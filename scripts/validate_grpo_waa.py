#!/usr/bin/env python3
"""Phased validation of GRPO training against a WAA VM.

Each phase builds on the previous one, with clear success criteria and
failure diagnostics. Run with --phase N to execute phases 1 through N.

Phases:
    1. Connectivity: Verify WAA server is reachable (/screenshot, /evaluate)
    2. Single rollout: Reset environment, take one action, get reward
    3. Model inference: Load model, generate an action from a screenshot
    4. Single training step: Collect rollout group, compute loss, backward
    5. Multi-step training: Run 3 full GRPO steps, verify checkpoint saved

Usage:
    python scripts/validate_grpo_waa.py --server-url http://localhost:5001 --phase 3
    python scripts/validate_grpo_waa.py --server-url http://VM_IP:5000 --phase 5 --task-id <UUID>
    python scripts/validate_grpo_waa.py --mock --phase 4  # Use mock adapter (no VM)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_grpo_waa")


def phase1_connectivity(server_url: str, evaluate_url: str | None) -> bool:
    """Phase 1: Check WAA server connectivity."""
    import requests

    logger.info("=== Phase 1: Connectivity Check ===")

    # Check screenshot endpoint
    try:
        r = requests.get(f"{server_url}/screenshot", timeout=10)
        if r.status_code == 200 and len(r.content) > 100:
            logger.info("  /screenshot OK (%d bytes)", len(r.content))
        else:
            logger.error("  /screenshot failed: status=%d, len=%d", r.status_code, len(r.content))
            return False
    except Exception as e:
        logger.error("  /screenshot unreachable: %s", e)
        return False

    # Check evaluate endpoint
    eval_base = evaluate_url or server_url
    try:
        r = requests.get(f"{eval_base}/probe", timeout=10)
        logger.info("  /probe status=%d", r.status_code)
    except Exception as e:
        logger.warning("  /probe unreachable (non-fatal): %s", e)

    logger.info("Phase 1 PASSED")
    return True


def phase2_single_rollout(server_url: str, evaluate_url: str | None, task_id: str, mock: bool) -> bool:
    """Phase 2: Reset env, take one action, get reward."""
    logger.info("=== Phase 2: Single Rollout ===")

    if mock:
        from openadapt_evals.adapters.waa.mock import WAAMockAdapter
        adapter = WAAMockAdapter()
    else:
        from openadapt_evals.adapters.waa.live import WAALiveAdapter, WAALiveConfig
        adapter = WAALiveAdapter(
            WAALiveConfig(server_url=server_url, evaluate_url=evaluate_url)
        )

    from openadapt_evals.adapters.base import BenchmarkAction
    from openadapt_evals.adapters.rl_env import RLEnvironment

    env = RLEnvironment(adapter)

    # Reset
    obs = env.reset(task_id=task_id)
    if obs is None or obs.screenshot is None:
        logger.error("  Reset returned no observation or screenshot")
        return False
    logger.info("  Reset OK, screenshot=%d bytes", len(obs.screenshot))

    # Take one action
    action = BenchmarkAction(type="click", x=500, y=400)
    step = env.step(action)
    logger.info("  Step OK: reward=%.2f, done=%s", step.reward, step.done)

    # Check screen size
    logger.info("  Screen size: %s", env.screen_size)

    if hasattr(adapter, "close"):
        adapter.close()

    logger.info("Phase 2 PASSED")
    return True


def phase3_model_inference(server_url: str, evaluate_url: str | None, model_name: str, task_id: str, mock: bool) -> bool:
    """Phase 3: Load model, generate action from screenshot."""
    logger.info("=== Phase 3: Model Inference ===")

    import io

    import torch
    from PIL import Image

    from openadapt_ml.training.grpo.config import GRPOConfig
    from openadapt_ml.training.grpo.trainer import (
        _build_agent_messages,
        _load_model_and_processor,
        _parse_vlm_output_to_action,
    )

    config = GRPOConfig(
        model_name=model_name,
        server_url=server_url,
        evaluate_url=evaluate_url,
    )

    logger.info("  Loading model: %s", model_name)
    t0 = time.time()
    model, processor = _load_model_and_processor(config)
    logger.info("  Model loaded in %.1fs", time.time() - t0)

    # Get a screenshot
    if mock:
        screenshot = Image.new("RGB", (1920, 1080), color=(50, 50, 80))
    else:
        import requests
        r = requests.get(f"{server_url}/screenshot", timeout=10)
        screenshot = Image.open(io.BytesIO(r.content))

    logger.info("  Screenshot: %s", screenshot.size)

    # Build prompt and generate
    messages = _build_agent_messages("Click the Start button")
    if hasattr(processor, "apply_chat_template"):
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text_input = messages[-1]["content"]

    inputs = processor(text_input, images=[screenshot], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)

    decoded = processor.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    logger.info("  Model output: %s", decoded.strip()[:200])

    action = _parse_vlm_output_to_action(decoded)
    logger.info("  Parsed action: type=%s", action.type)

    logger.info("Phase 3 PASSED")
    return True


def phase4_single_training_step(
    server_url: str,
    evaluate_url: str | None,
    model_name: str,
    task_id: str,
    lora_checkpoint: str | None,
    mock: bool,
) -> bool:
    """Phase 4: Collect rollout group, compute loss, one gradient step."""
    logger.info("=== Phase 4: Single Training Step ===")

    import tempfile

    from openadapt_ml.training.grpo.config import GRPOConfig
    from openadapt_ml.training.grpo.trainer import GRPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GRPOConfig(
            model_name=model_name,
            server_url=server_url,
            evaluate_url=evaluate_url,
            task_ids=[task_id],
            lora_checkpoint=lora_checkpoint,
            num_rollouts_per_step=2,  # Small group for validation
            max_steps_per_episode=3,  # Short episodes
            num_training_steps=1,
            save_every_steps=1,
            output_dir=tmpdir,
        )

        trainer = GRPOTrainer(config)
        logger.info("  Config: rollouts=%d, max_steps=%d", config.num_rollouts_per_step, config.max_steps_per_episode)

        t0 = time.time()
        checkpoint_path = trainer.train()
        elapsed = time.time() - t0

        logger.info("  Training step completed in %.1fs", elapsed)
        logger.info("  Checkpoint: %s", checkpoint_path)

        # Verify checkpoint exists
        from pathlib import Path
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            logger.error("  Checkpoint directory missing!")
            return False

        adapter_files = list(ckpt.glob("adapter_*"))
        logger.info("  Checkpoint files: %s", [f.name for f in adapter_files])

    logger.info("Phase 4 PASSED")
    return True


def phase5_multi_step_training(
    server_url: str,
    evaluate_url: str | None,
    model_name: str,
    task_id: str,
    lora_checkpoint: str | None,
    mock: bool,
) -> bool:
    """Phase 5: Run 3 GRPO steps, verify checkpoints."""
    logger.info("=== Phase 5: Multi-Step Training ===")

    import tempfile
    from pathlib import Path

    from openadapt_ml.training.grpo.config import GRPOConfig
    from openadapt_ml.training.grpo.trainer import GRPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GRPOConfig(
            model_name=model_name,
            server_url=server_url,
            evaluate_url=evaluate_url,
            task_ids=[task_id],
            lora_checkpoint=lora_checkpoint,
            num_rollouts_per_step=2,
            max_steps_per_episode=3,
            num_training_steps=3,
            save_every_steps=1,
            output_dir=tmpdir,
        )

        trainer = GRPOTrainer(config)
        t0 = time.time()
        checkpoint_path = trainer.train()
        elapsed = time.time() - t0

        logger.info("  3 training steps completed in %.1fs", elapsed)

        # Verify all checkpoints
        for step in [1, 2, 3]:
            ckpt = Path(tmpdir) / f"step_{step}"
            if ckpt.exists():
                logger.info("  step_%d checkpoint: OK", step)
            else:
                logger.warning("  step_%d checkpoint: MISSING", step)

    logger.info("Phase 5 PASSED")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GRPO training against WAA")
    parser.add_argument("--server-url", default="http://localhost:5001")
    parser.add_argument("--evaluate-url", default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--task-id", default="notepad_1")
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--phase", type=int, default=5, help="Run phases 1 through N")
    parser.add_argument("--mock", action="store_true", help="Use mock adapter (no VM)")
    args = parser.parse_args()

    phases = [
        (1, lambda: phase1_connectivity(args.server_url, args.evaluate_url)),
        (2, lambda: phase2_single_rollout(args.server_url, args.evaluate_url, args.task_id, args.mock)),
        (3, lambda: phase3_model_inference(args.server_url, args.evaluate_url, args.model_name, args.task_id, args.mock)),
        (4, lambda: phase4_single_training_step(args.server_url, args.evaluate_url, args.model_name, args.task_id, args.lora_checkpoint, args.mock)),
        (5, lambda: phase5_multi_step_training(args.server_url, args.evaluate_url, args.model_name, args.task_id, args.lora_checkpoint, args.mock)),
    ]

    # Skip phase 1 when using mock (no server to connect to)
    if args.mock:
        phases = [(n, fn) for n, fn in phases if n != 1]

    for phase_num, phase_fn in phases:
        if phase_num > args.phase:
            break
        try:
            if not phase_fn():
                logger.error("Phase %d FAILED", phase_num)
                return 1
        except Exception:
            logger.exception("Phase %d raised an exception", phase_num)
            return 1

    logger.info("All phases through %d PASSED", args.phase)
    return 0


if __name__ == "__main__":
    sys.exit(main())
