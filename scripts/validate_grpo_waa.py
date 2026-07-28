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
import json
import logging
import math
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_grpo_waa")


def _adapter_weight_issue(weight_path: Path) -> str | None:
    """Return why an adapter weight file is not a readable tensor state."""
    try:
        import torch

        def tensor_issue(tensor: torch.Tensor) -> str | None:
            if tensor.layout != torch.strided:
                return "adapter weights contain an unsupported tensor layout"
            if tensor.numel() == 0:
                return "adapter weights contain an empty tensor"
            if (
                tensor.is_floating_point() or tensor.is_complex()
            ) and not torch.isfinite(tensor).all():
                return "adapter weights contain a non-finite tensor"
            return None

        if weight_path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(weight_path), framework="pt", device="cpu") as tensors:
                keys = list(tensors.keys())
                if not keys:
                    return "adapter weights contain no tensors"
                for key in keys:
                    if (issue := tensor_issue(tensors.get_tensor(key))) is not None:
                        return issue
            return None

        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            return "adapter weights must contain a non-empty tensor state"
        if any(not isinstance(value, torch.Tensor) for value in state.values()):
            return "adapter weights contain a non-tensor value"
        for value in state.values():
            if (issue := tensor_issue(value)) is not None:
                return issue
    except Exception as exc:
        return f"adapter weights are unreadable: {type(exc).__name__}"
    return None


def _checkpoint_issue(checkpoint_dir: Path) -> str | None:
    """Return why a PEFT adapter checkpoint cannot be loaded."""
    if not checkpoint_dir.is_dir():
        return "checkpoint directory is missing"

    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        return "adapter_config.json is missing or empty"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "adapter_config.json is not valid JSON"
    if not isinstance(config, dict) or not config:
        return "adapter_config.json must contain a non-empty JSON object"
    try:
        from peft import PeftConfig

        PeftConfig.from_pretrained(str(checkpoint_dir))
    except Exception as exc:
        return f"adapter_config.json is not valid PEFT metadata: {type(exc).__name__}"

    weight_paths = [
        checkpoint_dir / "adapter_model.safetensors",
        checkpoint_dir / "adapter_model.bin",
    ]
    present_weights = [
        path for path in weight_paths if path.is_file() and path.stat().st_size > 0
    ]
    if not present_weights:
        return "adapter weights are missing or empty"
    for weight_path in present_weights:
        if (issue := _adapter_weight_issue(weight_path)) is not None:
            return issue
    return None


def _missing_checkpoints(output_dir: str, steps: list[int]) -> list[int]:
    """Return training steps whose adapter checkpoint is incomplete."""

    root = Path(output_dir)
    return [
        step for step in steps if _checkpoint_issue(root / f"step_{step}") is not None
    ]


def _evaluator_candidates(server_url: str, evaluate_url: str | None) -> list[str]:
    """Return evaluator URL candidates in deployment-preference order."""
    if evaluate_url:
        return [evaluate_url.rstrip("/")]

    parsed = urlsplit(server_url)
    if not parsed.scheme or not parsed.hostname:
        return [server_url.rstrip("/")]
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    candidates = [
        f"{parsed.scheme}://{host}:5050",
        f"{parsed.scheme}://{host}:5051",
        server_url.rstrip("/"),
    ]
    return list(dict.fromkeys(candidates))


def _detect_evaluator_url(
    server_url: str,
    evaluate_url: str | None,
    *,
    timeout: float = 10,
) -> str | None:
    """Find the endpoint that identifies itself as the evaluator service."""
    import requests

    for candidate in _evaluator_candidates(server_url, evaluate_url):
        try:
            response = requests.get(f"{candidate}/probe", timeout=timeout)
            payload = response.json() if response.status_code == 200 else None
        except Exception:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("status") == "ok"
            and payload.get("service") == "evaluate_server"
        ):
            return candidate
    return None


def _has_measured_training_update(
    trainer: object, expected_steps: set[int] | None = None
) -> bool:
    """Return whether each requested step has one measured optimizer update."""
    measured_steps: set[int] = set()
    found_update = False
    for metrics in getattr(trainer, "training_metrics", ()):
        if not isinstance(metrics, dict) or metrics.get("skipped") is not False:
            continue
        numeric_fields = ("reward_mean", "loss", "gradient_norm")
        if any(
            isinstance(metrics.get(field), bool)
            or not isinstance(metrics.get(field), (int, float))
            or not math.isfinite(metrics[field])
            for field in numeric_fields
        ):
            continue
        gradient_terms = metrics.get("num_gradient_terms")
        if (
            isinstance(gradient_terms, bool)
            or not isinstance(gradient_terms, int)
            or gradient_terms <= 0
            or metrics["gradient_norm"] <= 0.0
            or metrics.get("optimizer_step_applied") is not True
        ):
            continue
        found_update = True
        step = metrics.get("step")
        if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
            measured_steps.add(step)
    if expected_steps is None:
        return found_update
    return measured_steps == expected_steps


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
            logger.error(
                "  /screenshot failed: status=%d, len=%d", r.status_code, len(r.content)
            )
            return False
    except Exception as e:
        logger.error("  /screenshot unreachable: %s", e)
        return False

    evaluator_url = _detect_evaluator_url(server_url, evaluate_url)
    if evaluator_url is None:
        logger.error(
            "  no evaluator returned the canonical evaluate_server probe response"
        )
        return False
    logger.info("  evaluator ready at %s", evaluator_url)

    logger.info("Phase 1 PASSED")
    return True


def phase2_single_rollout(
    server_url: str, evaluate_url: str | None, task_id: str, mock: bool
) -> bool:
    """Phase 2: Reset env, take one action, get reward."""
    logger.info("=== Phase 2: Single Rollout ===")

    if mock:
        from openadapt_evals.adapters.waa.mock import WAAMockAdapter

        adapter = WAAMockAdapter()
    else:
        from openadapt_evals.adapters.waa.live import WAALiveAdapter, WAALiveConfig

        resolved_evaluate_url = _detect_evaluator_url(server_url, evaluate_url)
        if resolved_evaluate_url is None:
            logger.error("  No canonical evaluator endpoint is available")
            return False
        adapter = WAALiveAdapter(
            WAALiveConfig(
                server_url=server_url,
                evaluate_url=resolved_evaluate_url,
            )
        )

    from openadapt_evals.adapters.base import BenchmarkAction
    from openadapt_evals.adapters.rl_env import ResetConfig, RLEnvironment

    env = RLEnvironment(adapter)

    # Reset
    obs = env.reset(config=ResetConfig(task_id=task_id))
    if obs is None or obs.screenshot is None:
        logger.error("  Reset returned no observation or screenshot")
        return False
    logger.info("  Reset OK, screenshot=%d bytes", len(obs.screenshot))

    # Advance through a no-op so validation cannot click an unknown live target.
    action = BenchmarkAction(type="wait")
    step = env.step(action)
    logger.info("  Step OK: reward=%.2f, done=%s", step.reward, step.done)

    try:
        measured_score = env.evaluate()
    except Exception as exc:
        logger.error("  Evaluator failed: %s", exc)
        if hasattr(adapter, "close"):
            adapter.close()
        return False
    if (
        isinstance(measured_score, bool)
        or not isinstance(measured_score, (int, float))
        or not math.isfinite(measured_score)
        or not 0.0 <= measured_score <= 1.0
    ):
        logger.error("  Evaluator returned invalid score: %r", measured_score)
        if hasattr(adapter, "close"):
            adapter.close()
        return False
    logger.info("  Evaluator measured score=%.2f", measured_score)

    # Check screen size
    logger.info("  Screen size: %s", env.screen_size)

    if hasattr(adapter, "close"):
        adapter.close()

    logger.info("Phase 2 PASSED")
    return True


def phase3_model_inference(
    server_url: str, evaluate_url: str | None, model_name: str, task_id: str, mock: bool
) -> bool:
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
        outputs = model.generate(
            **inputs, max_new_tokens=100, temperature=0.7, do_sample=True
        )

    decoded = processor.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
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

    resolved_evaluate_url = evaluate_url
    if not mock:
        resolved_evaluate_url = _detect_evaluator_url(server_url, evaluate_url)
        if resolved_evaluate_url is None:
            logger.error("  No canonical evaluator endpoint is available")
            return False

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GRPOConfig(
            model_name=model_name,
            server_url=server_url,
            evaluate_url=resolved_evaluate_url,
            task_ids=[task_id],
            lora_checkpoint=lora_checkpoint,
            num_rollouts_per_step=2,  # Small group for validation
            max_steps_per_episode=3,  # Short episodes
            num_training_steps=1,
            save_every_steps=1,
            output_dir=tmpdir,
        )

        trainer = GRPOTrainer(config)
        logger.info(
            "  Config: rollouts=%d, max_steps=%d",
            config.num_rollouts_per_step,
            config.max_steps_per_episode,
        )

        t0 = time.time()
        checkpoint_path = trainer.train()
        elapsed = time.time() - t0

        logger.info("  Training step completed in %.1fs", elapsed)
        logger.info("  Checkpoint: %s", checkpoint_path)

        if not _has_measured_training_update(trainer, expected_steps={0}):
            logger.error("  Training produced no finite non-skipped optimizer update")
            return False

        ckpt = Path(checkpoint_path)
        issue = _checkpoint_issue(ckpt)
        if issue is not None:
            logger.error("  Checkpoint is incomplete: %s", issue)
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

    from openadapt_ml.training.grpo.config import GRPOConfig
    from openadapt_ml.training.grpo.trainer import GRPOTrainer

    resolved_evaluate_url = evaluate_url
    if not mock:
        resolved_evaluate_url = _detect_evaluator_url(server_url, evaluate_url)
        if resolved_evaluate_url is None:
            logger.error("  No canonical evaluator endpoint is available")
            return False

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GRPOConfig(
            model_name=model_name,
            server_url=server_url,
            evaluate_url=resolved_evaluate_url,
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
        trainer.train()
        elapsed = time.time() - t0

        logger.info("  3 training steps completed in %.1fs", elapsed)

        if not _has_measured_training_update(trainer, expected_steps={0, 1, 2}):
            logger.error("  Training did not update each of steps 0, 1, and 2")
            return False

        # Verify all checkpoints
        for step in [1, 2, 3]:
            ckpt = Path(tmpdir) / f"step_{step}"
            issue = _checkpoint_issue(ckpt)
            if issue is None:
                logger.info("  step_%d checkpoint: OK", step)
            else:
                logger.error("  step_%d checkpoint: %s", step, issue)

        missing = _missing_checkpoints(tmpdir, [1, 2, 3])
        if missing:
            logger.error("Phase 5 FAILED: missing checkpoints %s", missing)
            return False

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
        (
            2,
            lambda: phase2_single_rollout(
                args.server_url, args.evaluate_url, args.task_id, args.mock
            ),
        ),
        (
            3,
            lambda: phase3_model_inference(
                args.server_url,
                args.evaluate_url,
                args.model_name,
                args.task_id,
                args.mock,
            ),
        ),
        (
            4,
            lambda: phase4_single_training_step(
                args.server_url,
                args.evaluate_url,
                args.model_name,
                args.task_id,
                args.lora_checkpoint,
                args.mock,
            ),
        ),
        (
            5,
            lambda: phase5_multi_step_training(
                args.server_url,
                args.evaluate_url,
                args.model_name,
                args.task_id,
                args.lora_checkpoint,
                args.mock,
            ),
        ),
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
