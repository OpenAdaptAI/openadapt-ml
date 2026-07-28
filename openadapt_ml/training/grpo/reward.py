"""Reward functions for GRPO training.

Provides binary task-success rewards and group-relative advantage
computation following the GRPO algorithm (Shao et al., 2024).

GRPO computes advantages relative to the group mean rather than using
a learned value function, which is simpler and works well for sparse
binary rewards (task success/failure).

Also provides ``evaluate_milestones_screenshot``, a standalone utility
that evaluates milestone-based rewards from a screenshot without needing
the WAA /evaluate endpoint.  This is the local-evaluation path used by
the standalone GRPO trainer when ``--task-dir`` is set.
"""

from __future__ import annotations

import io
import logging
import math

from PIL import Image

logger = logging.getLogger(__name__)


class MilestoneEvaluationError(RuntimeError):
    """Raised when milestone rewards could not be computed.

    This exists so that "the agent did not reach any milestone" (reward 0.0)
    and "this evaluation could not run" are different outcomes. Returning 0.0
    for the second case makes an infrastructure failure look like a training
    signal: the trainer would compute advantages, take a gradient step, and
    log a reward mean, all from a number that measures nothing.
    """


def _unit_interval(value: object, name: str) -> float:
    """Return a finite numeric value in the closed unit interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def binary_task_success(score: float, threshold: float = 0.5) -> float:
    """Convert evaluator score to binary reward.

    Args:
        score: Raw evaluator score (0.0-1.0) from WAA environment.
        threshold: Score at or above which the task is considered successful.

    Returns:
        1.0 if score >= threshold, else 0.0.
    """
    measured_score = _unit_interval(score, "score")
    measured_threshold = _unit_interval(threshold, "threshold")
    return 1.0 if measured_score >= measured_threshold else 0.0


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Compute group-relative advantages for a batch of rollout rewards.

    GRPO normalizes rewards within each group:
        advantage[i] = (reward[i] - mean) / (std + eps)

    If all rewards are identical (no variance), returns all zeros. This
    avoids NaN from division by zero and correctly signals that there is
    no gradient signal when every rollout in the group has the same outcome.

    Args:
        rewards: List of scalar rewards for each rollout in the group.

    Returns:
        List of advantage values, same length as rewards.
    """
    n = len(rewards)
    if n == 0:
        return []

    measured_rewards = [
        _unit_interval(reward, f"rewards[{index}]")
        for index, reward in enumerate(rewards)
    ]

    mean = sum(measured_rewards) / n
    variance = sum((reward - mean) ** 2 for reward in measured_rewards) / n
    std = variance**0.5
    eps = 1e-8

    # No variance means no gradient signal: all advantages are zero
    if std < eps:
        return [0.0] * n

    return [(reward - mean) / (std + eps) for reward in measured_rewards]


def evaluate_milestones_screenshot(
    task_config: object,
    screenshot_bytes: bytes,
    vlm_model: str = "gpt-4.1-mini",
    vlm_provider: str = "openai",
) -> float:
    """Evaluate milestone-based rewards from a screenshot (no server needed).

    Iterates over the milestones in a TaskConfig and evaluates each
    ``screenshot``-type milestone using a VLM judge.  Non-screenshot
    milestones are skipped (they require a live server).

    This is a standalone utility that can be called independently of the
    trainer, e.g.::

        from openadapt_ml.training.grpo.reward import evaluate_milestones_screenshot
        reward = evaluate_milestones_screenshot(task_config, screenshot_bytes)

    Args:
        task_config: A ``TaskConfig`` instance (from ``openadapt_evals.task_config``).
            Must have a ``milestones`` attribute (list of ``Milestone`` objects).
        screenshot_bytes: PNG screenshot bytes to evaluate against.
        vlm_model: VLM model name for the judge.
        vlm_provider: VLM provider (``"openai"`` or ``"anthropic"``).

    Returns:
        Fraction of screenshot milestones that passed (0.0 to 1.0). 0.0 means
        every screenshot milestone was evaluated and none passed.

    Raises:
        MilestoneEvaluationError: If the evaluation could not be run at all --
            no milestones, no locally evaluable milestones, ``openadapt-evals``
            missing, a milestone with no description, or a VLM judge failure.
            These used to return (or silently contribute) 0.0, which is
            indistinguishable from a genuine failed rollout and would be fed
            straight into the GRPO advantage computation as if it were a
            measurement. Callers that want to continue must catch this and
            decide explicitly what to do with the missing measurement.
    """
    milestones = getattr(task_config, "milestones", None)
    if not milestones:
        raise MilestoneEvaluationError(
            f"Task config {getattr(task_config, 'id', task_config)!r} has no "
            "milestones, so there is nothing to score. This is not a reward "
            "of 0.0."
        )
    if not screenshot_bytes:
        raise MilestoneEvaluationError(
            "Screenshot milestone evaluation requires non-empty screenshot bytes"
        )
    try:
        with Image.open(io.BytesIO(screenshot_bytes)) as screenshot:
            screenshot.verify()
    except Exception as exc:
        raise MilestoneEvaluationError(
            "Screenshot milestone evaluation requires decodable image evidence"
        ) from exc

    # Only evaluate screenshot-type milestones locally
    screenshot_milestones = [
        ms for ms in milestones if getattr(ms.check, "check", None) == "screenshot"
    ]
    if not screenshot_milestones:
        raise MilestoneEvaluationError(
            f"Task config {getattr(task_config, 'id', task_config)!r} has "
            f"{len(milestones)} milestone(s) but none of type 'screenshot'. "
            "Local screenshot evaluation cannot score this task; use the WAA "
            "/evaluate endpoint instead."
        )

    try:
        from openadapt_evals.vlm_evaluator import vlm_judge
    except ImportError as exc:
        raise MilestoneEvaluationError(
            "openadapt-evals is not installed; screenshot milestones cannot be "
            "evaluated. Install with: pip install openadapt-evals"
        ) from exc

    passed = 0
    for ms in screenshot_milestones:
        description = getattr(ms.check, "description", None) or ""
        if not description:
            raise MilestoneEvaluationError(
                f"Milestone {getattr(ms, 'name', '?')!r} is a screenshot "
                "milestone with no description, so the VLM judge has nothing "
                "to check. Skipping it would leave it in the denominator and "
                "silently depress the reward."
            )
        try:
            success, confidence = vlm_judge(
                screenshot_bytes,
                description,
                model=vlm_model,
                provider=vlm_provider,
            )
        except Exception as exc:
            raise MilestoneEvaluationError(
                f"VLM judge failed on milestone {getattr(ms, 'name', '?')!r}: "
                f"{exc}. A judge failure is not a failed milestone."
            ) from exc
        if not isinstance(success, bool):
            raise MilestoneEvaluationError(
                f"VLM judge returned a non-boolean success value for milestone "
                f"{getattr(ms, 'name', '?')!r}"
            )
        try:
            _unit_interval(confidence, "VLM judge confidence")
        except ValueError as exc:
            raise MilestoneEvaluationError(
                f"VLM judge returned invalid confidence for milestone "
                f"{getattr(ms, 'name', '?')!r}"
            ) from exc
        if success:
            passed += 1
        logger.debug(
            "Milestone '%s': %s",
            getattr(ms, "name", "?"),
            "PASS" if success else "FAIL",
        )

    total = len(screenshot_milestones)
    score = passed / total
    logger.info(
        "Milestone evaluation: %d/%d screenshot milestones passed (%.2f)",
        passed,
        total,
        score,
    )
    return score
