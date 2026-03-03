"""Reward functions for GRPO training.

Provides binary task-success rewards and group-relative advantage
computation following the GRPO algorithm (Shao et al., 2024).

GRPO computes advantages relative to the group mean rather than using
a learned value function, which is simpler and works well for sparse
binary rewards (task success/failure).
"""

from __future__ import annotations


def binary_task_success(score: float, threshold: float = 0.5) -> float:
    """Convert evaluator score to binary reward.

    Args:
        score: Raw evaluator score (0.0-1.0) from WAA environment.
        threshold: Score at or above which the task is considered successful.

    Returns:
        1.0 if score >= threshold, else 0.0.
    """
    return 1.0 if score >= threshold else 0.0


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

    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance**0.5
    eps = 1e-8

    # No variance means no gradient signal: all advantages are zero
    if std < eps:
        return [0.0] * n

    return [(r - mean) / (std + eps) for r in rewards]
