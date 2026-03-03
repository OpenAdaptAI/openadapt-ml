"""Rollout collector for GRPO training.

Collects groups of N rollouts using the openadapt-evals RLEnvironment,
which wraps a live WAA server. Each rollout produces a trajectory of
(observation, action, reward) tuples and a terminal reward from the
WAA evaluator.

Currently sequential (single VM). Parallel VM support via
openadapt-evals PoolManager is future work.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from openadapt_ml.training.grpo.config import GRPOConfig
from openadapt_ml.training.grpo.reward import binary_task_success

logger = logging.getLogger(__name__)

# Deferred imports for openadapt-evals dependencies (optional at install time)
try:
    from openadapt_evals.adapters import (
        RLEnvironment,
        RolloutStep,
        WAALiveAdapter,
        WAALiveConfig,
    )
    from openadapt_evals.adapters.rl_env import ResetConfig
except ImportError:
    RLEnvironment = None  # type: ignore[assignment, misc]
    RolloutStep = None  # type: ignore[assignment, misc]
    WAALiveAdapter = None  # type: ignore[assignment, misc]
    WAALiveConfig = None  # type: ignore[assignment, misc]
    ResetConfig = None  # type: ignore[assignment, misc]


@dataclass
class Rollout:
    """Complete episode rollout with reward.

    Attributes:
        task_id: The WAA task that was executed.
        steps: List of RolloutStep objects from the RLEnvironment.
        reward: Binary reward (0.0 or 1.0) from the evaluator.
        num_steps: Number of steps taken in the episode.
        instruction: Task instruction text for prompt reconstruction
            during loss computation. Populated from the environment's
            current task after rollout collection.
    """

    task_id: str
    steps: list[Any] = field(default_factory=list)  # list[RolloutStep]
    reward: float = 0.0
    num_steps: int = 0
    instruction: str = ""


class GRPORolloutCollector:
    """Collects groups of rollouts using openadapt-evals RLEnvironment.

    Creates a WAALiveAdapter and RLEnvironment from the config, then
    provides methods to collect groups of N rollouts for GRPO training.
    Currently sequential (single VM); parallel VM support is future work.

    Args:
        config: GRPO training configuration.

    Raises:
        ImportError: If openadapt-evals is not installed.
    """

    def __init__(self, config: GRPOConfig) -> None:
        if RLEnvironment is None:
            raise ImportError(
                "openadapt-evals is required for rollout collection. "
                "Install it with: uv add openadapt-evals"
            )

        self._config = config
        self._adapter = WAALiveAdapter(
            WAALiveConfig(server_url=config.server_url)
        )
        self._env = RLEnvironment(self._adapter)

    @property
    def env(self) -> Any:
        """The underlying RLEnvironment instance."""
        return self._env

    def collect_group(
        self,
        agent_fn: Callable,
        task_id: str | None = None,
    ) -> list[Rollout]:
        """Collect N rollouts for one GRPO gradient step.

        Runs the agent N times on the same task (or a random task from
        config.task_ids if task_id is not specified). Each rollout resets
        the environment, runs the agent, and evaluates the result.

        Currently sequential (single VM). Parallel VM support via
        openadapt-evals PoolManager is future work.

        Args:
            agent_fn: Callable that takes a BenchmarkObservation and returns
                a BenchmarkAction. This is the model's predict function.
            task_id: Specific task ID, or None to pick from config.task_ids.

        Returns:
            List of N Rollout objects with binary rewards.
        """
        if task_id is None:
            if not self._config.task_ids:
                raise ValueError(
                    "No task_id provided and config.task_ids is empty."
                )
            task_id = random.choice(self._config.task_ids)

        rollouts: list[Rollout] = []

        for i in range(self._config.num_rollouts_per_step):
            logger.info(
                "Collecting rollout %d/%d for task %s",
                i + 1,
                self._config.num_rollouts_per_step,
                task_id,
            )

            # collect_rollout resets the environment internally with the
            # given task_id before running the agent
            steps = self._env.collect_rollout(
                agent_fn=agent_fn,
                max_steps=self._config.max_steps_per_episode,
                stuck_window=self._config.stuck_window,
                task_id=task_id,
            )

            # Extract terminal score from the last step's reward
            raw_score = steps[-1].reward if steps else 0.0
            reward = binary_task_success(raw_score)

            # CR-01: Extract task instruction from the environment's
            # current task (set during reset inside collect_rollout).
            instruction = ""
            task = getattr(self._env, "_current_task", None)
            if task is not None:
                instruction = getattr(task, "instruction", "") or ""

            rollout = Rollout(
                task_id=task_id,
                steps=steps,
                reward=reward,
                num_steps=len(steps),
                instruction=instruction,
            )
            rollouts.append(rollout)

            logger.info(
                "Rollout %d: %d steps, raw_score=%.2f, reward=%.1f",
                i + 1,
                len(steps),
                raw_score,
                reward,
            )

        return rollouts

    def close(self) -> None:
        """Clean up adapter resources."""
        if hasattr(self._adapter, "close"):
            self._adapter.close()
