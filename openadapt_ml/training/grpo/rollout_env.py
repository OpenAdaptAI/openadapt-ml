"""Minimal environment interface driven by GRPO training.

RL training in ``openadapt-ml`` needs to *drive* an environment (reset it,
step actions, observe, and collect whole rollouts), but the *concrete*
environment is an evaluation-harness concern that lives in
``openadapt-evals`` (e.g. ``RLEnvironment``, ``WAALiveAdapter``,
``WAADesktopEnv``).

To keep ``openadapt-ml`` a dependency **leaf** (no module-level import of
``openadapt-evals``), the trainer types against this thin ``RolloutEnv``
Protocol instead of a concrete class. The concrete adapters in
``openadapt-evals`` implement it structurally.

Only the surface the GRPO trainer/collector actually uses is declared here;
signatures are intentionally loose (``Any``) so the concrete evals
implementations conform structurally without importing evals-specific
config types into ml.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from openadapt_types import BenchmarkAction, BenchmarkObservation


@runtime_checkable
class RolloutEnv(Protocol):
    """Structural interface for an environment the GRPO trainer can drive.

    Implemented by ``openadapt_evals.adapters.rl_env.RLEnvironment`` (and
    compatible desktop environments). ml depends on this interface; evals
    provides the implementation.
    """

    @property
    def screen_size(self) -> tuple[int, int]:
        """Current environment screen size as (width, height)."""
        ...

    def reset(self, config: Any = None) -> BenchmarkObservation:
        """Reset the environment and return the initial observation."""
        ...

    def step(self, action: BenchmarkAction) -> Any:
        """Execute an action, returning a step result (obs/reward/done/info)."""
        ...

    def observe(self) -> BenchmarkObservation:
        """Return the current observation without stepping."""
        ...

    def collect_rollout(
        self,
        agent_fn: Any,
        max_steps: int = ...,
        stuck_window: int = ...,
        task_id: Any = None,
    ) -> list[Any]:
        """Run ``agent_fn`` to completion, returning the list of rollout steps."""
        ...


__all__ = ["RolloutEnv"]
