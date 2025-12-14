"""Evaluation runner for benchmarks.

This module provides functions to run agents on benchmarks and collect results.

Example:
    from openadapt_ml.benchmarks import WAAAdapter, PolicyAgent, evaluate_agent_on_benchmark

    adapter = WAAAdapter(waa_repo_path="/path/to/WAA")
    agent = PolicyAgent(policy)
    results = evaluate_agent_on_benchmark(agent, adapter, max_steps=50)

    print(f"Success rate: {sum(r.success for r in results) / len(results):.1%}")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from openadapt_ml.benchmarks.agent import BenchmarkAgent
from openadapt_ml.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkAction,
    BenchmarkObservation,
    BenchmarkResult,
    BenchmarkTask,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation.

    Attributes:
        max_steps: Maximum steps per task.
        parallel: Number of parallel workers (if supported).
        save_trajectories: Whether to save full trajectories in results.
        verbose: Whether to print progress.
        on_step: Optional callback called after each step.
        on_task_complete: Optional callback called after each task.
    """

    max_steps: int = 50
    parallel: int = 1
    save_trajectories: bool = True
    verbose: bool = True
    on_step: Callable[[BenchmarkObservation, BenchmarkAction, int], None] | None = None
    on_task_complete: Callable[[BenchmarkResult], None] | None = None


def evaluate_agent_on_benchmark(
    agent: BenchmarkAgent,
    adapter: BenchmarkAdapter,
    task_ids: list[str] | None = None,
    max_steps: int = 50,
    parallel: int = 1,
    config: EvaluationConfig | None = None,
) -> list[BenchmarkResult]:
    """Run agent on benchmark tasks and collect results.

    Args:
        agent: Agent to evaluate.
        adapter: Benchmark adapter.
        task_ids: Specific tasks to run (None = all tasks).
        max_steps: Maximum steps per task (overridden by config if provided).
        parallel: Number of parallel workers (overridden by config if provided).
        config: Full evaluation configuration.

    Returns:
        List of BenchmarkResult for each task.
    """
    if config is None:
        config = EvaluationConfig(max_steps=max_steps, parallel=parallel)

    # Load tasks
    if task_ids is not None:
        tasks = [adapter.load_task(tid) for tid in task_ids]
    else:
        tasks = adapter.list_tasks()

    if config.verbose:
        logger.info(f"Evaluating {len(tasks)} tasks on {adapter.name}")

    # Run evaluation
    if config.parallel > 1 and adapter.supports_parallel:
        results = _evaluate_parallel(agent, adapter, tasks, config)
    else:
        results = _evaluate_sequential(agent, adapter, tasks, config)

    # Log summary
    if config.verbose:
        success_count = sum(1 for r in results if r.success)
        success_rate = success_count / len(results) if results else 0
        avg_steps = sum(r.num_steps for r in results) / len(results) if results else 0
        logger.info(
            f"Evaluation complete: {success_count}/{len(results)} "
            f"({success_rate:.1%}) success, {avg_steps:.1f} avg steps"
        )

    return results


def _evaluate_sequential(
    agent: BenchmarkAgent,
    adapter: BenchmarkAdapter,
    tasks: list[BenchmarkTask],
    config: EvaluationConfig,
) -> list[BenchmarkResult]:
    """Run evaluation sequentially.

    Args:
        agent: Agent to evaluate.
        adapter: Benchmark adapter.
        tasks: Tasks to evaluate.
        config: Evaluation configuration.

    Returns:
        List of results.
    """
    results = []
    for i, task in enumerate(tasks):
        if config.verbose:
            logger.info(f"Task {i + 1}/{len(tasks)}: {task.task_id}")

        result = _run_single_task(agent, adapter, task, config)
        results.append(result)

        if config.on_task_complete:
            config.on_task_complete(result)

    return results


def _evaluate_parallel(
    agent: BenchmarkAgent,
    adapter: BenchmarkAdapter,
    tasks: list[BenchmarkTask],
    config: EvaluationConfig,
) -> list[BenchmarkResult]:
    """Run evaluation in parallel.

    Note: This requires the adapter to support parallel execution
    (e.g., via multiple VM instances).

    Args:
        agent: Agent to evaluate.
        adapter: Benchmark adapter.
        tasks: Tasks to evaluate.
        config: Evaluation configuration.

    Returns:
        List of results.
    """
    results = []

    with ThreadPoolExecutor(max_workers=config.parallel) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_run_single_task, agent, adapter, task, config): task
            for task in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)

                if config.on_task_complete:
                    config.on_task_complete(result)

                if config.verbose:
                    status = "SUCCESS" if result.success else "FAIL"
                    logger.info(f"Task {task.task_id}: {status}")

            except Exception as e:
                logger.error(f"Task {task.task_id} failed with error: {e}")
                results.append(
                    BenchmarkResult(
                        task_id=task.task_id,
                        success=False,
                        score=0.0,
                        error=str(e),
                    )
                )

    return results


def _run_single_task(
    agent: BenchmarkAgent,
    adapter: BenchmarkAdapter,
    task: BenchmarkTask,
    config: EvaluationConfig,
) -> BenchmarkResult:
    """Run a single task and return result.

    Args:
        agent: Agent to evaluate.
        adapter: Benchmark adapter.
        task: Task to run.
        config: Evaluation configuration.

    Returns:
        BenchmarkResult.
    """
    start_time = time.perf_counter()
    history: list[tuple[BenchmarkObservation, BenchmarkAction]] = []

    try:
        # Reset agent and environment
        agent.reset()
        obs = adapter.reset(task)

        done = False
        steps = 0
        max_steps = task.time_limit_steps or config.max_steps

        while not done and steps < max_steps:
            # Get action from agent
            action = agent.act(obs, task, history if config.save_trajectories else None)

            # Record step
            if config.save_trajectories:
                history.append((obs, action))

            if config.on_step:
                config.on_step(obs, action, steps)

            # Check for terminal action
            if action.type == "done":
                done = True
                break

            # Execute action
            obs, done, info = adapter.step(action)
            steps += 1

        # Evaluate result
        result = adapter.evaluate(task)

        # Update result with trajectory info
        result.steps = history if config.save_trajectories else []
        result.num_steps = steps
        result.total_time_seconds = time.perf_counter() - start_time

        return result

    except Exception as e:
        logger.error(f"Error running task {task.task_id}: {e}")
        return BenchmarkResult(
            task_id=task.task_id,
            success=False,
            score=0.0,
            steps=history if config.save_trajectories else [],
            num_steps=len(history),
            error=str(e),
            total_time_seconds=time.perf_counter() - start_time,
        )


def compute_metrics(results: list[BenchmarkResult]) -> dict:
    """Compute aggregate metrics from evaluation results.

    Args:
        results: List of BenchmarkResult from evaluation.

    Returns:
        Dict with aggregate metrics.
    """
    if not results:
        return {
            "num_tasks": 0,
            "success_rate": 0.0,
            "avg_score": 0.0,
            "avg_steps": 0.0,
            "avg_time_seconds": 0.0,
        }

    num_tasks = len(results)
    success_count = sum(1 for r in results if r.success)
    total_score = sum(r.score for r in results)
    total_steps = sum(r.num_steps for r in results)
    total_time = sum(r.total_time_seconds for r in results)

    return {
        "num_tasks": num_tasks,
        "success_rate": success_count / num_tasks,
        "avg_score": total_score / num_tasks,
        "avg_steps": total_steps / num_tasks,
        "avg_time_seconds": total_time / num_tasks,
        "success_count": success_count,
        "fail_count": num_tasks - success_count,
    }


def compute_domain_metrics(
    results: list[BenchmarkResult], tasks: list[BenchmarkTask]
) -> dict[str, dict]:
    """Compute per-domain metrics.

    Args:
        results: List of BenchmarkResult.
        tasks: List of BenchmarkTask (to get domain info).

    Returns:
        Dict mapping domain to metrics dict.
    """
    # Build task_id -> domain mapping
    task_domains = {t.task_id: t.domain for t in tasks}

    # Group results by domain
    domain_results: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        domain = task_domains.get(result.task_id, "unknown")
        if domain not in domain_results:
            domain_results[domain] = []
        domain_results[domain].append(result)

    # Compute metrics per domain
    return {domain: compute_metrics(res) for domain, res in domain_results.items()}
