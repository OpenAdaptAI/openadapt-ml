from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from openadapt_ml.runtime.policy import AgentPolicy
from openadapt_ml.schemas.sessions import Action, Episode


@dataclass
class EpisodeMetrics:
    episode_id: str
    step_matches: int
    step_total: int
    coord_errors: List[float]
    success_pred: bool
    success_gt: Optional[bool]
    click_hits: int
    click_total: int


@dataclass
class AggregateMetrics:
    num_episodes: int
    num_steps: int
    action_type_accuracy: float
    mean_coord_error: Optional[float]
    coord_error_count: int
    episode_success_rate: Optional[float]
    click_hit_rate: Optional[float]


def compute_coordinate_error(pred_action: Action, gt_action: Action) -> Optional[float]:
    """Compute normalized L2 distance between predicted and ground-truth coords.

    Returns None if either action is missing coordinates.
    """

    if (
        pred_action.x is None
        or pred_action.y is None
        or gt_action.x is None
        or gt_action.y is None
    ):
        return None

    dx = pred_action.x - gt_action.x
    dy = pred_action.y - gt_action.y
    return math.sqrt(dx * dx + dy * dy)


def evaluate_episode(
    policy: AgentPolicy,
    episode: Episode,
    samples: List[Dict[str, Any]],
    start_idx: int,
    log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_limit: Optional[int] = None,
    logged_count: int = 0,
) -> tuple[EpisodeMetrics, int, int]:
    """Evaluate a single episode offline using pre-built SFT samples.

    We assume `samples` were created by iterating episodes and steps in the
    same order as here (see `build_next_action_sft_samples`). `start_idx`
    indicates the index of the first sample corresponding to this episode's
    first step. The function returns the episode metrics and the next sample
    index after this episode.
    """

    step_matches = 0
    step_total = 0
    coord_errors: List[float] = []
    success_pred = True
    click_hits = 0
    click_total = 0

    sample_idx = start_idx

    for step_idx, step in enumerate(episode.steps):
        # Skip steps without an image; the dataset builder does the same.
        if not step.observation.image_path:
            continue

        if sample_idx >= len(samples):
            break

        sample = samples[sample_idx]
        sample_idx += 1

        pred_action, _ = policy.predict_action_from_sample(sample)
        gt_action = step.action

        if pred_action.type == gt_action.type:
            step_matches += 1
        else:
            success_pred = False

        coord_error: Optional[float] = None
        if gt_action.type in {"click", "drag"}:
            coord_error = compute_coordinate_error(pred_action, gt_action)
            if coord_error is not None:
                coord_errors.append(coord_error)
                click_total += 1
                if coord_error < 0.05:
                    click_hits += 1

        # Ensure DONE is correct at the DONE step.
        if gt_action.type == "done" and pred_action.type != "done":
            success_pred = False

        # Optional logging of this step.
        if log_fn is not None and (log_limit is None or logged_count < log_limit):
            messages = sample.get("messages", [])
            system_prompt = None
            user_prompt = None
            for m in messages:
                if m.get("role") == "system" and system_prompt is None:
                    system_prompt = m.get("content")
                if m.get("role") == "user" and user_prompt is None:
                    user_prompt = m.get("content")

            record: Dict[str, Any] = {
                "episode_id": episode.id,
                "step_index": step_idx,
                "goal": episode.goal,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model_output_raw": getattr(pred_action, "raw", {}).get("text")
                if pred_action.type == "failed"
                else None,
                "pred_action": {
                    "type": pred_action.type,
                    "x": pred_action.x,
                    "y": pred_action.y,
                    "text": pred_action.text,
                },
                "ground_truth_action": {
                    "type": gt_action.type,
                    "x": gt_action.x,
                    "y": gt_action.y,
                    "text": gt_action.text,
                },
                "correct_type": pred_action.type == gt_action.type,
                "coord_error_norm": coord_error,
            }

            log_fn(record)
            logged_count += 1

        step_total += 1

    metrics = EpisodeMetrics(
        episode_id=episode.id,
        step_matches=step_matches,
        step_total=step_total,
        coord_errors=coord_errors,
        success_pred=success_pred,
        success_gt=episode.success,
        click_hits=click_hits,
        click_total=click_total,
    )
    return metrics, sample_idx, logged_count


def aggregate_metrics(episodes_metrics: List[EpisodeMetrics]) -> AggregateMetrics:
    """Aggregate per-episode metrics into global metrics.

    - action_type_accuracy: total correct types / total steps.
    - mean_coord_error: mean of all collected coordinate errors.
    - episode_success_rate: fraction of episodes where success_pred is True,
      restricted to episodes that have at least one evaluated step.
    """

    num_episodes = len(episodes_metrics)
    num_steps = sum(m.step_total for m in episodes_metrics)

    total_matches = sum(m.step_matches for m in episodes_metrics)
    action_type_accuracy = (total_matches / num_steps) if num_steps > 0 else 0.0

    all_coord_errors: List[float] = []
    for m in episodes_metrics:
        all_coord_errors.extend(m.coord_errors)

    mean_coord_error: Optional[float]
    if all_coord_errors:
        mean_coord_error = sum(all_coord_errors) / len(all_coord_errors)
    else:
        mean_coord_error = None

    eval_episodes = [m for m in episodes_metrics if m.step_total > 0]
    if eval_episodes:
        success_count = sum(1 for m in eval_episodes if m.success_pred)
        episode_success_rate = success_count / len(eval_episodes)
    else:
        episode_success_rate = None

    total_click_hits = sum(m.click_hits for m in episodes_metrics)
    total_click_total = sum(m.click_total for m in episodes_metrics)
    if total_click_total > 0:
        click_hit_rate: Optional[float] = total_click_hits / total_click_total
    else:
        click_hit_rate = None

    return AggregateMetrics(
        num_episodes=num_episodes,
        num_steps=num_steps,
        action_type_accuracy=action_type_accuracy,
        mean_coord_error=mean_coord_error,
        coord_error_count=len(all_coord_errors),
        episode_success_rate=episode_success_rate,
        click_hit_rate=click_hit_rate,
    )


def evaluate_policy_on_episodes(
    policy: AgentPolicy,
    episodes: List[Episode],
    samples: List[Dict[str, Any]],
    log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_limit: Optional[int] = None,
) -> AggregateMetrics:
    """Evaluate a policy on a list of episodes given corresponding SFT samples.

    The `samples` list must have been produced from `episodes` using
    `build_next_action_sft_samples`, so that iterating episodes/steps in order
    aligns with iterating over `samples`.
    """

    episodes_metrics: List[EpisodeMetrics] = []
    sample_idx = 0
    logged_count = 0

    for episode in episodes:
        metrics, sample_idx, logged_count = evaluate_episode(
            policy,
            episode,
            samples,
            sample_idx,
            log_fn=log_fn,
            log_limit=log_limit,
            logged_count=logged_count,
        )
        episodes_metrics.append(metrics)

    return aggregate_metrics(episodes_metrics)
