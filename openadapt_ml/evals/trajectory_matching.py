from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openadapt_ml.runtime.policy import AgentPolicy
from openadapt_ml.schemas.sessions import Action, Episode


@dataclass
class MilestoneSpec:
    """Defines a semantic milestone for weak episode success evaluation.

    A milestone is achieved when, at a specific step, the predicted action
    matches certain criteria (type match + optional coord threshold).
    """
    name: str
    step_index: int  # Which step in the episode (0-indexed)
    expected_type: str  # Expected ground truth action type ("click", "type", "done", etc.)
    coord_threshold: Optional[float] = None  # If set, coord error must be < this for clicks


# Predefined milestone specs per scenario
# Updated for 6-step episode (no spurious WAIT):
# Step 0: click username, Step 1: type username, Step 2: click password,
# Step 3: type password, Step 4: click login, Step 5: done
LOGIN_MILESTONES = [
    MilestoneSpec("typed_username", step_index=1, expected_type="type"),
    MilestoneSpec("typed_password", step_index=3, expected_type="type"),
    MilestoneSpec("clicked_login", step_index=4, expected_type="click", coord_threshold=0.10),
    MilestoneSpec("emitted_done", step_index=5, expected_type="done"),
]

SETTINGS_MILESTONES = [
    # Placeholder - to be defined when settings scenario is implemented
    # MilestoneSpec("toggled_setting", step_index=..., expected_type="click", coord_threshold=0.10),
    # MilestoneSpec("clicked_save", step_index=..., expected_type="click", coord_threshold=0.10),
    # MilestoneSpec("emitted_done", step_index=..., expected_type="done"),
]


def get_milestones_for_scenario(scenario: str = "login") -> List[MilestoneSpec]:
    """Return milestone specs for a given scenario."""
    if scenario == "login":
        return LOGIN_MILESTONES
    elif scenario == "settings":
        return SETTINGS_MILESTONES
    else:
        return []  # Unknown scenario - no semantic milestones


@dataclass
class EpisodeMetrics:
    episode_id: str
    step_matches: int
    step_total: int
    coord_errors: List[float]
    success_pred: bool  # Strict: all steps must match
    success_gt: Optional[bool]
    click_hits: int  # Point-based: click within 5% of target center
    click_total: int
    # Semantic goal milestones (scenario-agnostic)
    milestones_achieved: Dict[str, bool] = field(default_factory=dict)
    # Full step correctness (type match + click hit when applicable)
    full_step_correct: int = 0
    # State-based weak success: from model's State: {"success": true/false}
    state_success: Optional[bool] = None
    # Bbox-based click evaluation: click anywhere within element bounds
    bbox_hits: int = 0
    bbox_total: int = 0
    # SoM element index accuracy: predicted index == GT index
    element_hits: int = 0
    element_total: int = 0


@dataclass
class AggregateMetrics:
    num_episodes: int
    num_steps: int
    action_type_accuracy: float
    mean_coord_error: Optional[float]
    coord_error_count: int
    episode_success_rate: Optional[float]  # Strict: all steps must match (renamed from success_pred)
    click_hit_rate: Optional[float]  # Point-based: within 5% of center
    mean_episode_progress: Optional[float]  # Partial credit: avg(step_matches/step_total)
    # New partial-credit metrics
    mean_episode_step_score: Optional[float]  # Strict partial: avg(full_step_correct/step_total)
    weak_episode_success_rate: Optional[float]  # Semantic milestones all achieved
    state_success_rate: Optional[float] = None  # From model's State: {"success": true}
    bbox_hit_rate: Optional[float] = None  # Bbox-based: click anywhere in element bounds
    element_accuracy: Optional[float] = None  # SoM element index accuracy


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


def is_click_in_bbox(pred_action: Action, gt_action: Action) -> Optional[bool]:
    """Check if predicted click falls within ground truth bounding box.

    Returns:
        - True if prediction is inside bbox
        - False if prediction is outside bbox
        - None if no bbox is available (fall back to coord distance)
    """
    if gt_action.bbox is None:
        return None

    if pred_action.x is None or pred_action.y is None:
        return False

    x_min, y_min, x_max, y_max = gt_action.bbox
    return (x_min <= pred_action.x <= x_max) and (y_min <= pred_action.y <= y_max)


def evaluate_episode(
    policy: AgentPolicy,
    episode: Episode,
    samples: List[Dict[str, Any]],
    start_idx: int,
    log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_limit: Optional[int] = None,
    logged_count: int = 0,
    milestones: Optional[List[MilestoneSpec]] = None,
    use_som: bool = False,
) -> tuple[EpisodeMetrics, int, int]:
    """Evaluate a single episode offline using pre-built SFT samples.

    We assume `samples` were created by iterating episodes and steps in the
    same order as here (see `build_next_action_sft_samples`). `start_idx`
    indicates the index of the first sample corresponding to this episode's
    first step. The function returns the episode metrics and the next sample
    index after this episode.

    Args:
        milestones: Optional list of MilestoneSpec to track for weak success.
                   If None, defaults to LOGIN_MILESTONES for backward compat.
        use_som: If True, evaluate using Set-of-Marks element index matching
                instead of coordinate-based evaluation.
    """
    if milestones is None:
        milestones = LOGIN_MILESTONES

    step_matches = 0
    step_total = 0
    coord_errors: List[float] = []
    success_pred = True
    click_hits = 0  # Point-based (5% threshold)
    click_total = 0
    bbox_hits = 0  # Bbox-based (anywhere in element)
    bbox_total = 0
    element_hits = 0  # SoM element index match
    element_total = 0
    # Generic milestone tracking
    milestones_achieved: Dict[str, bool] = {m.name: False for m in milestones}
    full_step_correct = 0
    # Track the last state's success flag
    last_state_success: Optional[bool] = None

    sample_idx = start_idx

    for step_idx, step in enumerate(episode.steps):
        # Skip steps without an image; the dataset builder does the same.
        if not step.observation.image_path:
            continue

        if sample_idx >= len(samples):
            break

        sample = samples[sample_idx]
        sample_idx += 1

        pred_action, _thought, pred_state, raw_text = policy.predict_action_from_sample(sample)
        gt_action = step.action

        # Track state-based success from final step
        if pred_state and isinstance(pred_state, dict):
            success_val = pred_state.get("success")
            if isinstance(success_val, bool):
                last_state_success = success_val

        type_match = pred_action.type == gt_action.type
        if type_match:
            step_matches += 1
        else:
            success_pred = False

        coord_error: Optional[float] = None
        click_hit = False
        bbox_hit = False
        element_hit = False

        # SoM mode: evaluate by element index for click/drag/type actions
        if use_som and gt_action.type in {"click", "drag", "type"}:
            if gt_action.element_index is not None:
                element_total += 1
                if pred_action.element_index == gt_action.element_index:
                    element_hits += 1
                    element_hit = True
        elif gt_action.type in {"click", "drag"}:
            # Coordinate mode: evaluate by coordinate distance
            coord_error = compute_coordinate_error(pred_action, gt_action)
            if coord_error is not None:
                coord_errors.append(coord_error)
                click_total += 1
                if coord_error < 0.05:
                    click_hits += 1
                    click_hit = True

            # Bbox-based evaluation (more lenient)
            in_bbox = is_click_in_bbox(pred_action, gt_action)
            if in_bbox is not None:
                bbox_total += 1
                if in_bbox:
                    bbox_hits += 1
                    bbox_hit = True

        # Full step correctness: type matches AND element/coord match for relevant actions
        if type_match:
            if use_som and gt_action.type in {"click", "drag", "type"}:
                # SoM mode: require element index match
                if element_hit:
                    full_step_correct += 1
            elif gt_action.type in {"click", "drag"}:
                # Coordinate mode: require click hit
                if click_hit:
                    full_step_correct += 1
            else:
                # Non-targeting actions (wait, done): type match is sufficient
                full_step_correct += 1

        # Track semantic milestones using the milestone spec
        for milestone in milestones:
            if step_idx == milestone.step_index and gt_action.type == milestone.expected_type:
                if pred_action.type == milestone.expected_type:
                    # Check coord threshold if specified (for click actions)
                    if milestone.coord_threshold is not None:
                        if coord_error is not None and coord_error < milestone.coord_threshold:
                            milestones_achieved[milestone.name] = True
                    else:
                        # No coord threshold - type match is sufficient
                        milestones_achieved[milestone.name] = True

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
                "model_output_raw": raw_text,
                "pred_action": {
                    "type": pred_action.type,
                    "x": pred_action.x,
                    "y": pred_action.y,
                    "text": pred_action.text,
                    "element_index": pred_action.element_index,
                },
                "ground_truth_action": {
                    "type": gt_action.type,
                    "x": gt_action.x,
                    "y": gt_action.y,
                    "text": gt_action.text,
                    "element_index": gt_action.element_index,
                },
                "correct_type": pred_action.type == gt_action.type,
                "coord_error_norm": coord_error,
                "element_match": pred_action.element_index == gt_action.element_index
                if gt_action.element_index is not None
                else None,
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
        milestones_achieved=milestones_achieved,
        full_step_correct=full_step_correct,
        state_success=last_state_success,
        bbox_hits=bbox_hits,
        bbox_total=bbox_total,
        element_hits=element_hits,
        element_total=element_total,
    )
    return metrics, sample_idx, logged_count


def aggregate_metrics(episodes_metrics: List[EpisodeMetrics]) -> AggregateMetrics:
    """Aggregate per-episode metrics into global metrics.

    Three-tier episodic success metrics (from least to most strict):

    1. **weak_episode_success_rate**: Semantic goal completion. For the login
       flow, requires: typed username, typed password, clicked login button
       (within 10% coord error), and emitted DONE. Allows intermediate mistakes.

    2. **mean_episode_step_score**: Strict partial credit. Average of
       (full_step_correct / step_total) per episode. A step is "full correct"
       if action type matches AND (not a click OR click within 5% threshold).

    3. **episode_success_rate**: Hard metric. All steps must match exactly
       (action type correct AND click hits where applicable). This is the
       long-horizon metric that only becomes meaningful at high step accuracy.

    Also computes:
    - action_type_accuracy: total correct types / total steps.
    - mean_coord_error: mean of all collected coordinate errors.
    - mean_episode_progress: avg(step_matches / step_total) - type matches only.
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

    # Partial credit: average episode progress (step_matches / step_total per episode)
    if eval_episodes:
        episode_progress_scores = [
            m.step_matches / m.step_total for m in eval_episodes
        ]
        mean_episode_progress = sum(episode_progress_scores) / len(episode_progress_scores)
    else:
        mean_episode_progress = None

    # Strict partial: avg(full_step_correct / step_total) - requires type match + click hit
    if eval_episodes:
        step_scores = [
            m.full_step_correct / m.step_total for m in eval_episodes
        ]
        mean_episode_step_score = sum(step_scores) / len(step_scores)
    else:
        mean_episode_step_score = None

    # Weak episode success: all milestones achieved
    if eval_episodes:
        weak_success_count = sum(
            1 for m in eval_episodes
            if m.milestones_achieved and all(m.milestones_achieved.values())
        )
        weak_episode_success_rate = weak_success_count / len(eval_episodes)
    else:
        weak_episode_success_rate = None

    # State-based success: from model's State: {"success": true}
    episodes_with_state = [m for m in eval_episodes if m.state_success is not None]
    if episodes_with_state:
        state_success_count = sum(1 for m in episodes_with_state if m.state_success)
        state_success_rate = state_success_count / len(episodes_with_state)
    else:
        state_success_rate = None

    # Bbox-based click evaluation (more lenient than point-based)
    total_bbox_hits = sum(m.bbox_hits for m in episodes_metrics)
    total_bbox_total = sum(m.bbox_total for m in episodes_metrics)
    if total_bbox_total > 0:
        bbox_hit_rate: Optional[float] = total_bbox_hits / total_bbox_total
    else:
        bbox_hit_rate = None

    # SoM element index accuracy
    total_element_hits = sum(m.element_hits for m in episodes_metrics)
    total_element_total = sum(m.element_total for m in episodes_metrics)
    if total_element_total > 0:
        element_accuracy: Optional[float] = total_element_hits / total_element_total
    else:
        element_accuracy = None

    return AggregateMetrics(
        num_episodes=num_episodes,
        num_steps=num_steps,
        action_type_accuracy=action_type_accuracy,
        mean_coord_error=mean_coord_error,
        coord_error_count=len(all_coord_errors),
        episode_success_rate=episode_success_rate,
        click_hit_rate=click_hit_rate,
        mean_episode_progress=mean_episode_progress,
        mean_episode_step_score=mean_episode_step_score,
        weak_episode_success_rate=weak_episode_success_rate,
        state_success_rate=state_success_rate,
        bbox_hit_rate=bbox_hit_rate,
        element_accuracy=element_accuracy,
    )


def evaluate_policy_on_episodes(
    policy: AgentPolicy,
    episodes: List[Episode],
    samples: List[Dict[str, Any]],
    log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_limit: Optional[int] = None,
    use_som: bool = False,
) -> AggregateMetrics:
    """Evaluate a policy on a list of episodes given corresponding SFT samples.

    The `samples` list must have been produced from `episodes` using
    `build_next_action_sft_samples`, so that iterating episodes/steps in order
    aligns with iterating over `samples`.

    Args:
        use_som: If True, evaluate using Set-of-Marks element index matching
                instead of coordinate-based evaluation.
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
            use_som=use_som,
        )
        episodes_metrics.append(metrics)

    return aggregate_metrics(episodes_metrics)
