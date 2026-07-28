"""Regression tests for failures that used to be reported as successful results.

Every test in this file guards one specific place where openadapt-ml could not
tell "I looked and the answer is empty/zero" apart from "I could not look", and
returned the first shape for the second case. In a package that produces
training rewards, grounding metrics and exported artifacts, that difference is
the difference between a number and a wrong number.

Each test is mutation-checked against the pre-fix code: reverting the
production change makes the matching test fail.
"""

from __future__ import annotations

import ast
import io
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _error_type(module_path: str, name: str) -> type[BaseException]:
    """Resolve an exception type by name, falling back to Exception.

    The tests below are mutation-checked by reverting the production file to its
    pre-fix form. If they imported the new exception class directly, that revert
    would make them fail with ImportError -- which proves only that a name was
    removed. Resolving the type dynamically means the revert makes them fail
    with "DID NOT RAISE", i.e. on the defect itself: a value was returned where
    a failure should have been reported.
    """
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name, Exception)


# ---------------------------------------------------------------------------
# 1. Duplicated dataclass field (`task_dir` declared twice in GRPOConfig)
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "openadapt_ml"


def _duplicate_annotated_fields(tree: ast.AST) -> list[tuple[str, str, int]]:
    """Return (class_name, field_name, lineno) for every re-declared field."""
    duplicates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        seen: dict[str, int] = {}
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            name = stmt.target.id
            if name in seen:
                duplicates.append((node.name, name, stmt.lineno))
            else:
                seen[name] = stmt.lineno
    return duplicates


def test_grpo_config_declares_task_dir_exactly_once():
    """`task_dir` was declared twice in GRPOConfig (ruff PIE794).

    Both declarations spelled `str | None = None`, so the surviving default was
    unchanged -- but a duplicate field is a live hazard: the *last* assignment
    silently wins the default while the *first* keeps the field's position in
    the generated __init__ signature, so the two halves of one field can drift
    apart with nothing to flag it. The docstring documents exactly one
    `task_dir`, and only the first declaration carried the explanatory comment
    that matches it.
    """
    source = (PACKAGE_ROOT / "training" / "grpo" / "config.py").read_text()
    duplicates = _duplicate_annotated_fields(ast.parse(source))
    assert duplicates == [], "GRPOConfig re-declares dataclass field(s): " + ", ".join(
        f"{cls}.{name} at line {line}" for cls, name, line in duplicates
    )


def test_no_dataclass_in_the_package_redeclares_a_field():
    """The same defect anywhere else in openadapt_ml/ is also a failure."""
    duplicates = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for cls, name, line in _duplicate_annotated_fields(tree):
            duplicates.append(f"{path}:{line}: {cls}.{name}")
    assert duplicates == [], "Re-declared class fields:\n" + "\n".join(duplicates)


def test_grpo_config_task_dir_keeps_its_documented_meaning():
    """The declaration that survived is the documented one."""
    from openadapt_ml.training.grpo.config import GRPOConfig

    task_dir_fields = [f for f in fields(GRPOConfig) if f.name == "task_dir"]
    assert len(task_dir_fields) == 1
    assert task_dir_fields[0].default is None
    assert GRPOConfig().task_dir is None
    # The docstring is the contract the surviving declaration must match.
    assert "task_dir: Path to a directory of YAML task config files" in (
        GRPOConfig.__doc__ or ""
    )


def test_grpo_loss_refuses_missing_or_unreadable_rollout_evidence():
    """Missing screenshots must not become a successful zero-loss update."""
    from openadapt_ml.training.grpo.config import GRPOConfig
    from openadapt_ml.training.grpo.rollout_collector import Rollout
    from openadapt_ml.training.grpo.trainer import GRPOTrainer, RolloutLossError

    trainer = object.__new__(GRPOTrainer)
    trainer._config = GRPOConfig()
    trainer._collector = None
    trainer._model = MagicMock()
    parameter = MagicMock(device="cpu")
    trainer._model.parameters.return_value = iter([parameter])

    with pytest.raises(RolloutLossError, match="no trainable"):
        trainer._compute_rollout_loss(Rollout(task_id="empty"), advantage=1.0)

    trainer._model.parameters.return_value = iter([parameter])
    broken = Rollout(
        task_id="broken",
        steps=[
            SimpleNamespace(
                observation=SimpleNamespace(screenshot=b"not an image"),
                action=SimpleNamespace(type="click", x=1, y=1),
            )
        ],
    )
    with pytest.raises(RolloutLossError, match="unreadable screenshot"):
        trainer._compute_rollout_loss(broken, advantage=1.0)


def test_grpo_collector_does_not_score_an_empty_rollout_as_zero():
    from openadapt_ml.training.grpo.config import GRPOConfig
    from openadapt_ml.training.grpo.rollout_collector import (
        GRPORolloutCollector,
        RolloutCollectionError,
    )

    collector = object.__new__(GRPORolloutCollector)
    collector._config = GRPOConfig(
        task_ids=["task-1"],
        num_rollouts_per_step=1,
    )
    collector._task_configs = {}
    collector._env = MagicMock()
    collector._env.collect_rollout.return_value = []

    with pytest.raises(RolloutCollectionError, match="no rollout steps"):
        collector.collect_group(agent_fn=MagicMock(), task_id="task-1")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_grpo_collector_rejects_invalid_terminal_score(score):
    from openadapt_ml.training.grpo.config import GRPOConfig
    from openadapt_ml.training.grpo.rollout_collector import (
        GRPORolloutCollector,
        RolloutCollectionError,
    )

    collector = object.__new__(GRPORolloutCollector)
    collector._config = GRPOConfig(task_ids=["task-1"], num_rollouts_per_step=1)
    collector._task_configs = {}
    collector._env = MagicMock()
    collector._env.collect_rollout.return_value = [SimpleNamespace(reward=score)]

    with pytest.raises(RolloutCollectionError, match="invalid terminal score"):
        collector.collect_group(agent_fn=MagicMock(), task_id="task-1")


# ---------------------------------------------------------------------------
# 2. GRPO milestone reward: "could not evaluate" must not be reward 0.0
# ---------------------------------------------------------------------------


class _Check:
    def __init__(self, check="screenshot", description="a description"):
        self.check = check
        self.description = description


class _Milestone:
    def __init__(self, name="m", check=None):
        self.name = name
        self.check = check if check is not None else _Check()


class _TaskConfig:
    def __init__(self, milestones, task_id="task-1"):
        self.milestones = milestones
        self.id = task_id


def test_milestone_reward_raises_when_evals_package_is_missing(monkeypatch):
    """Missing openadapt-evals used to log a warning and return reward 0.0.

    0.0 is a legal reward. GRPO would have normalised it into an advantage and
    taken a gradient step on a measurement that never happened.
    """
    from openadapt_ml.training.grpo.reward import evaluate_milestones_screenshot

    MilestoneEvaluationError = _error_type(
        "openadapt_ml.training.grpo.reward", "MilestoneEvaluationError"
    )

    monkeypatch.setitem(sys.modules, "openadapt_evals.vlm_evaluator", None)
    config = _TaskConfig([_Milestone()])

    with pytest.raises(MilestoneEvaluationError, match="openadapt-evals"):
        evaluate_milestones_screenshot(config, _png_bytes())


def test_milestone_reward_raises_when_the_vlm_judge_fails(monkeypatch):
    """A judge exception used to be caught per-milestone and counted as a FAIL.

    The milestone stayed in the denominator, so an API timeout depressed the
    reward by exactly as much as a genuinely unmet milestone.
    """
    import types

    from openadapt_ml.training.grpo.reward import evaluate_milestones_screenshot

    MilestoneEvaluationError = _error_type(
        "openadapt_ml.training.grpo.reward", "MilestoneEvaluationError"
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("429 rate limited")

    fake = types.ModuleType("openadapt_evals.vlm_evaluator")
    fake.vlm_judge = _boom
    monkeypatch.setitem(sys.modules, "openadapt_evals", types.ModuleType("oe"))
    monkeypatch.setitem(sys.modules, "openadapt_evals.vlm_evaluator", fake)

    config = _TaskConfig([_Milestone("m1"), _Milestone("m2")])

    with pytest.raises(MilestoneEvaluationError, match="not a failed milestone"):
        evaluate_milestones_screenshot(config, _png_bytes())


def test_milestone_reward_raises_on_a_milestone_with_no_description(monkeypatch):
    """An undescribed milestone used to be `continue`d past.

    It stayed in `total`, so the task could never score 1.0 and nothing said so.
    """
    import types

    from openadapt_ml.training.grpo.reward import evaluate_milestones_screenshot

    MilestoneEvaluationError = _error_type(
        "openadapt_ml.training.grpo.reward", "MilestoneEvaluationError"
    )

    fake = types.ModuleType("openadapt_evals.vlm_evaluator")
    fake.vlm_judge = lambda *a, **k: (True, 1.0)
    monkeypatch.setitem(sys.modules, "openadapt_evals", types.ModuleType("oe"))
    monkeypatch.setitem(sys.modules, "openadapt_evals.vlm_evaluator", fake)

    config = _TaskConfig(
        [_Milestone("described"), _Milestone("blank", _Check(description=""))]
    )

    with pytest.raises(MilestoneEvaluationError, match="no description"):
        evaluate_milestones_screenshot(config, _png_bytes())


def test_milestone_reward_raises_when_there_is_nothing_locally_evaluable():
    """No milestones, or none of type 'screenshot', used to return 0.0."""
    from openadapt_ml.training.grpo.reward import evaluate_milestones_screenshot

    MilestoneEvaluationError = _error_type(
        "openadapt_ml.training.grpo.reward", "MilestoneEvaluationError"
    )

    with pytest.raises(MilestoneEvaluationError, match="no milestones"):
        evaluate_milestones_screenshot(_TaskConfig([]), _png_bytes())

    server_only = _TaskConfig([_Milestone("m", _Check(check="http"))])
    with pytest.raises(MilestoneEvaluationError, match="none of type 'screenshot'"):
        evaluate_milestones_screenshot(server_only, _png_bytes())


def test_milestone_reward_rejects_empty_screenshot():
    from openadapt_ml.training.grpo.reward import (
        MilestoneEvaluationError,
        evaluate_milestones_screenshot,
    )

    with pytest.raises(MilestoneEvaluationError, match="non-empty screenshot"):
        evaluate_milestones_screenshot(_TaskConfig([_Milestone()]), b"")


def test_milestone_reward_rejects_undecodable_screenshot():
    from openadapt_ml.training.grpo.reward import (
        MilestoneEvaluationError,
        evaluate_milestones_screenshot,
    )

    with pytest.raises(MilestoneEvaluationError, match="decodable image"):
        evaluate_milestones_screenshot(_TaskConfig([_Milestone()]), b"not-an-image")


@pytest.mark.parametrize(
    "judge_result, message",
    [
        ((1, 0.9), "non-boolean success"),
        ((True, float("nan")), "invalid confidence"),
        ((False, 1.1), "invalid confidence"),
    ],
)
def test_milestone_reward_rejects_invalid_judge_evidence(
    monkeypatch, judge_result, message
):
    import types

    from openadapt_ml.training.grpo.reward import (
        MilestoneEvaluationError,
        evaluate_milestones_screenshot,
    )

    fake = types.ModuleType("openadapt_evals.vlm_evaluator")
    fake.vlm_judge = lambda *args, **kwargs: judge_result
    monkeypatch.setitem(sys.modules, "openadapt_evals", types.ModuleType("oe"))
    monkeypatch.setitem(sys.modules, "openadapt_evals.vlm_evaluator", fake)

    with pytest.raises(MilestoneEvaluationError, match=message):
        evaluate_milestones_screenshot(_TaskConfig([_Milestone()]), _png_bytes())


def test_milestone_reward_still_returns_a_real_score(monkeypatch):
    """The honest path is unchanged: a measured 0.5 is still 0.5."""
    import types

    from openadapt_ml.training.grpo.reward import evaluate_milestones_screenshot

    calls = []

    def _judge(screenshot, description, **kwargs):
        calls.append(description)
        return (len(calls) == 1, 1.0)

    fake = types.ModuleType("openadapt_evals.vlm_evaluator")
    fake.vlm_judge = _judge
    monkeypatch.setitem(sys.modules, "openadapt_evals", types.ModuleType("oe"))
    monkeypatch.setitem(sys.modules, "openadapt_evals.vlm_evaluator", fake)

    config = _TaskConfig([_Milestone("a"), _Milestone("b")])
    assert evaluate_milestones_screenshot(config, _png_bytes()) == 0.5
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# 3. Grounding: an API failure must not be reported as "no candidates matched"
# ---------------------------------------------------------------------------


def test_gemini_grounder_raises_instead_of_returning_no_candidates(monkeypatch):
    """`ground()` used to print the error and return [].

    `evaluate_grounder` scores [] as best_iou 0.0 and centroid_hit False, so an
    unreachable API was recorded as a grounding miss the model never made.
    """
    from PIL import Image

    from openadapt_ml.grounding.detector import GeminiGrounder

    GroundingError = _error_type("openadapt_ml.grounding.base", "GroundingError")

    class _ExplodingModel:
        def generate_content(self, *args, **kwargs):
            raise RuntimeError("503 backend unavailable")

    grounder = GeminiGrounder(api_key="test-key")
    monkeypatch.setattr(grounder, "_get_model", lambda: _ExplodingModel())

    with pytest.raises(GroundingError, match="Gemini grounding call failed"):
        grounder.ground(Image.new("RGB", (32, 32)), "the login button")


def test_gemini_grounder_raises_on_an_unreadable_response(monkeypatch):
    """A reply with no JSON in it used to be indistinguishable from no match."""
    from PIL import Image

    from openadapt_ml.grounding.detector import GeminiGrounder

    GroundingError = _error_type("openadapt_ml.grounding.base", "GroundingError")

    class _Reply:
        text = "I'm sorry, I can't help with that."

    class _ChattyModel:
        def generate_content(self, *args, **kwargs):
            return _Reply()

    grounder = GeminiGrounder(api_key="test-key")
    monkeypatch.setattr(grounder, "_get_model", lambda: _ChattyModel())

    with pytest.raises(GroundingError):
        grounder.ground(Image.new("RGB", (32, 32)), "the login button")


def test_gemini_grounder_still_returns_an_empty_list_for_a_real_no_match(
    monkeypatch,
):
    """ "I looked and matched nothing" must keep returning []."""
    from PIL import Image

    from openadapt_ml.grounding.detector import GeminiGrounder

    class _Reply:
        text = "[]"

    class _EmptyModel:
        def generate_content(self, *args, **kwargs):
            return _Reply()

    grounder = GeminiGrounder(api_key="test-key")
    monkeypatch.setattr(grounder, "_get_model", lambda: _EmptyModel())

    assert grounder.ground(Image.new("RGB", (32, 32)), "the login button") == []


def test_extract_ui_elements_raises_instead_of_claiming_an_empty_screen(
    monkeypatch,
):
    """Both handlers used to print and return [], i.e. "no elements on screen"."""
    import types

    from PIL import Image

    from openadapt_ml.grounding.detector import extract_ui_elements

    GroundingError = _error_type("openadapt_ml.grounding.base", "GroundingError")

    class _ExplodingModel:
        def generate_content(self, *args, **kwargs):
            raise RuntimeError("quota exceeded")

    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda **kwargs: None
    fake_genai.GenerativeModel = lambda *a, **k: _ExplodingModel()
    fake_genai.GenerationConfig = lambda **kwargs: None

    # `import google.generativeai as genai` binds from the parent package's
    # attribute when `google` is already imported, so patching sys.modules
    # alone leaves the real client in place -- and the test would make a live
    # API call. Patch both.
    import google

    monkeypatch.setattr(google, "generativeai", fake_genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    with pytest.raises(GroundingError, match="element extraction failed"):
        extract_ui_elements(Image.new("RGB", (32, 32)), api_key="test-key")


# ---------------------------------------------------------------------------
# 4. Grounding metrics: an unreadable screenshot must not shrink the denominator
# ---------------------------------------------------------------------------


def test_evaluate_grounder_on_episode_refuses_an_unreadable_screenshot(tmp_path):
    """A step whose screenshot cannot be opened used to be `continue`d past.

    Every GroundingMetrics property divides by len(results), so the sample
    silently left the denominator and the reported hit rate stayed high.
    """
    from openadapt_ml.evals.grounding import evaluate_grounder_on_episode
    from openadapt_ml.grounding.base import GroundingModule
    from openadapt_ml.schema import (
        Action,
        ActionType,
        BoundingBox,
        Episode,
        Observation,
        Step,
        UIElement,
    )

    class _NeverCalled(GroundingModule):
        def ground(self, image, target_description, k=1):
            raise AssertionError("grounder must not run on a broken episode")

    missing = tmp_path / "not-a-png.png"
    missing.write_text("this is not an image")

    step = Step(
        step_index=0,
        observation=Observation(screenshot_path=str(missing)),
        action=Action(
            type=ActionType.CLICK,
            element=UIElement(
                bounds=BoundingBox(x=1, y=2, width=3, height=4),
            ),
        ),
    )
    episode = Episode(
        episode_id="ep-1",
        instruction="click the login button",
        steps=[step],
    )

    with pytest.raises(ValueError, match="could not be"):
        evaluate_grounder_on_episode(_NeverCalled(), episode)


# ---------------------------------------------------------------------------
# 5. Parquet summary: "wrote nothing" must not look like "wrote the summary"
# ---------------------------------------------------------------------------


def test_write_summary_raises_when_pyarrow_is_missing(monkeypatch, tmp_path):
    """`except ImportError: return` made include_summary=True a no-op.

    to_parquet() returned None either way, so the caller could not tell a
    summary file that was written from one that never was.
    """
    from openadapt_ml.export.parquet import _write_summary

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)

    with pytest.raises(ImportError, match="pyarrow"):
        _write_summary([], str(tmp_path / "episodes.parquet"))


# ---------------------------------------------------------------------------
# 6. Azure login probe: "I could not tell" must not be reported as True
# ---------------------------------------------------------------------------


def test_check_az_logged_in_does_not_guess_true(monkeypatch):
    """An unrecognised az error used to return True ("assume logged in")."""
    import importlib.util
    import subprocess

    spec = importlib.util.spec_from_file_location(
        "_setup_azure_under_test",
        Path(__file__).resolve().parent.parent / "scripts" / "setup_azure.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, "az", stderr="ERROR: the resource provider is not registered"
        )

    monkeypatch.setattr(module, "run_cmd", _fail)

    with pytest.raises(subprocess.CalledProcessError):
        module.check_az_logged_in()


def test_check_az_logged_in_still_reports_false_for_a_known_auth_error(monkeypatch):
    """The recognised "not logged in" answers are unchanged."""
    import importlib.util
    import subprocess

    spec = importlib.util.spec_from_file_location(
        "_setup_azure_under_test_2",
        Path(__file__).resolve().parent.parent / "scripts" / "setup_azure.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _expired(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, "az", stderr="AADSTS700082: the refresh token has expired"
        )

    monkeypatch.setattr(module, "run_cmd", _expired)
    assert module.check_az_logged_in() is False
