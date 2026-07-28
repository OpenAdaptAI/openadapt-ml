"""Agent infrastructure and parse failures must not look like task completion."""

from __future__ import annotations

from openadapt_types import BenchmarkObservation, BenchmarkTask

from openadapt_ml.baselines import ParsedAction
from openadapt_ml.benchmarks.agent import APIBenchmarkAgent, UnifiedBaselineAgent
from openadapt_ml.experiments.waa_demo.runner import DemoConditionedAgent


class _FailingAdapter:
    def generate(self, *args, **kwargs):
        raise RuntimeError("provider unavailable")

    def predict(self, *args, **kwargs):
        raise RuntimeError("provider unavailable")


def _observation_and_task():
    return BenchmarkObservation(), BenchmarkTask(
        task_id="task-1", instruction="Complete the task", domain="desktop"
    )


def test_api_agent_preserves_provider_and_parse_failures():
    observation, task = _observation_and_task()
    agent = APIBenchmarkAgent()
    agent._adapter = _FailingAdapter()

    assert agent.act(observation, task).type == "error"
    assert agent._parse_response("not an action").type == "error"
    assert agent._parse_response("ACTION: UNKNOWN(foo)").type == "error"


def test_unified_agent_preserves_provider_and_parse_failures():
    observation, task = _observation_and_task()
    agent = UnifiedBaselineAgent()
    agent._adapter = _FailingAdapter()

    assert agent.act(observation, task).type == "error"
    invalid = ParsedAction(action_type="unknown", parse_error="not an action")
    assert agent._parsed_to_benchmark_action(invalid).type == "error"


def test_demo_agent_preserves_provider_and_parse_failures():
    observation, task = _observation_and_task()
    agent = DemoConditionedAgent()
    agent._adapter = _FailingAdapter()

    assert agent.act(observation, task).type == "error"
    assert agent._parse_response("not an action").type == "error"
    assert agent._parse_response("ACTION: UNKNOWN(foo)").type == "error"
