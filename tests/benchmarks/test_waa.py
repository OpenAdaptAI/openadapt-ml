"""Tests for WAA benchmark adapter."""

import pytest

from openadapt_ml.benchmarks import (
    BenchmarkAction,
    BenchmarkObservation,
    BenchmarkResult,
    BenchmarkTask,
    RandomAgent,
    ScriptedAgent,
    WAAMockAdapter,
    compute_metrics,
    evaluate_agent_on_benchmark,
)


class TestWAAMockAdapter:
    """Tests for WAAMockAdapter."""

    def test_list_tasks(self):
        """Test listing tasks."""
        adapter = WAAMockAdapter(num_tasks=10)
        tasks = adapter.list_tasks()
        assert len(tasks) == 10
        assert all(isinstance(t, BenchmarkTask) for t in tasks)

    def test_list_tasks_by_domain(self):
        """Test filtering tasks by domain."""
        adapter = WAAMockAdapter(num_tasks=10, domains=["browser", "notepad"])
        browser_tasks = adapter.list_tasks(domain="browser")
        assert len(browser_tasks) == 5
        assert all(t.domain == "browser" for t in browser_tasks)

    def test_load_task(self):
        """Test loading a specific task."""
        adapter = WAAMockAdapter(num_tasks=5, domains=["browser"])
        task = adapter.load_task("browser_1")
        assert task.task_id == "browser_1"
        assert task.domain == "browser"

    def test_load_task_not_found(self):
        """Test loading a non-existent task."""
        adapter = WAAMockAdapter(num_tasks=5)
        with pytest.raises(KeyError):
            adapter.load_task("nonexistent_task")

    def test_reset_returns_observation(self):
        """Test reset returns observation."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.list_tasks()[0]
        obs = adapter.reset(task)
        assert isinstance(obs, BenchmarkObservation)
        assert obs.viewport is not None
        assert obs.accessibility_tree is not None

    def test_step_returns_observation_done_info(self):
        """Test step returns correct tuple."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.list_tasks()[0]
        adapter.reset(task)

        action = BenchmarkAction(type="click", x=0.5, y=0.5)
        obs, done, info = adapter.step(action)

        assert isinstance(obs, BenchmarkObservation)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_done_on_max_steps(self):
        """Test step returns done after max steps."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.list_tasks()[0]
        adapter.reset(task)

        action = BenchmarkAction(type="click", x=0.5, y=0.5)
        for _ in range(14):
            obs, done, info = adapter.step(action)
            assert not done

        obs, done, info = adapter.step(action)
        assert done

    def test_step_done_on_done_action(self):
        """Test step returns done when action is 'done'."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.list_tasks()[0]
        adapter.reset(task)

        action = BenchmarkAction(type="done")
        obs, done, info = adapter.step(action)
        assert done

    def test_evaluate_returns_result(self):
        """Test evaluate returns BenchmarkResult."""
        adapter = WAAMockAdapter(num_tasks=5)
        task = adapter.list_tasks()[0]
        adapter.reset(task)

        result = adapter.evaluate(task)
        assert isinstance(result, BenchmarkResult)
        assert result.task_id == task.task_id
        assert result.score in [0.0, 1.0]

    def test_benchmark_properties(self):
        """Test benchmark properties."""
        adapter = WAAMockAdapter()
        assert adapter.name == "waa-mock"
        assert adapter.benchmark_type == "interactive"


class TestEvaluationRunner:
    """Tests for evaluation runner with mock adapter."""

    def test_evaluate_with_random_agent(self):
        """Test running evaluation with RandomAgent."""
        adapter = WAAMockAdapter(num_tasks=5)
        agent = RandomAgent(seed=42)

        results = evaluate_agent_on_benchmark(agent, adapter, max_steps=10)

        assert len(results) == 5
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_evaluate_specific_tasks(self):
        """Test evaluating specific tasks."""
        adapter = WAAMockAdapter(num_tasks=10)
        agent = RandomAgent(seed=42)

        results = evaluate_agent_on_benchmark(
            agent, adapter, task_ids=["browser_1", "browser_2"], max_steps=10
        )

        assert len(results) == 2
        assert results[0].task_id == "browser_1"
        assert results[1].task_id == "browser_2"

    def test_evaluate_with_scripted_agent(self):
        """Test running evaluation with ScriptedAgent."""
        adapter = WAAMockAdapter(num_tasks=3)

        # Create script that clicks then says done
        actions = [
            BenchmarkAction(type="click", x=0.5, y=0.5),
            BenchmarkAction(type="type", text="hello"),
            BenchmarkAction(type="done"),
        ]
        agent = ScriptedAgent(actions=actions)

        results = evaluate_agent_on_benchmark(agent, adapter, max_steps=10)

        assert len(results) == 3


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_empty(self):
        """Test metrics with empty results."""
        metrics = compute_metrics([])
        assert metrics["num_tasks"] == 0
        assert metrics["success_rate"] == 0.0

    def test_compute_metrics_all_success(self):
        """Test metrics with all successful tasks."""
        results = [
            BenchmarkResult(task_id=f"task_{i}", success=True, score=1.0, num_steps=5)
            for i in range(3)
        ]
        metrics = compute_metrics(results)
        assert metrics["num_tasks"] == 3
        assert metrics["success_rate"] == 1.0
        assert metrics["avg_score"] == 1.0

    def test_compute_metrics_all_fail(self):
        """Test metrics with all failed tasks."""
        results = [
            BenchmarkResult(task_id=f"task_{i}", success=False, score=0.0, num_steps=10)
            for i in range(3)
        ]
        metrics = compute_metrics(results)
        assert metrics["num_tasks"] == 3
        assert metrics["success_rate"] == 0.0
        assert metrics["avg_score"] == 0.0

    def test_compute_metrics_partial_success(self):
        """Test metrics with partial success."""
        results = [
            BenchmarkResult(task_id="task_1", success=True, score=1.0, num_steps=5),
            BenchmarkResult(task_id="task_2", success=False, score=0.0, num_steps=10),
            BenchmarkResult(task_id="task_3", success=True, score=1.0, num_steps=3),
            BenchmarkResult(task_id="task_4", success=False, score=0.0, num_steps=8),
        ]
        metrics = compute_metrics(results)
        assert metrics["num_tasks"] == 4
        assert metrics["success_rate"] == 0.5
        assert metrics["success_count"] == 2
        assert metrics["fail_count"] == 2
