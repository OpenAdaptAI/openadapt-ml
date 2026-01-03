"""Unit tests for WAA demo experiment module.

Tests for:
- tasks.py: Task definitions and helper functions
- demos.py: Demo content and retrieval functions
- Integration: Consistency between tasks and demos
"""

from __future__ import annotations

import pytest

from openadapt_ml.experiments.waa_demo.tasks import (
    TASKS,
    Difficulty,
    Domain,
    WATask,
    get_manual_tasks,
    get_recorded_tasks,
    get_task,
    get_tasks_by_method,
)
from openadapt_ml.experiments.waa_demo.demos import (
    DEMOS,
    format_demo_for_prompt,
    get_complete_demos,
    get_demo,
    get_placeholder_demos,
)


# =============================================================================
# TASKS.PY TESTS
# =============================================================================

class TestTaskDefinitions:
    """Tests for task definitions in tasks.py."""

    def test_all_10_tasks_defined(self):
        """Test that all 10 tasks are defined in TASKS dict."""
        assert len(TASKS) == 10
        # Check that tasks 1-10 exist
        for i in range(1, 11):
            assert str(i) in TASKS, f"Task {i} is missing from TASKS"

    def test_get_task_returns_correct_task_for_valid_numbers(self):
        """Test get_task() returns correct task for valid task numbers."""
        # Test with string
        task_1 = get_task("1")
        assert task_1 is not None
        assert isinstance(task_1, WATask)
        assert task_1.instruction == "Enable the 'Do Not Track' feature in Edge"

        # Test with integer
        task_5 = get_task(5)
        assert task_5 is not None
        assert isinstance(task_5, WATask)
        assert "Calculate monthly totals" in task_5.instruction

        # Test task 10
        task_10 = get_task(10)
        assert task_10 is not None
        assert "Details view" in task_10.instruction

    def test_get_task_returns_none_for_invalid_numbers(self):
        """Test get_task() returns None for invalid task numbers."""
        # Out of range
        assert get_task(0) is None
        assert get_task(11) is None
        assert get_task(-1) is None
        assert get_task(100) is None

        # Invalid string
        assert get_task("0") is None
        assert get_task("11") is None
        assert get_task("invalid") is None
        assert get_task("") is None

    def test_get_manual_tasks_returns_7_tasks(self):
        """Test get_manual_tasks() returns 7 tasks."""
        manual_tasks = get_manual_tasks()
        assert len(manual_tasks) == 7
        for task in manual_tasks:
            assert task.demo_method == "manual"

    def test_get_recorded_tasks_returns_3_tasks(self):
        """Test get_recorded_tasks() returns 3 tasks."""
        recorded_tasks = get_recorded_tasks()
        assert len(recorded_tasks) == 3
        for task in recorded_tasks:
            assert task.demo_method == "recorded"

    def test_task_fields_are_populated(self):
        """Test that all task fields are populated properly."""
        for task_num, task in TASKS.items():
            # Test instruction is non-empty string
            assert task.instruction, f"Task {task_num} has empty instruction"
            assert isinstance(task.instruction, str)
            assert len(task.instruction) > 5

            # Test domain is valid enum
            assert task.domain, f"Task {task_num} has no domain"
            assert isinstance(task.domain, Domain)

            # Test difficulty is valid enum
            assert task.difficulty, f"Task {task_num} has no difficulty"
            assert isinstance(task.difficulty, Difficulty)

            # Test task_id is non-empty
            assert task.task_id, f"Task {task_num} has empty task_id"

            # Test first_action_hint is non-empty
            assert task.first_action_hint, f"Task {task_num} has empty first_action_hint"

            # Test demo_method is valid
            assert task.demo_method in ("manual", "recorded"), \
                f"Task {task_num} has invalid demo_method: {task.demo_method}"

            # Test json_path is non-empty
            assert task.json_path, f"Task {task_num} has empty json_path"

    def test_get_tasks_by_method(self):
        """Test get_tasks_by_method() helper function."""
        manual = get_tasks_by_method("manual")
        recorded = get_tasks_by_method("recorded")
        invalid = get_tasks_by_method("invalid")

        assert len(manual) == 7
        assert len(recorded) == 3
        assert len(invalid) == 0

    def test_task_domains_coverage(self):
        """Test that tasks cover multiple domains."""
        domains = {task.domain for task in TASKS.values()}
        # Should have at least 4 different domains
        assert len(domains) >= 4

    def test_task_difficulties_coverage(self):
        """Test that tasks include various difficulty levels."""
        difficulties = {task.difficulty for task in TASKS.values()}
        # Should have all three difficulty levels
        assert Difficulty.EASY in difficulties
        assert Difficulty.MEDIUM in difficulties
        assert Difficulty.HARD in difficulties


# =============================================================================
# DEMOS.PY TESTS
# =============================================================================

class TestDemoDefinitions:
    """Tests for demo definitions in demos.py."""

    def test_all_10_demos_exist(self):
        """Test that all 10 demos exist in DEMOS dict."""
        assert len(DEMOS) == 10
        # Check that demos 1-10 exist
        for i in range(1, 11):
            assert str(i) in DEMOS, f"Demo {i} is missing from DEMOS"

    def test_get_demo_returns_string_for_valid_task_numbers(self):
        """Test get_demo() returns string for valid task numbers."""
        # Test with string
        demo_1 = get_demo("1")
        assert demo_1 is not None
        assert isinstance(demo_1, str)
        assert len(demo_1) > 100  # Should be substantial content

        # Test with integer
        demo_5 = get_demo(5)
        assert demo_5 is not None
        assert isinstance(demo_5, str)

        # Test task 10
        demo_10 = get_demo(10)
        assert demo_10 is not None
        assert isinstance(demo_10, str)

    def test_get_demo_returns_none_for_invalid_task_numbers(self):
        """Test get_demo() returns None for invalid task numbers."""
        assert get_demo(0) is None
        assert get_demo(11) is None
        assert get_demo(-1) is None
        assert get_demo("invalid") is None
        assert get_demo("") is None

    def test_get_complete_demos_returns_7_demos(self):
        """Test get_complete_demos() returns 7 demos (non-placeholder)."""
        complete_demos = get_complete_demos()
        assert len(complete_demos) == 7

        for task_num, demo in complete_demos.items():
            assert "[PLACEHOLDER" not in demo, \
                f"Demo {task_num} is a placeholder but returned as complete"

    def test_get_placeholder_demos_returns_3_demos(self):
        """Test get_placeholder_demos() returns 3 demos (with [PLACEHOLDER])."""
        placeholder_demos = get_placeholder_demos()
        assert len(placeholder_demos) == 3

        for task_num, demo in placeholder_demos.items():
            assert "[PLACEHOLDER" in demo, \
                f"Demo {task_num} should contain [PLACEHOLDER] marker"

    def test_placeholder_demo_tasks(self):
        """Test that the correct tasks have placeholder demos."""
        placeholder_demos = get_placeholder_demos()
        # Tasks 4, 5, and 9 should be placeholders (recorded demos)
        expected_placeholders = {"4", "5", "9"}
        actual_placeholders = set(placeholder_demos.keys())
        assert actual_placeholders == expected_placeholders

    def test_complete_demo_tasks(self):
        """Test that the correct tasks have complete demos."""
        complete_demos = get_complete_demos()
        # Tasks 1, 2, 3, 6, 7, 8, 10 should be complete (manual demos)
        expected_complete = {"1", "2", "3", "6", "7", "8", "10"}
        actual_complete = set(complete_demos.keys())
        assert actual_complete == expected_complete

    def test_demo_content_has_demonstration_header(self):
        """Test that all demos start with DEMONSTRATION: header."""
        for task_num, demo in DEMOS.items():
            assert demo.strip().startswith("DEMONSTRATION:"), \
                f"Demo {task_num} should start with 'DEMONSTRATION:'"

    def test_demo_content_has_goal(self):
        """Test that all demos contain a Goal line."""
        for task_num, demo in DEMOS.items():
            assert "Goal:" in demo, f"Demo {task_num} should contain 'Goal:'"

    def test_format_demo_for_prompt(self):
        """Test format_demo_for_prompt() generates valid prompt text."""
        demo = get_demo("1")
        task_instruction = "Complete a test task"

        formatted = format_demo_for_prompt(demo, task_instruction)

        # Check that it contains the demo
        assert demo in formatted

        # Check that it contains the task instruction
        assert task_instruction in formatted

        # Check for context header
        assert "demonstration shows how to complete" in formatted.lower()

        # Check for "Now complete this task" directive
        assert "Now complete this task:" in formatted

    def test_format_demo_for_prompt_with_all_demos(self):
        """Test format_demo_for_prompt() works with all demos."""
        for task_num in DEMOS.keys():
            demo = get_demo(task_num)
            task = get_task(task_num)
            formatted = format_demo_for_prompt(demo, task.instruction)

            assert isinstance(formatted, str)
            assert len(formatted) > len(demo)
            assert task.instruction in formatted


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestTaskDemoIntegration:
    """Integration tests for consistency between tasks and demos."""

    def test_task_numbers_match_between_tasks_and_demos(self):
        """Test that task numbers match between TASKS and DEMOS dicts."""
        task_keys = set(TASKS.keys())
        demo_keys = set(DEMOS.keys())

        assert task_keys == demo_keys, \
            f"Mismatch between task and demo keys. " \
            f"Only in TASKS: {task_keys - demo_keys}, " \
            f"Only in DEMOS: {demo_keys - task_keys}"

    def test_demo_method_matches_placeholder_status(self):
        """Test that demo_method in tasks matches placeholder status in demos."""
        for task_num, task in TASKS.items():
            demo = DEMOS[task_num]
            is_placeholder = "[PLACEHOLDER" in demo

            if task.demo_method == "manual":
                assert not is_placeholder, \
                    f"Task {task_num} has demo_method='manual' but demo is a placeholder"
            elif task.demo_method == "recorded":
                assert is_placeholder, \
                    f"Task {task_num} has demo_method='recorded' but demo is not a placeholder"

    def test_demo_goal_matches_task_instruction(self):
        """Test that demo Goal matches task instruction."""
        for task_num, task in TASKS.items():
            demo = DEMOS[task_num]
            # Extract the goal line from the demo
            goal_line = None
            for line in demo.split("\n"):
                if line.strip().startswith("Goal:"):
                    goal_line = line
                    break

            assert goal_line is not None, \
                f"Demo {task_num} missing Goal line"

            # The goal should match or be similar to the task instruction
            # Allow for slight differences in wording
            goal_text = goal_line.replace("Goal:", "").strip()
            # Check that key words from instruction appear in goal
            # (exact match not required, but should be related)
            instruction_words = set(task.instruction.lower().split())
            goal_words = set(goal_text.lower().split())
            # At least 2 significant words should match
            common_words = instruction_words & goal_words
            # Exclude common stop words
            stop_words = {"the", "a", "an", "in", "to", "for", "and", "or", "on", "at", "of"}
            significant_common = common_words - stop_words
            assert len(significant_common) >= 1, \
                f"Task {task_num}: Goal '{goal_text}' doesn't match instruction '{task.instruction}'"

    def test_manual_and_recorded_counts_consistency(self):
        """Test that manual + recorded tasks = total tasks = 10."""
        manual_tasks = get_manual_tasks()
        recorded_tasks = get_recorded_tasks()

        assert len(manual_tasks) + len(recorded_tasks) == 10
        assert len(manual_tasks) + len(recorded_tasks) == len(TASKS)

    def test_complete_and_placeholder_demos_consistency(self):
        """Test that complete + placeholder demos = total demos = 10."""
        complete_demos = get_complete_demos()
        placeholder_demos = get_placeholder_demos()

        assert len(complete_demos) + len(placeholder_demos) == 10
        assert len(complete_demos) + len(placeholder_demos) == len(DEMOS)

    def test_no_overlap_between_complete_and_placeholder_demos(self):
        """Test that no demo is in both complete and placeholder sets."""
        complete_keys = set(get_complete_demos().keys())
        placeholder_keys = set(get_placeholder_demos().keys())

        overlap = complete_keys & placeholder_keys
        assert len(overlap) == 0, f"Overlap between complete and placeholder demos: {overlap}"

    def test_task_retrieval_roundtrip(self):
        """Test that tasks can be retrieved and match original data."""
        for task_num in TASKS.keys():
            retrieved_task = get_task(task_num)
            original_task = TASKS[task_num]
            assert retrieved_task is original_task

    def test_demo_retrieval_roundtrip(self):
        """Test that demos can be retrieved and match original data."""
        for task_num in DEMOS.keys():
            retrieved_demo = get_demo(task_num)
            original_demo = DEMOS[task_num]
            assert retrieved_demo is original_demo
