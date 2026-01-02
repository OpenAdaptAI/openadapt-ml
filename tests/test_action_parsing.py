"""Tests for action DSL parsing in AgentPolicy."""

import pytest

from openadapt_ml.runtime.policy import (
    _CLICK_RE,
    _TYPE_RE,
    _WAIT_RE,
    _DONE_RE,
    AgentPolicy,
)
from openadapt_ml.schema import Action, ActionType


class MockAdapter:
    """Mock adapter for testing parsing without a real model."""

    def __init__(self, response: str):
        self._response = response

    def generate(self, sample, max_new_tokens=64):
        return self._response


class TestActionRegexPatterns:
    """Test individual regex patterns."""

    def test_click_re_basic(self):
        m = _CLICK_RE.search("CLICK(x=0.5, y=0.3)")
        assert m is not None
        assert float(m.group(1)) == 0.5
        assert float(m.group(2)) == 0.3

    def test_click_re_with_context(self):
        m = _CLICK_RE.search("Action: CLICK(x=0.123, y=0.456)")
        assert m is not None
        assert float(m.group(1)) == 0.123
        assert float(m.group(2)) == 0.456

    def test_type_re_basic(self):
        m = _TYPE_RE.search('TYPE(text="hello")')
        assert m is not None
        assert m.group(1) == "hello"

    def test_type_re_with_spaces(self):
        m = _TYPE_RE.search('TYPE(text="hello world")')
        assert m is not None
        assert m.group(1) == "hello world"

    def test_type_re_with_escaped_quote(self):
        m = _TYPE_RE.search('TYPE(text="say \\"hi\\"")')
        assert m is not None
        assert m.group(1) == 'say \\"hi\\"'

    def test_wait_re(self):
        assert _WAIT_RE.search("WAIT()") is not None
        assert _WAIT_RE.search("Action: WAIT()") is not None
        assert _WAIT_RE.search("WAIT (  )") is not None

    def test_done_re(self):
        assert _DONE_RE.search("DONE()") is not None
        assert _DONE_RE.search("Action: DONE()") is not None
        assert _DONE_RE.search("DONE (  )") is not None


class TestAgentPolicyParsing:
    """Test AgentPolicy._parse_action method."""

    def test_parse_click(self):
        policy = AgentPolicy(MockAdapter(""))
        action = policy._parse_action("CLICK(x=0.5, y=0.3)")
        assert action.type == ActionType.CLICK
        assert action.normalized_coordinates == (0.5, 0.3)

    def test_parse_click_clamps_coords(self):
        policy = AgentPolicy(MockAdapter(""))
        action = policy._parse_action("CLICK(x=1.5, y=-0.2)")
        assert action.type == ActionType.CLICK
        assert action.normalized_coordinates == (1.0, 0.0)  # Clamped

    def test_parse_type(self):
        policy = AgentPolicy(MockAdapter(""))
        action = policy._parse_action('TYPE(text="hello")')
        assert action.type == ActionType.TYPE
        assert action.text == "hello"

    def test_parse_type_with_escaped_quotes(self):
        policy = AgentPolicy(MockAdapter(""))
        action = policy._parse_action('TYPE(text="say \\"hi\\"")')
        assert action.type == ActionType.TYPE
        assert action.text == 'say "hi"'

    def test_parse_wait(self):
        policy = AgentPolicy(MockAdapter(""))
        action = policy._parse_action("WAIT()")
        assert action.type == ActionType.WAIT

    def test_parse_done(self):
        policy = AgentPolicy(MockAdapter(""))
        action = policy._parse_action("DONE()")
        assert action.type == ActionType.DONE

    def test_parse_failed_invalid(self):
        policy = AgentPolicy(MockAdapter(""))
        action = policy._parse_action("some random text")
        assert action.type == ActionType.FAIL
        assert action.raw == {"text": "some random text"}

    def test_parse_with_thought_action_format(self):
        """Test parsing when model outputs Thought: ... Action: ... format."""
        response = """Thought: I need to click the login button.
Action: CLICK(x=0.5, y=0.7)"""
        policy = AgentPolicy(MockAdapter(response))
        action, thought, state, raw_text = policy.predict_action_from_sample({})
        assert action.type == ActionType.CLICK
        assert action.normalized_coordinates == (0.5, 0.7)
        assert thought is not None
        assert "click the login button" in thought

    def test_parse_type_in_thought_action_format(self):
        """Test TYPE parsing with Thought/Action format."""
        response = """Thought: I need to enter the username.
Action: TYPE(text="user123")"""
        policy = AgentPolicy(MockAdapter(response))
        action, thought, state, raw_text = policy.predict_action_from_sample({})
        assert action.type == ActionType.TYPE
        assert action.text == "user123"

    def test_parse_wait_in_thought_action_format(self):
        """Test WAIT parsing with Thought/Action format."""
        response = """Thought: The page is loading.
Action: WAIT()"""
        policy = AgentPolicy(MockAdapter(response))
        action, thought, state, raw_text = policy.predict_action_from_sample({})
        assert action.type == ActionType.WAIT
