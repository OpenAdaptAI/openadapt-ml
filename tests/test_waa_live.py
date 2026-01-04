"""Tests for WAALiveAdapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from openadapt_ml.benchmarks.waa_live import WAALiveAdapter, WAALiveConfig
from openadapt_ml.benchmarks.base import BenchmarkAction, BenchmarkTask


class TestWAALiveConfig:
    """Tests for WAALiveConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WAALiveConfig()
        assert config.server_url == "http://localhost:5000"
        assert config.a11y_backend == "uia"
        assert config.screen_width == 1920
        assert config.screen_height == 1200
        assert config.max_steps == 15
        assert config.action_delay == 0.5
        assert config.timeout == 90.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = WAALiveConfig(
            server_url="http://192.168.1.100:5000",
            max_steps=20,
            action_delay=1.0,
        )
        assert config.server_url == "http://192.168.1.100:5000"
        assert config.max_steps == 20
        assert config.action_delay == 1.0


class TestWAALiveAdapter:
    """Tests for WAALiveAdapter."""

    def test_adapter_properties(self):
        """Test adapter properties."""
        adapter = WAALiveAdapter()
        assert adapter.name == "waa-live"
        assert adapter.benchmark_type == "interactive"
        assert adapter.supports_parallel is False

    @patch("openadapt_ml.benchmarks.waa_live.requests")
    def test_check_connection_success(self, mock_requests):
        """Test successful connection check."""
        mock_requests.get.return_value = Mock(status_code=200)

        adapter = WAALiveAdapter()
        assert adapter.check_connection() is True

        mock_requests.get.assert_called_once()

    @patch("openadapt_ml.benchmarks.waa_live.requests.get")
    def test_check_connection_failure(self, mock_get):
        """Test failed connection check."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")

        adapter = WAALiveAdapter()
        assert adapter.check_connection() is False


class TestActionTranslation:
    """Tests for action translation.

    The adapter uses element-based grounding via WAA's Computer class:
    - Click actions use computer.mouse.move_id(id) for element grounding
    - Keyboard actions use pyautogui (no grounding needed)
    - Scroll uses computer.mouse.scroll()
    """

    def test_click_with_element_id(self):
        """Test click with element ID uses move_id for grounding."""
        adapter = WAALiveAdapter()
        adapter._current_rects = {"5": [100, 200, 300, 400]}

        action = BenchmarkAction(type="click", target_node_id="5")
        command = adapter._translate_action(action)

        assert "computer.mouse.move_id('5')" in command
        assert "computer.mouse.single_click()" in command

    def test_click_fallback_to_coords(self):
        """Test click falls back to move_abs when no element ID."""
        adapter = WAALiveAdapter()
        adapter._current_rects = {}

        action = BenchmarkAction(type="click", x=500, y=300)
        command = adapter._translate_action(action)

        assert "computer.mouse.move_abs(500, 300)" in command
        assert "computer.mouse.single_click()" in command

    def test_click_normalized_coords_fallback(self):
        """Test click with normalized coordinates falls back to move_abs."""
        adapter = WAALiveAdapter(WAALiveConfig(screen_width=1920, screen_height=1080))
        adapter._current_rects = {}

        action = BenchmarkAction(type="click", x=0.5, y=0.5)
        command = adapter._translate_action(action)

        # Normalized coords passed to move_abs (WAA handles conversion)
        assert "computer.mouse.move_abs(0.5, 0.5)" in command
        assert "computer.mouse.single_click()" in command

    def test_double_click_with_element_id(self):
        """Test double click with element ID."""
        adapter = WAALiveAdapter()
        adapter._current_rects = {"7": [0, 0, 100, 50]}

        action = BenchmarkAction(type="double_click", target_node_id="7")
        command = adapter._translate_action(action)

        assert "computer.mouse.move_id('7')" in command
        assert "computer.mouse.double_click()" in command

    def test_type_action(self):
        """Test type action uses pyautogui (no grounding needed)."""
        adapter = WAALiveAdapter()

        action = BenchmarkAction(type="type", text="Hello World")
        command = adapter._translate_action(action)

        assert "pyautogui.write('Hello World'" in command

    def test_type_action_with_quotes(self):
        """Test type action with quotes escaped."""
        adapter = WAALiveAdapter()

        action = BenchmarkAction(type="type", text="It's a \"test\"")
        command = adapter._translate_action(action)

        # Should escape single quotes
        assert "\\'" in command

    def test_key_action(self):
        """Test key action uses pyautogui (no grounding needed)."""
        adapter = WAALiveAdapter()

        action = BenchmarkAction(type="key", key="Enter")
        command = adapter._translate_action(action)

        assert "pyautogui.press('enter')" in command

    def test_key_action_with_modifiers(self):
        """Test key action with modifiers."""
        adapter = WAALiveAdapter()

        action = BenchmarkAction(type="key", key="c", modifiers=["Control"])
        command = adapter._translate_action(action)

        assert "pyautogui.hotkey('ctrl', 'c')" in command

    def test_scroll_action_down(self):
        """Test scroll down uses computer.mouse.scroll."""
        adapter = WAALiveAdapter()

        action = BenchmarkAction(type="scroll", scroll_direction="down", scroll_amount=5)
        command = adapter._translate_action(action)

        assert "computer.mouse.scroll('down')" in command

    def test_scroll_action_up(self):
        """Test scroll up uses computer.mouse.scroll."""
        adapter = WAALiveAdapter()

        action = BenchmarkAction(type="scroll", scroll_direction="up", scroll_amount=3)
        command = adapter._translate_action(action)

        assert "computer.mouse.scroll('up')" in command

    def test_done_action(self):
        """Test done action returns None."""
        adapter = WAALiveAdapter()

        action = BenchmarkAction(type="done")
        command = adapter._translate_action(action)

        assert command is None

    def test_wait_action(self):
        """Test wait action."""
        adapter = WAALiveAdapter()

        action = BenchmarkAction(type="wait")
        command = adapter._translate_action(action)

        assert "time.sleep(1)" in command


class TestRectExtraction:
    """Tests for extracting element rects from a11y tree.

    The adapter extracts element IDs and bboxes from the a11y tree
    and sends them to WAA via /update_computer. WAA then handles
    the actual grounding when computer.mouse.move_id(id) is called.
    """

    def test_extract_rects_simple(self):
        """Test extracting rects from simple a11y tree."""
        adapter = WAALiveAdapter()
        a11y_tree = {
            "id": "root",
            "children": [
                {
                    "id": "5",
                    "bbox": [100, 200, 300, 400],
                }
            ]
        }

        rects = adapter._extract_rects_from_a11y(a11y_tree)

        assert "5" in rects
        assert rects["5"] == [100, 200, 300, 400]

    def test_extract_rects_nested(self):
        """Test extracting rects from nested a11y tree."""
        adapter = WAALiveAdapter()
        a11y_tree = {
            "id": "root",
            "children": [
                {
                    "id": "1",
                    "bbox": [0, 0, 500, 500],
                    "children": [
                        {
                            "id": "3",
                            "bbox": [50, 50, 150, 100],
                        }
                    ]
                }
            ]
        }

        rects = adapter._extract_rects_from_a11y(a11y_tree)

        assert "root" in rects or "1" in rects  # Depends on if root has bbox
        assert "1" in rects
        assert "3" in rects
        assert rects["3"] == [50, 50, 150, 100]

    def test_extract_rects_empty_tree(self):
        """Test extracting rects from empty a11y tree."""
        adapter = WAALiveAdapter()

        rects = adapter._extract_rects_from_a11y(None)
        assert rects == {}

        rects = adapter._extract_rects_from_a11y({})
        assert rects == {}

    def test_extract_rects_no_bbox(self):
        """Test elements without bbox are skipped."""
        adapter = WAALiveAdapter()
        a11y_tree = {
            "id": "root",
            "children": [{"id": "5", "name": "Button"}]  # No bbox
        }

        rects = adapter._extract_rects_from_a11y(a11y_tree)

        # Element without bbox should not be in rects
        assert "5" not in rects

    def test_click_element_not_in_rects_warns(self):
        """Test click with unknown element ID logs warning and uses coords."""
        adapter = WAALiveAdapter()
        adapter._current_rects = {"1": [0, 0, 100, 100]}  # Element 7 not here

        action = BenchmarkAction(
            type="click",
            target_node_id="7",  # Not in rects
            x=999, y=999
        )
        command = adapter._translate_action(action)

        # Should fall back to coordinate-based click
        assert "move_abs" in command
        assert "999" in command


class TestObservationFetching:
    """Tests for observation fetching."""

    @patch("openadapt_ml.benchmarks.waa_live.requests")
    def test_get_observation(self, mock_requests):
        """Test fetching observation from server."""
        # Mock screenshot response
        screenshot_response = Mock()
        screenshot_response.status_code = 200
        screenshot_response.content = b"fake_png_data"

        # Mock a11y response
        a11y_response = Mock()
        a11y_response.status_code = 200
        a11y_response.json.return_value = {"AT": {"id": "root"}}

        mock_requests.get.side_effect = [screenshot_response, a11y_response]

        adapter = WAALiveAdapter()
        obs = adapter._get_observation()

        assert obs.screenshot == b"fake_png_data"
        assert obs.accessibility_tree == {"id": "root"}
        assert obs.viewport == (1920, 1200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
