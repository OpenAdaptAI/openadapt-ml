"""Unit tests for convert_demos module.

Tests for:
- Action parsing (_parse_action_raw)
- Coordinate conversion (_coord_01_to_1000)
- Action formatting (_format_action_qwen)
- Think block building (_build_think_block)
- Step conversion (convert_step)
- Demo validation (_validate_demo)
- Bundle creation (create_bundle)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from openadapt_ml.training.convert_demos import (
    SYSTEM_PROMPT,
    _build_think_block,
    _coord_01_to_1000,
    _format_action_qwen,
    _parse_action_raw,
    _validate_coord,
    _validate_demo,
    convert_step,
    create_bundle,
    generate_screenshot_mapping,
    prepare_bundle,
)


class TestParseActionRaw:
    """Test action parsing from raw annotated demo format."""

    def test_click(self):
        action_type, params = _parse_action_raw("CLICK(0.294, 0.532)")
        assert action_type == "click"
        assert params["x"] == pytest.approx(0.294)
        assert params["y"] == pytest.approx(0.532)

    def test_click_case_insensitive(self):
        action_type, params = _parse_action_raw("click(0.5, 0.5)")
        assert action_type == "click"

    def test_double_click(self):
        action_type, params = _parse_action_raw("DOUBLE_CLICK(0.1, 0.9)")
        assert action_type == "double_click"
        assert params["x"] == pytest.approx(0.1)
        assert params["y"] == pytest.approx(0.9)

    def test_right_click(self):
        action_type, params = _parse_action_raw("RIGHT_CLICK(0.5, 0.5)")
        assert action_type == "right_click"

    def test_type_with_quotes(self):
        action_type, params = _parse_action_raw('TYPE("hello world")')
        assert action_type == "type"
        assert params["text"] == "hello world"

    def test_type_without_quotes(self):
        action_type, params = _parse_action_raw("TYPE(hello)")
        assert action_type == "type"
        assert params["text"] == "hello"

    def test_key(self):
        action_type, params = _parse_action_raw("KEY(Enter)")
        assert action_type == "press"
        assert params["keys"] == "Enter"

    def test_press(self):
        action_type, params = _parse_action_raw("PRESS(Ctrl+C)")
        assert action_type == "press"
        assert params["keys"] == "Ctrl+C"

    def test_scroll(self):
        action_type, params = _parse_action_raw("SCROLL(down)")
        assert action_type == "scroll"
        assert params["direction"] == "down"

    def test_drag(self):
        action_type, params = _parse_action_raw("DRAG(0.1, 0.2, 0.8, 0.9)")
        assert action_type == "drag"
        assert params["from_x"] == pytest.approx(0.1)
        assert params["from_y"] == pytest.approx(0.2)
        assert params["to_x"] == pytest.approx(0.8)
        assert params["to_y"] == pytest.approx(0.9)

    def test_done(self):
        action_type, _ = _parse_action_raw("DONE()")
        assert action_type == "done"

    def test_finished(self):
        action_type, _ = _parse_action_raw("FINISHED()")
        assert action_type == "finished"

    def test_wait(self):
        action_type, _ = _parse_action_raw("WAIT()")
        assert action_type == "wait"

    def test_unknown(self):
        action_type, params = _parse_action_raw("UNKNOWN_ACTION(foo)")
        assert action_type == "unknown"
        assert "raw" in params


class TestCoordConversion:
    """Test coordinate conversion from [0,1] to [0,1000]."""

    def test_zero(self):
        assert _coord_01_to_1000(0.0) == 0

    def test_one(self):
        assert _coord_01_to_1000(1.0) == 1000

    def test_half(self):
        assert _coord_01_to_1000(0.5) == 500

    def test_typical(self):
        assert _coord_01_to_1000(0.294) == 294

    def test_rounding(self):
        # 0.2945 * 1000 = 294.5, round() uses banker's rounding -> 294
        assert _coord_01_to_1000(0.2945) == 294
        # 0.2955 * 1000 = 295.5, banker's rounding -> 296
        assert _coord_01_to_1000(0.2955) == 296


class TestValidateCoord:
    """Test coordinate validation."""

    def test_valid(self):
        assert _validate_coord(0.5, "x", "CLICK(0.5, 0.5)") == 0.5

    def test_boundary_zero(self):
        assert _validate_coord(0.0, "x", "CLICK(0, 0)") == 0.0

    def test_boundary_one(self):
        assert _validate_coord(1.0, "x", "CLICK(1, 1)") == 1.0

    def test_out_of_range_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_coord(1.5, "x", "CLICK(1.5, 0)")
            assert len(w) == 1
            assert "outside [0, 1]" in str(w[0].message)


class TestFormatActionQwen:
    """Test Qwen3-VL action string formatting."""

    def test_click(self):
        result = _format_action_qwen("click", {"x": 0.294, "y": 0.532})
        assert result == "click(x=294, y=532)"

    def test_double_click(self):
        result = _format_action_qwen("double_click", {"x": 0.5, "y": 0.5})
        assert result == "double_click(x=500, y=500)"

    def test_right_click(self):
        result = _format_action_qwen("right_click", {"x": 0.1, "y": 0.9})
        assert result == "right_click(x=100, y=900)"

    def test_type(self):
        result = _format_action_qwen("type", {"text": "hello"})
        assert result == 'type(text="hello")'

    def test_press_single(self):
        result = _format_action_qwen("press", {"keys": "Enter"})
        assert result == 'press(keys=["Enter"])'

    def test_press_combo(self):
        result = _format_action_qwen("press", {"keys": "Ctrl+C"})
        assert result == 'press(keys=["Ctrl", "C"])'

    def test_scroll(self):
        result = _format_action_qwen("scroll", {"direction": "down"})
        assert result == 'scroll(direction="down", amount=3)'

    def test_drag(self):
        result = _format_action_qwen(
            "drag",
            {
                "from_x": 0.1,
                "from_y": 0.2,
                "to_x": 0.8,
                "to_y": 0.9,
            },
        )
        assert result == "drag(from_coord=[100, 200], to_coord=[800, 900])"

    def test_finished(self):
        assert _format_action_qwen("finished", {}) == "finished()"

    def test_wait(self):
        assert _format_action_qwen("wait", {}) == "wait()"

    def test_unknown(self):
        result = _format_action_qwen("unknown", {"raw": "FOO(bar)"})
        assert "unknown" in result


class TestBuildThinkBlock:
    """Test think block construction."""

    def test_observation_and_intent(self):
        step = {"observation": "I see a button.", "intent": "Click it."}
        result = _build_think_block(step)
        assert result.startswith("<think>")
        assert result.endswith("</think>\n")
        assert "I see a button." in result
        assert "Click it." in result

    def test_observation_only(self):
        step = {"observation": "Desktop visible.", "intent": ""}
        result = _build_think_block(step)
        assert "Desktop visible." in result

    def test_empty(self):
        step = {}
        result = _build_think_block(step)
        assert result == ""


class TestConvertStep:
    """Test single step conversion to SFT format."""

    def test_basic_click(self):
        step = {
            "step_index": 0,
            "action_raw": "CLICK(0.5, 0.5)",
            "observation": "A desktop.",
            "intent": "Click center.",
        }
        sample = convert_step(
            step=step,
            instruction="Do the task",
            previous_actions=[],
            screenshot_path="/path/to/img.png",
        )
        assert sample is not None
        assert sample["images"] == ["/path/to/img.png"]
        assert len(sample["messages"]) == 3
        assert sample["messages"][0]["role"] == "system"
        assert sample["messages"][0]["content"] == SYSTEM_PROMPT
        assert sample["messages"][1]["role"] == "user"
        assert "Do the task" in sample["messages"][1]["content"]
        assert "<image>" in sample["messages"][1]["content"]
        assert sample["messages"][2]["role"] == "assistant"
        assert "click(x=500, y=500)" in sample["messages"][2]["content"]

    def test_with_previous_actions(self):
        step = {
            "step_index": 1,
            "action_raw": "TYPE(hello)",
        }
        sample = convert_step(
            step=step,
            instruction="Type hello",
            previous_actions=["click(x=100, y=200)"],
        )
        assert "Previous actions:" in sample["messages"][1]["content"]
        assert "click(x=100, y=200)" in sample["messages"][1]["content"]

    def test_no_screenshot(self):
        step = {"step_index": 0, "action_raw": "CLICK(0.5, 0.5)"}
        sample = convert_step(step=step, instruction="task", previous_actions=[])
        assert "images" not in sample

    def test_empty_action_returns_none(self):
        step = {"step_index": 0, "action_raw": ""}
        sample = convert_step(step=step, instruction="task", previous_actions=[])
        assert sample is None

    def test_no_thinking(self):
        step = {
            "step_index": 0,
            "action_raw": "CLICK(0.5, 0.5)",
            "observation": "Obs",
            "intent": "Int",
        }
        sample = convert_step(
            step=step,
            instruction="task",
            previous_actions=[],
            include_thinking=False,
        )
        assert "<think>" not in sample["messages"][2]["content"]
        assert "Output exactly one action." in sample["messages"][1]["content"]


class TestValidateDemo:
    """Test demo JSON validation."""

    def test_valid_demo(self):
        demo = {
            "instruction": "Do something",
            "steps": [{"step_index": 0, "action_raw": "CLICK(0.5, 0.5)"}],
        }
        issues = _validate_demo(demo, Path("test.json"))
        assert issues == []

    def test_missing_instruction(self):
        demo = {"steps": [{"step_index": 0, "action_raw": "CLICK(0.5, 0.5)"}]}
        issues = _validate_demo(demo, Path("test.json"))
        assert any("instruction" in i for i in issues)

    def test_empty_steps(self):
        demo = {"instruction": "Do something", "steps": []}
        issues = _validate_demo(demo, Path("test.json"))
        assert any("empty" in i for i in issues)

    def test_missing_action_raw(self):
        demo = {
            "instruction": "Do something",
            "steps": [{"step_index": 0}],
        }
        issues = _validate_demo(demo, Path("test.json"))
        assert any("action_raw" in i for i in issues)


class TestCreateBundle:
    """Test bundle creation."""

    def test_bundle_creates_directory(self, tmp_path):
        # Create a fake image
        img_dir = tmp_path / "source_images"
        img_dir.mkdir()
        img_file = img_dir / "screenshot.png"
        img_file.write_bytes(b"fake png data")

        samples = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "click(x=500, y=500)"},
                ],
                "images": [str(img_file)],
            }
        ]

        bundle_dir = tmp_path / "bundle"
        jsonl_path = create_bundle(samples, bundle_dir)

        assert jsonl_path.exists()
        assert (bundle_dir / "images" / "screenshot.png").exists()

        # Verify JSONL has relative paths
        with open(jsonl_path) as f:
            line = json.loads(f.readline())
        assert line["images"] == ["images/screenshot.png"]

    def test_bundle_missing_image(self, tmp_path):
        samples = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "images": ["/nonexistent/path.png"],
            }
        ]

        bundle_dir = tmp_path / "bundle"
        jsonl_path = create_bundle(samples, bundle_dir)

        with open(jsonl_path) as f:
            line = json.loads(f.readline())
        assert line["images"] == []


class TestGenerateScreenshotMapping:
    """Test auto-generation of screenshot mapping from captures on disk."""

    def _make_capture(self, captures_dir, task_id, num_steps):
        """Create a fake capture directory with screenshots."""
        cap_dir = captures_dir / task_id / "screenshots"
        cap_dir.mkdir(parents=True)
        for i in range(num_steps):
            (cap_dir / f"capture_1_step_{i}.png").write_bytes(b"fake")
        return cap_dir

    def _make_demo(self, demo_dir, task_id, num_steps):
        """Create a fake annotated demo JSON."""
        demo = {
            "task_id": task_id,
            "instruction": "Do something",
            "steps": [
                {"step_index": i, "action_raw": f"CLICK(0.{i}, 0.{i})"}
                for i in range(num_steps)
            ],
        }
        demo_path = demo_dir / f"{task_id}.json"
        demo_path.write_text(json.dumps(demo))
        return demo_path

    def test_basic_mapping(self, tmp_path):
        demos = tmp_path / "demos"
        demos.mkdir()
        captures = tmp_path / "captures"
        captures.mkdir()

        self._make_demo(demos, "task-1", 3)
        self._make_capture(captures, "task-1", 3)

        mapping = generate_screenshot_mapping(demos, captures)
        assert "task-1" in mapping
        assert len(mapping["task-1"]) == 3
        for i in range(3):
            assert str(i) in mapping["task-1"]

    def test_missing_capture_warns(self, tmp_path):
        demos = tmp_path / "demos"
        demos.mkdir()
        captures = tmp_path / "captures"
        captures.mkdir()

        self._make_demo(demos, "task-missing", 2)
        # No capture dir created

        mapping = generate_screenshot_mapping(demos, captures)
        assert "task-missing" not in mapping

    def test_missing_screenshot_warns(self, tmp_path):
        demos = tmp_path / "demos"
        demos.mkdir()
        captures = tmp_path / "captures"
        captures.mkdir()

        self._make_demo(demos, "task-2", 3)
        self._make_capture(captures, "task-2", 2)  # only 2 screenshots for 3 steps

        mapping = generate_screenshot_mapping(demos, captures)
        assert "task-2" in mapping
        assert len(mapping["task-2"]) == 2  # only 2 found

    def test_multiple_tasks(self, tmp_path):
        demos = tmp_path / "demos"
        demos.mkdir()
        captures = tmp_path / "captures"
        captures.mkdir()

        for tid in ["a", "b", "c"]:
            self._make_demo(demos, tid, 2)
            self._make_capture(captures, tid, 2)

        mapping = generate_screenshot_mapping(demos, captures)
        assert len(mapping) == 3


class TestPrepareBundle:
    """Test end-to-end prepare_bundle pipeline."""

    def test_prepare_bundle_auto_mapping(self, tmp_path):
        demos = tmp_path / "demos"
        demos.mkdir()
        captures = tmp_path / "captures"
        captures.mkdir()

        # Create demo + capture
        demo = {
            "task_id": "t1",
            "instruction": "Click the button",
            "steps": [
                {
                    "step_index": 0,
                    "action_raw": "CLICK(0.5, 0.5)",
                    "observation": "I see a button",
                    "intent": "Click it",
                }
            ],
        }
        (demos / "t1.json").write_text(json.dumps(demo))
        ss_dir = captures / "t1" / "screenshots"
        ss_dir.mkdir(parents=True)
        (ss_dir / "capture_1_step_0.png").write_bytes(b"fake png")

        bundle_dir = tmp_path / "bundle"
        result = prepare_bundle(demos, captures, bundle_dir=bundle_dir)

        assert result == bundle_dir
        assert (bundle_dir / "training_data.jsonl").exists()
        assert (bundle_dir / "images").is_dir()
        assert (bundle_dir / "screenshot_mapping.json").exists()

        # Verify JSONL content
        with open(bundle_dir / "training_data.jsonl") as f:
            sample = json.loads(f.readline())
        assert sample["images"] == ["images/capture_1_step_0.png"]
        assert sample["messages"][0]["role"] == "system"

    def test_prepare_bundle_with_mapping(self, tmp_path):
        demos = tmp_path / "demos"
        demos.mkdir()
        captures = tmp_path / "captures"
        captures.mkdir()

        demo = {
            "task_id": "t2",
            "instruction": "Type hello",
            "steps": [{"step_index": 0, "action_raw": 'TYPE("hello")'}],
        }
        (demos / "t2.json").write_text(json.dumps(demo))

        # Create image and pre-computed mapping
        img = tmp_path / "img.png"
        img.write_bytes(b"fake")
        mapping = {"t2": {"0": str(img)}}
        mapping_path = tmp_path / "mapping.json"
        mapping_path.write_text(json.dumps(mapping))

        bundle_dir = tmp_path / "bundle"
        prepare_bundle(demos, captures, bundle_dir=bundle_dir, mapping_path=mapping_path)

        assert (bundle_dir / "training_data.jsonl").exists()
        # Should NOT have screenshot_mapping.json since we provided one
        assert not (bundle_dir / "screenshot_mapping.json").exists()

    def test_prepare_bundle_no_samples_raises(self, tmp_path):
        demos = tmp_path / "demos"
        demos.mkdir()
        captures = tmp_path / "captures"
        captures.mkdir()
        # Empty demo dir — no JSON files

        with pytest.raises(ValueError, match="No training samples"):
            prepare_bundle(demos, captures, bundle_dir=tmp_path / "bundle")
