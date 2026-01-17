"""Unit tests for openadapt_ml.baselines module.

Tests track configuration, model registry, response parsing, and prompt building.
"""

from __future__ import annotations

import pytest
from PIL import Image

from openadapt_ml.baselines import (
    BaselineConfig,
    TrackConfig,
    TrackType,
    ModelSpec,
    MODELS,
    get_model_spec,
    get_default_model,
    ParsedAction,
    UnifiedResponseParser,
    PromptBuilder,
)


class TestTrackConfigFactoryMethods:
    """Tests for TrackConfig factory methods."""

    def test_track_a_direct_coordinates(self):
        """Test TrackConfig.track_a() creates Track A config."""
        config = TrackConfig.track_a()

        assert config.track_type == TrackType.TRACK_A
        assert config.use_som is False
        assert config.use_a11y_tree is True
        assert config.include_reasoning is False
        assert "CLICK" in config.output_format
        assert '"x": float' in config.output_format

    def test_track_b_react_style(self):
        """Test TrackConfig.track_b() creates Track B config."""
        config = TrackConfig.track_b()

        assert config.track_type == TrackType.TRACK_B
        assert config.use_som is False
        assert config.use_a11y_tree is True
        assert config.include_reasoning is True
        assert "thought" in config.output_format

    def test_track_c_set_of_mark(self):
        """Test TrackConfig.track_c() creates Track C config."""
        config = TrackConfig.track_c()

        assert config.track_type == TrackType.TRACK_C
        assert config.use_som is True
        assert config.use_a11y_tree is True
        assert config.include_reasoning is False
        assert "element_id" in config.output_format

    def test_track_configs_have_defaults(self):
        """Test all track configs have default values."""
        for factory in [TrackConfig.track_a, TrackConfig.track_b, TrackConfig.track_c]:
            config = factory()
            assert config.include_history is True
            assert config.max_history_steps == 5
            assert config.max_a11y_elements == 50


class TestBaselineConfigFromAlias:
    """Tests for BaselineConfig.from_alias() method."""

    def test_from_alias_claude_opus(self):
        """Test from_alias creates correct config for claude-opus-4.5."""
        config = BaselineConfig.from_alias("claude-opus-4.5")

        assert config.provider == "anthropic"
        assert config.model == "claude-opus-4-5-20251101"

    def test_from_alias_gpt(self):
        """Test from_alias creates correct config for GPT models."""
        config = BaselineConfig.from_alias("gpt-5.2")

        assert config.provider == "openai"
        assert config.model == "gpt-5.2"

    def test_from_alias_gemini(self):
        """Test from_alias creates correct config for Gemini models."""
        config = BaselineConfig.from_alias("gemini-3-pro")

        assert config.provider == "google"
        assert config.model == "gemini-3-pro"

    def test_from_alias_with_custom_track(self):
        """Test from_alias with custom track config."""
        track = TrackConfig.track_c()
        config = BaselineConfig.from_alias("claude-opus-4.5", track=track)

        assert config.track.track_type == TrackType.TRACK_C
        assert config.track.use_som is True

    def test_from_alias_with_kwargs(self):
        """Test from_alias passes through additional kwargs."""
        config = BaselineConfig.from_alias(
            "claude-opus-4.5",
            temperature=0.5,
            max_tokens=2048,
            verbose=True,
        )

        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.verbose is True

    def test_from_alias_unknown_model_raises_error(self):
        """Test from_alias raises ValueError for unknown alias."""
        with pytest.raises(ValueError) as exc_info:
            BaselineConfig.from_alias("unknown-model")

        assert "Unknown model" in str(exc_info.value)

    def test_from_alias_defaults_to_track_a(self):
        """Test from_alias defaults to Track A when no track specified."""
        config = BaselineConfig.from_alias("claude-opus-4.5")

        assert config.track.track_type == TrackType.TRACK_A


class TestModelsRegistry:
    """Tests for MODELS registry."""

    def test_models_has_expected_entries(self):
        """Test MODELS registry contains expected models."""
        expected_models = [
            "claude-opus-4.5",
            "claude-sonnet-4.5",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-4o",
            "gemini-3-pro",
            "gemini-3-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ]
        for model in expected_models:
            assert model in MODELS, f"Missing model: {model}"

    def test_all_models_are_model_spec_instances(self):
        """Test all MODELS values are ModelSpec instances."""
        for name, spec in MODELS.items():
            assert isinstance(spec, ModelSpec), f"{name} is not ModelSpec"

    def test_all_models_have_valid_providers(self):
        """Test all models have valid provider names."""
        valid_providers = {"anthropic", "openai", "google"}
        for name, spec in MODELS.items():
            assert spec.provider in valid_providers, f"{name} has invalid provider"

    def test_each_provider_has_default_model(self):
        """Test each provider has exactly one default model."""
        defaults_by_provider = {}
        for name, spec in MODELS.items():
            if spec.is_default:
                if spec.provider in defaults_by_provider:
                    pytest.fail(f"Multiple defaults for {spec.provider}")
                defaults_by_provider[spec.provider] = name

        assert "anthropic" in defaults_by_provider
        assert "openai" in defaults_by_provider
        assert "google" in defaults_by_provider

    def test_get_model_spec_returns_correct_spec(self):
        """Test get_model_spec returns correct ModelSpec."""
        spec = get_model_spec("claude-opus-4.5")

        assert spec.provider == "anthropic"
        assert spec.model_id == "claude-opus-4-5-20251101"
        assert spec.display_name == "Claude Opus 4.5"

    def test_get_default_model_returns_default(self):
        """Test get_default_model returns the default for each provider."""
        anthropic_default = get_default_model("anthropic")
        assert anthropic_default.is_default is True
        assert anthropic_default.provider == "anthropic"

        openai_default = get_default_model("openai")
        assert openai_default.is_default is True
        assert openai_default.provider == "openai"

        google_default = get_default_model("google")
        assert google_default.is_default is True
        assert google_default.provider == "google"


class TestUnifiedResponseParserClickCoords:
    """Tests for UnifiedResponseParser parsing CLICK(x, y) format."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnifiedResponseParser()

    def test_parse_click_coordinates(self, parser):
        """Test parsing CLICK(0.5, 0.3) format."""
        action = parser.parse("CLICK(0.5, 0.3)")

        assert action.action_type == "click"
        assert action.x == pytest.approx(0.5)
        assert action.y == pytest.approx(0.3)
        assert action.is_valid

    def test_parse_click_coordinates_with_spaces(self, parser):
        """Test parsing CLICK with various spacing."""
        action = parser.parse("CLICK( 0.5 , 0.3 )")

        assert action.action_type == "click"
        assert action.x == pytest.approx(0.5)
        assert action.y == pytest.approx(0.3)

    def test_parse_click_coordinates_case_insensitive(self, parser):
        """Test parsing click is case insensitive."""
        action = parser.parse("click(0.5, 0.3)")
        assert action.action_type == "click"

        action = parser.parse("Click(0.5, 0.3)")
        assert action.action_type == "click"


class TestUnifiedResponseParserClickElement:
    """Tests for UnifiedResponseParser parsing CLICK([id]) format."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnifiedResponseParser()

    def test_parse_click_element_with_brackets(self, parser):
        """Test parsing CLICK([42]) format."""
        action = parser.parse("CLICK([42])")

        assert action.action_type == "click"
        assert action.element_id == 42
        assert action.x is None
        assert action.y is None
        assert action.is_valid

    def test_parse_click_element_without_brackets(self, parser):
        """Test parsing CLICK(42) format (no brackets)."""
        action = parser.parse("CLICK(42)")

        assert action.action_type == "click"
        assert action.element_id == 42

    def test_parse_click_element_with_spaces(self, parser):
        """Test parsing CLICK with spaces around element ID."""
        action = parser.parse("CLICK( [ 42 ] )")

        assert action.action_type == "click"
        assert action.element_id == 42


class TestUnifiedResponseParserType:
    """Tests for UnifiedResponseParser parsing TYPE format."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnifiedResponseParser()

    def test_parse_type_double_quotes(self, parser):
        """Test parsing TYPE("hello") format."""
        action = parser.parse('TYPE("hello")')

        assert action.action_type == "type"
        assert action.text == "hello"
        assert action.is_valid

    def test_parse_type_single_quotes(self, parser):
        """Test parsing TYPE('hello') format."""
        action = parser.parse("TYPE('hello')")

        assert action.action_type == "type"
        assert action.text == "hello"

    def test_parse_type_with_spaces(self, parser):
        """Test parsing TYPE with text containing spaces."""
        action = parser.parse('TYPE("hello world")')

        assert action.action_type == "type"
        assert action.text == "hello world"


class TestUnifiedResponseParserKey:
    """Tests for UnifiedResponseParser parsing KEY format."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnifiedResponseParser()

    def test_parse_key_enter(self, parser):
        """Test parsing KEY(enter) format."""
        action = parser.parse("KEY(enter)")

        assert action.action_type == "key"
        assert action.key == "enter"
        assert action.is_valid

    def test_parse_key_escape(self, parser):
        """Test parsing KEY(escape) format."""
        action = parser.parse("KEY(escape)")

        assert action.action_type == "key"
        assert action.key == "escape"

    def test_parse_key_case_insensitive(self, parser):
        """Test KEY parsing is case insensitive."""
        action = parser.parse("KEY(Enter)")
        assert action.key == "enter"

        action = parser.parse("KEY(ESCAPE)")
        assert action.key == "escape"


class TestUnifiedResponseParserScroll:
    """Tests for UnifiedResponseParser parsing SCROLL format."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnifiedResponseParser()

    def test_parse_scroll_down(self, parser):
        """Test parsing SCROLL(down) format."""
        action = parser.parse("SCROLL(down)")

        assert action.action_type == "scroll"
        assert action.direction == "down"
        assert action.is_valid

    def test_parse_scroll_up(self, parser):
        """Test parsing SCROLL(up) format."""
        action = parser.parse("SCROLL(up)")

        assert action.action_type == "scroll"
        assert action.direction == "up"

    def test_parse_scroll_case_insensitive(self, parser):
        """Test SCROLL parsing is case insensitive."""
        action = parser.parse("SCROLL(Down)")
        assert action.direction == "down"


class TestUnifiedResponseParserDone:
    """Tests for UnifiedResponseParser parsing DONE format."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnifiedResponseParser()

    def test_parse_done(self, parser):
        """Test parsing DONE() format."""
        action = parser.parse("DONE()")

        assert action.action_type == "done"
        assert action.is_valid

    def test_parse_done_with_spaces(self, parser):
        """Test parsing DONE with spaces."""
        action = parser.parse("DONE( )")

        assert action.action_type == "done"


class TestUnifiedResponseParserJSON:
    """Tests for UnifiedResponseParser parsing JSON format."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnifiedResponseParser()

    def test_parse_json_click_coords(self, parser):
        """Test parsing JSON click with coordinates."""
        action = parser.parse('{"action": "CLICK", "x": 0.5, "y": 0.3}')

        assert action.action_type == "click"
        assert action.x == pytest.approx(0.5)
        assert action.y == pytest.approx(0.3)

    def test_parse_json_click_element_id(self, parser):
        """Test parsing JSON click with element_id."""
        action = parser.parse('{"action": "CLICK", "element_id": 17}')

        assert action.action_type == "click"
        assert action.element_id == 17

    def test_parse_json_type(self, parser):
        """Test parsing JSON type action."""
        action = parser.parse('{"action": "TYPE", "text": "hello world"}')

        assert action.action_type == "type"
        assert action.text == "hello world"

    def test_parse_json_with_thought(self, parser):
        """Test parsing JSON with thought field (ReAct)."""
        action = parser.parse('{"thought": "I see a button", "action": "CLICK", "x": 0.5, "y": 0.3}')

        assert action.action_type == "click"
        assert action.thought == "I see a button"

    def test_parse_json_embedded_in_text(self, parser):
        """Test parsing JSON embedded in surrounding text."""
        response = 'Based on the screenshot, I will click the button. {"action": "CLICK", "x": 0.5, "y": 0.3}'
        action = parser.parse(response)

        assert action.action_type == "click"
        assert action.x == pytest.approx(0.5)


class TestUnifiedResponseParserInvalid:
    """Tests for UnifiedResponseParser handling invalid inputs."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return UnifiedResponseParser()

    def test_parse_unknown_action_returns_unknown(self, parser):
        """Test parsing unknown text returns unknown action."""
        action = parser.parse("I don't know what to do")

        assert action.action_type == "unknown"
        assert action.is_valid is False
        assert action.parse_error is not None

    def test_parse_empty_string(self, parser):
        """Test parsing empty string."""
        action = parser.parse("")

        assert action.action_type == "unknown"
        assert action.is_valid is False

    def test_parse_preserves_raw_response(self, parser):
        """Test parsing preserves original response."""
        response = "CLICK(0.5, 0.3)"
        action = parser.parse(response)

        assert action.raw_response == response


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_track_a_system_prompt(self):
        """Test PromptBuilder generates Track A system prompt."""
        builder = PromptBuilder(TrackConfig.track_a())
        prompt = builder.get_system_prompt()

        assert "GUI automation agent" in prompt
        # The prompt shows coordinates in JSON format
        assert "normalized" in prompt.lower()
        assert "0.0" in prompt and "1.0" in prompt  # Coordinate range explanation
        # Track A should NOT mention element IDs as primary method
        assert "element_id" not in prompt.lower() or "element ID" not in prompt.lower()

    def test_track_b_system_prompt(self):
        """Test PromptBuilder generates Track B system prompt."""
        builder = PromptBuilder(TrackConfig.track_b())
        prompt = builder.get_system_prompt()

        assert "ReAct" in prompt or "Reason + Act" in prompt
        assert "thought" in prompt.lower()
        assert "OBSERVE" in prompt or "THINK" in prompt

    def test_track_c_system_prompt(self):
        """Test PromptBuilder generates Track C system prompt."""
        builder = PromptBuilder(TrackConfig.track_c())
        prompt = builder.get_system_prompt()

        assert "numbered labels" in prompt or "[1]" in prompt
        assert "element ID" in prompt.lower() or "element_id" in prompt
        # Track C should mention using IDs not coordinates
        assert "NOT coordinates" in prompt or "element ID" in prompt.lower()

    def test_system_prompt_with_demo(self):
        """Test PromptBuilder includes demo when provided."""
        builder = PromptBuilder(TrackConfig.track_a())
        demo_text = "Example: User clicked login button"
        prompt = builder.get_system_prompt(demo=demo_text)

        assert demo_text in prompt
        assert "DEMONSTRATION" in prompt or "EXAMPLE" in prompt

    def test_build_user_content_includes_goal(self):
        """Test build_user_content includes goal."""
        builder = PromptBuilder(TrackConfig.track_a())
        content = builder.build_user_content(goal="Click the submit button")

        # Should have at least one text content item
        text_items = [c for c in content if c.get("type") == "text"]
        assert len(text_items) > 0
        assert "Click the submit button" in text_items[0]["text"]

    def test_build_user_content_includes_a11y_tree(self):
        """Test build_user_content includes accessibility tree."""
        track = TrackConfig.track_a()
        track.use_a11y_tree = True
        builder = PromptBuilder(track)

        a11y_tree = "[1] button: Submit\n[2] text: Hello"
        content = builder.build_user_content(goal="Click submit", a11y_tree=a11y_tree)

        text_content = content[0]["text"]
        assert "ACCESSIBILITY TREE" in text_content
        assert "Submit" in text_content

    def test_build_user_content_includes_history(self):
        """Test build_user_content includes action history."""
        track = TrackConfig.track_a()
        track.include_history = True
        builder = PromptBuilder(track)

        history = [
            {"type": "click", "x": 0.1, "y": 0.2},
            {"type": "type", "text": "hello"},
        ]
        content = builder.build_user_content(goal="Continue", history=history)

        text_content = content[0]["text"]
        assert "PREVIOUS ACTIONS" in text_content
        assert "CLICK" in text_content
        assert "TYPE" in text_content

    def test_build_user_content_with_screenshot(self):
        """Test build_user_content includes encoded screenshot."""
        builder = PromptBuilder(TrackConfig.track_a())
        test_image = Image.new("RGB", (100, 100), color="red")

        def mock_encode(img):
            return {"type": "image", "data": "encoded"}

        content = builder.build_user_content(
            goal="Click button",
            screenshot=test_image,
            encode_image_fn=mock_encode,
        )

        image_items = [c for c in content if c.get("type") == "image"]
        assert len(image_items) == 1

    def test_build_user_content_truncates_a11y_tree(self):
        """Test build_user_content truncates long accessibility trees."""
        track = TrackConfig.track_a()
        track.use_a11y_tree = True
        track.max_a11y_elements = 5
        builder = PromptBuilder(track)

        # Create a11y tree with more than max elements
        lines = [f"[{i}] button: Button {i}" for i in range(20)]
        a11y_tree = "\n".join(lines)

        content = builder.build_user_content(goal="Click", a11y_tree=a11y_tree)

        text_content = content[0]["text"]
        # Should be truncated - the implementation shows "showing X of Y elements"
        assert "showing 5 of 20" in text_content or "truncated" in text_content


class TestParsedAction:
    """Tests for ParsedAction dataclass."""

    def test_parsed_action_is_valid(self):
        """Test is_valid property for valid actions."""
        action = ParsedAction(action_type="click", x=0.5, y=0.3)
        assert action.is_valid is True

    def test_parsed_action_is_invalid_with_error(self):
        """Test is_valid property when parse_error is set."""
        action = ParsedAction(action_type="click", parse_error="Missing coordinates")
        assert action.is_valid is False

    def test_parsed_action_unknown_is_invalid(self):
        """Test is_valid property for unknown action type."""
        action = ParsedAction(action_type="unknown")
        assert action.is_valid is False

    def test_to_dict_click_coords(self):
        """Test to_dict for click with coordinates."""
        action = ParsedAction(action_type="click", x=0.5, y=0.3)
        d = action.to_dict()

        assert d["type"] == "click"
        assert d["x"] == 0.5
        assert d["y"] == 0.3

    def test_to_dict_click_element(self):
        """Test to_dict for click with element_id."""
        action = ParsedAction(action_type="click", element_id=42)
        d = action.to_dict()

        assert d["type"] == "click"
        assert d["element_id"] == 42
        assert "x" not in d
        assert "y" not in d

    def test_to_dict_type_action(self):
        """Test to_dict for type action."""
        action = ParsedAction(action_type="type", text="hello")
        d = action.to_dict()

        assert d["type"] == "type"
        assert d["text"] == "hello"

    def test_to_dict_with_thought(self):
        """Test to_dict includes thought when present."""
        action = ParsedAction(action_type="click", x=0.5, y=0.3, thought="I see a button")
        d = action.to_dict()

        assert d["thought"] == "I see a button"
