"""Unit tests for openadapt_ml.models.providers module.

Tests provider factory, model alias resolution, and image encoding for each provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import io
import base64

import pytest
from PIL import Image

from openadapt_ml.models.providers import (
    get_provider,
    resolve_model_alias,
    PROVIDERS,
    MODEL_ALIASES,
    AnthropicProvider,
    OpenAIProvider,
    GoogleProvider,
    BaseAPIProvider,
)


class TestGetProvider:
    """Tests for get_provider() factory function."""

    def test_get_anthropic_provider(self):
        """Test get_provider returns AnthropicProvider for 'anthropic'."""
        provider = get_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)
        assert provider.name == "anthropic"

    def test_get_openai_provider(self):
        """Test get_provider returns OpenAIProvider for 'openai'."""
        provider = get_provider("openai")
        assert isinstance(provider, OpenAIProvider)
        assert provider.name == "openai"

    def test_get_google_provider(self):
        """Test get_provider returns GoogleProvider for 'google'."""
        provider = get_provider("google")
        assert isinstance(provider, GoogleProvider)
        assert provider.name == "google"

    def test_get_provider_case_insensitive(self):
        """Test get_provider handles case-insensitive names."""
        provider_lower = get_provider("anthropic")
        provider_upper = get_provider("ANTHROPIC")
        provider_mixed = get_provider("Anthropic")

        assert all(isinstance(p, AnthropicProvider) for p in [provider_lower, provider_upper, provider_mixed])

    def test_get_provider_unknown_raises_error(self):
        """Test get_provider raises ValueError for unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            get_provider("unknown_provider")

        assert "Unknown provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)
        # Should list available providers
        assert "anthropic" in str(exc_info.value)

    def test_providers_registry_completeness(self):
        """Test PROVIDERS registry contains all expected providers."""
        assert "anthropic" in PROVIDERS
        assert "openai" in PROVIDERS
        assert "google" in PROVIDERS
        assert len(PROVIDERS) == 3

    def test_providers_are_subclasses_of_base(self):
        """Test all providers are subclasses of BaseAPIProvider."""
        for name, provider_class in PROVIDERS.items():
            assert issubclass(provider_class, BaseAPIProvider), f"{name} not a subclass of BaseAPIProvider"


class TestResolveModelAlias:
    """Tests for resolve_model_alias() function."""

    def test_resolve_claude_opus_alias(self):
        """Test resolving claude-opus-4.5 alias."""
        provider, model_id = resolve_model_alias("claude-opus-4.5")
        assert provider == "anthropic"
        assert model_id == "claude-opus-4-5-20251101"

    def test_resolve_claude_sonnet_alias(self):
        """Test resolving claude-sonnet-4.5 alias."""
        provider, model_id = resolve_model_alias("claude-sonnet-4.5")
        assert provider == "anthropic"
        assert model_id == "claude-sonnet-4-5-20250929"

    def test_resolve_gpt_aliases(self):
        """Test resolving GPT model aliases."""
        # GPT-5.2
        provider, model_id = resolve_model_alias("gpt-5.2")
        assert provider == "openai"
        assert model_id == "gpt-5.2"

        # GPT-5.1
        provider, model_id = resolve_model_alias("gpt-5.1")
        assert provider == "openai"
        assert model_id == "gpt-5.1"

        # GPT-4o
        provider, model_id = resolve_model_alias("gpt-4o")
        assert provider == "openai"
        assert model_id == "gpt-4o"

    def test_resolve_gemini_aliases(self):
        """Test resolving Gemini model aliases."""
        # Gemini 3 Pro
        provider, model_id = resolve_model_alias("gemini-3-pro")
        assert provider == "google"
        assert model_id == "gemini-3-pro"

        # Gemini 3 Flash
        provider, model_id = resolve_model_alias("gemini-3-flash")
        assert provider == "google"
        assert model_id == "gemini-3-flash"

        # Gemini 2.5 Pro
        provider, model_id = resolve_model_alias("gemini-2.5-pro")
        assert provider == "google"
        assert model_id == "gemini-2.5-pro"

    def test_resolve_unknown_alias_raises_error(self):
        """Test resolving unknown alias raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_model_alias("unknown-model-alias")

        assert "Unknown model alias" in str(exc_info.value)
        assert "unknown-model-alias" in str(exc_info.value)

    def test_model_aliases_completeness(self):
        """Test MODEL_ALIASES registry has expected entries."""
        expected_aliases = [
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
        for alias in expected_aliases:
            assert alias in MODEL_ALIASES, f"Missing alias: {alias}"


class TestAnthropicProviderEncodeImage:
    """Tests for AnthropicProvider.encode_image() method."""

    @pytest.fixture
    def provider(self):
        """Create AnthropicProvider instance."""
        return AnthropicProvider()

    @pytest.fixture
    def test_image(self):
        """Create a test PIL Image."""
        return Image.new("RGB", (100, 100), color="red")

    def test_encode_image_returns_correct_format(self, provider, test_image):
        """Test encode_image returns Anthropic-format dict."""
        result = provider.encode_image(test_image)

        assert isinstance(result, dict)
        assert result["type"] == "image"
        assert "source" in result

    def test_encode_image_has_base64_source(self, provider, test_image):
        """Test encode_image source has base64 type."""
        result = provider.encode_image(test_image)

        source = result["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "image/png"
        assert "data" in source

    def test_encode_image_data_is_valid_base64(self, provider, test_image):
        """Test encode_image data is valid base64."""
        result = provider.encode_image(test_image)

        data = result["source"]["data"]
        # Should be decodable base64
        decoded = base64.b64decode(data)
        assert len(decoded) > 0

        # Should be valid PNG (magic bytes)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


class TestOpenAIProviderEncodeImage:
    """Tests for OpenAIProvider.encode_image() method."""

    @pytest.fixture
    def provider(self):
        """Create OpenAIProvider instance."""
        return OpenAIProvider()

    @pytest.fixture
    def test_image(self):
        """Create a test PIL Image."""
        return Image.new("RGB", (100, 100), color="blue")

    def test_encode_image_returns_data_url_format(self, provider, test_image):
        """Test encode_image returns OpenAI data URL format."""
        result = provider.encode_image(test_image)

        assert isinstance(result, dict)
        assert result["type"] == "image_url"
        assert "image_url" in result

    def test_encode_image_url_has_data_prefix(self, provider, test_image):
        """Test encode_image URL has proper data: prefix."""
        result = provider.encode_image(test_image)

        url = result["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_encode_image_data_url_contains_valid_base64(self, provider, test_image):
        """Test encode_image data URL contains valid base64."""
        result = provider.encode_image(test_image)

        url = result["image_url"]["url"]
        # Extract base64 data after the prefix
        prefix = "data:image/png;base64,"
        base64_data = url[len(prefix):]

        # Should be decodable base64
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

        # Should be valid PNG
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


class TestGoogleProviderEncodeImage:
    """Tests for GoogleProvider.encode_image() method."""

    @pytest.fixture
    def provider(self):
        """Create GoogleProvider instance."""
        return GoogleProvider()

    @pytest.fixture
    def test_image(self):
        """Create a test PIL Image."""
        return Image.new("RGB", (100, 100), color="green")

    def test_encode_image_returns_pil_image(self, provider, test_image):
        """Test encode_image returns PIL image in dict."""
        result = provider.encode_image(test_image)

        assert isinstance(result, dict)
        assert result["type"] == "image"
        assert "image" in result

    def test_encode_image_preserves_original_image(self, provider, test_image):
        """Test encode_image preserves the original PIL Image."""
        result = provider.encode_image(test_image)

        # Google accepts PIL Images directly
        image = result["image"]
        assert image is test_image
        assert isinstance(image, Image.Image)

    def test_encode_image_no_base64_conversion(self, provider, test_image):
        """Test Google provider does not convert to base64."""
        result = provider.encode_image(test_image)

        # Should not have base64 data
        assert "data" not in result
        assert "url" not in result
        # Should have the actual image object
        assert isinstance(result["image"], Image.Image)


class TestProviderClientCreation:
    """Tests for provider client creation with mocks."""

    def test_anthropic_create_client(self):
        """Test AnthropicProvider.create_client creates Anthropic client."""
        # Mock the anthropic module
        mock_anthropic_module = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            provider = AnthropicProvider()
            client = provider.create_client("test-api-key")

            mock_anthropic_module.Anthropic.assert_called_once_with(api_key="test-api-key")
            assert client is mock_client

    def test_openai_create_client(self):
        """Test OpenAIProvider.create_client creates OpenAI client."""
        # Mock the openai module
        mock_openai_module = MagicMock()
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            provider = OpenAIProvider()
            client = provider.create_client("test-api-key")

            mock_openai_module.OpenAI.assert_called_once_with(api_key="test-api-key")
            assert client is mock_client

    def test_google_create_client(self):
        """Test GoogleProvider.create_client configures genai."""
        import sys

        # Mock the google.generativeai module
        mock_genai = MagicMock()
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        # Remove any cached google modules
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("google")]
        original_modules = {k: sys.modules.pop(k) for k in modules_to_remove if k in sys.modules}

        try:
            with patch.dict("sys.modules", {"google": mock_google, "google.generativeai": mock_genai}):
                provider = GoogleProvider()
                result = provider.create_client("test-api-key")

                mock_genai.configure.assert_called_once_with(api_key="test-api-key")
                assert result["api_key"] == "test-api-key"
                assert result["genai"] is mock_genai
        finally:
            # Restore original modules
            sys.modules.update(original_modules)


class TestProviderSendMessage:
    """Tests for provider send_message with mocks."""

    def test_anthropic_send_message(self):
        """Test AnthropicProvider.send_message calls API correctly."""
        # Setup mock
        mock_anthropic_module = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        # Mock response
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"action": "CLICK", "x": 0.5, "y": 0.3}'

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_client.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            provider = AnthropicProvider()
            client = provider.create_client("test-key")
            result = provider.send_message(
                client=client,
                model="claude-opus-4-5-20251101",
                system="You are a GUI agent.",
                content=[{"type": "text", "text": "Click submit"}],
                max_tokens=100,
                temperature=0.1,
            )

            mock_client.messages.create.assert_called_once()
            assert '{"action": "CLICK"' in result

    def test_openai_send_message(self):
        """Test OpenAIProvider.send_message calls API correctly."""
        # Setup mock
        mock_openai_module = MagicMock()
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        # Mock response
        mock_message = MagicMock()
        mock_message.content = '{"action": "TYPE", "text": "hello"}'

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            provider = OpenAIProvider()
            client = provider.create_client("test-key")
            result = provider.send_message(
                client=client,
                model="gpt-5.2",
                system="You are a GUI agent.",
                content=[{"type": "text", "text": "Type hello"}],
                max_tokens=100,
                temperature=0.1,
            )

            mock_client.chat.completions.create.assert_called_once()
            assert '{"action": "TYPE"' in result

    def test_google_send_message(self):
        """Test GoogleProvider.send_message calls API correctly."""
        import sys

        # Setup mock
        mock_genai = MagicMock()
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        mock_model_instance = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model_instance

        mock_response = MagicMock()
        mock_response.text = '{"action": "SCROLL", "direction": "down"}'
        mock_model_instance.generate_content.return_value = mock_response

        # Remove any cached google modules
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("google")]
        original_modules = {k: sys.modules.pop(k) for k in modules_to_remove if k in sys.modules}

        try:
            with patch.dict("sys.modules", {"google": mock_google, "google.generativeai": mock_genai}):
                provider = GoogleProvider()
                client = provider.create_client("test-key")
                result = provider.send_message(
                    client=client,
                    model="gemini-3-pro",
                    system="You are a GUI agent.",
                    content=[{"type": "text", "text": "Scroll down"}],
                    max_tokens=100,
                    temperature=0.1,
                )

                mock_genai.GenerativeModel.assert_called_once_with("gemini-3-pro")
                mock_model_instance.generate_content.assert_called_once()
                assert '{"action": "SCROLL"' in result
        finally:
            # Restore original modules
            sys.modules.update(original_modules)


class TestBaseAPIProviderHelpers:
    """Tests for BaseAPIProvider helper methods."""

    @pytest.fixture
    def provider(self):
        """Use a concrete provider to test base methods."""
        return AnthropicProvider()

    @pytest.fixture
    def test_image(self):
        """Create a test PIL Image."""
        return Image.new("RGB", (50, 50), color="white")

    def test_image_to_base64(self, provider, test_image):
        """Test image_to_base64 converts image correctly."""
        result = provider.image_to_base64(test_image, "PNG")

        # Should be base64 string
        assert isinstance(result, str)

        # Should be decodable
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_get_media_type_png(self, provider):
        """Test get_media_type returns correct MIME for PNG."""
        assert provider.get_media_type("PNG") == "image/png"
        assert provider.get_media_type("png") == "image/png"

    def test_get_media_type_jpeg(self, provider):
        """Test get_media_type returns correct MIME for JPEG."""
        assert provider.get_media_type("JPEG") == "image/jpeg"
        assert provider.get_media_type("JPG") == "image/jpeg"
        assert provider.get_media_type("jpg") == "image/jpeg"

    def test_get_media_type_gif(self, provider):
        """Test get_media_type returns correct MIME for GIF."""
        assert provider.get_media_type("GIF") == "image/gif"

    def test_get_media_type_webp(self, provider):
        """Test get_media_type returns correct MIME for WEBP."""
        assert provider.get_media_type("WEBP") == "image/webp"

    def test_get_media_type_unknown_defaults_to_png(self, provider):
        """Test get_media_type defaults to PNG for unknown formats."""
        assert provider.get_media_type("unknown") == "image/png"
