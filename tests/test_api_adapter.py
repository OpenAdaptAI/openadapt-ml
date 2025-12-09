from __future__ import annotations

from typing import Any, Dict

import builtins
import io
from unittest import mock

import pytest

from openadapt_ml.models.api_adapter import ApiVLMAdapter


@pytest.fixture
def dummy_sample(tmp_path) -> Dict[str, Any]:
    img_path = tmp_path / "dummy.png"
    img_path.write_bytes(b"fake-image-bytes")
    return {
        "images": [str(img_path)],
        "messages": [
            {"role": "system", "content": "You are a GUI automation agent."},
            {"role": "user", "content": "Goal: test. Current screen: see image."},
        ],
    }


@mock.patch("openadapt_ml.models.api_adapter.settings")
@mock.patch("openadapt_ml.models.api_adapter.OpenAI")
@mock.patch("openadapt_ml.models.api_adapter.os.getenv")
def test_openai_adapter_generate(mock_getenv, mock_openai, mock_settings, dummy_sample) -> None:
    mock_settings.openai_api_key = None
    mock_getenv.return_value = "dummy-key"
    client_instance = mock_openai.return_value
    client_instance.chat.completions.create.return_value = mock.Mock(
        choices=[mock.Mock(message=mock.Mock(content="CLICK(x=0.5, y=0.5)"))]
    )

    adapter = ApiVLMAdapter(provider="openai")
    text = adapter.generate(dummy_sample)
    assert "CLICK(" in text


@mock.patch("openadapt_ml.models.api_adapter.settings")
@mock.patch("openadapt_ml.models.api_adapter.Anthropic")
@mock.patch("openadapt_ml.models.api_adapter.os.getenv")
def test_anthropic_adapter_generate(mock_getenv, mock_anthropic, mock_settings, dummy_sample) -> None:
    mock_settings.anthropic_api_key = None
    mock_getenv.return_value = "dummy-key"
    client_instance = mock_anthropic.return_value
    # Simulate messages.create returning an object with a content list of blocks.
    block = mock.Mock()
    block.type = "text"
    block.text = "DONE()"
    resp = mock.Mock()
    resp.content = [block]
    client_instance.messages.create.return_value = resp

    adapter = ApiVLMAdapter(provider="anthropic")
    text = adapter.generate(dummy_sample)
    assert "DONE" in text
