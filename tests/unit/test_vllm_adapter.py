import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).parents[2] / "src"))

from adapters.inference.vllm_adapter import VllmAdapter, VllmConfig  # noqa: E402
from domain.value_objects.verification_result import MatchStatus, VlmResponseParser  # noqa: E402


@pytest.fixture
def mock_config():
    return VllmConfig(endpoint="http://localhost:8000/v1", model_name="test-model")


@pytest.fixture
def adapter(mock_config):
    with patch("adapters.inference.vllm_adapter.AsyncOpenAI"):
        return VllmAdapter(mock_config)


class TestVlmResponseParser:
    def test_parse_match_yes(self):
        parser = VlmResponseParser()
        result = parser.parse("The person is wearing blue. MATCH: yes")

        assert result.status == MatchStatus.CONFIRMED
        assert result.is_match is True

    def test_parse_match_no(self):
        parser = VlmResponseParser()
        result = parser.parse("No match found. MATCH: no")

        assert result.status == MatchStatus.REJECTED
        assert result.is_match is False

    def test_parse_empty_response(self):
        parser = VlmResponseParser()
        result = parser.parse("")

        assert result.status == MatchStatus.AMBIGUOUS


@pytest.mark.asyncio
async def test_verify_track(adapter):
    # Mock the OpenAI response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "MATCH: yes"
    mock_response.choices = [mock_choice]
    adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_package = MagicMock()
    mock_package.crops = ["path/to/image.jpg"]
    mock_package.features = None

    with patch.object(adapter, "_encode_image", return_value="base64data"):
        result = await adapter.verify_track(mock_package, "test question")

    assert result.is_match is True
