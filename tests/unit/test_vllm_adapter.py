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
async def test_verify_batch(adapter):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"1": {"match": true, "reason": "ok"}}'
    mock_response.choices = [mock_choice]
    adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

    pkg = MagicMock()
    pkg.track_id = 1
    pkg.frames = [0, 1, 2]
    pkg.bboxes = [(0, 0, 10, 10)]
    pkg.video_path = "/tmp/fake.mp4"
    pkg.meta = {"resolution": (1920, 1080)}
    pkg.features = MagicMock(
        norm_speed=0.5, linearity=0.9, scale_change=1.0, displacement_vec=(1, 0), duration_s=1.0
    )

    with patch.object(adapter, "_extract_frames", return_value=(["imgb64"], (1920, 1080))):
        results = await adapter.verify_batch([pkg], "test question")

    assert len(results) == 1
    assert results[0].is_match is True
