import asyncio

import pytest

from adapters.inference.mock_adapter import MockInferenceAdapter
from adapters.storage.memory_repo import InMemoryTrackRepository
from domain.entities.evidence import EvidencePackage
from domain.entities.track import Track
from ports.inference_port import InferencePort
from ports.storage_port import StoragePort


@pytest.mark.asyncio
async def test_mock_inference_adapter_matches_by_default():
    adapter: InferencePort = MockInferenceAdapter(default_match=True)
    track = Track(track_id=1, video_id="v1", frames=[1, 2, 3], bboxes=[(0, 0, 1, 1)], fps=30.0)
    package = EvidencePackage(track=track, crops=[], bboxes=track.bboxes)
    result = await adapter.verify_track(package, question="test")
    assert result.is_match
    assert adapter.call_count == 1


@pytest.mark.asyncio
async def test_in_memory_repo_save_and_get():
    repo: StoragePort = InMemoryTrackRepository()
    track = Track(track_id=2, video_id="v2", frames=[1, 5, 10], bboxes=[(0, 0, 2, 2)], fps=25.0)
    await repo.save_track(track)
    tracks = await repo.get_tracks_by_video("v2")
    assert len(tracks) == 1
    assert tracks[0].track_id == 2
