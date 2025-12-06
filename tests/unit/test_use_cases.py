import pytest

from adapters.inference.mock_adapter import MockInferenceAdapter
from adapters.storage.memory_repo import InMemoryTrackRepository
from application.use_cases.search_tracks import SearchTracksUseCase, SearchRequest
from domain.entities.track import Track


@pytest.mark.asyncio
async def test_search_use_case_matches_tracks():
    repo = InMemoryTrackRepository()
    # populate two tracks
    await repo.save_track(Track(track_id=1, video_id="vid", frames=[1, 3, 6], bboxes=[(0, 0, 1, 1)], fps=30.0))
    await repo.save_track(Track(track_id=2, video_id="vid", frames=[2, 4, 7], bboxes=[(0, 0, 2, 2)], fps=30.0))

    adapter = MockInferenceAdapter(default_match=True)
    use_case = SearchTracksUseCase(inference=adapter, track_repo=repo)

    response = await use_case.execute(SearchRequest(video_id="vid", question="who", top_k=1))
    assert len(response.matched_track_ids) == 1
    assert response.matched_track_ids[0] == 1
    assert adapter.call_count == 1  # stopped at top_k
