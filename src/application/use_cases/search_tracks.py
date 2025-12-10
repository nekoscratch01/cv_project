from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from adapters.inference.mock_adapter import MockInferenceAdapter
from domain.entities.evidence import EvidencePackage
from domain.entities.track import Track
from domain.value_objects.verification_result import InferenceResult
from ports.inference_port import InferencePort
from ports.storage_port import StoragePort
from adapters.storage.memory_repo import InMemoryTrackRepository


@dataclass
class SearchRequest:
    video_id: str
    question: str
    top_k: Optional[int] = None


@dataclass
class SearchResponse:
    matches: List[InferenceResult]
    matched_track_ids: List[int]


class SearchTracksUseCase:
    """Use case for verifying tracks against a question."""

    def __init__(self, inference: InferencePort, track_repo: StoragePort) -> None:
        self.inference = inference
        self.track_repo = track_repo

    async def execute(self, request: SearchRequest) -> SearchResponse:
        tracks = await self.track_repo.get_tracks_by_video(request.video_id)
        matches: List[InferenceResult] = []
        matched_ids: List[int] = []
        for track in tracks:
            package = EvidencePackage(track=track, crops=[], bboxes=track.bboxes)
            result = await self.inference.verify_track(
                package, request.question, plan_context=None
            )
            if result.is_match:
                matches.append(result)
                matched_ids.append(track.track_id)
                if request.top_k and len(matched_ids) >= request.top_k:
                    break
        return SearchResponse(matches=matches, matched_track_ids=matched_ids)


def build_default_use_case() -> SearchTracksUseCase:
    """Helper to build use case with in-memory defaults."""
    repo: StoragePort = InMemoryTrackRepository()
    inference: InferencePort = MockInferenceAdapter(default_match=True)
    return SearchTracksUseCase(inference=inference, track_repo=repo)
