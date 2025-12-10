from __future__ import annotations

from typing import Dict, List

from domain.entities.track import Track
from ports.storage_port import StoragePort


class InMemoryTrackRepository(StoragePort):
    """In-memory repository for tracks (test/demo only)."""

    def __init__(self) -> None:
        self._tracks: Dict[str, Track] = {}

    async def save_track(self, track: Track) -> None:
        self._tracks[f"{track.video_id}_{track.track_id}"] = track

    async def get_tracks_by_video(self, video_id: str) -> List[Track]:
        return [t for t in self._tracks.values() if t.video_id == video_id]
