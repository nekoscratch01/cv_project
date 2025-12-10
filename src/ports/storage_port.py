from __future__ import annotations

from typing import Protocol, List

from domain.entities.track import Track


class StoragePort(Protocol):
    """Storage abstraction for tracks/evidence metadata."""

    async def save_track(self, track: Track) -> None:
        ...

    async def get_tracks_by_video(self, video_id: str) -> List[Track]:
        ...
