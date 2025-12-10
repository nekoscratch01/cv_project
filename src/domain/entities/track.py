from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Track:
    """Core track entity."""

    track_id: int
    video_id: str
    frames: List[int]
    bboxes: List[Tuple[int, int, int, int]]
    fps: float

    @property
    def duration_seconds(self) -> float:
        if not self.frames or self.fps <= 0:
            return 0.0
        return (self.frames[-1] - self.frames[0]) / self.fps

    @property
    def is_valid(self) -> bool:
        return self.duration_seconds >= 0.5 and len(self.frames) >= 3
