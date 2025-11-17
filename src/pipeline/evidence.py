"""Evidence package utilities for question-driven retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .features import TrackFeatures
from .perception import TrackRecord, VideoMetadata


@dataclass
class EvidencePackage:
    video_id: str
    track_id: int
    frames: List[int]
    bboxes: List[Tuple[int, int, int, int]]
    crops: List[str]
    fps: float
    motion: TrackFeatures | None = None

    @property
    def start_frame(self) -> int:
        return self.frames[0] if self.frames else 0

    @property
    def end_frame(self) -> int:
        return self.frames[-1] if self.frames else 0

    @property
    def start_time_seconds(self) -> float:
        return self.start_frame / self.fps if self.fps > 0 else 0.0

    @property
    def end_time_seconds(self) -> float:
        return self.end_frame / self.fps if self.fps > 0 else 0.0


def build_evidence_packages(
    video_id: str,
    track_records: Dict[int, TrackRecord],
    metadata: VideoMetadata,
    features: Dict[int, TrackFeatures],
) -> Dict[int, EvidencePackage]:
    packages: Dict[int, EvidencePackage] = {}
    for track_id, record in track_records.items():
        packages[track_id] = EvidencePackage(
            video_id=video_id,
            track_id=track_id,
            frames=list(record.frames),
            bboxes=list(record.bboxes),
            crops=list(record.crops),
            fps=metadata.fps,
            motion=features.get(track_id),
        )
    return packages
