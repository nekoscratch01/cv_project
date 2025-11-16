"""Feature extraction utilities built on top of raw track records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import numpy as np

from .perception import TrackRecord, VideoMetadata


@dataclass
class TrackFeatures:
    track_id: int
    avg_speed_px_s: float
    max_speed_px_s: float
    path_length_px: float
    duration_s: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "avg_speed_px_s": self.avg_speed_px_s,
            "max_speed_px_s": self.max_speed_px_s,
            "path_length_px": self.path_length_px,
            "duration_s": self.duration_s,
        }


class TrackFeatureExtractor:
    """Compute simple motion features from trajectories."""

    def __init__(self, metadata: VideoMetadata):
        self.metadata = metadata

    def extract(self, tracks: Dict[int, TrackRecord]) -> Dict[int, TrackFeatures]:
        features: Dict[int, TrackFeatures] = {}
        fps = max(self.metadata.fps, 1e-3)

        for track_id, record in tracks.items():
            centers = [self._bbox_center(b) for b in record.bboxes]
            if len(centers) < 2:
                features[track_id] = TrackFeatures(
                    track_id, avg_speed_px_s=0.0, max_speed_px_s=0.0, path_length_px=0.0, duration_s=0.0
                )
                continue

            distances = []
            speeds = []
            total_length = 0.0

            for i in range(1, len(centers)):
                c_prev, c_curr = centers[i - 1], centers[i]
                dist = math.dist(c_prev, c_curr)
                frame_delta = max(record.frames[i] - record.frames[i - 1], 1)
                time_delta = frame_delta / fps
                if time_delta <= 0:
                    continue
                speed = dist / time_delta
                speeds.append(speed)
                distances.append(dist)
                total_length += dist

            duration_seconds = max((record.frames[-1] - record.frames[0]) / fps, 0.0)
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            max_speed = float(np.max(speeds)) if speeds else 0.0

            features[track_id] = TrackFeatures(
                track_id=track_id,
                avg_speed_px_s=avg_speed,
                max_speed_px_s=max_speed,
                path_length_px=total_length,
                duration_s=duration_seconds,
            )

        return features

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
