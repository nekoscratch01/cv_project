"""Phase 2 基础行为特征与事件检测工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math

from config import SystemConfig
from perception import TrackRecord, VideoMetadata


@dataclass
class RoiDwell:
    zone: str
    seconds: float


@dataclass
class FollowEvent:
    follower: int
    target: int
    start_s: float
    end_s: float
    min_distance: float


class BehaviorFeatureExtractor:
    """计算 ROI 停留时间等基础行为特征。"""

    def __init__(self, config: SystemConfig, metadata: VideoMetadata) -> None:
        self.config = config
        self.metadata = metadata

    def compute_roi_dwell(self, tracks: Dict[int, TrackRecord]) -> Dict[int, List[RoiDwell]]:
        rois = self.config.roi_zones
        if not rois:
            return {tid: [] for tid in tracks}

        fps = max(self.metadata.fps, 1e-3)
        result: Dict[int, List[RoiDwell]] = {}
        for tid, rec in tracks.items():
            dwell_list: List[RoiDwell] = []
            for name, (x1, y1, x2, y2) in rois:
                frames_inside = 0
                for (_, bbox) in zip(rec.frames, rec.bboxes):
                    bx1, by1, bx2, by2 = bbox
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        frames_inside += 1
                seconds_inside = frames_inside / fps
                if seconds_inside > 0:
                    dwell_list.append(RoiDwell(zone=name, seconds=seconds_inside))
            result[tid] = dwell_list
        return result


class EventDetector:
    """基础事件检测：跟随事件。"""

    def __init__(self, config: SystemConfig, metadata: VideoMetadata) -> None:
        self.config = config
        self.metadata = metadata

    def detect_follow_events(self, tracks: Dict[int, TrackRecord]) -> List[FollowEvent]:
        """简单的跟随事件：两个目标中心点在给定距离内持续一定帧数。"""
        fps = max(self.metadata.fps, 1e-3)
        dist_thresh = self.config.follow_distance_thresh
        min_frames = self.config.follow_min_frames

        track_ids = list(tracks.keys())
        events: List[FollowEvent] = []

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                tid_a = track_ids[i]
                tid_b = track_ids[j]
                rec_a = tracks[tid_a]
                rec_b = tracks[tid_b]

                # 基于帧号对齐，找到共同出现的帧
                frame_to_bbox_a = dict(zip(rec_a.frames, rec_a.bboxes))
                frame_to_bbox_b = dict(zip(rec_b.frames, rec_b.bboxes))
                common_frames = sorted(set(frame_to_bbox_a.keys()) & set(frame_to_bbox_b.keys()))
                if len(common_frames) < min_frames:
                    continue

                consecutive = 0
                min_dist = float("inf")
                start_frame = None
                end_frame = None

                for f in common_frames:
                    xa1, ya1, xa2, ya2 = frame_to_bbox_a[f]
                    xb1, yb1, xb2, yb2 = frame_to_bbox_b[f]
                    ca = ((xa1 + xa2) / 2.0, (ya1 + ya2) / 2.0)
                    cb = ((xb1 + xb2) / 2.0, (yb1 + yb2) / 2.0)
                    dist = math.dist(ca, cb)
                    if dist <= dist_thresh:
                        consecutive += 1
                        min_dist = min(min_dist, dist)
                        if start_frame is None:
                            start_frame = f
                        end_frame = f
                    else:
                        if consecutive >= min_frames and start_frame is not None and end_frame is not None:
                            events.append(
                                FollowEvent(
                                    follower=tid_a,
                                    target=tid_b,
                                    start_s=start_frame / fps,
                                    end_s=end_frame / fps,
                                    min_distance=min_dist,
                                )
                            )
                        consecutive = 0
                        start_frame = None
                        end_frame = None
                        min_dist = float("inf")

                # 检查尾段
                if consecutive >= min_frames and start_frame is not None and end_frame is not None:
                    events.append(
                        FollowEvent(
                            follower=tid_a,
                            target=tid_b,
                            start_s=start_frame / fps,
                            end_s=end_frame / fps,
                            min_distance=min_dist,
                        )
                    )

        return events
