"""Perception layer: detection, tracking, and visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from boxmot import create_tracker
from ultralytics import YOLO
from collections import defaultdict

from .config import SystemConfig


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    total_frames: int


@dataclass
class TrackRecord:
    track_id: int
    frames: List[int]
    bboxes: List[Tuple[int, int, int, int]]
    crops: List[str]


class VideoPerception:
    """Runs YOLO detection + ByteTrack tracking to collect per-track data."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.yolo = YOLO(config.yolo_model)
        self.tracker = create_tracker(
            tracker_type=config.tracker_type,
            tracker_config=None,
            reid_weights=None,
            device="cpu",
            half=False,
            per_class=False,
        )

    def process(self) -> tuple[Dict[int, TrackRecord], VideoMetadata]:
        cap = cv2.VideoCapture(str(self.config.video_path))
        metadata = VideoMetadata(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )

        track_records: Dict[int, TrackRecord] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = self.yolo.predict(
                source=frame,
                device=self.config.yolo_device,
                conf=self.config.yolo_conf,
                verbose=False,
                classes=[0],
            )[0]

            detections = []
            boxes = results.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    detections.append([x1, y1, x2, y2, conf, cls])

            if not detections:
                continue

            tracks = self.tracker.update(np.array(detections), frame)
            if tracks.size == 0:
                continue

            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])

                record = track_records.setdefault(
                    track_id,
                    TrackRecord(track_id=track_id, frames=[], bboxes=[], crops=[]),
                )
                record.frames.append(frame_idx)
                record.bboxes.append((x1, y1, x2, y2))

                # sampling for crops
                if len(record.frames) % self.config.sample_interval == 1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_path = (
                            self.config.crops_dir
                            / f"id{track_id:03d}_frame{frame_idx:05d}.jpg"
                        )
                        cv2.imwrite(str(crop_path), crop)
                        record.crops.append(str(crop_path))

        cap.release()

        filtered = {
            tid: rec
            for tid, rec in track_records.items()
            if len(rec.frames) >= self.config.min_track_length
        }

        return filtered, metadata

    def render_highlight_video(
        self,
        track_records: Dict[int, TrackRecord],
        metadata: VideoMetadata,
        target_ids: list[int],
        output_path,
        label_text: str,
    ) -> None:
        if not target_ids:
            print("   ⚠️  没有目标需要可视化，跳过视频导出")
            return

        target_set = set(target_ids)
        frame_map = defaultdict(list)
        for tid in target_set:
            record = track_records.get(tid)
            if not record:
                continue
            for frame_idx, bbox in zip(record.frames, record.bboxes):
                frame_map[frame_idx].append((tid, bbox))

        cap = cv2.VideoCapture(str(self.config.video_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path), fourcc, metadata.fps, (metadata.width, metadata.height)
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            tracks_this_frame = frame_map.get(frame_idx, [])
            for tid, bbox in tracks_this_frame:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    self.config.highlight_color,
                    3,
                )
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    self.config.highlight_color,
                    2,
                )

            if tracks_this_frame:
                cv2.putText(
                    frame,
                    f"Tracking {label_text}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    self.config.highlight_color,
                    3,
                )

            out.write(frame)

        cap.release()
        out.release()
