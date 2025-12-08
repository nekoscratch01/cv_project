"""Perception layer: detection, tracking, and visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from boxmot import create_tracker
from ultralytics import YOLO
from collections import defaultdict

from core.config import SystemConfig


@dataclass
class VideoMetadata:
    """
    视频的基础元数据，用于后续的时间对齐和视频生成。
    在内存中流转，主要被 FeatureExtractor (算速度) 和 EvidencePackage (算时间戳) 使用。
    """
    fps: float          # 帧率，用于将帧号转换为秒 (frame_idx / fps = seconds)
    width: int          # 视频画面宽度
    height: int         # 视频画面高度
    total_frames: int   # 总帧数



@dataclass
class TrackRecord:
    track_id: int
    frames: List[int]
    bboxes: List[Tuple[int, int, int, int]]
    crops: List[str]

# TrackRecord(
#     track_id=3,
#     frames=[10, 40, 70],
#     bboxes=[(100, 200, 150, 300), (105, 205, 155, 305), (110, 210, 160, 310)],
#     crops=["crops/id003_frame00010.jpg", "crops/id003_frame00070.jpg"]
# )

class VideoPerception:
    """Runs YOLO detection + ByteTrack tracking to collect per-track data."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.yolo = YOLO(config.yolo_model)
        # 允许多类别跟踪，后续下游可按类别过滤
        self.tracker = create_tracker(
            tracker_type=config.tracker_type,
            tracker_config=None,
            reid_weights=None,
            device="cpu",
            half=False,
            per_class=True,
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
                classes=[0],  # 仅检测 person 类
            )[0]  # predict 返回一个 list，针对单张图片我们只取第 0 个结果

            # --- 详细注释：YOLO 返回的 results 数据结构 (JSON 格式展示) ---
            # results 是一个 ultralytics.engine.results.Results 对象
            # 主要访问 results.boxes，包含检测到的所有框信息
            # 
            # 当有检测结果时的 JSON 结构示例 (假设检测到2个人):
            # {
            #     "results": {
            #         "boxes": {
            #             "xyxy": [
            #                 [100.0, 150.0, 200.0, 300.0],  // 第1人: 左上角(100,150), 右下角(200,300)
            #                 [300.0, 200.0, 400.0, 350.0]   // 第2人: 左上角(300,200), 右下角(400,350)
            #             ],
            #             "conf": [0.92, 0.87],           // 置信度分数 (0.0 ~ 1.0)
            #             "cls": [0.0, 0.0],              // 类别索引 (COCO: 0=person)
            #             "id": [1.0, 2.0]                // 跟踪ID (ByteTrack跟踪时才有, 否则null)
            #         },
            #         "masks": null,                      // 分割掩码 (未启用时为null)
            #         "keypoints": null,                  // 关键点 (未启用时为null)
            #         "probs": null,                      // 分类概率 (分类模式时使用)
            #         "orig_shape": [1080, 1920],         // 原始图像尺寸 [height, width]
            #         "names": {"0": "person"},           // 类别名称映射
            #         "path": "path/to/image.jpg"         // 图像路径
            #     }
            # }
            # 
            # 当无检测结果时:
            # {
            #     "results": {
            #         "boxes": null,
            #         "orig_shape": [1080, 1920],
            #         "names": {"0": "person"},
            #         "path": "path/to/image.jpg"
            #     }
            # }
            # 
            # 注意: 实际数据是 torch.Tensor，需要 .cpu().numpy() 转换为 Python 可读格式
            # ------------------------------------------

            detections = []
            boxes = results.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # .cpu().numpy() 把 GPU 上的 Tensor 转回 CPU 并变成 numpy 数组
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])  # 这里的 cls 肯定是 0，因为上面 classes=[0]
                    detections.append([x1, y1, x2, y2, conf, cls])

            if not detections:
                continue

            tracks = self.tracker.update(np.array(detections), frame)
            if tracks.size == 0:
                continue

            # ========== 核心逻辑：为每个跟踪目标构建轨迹记录 ==========
            # 遍历当前帧中 ByteTrack 返回的所有有效跟踪结果
            # 每个 track 是一个 numpy 数组，格式为: [x1, y1, x2, y2, track_id]
            # 其中前4个是边界框坐标，最后1个是跟踪ID
            for track in tracks:
                # 提取边界框坐标并转换为整数（像素坐标必须是int）
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])  # 跟踪ID，唯一标识同一个人

                # 使用 setdefault 模式初始化轨迹记录
                # 如果 track_id 不存在，则创建新的 TrackRecord 对象
                # TrackRecord 包含：frames(出现帧号列表)、bboxes(边界框列表)、crops(裁剪图路径列表)
                record = track_records.setdefault(
                    track_id,
                    TrackRecord(track_id=track_id, frames=[], bboxes=[], crops=[]),
                )
                
                # 记录当前帧的信息到该轨迹
                record.frames.append(frame_idx)      # 当前帧号
                record.bboxes.append((x1, y1, x2, y2))  # 当前边界框

                # ========== 裁剪图采样逻辑 ==========
                # 为了节省存储空间，不保存每帧的裁剪图，而是按固定间隔采样
                # self.config.sample_interval 通常设为 5-10，意思是每隔 N 帧保存一张代表性图片
                # 这样每个轨迹大约会有 3-6 张图片，用于后续 VLM 分析
                if len(record.frames) % self.config.sample_interval == 1:
                    # 从当前帧中裁剪出目标人物区域
                    # OpenCV 切片语法: frame[y1:y2, x1:x2]，注意是 (行, 列) 顺序
                    crop = frame[y1:y2, x1:x2]
                    
                    # 安全检查：确保裁剪区域不为空
                    if crop.size > 0:
                        # 生成唯一的裁剪图文件名
                        # 格式：id001_frame00023.jpg
                        # track_id 用 03d 格式化（补零到3位），frame_idx 用 05d（补零到5位）
                        crop_path = (
                            self.config.crops_dir
                            / f"id{track_id:03d}_frame{frame_idx:05d}.jpg"
                        )
                        
                        # 保存裁剪图到磁盘（crops_dir 是配置的输出目录）
                        cv2.imwrite(str(crop_path), crop)
                        
                        # 记录裁剪图的相对路径到轨迹记录中
                        # 后续 VLM 分析时会读取这些图片文件
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
            print("   ⚠️  No targets to visualize; skip video export")
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

        # 先用 mp4v（OpenCV 默认最稳定的 MP4 编码），失败再尝试 avc1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, metadata.fps, (metadata.width, metadata.height))
        if not out.isOpened():
            print("   ⚠️ Failed to open mp4v encoder, trying avc1 ...")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(str(output_path), fourcc, metadata.fps, (metadata.width, metadata.height))
        if not out.isOpened():
            print(f"   ❌ Cannot create output video file: {output_path}")
            cap.release()
            return

        frame_idx = 0
        written = 0
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
            written += 1

        cap.release()
        out.release()
        if written == 0:
            print(f"   ⚠️ Video {output_path} has 0 frames written; player may not open it.")
