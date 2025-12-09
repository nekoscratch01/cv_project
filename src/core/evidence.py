"""Evidence package utilities for question-driven retrieval."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.features import TrackFeatures
from core.perception import TrackRecord, VideoMetadata


@dataclass
class EvidencePackage:
    """
    证据包：一个人在视频中的完整档案（打包好的"人物卡片"）。
    
    把来自3个不同来源的信息打包在一起：
    1. perception.py 提供的：帧号、检测框、裁剪图路径
    2. features.py 提供的：运动特征（速度、时长等）
    3. metadata 提供的：fps（用于自动计算时间）
    
    这是整个系统的核心数据交换格式，后续所有模块（召回、VLM、可视化）
    都只需要拿 EvidencePackage，不用管数据来自哪里。
    
    Attributes:
        video_id: 视频的唯一标识符（通常是文件名），例如 "shopping_mall"
        track_id: 这个人的唯一编号，例如 1, 2, 3...
        frames: 这个人出现在哪些帧，例如 [1, 2, 3, ..., 900]
        bboxes: 每一帧中的检测框，例如 [(50,100,150,300), ...]
        crops: 保存的裁剪图文件路径列表，例如 ["crops/id001_frame00001.jpg", ...]
        fps: 视频帧率，用于将帧号转换成时间
        features: Atomic 8 特征对象（可选）
        meta: 额外元数据（video_id/fps/resolution 等）
        raw_trace: 对齐后的整段检测框序列
        embedding: 视觉向量（SigLIP/CLIP 预留字段）
    
    使用示例：
        package = evidence_map[1]
        print(f"Track {package.track_id} 出现在 {package.start_time_seconds:.1f}s 到 {package.end_time_seconds:.1f}s")
        print(f"平均速度: {package.motion.avg_speed_px_s:.1f} 像素/秒")
    """
    video_id: str
    track_id: int
    frames: List[int]
    bboxes: List[Tuple[int, int, int, int]]
    crops: List[str]
    fps: float
    video_path: str = ""
    best_bbox_index: int = -1  # 面积+中心性选出的最佳截图索引
    features: Optional[TrackFeatures] = None
    meta: Optional[Dict[str, Any]] = None
    raw_trace: Optional[List[Tuple[int, int, int, int]]] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {"video_id": self.video_id, "fps": self.fps}
        if self.raw_trace is None:
            # 默认让 raw_trace 是 bboxes 的一个副本，保持解耦
            self.raw_trace = list(self.bboxes)

    @property
    def motion(self) -> Optional[TrackFeatures]:
        """与旧代码兼容的别名。"""
        return self.features

    @motion.setter
    def motion(self, value: Optional[TrackFeatures]) -> None:
        self.features = value

    @property
    def start_frame(self) -> int:
        """
        获取起始帧号（第一次出现的帧）。
        
        Returns:
            int: 起始帧号，如果frames为空则返回0
        """
        return self.frames[0] if self.frames else 0

    @property
    def end_frame(self) -> int:
        """
        获取结束帧号（最后一次出现的帧）。
        
        Returns:
            int: 结束帧号，如果frames为空则返回0
        """
        return self.frames[-1] if self.frames else 0

    @property
    def start_time_seconds(self) -> float:
        """
        自动计算起始时间（秒）。
        
        Returns:
            float: 起始时间，单位秒。例如 0.03 表示第0.03秒
                  计算公式：start_frame / fps
        """
        return self.start_frame / self.fps if self.fps > 0 else 0.0

    @property
    def end_time_seconds(self) -> float:
        """
        自动计算结束时间（秒）。
        
        Returns:
            float: 结束时间，单位秒。例如 30.0 表示第30秒
                  计算公式：end_frame / fps
        """
        return self.end_frame / self.fps if self.fps > 0 else 0.0


def build_evidence_packages(
    video_id: str,
    track_records: Dict[int, TrackRecord],
    metadata: VideoMetadata,
    features: Dict[int, TrackFeatures],
    video_path: str | Path = "",
) -> Dict[int, EvidencePackage]:
    """
    构建证据包字典：把分散的数据打包成统一格式。
    
    这个函数是"打包工"，负责把前面3个阶段产生的数据整合在一起：
    - perception.py 生成的 track_records（帧号、框、裁剪图）
    - features.py 生成的 features（速度、时长等统计数据）
    - metadata（视频的fps信息）
    
    打包后，下游模块（召回、VLM、可视化）只需要拿 EvidencePackage，
    不用关心数据来自哪里，也不用自己计算时间戳。
    
    Args:
        video_id: 视频的唯一标识符（通常是文件名），例如 "shopping_mall"
        track_records: 轨迹记录字典，来自 VideoPerception.process()
                      格式：{1: TrackRecord(...), 2: TrackRecord(...), ...}
        metadata: 视频元数据，来自 VideoPerception.process()
                 主要需要 fps 字段
        features: 特征字典，来自 TrackFeatureExtractor.extract()
                 格式：{1: TrackFeatures(...), 2: TrackFeatures(...), ...}
    
    Returns:
        证据包字典，格式：{track_id: EvidencePackage, ...}
        例如：{1: EvidencePackage(video_id="demo", track_id=1, ...), ...}
        
    Note:
        - 输出字典的 key 和 track_records 的 key 完全一致
        - 如果某个 track_id 在 features 中不存在，对应的 motion 字段会是 None
        - 所有列表字段都会被复制（而不是引用），避免意外修改原始数据
    
    使用示例：
        # 前面3个阶段
        track_records, metadata = perception.process()
        features = feature_extractor.extract(track_records)
        
        # 打包
        evidence_map = build_evidence_packages("video1", track_records, metadata, features)
        
        # 使用
        package = evidence_map[1]
        print(f"找到 track {package.track_id}，出现时间 {package.start_time_seconds}s")
    """
    packages: Dict[int, EvidencePackage] = {}
    for track_id, record in track_records.items():
        best_idx = _select_best_bbox_index(record, metadata)
        packages[track_id] = EvidencePackage(
            video_id=video_id,
            video_path=str(video_path) if video_path else "",
            track_id=track_id,
            frames=list(record.frames),       # 复制列表，避免引用
            bboxes=list(record.bboxes),       # 复制列表
            crops=list(record.crops),         # 复制列表
            fps=metadata.fps,
            best_bbox_index=best_idx,
            features=features.get(track_id),  # 如果不存在则为 None
            meta={
                "video_id": video_id,
                "fps": metadata.fps,
                "resolution": (metadata.width, metadata.height),
            },
            raw_trace=list(record.bboxes),
        )
    return packages


def _select_best_bbox_index(record: TrackRecord, meta: VideoMetadata) -> int:
    """
    选择最佳截图索引：面积为主，中心性为辅，剔除贴边框。
    """
    if not record.bboxes:
        return -1

    cx, cy = meta.width / 2, meta.height / 2
    best_idx = -1
    best_score = -1.0

    for i, (x1, y1, x2, y2) in enumerate(record.bboxes):
        # 边缘剔除：避免截断人物
        if x1 < 5 or y1 < 5 or x2 > meta.width - 5 or y2 > meta.height - 5:
            continue
        area = (x2 - x1) * (y2 - y1)
        box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = ((box_cx - cx) ** 2 + (box_cy - cy) ** 2) ** 0.5
        max_dist = ((meta.width ** 2 + meta.height ** 2) ** 0.5) / 2
        centrality = 1.0 - dist / max(max_dist, 1.0)
        score = area * (1.0 + 0.5 * centrality)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        # 全部贴边被过滤，退回面积最大
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in record.bboxes]
        best_idx = int(np.argmax(areas)) if areas else -1
    return best_idx


def extract_best_crop_from_package(pkg: EvidencePackage, pad: float = 0.15) -> str:
    """
    从原视频中提取最佳特写（base64）。
    优先使用 best_bbox_index；缺失则取中间帧的框。
    """
    if not pkg.frames or not pkg.bboxes or not pkg.video_path:
        return ""

    idx = getattr(pkg, "best_bbox_index", -1)
    if idx < 0 or idx >= len(pkg.bboxes):
        idx = len(pkg.bboxes) // 2

    frame_id = pkg.frames[idx]
    x1, y1, x2, y2 = pkg.bboxes[idx]

    cap = cv2.VideoCapture(pkg.video_path)
    if not cap.isOpened():
        return ""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return ""

    h, w = frame.shape[:2]
    pad_x = int((x2 - x1) * pad)
    pad_y = int((y2 - y1) * pad)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)
    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return ""
    _, buffer = cv2.imencode(".jpg", crop)
    return base64.b64encode(buffer).decode("utf-8")
