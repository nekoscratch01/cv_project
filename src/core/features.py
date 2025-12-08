"""Feature extraction utilities built on top of raw track records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math

import numpy as np

from core.perception import TrackRecord, VideoMetadata


@dataclass
class TrackFeatures:
    """
    一个人在 Atomic 8 协议下的“体检报告”。
    
    v7 开始，这份报告除了速度/时长，还要提供几何真相：
        - 起止时间戳（秒）
        - 归一化的中心点轨迹（centroids 0~1）
        - 首尾位移向量（displacement_vec）
    
    这些字段都是纯几何量，不带任何语义标签，供 Hard Rules 与 VLM 复用。
    """

    track_id: int
    start_s: float
    end_s: float
    duration_s: float
    centroids: List[Tuple[float, float]]
    displacement_vec: Tuple[float, float]
    avg_speed_px_s: float
    max_speed_px_s: float
    path_length_px: float
    norm_speed: float = 0.0
    linearity: float = 0.0
    scale_change: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """
        将特征对象转换为字典格式，用于JSON序列化。
        
        Returns:
            Dict[str, Any]: 包含 Atomic 8 字段的字典。
        """
        return {
            "track_id": self.track_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "centroids": [[float(x), float(y)] for x, y in self.centroids],
            "displacement_vec": [float(self.displacement_vec[0]), float(self.displacement_vec[1])],
            "avg_speed_px_s": self.avg_speed_px_s,
            "max_speed_px_s": self.max_speed_px_s,
            "path_length_px": self.path_length_px,
            "norm_speed": self.norm_speed,
            "linearity": self.linearity,
            "scale_change": self.scale_change,
        }


class TrackFeatureExtractor:
    """
    轨迹特征提取器，从原始轨迹记录中计算运动特征。
    
    负责把一串GPS式的坐标点（轨迹）转换成可读的统计数字，
    例如：[900个坐标点] → [平均速度、最大速度、路径长度、持续时间]
    
    使用方法：
        extractor = TrackFeatureExtractor(metadata)
        features = extractor.extract(track_records)
    """

    def __init__(self, metadata: VideoMetadata):
        """
        初始化特征提取器。
        
        Args:
            metadata: 视频元数据，主要需要fps（帧率）来将帧数转换成时间（秒）
        """
        self.metadata = metadata

    def extract(self, tracks: Dict[int, TrackRecord]) -> Dict[int, TrackFeatures]:
        """
        从轨迹记录中提取运动特征。
        
        核心算法：
        1. 把检测框转换成中心点坐标（更准确表示"人的位置"）
        2. 计算相邻点之间的距离（移动了多远）
        3. 计算相邻点之间的时间（用了多久）
        4. 距离 ÷ 时间 = 速度
        5. 统计：平均速度、最大速度、累加总距离、计算总时长
        
        Args:
            tracks: 轨迹记录字典，key是track_id，value是TrackRecord对象
                   例如：{1: TrackRecord(...), 2: TrackRecord(...)}
        
        Returns:
            特征字典，key是track_id，value是TrackFeatures对象
            例如：{1: TrackFeatures(avg_speed=75.0, ...), 2: TrackFeatures(...)}
            
        Note:
            - 如果某个轨迹少于2个点，会返回全0的特征（无法计算速度）
            - 输出字典的key和输入字典的key完全一致（不会"丢人"）
        """
        features: Dict[int, TrackFeatures] = {}
        fps = max(self.metadata.fps, 1e-3)  # 防止fps为0导致除零错误
        width = max(self.metadata.width, 1)
        height = max(self.metadata.height, 1)

        for track_id, record in tracks.items():
            # 计算中心点和面积，方便后续语义特征
            centers = [self._bbox_center(b) for b in record.bboxes]
            centroids = [self._normalize_center(c, width, height) for c in centers]
            areas = [self._bbox_area(b) for b in record.bboxes]
            avg_height = sum((b[3] - b[1]) for b in record.bboxes) / max(len(record.bboxes), 1)

            start_frame = record.frames[0] if record.frames else 0
            end_frame = record.frames[-1] if record.frames else 0
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration_seconds = max(end_time - start_time, 0.0)
            displacement_vec = self._compute_displacement(centroids)

            if len(centers) < 2:
                features[track_id] = TrackFeatures(
                    track_id=track_id,
                    start_s=start_time,
                    end_s=end_time,
                    duration_s=duration_seconds,
                    centroids=centroids,
                    displacement_vec=displacement_vec,
                    avg_speed_px_s=0.0,
                    max_speed_px_s=0.0,
                    path_length_px=0.0,
                    norm_speed=0.0,
                    linearity=0.0,
                    scale_change=1.0,
                )
                continue
            # 计算每个边界框的中心点坐标之间的距
            distances = []  # 中心点坐标之间的距离
            speeds = []  # 速度
            total_length = 0.0  # 总长度

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

            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            max_speed = float(np.max(speeds)) if speeds else 0.0

            # 归一化速度（身长/秒）：用平均框高作为身长尺度，加保护下限
            norm_speed = 0.0
            if duration_seconds > 0:
                avg_height = max(avg_height, 10.0)
                norm_speed = (avg_speed) / avg_height if avg_height > 0 else 0.0

            # 线性度：位移 / 路径长度，极短路径视为徘徊
            linearity = 0.0
            if total_length >= 50.0:
                linearity = float(math.hypot(*displacement_vec)) / max(total_length, 1.0)

            # 尺度变化：前5帧平均面积 vs 后5帧平均面积，平滑抖动
            head_area = np.mean(areas[:5]) if areas else 1.0
            tail_area = np.mean(areas[-5:]) if areas else 1.0
            head_area = max(head_area, 1.0)
            scale_change = float(tail_area / head_area)

            features[track_id] = TrackFeatures(
                track_id=track_id,
                start_s=start_time,
                end_s=end_time,
                duration_s=duration_seconds,
                centroids=centroids,
                displacement_vec=displacement_vec,
                avg_speed_px_s=avg_speed,
                max_speed_px_s=max_speed,
                path_length_px=total_length,
                norm_speed=norm_speed,
                linearity=linearity,
                scale_change=scale_change,
            )

        return features

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        计算检测框的中心点坐标。
        
        Args:
            bbox: 检测框坐标，格式为 (x1, y1, x2, y2)
                 其中 (x1, y1) 是左上角，(x2, y2) 是右下角
                 例如：(100, 200, 150, 300)
        
        Returns:
            中心点坐标 (center_x, center_y)，格式为浮点数元组
            例如：(125.0, 250.0) 表示中心点在 x=125, y=250
            
        Note:
            返回float而不是int，因为中心点可能是小数
            例如框 (100, 200, 151, 301) 的中心是 (125.5, 250.5)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _normalize_center(center: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
        """把像素坐标归一化到 [0, 1] 区间，并剪裁异常值。"""
        cx, cy = center
        norm_x = TrackFeatureExtractor._clamp(cx / max(width, 1))
        norm_y = TrackFeatureExtractor._clamp(cy / max(height, 1))
        return (norm_x, norm_y)

    @staticmethod
    def _clamp(value: float) -> float:
        return float(min(max(value, 0.0), 1.0))

    @staticmethod
    def _compute_displacement(centroids: List[Tuple[float, float]]) -> Tuple[float, float]:
        if len(centroids) < 2:
            return (0.0, 0.0)
        start_x, start_y = centroids[0]
        end_x, end_y = centroids[-1]
        return (float(end_x - start_x), float(end_y - start_y))

    @staticmethod
    def _bbox_area(bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        return float(max(x2 - x1, 0) * max(y2 - y1, 0))
