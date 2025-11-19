"""Feature extraction utilities built on top of raw track records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import numpy as np

from perception import TrackRecord, VideoMetadata


@dataclass
class TrackFeatures:
    """
    一个人的运动特征统计（"体检报告"）。
    
    存储从轨迹中提取的数值特征，包括速度、路径长度、持续时间等。
    这些特征用于：
    1. 给VLM提供数值上下文（例如"这个人速度很慢，可能在徘徊"）
    2. 未来做行为检测（例如"找出快速移动的人"）
    3. 质量检查（例如过滤持续时间太短的轨迹）
    
    Attributes:
        track_id: 轨迹的唯一标识符
        avg_speed_px_s: 平均速度（像素/秒）。例如 75.0 表示平均每秒移动75像素
        max_speed_px_s: 最大速度（像素/秒）。例如 120.0 表示最快时每秒移动120像素
        path_length_px: 总路径长度（像素）。累加所有相邻帧之间的移动距离
        duration_s: 持续时间（秒）。从第一帧到最后一帧的时间跨度
    """
    track_id: int
    avg_speed_px_s: float
    max_speed_px_s: float
    path_length_px: float
    duration_s: float

    def to_dict(self) -> Dict[str, float]:
        """
        将特征对象转换为字典格式，用于JSON序列化。
        
        Returns:
            Dict[str, float]: 包含4个运动特征的字典，key是特征名，value是数值
        """
        return {
            "avg_speed_px_s": self.avg_speed_px_s,
            "max_speed_px_s": self.max_speed_px_s,
            "path_length_px": self.path_length_px,
            "duration_s": self.duration_s,
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

        for track_id, record in tracks.items():
            # 计算每个边界框的中心点坐标
            centers = [self._bbox_center(b) for b in record.bboxes]
            if len(centers) < 2:
                features[track_id] = TrackFeatures(
                    track_id, avg_speed_px_s=0.0, max_speed_px_s=0.0, path_length_px=0.0, duration_s=0.0
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
