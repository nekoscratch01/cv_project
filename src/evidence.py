"""Evidence package utilities for question-driven retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from features import TrackFeatures
from perception import TrackRecord, VideoMetadata


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
        motion: 运动特征对象（可选），包含速度、路径长度、持续时间等
    
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
    motion: TrackFeatures | None = None

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
        packages[track_id] = EvidencePackage(
            video_id=video_id,
            track_id=track_id,
            frames=list(record.frames),       # 复制列表，避免引用
            bboxes=list(record.bboxes),       # 复制列表
            crops=list(record.crops),         # 复制列表
            fps=metadata.fps,
            motion=features.get(track_id),    # 如果不存在则为 None
        )
    return packages
