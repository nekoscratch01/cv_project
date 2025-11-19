"""Centralized configuration for the semantic video system."""
# field：用来定义特殊的字段（比如"这个字段不在初始化时设置，而是后面自动生成"）
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemConfig:
    """Runtime configuration shared across pipeline modules."""

    # I/O
    video_path: Path = Path("/Users/neko_wen/my/代码/uw/cv/project/data/raw/core/MOT17-09.mp4")
    output_dir: Path = Path("output_full_system_MOT17_04")

    # Detection / tracking
    yolo_model: str = "yolo11n.pt"  # upgrade to YOLOv11 weights
    yolo_conf: float = 0.3
    yolo_device: str = "mps"
    tracker_type: str = "bytetrack"

    # Sampling
    sample_interval: int = 30
    min_track_length: int = 10

    # Semantic / VLM
    vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Visualization
    highlight_color: tuple[int, int, int] = (0, 0, 255)
    max_preview_tracks: int = 5

    # Behavior / ROI (Phase 2 hooks)
    # 定义 ROI 区域用于停留时间等行为分析，格式: [(name, (x1,y1,x2,y2)), ...]
    roi_zones: list[tuple[str, tuple[int, int, int, int]]] = field(default_factory=list)
    # 跟随事件检测阈值（像素距离）
    follow_distance_thresh: float = 80.0
    follow_min_frames: int = 30

    # Derived paths (initialized post-creation)
    crops_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir = self.output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
