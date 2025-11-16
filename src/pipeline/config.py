"""Centralized configuration for the semantic video system."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemConfig:
    """Runtime configuration shared across pipeline modules."""

    # I/O
    video_path: Path = Path("/Users/neko_wen/my/代码/uw/cv/project/data/raw/core/MOT17-04.mp4")
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

    # Derived paths (initialized post-creation)
    crops_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir = self.output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
