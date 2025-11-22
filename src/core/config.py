"""Centralized configuration for the semantic video system."""
# field：用来定义特殊的字段（比如"这个字段不在初始化时设置，而是后面自动生成"）
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemConfig:
    """Runtime configuration shared across pipeline modules."""

    # I/O
    video_path: Path = Path("data/raw/core/MOT17-09.mp4")
    output_dir: Path = Path("output")

    # Detection / tracking
    yolo_model: str = "yolo11n.pt"  # upgrade to YOLOv11 weights
    yolo_conf: float = 0.3
    yolo_device: str = "mps"
    tracker_type: str = "bytetrack"

    # Sampling
    sample_interval: int = 2   # 更密集采样，避免漏掉最清晰帧
    min_track_length: int = 1  # 放宽轨迹长度过滤，保留短轨迹

    # Semantic / VLM (v7 默认：单 Qwen3‑VL‑4B，经 transformers；保留 llama-cpp 接口)
    vlm_backend: str = "hf"  # 允许显式设为 "llama_cpp" 以备未来扩展
    vlm_gguf_path: Path | None = None
    vlm_context_size: int = 4096
    vlm_gpu_layers: int = -1
    vlm_cpu_threads: int | None = None
    vlm_temperature: float = 0.1
    vlm_max_new_tokens: int = 512
    # Router 默认也使用同一个 4B 模型；当前实现仅提供 transformers 版本
    router_backend: str = "hf"  # 允许保留 "llama_cpp" 占位
    router_gguf_path: Path | None = None
    router_max_new_tokens: int = 256
    router_temperature: float = 0.2
    siglip_model_name: str = "google/siglip-base-patch16-224"
    siglip_device: str = "mps"
    embedding_cache_dir: Path = Path("output/embeddings")

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
        self.embedding_cache_dir = Path(self.embedding_cache_dir)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
