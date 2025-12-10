"""Centralized configuration for the semantic video system."""
# field：用来定义特殊的字段（比如"这个字段不在初始化时设置，而是后面自动生成"）
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemConfig:
    """Runtime configuration shared across pipeline modules."""

    # I/O
    video_path: Path = Path(__file__).resolve().parents[2] / "data/raw/core/MOT17-09.mp4"
    output_dir: Path = Path("output")

    # Detection / tracking
    yolo_model: str = "yolo11n.pt"  # upgrade to YOLOv11 weights
    yolo_conf: float = 0.5
    yolo_device: str = "cuda"
    tracker_type: str = "bytetrack"

    # Sampling
    sample_interval: int = 5   # 更稀疏存图，聚焦关键帧
    min_track_length: int = 15  # 过滤闪烁短轨迹

    # Semantic / VLM (Phase1: 默认 vLLM 服务)
    vlm_backend: str = "vllm"  # 强制 vLLM
    vllm_endpoint: str = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_gguf_path: Path | None = None
    vlm_context_size: int = 8192
    vlm_gpu_layers: int = -1
    vlm_cpu_threads: int | None = None
    vlm_temperature: float = 0.1
    vlm_max_new_tokens: int = 1024  # 提高上限，避免批次回答被截断
    vlm_batch_size: int = 4
    # Router 默认改为轻量 simple（避免 transformers 依赖）
    router_backend: str = "simple"
    router_gguf_path: Path | None = None
    router_max_new_tokens: int = 256
    router_temperature: float = 0.2
    siglip_model_name: str = "google/siglip-base-patch16-224"
    siglip_device: str = "cuda"
    clip_filter_threshold: float = 0.05  # 放宽相似度阈值，避免误杀
    embedding_cache_dir: Path = Path("output/embeddings")
    # Filmstrip (Layer2) settings
    filmstrip_enabled: bool = True
    filmstrip_frame_count: int = 5
    filmstrip_max_width: int = 4096  # 最终拼接图的最大宽度
    enable_clip_filter: bool = False  # CLIP/SigLIP 召回预过滤开关（临时关闭以便直接进入 VLM）

    # Visualization
    highlight_color: tuple[int, int, int] = (0, 0, 255)
    max_preview_tracks: int = 10

    # Behavior / ROI (Phase 2 hooks)
    # 定义 ROI 区域用于停留时间等行为分析，格式: [(name, (x1,y1,x2,y2)), ...]
    roi_zones: list[tuple[str, tuple[int, int, int, int]]] = field(default_factory=list)
    # 跟随事件检测阈值（像素距离）
    follow_distance_thresh: float = 80.0
    follow_min_frames: int = 30

    # Derived paths (initialized post-creation)
    crops_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        # 允许从任意工作目录运行：相对路径统一相对项目根目录解析
        if not self.video_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            self.video_path = (project_root / self.video_path).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir = self.output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir = Path(self.embedding_cache_dir)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
