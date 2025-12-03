# 🚀 Edge-Detective 工业化路线图

> **文档性质**: 工程实施计划  
> **当前状态**: PoC（功能验证原型）  
> **目标**: 生产级视频语义检索系统

---

## 概览

```
Phase 1（MVP）：API 封装 + 数据库 + 基础监控
     ↓
Phase 2（生产可用）：向量库 + 批推理 + 容器化部署
     ↓
Phase 3（规模化）：实时流 + 多摄像头 + K8s + Re-ID
```

---

## Phase 1: MVP（最小可行产品）

> **目标**: 从 CLI 原型升级为可被外部系统调用的服务  
> **预计周期**: 2-3 周

### 1.1 服务化 API 层

**当前问题**:
```python
# 目前只有 CLI 入口
if __name__ == "__main__":
    run_demo()
```

**改造方案**: 使用 FastAPI 封装核心接口

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── routes/
│   │   ├── index.py         # POST /index - 视频索引
│   │   ├── search.py        # POST /search - 语义检索
│   │   └── health.py        # GET /health - 健康检查
│   ├── schemas/
│   │   ├── request.py       # Pydantic 请求模型
│   │   └── response.py      # Pydantic 响应模型
│   └── dependencies.py      # 依赖注入（VLM、数据库连接）
```

**核心接口设计**:

| 端点 | 方法 | 功能 | 异步 |
|------|------|------|------|
| `/api/v1/videos` | POST | 上传视频并触发索引 | ✅ |
| `/api/v1/videos/{id}/status` | GET | 查询索引进度 | ❌ |
| `/api/v1/search` | POST | 执行语义检索 | ❌ |
| `/api/v1/tracks/{id}` | GET | 获取单条轨迹详情 | ❌ |
| `/api/v1/tracks/{id}/video` | GET | 下载高亮视频片段 | ❌ |

**请求/响应示例**:

```python
# schemas/request.py
class SearchRequest(BaseModel):
    video_id: str
    question: str
    top_k: int = 5
    recall_limit: Optional[int] = None

# schemas/response.py
class TrackResult(BaseModel):
    track_id: int
    start_s: float
    end_s: float
    score: float
    reason: str
    thumbnail_url: Optional[str]

class SearchResponse(BaseModel):
    video_id: str
    question: str
    results: List[TrackResult]
    highlight_video_url: Optional[str]
    processing_time_ms: int
```

### 1.2 异步任务队列

**问题**: 视频索引是重型任务（5分钟视频可能需要10分钟处理），不能阻塞 API

**方案**: Celery + Redis

```python
# tasks/indexing.py
from celery import Celery

app = Celery('edge_detective', broker='redis://localhost:6379/0')

@app.task(bind=True)
def index_video_task(self, video_path: str, video_id: str):
    """后台索引任务"""
    self.update_state(state='PROCESSING', meta={'progress': 0})
    
    system = VideoSemanticSystem(config)
    # 分阶段更新进度
    system.perception.on_progress = lambda p: self.update_state(
        state='PROCESSING', meta={'progress': p}
    )
    system.build_index()
    
    return {'status': 'completed', 'track_count': len(system.track_records)}
```

**任务状态查询**:
```python
# routes/index.py
@router.get("/videos/{video_id}/status")
async def get_index_status(video_id: str):
    task = AsyncResult(video_id)
    return {
        "status": task.state,
        "progress": task.info.get('progress', 0) if task.info else 0
    }
```

### 1.3 数据库升级

**当前问题**:
```python
# 元数据存在 JSON 文件
db_path = self.config.output_dir / "semantic_database.json"
```

**方案**: PostgreSQL + SQLAlchemy

```python
# models/database.py
from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(String, primary_key=True)
    path = Column(String, nullable=False)
    fps = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    total_frames = Column(Integer)
    status = Column(String, default="pending")  # pending/processing/completed/failed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    tracks = relationship("Track", back_populates="video")

class Track(Base):
    __tablename__ = "tracks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.id"))
    track_id = Column(Integer)  # 原始 track_id（视频内唯一）
    start_s = Column(Float)
    end_s = Column(Float)
    duration_s = Column(Float)
    avg_speed_px_s = Column(Float)
    max_speed_px_s = Column(Float)
    centroids = Column(JSON)  # List[(x, y)]
    displacement_vec = Column(JSON)  # (vx, vy)
    crops_paths = Column(JSON)  # List[str]
    
    video = relationship("Video", back_populates="tracks")
```

**数据库迁移**: Alembic
```bash
alembic init migrations
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

### 1.4 基础监控

**结构化日志**:
```python
# utils/logging.py
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# 使用
logger.info("search_completed", 
    video_id=video_id, 
    question=question,
    result_count=len(results),
    latency_ms=elapsed
)
```

**健康检查端点**:
```python
@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    checks = {
        "database": await check_db(db),
        "redis": await check_redis(),
        "vlm_model": check_vlm_loaded(),
        "disk_space": check_disk_space()
    }
    status = "healthy" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks}
```

### 1.5 Phase 1 交付物

- [ ] FastAPI 服务骨架
- [ ] Celery 异步任务
- [ ] PostgreSQL 数据模型
- [ ] 结构化日志
- [ ] 健康检查接口
- [ ] Docker Compose 本地开发环境
- [ ] 基础 API 文档（Swagger）

---

## Phase 2: 生产可用

> **目标**: 性能优化 + 稳定部署  
> **预计周期**: 3-4 周

### 2.1 向量数据库集成

**当前问题**:
```python
# 向量存在 numpy 文件，每次检索都要全量加载
np.save(cache_path, emb)
```

**方案**: Milvus / Qdrant（推荐 Qdrant，轻量易部署）

```python
# storage/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "track_embeddings"
    
    def create_collection(self, dim: int = 768):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    
    def upsert_track(self, video_id: str, track_id: int, embedding: List[float]):
        point_id = f"{video_id}_{track_id}"
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=hash(point_id) % (2**63),
                vector=embedding,
                payload={"video_id": video_id, "track_id": track_id}
            )]
        )
    
    def search(self, query_vector: List[float], video_id: str, top_k: int = 50):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter={"must": [{"key": "video_id", "match": {"value": video_id}}]},
            limit=top_k
        )
```

**RecallEngine 改造**:
```python
class RecallEngine:
    def __init__(self, config, vector_store: VectorStore, siglip_client):
        self.vector_store = vector_store
        self.siglip = siglip_client
    
    def visual_filter(self, video_id: str, description: str, visual_tags: List[str], top_k: int):
        query_vec = self._encode_query(description, visual_tags)
        # 直接从向量库检索，无需全量加载
        results = self.vector_store.search(query_vec, video_id, top_k)
        return [r.payload["track_id"] for r in results]
```

### 2.2 推理性能优化

#### 2.2.1 VLM 批推理

**当前问题**:
```python
# 串行处理每条轨迹
for idx, package in enumerate(candidates):
    answer = self._query_package(package, question, plan)  # 100条 = 100次调用
```

**优化方案**: 批量构造 prompt，一次推理

```python
class Qwen3VL4BHFClient:
    def answer_batch(self, question: str, candidates: List[EvidencePackage], batch_size: int = 4):
        """批量推理，减少模型调用次数"""
        results = []
        for batch in chunked(candidates, batch_size):
            # 构造多轨迹的联合 prompt
            batch_prompt = self._build_batch_prompt(question, batch)
            batch_results = self._inference(batch_prompt)
            results.extend(self._parse_batch_results(batch_results, batch))
        return results
```

#### 2.2.2 模型量化部署

**当前问题**: HF transformers 加载原始权重，内存占用大

**方案**: 使用 GGUF 量化 + llama-cpp-python

```python
# config.py 已预留接口
vlm_backend: str = "llama_cpp"  # 切换到量化后端
vlm_gguf_path: Path = Path("models/qwen3-vl-4b-q4_k_m.gguf")
```

```python
# vlm_client_gguf.py
from llama_cpp import Llama

class Qwen3VLGGUFClient:
    def __init__(self, config: SystemConfig):
        self.llm = Llama(
            model_path=str(config.vlm_gguf_path),
            n_ctx=config.vlm_context_size,
            n_gpu_layers=config.vlm_gpu_layers,
            n_threads=config.vlm_cpu_threads or 4
        )
```

#### 2.2.3 SigLIP 批处理

```python
class SiglipClient:
    @torch.no_grad()
    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码图像"""
        all_embeddings = []
        for batch_paths in chunked(image_paths, batch_size):
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            emb = self.model.get_image_features(**inputs)
            emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)
```

### 2.3 对象存储

**当前问题**: 裁剪图存本地文件系统，不可扩展

**方案**: MinIO（S3 兼容）

```python
# storage/object_store.py
from minio import Minio

class ObjectStore:
    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        self.client = Minio(endpoint, access_key, secret_key, secure=False)
        self.bucket = "edge-detective-crops"
    
    def upload_crop(self, video_id: str, track_id: int, frame_idx: int, image_bytes: bytes) -> str:
        object_name = f"{video_id}/track_{track_id}/frame_{frame_idx:05d}.jpg"
        self.client.put_object(
            self.bucket, object_name, 
            io.BytesIO(image_bytes), len(image_bytes),
            content_type="image/jpeg"
        )
        return f"s3://{self.bucket}/{object_name}"
    
    def get_presigned_url(self, object_name: str, expires: int = 3600) -> str:
        return self.client.presigned_get_object(self.bucket, object_name, expires=timedelta(seconds=expires))
```

### 2.4 容器化部署

**Dockerfile**:
```dockerfile
# Dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/
COPY models/ ./models/

# 预下载模型权重（可选，或用 volume 挂载）
# RUN python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')"

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/edge_detective
      - REDIS_URL=redis://redis:6379/0
      - MINIO_ENDPOINT=minio:9000
      - QDRANT_HOST=qdrant
    depends_on:
      - postgres
      - redis
      - minio
      - qdrant
    volumes:
      - ./models:/app/models  # 模型权重挂载
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  celery_worker:
    build: .
    command: celery -A tasks.indexing worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/edge_detective
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: edge_detective
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  redis_data:
  minio_data:
  qdrant_data:
```

### 2.5 Phase 2 交付物

- [ ] Qdrant 向量库集成
- [ ] VLM 批推理实现
- [ ] GGUF 量化模型支持
- [ ] MinIO 对象存储
- [ ] Dockerfile + docker-compose
- [ ] 性能基准测试报告
- [ ] 部署文档

---

## Phase 3: 规模化

> **目标**: 支持实时流、多摄像头、高并发  
> **预计周期**: 4-6 周

### 3.1 实时流处理

**当前问题**:
```python
# 只能处理完整视频文件
cap = cv2.VideoCapture(str(self.config.video_path))
```

**方案**: 支持 RTSP/RTMP 流 + 滑动窗口追踪

```python
# streaming/stream_processor.py
import cv2
from collections import deque

class StreamProcessor:
    """实时流处理器"""
    
    def __init__(self, stream_url: str, config: SystemConfig):
        self.cap = cv2.VideoCapture(stream_url)
        self.config = config
        self.frame_buffer = deque(maxlen=300)  # 10秒 @ 30fps
        self.active_tracks: Dict[int, TrackRecord] = {}
        self.tracker = create_tracker(config.tracker_type)
        self.yolo = YOLO(config.yolo_model)
        
    def run(self, on_track_complete: Callable[[TrackRecord], None]):
        """主循环：持续处理流"""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_idx += 1
            self.frame_buffer.append((frame_idx, frame))
            
            # 检测 + 追踪
            detections = self._detect(frame)
            tracks = self.tracker.update(detections, frame)
            
            # 更新活跃轨迹
            active_ids = set()
            for track in tracks:
                tid = int(track[4])
                active_ids.add(tid)
                self._update_track(tid, frame_idx, track, frame)
            
            # 检查已结束的轨迹（消失超过阈值帧数）
            for tid in list(self.active_tracks.keys()):
                if tid not in active_ids:
                    record = self.active_tracks[tid]
                    if frame_idx - record.frames[-1] > 30:  # 消失超过1秒
                        on_track_complete(record)
                        del self.active_tracks[tid]
```

**实时索引更新**:
```python
class RealTimeIndexer:
    """实时索引更新器"""
    
    def __init__(self, vector_store: VectorStore, siglip: SiglipClient):
        self.vector_store = vector_store
        self.siglip = siglip
    
    def on_track_complete(self, record: TrackRecord, video_id: str):
        """轨迹结束时触发索引"""
        # 1. 计算特征
        features = self._compute_features(record)
        
        # 2. 生成 embedding
        crops = self._sample_crops(record)
        embeddings = self.siglip.encode_images(crops)
        avg_embedding = embeddings.mean(axis=0)
        
        # 3. 写入向量库
        self.vector_store.upsert_track(video_id, record.track_id, avg_embedding.tolist())
        
        # 4. 写入元数据库
        self._save_to_db(video_id, record, features)
```

### 3.2 多摄像头管理

```python
# cameras/manager.py
class CameraManager:
    """多摄像头管理器"""
    
    def __init__(self):
        self.cameras: Dict[str, StreamProcessor] = {}
        self.executor = ThreadPoolExecutor(max_workers=8)
    
    def add_camera(self, camera_id: str, stream_url: str, config: SystemConfig):
        processor = StreamProcessor(stream_url, config)
        self.cameras[camera_id] = processor
        # 启动独立线程处理
        self.executor.submit(self._run_camera, camera_id, processor)
    
    def remove_camera(self, camera_id: str):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
    
    def _run_camera(self, camera_id: str, processor: StreamProcessor):
        indexer = RealTimeIndexer(self.vector_store, self.siglip)
        processor.run(
            on_track_complete=lambda r: indexer.on_track_complete(r, camera_id)
        )
```

**跨摄像头检索**:
```python
# routes/search.py
@router.post("/search/cross-camera")
async def search_cross_camera(request: CrossCameraSearchRequest):
    """跨多个摄像头检索"""
    results = []
    for camera_id in request.camera_ids:
        camera_results = await search_single_camera(camera_id, request.question, request.top_k)
        results.extend(camera_results)
    
    # 按分数排序，去重
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:request.top_k]
```

### 3.3 Re-ID（跨镜追踪）

**问题**: 同一个人在不同摄像头出现，如何关联？

**方案**: 基于外观特征的 Re-ID

```python
# reid/matcher.py
class ReIDMatcher:
    """跨镜 Re-ID 匹配器"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        # 使用专门的 Re-ID 模型（如 OSNet、BoT）
        self.reid_model = self._load_reid_model()
    
    def extract_feature(self, crops: List[Image.Image]) -> np.ndarray:
        """提取 Re-ID 特征"""
        features = []
        for crop in crops:
            feat = self.reid_model(crop)
            features.append(feat)
        return np.mean(features, axis=0)
    
    def match_across_cameras(self, 
                             query_track: EvidencePackage,
                             candidate_tracks: List[EvidencePackage]) -> List[Tuple[int, float]]:
        """跨摄像头匹配"""
        query_feat = self.extract_feature(query_track.crops)
        matches = []
        for candidate in candidate_tracks:
            cand_feat = self.extract_feature(candidate.crops)
            similarity = np.dot(query_feat, cand_feat)
            if similarity > self.threshold:
                matches.append((candidate.track_id, similarity))
        return sorted(matches, key=lambda x: x[1], reverse=True)
```

### 3.4 Kubernetes 部署

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-detective-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: edge-detective-api
  template:
    metadata:
      labels:
        app: edge-detective-api
    spec:
      containers:
      - name: api
        image: edge-detective:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: edge-detective-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: edge-detective-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: edge-detective-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

### 3.5 事件触发与告警

```python
# events/detector.py
class EventTrigger:
    """实时事件触发器"""
    
    def __init__(self, rules: List[EventRule], webhook_url: str):
        self.rules = rules
        self.webhook_url = webhook_url
    
    async def check_track(self, track: EvidencePackage, camera_id: str):
        """检查轨迹是否触发告警规则"""
        for rule in self.rules:
            if rule.matches(track):
                await self._send_alert(rule, track, camera_id)
    
    async def _send_alert(self, rule: EventRule, track: EvidencePackage, camera_id: str):
        alert = {
            "rule_name": rule.name,
            "camera_id": camera_id,
            "track_id": track.track_id,
            "timestamp": datetime.utcnow().isoformat(),
            "description": rule.description,
            "thumbnail_url": self._get_thumbnail_url(track)
        }
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=alert)

# 示例规则：门口徘徊超过30秒
class LoiteringRule(EventRule):
    name = "loitering_alert"
    description = "检测到门口徘徊行为"
    
    def matches(self, track: EvidencePackage) -> bool:
        if not track.features:
            return False
        # 检查是否在门口 ROI 停留超过30秒
        roi_dwell = self._compute_roi_dwell(track, "entrance")
        return roi_dwell > 30.0
```

### 3.6 Phase 3 交付物

- [ ] RTSP/RTMP 流处理器
- [ ] 实时索引更新
- [ ] 多摄像头管理 API
- [ ] Re-ID 跨镜追踪
- [ ] Kubernetes 部署配置
- [ ] HPA 自动扩缩容
- [ ] 事件告警系统
- [ ] 运维 Runbook

---

## 附录：技术选型对比

### 向量数据库

| 方案 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **Qdrant** | 轻量、易部署、API 友好 | 社区较小 | MVP / 中小规模 |
| Milvus | 功能全、性能强 | 运维复杂 | 大规模生产 |
| Pinecone | 全托管、免运维 | 成本高、数据出境 | 快速上线 |
| Faiss | 纯库、性能最优 | 无持久化、无分布式 | 嵌入式场景 |

### 模型推理

| 方案 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **llama-cpp** | 低内存、CPU/GPU 灵活 | 需自行量化 | 边缘设备 |
| vLLM | 高吞吐、批推理 | 仅支持部分模型 | 云端高并发 |
| TensorRT | 极致性能 | 转换复杂、仅 NVIDIA | 固定模型生产 |
| ONNX Runtime | 跨平台 | 转换可能损失精度 | 多平台部署 |

---

## 里程碑检查点

| 阶段 | 关键指标 | 验收标准 |
|------|----------|----------|
| Phase 1 | API 可用 | 能通过 HTTP 完成索引+检索 |
| Phase 2 | 性能达标 | 检索 P99 < 3s，索引速度 > 2x 实时 |
| Phase 3 | 规模化 | 支持 10+ 摄像头并发，实时延迟 < 5s |
