# 🚀 Phase 3: 规模化 开发计划

> **目标**: 支持实时流、多摄像头、高并发、K8s 部署  
> **周期**: 4-6 周  
> **前置条件**: Phase 2 完成

---

## 目录

1. [Week 8-9: 实时流处理](#week-8-9-实时流处理)
2. [Week 10-11: Re-ID + 事件告警](#week-10-11-re-id--事件告警)
3. [Week 12-13: Kubernetes 部署](#week-12-13-kubernetes-部署)
4. [最终目录结构](#最终目录结构)
5. [完整删除清单](#完整删除清单)
6. [验收标准](#验收标准)

---

## Week 8-9: 实时流处理

### 实时流处理器

**`src/adapters/streaming/rtsp_processor.py`**
```python
"""RTSP 流处理器"""
from __future__ import annotations

import cv2
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Callable, Optional
from threading import Thread, Event

from ultralytics import YOLO
from boxmot import create_tracker

from core.perception import TrackRecord


@dataclass
class StreamConfig:
    """流配置"""
    url: str
    camera_id: str
    buffer_size: int = 300  # 10s @ 30fps
    detection_interval: int = 1  # 每帧检测


class RtspStreamProcessor:
    """
    RTSP 实时流处理器
    
    功能：
    - 实时读取 RTSP/RTMP 流
    - 持续检测和追踪
    - 轨迹完成时回调
    """
    
    def __init__(
        self,
        config: StreamConfig,
        yolo_model: str = "yolo11n.pt",
        tracker_type: str = "bytetrack",
    ):
        self.config = config
        self.cap = cv2.VideoCapture(config.url)
        self.yolo = YOLO(yolo_model)
        self.tracker = create_tracker(
            tracker_type=tracker_type,
            tracker_config=None,
            reid_weights=None,
            device="cuda",
            half=False,
            per_class=True,
        )
        
        self.frame_buffer = deque(maxlen=config.buffer_size)
        self.active_tracks: Dict[int, TrackRecord] = {}
        self.frame_idx = 0
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
    
    def start(self, on_track_complete: Callable[[TrackRecord], None]):
        """启动流处理"""
        self._stop_event.clear()
        self._thread = Thread(
            target=self._run_loop,
            args=(on_track_complete,),
            daemon=True
        )
        self._thread.start()
    
    def stop(self):
        """停止流处理"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.cap.release()
    
    def _run_loop(self, on_track_complete: Callable):
        """主循环"""
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            self.frame_idx += 1
            self.frame_buffer.append((self.frame_idx, frame))
            
            # 检测
            if self.frame_idx % self.config.detection_interval == 0:
                self._process_frame(frame, on_track_complete)
    
    def _process_frame(self, frame, on_track_complete: Callable):
        """处理单帧"""
        results = self.yolo.predict(
            source=frame,
            device="cuda",
            conf=0.5,
            verbose=False,
            classes=[0],
        )[0]
        
        detections = []
        if results.boxes is not None:
            for i in range(len(results.boxes)):
                x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy()
                conf = float(results.boxes.conf[i])
                detections.append([x1, y1, x2, y2, conf, 0])
        
        if not detections:
            return
        
        import numpy as np
        tracks = self.tracker.update(np.array(detections), frame)
        
        active_ids = set()
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            tid = int(track[4])
            active_ids.add(tid)
            
            if tid not in self.active_tracks:
                self.active_tracks[tid] = TrackRecord(
                    track_id=tid, frames=[], bboxes=[], crops=[]
                )
            
            record = self.active_tracks[tid]
            record.frames.append(self.frame_idx)
            record.bboxes.append((x1, y1, x2, y2))
        
        # 检查已结束的轨迹
        for tid in list(self.active_tracks.keys()):
            if tid not in active_ids:
                record = self.active_tracks[tid]
                if self.frame_idx - record.frames[-1] > 30:  # 1秒未出现
                    on_track_complete(record)
                    del self.active_tracks[tid]
```

### 多摄像头管理

**`src/adapters/streaming/camera_manager.py`**
```python
"""多摄像头管理器"""
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from adapters.streaming.rtsp_processor import RtspStreamProcessor, StreamConfig
from core.perception import TrackRecord


class CameraManager:
    """
    多摄像头管理器
    
    功能：
    - 管理多个摄像头流
    - 动态添加/移除摄像头
    - 统一的轨迹回调
    """
    
    def __init__(self, max_cameras: int = 16):
        self.processors: Dict[str, RtspStreamProcessor] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_cameras)
    
    def add_camera(
        self,
        camera_id: str,
        stream_url: str,
        on_track_complete=None
    ):
        """添加摄像头"""
        if camera_id in self.processors:
            raise ValueError(f"Camera {camera_id} already exists")
        
        config = StreamConfig(url=stream_url, camera_id=camera_id)
        processor = RtspStreamProcessor(config)
        self.processors[camera_id] = processor
        
        if on_track_complete:
            processor.start(on_track_complete)
    
    def remove_camera(self, camera_id: str):
        """移除摄像头"""
        if camera_id in self.processors:
            self.processors[camera_id].stop()
            del self.processors[camera_id]
    
    def list_cameras(self) -> list:
        """列出所有摄像头"""
        return list(self.processors.keys())
    
    def stop_all(self):
        """停止所有摄像头"""
        for proc in self.processors.values():
            proc.stop()
        self.processors.clear()
        self.executor.shutdown(wait=True)
```

### 实时索引更新

**`src/application/use_cases/realtime_index.py`**
```python
"""实时索引用例"""
from core.perception import TrackRecord
from adapters.vector.qdrant_adapter import QdrantAdapter


class RealtimeIndexer:
    """实时索引器"""
    
    def __init__(
        self,
        vector_store: QdrantAdapter,
        siglip_client,
    ):
        self.vector_store = vector_store
        self.siglip = siglip_client
    
    async def on_track_complete(
        self,
        camera_id: str,
        record: TrackRecord
    ):
        """轨迹完成时触发索引"""
        # 1. 计算特征
        # 2. 生成 embedding
        # 3. 写入向量库
        # 4. 写入元数据库
        pass
```

---

## Week 10-11: Re-ID + 事件告警

### Re-ID 跨镜追踪

**`src/adapters/reid/osnet_adapter.py`**
```python
"""OSNet Re-ID 适配器"""
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple


class OSNetReIDAdapter:
    """
    OSNet Re-ID 模型适配器
    
    用于跨摄像头人员重识别
    """
    
    def __init__(self, model_name: str = "osnet_x1_0"):
        import torchreid
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def extract_feature(self, images: List[Image.Image]) -> np.ndarray:
        """提取 Re-ID 特征"""
        # 预处理
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        tensors = [transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
        
        return features.cpu().numpy().mean(axis=0)
    
    def match(
        self,
        query_feature: np.ndarray,
        gallery_features: List[np.ndarray],
        threshold: float = 0.7
    ) -> List[Tuple[int, float]]:
        """匹配"""
        matches = []
        for i, feat in enumerate(gallery_features):
            similarity = np.dot(query_feature, feat) / (
                np.linalg.norm(query_feature) * np.linalg.norm(feat)
            )
            if similarity > threshold:
                matches.append((i, float(similarity)))
        return sorted(matches, key=lambda x: x[1], reverse=True)
```

### 事件告警系统

**`src/adapters/events/alert_system.py`**
```python
"""事件告警系统"""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import aiohttp


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    description: str
    condition: str  # 规则表达式
    severity: str  # info, warning, critical
    webhook_url: Optional[str] = None


@dataclass
class Alert:
    """告警"""
    rule_name: str
    camera_id: str
    track_id: int
    timestamp: datetime
    description: str
    thumbnail_url: Optional[str] = None


class AlertSystem:
    """
    事件告警系统
    
    功能：
    - 规则管理
    - 实时检测
    - Webhook 通知
    """
    
    def __init__(self, rules: List[AlertRule]):
        self.rules = {r.name: r for r in rules}
    
    async def check(self, camera_id: str, track, features) -> List[Alert]:
        """检查是否触发告警"""
        alerts = []
        for rule in self.rules.values():
            if self._evaluate_rule(rule, track, features):
                alert = Alert(
                    rule_name=rule.name,
                    camera_id=camera_id,
                    track_id=track.track_id,
                    timestamp=datetime.utcnow(),
                    description=rule.description,
                )
                alerts.append(alert)
                
                if rule.webhook_url:
                    await self._send_webhook(rule.webhook_url, alert)
        
        return alerts
    
    def _evaluate_rule(self, rule: AlertRule, track, features) -> bool:
        """评估规则（简化示例）"""
        # TODO: 实现规则引擎
        return False
    
    async def _send_webhook(self, url: str, alert: Alert):
        """发送 Webhook"""
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={
                "rule": alert.rule_name,
                "camera": alert.camera_id,
                "track_id": alert.track_id,
                "timestamp": alert.timestamp.isoformat(),
                "description": alert.description,
            })
```

---

## Week 12-13: Kubernetes 部署

### K8s 配置文件

**`deploy/k8s/api-deployment.yaml`**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-detective-api
  labels:
    app: edge-detective
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: edge-detective
      component: api
  template:
    metadata:
      labels:
        app: edge-detective
        component: api
    spec:
      containers:
      - name: api
        image: edge-detective:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: edge-detective-secrets
              key: database-url
        - name: VLLM_ENDPOINT
          value: "http://vllm-service:8000/v1"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: edge-detective-api
spec:
  selector:
    app: edge-detective
    component: api
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

**`deploy/k8s/vllm-deployment.yaml`**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  labels:
    app: edge-detective
    component: vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: edge-detective
      component: vllm
  template:
    metadata:
      labels:
        app: edge-detective
        component: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "Qwen/Qwen3-VL-4B-Instruct"
        - "--trust-remote-code"
        - "--host"
        - "0.0.0.0"
        - "--port"
        - "8000"
        - "--gpu-memory-utilization"
        - "0.90"
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "16Gi"
            nvidia.com/gpu: 1
          limits:
            memory: "24Gi"
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: edge-detective
    component: vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

**`deploy/k8s/hpa.yaml`**
```yaml
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
```

---

## 最终目录结构

> 完整重构后的目录结构，与 `final_upgrade_blueprint.md` 一致

```
src/
├── api/                          # 网关层
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── index.py
│   │   ├── search.py
│   │   └── tracks.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   └── responses.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── rate_limit.py
│   │   └── tracing.py
│   └── dependencies.py
│
├── application/                  # 应用层
│   ├── __init__.py
│   ├── use_cases/
│   │   ├── __init__.py
│   │   ├── index_video.py
│   │   ├── search_tracks.py
│   │   ├── generate_report.py
│   │   └── realtime_index.py
│   └── dto/
│       ├── __init__.py
│       ├── index_dto.py
│       └── search_dto.py
│
├── domain/                       # 领域层
│   ├── __init__.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── video.py
│   │   ├── track.py
│   │   └── evidence.py
│   ├── value_objects/
│   │   ├── __init__.py
│   │   ├── bounding_box.py
│   │   ├── trajectory.py
│   │   └── verification_result.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py
│   │   ├── motion_analyzer.py
│   │   └── hard_rule_engine.py
│   └── events/
│       ├── __init__.py
│       └── domain_events.py
│
├── ports/                        # 端口层
│   ├── __init__.py
│   ├── inference_port.py
│   ├── storage_port.py
│   ├── vector_store_port.py
│   └── message_queue_port.py
│
├── adapters/                     # 适配层
│   ├── __init__.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── vllm_adapter.py
│   │   ├── llamacpp_adapter.py
│   │   ├── model_registry.py
│   │   └── response_parser.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── postgres_repo.py
│   │   ├── minio_adapter.py
│   │   └── memory_repo.py
│   ├── vector/
│   │   ├── __init__.py
│   │   └── qdrant_adapter.py
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_adapter.py
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── rtsp_processor.py
│   │   └── camera_manager.py
│   ├── reid/
│   │   ├── __init__.py
│   │   └── osnet_adapter.py
│   └── events/
│       ├── __init__.py
│       └── alert_system.py
│
├── infrastructure/               # 基础设施
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── app_config.py
│   │   └── infra_config.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── migrations/
│   └── logging/
│       ├── __init__.py
│       └── structured_logger.py
│
├── tasks/                        # 异步任务
│   ├── __init__.py
│   ├── celery_app.py
│   └── indexing.py
│
└── tests/                        # 测试
    ├── __init__.py
    ├── conftest.py
    ├── unit/
    ├── integration/
    └── e2e/
```

---

## 完整删除清单

> Phase 3 完成后，删除所有旧代码

### 🔴 必须删除

| 文件/目录 | 理由 | 替代 |
|-----------|------|------|
| `src/pipeline/vlm_client_hf.py` | 直接加载模型，不符合分层架构 | `adapters/inference/vllm_adapter.py` |
| `src/pipeline/vlm_client_vllm.py` (如果有旧版) | 迁移到 adapters | `adapters/inference/vllm_adapter.py` |
| `src/pipeline/recall.py` | 迁移到应用层 | `application/use_cases/search_tracks.py` |
| `src/pipeline/router.py` | 迁移到应用层 | `application/use_cases/` |
| `src/pipeline/router_llm.py` | 同上 | `application/use_cases/` |
| `src/core/config.py` | 迁移到 infrastructure | `infrastructure/config/` |
| `src/core/perception.py` | 迁移到 adapters | `adapters/detection/yolo_adapter.py` |
| `src/core/features.py` | 迁移到 domain | `domain/services/feature_extractor.py` |
| `src/core/evidence.py` | 迁移到 domain | `domain/entities/evidence.py` |
| `src/core/hard_rules.py` | 迁移到 domain | `domain/services/hard_rule_engine.py` |
| `src/core/vlm_types.py` | 迁移到 domain | `domain/value_objects/` |

### 🟡 可选删除

| 文件/目录 | 理由 | 建议 |
|-----------|------|------|
| `src/pipeline/video_semantic_search.py` | 旧入口 | 保留作为参考，后删除 |
| `src/examples/` | 旧示例代码 | 更新或删除 |

### 最终删除命令

```bash
# Phase 3 完成后执行
rm -rf src/pipeline/
rm -rf src/core/
rm -rf src/examples/

# 确保新目录已创建
ls -la src/api/
ls -la src/application/
ls -la src/domain/
ls -la src/ports/
ls -la src/adapters/
ls -la src/infrastructure/
ls -la src/tasks/
```

---

## 验收标准

### 功能验收

- [ ] RTSP 实时流处理正常
- [ ] 多摄像头管理 API 可用
- [ ] Re-ID 跨镜匹配可用
- [ ] 事件告警触发正常
- [ ] Webhook 通知可达

### 性能验收

- [ ] 实时延迟 < 5s
- [ ] 支持 10+ 摄像头并发
- [ ] API P99 延迟 < 500ms
- [ ] 系统 CPU < 80%

### 部署验收

- [ ] K8s 部署成功
- [ ] HPA 自动扩缩容正常
- [ ] 健康检查通过
- [ ] 监控指标可采集
- [ ] 日志可检索

---

## 项目完成标志

当以下条件全部满足时，项目重构完成：

1. ✅ 目录结构与 `final_upgrade_blueprint.md` 一致
2. ✅ 所有旧代码已删除
3. ✅ 所有测试通过
4. ✅ 文档更新完成
5. ✅ K8s 部署成功
6. ✅ 监控告警正常

---

**恭喜！Edge-Detective 工业化升级完成！** 🎉

