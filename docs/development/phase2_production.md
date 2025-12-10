# 🏭 Phase 2: 生产可用 开发计划

> **目标**: 性能优化 + 稳定部署 + 量化模型支持  
> **周期**: 3-4 周  
> **前置条件**: Phase 1 完成

---

## 目录

1. [Week 4: 向量库 + 对象存储](#week-4-向量库--对象存储)
2. [Week 5: 量化模型 + ModelRegistry](#week-5-量化模型--modelregistry)
3. [Week 6: 前端骨架](#week-6-前端骨架)
4. [Week 7: 完整测试 + 性能优化](#week-7-完整测试--性能优化)
5. [待删除文件清单](#待删除文件清单)
6. [验收标准](#验收标准)

---

## Week 4: 向量库 + 对象存储

### Day 1-2: Qdrant 向量库集成

#### 需要创建的文件

```bash
src/ports/vector_store_port.py
src/adapters/vector/__init__.py
src/adapters/vector/qdrant_adapter.py
```

#### 代码实现

**`src/ports/vector_store_port.py`**
```python
"""向量存储端口"""
from typing import Protocol, List


class VectorStorePort(Protocol):
    """向量存储抽象接口"""
    
    async def upsert(
        self,
        video_id: str,
        track_id: int,
        embedding: List[float],
        metadata: dict
    ) -> None:
        """插入/更新向量"""
        ...
    
    async def search(
        self,
        video_id: str,
        query_vector: List[float],
        top_k: int = 50
    ) -> List[dict]:
        """向量检索"""
        ...
    
    async def delete_video(self, video_id: str) -> None:
        """删除视频的所有向量"""
        ...
```

**`src/adapters/vector/qdrant_adapter.py`**
```python
"""Qdrant 向量库适配器"""
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)

from ports.vector_store_port import VectorStorePort


class QdrantAdapter:
    """Qdrant 向量存储实现"""
    
    COLLECTION_PREFIX = "track_embeddings"
    VECTOR_DIM = 768  # SigLIP
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
    
    def _collection_name(self, video_id: str) -> str:
        return f"{self.COLLECTION_PREFIX}_{video_id}"
    
    async def ensure_collection(self, video_id: str):
        name = self._collection_name(video_id)
        collections = self.client.get_collections().collections
        if name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.VECTOR_DIM,
                    distance=Distance.COSINE
                )
            )
    
    async def upsert(
        self,
        video_id: str,
        track_id: int,
        embedding: List[float],
        metadata: dict
    ):
        await self.ensure_collection(video_id)
        self.client.upsert(
            collection_name=self._collection_name(video_id),
            points=[PointStruct(
                id=track_id,
                vector=embedding,
                payload={"video_id": video_id, "track_id": track_id, **metadata}
            )]
        )
    
    async def search(
        self,
        video_id: str,
        query_vector: List[float],
        top_k: int = 50
    ) -> List[dict]:
        results = self.client.search(
            collection_name=self._collection_name(video_id),
            query_vector=query_vector,
            limit=top_k,
        )
        return [{"track_id": h.id, "score": h.score, **h.payload} for h in results]
    
    async def delete_video(self, video_id: str):
        name = self._collection_name(video_id)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
```

### Day 3-4: MinIO 对象存储

**`src/ports/storage_port.py`**
```python
"""对象存储端口"""
from typing import Protocol, BinaryIO


class ObjectStoragePort(Protocol):
    """对象存储抽象接口"""
    
    async def upload(
        self,
        bucket: str,
        key: str,
        data: BinaryIO,
        content_type: str = "application/octet-stream"
    ) -> str:
        """上传文件，返回 URL"""
        ...
    
    async def download(self, bucket: str, key: str) -> bytes:
        """下载文件"""
        ...
    
    async def get_presigned_url(
        self,
        bucket: str,
        key: str,
        expires: int = 3600
    ) -> str:
        """获取预签名 URL"""
        ...
    
    async def delete(self, bucket: str, key: str) -> None:
        """删除文件"""
        ...
```

**`src/adapters/storage/minio_adapter.py`**
```python
"""MinIO 对象存储适配器"""
import io
from datetime import timedelta
from typing import BinaryIO

from minio import Minio

from ports.storage_port import ObjectStoragePort


class MinioAdapter:
    """MinIO 对象存储实现"""
    
    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
    
    async def ensure_bucket(self, bucket: str):
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
    
    async def upload(
        self,
        bucket: str,
        key: str,
        data: BinaryIO,
        content_type: str = "application/octet-stream"
    ) -> str:
        await self.ensure_bucket(bucket)
        data.seek(0, 2)
        size = data.tell()
        data.seek(0)
        self.client.put_object(bucket, key, data, size, content_type)
        return f"s3://{bucket}/{key}"
    
    async def download(self, bucket: str, key: str) -> bytes:
        response = self.client.get_object(bucket, key)
        return response.read()
    
    async def get_presigned_url(
        self,
        bucket: str,
        key: str,
        expires: int = 3600
    ) -> str:
        return self.client.presigned_get_object(
            bucket, key, expires=timedelta(seconds=expires)
        )
    
    async def delete(self, bucket: str, key: str):
        self.client.remove_object(bucket, key)
```

### Day 5: Docker Compose 整合

**`docker-compose.yml`**
```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  qdrant_data:
  minio_data:
  redis_data:
```

---

## Week 5: 量化模型 + ModelRegistry

### Day 1-2: llama.cpp 适配器

**`src/adapters/inference/llamacpp_adapter.py`**
```python
"""llama.cpp 量化模型适配器"""
from __future__ import annotations

import base64
import asyncio
from typing import List, Optional
from dataclasses import dataclass

from ports.inference_port import InferencePort
from domain.value_objects.verification_result import VerificationResult, VlmResponseParser
from core.evidence import EvidencePackage


@dataclass
class LlamaCppConfig:
    """llama.cpp 配置"""
    model_path: str
    clip_model_path: str = ""
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_threads: int = 4


class LlamaCppAdapter:
    """
    llama.cpp 量化模型适配器
    
    用于 GGUF 格式的量化模型，支持：
    - CPU/GPU 混合推理
    - 极低显存占用
    - 边缘设备部署
    """
    
    def __init__(self, config: LlamaCppConfig):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        
        self.config = config
        self._parser = VlmResponseParser()
        
        # 初始化视觉处理器
        chat_handler = None
        if config.clip_model_path:
            chat_handler = Llava15ChatHandler(
                clip_model_path=config.clip_model_path
            )
        
        self.llm = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.n_threads,
            chat_handler=chat_handler,
            verbose=False,
        )
    
    async def verify_track(
        self,
        package: EvidencePackage,
        question: str,
        plan_context: Optional[str] = None,
    ) -> VerificationResult:
        """验证单个轨迹"""
        messages = self._build_messages(package, question, plan_context)
        
        # llama.cpp 是同步的，用 run_in_executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.1,
            )
        )
        
        raw_text = response["choices"][0]["message"]["content"]
        return self._parser.parse(raw_text)
    
    async def verify_batch(
        self,
        packages: List[EvidencePackage],
        question: str,
        plan_context: Optional[str] = None,
        concurrency: int = 1,  # llama.cpp 不支持真正并发
    ) -> List[VerificationResult]:
        """批量验证（串行）"""
        results = []
        for pkg in packages:
            result = await self.verify_track(pkg, question, plan_context)
            results.append(result)
        return results
    
    def _build_messages(self, package, question, plan_context):
        crop_paths = package.crops[:3]  # 量化模型用更少图片
        
        image_contents = []
        for path in crop_paths:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
        
        return [
            {"role": "system", "content": "You are a video analysis assistant."},
            {
                "role": "user",
                "content": [
                    *image_contents,
                    {"type": "text", "text": f"Query: {question}\n\nDoes this person match? Answer with MATCH: yes or MATCH: no"}
                ]
            }
        ]
```

### Day 3-4: ModelRegistry 实现

**`src/adapters/inference/model_registry.py`**
```python
"""模型注册中心"""
from enum import Enum
from typing import Dict, List, Optional

from ports.inference_port import InferencePort


class InferencePriority(Enum):
    """推理优先级"""
    HIGH_ACCURACY = "high_accuracy"
    LOW_LATENCY = "low_latency"
    COST_SAVING = "cost_saving"


class ModelRegistry:
    """
    模型注册中心
    
    职责：
    1. 管理多个推理适配器
    2. 根据策略路由请求
    3. 支持运行时切换
    4. 实现 A/B 测试
    """
    
    def __init__(self):
        self._adapters: Dict[str, InferencePort] = {}
        self._priority_map: Dict[InferencePriority, str] = {}
        self._default: Optional[str] = None
    
    def register(
        self,
        name: str,
        adapter: InferencePort,
        priorities: Optional[List[InferencePriority]] = None,
        is_default: bool = False
    ):
        """注册适配器"""
        self._adapters[name] = adapter
        
        if priorities:
            for priority in priorities:
                self._priority_map[priority] = name
        
        if is_default or self._default is None:
            self._default = name
    
    def get_adapter(
        self,
        priority: Optional[InferencePriority] = None
    ) -> InferencePort:
        """根据优先级获取适配器"""
        if priority and priority in self._priority_map:
            name = self._priority_map[priority]
            return self._adapters[name]
        
        if self._default:
            return self._adapters[self._default]
        
        raise ValueError("No adapter registered")
    
    def get_by_name(self, name: str) -> InferencePort:
        """按名称获取"""
        if name not in self._adapters:
            raise ValueError(f"Adapter not found: {name}")
        return self._adapters[name]
    
    def list_adapters(self) -> List[str]:
        """列出所有已注册的适配器"""
        return list(self._adapters.keys())


def create_model_registry(config) -> ModelRegistry:
    """工厂函数：创建模型注册中心"""
    from adapters.inference.vllm_adapter import VllmAdapter, VllmConfig
    from adapters.inference.llamacpp_adapter import LlamaCppAdapter, LlamaCppConfig
    
    registry = ModelRegistry()
    
    # 注册 vLLM
    if getattr(config, "vllm_enabled", True):
        vllm_adapter = VllmAdapter(VllmConfig(
            endpoint=config.vllm_endpoint,
            model_name=config.vllm_model_name,
        ))
        registry.register(
            "vllm",
            vllm_adapter,
            [InferencePriority.HIGH_ACCURACY],
            is_default=True
        )
    
    # 注册量化模型
    if getattr(config, "quantized_enabled", False):
        quant_adapter = LlamaCppAdapter(LlamaCppConfig(
            model_path=config.quantized_model_path,
        ))
        registry.register(
            "quantized",
            quant_adapter,
            [InferencePriority.COST_SAVING, InferencePriority.LOW_LATENCY]
        )
    
    return registry
```

### Day 5: 应用层用例实现

**`src/application/use_cases/search_tracks.py`**
```python
"""轨迹检索用例"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.inference.model_registry import ModelRegistry, InferencePriority
    from ports.vector_store_port import VectorStorePort


@dataclass
class SearchRequest:
    video_id: str
    question: str
    top_k: int = 5
    recall_limit: int = 50
    model_priority: Optional[str] = None


@dataclass
class SearchResult:
    track_id: int
    start_seconds: float
    end_seconds: float
    score: float
    reason: str


@dataclass
class SearchResponse:
    video_id: str
    question: str
    results: List[SearchResult]
    latency_ms: int
    model_variant: str


class SearchTracksUseCase:
    """轨迹检索用例"""
    
    def __init__(
        self,
        model_registry: "ModelRegistry",
        vector_store: Optional["VectorStorePort"] = None,
    ):
        self.model_registry = model_registry
        self.vector_store = vector_store
    
    async def execute(self, request: SearchRequest) -> SearchResponse:
        start_time = time.time()
        
        # 获取适配器
        if request.model_priority:
            from adapters.inference.model_registry import InferencePriority
            priority = InferencePriority(request.model_priority)
            adapter = self.model_registry.get_adapter(priority)
        else:
            adapter = self.model_registry.get_adapter()
        
        # TODO: 向量召回、硬规则过滤、VLM 验证
        # 暂时返回空结果
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            video_id=request.video_id,
            question=request.question,
            results=[],
            latency_ms=elapsed_ms,
            model_variant=adapter.__class__.__name__,
        )
```

---

## Week 6: 前端骨架

### Day 1-2: Next.js 项目初始化

```bash
npx create-next-app@latest frontend --typescript --tailwind --app --src-dir
cd frontend
npm install @shadcn/ui video.js recharts zustand socket.io-client
```

### Day 3-5: 核心页面

详见 `final_upgrade_blueprint.md` 第七章。

---

## Week 7: 完整测试 + 性能优化

### 测试覆盖

```bash
src/tests/
├── unit/
│   ├── test_inference_port.py
│   ├── test_vllm_adapter.py
│   ├── test_llamacpp_adapter.py
│   ├── test_model_registry.py
│   ├── test_qdrant_adapter.py
│   └── test_minio_adapter.py
├── integration/
│   ├── test_search_use_case.py
│   └── test_api_routes.py
└── e2e/
    └── test_full_pipeline.py
```

---

## 待删除文件清单

> Phase 2 结束后删除

| 文件 | 状态 | 理由 |
|------|------|------|
| `src/pipeline/vlm_client_hf.py` | 🔴 DELETE | 已被 vLLM/llama.cpp 适配器替代 |
| `src/pipeline/recall.py` | 🟡 REFACTOR | 迁移到 `application/use_cases/` |
| `src/core/config.py` | 🟡 REFACTOR | 迁移到 `infrastructure/config/` |

---

## 验收标准

- [ ] 向量检索延迟 < 50ms
- [ ] VLM 推理延迟 < 30s/track (vLLM)
- [ ] 支持 vLLM / llama.cpp 切换
- [ ] MinIO 存储可用
- [ ] 前端可展示结果
- [ ] 测试覆盖率 > 70%

---

## 下一步

完成 Phase 2 后，进入 [Phase 3: 规模化](./phase3_scale.md)

