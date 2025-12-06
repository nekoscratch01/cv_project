# 🚀 Phase 1: MVP 开发计划

> **目标**: 从 CLI 原型升级为分层架构 + vLLM 推理服务  
> **周期**: 2-3 周  
> **核心交付**: `question_search()` 可通过 vLLM 运行

---

## 目录

1. [Week 1: vLLM 集成 + 核心分层](#week-1-vllm-集成--核心分层)
2. [Week 2: 基础设施搭建](#week-2-基础设施搭建)
3. [Week 3: API 层 + 异步任务](#week-3-api-层--异步任务)
4. [待删除文件清单](#待删除文件清单)
5. [验收标准](#验收标准)

---

## Week 1: vLLM 集成 + 核心分层

> **目标**: demo 能用 vLLM 跑起来

### Day 1-2: 目录结构 + 端口定义

#### 任务清单

- [ ] 创建新的目录结构
- [ ] 定义核心端口接口
- [ ] 创建领域实体

#### 需要创建的文件

```bash
# 端口层
src/ports/__init__.py
src/ports/inference_port.py

# 领域层 - 值对象
src/domain/__init__.py
src/domain/value_objects/__init__.py
src/domain/value_objects/verification_result.py
```

#### 代码实现

**`src/ports/__init__.py`**
```python
"""端口层：定义业务层依赖的抽象接口"""
from .inference_port import InferencePort

__all__ = ["InferencePort"]
```

**`src/ports/inference_port.py`**
```python
"""推理端口：业务层只依赖此抽象接口"""
from __future__ import annotations

from typing import Protocol, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from domain.value_objects.verification_result import VerificationResult
    from core.evidence import EvidencePackage


class InferencePort(Protocol):
    """
    推理端口协议
    
    设计原则：
    - 业务层只依赖此接口，不依赖具体实现
    - 支持 vLLM、量化模型、云端 API 等多种后端
    - 异步优先，支持高并发
    """
    
    async def verify_track(
        self,
        package: "EvidencePackage",
        question: str,
        plan_context: Optional[str] = None,
    ) -> "VerificationResult":
        """
        验证单个轨迹是否匹配查询
        
        Args:
            package: 证据包（包含图片路径、特征等）
            question: 用户查询
            plan_context: Router 生成的上下文信息
            
        Returns:
            VerificationResult: 包含 match 判断、置信度、原因
        """
        ...
    
    async def verify_batch(
        self,
        packages: List["EvidencePackage"],
        question: str,
        plan_context: Optional[str] = None,
        concurrency: int = 10,
    ) -> List["VerificationResult"]:
        """
        批量验证轨迹（真正的并发）
        
        与 HF transformers 的 Batch 不同：
        - HF Batch: 同一个 forward pass，受 padding 影响
        - vLLM 并发: 多个独立请求，vLLM 内部 Continuous Batching
        """
        ...
```

**`src/domain/value_objects/verification_result.py`**
```python
"""验证结果值对象"""
from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass


class MatchStatus(Enum):
    """匹配状态枚举"""
    CONFIRMED = "confirmed"      # 确认匹配
    REJECTED = "rejected"        # 确认不匹配
    AMBIGUOUS = "ambiguous"      # 模糊/无法判断


@dataclass(frozen=True)
class VerificationResult:
    """
    验证结果值对象（不可变）
    
    这是反腐败层(ACL)的输出，将 VLM 的自然语言响应
    转换为系统内部的结构化表示。
    """
    status: MatchStatus
    confidence: float
    reason: str
    raw_response: str
    
    @classmethod
    def confirmed(cls, confidence: float, reason: str, raw: str = "") -> "VerificationResult":
        return cls(MatchStatus.CONFIRMED, confidence, reason, raw)
    
    @classmethod
    def rejected(cls, reason: str, raw: str = "") -> "VerificationResult":
        return cls(MatchStatus.REJECTED, 0.0, reason, raw)
    
    @classmethod
    def error(cls, error_msg: str) -> "VerificationResult":
        return cls(MatchStatus.AMBIGUOUS, 0.0, f"Error: {error_msg}", "")
    
    @property
    def is_match(self) -> bool:
        return self.status == MatchStatus.CONFIRMED


class VlmResponseParser:
    """
    VLM 响应解析器（反腐败层实现）
    
    职责：
    1. 从自然语言响应中提取结构化信息
    2. 处理各种边界情况和异常格式
    3. 将不可靠的外部数据转换为可靠的内部表示
    """
    
    MATCH_PATTERN = re.compile(r"MATCH:\s*(yes|no)", re.IGNORECASE)
    CONFIDENCE_PATTERN = re.compile(r"confidence[:\s]+(\d+(?:\.\d+)?)", re.IGNORECASE)
    
    def parse(self, raw_response: str) -> VerificationResult:
        """解析 VLM 原始响应"""
        if not raw_response:
            return VerificationResult.error("Empty response")
        
        match_result = self._extract_match_marker(raw_response)
        confidence = self._extract_confidence(raw_response)
        status = self._determine_status(match_result, confidence)
        
        return VerificationResult(
            status=status,
            confidence=confidence,
            reason=self._extract_reason(raw_response),
            raw_response=raw_response
        )
    
    def _extract_match_marker(self, text: str) -> bool | None:
        match = self.MATCH_PATTERN.search(text)
        if match:
            return match.group(1).lower() == "yes"
        return None
    
    def _extract_confidence(self, text: str) -> float:
        match = self.CONFIDENCE_PATTERN.search(text)
        if match:
            try:
                conf = float(match.group(1))
                return min(max(conf, 0.0), 1.0)
            except ValueError:
                pass
        return 0.8 if self._extract_match_marker(text) is not None else 0.5
    
    def _determine_status(self, match_result: bool | None, confidence: float) -> MatchStatus:
        if match_result is True and confidence >= 0.6:
            return MatchStatus.CONFIRMED
        elif match_result is False:
            return MatchStatus.REJECTED
        else:
            return MatchStatus.AMBIGUOUS
    
    def _extract_reason(self, text: str) -> str:
        lines = text.strip().split("\n")
        reason_lines = [
            line for line in lines
            if not line.strip().lower().startswith("match:")
        ]
        return " ".join(reason_lines).strip()[:500]
```

---

### Day 3-4: vLLM 适配器实现

#### 任务清单

- [ ] 创建 vLLM 适配器
- [ ] 修改配置支持 vLLM
- [ ] 修改 `_build_vlm_client()` 工厂方法

#### 需要创建的文件

```bash
src/adapters/__init__.py
src/adapters/inference/__init__.py
src/adapters/inference/vllm_adapter.py
```

#### 代码实现

**`src/adapters/inference/vllm_adapter.py`**
```python
"""vLLM 推理适配器"""
from __future__ import annotations

import base64
import asyncio
from typing import List, Optional
from dataclasses import dataclass

from openai import AsyncOpenAI

from ports.inference_port import InferencePort
from domain.value_objects.verification_result import VerificationResult, VlmResponseParser
from core.evidence import EvidencePackage


@dataclass
class VllmConfig:
    """vLLM 配置"""
    endpoint: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 256
    timeout: float = 120.0
    max_retries: int = 3
    max_images_per_request: int = 5


class VllmAdapter:
    """
    vLLM 推理适配器
    
    实现 InferencePort 协议，通过 OpenAI 兼容 API 调用 vLLM 服务。
    
    特点：
    1. 完全无状态 - 可以在多个 worker 间共享
    2. 异步优先 - 支持高并发
    3. 重试机制 - 网络抖动容错
    """
    
    def __init__(self, config: VllmConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.endpoint,
            api_key="EMPTY",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self._parser = VlmResponseParser()
    
    async def verify_track(
        self,
        package: EvidencePackage,
        question: str,
        plan_context: Optional[str] = None,
    ) -> VerificationResult:
        """验证单个轨迹"""
        try:
            messages = self._build_messages(package, question, plan_context)
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw_text = response.choices[0].message.content
            return self._parser.parse(raw_text)
        except Exception as e:
            return VerificationResult.error(str(e))
    
    async def verify_batch(
        self,
        packages: List[EvidencePackage],
        question: str,
        plan_context: Optional[str] = None,
        concurrency: int = 10,
    ) -> List[VerificationResult]:
        """批量验证（真正的并发请求）"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _verify_with_limit(pkg: EvidencePackage) -> VerificationResult:
            async with semaphore:
                return await self.verify_track(pkg, question, plan_context)
        
        tasks = [_verify_with_limit(pkg) for pkg in packages]
        return await asyncio.gather(*tasks)
    
    def _build_messages(
        self,
        package: EvidencePackage,
        question: str,
        plan_context: Optional[str],
    ) -> List[dict]:
        """构造 OpenAI 格式消息"""
        crop_paths = self._sample_crops(package.crops)
        
        image_contents = []
        for path in crop_paths:
            base64_image = self._encode_image(path)
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        prompt = self._build_prompt(package, question, plan_context)
        
        return [
            {
                "role": "system",
                "content": "You are a video analysis assistant. Answer with reasoning, then end with 'MATCH: yes' or 'MATCH: no'."
            },
            {
                "role": "user",
                "content": [*image_contents, {"type": "text", "text": prompt}]
            }
        ]
    
    def _sample_crops(self, crops: List[str]) -> List[str]:
        max_crops = self.config.max_images_per_request
        if len(crops) <= max_crops:
            return list(crops)
        step = len(crops) / max_crops
        indices = [int(i * step) for i in range(max_crops)]
        return [crops[i] for i in indices]
    
    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _build_prompt(
        self,
        package: EvidencePackage,
        question: str,
        plan_context: Optional[str],
    ) -> str:
        motion_desc = self._build_motion_description(package)
        return f"""## Task
Verify if this person matches the query: "{question}"

## Evidence
### Appearance
The images show the person at different moments in the video.

### Motion Summary
{motion_desc}

### Constraints
{plan_context or "No additional constraints."}

## Instructions
1. Describe what you see in the images.
2. Check if the person matches the query criteria.
3. Final line must be: MATCH: yes or MATCH: no
"""
    
    def _build_motion_description(self, package: EvidencePackage) -> str:
        if not package.features:
            return "No motion data available."
        
        feats = package.features
        parts = []
        
        if feats.avg_speed_px_s < 50:
            parts.append("Standing still or barely moving")
        elif feats.avg_speed_px_s < 200:
            parts.append("Walking at normal pace")
        else:
            parts.append("Moving fast or running")
        
        dx, dy = feats.displacement_vec
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down (towards camera)" if dy > 0 else "up (away)"
        parts.append(f"Moving {direction}")
        
        return ". ".join(parts) + "."
```

#### 修改现有文件

**`src/core/config.py`** - 添加 vLLM 配置

```python
# 在 SystemConfig 类中添加：

# vLLM 配置
vlm_backend: str = "vllm"  # "hf" | "vllm"
vllm_endpoint: str = "http://localhost:8000/v1"
vllm_model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
```

**`src/pipeline/video_semantic_search.py`** - 修改工厂方法

```python
def _build_vlm_client(self):
    if self.config.vlm_backend == "vllm":
        from adapters.inference.vllm_adapter import VllmAdapter, VllmConfig
        return VllmAdapter(VllmConfig(
            endpoint=self.config.vllm_endpoint,
            model_name=self.config.vllm_model_name,
            temperature=self.config.vlm_temperature,
            max_tokens=self.config.vlm_max_new_tokens,
        ))
    elif self.config.vlm_backend in {"hf", "transformers"}:
        from pipeline.vlm_client_hf import Qwen3VL4BHFClient
        return Qwen3VL4BHFClient(self.config)
    else:
        raise RuntimeError(f"Unknown vlm_backend: {self.config.vlm_backend}")
```

---

### Day 5: vLLM 服务部署 + 端到端测试

#### 任务清单

- [ ] 在 Colab 上部署 vLLM 服务
- [ ] 编写 vLLM 启动脚本
- [ ] 端到端测试 `question_search()`
- [ ] 编写单元测试

#### vLLM 启动脚本

**`deploy/start_vllm.sh`**
```bash
#!/bin/bash
# vLLM 服务启动脚本

# 基础版本（FP16）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=5 \
    --enable-prefix-caching

# AWQ 量化版本（更快，显存更小）
# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-VL-4B-Instruct-AWQ \
#     --trust-remote-code \
#     --quantization awq \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --gpu-memory-utilization 0.90 \
#     --max-model-len 8192 \
#     --limit-mm-per-prompt image=5
```

#### Colab Notebook 代码

```python
# Cell 1: 安装依赖
!pip install vllm openai

# Cell 2: 启动 vLLM 服务（后台运行）
import subprocess
import time

process = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen3-VL-4B-Instruct",
    "--trust-remote-code",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--gpu-memory-utilization", "0.90",
    "--max-model-len", "8192",
    "--limit-mm-per-prompt", "image=5",
])

print("Waiting for vLLM to start...")
time.sleep(120)  # 等待模型加载

# Cell 3: 测试 vLLM 服务
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-4B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)
print(response.choices[0].message.content)

# Cell 4: 运行 demo
%cd /content/cv_project
!python -c "
from pipeline.video_semantic_search import VideoSemanticSystem

system = VideoSemanticSystem()
system.build_index()
system.question_search('Find the person in blue')
"
```

#### 单元测试

**`tests/unit/test_vllm_adapter.py`**
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from adapters.inference.vllm_adapter import VllmAdapter, VllmConfig
from domain.value_objects.verification_result import MatchStatus


@pytest.fixture
def mock_config():
    return VllmConfig(
        endpoint="http://localhost:8000/v1",
        model_name="test-model"
    )


@pytest.fixture
def adapter(mock_config):
    with patch("adapters.inference.vllm_adapter.AsyncOpenAI"):
        return VllmAdapter(mock_config)


class TestVlmResponseParser:
    def test_parse_match_yes(self):
        from domain.value_objects.verification_result import VlmResponseParser
        
        parser = VlmResponseParser()
        result = parser.parse("The person is wearing blue. MATCH: yes")
        
        assert result.status == MatchStatus.CONFIRMED
        assert result.is_match is True
    
    def test_parse_match_no(self):
        from domain.value_objects.verification_result import VlmResponseParser
        
        parser = VlmResponseParser()
        result = parser.parse("No match found. MATCH: no")
        
        assert result.status == MatchStatus.REJECTED
        assert result.is_match is False
    
    def test_parse_empty_response(self):
        from domain.value_objects.verification_result import VlmResponseParser
        
        parser = VlmResponseParser()
        result = parser.parse("")
        
        assert result.status == MatchStatus.AMBIGUOUS


@pytest.mark.asyncio
async def test_verify_track(adapter):
    # Mock the OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="MATCH: yes"))]
    adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Mock package
    mock_package = MagicMock()
    mock_package.crops = ["path/to/image.jpg"]
    mock_package.features = None
    
    with patch.object(adapter, "_encode_image", return_value="base64data"):
        result = await adapter.verify_track(mock_package, "test question")
    
    assert result.is_match is True
```

---

## Week 2: 基础设施搭建

> **目标**: 搭建数据库、日志、配置管理

### Day 1-2: 配置系统重构

#### 需要创建的文件

```bash
src/infrastructure/__init__.py
src/infrastructure/config/__init__.py
src/infrastructure/config/app_config.py
src/infrastructure/config/infra_config.py
```

#### 代码实现

**`src/infrastructure/config/app_config.py`**
```python
"""应用配置（业务层）"""
from pydantic_settings import BaseSettings
from typing import Optional


class AppConfig(BaseSettings):
    """应用配置"""
    
    # 服务
    service_name: str = "edge-detective"
    debug: bool = False
    
    # 推理
    vlm_backend: str = "vllm"
    vllm_endpoint: str = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    
    # 业务参数
    default_top_k: int = 5
    default_recall_limit: int = 50
    max_images_per_request: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### Day 3-4: 结构化日志

**`src/infrastructure/logging/structured_logger.py`**
```python
"""结构化日志配置"""
import structlog
import logging


def configure_logging(debug: bool = False):
    """配置结构化日志"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(level=log_level)


def get_logger(name: str = __name__):
    return structlog.get_logger(name)
```

### Day 5: Mock 存储适配器

**`src/adapters/storage/memory_repo.py`**
```python
"""内存存储（开发/测试用）"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class InMemoryVideoRepository:
    """视频仓储内存实现"""
    _videos: Dict[str, dict] = field(default_factory=dict)
    
    async def save(self, video_id: str, data: dict):
        self._videos[video_id] = data
    
    async def get(self, video_id: str) -> Optional[dict]:
        return self._videos.get(video_id)
    
    async def list_all(self) -> List[dict]:
        return list(self._videos.values())


@dataclass
class InMemoryTrackRepository:
    """轨迹仓储内存实现"""
    _tracks: Dict[str, dict] = field(default_factory=dict)
    
    async def save(self, video_id: str, track_id: int, data: dict):
        key = f"{video_id}_{track_id}"
        self._tracks[key] = data
    
    async def get_by_video(self, video_id: str) -> List[dict]:
        return [
            t for k, t in self._tracks.items()
            if k.startswith(f"{video_id}_")
        ]
```

---

## Week 3: API 层 + 异步任务

> **目标**: FastAPI 服务 + Celery 任务队列

### Day 1-2: FastAPI 骨架

**`src/api/main.py`**
```python
"""FastAPI 入口"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, search

app = FastAPI(
    title="Edge-Detective API",
    version="2.0.0",
    description="视频语义检索服务"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])


@app.on_event("startup")
async def startup():
    from infrastructure.logging.structured_logger import configure_logging
    configure_logging()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**`src/api/routes/health.py`**
```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness():
    return {"ready": True}


@router.get("/health/live")
async def liveness():
    return {"alive": True}
```

**`src/api/routes/search.py`**
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class SearchRequest(BaseModel):
    video_id: str
    question: str
    top_k: int = 5


class TrackResult(BaseModel):
    track_id: int
    start_s: float
    end_s: float
    score: float
    reason: str


class SearchResponse(BaseModel):
    video_id: str
    question: str
    results: List[TrackResult]
    latency_ms: int


@router.post("/search", response_model=SearchResponse)
async def search_tracks(request: SearchRequest):
    # TODO: 接入 SearchTracksUseCase
    raise HTTPException(status_code=501, detail="Not implemented yet")
```

### Day 3-4: Celery 任务队列

**`src/tasks/celery_app.py`**
```python
from celery import Celery

app = Celery(
    "edge_detective",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
```

**`src/tasks/indexing.py`**
```python
from tasks.celery_app import app


@app.task(bind=True)
def index_video_task(self, video_path: str, video_id: str):
    """后台索引任务"""
    self.update_state(state="PROCESSING", meta={"progress": 0})
    
    # TODO: 调用 IndexVideoUseCase
    
    return {"status": "completed", "video_id": video_id}
```

### Day 5: 整合测试

```bash
# 启动 Redis
docker run -d -p 6379:6379 redis:7-alpine

# 启动 Celery Worker
celery -A tasks.celery_app worker --loglevel=info

# 启动 FastAPI
uvicorn api.main:app --reload --port 8080

# 测试
curl http://localhost:8080/health
```

---

## 待删除文件清单

> Phase 1 结束后，以下文件标记为 **DEPRECATED**，Phase 2 开始时删除

| 文件 | 状态 | 理由 | 替代 |
|------|------|------|------|
| `src/pipeline/vlm_client_hf.py` | 🔴 DEPRECATED | 直接在业务进程加载模型 | `adapters/inference/vllm_adapter.py` |
| `src/api/` (旧版) | 🔴 DELETE | 如果存在旧 API 代码 | `src/api/` (新版) |
| `src/tasks/` (旧版) | 🔴 DELETE | 如果存在旧任务代码 | `src/tasks/` (新版) |

---

## 验收标准

### Week 1 验收

- [ ] `question_search()` 可通过 vLLM 运行
- [ ] `ports/inference_port.py` 定义完成
- [ ] `adapters/inference/vllm_adapter.py` 实现完成
- [ ] 单元测试通过

### Week 2 验收

- [ ] 配置系统重构完成
- [ ] 结构化日志可用
- [ ] Mock 存储适配器可用

### Week 3 验收

- [ ] FastAPI 服务可启动
- [ ] `/health` 端点可访问
- [ ] Celery 任务可执行
- [ ] Docker 环境可运行

---

## 下一步

完成 Phase 1 后，进入 [Phase 2: 生产可用](./phase2_production.md)

