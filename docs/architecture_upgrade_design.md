# 🏗️ Edge-Detective 系统架构升级与重构设计书 (Phase 2)

> **版本**: 2.0 (Draft)
> **对齐目标**: [Industrialization Roadmap - Phase 2](./industrialization_roadmap.md)
> **核心理念**: 解耦 (Decoupling)、异步 (Async)、服务化 (Service-Oriented)

---

## 1. 现状审计与"腐朽代码"清洗计划 (The Purge List)

为了达成工业化目标，我们需要痛下决心，清理不符合云原生架构的代码。

### 🚨 待删除/废弃组件 (Deprecated Components)

| 组件/文件 | 判定 | 理由 (Why it's bad) | 替代方案 |
| :--- | :--- | :--- | :--- |
| **`src/pipeline/vlm_client_hf.py`** | **🔥 DELETE** | **反模式核心**。直接在业务进程中加载 16GB 模型权重。导致无法水平扩展 API 服务，且受限于 Python GIL 和 PyTorch 调度瓶颈，引发 Padding 性能问题。 | **vLLM Service** (独立进程) + **HTTP Client** |
| `src/core/config.py` (部分字段) | **REFACTOR** | 混杂了"基础设施配置"（显卡型号）与"业务配置"。`yolo_device`, `vlm_batch_size` 等底层硬件参数不应由业务代码管理。 | 配置应拆分为 `AppConfig` (业务) 和 `InfraConfig` (K8s/Env)。硬件参数移交给 `docker-compose.yaml` 或启动脚本。 |
| `src/pipeline/recall.py` (同步逻辑) | **REWRITE** | 核心调度逻辑是同步阻塞的（Serial Blocking）。在调用 VLM 时整个线程卡死，无法处理并发请求。 | **Async/Await** 重写。使用 `asyncio.gather` 并发请求推理服务。 |
| 本地文件系统依赖 (File I/O) | **PHASE OUT** | 代码中大量出现的 `open(path)`。如果在 K8s 多节点部署，Worker A 无法读取 Worker B 存的图片。 | **Object Storage (MinIO)** 抽象层。 |

---

## 2. 目标架构设计 (Target Architecture)

我们将从 **Monolithic Script (单体脚本)** 转型为 **Model-as-a-Service (模型即服务)** 架构。

### 2.1 系统拓扑图

```mermaid
graph TD
    subgraph "Control Plane (CPU-Bound)"
        API[FastAPI Gateway] -->|Async HTTP| Orchestrator[Pipeline Orchestrator]
        Orchestrator -->|Read/Write| DB[(PostgreSQL)]
        Orchestrator -->|Read/Write| VectorDB[(Qdrant)]
        Orchestrator -->|Put/Get| ObjStore[(MinIO)]
    end

    subgraph "Inference Plane (GPU-Bound)"
        Orchestrator -- "OpenAI API Protocol" --> vLLM[vLLM Server (Qwen2-VL)]
        Orchestrator -- "gRPC/HTTP" --> Detection[YOLO/SigLIP Service]
    end

    subgraph "Infrastructure"
        vLLM -->|Mapped| GPU[L4 GPU]
    end
```

### 2.2 关键架构决策

1.  **推理与业务物理隔离**：
    *   **推理层 (Inference Layer)**：vLLM 独占 GPU。它只负责计算，不知道什么是 "Track" 或 "Evidence"，只知道 "Input Tokens -> Output Tokens"。
    *   **业务层 (Business Layer)**：Python 业务代码只负责逻辑判断。它不知道模型是跑在本地还是跑在火星上。

2.  **通信协议标准化**：
    *   所有 VLM 交互强制使用 **OpenAI API 兼容协议**。
    *   **优势**：如果我们明天想测试 GPT-4o 或 Claude 3.5 Sonnet，只需改一个 URL 配置，代码一行不用动。

3.  **IO 也是并行的**：
    *   在旧架构中，VLM 分析 Track A 时，CPU 是闲置的。
    *   在新架构中，Orchestrator 可以同时向 vLLM 发送 10 个 Track 的请求（vLLM 内部会自动做 Continuous Batching），同时向数据库写入元数据。

---

## 3. 重构实施方案 (Implementation Plan)

### Step 1: 部署推理服务 (Infrastructure)

我们需要先让模型作为独立服务跑起来。这是所有重构的前提。

**新建文件: `deploy/start_vllm.sh`**
```bash
# 工业级启动参数
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=5 \
    --enable-prefix-caching  # 开启前缀缓存，大幅加速相同 Prompt 的检索
```

### Step 2: 编写新的 VLM 客户端 (Code)

**新建文件: `src/pipeline/vlm_client_vllm.py`**
该客户端完全**无状态**，仅封装 HTTP 调用。

```python
import base64
from openai import AsyncOpenAI
from core.config import SystemConfig

class VLMClientVLLM:
    def __init__(self, config: SystemConfig):
        # 使用标准 OpenAI 客户端
        self.client = AsyncOpenAI(
            base_url=config.vlm_api_url,  # e.g., "http://localhost:8000/v1"
            api_key="EMPTY"
        )
        self.model = config.vlm_model_name

    def _encode_image(self, image_path: str) -> str:
        """工业化处理：未来这里可替换为直接生成 MinIO Presigned URL"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def answer_track(self, track: EvidencePackage, question: str) -> QueryResult:
        """异步单条处理 - 并发由调用方控制"""
        # 构造 OpenAI 格式消息...
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1
        )
        return self._parse(response)
```

### Step 3: 重构核心流水线 (Async Orchestration)

**修改: `src/pipeline/recall.py`**

将原本的 `for` 循环串行调用改为 `asyncio.gather`。

```python
# 旧代码 (The Bad)
# for package in candidates:
#     result = client.answer(package)  <-- 阻塞

# 新代码 (The Good)
import asyncio

class AsyncRecallEngine:
    async def process_candidates(self, candidates, question):
        tasks = []
        for package in candidates:
            # 创建异步任务
            tasks.append(self.vlm_client.answer_track(package, question))
        
        # 并发执行所有请求
        # vLLM 服务端会自动处理这些并发请求的 Batching (Continuous Batching)
        results = await asyncio.gather(*tasks)
        return results
```

---

## 4. 收益预测 (ROI)

通过这次架构升级，我们将获得：

1.  **速度质变**：
    *   不再受限于 Padding。vLLM 的 Continuous Batching 能让吞吐量提升 **3-5 倍**。
    *   Python 端不再阻塞，可以处理其他 I/O。
2.  **调试友好**：
    *   模型服务一直开着，调试 Python 代码时不需要每次都等 2 分钟加载模型。**修改代码 -> 运行** 的反馈循环缩短到 1 秒。
3.  **未来就绪**：
    *   直接对接 Phase 3 的 K8s 部署。FastAPI 容器和 vLLM 容器可以分别扩容。

---

## 5. 下一步行动建议

1.  **批准删除**: 确认废弃 `vlm_client_hf.py`。
2.  **环境准备**: 在您的 L4 机器上启动 vLLM Server。
3.  **代码替换**: 我将为您编写上述的 `vlm_client_vllm.py` 和异步版的 `recall.py`。

```