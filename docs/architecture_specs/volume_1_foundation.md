# Edge-Detective 工业级系统架构白皮书 (卷一)

> **文档密级**: 核心设计 (Core Design)
> **版本**: 2.0.0 (Volume I: Foundation & Inference)
> **总架构师**: Gemini
> **日期**: 2025-05-12

---

# 第一部分：架构总论与设计哲学 (Architecture Overview & Philosophy)

## 1.1 从原型到工业级的跨越 (The Industrial Leap)

Edge-Detective 当前的代码库处于典型的 "Script-based Prototype"（脚本化原型）阶段。其特征是：业务逻辑与基础设施紧密耦合，数据状态散落在文件系统中，计算资源（GPU）被单一进程独占。这种架构在处理单一视频 Demo 时表现尚可，但在面对多路视频流、高并发检索、以及异构硬件部署（既要跑在 L4 服务器，又要跑在 Jetson 边缘端）时，存在根本性的**架构缺陷**。

本白皮书旨在定义一套全新的架构标准，以实现以下工业化目标：

1.  **硬件无关性 (Hardware Agnosticism)**: 业务代码不需要知道底层是 NVIDIA A100 还是嵌入式 NPU，也不需要知道模型是 FP16 还是 INT4 量化版本。
2.  **弹性伸缩 (Elastic Scalability)**: 推理能力（Inference）与业务编排（Orchestration）必须能够独立扩缩容。
3.  **数据主权 (Data Sovereignty)**: 建立严格的数据分级存储标准，确保从 GB 级的视频到 KB 级的元数据都有明确的归宿。

## 1.2 核心设计哲学：异构计算分离 (Separation of Heterogeneous Compute)

在深度学习应用中，最核心的矛盾在于 **CPU 负载（IO密集/逻辑复杂）** 与 **GPU 负载（计算密集/吞吐敏感）** 的不对称性。

*   **现状（反模式）**: `vlm_client_hf.py` 直接在 Python 主进程中加载 16GB 的模型权重。这导致 Python 的 GIL 锁限制了 GPU 的数据喂入效率，且一旦业务逻辑崩溃（如解析 JSON 失败），整个昂贵的模型上下文也会随之丢失，重启需要数分钟。
*   **新架构（正交设计）**: 我们采用 **Client-Server** 架构彻底切分这两者。
    *   **Orchestrator (CPU)**: 负责“思考”。它是无状态的、轻量级的，负责任务分发、状态管理、数据库读写。
    *   **Inference Services (GPU)**: 负责“计算”。它是纯粹的数学计算单元，通过 HTTP/gRPC 暴露标准接口。

这种分离带来的直接价值是：**我们可以在不修改任何业务代码的情况下，将推理后端从本地的 HuggingFace 切换到远程的 vLLM 集群，或者边缘的 Quantized Llama.cpp 服务。**

---

# 第二部分：领域驱动设计 (Domain-Driven Design)

为了治理复杂度，我们引入 DDD 方法论，对系统进行战略建模。

## 2.1 限界上下文 (Bounded Contexts)

我们将系统严格划分为以下上下文，每个上下文拥有独立的通用语言 (Ubiquitous Language) 和数据模型。

| 限界上下文 | 类型 | 职责 | 核心实体 | 依赖关系 |
| :--- | :--- | :--- | :--- | :--- |
| **Recall Context** (召回) | Core Domain | 负责语义检索的核心逻辑，包括两阶段筛选（向量粗排 -> VLM 精排）。 | `Query`, `CandidateTrack`, `SearchResult` | 下游依赖 Inference, Indexing |
| **Indexing Context** (索引) | Core Domain | 负责视频的摄入、解码、特征提取和入库。是写密集型业务。 | `Video`, `Track`, `Embedding` | 下游依赖 Inference, Storage |
| **Inference Context** (推理) | Generic Subdomain | 提供通用的 AI 计算能力。不关心业务含义，只关心 Token 输入输出。 | `Model`, `Prompt`, `Completion` | 无业务依赖 |
| **Identity Context** (追踪) | Supporting Subdomain | 负责跨摄像头的 Re-ID 和轨迹关联。 | `Trajectory`, `PersonID` | 被 Recall 依赖 |

## 2.2 防腐层 (ACL) 的战略地位

在 **Recall Context** 与 **Inference Context** 之间，必须建立一道坚固的 **反腐败层 (Anti-Corruption Layer, ACL)**。

*   **问题**: VLM 模型（无论是 Qwen 还是 GPT-4）输出的本质是概率性的自然语言。它可能输出 "Yes", "yes", "Match: yes", 甚至 "I think it is a match"。
*   **策略**: ACL 负责将这些“脏数据”清洗为领域内绝对可信的 **值对象 (Value Object)**。
*   **实现**: 任何未经 ACL 清洗的数据，严禁进入数据库或返回给前端。

---

# 第三部分：万能推理网关 (The Universal Inference Gateway)

这是本架构中最具“工业味”的设计，也是**量化模型 (Quantization)** 能够无缝切入的关键。

## 3.1 架构意图

我们不希望业务逻辑层充斥着这样的代码：
```python
# ❌ 错误示范：业务逻辑被底层细节污染
if config.use_quantization:
    from llama_cpp import Llama
    model = Llama(...)
elif config.use_vllm:
    import vllm
    ...
```

我们希望业务逻辑层看到的是：
```python
# ✅ 正确示范：面向接口编程
result = await inference_gateway.chat(prompt, images)
```

## 3.2 核心抽象层 (The Core Abstraction)

我们在 `src/core/interfaces/` 下定义一套纯粹的 Python Protocol（接口）：

```python
from typing import Protocol, List, Optional
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    temperature: float = 0.1
    max_tokens: int = 256
    top_p: float = 0.9

@dataclass
class InferenceResponse:
    text: str
    token_usage: int
    latency_ms: float
    model_version: str  # 关键：用于追溯是哪个模型生成的

class VLMInterface(Protocol):
    """
    万能推理接口。
    无论后端是 vLLM, Llama.cpp 还是 OpenAI API，都必须实现此协议。
    """
    async def chat_completion(
        self, 
        system_prompt: str,
        user_prompt: str,
        images: List[bytes],  # 统一使用二进制流，解耦文件系统
        config: Optional[GenerationConfig] = None
    ) -> InferenceResponse:
        ...
```

## 3.3 适配器模式与量化支持 (Adapters & Quantization)

我们将通过 **适配器模式 (Adapter Pattern)** 来支持多种推理后端。这使得引入量化模型变得轻而易举。

### 3.3.1 High-Performance Adapter (vLLM)
用于云端/服务器环境，追求极致的并发吞吐。

```python
class VllmAdapter(VLMInterface):
    def __init__(self, endpoint: str):
        self.client = AsyncOpenAI(base_url=endpoint, api_key="EMPTY")
    
    async def chat_completion(self, ...) -> InferenceResponse:
        # 1. 转换 logic: 将 bytes 图片转为 vLLM 接受的 base64/url
        # 2. 调用 OpenAI 兼容接口
        # 3. 封装标准 Response
        pass
```

### 3.3.2 Edge-Quantized Adapter (Llama.cpp)
**这是您关注的焦点。** 用于边缘设备或低成本环境。

```python
class LlamaCppAdapter(VLMInterface):
    def __init__(self, gguf_path: str, n_gpu_layers: int = -1):
        # 这里加载量化模型 (INT4 / Q4_K_M)
        from llama_cpp import Llama
        self.model = Llama(
            model_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            verbose=False
        )

    async def chat_completion(self, ...) -> InferenceResponse:
        # 在这里执行 CPU/混合推理
        # 将 output 封装为与 vLLM 完全一致的 InferenceResponse
        # 业务层根本不知道这次推理只消耗了 2GB 显存！
        pass
```

## 3.4 动态路由策略 (Dynamic Routing Strategy)

在工业级场景中，我们甚至可以实现更高级的 **混合推理 (Hybrid Inference)**。

```python
class SmartInferenceGateway(VLMInterface):
    def __init__(self, primary: VllmAdapter, fallback: LlamaCppAdapter):
        self.primary = primary    # FP16, 高精度
        self.fallback = fallback  # INT4, 低成本

    async def chat_completion(self, ...) -> InferenceResponse:
        try:
            # 尝试走主通道 (vLLM)
            return await self.primary.chat_completion(...)
        except (NetworkError, OverloadError):
            # 降级容灾：如果 vLLM 挂了，自动切到本地量化模型
            logger.warning("vLLM unavailable, falling back to local INT4 model")
            return await self.fallback.chat_completion(...)
```

---

# 第四部分：遗留系统迁移策略 (The Strangler Fig Pattern)

面对现有的 `vlm_client_hf.py`，我们采取 **绞杀植物模式 (Strangler Fig Pattern)** 进行渐进式重构。我们不建议“停止一切业务进行重写”，而是采用以下步骤：

## 4.1 阶段一：建立新皮层 (The Shim Layer)
在 `src/pipeline/` 中引入新的 `VlmGateway` 类，但在其内部，我们暂时实例化旧的 `Qwen3VL4BHFClient`。
*   **目的**: 统一调用入口，让上层业务逻辑 (`recall.py`) 先习惯使用新接口。

## 4.2 阶段二：旁路建设 (Bypass Construction)
在服务器上部署 vLLM 服务（作为 Sidecar 运行）。同时开发 `VllmAdapter`。此时，旧代码仍在运行，新代码已就绪。

## 4.3 阶段三：金丝雀发布 (Canary Release)
通过配置标志 `USE_NEW_VLM_BACKEND = True`，在开发环境启用新链路。
*   此时，`VlmGateway` 内部会根据配置，选择实例化 `VllmAdapter` 还是旧的 `HFClient`。

## 4.4 阶段四：全面扼杀 (Strangulation)
当 vLLM 链路稳定运行一周后，我们将默认配置改为 `True`。随后，彻底删除 `vlm_client_hf.py` 文件及其所有依赖（如 `transformers` 库的直接依赖）。

---

# 卷一总结

通过卷一的设计，我们确立了 Edge-Detective 2.0 的核心骨架：
1.  **物理隔离**解决了 GIL 锁和显存争抢问题。
2.  **Universal Inference Gateway** 为引入 **量化模型** 提供了完美的架构插槽，实现了业务逻辑与模型精度的解耦。
3.  **ACL** 保证了数据的纯净性。

**(卷一 完)**

---

*待续：*
*   *卷二：数据架构与 CQRS (Volume II: Data Architecture & CQRS)* —— 将详细阐述如何设计 Redis/Qdrant/PostgreSQL 的三级缓存，以及 Schema Evolution 策略。
*   *卷三：前端与运维 (Volume III: Frontend Experience & DevOps)* —— 将详细阐述 React 状态管理、WebSocket 实时流以及 Kubernetes 部署清单。
