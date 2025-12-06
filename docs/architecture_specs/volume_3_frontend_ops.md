# Edge-Detective 工业级系统架构白皮书 (卷三)

> **文档密级**: 核心设计 (Core Design)
> **版本**: 2.0.0 (Volume III: Frontend Experience & DevOps)
> **总架构师**: Gemini
> **日期**: 2025-05-12

---

# 第九部分：前端体验架构 (Frontend Experience Architecture)

AI 系统的前端不仅仅是 API 的展示层，它是用户与智能体交互的**控制台 (Cockpit)**。我们不再构建传统的 CRUD 页面，而是构建一个**实时、状态驱动的单页应用 (Real-time SPA)**。

## 9.1 技术栈选型：现代全栈 (The Modern Stack)

*   **Framework**: **Next.js 14 (App Router)**。利用 Server Components (RSC) 直接在服务端聚合数据，减少客户端瀑布流请求。
*   **State Management**:
    *   **Server State**: **TanStack Query (React Query)**。管理 API 数据缓存、自动重试、乐观更新 (Optimistic Updates)。
    *   **Client State**: **Zustand**。轻量级管理 UI 状态（如视频播放进度、侧边栏开关）。
*   **Visuals**: **Tailwind CSS + Shadcn UI**。工业级组件库，极速构建仪表盘。

## 9.2 核心交互模式设计

### 9.2.1 流式响应 (Streaming UX)
VLM 的推理是耗时的（可能需要 2-5 秒）。传统的“转圈加载”会让用户感到焦虑。
我们采用 **流式 UI (Streaming UI)**：
1.  用户输入 "Find a man in red"。
2.  **Phase 1 (Instant)**: 立即展示 Qdrant 返回的 Top-20 粗排结果（缩略图），标记为 "Candidates"。
3.  **Phase 2 (Streaming)**: 建立 WebSocket 连接。随着后台 vLLM 逐个验证 Candidate，前端实时点亮通过验证的卡片，并自动重新排序。
4.  **Value**: 用户在 200ms 内看到反馈，在 2s 内看到最终结果，体验极其流畅。

### 9.2.2 视频时间轴可视化 (Canvas Timeline)
对于长视频，列表展示是低效的。我们需要一个类似 Adobe Premiere 的**时间轴控件**。
*   使用 HTML5 Canvas 绘制 24小时的时间轴。
*   将检索到的 Track 渲染为时间轴上的**高亮色块**。
*   用户点击色块，播放器自动跳转到该秒数 (Seek)。

---

# 第十部分：云原生部署架构 (Cloud-Native DevOps)

代码写得再好，跑不起来也是白搭。我们采用 **Kubernetes (K8s)** 作为唯一的交付标准。

## 10.1 声明式基础设施 (Infrastructure as Code)

我们不写 Shell 脚本，只写 YAML。

### 10.1.1 核心 Deployment 清单

**Inference Pod (GPU Workload)**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args: ["--model", "Qwen/Qwen2-VL-7B-Instruct", "--gpu-memory-utilization", "0.95"]
        resources:
          limits:
            nvidia.com/gpu: 1  # 独占 L4 GPU
```

**API Pod (CPU Workload)**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-detective-api
spec:
  replicas: 3  # 水平扩展应对高并发
  template:
    spec:
      containers:
      - name: api
        image: edge-detective-api:v2.0
        env:
        - name: VLM_API_ENDPOINT
          value: "http://vllm-server:8000/v1"  # 指向 K8s Service
```

## 10.2 混合云部署策略 (Hybrid Cloud Strategy)

利用 K8s 的 **Node Selector** 和 **Taints/Tolerations**，我们可以实现完美的云边协同：

*   **Edge Nodes (Factory)**: 打上 label `zone=edge`。部署 YOLO 检测器和 Llama.cpp 量化服务。数据就地处理，不占带宽。
*   **Cloud Nodes (Data Center)**: 打上 label `zone=cloud`。部署 vLLM 和 Qdrant。处理复杂的全局搜索请求。

---

# 第十一部分：可观测性与运维 (Observability & Ops)

分布式系统的噩梦是：“它慢了，但我不知道为什么”。

## 11.1 分布式链路追踪 (Distributed Tracing)

我们集成 **OpenTelemetry (OTEL)** 标准。一个 Trace ID 将贯穿：
`Web Frontend` -> `API Gateway` -> `Orchestrator` -> `Vector DB` -> `Inference Gateway` -> `vLLM`。

在 Jaeger 或 Grafana Tempo 中，我们可以清晰地看到瀑布图：
*   Total Latency: 2.5s
*   `qdrant_search`: 50ms
*   `postgres_metadata`: 20ms
*   `vlm_inference` (span): 2.4s
    *   `token_generation`: 2.3s
    *   `network_overhead`: 100ms

**结论**: 一眼看出瓶颈在 GPU 推理，而不是数据库。

## 11.2 监控大盘 (Metrics Dashboard)

基于 Prometheus + Grafana。核心监控指标 (Golden Signals)：

1.  **Latency**: P99 检索延迟。如果超过 5秒，触发 PagerDuty 告警。
2.  **Traffic**: QPS (每秒查询数)。
3.  **Errors**: API 5xx 错误率。
4.  **Saturation**:
    *   **GPU Utilization**: 如果长期 < 30%，说明资源浪费，建议缩容或切换到更小的卡。
    *   **VLLM Queue Length**: 如果 > 0，说明请求在排队，需要立刻扩容 GPU 节点。

---

# 全文总结 (Final Conclusion)

至此，**《Edge-Detective 工业级系统架构白皮书》** 三卷全部完成。

*   **卷一** 确立了**计算与逻辑分离**的哲学，设计了**万能推理网关**，解决了量化模型引入和异构计算难题。
*   **卷二** 构建了基于 **CQRS** 的数据底座，解决了高吞吐写入与低延迟查询的矛盾，并规划了 Schema 演进策略。
*   **卷三** 补全了 **用户体验** 和 **运维落地** 的拼图，确保系统不仅“能跑”，而且“好用”、“好管”。

这套架构设计不再是一个简单的 Python 脚本集合，而是一个具备**弹性、可观测性、演进能力**的现代 AI 平台。它足以支撑从现在的单机 Demo 演进到未来覆盖数千路摄像头的城市级监控系统。

**(全书完)**
