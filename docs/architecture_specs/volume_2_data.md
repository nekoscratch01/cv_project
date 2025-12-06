# Edge-Detective 工业级系统架构白皮书 (卷二)

> **文档密级**: 核心设计 (Core Design)
> **版本**: 2.0.0 (Volume II: Data Architecture & CQRS)
> **总架构师**: Gemini
> **日期**: 2025-05-12

---

# 第五部分：数据架构概论 (Data Architecture Overview)

在视频语义检索领域，数据不仅是“记录”，更是“流动的资产”。我们面临着极其特殊的数据特征：
*   **海量非结构化数据**: 原始视频、截图 (Crops)。
*   **高维向量数据**: 图像 Embedding (768维或更高)。
*   **读写极端不对称**:
    *   **写入端 (Index)**: 持续、高吞吐、批处理。一小时视频可能产生 10万+ 帧截图和向量。
    *   **读取端 (Search)**: 突发、低延迟、交互式。用户期望在 200ms 内得到 Top-K 结果。

基于此，传统的 CRUD（增删改查）模型完全失效。我们必须采用 **CQRS (Command Query Responsibility Segregation，命令查询职责分离)** 模式。

---

# 第六部分：CQRS 实战设计 (CQRS in Action)

## 6.1 Command Side: 索引流水线 (The Write Model)

**职责**: 负责“视频理解”。这是一个重计算、高吞吐的离线（或近线）过程。
**核心原则**: 最终一致性 (Eventual Consistency)。

### 6.1.1 架构组件
*   **Ingestion Worker**: 负责视频解码、YOLO 检测、SigLIP 特征提取。
*   **Write Buffer (Kafka/Redis Stream)**: 削峰填谷。当 100 个摄像头同时上传视频时，保护后端数据库不被击穿。
*   **Batch Writer**: 批量将向量写入 Qdrant，将元数据写入 PostgreSQL。

### 6.1.2 数据流 (Data Flow)
1.  **Command**: `IndexVideo(video_path)`
2.  **Process**:
    *   解码器提取关键帧 -> 存入 MinIO (Object Storage) -> 生成 `s3://...` 路径。
    *   YOLO 提取 BBox -> 存入 PostgreSQL (Metadata)。
    *   SigLIP 提取 Embedding -> 存入 Qdrant (Vector Data)。
3.  **Event**: 每一批次处理完，触发 `SegmentIndexed` 事件，更新索引进度。

**工业化细节**:
为了支持**量化模型**在边缘端运行，Command Side 可以部署在边缘设备（Jetson）。边缘设备只负责提取 Embedding 和 BBox，然后将这些轻量级数据（KB 级别）上传到云端，避免上传原始视频（GB 级别），极大节省带宽。

## 6.2 Query Side: 检索与重排 (The Read Model)

**职责**: 负责“语义匹配”。这是一个延迟敏感的在线过程。
**核心原则**: 读写分离，异构视图。

### 6.2.1 架构组件
*   **Search Aggregator**: 聚合层。
*   **Vector Engine (Qdrant)**: 提供粗排 (Recall)。
*   **Rerank Engine (vLLM)**: 提供精排 (Precision)。

### 6.2.2 散列-聚合模式 (Scatter-Gather)
当用户查询 "A man in red shirt running" 时：
1.  **Scatter**: 
    *   向 Qdrant 发起 ANN (Approximate Nearest Neighbor) 搜索，获取 Top-100 候选。
    *   并行查询 PostgreSQL，获取这 100 个候选的时间戳、位置等元数据。
2.  **Gather**:
    *   将 100 个候选打包。
3.  **Rerank**:
    *   调用 **Inference Gateway** (见卷一)，对 Top-100 进行 VLM 视觉验证。
    *   这里可以应用**动态剪枝**：如果前 10 个置信度都很高，就不需要验证后 90 个了。

---

# 第七部分：数据分级存储策略 (Tiered Storage Strategy)

为了平衡成本与性能，我们设计了严密的 L1/L2/L3 存储分级。

## 7.1 L1: Hot Tier (Redis Cluster)

*   **定位**: 瞬时状态、高频计数器、分布式锁。
*   **存储内容**:
    *   `task:{task_id}:progress`: 视频索引的实时进度条（前端轮询用）。
    *   `session:{user_id}:last_query`: 用户的上下文历史（用于多轮对话）。
    *   `dedup:{image_hash}`: 7天内的图片去重指纹，避免重复索引同一视频。
*   **TTL 策略**: 极其激进，通常为 1小时 - 24小时。

## 7.2 L2: Warm Tier (Qdrant & Memory Mapped)

*   **定位**: 核心检索索引。必须支持毫秒级 ANN 搜索。
*   **存储内容**:
    *   **Collections**: `frames_siglip_v1`, `tracks_reid_v2`.
    *   **Payloads**: 仅存储用于过滤的最小元数据（`camera_id`, `timestamp`, `track_id`）。严禁存储大段 JSON 或 URL。
*   **硬件要求**: 必须部署在 NVMe SSD 或大内存节点上。

## 7.3 L3: Cold Tier (PostgreSQL & MinIO)

*   **定位**: 事实真理来源 (Source of Truth)。成本最低，容量最大。
*   **PostgreSQL**:
    *   存储完整的业务对象：`Videos`, `Tracks`, `AuditLogs`。
    *   使用 **JSONB** 列存储非结构化的检测属性（因为 YOLO 版本更新可能增加新属性，如 'mask_on'）。
*   **MinIO (S3 Compatible)**:
    *   **Immutable Assets**: 原始视频文件、裁剪后的行人图片 (Crops)、生成的缩略图。
    *   **路径规范**: `/{date}/{camera_id}/{video_id}/{track_id}.jpg`。

---

# 第八部分：Schema Evolution (数据演进策略)

这是工业界最大的痛点之一。当模型升级时，历史数据怎么办？

## 8.1 向量版本控制 (Vector Versioning)

**场景**: 我们将 Embedding 模型从 `SigLIP-Base` (768维) 升级到了 `SigLIP-Large` (1024维)。

**策略**: **多版本共存 (Blue-Green Indexing)**。
1.  在 Qdrant 中创建新 Collection: `tracks_v2` (1024 dim)。
2.  **Write Path**: 新视频同时写入 `tracks_v1` 和 `tracks_v2`（双写），或者只写 `v2`。
3.  **Backfill**: 启动后台任务，从 MinIO 读取历史 Crops，用新模型重新推理，填补 `v2`。
4.  **Read Path**: 
    *   Inference Gateway 会根据配置，决定 Search 使用 v1 还是 v2 模型提取 Query Vector。
    *   一旦 `v2` 数据回填完成，切换配置，废弃 `v1`。

## 8.2 元数据模式演进 (Metadata Evolution)

**场景**: YOLO 模型升级，新增了 `has_backpack` (背包) 属性。旧数据没有这个字段。

**策略**: **宽表 + 默认值 (Flexible Schema)**。
*   PostgreSQL 中使用 JSONB 存储检测属性。
*   在应用层 (Python Pydantic Model) 处理向后兼容性：
    ```python
    class TrackAttributes(BaseModel):
        speed: float
        has_backpack: bool = Field(default=False)  # 旧数据读取时默认为 False
    ```
*   这避免了可怕的 `ALTER TABLE` 锁表操作。

---

# 卷二总结

通过卷二的设计，我们解决了“数据怎么存、怎么查、怎么变”的核心问题：
1.  **CQRS** 完美解耦了重型的索引计算和轻快的用户检索。
2.  **分级存储** 让我们在成本和性能之间找到了最优解（用 Redis 扛并发，用 MinIO 扛容量）。
3.  **Schema Evolution** 策略确保了系统不会因为模型升级而成为“数据遗留代码 (Data Legacy)”的受害者。

**(卷二 完)**

---

*待续：*
*   *卷三：前端与运维 (Volume III: Frontend Experience & DevOps)* —— 最后一卷将把视角拉到用户端和运维端，详细阐述 React/Next.js 的状态管理哲学，以及 Kubernetes 的声明式部署架构。
