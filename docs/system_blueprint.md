# 🧭 系统蓝图总览

> 这一份文档现在分成两层：  
> **Part I**：当前代码已经实现/正在实现的 Phase 0–2 蓝图（方便对照 src 目录和测试）。  
> **Part II**：更完整的 Edge‑Detective v5.0 架构（长期“施工图纸”，不要求一次到位）。

---

## Part I：现有实现蓝图（Phase 0–2）

### 1. 当前核心类型（和代码对应）

| 概念 | 文件 | 作用 |
|------|------|------|
| `TrackRecord` | `src/core/perception.py` | YOLO+ByteTrack 输出的原始轨迹：`track_id, frames, bboxes, crops` |
| `VideoMetadata` | `src/core/perception.py` | `fps, width, height, total_frames`，用于时间/尺寸对齐 |
| `TrackFeatures` | `src/core/features.py` | 轨迹的 Level‑0 几何特征：`duration_s, path_length_px, avg_speed_px_s, max_speed_px_s` |
| `EvidencePackage` | `src/core/evidence.py` | 面向各层模型的证据包：`video_id, track_id, frames, bboxes, crops, fps, motion(TrackFeatures)` |
| `QueryResult` | `src/vlm_client.py` | VLM 对每条轨迹在某个问题下的判断结果：`track_id, start_s, end_s, score, reason` |

这些类型已经在代码里实现，是目前所有流程的数据基础。

---

### 2. Phase 0 —— 感知与轨迹索引

**一句话目标**：  
把一条“乱糟糟的原始视频”切成一批**干净的轨迹对象**，并能随时在原视频上高亮其中任何一条轨迹。

可以把 Phase 0 理解成：  
> 只做基础设施，不回答任何问题，只保证切得干净、时间对齐、能画框。

**组件与文件（按处理顺序）**

- `VideoPerception` (`src/core/perception.py`)  
  - 输入：`SystemConfig.video_path`（当前通常指 MOT17 合成的视频路径）。  
  - 输出：`Dict[int, TrackRecord]` + `VideoMetadata`。  
  - 负责：
    - 用 YOLOv11 逐帧检测人；  
    - 用 ByteTrack 把检测结果串成 `track_id`；  
    - 对每条轨迹采样若干代表性裁剪图，写入 `config.crops_dir`。  
  - 直观理解：它负责回答 “这条视频里有哪些人？各自在第几帧在哪个框？”。

- `TrackFeatureExtractor` (`src/core/features.py`)  
  - 输入：`track_records` + `VideoMetadata.fps`。  
  - 输出：`Dict[int, TrackFeatures]`。  
  - 负责：为每条轨迹算最基础的几何量：
    - `duration_s`：出现在视频里的总时间；  
    - `path_length_px`：中心点走过的总路程；  
    - `avg_speed_px_s`：平均速度；  
    - `max_speed_px_s`：最大瞬时速度。  
  - 这些都是“几何真相”，不牵涉任何“嫌疑 / 偷东西”这种语义。

- `build_evidence_packages` (`src/core/evidence.py`)  
  - 输入：`video_id, track_records, metadata, features`。  
  - 输出：`Dict[int, EvidencePackage]`。  
  - 负责把 raw 轨迹 + 几何特征打包成统一的 `EvidencePackage`，为后面 Recall/VLM/行为模块提供统一接口。

- `render_highlight_video` (`VideoPerception.render_highlight_video`)  
  - 输入：`track_records, metadata, target_ids, output_path`。  
  - 输出：高亮轨迹视频（优先 H.264 编码，失败时回退）。  
  - 用途：人工检查感知与索引是否靠谱（人有没有“丢框”“跳框”）。

**如何只测 Phase 0？**

- 跑单测：  
  - `tests/test_phase1_components.py::test_build_evidence_packages_computes_timings`  
    - 检查 EvidencePackage 的时间戳（起止秒）与运动特征是否正确。  
- 手动调用：

  ```python
  system = VideoSemanticSystem(config)
  system.build_index()  # 只跑 Phase 0
  # 再用 VideoPerception.render_highlight_video 看几个 track 的高亮视频
  ```

---

### 3. Phase 1 —— 单视频、人检索、问题驱动 QA

**一句话目标**：  
用户用一句自然语言描述“想找的人”，系统在这条视频里把所有人拿出来问一遍 VLM，最后告诉你：谁最像、在哪几秒、为什么。

可以把 Phase 1 想象成：  
> 在 Phase 0 那堆 EvidencePackage 上，先粗选，再让 VLM 做“逐人问询”。

**组件与文件**

- `RecallEngine` (`src/pipeline/recall.py`)  
  - v7 版本：`visual_filter(tracks, description, visual_tags, top_k)` 使用 SigLIP 向量相似度从所有轨迹中选出 Top‑K 候选；  
  - 同时保留 `recall(question, evidence_map, limit)` 作为兼容入口，内部调用 `visual_filter`。

- `QueryResult` (`src/core/vlm_types.py`)  
  - VLM 在某个问题下对单条轨迹的判断结果：`track_id, start_s, end_s, score, reason`。

- `Qwen3VL4BHFClient` (`src/pipeline/vlm_client_hf.py`)  
  - 使用 transformers 直接加载 `Qwen/Qwen3-VL-4B-Instruct`，在 Verifier 阶段对候选轨迹做 Yes/No 判定并返回 `QueryResult` 列表（未来可替换为 llama-cpp GGUF）。

- `VideoSemanticSystem` (`src/pipeline/video_semantic_search.py`)  
  - `build_index()`：Phase 0 的 orchestrator + 构建 `evidence_map`。  
  - `question_search(question, top_k=5, recall_limit=None)`：
    1. 调用 Router 生成 ExecutionPlan；  
    2. 用 `RecallEngine.visual_filter` 进行 SigLIP 粗筛；  
    3. 用 Hard Rule Engine 在 Atomic 8 空间做几何过滤与排序；  
    4. 将剩余候选交给 Qwen3‑VL‑4B（transformers）做终审，并按 score 排序，取前 `top_k`；  
    5. 打印结果 + 调 `render_highlight_video` 生成两个视频：结果视频（只框匹配轨迹）和全量轨迹视频（便于对比调试）。
- VLM 提示与判定（v1.27 实际做法）：
  - 均匀采样 crops + 小地图（轨迹打点图）输入；动作叙事层把速度/方向/左右位置翻译成人话。
  - Prompt 不再强制 JSON，要求末行输出 `MATCH: yes/no`，解析仅看该行，避免小模型 JSON 格式不稳导致误判。
- 方向/动作几何过滤（TODO，建议）：
  - 在 Hard Rule Engine 增加 `direction` 约束（基于 displacement_vec 或首尾 x 差），先过滤明显反向/静止，再交给 VLM，减少“往右”误报。

**单独测试入口**

- `tests/test_phase1_components.py::test_recall_engine_limit`  
  - 构造一个小的 `evidence_map`，检查 `limit` 为 1、2、None 时输出长度是否符合预期。  
- `tests/test_phase1_components.py::test_question_search_uses_stub_vlm`  
  - 用 StubVLMClient 替代真实 VLM，验证 `question_search` 是否正确调用 Recall 和渲染函数，而不依赖模型本身。

---

### 4. Phase 2 —— 行为特征与基础事件 (ROI 停留 / 跟随)

**一句话目标**：  
在不改变 Phase 1 主流程的前提下，为后续“徘徊、尾随”等行为问题准备好一组可复用的数学积木。

**新组件与文件**

- `BehaviorFeatureExtractor` (`src/core/behavior.py`)  
  - `compute_roi_dwell(tracks)`：  
    - 基于 `SystemConfig.roi_zones` 中定义的矩形区域（如门口、收银台），统计每条轨迹在各 ROI 内停留的时间（秒）；  
    - 输出 `Dict[int, List[RoiDwell]]`，每个 `RoiDwell` 记录 ROI 名称和停留时间。  
  - 这些输出以后可以直接支持 v6 里的 `spatial_op="stay"` 等规则。

- `EventDetector` (`src/core/behavior.py`)  
  - `detect_follow_events(tracks)`：  
    - 对所有轨迹对 (i, j) 计算在重叠时间段内的中心点距离；  
    - 按 `follow_distance_thresh` + `follow_min_frames` 找出“持续靠得很近”的片段；  
    - 输出 `List[FollowEvent(follower, target, start_s, end_s, min_distance)]`。  
  - 这为未来的问题 “谁在跟着某个人？” 提供几何基础。

- `SystemConfig` (`src/config.py`) 新增 Phase 2 钩子：  
  - `roi_zones: List[(name, (x1,y1,x2,y2))]`：预定义场景里的关键位置；  
  - `follow_distance_thresh: float`：多近算“跟着”；  
  - `follow_min_frames: int`：要连续多少帧才算真的尾随。

**单独测试入口**

- `tests/test_phase2_behavior.py::test_roi_dwell_counts_seconds`  
  - 用手工设计的小轨迹验证：在 ROI 内的帧数是否正确转成秒数。  
- `tests/test_phase2_behavior.py::test_follow_event_detection`  
  - 构造两个“明显在一起走”的轨迹，检查是否能检测出跟随事件。

> 目前 Phase 2 模块尚未接入 `VideoSemanticSystem` 主流程，保持高解耦：  
> 未来可以在 Hard Rule Engine 或更高层逻辑中按需组合使用这些行为结果。

---

## Part II：通往 v6 的修改计划（Blueprint v6.0 Migration）

> v6 的完整“施工图纸”已经单独放在 `docs/edge_detective_blueprint_v6.md`。  
> 这一部分只回答三个问题：  
> 1）从现在的 Phase 0–2 怎么一步一步长成 v6？  
> 2）每一步尽量保持高解耦、可单测？  
> 3）未来接 live / Redis / 多视频 时不会推翻现有设计？

### 1. 数据协议对齐：从当前结构走向 Atomic 8

目标：在不破坏现有逻辑和测试的前提下，让代码里的 `TrackFeatures` / `EvidencePackage` 渐进式靠近 v6 中的“Atomic 8 + EvidencePackage” 协议。

- 在 `src/core/features.py` 中扩展 `TrackFeatures`：
  - 增加时间与空间字段：`start_s, end_s, centroids, displacement_vec`；
  - 全部从现有 `TrackRecord.frames + bboxes + VideoMetadata.fps` 推导出来；
  - 保留原有字段 `duration_s, path_length_px, avg_speed_px_s, max_speed_px_s`，不删不改。
- 在 `src/core/evidence.py` 中扩展 `EvidencePackage`：
  - 增加 `meta: {video_id, fps, resolution}`；
  - 增加 `raw_trace`（等价于现在的 `bboxes`，先做别名即可）；
  - 增加 `embedding` 字段，初期固定为 `None`，等接入 SigLIP 后再真正写入向量；
  - 保持现有字段名不变，保证 Phase 1/2 的调用与测试全部继续通过。

> 检查点：  
> - 所有已有测试 (`test_phase1_components.py`, `test_phase2_behavior.py`) 仍然通过；  
> - 新增字段可以在单独的小测试里验证数值正确性（如 `centroids` 是否在 0–1 之间，`displacement_vec` 是否等于终点减起点）。

### 2. 行为特征 → Hard Rule Engine：把 Phase 2 变成 v6 的 Tier 0

目标：复用现有 `BehaviorFeatureExtractor` / `EventDetector`，对上抽象成 v6 里的 Hard Rule Engine 接口，而不是堆在业务代码里。

- 新增模块：`src/core/hard_rules.py`，提供统一入口：

  ```python
  def apply_hard_rules(
      tracks: List[EvidencePackage],
      rules: Dict
  ) -> List[EvidencePackage]:
      ...
  ```

- 实现思路：
  - 利用 `EvidencePackage.features` 中的 `centroids / start_s / end_s / duration_s / avg_speed_px_s`，实现基础算子：
    - ROI 相关：`enter / exit / stay / cross` 对应当前 ROI 停留逻辑；
    - 跟随相关：在内部复用 `EventDetector.detect_follow_events`；
    - 排序相关：`time_desc / speed_desc / duration_desc`。
  - `rules` 为字典（由 Router 生成），但 Hard Rule Engine 本身不依赖任何 LLM。
- 单测策略：
  - 在 `tests/` 下新增 `test_hard_rules.py`；
  - 用手工构造的 `EvidencePackage`（或把现有行为测试里的简单场景重用）来验证：
    - `spatial_op="enter"` 时是否只返回进过 ROI 的轨迹；
    - `sort_op="speed_desc"` 是否真的按平均速度排序；
    - `limit` 是否生效。

这样做之后，Phase 2 的“行为/事件”逻辑就自然升级为 v6 里的 Tier 0（会计师），同时保持和上层完全解耦。

### 3. RecallEngine 升级：给未来 SigLIP 预留视觉筛选层

目标：在不引入新模型的前提下，把现有 `RecallEngine` 的接口调整为 v6 设计的 `visual_filter` 风格，方便以后直接塞 SigLIP。

- 在 `src/pipeline/recall.py` 中：
  - 保留当前的 `RecallEngine.recall(question, evidence_map, limit)` 以兼容旧代码；
  - 新增一个更通用的函数/方法：

    ```python
    def visual_filter(
        tracks: List[EvidencePackage],
        tags: List[str],
        top_k: int = 20,
    ) -> List[EvidencePackage]:
        ...
    ```

  - 当前实现可以是：
    - 如果 `tags` 为空：直接返回前 `top_k` 条（或所有）；
    - 如果 `tags` 非空：先用非常简单的 stub（例如根据 `video_id/track_id` 做伪相似度），只保证接口跑通，等接入 SigLIP 后再换成真向量检索。
- 在 `VideoSemanticSystem.question_search` 中，可以逐步从：
  - 直接调用 `RecallEngine.recall(...)` 过渡到：
  - 先根据问题提炼 `tags`（手写规则也可以），再调用 `visual_filter(...)`。

> 关键是：现在就把“召回层 → 精排层”的结构钉死，哪怕召回层暂时是 no‑op，也不要把所有逻辑都堆进 VLM 里。

### 4. 预留 Router / Thinking 模型的位置，但暂时用手写规则代替

目标：不在当前机器上硬上 Qwen3‑4B‑Thinking，但把 Router 这一层的“接口和职责”先安好，等硬件/模型准备好后可以平滑替换。

- 新增模块：`src/router.py`：
  - 定义 `ExecutionPlan` 的 Python 结构（`dataclass` 或 TypedDict），字段对齐 `edge_detective_blueprint_v6.md` 里的 JSON Schema；
  - 实现一个简单版本：

    ```python
    def build_execution_plan(user_query: str) -> ExecutionPlan:
        """
        v0: 纯手写规则/if-else，把常见问题映射到 ExecutionPlan。
        v1: 替换为 Qwen3-4B-Thinking 调用 + parse_router_output。
        """
    ```

  - 同时实现 `parse_router_output(raw_output: str) -> Tuple[ExecutionPlan, str]` 的空壳/伪实现，用于未来接 Thinking 模型。
- 在 `src/pipeline/video_semantic_search.py` 中，逐步将“解析用户问题”的逻辑迁移到 Router：
  - 当前可以在 `question_search` 里直接调用 `build_execution_plan(question)`；
  - 对于简单场景（“找穿红衣服的人”、“最后一个进店的人”），手写规则即可覆盖。

> 这一层的重点是“把自然语言 → 视觉标签 + 硬规则 + 验证 prompt”这个拆解职责独立出来，而不是立刻引入新模型。

### 5. 为 live / Redis / 多视频 做扩展预留（只改边缘，不动核心）

目标：保证将来接入 live 流、Redis 缓存、向量库、多视频检索时，核心抽象（TrackRecord / EvidencePackage / ExecutionPlan / Hard Rules / VLMClient）不需要改动，只是换实现。

短期可以在文档层面定几个约束（代码逐步靠拢）：

- 所有与“存储/缓存”相关的逻辑集中在一个薄层里（例如将来新增 `storage.py` / `index_store.py`），而不是散落在 `src/pipeline/video_semantic_search.py` 中；
- live 视频入口只负责把连续帧切成“片段 + TrackRecord 流”，对下游暴露的仍然是同一个 `Dict[int, TrackRecord] + VideoMetadata` 接口；
- 将来引入 Redis / 向量库时：
  - 只是在 Recall/Router/HardRules 层面增加“从远程索引/缓存拿 EvidencePackage/embedding”的路径；
  - 不改变 EvidencePackage 的字段定义，也不改变 VLM 调用接口。

> 简单理解：  
> - v6 的大蓝图放在 `edge_detective_blueprint_v6.md`；  
> - `system_blueprint.md` 负责记录“当前做到哪一步、下一步要改哪些模块”；  
> - 每次大改动前，先更新这里的 Migration 小节，再去动代码和测试。
