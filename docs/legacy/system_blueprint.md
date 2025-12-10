# 🧭 当前系统总览（v2.1，面向读者的“怎么跑、怎么用”）

这份文档只讲**当前落地的能力与流程**，不谈远期架构。目标：让不看代码的人也能理解这个系统是做什么的、怎么跑、输出什么。

---

## 1. 关键数据结构（看这些就能理解流水线）

| 概念 | 文件 | 作用 |
|------|------|------|
| `TrackRecord` | `src/core/perception.py` | YOLO+ByteTrack 输出的原始轨迹：`track_id, frames, bboxes, crops` |
| `VideoMetadata` | `src/core/perception.py` | `fps, width, height, total_frames`，用于时间/尺寸对齐 |
| `TrackFeatures` | `src/core/features.py` | 轨迹的 Level‑0 几何特征：`duration_s, path_length_px, avg_speed_px_s, max_speed_px_s` |
| `EvidencePackage` | `src/core/evidence.py` | 面向各层模型的证据包：`video_id, track_id, frames, bboxes, crops, fps, motion(TrackFeatures)` |
| `QueryResult` | `src/vlm_client.py` | VLM 对每条轨迹在某个问题下的判断结果：`track_id, start_s, end_s, score, reason` |

---

## 2. 感知与索引（Phase 0）

**目标**：把原始视频切成一批轨迹对象，并能画框检查。

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

## 3. 问题驱动检索（Phase 1）

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
  - v2.1：质量优先采样（分段取大框帧），轨迹叠加到真实帧（黄线、绿→红打点、START/END），结构化 Prompt（外观/动线/几何事实/约束，末行 `MATCH: yes/no`），解析更鲁棒（MATCH 行 + yes/no/中文 fallback）。

- `VideoSemanticSystem` (`src/pipeline/video_semantic_search.py`)  
  - `build_index()`：Phase 0 的 orchestrator + 构建 `evidence_map`。  
  - `question_search(question, top_k=5, recall_limit=None)`：
    1. 调用 Router 生成 ExecutionPlan；  
    2. 用 `RecallEngine.visual_filter` 进行 SigLIP 粗筛；  
    3. 用 Hard Rule Engine 在 Atomic 8 空间做几何过滤与排序；  
    4. 将剩余候选交给 Qwen3‑VL‑4B（transformers）做终审，并按 score 排序，取前 `top_k`；  
    5. 打印结果 + 调 `render_highlight_video` 生成两个视频：结果视频（只框匹配轨迹）和全量轨迹视频（便于对比调试）。
- VLM 提示与判定（v2.1）：
  - 质量优先采样 + 轨迹叠加真实帧；结构化 Prompt（外观、轨迹覆盖、几何事实、约束），末行 `MATCH: yes/no`。
- 方向/动作几何过滤（TODO，建议）：
  - Hard Rule Engine 已支持 `direction`，基于轨迹分段投票的主方向，先过滤反向/静止再交给 VLM，减少误报。

**单独测试入口**

- `tests/test_phase1_components.py::test_recall_engine_limit`  
  - 构造一个小的 `evidence_map`，检查 `limit` 为 1、2、None 时输出长度是否符合预期。  
- `tests/test_phase1_components.py::test_question_search_uses_stub_vlm`  
  - 用 StubVLMClient 替代真实 VLM，验证 `question_search` 是否正确调用 Recall 和渲染函数，而不依赖模型本身。

#### 近期改进建议（v1.28 思路）
- 轨迹提示方式：参考 TraceVLA，将轨迹线/打点直接叠加在真实帧上（半透明、绿→红渐变、标注 START/END），而不是白底抽象图，便于 VLM 理解场景语义。
- Prompt 结构：分块告知（外观图、叠加轨迹图），将几何计算出的方向/速度/位置变化作为“已确认事实”，最后一行仍用 `MATCH: yes/no`。
- 方向几何过滤：Hard Rule Engine 已支持 `direction`，建议使用主运动方向过滤（可考虑分段票选，抵消“进店又出”导致的位移接近零）。
- 采样质量：在均匀采样基础上，可优先选 bbox 面积大的帧或前中后分段抽样，避免模糊/遮挡帧。
- 解析鲁棒：`MATCH` 行已启用，可再增加 yes/no/是/否 的 fallback，避免格式漂移。
- 评分机制：考虑将 SigLIP 相似度、硬规则符合度、VLM 置信度做加权，或候选池模式降低分而非直接删除，减少漏召回。

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

# 4. 调试与工具

# 5. 运行/调试速查

- 运行主流程：`cd repo && PYTHONPATH=src python -m pipeline.video_semantic_search`
- 轨迹高亮（含路径）调试视频：`python tests/render_tracks_with_paths.py` 输出 `output/tracks_with_paths*.mp4`
- 单轨迹动线图（给 VLM 的那类叠加图）：`python tests/motion_plot_demo.py`
- 查看数据库：`output/semantic_database.json`

---

# 6. 未来思路/提升方向

- Recall 评分：SigLIP 相似度 + 硬规则符合度 + VLM 置信度做加权，或候选池（软过滤）避免漏召回。
- 解释性输出：结构化 reason（视觉/运动证据、匹配/不匹配原因、置信度）而不止一段 VLM 文本。
- 更精细采样：结合清晰度/遮挡度评分。
- 方向判定：分段票选已上线，如需可扩展掉头检测。
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
