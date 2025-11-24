# Stage 1 项目快照（现状总览）

> 目标：记录当前已落地的能力、框架和原理，方便快速对齐“我们现在有哪些东西、怎么跑、架构长什么样”。

## 现有能力（按链路）
- 感知 & 轨迹：YOLOv11（人检测）+ ByteTrack（多目标跟踪），在 `VideoPerception.process` (`src/core/perception.py`) 输出 `TrackRecord` 字典和 `VideoMetadata`，并保存轨迹裁剪图。
- 几何特征：`TrackFeatureExtractor` (`src/core/features.py`) 计算 Atomic 8 风格字段：`start_s/end_s/centroids/displacement_vec/duration_s/path_length_px/avg_speed_px_s/max_speed_px_s`。
- 证据打包：`build_evidence_packages` (`src/core/evidence.py`) 生成 `EvidencePackage`（含 meta/raw_trace/embedding 预留），作为下游统一输入。
- 召回：`RecallEngine.visual_filter` (`src/pipeline/recall.py`) 使用 SigLIP (`src/core/siglip_client.py`) 计算图文向量，相似度排序并缓存到 `embedding_cache_dir`。
- Hard Rules：`HardRuleEngine` (`src/core/hard_rules.py`) 在 Atomic 8 空间做 ROI 进入/停留、时间窗、阈值、排序、limit 等几何过滤。
- Router（规划）：`build_execution_plan` (`src/pipeline/router.py`) + `HFRouter` (`src/pipeline/router_llm.py`)，默认用 transformers 加载 Qwen3-VL-4B 生成 `ExecutionPlan`（含 tags/constraints/prompts）。
- Verifier（终审）：`Qwen3VL4BHFClient` (`src/pipeline/vlm_client_hf.py`) 读取候选的 crops + 摘要提示，给出 `QueryResult(track_id, start_s, end_s, score, reason)` 列表。
- Orchestrator：`VideoSemanticSystem` (`src/pipeline/video_semantic_search.py`) 负责 `build_index()`（Phase 0）与 `question_search()`（Router → Recall → Hard Rules → Verifier），并输出高亮视频。

## 核心协议与数据结构
- `TrackRecord` (`src/core/perception.py`)：`track_id, frames, bboxes, crops`。
- `VideoMetadata` (`src/core/perception.py`)：`fps, width, height, total_frames`。
- `TrackFeatures` (`src/core/features.py`)：Atomic 8 几何字段，含时间、中心点、位移和速度。
- `EvidencePackage` (`src/core/evidence.py`)：`video_id, track_id, frames, bboxes, crops, fps, features, meta{video_id,fps,resolution}, raw_trace, embedding`。
- `QueryResult` (`src/core/vlm_types.py`)：`track_id, start_s, end_s, score, reason`。
- `ExecutionPlan` (`src/pipeline/router.py`)：`tags, constraints, verifier_prompt` 等（Router 产物，供召回与 Hard Rules 使用）。

## Stage1 设计要点（近期动作）
- 优先级：1) VLM 输入优化 2) Router/Recall 策略 3) 配置重设 4) 验证闭环。
- VLM 输入优化（`src/pipeline/vlm_client_hf.py`）
  - Crop 采样：避免“前几帧陷阱”，对 `package.crops` 做均匀采样或跳过前 10%，限定 `self.max_crops`。
  - Prompt：三步结构（描述外观→对比约束→结论），输出 JSON 保留 `thinking/match/reason`，继续使用解析兜底。
  - 上下文翻译层：用现有几何数据生成标签喂给 VLM，而非裸数字——速度归一化后映射为 standing/walking/running/sprinting；bbox 中心映射 left/center/right；位移向量映射 moving left/right/up/down；面积占比映射 close/medium/far；可选徘徊度用于 wandering 描述。
- Router / Recall
  - Router Prompt 保持 JSON 要求并强调“仅 JSON、无前后缀”，继续保留正则兜底；约束/标签直传 Hard Rules 与 VLM。
  - Recall 调试模式允许 `recall_limit=None`；默认建议 20–50，保留 top_k 设置。必要时在 log 打印候选数与被过滤原因。
- 配置推荐值（`src/core/config.py`）
  - `min_track_length`: 15–30；`sample_interval`: 5–10；`yolo_conf`: 0.45–0.5。
  - `vlm_max_new_tokens`: 1024；`vlm_context_size`: 8192（显存足够时）；`max_preview_tracks`: 10。
  - `recall_limit` 默认 20–50，调试时设 None；其他参数保持可调但默认聚焦稳健性。
- 验证步骤
  - 单测：保持现有 `tests/test_phase1_components.py` 与 `tests/test_phase2_behavior.py` 必须全绿。
  - 手动：跑两类问题（颜色/动作、位置/方向），检查 log 中召回数量、VLM 原始输出、最终高亮视频；确认翻译层标签已出现在 VLM prompt 中。

## 模型与依赖
- 检测/跟踪：ultralytics YOLOv11、人类目标；ByteTrack 在同一文件完成关联。
- 召回：SigLIP `google/siglip-base-patch16-224`（向量检索与本地缓存）。
- Router & Verifier：Qwen/Qwen3-VL-4B-Instruct（transformers，默认 MPS/CPU）。
- 其他：OpenCV（I/O、渲染）、NumPy（几何计算）、PyTorch（模型推理）。

## 运行方式（单视频离线 Demo）
```bash
conda activate mvsys-py311  # 或你自己的环境
export PYTHONPATH=src
python -m pipeline.video_semantic_search
```
- 配置入口：`src/core/config.py`（视频路径、输出目录、模型名、设备、ROI 等）。
- 产物：`output/semantic_database.json`、`output/crops/`、`output/embeddings/<video_id>/track_*.npy`、`output/tracking_<question>.mp4`。

## 测试与验证
- 单测：`tests/test_phase1_components.py`（召回 limit、EvidencePackage 时间戳）、`tests/test_phase2_behavior.py`（ROI 停留、跟随事件）。
- 手动：`build_index()` 后用 `render_highlight_video` 抽查轨迹；`question_search("找穿红衣服的人")` 查看 Router → Recall → Hard Rules → Verifier 执行日志与渲染视频。

## 已知边界
- 场景：单视频、离线；未接 live/多视频/Redis。
- 性能：召回使用 SigLIP 逐轨迹编码，轨迹数多时需注意缓存命中；VLM 为 4B 级别，QPS 受硬件限制。
- 规则覆盖：Hard Rules 支持 ROI/时间窗/排序/速度跳变等基础几何规则，复杂语义仍依赖 VLM。
