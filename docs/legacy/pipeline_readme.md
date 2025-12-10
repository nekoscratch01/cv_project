# Edge-Detective 流水线白皮书（Phase 1，详细版）

本文面向架构师，按“工厂流水线”视角，完整拆解当前 Phase 1 的视频语义检索流程：原料是视频，成品是“哪些轨迹匹配问题 + 解释 + 可视化”。同时列出关键模块、数据形态、运行指引与升级待办，便于后续演进。

---

## 生产线总览（两大阶段）

1) **建索引（离线预处理）**：把原始视频加工成结构化的人物档案，准备好后续可重复查询。  
2) **问题检索（在线问答）**：接收自然语言问题，在档案库中筛选匹配的人，并给出理由和可视化。

这两段流水线彼此解耦：索引只需跑一次，查询可重复多次。

---

## 阶段一：建索引车间群

### 1. 感知车间（Detection + Tracking）
- **职责**：将视频切分出单个行人轨迹。  
- **工序**：YOLOv11 检测 + ByteTrack 跟踪。  
- **输出**：  
  - `TrackRecord`：帧序列、检测框序列、裁剪图路径。  
  - `VideoMetadata`：fps、分辨率、总帧数等。  
- **代码**：`core.perception.VideoPerception`

### 2. 特征车间（Atomic-8 运动体检）
- **职责**：把轨迹转成几何/运动“原子事实”。  
- **Atomic-8 八项指标**：  
  1) `start_s` 起始时间  
  2) `end_s` 结束时间  
  3) `duration_s` 时长  
  4) `centroids` 中心点序列（归一化轨迹）  
  5) `displacement_vec` 首尾位移向量  
  6) `avg_speed_px_s` 平均速度  
  7) `max_speed_px_s` 最大速度  
  8) `path_length_px` 轨迹总路径长度  
- **输出**：`TrackFeatures`（纯几何特征，不带语义标签）。  
- **代码**：`core.features.TrackFeatureExtractor`

### 3. 证据装配车间（Evidence Packaging）
- **职责**：把轨迹、裁剪图、Atomic-8、元数据打包成统一“人物档案”。  
- **输出**：`EvidencePackage` 字典，落盘到 `output/semantic_database.json`，便于恢复与调试。  
- **代码**：`core.evidence.build_evidence_packages`

---

## 阶段二：问题检索车间群

### 4. 路由车间（Router / Plan）
- **职责**：把自然语言问题转成执行计划。Phase 1 用简版 `SimpleRouter`，仅回显描述；后续可换成真正的语义规划器（解析服饰/物品/动作/ROI 约束）。  
- **输出**：`ExecutionPlan`（描述 + 约束占位）。  
- **代码**：`pipeline.router.SimpleRouter`

### 5. 召回车间（Recall）
- **职责**：快速粗筛，减少后端推理压力。  
- **工序**：SigLIP 图文向量相似度，过滤掉明显无关的轨迹。  
- **输出**：候选 `EvidencePackage` 列表。  
- **代码**：`pipeline.recall.RecallEngine`

### 6. 硬规则车间（Hard Rules）
- **职责**：在几何/时间维度做二次过滤，如时长阈值、位移方向、区域约束等。  
- **输出**：进一步收敛的候选列表。  
- **代码**：`core.hard_rules.HardRuleEngine`

### 7. VLM 终检车间（VLM Verification）
- **职责**：用多模态大模型逐个候选做最终判定。  
- **模型**：vLLM + Qwen3-VL-4B（OpenAI 兼容接口）。  
- **输入**：  
  - **视觉**：候选轨迹的裁剪图，均匀采样，最多 5 张（Base64）。  
  - **文本（视觉 prompt）**：问题 + Atomic-8 生成的运动摘要 + 约束。示例：  
    ```
    ## Task
    Verify if this person matches the query: "找穿蓝色衣服的人"

    ## Evidence
    ### Appearance
    The images show the person at different moments in the video.

    ### Motion Summary
    Walking at normal pace. Moving right. Duration: 5.3s.

    ### Constraints
    No additional constraints.

    ## Instructions
    1. Describe what you see in the images.
    2. Check if the person matches the query criteria.
    3. Final line must be: MATCH: yes or MATCH: no
    ```
- **解析/反腐败层**：`VlmResponseParser` 从自然语言解析出 `VerificationResult`（status + confidence + reason + raw）。  
- **代码**：  
  - 端口协议：`ports.inference_port.InferencePort`  
  - 适配器：`adapters.inference.vllm_adapter.VllmAdapter`  
  - 值对象与解析：`domain.value_objects.verification_result`

### 8. 分拣车间（Select）
- **职责**：按置信度排序，截取 Top-K。  
- **输出**：最终匹配的轨迹列表（track_id、时间区间、分数、理由）。

### 9. 出货车间（Visualization）
- **职责**：生成可视化成果，方便人工复核与对外交付。  
- **产出**：两段 MP4：  
  1) `tracking_<question>.mp4` —— 匹配轨迹高亮  
  2) `tracking_all_tracks_<question>.mp4` —— 全轨迹调试版（带 ByteTrack ID）  
- **代码**：`core.perception.VideoPerception.render_highlight_video`（在 orchestrator 内被调用）

---

## 关键工位与文件映射

- 总控/入口：`src/pipeline/video_semantic_search.py` (`VideoSemanticSystem`)
- 配置班长：`src/core/config.py` (`SystemConfig`) —— 视频路径、输出目录、vLLM 端点/模型名等
- 路由：`src/pipeline/router.py` (`SimpleRouter`, `ExecutionPlan`)
- 召回：`src/pipeline/recall.py` (`RecallEngine`)
- 硬规则：`src/core/hard_rules.py` (`HardRuleEngine`)
- VLM 端口：`src/ports/inference_port.py`（`verify_track/verify_batch` 协议）
- VLM 适配器：`src/adapters/inference/vllm_adapter.py`
- 反腐败层/值对象：`src/domain/value_objects/verification_result.py`（`VerificationResult`, `VlmResponseParser`）
- 证据装配：`src/core/evidence.py`（`build_evidence_packages`）

---

## 工厂 SOP（如何跑通 Phase 1 Demo）

1) **依赖**：`pip install -r requirements.txt`（flash-attn 可选；默认 SDPA）。  
2) **启动 vLLM**：`./deploy/start_vllm.sh`（Qwen3-VL-4B，端口 8000，已禁用 flash-attn，后续可手动开启）。  
3) **运行示例**：`./run_phase1_demo.sh` 或顺序执行 `phase1_demo.ipynb`。  
   - 视频：`data/raw/semantic/MOT17-12.mp4`  
   - 查询：`找穿蓝色衣服的人`  
   - 产出：`output/demo_run/` 下两段 MP4（匹配高亮 / 全轨迹调试）。

---

## 现状与升级待办（进入下阶段前的差距）

1) **路由智能度低**：当前仅回显描述，缺少服饰/物品/动作/ROI 等约束解析，需引入轻量 LLM Router 或规则解析。  
2) **服务化缺位**：未挂 FastAPI/Celery/健康检查/结构化日志/指标，尚不能对外提供稳态 API。  
3) **推理弹性不足**：vLLM 单端点，无健康探测、A/B 或弹性策略；可后续补多实例路由和监控。  
4) **可运维性**：日志/指标/告警链路缺失，难以在生产观测与调优。  
5) **扩展性**：暂无量化/云端兜底适配；仅硬绑 vLLM，本阶段可接受，后续可加模型注册表与策略路由。

---

## 作为架构师应关注的要点

- **数据形态清晰**：从 `TrackRecord` → `TrackFeatures (Atomic-8)` → `EvidencePackage`，每一步都可落盘调试。  
- **分层解耦**：端口（Ports）隔离业务与推理实现；适配器（Adapters）可替换 vLLM 后端；反腐败层保证外部模型输出被结构化。  
- **异步与并发**：VLM 适配器是异步的，批量走并发请求（vLLM 内部连续批处理），保障吞吐。  
- **可视化闭环**：每次检索都生成结果视频与全轨迹调试视频，方便人工复核和快速迭代。  
- **升级路径**：从最小可用的 CLI/Notebook 流水线起步，逐步补路由智能、服务化、监控与弹性推理。
