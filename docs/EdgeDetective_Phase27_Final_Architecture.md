# Edge‑Detective Phase 2.7：VLM Router + SigLIP Soft‑Rerank + Constraint‑Aware Dual Verifier
## Final Architecture Whitepaper (Phase 2.7)

> **文档性质**：系统架构设计白皮书（最终版）  
> **核心理念**：让非视频 VLM 具备“在视频中找人/找行为线索”的能力，同时把“物理/空间计算”交给确定性算法做拐杖  
> **写作风格**：沿用你现有 Phase 2.6 架构白皮书的结构与图示风格（漏斗、证据工场、Router、双轨验证、全景流转）  
> **本文不谈具体代码细节**：但会给出明确的数据契约、公式、模块边界，让你可以对照代码检查“是不是按这个实现的”。

---

## 一、核心设计哲学：四层漏斗 (The Funnel)

本系统的核心理念依然是 **“分级过滤，降本增效”**，但 Phase 2.7 把漏斗结构进一步“工程化”和“可解释化”：

- **上游不漏**：YOLO + ByteTrack 让每个人都进入档案库  
- **中游很便宜**：SigLIP/CLIP 只做相似度排序与候选压缩（不做误杀阈值）  
- **物理拐杖补弱项**：把 norm_speed / scale_change / linearity 等“模型不擅长的物理量”由系统计算，作为排序的第二信号与 Verifier 的事实注入  
- **下游高精度**：VLM 只审讯少量最有价值的候选（L1/L2 双轨）

```
                    ┌───────────────────────────────────────────────┐
                    │                                               │
     第一层         │   🔍 感知层 Perception (100%召回)               │
     Perception     │   YOLO + ByteTrack → TrackRecord               │
                    │   每个人 → 完整生命周期档案                     │
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                         ┌──────────────────────────────────┐
                         │                                  │
     第二层              │ ⚡ 外观召回 Recall (Soft‑Rerank)   │
     Recall              │ SigLIP: 排序 + 自适应候选压缩      │
                         │ 100% → 10~60 tracks（不硬阈值）   │
                         │                                  │
                         └──────────────────┬───────────────┘
                                            │
                                            ▼
                         ┌──────────────────────────────────┐
                         │                                  │
     第三层              │ 🧮 物理约束 Constraints (Crutch)  │
     Constraints         │ Atomic Facts → constraint_score   │
                         │ 只“轻微修正排序/提示”，不硬砍      │
                         │                                  │
                         └──────────────────┬───────────────┘
                                            │
                                            ▼
                              ┌────────────────────────┐
                              │                        │
     第四层                   │ 🎯 VLM Verification     │
     Verification             │ L1 Crop / L2 Filmstrip  │
                              │ 终审 + 可解释输出       │
                              │                        │
                              └────────────────────────┘
```

> 💡 **漏斗哲学的 Phase 2.7 版本**：  
> 让便宜的模型处理海量数据，让昂贵的模型只处理最有价值的候选；  
> 同时让“确定性算法”负责物理量与时空度量，VLM 负责语义理解与最终裁决。

---

## 二、证据工场：为每个人建立“数字档案” (Evidence Factory)

在任何大模型介入之前，我们必须把 **非结构化视频流** 变成 **结构化证据包**。  
证据工场是整个系统的地基：如果档案不稳定，后面所有智能都会“漂”。

### 2.1 全量抓拍：一个都不能漏

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│    📹 视频流                                                             │
│     │                                                                    │
│     ▼                                                                    │
│    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐   │
│    │   YOLO11n    │ ──────→ │  ByteTrack   │ ──────→ │  TrackRecord │   │
│    │   目标检测    │  检测框  │   多目标跟踪  │   轨迹   │   数字档案    │   │
│    └──────────────┘         └──────────────┘         └──────────────┘   │
│                                                                          │
│    设计原则：                                                            │
│    - 宁可误检，不可漏检（Perception 追求 100%召回）                     │
│    - 记录完整生命周期：每帧 bbox / frame_id / 时间戳                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 证件照选取：Best Crop + 多帧代表照（K=3）

Phase 2.6 你已经有 Best Crop Selection。Phase 2.7 继续保留，并新增一个关键点：

> 💡 **核心洞察（升级点）**：  
> 只用一张“最佳证件照”做相似度，会对遮挡/光照/朝向很敏感；  
> 所以我们为每条 track 预存 K 张代表照（默认 K=3：best/mid/end），用于 SigLIP 聚合打分。

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│    Track: 900 帧 bbox                                                    │
│     │                                                                    │
│     ▼                                                                    │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │  (A) Best Crop Selection                                        │   │
│    │      Score = area × (1 + 0.5 × centrality)                      │   │
│    │      - 面积越大越清晰                                            │   │
│    │      - 越居中越不易截断                                          │   │
│    └────────────────────────────────────────────────────────────────┘   │
│     │                                                                    │
│     ├────────────────────────────────────────────────────────────────┐   │
│     │                                                                │   │
│     │  (B) Multi‑Crop Bank (K=3)                                     │   │
│     │      - best: 最高分证件照                                      │   │
│     │      - mid : 轨迹中段（抗光照/姿态变化）                        │   │
│     │      - end : 轨迹末段（抗遮挡/回头等变化）                      │   │
│     │                                                                │   │
│     └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│    产出：                                                                │
│    - best_crop.jpg                                                      │
│    - crops_k/{best,mid,end}.jpg                                         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.3 物理特征计算：Atomic Facts 仪表盘（Phase 2.7 标准口径）

> 💡 **核心洞察**：  
> 4B VLM 在“速度/深度变化/轨迹度量”等方面很弱，容易 hallucinate。  
> 所以我们用确定性几何计算预先算好，作为 **crutch（拐杖）**：  
> - 中游用于约束评分（constraint_score）  
> - 下游注入给 VLM verifier（让它“读仪表盘”，不要瞎猜）

#### 2.3.1 原始序列定义（每帧）

bbox: (x1_t, y1_t, x2_t, y2_t), frame size: W×H, fps

- 中心点（像素）：
  - cx_t = (x1_t + x2_t)/2
  - cy_t = (y1_t + y2_t)/2
- 高度（像素）：
  - h_t = (y2_t - y1_t)

（可选：对 cx/cy/h 做 3~5 帧滑动均值；对极小 h 做 eps/clamp）

#### 2.3.2 norm_speed（身高/秒, heights/sec）

相邻帧中心点位移（像素）：
- dpx_t = sqrt((cx_t-cx_{t-1})^2 + (cy_t-cy_{t-1})^2)

逐帧归一化速度：
- speed_norm_t = (dpx_t / max(h_t, eps)) * fps

聚合（稳健）：
- norm_speed = median(speed_norm_t)

可选：
- norm_speed_p90 = P90(speed_norm_t) （捕捉短促冲刺）

#### 2.3.3 scale_change（靠近/远离）

使用 bbox 高度的首尾分段中位数（抗遮挡/抖动）：
- h_start = median(h_t in first 20% frames)
- h_end   = median(h_t in last  20% frames)
- scale_change = h_end / max(h_start, eps)

#### 2.3.4 linearity（轨迹线性度 0~1）

使用同单位坐标（像素或归一化都行，但必须一致）：

- D = ||c_last - c_first||
- L = sum_t ||c_t - c_{t-1}||
- linearity = D / max(L, eps)

#### 2.3.5 displacement_vec（首尾位移向量）

用归一化中心点（0~1）表达方向更直观：
- cxn_t = cx_t / W
- cyn_t = cy_t / H
- displacement_vec = (cxn_last - cxn_first, cyn_last - cyn_first)

---

### 2.4 EvidencePackage：系统的数据货币（统一数据契约）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                         EvidencePackage (Phase 2.7)                      │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  身份信息                                                       │   │
│   │  - video_id / track_id / video_path                              │   │
│   │                                                                 │   │
│   │  时空轨迹                                                       │   │
│   │  - frames[] / bboxes[] / fps                                     │   │
│   │                                                                 │   │
│   │  视觉证据                                                       │   │
│   │  - best_crop_uri                                                │   │
│   │  - crops_k_uris (K=3: best/mid/end)                              │   │
│   │                                                                 │   │
│   │  Embedding 资产（索引期预存）                                     │   │
│   │  - siglip_img_embeds[K]                                          │   │
│   │                                                                 │   │
│   │  Atomic Facts 仪表盘（索引期预存）                                │   │
│   │  - norm_speed / scale_change / linearity / displacement_vec ...  │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   💡 设计意图：                                                          │
│   - 这是系统的“数据货币”                                                 │
│   - 下游模块只消费 EvidencePackage，不关心上游怎么计算/怎么取图          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 三、Query 编译器：VLM Router → QuerySpec（Phase 2.7 关键升级）

Phase 2.6 的 Router 输出是 ExecutionPlan，包含 hard_rules 数值阈值并需要“约束清洗”。  
Phase 2.7 的 Router **彻底收敛职责**：

> Router 只做“语义编译”：  
> - 输出外观 prompts（给 SigLIP）  
> - 输出 need_context（决定 L2）  
> - 输出 constraint_intents（只输出意图标签，不输出任何阈值）

### 3.1 QuerySpec（Router 唯一输出、严格 JSON schema）

```json
{
  "positive_prompts": ["person wearing a blue shirt"],
  "negative_prompts": [],
  "need_context": false,
  "constraint_intents": []
}
```

字段语义：
- positive_prompts：1~5 条，<= 8 words/条，只写外观（color/clothes/accessory）
- negative_prompts：0~3 条，仅当 query 明确否定（not/without/excluding）时输出
- need_context：bool（涉及动作/方向/场景/交互 → true）
- constraint_intents：0~5 个标签（只从白名单选，不输出数字）

### 3.2 Router 决策树（need_context + intent 的来源）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                       VLM Router 语义编译决策树                          │
│                                                                          │
│  输入：英文 query                                                        │
│    │                                                                     │
│    ▼                                                                     │
│  1) 外观抽取 → positive_prompts（appearance-only）                        │
│     - color: blue/red/black...                                           │
│     - clothing: shirt/hoodie/jacket...                                   │
│     - accessory: backpack/hat/glasses...                                 │
│                                                                          │
│  2) need_context 判定                                                     │
│     - 有动作/时序：run/walk/leave/enter/approach/wander...  → true       │
│     - 有方向/关系：left/right/toward/near/beside/follow...   → true      │
│     - 纯外观：blue shirt / backpack / glasses               → false      │
│                                                                          │
│  3) constraint_intents（只输出标签，不输出阈值）                          │
│     - RUNNING / WALKING / STOPPED                                        │
│     - APPROACHING / LEAVING                                              │
│     - WANDERING (linearity low)                                          │
│     - MOVING_LEFT / MOVING_RIGHT (optional)                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3 为什么 Router 不能输出 hard_rules 数值阈值？
因为 VLM（尤其是小模型）会“过度解读”并 hallucinate 数值区间。  
Phase 2.7 的设计是：

> **数值阈值/公式永远由系统掌控（Constraint Engine），Router 只输出意图标签。**

这让“约束清洗”从一个复杂补丁，降级为简单 schema 校验（字段白名单+禁止阈值结构）。

---

## 四、外观召回：SigLIP Soft‑Rerank（不再阈值误杀）

Phase 2.6 的 CLIP 过滤有“红线阈值误杀”风险，并且默认关闭。  
Phase 2.7 的 SigLIP 召回层改成：

> **只做打分排序 + 自适应候选池压缩**（默认开启）  
> 不用硬阈值过滤，避免 recall loss。

### 4.1 SigLIP 相似度：cosine

- 对文本 prompts 编码：txt_embed[p]
- 对每条 track 的 K 张代表照预存 img_embed[i]

余弦相似度：
- sim[p,i] = cosine(txt_embed[p], img_embed[i])

### 4.2 track 级聚合（稳健：top‑m mean）

对每条 prompt：
- score_prompt = mean(top2(sim[p,:]))   （m=2 默认）

对多 prompt：
- siglip_score(track) = max(score_prompt over prompts)

（可选 negative_prompts）
- siglip_score = pos_score − λ * neg_score

> 💡 为什么是 top2 mean：  
> - 只取 max 太敏感（偶然一帧光照正好会“虚高”）  
> - 取 mean 太钝（被坏帧稀释）  
> - top2 mean 是兼顾稳定与尖锐度的折中

### 4.3 候选池选择：margin 自适应 + clamp

不固定 topK，而是用分数分布自适应：

1) best = max(siglip_score)
2) keep all tracks with score >= best − Δ
3) clamp 到 [K_min, K_max]

默认：
- Δ = 0.08
- K_min = 10
- K_max = 60

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│      自适应候选池（避免“固定 topK”过松/过紧）                            │
│                                                                          │
│   score >= best - Δ  → 都保留                                             │
│   太少 → 补到 K_min                                                      │
│   太多 → 截到 K_max                                                      │
│                                                                          │
│   结果：候选数量随“任务难度/分数尖锐度”自动调整                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 五、物理约束：Constraint Engine（hard_rules 的 Phase 2.7 正确形态）

这一层的核心是：把你自己算出来的 Atomic Facts 变成可控的“约束信号”，同时避免一棍子打死。

### 5.1 输入与输出

输入：
- constraint_intents（来自 Router）
- atomic_facts（来自 EvidencePackage）

输出：
- constraint_score(track) ∈ [0,1]
- breakdown（用于日志与可解释性）

### 5.2 soft scoring：用 sigmoid 代替硬阈值

定义 sigmoid：
- σ(x) = 1 / (1 + exp(-x))

#### RUNNING（中心 1.8）
- score_run = σ((norm_speed − 1.8) / 0.2)

#### APPROACHING（中心 1.2）
- score_app = σ((scale_change − 1.2) / 0.1)

#### WANDERING（线性度低）
- score_wand = σ((0.3 − linearity) / 0.1)

#### STOPPED（低速）
- score_stop = σ((0.4 − norm_speed) / 0.1)

多 intent 合并（AND 语义）：
- constraint_score = min(score_intent_1, score_intent_2, ...)

> 💡 设计原则：  
> - intent 为空 → constraint_score = 0（完全不影响纯外观查询）  
> - 约束只“轻微修正排序”，不做硬过滤（避免 recall 崩）

### 5.3 与 SigLIP 融合排序（Stage2.5）

最终排序分：
- final_rank = 1.0*siglip_score + 0.2*constraint_score + 0.05*quality

---

## 六、核心审讯：VLM 双轨验证（L1 Crop / L2 Filmstrip）

### 6.1 双轨设计哲学

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│     外观类问题：需要高清细节 → Layer 1 Crop                              │
│     行为/关系/场景类问题：需要时序+背景 → Layer 2 Filmstrip              │
│                                                                          │
│     Router 输出 need_context:                                             │
│     - false → 只跑 L1                                                    │
│     - true  → 跑 L2（可选同时跑 L1 取外观细节）                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Layer 1：Crop Mode（外观特写）
输入：
- best_crop（或高分 crop/panel）

输出（建议固定 JSON）：
- match: true/false
- reason: string
- confidence: float

### 6.3 Layer 2：Filmstrip Context Mode（全景时序）
输入三件套：
1) reference crop（通缉令：告诉模型要找谁）
2) filmstrip（多帧拼接 + 红框 burn‑in）
3) telemetry facts（Atomic Facts + constraint intents）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   Layer 2 输入组合拳：                                                    │
│                                                                          │
│   ① Reference Crop：这个人长这样                                         │
│   ② Filmstrip：t0|t1|t2|t3|t4 并排（红框锁定同一人）                     │
│   ③ Facts：norm_speed/scale_change/linearity/...                         │
│                                                                          │
│   💡 关键要求：                                                          │
│   - Facts 是“拐杖”：模型必须使用系统给定事实，不要自己估速度/阈值        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 七、系统流转全景图（离线索引 vs 在线检索）

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                     Edge‑Detective Phase 2.7 完整流程                            │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  离线阶段：建立档案库 (Index Once, Query Many)                              │  │
│  │                                                                            │  │
│  │  📹 video → YOLO → ByteTrack → EvidenceFactory                             │  │
│  │                       ├─ best_crop + K-crops                              │  │
│  │                       ├─ SigLIP image embeddings (precompute)             │  │
│  │                       └─ Atomic Facts (precompute)                         │  │
│  │                                                                            │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                         │
│                                          ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  在线阶段：Query‑Driven Search                                              │  │
│  │                                                                            │  │
│  │  English Query                                                             │  │
│  │    │                                                                       │  │
│  │    ▼                                                                       │  │
│  │  VLM Router (text-only) → QuerySpec {prompts, need_context, intents}       │  │
│  │    │                                                                       │  │
│  │    ▼                                                                       │  │
│  │  SigLIP Soft‑Rerank (appearance) → siglip_score(track)                     │  │
│  │    │                                                                       │  │
│  │    ▼                                                                       │  │
│  │  Constraint Engine (atomic facts) → constraint_score(track)                │  │
│  │    │                                                                       │  │
│  │    ▼                                                                       │  │
│  │  final_rank fusion + margin clamp → candidate tracks (10~60)               │  │
│  │    │                                                                       │  │
│  │    ▼                                                                       │  │
│  │  VLM Verifier: L1 crop, L2 filmstrip if need_context=true                  │  │
│  │                                                                            │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 八、可观察性与可回放（为了让你对照代码验证“有没有按架构实现”）

必须记录的日志（每次 query）：

- QuerySpec（Router 输出）  
- SigLIP 排序表（top30：track_id, siglip_score, rank）  
- Constraint breakdown（intent→score）  
- final_rank 表（top30：siglip + 0.2*constraint + 0.05*quality）  
- 候选池大小（margin 后数量）  
- Verifier 输出（L1/L2 的 JSON + confidence）  

> 💡 你实现完后，只要能把这些信息打出来，你就能非常快速判断：  
> - “Router 有没有乱产字段？”  
> - “SigLIP 是否真的在工作？”  
> - “约束是不是在帮倒忙？”  
> - “为什么这个 track 被排到前面/后面？”

---

## 九、模块边界（对照代码时最容易核对的点）

（不写具体实现，只写“应该在哪里发生什么”）

- EvidenceFactory：TrackRecord → EvidencePackage（crops + embeddings + atomic facts）
- Router：English query → QuerySpec（strict JSON）
- Recall：SigLIP soft rerank（不阈值过滤）
- Constraints：intent + facts → constraint_score（soft）
- Verifier：L1/L2 双轨 + facts 注入（只在 need_context/intents 存在时注入）

---

## 十、设计决策总结（Phase 2.7）

| 决策点 | Phase 2.7 选择 | 设计理由 |
|------|----------------|---------|
| Router 输出 | QuerySpec + constraint_intents | VLM 负责语义编译，系统负责数值执行 |
| 外观召回 | SigLIP soft rerank + margin 选候选 | 不误杀、延迟更稳定 |
| 物理约束 | Atomic Facts → soft constraint_score | 作为拐杖补模型弱项，不一刀切 |
| VLM 验证 | L1 Crop / L2 Filmstrip 双轨 | 证据匹配问题类型，节省成本 |
| 时序理解 | Filmstrip | 让非视频模型理解运动 |
| 可解释性 | breakdown + 日志 | 方便调参和定位 badcase |

---

## 十一、技术栈（建议）

- Perception：YOLO11n + ByteTrack
- Recall：SigLIP（text encoder online / image embeddings offline）
- VLM：Qwen3‑VL‑4B（Router + Verifier）
- Serving：vLLM（L4 上可部署）
- Core：opencv/numpy + 结构化证据包

---

> ✅ **一句话总结**：  
> Phase 2.7 把系统变成“可控的编译器 + 可解释的打分器 + 双轨审讯器”：  
> VLM 只做它最擅长的语义编译与最终裁决，  
> 物理量与时空度量交给确定性算法，  
> SigLIP 负责把大海量候选压缩到 VLM 能承受的规模。
