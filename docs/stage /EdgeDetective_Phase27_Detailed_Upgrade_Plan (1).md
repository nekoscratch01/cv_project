# Edge‑Detective Phase 2.7 详细升级方案（VLM Router → QuerySpec → SigLIP Rerank → Constraints → Verifier）
> 目标：在你 Phase 2.6 的三层漏斗基础上升级（Perception → Recall → Verification）fileciteturn4file6L19-L46  
> 重点：把 hard_rules 从“VLM 生成阈值”改成“系统基于 Atomic Facts 自己算的拐杖”，并且让它同时服务 **排序** 和 **Verifier 提示**。

---

## 0. 你现有体系（Phase 2.6）里 hard_rules / Atomic Facts 到底是什么

### 0.1 你文档里的 Atomic 8（你现在想“救回来”的那部分）
在 `docs/architecture.md` 里，Atomic 8 的核心是：  
- 时空：`start_s, end_s, duration_s`  
- 几何：`centroids(0~1), displacement_vec`  
- 运动语义：`norm_speed(身高/秒), linearity(0~1), scale_change(尺度比)`，并给了阈值语义解释（跑步>1.8、靠近>1.2、徘徊<0.3 等）fileciteturn4file9L10-L20

### 0.2 你现有代码/旧白皮书里 TrackFeatureExtractor 目前算的还是“Level‑0”
在 `docs/legacy/pipeline_readme.md` 里，Atomic-8（Phase1）列的是 `avg_speed_px_s, max_speed_px_s, path_length_px` 等几何量fileciteturn4file2L26-L38，这说明：
- **你 Phase 2.6 文档里的 norm_speed/linearity/scale_change 可能还没完全成为“统一实现标准”**
- Phase 2.7 需要把这部分 **标准化成一个真正的可复用模块**（并且写进 README/TODO）

### 0.3 你当前 router_vlm.py 的 hard_rules 痛点（你自己也写了清洗）
当前 `src/pipeline/router_vlm.py` 的 system prompt 里，确实要求 VLM 输出 hard_rules（norm_speed/linearity/scale_change）并给了阈值示例fileciteturn4file0L22-L29。  
同时代码里做了“无动作词就清空约束、并移除 0.8–1.2 这种常见幻觉尺度约束”fileciteturn4file1L9-L25。  
这就是你现在“hard_rules 像狗皮膏药”的根因：**Router 既想输出约束，又怕幻觉，只能不断打补丁**。

---

## 1. Phase 2.7 的最终目标架构（你想要的链路 + hard_rules 正确位置）

你确认的主链路是：

**Query(text-only) → VLM Router → QuerySpec(prompts + need_context + constraint_intents) → SigLIP rerank →（插入 hard_rules / constraints）→ margin → VLM verifier(L1/L2)**

### hard_rules 在 Phase 2.7 的“正确位置”
**SigLIP 后、Verifier 前（用于排序/筛选） + Verifier prompt 里（用于拐杖提示）**。  
这会把 hard_rules 从“VLM 生成阈值”变成“系统用 Atomic Facts 计算出来的信号”。

---

## 2. 关键设计：hard_rules 不再是“VLM 输出阈值”，而是“系统计算的约束信号”

### 2.1 Router 只输出 3 类东西（全部可校验、不会胡编数值）
**(A) prompts（给 SigLIP 用）**：appearance-only 短句  
**(B) need_context（bool）**：决定是否跑 Layer2（filmstrip）  
**(C) constraint_intents（标签）**：告诉系统“我需要哪种物理拐杖”，但**不含任何数值阈值**

> 这相当于保留了你文档里“约束清洗防幻觉”的意图fileciteturn4file1L9-L25，但把“防幻觉”从事后补丁升级为“源头不让它输出数字”。

### 2.2 Constraint Engine（新模块）做两件事
1) **把 intent 映射到 Atomic Facts**（norm_speed/scale_change/linearity/dir 等）  
2) 输出一个 **constraint_score（0~1）**（软评分，默认不一刀切）

> 你文档里说“只存储纯几何量，不存语义标签；语义在原子事实之上推导”fileciteturn4file9L10-L14。  
> Phase 2.7 就是把“推导”显式化为一个可测试的模块。

---

## 3. 先把 Atomic Facts 的计算过程“定标准”（你说你忘了当时怎么算）

下面是 Phase 2.7 建议的 **标准定义**（工程同学照做即可）。  
输入：每个 track 的 bboxes 序列 + fps + frame_w/h。

### 3.1 统一预处理（所有指标都依赖）
对每一帧 bbox `(x1,y1,x2,y2)`：
- `cx = (x1+x2)/2`, `cy = (y1+y2)/2`
- `h = (y2-y1)`
- 归一化：`cxn=cx/W`, `cyn=cy/H`, `hn=h/H`

可选但强烈建议：
- 对 `cxn,cyn,hn` 做 3~5 帧滑动均值（抗 ByteTrack jitter）
- 丢弃 `hn` 极小的帧（远处噪声大，速度会爆）

### 3.2 norm_speed（身高/秒，匹配你文档阈值）
你文档的阈值语义：`>1.8` 跑步，`0.1~1.8` 行走，`<0.1` 静止fileciteturn4file9L39-L44。  
推荐计算（稳定、可解释）：

对相邻帧：
- `dpx = sqrt((cx_t-cx_{t-1})^2 + (cy_t-cy_{t-1})^2)`（像素）
- `speed_norm_t = (dpx / max(h_t, eps)) * fps`

聚合：
- `norm_speed = median(speed_norm_t)`（稳）
- 可选：`norm_speed_p90`（抓短促冲刺）

### 3.3 scale_change（靠近/远离）
你文档阈值：`>1.2` 靠近，`<0.8` 远离fileciteturn4file9L4-L6。  
推荐计算（抗噪）：
- `h_start = median(hn in first 20% frames)`
- `h_end   = median(hn in last  20% frames)`
- `scale_change = h_end / max(h_start, eps)`

可选增强（更鲁棒）：
- `scale_slope = slope( log(hn) vs time )`

### 3.4 linearity（徘徊/直线通过）
你文档阈值：`<0.3` 徘徊，`>0.7` 直线通过fileciteturn4file9L45-L47。  
推荐最简单可靠定义：
- `D = ||c_last - c_first||`
- `L = sum ||c_t - c_{t-1}||`
- `linearity = D / max(L, eps)` （0~1）

### 3.5 displacement_vec / direction（可选但你后面会很需要）
你文档已有 `displacement_vec` 用来描述“从左到右走的人”fileciteturn4file9L35-L38。  
建议计算：
- `dx = cxn_last - cxn_first`
- `dy = cyn_last - cyn_first`
- `displacement_vec = (dx, dy)`

可选：主方向票选（解决“进店又出”导致 dx≈0）
- 把轨迹分成 3 段，分别算 dx，取多数方向

---

## 4. VLM Router（text-only）改造：ExecutionPlan → QuerySpec（保留 need_context，但干掉数字 hard_rules）

### 4.1 现状（你现在 router_vlm.py）
它会让 VLM 输出 hard_rules 数值（并用代码清洗）fileciteturn4file0L22-L29 fileciteturn4file1L9-L25。

### 4.2 Phase 2.7 目标：Router 输出“意图标签”而不是阈值
**新 QuerySpec schema**（Router 必须且只输出这些字段）：

```json
{
  "positive_prompts": ["person wearing a blue shirt"],
  "negative_prompts": [],
  "need_context": false,
  "constraint_intents": []
}
```

**constraint_intents 白名单**（起步版）
- RUNNING, WALKING, STOPPED
- APPROACHING, LEAVING
- WANDERING（对应 linearity 低）
- MOVING_LEFT, MOVING_RIGHT（可选，若你愿意引入方向）

### 4.3 Router system prompt（关键：禁止数字、禁止把动作写进 prompts）
- prompts 只能写 appearance（color/clothes/accessories）
- need_context bool：包含环境/方向/出入/交互就 true，否则 false
- constraint_intents：只选标签，不输出 min/max，不输出公式

> 你之前的“约束清洗防幻觉”可以彻底简化：  
> 只保留 schema 校验（字段白名单 + 禁止出现数字/符号），不再做复杂清洗。

---

## 5. Stage2：SigLIP 从“红线阈值过滤”升级为“soft rerank + margin 选候选”
你文档里强调红线阈值不能太高、宁可多放进 VLM 也别误杀fileciteturn4file10L24-L25。  
Phase 2.7 的改造目标就是：**不硬删，只排序压缩**。

---

## 6. Constraint Engine：把 constraint_intents → constraint_score（0~1），并融合排序

### 6.1 评分函数（推荐 sigmoid，比硬阈值鲁棒）
- RUNNING：`score_run = sigmoid((norm_speed - 1.8)/0.2)`fileciteturn4file9L39-L44
- APPROACHING：`score_app = sigmoid((scale_change - 1.2)/0.1)`fileciteturn4file9L4-L6
- WANDERING：`score_wand = sigmoid((0.3 - linearity)/0.1)`fileciteturn4file9L45-L47

多个 intent：
- 默认 `min()` 合并（AND）

### 6.2 融合排序（默认弱权重，不抢 SigLIP 主导）
`final_rank = siglip_score + 0.2*constraint_score + 0.05*quality`

---

## 7. 候选选择：margin 自适应 + clamp（稳定延迟 + 不漏）

对所有 track 得到 `final_rank_score`：
- `best = max(score)`
- keep all `score >= best - Δ`
- clamp 到 `[K_min, K_max]`

默认：
- `Δ=0.08`
- `K_min=10`
- `K_max=60`

---

## 8. Verifier：need_context 的执行策略（按你的偏好：bool，无 auto）

**Phase 2.7 推荐策略**
- need_context=false：只跑 L1（crop）
- need_context=true：同一批候选 **L1 + L2 一起跑**（或至少 topM=5 跑 L2）

---

## 9. 代码落地：按你 repo 结构写的 TODO（极具体）
文件映射来自你 repo 文档fileciteturn4file4L7-L16。

### 9.1 修改 `src/pipeline/router_vlm.py`
- [ ] 输出 schema 改成 QuerySpec（prompts + need_context + constraint_intents）
- [ ] 删除 hard_rules 数值输出（不再让 VLM 产 min/max）
- [ ] 旧清洗逻辑简化成 schema 校验（出现数字/extra keys 就重试）

### 9.2 修改 `src/core/features.py`（TrackFeatureExtractor）
- [ ] 新增：norm_speed / scale_change / linearity / centroids / displacement_vec
- [ ] 单测覆盖（直线/徘徊/靠近）

### 9.3 修改 `src/core/evidence.py`
- [ ] 把新增 atomic facts 写入 EvidencePackage（保持“数据货币”原则fileciteturn4file10L8-L12）

### 9.4 新增 `src/core/constraints.py`
- [ ] intent → soft score（sigmoid）
- [ ] debug_breakdown 输出

### 9.5 修改 orchestrator `src/pipeline/video_semantic_search.py`
旧链路在 system_blueprint 中是：SigLIP 粗筛 → HardRuleEngine → VLM 终审fileciteturn4file13L1-L6。  
Phase 2.7 改成：
- [ ] SigLIP 得 siglip_score（不硬删）
- [ ] constraints 得 constraint_score
- [ ] final_rank 融合
- [ ] margin+clamp 选候选
- [ ] need_context 决定是否 L2 双跑

### 9.6 修改 `src/adapters/inference/vllm_adapter.py`
- [ ] L2 prompt 注入 facts（norm_speed/scale_change/linearity）+ constraint_intents
- [ ] system 强制：use provided facts, do not invent thresholds

---

## 10. 验收标准（DoD）
- [ ] 进入 verifier 的候选数明显下降，但不会因误杀导致漏召回
- [ ] need_context=true 的 query 会触发 L2（符合 router 的定义fileciteturn4file0L16-L21）
- [ ] constraints 在 RUNNING/APPROACHING 类 query 上能明显帮助排序或让 VLM 输出更稳定
