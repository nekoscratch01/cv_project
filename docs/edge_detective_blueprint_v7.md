这份文档是 **Edge‑Detective v7.0** 的**技术规格说明书 (Engineering Spec)**。  
它在 v6 的“理想三模型架构”基础上，收敛成一个**在 Mac M4（16GB）上真能跑起来的版本**：

- 保留：Atomic 8 + EvidencePackage 协议、四层结构（Router / Recall / Hard Rules / Verifier）、场景 A/B/C 的工作流；
- 调整：模型栈从“Qwen3-4B-Thinking + Qwen3-VL-2B + SigLIP 三件套”简化为：
  - **一个 Qwen3‑VL‑4B（Int4）** 同时担任 Router + Verifier；
  - **一个 CLIP/SigLIP** 负责向量粗筛。

-----

# 🏗️ Edge-Detective v7.0 技术规格说明书 (M4 实施版)

> **文档性质**: 实施标准 / 架构定义（M4 可运行版本）  
> **适用硬件**: Mac M4 (16GB RAM) / 近似算力边缘设备  
> **核心模型**: Qwen3‑VL‑4B‑Instruct‑GGUF (单模型双角色) + CLIP/SigLIP  
> **量化标准**: 4B VLM 使用 **GGUF Int4**

核心思路一句话：  
> 一个 4B VLM 负责“理解 + 规划 + 最终判断”，几何真相由 Atomic 8 保证，CLIP 只在中间做一次高召回粗筛。

-----

## 1. 数据协议层 (Data Protocol Layer) —— 沿用 v6 的“宪法”

v7 **完全沿用** v6 的 Atomic 8 与 EvidencePackage 协议，不做任何削弱，只在实现上把这些字段算出来并存盘。

### 1.1 原子事实 (Atomic 8 Features)

- **存储位置**: `EvidencePackage.features`  
- **生成时机**: 感知阶段 (YOLO + ByteTrack 后处理)

| 字段名 | 类型 | 物理含义 | 逻辑用途示例 |
| :--- | :--- | :--- | :--- |
| `track_id` | `int` | 唯一标识符 (在某个 `video_id` 内唯一) | 锁定目标 |
| `start_s` | `float` | 首帧时间戳 | “最早出现的…” / “先后顺序” |
| `end_s` | `float` | 末帧时间戳 | “最后离开的…” / “谁最后进门” |
| `duration_s` | `float` | $end - start$ | “停留超过 30 秒的人” |
| `centroids` | `List[(x,y)]` | 归一化中心点轨迹 (0.0–1.0) | ROI 判定 / 边缘检测 / 徘徊检测 / 跟随 |
| `avg_speed_px_s` | `float` | 平均速度 (px/s) | “跑得最快的人” |
| `max_speed_px_s` | `float` | 最大瞬时速度 | “突然加速的人” |
| `displacement_vec` | `(vx, vy)` | 首尾位移向量 | “从左往右 / 从门口进店 / 朝出口方向离开” |

> 约束：  
> - 不在这里写任何“语义字段”（如 `is_thief`, `is_fighting`）；  
> - 所有行为语义都要通过上层规则 + VLM 在这些原子事实之上推导。

### 1.2 视觉证据 (Visual Evidence)

- **存储位置**: `EvidencePackage.crops_paths`  
- **生成时机**: 感知阶段，对 `TrackRecord` 采样后保存为 jpg。

采样策略（保持与 v6 一致，只是落地到 M4）：

- 过滤分辨率 \< 50×50 的碎片；
- 用 Laplacian 清晰度评分，从该轨迹所有帧中挑出 Top‑3 ~ Top‑5；
- 尽量覆盖时间轴（起点、中段、终点），避免只看一瞬间。

### 1.3 完整证据包 (EvidencePackage)

```python
class EvidencePackage:
    meta: Dict[str, Any]      # {'video_id': str, 'fps': float, 'resolution': (w, h)}
    raw_trace: List[Box]      # [x1, y1, x2, y2] per frame
    frames: List[int]         # 对应帧号
    crops_paths: List[str]    # 若干精选裁剪图路径
    embedding: Optional[List[float]]  # CLIP/SigLIP 向量 (可延迟计算)
    features: TrackFeatures   # 上面的 Atomic 8
```

-----

## 2. 模型栈 (Model Stack) —— 单 4B VLM + CLIP

和 v6 最大的不同点在这里：**不再假设现有环境里能同时常驻 2–3 个大模型**。

### 2.1 VLM (Router + Verifier 共用)

- **模型**: `Qwen3‑VL‑4B‑Instruct‑GGUF` (Int4)  
- **角色**：
  1. **Router 模式**（纯文本，Thinking）：
     - 输入：原始用户 query（中文） + 若干 Few‑Shot 示例；  
     - 输出：ExecutionPlan（后文定义）；
  2. **Verifier 模式**（多图 + 文本）：
     - 输入：若干候选 EvidencePackage 的裁剪图 + 原 query + 部分原子事实摘要；  
     - 输出：逐条轨迹的 Yes/No 判定 + 解释（reason）。

> 实现注意：  
> - 这两个“模式”只是 prompt 不同，都走同一个 4B 模型实例；  
> - 为节省显存，可以在 Router 阶段只加载文本部分（不送图），Verifier 阶段再喂图。

### 2.2 Recall 模型：CLIP / SigLIP

- **模型**: 任意 300–400M 级的 CLIP / SigLIP（FP16 / BF16）；  
- **用途**：
  - 将 ExecutionPlan 里的短描述 / visual_tags 映射到 embedding 空间；  
- 对所有 EvidencePackage 的 crops 做 embedding，计算相似度，选 Top‑K 作为候选；
- **不做推理，只做召回**。

### 2.3 资源预算（Mac M4 16GB）

粗略估算（Int4 + FP16）：

- Qwen3‑VL‑4B‑Int4：约 3–3.5GB；  
- CLIP‑400M‑FP16：约 0.6–0.8GB；  
- Python + 代码逻辑 + 缓冲：4–5GB；  
- 余量：约 6–7GB（足够 cache crops / embedding / 中间结果）。

-----

## 3. ExecutionPlan：从“长 query”变成“可执行计划”

v6 里的 ExecutionPlan 只有 `visual_tags + hard_rules + verification` 三类字段，  
在 v7 里，我们扩展成更适合单 4B VLM + CLIP 的形状。

### 3.1 ExecutionPlan Schema (v7)

```json
{
  "description": "a person in a blue shirt near the store entrance",
  "visual_tags": ["blue shirt", "near entrance"],
  "needed_facts": ["start_s", "end_s", "centroids", "avg_speed_px_s"],
  "constraints": {
    "roi": "shop_door",
    "event_type": "enter_then_run",
    "time_window": [0.0, 120.0],
    "sort_by": "end_s",
    "sort_order": "desc",
    "limit": 5,
    "min_speed_jump": 2.0
  },
  "verification_prompt": "Given the original question, is this track a plausible match? Answer Yes or No."
}
```

字段说明：

- `description: str`  
  - 为 CLIP / VLM 准备的、**简化版的英文/中英混合描述**；  
  - 例：`"a person in a red hoodie carrying a backpack"`。

- `visual_tags: List[str]`  
  - 更细粒度的标签列表，方便 Router 显式列出关键属性；  
  - 例：`["red hoodie", "backpack"]`。  
  - Recall Engine 可以用 `(description + visual_tags)` 拼成检索文本。

- `needed_facts: List[str]`  
  - 告诉 Hard Rule Engine / Verifier 这次判断需要哪些 Atomic 8 字段；  
  - 例：`["start_s", "end_s", "centroids"]`（找“最后进入门口的人”）  
        `["avg_speed_px_s", "max_speed_px_s"]`（找“跑得最快的人”）。

- `constraints: Dict`  
  - 描述在 Atomic 8 空间里要怎样筛选/排序：
    - `roi: str`：使用哪个预定义 ROI（door / shop / cashier 等）；  
    - `event_type: str`：像 `"enter"`, `"stay"`, `"enter_then_run"`, `"follow"` 这样的小枚举，指导 Hard Rule Engine 选用哪一套算子组合；  
    - `time_window: [start, end]`：限定只在部分时间段搜索；  
    - `sort_by: str`：`"start_s" | "end_s" | "duration_s" | "avg_speed_px_s" | "max_speed_px_s"`；  
    - `sort_order: "asc" | "desc"`；  
    - `limit: int`：最多保留多少条轨迹供下一步验证；  
    - 以及一些可选阈值，如 `min_speed_jump`, `min_dwell_s` 等。

- `verification_prompt: str`  
  - 给 Verifier 模式的 4B VLM 用的 Yes/No 问题；  
  - 格式统一为：  
    `"Given the original question, is this track a plausible match? Answer Yes or No."`  
    或加上一句细化说明。

### 3.2 Router（4B VLM 的“规划模式”）

- **输入**：用户原始 query（中文为主） + 若干场景 A/B/C 的 Few‑Shot 示例；  
- **输出**：`ExecutionPlan`（JSON） + `<think>` 中间推理文本。

Prompt 设计沿用 v6 的思路：

- System Prompt 强调：
  - 只能使用定义好的字段（description / visual_tags / needed_facts / constraints / verification_prompt）；  
  - 不允许发明新的字段；  
  - 所有关于“最后 / 最快 / 跟随 / 进入 / 停留”等词，要转化到 Atomic 8 + ROI 的语言中。

- Few‑Shot 示例直接使用场景 A/B/C 的问法，让模型学会：
  - 纯逻辑问题（不提外观）的 ExecutionPlan 怎么写；  
  - 纯外观问题（不提时间/地点）的 ExecutionPlan 怎么写；  
  - 视觉 + 几何混合问题（既要看颜色，又要看谁跑得最快）的 ExecutionPlan 怎么写。

解析函数与 v6 类似：  
`parse_router_output(raw_output) -> (ExecutionPlan, think_log)`

-----

## 4. Recall Engine：CLIP/SigLIP 粗筛层

Recall Engine 的职责很简单：

> 根据 ExecutionPlan.description / visual_tags，从所有轨迹中找出一小批“看起来最可能相关”的候选，供 Hard Rule Engine + Verifier 深挖。

### 4.1 接口（逻辑级）

```python
def visual_filter(
    tracks: List[EvidencePackage],
    description: str,
    visual_tags: List[str],
    top_k: int = 50,
) -> List[EvidencePackage]:
    ...
```

### 4.2 算法步骤

1. 将 `description` 和 `visual_tags` 拼成一条检索文本，例如：  
   `"a person in a red hoodie carrying a backpack, near the shop entrance"`；
2. 用 CLIP/SigLIP 编码成文本向量 `q`；  
3. 对每条轨迹：
   - 对其所有 `crops_paths` 编码成图像向量 `v_i`；  
   - 计算 `max_i cos(q, v_i)` 作为该轨迹的相似度分数；  
4. 按分数排序，保留 Top‑K 条（默认 50）；  
5. 返回这些 EvidencePackage。

> 注意：如果 ExecutionPlan 里没有任何视觉约束（description 非常抽象，或者用户压根没提外观），Recall Engine 可以退化为：  
> - `visual_filter` 直接返回全量轨迹（不做筛选），把工作交给 Hard Rules + Verifier。

-----

## 5. Hard Rule Engine：原子事实上的“会计师”

Hard Rule Engine 只干一件事：

> 在 Recall Engine 选出的候选集合里，用 Atomic 8 做数学过滤与排序，把明显不可能的都排除掉，再把 Top‑N 交给 Verifier。

### 5.1 核心算子（与 v6 一致，只是绑定 ExecutionPlan.constraints）

1. ROI 过滤：  
   - `enter / stay / cross` → 基于 `centroids` 与 ROI 多边形关系；
2. 时间过滤：  
   - `time_window` → 在 `[start_s, end_s]` 范围外的轨迹直接扔掉；
3. 排序：  
   - `sort_by / sort_order` → 基于 `start_s / end_s / duration_s / avg_speed_px_s / max_speed_px_s`；
4. 阈值过滤：  
   - `min_speed_jump` → 比较轨迹前段/后段速度差；
   - `min_dwell_s` → 在某 ROI 内停留时长是否超过阈值；
   - 以后还可以扩 `follow_min_overlap_s` 等等。

### 5.2 总入口接口

```python
def apply_constraints(
    tracks: List[EvidencePackage],
    plan: ExecutionPlan,
) -> List[EvidencePackage]:
    """
    按 ExecutionPlan.constraints 里的字段，调用 ROI / 时间 / 排序 / 阈值 等算子，
    输出满足约束条件、数量不超过 plan.constraints.limit 的轨迹列表。
    """
```

-----

## 6. Verifier：4B VLM 的“终审模式”

Verifier 使用与 Router 相同的 Qwen3‑VL‑4B 模型，只是 prompt 变成 “看图 + 决策” 模式。

### 6.1 输入组成

对每条候选轨迹，我们提供：

- 若干裁剪图（`crops_paths`）；  
- 原始用户 query（中文）；  
- ExecutionPlan 的摘要（description + constraints 简述）；  
- 部分 Atomic 8 的数值（start/end/duration/speed 等）。

### 6.2 Prompt 结构示意

（伪代码，自然语言可中英混合）

```text
You are a video analysis assistant.

User question:
{user_query}

System facts for this track:
- start time: {start_s} seconds
- end time: {end_s} seconds
- duration: {duration_s} seconds
- average speed: {avg_speed_px_s} px/s
- max speed: {max_speed_px_s} px/s

High-level description from planner:
{plan.description}

Look at the following images of this track and
think step-by-step whether this track matches the user question.

Finally, answer strictly "Yes" or "No" on the first line,
then give a short explanation in 1-2 sentences.
```

解析逻辑：

```python
def verify_candidate(track: EvidencePackage, plan: ExecutionPlan, user_query: str) -> Tuple[bool, str]:
    # 1. 构造上面的 prompt + 多张图
    # 2. 调用 Qwen3‑VL‑4B
    # 3. 从输出第一行解析 Yes/No → bool
    # 4. 剩余部分作为 reason 返回
```

-----

## 7. 端到端场景（v7 版本）

### 场景 A：纯逻辑 ——「帮我找最后一个进店的人」

1. Router 模式（4B 文本）
   - `description = "people entering the shop"`  
   - `visual_tags = []`  
   - `needed_facts = ["start_s", "end_s", "centroids"]`  
   - `constraints = { "roi": "shop_door", "event_type": "enter", "sort_by": "end_s", "sort_order": "desc", "limit": 1 }`  
   - `verification_prompt = ""`（纯几何题，不需要视觉确认）
2. Recall Engine（CLIP）
   - `visual_tags` 为空 → 直接透传所有轨迹。
3. Hard Rule Engine
   - 用 ROI + `end_s` 排序，取最后一个“进入 shop_door”的轨迹。
4. Verifier
   - `verification_prompt` 为空 → 直接把这条轨迹作为最终答案。

### 场景 B：纯视觉 ——「找穿红衣服、背书包的人」

1. Router
   - `description = "a person wearing red clothes and carrying a backpack"`  
   - `visual_tags = ["red clothes", "backpack"]`  
   - `needed_facts = []`（这次完全靠外观）  
   - `constraints = {"limit": 50}`  
   - `verification_prompt` 要求严格 Yes/No。
2. Recall Engine
   - CLIP 用 description + visual_tags 找出 Top‑50 轨迹。
3. Hard Rules
   - constraints 除了 limit 没别的 → 透传。
4. Verifier
   - 对这 50 条逐个问 “Yes/No”，保留 `match=True` 的轨迹 + reason。

### 场景 C：混合 ——「谁是跑得最快的红衣人？」

1. Router
   - `description = "a person wearing red clothes"`  
   - `visual_tags = ["red clothes"]`  
   - `needed_facts = ["avg_speed_px_s"]`  
   - `constraints = { "sort_by": "avg_speed_px_s", "sort_order": "desc", "limit": 1 }`
2. Recall Engine
   - CLIP 先找出 Top‑50 “可能是红衣人”的轨迹。
3. Hard Rules
   - 在这 50 条里按 `avg_speed_px_s` 排序，取第一名。
4. Verifier
   - 用 VLM 再确认一次“是否穿红衣服”，避免几何误选。

-----

## 8. 小结

v7 相当于给 v6 做了一个 **“M4 可运行版压缩”**：

- 协议层（Atomic 8 + EvidencePackage）完全保留，不牺牲泛化能力；  
- 模型层从“三模型豪华版”收敛为“单 4B VLM + 一个 CLIP”；  
- ExecutionPlan 从简单的 `visual_tags + hard_rules + verification` 升级为更加明确的  
  `description + visual_tags + needed_facts + constraints + verification_prompt`；  
- 整体流程仍然是：  
  自然语言 → ExecutionPlan → CLIP 粗筛 → Hard Rules 物理过滤 → VLM 终审 + 解释。

接下来，代码侧只需要按这个 Spec 逐步把：

- `TrackFeatures` 扩展到 Atomic 8；  
- `EvidencePackage` 补齐 meta/raw_trace/embedding；  
- 加一个 ExecutionPlan 类型 + Router 模块；  
- 把现有 Recall/VLMClient/behavior 模块按照这里的接口慢慢对齐，

你就可以在当前的 Mac 上先跑出一个真正“按问题找人/找行为”的 v1 系统了。

