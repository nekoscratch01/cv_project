这份文档是 **Edge‑Detective v6.0** 的**技术规格说明书 (Engineering Spec)**。  
它是这个项目的**最终目标架构**，所有长期设计和重构都要向这里对齐。

-----

# 🏗️ Edge-Detective v6.0 技术规格说明书 (Engineering Spec)

> **文档性质**: 实施标准 / 架构定义  
> **适用硬件**: Mac M4 (16GB RAM) / Jetson Orin (Edge Native)  
> **核心模型**: Qwen3-4B-Thinking (Router) + Qwen3-VL-Thinking (Verifier) + SigLIP  
> **量化标准**: 全链路 **GGUF Int4**

核心思想一句话：  
> **不让大模型“猜”，所有事实都来自几何 + 规则，模型只做路由和解释。**

-----

## 1. 数据协议层 (Data Protocol Layer)

这是系统的地基。所有上层逻辑（Router / Hard Rules / Recall / Verifier）**只能**基于这一层的字段进行计算。  
**严禁**在此层之外增加业务标签（如 `is_suspicious`、`is_thief`），这些都必须在上层由规则 + 模型推理出来。

### 1.1 原子事实 (The Atomic 8 Features)

- **存储位置**: `EvidencePackage.features`  
- **生成时机**: Phase 1 感知阶段 (YOLO + ByteTrack 后处理)  

我们为每条轨迹计算 8 个原子事实（Atomic 8）：

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

> 直觉：  
> 给你一条 `TrackFeatures`，你就可以在纸上还原“这个人从哪来，到哪去，走了多久，多快，多曲折”，完全不看像素。  
> 所有复杂行为（进门、徘徊、尾随、同行、打架）都通过组合这 8 个事实来定义。

### 1.2 视觉证据 (Visual Evidence)

- **存储位置**: `EvidencePackage.crops_paths`  
- **生成时机**: 感知阶段，对 `TrackRecord` 采样后保存为图片文件。

**采样策略: Quality-Based Sampling**

- 对每条轨迹：
  - 过滤分辨率 \< 50×50 的碎片框；
  - 用 Laplacian 清晰度打分；
  - 保留 **Top-3 ~ Top-5** 张最清晰的图；
  - 尽量覆盖不同时间点（开头 / 中间 / 结尾）。

**用途**

- 仅供：
  - Tier 1：SigLIP 做向量召回；  
  - Tier 2：Qwen3-VL-Thinking 做 Yes/No 视觉验证。
- 原视频小片段（高亮轨迹视频）只在**输出可视化时按需重建**，不写死在 EvidencePackage 里。

### 1.3 完整证据包 (EvidencePackage)

这是在模块间流转的**唯一对象**。

```python
class EvidencePackage:
    # 1. 元信息 (Meta)
    meta: Dict[str, Any]    # {'video_id': str, 'fps': float, 'resolution': (w, h)}
    
    # 2. 原始轨迹 (Raw Trace) - 用于画图/视频生成
    raw_trace: List[Box]    # 每一帧的 bbox [x1, y1, x2, y2]
    frames: List[int]       # 对应的帧号
    
    # 3. 视觉证据 (Visual Evidence) - 给 SigLIP / VLM 看的
    crops_paths: List[str]  # 若干精选裁剪图的路径
    
    # 4. 向量索引 (Embedding, Optional)
    embedding: Optional[List[float]]  # SigLIP 向量 (可选, 可延迟计算)
    
    # 5. 几何真相 (Atomic 8)
    features: TrackFeatures
```

> 设计原则：  
> - **事实分层**：几何真相 (Atomic 8) 与视觉证据 (crops / embedding) 分开；  
> - **可恢复性**：只要有 `raw_trace + frames + meta`，就能重建高亮视频；  
> - **扩展性**：将来加行为摘要、事件摘要，只能附加，不允许修改 Atomic 8。

-----

## 2. 逻辑路由层 (Tier 3: The Thinking Router)

Router 的职责是：  
> 把“用户问话”翻译成 “视觉 + 数学 + 验证” 三类指令：
> - 要看哪些外观特征？ → `visual_tags`  
> - 要执行哪些硬规则？ → `hard_rules` (基于 Atomic 8)  
> - 最后需要 VLM 回答什么 Yes/No 问题？ → `verification`

### 2.1 模型配置

- **Model**: `Qwen3-4B-Thinking-Instruct-GGUF` (Int4)  
- **Context**: 4096 tokens  
- **Grammar**: 使用 `llama.cpp` / `llama-cpp-python` 的 grammar 功能，对 Router 输出施加 JSON 约束。

### 2.2 思维链 + 语法约束 (CoT + Grammar)

让 4B 模型变强靠三招：

1. **In-Context Learning (Few-Shot)**：在 System Prompt 里塞 3–5 个真实范例，让它照着来；  
2. **CoT (思维链)**：利用 Qwen3 的 `<think>` 标签，让它先写“想法”，再写 JSON；  
3. **Grammar Constraint**：用 grammar/response_format 把输出锁死成合法 JSON。

### 2.3 ExecutionPlan Schema

Router 输出的 JSON 结构固定为：

```json
{
  "visual_tags": ["red clothes", "backpack"],
  "hard_rules": {
    "roi_op": "enter",
    "roi_name": "shop_door",
    "sort": "time_desc",
    "limit": 1,
    "time_range": [0.0, 60.0]
  },
  "verification": "Is this person wearing red clothes and a backpack?"
}
```

- `visual_tags: List[str]`  
  - 给 SigLIP / VLM 用的外观描述；
  - 如果用户没提外观，可以是 `[]`。
- `hard_rules: Dict`  
  - 给 Hard Rule Engine 用的“物理过滤规则”，只能包含预定义字段：
    - `roi_op`: `"enter" | "exit" | "stay" | "cross"`  
    - `roi_name`: 预定义 ROI 名（如 `"door"`, `"shop"`, `"cashier"`）  
    - `sort`: `"time_desc" | "time_asc" | "speed_desc" | "duration_desc"`  
    - `limit`: `int`，返回多少个候选  
    - `time_range`: `[start_s, end_s]`，可选时间窗  
    - 将来扩展时，只能新增字段，不允许修改已有含义。
- `verification: str`  
  - 给 Verifier (Qwen3-VL) 的 Yes/No 问题；  
  - 可以为空字符串，表示“不需要 VLM 最终确认”（纯逻辑题）。

### 2.4 System Prompt（含 Few-Shot 范例）

下面是 Router 的核心 Prompt，体现了 “原子能力 + 示例” 的思路（简化版）：

```text
SYSTEM_PROMPT = """
你是一个视频数据查询编译器。你的任务是将用户的自然语言查询，转换为结构化的 JSON 执行计划。

【原子能力定义】
你只能使用以下原子能力，不可捏造字段：
1. visual_tags: 外观描述列表 (给 SigLIP / VLM 用)
2. hard_rules: 物理过滤规则 (给 Python 代码用)
   - sort: "time_desc" (最晚), "time_asc" (最早), "speed_desc" (最快), "duration_desc" (最久)
   - roi_op: "enter" (进入), "stay" (停留), "cross" (穿越)
   - roi_name: 区域名称 (如 "shop", "door")
   - limit: 返回数量限制
   - time_range: [start_s, end_s]
3. verification: 视觉验证问题 (Yes/No)

【思维链范例 (Few-Shot)】

User: "帮我找最后一个进店的人"
Output:
<think>
1. 分析意图: 用户找"人"。逻辑是"进店"(ROI) + "最后"(Time Sort)。
2. 视觉特征: 没提外观，visual_tags 为空。
3. 硬规则:
   - 动作: 进入(enter) -> 区域: shop
   - 排序: 时间倒序(time_desc) -> 取 1 个
4. verification 不需要，因为这道题靠几何就能回答。
</think>
```json
{
  "visual_tags": [],
  "hard_rules": {"roi_op": "enter", "roi_name": "shop", "sort": "time_desc", "limit": 1},
  "verification": ""
}
```

User: "找穿红衣服、背书包的人"
Output:
<think>
1. 分析意图: 用户关心外观(红衣服, 背包)，没有明确的时间和区域限制。
2. visual_tags: ["red clothes", "backpack"]。
3. 硬规则: 不需要 ROI 或排序，只要返回所有可能候选 -> hard_rules = {}。
4. verification: 需要让 VLM 严格判断是否同时满足"红衣服"+"背书包"。
</think>
```json
{
  "visual_tags": ["red clothes", "backpack"],
  "hard_rules": {},
  "verification": "Is this person wearing red clothes and a backpack? Answer Yes or No."
}
```

User: "有没有人鬼鬼祟祟躲避摄像头？"
Output:
<think>
1. 分析意图: "躲避摄像头"是主观行为，不能直接从 Atomic 8 得到。
2. 转化为物理特征:
   - 轨迹贴近画面边缘 (centroids 靠近边界)。
   - 可能伴随高速度变化或突然折返。
3. 视觉上: 可能有"低头"、"遮挡脸"等外观特征。
4. 策略:
   - hard_rules: 先按"edge_stay"和"tortuosity_desc"等规则挑出 10 条最可疑的轨迹。
   - visual_tags: ["hiding face", "looking away from camera"]。
   - verification: 让 VLM 判断是否有刻意躲避摄像头的行为。
</think>
```json
{
  "visual_tags": ["hiding face", "looking away from camera"],
  "hard_rules": {"sort": "tortuosity_desc", "limit": 10},
  "verification": "Does this person seem to avoid the camera by turning away or hiding their face? Answer Yes or No."
}
```

你的回答必须严格遵守上面的 JSON 结构。
"""
```

### 2.5 Router 输出解析

Router 的原始输出形如：

```text
<think>
... 一大段中文/英文推理 ...
</think>
```json
{ ... ExecutionPlan ... }
```

解析函数（逻辑）：

```python
def parse_router_output(raw_output: str) -> Tuple[ExecutionPlan, str]:
    """
    1. 提取 <think>...</think> 之间的内容，作为 log_text。
    2. 提取 ```json 代码块，反序列化为 ExecutionPlan。
    3. 返回 (ExecutionPlan, log_text)。
    """
```

-----

## 3. 硬规则引擎层 (Tier 0: The Math Engine)

Hard Rule Engine 完全不用模型，只靠 Python + 数学运算执行 Router 的 `hard_rules`。它的输入是：

- `tracks: List[EvidencePackage]`  
- `rules: Dict`（即 ExecutionPlan.hard_rules）

输出仍然是 `List[EvidencePackage]`（被筛选/排序后的子集）。

### 3.1 核心算子 (Operators)

所有复杂逻辑都由这些基础算子组合出来。

#### 3.1.1 ROI 相关：`op_filter_roi`

```python
def op_filter_roi(tracks, roi_poly, mode: str) -> List[EvidencePackage]:
    ...
```

- 输入：轨迹列表 + ROI 多边形（由 `SystemConfig.roi_zones` 提供）；
- 基于 `centroids` 判断：
  - `mode="enter"`：首段在 ROI 外，后续某一帧进入 ROI 内；
  - `mode="stay"`：超过一定比例的点都在 ROI 内（例如 >80%）；
  - `mode="cross"`：从 ROI 一侧进入，从另一侧离开（穿过某个门）。

#### 3.1.2 排序相关：`op_sort`

```python
def op_sort(tracks, key: str, reverse: bool) -> List[EvidencePackage]:
    ...
```

- `key="time_start"`：按 `features.start_s` 排序；
- `key="time_end"`：按 `features.end_s` 排序；
- `key="speed"`：按 `features.avg_speed_px_s` 排序；
- `key="duration"`：按 `features.duration_s` 排序。

#### 3.1.3 交互相关：`op_interaction`

```python
def op_interaction(tracks, dist_thresh: float) -> List[Tuple[EvidencePackage, EvidencePackage]]:
    ...
```

- 计算任意两条轨迹在同一时间段的中心点距离；
- 如果存在一段连续时间，距离持续低于 `dist_thresh`，则认为这两人“有交互”（用于同行/打架/尾随等上层事件）。

### 3.2 apply_hard_rules 总入口

```python
def apply_hard_rules(
    tracks: List[EvidencePackage],
    rules: Dict,
) -> List[EvidencePackage]:
    """
    根据 ExecutionPlan.hard_rules 字段，组合调用 ROI / 排序 / 交互等算子。
    """
```

示例：

- `{"roi_op": "enter", "roi_name": "shop", "sort": "time_desc", "limit": 1}`：
  1. 用 `roi_name` 找到 ROI 多边形；  
  2. 调 `op_filter_roi(mode="enter")` 找出所有“进店”的轨迹；  
  3. 按 `end_s` 或 `start_s` 倒序排序；  
  4. 截断到 1 条。

-----

## 4. 视觉验证层 (Tier 1 & Tier 2)

### 4.1 侦察兵：SigLIP Recall Engine (Tier 1)

- 模型：`SigLIP-So400M` (FP16, PyTorch)  
- 输入：
  - 轨迹的 `crops_paths`（多张图）；  
  - Router 的 `visual_tags`（文本）。
- 输出：按相似度排序的轨迹子集。

逻辑：

1. 如果 `visual_tags` 为空，直接返回原始轨迹列表（不做筛选）；  
2. 否则：
   - 把 `visual_tags` 拼成一句短英文描述，编码成文本向量；
   - 对每条轨迹，取其所有 crops 编码成图像向量，取最大相似度作为该轨迹的得分；
   - 按得分排序，保留 Top-K（例如 20）。

对应 v6 接口（伪代码）：

```python
def visual_filter(
    tracks: List[EvidencePackage],
    tags: List[str],
    top_k: int = 20,
) -> List[EvidencePackage]:
    ...
```

### 4.2 狙击手：Qwen3-VL-Thinking (Tier 2)

- 模型：`Qwen3-VL-2B-Thinking-Instruct-GGUF` (Int4)  
- 职责：对候选轨迹做最终视觉确认，并给出一步一步的视觉分析。

接口：

```python
def verify_candidate(
    track: EvidencePackage,
    prompt: str,
) -> Tuple[bool, str]:
    """
    返回:
    - match: bool → 是否符合描述
    - reason: str → 模型的视觉推理过程
    """
```

Prompt 策略（示意）：

```text
"Look at these images carefully.
Think step-by-step about visual details to answer: {prompt}
Finally, answer strictly 'Yes' or 'No'."
```

解析逻辑：

1. 模型会先输出一大段分析，比如：  
   `"I see a person wearing a red jacket and a backpack..."`  
2. 最后输出 `"Yes"` 或 `"No"`；  
3. 我们提取最后一个 Yes/No 作为布尔结果，把前面的分析原样保留为 `reason`。

-----

## 5. 端到端场景穿透 (End-to-End Scenarios)

用几个代表性问题，说明 Router / Hard Rules / SigLIP / Verifier 如何协作。

### 场景 A：纯逻辑题 ——「帮我找最后一个进店的人」

1. **Router**  
   - `visual_tags = []`  
   - `hard_rules = {"roi_op": "enter", "roi_name": "shop", "sort": "time_desc", "limit": 1}`  
   - `verification = ""`

2. **SigLIP**  
   - `visual_tags` 为空 → 直接透传所有轨迹。

3. **Hard Rules**  
   - 用 ROI + `start_s/end_s` 找出所有进店的人；  
   - 按时间倒序排序，取第一个。

4. **Verifier**  
   - `verification` 为空 → 跳过；  
   - 输出这个轨迹的 `track_id + time range`。

### 场景 B：纯视觉题 ——「找穿红衣服、背书包的人」

1. **Router**
   - `visual_tags = ["red clothes", "backpack"]`  
   - `hard_rules = {}`  
   - `verification = "Is this person wearing red clothes and a backpack? Answer Yes or No."`

2. **SigLIP**  
   - 用 `visual_tags` 做向量检索，保留 Top-20 候选轨迹。

3. **Hard Rules**  
   - `hard_rules` 为空 → 直接透传。

4. **Verifier**  
   - 对这 20 条轨迹逐个调用 `verify_candidate`；  
   - 保留所有 `match=True` 的轨迹，附上 reason。

### 场景 C：混合题 ——「谁是跑得最快的红衣人？」

1. **Router**
   - `visual_tags = ["red clothes"]`  
   - `hard_rules = {"sort": "speed_desc", "limit": 1}`  
   - `verification` 可选。

2. **SigLIP**  
   - 找出最可能穿红衣服的若干轨迹。

3. **Hard Rules**  
   - 在候选集合内部按 `avg_speed_px_s` 排序，取第一名。

4. **Verifier**  
   - 如果 `verification` 不为空，再确认一次“是不是红衣人”。

### 场景 D：行为题 ——「谁和谁打架了？」

1. **Router**  
   - 推理：打架 ≈ 高交互 + 高速度变化；  
   - 输出：

     ```json
     {
       "visual_tags": ["two people fighting", "physical conflict"],
       "hard_rules": {"logic": "interaction", "filter": "high_motion", "limit": 5},
       "verification": "Are these two people fighting or hitting each other?"
     }
     ```

2. **SigLIP**  
   - 按 `visual_tags` 找出疑似“有人靠得很近且姿态激烈”的片段对应的轨迹对；

3. **Hard Rules**  
   - 用 `op_interaction` 找出距离极近、交互时间长的轨迹对；  
   - 用 `max_speed_px_s` / 速度变化筛掉“站着聊天”的，把“剧烈运动”的对留下。

4. **Verifier**  
   - 对每一对轨迹合成大框裁剪图，问：  
     `"Are these two people fighting or hitting each other? Answer Yes or No."`  
   - 只保留回答为 Yes 的对，并记录模型的解释。

-----

## 6. 硬件与落地检查清单 (Hardware & Checklist)

### 6.1 Mac M4 16GB 生存策略

- Router (4B) + Verifier (2B) 都用 **Int4 GGUF**；
- SigLIP 用 FP16，占用 \~0.6GB，可常驻；
- Router 与 Verifier 尽量串行调用，不在同一时刻跑两个大模型；
- 对 crops 做分辨率和清晰度过滤，避免 VLM 在垃圾图上浪费显存与算力。

### 6.2 Day‑1 Checklist

在写任何代码之前，请确认你拥有：

1. **模型文件**
   - `Qwen3-4B-Thinking-Instruct-q4_k_m.gguf`  
   - `Qwen3-VL-2B-Thinking-Instruct-q4_k_m.gguf`（或当前可用的最小 Thinking‑VL 版本）
2. **库依赖**
   - `llama-cpp-python`（带 Metal 支持）；  
   - `torch`, `numpy`;  
   - `shapely`（用于 ROI 多边形计算）。
3. **测试素材**
   - 至少一段 1 分钟的视频，包含：  
     - 进出门场景；  
     - 两人同行；  
     - 一人跑动或剧烈运动。

> 这份 v6 Spec 不直接告诉你“怎么写代码”，而是告诉你：  
> - 数据协议永远长什么样；  
> - Router / Hard Rules / SigLIP / Verifier 各自的黑盒边界；  
> - 四个层次如何组合起来回答从“找人”到“打架”这类复杂问题。  
> 你可以把它当成以后所有重构的“宪法”。

