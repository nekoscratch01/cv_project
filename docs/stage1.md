# Stage 1 改动思路（以“把人找准”为目标）

- 场景定位：先把单人检索稳住（颜色/服饰/动作），不追求复杂交互。
- 重点动作：给 VLM 喂清晰图 + 可读的动作线索，过滤噪声轨迹，保留轻量可控的参数。

## 已落地的核心改动
- VLM 输入（`src/pipeline/vlm_client_hf.py`）
  - 裁剪采样：均匀采样 `package.crops`，不再只取前几帧，确保看到最清晰帧。
  - 语义叙事：用几何数据翻译为标签（速度、方向、左右位置），Prompt 要求输出 `thinking/match/reason` JSON。
  - 视觉提示：生成轨迹小地图（白底、灰线、0.5s 彩色打点、绿到红渐变、终点箭头），作为额外一张图帮助理解方向/快慢/停滞。
- 配置收紧（`src/core/config.py`）
  - 质量门槛：`yolo_conf=0.5`，`min_track_length=15`，`sample_interval=5`。
  - VLM 空间：`vlm_max_new_tokens=1024`，`vlm_context_size=8192`（显存足够时），`max_preview_tracks=10`。

## 期望效果
- 颜色/服饰类查询：均匀采样保证 VLM 看到中段清晰帧，红衣/蓝裤更容易命中。
- 动作/方向类查询：小地图 + 叙事标签让 4B 模型用“看图题”理解快慢/走向。
- 噪声过滤：短闪烁和低置信度轨迹被挡在入口，召回/终审更聚焦。

## 建议验证
- 跑 `python -m pipeline.video_semantic_search`，测试：
  - 外观：如“找红衣服的人”——检查均匀采样帧是否明显展示红色。
  - 动作/方向：如“往右跑的人”——检查日志中 prompt 是否包含小地图说明，输出是否提到速度/方向。
- 如需进一步调优：`sample_interval` 在 5–10 之间微调，`min_track_length` 视场景密度在 15–30 之间调节。 
