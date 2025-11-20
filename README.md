# 🚀 Edge‑Detective v7（M4 实施版）

基于 YOLO + ByteTrack + Atomic 8 + 单 Qwen3‑VL‑4B VLM + SigLIP 的视频行人语义检索系统。  
支持在本地 Mac M 系列（特别是 M4 16GB）上，按自然语言问题在视频中“找人/找行为”。

---

## 📌 项目特点（v7 架构）

- ✅ **真正 v7**：单个 Qwen3‑VL‑4B‑Instruct‑GGUF（Int4）同时担任 Router + Verifier。  
- ✅ **SigLIP 召回**：`google/siglip-base-patch16-224` 做高召回粗筛，减少 VLM 压力。  
- ✅ **Atomic 8 协议**：所有行为判断都建立在几何“原子事实”之上（时间、速度、位移、轨迹等）。  
- ✅ **四层流水线**：Router → Recall → Hard Rules → Verifier，清晰解耦。  
- ✅ **自动下模型**：使用 `huggingface_hub` 自动下载 `unsloth/Qwen3-VL-4B-Instruct-GGUF`，无需手动找链接。

---

## 🧱 核心流水线（v7）

1. **Perception（感知）**  
   - 技术：YOLOv11（人检测）+ ByteTrack（多目标跟踪）。  
   - 输出：`TrackRecord` + `VideoMetadata`（`src/core/perception.py`）。

2. **Features（几何特征 / Atomic 8）**  
   - 技术：几何运算 + 轨迹插值。  
   - 输出：`TrackFeatures`（带 `start_s/end_s/centroids/displacement_vec/avg_speed/...`），在 `src/core/features.py`。

3. **Evidence（证据包）**  
   - 技术：数据打包。  
   - 输出：`EvidencePackage`（轨迹 + Atomic 8 + crops/meta/raw_trace/embedding），在 `src/core/evidence.py`。

4. **Router（规划层）**  
   - 默认：`HFRouter` 直接使用 `Qwen/Qwen3-VL-4B-Instruct`（transformers）解析自然语言问题 → `ExecutionPlan`（`src/pipeline/router_llm.py`）；  
   - 未来若需要 GGUF / llama-cpp，可在此接口上扩展，但当前实现已经完全由 VLM 端到端负责语义规划。

5. **Recall（SigLIP 粗筛）**  
   - 技术：`google/siglip-base-patch16-224` → 图文 embedding 相似度。  
   - 输出：Top‑K 候选轨迹列表，`src/pipeline/recall.py`。

6. **Hard Rules（几何会计师）**  
   - 技术：在 Atomic 8 空间执行 ROI / 时间窗 / 排序 / 阈值等规则。  
   - 输出：满足约束的少量轨迹，`src/core/hard_rules.py`。

7. **Verifier（终审）**  
   - 技术：同一个 Qwen3‑VL‑4B（transformers）模型，看多张 crops + Atomic 8 摘要，对每条轨迹做 Yes/No 判定并给出 reason。  
   - 输出：`QueryResult(track_id, start_s, end_s, score, reason)` 列表，`src/pipeline/vlm_client_hf.py`。

8. **VideoSemanticSystem（总 orchestrator）**  
   - 入口：`src/pipeline/video_semantic_search.py`  
   - API：  
     - `build_index()`：跑 Perception + Features + Evidence，写出 `semantic_database.json`。  
     - `question_search(question)`：跑 Router → Recall → Hard Rules → Verifier，并导出高亮视频。  

---

## 🛠️ 环境配置

### 硬件要求
- **CPU**：Intel/AMD x86_64 或 Apple Silicon (M1/M2/M4)
- **内存**：16GB 推荐
- **GPU**：可选（NVIDIA CUDA / Apple MPS）

### 软件依赖（v7）

```bash
# 创建虚拟环境
conda create -n mvsys-py311 python=3.11
conda activate mvsys-py311

# 安装 v7 依赖
pip install -r requirements.txt
```

> 注意：首次运行需要从 Hugging Face 下载 Qwen3-VL-4B，确保网络通畅或提前配置镜像。

---

## 🚀 快速开始

### 1. 准备视频数据
```bash
# 将视频放入data/snippets/目录
cp your_video.mp4 data/snippets/debug_15s.mp4
```

### 2. 运行 v7 全流程（单视频 Demo）

1. 编辑 `src/core/config.py`（至少改两项）：  
   ```python
   video_path: Path = Path("data/snippets/debug_15s.mp4")  # 你的输入视频
   output_dir: Path = Path("output")                       # 输出目录

   vlm_backend: str = "hf"
   router_backend: str = "hf"
   ```

2. 运行 demo（项目根目录）：  
   ```bash
   export PYTHONPATH=src  # Windows 使用 set PYTHONPATH=src
   python -m pipeline.video_semantic_search
   ```

   首次运行时：
   - 会自动从 Hugging Face 下载：  
     - `unsloth/Qwen3-VL-4B-Instruct-GGUF`（GGUF 文件，用于 Router + Verifier）  
     - `google/siglip-base-patch16-224`（SigLIP 召回模型）  
   - 会在 `output/` 下生成：  
     - `semantic_database.json`（索引数据库）  
     - `crops/`（轨迹裁剪图）  
     - `embeddings/<video_id>/track_*.npy`（SigLIP embedding cache）  
     - `tracking_找出穿紫色衣服的人.mp4`（高亮结果视频）

---

## 📊 性能指标

| 阶段 | 处理速度 (15秒视频) | 准确率 | 硬件 |
|------|-------------------|--------|------|
| **检测** | 约30秒 | 高（YOLOv8） | M4 MPS |
| **跟踪** | 约45秒 | 高（ByteTrack） | M4 MPS |
| **统计** | 约50秒 | 高（几何算法） | M4 MPS |
| **CLIP查询** | 3-5秒/35人 | 20-30% | M4 MPS |
| **VLM查询** | 60-90秒/35人 | 70-85% | M4 MPS |

---

## 📁 项目结构

```
project/
├── src/
│   ├── core/                     # 协议 & 底层组件
│   │   ├── config.py             # SystemConfig（视频路径、VLM GGUF、SigLIP 等）
│   │   ├── perception.py         # YOLO + ByteTrack → TrackRecord, VideoMetadata
│   │   ├── features.py           # TrackFeatures, TrackFeatureExtractor（Atomic 8）
│   │   ├── evidence.py           # EvidencePackage, build_evidence_packages
│   │   ├── behavior.py           # BehaviorFeatureExtractor, EventDetector
│   │   ├── hard_rules.py         # HardRuleEngine
│   │   ├── siglip_client.py      # SigLIP 封装
│   │   └── vlm_types.py          # QueryResult
│   └── pipeline/                 # 高层流水线（v7）
│       ├── router.py             # ExecutionPlan schema + parse_router_output
│       ├── router_llm.py         # HFRouter（Qwen3-VL-4B transformers 规划）
│       ├── recall.py             # RecallEngine（SigLIP 粗筛）
│       ├── vlm_client_hf.py      # Qwen3VL4BHFClient（Verifier）
│       └── video_semantic_search.py  # VideoSemanticSystem（入口）
│
├── data/                         # 数据目录
│   └── snippets/                 # 测试视频
│
├── docs/                         # 文档
│   ├── project_summary.md        # 项目总结
│   ├── vlm_guide.md              # VLM使用指南
│   └── vlm_mps_guide.md          # M4优化指南
│
├── requirements.txt              # 基础依赖
├── requirements_vlm.txt          # VLM扩展依赖
└── README.md                     # 本文件
```

---

## 🎓 核心技术栈（v7）

| 技术 | 版本 | 用途 |
|------|------|------|
| **YOLOv11** (ultralytics) | - | 人体检测 |
| **ByteTrack** (boxmot) | - | 多目标跟踪 |
| **SigLIP** (`google/siglip-base-patch16-224`) | - | 视觉召回（图文 embedding） |
| **Qwen3‑VL‑4B‑Instruct** (`Qwen/...`) | transformers (MPS/CPU) | Router + Verifier |
| **PyTorch** | 2.0+ | YOLO / SigLIP 依赖 |
| **OpenCV** | 4.8+ | 视频 I/O、画框 |
| **NumPy** | 1.24+ | 几何计算 |

---

## 💡 使用示例

### 语义查询示例

```python
# CLIP查询
search("a person wearing red clothes")
search("a person with a backpack")
search("a person wearing blue pants")

# VLM查询
# 在semantic_vlm_vllm.py中修改test_queries
test_queries = [
    ("穿红色衣服的人", ["red"]),
    ("背背包的人", ["backpack", "bag"]),
    ("戴帽子的人", ["hat", "cap"]),
]
```

---

## 🐛 故障排除

### Q: Transformers 下载模型太慢？
```bash
export HF_ENDPOINT=https://hf-mirror.com  # 或者使用本地缓存
pip install -U huggingface_hub
```

### Q: ModuleNotFoundError
```bash
# 确保在正确的虚拟环境
conda activate mvsys-py311
pip install -r requirements.txt
```

### Q: MPS不可用
```bash
# 检查PyTorch MPS支持
python src/check_mps.py

# 如果不可用，代码会自动降级到CPU
```

### Q: 视频无法打开
```bash
# 检查视频编码
ffmpeg -i your_video.mp4

# 确保使用支持的格式（MP4/AVI）
```

---

## 📚 学习资源

### 论文
- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **ByteTrack**: [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Qwen2-VL**: [Qwen2-VL Technical Report](https://github.com/QwenLM/Qwen2-VL)

### 代码库
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [BoxMOT](https://github.com/mikel-brostrom/boxmot)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)

---

## 🔧 进阶开发

### 优化建议
1. **性能优化**：调整跳帧参数、批处理大小
2. **准确率提升**：Fine-tune模型、调整阈值
3. **功能扩展**：添加Re-ID、行为识别、多摄像头融合

### 自定义配置
- 修改 `SKIP_FRAMES` 调整处理速度
- 修改检测阈值 `conf=0.5` 调整灵敏度
- 修改统计线位置实现不同区域统计

---

## 📄 许可证

本项目仅供学习研究使用。

---

## 👥 作者

UW Computer Vision Project

---

## 🙏 致谢

感谢以下开源项目：
- Ultralytics YOLOv8
- BoxMOT
- OpenAI CLIP
- Alibaba Qwen2-VL
- OpenCV Community

---

**🎯 一个完整的、可扩展的、前沿的视频分析系统！**
