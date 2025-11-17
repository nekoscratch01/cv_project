# 🚀 智能行人分析系统

基于深度学习的视频分析系统，实现行人检测、跟踪、统计和语义查询。

---

## 📌 项目特点

- ✅ **完整Pipeline**：从视频输入到智能查询的端到端系统
- ✅ **前沿技术**：集成YOLOv8、ByteTrack、CLIP、VLM等前沿算法
- ✅ **M4优化**：针对Apple Silicon进行MPS加速优化
- ✅ **模块化设计**：每个阶段独立运行，便于调试和扩展

---

## 🎯 四大核心功能

### Stage 1: 目标检测
- **技术**：YOLOv8
- **功能**：识别视频中的每个行人
- **输出**：边界框、置信度、检测结果CSV

### Stage 2: 多目标跟踪  
- **技术**：YOLOv8 + ByteTrack
- **功能**：为每个人分配唯一ID并追踪移动
- **输出**：轨迹文件（MOT格式）、带ID的可视化视频

### Stage 3: 越线统计
- **技术**：几何算法（叉积）+ 状态机
- **功能**：统计穿过指定区域的人数
- **输出**：统计视频、JSON数据

### Stage 4: 语义查询
- **技术**：CLIP（快速）+ VLM（准确）
- **功能**：用自然语言查询特定行人
- **输出**：匹配结果、可视化图片

---

## 🛠️ 环境配置

### 硬件要求
- **CPU**：Intel/AMD x86_64 或 Apple Silicon (M1/M2/M4)
- **内存**：16GB 推荐
- **GPU**：可选（NVIDIA CUDA / Apple MPS）

### 软件依赖

```bash
# 创建虚拟环境
conda create -n mvsys-py311 python=3.11
conda activate mvsys-py311

# 安装基础依赖
pip install -r requirements.txt

# （可选）安装VLM扩展
pip install -r requirements_vlm.txt
```

---

## 🚀 快速开始

### 1. 准备视频数据
```bash
# 将视频放入data/snippets/目录
cp your_video.mp4 data/snippets/debug_15s.mp4
```

### 2. 运行完整流程

**Stage 1: 检测**
```bash
cd src
python detect_v3_complete.py
```
输出：`detections.csv`, `output_video.mp4`

**Stage 2: 跟踪**
```bash
python track_v2_complete.py
```
输出：`tracks.txt`, `tracks_detail.csv`, `track_result.mp4`

**Stage 3: 统计**
```bash
python count_v1_complete.py
```
输出：`count_result.mp4`, `count_stats.json`

**Stage 4: 语义查询**
```bash
# 方式1：CLIP（快速）
python semantic_search_complete.py

# 方式2：VLM（准确）
python semantic_vlm_vllm.py
```
输出：`output_semantic/` 目录下的查询结果

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
├── src/                          # 源代码
│   ├── detect_v3_complete.py     # Stage 1: 检测
│   ├── track_v2_complete.py      # Stage 2: 跟踪
│   ├── count_v1_complete.py      # Stage 3: 统计
│   ├── semantic_search_complete.py  # Stage 4: CLIP查询
│   ├── semantic_vlm_vllm.py      # Stage 4: VLM查询
│   └── learn_*.py                # 学习辅助脚本
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

## 🎓 核心技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **YOLOv8** | 8.0+ | 目标检测 |
| **ByteTrack** | - | 多目标跟踪 |
| **OpenCV** | 4.8+ | 视频I/O、图像处理 |
| **CLIP** | - | 图像-文本匹配 |
| **Qwen2-VL** | 2B | 视觉语言理解 |
| **PyTorch** | 2.0+ | 深度学习框架 |
| **NumPy** | 1.24+ | 数值计算 |

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



