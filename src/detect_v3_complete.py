# Version 3: 完整版 - 视频检测 + 数据保存
# 新增功能：
#   1. 保存检测结果为CSV（机器可读）
#   2. 性能统计（每帧耗时、检测数量）
#   3. 跳帧优化

import cv2
import time
import csv
from pathlib import Path
from ultralytics import YOLO

# ===== 配置区域 =====
VIDEO_PATH = "/Users/neko_wen/my/代码/uw/cv/project/data/snippets/debug_15s.mp4"
OUTPUT_DIR = Path("output_detect")  # 输出目录
SKIP_FRAMES = 2  # 每隔几帧检测一次（1=不跳帧，2=跳一半）
CONF_THRESHOLD = 0.3  # 置信度阈值

# ===== 1. 初始化 =====
print("=" * 60)
print("🔄 正在加载YOLO模型...")
model = YOLO("yolov8n.pt")
print("✅ 模型加载完成！")
print("=" * 60 + "\n")

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== 2. 打开视频 =====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 错误：无法打开视频 {VIDEO_PATH}")
    exit()

# 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("📹 视频信息:")
print(f"   文件: {Path(VIDEO_PATH).name}")
print(f"   分辨率: {width}x{height}")
print(f"   帧率: {fps} FPS")
print(f"   总帧数: {total_frames}")
print(f"   时长: {total_frames/fps:.1f} 秒")
print(f"\n⚙️  处理设置:")
print(f"   跳帧: 每 {SKIP_FRAMES} 帧处理一次")
print(f"   实际处理: {total_frames // SKIP_FRAMES} 帧")
print(f"   置信度: {CONF_THRESHOLD}\n")

# ===== 3. 准备输出文件 =====
# 3.1 输出视频
output_video = OUTPUT_DIR / "result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

if not out.isOpened():
    print("⚠️  avc1编码器失败，尝试mp4v...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

# 3.2 CSV文件（保存检测数据）
csv_file = OUTPUT_DIR / "detections.csv"
f_csv = open(csv_file, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(f_csv)
# CSV表头
csv_writer.writerow([
    'frame_id',      # 帧号
    'object_id',     # 该帧中第几个物体
    'class_name',    # 类别名称（person/car/...）
    'confidence',    # 置信度
    'x1', 'y1',      # 左上角坐标
    'x2', 'y2',      # 右下角坐标
    'width', 'height' # 边界框宽高
])

# ===== 4. 核心循环 =====
frame_count = 0
detect_count = 0  # 总检测数
total_objects = 0  # 总目标数

start_time = time.time()
print("🚀 开始处理...\n")

# 用于存储上一次检测的帧（跳帧时复用）
last_annotated = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 判断是否需要检测这一帧
    if frame_count % SKIP_FRAMES == 0 or frame_count == 1:
        # ===== 执行检测 =====
        detect_count += 1
        
        results = model.predict(
            source=frame,
            device="mps",
            conf=CONF_THRESHOLD,
            verbose=False
        )
        
        # 获取检测结果
        boxes = results[0].boxes
        num_objects = len(boxes) if boxes is not None else 0
        total_objects += num_objects
        
        # 画框
        annotated_frame = results[0].plot()
        last_annotated = annotated_frame.copy()
        
        # ===== 保存检测数据到CSV =====
        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                # 提取数据
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                
                # 写入CSV
                csv_writer.writerow([
                    frame_count,
                    idx + 1,
                    cls_name,
                    f"{conf:.3f}",
                    int(x1), int(y1),
                    int(x2), int(y2),
                    int(x2 - x1), int(y2 - y1)
                ])
        
        # 进度显示
        if detect_count % 5 == 0:
            elapsed = time.time() - start_time
            fps_proc = detect_count / elapsed
            eta = (total_frames // SKIP_FRAMES - detect_count) / fps_proc
            print(f"⏳ 帧:{frame_count}/{total_frames} | "
                  f"检测:{detect_count} | "
                  f"速度:{fps_proc:.1f} FPS | "
                  f"检出:{num_objects}个 | "
                  f"剩余:{eta:.0f}s")
    
    else:
        # 跳帧：使用上一帧的检测结果
        annotated_frame = last_annotated if last_annotated is not None else frame
    
    # 写入输出视频
    out.write(annotated_frame)

# ===== 5. 清理和统计 =====
cap.release()
out.release()
f_csv.close()

total_time = time.time() - start_time
avg_fps = detect_count / total_time

print("\n" + "=" * 60)
print("✅ 处理完成！")
print("=" * 60)
print(f"\n📊 统计信息:")
print(f"   总帧数: {frame_count}")
print(f"   检测帧数: {detect_count} (跳过 {frame_count - detect_count} 帧)")
print(f"   检测到目标: {total_objects} 个")
print(f"   平均每帧: {total_objects / detect_count:.1f} 个")
print(f"\n⏱️  性能:")
print(f"   总耗时: {total_time:.1f} 秒")
print(f"   平均速度: {avg_fps:.2f} FPS")
print(f"   加速比: {avg_fps / fps:.2f}x")
print(f"\n📁 输出文件:")
print(f"   视频: {output_video}")
print(f"   数据: {csv_file}")
print(f"   大小: 视频 {output_video.stat().st_size / 1024 / 1024:.1f} MB")
print(f"         数据 {csv_file.stat().st_size / 1024:.1f} KB")
print("=" * 60)

