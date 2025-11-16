# Version 2: 完整跟踪系统
# 目标：YOLO检测 + 跟踪 + 保存MOT格式轨迹文件

import cv2
import csv
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import create_tracker

print("=" * 70)
print("🎯 完整跟踪系统：检测 → 跟踪 → 保存轨迹")
print("=" * 70)

# ===== 配置区域 =====
VIDEO_PATH = Path("/Users/neko_wen/my/代码/uw/cv/project/data/snippets/debug_15s.mp4")
OUTPUT_DIR = Path("output_track")
SKIP_FRAMES = 1  # 跟踪不建议跳帧（会影响ID连续性）

# ===== 1. 初始化 =====
print("\n📦 初始化...")
OUTPUT_DIR.mkdir(exist_ok=True)

model = YOLO("yolov8n.pt")
tracker = create_tracker(
    tracker_type='bytetrack',
    tracker_config=None,
    reid_weights=None,
    device='cpu',
    half=False,
    per_class=False
)

print("✅ 模型和跟踪器加载完成")

# ===== 2. 打开视频 =====
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    print(f"❌ 无法打开视频: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n📹 视频信息:")
print(f"   文件: {VIDEO_PATH.name}")
print(f"   分辨率: {width}x{height}")
print(f"   帧率: {fps} FPS")
print(f"   总帧数: {total_frames}")

# ===== 3. 准备输出文件 =====
# 3.1 可视化视频
output_video = OUTPUT_DIR / "result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

# 3.2 MOT格式轨迹文件
mot_file = OUTPUT_DIR / "tracks.txt"
f_mot = open(mot_file, 'w')

# MOT格式说明：
# 每行格式：<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
# frame: 帧号（从1开始）
# id: track_id
# bb_left, bb_top: 左上角坐标
# bb_width, bb_height: 宽高
# conf: 置信度
# x,y,z: 3D坐标（2D视频设为-1）

# 3.3 详细CSV（可选，便于分析）
csv_file = OUTPUT_DIR / "tracks_detail.csv"
f_csv = open(csv_file, 'w', newline='')
csv_writer = csv.writer(f_csv)
csv_writer.writerow([
    'frame_id', 'track_id', 'class_name', 'confidence',
    'x1', 'y1', 'x2', 'y2', 'width', 'height',
    'center_x', 'center_y'
])

# ===== 4. 跟踪主循环 =====
frame_count = 0
track_ids_seen = set()
total_tracks = 0

start_time = time.time()
print("\n🚀 开始跟踪...\n")

# 用于给不同ID分配不同颜色
def get_color(track_id):
    """根据track_id生成固定颜色"""
    np.random.seed(int(track_id))
    return tuple(map(int, np.random.randint(0, 255, 3)))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 跳帧处理
    if (frame_count - 1) % SKIP_FRAMES != 0 and frame_count != 1:
        out.write(frame)
        continue
    
    # ===== 步骤1: YOLO检测 =====
    results = model.predict(
        source=frame,
        device="mps",
        conf=0.3,
        classes=[0],  # 只检测person
        verbose=False
    )
    
    # ===== 步骤2: 提取检测结果 =====
    detections = []
    boxes = results[0].boxes
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls_id])
    
    # ===== 步骤3: 跟踪器更新 =====
    if len(detections) > 0:
        detections = np.array(detections)
        tracks = tracker.update(detections, frame)
        
        if tracks.size > 0:
            total_tracks += len(tracks)
            
            # ===== 步骤4: 保存和可视化 =====
            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                conf = float(track[5])
                cls_id = int(track[6])
                
                # 记录ID
                track_ids_seen.add(track_id)
                
                # 计算宽高和中心
                bb_width = x2 - x1
                bb_height = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # ===== 保存MOT格式 =====
                # 格式：<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1
                mot_line = f"{frame_count},{track_id},{x1},{y1},{bb_width},{bb_height},{conf:.3f},-1,-1,-1\n"
                f_mot.write(mot_line)
                
                # ===== 保存详细CSV =====
                csv_writer.writerow([
                    frame_count, track_id, 'person', f"{conf:.3f}",
                    x1, y1, x2, y2, bb_width, bb_height,
                    center_x, center_y
                ])
                
                # ===== 可视化 =====
                color = get_color(track_id)
                
                # 画边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 画轨迹点（中心）
                cv2.circle(frame, (center_x, center_y), 3, color, -1)
                
                # 显示ID和置信度
                label = f"ID:{track_id} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 文字背景
                cv2.rectangle(frame, 
                             (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1),
                             color, -1)
                
                # 文字
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 显示统计信息
    info_text = [
        f"Frame: {frame_count}/{total_frames}",
        f"Unique IDs: {len(track_ids_seen)}",
        f"Current: {len(tracks) if len(detections) > 0 and tracks.size > 0 else 0}"
    ]
    
    y_offset = 30
    for text in info_text:
        cv2.putText(frame, text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 35
    
    out.write(frame)
    
    # 进度显示
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_proc = frame_count / elapsed
        eta = (total_frames - frame_count) / fps_proc
        print(f"⏳ 帧:{frame_count}/{total_frames} | "
              f"唯一ID:{len(track_ids_seen)} | "
              f"速度:{fps_proc:.1f} FPS | "
              f"剩余:{eta:.0f}s")

# ===== 5. 清理 =====
cap.release()
out.release()
f_mot.close()
f_csv.close()

total_time = time.time() - start_time

print("\n" + "=" * 70)
print("✅ 跟踪完成！")
print("=" * 70)

# ===== 6. 统计分析 =====
print(f"\n📊 跟踪统计:")
print(f"   总帧数: {frame_count}")
print(f"   唯一ID数: {len(track_ids_seen)}")
print(f"   ID列表: {sorted(track_ids_seen)}")
print(f"   总跟踪数: {total_tracks}")
print(f"   平均每帧: {total_tracks / frame_count:.1f} 个目标")

print(f"\n⏱️  性能:")
print(f"   总耗时: {total_time:.1f} 秒")
print(f"   平均速度: {frame_count / total_time:.2f} FPS")

print(f"\n📁 输出文件:")
print(f"   视频: {output_video}")
print(f"   MOT轨迹: {mot_file}")
print(f"   详细CSV: {csv_file}")

print(f"\n💡 数据说明:")
print(f"   - MOT格式: 标准的多目标跟踪数据格式")
print(f"   - 可用于评估算法、后续分析")
print(f"   - 每行代表一个目标在一帧中的位置")

# ===== 7. 轨迹分析 =====
print(f"\n📈 轨迹分析:")

# 读取MOT文件统计每个ID的轨迹长度
track_lengths = {}
with open(mot_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        track_id = int(parts[1])
        track_lengths[track_id] = track_lengths.get(track_id, 0) + 1

print(f"   最长轨迹: ID:{max(track_lengths, key=track_lengths.get)} "
      f"({max(track_lengths.values())} 帧 = {max(track_lengths.values())/fps:.1f}秒)")
print(f"   最短轨迹: ID:{min(track_lengths, key=track_lengths.get)} "
      f"({min(track_lengths.values())} 帧 = {min(track_lengths.values())/fps:.1f}秒)")
print(f"   平均轨迹长度: {sum(track_lengths.values()) / len(track_lengths):.1f} 帧")

print("\n" + "=" * 70)
print("🎓 学习要点:")
print("   1. track_id 是每个人的唯一身份证")
print("   2. MOT格式是标准的轨迹数据格式")
print("   3. 轨迹长度 = 该人在视频中出现的时长")
print("=" * 70)

