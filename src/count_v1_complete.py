# Version 1: 完整的进出统计系统
# 目标：跟踪 + 统计线 + 进出计数

import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import create_tracker

print("=" * 70)
print("🎯 进出统计系统：跟踪 → 判断穿越 → 统计")
print("=" * 70)

# ===== 配置区域 =====
VIDEO_PATH = Path("/Users/neko_wen/my/代码/uw/cv/project/data/snippets/debug_15s.mp4")
OUTPUT_DIR = Path("output_counting")

# 统计线定义（竖直线，在画面中央）
# 格式：(x, y) 两个点定义一条线
LINE_START = (960, 0)      # 画面中央，从顶部
LINE_END = (960, 1080)     # 到底部
# 左边 = 进入前，右边 = 进入后

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

# ===== 2. 打开视频 =====
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    print(f"❌ 无法打开视频")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n📹 视频信息:")
print(f"   分辨率: {width}x{height}")
print(f"   总帧数: {total_frames}")

print(f"\n📐 统计线设置:")
print(f"   起点: {LINE_START}")
print(f"   终点: {LINE_END}")
print(f"   规则: 左→右=进入, 右→左=离开")

# ===== 3. 准备输出 =====
output_video = OUTPUT_DIR / "result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

# ===== 4. 穿越判断函数 =====
def cross_product(line_start, line_end, point):
    """计算叉积"""
    x1, y1 = line_start
    x2, y2 = line_end
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def get_point_side(line_start, line_end, point):
    """判断点在直线的哪一边"""
    cross = cross_product(line_start, line_end, point)
    if cross > 0:
        return "left"
    elif cross < 0:
        return "right"
    else:
        return "on_line"

# ===== 5. 统计数据结构 =====
track_history = {}      # {track_id: [(center_x, center_y), ...]}
track_crossed = {}      # {track_id: 最后穿越时间（帧号）}
enter_count = 0         # 进入计数
leave_count = 0         # 离开计数
crossing_events = []    # 穿越事件记录

# 防重复计数：同一ID在60帧内只计数一次
COOLDOWN_FRAMES = 60

# ===== 6. 主循环 =====
frame_count = 0
start_time = time.time()

print("\n🚀 开始处理...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 画统计线
    cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 3)
    
    # YOLO检测
    results = model.predict(
        source=frame,
        device="mps",
        conf=0.3,
        classes=[0],
        verbose=False
    )
    
    # 提取检测
    detections = []
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls_id])
    
    # 跟踪
    if len(detections) > 0:
        detections = np.array(detections)
        tracks = tracker.update(detections, frame)
        
        if tracks.size > 0:
            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                conf = float(track[5])
                
                # 计算中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)
                
                # 记录轨迹历史
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(center)
                
                # 只保留最近10个点
                if len(track_history[track_id]) > 10:
                    track_history[track_id].pop(0)
                
                # ===== 穿越判断 =====
                if len(track_history[track_id]) >= 2:
                    prev_center = track_history[track_id][-2]
                    curr_center = track_history[track_id][-1]
                    
                    prev_side = get_point_side(LINE_START, LINE_END, prev_center)
                    curr_side = get_point_side(LINE_START, LINE_END, curr_center)
                    
                    # 检查冷却时间（防止重复计数）
                    last_crossed_frame = track_crossed.get(track_id, -9999)
                    can_count = (frame_count - last_crossed_frame) > COOLDOWN_FRAMES
                    
                    crossing_type = None
                    
                    # 左→右：进入
                    if prev_side == "left" and curr_side == "right" and can_count:
                        enter_count += 1
                        track_crossed[track_id] = frame_count
                        crossing_type = "ENTER"
                        crossing_events.append({
                            "frame": frame_count,
                            "track_id": track_id,
                            "type": "enter",
                            "position": center
                        })
                    
                    # 右→左：离开
                    elif prev_side == "right" and curr_side == "left" and can_count:
                        leave_count += 1
                        track_crossed[track_id] = frame_count
                        crossing_type = "LEAVE"
                        crossing_events.append({
                            "frame": frame_count,
                            "track_id": track_id,
                            "type": "leave",
                            "position": center
                        })
                    
                    # 可视化穿越事件
                    if crossing_type:
                        color = (0, 255, 0) if crossing_type == "ENTER" else (0, 0, 255)
                        cv2.putText(frame, crossing_type, 
                                   (center_x - 30, center_y - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                # 画边界框
                color = (0, 255, 0) if get_point_side(LINE_START, LINE_END, center) == "left" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 画轨迹
                if len(track_history[track_id]) > 1:
                    points = np.array(track_history[track_id], dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 2)
    
    # ===== 显示统计信息 =====
    current_inside = enter_count - leave_count
    
    stats_bg = np.zeros((150, 400, 3), dtype=np.uint8)
    stats_bg[:] = (50, 50, 50)
    
    cv2.putText(stats_bg, f"Enter: {enter_count}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(stats_bg, f"Leave: {leave_count}", (20, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(stats_bg, f"Inside: {current_inside}", (20, 130),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    
    frame[20:170, 20:420] = stats_bg
    
    # 标注统计线两侧
    cv2.putText(frame, "OUTSIDE", (LINE_START[0]-180, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, "INSIDE", (LINE_END[0]+20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    out.write(frame)
    
    if frame_count % 30 == 0:
        print(f"⏳ 帧:{frame_count}/{total_frames} | 进入:{enter_count} | 离开:{leave_count} | 在场:{current_inside}")

# ===== 7. 清理和保存 =====
cap.release()
out.release()

total_time = time.time() - start_time

print("\n" + "=" * 70)
print("✅ 统计完成！")
print("=" * 70)

print(f"\n📊 最终统计:")
print(f"   进入人数: {enter_count}")
print(f"   离开人数: {leave_count}")
print(f"   当前在场: {enter_count - leave_count}")
print(f"   穿越事件: {len(crossing_events)} 次")

print(f"\n⏱️  性能:")
print(f"   总耗时: {total_time:.1f} 秒")
print(f"   处理速度: {frame_count / total_time:.2f} FPS")

# 保存统计数据
stats = {
    "video": VIDEO_PATH.name,
    "total_frames": frame_count,
    "enter_count": enter_count,
    "leave_count": leave_count,
    "current_inside": enter_count - leave_count,
    "crossing_events": crossing_events,
    "line_definition": {
        "start": LINE_START,
        "end": LINE_END
    }
}

stats_file = OUTPUT_DIR / "statistics.json"
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n📁 输出文件:")
print(f"   视频: {output_video}")
print(f"   统计: {stats_file}")

print("\n" + "=" * 70)
print("🎓 理解要点:")
print("   1. 统计线把画面分成两部分")
print("   2. 轨迹穿越统计线时触发计数")
print("   3. 左→右=进入，右→左=离开")
print("   4. 防重复计数：60帧冷却时间")
print("=" * 70)

