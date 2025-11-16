# 完整的语义查询系统
# 目标：跟踪 → 提取人物 → CLIP编码 → 语义查询

import cv2
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from boxmot import create_tracker
from transformers import CLIPProcessor, CLIPModel

print("=" * 70)
print("🎯 语义查询系统：用人话找人")
print("=" * 70)

# ===== 配置 =====
VIDEO_PATH = Path("/Users/neko_wen/my/代码/uw/cv/project/data/snippets/debug_15s.mp4")
OUTPUT_DIR = Path("output_semantic")
SAMPLE_FRAMES = 5  # 每个track_id采样几帧

OUTPUT_DIR.mkdir(exist_ok=True)

# ===== 第一步：跟踪 + 提取人物图片 =====
print("\n" + "=" * 70)
print("📹 第一步：跟踪并提取人物图片")
print("=" * 70)

# 初始化
model = YOLO("yolov8n.pt")
tracker = create_tracker(
    tracker_type='bytetrack',
    tracker_config=None,
    reid_weights=None,
    device='cpu',
    half=False,
    per_class=False
)

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n📹 视频: {VIDEO_PATH.name}")
print(f"   总帧数: {total_frames}")

# 存储每个track_id的图片
track_images = {}  # {track_id: [图片1, 图片2, ...]}
track_frames = {}  # {track_id: [帧号1, 帧号2, ...]}

frame_count = 0
crops_dir = OUTPUT_DIR / "crops"
crops_dir.mkdir(exist_ok=True)

print("\n🚀 开始跟踪和提取...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # YOLO检测
    results = model.predict(
        source=frame,
        device="mps",
        conf=0.3,
        classes=[0],
        verbose=False
    )
    
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
                
                # 初始化track_id
                if track_id not in track_images:
                    track_images[track_id] = []
                    track_frames[track_id] = []
                
                # 采样：每个ID只保存SAMPLE_FRAMES张图
                if len(track_images[track_id]) < SAMPLE_FRAMES:
                    # 裁剪人物图片
                    # 扩大边界框10%
                    pad = 10
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(width, x2 + pad)
                    y2 = min(height, y2 + pad)
                    
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        # 保存
                        track_images[track_id].append(crop)
                        track_frames[track_id].append(frame_count)
                        
                        # 保存到文件
                        img_path = crops_dir / f"id{track_id:03d}_frame{frame_count:04d}.jpg"
                        cv2.imwrite(str(img_path), crop)
    
    if frame_count % 50 == 0:
        print(f"⏳ 处理帧:{frame_count}/{total_frames} | 已提取ID数:{len(track_images)}")

cap.release()

print(f"\n✅ 提取完成！")
print(f"   唯一ID数: {len(track_images)}")
print(f"   总图片数: {sum(len(imgs) for imgs in track_images.values())}")

# ===== 第二步：CLIP特征提取 =====
print("\n" + "=" * 70)
print("🧠 第二步：CLIP特征提取")
print("=" * 70)

print("\n📦 加载CLIP模型...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP加载完成")

# 为每个track_id的每张图片生成特征向量
features_db = {}  # {track_id: [向量1, 向量2, ...]}

print("\n🔄 提取特征向量...")
for track_id, images in track_images.items():
    features_db[track_id] = []
    
    for img in images:
        # OpenCV BGR → PIL RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # CLIP编码
        inputs = clip_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features_db[track_id].append(image_features[0].numpy())
    
    if track_id % 5 == 0:
        print(f"   处理ID:{track_id} ({len(images)}张图)")

print(f"\n✅ 特征提取完成！")

# 保存特征数据库
features_file = OUTPUT_DIR / "features.npz"
np.savez(features_file, **{f"id_{k}": np.array(v) for k, v in features_db.items()})
print(f"   特征库已保存: {features_file}")

# 保存元数据
metadata = {
    "track_ids": list(track_images.keys()),
    "num_images_per_id": {k: len(v) for k, v in track_images.items()},  # 修复：改为.items()
    "video": str(VIDEO_PATH)
}
with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# ===== 第三步：语义查询 =====
print("\n" + "=" * 70)
print("🔍 第三步：语义查询")
print("=" * 70)

def search(query_text, top_k=5, threshold=0.25):
    """
    语义查询函数
    
    参数:
        query_text: 查询文本（如："穿红色衣服的人"）
        top_k: 返回前k个结果
        threshold: 相似度阈值（只返回高于此值的结果）
    """
    # 编码查询文本
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features[0].numpy()
    
    # 计算每个track_id的相似度
    results = []
    for track_id, feature_list in features_db.items():
        # 对每个ID的多张图片，取最大相似度
        similarities = []
        for img_feature in feature_list:
            sim = np.dot(text_features, img_feature)
            similarities.append(sim)
        
        max_sim = max(similarities)
        avg_sim = np.mean(similarities)
        
        # 🔥 关键改进：只保留相似度高于阈值的结果
        if max_sim >= threshold:
            results.append({
                "track_id": track_id,
                "max_similarity": float(max_sim),
                "avg_similarity": float(avg_sim),
                "num_images": len(feature_list),
                "confidence": "high" if max_sim > 0.35 else "medium"
            })
    
    # 按相似度排序
    results.sort(key=lambda x: x["max_similarity"], reverse=True)
    
    # 返回结果（如果没找到，返回空列表）
    return results[:top_k]

# 测试查询
print("\n🧪 测试查询：")
print("-" * 70)

test_queries = [
    "a person wearing red clothes",
    "a person wearing blue pants",
    "a person with a backpack",
    "a person wearing white shirt",
    "a person wearing purple hat",  # 可能找不到
]

for query in test_queries:
    print(f"\n查询: \"{query}\"")
    results = search(query, top_k=3, threshold=0.25)
    
    if len(results) == 0:
        print("  ❌ 未找到匹配结果（相似度均低于阈值0.25）")
    else:
        for i, result in enumerate(results, 1):
            track_id = result["track_id"]
            similarity = result["max_similarity"]
            confidence = result["confidence"]
            emoji = "✅" if confidence == "high" else "⚠️"
            print(f"  {emoji} {i}. ID:{track_id:3d} | 相似度:{similarity:.3f} ({confidence}) | "
                  f"{len(track_images[track_id])}张图")

# ===== 第四步：可视化查询结果 =====
print("\n" + "=" * 70)
print("📊 第四步：可视化查询结果")
print("=" * 70)

def visualize_search_result(query, top_k=5, threshold=0.25):
    """可视化查询结果"""
    results = search(query, top_k, threshold)
    
    # 创建结果图
    result_images = []
    for result in results:
        track_id = result["track_id"]
        similarity = result["max_similarity"]
        
        # 取该ID的第一张图
        img = track_images[track_id][0].copy()
        
        # 添加文字标注
        label = f"ID:{track_id} ({similarity:.2f})"
        cv2.putText(img, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        result_images.append(img)
    
    # 拼接成一张大图
    if result_images:
        # 统一尺寸
        h, w = 200, 150
        resized = [cv2.resize(img, (w, h)) for img in result_images]
        
        # 横向拼接
        result_img = np.hstack(resized)
        
        # 保存
        query_safe = query.replace(" ", "_")[:30]
        output_path = OUTPUT_DIR / f"query_{query_safe}.jpg"
        cv2.imwrite(str(output_path), result_img)
        
        print(f"✅ 查询结果已保存: {output_path}")
        return output_path
    
    return None

# 可视化一个查询
query_example = "a person wearing red clothes"
print(f"\n可视化查询: \"{query_example}\"")
visualize_search_result(query_example)

# ===== 总结 =====
print("\n" + "=" * 70)
print("✅ 语义查询系统构建完成！")
print("=" * 70)

print(f"\n📊 系统统计:")
print(f"   索引的ID数: {len(features_db)}")
print(f"   总图片数: {sum(len(imgs) for imgs in track_images.values())}")
print(f"   特征向量维度: 512")

print(f"\n📁 输出文件:")
print(f"   人物图片: {crops_dir}/")
print(f"   特征数据库: {features_file}")
print(f"   查询结果: {OUTPUT_DIR}/")

print(f"\n💡 使用方法:")
print(f"   1. 加载特征数据库")
print(f"   2. 调用 search(\"你的查询\")")
print(f"   3. 获得匹配的track_id列表")

print("\n" + "=" * 70)
print("🎓 核心流程回顾:")
print("=" * 70)
print("""
1. 跟踪视频 → 为每个人分配ID
2. 提取图片 → 每个ID采样若干张
3. CLIP编码 → 图片变成512维向量
4. 保存特征库 → 建立索引
5. 查询 → 文字编码成向量 → 匹配 → 返回ID
""")

print("=" * 70)
print("🎯 现在你可以用自然语言找人了！")
print("   例如: \"穿红色衣服的人\"")
print("   例如: \"背着背包的人\"")
print("   例如: \"穿蓝色裤子的人\"")
print("=" * 70)

