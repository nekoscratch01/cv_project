"""
🎯 视频中找人系统 (Find Person in Video)
Video Person Search System

真实场景: 监控视频中根据描述找人
例如: "找出穿红色衣服的人" / "找出背背包的人"

作者: 一起学习的产物
日期: 2025-11
"""

import cv2
import torch
import json
import time
from pathlib import Path
from ultralytics import YOLO
from boxmot import create_tracker
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===== 配置 =====
VIDEO_PATH = Path("../data/snippets/debug_15s.mp4")
OUTPUT_DIR = Path("output_search_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 1. 视频处理：检测+跟踪+裁剪 =====
def process_video_and_extract_people(video_path, output_crops_dir):
    """
    处理视频，提取所有人物
    
    返回:
        people: {
            track_id: {
                "image": "crops/id001.jpg",  # 代表性图片
                "frames": [1, 2, 3, ...],    # 出现的帧号
                "first_bbox": (x1, y1, x2, y2)  # 第一次出现的位置
            }
        }
    """
    print("\n" + "=" * 70)
    print("📹 阶段1: 处理视频，提取所有人物")
    print("=" * 70)
    
    output_crops_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("\n🔧 加载YOLO和跟踪器...")
    yolo = YOLO("yolov8n.pt")
    tracker = create_tracker(
        tracker_type='bytetrack',
        tracker_config=None,
        reid_weights=None,
        device='cpu',
        half=False,
        per_class=False
    )
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   视频: {video_path.name}")
    print(f"   总帧数: {total_frames}")
    print(f"   帧率: {fps:.2f} FPS")
    
    people = {}  # 存储每个人的信息
    frame_idx = 0
    
    print("\n🔄 开始处理...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_idx / elapsed
            eta = (total_frames - frame_idx) / fps_processing
            print(f"   进度: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | "
                  f"速度: {fps_processing:.1f} FPS | ETA: {eta:.0f}秒", end='\r')
        
        # YOLO检测（只检测人）
        results = yolo.predict(
            source=frame,
            device="mps",
            conf=0.3,
            verbose=False,
            classes=[0]  # 只检测人（class 0）
        )[0]
        
        # 提取检测结果
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                detections.append([x1, y1, x2, y2, conf, cls])
        
        # ByteTrack跟踪
        if len(detections) > 0:
            detections = np.array(detections)
            tracks = tracker.update(detections, frame)
            
            if tracks.size > 0:
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    
                    # 记录这个人
                    if track_id not in people:
                        # 第一次出现，裁剪并保存图片
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_path = output_crops_dir / f"id{track_id:03d}.jpg"
                            cv2.imwrite(str(crop_path), crop)
                            
                            people[track_id] = {
                                "image": str(crop_path),
                                "frames": [frame_idx],
                                "first_bbox": (x1, y1, x2, y2),
                                "all_bboxes": [(frame_idx, x1, y1, x2, y2)]
                            }
                    else:
                        # 已经存在，只记录帧号
                        people[track_id]["frames"].append(frame_idx)
                        people[track_id]["all_bboxes"].append((frame_idx, x1, y1, x2, y2))
    
    cap.release()
    elapsed = time.time() - start_time
    
    print(f"\n\n✅ 视频处理完成！")
    print(f"   用时: {elapsed:.1f}秒")
    print(f"   检测到 {len(people)} 个不同的人")
    
    # 过滤出现太短的人（可能是误检）
    min_frames = 5
    filtered_people = {
        tid: data for tid, data in people.items()
        if len(data["frames"]) >= min_frames
    }
    
    print(f"   过滤后剩余 {len(filtered_people)} 个有效目标")
    
    return filtered_people


# ===== 2. VLM查询模块 =====
class VLMQueryEngine:
    """VLM查询引擎"""
    
    def __init__(self):
        print("\n🔧 加载VLM模型...")
        
        # 检测设备
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # 加载Qwen2-VL
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"   ✅ 模型加载完成 (设备: {self.device.upper()})")
    
    def ask(self, image_path, question):
        """对图片提问"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=30)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return answer.strip()


# ===== 3. 搜索引擎 =====
def search_person_by_description(people, vlm_engine, description):
    """
    根据描述在视频中找人
    
    参数:
        people: process_video_and_extract_people()返回的数据
        vlm_engine: VLM查询引擎
        description: 用户描述，例如"穿红色衣服的人"
    
    返回:
        匹配的track_id列表
    """
    print("\n" + "=" * 70)
    print(f"🔍 阶段2: 根据描述找人")
    print("=" * 70)
    print(f"   查询: {description}")
    print(f"   候选人数: {len(people)}\n")
    
    # 构建问题（根据描述类型）
    if "红色" in description or "蓝色" in description or "绿色" in description or "颜色" in description:
        question = "What is the main color of this person's clothing? Answer with one word only."
        target_keyword = None
        if "红色" in description:
            target_keyword = "red"
        elif "蓝色" in description:
            target_keyword = "blue"
        elif "绿色" in description:
            target_keyword = "green"
        elif "黑色" in description:
            target_keyword = "black"
        elif "白色" in description:
            target_keyword = "white"
    
    elif "背包" in description or "backpack" in description.lower():
        question = "Is this person carrying a backpack? Answer yes or no."
        target_keyword = "yes"
    
    elif "帽子" in description or "hat" in description.lower():
        question = "Is this person wearing a hat? Answer yes or no."
        target_keyword = "yes"
    
    else:
        # 通用描述，直接用自然语言
        question = f"Does this person match this description: '{description}'? Answer yes or no."
        target_keyword = "yes"
    
    print(f"   VLM问题: {question}")
    print(f"   匹配关键词: {target_keyword}\n")
    
    # 遍历所有人，逐个提问
    results = []
    
    for idx, (track_id, data) in enumerate(people.items(), 1):
        image_path = data["image"]
        
        print(f"[{idx}/{len(people)}] ID {track_id:3d} ... ", end='')
        
        # 提问VLM
        answer = vlm_engine.ask(image_path, question)
        answer_lower = answer.lower().strip()
        
        # 判断是否匹配
        is_match = False
        if target_keyword:
            is_match = target_keyword.lower() in answer_lower
        
        if is_match:
            results.append(track_id)
            print(f"✅ 匹配！ (回答: {answer})")
        else:
            print(f"❌ 不匹配 (回答: {answer})")
    
    return results


# ===== 4. 结果可视化 =====
def visualize_results(video_path, people, matched_ids, output_path):
    """
    可视化搜索结果
    
    1. 在视频中标注匹配的人
    2. 拼接匹配的人物图片
    """
    if not matched_ids:
        print("\n⚠️  没有找到匹配的人")
        return
    
    print(f"\n📊 可视化结果...")
    
    # 1. 拼接人物图片
    print("   生成人物拼图...")
    images = []
    for tid in matched_ids[:5]:  # 最多显示5个
        img = cv2.imread(people[tid]["image"])
        if img is not None:
            # 调整大小
            h, w = img.shape[:2]
            target_h = 200
            scale = target_h / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, target_h))
            
            # 添加ID标签
            cv2.rectangle(img, (0, 0), (new_w, 40), (0, 0, 0), -1)
            cv2.putText(
                img, f"ID: {tid}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )
            images.append(img)
    
    if images:
        collage = np.hstack(images)
        collage_path = output_path.parent / f"{output_path.stem}_collage.jpg"
        cv2.imwrite(str(collage_path), collage)
        print(f"   ✅ 人物拼图: {collage_path}")
    
    # 2. 在视频中标注（取第一帧）
    print("   生成标注视频帧...")
    cap = cv2.VideoCapture(str(video_path))
    
    # 找到第一个匹配的人第一次出现的帧
    first_match_id = matched_ids[0]
    first_frame_idx = people[first_match_id]["frames"][0]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx - 1)
    ret, frame = cap.read()
    
    if ret:
        # 标注所有匹配的人（如果在这一帧出现）
        for tid in matched_ids:
            if first_frame_idx in people[tid]["frames"]:
                # 找到这一帧的bbox
                for frame_num, x1, y1, x2, y2 in people[tid]["all_bboxes"]:
                    if frame_num == first_frame_idx:
                        # 绘制绿色框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(
                            frame, f"ID:{tid}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2
                        )
        
        frame_path = output_path
        cv2.imwrite(str(frame_path), frame)
        print(f"   ✅ 标注帧: {frame_path}")
    
    cap.release()


# ===== 5. 主函数 =====
def main():
    """
    主流程：根据描述在视频中找人
    """
    print("\n" + "=" * 70)
    print("🎯 视频中找人系统")
    print("=" * 70)
    
    # 阶段1: 处理视频，提取所有人
    crops_dir = OUTPUT_DIR / "crops"
    people = process_video_and_extract_people(VIDEO_PATH, crops_dir)
    
    # 保存人物数据
    people_db_path = OUTPUT_DIR / "people_database.json"
    with open(people_db_path, 'w', encoding='utf-8') as f:
        json.dump({
            str(k): {
                "image": v["image"],
                "num_frames": len(v["frames"]),
                "first_frame": v["frames"][0],
                "first_bbox": v["first_bbox"]
            }
            for k, v in people.items()
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 人物数据已保存: {people_db_path}")
    
    # 阶段2: 加载VLM
    vlm_engine = VLMQueryEngine()
    
    # 阶段3: 执行搜索
    test_queries = [
        "穿红色衣服的人",
        "穿蓝色衣服的人",
        "背背包的人",
    ]
    
    for query in test_queries:
        matched_ids = search_person_by_description(people, vlm_engine, query)
        
        print("\n" + "=" * 70)
        print(f"📋 查询结果: {query}")
        print("=" * 70)
        
        if matched_ids:
            print(f"✅ 找到 {len(matched_ids)} 个匹配的人:")
            for tid in matched_ids:
                print(f"   - Track ID: {tid}")
                print(f"     出现帧数: {len(people[tid]['frames'])}")
                print(f"     首次出现: 第 {people[tid]['frames'][0]} 帧")
            
            # 可视化
            output_path = OUTPUT_DIR / f"result_{query}.jpg"
            visualize_results(VIDEO_PATH, people, matched_ids, output_path)
        else:
            print("❌ 未找到匹配的人")
        
        print()
    
    # 最终总结
    print("\n" + "=" * 70)
    print("✅ 搜索完成！")
    print("=" * 70)
    print(f"\n📁 输出目录: {OUTPUT_DIR}")
    print(f"   - crops/              人物裁剪图片")
    print(f"   - people_database.json  人物数据库")
    print(f"   - result_*.jpg         搜索结果可视化")


if __name__ == '__main__':
    main()

