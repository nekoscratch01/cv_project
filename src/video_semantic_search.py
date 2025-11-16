"""
🎯 完整的视频语义检索系统
Video Semantic Search System

功能流程:
    视频 → YOLO检测 → ByteTrack跟踪 → 提取关键帧 → VLM属性提取 → 建立数据库 → 语义查询

作者: 一起学习的产物
日期: 2025-11
"""

import cv2
import torch
import json
import time
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from boxmot import create_tracker
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===== 1. 配置 =====
class Config:
    """系统配置"""
    # 输入视频
    VIDEO_PATH = Path("../data/snippets/debug_15s.mp4")
    
    # 输出目录
    OUTPUT_DIR = Path("output_full_system")
    CROPS_DIR = OUTPUT_DIR / "crops"
    
    # YOLO配置
    YOLO_MODEL = "yolov8n.pt"
    YOLO_CONF = 0.3
    YOLO_DEVICE = "mps"
    
    # 跟踪配置
    TRACKER_TYPE = "bytetrack"
    
    # VLM配置
    VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    # 采样配置
    SAMPLE_INTERVAL = 30  # 每30帧采样一次（避免重复）
    MIN_TRACK_LENGTH = 10  # 至少出现10帧才处理
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CROPS_DIR.mkdir(parents=True, exist_ok=True)


# ===== 2. 视频处理模块 =====
class VideoProcessor:
    """视频检测与跟踪"""
    
    def __init__(self, config: Config):
        self.config = config
        print("\n🔧 初始化检测与跟踪模块...")
        
        # 加载YOLO
        self.yolo = YOLO(config.YOLO_MODEL)
        
        # 加载跟踪器
        self.tracker = create_tracker(
            tracker_type=config.TRACKER_TYPE,
            tracker_config=None,
            reid_weights=None,
            device='cpu',
            half=False,
            per_class=False
        )
        
        print(f"   ✅ YOLO: {config.YOLO_MODEL}")
        print(f"   ✅ Tracker: {config.TRACKER_TYPE}")
    
    def process_video(self):
        """
        处理视频，提取每个track的关键帧
        
        返回:
            track_data: {
                track_id: {
                    "frames": [frame_idx1, frame_idx2, ...],
                    "crops": [crop_path1, crop_path2, ...],
                    "bboxes": [(x1,y1,x2,y2), ...]
                }
            }
        """
        print(f"\n📹 开始处理视频: {self.config.VIDEO_PATH}")
        
        cap = cv2.VideoCapture(str(self.config.VIDEO_PATH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        track_data = {}  # 存储每个track的信息
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # 进度显示
            if frame_idx % 30 == 0:
                print(f"   处理中: {frame_idx}/{total_frames} 帧 ({frame_idx/total_frames*100:.1f}%)", end='\r')
            
            # YOLO检测
            results = self.yolo.predict(
                source=frame,
                device=self.config.YOLO_DEVICE,
                conf=self.config.YOLO_CONF,
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
            
            # 跟踪
            if len(detections) > 0:
                detections = np.array(detections)
                tracks = self.tracker.update(detections, frame)
                
                if tracks.size > 0:
                    for track in tracks:
                        x1, y1, x2, y2 = map(int, track[:4])
                        track_id = int(track[4])
                        
                        # 初始化track记录
                        if track_id not in track_data:
                            track_data[track_id] = {
                                "frames": [],
                                "crops": [],
                                "bboxes": []
                            }
                        
                        track_data[track_id]["frames"].append(frame_idx)
                        track_data[track_id]["bboxes"].append((x1, y1, x2, y2))
                        
                        # 采样关键帧（每N帧保存一次）
                        if len(track_data[track_id]["frames"]) % self.config.SAMPLE_INTERVAL == 1:
                            # 裁剪人物图片
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                crop_path = self.config.CROPS_DIR / f"id{track_id:03d}_frame{frame_idx:05d}.jpg"
                                cv2.imwrite(str(crop_path), crop)
                                track_data[track_id]["crops"].append(str(crop_path))
        
        cap.release()
        print(f"\n   ✅ 处理完成: {total_frames} 帧")
        
        # 过滤短track
        filtered_data = {
            tid: data for tid, data in track_data.items()
            if len(data["frames"]) >= self.config.MIN_TRACK_LENGTH
        }
        
        print(f"   📊 总共检测到 {len(track_data)} 个目标")
        print(f"   📊 过滤后剩余 {len(filtered_data)} 个有效目标")
        
        return filtered_data

    def render_highlight_video(self, track_data, target_track_ids, output_path, label_text="target"):
        """将满足条件的track在原视频上高亮并导出"""
        if not target_track_ids:
            print("   ⚠️  没有目标需要可视化，跳过视频导出")
            return

        target_ids = set(target_track_ids)
        frame_map = defaultdict(list)
        for tid in target_ids:
            data = track_data.get(tid)
            if not data:
                continue
            for frame_idx, bbox in zip(data["frames"], data["bboxes"]):
                frame_map[frame_idx].append((tid, bbox))

        cap = cv2.VideoCapture(str(self.config.VIDEO_PATH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        highlight_color = (0, 0, 255)  # 红色边框

        print(f"\n📼 导出高亮视频: {output_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            tracks_this_frame = frame_map.get(frame_idx, [])
            for tid, bbox in tracks_this_frame:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), highlight_color, 3)
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    highlight_color,
                    2
                )

            if tracks_this_frame:
                cv2.putText(
                    frame,
                    f"Tracking {label_text}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            out.write(frame)

        cap.release()
        out.release()
        print("   ✅ 已生成高亮视频")


# ===== 3. VLM属性提取模块 =====
class AttributeExtractor:
    """使用VLM提取每个track的属性"""
    
    def __init__(self, config: Config):
        self.config = config
        print("\n🔧 初始化VLM模块...")
        
        # 检测设备
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # 加载模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(config.VLM_MODEL)
        
        print(f"   ✅ 模型: {config.VLM_MODEL}")
        print(f"   ✅ 设备: {self.device.upper()}")
    
    def query_image(self, image_path, question):
        """对单张图片提问"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        
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
    
    def extract_attributes(self, track_data):
        """
        为每个track提取属性
        
        参数:
            track_data: VideoProcessor返回的跟踪数据
        
        返回:
            attributes: {
                track_id: {
                    "color": "red",
                    "has_backpack": True,
                    "gender": "male",
                    ...
                }
            }
        """
        print(f"\n🔍 开始提取属性 ({len(track_data)} 个目标)...")
        
        attributes = {}
        
        # 定义要问的问题
        questions = [
            ("color", "What is the main color of this person's clothing? Answer with one color word only."),
            ("has_backpack", "Is this person carrying a backpack? Answer yes or no."),
            ("upper_color", "What color is this person's upper body clothing? Answer with one color word only."),
        ]
        
        for idx, (track_id, data) in enumerate(track_data.items(), 1):
            print(f"\n[{idx}/{len(track_data)}] 处理 Track ID: {track_id}")
            
            # 只用第一张crop（代表性图片）
            if not data["crops"]:
                print(f"   ⚠️  没有可用图片，跳过")
                continue
            
            crop_path = data["crops"][0]
            print(f"   📷 使用图片: {Path(crop_path).name}")
            
            # 提取属性
            attrs = {}
            for attr_name, question in questions:
                try:
                    answer = self.query_image(crop_path, question)
                    
                    # 处理yes/no问题
                    if attr_name.startswith("has_"):
                        attrs[attr_name] = any(
                            word in answer.lower() 
                            for word in ['yes', 'yeah', 'yep', 'true']
                        )
                    else:
                        attrs[attr_name] = answer.lower().strip()
                    
                    print(f"   ✅ {attr_name}: {attrs[attr_name]}")
                
                except Exception as e:
                    print(f"   ❌ {attr_name}: 提取失败 ({str(e)})")
                    attrs[attr_name] = None
            
            attributes[track_id] = attrs
        
        return attributes


# ===== 4. 语义查询模块 =====
class SemanticSearchEngine:
    """语义查询引擎"""
    
    def __init__(self, track_data, attributes):
        self.track_data = track_data
        self.attributes = attributes
        
        print("\n🔧 初始化查询引擎...")
        print(f"   ✅ 加载 {len(attributes)} 个目标的属性数据")
    
    def search(self, query_type, query_value):
        """
        语义查询
        
        参数:
            query_type: 查询类型（color, has_backpack, upper_color等）
            query_value: 查询值（如"red", True等）
        
        返回:
            匹配的track_id列表
        """
        print(f"\n🔍 查询: {query_type} = {query_value}")
        
        results = []
        for track_id, attrs in self.attributes.items():
            if query_type not in attrs:
                continue
            
            attr_value = attrs[query_type]
            
            # 布尔值直接比较
            if isinstance(query_value, bool):
                if attr_value == query_value:
                    results.append(track_id)
            # 字符串模糊匹配
            elif isinstance(query_value, str):
                if query_value.lower() in str(attr_value).lower():
                    results.append(track_id)
        
        print(f"   📋 找到 {len(results)} 个匹配结果: {results}")
        return results
    
    def complex_search(self, conditions):
        """
        复合条件查询
        
        参数:
            conditions: [(query_type, query_value), ...]
        
        示例:
            conditions = [("color", "red"), ("has_backpack", True)]
        """
        print(f"\n🔍 复合查询: {conditions}")
        
        # 初始为所有track_id
        result_set = set(self.attributes.keys())
        
        # 逐个条件过滤
        for query_type, query_value in conditions:
            matched = set(self.search(query_type, query_value))
            result_set = result_set.intersection(matched)
        
        results = list(result_set)
        print(f"\n   🎯 最终匹配: {len(results)} 个结果 {results}")
        return results
    
    def visualize_results(self, track_ids, output_path):
        """可视化查询结果"""
        if not track_ids:
            print("   ⚠️  没有结果可视化")
            return
        
        print(f"\n📊 生成结果可视化...")
        
        # 收集图片
        images = []
        for tid in track_ids[:5]:  # 最多显示5个
            if tid in self.track_data and self.track_data[tid]["crops"]:
                crop_path = self.track_data[tid]["crops"][0]
                img = cv2.imread(crop_path)
                if img is not None:
                    # 添加ID标签
                    cv2.putText(
                        img, f"ID:{tid}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )
                    images.append(img)
        
        if not images:
            print("   ⚠️  没有可用图片")
            return
        
        # 拼接图片
        if len(images) == 1:
            result = images[0]
        else:
            # 水平拼接
            max_height = max(img.shape[0] for img in images)
            resized = []
            for img in images:
                h, w = img.shape[:2]
                scale = max_height / h
                new_w = int(w * scale)
                resized.append(cv2.resize(img, (new_w, max_height)))
            result = np.hstack(resized)
        
        cv2.imwrite(str(output_path), result)
        print(f"   ✅ 保存到: {output_path}")


# ===== 5. 主系统 =====
class VideoSemanticSearchSystem:
    """完整的视频语义检索系统"""
    
    def __init__(self, config: Config):
        self.config = config
        
        print("=" * 70)
        print("🚀 视频语义检索系统")
        print("=" * 70)
        
        # 初始化模块
        self.video_processor = VideoProcessor(config)
        self.attribute_extractor = AttributeExtractor(config)
        
        self.track_data = None
        self.attributes = None
        self.search_engine = None
    
    def build_index(self):
        """构建索引（处理视频+提取属性）"""
        print("\n" + "=" * 70)
        print("📦 阶段1: 构建索引")
        print("=" * 70)
        
        # Step 1: 处理视频
        self.track_data = self.video_processor.process_video()
        
        # Step 2: 提取属性
        self.attributes = self.attribute_extractor.extract_attributes(self.track_data)
        
        # Step 3: 保存数据库
        db_path = self.config.OUTPUT_DIR / "attribute_database.json"
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump({
                "track_data": {
                    str(k): {
                        "frames": v["frames"],
                        "crops": v["crops"],
                        "num_bboxes": len(v["bboxes"])
                    }
                    for k, v in self.track_data.items()
                },
                "attributes": {str(k): v for k, v in self.attributes.items()}
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 数据库已保存: {db_path}")
        
        # Step 4: 初始化查询引擎
        self.search_engine = SemanticSearchEngine(self.track_data, self.attributes)
        
        print("\n✅ 索引构建完成！")
    
    def search(self, query_description, conditions):
        """
        执行查询
        
        参数:
            query_description: 查询描述（用于文件名）
            conditions: [(query_type, query_value), ...]
        """
        if self.search_engine is None:
            print("❌ 请先运行 build_index()")
            return []
        
        print("\n" + "=" * 70)
        print("🔍 阶段2: 语义查询")
        print("=" * 70)
        
        results = self.search_engine.complex_search(conditions)
        
        # 可视化
        if results:
            output_path = self.config.OUTPUT_DIR / f"result_{query_description}.jpg"
            self.search_engine.visualize_results(results, output_path)

            # 生成高亮视频，展示“帮我跟踪这些人”能力
            safe_name = query_description.replace("/", "_").replace(" ", "_")
            tracking_video = self.config.OUTPUT_DIR / f"tracking_{safe_name}.mp4"
            self.video_processor.render_highlight_video(
                self.track_data,
                results,
                tracking_video,
                label_text=query_description
            )
        
        return results


# ===== 6. 主函数 =====
def main():
    """主流程"""
    
    # 配置
    config = Config()
    
    # 创建系统
    system = VideoSemanticSearchSystem(config)
    
    # 构建索引
    system.build_index()
    
    # 执行查询
    print("\n" + "=" * 70)
    print("🎯 测试查询")
    print("=" * 70)
    
    # 查询1: 穿红色衣服的人
    system.search(
        query_description="穿红色衣服的人",
        conditions=[("color", "red")]
    )
    
    # 查询2: 背背包的人
    system.search(
        query_description="背背包的人",
        conditions=[("has_backpack", True)]
    )
    
    # 查询3: 复合查询（穿蓝色衣服且背背包）
    system.search(
        query_description="穿蓝色衣服且背背包",
        conditions=[("color", "blue"), ("has_backpack", True)]
    )
    
    print("\n" + "=" * 70)
    print("✅ 系统运行完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
