"""Question-driven VLM client for person retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from config import SystemConfig
from evidence import EvidencePackage


@dataclass
class QueryResult:
    """
    VLM的查询结果：一个匹配的轨迹及其详细信息。
    
    当VLM判断某个轨迹匹配用户的查询问题时，返回这个结果对象。
    包含：这是谁、什么时候出现、匹配度多高、为什么匹配。
    
    Attributes:
        track_id: 匹配的轨迹ID，例如 3
        start_s: 起始时间（秒），例如 12.3 表示第12.3秒
        end_s: 结束时间（秒），例如 27.8 表示第27.8秒
        score: 匹配分数，范围 0.0-1.0。1.0 表示确定匹配，0.0 表示不匹配
        reason: 匹配理由（自然语言），例如 "wearing a purple jacket and black pants"
    
    使用示例：
        result = QueryResult(
            track_id=3,
            start_s=12.3,
            end_s=27.8,
            score=0.94,
            reason="The person wears a purple jacket and carries a backpack."
        )
        print(f"找到 Track {result.track_id}，出现在 {result.start_s:.1f}s-{result.end_s:.1f}s")
        print(f"匹配度：{result.score:.0%}，理由：{result.reason}")
    """
    track_id: int
    start_s: float
    end_s: float
    score: float
    reason: str


class QwenVLMClient:
    """
    VLM客户端：使用视觉-语言模型进行问题驱动的人物检索。
    
    这是系统的"大脑"，负责：
    1. 看每个候选人的照片（多张裁剪图）
    2. 读用户的查询问题（自然语言）
    3. 回答："这个人是否匹配描述？为什么？"
    
    工作流程：
        对于每个候选轨迹：
        - 拿出证据包中的3张照片 + 运动特征数据
        - 构造prompt："这个人是否匹配'穿紫色衣服'的描述？"
        - 让Qwen2-VL模型回答，期望JSON格式：{"match": "yes", "reason": "..."}
        - 解析答案，提取匹配结果和理由
    
    技术细节：
        - 使用 Qwen2-VL-2B-Instruct 模型（20亿参数的视觉-语言模型）
        - 支持 Apple Silicon GPU (mps) 加速
        - 自动处理多图输入（同一人的多帧采样）
        - 鲁棒的答案解析（支持JSON格式和自然语言）
    
    设计原则：
        - VLM只负责"是不是 + 为什么"，不自己算时间戳
        - 时间戳由系统根据 frames + fps 自动计算
        - max_crops 控制输入图片数量，平衡效果和速度
    
    使用示例：
        client = QwenVLMClient(config, max_crops=3)
        results = client.answer("找穿红衣服的人", candidates)
        for r in results:
            print(f"Track {r.track_id}: {r.reason}")
    """

    def __init__(self, config: SystemConfig, max_crops: int = 3) -> None:
        """
        初始化VLM客户端，加载模型和处理器。
        
        Args:
            config: 系统配置对象，需要 vlm_model 字段（模型名称）
                   例如："Qwen/Qwen2-VL-2B-Instruct"
            max_crops: 每个轨迹最多使用多少张裁剪图，默认3张
                      越多越准确但越慢。3张是效果和速度的平衡点
        
        Note:
            - 模型会自动选择最佳设备：Apple Silicon GPU (mps) > CPU
            - 首次运行会从Hugging Face下载模型（约4GB），后续会使用缓存
            - 如果内存不足，可以减小 max_crops 或使用更小的模型
        """
        self.max_crops = max_crops
        # 自动选择设备：Apple Silicon GPU > CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # 加载视觉-语言生成模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vlm_model,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,  # mps用FP16加速
            device_map=self.device,
        )
        # 加载文本和图像处理器（tokenizer + image processor）
        self.processor = AutoProcessor.from_pretrained(config.vlm_model)

    def answer(
        self,
        question: str,
        candidates: Iterable[EvidencePackage],
        top_k: int | None = None,
    ) -> List[QueryResult]:
        """
        对候选轨迹进行VLM判断，返回匹配的结果列表。
        
        这是VLM客户端的主入口函数，负责：
        1. 遍历每个候选轨迹
        2. 调用 _query_package 让VLM判断是否匹配
        3. 调用 _parse_answer 解析VLM的回答
        4. 过滤出匹配的轨迹，构造结果列表
        
        Args:
            question: 用户的查询问题（自然语言）
                     例如："找出穿紫色衣服的人"、"找戴牛仔帽的人"
            candidates: 候选证据包的可迭代对象（通常来自召回引擎）
                       例如：[EvidencePackage(...), EvidencePackage(...), ...]
            top_k: 可选的早停参数。如果指定，找到 top_k 个匹配就停止
                  例如：top_k=5 表示找到5个匹配就返回（节省计算）
                  如果为 None，会遍历所有候选
        
        Returns:
            匹配结果列表，格式：[QueryResult, QueryResult, ...]
            每个结果包含：track_id, start_s, end_s, score, reason
            列表按照候选轨迹的顺序返回（不排序）
            
        Note:
            - 如果候选的 crops 为空（没有裁剪图），会跳过
            - VLM回答 "no" 或无法解析的轨迹会被过滤掉
            - 时间戳（start_s, end_s）由证据包自动计算，VLM不参与
            - top_k 用于早停优化，但结果顺序取决于候选顺序
        
        使用示例：
            # 遍历所有候选，返回所有匹配
            all_results = client.answer("找穿红衣服的人", candidates)
            
            # 找到5个匹配就停止（适合只需要前几个结果的场景）
            top_results = client.answer("找穿红衣服的人", candidates, top_k=5)
        """
        results: List[QueryResult] = []
        for package in candidates:
            # 早停优化：已经找到足够多的结果
            if top_k is not None and len(results) >= top_k:
                break
            # 跳过没有裁剪图的轨迹（无法判断）
            if not package.crops:
                continue
            # 让VLM判断这个轨迹是否匹配
            answer = self._query_package(package, question)
            # 解析VLM的回答
            parsed = self._parse_answer(answer)
            # 如果不匹配（parsed[0]=False），跳过
            if not parsed[0]:
                continue
            # 匹配成功，构造结果对象
            results.append(
                QueryResult(
                    track_id=package.track_id,
                    start_s=package.start_time_seconds,  # 自动计算的时间戳
                    end_s=package.end_time_seconds,      # 自动计算的时间戳
                    score=parsed[1],                     # 匹配分数
                    reason=parsed[2],                    # 匹配理由
                )
            )
        return results

    def _query_package(self, package: EvidencePackage, question: str) -> str:
        """
        查询单个证据包：构造prompt并调用VLM推理。
        
        这是VLM推理的核心函数，负责：
        1. 从证据包中提取裁剪图（最多 max_crops 张）
        2. 提取运动特征作为数值上下文（可选）
        3. 构造多模态prompt（图片 + 文字指令）
        4. 调用Qwen2-VL模型进行推理
        5. 返回模型的文本回答
        
        Prompt结构：
            - 系统指令：告诉模型任务是什么（判断是否匹配）
            - 用户描述：具体的查询条件（例如"穿紫色衣服"）
            - 数值上下文：运动特征（速度、时长），帮助模型理解行为
            - 图片输入：同一人的多张裁剪图（多角度、多时刻）
        
        Args:
            package: 证据包对象，包含裁剪图路径、运动特征等
            question: 用户的查询问题，例如 "找穿紫色衣服的人"
        
        Returns:
            VLM的文本回答，例如：
            - 成功：'{"match": "yes", "reason": "wearing a purple jacket"}'
            - 失败：'{"match": "no", "reason": "wearing black clothes"}'
            - 空字符串：如果没有裁剪图
        
        Note:
            - 限制图片数量（max_crops）是为了平衡效果和速度
            - 运动特征是可选的上下文，帮助模型理解"徘徊"、"快速移动"等行为
            - 使用 with torch.no_grad() 避免计算梯度（推理不需要）
        """
        # 限制裁剪图数量（例如只用前3张）
        limited_crops = package.crops[: self.max_crops]
        if not limited_crops:
            return ""
        
        # 构造运动特征上下文（如果有的话）
        motion_context = ""
        if package.motion:
            motion_context = (
                f" avg_speed={package.motion.avg_speed_px_s:.2f}px/s,"
                f" duration={package.motion.duration_s:.1f}s"
            )

        # 构造系统指令：告诉模型任务和输出格式
        instruction = (
            "You are given crops of the same person extracted from a video. "
            "Answer whether this person matches the following description. "
            "Respond strictly in JSON with keys match (yes/no) and reason."
        )
        # 构造完整prompt：指令 + 用户描述 + 数值上下文
        prompt = (
            f"{instruction}\nDescription: {question}\n"
            f"Additional numeric context:{motion_context if motion_context else ' none.'}"
        )

        # 构造多模态content：图片 + 文字
        content = []
        for crop in limited_crops:
            content.append({"type": "image", "image": crop})  # 添加图片
        content.append({"type": "text", "text": prompt})      # 添加文字prompt

        # 构造消息格式（Qwen2-VL的标准格式）
        messages = [{"role": "user", "content": content}]
        
        # 应用chat模板：把消息转换成模型能理解的文本格式
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理视觉信息：提取图片和视频输入
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 处理所有输入（文本 + 图片）：tokenize + resize + normalize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",  # 返回PyTorch tensor
        )
        
        # 把所有tensor移动到GPU/CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理阶段：不计算梯度（节省内存和计算）
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)

        # 裁剪生成的token：去掉输入部分，只保留生成的新内容
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        # 解码：把token ID转回文本
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,          # 跳过 <|endoftext|> 等特殊token
            clean_up_tokenization_spaces=True,  # 清理多余空格
        )[0]
        return answer.strip()  # 去除首尾空白

    def _parse_answer(self, answer: str) -> tuple[bool, float, str]:
        """
        解析VLM的回答，提取匹配结果、分数和理由。
        
        VLM的回答可能是多种格式，这个函数要鲁棒地处理各种情况：
        1. 标准JSON格式：{"match": "yes", "reason": "..."}
        2. 自然语言：Yes, the person is wearing...
        3. 混合格式：Match: yes. Reason: ...
        
        解析策略：
        - 优先尝试JSON解析（最可靠）
        - JSON失败则用关键词匹配（match/yes/no）
        - 提取理由文本（整段回答或JSON中的reason字段）
        
        Args:
            answer: VLM的原始回答文本
                   例如：'{"match": "yes", "reason": "wearing purple clothes"}'
                   或者：'Yes, this person matches the description.'
        
        Returns:
            三元组 (match, score, reason)：
            - match (bool): 是否匹配。True表示匹配，False表示不匹配
            - score (float): 匹配分数。匹配=1.0，不匹配=0.0
            - reason (str): 理由文本，例如 "wearing a purple jacket"
            
        Note:
            - 如果answer为空，返回 (False, 0.0, "")
            - JSON解析失败时会自动降级到关键词匹配
            - 关键词匹配会检查 "yes"/"no"/"match" 等词
            - reason默认是整段回答，如果JSON中有reason字段则使用该字段
        
        示例：
            # JSON格式
            _parse_answer('{"match": "yes", "reason": "red shirt"}')
            # 返回：(True, 1.0, "red shirt")
            
            # 自然语言
            _parse_answer('Yes, the person is wearing red.')
            # 返回：(True, 1.0, "Yes, the person is wearing red.")
            
            # 不匹配
            _parse_answer('{"match": "no", "reason": "wearing black"}')
            # 返回：(False, 0.0, "wearing black")
        """
        if not answer:
            return False, 0.0, ""
        
        match = False
        reason = answer  # 默认理由是整段回答
        score = 0.0

        # 策略1：尝试JSON解析（最可靠）
        json_match = re.search(r"\{.*\}", answer, re.S)  # re.S让.匹配换行符
        if json_match:
            candidate = json_match.group(0)
            try:
                payload = json.loads(candidate)
                # 提取match字段：yes/true/1/是 都算匹配
                flag = str(payload.get("match", "")).lower()
                match = flag in {"yes", "true", "1", "是"}
                # 提取reason字段（如果有）
                reason = payload.get("reason", answer)
            except json.JSONDecodeError:
                pass  # JSON解析失败，继续下面的关键词匹配

        # 策略2：JSON解析失败，用关键词匹配
        if not match:
            lowered = answer.lower()
            if "match" in lowered:
                # 检查 "match" 后面6个字符内有没有 "no"
                # 例如："match: yes" ✓  "match: no" ✗
                match = "match" in lowered and "no" not in lowered.split("match")[-1][:6]
            else:
                # 检查 "yes" 前面有没有 "no"
                # 例如："yes" ✓  "no, not yes" ✗
                match = "yes" in lowered and "no" not in lowered[: lowered.find("yes")]

        # 匹配成功，设置分数为1.0
        if match:
            score = 1.0
        return match, score, reason.strip()
