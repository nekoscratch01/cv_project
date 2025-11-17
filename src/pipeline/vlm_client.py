"""Question-driven VLM client for person retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .config import SystemConfig
from .evidence import EvidencePackage


@dataclass
class QueryResult:
    track_id: int
    start_s: float
    end_s: float
    score: float
    reason: str


class QwenVLMClient:
    """Wraps Qwen2-VL for question-driven yes/no decisions per track."""

    def __init__(self, config: SystemConfig, max_crops: int = 3) -> None:
        self.max_crops = max_crops
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vlm_model,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(config.vlm_model)

    def answer(
        self,
        question: str,
        candidates: Iterable[EvidencePackage],
        top_k: int | None = None,
    ) -> List[QueryResult]:
        results: List[QueryResult] = []
        for package in candidates:
            if top_k is not None and len(results) >= top_k:
                break
            if not package.crops:
                continue
            answer = self._query_package(package, question)
            parsed = self._parse_answer(answer)
            if not parsed[0]:
                continue
            results.append(
                QueryResult(
                    track_id=package.track_id,
                    start_s=package.start_time_seconds,
                    end_s=package.end_time_seconds,
                    score=parsed[1],
                    reason=parsed[2],
                )
            )
        return results

    def _query_package(self, package: EvidencePackage, question: str) -> str:
        limited_crops = package.crops[: self.max_crops]
        if not limited_crops:
            return ""
        motion_context = ""
        if package.motion:
            motion_context = (
                f" avg_speed={package.motion.avg_speed_px_s:.2f}px/s,"
                f" duration={package.motion.duration_s:.1f}s"
            )

        instruction = (
            "You are given crops of the same person extracted from a video. "
            "Answer whether this person matches the following description. "
            "Respond strictly in JSON with keys match (yes/no) and reason."
        )
        prompt = (
            f"{instruction}\nDescription: {question}\n"
            f"Additional numeric context:{motion_context if motion_context else ' none.'}"
        )

        content = []
        for crop in limited_crops:
            content.append({"type": "image", "image": crop})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return answer.strip()

    def _parse_answer(self, answer: str) -> tuple[bool, float, str]:
        if not answer:
            return False, 0.0, ""
        match = False
        reason = answer
        score = 0.0

        # try JSON
        json_match = re.search(r"\{.*\}", answer, re.S)
        if json_match:
            candidate = json_match.group(0)
            try:
                payload = json.loads(candidate)
                flag = str(payload.get("match", "")).lower()
                match = flag in {"yes", "true", "1", "是"}
                reason = payload.get("reason", answer)
            except json.JSONDecodeError:
                pass

        if not match:
            lowered = answer.lower()
            if "match" in lowered:
                match = "match" in lowered and "no" not in lowered.split("match")[-1][:6]
            else:
                match = "yes" in lowered and "no" not in lowered[: lowered.find("yes")]

        if match:
            score = 1.0
        return match, score, reason.strip()
