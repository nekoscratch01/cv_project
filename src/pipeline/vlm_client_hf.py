"""Qwen3-VL-4B client powered by transformers (当前默认实现)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from core.config import SystemConfig
from core.evidence import EvidencePackage
from core.vlm_types import QueryResult
from pipeline.router import ExecutionPlan, DEFAULT_VERIFICATION_PROMPT


@dataclass
class Qwen3VL4BHFClient:
    """
    使用 transformers 加载 `Qwen/Qwen3-VL-4B-Instruct` 的 VLM 客户端。

    - 模型仓库：Qwen/Qwen3-VL-4B-Instruct
    - 后端：transformers + PyTorch (自动选择设备 / dtype)
    """

    config: SystemConfig
    max_crops: int = 3

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "缺少 transformers 或 Qwen3-VL 支持，无法使用非量化 Qwen3-VL-4B。\n"
                "请先执行：pip install -U transformers"
            ) from exc

        repo_id = "Qwen/Qwen3-VL-4B-Instruct"
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(repo_id)
        # 按官方 README 推荐：dtype=\"auto\" + device_map=\"auto\"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo_id,
            dtype="auto",
            device_map="auto",
        )
        self.temperature = self.config.vlm_temperature
        self.max_new_tokens = self.config.vlm_max_new_tokens

    def compose_final_answer(self, question: str, results: List[QueryResult]) -> str:
        """
        用同一个 4B VLM 对筛选后的轨迹做一次总回答，直接返回给用户。
        """
        if not results:
            return "未找到匹配轨迹。"

        summary_lines = []
        for item in results:
            summary_lines.append(
                f"- track {item.track_id}: {item.start_s:.1f}s–{item.end_s:.1f}s | reason: {item.reason}"
            )
        summary = "\n".join(summary_lines)

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an assistant summarizing video retrieval results. "
                            "Given the original question and the shortlisted tracks with reasons, "
                            "provide a concise final answer in Chinese that tells the user which tracks match. "
                            "If none, say no match."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {question}\nShortlisted:\n{summary}"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=self.temperature,
            )
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        answers = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return answers[0] if answers else ""

    def answer(
        self,
        question: str,
        candidates: Iterable[EvidencePackage],
        *,
        plan: ExecutionPlan | None = None,
        top_k: int | None = None,
    ) -> List[QueryResult]:
        results: List[QueryResult] = []
        for idx, package in enumerate(candidates, start=1):
            if top_k is not None and len(results) >= top_k:
                break
            if not package.crops:
                continue
            print(f"[HF VLM] checking track {package.track_id} ({idx}) ...")
            answer = self._query_package(package, question, plan)
            if not answer:
                continue
            match, score, reason = self._parse_answer(answer)
            if not match:
                continue
            results.append(
                QueryResult(
                    track_id=package.track_id,
                    start_s=package.start_time_seconds,
                    end_s=package.end_time_seconds,
                    score=score,
                    reason=reason,
                )
            )
        return results

    def _query_package(
        self,
        package: EvidencePackage,
        question: str,
        plan: ExecutionPlan | None,
    ) -> str:
        limited_crops = package.crops[: self.max_crops]
        if not limited_crops:
            return ""

        motion_context = ""
        if package.motion:
            motion_context = (
                f" avg_speed={package.motion.avg_speed_px_s:.2f}px/s,"
                f" duration={package.motion.duration_s:.1f}s"
            )
        plan_context = self._build_plan_context(plan)
        verification_prompt = (
            plan.verification_prompt if plan and plan.verification_prompt else DEFAULT_VERIFICATION_PROMPT
        )

        prompt = (
            "You are a video analysis assistant.\n"
            f"Original question: {question}\n"
            f"{verification_prompt}\n"
            f"Planner summary:\n{plan_context}\n"
            f"Additional numeric context:{motion_context if motion_context else ' none.'}"
        )

        # 构造 Qwen3-VL 聊天消息（本地 PIL 图像 + 文本）
        user_content = []
        for crop in limited_crops:
            img = Image.open(Path(crop)).convert("RGB")
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You answer strictly in JSON with keys match (yes/no) and reason.",
                    }
                ],
            },
            {"role": "user", "content": user_content},
        ]

        # 参照官方 README：processor.apply_chat_template + model.generate
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with self._torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        # 去掉提示部分，只保留生成的新 token
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return answers[0] if answers else ""

    @staticmethod
    def _parse_answer(answer: str) -> tuple[bool, float, str]:
        if not answer:
            return False, 0.0, ""

        import json
        import re

        match_flag = False
        reason = answer
        score = 0.0

        json_match = re.search(r"\{.*\}", answer, re.S)
        if json_match:
            candidate = json_match.group(0)
            try:
                payload = json.loads(candidate)
                flag = str(payload.get("match", "")).lower()
                match_flag = flag in {"yes", "true", "1", "是"}
                reason = payload.get("reason", answer)
            except json.JSONDecodeError:
                pass

        if not match_flag:
            lowered = answer.lower()
            if "match" in lowered:
                match_flag = "match" in lowered and "no" not in lowered.split("match")[-1][:6]
            else:
                if "yes" in lowered:
                    idx = lowered.find("yes")
                    match_flag = "no" not in lowered[:idx]

        if match_flag:
            score = 1.0
        return match_flag, score, reason.strip()

    @staticmethod
    def _build_plan_context(plan: ExecutionPlan | None) -> str:
        if plan is None:
            return "No extra plan context."
        lines = [f"- description: {plan.description}"]
        if plan.visual_tags:
            lines.append(f"- visual tags: {', '.join(plan.visual_tags)}")
        if plan.needed_facts:
            lines.append(f"- needed facts: {', '.join(plan.needed_facts)}")
        if plan.constraints:
            parts = []
            for key, value in plan.constraints.items():
                parts.append(f"{key}={value}")
            lines.append(f"- constraints: {'; '.join(parts)}")
        return "\n".join(lines)
