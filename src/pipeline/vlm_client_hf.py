"""Qwen3-VL-4B client (transformers) for verifier stage.

v2.1 highlights:
- Quality-aware crop sampling (per-segment largest box).
- Trajectory overlay on real frame with time-coded dots and START/END.
- Structured prompt (appearance / trajectory / geometric facts / constraints), final line MATCH: yes/no.
- Robust parsing (MATCH line, yes/no, 中文 是/否 fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
import concurrent.futures
from pathlib import Path
from typing import Iterable, List, Tuple
import time

import numpy as np
from PIL import Image

from core.config import SystemConfig
from core.evidence import EvidencePackage
from core.vlm_types import QueryResult
from pipeline.router import ExecutionPlan, DEFAULT_VERIFICATION_PROMPT

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


@dataclass
class Qwen3VL4BHFClient:
    """
    使用 transformers 加载 `Qwen/Qwen3-VL-4B-Instruct` 的 VLM 客户端。

    - 模型仓库：Qwen/Qwen3-VL-4B-Instruct
    - 后端：transformers + PyTorch (自动选择设备 / dtype)
    """

    config: SystemConfig
    max_crops: int = 5
    minimap_size: Tuple[int, int] = (336, 336)
    minimap_time_step_s: float = 0.5
    batch_size: int = 8

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Missing transformers or Qwen3-VL support; non-quantized Qwen3-VL-4B cannot be used.\n"
                "Please run: pip install -U transformers"
            ) from exc

        repo_id = "Qwen/Qwen3-VL-4B-Instruct"
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(repo_id)
        # Prefer fast image processor if available.
        if hasattr(self.processor, "image_processor"):
            setattr(self.processor.image_processor, "use_fast", True)
        # Decoder-only: ensure left padding to avoid warning / misalignment.
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        # 按官方 README 推荐：dtype=\"auto\" + device_map=\"auto\"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo_id,
            dtype="auto",
            device_map="auto",
        )
        print(f"[HF VLM] device map: {getattr(self.model, 'hf_device_map', 'n/a')}")
        self.temperature = self.config.vlm_temperature
        self.max_new_tokens = self.config.vlm_max_new_tokens
        # Use config-provided batch size to keep a single source of truth.
        self.batch_size = self.config.vlm_batch_size

    def compose_final_answer(self, question: str, results: List[QueryResult]) -> str:
        """
        用同一个 4B VLM 对筛选后的轨迹做一次总回答，直接返回给用户。
        """
        if not results:
            return "No matching tracks found."

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
                            "return a single concise answer (1-2 sentences) in English. "
                            "Do NOT repeat per-track bullet points; synthesize the conclusion "
                            "and mention track IDs briefly. If none match, say no match."
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
                max_new_tokens=80,
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
        batch_messages: list[list[dict]] = []
        batch_packages: list[EvidencePackage] = []
        t_start = time.monotonic()

        def flush_batch() -> bool:
            """Run one VLM batch and append matches to results. Returns True if top_k reached."""
            if not batch_messages:
                return False
            t_batch_start = time.monotonic()
            try:
                batch_answers = self._run_batch_inference(batch_messages)
            except Exception as exc:
                print(f"[HF VLM] batch of {len(batch_messages)} failed: {exc}")
                batch_messages.clear()
                batch_packages.clear()
                return False
            t_batch_end = time.monotonic()
            print(
                f"[HF VLM] batch size={len(batch_messages)} time={t_batch_end - t_batch_start:.2f}s "
                f"(total so far {t_batch_end - t_start:.2f}s)"
            )

            for package, answer_text in zip(batch_packages, batch_answers):
                match, score, reason = self._parse_answer(answer_text)
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
                if top_k is not None and len(results) >= top_k:
                    batch_messages.clear()
                    batch_packages.clear()
                    return True

            batch_messages.clear()
            batch_packages.clear()
            return False

        for idx, package in enumerate(candidates, start=1):
            if top_k is not None and len(results) >= top_k:
                break
            if not package.crops:
                continue
            print(f"[HF VLM] queue track {package.track_id} ({idx}) ...")
            messages = self._build_messages(package, question, plan)
            if not messages:
                continue
            batch_packages.append(package)
            batch_messages.append(messages)
            if len(batch_messages) >= self.batch_size:
                if flush_batch():
                    break

        if batch_messages and (top_k is None or len(results) < top_k):
            flush_batch()
        print(f"[HF VLM] total time={time.monotonic() - t_start:.2f}s for {len(results)} matches")

        return results

    def _build_messages(
        self,
        package: EvidencePackage,
        question: str,
        plan: ExecutionPlan | None,
    ) -> list[dict] | None:
        """Construct chat messages (images + prompt) for one package."""
        limited_crops = self._sample_crops(package)
        if not limited_crops:
            return None

        minimap_img = self._render_minimap(package)
        prompt = self._build_verification_prompt(
            package=package,
            question=question,
            plan=plan,
            appearance_count=len(limited_crops),
            minimap_present=minimap_img is not None,
        )

        user_content = []
        # Parallelize image loading for I/O bound speedup.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(limited_crops) or 1)
        ) as executor:
            images = list(executor.map(lambda p: Image.open(Path(p)).convert("RGB"), limited_crops))
        for img in images:
            user_content.append({"type": "image", "image": img})
        if minimap_img is not None:
            user_content.append({"type": "image", "image": minimap_img})
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer in free text, but end with 'MATCH: yes' or 'MATCH: no'.",
                    }
                ],
            },
            {"role": "user", "content": user_content},
        ]
        return messages

    def _run_batch_inference(self, batch_messages: list[list[dict]]) -> list[str]:
        """Run VLM inference for a batch of messages."""
        if not batch_messages:
            return []

        t0 = time.monotonic()
        texts = [
            self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in batch_messages
        ]
        t_templates = time.monotonic()

        batch_images: list[list[Image.Image]] = []
        for msgs in batch_messages:
            user_content = next(m["content"] for m in msgs if m.get("role") == "user")
            imgs = [item["image"] for item in user_content if item.get("type") == "image"]
            batch_images.append(imgs)
        t_images = time.monotonic()

        inputs = self.processor(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        )
        t_proc = time.monotonic()
        inputs = inputs.to(self.model.device)
        t_to_device = time.monotonic()

        with self._torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        t_generate = time.monotonic()

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        t_decode = time.monotonic()
        print(
            "[HF VLM timing] templates={:.2f}s images={:.2f}s proc={:.2f}s to_device={:.2f}s "
            "generate={:.2f}s decode={:.2f}s total={:.2f}s".format(
                t_templates - t0,
                t_images - t_templates,
                t_proc - t_images,
                t_to_device - t_proc,
                t_generate - t_to_device,
                t_decode - t_generate,
                t_decode - t0,
            )
        )
        return answers

    def _query_package(
        self,
        package: EvidencePackage,
        question: str,
        plan: ExecutionPlan | None,
    ) -> str:
        messages = self._build_messages(package, question, plan)
        if not messages:
            return ""
        answers = self._run_batch_inference([messages])
        return answers[0] if answers else ""

    @staticmethod
    def _parse_answer(answer: str) -> tuple[bool, float, str]:
        if not answer:
            return False, 0.0, ""

        import re

        match_flag = False
        reason = answer
        score = 0.0

        lowered = answer.lower()
        match_line = None
        for line in lowered.splitlines():
            if line.strip().startswith("match:"):
                match_line = line.strip()
                break
        if match_line:
            if "yes" in match_line and "no" not in match_line:
                match_flag = True
            elif "no" in match_line:
                match_flag = False
        else:
            # fallback to explicit yes/no cues
            if re.search(r"\byes\b", lowered) and not re.search(r"\bno\b", lowered):
                match_flag = True
            elif "是" in answer and "不" not in answer:
                match_flag = True
            else:
                negative = (
                    "not match" in lowered
                    or "no match" in lowered
                    or "does not" in lowered
                    or "not a plausible" in lowered
                    or "not " in lowered
                    or " no " in lowered
                )
                match_flag = False if negative else match_flag

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

    def _sample_crops(self, package: EvidencePackage) -> List[str]:
        """质量优先的均匀采样：分段取 bbox 面积较大的帧，覆盖前中后。"""
        crops = package.crops
        if not crops:
            return []
        if len(crops) <= self.max_crops:
            return list(crops)
        areas = []
        for bbox in package.bboxes:
            x1, y1, x2, y2 = bbox
            areas.append(max((x2 - x1) * (y2 - y1), 1))
        indices: List[int] = []
        segments = np.array_split(range(len(crops)), self.max_crops)
        for seg in segments:
            if len(seg) == 0:
                continue
            best_idx = max(seg, key=lambda i: areas[i])
            indices.append(best_idx)
        indices = sorted(set(indices))
        if len(indices) > self.max_crops:
            indices = indices[: self.max_crops]
        return [crops[i] for i in indices]

    def _build_motion_narrative(self, package: EvidencePackage) -> str:
        feats = package.features
        if not feats:
            return "No motion data."

        # 基于画面宽度做速度归一化，转成模型易懂的标签
        width = 1920
        if package.meta and package.meta.get("resolution"):
            width = max(int(package.meta["resolution"][0] or 1920), 1)

        norm_speed = feats.avg_speed_px_s / float(width)
        if norm_speed < 0.02:
            speed_desc = "standing still or barely moving"
        elif norm_speed < 0.10:
            speed_desc = "walking at normal pace"
        elif norm_speed < 0.25:
            speed_desc = "moving fast or running"
        else:
            speed_desc = "sprinting or moving very fast"

        dir_desc = ""
        dx, dy = feats.displacement_vec
        if feats.path_length_px > 10.0:
            if abs(dx) >= abs(dy):
                dir_desc = "moving right" if dx > 0 else "moving left"
            else:
                dir_desc = "moving down (towards camera)" if dy > 0 else "moving up (away)"

        pos_desc = ""
        if feats.centroids:
            start_side = self._side_from_ratio(feats.centroids[0][0])
            end_side = self._side_from_ratio(feats.centroids[-1][0])
            pos_desc = f"start at {start_side}, end at {end_side}"

        parts = [f"Speed: {speed_desc}."]
        if dir_desc:
            parts.append(f"Direction: {dir_desc}.")
        if pos_desc:
            parts.append(f"Position: {pos_desc}.")
        return " ".join(parts)

    @staticmethod
    def _side_from_ratio(x_ratio: float) -> str:
        if x_ratio < 0.33:
            return "left side"
        if x_ratio > 0.66:
            return "right side"
        return "center"

    def _build_verification_prompt(
        self,
        package: EvidencePackage,
        question: str,
        plan: ExecutionPlan | None,
        appearance_count: int,
        minimap_present: bool,
    ) -> str:
        plan_context = self._build_plan_context(plan)
        motion_context = self._build_motion_narrative(package)
        minimap_index = appearance_count + 1 if minimap_present else None
        minimap_desc = ""
        if minimap_index:
            minimap_desc = (
                f"Image {minimap_index} is a scene frame with the person's motion path overlaid "
                f"(yellow path, dots every {self.minimap_time_step_s:.1f}s, green=start, red=end, labeled START/END). "
                "Trust this overlaid path for direction/position."
            )

        return (
            f"## Task\n"
            f'Verify if this person matches the query: "{question}"\n\n'
            f"## Evidence\n"
            f"### Appearance (images 1-{appearance_count})\n"
            f"{'Images show the person at different moments.'}\n\n"
            f"### Motion Summary (pre-computed facts, do NOT re-judge)\n"
            f"{motion_context if motion_context else 'unknown'}\n\n"
            f"### Trajectory Overlay\n"
            f"{minimap_desc if minimap_desc else 'No overlaid path available.'}\n\n"
            f"### Constraints from planner\n{plan_context}\n\n"
            f"## Instructions\n"
            f"1) Describe briefly what you see in the images.\n"
            f"2) Check each constraint (visual, motion). State if they match or not.\n"
            f"3) Final line: MATCH: yes or MATCH: no\n"
        )

    def _render_minimap(self, package: EvidencePackage) -> Image.Image | None:
        """将轨迹叠加在真实帧上，带时间打点，辅助 VLM 理解场景内的路径。"""
        if cv2 is None:
            return None
        feats = package.features
        if not feats or not feats.centroids or not package.frames:
            return None

        # 选取轨迹中间帧
        mid_idx = len(package.frames) // 2
        target_frame_idx = max(package.frames[mid_idx] - 1, 0)

        cap = cv2.VideoCapture(str(self.config.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None

        height, width = frame.shape[:2]
        points = [
            (int(cx * (width - 1)), int(cy * (height - 1)))
            for cx, cy in feats.centroids
        ]
        overlay = frame.copy()

        # 路径线
        if len(points) >= 2:
            cv2.polylines(
                overlay,
                [np.array(points)],
                isClosed=False,
                color=(0, 215, 255),  # BGR yellow-ish
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        # 时间打点
        fps = max(package.fps, 1e-3)
        step = max(int(fps * self.minimap_time_step_s), 1)
        for idx in range(0, len(points), step):
            pt = points[idx]
            progress = idx / max(len(points) - 1, 1)
            color = (0, int(255 * (1 - progress)), int(255 * progress))  # green->red
            cv2.circle(overlay, pt, radius=5, color=color, thickness=-1)

        # 起止标记
        if len(points) >= 1:
            cv2.circle(overlay, points[0], 7, (0, 255, 0), 2)
            cv2.putText(overlay, "START", (points[0][0] + 5, points[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if len(points) >= 2:
            cv2.arrowedLine(overlay, points[-2], points[-1], (0, 0, 255), 3, tipLength=0.3)
            cv2.putText(overlay, "END", (points[-1][0] + 5, points[-1][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        blended = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
