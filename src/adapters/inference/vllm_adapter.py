"""vLLM 推理适配器，实现 InferencePort 协议（Phase 2 全图 Grounding 版）。"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency guard
    AsyncOpenAI = None  # type: ignore

from domain.value_objects.verification_result import VerificationResult, VlmResponseParser
from ports.inference_port import InferencePort

if TYPE_CHECKING:
    from core.evidence import EvidencePackage


@dataclass
class VllmConfig:
    """vLLM 配置"""

    endpoint: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 256
    timeout: float = 120.0
    max_retries: int = 3
    max_images_per_request: int = 5


SYSTEM_PROMPT = (
    "You are an intelligent video surveillance analyst. "
    "You will be provided with video frames and multiple candidate targets. "
    "For each target, focus ONLY on the region specified by its bounding box and the provided motion telemetry. "
    "Return a clear MATCH decision with a concise reason for each target."
)


class VllmAdapter:
    """
    vLLM 推理适配器

    特点：
    1. 无状态，可在多个 worker 间复用
    2. 异步优先，支持高并发
    3. 通过 OpenAI 兼容接口调用 vLLM
    """

    def __init__(self, config: VllmConfig):
        self.config = config
        if AsyncOpenAI is None:
            raise RuntimeError("Missing dependency 'openai'. Please install openai>=1.37.0.")
        self.client = AsyncOpenAI(
            base_url=config.endpoint,
            api_key="EMPTY",  # vLLM 的默认 OpenAI 兼容 API 无需真实 key
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self._parser = VlmResponseParser()

    async def verify_batch(
        self,
        packages: List["EvidencePackage"],
        question: str,
        plan_context: Optional[str] = None,
        concurrency: int = 3,
    ) -> List[VerificationResult]:
        """
        Phase 2 核心：全图 Grounding 批处理。
        """
        results: List[VerificationResult] = []
        batch_size = max(1, min(concurrency, len(packages)))

        for i in range(0, len(packages), batch_size):
            batch = packages[i : i + batch_size]
            try:
                frames_b64, res_info = self._extract_frames(batch)
                if not frames_b64:
                    err = VerificationResult.error("Video frame extraction failed")
                    results.extend([err] * len(batch))
                    continue

                messages = self._build_messages(batch, question, frames_b64, res_info, plan_context)
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                raw_text = response.choices[0].message.content if response.choices else ""
                parsed_map = self._parse_batch_response(raw_text or "", [p.track_id for p in batch])

                for pkg in batch:
                    vr = parsed_map.get(str(pkg.track_id))
                    if vr:
                        results.append(vr)
                    else:
                        results.append(VerificationResult.error("Missing result for track"))
            except Exception as exc:  # noqa: BLE001
                err = VerificationResult.error(str(exc))
                results.extend([err] * len(batch))
        return results

    def _build_messages(
        self,
        packages: List["EvidencePackage"],
        question: str,
        frames_b64: List[str],
        res_info: Tuple[int, int],
        plan_context: Optional[str],
    ) -> List[dict]:
        """构造批量 Prompt：全图帧 + 多候选 BBox + 遥测。"""
        image_contents = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in frames_b64
        ]
        prompt = self._build_prompt(packages, question, plan_context, res_info)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [*image_contents, {"type": "text", "text": prompt}]},
        ]

    def _sample_crops(self, crops: List[str]) -> List[str]:
        """均匀采样图片，避免超出 vLLM 每次请求限制。"""
        max_crops = max(self.config.max_images_per_request, 1)
        if len(crops) <= max_crops:
            return list(crops)
        step = len(crops) / max_crops
        indices = [min(int(i * step), len(crops) - 1) for i in range(max_crops)]
        return [crops[i] for i in indices]

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_prompt(
        self,
        packages: List["EvidencePackage"],
        question: str,
        plan_context: Optional[str],
        resolution: Tuple[int, int],
    ) -> str:
        telemetry_lines: List[str] = []
        res_w, res_h = resolution

        for pkg in packages:
            feats = getattr(pkg, "features", None)
            motion_desc = self._build_motion_description(feats)
            bbox = self._pick_mid_bbox(pkg.bboxes)
            norm_box = self._normalize_bbox(bbox, res_w, res_h)
            telemetry_lines.append(
                f"### Target {pkg.track_id}\n"
                f"- Location: <box2d>{norm_box}</box2d>\n"
                f"- Motion: {motion_desc}"
            )

        constraints = plan_context or "No additional constraints."
        telemetry_block = "\n".join(telemetry_lines)

        return f"""## User Query
"{question}"

## Candidates Telemetry
{telemetry_block}

## Constraints
{constraints}

## Instructions
1. For each target, focus ONLY on the region inside <box2d>.
2. Verify appearance against the query and cross-check motion telemetry.
3. Respond in JSON: {{"<track_id>": {{"match": true/false, "reason": "text", "confidence": <0-1 optional>}}}}
"""

    def _build_motion_description(self, feats) -> str:
        if not feats:
            return "No motion data available."

        parts: List[str] = []
        # Speed描述基于 norm_speed
        if feats.norm_speed < 0.1:
            parts.append("Static or barely moving")
        elif feats.norm_speed < 1.0:
            parts.append(f"Walking (norm_speed {feats.norm_speed:.1f})")
        else:
            parts.append(f"Running/fast (norm_speed {feats.norm_speed:.1f})")

        # 线性度
        if feats.linearity < 0.3:
            parts.append("Wandering/loitering path (low linearity)")
        elif feats.linearity > 0.8:
            parts.append("Direct path (high linearity)")

        # 方向/位移
        dx, dy = feats.displacement_vec
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down (towards camera)" if dy > 0 else "up (away)"
        parts.append(f"Moving {direction}")

        # 尺度变化
        if feats.scale_change > 1.2:
            parts.append("Approaching the camera (scale increasing)")
        elif feats.scale_change < 0.8:
            parts.append("Leaving the camera (scale decreasing)")

        duration = getattr(feats, "duration_s", None)
        if duration is not None:
            parts.append(f"Duration: {duration:.1f}s")

        return ". ".join(parts) + "."

    def _extract_frames(self, packages: List["EvidencePackage"]) -> Tuple[List[str], Tuple[int, int]]:
        """提取全图帧，按首个包的轨迹区间均匀采样。"""
        if not packages:
            return [], (0, 0)
        pkg0 = packages[0]
        video_path = getattr(pkg0, "video_path", "") or ""
        if not video_path:
            return [], (0, 0)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], (0, 0)

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        start_f = min((pkg.frames[0] for pkg in packages if pkg.frames), default=0)
        end_f = max((pkg.frames[-1] for pkg in packages if pkg.frames), default=start_f)
        if end_f < start_f:
            end_f = start_f
        count = max(self.config.frame_sampling_count, 1)
        step = max(1, (end_f - start_f + 1) // count)
        target_indices = list(range(start_f, end_f + 1, step))[:count]

        images_b64: List[str] = []
        for idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = self._resize_long(frame, self.config.image_resize_long)
            _, buffer = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buffer).decode("utf-8")
            images_b64.append(b64)
        cap.release()
        return images_b64, (w, h)

    @staticmethod
    def _resize_long(frame, target_long: int):
        h, w = frame.shape[:2]
        if max(h, w) <= target_long:
            return frame
        scale = target_long / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h))

    @staticmethod
    def _pick_mid_bbox(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        if not bboxes:
            return (0, 0, 0, 0)
        return bboxes[len(bboxes) // 2]

    @staticmethod
    def _normalize_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> List[int]:
        w = max(width, 1)
        h = max(height, 1)
        x1, y1, x2, y2 = bbox
        norm = [
            int(x1 / w * 1000),
            int(y1 / h * 1000),
            int(x2 / w * 1000),
            int(y2 / h * 1000),
        ]
        return [max(0, min(1000, v)) for v in norm]

    def _parse_batch_response(self, text: str, expected_ids: List[int]) -> Dict[str, VerificationResult]:
        """
        解析形如 {"123": {"match": true, "reason": "...", "confidence": 0.8}, ...} 的 JSON。
        """
        try:
            match = re.search(r"\{.*\}", text, re.S)
            json_str = match.group(0) if match else text
            data = json.loads(json_str)
            results: Dict[str, VerificationResult] = {}
            for key, val in data.items():
                match_flag = bool(val.get("match") or str(val.get("match", "")).lower() == "yes")
                conf = float(val.get("confidence", 0.0)) if isinstance(val, dict) else 0.0
                reason = val.get("reason", "") if isinstance(val, dict) else str(val)
                if match_flag:
                    results[str(key)] = VerificationResult.confirmed(conf or 0.8, reason, raw=text)
                else:
                    results[str(key)] = VerificationResult.rejected(reason or "No match", raw=text)
            if results:
                return results
        except Exception:
            pass
        # Fallback：整段用单对象解析，全部共享判定
        shared = self._parser.parse(text)
        return {str(tid): shared for tid in expected_ids}


__all__ = ["VllmAdapter", "VllmConfig"]
