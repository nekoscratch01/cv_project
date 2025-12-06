"""vLLM 推理适配器，实现 InferencePort 协议。"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

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

    async def verify_track(
        self,
        package: "EvidencePackage",
        question: str,
        plan_context: Optional[str] = None,
    ) -> VerificationResult:
        """验证单个轨迹"""
        try:
            messages = self._build_messages(package, question, plan_context)
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw_text = response.choices[0].message.content if response.choices else ""
            return self._parser.parse(raw_text or "")
        except Exception as exc:  # noqa: BLE001
            return VerificationResult.error(str(exc))

    async def verify_batch(
        self,
        packages: List["EvidencePackage"],
        question: str,
        plan_context: Optional[str] = None,
        concurrency: int = 10,
    ) -> List[VerificationResult]:
        """批量验证（真正的并发请求）"""
        semaphore = asyncio.Semaphore(concurrency)

        async def _verify_with_limit(pkg: "EvidencePackage") -> VerificationResult:
            async with semaphore:
                return await self.verify_track(pkg, question, plan_context)

        return await asyncio.gather(*(_verify_with_limit(pkg) for pkg in packages))

    def _build_messages(
        self,
        package: "EvidencePackage",
        question: str,
        plan_context: Optional[str],
    ) -> List[dict]:
        """构造 OpenAI chat 完整消息"""
        crop_paths = self._sample_crops(package.crops)

        image_contents = []
        for path in crop_paths:
            base64_image = self._encode_image(path)
            image_contents.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            )

        prompt = self._build_prompt(package, question, plan_context)

        return [
            {
                "role": "system",
                "content": "You are a video analysis assistant. Answer with reasoning, then end with 'MATCH: yes' or 'MATCH: no'.",
            },
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
        package: "EvidencePackage",
        question: str,
        plan_context: Optional[str],
    ) -> str:
        motion_desc = self._build_motion_description(package)
        return f"""## Task
Verify if this person matches the query: "{question}"

## Evidence
### Appearance
The images show the person at different moments in the video.

### Motion Summary
{motion_desc}

### Constraints
{plan_context or "No additional constraints."}

## Instructions
1. Describe what you see in the images.
2. Check if the person matches the query criteria.
3. Final line must be: MATCH: yes or MATCH: no
"""

    def _build_motion_description(self, package: "EvidencePackage") -> str:
        feats = getattr(package, "features", None)
        if not feats:
            return "No motion data available."

        parts: List[str] = []

        if feats.avg_speed_px_s < 50:
            parts.append("Standing still or barely moving")
        elif feats.avg_speed_px_s < 200:
            parts.append("Walking at normal pace")
        else:
            parts.append("Moving fast or running")

        dx, dy = feats.displacement_vec
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down (towards camera)" if dy > 0 else "up (away)"
        parts.append(f"Moving {direction}")

        duration = getattr(feats, "duration_s", None)
        if duration is not None:
            parts.append(f"Duration: {duration:.1f}s")

        return ". ".join(parts) + "."


__all__ = ["VllmAdapter", "VllmConfig"]
