from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol

if TYPE_CHECKING:
    from domain.value_objects.verification_result import VerificationResult
    from core.evidence import EvidencePackage


class InferencePort(Protocol):
    """
    推理端口协议

    设计原则：
    - 业务层只依赖此接口，不依赖具体实现
    - 支持 vLLM、量化模型、云端 API 等多种后端
    - 异步优先，支持高并发
    """

    async def verify_track(
        self,
        package: "EvidencePackage",
        question: str,
        plan_context: Optional[str] = None,
    ) -> "VerificationResult":
        ...

    async def verify_batch(
        self,
        packages: List["EvidencePackage"],
        question: str,
        plan_context: Optional[str] = None,
        concurrency: int = 10,
    ) -> List["VerificationResult"]:
        """
        批量验证轨迹（真正的并发）

        与 HF transformers 的 Batch 不同：
        - HF Batch: 同一个 forward pass，受 padding 影响
        - vLLM 并发: 多个独立请求，vLLM 内部 Continuous Batching
        """
        ...
