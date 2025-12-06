from __future__ import annotations

from typing import Optional

from domain.value_objects.verification_result import (
    InferenceResult,
    MatchStatus,
)
from domain.entities.evidence import EvidencePackage
from ports.inference_port import InferencePort


class MockInferenceAdapter(InferencePort):
    """Test/mock adapter that returns a fixed match decision."""

    def __init__(self, default_match: bool = True):
        self.default_match = default_match
        self.call_count = 0

    async def verify_track(
        self,
        package: EvidencePackage,
        question: str,
        plan_context: Optional[str] = None,
    ) -> InferenceResult:
        self.call_count += 1
        status = MatchStatus.CONFIRMED if self.default_match else MatchStatus.REJECTED
        return InferenceResult(
            status=status,
            confidence=0.9,
            reason="Mock response",
            raw_response="",
        )
