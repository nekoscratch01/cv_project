from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class MatchStatus(Enum):
    """匹配状态枚举"""

    CONFIRMED = "confirmed"  # 确认匹配
    REJECTED = "rejected"  # 确认不匹配
    AMBIGUOUS = "ambiguous"  # 模糊/无法判断


@dataclass(frozen=True)
class VerificationResult:
    """
    验证结果值对象（不可变）

    这是反腐败层(ACL)的输出，将 VLM 的自然语言响应
    转换为系统内部的结构化表示。
    """

    status: MatchStatus
    confidence: float
    reason: str
    raw_response: str

    @classmethod
    def confirmed(cls, confidence: float, reason: str, raw: str = "") -> "VerificationResult":
        return cls(MatchStatus.CONFIRMED, confidence, reason, raw)

    @classmethod
    def rejected(cls, reason: str, raw: str = "") -> "VerificationResult":
        return cls(MatchStatus.REJECTED, 0.0, reason, raw)

    @classmethod
    def error(cls, error_msg: str) -> "VerificationResult":
        return cls(MatchStatus.AMBIGUOUS, 0.0, f"Error: {error_msg}", "")

    @property
    def is_match(self) -> bool:
        return self.status == MatchStatus.CONFIRMED


class VlmResponseParser:
    """
    VLM 响应解析器（反腐败层实现）

    职责：
    1. 从自然语言响应中提取结构化信息
    2. 处理各种边界情况和异常格式
    3. 将不可靠的外部数据转换为可靠的内部表示
    """

    MATCH_PATTERN = re.compile(r"MATCH:\s*(yes|no)", re.IGNORECASE)
    CONFIDENCE_PATTERN = re.compile(r"confidence[:\s]+(\d+(?:\.\d+)?)", re.IGNORECASE)

    def parse(self, raw_response: str) -> VerificationResult:
        """解析 VLM 原始响应"""
        if not raw_response:
            return VerificationResult.error("Empty response")

        match_result = self._extract_match_marker(raw_response)
        confidence = self._extract_confidence(raw_response)
        status = self._determine_status(match_result, confidence)

        return VerificationResult(
            status=status,
            confidence=confidence,
            reason=self._extract_reason(raw_response),
            raw_response=raw_response,
        )

    def _extract_match_marker(self, text: str) -> bool | None:
        match = self.MATCH_PATTERN.search(text)
        if match:
            return match.group(1).lower() == "yes"
        return None

    def _extract_confidence(self, text: str) -> float:
        match = self.CONFIDENCE_PATTERN.search(text)
        if match:
            try:
                conf = float(match.group(1))
                return min(max(conf, 0.0), 1.0)
            except ValueError:
                pass
        return 0.8 if self._extract_match_marker(text) is not None else 0.5

    def _determine_status(self, match_result: bool | None, confidence: float) -> MatchStatus:
        if match_result is True and confidence >= 0.6:
            return MatchStatus.CONFIRMED
        elif match_result is False:
            return MatchStatus.REJECTED
        else:
            return MatchStatus.AMBIGUOUS

    def _extract_reason(self, text: str) -> str:
        lines = text.strip().split("\n")
        reason_lines = [line for line in lines if not line.strip().lower().startswith("match:")]
        return " ".join(reason_lines).strip()[:500]


# Backward compatibility: keep old name used in existing modules/tests.
InferenceResult = VerificationResult

__all__ = ["VerificationResult", "VlmResponseParser", "MatchStatus", "InferenceResult"]
