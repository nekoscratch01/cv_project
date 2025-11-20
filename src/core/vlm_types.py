"""Shared VLM result datatypes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryResult:
    """
    VLM 的查询结果：一个匹配的轨迹及其详细信息。

    当前 v7 中，所有具体 VLM 客户端（例如 Qwen3VL4BHFClient）都会返回该类型。
    """

    track_id: int
    start_s: float
    end_s: float
    score: float
    reason: str
