"""ExecutionPlan helpers shared by router implementations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


DEFAULT_VERIFICATION_PROMPT = (
    "Given the original question, is this track a plausible match? Answer Yes or No."
)

class SimpleRouter:
    """Minimal router that echoes the question into an ExecutionPlan."""

    def build_plan(self, question: str) -> ExecutionPlan:
        return ExecutionPlan(description=question or "a person")


@dataclass
class ExecutionPlan:
    description: str
    visual_tags: List[str] = field(default_factory=list)
    needed_facts: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    verification_prompt: str = DEFAULT_VERIFICATION_PROMPT
    meta: Dict[str, Any] = field(default_factory=dict)  # 可附加 need_context 等决策信号

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "visual_tags": list(self.visual_tags),
            "needed_facts": list(self.needed_facts),
            "constraints": dict(self.constraints),
            "verification_prompt": self.verification_prompt,
            "meta": dict(self.meta),
        }


def parse_router_output(raw_output: str) -> Tuple[ExecutionPlan, str]:
    """
    解析 Router 输出（预留将来对接 LLM）。当前实现尝试 JSON → 回退为朴素文本。
    """
    think_log = ""
    payload: Dict[str, Any]
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
    if not payload:
        plan = ExecutionPlan(description=raw_output.strip() or "a person")
        return plan, think_log
    plan = ExecutionPlan(
        description=payload.get("description") or "a person",
        visual_tags=list(payload.get("visual_tags", [])),
        needed_facts=list(payload.get("needed_facts", [])),
        constraints=dict(payload.get("constraints", {})),
        verification_prompt=payload.get("verification_prompt") or DEFAULT_VERIFICATION_PROMPT,
        meta=dict(payload.get("meta", {})),
    )
    think_log = payload.get("thoughts", "")
    return plan, think_log
