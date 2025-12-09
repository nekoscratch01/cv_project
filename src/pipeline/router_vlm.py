"""vLLM 驱动的文本路由器：用 Qwen3-VL-4B (text-only) 生成执行计划。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from openai import AsyncOpenAI

from pipeline.router import ExecutionPlan

ROUTER_SYSTEM_PROMPT = """
You are the Search Planner for a video surveillance system.
Analyze the User Query to determine the search strategy.

Decision Logic:
1. need_context = false:
   - Query focuses ONLY on static appearance (clothing, color, gender, carried object).
   - Examples: "man in blue shirt", "person with a red bag".
2. need_context = true:
   - Query involves motion, environment, or interaction.
   - Examples: "running person", "leaving the shop", "wandering at the door".

Constraints (only set if explicitly implied by the query):
- norm_speed (body_heights/s): Fast >1.8; Walk 0.0-1.8; Static <0.1
  * If query does NOT mention motion speed, DO NOT set this.
- linearity (0-1): Wandering <0.3; Direct >0.7
  * Default unset if query doesn’t imply path shape.
- scale_change: Approaching >1.2; Leaving <0.8
  * Default unset if query doesn’t imply depth change.

Output JSON ONLY:
{
  "visual_description": "string (appearance only)",
  "need_context": boolean,
  "hard_rules": {
    "norm_speed": {"min": float, "max": float},
    "linearity": {"min": float, "max": float},
    "scale_change": {"min": float, "max": float}
  }
}
"""


class VlmRouter:
    """Router that uses vLLM (OpenAI compatible) to generate an ExecutionPlan."""

    def __init__(self, base_url: str, model: str = "Qwen/Qwen3-VL-4B-Instruct", api_key: str = "EMPTY"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def build_plan(self, query: str) -> ExecutionPlan:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            content = response.choices[0].message.content or ""
            payload = self._parse_json(content)
            return ExecutionPlan(
                description=payload.get("visual_description") or query,
                visual_tags=[],
                needed_facts=[],
                constraints=payload.get("hard_rules") or {},
                meta={"need_context": payload.get("need_context", False)},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Router] vLLM routing failed: {exc}. Fallback to echo plan.")
            return ExecutionPlan(description=query, constraints={})

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


__all__ = ["VlmRouter"]
