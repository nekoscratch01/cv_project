"""vLLM 驱动的文本路由器：用 Qwen3-VL-4B (text-only) 生成执行计划。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from openai import AsyncOpenAI

from pipeline.router import ExecutionPlan

ROUTER_SYSTEM_PROMPT = """
You are the Search Planner for a video surveillance system.
Analyze the User Query and produce a JSON execution plan.

Decision Logic:
1) need_context = false: query only talks about appearance (color/clothes/gender/object).
   Examples: "person wearing a blue shirt", "child with a red backpack".
2) need_context = true: query involves motion/environment/interaction.
   Examples: "person running", "leaving the shop", "wandering at the door".

Constraints (only set if explicitly implied by the query):
- norm_speed (body_heights/s): Fast >1.8; Walk 0.0-1.8; Static <0.1
  * If the query does NOT mention motion speed, DO NOT set this.
- linearity (0-1): Wandering <0.3; Direct >0.7
  * Default unset if the query doesn't imply path shape.
- scale_change: Approaching >1.2; Leaving <0.8
  * Default unset if the query doesn't imply depth change.

### EXAMPLES (follow strictly)
User: "Find the person wearing a blue shirt."
Reasoning: Appearance only, no motion.
JSON:
{
  "visual_description": "person wearing a blue shirt",
  "visual_tags": ["blue shirt"],
  "hard_rules": {},
  "need_context": false
}

User: "Find the man running fast towards the camera."
Reasoning: High speed + approaching.
JSON:
{
  "visual_description": "man running fast",
  "visual_tags": ["man", "running"],
  "hard_rules": {
    "norm_speed": {"min": 1.8, "max": 10.0},
    "scale_change": {"min": 1.2, "max": 10.0}
  },
  "need_context": true
}

User: "Where is the child with the red backpack?"
Reasoning: Appearance only, no motion.
JSON:
{
  "visual_description": "child with red backpack",
  "visual_tags": ["child", "red backpack"],
  "hard_rules": {},
  "need_context": false
}

Remember: If the user does NOT mention speed/path/depth, the 'hard_rules' object MUST be empty.
Output JSON ONLY.
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
            constraints = payload.get("hard_rules") or {}

            # 兜底清洗：若查询无动作关键词则清空速度/路径/尺度约束
            ql = query.lower()
            motion_keywords = [
                "run", "running", "walk", "walking", "fast", "slow",
                "move", "moving", "leave", "leaving", "approach", "approaching",
                "wander", "wandering", "loiter", "loitering", "stand", "standing", "static",
                "enter", "entering", "exit", "exiting", "cross", "crossing", "pass", "passing",
                "chase", "chasing", "follow", "following", "stop", "stopped",
            ]
            if not any(k in ql for k in motion_keywords):
                constraints = {}
            else:
                # 移除常见幻觉的 0.8-1.2 尺度约束
                sc = constraints.get("scale_change")
                if sc:
                    try:
                        mn = float(sc.get("min", -1))
                        mx = float(sc.get("max", -1))
                        if 0.79 <= mn <= 0.81 and 1.19 <= mx <= 1.21:
                            constraints.pop("scale_change", None)
                    except Exception:
                        constraints.pop("scale_change", None)

            return ExecutionPlan(
                description=payload.get("visual_description") or query,
                visual_tags=[],
                needed_facts=[],
                constraints=constraints,
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
