"""vLLM 驱动的文本路由器：用 Qwen3-VL-4B (text-only) 生成执行计划。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from openai import AsyncOpenAI

from core.constraints import VALID_INTENTS
from pipeline.router import ExecutionPlan

ROUTER_SYSTEM_PROMPT = """
You are the QuerySpec router for a video search system.
Analyze the user query and output JSON ONLY with this schema:

{
  "positive_prompts": ["person wearing a blue shirt"],
  "negative_prompts": [],
  "need_context": false,
  "constraint_intents": []
}

Rules:
1) positive_prompts must describe APPEARANCE ONLY (color/clothes/accessories).
2) positive_prompts: 1-5 items, each <= 8 words.
3) negative_prompts: 0-3 items, only when the query explicitly negates.
4) need_context = true only if the query explicitly mentions motion, direction, interaction, or environment.
5) constraint_intents: 0-5 labels ONLY (no numbers, no thresholds).
6) Do NOT output hard_rules, min/max, or any threshold structures.
7) Output JSON only (no extra text).

Allowed constraint_intents:
RUNNING, WALKING, STOPPED, APPROACHING, LEAVING, WANDERING, MOVING_LEFT, MOVING_RIGHT
"""

ROUTER_REPAIR_PROMPT = """
Your previous output was invalid. Fix it and output ONLY valid JSON for the schema:

{
  "positive_prompts": ["person wearing a blue shirt"],
  "negative_prompts": [],
  "need_context": false,
  "constraint_intents": []
}

Rules:
- No hard_rules or numeric thresholds.
- positive_prompts: 1-5 items, each <= 8 words.
- negative_prompts: 0-3 items.
- constraint_intents: 0-5 items.
- No extra keys.
- Output JSON only.
"""


class VlmRouter:
    """Router that uses vLLM (OpenAI compatible) to generate an ExecutionPlan."""

    def __init__(self, base_url: str, model: str = "Qwen/Qwen3-VL-4B-Instruct", api_key: str = "EMPTY"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def build_plan(self, query: str) -> ExecutionPlan:
        payload, raw_text = await self._request_payload(query, ROUTER_SYSTEM_PROMPT)
        ok, reason = self._validate_payload(payload, raw_text)
        if not ok:
            payload, raw_text = await self._request_payload(query, ROUTER_REPAIR_PROMPT)
            ok, reason = self._validate_payload(payload, raw_text)
            if not ok:
                raise RuntimeError(f"Router output invalid: {reason}")

        payload = self._normalize_payload(payload)
        query_spec = {
            "positive_prompts": payload["positive_prompts"],
            "negative_prompts": payload.get("negative_prompts", []),
            "need_context": bool(payload.get("need_context", False)),
            "constraint_intents": payload.get("constraint_intents", []),
        }
        description = query_spec["positive_prompts"][0] if query_spec["positive_prompts"] else query
        return ExecutionPlan(
            description=description or query,
            visual_tags=[],
            needed_facts=[],
            constraints={},
            meta={
                "need_context": query_spec["need_context"],
                "query_spec": query_spec,
            },
        )

    async def _request_payload(self, query: str, system_prompt: str) -> Tuple[Dict[str, Any], str]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        return self._parse_json(content), content

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        positive = [str(p).strip() for p in payload.get("positive_prompts", []) if str(p).strip()]
        negative = [str(p).strip() for p in payload.get("negative_prompts", []) if str(p).strip()]
        intents = [str(i).strip().upper() for i in payload.get("constraint_intents", []) if str(i).strip()]
        return {
            "positive_prompts": positive,
            "negative_prompts": negative,
            "need_context": bool(payload.get("need_context", False)),
            "constraint_intents": intents,
        }

    @staticmethod
    def _validate_payload(payload: Dict[str, Any], raw_text: str) -> Tuple[bool, str]:
        if not payload:
            return False, "empty_json"

        allowed_keys = {"positive_prompts", "negative_prompts", "need_context", "constraint_intents"}
        unknown = [k for k in payload.keys() if k not in allowed_keys]
        if unknown:
            return False, f"unknown_keys={unknown}"

        raw_stripped = raw_text.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.S)
        if not json_match:
            return False, "missing_json_block"
        if raw_stripped != json_match.group(0).strip():
            return False, "extra_text"

        if "hard_rules" in raw_text:
            return False, "hard_rules_forbidden"
        if "norm_speed" in raw_text or "linearity" in raw_text or "scale_change" in raw_text:
            return False, "threshold_fields_forbidden"
        if re.search(r"\"(min|max)\"", raw_text):
            return False, "threshold_fields_forbidden"
        if ">" in raw_text or "<" in raw_text:
            return False, "threshold_operators_forbidden"

        positive = payload.get("positive_prompts")
        if not isinstance(positive, list) or not positive:
            return False, "positive_prompts_missing"
        if len(positive) > 5:
            return False, "positive_prompts_too_many"
        if any(not isinstance(p, str) or not p.strip() for p in positive):
            return False, "positive_prompts_invalid"
        if any(len(p.split()) > 8 for p in positive):
            return False, "positive_prompts_too_long"

        negative = payload.get("negative_prompts", [])
        if not isinstance(negative, list) or any(not isinstance(p, str) for p in negative):
            return False, "negative_prompts_invalid"
        if len(negative) > 3:
            return False, "negative_prompts_too_many"

        if not isinstance(payload.get("need_context"), bool):
            return False, "need_context_invalid"

        intents = payload.get("constraint_intents", [])
        if not isinstance(intents, list) or any(not isinstance(i, str) for i in intents):
            return False, "constraint_intents_invalid"
        if len(intents) > 5:
            return False, "constraint_intents_too_many"
        intents_upper = [i.strip().upper() for i in intents if i.strip()]
        invalid_intents = [i for i in intents_upper if i not in VALID_INTENTS]
        if invalid_intents:
            return False, f"invalid_intents={invalid_intents}"

        return True, ""


__all__ = ["VlmRouter"]
