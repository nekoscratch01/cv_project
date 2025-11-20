"""LLM-powered router (transformers-only)."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from core.config import SystemConfig
from pipeline.router import ExecutionPlan, parse_router_output


class HFRouter:
    """Router powered by transformers (共享 Qwen3-VL-4B 模型，文本模式)."""

    def __init__(self, config: SystemConfig, hf_client: Any | None = None) -> None:
        self.config = config
        self._external_client = hf_client
        if hf_client is not None and hasattr(hf_client, "model"):
            self.processor = getattr(hf_client, "processor")
            self.model = hf_client.model
        else:
            from pipeline.vlm_client_hf import Qwen3VL4BHFClient

            self._external_client = Qwen3VL4BHFClient(config)
            self.processor = self._external_client.processor
            self.model = self._external_client.model

        import torch

        self._torch = torch

    def build_plan(self, user_query: str) -> ExecutionPlan:
        prompt = _build_router_prompt(user_query.strip())
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Edge-Detective Router. Reply with only JSON ExecutionPlan.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.router_max_new_tokens,
                temperature=self.config.router_temperature,
            )
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        plan, _ = parse_router_output(text)
        return plan


def _build_router_prompt(user_query: str) -> str:
    few_shot = [
        {
            "question": "帮我找最后一个进店的人",
            "plan": {
                "description": "people entering the shop",
                "visual_tags": [],
                "needed_facts": ["start_s", "end_s", "centroids"],
                "constraints": {
                    "roi": "shop_door",
                    "event_type": "enter",
                    "sort_by": "end_s",
                    "sort_order": "desc",
                    "limit": 1,
                },
                "verification_prompt": "",
            },
        },
        {
            "question": "找穿红衣服背书包的人",
            "plan": {
                "description": "a person wearing red clothes and carrying a backpack",
                "visual_tags": ["red clothes", "backpack"],
                "needed_facts": [],
                "constraints": {"limit": 50},
                "verification_prompt": (
                    "Given the original question, is this track a plausible match? Answer Yes or No."
                ),
            },
        },
        {
            "question": "谁是跑得最快的红衣人？",
            "plan": {
                "description": "a person wearing red clothes",
                "visual_tags": ["red clothes"],
                "needed_facts": ["avg_speed_px_s"],
                "constraints": {"sort_by": "avg_speed_px_s", "sort_order": "desc", "limit": 1},
                "verification_prompt": (
                    "Given the original question, is this track a plausible match? Answer Yes or No."
                ),
            },
        },
    ]
    examples = "\n".join(
        f"Question: {item['question']}\nPlan JSON:\n{json.dumps(item['plan'], ensure_ascii=False)}\n"
        for item in few_shot
    )
    template = dedent(
        """
        Convert user questions into a JSON ExecutionPlan with keys:
        description, visual_tags, needed_facts, constraints, verification_prompt.
        Use only the defined keys. Examples:
        {examples}

        User query: {query}
        Respond ONLY with JSON.
        """
    ).strip()
    return template.format(examples=examples, query=user_query or "找一个人")
