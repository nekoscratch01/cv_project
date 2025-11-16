"""Semantic layer: convert crops + features into human-readable profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info

from .config import SystemConfig
from .perception import TrackRecord
from .features import TrackFeatures


@dataclass
class SemanticProfile:
    track_id: int
    attributes: Dict[str, Optional[str]]
    description: str
    features: Optional[TrackFeatures]

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "attributes": self.attributes,
            "description": self.description,
        }
        if self.features is not None:
            payload["features"] = self.features.to_dict()
        return payload


class SemanticDescriptor:
    """Leverages a VLM to annotate each track with semantic attributes."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vlm_model,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(config.vlm_model)

    def describe_tracks(
        self,
        track_records: Dict[int, TrackRecord],
        features: Dict[int, TrackFeatures],
    ) -> Dict[int, SemanticProfile]:
        profiles: Dict[int, SemanticProfile] = {}

        for track_id, record in track_records.items():
            if not record.crops:
                continue

            crop_path = record.crops[0]
            attributes = self._extract_attributes(crop_path)
            description = self._generate_description(crop_path)
            profile = SemanticProfile(
                track_id=track_id,
                attributes=attributes,
                description=description,
                features=features.get(track_id),
            )
            profiles[track_id] = profile

        return profiles

    def _extract_attributes(self, image_path: str) -> Dict[str, Optional[str]]:
        questions = {
            "color": "What is the main color of this person's clothing? Answer with one color word only.",
            "upper_color": "What color is this person's upper body clothing? Answer with one color word only.",
            "has_backpack": "Is this person carrying a backpack? Answer yes or no.",
        }

        attrs: Dict[str, Optional[str]] = {}
        for key, question in questions.items():
            answer = self._query_image(image_path, question)
            if key.startswith("has_"):
                attrs[key] = "yes" if any(
                    term in answer.lower() for term in ["yes", "yeah", "true", "yep"]
                ) else "no"
            else:
                attrs[key] = answer.lower().strip()
        return attrs

    def _generate_description(self, image_path: str) -> str:
        question = (
            "Describe this person's clothing style and visible belongings in one concise sentence."
        )
        return self._query_image(image_path, question)

    def _query_image(self, image_path: str, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return answer.strip()
