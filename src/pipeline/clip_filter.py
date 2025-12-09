"""CLIP/SigLIP based quick recall filter."""
from __future__ import annotations

import base64
import io
from typing import List, Sequence

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from core.evidence import extract_best_crop_from_package


class ClipFilter:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def score(self, text: str, image: Image.Image) -> float:
        inputs = self.processor(images=image, text=[text], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        img = outputs.image_embeds
        txt = outputs.text_embeds
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return float((img @ txt.T).squeeze().item())

    def filter_candidates(self, text: str, packages: Sequence, threshold: float = 0.2) -> List:
        kept = []
        for pkg in packages:
            crop_b64 = getattr(pkg, "best_crop_b64", "") or ""
            if not crop_b64:
                crop_b64 = extract_best_crop_from_package(pkg)
                setattr(pkg, "best_crop_b64", crop_b64)
            if not crop_b64:
                continue
            try:
                image = _b64_to_image(crop_b64)
            except Exception as e:
                print(f"[CLIP] Error processing track {getattr(pkg, 'track_id', '?')}: {e}")
                continue
            s = self.score(text, image)
            if s >= threshold:
                kept.append(pkg)
        return kept


def _b64_to_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
