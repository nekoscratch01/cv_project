"""SigLIP wrapper for image/text embeddings and caching support."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor


class SiglipClient:
    """Helper around SigLIP base checkpoints.

    Provides normalized embeddings for images and text separately. The processor handles
    both modalities; we expose encode_images and encode_text to upstream code.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.projection_dim)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 只通过视觉分支获取图像特征，避免触发文本分支对 input_ids 的要求
        emb = self.model.get_image_features(**inputs)  # [B, D]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 只通过文本分支获取文本特征
        emb = self.model.get_text_features(**inputs)  # [B, D]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    @staticmethod
    def load_image(path: str | Path) -> Image.Image:
        return Image.open(path).convert("RGB")
