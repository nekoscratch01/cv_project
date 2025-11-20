"""Recall stage for candidate track selection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image

from core.config import SystemConfig
from core.evidence import EvidencePackage
from core.siglip_client import SiglipClient


class RecallEngine:
    """
    召回引擎：快速筛选候选轨迹，减少VLM的工作量。
    
    在"问题驱动检索"的两阶段架构中，召回是第一阶段：
    - 第一阶段（召回）：快速粗筛，从所有轨迹中选出候选集（例如从100条筛到20条）
    - 第二阶段（VLM精排）：慢速精判，让VLM仔细看每个候选，给出最终答案
    
    Phase 1 的召回策略：
        v0（当前实现）：直接返回所有轨迹（即"无召回"的退化版本）
        这是最简单但最保险的方案，保证100%召回率，适合小规模场景（几十条轨迹）
    
    未来可能的增强（Phase 2+）：
        v1：用 CLIP 做图文相似度匹配，过滤掉明显不相关的轨迹
        v2：用颜色直方图、运动特征等做规则过滤
        v3：用向量数据库做语义检索
    
    设计原则：
        - 召回阶段只负责"减负"，不做最终决策
        - 宁可多召回（高召回率），也不要漏掉真正的目标（避免错杀）
        - 接口保持稳定，内部实现可以随时升级
    
    使用示例：
        engine = RecallEngine()
        candidates = engine.recall("找穿红衣服的人", evidence_map, limit=20)
        # 从 evidence_map 中选出最多20个候选，交给VLM精排
    """

    def __init__(
        self,
        config: SystemConfig | None = None,
        siglip_client: SiglipClient | None = None,
    ) -> None:
        self.config = config or SystemConfig()
        self._siglip_client: SiglipClient | None = siglip_client

    def recall(
        self,
        question: str,
        evidence_map: Dict[int, EvidencePackage],
        limit: int | None = None,
    ) -> List[EvidencePackage]:
        """
        Phase 1 兼容函数：直接调用 visual_filter（描述=question，tags=[]）。
        """
        top_k = limit if limit is not None else len(evidence_map)
        return self.visual_filter(
            list(evidence_map.values()),
            description=question,
            visual_tags=[],
            top_k=top_k,
        )

    def visual_filter(
        self,
        tracks: Sequence[EvidencePackage],
        description: str,
        visual_tags: List[str],
        top_k: int | None = 50,
    ) -> List[EvidencePackage]:
        """
        v7 风格接口：根据描述/标签选出 Top-K 候选（当前实现为 stub）。
        
        Args:
            tracks: 候选轨迹列表
            description: Router 生成的简化描述
            visual_tags: 关键视觉标签
            top_k: 最多返回多少条（None/<=0 表示不过滤）
        
        Returns:
            List[EvidencePackage]: 长度 <= top_k 的候选集合
        """
        if not tracks:
            return []

        if not description and not visual_tags:
            return tracks[: top_k] if top_k else list(tracks)

        embeddings = self._ensure_track_embeddings(tracks)
        query_vec = self._encode_query(description, visual_tags)
        if query_vec is None:
            return tracks[: top_k] if top_k else list(tracks)

        scores = []
        for pkg in tracks:
            track_vecs = embeddings.get(pkg.track_id)
            if track_vecs is None or len(track_vecs) == 0:
                scores.append((0.0, pkg))
                continue
            sims = track_vecs @ query_vec
            scores.append((float(np.max(sims)), pkg))

        scores.sort(key=lambda x: x[0], reverse=True)

        if top_k is None or top_k <= 0:
            return [pkg for _, pkg in scores]
        return [pkg for _, pkg in scores[: min(top_k, len(scores))]]

    def _ensure_siglip(self) -> SiglipClient:
        if self._siglip_client is None:
            self._siglip_client = SiglipClient(
                model_name=self.config.siglip_model_name,
                device=self.config.siglip_device,
            )
        return self._siglip_client

    def _ensure_track_embeddings(
        self, tracks: Sequence[EvidencePackage]
    ) -> Dict[int, np.ndarray]:
        cache: Dict[int, np.ndarray] = {}
        siglip = self._ensure_siglip()

        for pkg in tracks:
            if pkg.embedding:
                arr = np.array(pkg.embedding, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                cache[pkg.track_id] = arr
                continue

            cache_path = self._embedding_cache_path(pkg)
            if cache_path.exists():
                cache[pkg.track_id] = np.load(cache_path, allow_pickle=False)
                continue

            images = self._load_crop_images(pkg, max_images=3)
            if not images:
                cache[pkg.track_id] = np.zeros((0, siglip.embedding_dim), dtype=np.float32)
                continue
            emb = siglip.encode_images(images)
            cache[pkg.track_id] = emb
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, emb)

        return cache

    def _encode_query(self, description: str, visual_tags: List[str]) -> np.ndarray | None:
        text_parts = []
        if description:
            text_parts.append(description)
        if visual_tags:
            text_parts.append(", ".join(visual_tags))
        if not text_parts:
            return None
        query_text = ". ".join(text_parts)
        vec = self._ensure_siglip().encode_text([query_text])
        if vec.size == 0:
            return None
        return vec[0]

    def _embedding_cache_path(self, package: EvidencePackage) -> Path:
        video_id = None
        if package.meta:
            video_id = package.meta.get("video_id")
        if not video_id:
            video_id = package.video_id
        base = self.config.embedding_cache_dir / str(video_id)
        return base / f"track_{package.track_id}.npy"

    def _load_crop_images(self, package: EvidencePackage, max_images: int) -> List[Image.Image]:
        images = []
        for crop_path in package.crops[:max_images]:
            try:
                with Image.open(crop_path) as img:
                    images.append(img.convert("RGB"))
            except FileNotFoundError:
                continue
        return images
