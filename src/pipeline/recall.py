"""Recall stage for candidate track selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from core.config import SystemConfig
from core.constraints import score_constraints
from core.evidence import EvidencePackage
from core.query_spec import QuerySpec, QuerySpecExtractor


@dataclass
class SiglipRankDebug:
    track_id: int
    final_rank: float
    siglip_score: float
    constraint_score: float
    quality_score: float
    constraint_breakdown: Dict[str, float]
from core.siglip_client import SiglipClient


class RecallEngine:
    """
    召回引擎：快速筛选候选轨迹，减少VLM的工作量。
    
    在"问题驱动检索"的两阶段架构中，召回是第一阶段：
    - 第一阶段（召回）：快速粗筛，从所有轨迹中选出候选集（例如从100条筛到20条）
    - 第二阶段（VLM精排）：慢速精判，让VLM仔细看每个候选，给出最终答案
    
    Phase 2.7 的召回策略：
        v1：QuerySpec 生成 CLIP-friendly prompts
        v2：SigLIP Soft-Rerank（top-m mean + margin 自适应候选池）
        目标：不做硬阈值误杀，只做排序压缩
    
    未来可能的增强（Phase 3+）：
        v3：颜色/质量等特征融合加权
        v4：向量数据库 + ANN 检索
    
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
        self._query_spec_extractor = QuerySpecExtractor(self.config)

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

    def precompute_embeddings(
        self,
        tracks: Sequence[EvidencePackage],
        *,
        frames_per_track: int | None = None,
    ) -> int:
        max_images = frames_per_track if frames_per_track is not None else getattr(self.config, "siglip_frames_per_track", 3)
        embeddings = self._ensure_track_embeddings(tracks, max_images=max_images)
        return len(embeddings)

    def visual_filter(
        self,
        tracks: Sequence[EvidencePackage],
        description: str,
        visual_tags: List[str],
        top_k: int | None = 50,
    ) -> List[EvidencePackage]:
        """
        v7 风格接口：根据描述/标签做 SigLIP 排序并返回 Top-K 候选。
        
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

        query_parts = []
        if description:
            query_parts.append(description)
        if visual_tags:
            query_parts.append(", ".join(visual_tags))
        query_text = ". ".join(query_parts)
        query_spec = self._query_spec_extractor.extract(query_text)

        scores = self.score_tracks(
            tracks,
            positive_prompts=query_spec.positive_prompts,
            negative_prompts=query_spec.negative_prompts,
            frames_per_track=getattr(self.config, "siglip_frames_per_track", 3),
            topm=getattr(self.config, "siglip_topm_frames", 2),
            neg_lambda=getattr(self.config, "siglip_neg_prompt_lambda", 0.5),
        )
        scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is None or top_k <= 0:
            return [pkg for pkg, _ in scores]
        return [pkg for pkg, _ in scores[: min(top_k, len(scores))]]

    def siglip_soft_rerank(
        self, tracks: Sequence[EvidencePackage], query_spec: QuerySpec
    ) -> tuple[List[EvidencePackage], List[SiglipRankDebug]]:
        if not tracks:
            return [], []
        budget = query_spec.budget
        siglip_scores = self.score_tracks(
            tracks,
            positive_prompts=query_spec.positive_prompts,
            negative_prompts=query_spec.negative_prompts,
            frames_per_track=budget.frames_per_track,
            topm=budget.topm_frames,
            neg_lambda=getattr(self.config, "siglip_neg_prompt_lambda", 0.5),
        )
        enable_constraints = getattr(self.config, "enable_constraints", True)
        constraint_weight = getattr(self.config, "constraint_weight", 0.2)
        quality_weight = getattr(self.config, "quality_weight", 0.05)
        intents = list(query_spec.constraint_intents) if query_spec.constraint_intents else []

        ranked_items: list[tuple[EvidencePackage, float, float, float, float, Dict[str, float]]] = []
        for pkg, siglip_score in siglip_scores:
            constraint_score = 0.0
            breakdown: Dict[str, float] = {}
            if enable_constraints and intents:
                constraint_score, breakdown = score_constraints(intents, pkg.features)
            quality_score = self._estimate_quality_score(pkg)
            final_rank = siglip_score + constraint_weight * constraint_score + quality_weight * quality_score
            ranked_items.append((pkg, final_rank, siglip_score, constraint_score, quality_score, breakdown))

        ranked_items.sort(key=lambda x: x[1], reverse=True)
        selected = self._select_candidates(
            [(pkg, final_rank) for pkg, final_rank, _, _, _, _ in ranked_items],
            k_min=budget.k_min,
            k_max=budget.k_max,
            margin_delta=budget.margin_delta,
        )
        candidates = [pkg for pkg, _ in selected]
        ranked_scores = [
            SiglipRankDebug(
                track_id=pkg.track_id,
                final_rank=final_rank,
                siglip_score=siglip_score,
                constraint_score=constraint_score,
                quality_score=quality_score,
                constraint_breakdown=breakdown,
            )
            for pkg, final_rank, siglip_score, constraint_score, quality_score, breakdown in ranked_items
        ]
        return candidates, ranked_scores

    def score_tracks(
        self,
        tracks: Sequence[EvidencePackage],
        *,
        positive_prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        frames_per_track: int | None = None,
        topm: int | None = None,
        neg_lambda: float | None = None,
    ) -> List[tuple[EvidencePackage, float]]:
        if not tracks:
            return []

        pos_prompts = [p for p in positive_prompts if p]
        if not pos_prompts:
            return [(pkg, 0.0) for pkg in tracks]

        max_images = frames_per_track if frames_per_track is not None else getattr(self.config, "siglip_frames_per_track", 3)
        topm_val = topm if topm is not None else getattr(self.config, "siglip_topm_frames", 2)
        lam = neg_lambda if neg_lambda is not None else getattr(self.config, "siglip_neg_prompt_lambda", 0.5)

        pos_vecs = self._encode_prompts(pos_prompts)
        neg_vecs = self._encode_prompts(negative_prompts) if negative_prompts else None

        embeddings = self._ensure_track_embeddings(tracks, max_images=max_images)

        scores: List[tuple[EvidencePackage, float]] = []
        for pkg in tracks:
            img = embeddings.get(pkg.track_id)
            score = self._score_track(img, pos_vecs, neg_vecs, topm=topm_val, lam=lam)
            scores.append((pkg, score))
        return scores

    def _score_track(
        self,
        img_embeds: np.ndarray | None,
        pos_txt_embeds: np.ndarray | None,
        neg_txt_embeds: np.ndarray | None,
        *,
        topm: int,
        lam: float,
    ) -> float:
        if img_embeds is None or img_embeds.size == 0 or pos_txt_embeds is None or pos_txt_embeds.size == 0:
            return 0.0

        img = self._l2norm(img_embeds)
        pos = self._l2norm(pos_txt_embeds)
        sims = pos @ img.T
        topm = min(max(topm, 1), sims.shape[1])
        top = np.partition(sims, -topm, axis=1)[:, -topm:]
        pos_score = float(top.mean(axis=1).max())

        if neg_txt_embeds is None or neg_txt_embeds.size == 0:
            return pos_score

        neg = self._l2norm(neg_txt_embeds)
        nsims = neg @ img.T
        ntop = np.partition(nsims, -topm, axis=1)[:, -topm:]
        neg_score = float(ntop.mean(axis=1).max())
        return pos_score - lam * neg_score

    def _encode_prompts(self, prompts: Sequence[str] | None) -> np.ndarray | None:
        if not prompts:
            return None
        clean = [p for p in prompts if p]
        if not clean:
            return None
        vec = self._ensure_siglip().encode_text(list(clean))
        if vec.size == 0:
            return None
        return self._l2norm(vec)

    @staticmethod
    def _select_candidates(
        ranked: Sequence[tuple[EvidencePackage, float]],
        *,
        k_min: int,
        k_max: int,
        margin_delta: float,
    ) -> List[tuple[EvidencePackage, float]]:
        if not ranked:
            return []

        k_min = max(0, k_min)
        k_max = max(k_min, k_max)
        k_max = min(k_max, len(ranked))

        best = ranked[0][1]
        keep = [item for item in ranked if item[1] >= best - margin_delta]

        if len(keep) < k_min:
            keep = list(ranked[:k_min])
        if len(keep) > k_max:
            keep = list(keep[:k_max])
        return keep

    @staticmethod
    def _l2norm(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        if arr.ndim == 1:
            arr = arr[None, :]
        denom = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-12
        return arr / denom

    def _ensure_siglip(self) -> SiglipClient:
        if self._siglip_client is None:
            self._siglip_client = SiglipClient(
                model_name=self.config.siglip_model_name,
                device=self.config.siglip_device,
            )
        return self._siglip_client

    def _ensure_track_embeddings(
        self, tracks: Sequence[EvidencePackage], *, max_images: int | None = None
    ) -> Dict[int, np.ndarray]:
        cache: Dict[int, np.ndarray] = {}
        siglip = self._ensure_siglip()
        max_images = max_images if max_images is not None else getattr(self.config, "siglip_frames_per_track", 3)

        for pkg in tracks:
            if getattr(pkg, "siglip_img_embeds", None):
                arr = np.array(pkg.siglip_img_embeds, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                cache[pkg.track_id] = self._l2norm(arr)
                continue

            if pkg.embedding:
                arr = np.array(pkg.embedding, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                cache[pkg.track_id] = self._l2norm(arr)
                continue

            cache_path = self._embedding_cache_path(pkg)
            if cache_path.exists():
                arr = np.load(cache_path, allow_pickle=False)
                cache[pkg.track_id] = self._l2norm(arr)
                pkg.siglip_img_embeds = arr.tolist()
                continue

            images = self._load_crop_images(pkg, max_images=max_images)
            if not images:
                cache[pkg.track_id] = np.zeros((0, siglip.embedding_dim), dtype=np.float32)
                continue
            emb = siglip.encode_images(images)
            cache[pkg.track_id] = emb
            pkg.siglip_img_embeds = emb.tolist()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, emb)

        return cache

    def _embedding_cache_path(self, package: EvidencePackage) -> Path:
        video_id = None
        if package.meta:
            video_id = package.meta.get("video_id")
        if not video_id:
            video_id = package.video_id
        base = self.config.embedding_cache_dir / str(video_id)
        return base / f"track_{package.track_id}.npy"

    def _sample_crop_paths(self, crops: Sequence[str], max_images: int) -> List[str]:
        if max_images <= 0 or not crops:
            return []
        if len(crops) <= max_images:
            return list(crops)
        if max_images == 1:
            return [crops[len(crops) // 2]]
        indices = np.linspace(0, len(crops) - 1, num=max_images)
        idxs: List[int] = []
        for idx in indices:
            ridx = int(round(float(idx)))
            if ridx not in idxs:
                idxs.append(ridx)
        return [crops[i] for i in idxs if 0 <= i < len(crops)]

    def _load_crop_images(self, package: EvidencePackage, max_images: int) -> List[Image.Image]:
        images = []
        crop_paths = self._get_crop_paths(package)
        for crop_path in self._sample_crop_paths(crop_paths, max_images):
            try:
                with Image.open(crop_path) as img:
                    images.append(img.convert("RGB"))
            except FileNotFoundError:
                continue
        return images

    @staticmethod
    def _get_crop_paths(package: EvidencePackage) -> List[str]:
        crops_k = getattr(package, "crops_k", None)
        if crops_k:
            return list(crops_k)
        return list(package.crops) if package.crops else []

    @staticmethod
    def _estimate_quality_score(package: EvidencePackage) -> float:
        meta = package.meta or {}
        resolution = meta.get("resolution")
        if not resolution:
            return 0.0
        width, height = resolution
        if not width or not height:
            return 0.0
        areas = []
        for x1, y1, x2, y2 in package.bboxes:
            area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if area > 0:
                areas.append(area)
        if not areas:
            return 0.0
        median_area = float(np.median(areas))
        return float(min(max(median_area / float(width * height), 0.0), 1.0))
