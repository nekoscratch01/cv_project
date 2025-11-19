"""Recall stage for candidate track selection."""

from __future__ import annotations

from typing import Dict, List

from evidence import EvidencePackage


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

    def recall(
        self,
        question: str,
        evidence_map: Dict[int, EvidencePackage],
        limit: int | None = None,
    ) -> List[EvidencePackage]:
        """
        从所有轨迹中召回候选集。
        
        Phase 1 实现：直接返回所有轨迹（或前N条）。
        这是最保险的"无召回"版本，保证不会漏掉任何目标。
        
        Args:
            question: 用户的查询问题，例如 "找穿紫色衣服的人"
                     Phase 1 暂不使用这个参数，但预留接口供未来版本使用
            evidence_map: 所有轨迹的证据包字典，格式：{track_id: EvidencePackage}
                         例如：{1: EvidencePackage(...), 2: EvidencePackage(...), ...}
            limit: 可选的召回数量限制。如果指定，最多返回前 limit 条轨迹
                  例如：limit=20 表示最多返回20个候选
                  如果为 None，返回所有轨迹
        
        Returns:
            候选证据包列表，格式：[EvidencePackage, EvidencePackage, ...]
            如果指定了 limit，列表长度 <= limit
            如果未指定 limit，列表长度 = len(evidence_map)
        
        Note:
            - Phase 1 返回的顺序是字典的插入顺序（通常是 track_id 的顺序）
            - 未来版本可能会根据 question 做相似度排序，返回最相关的前N条
            - limit 参数用于控制VLM的工作量，避免一次处理太多候选
        
        使用示例：
            # 不限制数量，返回所有轨迹（适合小规模场景）
            all_candidates = engine.recall("找人", evidence_map)
            
            # 限制数量，最多返回20个候选（适合大规模场景）
            top_candidates = engine.recall("找穿红衣服的人", evidence_map, limit=20)
        """
        candidates = list(evidence_map.values())
        if limit is not None:
            return candidates[:limit]
        return candidates
