"""Recall stage for candidate track selection."""

from __future__ import annotations

from typing import Dict, List

from .evidence import EvidencePackage


class RecallEngine:
    """Lightweight recall. Phase 1 baseline simply returns all evidence."""

    def recall(
        self,
        question: str,
        evidence_map: Dict[int, EvidencePackage],
        limit: int | None = None,
    ) -> List[EvidencePackage]:
        candidates = list(evidence_map.values())
        if limit is not None:
            return candidates[:limit]
        return candidates
