from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from domain.entities.track import Track


@dataclass
class EvidencePackage:
    """Evidence bundle for a person track."""

    track: Track
    crops: List[str]
    bboxes: List[Tuple[int, int, int, int]]
    features: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
