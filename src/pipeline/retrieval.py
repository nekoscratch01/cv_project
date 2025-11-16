"""Retrieval/query layer built on semantic profiles."""

from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path

import cv2
import numpy as np

from .config import SystemConfig
from .perception import TrackRecord
from .semantic import SemanticProfile


class SemanticRetrievalEngine:
    def __init__(
        self,
        config: SystemConfig,
        track_records: Dict[int, TrackRecord],
        profiles: Dict[int, SemanticProfile],
    ) -> None:
        self.config = config
        self.track_records = track_records
        self.profiles = profiles

    def search_structured(self, conditions: List[Tuple[str, str]]) -> List[int]:
        """Simple AND filtering over attribute key/value pairs."""
        candidate = set(self.profiles.keys())
        for key, expected in conditions:
            expected_lower = expected.lower()
            matched = {
                tid
                for tid, profile in self.profiles.items()
                if key in profile.attributes
                and expected_lower in str(profile.attributes[key]).lower()
            }
            candidate &= matched
        return sorted(candidate)

    def search_text(self, query: str) -> List[int]:
        query_lower = query.lower()
        matches = [
            tid
            for tid, profile in self.profiles.items()
            if query_lower in profile.description.lower()
        ]
        return matches

    def visualize(self, target_ids: List[int], output_path: Path) -> None:
        if not target_ids:
            print("   ⚠️  没有结果可视化")
            return

        images = []
        for tid in target_ids[: self.config.max_preview_tracks]:
            record = self.track_records.get(tid)
            if not record or not record.crops:
                continue
            img = cv2.imread(record.crops[0])
            if img is None:
                continue
            cv2.putText(
                img,
                f"ID:{tid}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            images.append(img)

        if not images:
            print("   ⚠️  没有可用图片")
            return

        max_height = max(img.shape[0] for img in images)
        resized = []
        for img in images:
            h, w = img.shape[:2]
            scale = max_height / max(h, 1)
            resized.append(cv2.resize(img, (int(w * scale), max_height)))

        result_img = np.hstack(resized)
        cv2.imwrite(str(output_path), result_img)
        print(f"   ✅ 结果保存: {output_path}")
