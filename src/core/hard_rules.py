"""Hard Rule Engine operating on Atomic 8 features (with direction voting)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from core.config import SystemConfig
from core.evidence import EvidencePackage
from core.perception import VideoMetadata


@dataclass
class HardRuleEngine:
    """
    在 Atomic 8 空间中执行几何/时间约束。

    设计目标：
        - 与 ExecutionPlan.constraints 保持一致（ROI、时间窗、排序、limit 等）
        - 纯数学判断，不做任何语义推理
        - 兼容“纯字典”或未来的 ExecutionPlan 对象
    """

    config: SystemConfig
    metadata: Optional[VideoMetadata] = None

    def apply_constraints(
        self,
        tracks: Sequence[EvidencePackage],
        plan_or_constraints: Any = None,
    ) -> List[EvidencePackage]:
        """按约束过滤+排序候选轨迹。"""
        constraints = self._extract_constraints(plan_or_constraints)
        filtered = list(tracks)
        if not constraints:
            return filtered

        filtered = self._apply_roi_constraints(filtered, constraints)
        filtered = self._apply_time_window(filtered, constraints)
        filtered = self._apply_direction_filter(filtered, constraints)
        filtered = self._apply_thresholds(filtered, constraints)
        filtered = self._apply_sorting(filtered, constraints)
        limit = constraints.get("limit")
        if isinstance(limit, int) and limit > 0:
            filtered = filtered[:limit]
        return filtered

    def _extract_constraints(self, payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
        constraints = getattr(payload, "constraints", None)
        if constraints is None:
            return {}
        return dict(constraints)

    def _apply_roi_constraints(
        self,
        tracks: List[EvidencePackage],
        constraints: Mapping[str, Any],
    ) -> List[EvidencePackage]:
        roi_name = constraints.get("roi")
        if not roi_name or not self.config.roi_zones:
            return tracks
        event_type = constraints.get("event_type") or "enter"
        min_dwell = float(constraints.get("min_dwell_s", 0.0))
        result: List[EvidencePackage] = []
        for pkg in tracks:
            features = pkg.features
            if not features or not features.centroids:
                continue
            bounds = self._normalized_roi_bounds(roi_name, pkg)
            if not bounds:
                continue
            inside_mask = [self._point_inside(pt, bounds) for pt in features.centroids]
            if not any(inside_mask):
                continue
            if event_type == "stay":
                dwell = self._estimate_dwell_seconds(pkg, inside_mask)
                if dwell + 1e-6 < min_dwell:
                    continue
            elif event_type == "cross":
                if inside_mask[0] == inside_mask[-1]:
                    # 没有明显的进出变化
                    continue
            elif event_type == "exit":
                if not inside_mask[-1]:
                    continue
            # enter/default: 只要进入过即可
            result.append(pkg)
        return result

    def _apply_time_window(
        self,
        tracks: List[EvidencePackage],
        constraints: Mapping[str, Any],
    ) -> List[EvidencePackage]:
        time_window = constraints.get("time_window")
        if not time_window or len(time_window) != 2:
            return tracks
        start, end = float(time_window[0]), float(time_window[1])
        if start > end:
            start, end = end, start
        result = []
        for pkg in tracks:
            features = pkg.features
            if not features:
                continue
            if features.end_s < start or features.start_s > end:
                continue
            result.append(pkg)
        return result

    def _apply_thresholds(
        self,
        tracks: List[EvidencePackage],
        constraints: Mapping[str, Any],
    ) -> List[EvidencePackage]:
        result = []
        for pkg in tracks:
            features = pkg.features
            if not features:
                continue

            # norm_speed 区间过滤
            ns = constraints.get("norm_speed")
            if ns:
                ns_min = float(ns.get("min", float("-inf")))
                ns_max = float(ns.get("max", float("inf")))
                if features.norm_speed + 1e-6 < ns_min:
                    continue
                if features.norm_speed - 1e-6 > ns_max:
                    continue

            # linearity 区间过滤
            lin = constraints.get("linearity")
            if lin:
                lin_min = float(lin.get("min", float("-inf")))
                lin_max = float(lin.get("max", float("inf")))
                if features.linearity + 1e-6 < lin_min:
                    continue
                if features.linearity - 1e-6 > lin_max:
                    continue

            # scale_change 区间过滤
            sc = constraints.get("scale_change")
            if sc:
                sc_min = float(sc.get("min", float("-inf")))
                sc_max = float(sc.get("max", float("inf")))
                if features.scale_change + 1e-6 < sc_min:
                    continue
                if features.scale_change - 1e-6 > sc_max:
                    continue

            result.append(pkg)
        return result

    def _apply_direction_filter(
        self,
        tracks: List[EvidencePackage],
        constraints: Mapping[str, Any],
        min_step: float = 0.005,
    ) -> List[EvidencePackage]:
        """按主运动方向过滤：left/right/up/down，基于轨迹分段投票的主方向."""
        direction = constraints.get("direction")
        if not direction:
            return tracks
        direction = str(direction).lower()
        allowed = {"left", "right", "up", "down"}
        if direction not in allowed:
            return tracks

        result: List[EvidencePackage] = []
        for pkg in tracks:
            feats = pkg.features
            if not feats or not feats.centroids or len(feats.centroids) < 2:
                continue
            votes = {"left": 0, "right": 0, "up": 0, "down": 0}
            pts = feats.centroids
            for i in range(1, len(pts)):
                dx = pts[i][0] - pts[i - 1][0]
                dy = pts[i][1] - pts[i - 1][1]
                if abs(dx) < min_step and abs(dy) < min_step:
                    continue
                if abs(dx) >= abs(dy):
                    votes["right" if dx > 0 else "left"] += 1
                else:
                    votes["down" if dy > 0 else "up"] += 1
            if not any(votes.values()):
                continue
            actual = max(votes.items(), key=lambda kv: kv[1])[0]
            if actual == direction:
                result.append(pkg)
        return result

    def _apply_sorting(
        self,
        tracks: List[EvidencePackage],
        constraints: Mapping[str, Any],
    ) -> List[EvidencePackage]:
        sort_by = constraints.get("sort_by")
        if not sort_by:
            return tracks
        order = constraints.get("sort_order", "desc").lower()
        reverse = order != "asc"

        def key(pkg: EvidencePackage) -> float:
            features = pkg.features
            if not features:
                return 0.0
            return float(
                getattr(
                    features,
                    sort_by,
                    0.0,
                )
            )

        return sorted(tracks, key=key, reverse=reverse)

    def _normalized_roi_bounds(
        self,
        roi_name: str,
        package: EvidencePackage,
    ) -> Optional[Tuple[float, float, float, float]]:
        roi = next((rect for name, rect in self.config.roi_zones if name == roi_name), None)
        if not roi:
            return None
        width, height = self._resolve_resolution(package)
        x1, y1, x2, y2 = roi
        width = max(width, 1)
        height = max(height, 1)
        return (
            min(max(x1 / width, 0.0), 1.0),
            min(max(y1 / height, 0.0), 1.0),
            min(max(x2 / width, 0.0), 1.0),
            min(max(y2 / height, 0.0), 1.0),
        )

    def _resolve_resolution(self, package: EvidencePackage) -> Tuple[int, int]:
        if package.meta and "resolution" in package.meta:
            return package.meta["resolution"]
        if self.metadata:
            return (self.metadata.width, self.metadata.height)
        return (1920, 1080)

    @staticmethod
    def _point_inside(point: Tuple[float, float], bounds: Tuple[float, float, float, float]) -> bool:
        x, y = point
        x1, y1, x2, y2 = bounds
        return x1 <= x <= x2 and y1 <= y <= y2

    def _estimate_dwell_seconds(
        self,
        package: EvidencePackage,
        inside_mask: List[bool],
    ) -> float:
        fps = package.meta.get("fps") if package.meta else package.fps
        fps = max(float(fps or 0.0), 1e-3)
        frames = package.frames
        if not frames or len(frames) < 2:
            return 0.0
        limit = min(len(frames), len(inside_mask))
        frames = frames[:limit]
        mask = inside_mask[:limit]
        dwell = 0.0
        for idx in range(1, limit):
            delta_frames = max(frames[idx] - frames[idx - 1], 1)
            delta_time = delta_frames / fps
            if mask[idx] or mask[idx - 1]:
                dwell += delta_time
        return dwell
