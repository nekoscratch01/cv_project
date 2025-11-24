import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.config import SystemConfig  # noqa: E402
from core.evidence import EvidencePackage  # noqa: E402
from core.features import TrackFeatures  # noqa: E402
from core.hard_rules import HardRuleEngine  # noqa: E402
from core.perception import VideoMetadata  # noqa: E402


def _make_package(
    track_id: int,
    *,
    frames: list[int],
    centroids: list[tuple[float, float]],
    fps: float = 10.0,
    resolution: tuple[int, int] = (100, 100),
    avg_speed: float = 1.0,
    max_speed: float = 2.0,
) -> EvidencePackage:
    features = TrackFeatures(
        track_id=track_id,
        start_s=frames[0] / fps,
        end_s=frames[-1] / fps,
        duration_s=(frames[-1] - frames[0]) / fps,
        centroids=centroids,
        displacement_vec=(centroids[-1][0] - centroids[0][0], centroids[-1][1] - centroids[0][1]),
        avg_speed_px_s=avg_speed,
        max_speed_px_s=max_speed,
        path_length_px=0.0,
    )
    bbox = [(0, 0, 10, 10)] * len(frames)
    return EvidencePackage(
        video_id="demo",
        track_id=track_id,
        frames=frames,
        bboxes=bbox,
        crops=[],
        fps=fps,
        features=features,
        meta={"video_id": "demo", "fps": fps, "resolution": resolution},
        raw_trace=bbox,
    )


def test_roi_enter_and_stay():
    config = SystemConfig()
    config.roi_zones = [("door", (0, 0, 50, 50))]
    metadata = VideoMetadata(fps=10.0, width=100, height=100, total_frames=0)
    engine = HardRuleEngine(config, metadata)

    pkg_enter = _make_package(1, frames=[0, 10], centroids=[(0.8, 0.8), (0.4, 0.4)])
    pkg_stay_long = _make_package(
        2,
        frames=[0, 10, 20],
        centroids=[(0.2, 0.2), (0.25, 0.25), (0.2, 0.2)],
    )
    pkg_stay_short = _make_package(
        3,
        frames=[0, 10, 20],
        centroids=[(0.7, 0.7), (0.7, 0.7), (0.2, 0.2)],
    )

    filtered_enter = engine.apply_constraints([pkg_enter], {"roi": "door", "event_type": "enter"})
    assert [p.track_id for p in filtered_enter] == [1]

    filtered_stay = engine.apply_constraints(
        [pkg_stay_long, pkg_stay_short],
        {"roi": "door", "event_type": "stay", "min_dwell_s": 1.5},
    )
    assert [p.track_id for p in filtered_stay] == [2]


def test_time_window_sort_and_limit():
    config = SystemConfig()
    metadata = VideoMetadata(fps=10.0, width=100, height=100, total_frames=0)
    engine = HardRuleEngine(config, metadata)
    pkg1 = _make_package(1, frames=[0, 10], centroids=[(0.1, 0.1), (0.1, 0.1)])
    pkg2 = _make_package(2, frames=[15, 25], centroids=[(0.2, 0.2), (0.2, 0.2)])
    pkg3 = _make_package(3, frames=[40, 50], centroids=[(0.3, 0.3), (0.3, 0.3)])

    filtered = engine.apply_constraints(
        [pkg1, pkg2, pkg3],
        {"time_window": [1.1, 3.5], "sort_by": "end_s", "sort_order": "asc", "limit": 1},
    )
    assert [p.track_id for p in filtered] == [2]


def test_speed_jump_threshold():
    config = SystemConfig()
    metadata = VideoMetadata(fps=10.0, width=100, height=100, total_frames=0)
    engine = HardRuleEngine(config, metadata)
    fast = _make_package(1, frames=[0, 10], centroids=[(0.1, 0.1), (0.9, 0.9)], avg_speed=1.0, max_speed=4.0)
    slow = _make_package(2, frames=[0, 10], centroids=[(0.2, 0.2), (0.3, 0.3)], avg_speed=1.0, max_speed=1.5)
    filtered = engine.apply_constraints([fast, slow], {"min_speed_jump": 2.0})
    assert [p.track_id for p in filtered] == [1]
