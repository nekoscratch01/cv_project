import sys
from pathlib import Path

# 保证无论从项目根目录还是 tests/ 目录运行，都能导入 src 下的模块
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.behavior import BehaviorFeatureExtractor, EventDetector  # noqa: E402
from core.config import SystemConfig  # noqa: E402
from core.perception import TrackRecord, VideoMetadata  # noqa: E402


def _simple_track(track_id: int, frames, bboxes) -> TrackRecord:
    return TrackRecord(track_id=track_id, frames=frames, bboxes=bboxes, crops=[])


def test_roi_dwell_counts_seconds():
    config = SystemConfig()
    config.roi_zones = [("door", (0, 0, 10, 10))]
    metadata = VideoMetadata(fps=10.0, width=640, height=480, total_frames=50)
    # 3 帧在 ROI 内，fps=10 → 0.3s
    track = _simple_track(1, frames=[1, 2, 3], bboxes=[(1, 1, 2, 2), (5, 5, 6, 6), (20, 20, 30, 30)])
    extractor = BehaviorFeatureExtractor(config, metadata)
    dwells = extractor.compute_roi_dwell({1: track})
    assert len(dwells[1]) == 2  # 2 帧在 ROI 内（第1和第2帧的中心在 0~10 范围内）
    total_seconds = sum(d.seconds for d in dwells[1])
    assert abs(total_seconds - 0.2) < 1e-6


def test_follow_event_detection():
    config = SystemConfig(follow_distance_thresh=5.0, follow_min_frames=3)
    metadata = VideoMetadata(fps=10.0, width=640, height=480, total_frames=50)
    # track A and B stay within distance <=5 for 3 common frames
    track_a = _simple_track(
        1,
        frames=[1, 2, 3, 4],
        bboxes=[(0, 0, 2, 2), (0, 0, 2, 2), (0, 0, 2, 2), (100, 100, 102, 102)],
    )
    track_b = _simple_track(
        2,
        frames=[1, 2, 3, 4],
        bboxes=[(1, 1, 3, 3), (1, 1, 3, 3), (1, 1, 3, 3), (200, 200, 202, 202)],
    )
    detector = EventDetector(config, metadata)
    events = detector.detect_follow_events({1: track_a, 2: track_b})
    assert len(events) == 1
    ev = events[0]
    assert ev.follower == 1 and ev.target == 2
    assert ev.start_s == 0.1 and ev.end_s == 0.3  # 3 帧，fps=10
