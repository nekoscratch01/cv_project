import math
import sys
from pathlib import Path

# 保证 src 在 import 路径里
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.evidence import EvidencePackage, build_evidence_packages  # noqa: E402
from core.features import TrackFeatureExtractor, TrackFeatures  # noqa: E402
from core.perception import TrackRecord, VideoMetadata  # noqa: E402


def test_track_feature_extractor_computes_motion_and_normalization():
    metadata = VideoMetadata(fps=10.0, width=100, height=50, total_frames=200)
    track = TrackRecord(
        track_id=1,
        frames=[0, 10, 20],
        bboxes=[(0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10)],
        crops=[],
    )
    extractor = TrackFeatureExtractor(metadata)
    feats = extractor.extract({1: track})[1]

    assert math.isclose(feats.start_s, 0.0)
    assert math.isclose(feats.end_s, 2.0)
    assert math.isclose(feats.duration_s, 2.0)
    assert len(feats.centroids) == 3
    assert math.isclose(feats.centroids[0][0], 0.05)  # 5 / 100
    assert math.isclose(feats.centroids[0][1], 0.1)   # 5 / 50
    assert math.isclose(feats.displacement_vec[0], 0.2)  # (25-5)/100
    assert math.isclose(feats.displacement_vec[1], 0.0)
    assert math.isclose(feats.avg_speed_px_s, 10.0)
    assert math.isclose(feats.max_speed_px_s, 10.0)
    assert math.isclose(feats.path_length_px, 20.0)


def test_build_evidence_packages_includes_meta_and_raw_trace():
    track_records = {
        1: TrackRecord(
            track_id=1,
            frames=[0, 1],
            bboxes=[(0, 0, 10, 10), (1, 1, 11, 11)],
            crops=["c1.jpg"],
        )
    }
    metadata = VideoMetadata(fps=30.0, width=640, height=480, total_frames=100)
    features = {
        1: TrackFeatures(
            track_id=1,
            start_s=0.0,
            end_s=0.03,
            duration_s=0.03,
            centroids=[(0.01, 0.01)],
            displacement_vec=(0.0, 0.0),
            avg_speed_px_s=1.0,
            max_speed_px_s=1.0,
            path_length_px=1.0,
        )
    }
    packages = build_evidence_packages("demo_video", track_records, metadata, features)
    pkg = packages[1]

    assert pkg.meta["resolution"] == (640, 480)
    assert pkg.meta["video_id"] == "demo_video"
    assert pkg.raw_trace == track_records[1].bboxes
    assert pkg.raw_trace is not track_records[1].bboxes  # 确保是拷贝
    assert math.isclose(pkg.start_time_seconds, pkg.start_frame / metadata.fps)
    assert pkg.motion.avg_speed_px_s == 1.0
