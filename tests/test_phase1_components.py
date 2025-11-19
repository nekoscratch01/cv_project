import math
import sys
from pathlib import Path

# 将 src 目录加入 sys.path，解决 ModuleNotFoundError
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import SystemConfig
from evidence import EvidencePackage, build_evidence_packages
from features import TrackFeatures
from perception import TrackRecord, VideoMetadata
from recall import RecallEngine
from video_semantic_search import VideoSemanticSystem
from vlm_client import QueryResult


def _make_track_record(track_id: int) -> TrackRecord:
    frames = [1, 11, 21]
    bboxes = [(0, 0, 10, 10), (5, 5, 15, 15), (10, 10, 20, 20)]
    crops = [f"crop_{track_id}_1.jpg", f"crop_{track_id}_2.jpg"]
    return TrackRecord(track_id=track_id, frames=frames, bboxes=bboxes, crops=crops)


class StubVLMClient:
    """Simple VLM stub that marks pre-defined track ids as matches."""

    def __init__(self, matches: dict[int, str], score: float = 0.9) -> None:
        self.matches = matches
        self.score = score

    def answer(self, question: str, candidates, top_k: int | None = None):
        results = []
        for package in candidates:
            if top_k is not None and len(results) >= top_k:
                break
            if package.track_id not in self.matches:
                continue
            results.append(
                QueryResult(
                    track_id=package.track_id,
                    start_s=package.start_time_seconds,
                    end_s=package.end_time_seconds,
                    score=self.score,
                    reason=self.matches[package.track_id],
                )
            )
        return results


def test_build_evidence_packages_computes_timings():
    track_records = {7: _make_track_record(7)}
    metadata = VideoMetadata(fps=30.0, width=1920, height=1080, total_frames=100)
    features = {
        7: TrackFeatures(track_id=7, avg_speed_px_s=2.0, max_speed_px_s=3.2, path_length_px=12.0, duration_s=0.66)
    }
    packages = build_evidence_packages("demo", track_records, metadata, features)
    package = packages[7]
    assert math.isclose(package.start_time_seconds, package.start_frame / 30.0)
    assert math.isclose(package.end_time_seconds, package.end_frame / 30.0)
    assert package.motion.avg_speed_px_s == 2.0


def test_recall_engine_limit():
    packages = {
        i: EvidencePackage("demo", i, [1], [(0, 0, 1, 1)], [f"c{i}.jpg"], 30.0, None) for i in range(5)
    }
    engine = RecallEngine()
    limited = engine.recall("anything", packages, limit=2)
    assert len(limited) == 2


def test_question_search_uses_stub_vlm(tmp_path):
    config = SystemConfig(
        video_path=tmp_path / "fake.mp4",
        output_dir=tmp_path / "outputs",
    )
    system = VideoSemanticSystem(config=config, vlm_client=StubVLMClient({1: "wearing red clothes"}))

    track_records = {1: _make_track_record(1), 2: _make_track_record(2)}
    metadata = VideoMetadata(fps=24.0, width=1280, height=720, total_frames=250)
    features = {
        tid: TrackFeatures(track_id=tid, avg_speed_px_s=1.0 + tid, max_speed_px_s=2.0 + tid,
                           path_length_px=10.0 * tid, duration_s=1.5)
        for tid in track_records
    }
    system.track_records = track_records
    system.metadata = metadata
    system.features = features
    system.evidence_map = build_evidence_packages("demo", track_records, metadata, features)
    system.perception.render_highlight_video = lambda *args, **kwargs: None

    results = system.question_search("找红色衣服的人", top_k=5)
    assert len(results) == 1
    assert results[0].track_id == 1
    assert "red" in results[0].reason
