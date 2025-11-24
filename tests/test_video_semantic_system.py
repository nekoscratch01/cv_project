import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.config import SystemConfig  # noqa: E402
from core.evidence import build_evidence_packages  # noqa: E402
from core.features import TrackFeatures  # noqa: E402
from core.perception import TrackRecord, VideoMetadata  # noqa: E402
from core.vlm_types import QueryResult  # noqa: E402
from pipeline.router import ExecutionPlan  # noqa: E402
from pipeline.video_semantic_search import VideoSemanticSystem  # noqa: E402


class StubRecallEngine:
    def visual_filter(self, tracks, description, visual_tags, top_k=None):
        if top_k is None or top_k <= 0:
            return list(tracks)
        return list(tracks)[:top_k]


class StubVLMClient:
    def __init__(self, match_id: int):
        self.match_id = match_id

    def answer(self, question, candidates, *, plan=None, top_k=None):
        results = []
        for pkg in candidates:
            if pkg.track_id == self.match_id:
                results.append(
                    QueryResult(
                        track_id=pkg.track_id,
                        start_s=pkg.start_time_seconds,
                        end_s=pkg.end_time_seconds,
                        score=1.0,
                        reason="stub match",
                    )
                )
        return results

    def compose_final_answer(self, question, results):
        return f"found {len(results)} match"


class StubRouter:
    def build_plan(self, question):
        return ExecutionPlan(description=question, visual_tags=["red"], needed_facts=[], constraints={})


def test_question_search_with_stubs(tmp_path):
    config = SystemConfig(
        video_path=tmp_path / "fake.mp4",
        output_dir=tmp_path / "outputs",
        embedding_cache_dir=tmp_path / "embeddings",
    )
    system = VideoSemanticSystem(
        config=config,
        recall_engine=StubRecallEngine(),
        vlm_client=StubVLMClient(match_id=1),
        router=StubRouter(),
    )

    # 准备最小的索引数据
    track_records = {
        1: TrackRecord(track_id=1, frames=[0, 10], bboxes=[(0, 0, 10, 10), (5, 5, 15, 15)], crops=[]),
        2: TrackRecord(track_id=2, frames=[0, 10], bboxes=[(20, 20, 30, 30), (25, 25, 35, 35)], crops=[]),
    }
    metadata = VideoMetadata(fps=10.0, width=100, height=100, total_frames=20)
    features = {
        tid: TrackFeatures(
            track_id=tid,
            start_s=0.0,
            end_s=1.0,
            duration_s=1.0,
            centroids=[(0.1 * tid, 0.1)],
            displacement_vec=(0.0, 0.0),
            avg_speed_px_s=1.0,
            max_speed_px_s=2.0,
            path_length_px=1.0,
        )
        for tid in track_records
    }
    system.track_records = track_records
    system.metadata = metadata
    system.features = features
    system.evidence_map = build_evidence_packages("demo", track_records, metadata, features)
    system.perception.render_highlight_video = lambda *args, **kwargs: None

    results = system.question_search("找红衣服的人", top_k=2)
    assert len(results) == 1
    assert results[0].track_id == 1
