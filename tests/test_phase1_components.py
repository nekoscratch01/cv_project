import math
import sys
from pathlib import Path

import numpy as np

# 将 src 目录加入 sys.path，解决 ModuleNotFoundError
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.config import SystemConfig
from core.evidence import EvidencePackage, build_evidence_packages
from core.features import TrackFeatureExtractor, TrackFeatures
from core.constraints import score_constraints
from core.perception import TrackRecord, VideoMetadata
from core.query_spec import QuerySpec, QuerySpecExtractor, QueryBudget
from pipeline.router import ExecutionPlan
from pipeline.recall import RecallEngine
from pipeline.video_semantic_search import VideoSemanticSystem
from core.vlm_types import QueryResult


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

    def answer(self, question: str, candidates, *, plan=None, top_k: int | None = None):
        results = []
        for package in candidates:
            if top_k is not None and len(results) >= top_k:
                break
            if package.track_id not in self.matches:
                continue
            reason = self.matches[package.track_id]
            if plan and plan.description:
                reason = f"{reason} ({plan.description})"
            results.append(
                QueryResult(
                    track_id=package.track_id,
                    start_s=package.start_time_seconds,
                    end_s=package.end_time_seconds,
                    score=self.score,
                    reason=reason,
                )
            )
        return results


class StubRouter:
    """Minimal router stub for tests (no external vLLM dependency)."""

    def build_plan(self, question: str) -> ExecutionPlan:
        query_spec = {
            "positive_prompts": [question or "a person"],
            "negative_prompts": [],
            "need_context": False,
            "constraint_intents": [],
        }
        return ExecutionPlan(
            description=question or "a person",
            meta={"need_context": False, "query_spec": query_spec},
        )


class DummySiglipClient:
    embedding_dim = 2

    def encode_text(self, texts):
        if not texts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def encode_images(self, images):
        return np.ones((len(images), 2), dtype=np.float32)


def test_query_spec_extractor_color_garment():
    extractor = QuerySpecExtractor(SystemConfig())
    spec = extractor.extract("Find the person in a blue shirt")
    assert spec.positive_prompts[0] == "person wearing a blue shirt"


def test_query_spec_extractor_negative_prompt():
    extractor = QuerySpecExtractor(SystemConfig())
    spec = extractor.extract("Find a person not wearing red")
    assert "person wearing red clothing" in spec.negative_prompts
    assert spec.positive_prompts[0] != "person wearing red clothing"


def test_score_tracks_topm_mean(tmp_path):
    config = SystemConfig(embedding_cache_dir=tmp_path / "embeddings")
    engine = RecallEngine(config=config, siglip_client=DummySiglipClient())
    pkg = EvidencePackage(
        "demo",
        1,
        [1],
        [(0, 0, 1, 1)],
        ["c1.jpg"],
        30.0,
        None,
        siglip_img_embeds=[[1.0, 0.0], [0.0, 1.0]],
    )
    scores = engine.score_tracks([pkg], positive_prompts=["anything"], frames_per_track=2, topm=2)
    assert math.isclose(scores[0][1], 0.5)


def test_siglip_soft_rerank_margin_clamps(tmp_path):
    config = SystemConfig(embedding_cache_dir=tmp_path / "embeddings")
    engine = RecallEngine(config=config, siglip_client=DummySiglipClient())
    pkgs = [
        EvidencePackage(
            "demo",
            1,
            [1],
            [(0, 0, 1, 1)],
            ["c1.jpg"],
            30.0,
            None,
            siglip_img_embeds=[[1.0, 0.0]],
        ),
        EvidencePackage(
            "demo",
            2,
            [1],
            [(0, 0, 1, 1)],
            ["c2.jpg"],
            30.0,
            None,
            siglip_img_embeds=[[0.5, 0.0]],
        ),
        EvidencePackage(
            "demo",
            3,
            [1],
            [(0, 0, 1, 1)],
            ["c3.jpg"],
            30.0,
            None,
            siglip_img_embeds=[[0.0, 1.0]],
        ),
    ]
    query_spec = QuerySpec(
        raw_query="anything",
        positive_prompts=["anything"],
        negative_prompts=[],
        budget=QueryBudget(k_min=2, k_max=3, margin_delta=0.1, frames_per_track=1, topm_frames=1),
    )
    candidates, ranked = engine.siglip_soft_rerank(pkgs, query_spec)
    assert len(ranked) == 3
    assert len(candidates) == 2


def test_build_evidence_packages_computes_timings():
    track_records = {7: _make_track_record(7)}
    metadata = VideoMetadata(fps=30.0, width=1920, height=1080, total_frames=100)
    features = {
        7: TrackFeatures(
            track_id=7,
            start_s=0.0,
            end_s=0.66,
            duration_s=0.66,
            centroids=[(0.1, 0.1)],
            displacement_vec=(0.0, 0.0),
            avg_speed_px_s=2.0,
            max_speed_px_s=3.2,
            path_length_px=12.0,
        )
    }
    packages = build_evidence_packages("demo", track_records, metadata, features)
    package = packages[7]
    assert math.isclose(package.start_time_seconds, package.start_frame / 30.0)
    assert math.isclose(package.end_time_seconds, package.end_frame / 30.0)
    assert package.motion.avg_speed_px_s == 2.0
    assert package.meta["resolution"] == (1920, 1080)
    assert package.raw_trace == track_records[7].bboxes


def test_recall_engine_limit(tmp_path):
    packages = {
        i: EvidencePackage("demo", i, [1], [(0, 0, 1, 1)], [f"c{i}.jpg"], 30.0, None) for i in range(5)
    }
    config = SystemConfig(embedding_cache_dir=tmp_path / "embeddings")
    engine = RecallEngine(config=config, siglip_client=DummySiglipClient())
    limited = engine.recall("anything", packages, limit=2)
    assert len(limited) == 2


def test_visual_filter_uses_embeddings(tmp_path):
    config = SystemConfig(embedding_cache_dir=tmp_path / "embeddings")
    engine = RecallEngine(config=config, siglip_client=DummySiglipClient())
    pkg1 = EvidencePackage("demo", 1, [1], [(0, 0, 1, 1)], ["c1.jpg"], 30.0, None, embedding=[1.0, 0.0])
    pkg2 = EvidencePackage("demo", 2, [1], [(0, 0, 1, 1)], ["c2.jpg"], 30.0, None, embedding=[0.0, 1.0])
    filtered = engine.visual_filter([pkg1, pkg2], "person", ["red"], top_k=1)
    assert len(filtered) == 1
    assert filtered[0].track_id == 1


def test_visual_filter_stub_respects_top_k(tmp_path):
    packages = [
        EvidencePackage("demo", i, [1], [(0, 0, 1, 1)], [f"c{i}.jpg"], 30.0, None)
        for i in range(6)
    ]
    config = SystemConfig(embedding_cache_dir=tmp_path / "embeddings")
    engine = RecallEngine(config=config, siglip_client=DummySiglipClient())
    filtered = engine.visual_filter(packages, "desc", ["tag"], top_k=3)
    assert len(filtered) == 3


def test_question_search_uses_stub_vlm(tmp_path):
    config = SystemConfig(
        video_path=tmp_path / "fake.mp4",
        output_dir=tmp_path / "outputs",
        embedding_cache_dir=tmp_path / "embeddings",
    )
    recall_engine = RecallEngine(config=config, siglip_client=DummySiglipClient())
    system = VideoSemanticSystem(
        config=config,
        recall_engine=recall_engine,
        vlm_client=StubVLMClient({1: "wearing red clothes"}),
        router=StubRouter(),
    )

    track_records = {1: _make_track_record(1), 2: _make_track_record(2)}
    metadata = VideoMetadata(fps=24.0, width=1280, height=720, total_frames=250)
    features = {
        tid: TrackFeatures(
            track_id=tid,
            start_s=0.0,
            end_s=1.5,
            duration_s=1.5,
            centroids=[(0.1 * tid, 0.1)],
            displacement_vec=(0.0, 0.0),
            avg_speed_px_s=1.0 + tid,
            max_speed_px_s=2.0 + tid,
            path_length_px=10.0 * tid,
        )
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


def test_track_feature_extractor_outputs_atomic_fields():
    metadata = VideoMetadata(fps=10.0, width=100, height=50, total_frames=200)
    track = TrackRecord(
        track_id=1,
        frames=[0, 10, 20],
        bboxes=[(0, 0, 10, 10), (20, 10, 30, 20), (40, 10, 50, 20)],
        crops=[],
    )
    extractor = TrackFeatureExtractor(metadata)
    features = extractor.extract({1: track})[1]

    assert math.isclose(features.start_s, 0.0)
    assert math.isclose(features.end_s, 2.0)
    assert len(features.centroids) == 3
    assert math.isclose(features.centroids[0][0], 0.05)
    assert math.isclose(features.centroids[0][1], 0.1)
    assert math.isclose(features.displacement_vec[0], 0.4)
    assert math.isclose(features.displacement_vec[1], 0.2)


def test_linearity_unit_consistency():
    metadata = VideoMetadata(fps=10.0, width=100, height=100, total_frames=100)
    straight = TrackRecord(
        track_id=1,
        frames=[0, 1, 2, 3],
        bboxes=[(0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10), (30, 0, 40, 10)],
        crops=[],
    )
    back_and_forth = TrackRecord(
        track_id=2,
        frames=[0, 1, 2, 3],
        bboxes=[(0, 0, 10, 10), (10, 0, 20, 10), (0, 0, 10, 10), (10, 0, 20, 10)],
        crops=[],
    )
    extractor = TrackFeatureExtractor(metadata)
    straight_feat = extractor.extract({1: straight})[1]
    back_feat = extractor.extract({2: back_and_forth})[2]

    assert straight_feat.linearity > 0.95
    assert back_feat.linearity < 0.6


def test_scale_change_uses_height_ratio():
    metadata = VideoMetadata(fps=10.0, width=100, height=100, total_frames=100)
    track = TrackRecord(
        track_id=1,
        frames=[0, 1, 2, 3, 4, 5],
        bboxes=[
            (0, 0, 10, 10),
            (0, 0, 10, 10),
            (0, 0, 10, 10),
            (0, 0, 10, 20),
            (0, 0, 10, 20),
            (0, 0, 10, 20),
        ],
        crops=[],
    )
    features = TrackFeatureExtractor(metadata).extract({1: track})[1]
    assert math.isclose(features.scale_change, 2.0, rel_tol=1e-2)


def test_constraint_score_intent_gating():
    feats = TrackFeatures(
        track_id=1,
        start_s=0.0,
        end_s=1.0,
        duration_s=1.0,
        centroids=[(0.1, 0.1)],
        displacement_vec=(0.2, 0.0),
        avg_speed_px_s=10.0,
        max_speed_px_s=10.0,
        path_length_px=10.0,
        norm_speed=2.0,
        linearity=0.9,
        scale_change=1.0,
    )
    score_empty, _ = score_constraints([], feats)
    score_run, breakdown = score_constraints(["RUNNING"], feats)
    assert score_empty == 0.0
    assert score_run > 0.5
    assert "RUNNING" in breakdown
