"""Phase 1 entry point: question-driven person retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import (
    SystemConfig,
    VideoPerception,
    TrackFeatureExtractor,
    build_evidence_packages,
    RecallEngine,
    QwenVLMClient,
)


class VideoSemanticSystem:
    """High-level orchestrator exposing build + query APIs."""

    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or SystemConfig()
        self.perception = VideoPerception(self.config)
        self.track_records = None
        self.metadata = None
        self.features = None
        self.evidence_map = None
        self.recall_engine = RecallEngine()
        self.vlm_client = QwenVLMClient(self.config)

    def build_index(self) -> None:
        print("\n=== Stage 1: Perception ===")
        self.track_records, self.metadata = self.perception.process()
        print(f"   ✅ 有效 track 数: {len(self.track_records)}")

        print("\n=== Stage 2: Feature Extraction ===")
        feature_extractor = TrackFeatureExtractor(self.metadata)
        self.features = feature_extractor.extract(self.track_records)
        print("   ✅ 轨迹特征完成")

        print("\n=== Stage 3: 构建证据包 ===")
        video_id = Path(self.config.video_path).stem
        self.evidence_map = build_evidence_packages(
            video_id, self.track_records, self.metadata, self.features
        )
        print(f"   ✅ 构建 {len(self.evidence_map)} 个证据包")

        self._persist_database()

    def question_search(self, question: str, *, top_k: int = 5, recall_limit: int | None = None):
        if self.evidence_map is None:
            raise RuntimeError("请先运行 build_index()")

        print("\n=== 查询: 问题驱动检索 ===")
        print(f"描述: {question}")

        candidates = self.recall_engine.recall(question, self.evidence_map, recall_limit)
        print(f"   🔎 候选轨迹数: {len(candidates)}")

        vlm_results = self.vlm_client.answer(question, candidates)
        if not vlm_results:
            print("   ❌ 未找到匹配轨迹")
            return []

        vlm_results.sort(key=lambda r: r.score, reverse=True)
        selected = vlm_results[:top_k]

        print("   ✅ VLM 匹配结果:")
        for item in selected:
            print(
                f"      - Track {item.track_id}: {item.start_s:.1f}s → {item.end_s:.1f}s | 理由: {item.reason}"
            )

        track_ids = [item.track_id for item in selected]
        safe_name = question.replace(" ", "_")
        video_output = self.config.output_dir / f"tracking_{safe_name}.mp4"
        self.perception.render_highlight_video(
            self.track_records,
            self.metadata,
            track_ids,
            video_output,
            label_text=question,
        )

        return selected

    def _persist_database(self) -> None:
        db_path = self.config.output_dir / "semantic_database.json"
        feature_payload = (
            {str(tid): feature.to_dict() for tid, feature in self.features.items()}
            if self.features
            else {}
        )
        payload = {
            "video": str(self.config.video_path),
            "tracks": {
                str(tid): {
                    "frames": record.frames,
                    "bboxes": record.bboxes,
                    "crops": record.crops,
                }
                for tid, record in self.track_records.items()
            },
            "features": feature_payload,
        }
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"   💾 数据库存储: {db_path}")


def run_demo() -> None:
    system = VideoSemanticSystem()
    system.build_index()

    print("\n=== Demo Queries ===")
    system.question_search("找出穿紫色衣服的人", top_k=5)


if __name__ == "__main__":
    run_demo()
