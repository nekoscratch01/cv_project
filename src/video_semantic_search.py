"""Entry point for the refactored perception → semantic → retrieval pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import (
    SystemConfig,
    VideoPerception,
    TrackFeatureExtractor,
    SemanticDescriptor,
    SemanticRetrievalEngine,
)


class VideoSemanticSystem:
    """High-level orchestrator exposing build + query APIs."""

    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or SystemConfig()
        self.perception = VideoPerception(self.config)
        self.track_records = None
        self.metadata = None
        self.features = None
        self.profiles = None
        self.retrieval = None

    def build_index(self) -> None:
        print("\n=== Stage 1: Perception ===")
        self.track_records, self.metadata = self.perception.process()
        print(f"   ✅ 有效 track 数: {len(self.track_records)}")

        print("\n=== Stage 2: Feature Extraction ===")
        feature_extractor = TrackFeatureExtractor(self.metadata)
        self.features = feature_extractor.extract(self.track_records)
        print("   ✅ 轨迹特征完成")

        print("\n=== Stage 3: Semantic Annotation ===")
        descriptor = SemanticDescriptor(self.config)
        self.profiles = descriptor.describe_tracks(self.track_records, self.features)
        print(f"   ✅ 生成语义 profile 数: {len(self.profiles)}")

        print("\n=== Stage 4: Retrieval Setup ===")
        self.retrieval = SemanticRetrievalEngine(
            self.config, self.track_records, self.profiles
        )
        print("   ✅ 检索引擎就绪")

        self._persist_database()

    def query(self, query_name: str, *, structured=None, text: str | None = None) -> list[int]:
        if self.retrieval is None:
            raise RuntimeError("请先 build_index() 再查询")

        if structured:
            track_ids = self.retrieval.search_structured(structured)
        elif text:
            track_ids = self.retrieval.search_text(text)
        else:
            raise ValueError("必须提供 structured 或 text 查询条件")

        if not track_ids:
            print(f"   ❌ 查询 {query_name} 没有结果")
            return []

        image_output = self.config.output_dir / f"result_{query_name}.jpg"
        self.retrieval.visualize(track_ids, image_output)

        video_output = self.config.output_dir / f"tracking_{query_name}.mp4"
        self.perception.render_highlight_video(
            self.track_records,
            self.metadata,
            track_ids,
            video_output,
            label_text=query_name,
        )

        return track_ids

    def _persist_database(self) -> None:
        db_path = self.config.output_dir / "semantic_database.json"
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
            "profiles": {
                str(tid): profile.to_dict() for tid, profile in self.profiles.items()
            },
        }
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"   💾 数据库存储: {db_path}")


def run_demo() -> None:
    system = VideoSemanticSystem()
    system.build_index()

    print("\n=== Demo Queries ===")
    # 穿紫色衣服的人（基于 VLM 颜色识别）
    system.query("穿紫色衣服的人", structured=[("color", "purple")])


if __name__ == "__main__":
    run_demo()
