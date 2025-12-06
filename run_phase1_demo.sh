#!/bin/bash
# Phase 1 MVP demo runner (vLLM, Qwen3-VL-4B)
set -euo pipefail

export PYTHONPATH="$(pwd)/src"

python - <<'PY'
from pathlib import Path

from core.config import SystemConfig
from pipeline.video_semantic_search import VideoSemanticSystem

config = SystemConfig(
    video_path=Path("data/raw/semantic/MOT17-12.mp4"),
    output_dir=Path("output/demo_run"),
    vlm_backend="vllm",
    vllm_endpoint="http://localhost:8000/v1",
    vllm_model_name="Qwen/Qwen3-VL-4B-Instruct",
)

system = VideoSemanticSystem(config=config)

print("=== Building index on MOT17-12.mp4 ===")
system.build_index()
print(f"Evidence packages: {len(system.evidence_map)}")

question = "找穿蓝色衣服的人"
print(f"=== Running query: {question} ===")
results = system.question_search(question, top_k=3)
print("Matches:", len(results))
for r in results:
    print(f"track {r.track_id}: {r.start_s:.1f}s–{r.end_s:.1f}s | score={r.score:.2f} | {r.reason}")
print("Outputs in output/demo_run/: tracking_<question>.mp4 and tracking_all_tracks_<question>.mp4")
PY
