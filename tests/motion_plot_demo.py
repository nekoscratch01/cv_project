"""
Quick demo: build index, generate a motion minimap with colored dots for one track.

Run from repo root:
    PYTHONPATH=src python tests/motion_plot_demo.py
Outputs:
    - output/minimap_track_<id>.png : white canvas with gray path + 0.5s colored dots (green->red)
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.video_semantic_search import VideoSemanticSystem  # type: ignore  # noqa: E402
from pipeline.vlm_client_hf import Qwen3VL4BHFClient  # type: ignore  # noqa: E402


def main() -> None:
    system = VideoSemanticSystem()
    system.build_index()
    if not system.evidence_map:
        print("No evidence packages found. Check video_path or perception output.")
        return

    # Pick the first track to visualize
    pkg = next(iter(system.evidence_map.values()))

    client = Qwen3VL4BHFClient(system.config)
    minimap = client._render_minimap(pkg)  # type: ignore[attr-defined]
    if minimap is None:
        print("Minimap not available (cv2 missing or no centroids).")
        return

    out_path = Path(system.config.output_dir) / f"minimap_track_{pkg.track_id}.png"
    minimap.save(out_path)
    print(f"Saved motion minimap: {out_path}")


if __name__ == "__main__":
    main()
