"""
Utility script: run YOLO + ByteTrack on the configured video and render all tracks.

Usage (from repo root):
    PYTHONPATH=src python tests/perception_render_demo.py

Notes:
- Forces CPU (`yolo_device="cpu"`), keeps thresholds modest to ensure detections show up.
- Outputs a debug video with all tracks highlighted at `output/debug_all_tracks.mp4`.
"""

from pathlib import Path

from core.config import SystemConfig
from core.perception import VideoPerception


def main() -> None:
    cfg = SystemConfig(
        yolo_device="cpu",
        yolo_conf=0.3,       # loosen for visibility
        min_track_length=1,  # keep short tracks for inspection
        sample_interval=2,
    )
    print(f"[perception] loading video: {cfg.video_path}")
    perception = VideoPerception(cfg)
    tracks, meta = perception.process()
    print(f"[perception] tracks: {len(tracks)}, frames: {meta.total_frames}")

    out_path = cfg.output_dir / "debug_all_tracks.mp4"
    track_ids = list(tracks.keys())
    perception.render_highlight_video(
        tracks,
        meta,
        track_ids,
        out_path,
        label_text="all tracks",
    )
    print(f"[perception] rendered video: {out_path} (track ids: {track_ids[:10]}{'...' if len(track_ids) > 10 else ''})")


if __name__ == "__main__":
    main()
