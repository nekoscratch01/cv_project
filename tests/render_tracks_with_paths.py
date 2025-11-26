"""
Render a debug video with bounding boxes AND trajectory paths (colored lines/dots) for all tracks.

Usage (from repo root):
    python tests/render_tracks_with_paths.py

Output:
    output/tracks_with_paths.mp4  (fall back to mp4v if avc1 is unavailable)

Notes:
    - Uses VideoSemanticSystem.build_index() to reuse tracked results.
    - Draws per-track boxes and the full path up to the current frame.
    - Colors are deterministic per track_id.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.video_semantic_search import VideoSemanticSystem  # noqa: E402
from core.perception import TrackRecord  # noqa: E402


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color for a track_id."""
    rng = np.random.default_rng(seed=track_id)
    r, g, b = (int(x) for x in rng.integers(80, 255, size=3))
    return (b, g, r)


def main() -> None:
    system = VideoSemanticSystem()
    system.build_index()
    if not system.track_records or not system.metadata:
        print("No track records found. Ensure perception produced tracks.")
        return

    video_path = Path(system.config.video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = system.metadata.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(system.metadata.width or cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(system.metadata.height or cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_path = Path(system.config.output_dir) / "tracks_with_paths.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = Path(system.config.output_dir) / "tracks_with_paths_mp4v.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Failed to create VideoWriter.")
        return

    # Precompute centers per track
    track_points: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for tid, rec in system.track_records.items():
        pts = []
        for f, bbox in zip(rec.frames, rec.bboxes):
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            pts.append((f, (cx, cy)))
        track_points[tid] = pts

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        for tid, rec in system.track_records.items():
            # draw bbox if present for this frame
            if frame_idx in rec.frames:
                i = rec.frames.index(frame_idx)
                x1, y1, x2, y2 = rec.bboxes[i]
                color = _color_for_track(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # draw path up to current frame
        for tid, pts in track_points.items():
            color = _color_for_track(tid)
            path = [p for f, p in pts if f <= frame_idx]
            if len(path) >= 2:
                cv2.polylines(frame, [np.array(path)], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
                # draw last point
                cv2.circle(frame, path[-1], 4, color, -1)

        writer.write(frame)
        if total_frames and frame_idx % 100 == 0:
            print(f"Rendered {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"Saved debug video with paths: {out_path}")


if __name__ == "__main__":
    main()
