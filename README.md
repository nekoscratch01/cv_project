# Edge-Detective — Phase 2.6 (Dual-Stream VLM for Video)

We give non-video multimodal models a full “find people in video” ability:
- **Two-layer verifier:**  
  - **Layer 1** (crop only) → fast appearance check.  
  - **Layer 2** (context + filmstrip) → appearance + motion/direction in one stitched timeline image (left=past → right=future) with red boxes.
- **Works with small VLMs (Qwen3-VL-4B)** by tight prompt, filmstrip trick, and robust parsing.
- **Hard Rules removed**, CLIP/SigLIP kept as a switch (default OFF) to avoid accidental filtering.

Detailed architecture diagrams live in `docs/`. This README focuses on what we solve and how to run.

---

## What we solved
- **Non-video VLMs can now search video:** filmstrip turns time into space; red-boxed frames + close-up crops let the VLM judge who/where/what action without native video support.
- **Small models behave better:** strict JSON prompt, per-ID robust parsing, and higher token budget prevent “good matches lost” due to truncated/garbled outputs.
- **Safe defaults:** Hard Rules removed; CLIP optional and off by default; router sanitizes away motion constraints if the query doesn’t mention motion.

---

## Quick start

### Install
```bash
pip install -r requirements.txt
```

### Start vLLM (Qwen3-VL-4B)
```bash
./deploy/start_vllm.sh
```

### Run the demo (choose one)
- **Shell**  
  ```bash
  export PYTHONPATH=src
  python src/pipeline/video_semantic_search.py
  ```
- **Notebook**  
  Open `phase1_demo_onecell.ipynb` and run the single cell (it sets `PYTHONPATH`, builds index, runs a query).

Outputs: `output/demo_run/semantic_database.json`, `tracking_<question>.mp4`, `tracking_all_tracks_<question>.mp4`.

---

## Key config (src/core/config.py)
- `video_path`: absolute path to your test video (default: MOT17 sample).
- `enable_clip_filter=False`: keep CLIP off unless you trust video paths/crops.
- `filmstrip_enabled=True`, `filmstrip_frame_count=5`, `filmstrip_max_width=4096`.
- `vlm_max_new_tokens=1024`: enough budget for batch JSON replies.

---

## Code map (where to look)
- `src/pipeline/video_semantic_search.py` — orchestrator (index + query), CLIP switch, no Hard Rules.  
- `src/pipeline/router_vlm.py` — vLLM router with few-shot prompt & constraint sanitization.  
- `src/adapters/inference/vllm_adapter.py` — Layer1/Layer2, filmstrip stitching, robust per-ID parsing.  
- `src/core/config.py` — runtime knobs.  
- `src/core/evidence.py` — evidence packaging, best-crop extraction.

For deep architecture details, see `docs/` (e.g., edge_detective_blueprint_v7.md).
