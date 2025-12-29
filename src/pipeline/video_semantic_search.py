"""Phase 1 entry point: question-driven person retrieval (v2.1).

Pipeline:
- build_index: perception -> features -> evidence map.
- question_search: router -> recall -> VLM verifier (MATCH line).
- Outputs both result video (matched tracks) and all-tracks debug video.
"""

from __future__ import annotations

import json
import asyncio
import inspect
from pathlib import Path
import cv2

from core.config import SystemConfig
from core.perception import VideoPerception
from core.features import TrackFeatureExtractor
from core.evidence import build_evidence_packages
from pipeline.recall import RecallEngine
from pipeline.clip_filter import ClipFilter
from core.vlm_types import QueryResult
from typing import Any

VERSION = "v2.1"


class VideoSemanticSystem:
    """
    视频语义检索系统：问题驱动的人物检索主入口。
    
    这是整个系统的"总指挥官"，负责协调所有模块，对外提供两个核心API：
    1. build_index()：建立视频索引（离线阶段，只需运行一次）
    2. question_search()：问题驱动检索（在线阶段，可以多次查询）
    
    系统架构（两阶段）：
    
    【阶段一：建立索引】build_index()
        视频文件
          ↓
        1. Perception（感知层）：YOLO + ByteTrack → 每个人的轨迹记录
          ↓
        2. Features（特征层）：计算运动特征 → 速度、时长等统计数据
          ↓
        3. Evidence（证据层）：打包所有信息 → 每个人的完整档案
          ↓
        存储到 semantic_database.json
    
    【阶段二：问题检索】question_search(question)
        用户问题："找穿紫色衣服的人"
          ↓
        1. Recall（召回层）：快速筛选 → 选出候选人（Phase 1返回所有人）
          ↓
        2. VLM（精排层）：AI判断 → 哪些人匹配？为什么？
          ↓
        3. Visualization（可视化）：画红框 → 导出高亮视频
          ↓
        返回匹配结果 + 时间区间 + 理由 + 视频
    
    设计原则：
        - 索引阶段和查询阶段完全分离（索引一次，查询多次）
        - 所有中间结果都保存在内存和磁盘（可调试、可恢复）
        - 模块之间通过 EvidencePackage 交换数据（统一接口）
        - 支持依赖注入（recall_engine 和 vlm_client 可以替换）
    
    使用示例：
        # 创建系统
        system = VideoSemanticSystem()
        
        # 建立索引（只需运行一次）
        system.build_index()
        
        # 多次查询
        system.question_search("找穿紫色衣服的人")
        system.question_search("找戴帽子的人")
        system.question_search("找背包的人")
    """

    def __init__(
        self,
        config: SystemConfig | None = None,
        recall_engine: RecallEngine | None = None,
        vlm_client: object | None = None,
        router: Any | None = None,
    ) -> None:
        """
        初始化视频语义检索系统。
        
        Args:
            config: 系统配置对象。如果为 None，使用默认配置
                   包含视频路径、模型名称、输出目录等所有配置项
            recall_engine: 召回引擎。如果为 None，使用默认的 RecallEngine
                          支持依赖注入，方便测试和替换召回策略
            vlm_client: VLM客户端。如果为 None，使用默认的 GGUF VLM 客户端
                       支持依赖注入，方便测试和使用不同的VLM实现
        
        Note:
            - 初始化时只创建对象，不加载视频或模型（延迟加载）
            - 中间结果（track_records, features等）初始化为 None
            - 调用 build_index() 后，这些中间结果才会被填充
        """
        self.config = config or SystemConfig()
        self.perception = VideoPerception(self.config)
        # 中间结果，初始化为 None，调用 build_index() 后填充
        self.track_records = None  # 感知层输出：轨迹记录
        self.metadata = None       # 感知层输出：视频元数据
        self.features = None       # 特征层输出：运动特征
        self.evidence_map = None   # 证据层输出：证据包字典
        # 召回引擎和VLM客户端（支持依赖注入）
        self.recall_engine = recall_engine or RecallEngine(config=self.config)
        self.vlm_client = vlm_client or self._build_vlm_client()
        self.router = router or self._build_router()
        self.clip_filter = None  # 延迟加载

    def build_index(self) -> None:
        """
        Build the offline index for a video (run once, query many times).

        Stages:
        1) Perception: YOLO + ByteTrack → track_records + metadata.
        2) Features: motion stats (speed/linearity/scale) per track.
        3) Evidence: package tracks + features into evidence_map.

        Persists:
        - semantic_database.json under config.output_dir

        Returns:
            None. Results are stored on self.track_records, self.features, self.evidence_map.
        """
        print("\n=== Stage 1: Perception ===")
        # 感知层：检测和跟踪
        self.track_records, self.metadata = self.perception.process()
        print(f"   ✅ Valid tracks: {len(self.track_records)}")

        print("\n=== Stage 2: Feature Extraction ===")
        # 特征层：计算运动特征
        feature_extractor = TrackFeatureExtractor(self.metadata)
        self.features = feature_extractor.extract(self.track_records)
        print("   ✅ Track features computed")

        print("\n=== Stage 3: Build evidence packages ===")
        # 证据层：打包所有信息
        video_id = Path(self.config.video_path).stem  # 提取文件名作为video_id
        self.evidence_map = build_evidence_packages(
            video_id,
            self.track_records,
            self.metadata,
            self.features,
            video_path=str(self.config.video_path),
        )
        print(f"   ✅ Built {len(self.evidence_map)} evidence packages")

        # 持久化：保存到磁盘
        self._persist_database()

    def question_search(self, question: str, *, top_k: int = 5, recall_limit: int | None = None):
        """
        问题驱动检索：用自然语言查询，找出匹配的人。
        
        这是系统的"查询阶段"，负责根据用户问题找出匹配的轨迹。
        可以多次调用，不需要重新建立索引。
        
        工作流程（4个步骤）：
        
        Step 1: 召回（Recall）
            - 输入：问题 + 所有证据包
            - 处理：快速筛选候选人（Phase 1返回所有人）
            - 输出：候选证据包列表
            - 目的：减少VLM的工作量（未来版本会做真正的过滤）
        
        Step 2: VLM精排（VLM Ranking）
            - 输入：问题 + 候选证据包
            - 处理：VLM逐个判断是否匹配
            - 输出：匹配的轨迹 + 分数 + 理由
            - 目的：准确判断哪些人符合描述
        
        Step 4: 可视化（Visualization）
            - 输入：匹配的track_id列表
            - 处理：在原视频上画红框
            - 输出：高亮视频文件（tracking_xxx.mp4）
            - 目的：让用户直观看到结果
        
        Args:
            question: 用户的查询问题（自然语言）
                     例如："找出穿紫色衣服的人"
                          "找戴牛仔帽的人"
                          "找背圆形背包的人"
            recall_limit: 召回阶段的候选数量限制（可选）
                         例如：recall_limit=20 表示最多给VLM看20个候选
                         如果为 None，Phase 1会返回所有轨迹
        
        Returns:
            匹配结果列表，格式：[QueryResult, QueryResult, ...]（全部匹配，不截断）
            每个结果包含：track_id, start_s, end_s, score, reason
            如果没找到匹配，返回空列表 []
        
        Raises:
            RuntimeError: 如果还没调用 build_index()
        
        Side Effects:
            - 在 config.output_dir 下生成高亮视频：tracking_<question>.mp4
            - 打印查询过程和结果到控制台
        
        Note:
            - 必须先调用 build_index() 建立索引
            - question 中的空格会被替换成下划线（用于视频文件名）
            - 如果没有匹配结果，不会生成视频文件
            - VLM推理是主要耗时（每个候选约1-3秒）
        
        使用示例：
            # 先建立索引
            system = VideoSemanticSystem()
            system.build_index()
            
            # 查询1：找穿紫色衣服的人
            results = system.question_search("找穿紫色衣服的人", top_k=5)
            for r in results:
                print(f"Track {r.track_id}: {r.reason}")
            
            # 查询2：找戴帽子的人（不需要重新索引）
            results = system.question_search("找戴帽子的人", top_k=3)
        """
        # 检查是否已经建立索引
        if self.evidence_map is None:
            raise RuntimeError("Please run build_index() first")

        print(f"\n=== Version: {VERSION} ===")
        print("\n=== Query: Question-driven retrieval ===")
        print(f"Query: {question}")

        plan = self.router.build_plan(question)
        if inspect.isawaitable(plan):
            plan = self._run_coroutine(plan)
        print("   🧭 Routing plan:", plan.to_dict())

        # Step 1: 召回阶段（筛选候选）
        all_tracks = list(self.evidence_map.values())
        recall_top_k = recall_limit or len(all_tracks)
        plan.constraints["limit"] = len(all_tracks)
        candidates = self.recall_engine.visual_filter(
            all_tracks,
            description=plan.description,
            visual_tags=plan.visual_tags,
            top_k=recall_top_k,
        )
        print(f"   🔎 Candidate tracks: {len(candidates)}")

        # Step 1.1: CLIP/SigLIP 预过滤（外观快速筛，默认关闭，由配置控制）
        if getattr(self.config, "enable_clip_filter", False):
            if self.clip_filter is None:
                try:
                    self.clip_filter = ClipFilter(model_name=self.config.siglip_model_name, device=self.config.siglip_device)
                except Exception as exc:  # noqa: BLE001
                    print(f"   ⚠️ CLIP filter init failed: {exc}")
                    self.clip_filter = None
            if self.clip_filter is not None:
                before = len(candidates)
                threshold = getattr(self.config, "clip_filter_threshold", 0.05)
                candidates = self.clip_filter.filter_candidates(plan.description or plan.visual_tags, candidates, threshold=threshold)
                print(f"   🧊 After CLIP filter: {len(candidates)} (filtered {before - len(candidates)}, thr={threshold})")
        else:
            print("   🧊 CLIP filter disabled (enable_clip_filter=False)")

        # Step 1.5: Hard Rule Engine（已移除，保留 CLIP + VLM 双层过滤）
        if not candidates:
            print("   ❌ No candidates after CLIP")
            return []

        # Step 2: VLM精排阶段（AI判断）
        vlm_results = self._run_vlm_verification(question, candidates, plan, top_k=None)
        if not vlm_results:
            print("   ❌ No matching tracks")
            safe_name = question.replace(" ", "_")
            video_output = self.config.output_dir / f"tracking_{safe_name}.mp4"
            debug_output = self.config.output_dir / f"tracking_all_tracks_{safe_name}.mp4"

            # 候选高亮（如果有候选则画框，没有则跳过）
            candidate_ids = [c.track_id for c in candidates]
            if candidate_ids:
                self.perception.render_highlight_video(
                    self.track_records,
                    self.metadata,
                    candidate_ids,
                    video_output,
                    label_text=f"candidates: {question}",
                )
                print(f"   🎞️ Candidate video: {video_output}")
            else:
                self._write_raw_video(video_output)
                print(f"   🎞️ Candidate video (raw, no candidates): {video_output}")

            # 全轨迹调试：总是画出所有轨迹，便于比对
            all_track_ids = list(self.track_records.keys())
            self.perception.render_highlight_video(
                self.track_records,
                self.metadata,
                all_track_ids,
                debug_output,
                label_text="all tracks",
            )
            print(f"   🎞️ All-tracks video: {debug_output}")
            return []

        # Step 3: 保留全部匹配（不截断），仅用于展示排序
        vlm_results.sort(key=lambda r: r.score, reverse=True)
        matches = vlm_results
        print("   ✅ VLM matches (all is_match):")
        for item in matches:
            print(
                f"      - Track {item.track_id}: {item.start_s:.1f}s → {item.end_s:.1f}s | score={item.score:.2f} | reason: {item.reason}"
            )

        # 汇总一句话回答
        if matches:
            summary_parts = [
                f"track {m.track_id} ({m.start_s:.1f}s–{m.end_s:.1f}s): {m.reason}"
                for m in matches
            ]
            final_answer = f"Found {len(matches)} matches. " + " | ".join(summary_parts)
        else:
            final_answer = "No matching tracks found."
        print(f"\n📝 Final answer: {final_answer}")

        # Step 4: 可视化（仅高亮匹配轨迹）
        track_ids = [item.track_id for item in matches]
        safe_name = question.replace(" ", "_")  # 空格替换成下划线
        video_output = self.config.output_dir / f"tracking_{safe_name}.mp4"
        self.perception.render_highlight_video(
            self.track_records,
            self.metadata,
            track_ids,
            video_output,
            label_text=question,
        )

        # 额外输出：全量轨迹调试视频，便于比对
        all_track_ids = list(self.track_records.keys())
        debug_output = self.config.output_dir / f"tracking_all_tracks_{safe_name}.mp4"
        self.perception.render_highlight_video(
            self.track_records,
            self.metadata,
            all_track_ids,
            debug_output,
            label_text="all tracks",
        )
        print(f"   🎞️ Result video: {video_output}")
        print(f"   🎞️ All-tracks video: {debug_output}")

        return matches

    def _run_vlm_verification(self, question: str, candidates, plan, top_k: int | None):
        """
        在 vLLM 适配器（InferencePort）与旧版 HF 客户端之间做桥接。
        """
        if hasattr(self.vlm_client, "verify_batch"):
            plan_context = self._build_plan_context(plan)
            results: list[QueryResult] = []
            batch_size = max(1, min(3, getattr(self.config, "vlm_batch_size", 3)))
            for i in range(0, len(candidates), batch_size):
                chunk = candidates[i : i + batch_size]
                verifications = self._run_coroutine(
                    self.vlm_client.verify_batch(
                        packages=chunk,
                        question=question,
                        plan_context=plan_context,
                        concurrency=batch_size,
                    )
                )
                for package, verdict in zip(chunk, verifications):
                    if not verdict.is_match:
                        continue
                    results.append(
                        QueryResult(
                            track_id=package.track_id,
                            start_s=package.start_time_seconds,
                            end_s=package.end_time_seconds,
                            score=verdict.confidence,
                            reason=verdict.reason,
                        )
                    )
            return results

        # 兼容旧 HF 客户端接口
        return self.vlm_client.answer(question, candidates, plan=plan, top_k=top_k)  # type: ignore[no-any-return]

    @staticmethod
    def _build_plan_context(plan) -> str:
        try:
            return json.dumps(plan.to_dict(), ensure_ascii=False)
        except Exception:
            return ""

    def _run_coroutine(self, coro):
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                raise
            if loop.is_running():
                raise RuntimeError(
                    "vLLM verification requires a non-async context; please call the async adapter directly."
                ) from exc
            return loop.run_until_complete(coro)

    def _write_raw_video(self, output_path: Path) -> None:
        """把原视频直接拷贝为 MP4（无任何标注），用于空结果时的占位输出。"""
        cap = cv2.VideoCapture(str(self.config.video_path))
        if not cap.isOpened():
            print(f"   ⚠️ Cannot open video for copy: {self.config.video_path}")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.metadata.fps if self.metadata else cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"   ⚠️ Cannot create raw video file: {output_path}")
            cap.release()
            return
        frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames += 1
        cap.release()
        out.release()
        if frames == 0:
            print(f"   ⚠️ Raw video copy has 0 frames: {output_path}")

    def _build_router(self):
        if self.config.router_backend == "simple":
            from pipeline.router import SimpleRouter
            return SimpleRouter()
        if self.config.router_backend == "vllm":
            from pipeline.router_vlm import VlmRouter

            return VlmRouter(base_url=self.config.vllm_endpoint, model=self.config.vllm_model_name)
        raise RuntimeError(f"Unknown router_backend: {self.config.router_backend!r}")

    def _build_vlm_client(self):
        if self.config.vlm_backend != "vllm":
            raise RuntimeError("vlm_backend must be 'vllm' (no downgrade fallback).")

        from adapters.inference.vllm_adapter import VllmAdapter, VllmConfig

        return VllmAdapter(
            VllmConfig(
                endpoint=self.config.vllm_endpoint,
                model_name=self.config.vllm_model_name,
                temperature=self.config.vlm_temperature,
                max_tokens=self.config.vlm_max_new_tokens,
                max_images_per_request=getattr(self.config, "vlm_batch_size", 5),
            )
        )

    def _persist_database(self) -> None:
        """
        持久化数据库：保存所有轨迹和特征到JSON文件。
        
        这是一个内部方法（私有方法），由 build_index() 调用。
        把内存中的所有数据保存到磁盘，方便：
        1. 调试：查看中间结果，定位问题
        2. 恢复：程序崩溃后可以从文件恢复
        3. 分析：用其他工具分析轨迹数据
        
        保存的内容：
            - video: 视频文件路径
            - tracks: 所有轨迹的原始数据（帧号、框、裁剪图路径）
            - features: 所有轨迹的运动特征（速度、时长等）
        
        文件格式：
            JSON格式，UTF-8编码，带缩进（方便人类阅读）
            文件名：semantic_database.json
            位置：config.output_dir / "semantic_database.json"
        
        Note:
            - 这是私有方法，不应该被外部直接调用
            - 如果文件已存在，会被覆盖
            - 保存的是文件路径（crops），不是图片本身
            - track_id 会被转成字符串（JSON的key必须是字符串）
        
        文件结构示例：
            {
              "video": "/path/to/video.mp4",
              "tracks": {
                "1": {
                  "frames": [1, 2, 3, ...],
                  "bboxes": [[50,100,150,300], ...],
                  "crops": ["crops/id001_frame00001.jpg", ...]
                },
                "2": {...}
              },
              "features": {
                "1": {
                  "avg_speed_px_s": 75.0,
                  "max_speed_px_s": 120.0,
                  "path_length_px": 636.0,
                  "duration_s": 29.97
                },
                "2": {...}
              }
            }
        """
        db_path = self.config.output_dir / "semantic_database.json"
        
        # 转换 features 为字典格式（track_id必须是字符串）
        feature_payload = (
            {str(tid): feature.to_dict() for tid, feature in self.features.items()}
            if self.features
            else {}
        )
        
        # 构造完整的数据结构
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
        
        # 写入JSON文件
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,              # 缩进2个空格（美观）
                ensure_ascii=False     # 允许中文等非ASCII字符
            )
        print(f"   💾 Database saved: {db_path}")


def run_demo() -> None:
    """
    运行演示程序：展示系统的完整工作流程。
    
    这是一个演示函数，展示如何使用 VideoSemanticSystem：
    1. 创建系统实例（使用默认配置）
    2. 建立索引（处理视频）
    3. 执行查询（问题驱动检索）
    
    演示查询：
        "找出穿紫色衣服的人"
    
    输出：
        - 控制台打印：处理进度、匹配结果
        - 文件输出：
            * crops/：裁剪图文件夹
            * semantic_database.json：数据库文件
            * tracking_找出穿紫色衣服的人.mp4：高亮视频
    
    Note:
        - 使用默认配置（config.py 中的配置）
        - 如果要修改视频路径或其他配置，需要修改 config.py
        - 这个函数主要用于快速测试和演示
    
    使用方法：
        python video_semantic_search.py
    """
    # 创建系统实例
    system = VideoSemanticSystem()
    
    # 建立索引（处理视频）
    system.build_index()

    # 执行演示查询
    print("\n=== Demo Queries ===")
    system.question_search("Find the person in blue moving left — are they running or walking?", top_k=5)


if __name__ == "__main__":
    run_demo()
