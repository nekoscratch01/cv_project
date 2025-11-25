"""Phase 1 entry point: question-driven person retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from core.config import SystemConfig
from core.perception import VideoPerception
from core.features import TrackFeatureExtractor
from core.evidence import build_evidence_packages
from pipeline.recall import RecallEngine
from core.hard_rules import HardRuleEngine
from typing import Any

VERSION = "v1.27"


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
        hard_rule_engine: HardRuleEngine | None = None,
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
        self.hard_rule_engine = hard_rule_engine

    def build_index(self) -> None:
        """
        建立视频索引：离线处理视频，生成所有人的证据包。
        
        这是系统的"索引阶段"，负责把原始视频处理成结构化数据。
        只需要运行一次，处理完后可以多次查询。
        
        工作流程（3个阶段）：
        
        Stage 1: Perception（感知）
            - 输入：视频文件
            - 处理：YOLO检测 + ByteTrack跟踪
            - 输出：track_records（每个人的帧号、框、裁剪图）
                   metadata（视频的fps、分辨率等）
            - 耗时：主要瓶颈（需要逐帧处理视频）
        
        Stage 2: Feature Extraction（特征提取）
            - 输入：track_records + metadata
            - 处理：计算运动特征（速度、路径长度、持续时间）
            - 输出：features（每个人的运动统计数据）
            - 耗时：很快（只是数值计算）
        
        Stage 3: Evidence Building（证据包构建）
            - 输入：track_records + metadata + features
            - 处理：把分散的数据打包成统一格式
            - 输出：evidence_map（每个人的完整档案）
            - 耗时：很快（只是数据重组）
        
        最后：持久化到磁盘
            - 保存 semantic_database.json（包含所有轨迹和特征）
            - 方便调试、恢复、分析
        
        Returns:
            None（结果保存在 self.track_records, self.features, self.evidence_map）
        
        Raises:
            可能的异常：
            - 视频文件不存在或损坏
            - YOLO模型加载失败
            - 磁盘空间不足（裁剪图和数据库文件）
        
        Note:
            - 这个方法会修改 self 的多个属性（track_records等）
            - 处理时间取决于视频长度和分辨率（例如5分钟视频约需5-10分钟）
            - 输出目录会自动创建（config.output_dir）
            - 可以多次调用，会覆盖之前的结果
        
        使用示例：
            system = VideoSemanticSystem()
            system.build_index()  # 处理视频，生成索引
            # 之后可以多次查询，不需要重新索引
        """
        print("\n=== Stage 1: Perception ===")
        # 感知层：检测和跟踪
        self.track_records, self.metadata = self.perception.process()
        print(f"   ✅ 有效 track 数: {len(self.track_records)}")

        print("\n=== Stage 2: Feature Extraction ===")
        # 特征层：计算运动特征
        feature_extractor = TrackFeatureExtractor(self.metadata)
        self.features = feature_extractor.extract(self.track_records)
        print("   ✅ 轨迹特征完成")

        print("\n=== Stage 3: 构建证据包 ===")
        # 证据层：打包所有信息
        video_id = Path(self.config.video_path).stem  # 提取文件名作为video_id
        self.evidence_map = build_evidence_packages(
            video_id, self.track_records, self.metadata, self.features
        )
        print(f"   ✅ 构建 {len(self.evidence_map)} 个证据包")

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
        
        Step 3: 排序与截断（Sort & Select）
            - 输入：所有VLM匹配结果
            - 处理：按分数排序，取前 top_k 个
            - 输出：最终匹配列表
            - 目的：只返回最相关的前几个结果
        
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
            top_k: 最多返回几个匹配结果，默认5个
                  即使VLM找到10个匹配，也只返回分数最高的前5个
            recall_limit: 召回阶段的候选数量限制（可选）
                         例如：recall_limit=20 表示最多给VLM看20个候选
                         如果为 None，Phase 1会返回所有轨迹
        
        Returns:
            匹配结果列表，格式：[QueryResult, QueryResult, ...]
            每个结果包含：track_id, start_s, end_s, score, reason
            列表按分数降序排列（分数最高的在前）
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
            raise RuntimeError("请先运行 build_index()")

        print(f"\n=== Version: {VERSION} ===")
        print("\n=== 查询: 问题驱动检索 ===")
        print(f"描述: {question}")

        plan = self.router.build_plan(question)
        print("   🧭 路由计划:", plan.to_dict())

        # Step 1: 召回阶段（筛选候选）
        all_tracks = list(self.evidence_map.values())
        # 放宽召回：默认看全量，避免目标被 limit 过滤掉
        recall_top_k = recall_limit or len(all_tracks)
        plan.constraints["limit"] = len(all_tracks)
        candidates = self.recall_engine.visual_filter(
            all_tracks,
            description=plan.description,
            visual_tags=plan.visual_tags,
            top_k=recall_top_k,
        )
        print(f"   🔎 候选轨迹数: {len(candidates)}")

        # Step 1.5: Hard Rule Engine
        hard_engine = self._ensure_hard_rule_engine()
        candidates = hard_engine.apply_constraints(candidates, plan)
        print(f"   📐 硬规则过滤后: {len(candidates)}")
        if not candidates:
            print("   ❌ 无满足硬规则的候选")
            return []

        # Step 2: VLM精排阶段（AI判断）
        vlm_results = self.vlm_client.answer(question, candidates, plan=plan)
        if not vlm_results:
            print("   ❌ 未找到匹配轨迹")
            return []

        # Step 3: 排序与截断
        vlm_results.sort(key=lambda r: r.score, reverse=True)  # 按分数降序
        selected = vlm_results[:top_k]  # 取前 top_k 个

        # 打印匹配结果
        print("   ✅ VLM 匹配结果:")
        for item in selected:
            print(
                f"      - Track {item.track_id}: {item.start_s:.1f}s → {item.end_s:.1f}s | 理由: {item.reason}"
            )

        # 汇总一句话回答：用同一 4B VLM 生成最终答复
        final_answer = ""
        if hasattr(self.vlm_client, "compose_final_answer"):
            try:
                final_answer = self.vlm_client.compose_final_answer(question, selected)  # type: ignore
            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️  汇总回答失败: {exc}")
        if not final_answer:
            if selected:
                summary_text = "，".join(
                    f"轨迹{item.track_id}（{item.start_s:.1f}s–{item.end_s:.1f}s）" for item in selected
                )
                final_answer = f"最可能匹配：{summary_text}。"
            else:
                final_answer = "未找到匹配轨迹。"
        print(f"\n📝 汇总回答：{final_answer}")

        # Step 4: 可视化（画红框视频）
        track_ids = [item.track_id for item in selected]
        safe_name = question.replace(" ", "_")  # 空格替换成下划线
        video_output = self.config.output_dir / f"tracking_{safe_name}.mp4"
        # 渲染最终匹配
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
        print(f"   🎞️ 结果视频: {video_output}")
        print(f"   🎞️ 全量轨迹: {debug_output}")

        return selected

    def _ensure_hard_rule_engine(self) -> HardRuleEngine:
        if self.hard_rule_engine is None:
            if self.metadata is None:
                raise RuntimeError("缺少 metadata，无法初始化 HardRuleEngine")
            self.hard_rule_engine = HardRuleEngine(self.config, self.metadata)
        elif self.hard_rule_engine.metadata is None and self.metadata is not None:
            self.hard_rule_engine.metadata = self.metadata
        return self.hard_rule_engine

    def _build_router(self):
        if self.config.router_backend in {"hf", "transformers", "llama_cpp"}:
            from pipeline.router_llm import HFRouter
            from pipeline.vlm_client_hf import Qwen3VL4BHFClient

            hf_client = self.vlm_client if isinstance(self.vlm_client, Qwen3VL4BHFClient) else None
            return HFRouter(self.config, hf_client=hf_client)
        raise RuntimeError(f"未知 router_backend: {self.config.router_backend!r}")

    def _build_vlm_client(self):
        if self.config.vlm_backend in {"hf", "transformers", "llama_cpp"}:
            from pipeline.vlm_client_hf import Qwen3VL4BHFClient

            return Qwen3VL4BHFClient(self.config)
        raise RuntimeError(
            "当前仅支持 vlm_backend in {'hf', 'transformers'}，旧的 GGUF / 2B 模型路径已移除。"
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
        print(f"   💾 数据库存储: {db_path}")


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
    system.question_search("找出往左边走的穿蓝色衣服的人，他是在跑还是走路？", top_k=5)


if __name__ == "__main__":
    run_demo()
