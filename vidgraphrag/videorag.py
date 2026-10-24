import os
import sys
import json
import shutil
import asyncio
import multiprocessing
import torch
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer
import tiktoken


from ._llm import (
    LLMConfig,
    openai_config,
    azure_openai_config,
    ollama_config
)
from ._op import (
    chunking_by_video_segments,
    extract_entities,
    get_chunks,
    videorag_query,
    videorag_query_multiple_choice,
    videorag_query_fusion, # 融合查询入口
    generate_community_reports,
)
from ._storage import (
    JsonKVStorage,
    ShotKVStorage,
    SegmentKVStorage,
    CommunityReportStorage,
    ShotBM25Storage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    MultiModalVectorDB,
    NetworkXStorage,
)
from ._rerank import dashscope_rerank, RERANK_AVAILABLE
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    encode_string_by_tiktoken,
    limit_async_func_call,
    wrap_embedding_func_with_attrs,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._videoutil import(
    split_video,
    speech_to_text,
    segment_caption,
    merge_segment_information,
    saving_video_segments,
    detect_shots_with_scene_change,
    merge_shots_to_segments,
    shot_vlm_analysis,
    segment_narrative_understanding,
    encode_shot_features,
    encode_string_query,
)
from ._wiki import WikiConfig, WikiKnowledgeBase, WikiScale
from ._wiki_ingest import WikiIngestWriter
from ._wiki_query import WikiQueryRetriever, WikiQueryResult
from ._wiki_lint import WikiLinter


@dataclass
# 自动为类生成 __init__(), __repr__(), __eq__() 等方法，减少样板代码。
class VideoRAG:
    """
        VideoRAG 类是一个多模态检索增强生成系统，用于处理、索引和查询视频内容。
        它结合了视频分割、语音识别、视觉描述生成、知识图谱构建和向量检索功能。
    """
    working_dir: str = field(
        # 工作目录路径，用于存储处理后的视频数据
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # Fusion mode configuration
    enable_shot_level: bool = True  # 是否启用 Shot 级处理
    """是否启用 Shot 级细粒度处理（双层架构）"""
    shot_aggregation_strategy: str = 'temporal'  # 聚合策略
    """Shot 聚合为 Segment 的策略：'temporal'（按时长）或 'semantic'（按语义）"""
    
    # video
    threads_for_split: int = 10
    """视频分割时使用的线程数"""
    video_segment_length: int = 30 # seconds
    """每个视频片段的长度（秒）"""
    rough_num_frames_per_segment: int = 5 # frames
    """粗略分割时每片段的帧数"""
    fine_num_frames_per_segment: int = 15 # frames
    """精细分割时每片段的帧数"""
    video_output_format: str = "mp4"
    """视频输出格式"""
    audio_output_format: str = "mp3"
    """音频输出格式"""
    video_embedding_batch_num: int = 2
    """视频嵌入的批量处理数量"""
    segment_retrieval_top_k: int = 4
    """片段检索时返回的 top-k 数量"""
    video_embedding_dim: int = 1024
    """视频嵌入的维度"""
    
    # query
    retrieval_topk_chunks: int = 2
    """查询时检索的文本块 top-k 数量"""
    query_better_than_threshold: float = 0.2
    """查询相似度阈值"""

    # graph mode
    enable_local: bool = True
    """是否启用本地实体提取模式"""
    enable_naive_rag: bool = True
    """是否启用基础 RAG 模式"""

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_video_segments
    """文本分块函数"""

    chunk_token_size: int = 1200
    """每个文本块的最大 token 数"""
    # chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"
    """用于 token 计数的模型名称"""

    # entity extraction
    entity_extract_max_gleaning: int = 2
    """实体提取的最大尝试次数"""
    entity_summary_to_max_tokens: int = 500
    """实体摘要的最大 token 数"""

    # Change to your LLM provider
    llm: LLMConfig = field(default_factory=openai_config)
    """大语言模型配置"""
    
    # entity extraction
    entity_extraction_func: callable = extract_entities
    """实体提取函数"""

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    """键值对存储类"""
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    """向量数据库存储类"""
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    """视频片段特征向量存储类"""
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    """向量数据库存储类的额外参数"""
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    """图存储类"""
    enable_llm_cache: bool = True
    """是否启用 LLM 响应缓存"""

    # Wiki (Karpathy Memex)
    wiki_config: WikiConfig = field(default_factory=WikiConfig)
    """Wiki 知识库配置（LLM Memex 模式）"""

    # extension
    always_create_working_dir: bool = True
    """是否总是创建工作目录"""
    addon_params: dict = field(default_factory=dict)
    """附加参数"""
    convert_response_to_json_func: callable = convert_response_to_json
    """将响应转换为 JSON 的函数"""

    def load_caption_model(self, debug=False):
        """
            加载视频描述生成模型（MiniCPM-V）。

            参数:
                debug (bool): 是否为调试模式。True 时不加载模型，False 时加载完整模型。

            返回:
                None
        """
        # caption model
        if not debug:
            self.caption_model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
            self.caption_tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
            self.caption_model.eval()
        else:
            self.caption_model = None
            self.caption_tokenizer = None
    
    def __post_init__(self):
        """
            初始化后的配置处理。
            创建所有必要的存储实例、嵌入函数和 LLM 函数包装器。

            主要初始化内容:
                - 各种存储实例（视频路径、片段、文本块、LLM 缓存、知识图谱）
                - 嵌入函数及其限流包装
                - 实体向量数据库和文本块向量数据库
                - 视频片段特征向量数据库
                - LLM 模型的限流包装
                - [新增] Shot 级存储和 Segment 级存储（双层架构）
        """
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"VideoRAG init with param:\n\n  {_print_config}\n")

        # 工作路径是否存在，不存在创建工作路径
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # 初始化视频路径数据库，用于存储视频名称到文件路径的映射
        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )

        # 初始化视频片段存储，保存每个视频的片段详细信息
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

        # 初始化文本块存储，用于 RAG 检索的文本分块数据
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        # 根据配置决定是否启用 LLM 响应缓存，避免重复调用
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        # 初始化知识图谱存储，用于保存实体和关系
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        # 包装嵌入函数，添加维度限制和异步调用限制
        # 这是高阶函数的链式调用，也叫函数装饰器模式。
        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(wrap_embedding_func_with_attrs(
                embedding_dim = self.llm.embedding_dim,
                max_token_size = self.llm.embedding_max_token_size,
                model_name = self.llm.embedding_model_name)(self.llm.embedding_func))

        # 根据配置初始化实体向量数据库，支持本地实体检索
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )

        # 根据配置初始化文本块向量数据库，支持基础 RAG 检索
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        # 初始化视频片段特征向量数据库，嵌入过程在 insert() 中实现
        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=asdict(self),
                embedding_func=None, # we code the embedding process inside the insert() function.
            )
        )
        
        # [新增] 如果启用 Shot 级处理，初始化双层存储
        if self.enable_shot_level:
            logger.info("Initializing dual-layer storage (Shot + Segment)")
            
            # Shot 级 KV 存储
            self.shot_storage = ShotKVStorage(
                namespace="shots",
                global_config=asdict(self)
            )
            
            # Shot 级多模态向量数据库（不需要 embedding_func，因为我们直接传入向量）
            self.shot_vdb = MultiModalVectorDB(
                namespace="shot_vectors",
                global_config=asdict(self),
                embedding_func=None  # 我们手动编码向量
            )
            
            # Segment 级 KV 存储（增强版）
            self.segment_storage = SegmentKVStorage(
                namespace="segments_enhanced",
                global_config=asdict(self)
            )
            
            # 社区报告存储
            self.community_reports = CommunityReportStorage(
                namespace="community_reports",
                global_config=asdict(self)
            )

            # BM25 稀疏检索索引（基于 ASR 文本）
            self.shot_bm25 = ShotBM25Storage(
                namespace="shot_bm25",
                global_config=asdict(self)
            )

            # Cross-encoder Reranker（DashScope gte-rerank）
            self.reranker = dashscope_rerank if RERANK_AVAILABLE else None

            logger.info("Dual-layer storage initialized successfully")
        else:
            logger.info("Shot-level processing disabled, using legacy mode")
            self.shot_storage = None
            self.shot_vdb = None
            self.segment_storage = None
            self.community_reports = None
            self.shot_bm25 = None
            self.reranker = None

        # 为 LLM 的最佳模型和廉价模型函数添加异步调用限制和缓存功能
        self.llm.best_model_func = limit_async_func_call(self.llm.best_model_max_async)(
            # partial 是 Python 的偏函数工具，来自 functools 模块。
            # 固定函数的部分参数，生成一个新函数，减少后续调用的参数数量。
            partial(self.llm.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.llm.cheap_model_func = limit_async_func_call(self.llm.cheap_model_max_async)(
            partial(self.llm.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

        # [Wiki] 初始化 Wiki 知识库
        if self.wiki_config.enabled:
            logger.info("Initializing Wiki Knowledge Base (Karpathy Memex mode)")
            self.wiki = WikiKnowledgeBase(self.working_dir, self.wiki_config)
            self.wiki_ingest_writer = WikiIngestWriter(self.wiki, self.llm.best_model_func)
            self.wiki_query_retriever = WikiQueryRetriever(self.wiki, self.llm.best_model_func)
            self.wiki_linter = WikiLinter(self.wiki, self.llm.cheap_model_func)
            if self.wiki_config.git_versioning:
                self.wiki.git_init()
            logger.info(f"Wiki initialized: scale={self.wiki.current_scale().value}, "
                        f"pages={self.wiki.page_count()}")
        else:
            self.wiki = None
            self.wiki_ingest_writer = None
            self.wiki_query_retriever = None
            self.wiki_linter = None

    def insert_video(self, video_path_list=None):
        """
            插入并处理视频列表，将其转换为可检索的知识库。

            处理流程（8 个步骤）:
                Step 0: 检查视频是否已存在于存储中，存在则跳过
                Step 1: 将视频分割为固定长度的片段，并提取关键帧
                Step 2: 使用 Whisper 对每个片段进行语音识别，生成文本稿
                Step 3: 并行处理：保存视频片段文件 + 使用 VLM 生成视觉描述
                Step 4: 合并所有片段信息（时间戳、文本、描述）并存入存储
                Step 5: 编码视频片段特征到向量数据库
                Step 6: 清理临时缓存文件
                Step 7: 保存当前视频的所有索引信息
                Final: 执行文本分块和实体提取，构建知识图谱

            参数:
                video_path_list (List[str], optional): 视频文件路径列表。默认为 None。

            返回:
                None

            异常:
                RuntimeError: 当并行处理过程中出现错误时抛出，并记录到日志文件
            """
        # 根据配置选择处理方式
        if self.enable_shot_level:
            logger.info("Using fusion mode (Shot-Segment dual-layer)")
            return self.insert_video_fusion(video_path_list)
        else:
            logger.info("Using legacy mode")
            return self._insert_video_legacy(video_path_list)
    
    def insert_video_fusion(self, video_path_list=None):
        """
        融合模式下的视频插入流程（双层架构）
        
        流程：
        1. 分镜检测 → Shots
        2. 并行处理每个 Shot:
           - VLM 结构化分析
           - 三路特征编码
           - 写入 Shot 存储
        3. 聚合 Shots → Segments
        4. Segment 级叙事理解
        5. 构建知识图谱
        """
        loop = always_get_an_event_loop()
        
        for video_path in video_path_list:
            video_name = os.path.basename(video_path).split('.')[0]
            
            # Step 0: 检查视频是否存在
            if video_name in self.video_segments._data:
                logger.info(f"Find the video named {os.path.basename(video_path)} in storage and skip it.")
                continue
            
            logger.info(f"Processing video: {video_name} (Fusion Mode)")
            
            # Step 1: 分镜检测
            logger.info("Step 1: Detecting shots...")
            try:
                shots = detect_shots_with_scene_change(video_path)
                logger.info(f"Detected {len(shots)} shots")
            except Exception as e:
                logger.error(f"Shot detection failed: {e}, falling back to legacy mode")
                # 降级到传统分段方式
                return self._insert_video_legacy([video_path])
            
            if not shots:
                logger.warning("No shots detected, skipping video")
                continue
            
            # Step 2: 获取 API Key（用于 VLM 和 Embedding）
            # 优先级：_shot_api.py 注入的 ali_dashscope_api_key > 环境变量 > LLMConfig.api_key
            api_key = (
                getattr(self, 'ali_dashscope_api_key', None)
                or os.getenv("DASHSCOPE_API_KEY")
                or (self.llm.api_key if hasattr(self.llm, 'api_key') else None)
            )
            if not api_key:
                logger.warning("API key not found, some features may be limited")
            
            # Step 2.5: 全视频 ASR（一次调用，带时间戳）
            logger.info("Step 2.5: Full-video ASR...")
            full_asr_segments = loop.run_until_complete(
                self._full_video_asr(video_path, api_key)
            )
            logger.info(f"Full-video ASR completed: {len(full_asr_segments)} segments with timestamps")

            # Step 3: 并行处理所有 Shots
            logger.info(f"Step 3: Processing {len(shots)} shots...")
            processed_shots = loop.run_until_complete(
                self._process_all_shots(
                    video_path=video_path,
                    video_name=video_name,
                    shots=shots,
                    api_key=api_key,
                    full_asr_segments=full_asr_segments
                )
            )
            
            # Step 4: 写入 Shot 存储
            logger.info("Step 3: Saving shots to storage...")
            loop.run_until_complete(self.shot_storage.upsert(processed_shots))
            
            # 准备向量数据
            shot_vectors = {
                shot_id: data['vectors']
                for shot_id, data in processed_shots.items()
            }
            loop.run_until_complete(self.shot_vdb.upsert(shot_vectors))
            # 持久化 ShotKVStorage 和 MultiModalVectorDB 到磁盘
            # （_insert_done 的遍历列表未包含这两个存储，必须在此显式落盘，
            # 否则进程退出后 query worker 加载到空 namespace，导致 shot 检索召回为空）
            loop.run_until_complete(self.shot_storage.index_done_callback())
            loop.run_until_complete(self.shot_vdb.index_done_callback())

            # 构建 BM25 索引（基于 ASR 文本）
            if self.shot_bm25 is not None:
                loop.run_until_complete(self.shot_bm25.build_from_shots(processed_shots))
                loop.run_until_complete(self.shot_bm25.index_done_callback())

            # Step 5: 聚合为 Segments（语义感知）
            logger.info("Step 4: Aggregating shots to segments...")
            shot_embeddings = {
                shot_id: data['vectors']['visual_vec']
                for shot_id, data in processed_shots.items()
                if data.get('vectors', {}).get('visual_vec') is not None
            }
            segments = merge_shots_to_segments(
                shots,
                max_duration=self.video_segment_length,
                min_duration=10,
                shot_embeddings=shot_embeddings if shot_embeddings else None,
                similarity_threshold=0.65,
            )
            segment_data = loop.run_until_complete(
                self._build_segment_data(
                    segments=segments,
                    processed_shots=processed_shots,
                    api_key=api_key
                )
            )
            
            # Step 6: 写入 Segment 存储
            logger.info("Step 5: Saving segments...")
            loop.run_until_complete(self.segment_storage.upsert(segment_data))
            # 持久化 SegmentKVStorage 到磁盘 (kv_store_segments_enhanced.json)
            loop.run_until_complete(self.segment_storage.index_done_callback())
            
            # Step 7: 保存视频元信息（向后兼容）
            loop.run_until_complete(self.video_path_db.upsert({video_name: video_path}))
            loop.run_until_complete(self.video_segments.upsert({video_name: segments}))
            loop.run_until_complete(self._save_video_segments())
            
            # Step 8: 构建知识图谱
            logger.info("Step 6: Building knowledge graph...")
            loop.run_until_complete(self.ainsert_from_segments(segment_data))
            
            # Step 9: 清理临时文件
            self._cleanup_cache(video_name)

            # Step 10: Wiki Ingest — 从底层产出物蒸馏 Wiki 页
            if self.wiki_ingest_writer is not None:
                logger.info("Step 10: Wiki ingest (Karpathy Memex)...")
                try:
                    graph_data = loop.run_until_complete(self._extract_graph_data_for_wiki())
                    loop.run_until_complete(
                        self.wiki_ingest_writer.ingest_video(
                            video_name=video_name,
                            segment_data=segment_data,
                            graph_data=graph_data,
                        )
                    )
                    logger.info(f"Wiki ingest completed: {self.wiki.page_count()} total pages")
                except Exception as e:
                    logger.warning(f"Wiki ingest failed (non-fatal): {e}")

            logger.info(f"Video {video_name} processing completed (Fusion Mode)")
    
    async def _full_video_asr(
        self,
        video_path: str,
        api_key: Optional[str] = None
    ) -> list:
        """
        对完整视频做一次 ASR，返回带时间戳的句子列表。

        返回格式: [{'text': str, 'begin_time': float, 'end_time': float}, ...]
        单位为秒。如果 ASR 失败则返回空列表。
        """
        import tempfile
        from moviepy.editor import VideoFileClip

        try:
            with VideoFileClip(video_path) as video:
                audio_clip = video.audio
                if audio_clip is None:
                    logger.info("Video has no audio track, skipping ASR")
                    return []

                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio_tmp:
                    audio_path = audio_tmp.name

                audio_clip.write_audiofile(audio_path, codec='mp3', verbose=False, logger=None)

            asr_model_name = getattr(self, 'asr_model', None)

            if asr_model_name and api_key and asr_model_name.startswith('fun-asr'):
                import dashscope
                from dashscope.audio.asr import Recognition

                dashscope.api_key = api_key
                recognition = Recognition(
                    model=asr_model_name,
                    format='mp3',
                    sample_rate=16000,
                    language_hints=['zh', 'en', 'ja'],
                    callback=None
                )
                result = await asyncio.get_event_loop().run_in_executor(
                    None, recognition.call, audio_path
                )

                segments = []
                if result and result.status_code == 200 and result.output:
                    for sentence in result.output.get('sentence', []):
                        segments.append({
                            'text': sentence.get('text', ''),
                            'begin_time': sentence.get('begin_time', 0) / 1000.0,
                            'end_time': sentence.get('end_time', 0) / 1000.0,
                        })
                else:
                    logger.warning(f"DashScope full-video ASR failed: {getattr(result, 'message', 'unknown')}")
            else:
                try:
                    from faster_whisper import WhisperModel
                    import logging as _logging

                    if not hasattr(self, '_whisper_model'):
                        self._whisper_model = WhisperModel("Systran/faster-distil-whisper-large-v3")
                        self._whisper_model.logger.setLevel(_logging.WARNING)

                    whisper_segments, _ = self._whisper_model.transcribe(audio_path)
                    segments = [
                        {'text': seg.text, 'begin_time': seg.start, 'end_time': seg.end}
                        for seg in whisper_segments
                    ]
                except Exception as e:
                    logger.warning(f"Whisper full-video ASR failed: {e}")
                    segments = []

            if os.path.exists(audio_path):
                os.unlink(audio_path)

            return segments

        except Exception as e:
            logger.warning(f"Full-video ASR failed: {e}")
            return []

    @staticmethod
    def _slice_asr_by_time(
        full_asr_segments: list,
        time_start: float,
        time_end: float
    ) -> str:
        """从全视频 ASR 结果中按时间范围切出对应文本"""
        parts = []
        for seg in full_asr_segments:
            seg_start = seg.get('begin_time', 0)
            seg_end = seg.get('end_time', 0)
            if seg_end <= time_start:
                continue
            if seg_start >= time_end:
                break
            parts.append(seg.get('text', ''))
        return ' '.join(parts).strip()

    async def _process_all_shots(
        self,
        video_path: str,
        video_name: str,
        shots: List[Dict],
        api_key: Optional[str] = None,
        full_asr_segments: Optional[list] = None
    ) -> Dict[str, Dict]:
        """
        并行处理所有 Shots
        
        优化：
        - 批量切割视频片段
        - 批量 VLM 推理
        - 批量特征编码（在 Shot 处理后统一编码）
        """
        import tempfile
        from moviepy.editor import VideoFileClip
        import numpy as np
        
        processed_shots = {}
        shot_video_paths = {}  # 保存临时文件路径，用于后续清理
        
        # 分批处理（每批 10 个 Shots）
        batch_size = 10
        total_batches = (len(shots) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(shots))
            batch_shots = shots[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_shots)} shots)")
            
            # 并行处理批次内的 Shots（只切割视频和 VLM 分析）
            tasks = [
                self._process_single_shot(video_path, video_name, shot, api_key, full_asr_segments)
                for shot in batch_shots
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for shot, result in zip(batch_shots, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process shot {shot['shot_id']}: {result}")
                    continue
                processed_shots[shot['shot_id']] = result
                # 保存临时文件路径
                shot_video_paths[shot['shot_id']] = result.get('_temp_video_path', '')
            
            logger.info(f"Batch {batch_idx + 1} completed: {len(batch_results)} shots processed")
        
        # 批量编码三路特征
        if processed_shots and api_key:
            logger.info("Encoding shot features (visual + text embeddings)...")
            try:
                # 准备批量编码的输入
                valid_shot_ids = []
                shot_paths = []
                vlm_captions = []
                asr_texts = []
                
                for shot_id, shot_data in processed_shots.items():
                    temp_path = shot_video_paths.get(shot_id, '')
                    if temp_path and os.path.exists(temp_path):
                        valid_shot_ids.append(shot_id)
                        shot_paths.append(temp_path)
                        vlm_captions.append(shot_data['content']['vlm_shot_summary'].get('caption', ''))
                        asr_texts.append(shot_data['content'].get('asr_text', ''))
                
                if shot_paths:
                    # 加载 ImageBind 模型（单例模式，避免重复加载）
                    if not hasattr(self, '_imagebind_model'):
                        logger.info("Loading ImageBind model...")
                        from ._videoutil import load_imagebind_model
                        self._imagebind_model = load_imagebind_model()
                        logger.info("ImageBind model loaded")
                    
                    # 批量编码
                    encoded_features = encode_shot_features(
                        shot_paths=shot_paths,
                        embedder=self._imagebind_model,
                        vlm_captions=vlm_captions,
                        asr_texts=asr_texts,
                        api_key=api_key
                    )
                    
                    # 更新 processed_shots 中的向量
                    for shot_id, features in encoded_features.items():
                        if shot_id in processed_shots:
                            processed_shots[shot_id]['vectors'] = features
                    
                    logger.info(f"Encoded {len(encoded_features)} shots with 3-way features")
            except Exception as e:
                logger.error(f"Feature encoding failed: {e}, using zero vectors as fallback")
                import traceback
                traceback.print_exc()
                # 为所有 Shot 填充零向量
                for shot_id in processed_shots:
                    if processed_shots[shot_id].get('vectors') is None:
                        processed_shots[shot_id]['vectors'] = {
                            'visual_vec': np.zeros(1024),
                            'audio_vec': np.zeros(1024),
                            'vlm_vec': np.zeros(1536)
                        }
        else:
            # 没有 API Key 或没有 Shots，填充零向量
            for shot_id in processed_shots:
                if processed_shots[shot_id].get('vectors') is None:
                    processed_shots[shot_id]['vectors'] = {
                        'visual_vec': np.zeros(1024),
                        'audio_vec': np.zeros(1024),
                        'vlm_vec': np.zeros(1536)
                    }
        
        # 清理临时视频文件和 _temp_video_path 字段
        for shot_id, temp_path in shot_video_paths.items():
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            # 删除内部使用的临时字段
            if shot_id in processed_shots and '_temp_video_path' in processed_shots[shot_id]:
                del processed_shots[shot_id]['_temp_video_path']
        
        logger.info(f"All shots processed: {len(processed_shots)} shots")
        return processed_shots
    
    async def _process_single_shot(
        self,
        video_path: str,
        video_name: str,
        shot: Dict,
        api_key: Optional[str] = None,
        full_asr_segments: Optional[list] = None
    ) -> Dict:
        """
        处理单个 Shot
        
        返回：完整的 Shot 数据结构
        """
        import tempfile
        from moviepy.editor import VideoFileClip
        import numpy as np
        
        shot_id = shot['shot_id']
        shot_start = shot['start']
        shot_end = shot['end']
        shot_duration = shot['duration']
        
        try:
            # Step 1: 切割视频片段
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                shot_video_path = tmp_file.name
            
            with VideoFileClip(video_path) as video:
                shot_clip = video.subclip(shot_start, shot_end)
                shot_clip.write_videofile(
                    shot_video_path,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                shot_clip.close()
            
            # Step 2: 提取关键帧（简化版：均匀采样 10 帧）
            keyframes = []
            with VideoFileClip(shot_video_path) as shot_video:
                duration = shot_video.duration
                num_frames = min(10, int(duration * 2))  # 最多 10 帧，最少每秒 2 帧
                if num_frames == 0:
                    num_frames = 1
                
                for i in range(num_frames):
                    t = (i + 0.5) * duration / num_frames
                    frame = shot_video.get_frame(t)
                    keyframes.append(frame)
            
            # Step 2.5: Shot 质量过滤
            from ._videoutil.shot_filter import assess_shot_quality
            quality = assess_shot_quality(keyframes, shot_duration)

            # Step 3: 从全视频 ASR 结果中按时间范围切出对应文本
            if full_asr_segments:
                asr_text = self._slice_asr_by_time(full_asr_segments, shot_start, shot_end)
                logger.debug(f"Shot {shot_id} ASR (sliced): {len(asr_text)} chars")
            else:
                asr_text = ""
                logger.debug(f"Shot {shot_id}: no ASR data available")

            # Step 4: VLM 结构化分析（低信息量 Shot 跳过 VLM 调用）
            if quality["is_low_info"]:
                logger.info(f"Shot {shot_id} skipped VLM: {quality['reason']}")
                vlm_summary = {
                    "subject": [], "action": [], "environment": [], "ocr": [],
                    "caption": f"Low-info shot ({quality['reason']})",
                }
            else:
                vlm_summary = shot_vlm_analysis(
                    model_or_key=api_key or "",
                    shot_path=shot_video_path,
                    keyframes=keyframes,
                    asr_context=asr_text
                )
            
            # Step 5: 返回结果（不包含向量，向量稍后批量编码）
            result = {
                "shot_id": shot_id,
                "video_id": video_name,
                "time_start": shot_start,
                "time_end": shot_end,
                "duration": shot_duration,
                
                "vectors": None,  # 占位，稍后批量编码时填充
                
                "content": {
                    "asr_text": asr_text,
                    "vlm_shot_summary": vlm_summary
                },
                
                "cognitive_links": {
                    "segment_id": None,  # 稍后填充
                    "narrative_summary": None,
                    "entity_mentions": vlm_summary.get('subject', []),
                    "event_type": "unknown"
                },
                
                "_temp_video_path": shot_video_path  # 临时文件路径，用于后续编码和清理
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing shot {shot_id}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _build_segment_data(
        self,
        segments: List[Dict],
        processed_shots: Dict[str, Dict],
        api_key: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        构建 Segment 数据（包含叙事理解）
        """
        import numpy as np
        
        segment_data = {}
        
        for segment in segments:
            segment_id = segment['segment_id']
            shot_ids = segment['shot_ids']
            
            # 收集该 Segment 包含的所有 Shots 信息
            segment_shots = [processed_shots[sid] for sid in shot_ids if sid in processed_shots]
            
            if not segment_shots:
                logger.warning(f"Segment {segment_id} has no valid shots, skipping")
                continue
            
            # 聚合 ASR 文本
            aggregated_asr = " ".join([shot['content']['asr_text'] for shot in segment_shots if shot['content']['asr_text']])
            
            # 聚合 Shot summaries
            shot_summaries_list = [shot['content']['vlm_shot_summary'] for shot in segment_shots]
            
            # Segment 级叙事理解
            # 多源 fallback：参数 api_key > self 注入 > env > dashscope SDK 全局
            _seg_key = api_key or getattr(self, 'ali_dashscope_api_key', None) or os.getenv("DASHSCOPE_API_KEY") or ""
            if not _seg_key:
                try:
                    import dashscope as _ds
                    _seg_key = getattr(_ds, 'api_key', '') or ""
                except ImportError:
                    pass
            if not _seg_key:
                logger.warning(f"Segment {segment_id} narrative skipped: no DASHSCOPE_API_KEY available")
                narrative_text = " ".join([s.get('caption', '') for s in shot_summaries_list if isinstance(s, dict)])[:500]
            else:
                narrative_text = segment_narrative_understanding(
                    model_or_key=_seg_key,
                    shot_summaries_list=shot_summaries_list,
                    aggregated_asr=aggregated_asr
                )
            
            # 从 shots 的 vlm_shot_summary 聚合实体名（subject + ocr + environment）
            # 这样 SegmentKVStorage.get_related_segments(entity_names) 才能命中
            entity_name_set = set()
            for shot in segment_shots:
                summary = shot.get('content', {}).get('vlm_shot_summary', {})
                # 防御：VLM 失败时可能返回非 dict（如 None / error string）
                if not isinstance(summary, dict):
                    continue
                for field in ('subject', 'ocr', 'environment'):
                    val = summary.get(field, [])
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str) and item.strip():
                                entity_name_set.add(item.strip())
                    elif isinstance(val, str) and val.strip():
                        entity_name_set.add(val.strip())
            entities = [{"name": name} for name in sorted(entity_name_set)]
            relations = []
            themes = []
            
            # 编码叙事向量 / Encode narrative vector
            narrative_vec = np.zeros(1536)
            if narrative_text and self.embedding_func:
                try:
                    # 使用阿里云百炼或配置的 embedding 模型 / Use DashScope or configured embedding model
                    # Embedding function expects a list of texts, not a single string
                    narrative_vec = await self.embedding_func([narrative_text])
                    if isinstance(narrative_vec, list):
                        narrative_vec = np.array(narrative_vec)
                    # Extract first embedding if we got a batch result
                    if narrative_vec.ndim > 1:
                        narrative_vec = narrative_vec[0]
                    logger.debug(f"Segment {segment_id} narrative encoded: {narrative_vec.shape}")
                except Exception as e:
                    logger.warning(f"Failed to encode narrative for segment {segment_id}: {e}")
            
            segment_data[segment_id] = {
                "segment_id": segment_id,
                "video_id": processed_shots[shot_ids[0]]['video_id'],
                "time_start": segment['start'],
                "time_end": segment['end'],
                "duration": segment['duration'],
                
                "shot_ids": shot_ids,
                
                "cognitive_content": {
                    "narrative_text": narrative_text,
                    "entities": entities,
                    "relations": relations,
                    "themes": themes
                },
                
                "summary_vectors": {
                    "narrative_vec": narrative_vec
                }
            }
            
            # 更新 Shot 的 cognitive_links
            for shot_id in shot_ids:
                if shot_id in processed_shots:
                    processed_shots[shot_id]['cognitive_links']['segment_id'] = segment_id
                    processed_shots[shot_id]['cognitive_links']['narrative_summary'] = narrative_text
        
        logger.info(f"Built {len(segment_data)} segments")
        return segment_data
    
    async def ainsert_from_segments(self, segment_data: Dict[str, Dict]):
        """
        从 Segment 数据构建知识图谱（适配双层架构）
        """
        await self._insert_start()
        try:
            # 将 Segment 数据转换为文本块格式
            inserting_chunks = {}
            for seg_id, seg_data in segment_data.items():
                narrative = seg_data.get('cognitive_content', {}).get('narrative_text', '')
                if narrative:
                    chunk_id = compute_mdhash_id(narrative, prefix="chunk-")
                    inserting_chunks[chunk_id] = {
                        "content": narrative,
                        "video_segment_id": [seg_id],
                        "tokens": len(encode_string_by_tiktoken(narrative)),
                        "time_start": seg_data.get('time_start', 0),
                        "time_end": seg_data.get('time_end', 0),
                    }
            
            if not inserting_chunks:
                logger.warning("No chunks to insert")
                return
            
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks from segments")
            
            # 插入文本块到向量数据库
            if self.enable_naive_rag and self.chunks_vdb:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)
            
            # 实体提取和图谱构建
            logger.info("[Entity Extraction]...")
            maybe_new_kg, all_entities_data, _ = await self.entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return

            self.chunk_entity_relation_graph = maybe_new_kg

            # 跨 Segment 实体对齐
            if all_entities_data and len(inserting_chunks) > 1:
                seg_entity_map = defaultdict(list)
                for ent in all_entities_data:
                    seg_ids = ent.get("segment_ids", "[]")
                    if isinstance(seg_ids, str):
                        try:
                            seg_ids = json.loads(seg_ids)
                        except (json.JSONDecodeError, TypeError):
                            seg_ids = []
                    for sid in seg_ids:
                        seg_entity_map[sid].append(ent["entity_name"])
                if len(seg_entity_map) > 1:
                    logger.info(f"[Entity Alignment] aligning entities across {len(seg_entity_map)} segments")
                    seg_entity_lists = list(seg_entity_map.values())
                    aligned = await self.chunk_entity_relation_graph.align_entities_across_shots(
                        seg_entity_lists,
                        embedding_func=self.embedding_func,
                    )
                    await self.chunk_entity_relation_graph.merge_aligned_entities_into_graph(aligned)

            # 社区聚类 + Report 生成
            if self.community_reports is not None:
                logger.info("[Community Detection]...")
                communities = await self.chunk_entity_relation_graph.detect_communities()
                if communities:
                    use_llm = self.llm.best_model_func
                    reports = await generate_community_reports(
                        communities, self.chunk_entity_relation_graph, use_llm
                    )
                    community_data = {r['community_id']: r for r in reports}
                    await self.community_reports.upsert(community_data)
                    logger.info(f"Generated {len(reports)} community reports")

            # 提交更新
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()
    
    def _cleanup_cache(self, video_name: str):
        """清理临时缓存文件"""
        import shutil
        video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
        if os.path.exists(video_segment_cache_path):
            shutil.rmtree(video_segment_cache_path)
            logger.debug(f"Cleaned up cache for {video_name}")
    
    def _insert_video_legacy(self, video_path_list=None):
        """
        原有的视频插入流程（向后兼容）
        """
        loop: asyncio.AbstractEventLoop = always_get_an_event_loop() # 获取事件循环
        # 用同步方式调用异步方法,遍历视频列表
        for video_path in video_path_list:
            # Step0: 检查视频是否存在
            video_name = os.path.basename(video_path).split('.')[0]
            if video_name in self.video_segments._data:
                logger.info(f"Find the video named {os.path.basename(video_path)} in storage and skip it.")
                continue
            loop.run_until_complete(self.video_path_db.upsert(
                {video_name: video_path}
            ))
            
            # Step1: 分割视频
            segment_index2name, segment_times_info = split_video(
                video_path, 
                self.working_dir, 
                self.video_segment_length,
                self.rough_num_frames_per_segment,
                self.audio_output_format,
            )
            
            # Step2: 语音转文字ASR
            transcripts = speech_to_text(
                video_name, 
                self.working_dir, 
                segment_index2name,
                self.audio_output_format,
                asdict(self)  # Pass global config for ASR API key
            )
            
            # Step3: saving video segments **as well as** obtain caption with vision language model
            manager = multiprocessing.Manager() # 创建一个多进程管理器
            captions = manager.dict() # 创建一个字典对象，用于存储视频片段的描述
            error_queue = manager.Queue() # 创建一个队列对象，用于存储错误信息
            
            process_saving_video_segments = multiprocessing.Process(
                target=saving_video_segments,
                args=(
                    video_name,
                    video_path,
                    self.working_dir,
                    segment_index2name,
                    segment_times_info,
                    error_queue,
                    self.video_output_format,
                )
            ) # 创建一个进程对象，用于保存视频片段
            
            process_segment_caption = multiprocessing.Process(
                target=segment_caption,
                args=(
                    video_name,
                    video_path,
                    segment_index2name,
                    transcripts,
                    segment_times_info,
                    captions,
                    error_queue,
                )
            ) # 创建一个进程对象，用于处理视频片段
            
            process_saving_video_segments.start()
            process_segment_caption.start()
            process_saving_video_segments.join()
            process_segment_caption.join() # 等待两个进程结束
            
            # if raise error in this two, stop the processing
            while not error_queue.empty(): # 如果队列不为空，则处理错误信息
                error_message = error_queue.get()
                with open('error_log_videorag.txt', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Video Name:{video_name} Error processing:\n{error_message}\n\n")
                raise RuntimeError(error_message)
            
            # Step4: insert video segments information
            segments_information = merge_segment_information(
                segment_index2name,
                segment_times_info,
                transcripts,
                captions,
            ) # 合并所有片段信息

            manager.shutdown() # 关闭管理器
            loop.run_until_complete(self.video_segments.upsert(
                {video_name: segments_information}
            )) # 将完整的片段信息存入数据库
            
            # Step5: encode video segment features
            loop.run_until_complete(self.video_segment_feature_vdb.upsert(
                video_name,
                segment_index2name,
                self.video_output_format,
            )) # 将视频片段特征编码后存入向量数据库
            
            # Step6: delete the cache file 清理缓存
            video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            if os.path.exists(video_segment_cache_path):
                shutil.rmtree(video_segment_cache_path)
            
            # Step 7: saving current video information 保存索引
            loop.run_until_complete(self._save_video_segments())

        # Step8: insert video segments information into vector database 构建知识图谱
        loop.run_until_complete(self.ainsert(self.video_segments._data))

        # Step9: Wiki Ingest (legacy mode)
        if self.wiki_ingest_writer is not None:
            logger.info("Wiki ingest (legacy mode)...")
            for video_path in video_path_list:
                video_name = os.path.basename(video_path).split('.')[0]
                try:
                    seg_data = self._legacy_segments_to_wiki_format(video_name)
                    if seg_data:
                        graph_data = loop.run_until_complete(self._extract_graph_data_for_wiki())
                        loop.run_until_complete(
                            self.wiki_ingest_writer.ingest_video(
                                video_name=video_name,
                                segment_data=seg_data,
                                graph_data=graph_data,
                            )
                        )
                except Exception as e:
                    logger.warning(f"Wiki ingest failed for {video_name} (non-fatal): {e}")

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        # [Wiki] 先尝试 Wiki 检索路径
        if self.wiki_query_retriever is not None and param.mode == "videorag":
            wiki_result = await self.wiki_query_retriever.query(query)
            if wiki_result.confidence >= self.wiki_config.query_confidence_threshold:
                logger.info(f"Wiki query hit: confidence={wiki_result.confidence:.2f}")
                await self._query_done()
                return wiki_result.answer
            else:
                logger.info(f"Wiki query miss (confidence={wiki_result.confidence:.2f}), "
                            f"falling back to engine")

        if param.mode == "videorag":
            # 根据配置选择查询模式
            if self.enable_shot_level and hasattr(self, 'shot_storage'):
                # 融合模式：使用双层架构查询
                logger.info("Using fusion query mode (Shot-Segment dual-layer)")
                response = await videorag_query_fusion(
                    query,
                    param,
                    asdict(self),
                    shot_storage=self.shot_storage,
                    shot_vdb=self.shot_vdb,
                    shot_bm25=self.shot_bm25,
                    reranker=self.reranker,
                    segment_storage=self.segment_storage,
                    graph_db=self.chunk_entity_relation_graph,
                    embedding_func=self.embedding_func,
                    text_chunks_db=self.text_chunks,
                    entities_vdb=self.entities_vdb,
                    community_reports=getattr(self, 'community_reports', None),
                    visual_embedder=getattr(self, 'visual_embedder', None) or getattr(self, '_imagebind_model', None),
                )
            else:
                # 传统模式：使用原有查询逻辑
                logger.info("Using legacy query mode")
                response = await videorag_query(
                    query, # 用户输入的查询
                    self.entities_vdb, # 实体向量数据库
                    self.text_chunks,# 文本块
                    self.chunks_vdb, # 文本块向量数据库
                    self.video_path_db, # 视频路径数据库
                    self.video_segments, # 视频片段信息
                    self.video_segment_feature_vdb, # 视频片段特征向量数据库
                    self.chunk_entity_relation_graph, # 知识图谱
                    self.caption_model, # 捕获模型
                    self.caption_tokenizer, # 捕获分词器
                    param,
                    asdict(self),
                )
        # NOTE: update here
        elif param.mode == "videorag_multiple_choice": # 多项选择查询模式
            response = await videorag_query_multiple_choice(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model, 
                self.caption_tokenizer,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    def query_by_image(self, image_path: str, top_k: int = 5) -> list:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_by_image(image_path, top_k))

    async def aquery_by_image(self, image_path: str, top_k: int = 5) -> list:
        """以图搜视频：输入图片，返回视觉上最相似的 Shot 列表。"""
        if not self.enable_shot_level or not hasattr(self, 'shot_vdb'):
            raise ValueError("Image query requires shot-level mode")

        visual_embedder = getattr(self, 'visual_embedder', None) or getattr(self, '_imagebind_model', None)
        if visual_embedder is None:
            raise ValueError("Image query requires ImageBind model")

        from ._videoutil.feature import encode_image_query
        query_vec = encode_image_query(image_path, visual_embedder)

        results = await self.shot_vdb.search_visual(query_vec, top_k=top_k)

        shot_details = []
        for shot_id, score in results:
            shot_info = await self.shot_storage.get_by_id(shot_id)
            if shot_info:
                shot_info['retrieval_score'] = score
                shot_details.append(shot_info)

        return shot_details

    def _require_shot_embedder(self):
        """获取 ImageBind 编码器（visual/audio/video 查询共用），缺失则报错。"""
        if not self.enable_shot_level or not hasattr(self, 'shot_vdb'):
            raise ValueError("Media query requires shot-level mode")
        embedder = getattr(self, 'visual_embedder', None) or getattr(self, '_imagebind_model', None)
        if embedder is None:
            raise ValueError("Media query requires ImageBind model (install [imagebind] extra)")
        return embedder

    async def _collect_shot_details(self, results: list) -> list:
        """将 (shot_id, score) 列表回填为完整 Shot 信息。"""
        shot_details = []
        for shot_id, score in results:
            shot_info = await self.shot_storage.get_by_id(shot_id)
            if shot_info:
                shot_info['retrieval_score'] = score
                shot_details.append(shot_info)
        return shot_details

    def query_by_audio(self, audio_path: str, top_k: int = 5) -> list:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_by_audio(audio_path, top_k))

    async def aquery_by_audio(self, audio_path: str, top_k: int = 5) -> list:
        """以音搜视频：输入音频，返回声学上最相似的 Shot 列表。"""
        embedder = self._require_shot_embedder()
        from ._videoutil.feature import encode_audio_query
        query_vec = encode_audio_query(audio_path, embedder)
        results = await self.shot_vdb.search_audio(query_vec, top_k=top_k)
        return await self._collect_shot_details(results)

    def query_by_video_clip(self, video_path: str, top_k: int = 5) -> list:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_by_video_clip(video_path, top_k))

    async def aquery_by_video_clip(self, video_path: str, top_k: int = 5) -> list:
        """以视频片段搜视频：输入短片视频，返回视觉上最相似的 Shot 列表。"""
        embedder = self._require_shot_embedder()
        from ._videoutil.feature import encode_video_clip_query
        query_vec = encode_video_clip_query(video_path, embedder)
        results = await self.shot_vdb.search_visual(query_vec, top_k=top_k)
        return await self._collect_shot_details(results)

    async def ainsert(self, new_video_segment): # new_video_segment 新的视频片段数据字典
        # 定义异步插入方法
        await self._insert_start() # 启动索引前的准备工作
        try:
            # ---------- chunking
            inserting_chunks = get_chunks(
                new_videos=new_video_segment,
                chunk_func=self.chunk_func, # 文本块分割函数
                max_token_size=self.chunk_token_size, # 最大分块大小
            ) # 将视频内容分割为文本块的函数

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            ) # 检查文本块是否已经存在，过滤掉已存在的键

            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            } # {...} 字典推导式，创建新的过滤后的字典；只保留已存在的键对应的值

            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks) # 批量插入或更新文本块到向量数据库

            # TODO: no incremental update for communities now, so just drop all
            # await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            # 实体提取函数
            maybe_new_kg, _, _ = await self.entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            # 更新实例的知识图谱属性为新返回的实例
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- commit upsertings and indexing
            await self.text_chunks.upsert(inserting_chunks) # 将文本块持久化到存储
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks) # 并发执行多个协程，等待所有任务完成

    async def _save_video_segments(self): # 保存视频片段相关的所有存储
        tasks = []
        for storage_inst in [
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
            # 调用每个存储的索引完成回调，加入任务列表
        await asyncio.gather(*tasks)
    
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
            getattr(self, 'community_reports', None),
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks) # 并发执行多个协程，等待所有任务完成

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    # ---- Wiki 辅助方法 ----

    async def _extract_graph_data_for_wiki(self) -> dict:
        """从 NetworkX 图谱中提取节点和边，供 Wiki Ingest 使用"""
        graph = self.chunk_entity_relation_graph
        nodes = []
        edges = []
        try:
            if hasattr(graph, '_graph'):
                g = graph._graph
                for node_id, data in g.nodes(data=True):
                    nodes.append({"entity_name": node_id, **data})
                for src, tgt, data in g.edges(data=True):
                    edges.append({"src_id": src, "tgt_id": tgt, **data})
        except Exception as e:
            logger.warning(f"Failed to extract graph data for wiki: {e}")
        return {"nodes": nodes, "edges": edges}

    def _legacy_segments_to_wiki_format(self, video_name: str) -> dict:
        """将 legacy 模式的 segment 数据转为 Wiki Ingest 格式"""
        if video_name not in self.video_segments._data:
            return {}
        raw = self.video_segments._data[video_name]
        result = {}
        for seg_id, seg_info in raw.items():
            content = seg_info.get("content", "")
            time_str = seg_info.get("time", "0-0")
            parts = time_str.split("-")
            t_start = float(parts[0]) if len(parts) > 0 else 0
            t_end = float(parts[1]) if len(parts) > 1 else 0
            result[f"{video_name}_{seg_id}"] = {
                "segment_id": f"{video_name}_{seg_id}",
                "video_id": video_name,
                "time_start": t_start,
                "time_end": t_end,
                "duration": t_end - t_start,
                "cognitive_content": {
                    "narrative_text": content,
                    "entities": [],
                    "relations": [],
                },
            }
        return result

    def lint(self):
        """运行 Wiki 体检"""
        if self.wiki_linter is None:
            logger.warning("Wiki is not enabled, lint skipped")
            return []
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.wiki_linter.lint())

    def wiki_stats(self) -> dict:
        """返回 Wiki 知识库统计信息"""
        if self.wiki is None:
            return {"enabled": False}
        stats = self.wiki.stats()
        stats["enabled"] = True
        stats["config"] = self.wiki_config.to_dict()
        return stats
