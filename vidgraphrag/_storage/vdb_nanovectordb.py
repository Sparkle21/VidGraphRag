import asyncio
import os
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from imagebind.models import imagebind_model

try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    hnswlib = None

from .._utils import logger
from ..base import BaseVectorStorage
from .._videoutil import encode_video_segments, encode_string_query


def _hnsw_stored_dim(index_path: str, M: int = 16) -> int:
    """从 hnswlib 二进制索引文件头精确读出向量维度。

    hnswlib 头部前 4 个 uint64 依次为: [0]=起始标签偏移, [8]=max_elements,
    [16]=num_elements, [24]=size_data_per_element。而
    size_data_per_element = dim*4 + (M0*4 + 4) + 8, 其中 level0 链接数 M0 = 2*M。
    故 dim = (size_data_per_element - (2*M*4 + 12)) // 4。
    用途: load_index 对维度不符不报错而会静默错乱(返回全同距离), 需在加载前显式校验。
    """
    import struct
    with open(index_path, "rb") as f:
        head = f.read(32)
    if len(head) < 32:
        raise ValueError(f"unexpected hnsw header size: {len(head)}")
    _f0, _maxel, _numel, sdpe = struct.unpack("<4Q", head[:32])
    overhead = 2 * M * 4 + 12
    return (int(sdpe) - overhead) // 4


# 基于 NanoVectorDB 的轻量级向量数据库
# 两个类：
# 1. NanoVectorDBStorage         → 存储文本嵌入（用于检索）
# 2. NanoVectorDBVideoSegmentStorage → 存储视频片段嵌入（ImageBind 多模态）
# 这是基于 NanoVectorDB 的轻量级向量存储，一个存文本嵌入用于常规检索，另一个用 ImageBind 编码视频片段实现文本搜视频的多模态检索。
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    
    def __post_init__(self):
        # 文件路径: working_dir/vdb_namespace.json

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["llm"]["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        # 相似度阈值：只返回余弦相似度 > 0.2 的结果
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        # 批量计算嵌入（并发）
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        # 存入 NanoVectorDB
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        # 步骤 1: 计算查询向量
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        # 步骤 2: 相似度搜索
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,# 过滤低相似度
        )
        # 步骤 3: 格式化返回
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NanoVectorDBVideoSegmentStorage(BaseVectorStorage):
    embedding_func = None
    segment_retrieval_top_k: float = 2
    
    def __post_init__(self):
        # 使用 ImageBind 模型生成多模态嵌入
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["video_embedding_batch_num"]
        self._client = NanoVectorDB(
            self.global_config["video_embedding_dim"], storage_file=self._client_file_name
        )
        self.top_k = self.global_config.get(
            "segment_retrieval_top_k", self.segment_retrieval_top_k
        )
    
    async def upsert(self, video_name, segment_index2name, video_output_format):
        # 步骤 1: 加载 ImageBind 模型（GPU）
        embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
        embedder.eval()
        
        logger.info(f"Inserting {len(segment_index2name)} segments to {self.namespace}")
        if not len(segment_index2name):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data, video_paths = [], []
        cache_path = os.path.join(self.global_config["working_dir"], '_cache', video_name)
        index_list = list(segment_index2name.keys())
        # 步骤 3: 批量编码视频片段（ImageBind）
        for index in index_list:
            list_data.append({
                "__id__": f"{video_name}_{index}",
                "__video_name__": video_name,
                "__index__": index,
            })
            segment_name = segment_index2name[index]
            video_file = os.path.join(cache_path, f"{segment_name}.{video_output_format}")
            video_paths.append(video_file)
        batches = [
            video_paths[i: i + self._max_batch_size]
            for i in range(0, len(video_paths), self._max_batch_size)
        ]
        embeddings = []
        # 步骤 4: 组装数据
        for _batch in tqdm(batches, desc=f"Encoding Video Segments {video_name}"):
            batch_embeddings = encode_video_segments(_batch, embedder)
            embeddings.append(batch_embeddings)
        embeddings = torch.concat(embeddings, dim=0)
        embeddings = embeddings.numpy()
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        # 步骤 5: 存入数据库
        results = self._client.upsert(datas=list_data)
        return results
    
    async def query(self, query: str):
        # 步骤 1: 用 ImageBind 编码查询文本
        embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
        embedder.eval()
        
        embedding = encode_string_query(query, embedder)
        # 文本和视频映射到同一空间！
        embedding = embedding[0]
        # 步骤 2: 跨模态检索
        results = self._client.query(
            query=embedding,
            top_k=self.top_k,
            better_than_threshold=-1,# 不过滤
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    async def index_done_callback(self):
        self._client.save()


@dataclass
class MultiModalVectorDB(NanoVectorDBStorage):
    """
    支持三路向量的存储和检索
    
    索引结构：
    - visual_index: HNSW 索引，1024 维，用于视觉相似检索
    - audio_index: HNSW 索引，1536 维，用于对话/声音检索
    - vlm_index: HNSW 索引，1536 维，用于语义属性检索
    """
    
    # 覆盖父类的 embedding_func，使其可选
    embedding_func = None
    
    def __post_init__(self):
        # 不调用父类的 __post_init__，因为我们不需要 NanoVectorDB
        # 直接初始化 StorageNameSpace 的基础字段
        if not hasattr(self, 'namespace'):
            raise ValueError("namespace is required")
        if not hasattr(self, 'global_config'):
            raise ValueError("global_config is required")
        
        if not HNSW_AVAILABLE:
            raise ImportError(
                "hnswlib is required for MultiModalVectorDB. "
                "Install it with: pip install hnswlib"
            )
        
        # 初始化三个独立的索引（维度可经 global_config 覆盖，便于模态升级时迁移）
        # audio 现为 ImageBind 声学向量（1024），与 visual 同处 ImageBind 共享空间
        self.visual_dim = int(self.global_config.get("visual_vec_dim", 1024))
        self.audio_dim = int(self.global_config.get("audio_vec_dim", 1024))
        self.vlm_dim = int(self.global_config.get("vlm_vec_dim", 1536))
        self.visual_index = hnswlib.Index(space='cosine', dim=self.visual_dim)
        self.audio_index = hnswlib.Index(space='cosine', dim=self.audio_dim)
        self.vlm_index = hnswlib.Index(space='cosine', dim=self.vlm_dim)
        
        # 配置 HNSW 参数
        ef_construction = self.global_config.get("hnsw_ef_construction", 200)
        M = self.global_config.get("hnsw_M", 16)
        
        # 初始化索引（需要先设置参数再初始化）
        self.visual_index.init_index(max_elements=100000, ef_construction=ef_construction, M=M)
        self.audio_index.init_index(max_elements=100000, ef_construction=ef_construction, M=M)
        self.vlm_index.init_index(max_elements=100000, ef_construction=ef_construction, M=M)
        
        # 设置查询时的 ef 参数
        ef_search = self.global_config.get("hnsw_ef_search", 50)
        self.visual_index.set_ef(ef_search)
        self.audio_index.set_ef(ef_search)
        self.vlm_index.set_ef(ef_search)
        
        # 跟踪已添加的元素数量和 ID 映射
        self.element_count = 0
        self.id_to_shot_id = {}  # int_id -> shot_id 映射
        
        # 自动从磁盘加载已持久化的索引（如存在）。
        # 关键：hnswlib 的 load_index 对维度不符不报错，而是静默错乱解读
        # （任何查询返回全同距离的垃圾），比崩溃更危险。因此逐模态用文件头
        # 读出的存储维度与期望维度比对，不符则跳过该模态并提示重嵌入。
        try:
            working_dir = self.global_config.get("working_dir")
            if working_dir:
                index_dir = os.path.join(working_dir, f"hnsw_{self.namespace}")
                mapping_file = os.path.join(index_dir, "id_mapping.json")
                if os.path.exists(mapping_file):
                    modalities = [
                        ("visual", "visual.index", self.visual_dim, self.visual_index),
                        ("audio", "audio.index", self.audio_dim, self.audio_index),
                        ("vlm", "vlm.index", self.vlm_dim, self.vlm_index),
                    ]
                    for name, fname, exp_dim, index in modalities:
                        ipath = os.path.join(index_dir, fname)
                        if not os.path.exists(ipath):
                            continue
                        try:
                            stored = _hnsw_stored_dim(ipath, M)
                        except Exception as e:
                            logger.warning(f"[{name}] cannot read stored dim ({e}); skipping load")
                            continue
                        if stored != exp_dim:
                            logger.warning(
                                f"[{name}] index stored dim {stored} != expected {exp_dim}; "
                                f"skipping this modality (querying it would return garbage). "
                                f"If this workdir predates the P1 acoustic change "
                                f"(audio_vec 1536->1024), re-ingest to regenerate vectors."
                            )
                            continue
                        index.load_index(ipath)
                    import json as _json
                    with open(mapping_file, 'r') as f:
                        str_mapping = _json.load(f)
                        self.id_to_shot_id = {int(k): v for k, v in str_mapping.items()}
                    self.element_count = len(self.id_to_shot_id)
                    logger.info(
                        f"MultiModalVectorDB auto-loaded {self.element_count} shots from {index_dir} "
                        f"(visual={self.visual_index.element_count}, audio={self.audio_index.element_count}, "
                        f"vlm={self.vlm_index.element_count})"
                    )
        except Exception as e:
            logger.warning(
                f"MultiModalVectorDB auto-load failed (continuing with empty index): {e}. "
                f"If this workdir was built with the old ASR-text audio_vec (dim 1536), re-ingest "
                f"the videos to regenerate the acoustic audio_vec (dim {self.audio_dim})."
            )
        
        logger.info(f"MultiModalVectorDB initialized: namespace={self.namespace}, element_count={self.element_count}")
    
    async def upsert(self, shot_vectors_dict: Dict[str, Dict[str, np.ndarray]]):
        """
        写入 Shot 的三路向量
        
        输入格式：
        {
          "s_001": {
            "visual_vec": np.array([...]),
            "audio_vec": np.array([...]),
            "vlm_vec": np.array([...])
          },
          ...
        }
        """
        if not shot_vectors_dict:
            logger.warning("Empty data to insert")
            return []
        
        visual_vectors = []
        audio_vectors = []
        vlm_vectors = []
        visual_ids = []
        audio_ids = []
        vlm_ids = []
        
        # HNSW 要求 IDs 是整数，我们使用索引作为 ID，并维护一个映射表
        start_idx = self.element_count
        
        for idx, (shot_id, vectors) in enumerate(shot_vectors_dict.items()):
            int_id = start_idx + idx
            
            # 保存 ID 映射
            self.id_to_shot_id[int_id] = shot_id
            
            # 验证向量维度
            if "visual_vec" in vectors:
                assert vectors["visual_vec"].shape == (self.visual_dim,), \
                    f"Visual vector dimension mismatch: {vectors['visual_vec'].shape}"
                visual_vectors.append(vectors["visual_vec"])
                visual_ids.append(int_id)
            
            if "audio_vec" in vectors:
                assert vectors["audio_vec"].shape == (self.audio_dim,), \
                    f"Audio vector dimension mismatch: {vectors['audio_vec'].shape}"
                audio_vectors.append(vectors["audio_vec"])
                audio_ids.append(int_id)
            
            if "vlm_vec" in vectors:
                assert vectors["vlm_vec"].shape == (self.vlm_dim,), \
                    f"VLM vector dimension mismatch: {vectors['vlm_vec'].shape}"
                vlm_vectors.append(vectors["vlm_vec"])
                vlm_ids.append(int_id)
        
        # 批量添加到索引
        if visual_vectors:
            visual_array = np.stack(visual_vectors)
            self.visual_index.add_items(data=visual_array, ids=visual_ids)
        
        if audio_vectors:
            audio_array = np.stack(audio_vectors)
            self.audio_index.add_items(data=audio_array, ids=audio_ids)
        
        if vlm_vectors:
            vlm_array = np.stack(vlm_vectors)
            self.vlm_index.add_items(data=vlm_array, ids=vlm_ids)
        
        self.element_count += len(shot_vectors_dict)
        logger.info(f"Inserted {len(shot_vectors_dict)} shots into MultiModalVectorDB")
        
        return list(shot_vectors_dict.keys())
    
    async def search_visual(self, query_vector: np.ndarray, top_k=20) -> List[Tuple[str, float]]:
        """视觉相似性检索"""
        n = self.visual_index.element_count
        if n == 0:
            return []
        
        query_vec = query_vector.reshape(1, -1)
        labels, distances = self.visual_index.knn_query(query_vec, k=min(top_k, n))
        
        results = []
        for label, distance in zip(labels[0], distances[0]):
            shot_id = self.id_to_shot_id.get(int(label), f"unknown_{label}")
            # HNSW cosine distance 转换为 similarity score
            similarity = 1.0 - distance
            results.append((shot_id, similarity))
        
        return results
    
    async def search_audio(self, query_vector: np.ndarray, top_k=20) -> List[Tuple[str, float]]:
        """声学检索（ImageBind audio 空间，与文本查询向量跨模态对齐）"""
        n = self.audio_index.element_count
        if n == 0:
            return []
        
        query_vec = query_vector.reshape(1, -1)
        labels, distances = self.audio_index.knn_query(query_vec, k=min(top_k, n))
        
        results = []
        for label, distance in zip(labels[0], distances[0]):
            shot_id = self.id_to_shot_id.get(int(label), f"unknown_{label}")
            similarity = 1.0 - distance
            results.append((shot_id, similarity))
        
        return results
    
    async def search_vlm(self, query_vector: np.ndarray, top_k=20) -> List[Tuple[str, float]]:
        """语义属性检索"""
        n = self.vlm_index.element_count
        if n == 0:
            return []
        
        query_vec = query_vector.reshape(1, -1)
        labels, distances = self.vlm_index.knn_query(query_vec, k=min(top_k, n))
        
        results = []
        for label, distance in zip(labels[0], distances[0]):
            shot_id = self.id_to_shot_id.get(int(label), f"unknown_{label}")
            similarity = 1.0 - distance
            results.append((shot_id, similarity))
        
        return results
    
    async def multi_channel_search(
        self,
        query_vectors: Dict[str, np.ndarray],
        top_k=20,
        fusion_strategy='weighted'
    ) -> List[Tuple[str, float]]:
        """
        多路召回 + 结果融合
        
        支持的融合策略：
        - 'rrf': Reciprocal Rank Fusion（倒数排名融合）
        - 'weighted': 加权分数融合（默认）
        - 'semantic_first': 优先 vlm 结果
        """
        all_results = {}
        
        # 并行三路检索
        tasks = []
        
        if query_vectors.get('visual_vec') is not None:
            tasks.append(('visual', self.search_visual(query_vectors['visual_vec'], top_k * 2)))
        
        if query_vectors.get('audio_vec') is not None:
            tasks.append(('audio', self.search_audio(query_vectors['audio_vec'], top_k * 2)))
        
        if query_vectors.get('vlm_vec') is not None:
            tasks.append(('vlm', self.search_vlm(query_vectors['vlm_vec'], top_k * 2)))
        
        if not tasks:
            logger.warning("No query vectors provided")
            return []
        
        # 并发执行检索
        results_list = await asyncio.gather(*[task[1] for task in tasks])
        results_dict = dict(zip([task[0] for task in tasks], results_list))
        
        # 根据融合策略合并结果
        if fusion_strategy == 'rrf':
            all_results = self._fuse_rrf(results_dict, top_k)
        elif fusion_strategy == 'weighted':
            all_results = self._fuse_weighted(results_dict, top_k)
        elif fusion_strategy == 'semantic_first':
            all_results = self._fuse_semantic_first(results_dict, top_k)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # 排序返回
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    # Weighted RRF 通道权重（balanced profile）
    # TODO: 未来可根据 query 意图动态选择 profile（LLM 判断或规则分类）
    CHANNEL_WEIGHTS = {
        'visual': 0.3,
        'audio': 0.2,
        'vlm': 0.3,
        'bm25': 0.2,
    }

    def _fuse_rrf(self, results_dict: Dict[str, List[Tuple[str, float]]], top_k: int) -> Dict[str, float]:
        """
        Weighted RRF 融合策略

        公式: score = sum(w_channel * 1 / (k + rank))
        w_channel 来自 CHANNEL_WEIGHTS，k=60
        """
        rrf_scores = {}
        k_constant = 60

        for channel, results in results_dict.items():
            w = self.CHANNEL_WEIGHTS.get(channel, 0.25)
            for rank, (shot_id, _) in enumerate(results, start=1):
                rrf_score = w * 1.0 / (k_constant + rank)
                rrf_scores[shot_id] = rrf_scores.get(shot_id, 0) + rrf_score

        return rrf_scores
    
    def _fuse_weighted(self, results_dict: Dict[str, List[Tuple[str, float]]], top_k: int) -> Dict[str, float]:
        """
        加权分数融合策略
        
        权重分配：
        - visual: 0.4
        - audio: 0.3
        - vlm: 0.3
        """
        weights = {
            'visual': 0.4,
            'audio': 0.3,
            'vlm': 0.3,
            'bm25': 0.15,
        }
        
        fused_scores = {}
        
        for channel, results in results_dict.items():
            weight = weights.get(channel, 0.3)
            for shot_id, score in results:
                fused_scores[shot_id] = fused_scores.get(shot_id, 0) + score * weight
        
        return fused_scores
    
    def _fuse_semantic_first(self, results_dict: Dict[str, List[Tuple[str, float]]], top_k: int) -> Dict[str, float]:
        """
        语义优先融合策略
        这个就是 Weighted 的变种，只是权重分配不同
        权重分配：
        - visual: 0.25
        - audio: 0.25
        - vlm: 0.5
        优先使用 VLM 结果，其他通道作为补充
        """
        fused_scores = {}
        
        # VLM 通道权重更高
        weights = {
            'visual': 0.25,
            'audio': 0.25,
            'vlm': 0.5,
            'bm25': 0.15,
        }
        
        for channel, results in results_dict.items():
            weight = weights.get(channel, 0.3)
            for shot_id, score in results:
                fused_scores[shot_id] = fused_scores.get(shot_id, 0) + score * weight
        
        return fused_scores
    
    async def index_done_callback(self):
        """保存索引到磁盘"""
        # HNSW 索引持久化
        index_dir = os.path.join(self.global_config["working_dir"], f"hnsw_{self.namespace}")
        os.makedirs(index_dir, exist_ok=True)
        
        self.visual_index.save_index(os.path.join(index_dir, "visual.index"))
        self.audio_index.save_index(os.path.join(index_dir, "audio.index"))
        self.vlm_index.save_index(os.path.join(index_dir, "vlm.index"))
        
        # 保存 ID 映射表
        import json
        mapping_file = os.path.join(index_dir, "id_mapping.json")
        with open(mapping_file, 'w') as f:
            # 将 int key 转换为 string 以便 JSON 序列化
            str_mapping = {str(k): v for k, v in self.id_to_shot_id.items()}
            json.dump(str_mapping, f)
        
        logger.info(f"MultiModalVectorDB indexes saved to {index_dir}")
    
    @classmethod
    def load_from_disk(cls, namespace: str, global_config: dict, **kwargs):
        """
        从磁盘加载已保存的索引
        
        注意：这是一个同步方法，需要在异步上下文中调用
        """
        instance = cls(namespace=namespace, global_config=global_config, **kwargs)
        
        index_dir = os.path.join(global_config["working_dir"], f"hnsw_{namespace}")
        
        if os.path.exists(index_dir):
            visual_index_path = os.path.join(index_dir, "visual.index")
            audio_index_path = os.path.join(index_dir, "audio.index")
            vlm_index_path = os.path.join(index_dir, "vlm.index")
            
            if os.path.exists(visual_index_path):
                instance.visual_index.load_index(visual_index_path)
            if os.path.exists(audio_index_path):
                instance.audio_index.load_index(audio_index_path)
            if os.path.exists(vlm_index_path):
                instance.vlm_index.load_index(vlm_index_path)
            
            # 加载 ID 映射表
            import json
            mapping_file = os.path.join(index_dir, "id_mapping.json")
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    str_mapping = json.load(f)
                    # 将 string key 转换回 int
                    instance.id_to_shot_id = {int(k): v for k, v in str_mapping.items()}
                instance.element_count = len(instance.id_to_shot_id)
                logger.info(f"Loaded {instance.element_count} shots from disk")
            
            logger.info(f"MultiModalVectorDB loaded from {index_dir}")
        else:
            logger.warning(f"No saved indexes found at {index_dir}")
        
        return instance
