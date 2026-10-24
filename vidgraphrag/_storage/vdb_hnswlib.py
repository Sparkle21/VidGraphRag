import asyncio
import os
from dataclasses import dataclass, field
from typing import Any
import pickle
import hnswlib
import numpy as np
import xxhash

from .._utils import logger
from ..base import BaseVectorStorage

# 基于 HNSWLib 的近似最近邻搜索（ANN）向量数据库
# 作用：存储向量嵌入，支持快速相似度检索
#      -这是一个基于 HNSW 算法的高性能内存向量数据库，用余弦相似度做快速检索，
#       通过哈希映射管理 ID，支持持久化到本地文件。
#       HNSWLib 解决"找什么"（语义匹配），Neo4j 解决"有什么关系"（结构遍历）。
#       两者结合才能实现智能检索：先用向量找到相关实体，再用图谱挖掘深度信息。

@dataclass
class HNSWVectorStorage(BaseVectorStorage):
    """基于 HNSW (Hierarchical Navigable Small World) 算法的高性能向量数据库实现，支持近似最近邻搜索"""
    # 核心组件:
    # 1. self._index: HNSW 索引（内存中）
    # 2. self._metadata: 元数据缓存（ID → 原始信息映射）
    # 3. 持久化文件: .index (索引) + .pkl (元数据)
    ef_construction: int = 100
    M: int = 16
    max_elements: int = 1000000
    ef_search: int = 50
    num_threads: int = -1
    _index: Any = field(init=False)
    _metadata: dict[str, dict] = field(default_factory=dict)
    _current_elements: int = 0

    def __post_init__(self):
        self._index_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_hnsw.index"
        )
        self._metadata_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_hnsw_metadata.pkl"
        )
        self._embedding_batch_num = self.global_config.get("embedding_batch_num", 20)

        hnsw_params = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.ef_construction = hnsw_params.get("ef_construction", self.ef_construction)
        self.M = hnsw_params.get("M", self.M)
        self.max_elements = hnsw_params.get("max_elements", self.max_elements)
        self.ef_search = hnsw_params.get("ef_search", self.ef_search)
        self.num_threads = hnsw_params.get("num_threads", self.num_threads)
        # 初始化 HNSW 索引，使用余弦相似度作为距离度量
        self._index = hnswlib.Index(
            space="cosine", dim=self.embedding_func.embedding_dim
        )

        if os.path.exists(self._index_file_name) and os.path.exists(
            self._metadata_file_name
        ):
            # 加载已有的索引和元数据
            self._index.load_index(
                self._index_file_name, max_elements=self.max_elements
            )
            with open(self._metadata_file_name, "rb") as f:
                self._metadata, self._current_elements = pickle.load(f)
            logger.info(
                f"Loaded existing index for {self.namespace} with {self._current_elements} elements"
            )
        else:
            # 创建新的空索引
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.set_ef(self.ef_search)
            self._metadata = {}
            self._current_elements = 0
            logger.info(f"Created new index for {self.namespace}")

    async def upsert(self, data: dict[str, dict]) -> np.ndarray:
        """
            将文本内容转换为向量并插入索引中

            参数：
                data: 待插入的数据字典，键为 ID，值为包含 'content' 字段的字典

            返回：
                np.ndarray: 插入成功后返回每个条目的整数 ID 数组
        """
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not data:
            logger.warning("You insert an empty data to vector DB")
            return []

        if self._current_elements + len(data) > self.max_elements:
            raise ValueError(
                f"Cannot insert {len(data)} elements. Current: {self._current_elements}, Max: {self.max_elements}"
            )

        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        # 分批计算嵌入向量以提高效率
        batch_size = min(self._embedding_batch_num, len(contents))
        embeddings = np.concatenate(
            await asyncio.gather(
                *[
                    self.embedding_func(contents[i : i + batch_size])
                    for i in range(0, len(contents), batch_size)
                ]
            )
        )

        # 使用 xxhash 将字符串 ID 映射为唯一的 32 位整数 ID
        ids = np.fromiter(
            (xxhash.xxh32_intdigest(d["id"].encode()) for d in list_data),
            dtype=np.uint32,
            count=len(list_data),
        )
        # 更新元数据缓存，保存原始 ID 和其他字段信息
        self._metadata.update(
            {
                id_int: {
                    k: v for k, v in d.items() if k in self.meta_fields or k == "id"
                }
                for id_int, d in zip(ids, list_data)
            }
        )
        # 向 HNSW 索引中添加向量数据
        self._index.add_items(data=embeddings, ids=ids, num_threads=self.num_threads)
        self._current_elements = self._index.get_current_count()
        return ids

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        """
            执行向量相似度检索，找出与查询文本最相似的前 K 个条目

           参数：
               query: 查询文本字符串
               top_k: 返回的最相似条目数量，默认为 5

               返回：
                list[dict]: 包含检索结果及其相似度分数的字典列表，按相似度降序排列
       """
        if self._current_elements == 0:
            return []

        top_k = min(top_k, self._current_elements)

        if top_k > self.ef_search:
            logger.warning(
                f"Setting ef_search to {top_k} because top_k is larger than ef_search"
            )
            self._index.set_ef(top_k)

        # 计算查询文本的嵌入向量
        embedding = await self.embedding_func([query])
        # 在 HNSW 索引中执行 K 近邻搜索
        labels, distances = self._index.knn_query(
            data=embedding[0], k=top_k, num_threads=self.num_threads
        )

        return [
            {
                **self._metadata.get(label, {}),
                "distance": distance,
                "similarity": 1 - distance,
            }
            for label, distance in zip(labels[0], distances[0])
        ]

    async def index_done_callback(self):
        """索引构建完成后的回调，负责将索引和元数据持久化到磁盘"""
        self._index.save_index(self._index_file_name)
        with open(self._metadata_file_name, "wb") as f:
            pickle.dump((self._metadata, self._current_elements), f)
