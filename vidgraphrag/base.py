from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar

import numpy as np

from ._utils import EmbeddingFunc


@dataclass
class QueryParam:
    """
    查询参数配置类，用于定义视频检索和问答的各种参数。

    属性:
        mode (Literal["local", "global", "naive"]): 查询模式
            - "local": 基于本地知识图谱的查询
            - "global": 基于全局知识图谱的查询（默认）
            - "naive": 基础 RAG 检索
        only_need_context (bool): 是否仅需要上下文信息，不需要生成回答
        response_type (str): 回答类型，如"Multiple Paragraphs"（多段落格式）
        level (int): 知识图谱的层级深度，控制检索范围（默认 2 层）
        top_k (int): 检索返回的 top-k 结果数量（默认 20 个）
        naive_max_token_for_text_unit (int): 基础 RAG 模式下文本块的最大 token 数（默认 12000）
    """
    mode: Literal["local", "global", "naive"] = "global"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 5
    # naive search
    naive_max_token_for_text_unit = 12000
    # videorag search
    only_need_context: bool = False


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {
        "tokens": int,  # 文本块的 token 数量
        "content": str,  # 文本块的实际内容
        "video_segment_id": str,  # 所属视频片段的 ID
        "chunk_order_index": int,  # 在视频中的分块顺序索引
        "time_start": float,  # 片段起始时间（秒）
        "time_end": float,  # 片段结束时间（秒）
    },
)

SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,  # 社区层级（数字越小层级越高）
        "title": str,  # 社区标题
        "edges": list[list[str, str]],  # 边列表，每个元素为 [源节点，目标节点]
        "nodes": list[str],  # 节点（实体）ID 列表
        "chunk_ids": list[str],  # 关联的文本块 ID 列表
        "occurrence": float,  # 社区出现频率/权重
        "sub_communities": list[str]  # 子社区标题列表
    },
)


class CommunitySchema(SingleCommunitySchema):
    """
    社区报告 Schema，继承 SingleCommunitySchema 并添加报告字段。
    用于存储知识图谱社区聚类的结果和生成的摘要报告。

    新增属性:
        report_string (str): 社区报告的字符串形式
        report_json (dict): 社区报告的结构化 JSON 数据
    """
    report_string: str
    report_json: dict


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    """
    命名空间基类，所有存储组件都继承此类。
    提供统一的命名空间和回调接口。

    属性:
        namespace (str): 存储空间的命名标识，用于区分不同类型的存储
        global_config (dict): 全局配置字典，包含系统级别的各项参数

    方法:
        index_start_callback(): 索引开始前的回调函数（空实现）
        index_done_callback(): 索引完成后的回调函数，用于持久化数据
        query_done_callback(): 查询完成后的回调函数，用于清理缓存
    """
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    """
    向量存储基类，继承自 StorageNameSpace。
    定义了向量数据库的标准接口，用于存储和检索嵌入向量。

    属性:
        embedding_func (EmbeddingFunc): 嵌入函数，用于将文本转换为向量
        meta_fields (set): 元数据字段集合，存储额外的元信息

    方法:
        query(): 根据查询字符串检索最相似的 top-k 个结果
        upsert(): 批量插入或更新向量数据
    """
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        """
        查询与给定查询最相似的向量。

        参数:
            query (str): 查询字符串，将被嵌入后与库中向量比较
            top_k (int): 返回最相似的 top-k 个结果

        返回:
            list[dict]: 包含匹配结果的列表，每个元素为字典，包含 ID、相似度等信息

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """
        批量插入或更新向量数据。

        参数:
            data (dict[str, dict]): 待存储的数据字典
                - key: 用作向量的唯一标识符
                - value: 包含以下字段的字典：
                    - 'content': 用于嵌入的文本内容
                    - 'embedding': 如果 embedding_func 为 None 时使用预计算向量

        返回:
            None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """
    键值存储基类，继承自 StorageNameSpace，支持泛型。
    定义了键值对存储的标准接口，用于存储结构化数据。

    泛型参数:
        T: 存储值的类型

    属性:
        namespace (str): 存储空间名称
        global_config (dict): 全局配置

    方法:
        all_keys(): 获取所有键
        get_by_id(): 根据单个 ID 获取值
        get_by_ids(): 根据多个 ID 批量获取值
        filter_keys(): 过滤掉已存在的键，返回不存在的键
        upsert(): 批量插入或更新键值对
        drop(): 删除所有数据
    """

    async def all_keys(self) -> list[str]:
        """
        获取存储中的所有键。

        返回:
            list[str]: 所有键的列表

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        """
        根据单个 ID 获取对应的值。

        参数:
            id (str): 要查询的键

        返回:
            Union[T, None]: 如果找到则返回对应的值，否则返回 None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def get_by_ids(
            self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        """
        根据多个 ID 批量获取值。

        参数:
            ids (list[str]): 要查询的键列表
            fields (Union[set[str], None], optional): 指定要获取的字段集合，None 表示获取全部

        返回:
            list[Union[T, None]]: 值的列表，对应输入的 IDs，不存在的键返回 None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """
        过滤掉已存在的键，返回不存在的键集合。
        用于增量更新时判断哪些键需要新建。

        参数:
            data (list[str]): 待检查的键列表

        返回:
            set[str]: 不存在于存储中的键集合

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        """
        批量插入或更新键值对。

        参数:
            data (dict[str, T]): 待存储的键值对字典

        返回:
            None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def drop(self):
        """
        删除存储中的所有数据。

        返回:
            None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    """
    图存储基类，继承自 StorageNameSpace。
    定义了知识图谱存储的标准接口，用于存储实体和关系。

    属性:
        namespace (str): 存储空间名称
        global_config (dict): 全局配置

    方法:
        has_node(): 检查节点是否存在
        has_edge(): 检查边是否存在
        node_degree(): 获取节点的度（连接数）
        edge_degree(): 获取边的度
        get_node(): 获取节点信息
        get_edge(): 获取边信息
        get_node_edges(): 获取节点的所有相邻边
        upsert_node(): 插入或更新节点
        upsert_edge(): 插入或更新边
        clustering(): 执行图聚类算法
        community_schema(): 获取社区的 Schema 表示
        embed_nodes(): 计算节点嵌入向量
    """

    async def has_node(self, node_id: str) -> bool:
        """
        检查图中是否存在指定节点。

        参数:
            node_id (str): 节点 ID

        返回:
            bool: 节点存在返回 True，否则返回 False

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        检查图中是否存在指定边。

        参数:
            source_node_id (str): 源节点 ID
            target_node_id (str): 目标节点 ID

        返回:
            bool: 边存在返回 True，否则返回 False

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        """
        获取节点的度（与该节点相连的边数）。

        参数:
            node_id (str): 节点 ID

        返回:
            int: 节点的度数

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        获取边的度（源节点度和目标节点度的总和）。

        参数:
            src_id (str): 源节点 ID
            tgt_id (str): 目标节点 ID

        返回:
            int: 边的度数

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """
        获取节点的详细信息。

        参数:
            node_id (str): 节点 ID

        返回:
            Union[dict, None]: 节点信息字典（包含属性等），不存在则返回 None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def get_edge(
            self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """
        获取边的详细信息。

        参数:
            source_node_id (str): 源节点 ID
            target_node_id (str): 目标节点 ID

        返回:
            Union[dict, None]: 边信息字典（包含权重、关系类型等），不存在则返回 None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def get_node_edges(
            self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        """
        获取与指定节点相连的所有边。

        参数:
            source_node_id (str): 节点 ID

        返回:
            Union[list[tuple[str, str]], None]: 边列表，每个元素为 (源节点，目标节点) 元组

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """
        插入新节点或更新已有节点的信息。

        参数:
            node_id (str): 节点 ID
            node_data (dict[str, str]): 节点属性字典

        返回:
            None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """
        插入新边或更新已有边的信息。

        参数:
            source_node_id (str): 源节点 ID
            target_node_id (str): 目标节点 ID
            edge_data (dict[str, str]): 边属性字典（如关系类型、权重等）

        返回:
            None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        """
        对图执行聚类算法，发现社区结构。

        参数:
            algorithm (str): 聚类算法名称（如 "leiden", "louvain" 等）

        返回:
            None

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """
        Return the community representation with report and nodes.
        获取社区的 Schema 表示，包含报告和节点信息。

        返回:
            dict[str, SingleCommunitySchema]: 社区字典，key 为社区标题，value 为社区 Schema

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """
        计算节点的嵌入向量。

        参数:
            algorithm (str): 节点嵌入算法名称

        返回:
            tuple[np.ndarray, list[str]]:
                - 第一个元素：节点嵌入向量数组 (n_nodes × embedding_dim)
                - 第二个元素：节点 ID 列表，与向量数组的行对应

        异常:
            NotImplementedError: 子类必须实现此方法

        Note:
            在 nano-graphrag 中未使用此功能
        """
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")
