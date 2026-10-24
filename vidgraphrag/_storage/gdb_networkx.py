import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, List, Dict, Optional
import networkx as nx
import numpy as np

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP

# 这是一个基于 NetworkX 的内存图存储实现
# 作用：在内存中操作知识图谱，支持持久化到 GraphML 文件
@dataclass
class NetworkXStorage(BaseGraphStorage):
    """基于 NetworkX 的本地图存储实现，支持图的持久化、聚类及节点嵌入"""
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        """从 GraphML 文件加载图数据"""
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)  # 读取 XML 格式的图数据
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        """将图数据写入 GraphML 文件进行持久化"""
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)
        # 例如: <node id="特斯拉"><data key="entity_type">COMPANY</data></node>

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        """提取图中最大的连通分量，并对节点和边进行标准化处理以确保稳定性"""
        # 目的：提取最大连通分量 + 标准化节点名称
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        # 步骤 1: 提取最大的连通子图
        graph = cast(nx.Graph, largest_connected_component(graph))
        # 步骤 2: 统一节点名称（转大写 + 去除 HTML 转义字符）
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        # 步骤 3: 排序节点和边（确保每次运行结果一致）
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        """通过排序节点和边来稳定图结构，确保相同内容的图在读取时具有一致的内部表示"""
        # 对节点和边排序，保证确定性
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        # 重新构建图
        # 为什么要重新构建图呢？
        # 问题：同样的数据，每次运行聚类结果可能不同
        # 原因：NetworkX 内部节点顺序不确定

        # 解决：强制排序节点和边
        # 前: {"tesla", "elon", "alice"}  ← 随机顺序
        # 后: ["ALICE", "ELON", "TESLA"]  ← 固定顺序
        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        # 确定文件路径
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        # 尝试加载已有图文件
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        # 如果文件不存在，创建空图
        self._graph = preloaded_graph or nx.Graph()
        # 注册算法
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        """索引构建完成后的回调，负责将内存中的图保存到磁盘"""
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def clustering(self, algorithm: str):
        """根据指定的算法对图执行聚类分析"""
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """构建并返回多层级的社区结构 Schema"""
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        # 步骤 1: 遍历所有节点，按社区分组
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        # 步骤 2: 计算子社区关系
        # 社区 1 (层级 0) 包含 社区 5 (层级 1) 的所有节点
        # → 社区 5 是 社区 1 的子社区
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        # 步骤 1: 获取稳定的最大连通分量
        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        # 步骤 2: 执行分层 Leiden 算法
        # Leiden 算法是一种社区发现（Community Detection）算法，用于在图中找出紧密连接的节点群
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"], # 最大层级
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        # 步骤 4: 将结果存入节点属性
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        # 步骤 5: 保存到图中
        # 聚类前:
        # 图结构: 特斯拉 --connected--> Elon Musk

        # 聚类后:
        # 节点属性:
        # 特斯拉.clusters = '[{"level": 0, "cluster": 1}, {"level": 1, "cluster": 5}]'
        # Elon Musk.clusters = '[{"level": 0, "cluster": 1}]'
        # ↑ 它们在同一层级属于同一个社区
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        # 使用 Node2Vec 算法生成节点向量
        embeddings, nodes = embed.node2vec_embed(
            self._graph, # 图结构
            **self.global_config["node2vec_params"],
        )

        # embeddings: shape=(100, 128)  ← 100 个节点，每个 128 维向量
        # nodes: ["特斯拉", "Elon Musk", ...]
        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    # ==================== Shot 级知识图谱增强功能 ====================
    
    async def add_shot_node(self, shot_id: str, shot_info: Dict):
        """
        添加 Shot 节点到图谱，添加镜头节点，包含时间、VLM摘要等完整信息
        
        Args:
            shot_id: 镜头ID (如 "s_001")
            shot_info: 包含时间、caption等信息的字典
                {
                    'time_start': 0.0,
                    'time_end': 3.5,
                    'duration': 3.5,
                    'vlm_summary': {'caption': '...', 'subject': [...]},
                    'video_id': 'v_001'
                }
        """
        node_id = f"shot_{shot_id}"
        
        # 构建节点属性
        node_data = {
            'type': 'SHOT',
            'shot_id': shot_id,
            'time_start': shot_info.get('time_start', 0.0),
            'time_end': shot_info.get('time_end', 0.0),
            'duration': shot_info.get('duration', 0.0),
            'video_id': shot_info.get('video_id', ''),
        }
        
        # 添加 VLM 摘要信息
        if 'vlm_summary' in shot_info:
            vlm_summary = shot_info['vlm_summary']
            node_data['caption'] = vlm_summary.get('caption', '')
            node_data['subjects'] = json.dumps(vlm_summary.get('subject', []))
            node_data['actions'] = json.dumps(vlm_summary.get('action', []))
            node_data['environment'] = json.dumps(vlm_summary.get('environment', []))
        
        # 添加到图中
        self._graph.add_node(node_id, **node_data)
        logger.debug(f"Added shot node: {node_id}")
    
    async def add_shot_to_segment_link(self, shot_id: str, segment_id: str):
        """
        建立 Shot 到 Segment 的归属关系
        
        Args:
            shot_id: 镜头ID
            segment_id: 片段ID
        """
        shot_node_id = f"shot_{shot_id}"
        segment_node_id = f"segment_{segment_id}"
        
        # 确保两个节点都存在
        if not self._graph.has_node(shot_node_id):
            logger.warning(f"Shot node {shot_node_id} does not exist")
            return
        
        if not self._graph.has_node(segment_node_id):
            logger.warning(f"Segment node {segment_node_id} does not exist")
            return
        
        # 添加归属关系边
        self._graph.add_edge(
            shot_node_id,
            segment_node_id,
            relation='BELONGS_TO',
            edge_type='hierarchical'
        )
        logger.debug(f"Linked shot {shot_id} to segment {segment_id}")
    
    async def get_subgraph_by_time_range(self, start: float, end: float) -> nx.Graph:
        """
        按时间范围获取子图
        
        Args:
            start: 起始时间（秒）
            end: 结束时间（秒）
        
        Returns:
            包含该时间段内所有 Shot 节点及其相关实体的子图
        """
        subgraph = nx.Graph()
        
        # 找出时间范围内的所有 Shot 节点
        shot_nodes_in_range = []
        for node_id, node_data in self._graph.nodes(data=True):
            if node_data.get('type') == 'SHOT':
                node_start = node_data.get('time_start', 0.0)
                node_end = node_data.get('time_end', 0.0)
                
                # 判断是否有时间重叠
                if node_start < end and node_end > start:
                    shot_nodes_in_range.append(node_id)
                    subgraph.add_node(node_id, **node_data)
        
        logger.info(f"Found {len(shot_nodes_in_range)} shots in time range [{start}, {end}]")
        
        # 添加这些 Shot 之间的边
        for u, v, edge_data in self._graph.edges(data=True):
            if u in shot_nodes_in_range and v in shot_nodes_in_range:
                subgraph.add_edge(u, v, **edge_data)
            elif u in shot_nodes_in_range or v in shot_nodes_in_range:
                # 如果一边是 Shot，另一边是 Entity，也加入
                other_node = v if u in shot_nodes_in_range else u
                if other_node in self._graph.nodes:
                    subgraph.add_node(other_node, **self._graph.nodes[other_node])
                    subgraph.add_edge(u, v, **edge_data)
        
        logger.info(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        return subgraph
    
    async def build_shot_temporal_graph(self, shots_sequence: List[Dict]) -> nx.DiGraph:
        """
        构建 Shot 间的时序关系图
        
        Args:
            shots_sequence: 按时间排序的 Shot 列表
                [
                    {
                        'shot_id': 's_001',
                        'time_start': 0.0,
                        'time_end': 3.24,
                        'vlm_summary': {
                            'caption': '...',
                            'subject': ['钢铁侠']
                        }
                    },
                    ...
                ]
        
        Returns:
            包含时序关系的有向图
        """
        G = nx.DiGraph()
        
        # 添加 Shot 节点
        for i, shot in enumerate(shots_sequence):
            shot_id = shot['shot_id']
            node_id = f"shot_{shot_id}"
            
            # 提取 VLM 摘要
            vlm_summary = shot.get('vlm_summary', {})
            caption = vlm_summary.get('caption', '')
            
            G.add_node(
                node_id,
                type='SHOT',
                shot_id=shot_id,
                time_start=shot.get('time_start', 0.0),
                time_end=shot.get('time_end', 0.0),
                duration=shot.get('duration', 0.0),
                caption=caption,
                subjects=json.dumps(vlm_summary.get('subject', [])),
                actions=json.dumps(vlm_summary.get('action', [])),
                environment=json.dumps(vlm_summary.get('environment', []))
            )
            
            # 添加时序边（NEXT 关系）
            if i > 0:
                prev_shot_id = shots_sequence[i-1]['shot_id']
                prev_node_id = f"shot_{prev_shot_id}"
                temporal_gap = shot.get('time_start', 0.0) - shots_sequence[i-1].get('time_end', 0.0)
                
                G.add_edge(
                    prev_node_id,
                    node_id,
                    relation='NEXT',
                    temporal_gap=temporal_gap,
                    edge_type='temporal'
                )
        
        # 添加实体节点和 APPEAR_IN 边
        for shot in shots_sequence:
            shot_id = shot['shot_id']
            shot_node_id = f"shot_{shot_id}"
            entities = shot.get('vlm_summary', {}).get('subject', [])
            
            for entity in entities:
                if not entity or not entity.strip():
                    continue
                
                entity_node_id = f"entity_{entity}"
                
                # 添加实体节点（如果不存在）
                if not G.has_node(entity_node_id):
                    G.add_node(
                        entity_node_id,
                        type='ENTITY',
                        entity_name=entity
                    )
                
                # 添加 APPEAR_IN 边
                G.add_edge(
                    entity_node_id,
                    shot_node_id,
                    relation='APPEAR_IN',
                    edge_type='semantic'
                )
        
        logger.info(f"Built temporal graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    async def align_entities_across_shots(
        self,
        shot_entities_list: List[List[str]],
        similarity_threshold: float = 0.8,
        embedding_func=None,
        semantic_threshold: float = 0.85,
    ) -> Dict[str, List[str]]:
        """
        跨 Shot 实体对齐

        策略（分两步）：
        1. 字符串匹配（编辑距离 + 子串包含）— 快，处理"张教授"≈"张老师"
        2. 语义匹配（embedding 余弦相似度）— 处理"钢铁侠"≈"穿盔甲的男人"

        Args:
            shot_entities_list: 每个 Shot 的实体列表
            similarity_threshold: 字符串相似度阈值（0-1）
            embedding_func: 文本嵌入函数（可选，传入后启用语义匹配）
            semantic_threshold: 语义相似度阈值（0-1）

        Returns:
            {规范名称: [提及形式列表]}
        """
        aligned_entities = {}

        # Step 1: 字符串匹配
        for shot_idx, entities in enumerate(shot_entities_list):
            for entity in entities:
                if not entity or not entity.strip():
                    continue

                entity_clean = entity.strip()
                found_match = False

                for canonical_name, mentions in aligned_entities.items():
                    if self._are_entities_equivalent(entity_clean, mentions[-1], similarity_threshold):
                        mentions.append(entity_clean)
                        found_match = True
                        logger.debug(f"Aligned '{entity_clean}' to canonical name '{canonical_name}'")
                        break

                if not found_match:
                    aligned_entities[entity_clean] = [entity_clean]

        # Step 2: 语义匹配（批量 embedding，分 batch 避免超过 API 限制）
        remaining = list(aligned_entities.keys())
        if len(remaining) > 1 and embedding_func is not None:
            try:
                BATCH_SIZE = 20
                all_embeddings = []
                for batch_start in range(0, len(remaining), BATCH_SIZE):
                    batch = remaining[batch_start:batch_start + BATCH_SIZE]
                    batch_emb = await embedding_func(batch)
                    all_embeddings.append(batch_emb)
                embeddings = np.concatenate(all_embeddings, axis=0)
                merge_pairs = []
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                        )
                        if sim >= semantic_threshold:
                            merge_pairs.append((remaining[i], remaining[j]))

                for name_keep, name_merge in merge_pairs:
                    if name_merge in aligned_entities and name_keep in aligned_entities:
                        aligned_entities[name_keep].extend(aligned_entities.pop(name_merge))
                        logger.debug(f"Semantic aligned '{name_merge}' → '{name_keep}'")

                if merge_pairs:
                    logger.info(f"Semantic alignment merged {len(merge_pairs)} entity pairs")
            except Exception as e:
                logger.warning(f"Semantic alignment failed: {e}")

        logger.info(f"Entity alignment: {len(aligned_entities)} unique entities from {sum(len(e) for e in shot_entities_list)} mentions")
        return aligned_entities
    
    def _are_entities_equivalent(
        self,
        entity1: str,
        entity2: str,
        threshold: float = 0.8
    ) -> bool:
        """
        判断两个实体是否等价
        
        策略：
        1. 完全匹配
        2. 包含关系（短串是长串的子串）
        3. 编辑距离相似度
        
        Args:
            entity1: 实体1
            entity2: 实体2
            threshold: 相似度阈值
        
        Returns:
            是否等价
        """
        # 标准化
        e1 = entity1.lower().strip()
        e2 = entity2.lower().strip()
        
        # 完全匹配
        if e1 == e2:
            return True
        
        # 包含关系（避免误匹配，要求较短的至少3个字符）
        if len(e1) >= 3 and len(e2) >= 3:
            if e1 in e2 or e2 in e1:
                return True
        
        # 编辑距离相似度
        similarity = self._string_similarity(e1, e2)
        return similarity >= threshold
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        计算两个字符串的相似度（基于编辑距离）
        
        Args:
            s1: 字符串1
            s2: 字符串2
        
        Returns:
            相似度分数（0-1）
        """
        if not s1 or not s2:
            return 0.0
        
        # 计算编辑距离
        len1, len2 = len(s1), len(s2)
        
        # 动态规划计算编辑距离
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 删除
                    dp[i][j-1] + 1,      # 插入
                    dp[i-1][j-1] + cost  # 替换
                )
        
        edit_distance = dp[len1][len2]
        max_len = max(len1, len2)
        
        # 转换为相似度
        similarity = 1.0 - (edit_distance / max_len)
        return similarity
    
    async def merge_aligned_entities_into_graph(
        self,
        aligned_entities: Dict[str, List[str]]
    ):
        """
        将对齐后的实体合并到图中

        主图中实体节点的 ID 就是实体名本身（大写），不带前缀。
        合并时将 mention 节点的边转移到 canonical 节点，然后删除 mention 节点。

        Args:
            aligned_entities: 对齐结果 {规范名称: [提及形式列表]}
        """
        merged_count = 0
        for canonical_name, mentions in aligned_entities.items():
            if len(mentions) <= 1:
                continue

            if not self._graph.has_node(canonical_name):
                self._graph.add_node(
                    canonical_name,
                    entity_type='UNKNOWN',
                    description='',
                    source_id='',
                    aliases=json.dumps(mentions),
                )
            else:
                existing_aliases = self._graph.nodes[canonical_name].get('aliases', '[]')
                try:
                    existing_list = json.loads(existing_aliases)
                except (json.JSONDecodeError, TypeError):
                    existing_list = []
                all_aliases = list(set(existing_list + mentions))
                self._graph.nodes[canonical_name]['aliases'] = json.dumps(all_aliases)

            for mention in mentions:
                if mention == canonical_name:
                    continue
                if not self._graph.has_node(mention):
                    continue
                for neighbor in list(self._graph.neighbors(mention)):
                    if neighbor == canonical_name:
                        continue
                    edge_data = dict(self._graph.edges[mention, neighbor])
                    if not self._graph.has_edge(canonical_name, neighbor):
                        self._graph.add_edge(canonical_name, neighbor, **edge_data)
                self._graph.remove_node(mention)
                merged_count += 1
                logger.debug(f"Merged entity '{mention}' into '{canonical_name}'")

        logger.info(f"Entity alignment: merged {merged_count} mention nodes into canonical nodes")

    async def detect_communities(
        self,
        resolution: float = 1.0,
        min_community_size: int = 2,
    ) -> list:
        """
        Louvain 社区发现

        Returns:
            [{"community_id": "c_001", "entities": [...], "edges": [(u, v, rel), ...], "size": int}, ...]
        """
        from networkx.algorithms.community import louvain_communities

        if self._graph.number_of_nodes() < min_community_size:
            logger.info("Graph too small for community detection")
            return []

        undirected = self._graph.to_undirected()
        try:
            communities = louvain_communities(undirected, resolution=resolution, seed=42)
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return []

        result = []
        for idx, community_nodes in enumerate(communities):
            if len(community_nodes) < min_community_size:
                continue
            entities = list(community_nodes)
            subgraph = self._graph.subgraph(entities)
            edges = []
            for u, v, data in subgraph.edges(data=True):
                edges.append((u, v, data.get('description', 'related_to')))
            result.append({
                "community_id": f"c_{idx + 1:03d}",
                "entities": entities,
                "edges": edges,
                "size": len(entities),
            })

        result.sort(key=lambda x: x['size'], reverse=True)
        logger.info(f"Community detection: {len(result)} communities from {self._graph.number_of_nodes()} nodes")
        return result
