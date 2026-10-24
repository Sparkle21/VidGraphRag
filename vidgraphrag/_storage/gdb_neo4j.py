import json
import asyncio
from collections import defaultdict
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from typing import Union
from ..base import BaseGraphStorage, SingleCommunitySchema
from .._utils import logger
from ..prompt import GRAPH_FIELD_SEP

# 创建异步锁，用于在并发环境下保护 Neo4j 数据库操作的线程安全
neo4j_lock = asyncio.Lock()


def make_path_idable(path):
    return path.replace(".", "_").replace("/", "__").replace("-", "_")


# 这是一个 Neo4j 图数据库的实现类
# 作用：存储和查询知识图谱（实体-关系结构）
@dataclass
class Neo4jStorage(BaseGraphStorage):
    # 核心组件:
    # 1. async_driver: Neo4j 异步驱动
    # 2. namespace: 命名空间（用于隔离不同项目的数据）
    # 3. 所有操作都是异步的（async/await）
    def __post_init__(self):
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        # 创建异步驱动
        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url, auth=self.neo4j_auth
        )

    # async def create_database(self):
    #     async with self.async_driver.session() as session:
    #         try:
    #             constraints = await session.run("SHOW CONSTRAINTS")
    #             # TODO I don't know why CREATE CONSTRAINT IF NOT EXISTS still trigger error
    #             # so have to check if the constrain exists
    #             constrain_exists = False

    #             async for record in constraints:
    #                 if (
    #                     self.namespace in record["labelsOrTypes"]
    #                     and "id" in record["properties"]
    #                     and record["type"] == "UNIQUENESS"
    #                 ):
    #                     constrain_exists = True
    #                     break
    #             if not constrain_exists:
    #                 await session.run(
    #                     f"CREATE CONSTRAINT FOR (n:{self.namespace}) REQUIRE n.id IS UNIQUE"
    #                 )
    #                 logger.info(f"Add constraint for namespace: {self.namespace}")

    #         except Exception as e:
    #             logger.error(f"Error accessing or setting up the database: {str(e)}")
    #             raise

    async def _init_workspace(self):
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()
        # TODOLater: create database if not exists always cause an error when async
        # await self.create_database()

    async def index_start_callback(self):
        logger.info("Init Neo4j workspace")
        await self._init_workspace()

    # 检查节点是否存在
    async def has_node(self, node_id: str) -> bool:
        # Cypher: MATCH (n:namespace) WHERE n.id = node_id RETURN COUNT(n) > 0
        # 返回: True/False
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    # 检查边是否存在
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        # Cypher: MATCH (s)-[r]->(t) WHERE s.id=source AND t.id=target
        # 返回: True/False
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN COUNT(r) > 0 AS exists",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    # 节点度数（连接数)
    async def node_degree(self, node_id: str) -> int:
        # Cypher: MATCH (n) WHERE n.id=node RETURN COUNT{(n)-[]-()} AS degree
        # 例如: Elon Musk 的度数 = 25（连接了 25 个其他实体）
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id "
                f"RETURN COUNT {{(n)-[]-(:{self.namespace})}} AS degree",
                node_id=node_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    # 边的度数（两端节点的总连接数）
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        # Cypher: 计算源节点和目标节点的度数之和
        # 用于衡量关系的重要性
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $src_id AND t.id = $tgt_id "
                f"RETURN COUNT {{(s)-[]-(:{self.namespace})}} + COUNT {{(t)-[]-(:{self.namespace})}} AS degree",
                src_id=src_id,
                tgt_id=tgt_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    # 获取节点数据
    async def get_node(self, node_id: str) -> Union[dict, None]:
        # Cypher: MATCH (n:namespace) WHERE n.id = node_id RETURN properties(n)
        # 返回: {"entity_name": "特斯拉", "entity_type": "COMPANY", ...}
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN properties(n) AS node_data",
                node_id=node_id,
            )
            record = await result.single()
            raw_node_data = record["node_data"] if record else None
        if raw_node_data is None:
            return None
        raw_node_data["clusters"] = json.dumps(
            [
                {
                    "level": index,
                    "cluster": cluster_id,
                }
                for index, cluster_id in enumerate(
                    raw_node_data.get("communityIds", [])
                )
            ]
        )
        return raw_node_data

    # 获取边数据
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        # Cypher: MATCH (s)-[r]->(t) RETURN properties(r)
        # 返回: {"relationship_description": "...", "weight": 0.9}
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN properties(r) AS edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["edge_data"] if record else None

    # 获取节点的所有边
    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        # Cypher: MATCH (s)-[r]->(t) WHERE s.id=source RETURN s.id, t.id
        # 返回: [("特斯拉", "Elon Musk"), ("特斯拉", "上海工厂")]
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) WHERE s.id = $source_id "
                "RETURN s.id AS source, t.id AS target",
                source_id=source_node_id,
            )
            edges = []
            async for record in result:
                edges.append((record["source"], record["target"]))
            return edges

    # 插入/更新数据
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        # Cypher: MERGE (n:namespace:NodeType {id: node_id}) SET n += node_data
        # MERGE = 存在就更新，不存在就创建
        # 例如: 创建"特斯拉"节点，类型为 COMPANY
        node_type = node_data.get("entity_type", "UNKNOWN").strip('"')
        async with self.async_driver.session() as session:
            await session.run(
                f"MERGE (n:{self.namespace}:{node_type} {{id: $node_id}}) "
                "SET n += $node_data",
                node_id=node_id,
                node_data=node_data,
            )

    # 插入/更新边
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        # Cypher: MATCH (s), (t) MERGE (s)-[r:RELATED]->(t) SET r += edge_data
        # 创建"特斯拉"-RELATED->"Elon Musk"的关系
        edge_data.setdefault("weight", 0.0)
        async with self.async_driver.session() as session:
            await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "MERGE (s)-[r:RELATED]->(t) "  # Added relationship type 'RELATED'
                "SET r += $edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
                edge_data=edge_data,
            )

    async def clustering(self, algorithm: str):
        """
            对知识图谱执行社区发现算法，将节点聚类到不同的层级社区中

            参数：
                algorithm: 聚类算法名称，当前仅支持 "leiden"

            异常：
                ValueError: 当传入不支持的算法时抛出
        """
        if algorithm != "leiden":
            raise ValueError(
                f"Clustering algorithm {algorithm} not supported in Neo4j implementation"
            )

        # 步骤 1: 投影图（将图复制到内存）
        random_seed = self.global_config["graph_cluster_seed"] # 随机种子
        max_level = self.global_config["max_graph_cluster_size"] # 最大层级
        async with self.async_driver.session() as session:
            try:
                # Project the graph with undirected relationships
                # 将 Neo4j 中的图数据投影到内存中，构建无向加权图供
                await session.run(
                    f"""
                    CALL gds.graph.project(
                        'graph_{self.namespace}',
                        ['{self.namespace}'],
                        {{
                            RELATED: {{
                                orientation: 'UNDIRECTED',
                                properties: ['weight']
                            }}
                        }}
                    )
                    """
                )

                # Run Leiden algorithm
                # 执行 Leiden 社区发现算法，将计算得到的社区 ID 写回节点属性
                result = await session.run(
                    f"""
                    CALL gds.leiden.write(
                        'graph_{self.namespace}',
                        {{
                            writeProperty: 'communityIds',
                            includeIntermediateCommunities: True,
                            relationshipWeightProperty: "weight",
                            maxLevels: {max_level},
                            tolerance: 0.0001,
                            gamma: 1.0,
                            theta: 0.01,
                            randomSeed: {random_seed}
                        }}
                    )
                    YIELD communityCount, modularities;
                    """
                )
                result = await result.single()
                community_count: int = result["communityCount"]
                modularities = result["modularities"]
                logger.info(
                    f"Performed graph clustering with {community_count} communities and modularities {modularities}"
                )
            finally:
                # Drop the projected graph
                # 释放内存中的投影图，避免资源泄漏
                await session.run(f"CALL gds.graph.drop('graph_{self.namespace}')")

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
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

        async with self.async_driver.session() as session:
            # Fetch community data
            # 查询所有节点的社区归属及其关联的节点信息
            result = await session.run(
                f"""
                MATCH (n:{self.namespace})
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:{self.namespace}) | m.id] AS connected_nodes
                RETURN n.id AS node_id, n.source_id AS source_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            # records = await result.fetch()

            max_num_ids = 0
            async for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    source_id = record["source_id"]
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    results[cluster_key]["chunk_ids"].update(chunk_ids)
                    max_num_ids = max(
                        max_num_ids, len(results[cluster_key]["chunk_ids"])
                    )

            # Process results
            # 将集合类型转换为列表，并计算每个社区的相对出现频率
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
                v["occurrence"] = len(v["chunk_ids"]) / max_num_ids

            # Compute sub-communities (this is a simplified approach)
            # 识别并记录每个社区的子社区（基于节点包含关系）
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    async def index_done_callback(self):
        """索引构建完成后的回调函数，用于关闭数据库连接"""
        await self.async_driver.close()

    async def _debug_delete_all_node_edges(self):
        """调试专用方法：清空当前命名空间下的所有节点和关系"""
        async with self.async_driver.session() as session:
            try:
                # Delete all relationships in the namespace
                await session.run(f"MATCH (n:{self.namespace})-[r]-() DELETE r")

                # Delete all nodes in the namespace
                await session.run(f"MATCH (n:{self.namespace}) DELETE n")

                logger.info(
                    f"All nodes and edges in namespace '{self.namespace}' have been deleted."
                )
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {str(e)}")
                raise
