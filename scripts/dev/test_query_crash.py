#!/usr/bin/env python3
"""测试查询崩溃问题"""
import sys
import os
sys.path.insert(0, 'Vimo-desktop/python_backend')

# 设置环境变量
os.environ['DASHSCOPE_API_KEY'] = 'sk-test'  # 占位符

from vidgraphrag._storage import NetworkXStorage, NanoVectorDBStorage
from vidgraphrag._utils import wrap_embedding_func_with_attrs, limit_async_func_call
import asyncio

async def test_query_crash():
    """测试查询流程"""
    working_dir = 'storage/chat-chat-1776138859434'
    
    print("=" * 60)
    print("测试知识图谱加载...")
    print("=" * 60)
    
    # 1. 加载知识图谱
    graph_storage = NetworkXStorage(
        namespace='chunk_entity_relation',
        global_config={'working_dir': working_dir}
    )
    
    print(f" 图加载成功: {graph_storage._graph.number_of_nodes()} 节点, {graph_storage._graph.number_of_edges()} 边")
    print(f"  示例节点: {list(graph_storage._graph.nodes())[:3]}")
    
    # 2. 模拟实体检索结果(带引号)
    print("\n" + "=" * 60)
    print("测试实体查询(带引号)...")
    print("=" * 60)
    
    mock_entity_results = [
        {"entity_name": '"THE FAMILY"'},
        {"entity_name": '"CHRISTMAS TREE"'},
    ]
    
    for entity in mock_entity_results:
        entity_name = entity["entity_name"]
        node = await graph_storage.get_node(entity_name)
        degree = await graph_storage.node_degree(entity_name)
        
        print(f"  实体: {entity_name}")
        print(f"    - 节点存在: {node is not None}")
        print(f"    - 度数: {degree}")
        if node:
            print(f"    - 描述: {node.get('description', '')[:50]}...")
    
    # 3. 测试不存在的实体
    print("\n" + "=" * 60)
    print("测试不存在的实体(不带引号)...")
    print("=" * 60)
    
    wrong_entity = "THE FAMILY"  # 不带引号
    node = await graph_storage.get_node(wrong_entity)
    print(f"  实体: {wrong_entity}")
    print(f"    - 节点存在: {node is not None}")
    print(f"    -  这会导致后续代码崩溃!")
    
    # 4. 检查 _find_most_related_segments_from_entities 可能的问题
    print("\n" + "=" * 60)
    print("测试节点数据访问...")
    print("=" * 60)
    
    node_datas = []
    for entity in mock_entity_results:
        node = await graph_storage.get_node(entity["entity_name"])
        if node:
            node_data = {
                **node,
                "entity_name": entity["entity_name"],
                "rank": await graph_storage.node_degree(entity["entity_name"])
            }
            node_datas.append(node_data)
    
    print(f"  有效节点数据: {len(node_datas)}")
    
    # 测试访问 source_id
    for i, nd in enumerate(node_datas):
        source_id = nd.get('source_id', '')
        print(f"  节点 {i+1}: source_id = {source_id}")
        
        # 这里可能崩溃:如果 source_id 不存在或格式不对
        if not source_id:
            print(f"     source_id 为空!")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    asyncio.run(test_query_crash())
