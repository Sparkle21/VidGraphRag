#!/usr/bin/env python3
import json
import networkx as nx

storage_dir = "storage/chat-chat-1776138859434"

print("=" * 60)
print(" 数据存储概览")
print("=" * 60)

# 1. 视频片段
with open(f"{storage_dir}/kv_store_video_segments.json", 'r') as f:
    segments = json.load(f)
print(f"\n 视频片段数: {len(segments)}")
for seg_id, seg in list(segments.items())[:2]:
    print(f"   - {seg_id}: shots={len(seg.get('shots', []))}, duration={seg.get('duration_seconds', 0):.1f}s")

# 2. Text Chunks
with open(f"{storage_dir}/kv_store_text_chunks.json", 'r') as f:
    chunks = json.load(f)
print(f"\n Text Chunks: {len(chunks)}")
for chunk_id, chunk in list(chunks.items())[:2]:
    content = chunk.get('content', '')[:100]
    print(f"   - {chunk_id}: {content}...")

# 3. Entities
with open(f"{storage_dir}/vdb_entities.json", 'r') as f:
    entities_data = json.load(f)
entities = entities_data.get('data', [])
print(f"\n 实体数量: {len(entities)}")
for entity in entities[:3]:
    print(f"   - {entity['entity_name']}: type={entity.get('entity_type')}")

# 4. Graph
graph_path = f"{storage_dir}/graph_chunk_entity_relation.graphml"
G = nx.read_graphml(graph_path)
print(f"\n  知识图谱:")
print(f"   节点数: {G.number_of_nodes()}")
print(f"   边数: {G.number_of_edges()}")
print(f"   节点类型分布:")
node_types = {}
for node, data in G.nodes(data=True):
    etype = data.get('entity_type', 'unknown')
    node_types[etype] = node_types.get(etype, 0) + 1
for etype, count in sorted(node_types.items()):
    print(f"     - {etype}: {count}")

# 5. Video Features
with open(f"{storage_dir}/vdb_video_segment_feature.json", 'r') as f:
    features_data = json.load(f)
features = features_data.get('data', [])
print(f"\n 视频片段特征向量: {len(features)}")
if features:
    print(f"   示例维度: {len(features[0].get('vector', []))}")

print("\n" + "=" * 60)
print(" 数据完整性检查完成")
print("=" * 60)
