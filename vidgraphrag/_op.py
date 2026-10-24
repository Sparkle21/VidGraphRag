import re
import json
import openai
import asyncio
import tiktoken
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from ._videoutil import (
    retrieved_segment_caption,
)

# 尝试导入 transformers（方案1：轻量级分类模型）
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, falling back to LLM-based classification")
# 这是 VideoRAG 的核心查询处理模块，实现基于知识图谱和视频检索的智能问答：
# 主要功能：
# 文本分块 (chunking_*): 三种策略切分视频内容（按 token 数、视频片段、分隔符）
# 实体抽取 (extract_entities): 从文本块中提取实体和关系，构建知识图谱
# 混合检索 (videorag_query):
#   语义检索（向量数据库）
#   图谱检索（实体关联）
#   视觉检索（ImageBind 特征匹配）
# 动态 Caption 生成: 对检索到的视频片段按需生成详细描述
# 多轮优化: 支持多次提炼查询和结果过滤

def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    """
        按 token 数量对文本进行分块处理
        参数：
            tokens_list: token 列表的列表，每个元素是一个文档的 token 序列
            doc_keys: 文档键列表，与 tokens_list 一一对应
            tiktoken_model: tiktoken 编码模型实例
            overlap_token_size: 重叠部分的 token 数量，默认 128
            max_token_size: 每个块的最大 token 数量，默认 1024

        返回：
            results: 分块结果列表，每个元素包含 tokens( token 数)、content(内容)、
                    chunk_order_index(块索引)、full_doc_id(文档 ID)
        """
    results = []
    for index, tokens in enumerate(tokens_list): # enumerate是同时获取列表的索引和元素值。
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        # decode_batch()方法将多个token序列解码为字符串列表。
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_video_segments(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    max_token_size=1024,
):
    """
       按视频片段进行分块处理，确保每个片段不超过最大 token 数

       参数：
           tokens_list: token 列表的列表，每个元素是一个视频片段的 token 序列
           doc_keys: 片段键列表，格式为"video_name_index"
           tiktoken_model: tiktoken 编码模型实例
           max_token_size: 每个块的最大 token 数量，默认 1024

       返回：
           results: 分块结果列表，每个元素包含 tokens( token 数)、content(内容)、
                   chunk_order_index(块索引)、video_segment_id(所属视频片段 ID 列表)
    """
    # make sure each segment is not larger than max_token_size
    for index in range(len(tokens_list)):
        if len(tokens_list[index]) > max_token_size:
            tokens_list[index] = tokens_list[index][:max_token_size]
    
    results = []
    chunk_token = []
    chunk_segment_ids = []
    chunk_order_index = 0
    for index, tokens in enumerate(tokens_list):
        
        if len(chunk_token) + len(tokens) <= max_token_size:
            # add new segment
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
        else:
            # save the current chunk
            chunk = tiktoken_model.decode(chunk_token)
            # decode_batch()方法将多个token序列解码为字符串列表。
            results.append(
                {
                    "tokens": len(chunk_token),
                    "content": chunk.strip(), # strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                    "chunk_order_index": chunk_order_index,
                    "video_segment_id": chunk_segment_ids,
                }
            )
            # new chunk with current segment as begin
            chunk_token = []
            chunk_segment_ids = []
            chunk_token += tokens.copy() # copy()方法用于复制列表。
            chunk_segment_ids.append(doc_keys[index])
            chunk_order_index += 1
    
    # save the last chunk
    if len(chunk_token) > 0:
        chunk = tiktoken_model.decode(chunk_token)
        results.append(
            {
                "tokens": len(chunk_token),
                "content": chunk.strip(),
                "chunk_order_index": chunk_order_index,
                "video_segment_id": chunk_segment_ids,
            }
        )
    
    return results
    
    
def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    """
       使用预定义分隔符对文本进行智能分块

       参数：
           tokens_list: token 列表的列表，每个元素是一个文档的 token 序列
           doc_keys: 文档键列表，与 tokens_list 一一对应
           tiktoken_model: tiktoken 编码模型实例
           overlap_token_size: 重叠部分的 token 数量，默认 128
           max_token_size: 每个块的最大 token 数量，默认 1024

       返回：
           results: 分块结果列表，每个元素包含 tokens( token 数)、content(内容)、
                   chunk_order_index(块索引)、full_doc_id(文档 ID)
       """
    splitter = SeparatorSplitter(
        # 使用预定义的分隔符智能切分
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_videos, chunk_func=chunking_by_video_segments, **chunk_func_params):
    """
        获取新视频的分块结果

        参数：
            new_videos: 新视频字典，结构为 {video_name: {segment_id: {content: ...}}}
            chunk_func: 分块函数，默认为 chunking_by_video_segments
            **chunk_func_params: 分块函数的额外参数

        返回：
            inserting_chunks: 分块结果字典，键为 MD5 哈希 ID，值为分块数据
    """
    inserting_chunks = {}

    new_videos_list = list(new_videos.keys())
    for video_name in new_videos_list:
        segment_id_list = list(new_videos[video_name].keys())
        docs = [new_videos[video_name][index]["content"] for index in segment_id_list]
        doc_keys = [f'{video_name}_{index}' for index in segment_id_list]

        ENCODER = tiktoken.encoding_for_model("gpt-4o")
        tokens = ENCODER.encode_batch(docs, num_threads=16)
        # 获取分块结果 chunking_by_video_segments
        chunks = chunk_func(
            tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
        )

        for chunk in chunks:
            inserting_chunks.update(
                {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
            ) # compute_mdhash_id() 函数用于计算 MD5 哈希值。

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """
        使用 LLM 对实体或关系的描述进行摘要

        参数：
            entity_or_relation_name: 实体或关系名称
            description: 需要摘要的描述文本
            global_config: 全局配置字典

        返回：
            摘要后的描述文本，如果原文本长度小于阈值则返回原文本
    """
    # 类型注解：声明 use_llm_func 是一个可调用函数
    use_llm_func: callable = global_config["llm"]["cheap_model_func"] # 模型函数
    llm_max_tokens = global_config["llm"]["cheap_model_max_token_size"] # 模型最大 token 数
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"] # 摘要最大 token 数

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    # 编码字符串
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    ) # 解码 token
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base) # 生成提示
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens) # 使用 LLM 进行摘要
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    chunk_meta: dict = None,
):
    """
        处理单个实体抽取记录

        参数：
            record_attributes: 记录属性列表，格式为 ['"entity"', 名称，类型，描述，...]
            chunk_key: 来源块的 ID
            chunk_meta: chunk 的元数据（video_segment_id, time_start, time_end）

        返回：
            实体信息字典，包含 entity_name、entity_type、description、source_id、
            segment_ids、time_ranges；如果格式不正确则返回 None
    """
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1])
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    result = dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )
    if chunk_meta:
        result["segment_ids"] = chunk_meta.get("video_segment_id", [])
        t_start = chunk_meta.get("time_start", 0)
        t_end = chunk_meta.get("time_end", 0)
        result["time_ranges"] = [[t_start, t_end]] if t_start or t_end else []
    return result


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
        处理单个关系抽取记录

        参数：
            record_attributes: 记录属性列表，格式为 ['"relationship"', 源实体，目标实体，描述，权重]
            chunk_key: 来源块的 ID

        返回：
            关系信息字典，包含 src_id、tgt_id、weight、description、source_id；
            如果格式不正确则返回 None
    """
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1])
    target = clean_str(record_attributes[2])
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
        合并多个节点数据并更新到知识图谱

        参数：
            entity_name: 实体名称
            nodes_data: 节点数据列表，每个元素包含 entity_type、description、source_id
            knowledge_graph_inst: 知识图谱存储实例
            global_config: 全局配置字典

        返回：
            合并后的节点数据字典，包含 entity_name、entity_type、description、source_id、
            segment_ids、time_ranges、first_appearance、last_appearance
    """
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    already_segment_ids = set()
    already_time_ranges = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        if "segment_ids" in already_node:
            try:
                already_segment_ids.update(json.loads(already_node["segment_ids"]))
            except (json.JSONDecodeError, TypeError):
                pass
        if "time_ranges" in already_node:
            try:
                already_time_ranges.extend(json.loads(already_node["time_ranges"]))
            except (json.JSONDecodeError, TypeError):
                pass

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )

    # 合并时间元数据
    all_segment_ids = set(already_segment_ids)
    all_time_ranges = list(already_time_ranges)
    for dp in nodes_data:
        for sid in dp.get("segment_ids", []):
            all_segment_ids.add(sid)
        for tr in dp.get("time_ranges", []):
            if tr not in all_time_ranges:
                all_time_ranges.append(tr)
    all_time_ranges.sort(key=lambda x: x[0])

    first_appearance = min((tr[0] for tr in all_time_ranges), default=0)
    last_appearance = max((tr[1] for tr in all_time_ranges), default=0)

    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        segment_ids=json.dumps(sorted(all_segment_ids)),
        time_ranges=json.dumps(all_time_ranges),
        first_appearance=first_appearance,
        last_appearance=last_appearance,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
        合并多个边数据并更新到知识图谱

        参数：
            src_id: 源实体 ID
            tgt_id: 目标实体 ID
            edges_data: 边数据列表，每个元素包含 weight、description、source_id、order
            knowledge_graph_inst: 知识图谱存储实例
            global_config: 全局配置字典

        返回：
            合并后的边数据字典，包含 src_tgt、description、weight
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            # 使用多个分隔符分割字符串
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )
    return_edge_data = dict(
        src_tgt=(src_id, tgt_id),
        description=description,
        weight=weight
    )
    return return_edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
       从文本块中抽取实体和关系，构建知识图谱

       参数：
           chunks: 文本块字典，键为块 ID，值为包含 content 等字段的 schema
           knowledge_graph_inst: 知识图谱存储实例
           entity_vdb: 实体向量数据库实例
           global_config: 全局配置字典

       返回：
           三元组 (knowledge_graph_inst, all_entities_data, all_edges_data)，
           如果没有抽取出任何实体则返回 None
    """
    use_llm_func: callable = global_config["llm"]["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    
    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """
                处理单个文本块，抽取其中的实体和关系

                参数:
                    chunk_key_dp: 元组，包含 (chunk_id, chunk 数据)

                返回:
                    dict(maybe_nodes): 实体字典 {实体名：[实体数据列表]}
                    dict(maybe_edges): 关系字典 {(源实体，目标实体): [关系数据列表]}
        """
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # 初始抽取
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt) # 调用LLM抽取实体

        # 多轮补充抽取（gleaning）
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            # 请求 LLM 继续抽取更多实体/关系
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            # 更新对话历史
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result

            # 如果达到最大轮次，强制退出
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            # 询问 LLM 是否还有遗漏
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        # 解析 LLM 返回的结果
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            # 尝试解析成实体
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, chunk_meta=chunk_dp
            )
            if if_entities is not None:
                # 成功解析为实体，添加到节点字典
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            # 如果不是实体，尝试解析成关系
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                # 成功解析为关系，添加到边字典
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        # 更新进度统计
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    ) # 同时处理所有的文本块（并发执行），等所有块都处理完后，收集所有结果。
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    ) # 批量插入实体数据
    all_edges_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    ) # 批量插入关系数据
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        } # 批量插入实体数据到 VDB
        await entity_vdb.upsert(data_for_vdb)
    return knowledge_graph_inst, all_entities_data, all_edges_data


async def _find_most_related_segments_from_entities(
    topk_chunks: int,
    node_datas: list[dict],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    """
        从实体节点找到最相关的视频片段

        参数：
            topk_chunks: 返回的块数量上限
            node_datas: 节点数据列表，每个元素包含 source_id 等信息
            text_chunks_db: 文本块存储实例
            knowledge_graph_inst: 知识图谱存储实例

        返回：
            chunk_related_segments: 相关视频片段 ID 集合
        """
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ] # 获取节点的文本单元
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    ) # 获取节点的边
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    ) # 获取一跳邻居节点数据
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    } # 创建一跳邻居节点的文本单元集合
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        # 创建节点的文本单元集合
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            # 遍历该节点的边
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup # 邻居节点在文本单元集合中
                    and c_id in all_one_hop_text_units_lookup[e[1]] # 当前节点在邻居节点的文本单元集合中
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            } # 创建节点的文本单元数据
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ] # 创建节点的文本单元数据
    sorted_text_units = sorted(
        all_text_units, key=lambda x: -x["relation_counts"]
    )[:topk_chunks] # 按关系数量排序，取前 topk_chunks 个
    # 提取相关视频片段 ID
    chunk_related_segments = set()
    for _chunk_data in sorted_text_units:
        for s_id in _chunk_data['data']['video_segment_id']:
            chunk_related_segments.add(s_id)
    # 返回相关视频片段 ID 集合
    return chunk_related_segments

async def _refine_entity_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    """
        使用 LLM 优化用于实体检索的查询

        参数：
            query: 原始查询文本
            query_param: 查询参数对象
            global_config: 全局配置字典

        返回：
            优化后的查询文本
        """
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite_for_entity_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    return final_result

async def _refine_visual_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    """
        使用 LLM 优化用于视觉检索的查询

        参数：
            query: 原始查询文本
            query_param: 查询参数对象
            global_config: 全局配置字典

        返回：
            优化后的查询文本
        """
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    return final_result

async def _extract_keywords_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    """
        使用 LLM 从查询中提取关键词

        参数：
            query: 原始查询文本
            query_param: 查询参数对象
            global_config: 全局配置字典

        返回：
            提取的关键词文本
        """
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    keywords_prompt = PROMPTS["keywords_extraction"]
    keywords_prompt = keywords_prompt.format(input_text=query)
    final_result = await use_llm_func(keywords_prompt)
    return final_result

# todo core query
async def videorag_query(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    caption_model,
    caption_tokenizer,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
        VideoRAG 核心查询函数，实现基于知识图谱和视频检索的智能问答

        参数：
            query: 用户查询文本
            entities_vdb: 实体向量数据库实例
            text_chunks_db: 文本块数据库实例
            chunks_vdb: 块向量数据库实例
            video_path_db: 视频路径数据库实例
            video_segments: 视频片段数据实例
            video_segment_feature_vdb: 视频片段特征向量数据库实例
            knowledge_graph_inst: 知识图谱存储实例
            caption_model: 视频 caption 生成模型
            caption_tokenizer: caption 分词器
            query_param: 查询参数对象
            global_config: 全局配置字典

        返回：
            生成的回答文本
        """
    # 用户提问
    #     ↓
    # 【阶段 1】朴素文本检索 → 获取相关文本块
    #     ↓
    # 【阶段 2】图谱检索 → 通过实体找到相关视频片段
    #     ↓
    # 【阶段 3】视觉检索 → 通过特征匹配找到相关视频片段
    #     ↓
    # 【阶段 4】相关性过滤 → 筛选最相关的片段
    #     ↓
    # 【阶段 5】精细 Caption 生成 → 生成详细描述
    #     ↓
    # 【阶段 6】整合回答 → LLM 生成最终答案
    #     ↓
    # 返回给用户
    use_model_func = global_config["llm"]["best_model_func"] # 初始化配置 获取模型调用函数
    query = query
    
    # 朴素文本块检索（Naive Retrieval）
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
    retreived_chunk_context = section
    
    # 图谱检索（Entity-based Retrieval）
    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query,
        query_param,
        global_config,
    )
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)
    entity_retrieved_segments = set()
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        ) # 获取节点数据
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        ) # 获取节点度数
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(entity_results, node_datas, node_degrees)
            if n is not None
        ]
        entity_retrieved_segments = entity_retrieved_segments.union(await _find_most_related_segments_from_entities(
            # 通过实体找到相关视频片段
            global_config["retrieval_topk_chunks"], node_datas, text_chunks_db, knowledge_graph_inst
        ))
    
    # 视觉检索（Visual Retrieval）
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query,
        query_param,
        global_config,
    ) # 获取视频片段特征向量 优化视觉查询
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    # 查询视频片段特征向量库
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])# 收集所有匹配的视频片段 ID
    
    # 合并和排序检索结果
    retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    ) # 排序：按视频名 + 片段索引
    print(query_for_entity_retrieval)
    print(f"Retrieved Text Segments {entity_retrieved_segments}")
    print(query_for_visual_retrieval)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    
    already_processed = 0

    # 相关性过滤（Relevance Filtering）
    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        """
            判断这个视频片段是否与查询相关

            示例 Prompt:
            ---
            请判断以下视频片段是否与问题相关:

            问题：特斯拉的自动驾驶技术怎么样？
            片段描述：一辆银色特斯拉在高速公路上行驶

            如果相关，回答"yes"；否则回答"no"。
            ---
        """
        nonlocal use_model_func, already_processed
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = PROMPTS["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Checked {already_processed} segments\r",
            end="",
            flush=True,
        )
        return (segment_key, result)

    # 准备粗略描述
    rough_captions = {}
    for s_id in retrieved_segments:
        video_name = '_'.join(s_id.split('_')[:-1]) # 视频名称
        index = s_id.split('_')[-1] # 片段索引
        # 从视频片段数据中获取粗略描述
        rough_captions[s_id] = video_segments._data[video_name][index]["content"]
    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    ) # 相关性过滤
    # 从结果中获取所有相关片段 筛选出相关的片段
    remain_segments = [x[0] for x in results if 'yes' in x[1].lower()]
    print(f"{len(remain_segments)} Video Segments remain after filtering")
    if len(remain_segments) == 0:
        # # 如果全部被过滤掉，就回退到使用所有检索到的片段
        print("Since no segments remain after filtering, we utilized all the retrieved segments.")
        remain_segments = retrieved_segments
    print(f"Remain segments {remain_segments}")
    
    # 生成精细 Caption visual retrieval
    keywords_for_caption = await _extract_keywords_query(
        query,
        query_param,
        global_config,
    ) # 获取关键词
    print(f"Keywords: {keywords_for_caption}")
    # 为每个片段生成详细的 caption
    caption_results = retrieved_segment_caption(
        caption_model,
        caption_tokenizer,
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment']
    )

    # 构建 CSV 格式的上下文 data table
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1]) # 视频名称
        index = s_id.split('_')[-1] # 片段索引
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0]) # 开始时间
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1]) # 结束时间
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}" # 格式化时间
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}" # 格式化时间
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]]) # 添加到数据表
    text_units_context = list_of_list_to_csv(text_units_section_list) # 转换为 CSV

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    if query_param.wo_reference:
        sys_prompt_temp = PROMPTS["videorag_response_wo_reference"]
    else:
        sys_prompt_temp = PROMPTS["videorag_response"]
        
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

# todo core multi-query
async def videorag_query_multiple_choice(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    caption_model,
    caption_tokenizer,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    VideoRAG 多项选择题查询函数

    参数：
        query: 用户查询文本
        entities_vdb: 实体向量数据库实例
        text_chunks_db: 文本块数据库实例
        chunks_vdb: 块向量数据库实例
        video_path_db: 视频路径数据库实例
        video_segments: 视频片段数据实例
        video_segment_feature_vdb: 视频片段特征向量数据库实例
        knowledge_graph_inst: 知识图谱存储实例
        caption_model: 视频 caption 生成模型
        caption_tokenizer: caption 分词器
        query_param: 查询参数对象
        global_config: 全局配置字典

    返回：
        JSON 格式的回答，包含"Answer"和"Explanation"字段
    """
    use_model_func = global_config["llm"]["best_model_func"]
    query = query
    
    # naive chunks
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    # NOTE: I update here, not len results can also process
    if len(results):
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.naive_max_token_for_text_unit,
        ) # 根据 token 数量截断列表
        logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
        section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
        # 生成新的 chunk
        retreived_chunk_context = section
    else:
        retreived_chunk_context = "No Content"
        
    # visual retrieval
    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query,
        query_param,
        global_config,
    ) # 获取实体检索查询
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)
    entity_retrieved_segments = set()
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        ) # 获取实体数据
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        ) # 获取实体度量
        # 实体数据 (node_datas) = 实体的详细信息；实体度量 (node_degrees) = 实体的重要性评分
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(entity_results, node_datas, node_degrees)
            if n is not None
        ] # 添加 rank
        entity_retrieved_segments = entity_retrieved_segments.union(await _find_most_related_segments_from_entities(
            global_config["retrieval_topk_chunks"], node_datas, text_chunks_db, knowledge_graph_inst
        ))
    
    # visual retrieval
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query,
        query_param,
        global_config,
    ) # 获取视频检索查询
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])
    
    # caption
    retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    ) # 排序
    print(query_for_entity_retrieval)
    print(f"Retrieved Text Segments {entity_retrieved_segments}")
    print(query_for_visual_retrieval)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    
    already_processed = 0
    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        # 用 LLM 判断单个视频片段是否与查询问题相关
        # nonlocal 的作用：允许内部函数修改外部函数的变量
        nonlocal use_model_func, already_processed
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1] # 从元组中提取片段 ID 和内容。
        filter_prompt = PROMPTS["filtering_segment"] # 获取过滤提示模板
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        # 使用片段内容生成过滤提示
        result = await use_model_func(filter_prompt) # 使用模型进行判断
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ] # 获取进度提示
        print(
            f"{now_ticks} Checked {already_processed} segments\r",
            end="",
            flush=True,
        )
        return (segment_key, result)
    
    rough_captions = {}
    for s_id in retrieved_segments:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        rough_captions[s_id] = video_segments._data[video_name][index]["content"]
    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    ) # 并行处理
    remain_segments = [x[0] for x in results if 'yes' in x[1].lower()]
    print(f"{len(remain_segments)} Video Segments remain after filtering")
    if len(remain_segments) == 0:
        print("Since no segments remain after filtering, we utilized all the retrieved segments.")
        remain_segments = retrieved_segments
    print(f"Remain segments {remain_segments}")
    
    # visual retrieval
    keywords_for_caption = await _extract_keywords_query(
        query,
        query_param,
        global_config,
    ) # 使用 LLM 从查询中提取关键词
    print(f"Keywords: {keywords_for_caption}")
    caption_results = retrieved_segment_caption(
        caption_model,
        caption_tokenizer,
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment']
    ) # 针对检索到的视频片段，根据特定查询生成详细的视觉描述和信息抽取

    ## data table
    # 构建视频检索结果的 CSV 表格，包含视频名称、起止时间和详细内容描述
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0])
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1])
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
    # 将二维列表转换为 CSV 格式的字符串。
    text_units_context = list_of_list_to_csv(text_units_section_list)

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    # NOTE: I update here to use a different prompt
    # 构建视频检索上下文，使用 Markdown 代码块格式包裹 CSV 数据
    sys_prompt_temp = PROMPTS["videorag_response_for_multiple_choice_question"]
        
    # 填充提示模板，整合视频内容、文本资料和回答要求
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        use_cache=False,
    )
    while True:
        try:
            json_response = json.loads(response)
            assert "Answer" in json_response and "Explanation" in json_response
            return json_response
        except Exception as e:
            logger.info(f"Response is not valid JSON for query {query}. Found {e}. Retrying...")
            response = await use_model_func(
                query,
                system_prompt=sys_prompt,
                use_cache=False,
            )


# ============================================================================
# Task 3.1: 查询路由分析器 - Shot-Segment 双层融合查询
# ============================================================================

# ==================== 查询意图识别模块 ====================

# 全局变量：缓存分类器实例
_query_classifier = None

def _get_query_classifier():
    """
    获取查询分类器实例（单例模式）
    使用中文文本分类模型进行意图识别
    """
    global _query_classifier
    
    if _query_classifier is None and TRANSFORMERS_AVAILABLE:
        try:
            logger.info("Loading query intent classifier...")
            # 使用轻量级的中文分类模型
            # 注意：这是一个通用模型，实际生产环境应该使用在视频查询数据上微调的模型
            _query_classifier = pipeline(
                "text-classification",
                model="uer/roberta-base-finetuned-jd-binary-chinese",
                top_k=3  # 返回 top-3 类别
            )
            logger.info("Query classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}, will use LLM-based method")
            _query_classifier = False  # 标记为加载失败
    
    return _query_classifier


async def _analyze_query_type_with_llm(query: str, llm_config: dict) -> str:
    """
    方案2：使用 LLM 进行 Few-shot 意图识别
    
    Args:
        query: 用户查询
        llm_config: LLM 配置，包含 cheap_model_func
    
    Returns:
        'precise', 'reasoning', 或 'hybrid'
    """
    llm_func = llm_config.get('cheap_model_func')
    if not llm_func:
        logger.warning("LLM function not available, using rule-based fallback")
        return _analyze_query_type_rules(query)
    
    prompt = f"""你是一个查询意图分析器。请分析以下查询的类型：

查询类型定义：
- precise: 需要定位具体镜头、画面、时间点的查询（如“找XX的镜头”、“第几分钟有XX”）
- reasoning: 需要理解、分析、推理的查询（如“XX和XX什么关系”、“为什么XX”）
- hybrid: 同时需要定位和分析的查询（如“XX是怎么做的？具体在哪些镜头？”）

示例：
- “找钢铁侠的镜头” → precise
- “钢铁侠和蜘蛛侠什么关系？” → reasoning
- “钢铁侠怎么教蜘蛛侠的？具体镜头在哪？” → hybrid
- “视频里有哪些战斗场景？” → precise
- “这段视频的主题是什么？” → reasoning
- “详细分析一下他们的对话内容” → hybrid

请分析以下查询（只返回类型名称，不要其他内容）：
查询：“{query}”
类型："""
    
    try:
        response = await llm_func(prompt)
        response_lower = response.strip().lower()
        
        # 解析响应
        if 'precise' in response_lower:
            return 'precise'
        elif 'reasoning' in response_lower:
            return 'reasoning'
        else:
            return 'hybrid'
    except Exception as e:
        logger.warning(f"LLM-based classification failed: {e}, using rule-based fallback")
        return _analyze_query_type_rules(query)


def _analyze_query_type_rules(query: str) -> str:
    """
    基于规则的意图识别（降级方案）
    
    Args:
        query: 用户查询
    
    Returns:
        'precise', 'reasoning', 或 'hybrid'
    """
    query_lower = query.lower()
    
    # 规则匹配
    precise_keywords = [
        '镜头', '画面', '场景', '第几分钟', '什么时候', '哪里',
        '找', '定位', '截取', '片段', '时间点', '出现'
    ]
    reasoning_keywords = [
        '关系', '为什么', '怎么', '如何', '原因', '意义', '主题',
        '特点', '分析', '理解', '评价', '比较', '对比', '总结',
        '概括', '解释', '说明'
    ]
    hybrid_keywords = ['具体', '哪些', '举例', '详细', '并且', '以及', '还有']
    
    has_precise = any(kw in query_lower for kw in precise_keywords)
    has_reasoning = any(kw in query_lower for kw in reasoning_keywords)
    has_hybrid = any(kw in query_lower for kw in hybrid_keywords)
    
    # 决策逻辑
    if has_precise and has_reasoning:
        return 'hybrid'
    elif has_reasoning:
        return 'reasoning'
    elif has_precise or has_hybrid:
        return 'precise'
    else:
        return 'hybrid'  # 默认混合查询


def analyze_query_type(query: str, llm_config: dict = None, use_llm_for_classification: bool = False) -> str:
    """
    查询意图分析（主入口）
    
    优先级：
    1. 方案1：轻量级分类模型（transformers）- 如果可用
    2. 方案2：LLM Few-shot 分类 - 需要显式启用 (use_llm_for_classification=True)
    3. 默认：基于规则的关键词匹配（最可靠、最快）
    
    返回类型：
    - 'precise': 精确检索（走 Shot 层）
      特征：需要定位具体镜头、画面、时间点
      示例：“找所有黄昏的场景”
    
    - 'reasoning': 深度推理（走 Segment+ 图谱）
      特征：需要理解、分析、推理
      示例：“钢铁侠和蜘蛛侠什么关系”
    
    - 'hybrid': 混合查询（双路协作）
      特征：同时需要定位和理解
      示例：“钢铁侠是怎么教蜘蛛侠的？具体镜头”
    
    Args:
        query: 用户查询字符串
        llm_config: LLM 配置（可选）
        use_llm_for_classification: 是否使用 LLM 进行分类（默认 False，使用规则）
    
    Returns:
        查询类型：'precise', 'reasoning', 或 'hybrid'
    """
    # 方案1：尝试使用轻量级分类模型（如果可用且已加载）
    if TRANSFORMERS_AVAILABLE:
        classifier = _get_query_classifier()
        if classifier and classifier is not False:
            try:
                result = classifier(query)
                logger.debug(f"Classifier result: {result}")
                
                # 由于通用模型的标签与我们的不匹配，暂时使用规则
                # TODO: 训练专门的视频查询分类模型后替换这里
                logger.info("Using rule-based method (model available but not trained for video queries)")
                return _analyze_query_type_rules(query)
                
            except Exception as e:
                logger.warning(f"Model-based classification failed: {e}")
    
    # 方案2：使用 LLM Few-shot 分类（仅在显式启用时）
    if use_llm_for_classification and llm_config:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在事件循环中，创建 task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        lambda: asyncio.run(_analyze_query_type_with_llm(query, llm_config))
                    )
                    return future.result(timeout=5)
            else:
                return loop.run_until_complete(_analyze_query_type_with_llm(query, llm_config))
        except Exception as e:
            logger.warning(f"LLM-based classification failed: {e}")
    
    # 方案3：默认使用规则匹配（最可靠、最快）
    return _analyze_query_type_rules(query)


async def shot_level_multi_channel_search(
    query: str,
    shot_storage,
    shot_vdb,
    top_k: int = 5,
    score_threshold: float = 0.01,
    embedding_func=None,
    visual_embedder=None,
    shot_bm25=None,
    reranker=None,
) -> list:
    """
    Shot 级多路召回
    
    流程：
    1. 编码查询为三路向量
    2. 多路召回
    3. 结果融合
    4. 获取完整 Shot 信息
    
    参数：
        query: 用户查询文本
        shot_storage: ShotKVStorage 实例
        shot_vdb: MultiModalVectorDB 实例
        top_k: 返回的 Shot 数量
        embedding_func: 文本嵌入函数（已包装的 EmbeddingFunc，签名为 async (texts: list[str])）
        visual_embedder: ImageBind 模型实例（可选，None 时视觉通路降级为 None）
    
    返回：
        shot_details: Shot 详细信息列表，每个元素包含 shot_id、retrieval_score 等
    """
    from ._videoutil.feature import encode_string_query
    import numpy as np
    
    # Step 1: 编码查询
    query_vectors = {}

    # 先算一份 ImageBind 文本向量：vision 与 audio 同处 ImageBind 1024 共享空间，
    # 同一文本查询向量可同时驱动两条跨模态通路（视觉检索 + 声学检索）。
    # 鸭子类型兼容：
    #   1) ImageBindModel 实例（本进程加载的原生模型） → 调 feature.encode_string_query
    #   2) HTTP 客户端（子进程中走服务） → 调 visual_embedder.encode_string_query(query)
    ib_text_vec = None
    if visual_embedder is not None:
        try:
            if hasattr(visual_embedder, 'encode_string_query') and not hasattr(visual_embedder, 'parameters'):
                # HTTP 客户端路径：返回 np.ndarray，shape=(1, dim) 或 (dim,)
                visual_emb = visual_embedder.encode_string_query(query)
                arr = np.asarray(visual_emb)
                ib_text_vec = arr[0] if arr.ndim == 2 else arr
            else:
                # 原生 ImageBind 模型路径，返回 torch.Tensor，shape=(1, dim)
                ib_text_vec = encode_string_query(query, visual_embedder).numpy()[0]
        except Exception as e:
            logger.warning(f"ImageBind text encoding failed: {e}")
            ib_text_vec = None
    else:
        logger.info("Visual/acoustic encoding skipped: no ImageBind embedder provided")

    # 视觉通路：ImageBind TEXT → visual_index（跨模态，1024）
    query_vectors['visual_vec'] = ib_text_vec

    # 声学通路：ImageBind TEXT → audio_index（与真声学向量同处 ImageBind 空间，跨模态对齐）
    #          检索的是"怎么发声"的意图；语音"内容"由下方 BM25(ASR 文本) 通道覆盖
    query_vectors['audio_vec'] = ib_text_vec

    # 语义通路：百炼 text embedding → vlm_index（1536 caption 空间），与上两路正交
    if embedding_func is not None:
        try:
            vlm_emb = await embedding_func([query])
            # embedding_func 返回 np.ndarray，shape=(1, dim)
            query_vectors['vlm_vec'] = vlm_emb[0]
        except Exception as e:
            logger.warning(f"VLM text encoding failed: {e}")
            query_vectors['vlm_vec'] = None
    else:
        logger.warning("VLM encoding skipped: no embedding_func provided")
        query_vectors['vlm_vec'] = None
    
    # Step 2: 动态 top_k — 短视频不应返回过多 shots
    actual_top_k = min(top_k, max(3, int(shot_vdb.element_count * 0.3)))

    # Step 3: 多路召回
    fused_results = await shot_vdb.multi_channel_search(
        query_vectors=query_vectors,
        top_k=actual_top_k * 3,
        fusion_strategy='rrf'
    )

    # Step 3.5: BM25 稀疏检索（第 4 通道）
    if shot_bm25 is not None:
        try:
            bm25_results = await shot_bm25.search(query, top_k=actual_top_k * 3)
            if bm25_results:
                k_constant = 60
                bm25_weight = 0.2  # 与 MultiModalVectorDB.CHANNEL_WEIGHTS['bm25'] 保持一致
                fused_dict = dict(fused_results)
                for rank, (shot_id, _) in enumerate(bm25_results, start=1):
                    fused_dict[shot_id] = fused_dict.get(shot_id, 0) + bm25_weight * 1.0 / (k_constant + rank)
                fused_results = sorted(fused_dict.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"BM25 channel: {len(bm25_results)} hits merged")
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")

    # Step 3.8: Cross-encoder Rerank
    if reranker is not None and fused_results:
        try:
            candidate_ids = [sid for sid, _ in fused_results[:actual_top_k * 3]]
            passages = []
            for sid in candidate_ids:
                shot_info = await shot_storage.get_by_id(sid)
                if shot_info:
                    asr = shot_info.get('content', {}).get('asr_text', '')
                    caption = shot_info.get('content', {}).get('vlm_shot_summary', {}).get('caption', '')
                    passages.append(f"{asr} {caption}".strip() or "empty")
                else:
                    passages.append("empty")
            rerank_results = await reranker(query, passages, top_k=actual_top_k)
            if rerank_results is not None:
                fused_results = [
                    (candidate_ids[idx], score) for idx, score in rerank_results
                ]
                logger.info(f"Rerank: {len(fused_results)} shots reranked")
        except Exception as e:
            logger.warning(f"Rerank failed, keeping original order: {e}")

    # Step 4: score threshold 过滤
    if score_threshold > 0:
        fused_results = [(sid, s) for sid, s in fused_results if s >= score_threshold]

    # Step 5: 获取完整信息
    shot_details = []
    for shot_id, score in fused_results:
        shot_info = await shot_storage.get_by_id(shot_id)
        if shot_info:
            shot_info['retrieval_score'] = score
            shot_details.append(shot_info)

    logger.info(f"Shot retrieval: {len(shot_details)} shots after filtering (total={shot_vdb.element_count}, top_k={actual_top_k})")
    return shot_details[:actual_top_k]


async def _kg_guided_shot_rerank(
    shot_results: list,
    graph_db,
    entities_vdb,
    query: str,
    top_k: int = 5,
) -> list:
    """用知识图谱实体对 shot 做重排序"""
    if not shot_results:
        return shot_results

    kg_keywords = set()
    try:
        ev_results = await entities_vdb.query(query, top_k=5)
        for r in ev_results:
            ent_name = r.get('entity_name', '')
            if ent_name:
                kg_keywords.add(ent_name.lower())
                node = await graph_db.get_node(ent_name)
                if node:
                    for word in ent_name.lower().split():
                        if len(word) > 2:
                            kg_keywords.add(word)
                edges = await graph_db.get_node_edges(ent_name)
                if edges:
                    for src, tgt in edges:
                        kg_keywords.add(src.lower())
                        kg_keywords.add(tgt.lower())
    except Exception as e:
        logger.warning(f"KG rerank entity retrieval failed: {e}")
        return shot_results[:top_k]

    if not kg_keywords:
        return shot_results[:top_k]

    for shot in shot_results:
        caption = shot.get('content', {}).get('vlm_shot_summary', {}).get('caption', '').lower()
        if not caption:
            continue
        matched = sum(1 for kw in kg_keywords if kw in caption)
        match_ratio = matched / len(kg_keywords) if kg_keywords else 0
        shot['retrieval_score'] = shot.get('retrieval_score', 0) + 0.5 * match_ratio

    shot_results.sort(key=lambda x: x.get('retrieval_score', 0), reverse=True)
    logger.info(f"KG rerank: {len(kg_keywords)} keywords, top score={shot_results[0].get('retrieval_score', 0):.4f}")
    return shot_results[:top_k]


async def segment_level_graph_reasoning(
    query: str,
    segment_storage,
    graph_db,
    llm_config: dict,
    text_chunks_db=None,
    entities_vdb=None,
    community_reports=None,
    top_k_entities: int = 10,
) -> dict:
    """
    Segment 级图谱推理
    
    流程：
    1. 实体提取（关键词 + 实体 VDB 向量召回合并）
    2. 图谱检索（segment_storage / fallback to text_chunks）
    3. 子图遵历 + 邻接边提取
    4. LLM 生成答案（entity description + relations 拼入 prompt）
    
    参数：
        query: 用户查询文本
        segment_storage: SegmentKVStorage 实例
        graph_db: NetworkXStorage 实例
        llm_config: LLM 配置字典
        text_chunks_db: 文本块库（可选、fallback 用）
        entities_vdb: 实体向量数据库（可选、启用后会走实体语义召回 + 邻接拼 prompt）
        top_k_entities: 实体召回 top_k
    
    返回：
        {
            'answer': 生成的答案文本,
            'evidence_segments': 证据 Segment ID 列表,
            'reasoning_paths': 推理路径（可选）
        }
    """
    # Step 1a: LLM 提取关键词（备用）
    try:
        entities_text = await _extract_keywords_query(
            query,
            QueryParam(),
            {'llm': llm_config}
        )
        # 简单解析实体（逗号分隔）
        entities = [e.strip() for e in entities_text.split(',') if e.strip()]
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        entities = []
    
    # Step 1b: 实体 VDB 向量召回（补充 entity 名称，以免关键词与图谱中实体名不匹配）
    retrieved_entity_names = []
    if entities_vdb is not None:
        try:
            ev_results = await entities_vdb.query(query, top_k=top_k_entities)
            retrieved_entity_names = [r['entity_name'] for r in ev_results if r.get('entity_name')]
        except Exception as e:
            logger.warning(f"entities_vdb query failed: {e}")
    
    # 合并去重（保留順序：先 VDB 召回后关键词）
    seen = set()
    merged_entities = []
    for name in retrieved_entity_names + entities:
        if name and name not in seen:
            seen.add(name)
            merged_entities.append(name)
    entities = merged_entities
    
    # Step 2: 图谱检索
    related_segments = []
    if hasattr(segment_storage, 'get_related_segments'):
        related_segments = await segment_storage.get_related_segments(entities)
    
    # Step 2.5: Fallback —— 只选与 query 实体相关的 chunks（不再全量 dump）
    fallback_used = False
    if not related_segments and text_chunks_db is not None:
        try:
            all_chunk_keys = await text_chunks_db.all_keys()
            chunk_records = await text_chunks_db.get_by_ids(all_chunk_keys)
            entity_keywords = set(e.lower() for e in entities if e)
            scored_chunks = []
            for ck_id, ck_data in zip(all_chunk_keys, chunk_records):
                if not ck_data:
                    continue
                content = ck_data.get('content', '')
                seg_ids = ck_data.get('video_segment_id', [])
                if not content:
                    continue
                hit_count = sum(1 for kw in entity_keywords if kw in content.lower())
                scored_chunks.append((hit_count, {
                    'segment_id': seg_ids[0] if seg_ids else ck_id,
                    'cognitive_content': {'narrative_text': content}
                }))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            related_segments = [c[1] for c in scored_chunks[:2]]
            fallback_used = True
            logger.warning(
                f"segment_level_graph_reasoning: SegmentKVStorage empty, "
                f"fallback to top-2 relevant chunks (of {len(scored_chunks)})"
            )
        except Exception as e:
            logger.warning(f"text_chunks fallback failed: {e}")
    
    # Step 3: 提取 entity description + 邻接边作为推理上下文
    entity_context_lines = []   # entity description 列表
    relation_context_lines = []  # entity-relation-entity 列表（P2a 文本化 subgraph）
    reasoning_paths = []
    if graph_db is not None and entities:
        seen_edges = set()
        for ent_name in entities:
            try:
                node = await graph_db.get_node(ent_name)
            except Exception:
                node = None
            if not node:
                continue
            # entity description
            desc = node.get('description') or node.get('entity_type') or ''
            if desc:
                temporal_info = ""
                seg_ids_str = node.get('segment_ids', '')
                time_ranges_str = node.get('time_ranges', '')
                if seg_ids_str:
                    try:
                        seg_ids = json.loads(seg_ids_str)
                        if seg_ids:
                            temporal_info += f"出现片段: {', '.join(seg_ids)}"
                    except (json.JSONDecodeError, TypeError):
                        pass
                if time_ranges_str:
                    try:
                        time_ranges = json.loads(time_ranges_str)
                        if time_ranges:
                            time_strs = [f"{tr[0]:.1f}s-{tr[1]:.1f}s" for tr in time_ranges]
                            temporal_info += (" | " if temporal_info else "") + f"时间: {', '.join(time_strs)}"
                    except (json.JSONDecodeError, TypeError):
                        pass
                if temporal_info:
                    entity_context_lines.append(f"- {ent_name}: {desc} [{temporal_info}]")
                else:
                    entity_context_lines.append(f"- {ent_name}: {desc}")
            # 1-hop: 邻接边→拼成 <e1>—[relation]—<e2>
            try:
                edges = await graph_db.get_node_edges(ent_name)
            except Exception:
                edges = None
            if not edges:
                continue
            hop1_neighbors = set()
            for src, tgt in edges:
                edge_key = tuple(sorted([src, tgt]))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                try:
                    edge_data = await graph_db.get_edge(src, tgt) or {}
                except Exception:
                    edge_data = {}
                rel_desc = edge_data.get('description') or edge_data.get('keywords') or 'related_to'
                relation_context_lines.append(f"- {src} —[{rel_desc}]— {tgt}")
                reasoning_paths.append({'source': src, 'target': tgt, 'relation': rel_desc})
                neighbor = tgt if src == ent_name else src
                hop1_neighbors.add(neighbor)

            # 2-hop: 从 1-hop 邻居再展开一层
            for neighbor in hop1_neighbors:
                try:
                    hop2_edges = await graph_db.get_node_edges(neighbor)
                except Exception:
                    hop2_edges = None
                if not hop2_edges:
                    continue
                for src2, tgt2 in hop2_edges:
                    edge_key2 = tuple(sorted([src2, tgt2]))
                    if edge_key2 in seen_edges:
                        continue
                    seen_edges.add(edge_key2)
                    try:
                        edge_data2 = await graph_db.get_edge(src2, tgt2) or {}
                    except Exception:
                        edge_data2 = {}
                    rel_desc2 = edge_data2.get('description') or edge_data2.get('keywords') or 'related_to'
                    relation_context_lines.append(f"- {src2} —[{rel_desc2}]— {tgt2}")
                    reasoning_paths.append({'source': src2, 'target': tgt2, 'relation': rel_desc2})

        logger.info(f"Graph traversal: {len(entity_context_lines)} entities, {len(reasoning_paths)} paths (1-hop + 2-hop)")

    # Step 4: 构建上下文并调用 LLM（带编号以支持引用）
    context_segments = []
    segment_ref_lines = []
    for idx, seg in enumerate(related_segments, 1):
        if 'cognitive_content' in seg and 'narrative_text' in seg['cognitive_content']:
            context_segments.append(f"[片段{idx}] {seg['cognitive_content']['narrative_text']}")
            seg_id = seg.get('segment_id', 'unknown')
            segment_ref_lines.append(f"[片段{idx}] {seg_id}")

    context_text = "\n\n".join(context_segments) if context_segments else "No relevant segments found."
    entity_block = "\n".join(entity_context_lines) if entity_context_lines else "无"
    # 限制送入 LLM 的关系数量，避免上下文过长导致回答冗长
    relation_context_lines = relation_context_lines[:10]
    reasoning_paths = reasoning_paths[:10]
    relation_block = "\n".join(relation_context_lines) if relation_context_lines else "无"

    # Step 4.5: 社区主题报告
    community_block = "无"
    if community_reports is not None:
        try:
            all_keys = await community_reports.all_keys()
            if all_keys:
                report_data = await community_reports.get_by_ids(all_keys)
                scored = []
                for rd in report_data:
                    if rd and rd.get('report'):
                        hit = sum(1 for ent in entities if ent.lower() in rd['report'].lower())
                        scored.append((hit, rd))
                scored.sort(key=lambda x: x[0], reverse=True)
                top_reports = [s[1]['report'] for s in scored[:3] if s[0] > 0]
                if top_reports:
                    community_block = "\n\n".join(top_reports)
                    logger.info(f"Added {len(top_reports)} community reports to reasoning context")
        except Exception as e:
            logger.warning(f"Community report retrieval failed: {e}")

    reasoning_prompt = f"""你是一个视频内容分析专家。请基于以下多源信息回答问题。

## 问题
{query}

## 信息源 1：视频片段叙事
{context_text}

## 信息源 2：知识图谱实体
{entity_block}

## 信息源 3：实体关系网络
{relation_block}

## 信息源 4：社区主题报告
{community_block}

## 回答要求
1. 优先使用知识图谱中的实体关系来构建回答框架
2. 用视频片段叙事补充具体细节
3. 如果图谱中有实体的时间信息，指出实体在视频中的出现位置
4. 如果社区主题报告提供了全局性的主题信息，整合到回答中
5. 明确区分"图谱中有明确记录的信息"和"基于上下文推断的信息"
6. 引用视频片段时使用 [片段N] 标注，例如「主角在这一幕中表达了愤怒[片段1]」

请提供详细分析："""
    
    use_llm_func = llm_config.get('best_model_func') or llm_config.get('cheap_model_func')
    if not use_llm_func:
        raise ValueError("LLM config must contain 'best_model_func' or 'cheap_model_func'")
    
    answer = await use_llm_func(reasoning_prompt)
    
    return {
        'answer': answer,
        'evidence_segments': [seg.get('segment_id', 'unknown') for seg in related_segments],
        'reasoning_paths': reasoning_paths if reasoning_paths else None
    }


def _build_evidence_and_refs(shot_results: list):
    """构建带编号的证据列表和引用列表。"""
    evidence_lines = []
    ref_lines = []
    for i, shot in enumerate(shot_results[:5], 1):
        time_start = shot.get('time_start', 0)
        time_end = shot.get('time_end', 0)
        video_name = shot.get('video_name', '未知视频')
        asr = shot.get('content', {}).get('asr_text', '')
        caption = shot.get('content', {}).get('vlm_shot_summary', {}).get('caption', '')
        desc = caption or asr or '无描述'
        asr_part = f" ASR: {asr}" if asr and asr != caption else ""
        evidence_lines.append(f"[{i}] ({time_start:.1f}s-{time_end:.1f}s): {desc}{asr_part}")
        ref_lines.append(f"[{i}] {video_name}, {time_start:.1f}s-{time_end:.1f}s")
    return evidence_lines, ref_lines


def _format_shot_results_plain(query: str, shot_results: list) -> str:
    """LLM 不可用时的降级纯拼接格式。"""
    shot_descriptions = []
    for i, shot in enumerate(shot_results[:5], 1):
        time_start = shot.get('time_start', 0)
        time_end = shot.get('time_end', 0)
        caption = shot.get('content', {}).get('vlm_shot_summary', {}).get('caption', '无描述')
        shot_descriptions.append(f"镜头 {i} ({time_start:.1f}s - {time_end:.1f}s): {caption}")
    answer = f"根据您的问题 \"{query}\"，我找到了以下相关镜头：\n\n"
    answer += "\n".join(shot_descriptions)
    answer += f"\n\n共找到 {len(shot_results)} 个相关镜头。"
    return answer


async def organize_shot_results_answer(query: str, shot_results: list, llm_func=None) -> str:
    if not shot_results:
        return "未找到相关的视频镜头。"

    evidence_lines, ref_lines = _build_evidence_and_refs(shot_results)

    if llm_func is None:
        return _format_shot_results_plain(query, shot_results)

    prompt = f"""基于以下视频证据回答问题。在答案中用 [编号] 标注引用来源。

## 问题
{query}

## 视频证据
{chr(10).join(evidence_lines)}

## 要求
1. 用自然语言回答，在关键论述后用 [1]、[2] 等标注来源
2. 答案末尾附上「引用」列表，格式为 [编号] 视频名, 起始时间-结束时间"""

    try:
        answer = await llm_func(prompt)
        if not any(f"[{i}]" in answer for i in range(1, len(ref_lines) + 1)):
            answer += "\n\n#### 引用\n" + "\n".join(ref_lines)
        return answer
    except Exception as e:
        logger.warning(f"LLM citation generation failed: {e}")
        return _format_shot_results_plain(query, shot_results)


async def fuse_hybrid_results(query: str, reasoning_result: dict, shot_results: list, llm_func=None) -> str:
    evidence_lines, ref_lines = _build_evidence_and_refs(shot_results) if shot_results else ([], [])

    reasoning_text = ""
    if reasoning_result and 'answer' in reasoning_result:
        reasoning_text = reasoning_result['answer']
        paths = reasoning_result.get('reasoning_paths') or []
        if paths:
            path_strs = [f"- {p['source']} —[{p['relation']}]→ {p['target']}" for p in paths[:5]]
            reasoning_text += "\n\n推理路径:\n" + "\n".join(path_strs)

    if llm_func is None or (not reasoning_text and not evidence_lines):
        return _fuse_hybrid_plain(reasoning_result, shot_results)

    prompt = f"""基于以下多源信息回答问题。在答案中用 [编号] 标注视频镜头引用来源。

## 问题
{query}

## 图谱推理分析
{reasoning_text or '无'}

## 视频镜头证据
{chr(10).join(evidence_lines) if evidence_lines else '无'}

## 要求
1. 综合图谱推理和镜头证据，用自然语言回答
2. 引用镜头证据时用 [1]、[2] 等标注
3. 答案末尾附上「引用」列表"""

    try:
        answer = await llm_func(prompt)
        if ref_lines and not any(f"[{i}]" in answer for i in range(1, len(ref_lines) + 1)):
            answer += "\n\n#### 引用\n" + "\n".join(ref_lines)
        return answer
    except Exception as e:
        logger.warning(f"LLM hybrid fusion failed: {e}")
        return _fuse_hybrid_plain(reasoning_result, shot_results)


def _fuse_hybrid_plain(reasoning_result: dict, shot_results: list) -> str:
    """LLM 不可用时的降级纯拼接格式。"""
    answer_parts = []
    if reasoning_result and 'answer' in reasoning_result:
        answer_parts.append("## 深度分析\n")
        answer_parts.append(reasoning_result['answer'])
        paths = reasoning_result.get('reasoning_paths') or []
        if paths:
            answer_parts.append("\n### 知识图谱推理依据\n")
            for path in paths[:5]:
                answer_parts.append(f"- {path['source']} —[{path['relation']}]→ {path['target']}")
    if shot_results:
        answer_parts.append("\n## 相关镜头\n")
        for i, shot in enumerate(shot_results[:5], 1):
            time_start = shot.get('time_start', 0)
            time_end = shot.get('time_end', 0)
            caption = shot.get('content', {}).get('vlm_shot_summary', {}).get('caption', '无描述')
            score = shot.get('retrieval_score', 0)
            answer_parts.append(f"- **镜头 {i}** ({time_start:.1f}s-{time_end:.1f}s, 相关性:{score:.3f}): {caption}")
    return "\n".join(answer_parts)


async def generate_community_reports(
    communities: list,
    graph_db,
    llm_func,
) -> list:
    """为每个社区生成主题报告"""
    reports = []
    for comm in communities:
        entity_descs = []
        for ent_name in comm['entities']:
            try:
                node = await graph_db.get_node(ent_name)
            except Exception:
                node = None
            desc = node.get('description', '') if node else ''
            entity_descs.append(f"- {ent_name}: {desc}")

        edge_descs = [f"- {u} —[{rel}]→ {v}" for u, v, rel in comm['edges']]

        prompt = f"""请为以下实体群体生成一段简洁的主题摘要（100-200字）。

## 实体
{chr(10).join(entity_descs)}

## 关系
{chr(10).join(edge_descs)}

## 要求
1. 总结这组实体构成的主题或事件
2. 指出核心实体及其角色
3. 一段话即可

摘要："""

        try:
            report = await llm_func(prompt)
        except Exception as e:
            logger.warning(f"Community report generation failed for {comm['community_id']}: {e}")
            report = f"社区包含实体: {', '.join(comm['entities'])}"

        reports.append({**comm, "report": report})

    logger.info(f"Generated {len(reports)} community reports")
    return reports


async def videorag_query_fusion(
    query: str,
    param: QueryParam,
    config: dict,
    **kwargs
) -> dict:
    """
    融合查询入口
    
    新流程：
    1. 分析查询类型
    2. 根据类型选择检索路径
    3. 执行检索/推理
    4. 生成最终答案
    
    参数：
        query: 用户查询文本
        param: 查询参数对象
        config: 全局配置字典
        kwargs: 额外参数，必须包含：
            - shot_storage: ShotKVStorage 实例
            - shot_vdb: MultiModalVectorDB 实例
            - segment_storage: SegmentKVStorage 实例
            - graph_db: NetworkXStorage 实例
    
    返回：
        {
            'answer': 答案文本,
            'evidence_shots': Shot 证据列表（可选）,
            'evidence_segments': Segment 证据列表（可选）,
            'query_type': 查询类型 ('precise'/'reasoning'/'hybrid')
        }
    """
    # Step 1: 查询分析
    query_type = analyze_query_type(query, llm_config=config.get('llm', {}))
    logger.info(f"Query type classified as: {query_type}")
    
    # Step 2: 路由执行
    if query_type == 'precise':
        # 只走 Shot 层
        shot_storage = kwargs.get('shot_storage')
        shot_vdb = kwargs.get('shot_vdb')
        
        if not shot_storage or not shot_vdb:
            raise ValueError("Precise query requires shot_storage and shot_vdb")
        
        shot_results = await shot_level_multi_channel_search(
            query=query,
            shot_storage=shot_storage,
            shot_vdb=shot_vdb,
            top_k=param.top_k,
            embedding_func=kwargs.get('embedding_func'),
            visual_embedder=kwargs.get('visual_embedder'),
            shot_bm25=kwargs.get('shot_bm25'),
            reranker=kwargs.get('reranker'),
        )

        llm_func = config.get('llm', {}).get('cheap_model_func')
        answer = await organize_shot_results_answer(query, shot_results, llm_func=llm_func)
        
        return {
            'answer': answer,
            'evidence_shots': shot_results,
            'query_type': 'precise'
        }
    
    elif query_type == 'reasoning':
        # 只走 Segment 层
        segment_storage = kwargs.get('segment_storage')
        graph_db = kwargs.get('graph_db')
        text_chunks_db = kwargs.get('text_chunks_db')
        entities_vdb = kwargs.get('entities_vdb')
        
        if not segment_storage or not graph_db:
            raise ValueError("Reasoning query requires segment_storage and graph_db")
        
        reasoning_result = await segment_level_graph_reasoning(
            query=query,
            segment_storage=segment_storage,
            graph_db=graph_db,
            llm_config=config.get('llm', {}),
            text_chunks_db=text_chunks_db,
            entities_vdb=entities_vdb,
            community_reports=kwargs.get('community_reports'),
        )

        return {
            'answer': reasoning_result['answer'],
            'evidence_segments': reasoning_result['evidence_segments'],
            'query_type': 'reasoning'
        }
    
    else:  # hybrid
        # 双路协作
        shot_storage = kwargs.get('shot_storage')
        shot_vdb = kwargs.get('shot_vdb')
        segment_storage = kwargs.get('segment_storage')
        graph_db = kwargs.get('graph_db')
        text_chunks_db = kwargs.get('text_chunks_db')
        entities_vdb = kwargs.get('entities_vdb')
        
        # Step A: 先用 Segment 层理解语义
        reasoning_result = None
        if segment_storage and graph_db:
            try:
                reasoning_result = await segment_level_graph_reasoning(
                    query=query,
                    segment_storage=segment_storage,
                    graph_db=graph_db,
                    llm_config=config.get('llm', {}),
                    text_chunks_db=text_chunks_db,
                    entities_vdb=entities_vdb,
                    community_reports=kwargs.get('community_reports'),
                )
            except Exception as e:
                logger.warning(f"Segment reasoning failed: {e}")
        
        # Step B: 再用 Shot 层精确定位
        shot_results = []
        if shot_storage and shot_vdb:
            try:
                shot_results = await shot_level_multi_channel_search(
                    query=query,
                    shot_storage=shot_storage,
                    shot_vdb=shot_vdb,
                    top_k=param.top_k,
                    embedding_func=kwargs.get('embedding_func'),
                    visual_embedder=kwargs.get('visual_embedder'),
                    shot_bm25=kwargs.get('shot_bm25'),
                    reranker=kwargs.get('reranker'),
                )
            except Exception as e:
                logger.warning(f"Shot search failed: {e}")

        # Step B2: KG 引导重排序
        if graph_db and entities_vdb and shot_results:
            try:
                shot_results = await _kg_guided_shot_rerank(
                    shot_results, graph_db, entities_vdb, query, top_k=param.top_k
                )
            except Exception as e:
                logger.warning(f"KG rerank failed: {e}")

        # Step C: 融合结果
        llm_func = config.get('llm', {}).get('cheap_model_func')
        final_answer = await fuse_hybrid_results(query, reasoning_result, shot_results, llm_func=llm_func)
        
        return {
            'answer': final_answer,
            'evidence_shots': shot_results,
            'evidence_segments': reasoning_result['evidence_segments'] if reasoning_result else [],
            'query_type': 'hybrid'
        }
    