import asyncio
import html
import json
import logging
import os
import re
import numbers
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union

import numpy as np
import tiktoken

logger = logging.getLogger("vidgraphrag")
ENCODER = None


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    获取或创建一个异步事件循环。

    在主线程中返回已存在的事件循环，在子线程中创建新的事件循环。
    解决子线程无法访问主线程事件循环的问题。

    返回:
        asyncio.AbstractEventLoop: 可用的异步事件循环对象

    异常处理:
        RuntimeError: 当在子线程中调用 asyncio.get_event_loop() 时抛出，此时创建新循环
    """
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """
    从字符串中提取 JSON 格式的内容。

    使用正则表达式查找第一个完整的 JSON 对象（从 { 到 }）。
    用于从 LLM 响应中提取结构化数据。

    参数:
        content (str): 可能包含 JSON 的字符串

    返回:
        Union[str, None]: 提取到的 JSON 字符串，如果未找到则返回 None

    示例:
        >>> locate_json_string_body_from_string('Some text {"key": "value"} more text')
        '{"key": "value"}'
    """
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


def convert_response_to_json(response: str) -> dict:
    """
    将 LLM 响应字符串转换为 JSON 字典。

    先定位 JSON 字符串位置，然后解析为 Python 字典。
    用于处理实体提取、关系提取等任务的 LLM 输出。

    参数:
        response (str): LLM 返回的响应字符串，应包含 JSON 格式内容

    返回:
        dict: 解析后的字典对象

    异常:
        AssertionError: 当无法从响应中找到 JSON 时抛出
        json.JSONDecodeError: 当 JSON 格式无效时抛出
    """
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    """
    使用 tiktoken 将文本编码为 token ID 列表。

    tiktoken 是 OpenAI 开发的高效 token 编码器，支持多种模型。

    参数:
        content (str): 要编码的文本内容
        model_name (str, optional): 使用的模型名称，默认 "gpt-4o"

    返回:
        list[int]: token ID 列表

    优化:
        使用全局 ENCODER 变量避免重复初始化，提升性能
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    """
    使用 tiktoken 将 token ID 列表解码回文本。

    与 encode_string_by_tiktoken 互为逆操作。

    参数:
        tokens (list[int]): token ID 列表
        model_name (str, optional): 使用的模型名称，默认 "gpt-4o"

    返回:
        str: 解码后的原始文本
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """
    根据 token 数量截断列表。

    用于控制上下文长度，确保不超过 LLM 的 token 限制。
    逐个累加元素的 token 数，直到达到上限。

    参数:
        list_data (list): 待截断的数据列表
        key (callable): 提取文本内容的函数，输入列表元素，返回字符串
        max_token_size (int): 最大允许的 token 数量

    返回:
        list: 截断后的列表，总 token 数不超过 max_token_size

    示例:
        >>> truncate_list_by_token_size(
        ...     [{"text": "hello"}, {"text": "world"}],
        ...     lambda x: x["text"],
        ...     max_token_size=10
        ... )
        [{'text': 'hello'}, {'text': 'world'}]
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    """
    计算内容的 MD5 哈希值作为唯一标识符。

    用于生成 chunk ID、entity ID 等，确保相同内容总是生成相同 ID。

    参数:
        content: 要哈希的内容（字符串或其他可序列化对象）
        prefix (str, optional): ID 前缀，用于区分不同类型的数据，默认空字符串

    返回:
        str: 前缀 + MD5 哈希值（32 位十六进制）

    示例:
        >>> compute_mdhash_id("hello world", prefix="chunk-")
        'chunk-5eb63bbbe01eeed093cb22bb8f5acdc3'
    """
    return prefix + md5(content.encode()).hexdigest()


def write_json(json_obj, file_name):
    """
    将 Python 对象写入 JSON 文件。

    参数:
        json_obj: 要保存的 Python 对象（dict、list 等）
        file_name (str): 输出文件路径

    注意:
        - 使用 UTF-8 编码，支持中文
        - indent=2 使文件格式化显示，便于阅读
        - ensure_ascii=False 保留非 ASCII 字符原样显示
        - default=_json_default 支持 numpy ndarray / numpy 标量 等非标准类型
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj):
    """为 json.dump 提供默认序列化处理，支持 numpy / 集合等常见非标准类型"""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_json(file_name):
    """
    从 JSON 文件加载 Python 对象。

    参数:
        file_name (str): JSON 文件路径

    返回:
        dict 或 list: 加载的 Python 对象，如果文件不存在则返回 None

    注意:
        文件不存在时不抛出异常，直接返回 None
    """
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def get_imagebind_device():
    """
    Get the best device for ImageBind models
    ImageBind uses Conv3D which is not supported on MPS, so fall back to CPU on Mac
    Priority: CUDA > CPU (MPS not supported)
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        # Force CPU even if MPS is available because ImageBind uses Conv3D
        return torch.device("cpu")


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(*args: str):
    """
    将交替的用户和助手消息打包成 OpenAI API 格式。

    自动按顺序分配 user 和 assistant 角色。

    参数:
        *args (str): 可变参数，交替的用户消息和助手消息

    返回:
        list[dict]: OpenAI 消息格式列表，每个元素包含 role 和 content

    示例:
        >>> pack_user_ass_to_openai_messages("你好", "你好！有什么可以帮助你的？", "再见")
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
            {"role": "user", "content": "再见"}
        ]
    """
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value):
    """
    使用正则表达式判断字符串是否为浮点数。

    参数:
        value (str): 待检查的字符串

    返回:
        bool: 如果是有效的浮点数格式返回 True，否则返回 False

    匹配模式:
        - 可选的正负号 (+/-)
        - 整数部分（可选）
        - 小数点（可选）
        - 小数部分（可选，但至少有一位数字）
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    """
    计算参数的哈希值，用于缓存键生成。

    参数:
        *args: 任意数量的参数

    返回:
        str: 参数的 MD5 哈希值（32 位十六进制）

    用途:
        在 LLM 缓存中，相同的输入参数生成相同的缓存键
    """
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """
    使用多个分隔符分割字符串。

    参数:
        content (str): 待分割的字符串
        markers (list[str]): 分隔符列表

    返回:
        list[str]: 分割后的字符串列表，已去除首尾空白

    实现:
        使用正则表达式的 OR 逻辑连接所有分隔符
    """
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: Any) -> str:
    """
    用引号包围内容，数字类型不加引号。

    用于生成 CSV 或 SQL 等格式的字符串表示。

    参数:
        content (Any): 任意类型的内容

    返回:
        str:
            - 数字类型：直接转为字符串，不加引号
            - 其他类型：去除首尾引号后，再用双引号包围

    示例:
        >>> enclose_string_with_quotes(123)
        '123'
        >>> enclose_string_with_quotes("hello")
        '"hello"'
    """
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    """
    将二维列表转换为 CSV 格式的字符串。

    参数:
        data (list[list]): 二维列表，每个子列表代表一行

    返回:
        str: CSV 格式的字符串，行内用制表符分隔，行间用换行符分隔

    示例:
        >>> list_of_list_to_csv([["a", "b"], ["c", "d"]])
        '"a",\t"b"\n"c",\t"d"'
    """
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """
    清理输入字符串，移除 HTML 转义字符、控制字符等不需要的字符。

    参考 Microsoft GraphRAG 的官方实现。

    参数:
        input (Any): 输入内容，如果不是字符串则原样返回

    返回:
        str: 清理后的字符串

    清理步骤:
        1. HTML 转义字符还原（如 &amp; → &）
        2. 移除控制字符（ASCII 0x00-0x1F 和 0x7F-0x9F）
    """
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    """
    嵌入函数的包装类，带有元数据属性。

    用于统一管理不同嵌入模型的接口和配置。

    属性:
        embedding_dim (int): 嵌入向量的维度（如 1024）
        max_token_size (int): 单次处理的最大 token 数
        model_name (str): 嵌入模型名称
        func (callable): 实际的嵌入函数（异步协程）

    方法:
        __call__: 使实例可以像函数一样调用
    """
    embedding_dim: int
    max_token_size: int
    model_name: str
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        调用嵌入函数，自动传递模型名称。

        参数:
            *args: 位置参数（通常是 texts 列表）
            **kwargs: 关键字参数

        返回:
            np.ndarray: 嵌入向量数组（形状：n_texts × embedding_dim）

        特殊处理:
            - 自动将 model_name 添加到 kwargs
            - 将位置参数转换为关键字参数 'texts'
        """
        # Had to fix this as the embedding function took only one named argument put it's passed in
        # positionally, now we need to pass both
        kwargs['model_name'] = self.model_name
        
        # If there are positional arguments, convert them to keyword arguments
        if args:
            # Assuming the first positional argument is always 'texts'
            if len(args) == 1 and isinstance(args[0], list):
                kwargs['texts'] = args[0]
            else:
                raise ValueError("Unexpected positional arguments. Expected a single list of texts")
        # Call the function with the updated keyword arguments
        return await self.func(**kwargs)        

    # Decorators ------------------------------------------------------------------------


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """
    限制异步函数的并发调用次数。

    通过手动计数实现限流，避免使用 asyncio.Semaphore 需要 nest-asyncio。

    参数:
        max_size (int): 最大并发数
        waitting_time (float, optional): 等待间隔（秒），默认 0.0001 秒

    返回:
        callable: 装饰器函数

    工作原理:
        1. 维护当前执行数 __current_size
        2. 超过限制时循环等待
        3. 执行完成后递减计数

    使用场景:
        - 限制 LLM API 调用频率
        - 控制嵌入函数的并发量
        - 避免资源耗尽
    """

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """
    为普通函数添加属性，包装成 EmbeddingFunc 对象。

    这是一个装饰器工厂，用于创建带有元数据的嵌入函数。

    参数:
        **kwargs: 要设置的属性键值对
            - embedding_dim: 向量维度
            - max_token_size: 最大 token 数
            - model_name: 模型名称
            - func: 被包装的函数

    返回:
        callable: 装饰器函数，返回 EmbeddingFunc 实例

    使用示例:
        @wrap_embedding_func_with_attrs(
            embedding_dim=1024,
            max_token_size=8192,
            model_name="text-embedding-3-large"
        )
        async def my_embedding_func(texts):
            ...
    """

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
