import numpy as np

from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
from ollama import AsyncClient
from dataclasses import asdict, dataclass, field
import dashscope
from http import HTTPStatus

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

# 全局变量用于存储单例的异步客户端实例
global_openai_async_client = None
global_azure_openai_async_client = None
global_ollama_client = None
global_dashscope_async_client = None

def get_openai_async_client_instance():
    """
    获取 OpenAI 异步客户端的单例实例
    
    Returns:
        AsyncOpenAI: OpenAI 异步客户端实例
    """
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    """
    获取 Azure OpenAI 异步客户端的单例实例
    
    Returns:
        AsyncAzureOpenAI: Azure OpenAI 异步客户端实例
    """
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client

def get_ollama_async_client_instance():
    """
    获取 Ollama 异步客户端的单例实例
    
    Returns:
        AsyncClient: Ollama 异步客户端实例
    """
    global global_ollama_client
    if global_ollama_client is None:
        # set OLLAMA_HOST or pass in host="http://127.0.0.1:11434"
        global_ollama_client = AsyncClient()  # Adjust base URL if necessary        
    return global_ollama_client

def get_dashscope_async_client_instance(api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
    """
    获取 DashScope 异步客户端的单例实例（OpenAI 兼容接口）
    
    Args:
        api_key: DashScope API Key
        base_url: DashScope API 基础 URL
        
    Returns:
        AsyncOpenAI: DashScope 异步客户端实例
    """
    global global_dashscope_async_client
    if global_dashscope_async_client is None:
        global_dashscope_async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    return global_dashscope_async_client

# Setup LLM Configuration.
@dataclass
class LLMConfig:
    """
    LLM 配置数据类，用于统一管理语言模型和嵌入模型的配置参数
    
    Attributes:
        embedding_func_raw (callable): 原始嵌入函数，需要接受 model_name 和 texts 参数
        embedding_model_name (str): 嵌入模型名称
        embedding_dim (int): 嵌入向量的维度
        embedding_max_token_size (int): 嵌入模型支持的最大 token 数
        embedding_batch_num (int): 嵌入处理的批量大小
        embedding_func_max_async (int): 嵌入函数最大异步并发数
        query_better_than_threshold (float): 查询相似度阈值，用于判断检索质量
        
        best_model_func_raw (callable): 高级模型完成函数，需要接受 model_name 和 prompt 等参数
        best_model_name (str): 高级模型名称（如 gpt-4o）
        best_model_max_token_size (int): 高级模型支持的最大 token 数
        best_model_max_async (int): 高级模型最大异步并发数
        
        cheap_model_func_raw (callable): 经济模型完成函数，需要接受 model_name 和 prompt 等参数
        cheap_model_name (str): 经济模型名称（如 gpt-4o-mini）
        cheap_model_max_token_size (int): 经济模型支持的最大 token 数
        cheap_model_max_async (int): 经济模型最大异步并发数
        
        embedding_func (EmbeddingFunc): 经过包装的嵌入函数，在 __post_init__ 中赋值
        best_model_func (callable): 绑定了模型名称的高级模型函数，在 __post_init__ 中赋值
        cheap_model_func (callable): 绑定了模型名称的经济模型函数，在 __post_init__ 中赋值
    """
    # To be set
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int    
    embedding_func_max_async: int 
    query_better_than_threshold: float
    
    best_model_func_raw: callable
    best_model_name: str    
    best_model_max_token_size: int
    best_model_max_async: int
    
    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int
    
    # Caption model configuration
    caption_model_func_raw: callable = None
    caption_model_name: str = None
    caption_model_max_async: int = 3

    # Assigned in post init
    embedding_func: EmbeddingFunc  = None    
    best_model_func: callable = None    
    cheap_model_func: callable = None
    caption_model_func: callable = None
    

    def __post_init__(self):
        """
        在数据类初始化后执行，用于包装和绑定模型函数
        将原始函数包装为不需要传递 model_name 的便捷函数
        """
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim = self.embedding_dim,
            max_token_size = self.embedding_max_token_size,
            model_name = self.embedding_model_name)
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )

        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )

        # Initialize caption_model_func if caption_model_func_raw is provided
        if hasattr(self, 'caption_model_func_raw') and self.caption_model_func_raw is not None:
            self.caption_model_func = lambda content_list, *args, **kwargs: self.caption_model_func_raw(
                self.caption_model_name, content_list, *args, **kwargs
            )

##### OpenAI Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    基于 OpenAI 的文本补全函数，支持缓存机制
    
    Args:
        model (str): 使用的模型名称
        prompt (str): 用户输入的提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他参数，包括 hashing_kv（缓存存储）和 use_cache（是否使用缓存）
        
    Returns:
        str: 模型生成的回复内容
        
    Raises:
        RateLimitError: 当遇到速率限制错误时重试
        APIConnectionError: 当遇到 API 连接错误时重试
    """
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    使用 GPT-4o 模型进行文本补全的便捷函数
    
    Args:
        model_name (str): 模型名称
        prompt (str): 用户提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他传递给底层 API 的参数
        
    Returns:
        str: GPT-4o 模型生成的回复
    """
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    使用 GPT-4o-mini 模型进行文本补全的便捷函数
    
    Args:
        model_name (str): 模型名称
        prompt (str): 用户提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他传递给底层 API 的参数
        
    Returns:
        str: GPT-4o-mini 模型生成的回复
    """
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    """
    使用 OpenAI 嵌入模型将文本转换为向量表示
    
    Args:
        model_name (str): 嵌入模型名称
        texts (list[str]): 待嵌入的文本列表
        
    Returns:
        np.ndarray: 嵌入向量数组，形状为 (len(texts), embedding_dim)
        
    Raises:
        RateLimitError: 当遇到速率限制错误时重试
        APIConnectionError: 当遇到 API 连接错误时重试
    """
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

# OpenAI 默认配置：使用 GPT-4o 作为高级模型，GPT-4o-mini 作为经济模型
openai_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 20,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

# OpenAI GPT-4o-mini 配置：高级模型和经济模型都使用 GPT-4o-mini
openai_4o_mini_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 20,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

###### Azure OpenAI Configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    基于 Azure OpenAI 的文本补全函数，支持缓存机制
    
    Args:
        deployment_name (str): Azure OpenAI 部署名称
        prompt (str): 用户输入的提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他参数，包括 hashing_kv（缓存存储）和 use_cache（是否使用缓存）
        
    Returns:
        str: 模型生成的回复内容
        
    Raises:
        RateLimitError: 当遇到速率限制错误时重试
        APIConnectionError: 当遇到 API 连接错误时重试
    """
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    使用 Azure GPT-4o 模型进行文本补全的便捷函数
    
    Args:
        model_name (str): 模型名称
        prompt (str): 用户提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他传递给底层 API 的参数
        
    Returns:
        str: Azure GPT-4o 模型生成的回复
    """
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    使用 Azure GPT-4o-mini 模型进行文本补全的便捷函数
    
    Args:
        model_name (str): 模型名称
        prompt (str): 用户提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他传递给底层 API 的参数
        
    Returns:
        str: Azure GPT-4o-mini 模型生成的回复
    """
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    """
    使用 Azure OpenAI 嵌入模型将文本转换为向量表示
    
    Args:
        model_name (str): 嵌入模型名称
        texts (list[str]): 待嵌入的文本列表
        
    Returns:
        np.ndarray: 嵌入向量数组，形状为 (len(texts), embedding_dim)
        
    Raises:
        RateLimitError: 当遇到速率限制错误时重试
        APIConnectionError: 当遇到 API 连接错误时重试
    """
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


# Azure OpenAI 配置：使用 GPT-4o 作为高级模型，GPT-4o-mini 作为经济模型
azure_openai_config = LLMConfig(
    embedding_func_raw = azure_openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size = 8192,    
    embedding_batch_num = 20,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    best_model_func_raw = azure_gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,

    cheap_model_func_raw  = azure_gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)


######  Ollama configuration

async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    基于 Ollama 的文本补全函数，支持缓存机制
    
    Args:
        model (str): 使用的模型名称
        prompt (str): 用户输入的提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他参数，包括 hashing_kv（缓存存储）和 use_cache（是否使用缓存）
        
    Returns:
        str: 模型生成的回复内容
    """
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Send the request to Ollama
    response = await ollama_client.chat(
        model=model,
        messages=messages
    )
    # print(messages)
    # print(response['message']['content'])

    
    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": response['message']['content'], "model": model}}
        )
        await hashing_kv.index_done_callback()

    return response['message']['content']



async def ollama_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """
    使用 Ollama 通用模型进行文本补全的便捷函数
    
    Args:
        model_name (str): 模型名称
        prompt (str): 用户提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他传递给底层 API 的参数
        
    Returns:
        str: Ollama 模型生成的回复
    """
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

async def ollama_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """
    使用 Ollama 轻量级模型进行文本补全的便捷函数
    
    Args:
        model_name (str): 模型名称
        prompt (str): 用户提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他传递给底层 API 的参数
        
    Returns:
        str: Ollama 轻量级模型生成的回复
    """
    return await ollama_complete_if_cache(
        # "deepseek-r1:latest",  # For now select your model
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    """
    使用 Ollama 嵌入模型将文本转换为向量表示
    
    Args:
        model_name (str): 嵌入模型名称
        texts (list[str]): 待嵌入的文本列表
        
    Returns:
        np.ndarray: 嵌入向量数组，形状为 (len(texts), embedding_dim)
        
    Raises:
        RateLimitError: 当遇到速率限制错误时重试
        APIConnectionError: 当遇到 API 连接错误时重试
    """
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    # Send the request to Ollama for embeddings
    response = await ollama_client.embed(
        model=model_name,  
        input=texts
    )

    # Extract embeddings from the response
    embeddings = response['embeddings']

    return np.array(embeddings)

# Ollama 配置：使用 gemma2 作为高级模型，olmo2 作为经济模型
ollama_config = LLMConfig(
    embedding_func_raw = ollama_embedding,
    embedding_model_name = "nomic-embed-text",
    embedding_dim = 768,
    embedding_max_token_size=8192,
    embedding_batch_num = 1,
    embedding_func_max_async = 1,
    query_better_than_threshold = 0.2,
    best_model_func_raw = ollama_complete ,
    best_model_name = "gemma2:latest", # need to be a solid instruct model
    best_model_max_token_size = 32768,
    best_model_max_async  = 1,
    cheap_model_func_raw = ollama_mini_complete,
    cheap_model_name = "olmo2",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 1
)
###### DeepSeek Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    基于 DeepSeek API 的文本补全函数，支持缓存机制
    
    Args:
        model (str): 使用的模型名称
        prompt (str): 用户输入的提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他参数，包括 hashing_kv（缓存存储）和 use_cache（是否使用缓存）
        
    Returns:
        str: 模型生成的回复内容
        
    Raises:
        RateLimitError: 当遇到速率限制错误时重试
        APIConnectionError: 当遇到 API 连接错误时重试
    """
    # 使用 DeepSeek API
    import httpx
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # DeepSeek API 调用
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', 'sk-*******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4096)
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model}}
        )
        await hashing_kv.index_done_callback()

    return content

async def deepseek_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """
    使用 DeepSeek 模型进行文本补全的便捷函数
    
    Args:
        model_name (str): 模型名称
        prompt (str): 用户提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他传递给底层 API 的参数
        
    Returns:
        str: DeepSeek 模型生成的回复
    """
    return await deepseek_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def bge_m3_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    """
    使用硅基流动的 BAAI/bge-m3 嵌入模型将文本转换为向量表示
    
    Args:
        model_name (str): 嵌入模型名称
        texts (list[str]): 待嵌入的文本列表
        
    Returns:
        np.ndarray: 嵌入向量数组，形状为 (len(texts), embedding_dim)
        
    Raises:
        RateLimitError: 当遇到速率限制错误时重试
        APIConnectionError: 当遇到 API 连接错误时重试
    """
    # 使用硅基流动的 BAAI/bge-m3 嵌入模型
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY', 'sk-******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "BAAI/bge-m3",
                "input": texts,
                "encoding_format": "float"
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def dashscope_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    """
    使用阿里云 DashScope 嵌入模型将文本转换为向量表示
    
    Args:
        model_name (str): 嵌入模型名称（如 text-embedding-v2）
        texts (list[str]): 待嵌入的文本列表
        
    Returns:
        np.ndarray: 嵌入向量数组，形状为 (len(texts), embedding_dim)
    """
    from dashscope import TextEmbedding
    
    # DashScope SDK 是同步的，需要在异步环境中执行
    import asyncio
    loop = asyncio.get_event_loop()
    
    def _call_embedding():
        response = TextEmbedding.call(
            model=model_name,
            input=texts
        )
        if response.status_code == HTTPStatus.OK:
            embeddings = [item['embedding'] for item in response.output['embeddings']]
            return np.array(embeddings)
        else:
            raise Exception(f"DashScope embedding failed: {response.code} - {response.message}")
    
    return await loop.run_in_executor(None, _call_embedding)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
)
async def dashscope_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    基于阿里云 DashScope 的文本补全函数，支持缓存机制
    
    Args:
        model (str): 使用的模型名称（如 qwen-max, qwen-plus）
        prompt (str): 用户输入的提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他参数，包括 hashing_kv（缓存存储）和 use_cache（是否使用缓存）
        
    Returns:
        str: 模型生成的回复内容
    """
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # DashScope SDK 是同步的，需要在异步环境中执行
    import asyncio
    loop = asyncio.get_event_loop()
    
    def _call_dashscope():
        from dashscope import Generation
        # 强制关闭 thinking 模式：qwen3 系列推理型模型默认会输出大量 reasoning_tokens，
        # 在 RAG 索引/生成场景下只增加成本与延迟，不需要。
        response = Generation.call(
            model=model,
            messages=messages,
            result_format='message',
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
            enable_thinking=False,
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content
        else:
            raise Exception(f"DashScope API failed: {response.code} - {response.message}")
    
    content = await loop.run_in_executor(None, _call_dashscope)

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    
    return content


async def dashscope_qwen_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    使用阿里云通义千问模型进行文本补全的便捷函数
    
    Args:
        model_name (str): 模型名称（如 qwen-max, qwen-plus, qwen-turbo）
        prompt (str): 用户提示词
        system_prompt (str, optional): 系统提示词，默认为 None
        history_messages (list, optional): 历史消息列表，默认为空列表
        **kwargs: 其他传递给底层 API 的参数
        
    Returns:
        str: 通义千问模型生成的回复
    """
    return await dashscope_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

# DashScope 默认配置：使用 Qwen-Max 作为高级模型，Qwen-Plus 作为经济模型
dashscope_config = LLMConfig(
    embedding_func_raw=dashscope_embedding,
    embedding_model_name="text-embedding-v2",
    embedding_dim=1536,
    embedding_max_token_size=2048,
    embedding_batch_num=20,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,
    
    best_model_func_raw=dashscope_qwen_complete,
    best_model_name="qwen3-8b",
    best_model_max_token_size=32768,
    best_model_max_async=16,

    cheap_model_func_raw=dashscope_qwen_complete,
    cheap_model_name="qwen3-8b",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def dashscope_caption_complete(
    model_name, content_list, **kwargs
) -> str:
    """
    DashScope vision model completion for video caption (OpenAI 兼容接口)
    
    Args:
        model_name: 模型名称（如 qwen-vl-plus-latest）
        content_list: list of {"type": "image_url", "image_url": {"url": "..."}} and {"type": "text", "text": "..."}
        **kwargs: 包含 ali_dashscope_api_key 和 ali_dashscope_base_url
        
    Returns:
        str: 模型生成的回复内容
    """
    api_key = kwargs.get("ali_dashscope_api_key") or kwargs.get("global_config", {}).get("ali_dashscope_api_key")
    base_url = kwargs.get("ali_dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not base_url.endswith("/v1") and not base_url.endswith("/compatible-mode/v1"):
        base_url = base_url.rstrip("/") + "/compatible-mode/v1"
    
    if not api_key:
        raise ValueError("DashScope API key is required for caption generation")
    
    dashscope_async_client = get_dashscope_async_client_instance(api_key, base_url)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content_list}
    ]
    
    response = await dashscope_async_client.chat.completions.create(
        model=model_name, 
        messages=messages, 
    )
    
    return response.choices[0].message.content

# DeepSeek + BAAI/bge-m3 配置：使用 DeepSeek 作为语言模型，BGE-M3 作为嵌入模型
deepseek_bge_config = LLMConfig(
    embedding_func_raw = bge_m3_embedding,
    embedding_model_name = "BAAI/bge-m3",
    embedding_dim = 1024,  # bge-m3 的嵌入维度
    embedding_max_token_size = 8192,
    embedding_batch_num = 20,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,
    
    best_model_func_raw = deepseek_complete,
    best_model_name = "deepseek-chat",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
    
    cheap_model_func_raw = deepseek_complete,
    cheap_model_name = "deepseek-chat",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

