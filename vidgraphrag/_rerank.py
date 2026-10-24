import asyncio
import logging
from typing import List, Tuple, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger("vidgraphrag")

try:
    from dashscope import TextReRanking
    from http import HTTPStatus
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception),
)
async def dashscope_rerank(
    query: str,
    passages: List[str],
    top_k: int = 5,
    model: str = "gte-rerank",
) -> Optional[List[Tuple[int, float]]]:
    """
    调用 DashScope gte-rerank API 对 query-passage pairs 重排序。

    返回: [(original_index, relevance_score), ...] 按 score 降序。
          API 失败或未安装 dashscope 时返回 None（调用方保持原排序）。
    """
    if not RERANK_AVAILABLE:
        return None

    if not passages:
        return None

    loop = asyncio.get_event_loop()

    def _call_rerank():
        response = TextReRanking.call(
            model=model,
            query=query,
            documents=passages,
            top_n=min(top_k, len(passages)),
        )
        if response.status_code == HTTPStatus.OK:
            results = []
            for item in response.output.results:
                results.append((item.index, item.relevance_score))
            return results
        else:
            raise Exception(f"DashScope rerank failed: {response.code} - {response.message}")

    return await loop.run_in_executor(None, _call_rerank)
