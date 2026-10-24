import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ..base import StorageNameSpace
from .._utils import load_json, write_json

logger = logging.getLogger("vidgraphrag")

try:
    from rank_bm25 import BM25Okapi
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


@dataclass
class ShotBM25Storage(StorageNameSpace):
    """
    BM25 稀疏检索索引，基于 Shot 的 ASR 转录文本构建。
    作为第 4 路检索通道与 visual/audio/vlm 向量检索互补，
    擅长精确关键词匹配（人名、术语、特定台词）。

    持久化格式: {working_dir}/bm25_{namespace}.json
    存储内容: shot_ids 列表 + 分词后的语料库
    """

    def __post_init__(self):
        if not BM25_AVAILABLE:
            logger.warning("rank_bm25 or jieba not installed — BM25 retrieval disabled")

        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"bm25_{self.namespace}.json")
        self._shot_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25 = None

        data = load_json(self._file_name)
        if data and data.get("shot_ids") and BM25_AVAILABLE:
            self._shot_ids = data["shot_ids"]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info(f"Load BM25 index {self.namespace} with {len(self._shot_ids)} shots")
        else:
            logger.info(f"BM25 index {self.namespace} initialized empty")

    async def build_from_shots(self, shots_data: Dict[str, Dict]):
        """
        从 Shot 数据增量构建 BM25 索引。

        提取每个 Shot 的 content.asr_text，用 jieba 分词后追加到语料库。
        已存在的 shot_id 会被跳过，支持多视频逐个索引。
        """
        if not BM25_AVAILABLE:
            return

        existing_keys = set(self._shot_ids)
        added = 0

        for shot_id, shot_data in shots_data.items():
            asr_text = shot_data.get("content", {}).get("asr_text", "").strip()
            ocr_list = shot_data.get("content", {}).get("vlm_shot_summary", {}).get("ocr", [])
            if isinstance(ocr_list, list):
                ocr_text = " ".join(str(item) for item in ocr_list if item).strip()
            elif isinstance(ocr_list, str):
                ocr_text = ocr_list.strip()
            else:
                ocr_text = ""

            asr_key = f"{shot_id}#asr"
            if asr_key not in existing_keys and asr_text:
                tokens = [t for t in jieba.lcut(asr_text) if t.strip()]
                if tokens:
                    self._shot_ids.append(asr_key)
                    self._tokenized_corpus.append(tokens)
                    existing_keys.add(asr_key)
                    added += 1

            ocr_key = f"{shot_id}#ocr"
            if ocr_key not in existing_keys and ocr_text:
                tokens = [t for t in jieba.lcut(ocr_text) if t.strip()]
                if tokens:
                    self._shot_ids.append(ocr_key)
                    self._tokenized_corpus.append(tokens)
                    existing_keys.add(ocr_key)
                    added += 1

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(
            f"BM25 index updated: +{added} shots, {len(self._shot_ids)} total"
        )

    async def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        BM25 检索，返回 (shot_id, bm25_score) 列表。
        输出格式与 MultiModalVectorDB._fuse_rrf 兼容。
        """
        if not BM25_AVAILABLE or self._bm25 is None or not self._shot_ids:
            return []

        query_tokens = [t for t in jieba.lcut(query) if t.strip()]
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-top_k * 2:][::-1]

        deduped: Dict[str, float] = {}
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            shot_id = self._shot_ids[idx].split("#")[0]
            if shot_id not in deduped or scores[idx] > deduped[shot_id]:
                deduped[shot_id] = float(scores[idx])

        ranked = sorted(deduped.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    async def index_done_callback(self):
        write_json(
            {"shot_ids": self._shot_ids, "tokenized_corpus": self._tokenized_corpus},
            self._file_name,
        )

    async def drop(self):
        self._shot_ids = []
        self._tokenized_corpus = []
        self._bm25 = None
