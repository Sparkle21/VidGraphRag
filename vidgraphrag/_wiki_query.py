"""
Wiki Query Retriever — index → 页面 → 回答 → 回填

流程：
1. LLM 读 index.md → 输出需要阅读的页面路径列表
2. LLM 读取页面内容 → 综合生成答案
3. LLM 自评置信度
4. 好答案回填为新 Wiki 页（queries/）
"""

import json
import re
from typing import Optional

from ._wiki import WikiKnowledgeBase, WikiScale, slugify
from ._utils import logger, encode_string_by_tiktoken


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_INDEX_LOOKUP = """你是一个知识检索助手。根据用户的问题，从 Wiki 目录中选择需要阅读的页面。

## 用户问题
{query}

## Wiki 目录
{index_content}

## 输出要求
以 JSON 格式列出需要阅读的页面路径（最多 8 个，按相关性排序）：
{{"pages": ["videos/xxx.md", "entities/yyy.md", ...]}}

只返回 JSON，不要其他内容。如果没有相关页面，返回空数组。"""

PROMPT_WIKI_ANSWER = """你是一个基于 Wiki 知识库的问答助手。请根据以下 Wiki 页面内容回答用户的问题。

## 用户问题
{query}

## Wiki 页面内容
{pages_content}

## 回答要求
1. 基于 Wiki 内容回答，引用具体页面
2. 如果信息不足，明确说明需要更深入检索
3. 使用 [[实体名]] 格式引用实体

## 回答格式
先回答问题，然后在最后一行用 JSON 格式给出置信度：
{{"confidence": 0.0到1.0的数字, "cited_pages": ["page1.md", "page2.md"]}}"""

PROMPT_BACKFILL_PAGE = """你是一个知识库编辑。请将以下问答对整理成一个 Wiki 知识页。

## 问题
{query}

## 回答
{answer}

## 引用页面
{cited_pages}

## 输出要求
生成一个结构化的 Markdown 页面：
# {title}

## 问题
（原始问题）

## 回答
（整理后的回答，保留 [[实体名]] 链接）

## 来源
（引用的 Wiki 页面列表）

请直接输出 Markdown 内容。"""


# ---------------------------------------------------------------------------
# WikiQueryResult
# ---------------------------------------------------------------------------

class WikiQueryResult:
    def __init__(self, answer: str, confidence: float, cited_pages: list[str],
                 query_type: str = "wiki", backfilled: bool = False):
        self.answer = answer
        self.confidence = confidence
        self.cited_pages = cited_pages
        self.query_type = query_type
        self.backfilled = backfilled

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "cited_pages": self.cited_pages,
            "query_type": self.query_type,
            "backfilled": self.backfilled,
        }


# ---------------------------------------------------------------------------
# WikiQueryRetriever
# ---------------------------------------------------------------------------

class WikiQueryRetriever:
    """Wiki 检索器：index → 页面 → 回答 → 回填"""

    def __init__(self, wiki: WikiKnowledgeBase, llm_func: callable):
        """
        Args:
            wiki: WikiKnowledgeBase 实例
            llm_func: 异步 LLM 函数
        """
        self.wiki = wiki
        self.llm_func = llm_func

    async def query(self, query: str) -> WikiQueryResult:
        """
        Wiki 查询主入口

        Returns:
            WikiQueryResult，包含 answer, confidence, cited_pages
        """
        scale = self.wiki.current_scale()
        logger.info(f"[Wiki Query] scale={scale.value}, query={query[:50]}...")

        # Step 1: 定位相关页面
        pages = await self._locate_pages(query)
        if not pages:
            logger.info("[Wiki Query] No relevant pages found")
            return WikiQueryResult(
                answer="", confidence=0.0, cited_pages=[], query_type="wiki_miss"
            )

        # Step 2: 读取页面内容
        pages_content = self._read_pages(pages)
        if not pages_content:
            return WikiQueryResult(
                answer="", confidence=0.0, cited_pages=[], query_type="wiki_miss"
            )

        # Step 3: 生成答案
        result = await self._generate_answer(query, pages_content)

        # Step 4: 好答案回填
        if (result.confidence >= self.wiki.config.query_confidence_threshold
                and self.wiki.config.backfill_enabled):
            await self._backfill(query, result)
            result.backfilled = True

        return result

    async def _locate_pages(self, query: str) -> list[str]:
        """用 LLM 从 index.md 中定位相关页面"""
        index_content = self.wiki.read_index()

        index_tokens = len(encode_string_by_tiktoken(index_content))
        if index_tokens > self.wiki.config.index_max_tokens:
            logger.warning(f"[Wiki Query] index.md too large ({index_tokens} tokens), consider tiered indexing")

        prompt = PROMPT_INDEX_LOOKUP.format(
            query=query,
            index_content=index_content,
        )
        response = await self.llm_func(prompt)

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                pages = data.get("pages", [])
                logger.info(f"[Wiki Query] Located {len(pages)} pages")
                return pages
        except (json.JSONDecodeError, AttributeError):
            logger.warning("[Wiki Query] Failed to parse page list from LLM")
        return []

    def _read_pages(self, page_paths: list[str]) -> str:
        """读取多个 Wiki 页面，拼成上下文"""
        sections = []
        for path in page_paths:
            path = path.strip()
            if not path.endswith(".md"):
                continue
            parts = path.replace("\\", "/").split("/")
            if len(parts) != 2:
                continue
            category, filename = parts[0], parts[1][:-3]
            content = self.wiki.read_page(category, filename)
            if content:
                sections.append(f"### [{path}]\n{content}")
            else:
                logger.debug(f"[Wiki Query] Page not found: {path}")

        combined = "\n\n---\n\n".join(sections)

        max_context_tokens = 8000
        tokens = encode_string_by_tiktoken(combined)
        if len(tokens) > max_context_tokens:
            from ._utils import decode_tokens_by_tiktoken
            combined = decode_tokens_by_tiktoken(tokens[:max_context_tokens])
            logger.info(f"[Wiki Query] Truncated context to {max_context_tokens} tokens")

        return combined

    async def _generate_answer(self, query: str, pages_content: str) -> WikiQueryResult:
        """用 LLM 基于 Wiki 页面内容生成答案"""
        prompt = PROMPT_WIKI_ANSWER.format(
            query=query,
            pages_content=pages_content,
        )
        response = await self.llm_func(prompt)

        confidence = 0.5
        cited_pages = []
        answer = response

        json_match = re.search(r"\{[^{}]*\"confidence\"[^{}]*\}", response)
        if json_match:
            try:
                meta = json.loads(json_match.group(0))
                confidence = float(meta.get("confidence", 0.5))
                cited_pages = meta.get("cited_pages", [])
                answer = response[:json_match.start()].strip()
            except (json.JSONDecodeError, ValueError):
                pass

        return WikiQueryResult(
            answer=answer,
            confidence=confidence,
            cited_pages=cited_pages,
            query_type="wiki",
        )

    async def _backfill(self, query: str, result: WikiQueryResult):
        """将好答案回填为新 Wiki 页"""
        title_slug = slugify(query[:60])
        if self.wiki.page_exists("queries", title_slug):
            logger.debug(f"[Wiki Query] Backfill skipped: queries/{title_slug}.md already exists")
            return

        prompt = PROMPT_BACKFILL_PAGE.format(
            query=query,
            answer=result.answer[:2000],
            cited_pages=", ".join(result.cited_pages),
            title=query[:60],
        )
        page_content = await self.llm_func(prompt)
        self.wiki.write_page("queries", title_slug, page_content)

        if not self.wiki.index_contains(title_slug):
            self.wiki.add_index_entry("Answered Questions",
                                      f"- [{query[:60]}](queries/{title_slug}.md)")

        self.wiki.append_log([
            f"**Query**: \"{query[:80]}\" — confidence={result.confidence:.2f}",
            f"**Backfilled**: queries/{title_slug}.md",
        ])

        self.wiki.git_commit(f"backfill: {query[:40]}")
        logger.info(f"[Wiki Query] Backfilled: queries/{title_slug}.md")
