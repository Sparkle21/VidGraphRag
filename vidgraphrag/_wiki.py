"""
Wiki Knowledge Base — Karpathy Memex 模式的核心基础设施

职责：
- WikiConfig：配置类，含规模降级开关
- WikiKnowledgeBase：index.md / log.md 管理、页面 CRUD、交叉引用
"""

import os
import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from ._utils import logger, compute_mdhash_id


class WikiScale(str, Enum):
    """Wiki 规模等级，决定检索策略"""
    PURE_WIKI = "pure_wiki"
    WIKI_PLUS_VECTOR = "wiki_plus_vector"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class WikiConfig:
    """
    Wiki 层配置

    规模降级策略：
    - pure_wiki: 纯 Wiki 检索（< 100 视频）
    - wiki_plus_vector: 分级索引 + 轻量向量辅助（100-500 视频）
    - hybrid: Wiki 高层理解 + 底层引擎精确检索（> 500 视频）
    - auto: 根据 Wiki 页面数自动选择
    """
    enabled: bool = True
    wiki_dir: str = "wiki"
    scale_mode: WikiScale = WikiScale.AUTO

    auto_scale_thresholds: dict = field(default_factory=lambda: {
        "pure_wiki_max_pages": 300,
        "wiki_plus_vector_max_pages": 1500,
    })

    index_max_tokens: int = 4000
    query_confidence_threshold: float = 0.8
    backfill_enabled: bool = True
    lint_on_ingest: bool = False
    git_versioning: bool = False

    def resolve_scale(self, page_count: int) -> WikiScale:
        if self.scale_mode != WikiScale.AUTO:
            return self.scale_mode
        t = self.auto_scale_thresholds
        if page_count <= t["pure_wiki_max_pages"]:
            return WikiScale.PURE_WIKI
        elif page_count <= t["wiki_plus_vector_max_pages"]:
            return WikiScale.WIKI_PLUS_VECTOR
        else:
            return WikiScale.HYBRID

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "wiki_dir": self.wiki_dir,
            "scale_mode": self.scale_mode.value,
            "auto_scale_thresholds": self.auto_scale_thresholds,
            "index_max_tokens": self.index_max_tokens,
            "query_confidence_threshold": self.query_confidence_threshold,
            "backfill_enabled": self.backfill_enabled,
            "lint_on_ingest": self.lint_on_ingest,
            "git_versioning": self.git_versioning,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WikiConfig":
        d = dict(d)
        if "scale_mode" in d:
            d["scale_mode"] = WikiScale(d["scale_mode"])
        return cls(**d)

    @classmethod
    def from_file(cls, path: str) -> "WikiConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Wiki page helpers
# ---------------------------------------------------------------------------

def _slugify(name: str, max_len: int = 80) -> str:
    """将名称转为文件名安全的 slug，截断到 max_len 字符"""
    s = name.strip().lower()
    s = re.sub(r"[/\\:*?\"<>|]", "-", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "unnamed"


def _extract_wikilinks(text: str) -> list[str]:
    """提取 markdown 中的 [[wikilink]] 引用"""
    return re.findall(r"\[\[([^\]]+)\]\]", text)


# ---------------------------------------------------------------------------
# WikiKnowledgeBase
# ---------------------------------------------------------------------------

class WikiKnowledgeBase:
    """
    Wiki 知识库核心类

    目录结构：
        wiki/
        ├── index.md
        ├── log.md
        ├── videos/
        ├── entities/
        ├── topics/
        ├── relations/
        └── queries/
    """

    SUBDIRS = ("videos", "entities", "topics", "relations", "queries")

    def __init__(self, working_dir: str, config: WikiConfig | None = None):
        self.config = config or WikiConfig()
        self.wiki_root = os.path.join(working_dir, self.config.wiki_dir)
        self._ensure_dirs()

    # ---- 目录初始化 ----

    def _ensure_dirs(self):
        os.makedirs(self.wiki_root, exist_ok=True)
        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(self.wiki_root, sub), exist_ok=True)
        if not os.path.exists(self.index_path):
            self._write_file(self.index_path, "# Video Knowledge Wiki\n")
        if not os.path.exists(self.log_path):
            self._write_file(self.log_path, "# Change Log\n")

    @property
    def index_path(self) -> str:
        return os.path.join(self.wiki_root, "index.md")

    @property
    def log_path(self) -> str:
        return os.path.join(self.wiki_root, "log.md")

    # ---- 页面 CRUD ----

    def page_path(self, category: str, slug: str) -> str:
        return os.path.join(self.wiki_root, category, f"{slug}.md")

    def page_exists(self, category: str, slug: str) -> bool:
        return os.path.exists(self.page_path(category, slug))

    def read_page(self, category: str, slug: str) -> str | None:
        path = self.page_path(category, slug)
        if not os.path.exists(path):
            return None
        return self._read_file(path)

    def write_page(self, category: str, slug: str, content: str):
        path = self.page_path(category, slug)
        self._write_file(path, content)
        logger.debug(f"Wiki page written: {category}/{slug}.md")

    def append_to_page(self, category: str, slug: str, section: str):
        existing = self.read_page(category, slug) or ""
        self.write_page(category, slug, existing.rstrip("\n") + "\n\n" + section + "\n")

    def delete_page(self, category: str, slug: str):
        path = self.page_path(category, slug)
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Wiki page deleted: {category}/{slug}.md")

    def list_pages(self, category: str) -> list[str]:
        d = os.path.join(self.wiki_root, category)
        if not os.path.isdir(d):
            return []
        return [f[:-3] for f in os.listdir(d) if f.endswith(".md")]

    # ---- index.md 管理 ----

    def read_index(self) -> str:
        return self._read_file(self.index_path)

    def write_index(self, content: str):
        self._write_file(self.index_path, content)

    def add_index_entry(self, section: str, line: str):
        """在 index.md 的指定 section 下追加一行条目"""
        index = self.read_index()
        section_header = f"## {section}"

        if section_header not in index:
            index = index.rstrip("\n") + f"\n\n{section_header}\n{line}\n"
        else:
            parts = index.split(section_header, 1)
            after = parts[1]
            next_section = re.search(r"\n## ", after)
            if next_section:
                insert_pos = next_section.start()
                after = after[:insert_pos] + f"\n{line}" + after[insert_pos:]
            else:
                after = after.rstrip("\n") + f"\n{line}\n"
            index = parts[0] + section_header + after

        self.write_index(index)

    def index_contains(self, slug: str) -> bool:
        return slug in self.read_index()

    # ---- log.md 管理 ----

    def read_log(self) -> str:
        return self._read_file(self.log_path)

    def append_log(self, entries: list[str]):
        """追加一条带时间戳的日志"""
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
        block = f"\n## {ts}\n" + "\n".join(f"- {e}" for e in entries) + "\n"
        existing = self._read_file(self.log_path)
        self._write_file(self.log_path, existing.rstrip("\n") + "\n" + block)

    # ---- 统计 ----

    def page_count(self) -> int:
        total = 0
        for sub in self.SUBDIRS:
            total += len(self.list_pages(sub))
        return total

    def current_scale(self) -> WikiScale:
        return self.config.resolve_scale(self.page_count())

    def stats(self) -> dict:
        counts = {sub: len(self.list_pages(sub)) for sub in self.SUBDIRS}
        return {
            "total_pages": sum(counts.values()),
            "by_category": counts,
            "scale_mode": self.current_scale().value,
            "index_size_chars": len(self.read_index()),
        }

    # ---- 交叉引用 ----

    def find_broken_wikilinks(self) -> list[dict]:
        """扫描所有页面，找出引用了不存在页面的 [[wikilink]]"""
        all_slugs = set()
        for sub in self.SUBDIRS:
            for slug in self.list_pages(sub):
                all_slugs.add(slug)

        broken = []
        for sub in self.SUBDIRS:
            for slug in self.list_pages(sub):
                content = self.read_page(sub, slug) or ""
                for link in _extract_wikilinks(content):
                    link_slug = _slugify(link)
                    if link_slug not in all_slugs:
                        broken.append({
                            "source": f"{sub}/{slug}.md",
                            "broken_link": link,
                            "slug": link_slug,
                        })
        return broken

    def find_orphan_pages(self) -> list[str]:
        """找出 index.md 中未引用的页面"""
        index = self.read_index()
        orphans = []
        for sub in self.SUBDIRS:
            for slug in self.list_pages(sub):
                ref = f"{sub}/{slug}.md"
                if ref not in index and slug not in index:
                    orphans.append(ref)
        return orphans

    # ---- Git 版本控制 ----

    def git_commit(self, message: str):
        if not self.config.git_versioning:
            return
        import subprocess
        try:
            subprocess.run(["git", "add", "."], cwd=self.wiki_root, check=True,
                           capture_output=True)
            subprocess.run(["git", "commit", "-m", message], cwd=self.wiki_root,
                           check=True, capture_output=True)
            logger.debug(f"Wiki git commit: {message}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Wiki git commit failed: {e}")
        except FileNotFoundError:
            logger.warning("git not found, skipping wiki versioning")

    def git_init(self):
        if not self.config.git_versioning:
            return
        git_dir = os.path.join(self.wiki_root, ".git")
        if os.path.exists(git_dir):
            return
        import subprocess
        try:
            subprocess.run(["git", "init"], cwd=self.wiki_root, check=True,
                           capture_output=True)
            logger.info(f"Wiki git repo initialized at {self.wiki_root}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Wiki git init failed: {e}")

    # ---- 底层 IO ----

    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _write_file(path: str, content: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


# 导出 slugify 供其他模块使用
slugify = _slugify
