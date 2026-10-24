"""
Wiki Lint — 定期体检

检查项：
1. 矛盾检测：同一实体在不同页面的描述矛盾
2. 过时内容：长期未更新的页面
3. 孤儿页：index.md 中未引用的页面
4. 缺失交叉引用：页面提到实体但没有链接
5. index 膨胀：index.md 超过阈值，建议拆分
"""

import re
from datetime import datetime
from typing import Optional

from ._wiki import WikiKnowledgeBase, slugify, _extract_wikilinks
from ._utils import logger, encode_string_by_tiktoken


PROMPT_CONTRADICTION_CHECK = """你是一个知识库质量检查员。请检查以下两段关于同一实体的描述是否存在矛盾。

## 实体名称
{entity_name}

## 描述 1（来自 {source1}）
{desc1}

## 描述 2（来自 {source2}）
{desc2}

## 输出要求
以 JSON 格式回答：
{{"has_contradiction": true/false, "detail": "矛盾描述（如无矛盾则为空字符串）", "suggestion": "修复建议"}}

只返回 JSON。"""


class LintIssue:
    def __init__(self, level: str, category: str, message: str,
                 source: str = "", suggestion: str = ""):
        self.level = level      # "error", "warning", "info"
        self.category = category
        self.message = message
        self.source = source
        self.suggestion = suggestion

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "source": self.source,
            "suggestion": self.suggestion,
        }

    def __str__(self):
        icon = {"error": "[E]", "warning": "[W]", "info": "[I]"}.get(self.level, "[?]")
        return f"{icon} [{self.category}] {self.message} ({self.source})"


class WikiLinter:
    """Wiki 体检器"""

    def __init__(self, wiki: WikiKnowledgeBase, llm_func: Optional[callable] = None):
        self.wiki = wiki
        self.llm_func = llm_func

    async def lint(self) -> list[LintIssue]:
        """运行所有检查，返回问题列表"""
        issues = []

        issues.extend(self._check_orphan_pages())
        issues.extend(self._check_broken_wikilinks())
        issues.extend(self._check_missing_crossrefs())
        issues.extend(self._check_index_bloat())

        if self.llm_func:
            issues.extend(await self._check_contradictions())

        self._write_report(issues)
        return issues

    def _check_orphan_pages(self) -> list[LintIssue]:
        orphans = self.wiki.find_orphan_pages()
        return [
            LintIssue(
                level="warning",
                category="orphan",
                message=f"Page not referenced in index.md: {page}",
                source=page,
                suggestion=f"Add to index.md or merge into another page",
            )
            for page in orphans
        ]

    def _check_broken_wikilinks(self) -> list[LintIssue]:
        broken = self.wiki.find_broken_wikilinks()
        return [
            LintIssue(
                level="warning",
                category="broken_link",
                message=f"Broken wikilink [[{b['broken_link']}]]",
                source=b["source"],
                suggestion=f"Create page for '{b['broken_link']}' or fix the link",
            )
            for b in broken
        ]

    def _check_missing_crossrefs(self) -> list[LintIssue]:
        """检查页面中提到的实体名但没有用 [[]] 链接的情况"""
        issues = []
        entity_names = self._extract_entity_names_from_summary()

        for sub in self.wiki.SUBDIRS:
            for page_slug in self.wiki.list_pages(sub):
                if sub == "entities" and page_slug == "_summary":
                    continue
                content = self.wiki.read_page(sub, page_slug) or ""
                existing_links = {link.lower() for link in _extract_wikilinks(content)}

                for ent_name in entity_names:
                    if ent_name in content.lower() and ent_name not in existing_links:
                        issues.append(LintIssue(
                            level="info",
                            category="missing_crossref",
                            message=f"Mentions '{ent_name}' without [[link]]",
                            source=f"{sub}/{page_slug}.md",
                            suggestion=f"Add [[{ent_name}]] link",
                        ))
        return issues

    def _extract_entity_names_from_summary(self) -> set[str]:
        """从 entities/_summary.md 的三级标题中提取实体名"""
        content = self.wiki.read_page("entities", "_summary") or ""
        names = set()
        for line in content.split("\n"):
            if line.startswith("### "):
                name = line[4:].strip()
                if name:
                    names.add(name.lower())
        return names

    def _check_index_bloat(self) -> list[LintIssue]:
        index_content = self.wiki.read_index()
        token_count = len(encode_string_by_tiktoken(index_content))
        threshold = self.wiki.config.index_max_tokens

        if token_count > threshold:
            return [LintIssue(
                level="warning",
                category="index_bloat",
                message=f"index.md has {token_count} tokens (threshold: {threshold})",
                source="index.md",
                suggestion="Split into tiered index: index/videos.md, index/entities.md, index/topics.md",
            )]

        if token_count > threshold * 0.8:
            return [LintIssue(
                level="info",
                category="index_bloat",
                message=f"index.md approaching limit: {token_count}/{threshold} tokens",
                source="index.md",
                suggestion="Consider splitting soon",
            )]

        return []

    async def _check_contradictions(self) -> list[LintIssue]:
        """用 LLM 检查实体描述矛盾（基于汇总页 vs 视频/主题页）"""
        issues = []
        entity_names = self._extract_entity_names_from_summary()
        summary_content = self.wiki.read_page("entities", "_summary") or ""
        if not summary_content:
            return issues

        for entity_name in list(entity_names)[:20]:
            entity_context = self._extract_context(summary_content, entity_name, window=300)
            if not entity_context:
                continue

            mentions = []
            for sub in ("videos", "topics", "queries"):
                for page_slug in self.wiki.list_pages(sub):
                    page_content = self.wiki.read_page(sub, page_slug) or ""
                    if entity_name in page_content.lower():
                        relevant = self._extract_context(page_content, entity_name, window=200)
                        if relevant:
                            mentions.append((f"{sub}/{page_slug}.md", relevant))

            for source, mention_text in mentions[:3]:
                try:
                    prompt = PROMPT_CONTRADICTION_CHECK.format(
                        entity_name=entity_name,
                        source1="entities/_summary.md",
                        desc1=entity_context[:500],
                        source2=source,
                        desc2=mention_text[:500],
                    )
                    response = await self.llm_func(prompt)
                    import json
                    json_match = re.search(r"\{.*\}", response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        if data.get("has_contradiction"):
                            issues.append(LintIssue(
                                level="error",
                                category="contradiction",
                                message=f"Contradiction about '{entity_name}': {data.get('detail', '')}",
                                source=f"entities/_summary.md vs {source}",
                                suggestion=data.get("suggestion", ""),
                            ))
                except Exception as e:
                    logger.warning(f"[Wiki Lint] Contradiction check failed for {entity_name}: {e}")

        return issues

    @staticmethod
    def _extract_context(text: str, keyword: str, window: int = 200) -> str:
        idx = text.lower().find(keyword.lower())
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        return text[start:end]

    def _write_report(self, issues: list[LintIssue]):
        if not issues:
            report = "# Wiki Lint Report\n\nNo issues found.\n"
        else:
            errors = [i for i in issues if i.level == "error"]
            warnings = [i for i in issues if i.level == "warning"]
            infos = [i for i in issues if i.level == "info"]

            lines = [
                "# Wiki Lint Report",
                "",
                f"**Total**: {len(issues)} issues — {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info",
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
            ]

            if errors:
                lines.append("## Errors")
                for i in errors:
                    lines.append(f"- **{i.source}**: {i.message}")
                    if i.suggestion:
                        lines.append(f"  - Suggestion: {i.suggestion}")
                lines.append("")

            if warnings:
                lines.append("## Warnings")
                for i in warnings:
                    lines.append(f"- **{i.source}**: {i.message}")
                    if i.suggestion:
                        lines.append(f"  - Suggestion: {i.suggestion}")
                lines.append("")

            if infos:
                lines.append("## Info")
                for i in infos:
                    lines.append(f"- **{i.source}**: {i.message}")
                lines.append("")

            report = "\n".join(lines)

        report_path = self.wiki.wiki_root
        WikiKnowledgeBase._write_file(
            f"{report_path}/lint-report.md", report
        )
        logger.info(f"[Wiki Lint] Report written: {len(issues)} issues")
