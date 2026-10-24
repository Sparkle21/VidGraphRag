"""
Wiki Ingest Writer — 从 VidGraphRAG 底层产出物蒸馏 Wiki 页

流程（单个视频 Ingest 后调用）：
1. 读 index.md（了解已有知识全貌）
2. 生成/更新 Wiki 页：
   a) videos/{name}.md — 视频总结页
   b) entities/_summary.md — 实体汇总页（全部实体一页，按类型分组）
   c) topics/{name}.md — 跨视频主题页（LLM 判断）
   d) relations/{a}-{b}.md — 高价值关系页
3. 更新 index.md
4. 追加 log.md

单次 Ingest 典型 LLM 调用 ~3-5 次
"""

import json
import re
from typing import Optional

from ._wiki import WikiKnowledgeBase, slugify
from ._utils import logger


def _normalize_entity_name(name: str) -> str:
    """归一化实体名：去引号、去首尾空白"""
    name = name.strip().strip("\"'""''`")
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_VIDEO_SUMMARY = """你是一个视频内容分析专家。请根据以下视频处理结果，生成一份结构化的视频总结页（Markdown 格式）。

## 视频名称
{video_name}

## Segment 叙事（按时间顺序）
{segment_narratives}

## 识别到的实体
{entity_list}

## 实体关系
{relation_list}

## 输出要求
请生成以下结构的 Markdown：

# {video_name}

## 概述
（2-3 句话总结视频核心内容）

## 时间轴
（按时间顺序列出关键场景/事件，格式：`- **MM:SS - MM:SS**: 描述`）

## 关键人物/实体
（列出主要出场实体，用 [[实体名]] 格式链接）

## 主题
（1-2 个核心主题标签）

请直接输出 Markdown 内容，不要加额外解释。"""

PROMPT_ENTITIES_OVERVIEW = """你是一个知识库编辑。请将以下实体列表汇总为一个结构化的 Markdown 实体总览页。

## 视频来源
{video_name}

## 实体数据
{entities_data}

## 输出要求
请按实体类型分组，生成以下结构的 Markdown：

# 实体总览

## [类型名]（如：人物、地点、组织等）

### 实体名
- **描述**: 一句话描述
- **出现**: [[视频名]] 时间段
- **关联**: [[关联实体]]（关系描述）

要求：
- 按类型分组，每个类型一个二级标题
- 每个实体一个三级标题，保持简洁（描述不超过两句话）
- 用 [[实体名]] 格式做交叉链接
- 直接输出 Markdown，不要额外解释"""

PROMPT_ENTITIES_MERGE = """你是一个知识库编辑。以下是已有的实体总览页和新视频中发现的新实体。请将新实体合并进已有页面。

## 已有实体总览
{existing_overview}

## 新视频
{video_name}

## 新增/更新的实体
{new_entities_data}

## 输出要求
- 新实体：追加到对应类型分组下
- 已有实体：在"出现"中补充新视频记录，如有新关联也补充
- 保持按类型分组的结构，保持简洁
- 如有描述矛盾，以新信息为准
- 直接输出完整的更新后 Markdown 页面"""

PROMPT_TOPIC_DISCOVERY = """你是一个视频内容分析专家。请判断以下新视频是否应该触发新的跨视频主题页。

## 新视频
{video_name}: {video_summary}

## 已有主题
{existing_topics}

## 已有视频
{existing_videos}

## 输出要求
以 JSON 格式回答：
{{
  "new_topics": [
    {{"name": "主题名", "description": "简要描述", "related_videos": ["视频1", "视频2"]}}
  ],
  "updated_topics": [
    {{"name": "已有主题名", "add_video": "{video_name}", "reason": "为什么相关"}}
  ]
}}

如果没有新主题也没有需要更新的主题，返回空数组。只返回 JSON，不要其他内容。"""


# ---------------------------------------------------------------------------
# WikiIngestWriter
# ---------------------------------------------------------------------------

class WikiIngestWriter:
    """从 VidGraphRAG 处理产出物蒸馏 Wiki 页"""

    def __init__(self, wiki: WikiKnowledgeBase, llm_func: callable):
        """
        Args:
            wiki: WikiKnowledgeBase 实例
            llm_func: 异步 LLM 函数，签名 async (prompt, **kwargs) -> str
        """
        self.wiki = wiki
        self.llm_func = llm_func

    async def ingest_video(
        self,
        video_name: str,
        segment_data: dict,
        graph_data: Optional[dict] = None,
        community_reports: Optional[list] = None,
    ):
        """
        完整的 Ingest 流程

        Args:
            video_name: 视频文件名（不含扩展名）
            segment_data: {segment_id: {cognitive_content: {narrative_text, entities, ...}, ...}}
            graph_data: 知识图谱数据 {nodes: [...], edges: [...]}（可选）
            community_reports: 社区报告列表（可选）
        """
        log_entries = []
        slug = slugify(video_name)

        # 收集 segment narratives + segment 实体候选（仅用于给图谱实体补时间戳）
        narratives = []
        _seg_candidates = {}  # normalized_lower → {segments: [...]}
        all_relations = []

        for seg_id, seg in segment_data.items():
            cc = seg.get("cognitive_content", {})
            narrative = cc.get("narrative_text", "")
            if narrative:
                narratives.append({
                    "segment_id": seg_id,
                    "time_start": seg.get("time_start", 0),
                    "time_end": seg.get("time_end", 0),
                    "text": narrative,
                })
            for ent in cc.get("entities", []):
                name = _normalize_entity_name(ent.get("name", ""))
                if not name:
                    continue
                key = name.lower()
                if key not in _seg_candidates:
                    _seg_candidates[key] = []
                _seg_candidates[key].append({
                    "segment_id": seg_id,
                    "time_start": seg.get("time_start", 0),
                    "time_end": seg.get("time_end", 0),
                })

        # 只取图谱实体（经过 NER 的高质量实体），segment 实体仅用于补出现时间
        all_entities = {}
        _norm_to_canonical = {}

        if graph_data:
            for node in graph_data.get("nodes", []):
                name = _normalize_entity_name(node.get("entity_name", ""))
                if not name:
                    continue
                norm_key = name.lower()
                _norm_to_canonical[norm_key] = name
                all_entities[name] = {
                    "type": node.get("entity_type", "") or "UNKNOWN",
                    "description": node.get("description", ""),
                    "segments": _seg_candidates.get(norm_key, []),
                    "relations": [],
                }

            for edge in graph_data.get("edges", []):
                src = _normalize_entity_name(edge.get("src_id", edge.get("source", "")))
                tgt = _normalize_entity_name(edge.get("tgt_id", edge.get("target", "")))
                desc = edge.get("description", "")
                src = _norm_to_canonical.get(src.lower(), src)
                tgt = _norm_to_canonical.get(tgt.lower(), tgt)
                if src and tgt:
                    all_relations.append({"source": src, "target": tgt, "description": desc})
                    if src in all_entities:
                        all_entities[src]["relations"].append({"target": tgt, "description": desc})
                    if tgt in all_entities:
                        all_entities[tgt]["relations"].append({"target": src, "description": desc})

        seg_only_count = len(_seg_candidates) - len(
            set(_seg_candidates) & {k.lower() for k in all_entities}
        )
        logger.info(
            f"[Wiki Ingest] Entities: {len(all_entities)} from graph, "
            f"{seg_only_count} segment-only discarded"
        )

        # Step 1: 生成视频总结页
        logger.info(f"[Wiki Ingest] Generating video page: {slug}")
        video_page = await self._generate_video_page(video_name, narratives, all_entities, all_relations)
        self.wiki.write_page("videos", slug, video_page)
        log_entries.append(f"**Created**: videos/{slug}.md")

        # Step 2: 生成/更新实体汇总页（一页 entities/_summary.md）
        if all_entities:
            await self._write_entities_overview(video_name, all_entities)
            log_entries.append(f"**Entities**: entities/_summary.md — {len(all_entities)} entities")

        # Step 3: 发现主题
        topic_changes = await self._discover_topics(video_name, video_page)
        for tc in topic_changes.get("new_topics", []):
            t_slug = slugify(tc["name"])
            self.wiki.write_page("topics", t_slug, f"# {tc['name']}\n\n{tc['description']}\n\n## Related Videos\n- [[{video_name}]]\n")
            log_entries.append(f"**Discovered**: topics/{t_slug}.md")
        for tc in topic_changes.get("updated_topics", []):
            t_slug = slugify(tc["name"])
            if self.wiki.page_exists("topics", t_slug):
                self.wiki.append_to_page("topics", t_slug, f"- [[{video_name}]]: {tc.get('reason', '')}")
                log_entries.append(f"**Updated**: topics/{t_slug}.md")

        # Step 4: 关系汇总页（一页 relations/_summary.md）
        if all_relations:
            self._write_relations_overview(video_name, all_relations)
            log_entries.append(f"**Relations**: relations/_summary.md — {len(all_relations)} relations")

        # Step 5: 更新 index.md
        self._update_index(video_name, slug, all_entities, all_relations, topic_changes)
        log_entries.append("**Updated**: index.md")

        # Step 6: 追加 log.md
        seg_count = len(segment_data)
        total_duration = sum(s.get("duration", 0) for s in segment_data.values())
        log_entries.insert(0, f"**Ingested**: {video_name} — {seg_count} segments, {total_duration:.0f}s")
        self.wiki.append_log(log_entries)

        # Step 7: Git commit
        self.wiki.git_commit(f"ingest: {video_name}")

        logger.info(f"[Wiki Ingest] Completed: {video_name} — {len(log_entries)} changes")

    # ---- 内部方法 ----

    async def _generate_video_page(
        self, video_name: str, narratives: list, entities: dict, relations: list
    ) -> str:
        seg_text = "\n".join(
            f"- **{self._format_time(n['time_start'])} - {self._format_time(n['time_end'])}**: {n['text']}"
            for n in narratives
        ) or "（无叙事内容）"

        ent_text = "\n".join(
            f"- {name} ({data['type']}): {data['description'][:100]}"
            for name, data in list(entities.items())[:20]
        ) or "（无实体）"

        rel_text = "\n".join(
            f"- {r['source']} → {r['target']}: {r['description'][:80]}"
            for r in relations[:15]
        ) or "（无关系）"

        prompt = PROMPT_VIDEO_SUMMARY.format(
            video_name=video_name,
            segment_narratives=seg_text,
            entity_list=ent_text,
            relation_list=rel_text,
        )
        return await self.llm_func(prompt)

    async def _write_entities_overview(
        self, video_name: str, entities: dict
    ):
        """生成或合并实体汇总页（entities/_summary.md），一次 LLM 调用"""
        entities_data = self._format_entities_for_prompt(entities, video_name)
        existing = self.wiki.read_page("entities", "_summary")

        if existing:
            prompt = PROMPT_ENTITIES_MERGE.format(
                existing_overview=existing[:6000],
                video_name=video_name,
                new_entities_data=entities_data,
            )
        else:
            prompt = PROMPT_ENTITIES_OVERVIEW.format(
                video_name=video_name,
                entities_data=entities_data,
            )

        content = await self.llm_func(prompt)
        self.wiki.write_page("entities", "_summary", content)
        logger.info(f"[Wiki Ingest] Entities overview: {len(entities)} entities → entities/_summary.md")

    def _format_entities_for_prompt(self, entities: dict, video_name: str) -> str:
        """将实体字典格式化为 prompt 友好的文本"""
        lines = []
        by_type = {}
        for name, data in entities.items():
            etype = data.get("type", "UNKNOWN")
            by_type.setdefault(etype, []).append((name, data))

        for etype, ent_list in by_type.items():
            lines.append(f"\n### {etype}")
            for name, data in ent_list:
                desc = data.get("description", "")[:200]
                segs = ", ".join(
                    f"{self._format_time(s['time_start'])}-{self._format_time(s['time_end'])}"
                    for s in data.get("segments", [])
                ) or "未知"
                rels = "; ".join(
                    f"{r['target']}({r['description'][:40]})"
                    for r in data.get("relations", [])[:5]
                ) or "无"
                lines.append(f"- **{name}**: {desc}")
                lines.append(f"  出现: [[{video_name}]] {segs}")
                lines.append(f"  关联: {rels}")
        return "\n".join(lines)

    def _write_relations_overview(self, video_name: str, relations: list):
        """生成或追加关系汇总页（relations/_summary.md），纯模板，不调 LLM"""
        existing = self.wiki.read_page("relations", "_summary") or "# 关系总览\n"

        new_lines = [f"\n## 来自 [[{video_name}]]"]
        for rel in relations:
            src, tgt = rel["source"], rel["target"]
            desc = rel.get("description", "")
            new_lines.append(f"- [[{src}]] → [[{tgt}]]: {desc}")

        self.wiki.write_page(
            "relations", "_summary",
            existing.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"
        )
        logger.info(f"[Wiki Ingest] Relations overview: {len(relations)} relations → relations/_summary.md")

    async def _discover_topics(self, video_name: str, video_summary: str) -> dict:
        existing_topics = self.wiki.list_pages("topics")
        existing_videos = self.wiki.list_pages("videos")

        topics_str = ", ".join(existing_topics) if existing_topics else "（暂无主题）"
        videos_str = ", ".join(existing_videos) if existing_videos else "（暂无其他视频）"

        prompt = PROMPT_TOPIC_DISCOVERY.format(
            video_name=video_name,
            video_summary=video_summary[:1000],
            existing_topics=topics_str,
            existing_videos=videos_str,
        )
        response = await self.llm_func(prompt)

        try:
            # 提取 JSON
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"[Wiki Ingest] Failed to parse topic discovery response")
        return {"new_topics": [], "updated_topics": []}

    def _update_index(self, video_name: str, slug: str, entities: dict,
                      relations: list, topic_changes: dict):
        if not self.wiki.index_contains(slug):
            self.wiki.add_index_entry("Videos", f"- [{video_name}](videos/{slug}.md)")

        if entities and not self.wiki.index_contains("entities/_summary"):
            self.wiki.add_index_entry("Entities", f"- [实体总览](entities/_summary.md)")

        if relations and not self.wiki.index_contains("relations/_summary"):
            self.wiki.add_index_entry("Relations", f"- [关系总览](relations/_summary.md)")

        for tc in topic_changes.get("new_topics", []):
            t_slug = slugify(tc["name"])
            if not self.wiki.index_contains(t_slug):
                self.wiki.add_index_entry("Topics", f"- [{tc['name']}](topics/{t_slug}.md)")

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
