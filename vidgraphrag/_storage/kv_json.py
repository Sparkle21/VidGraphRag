import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .._utils import load_json, logger, write_json
from ..base import (
    BaseKVStorage,
)


@dataclass
class ShotKVStorage(BaseKVStorage):
    """
    Shot 级详细信息存储
    
    数据结构：
    {
      "s_042": {
        "shot_id": "s_042",
        "video_id": "v_1001",
        "time_start": 73.0,
        "time_end": 77.0,
        "duration": 4.0,
        
        "vectors": {
          "visual_vec": [...],    # 1024 维
          "audio_vec": [...],     # 1536 维
          "vlm_vec": [...]        # 1536 维
        },
        
        "content": {
          "asr_text": "...",
          "vlm_shot_summary": {
            "subject": ["钢铁侠"],
            "action": ["飞行"],
            "environment": ["城市上空"],
            "ocr": [],
            "caption": "钢铁侠在黄昏时分飞向斯塔克大厦"
          }
        },
        
        "cognitive_links": {
          "segment_id": "seg_002",
          "narrative_summary": "钢铁侠教导蜘蛛侠飞行",
          "entity_mentions": ["钢铁侠", "蜘蛛侠"],
          "event_type": "教学事件"
        }
      }
    }
    """
    
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load Shot KV Storage {self.namespace} with {len(self._data)} shots")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, shot_id: str) -> Optional[Dict]:
        """按 ID 获取 Shot"""
        return self._data.get(shot_id, None)

    async def get_by_ids(self, ids: list[str], fields=None) -> list[Optional[Dict]]:
        if fields is None:
            return [self._data.get(shot_id, None) for shot_id in ids]
        return [
            (
                {k: v for k, v in self._data[shot_id].items() if k in fields}
                if self._data.get(shot_id, None)
                else None
            )
            for shot_id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: Dict[str, Dict]):
        """写入 Shot 数据"""
        self._data.update(data)

    async def drop(self):
        self._data = {}

    async def filter_by_time_range(self, start: float, end: float) -> List[Dict]:
        """按时间范围过滤"""
        results = []
        for shot_id, shot_data in self._data.items():
            shot_start = shot_data.get('time_start', 0)
            shot_end = shot_data.get('time_end', 0)
            # 检查是否有重叠
            if shot_start < end and shot_end > start:
                results.append(shot_data)
        # 按时间排序
        results.sort(key=lambda x: x.get('time_start', 0))
        return results

    async def search_by_entity(self, entity_name: str) -> List[Dict]:
        """按实体名检索（通过 cognitive_links）"""
        results = []
        for shot_id, shot_data in self._data.items():
            cognitive_links = shot_data.get('cognitive_links', {})
            entity_mentions = cognitive_links.get('entity_mentions', [])
            # 检查实体是否在 mentions 中
            if entity_name in entity_mentions:
                results.append(shot_data)
        # 按时间排序
        results.sort(key=lambda x: x.get('time_start', 0))
        return results


@dataclass
class SegmentKVStorage(BaseKVStorage):
    """
    Segment 级认知内容存储
    
    数据结构：
    {
      "seg_002": {
        "segment_id": "seg_002",
        "video_id": "v_1001",
        "time_start": 60.0,
        "time_end": 90.0,
        "duration": 30.0,
        
        "shot_ids": ["s_035", "s_036", ..., "s_044"],
        
        "cognitive_content": {
          "narrative_text": "钢铁侠耐心指导蜘蛛侠学习飞行...",
          "entities": [
            {"name": "钢铁侠", "type": "PERSON"},
            {"name": "蜘蛛侠", "type": "PERSON"},
            {"name": "飞行教学", "type": "EVENT"}
          ],
          "relations": [
            {"head": "钢铁侠", "relation": "MENTOR_OF", "tail": "蜘蛛侠"}
          ],
          "themes": ["成长", "传承", "责任"]
        },
        
        "summary_vectors": {
          "narrative_vec": [...],    # narrative_text 的 embedding
          "theme_vec": [...]         # themes 的 embedding
        }
      }
    }
    """
    
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load Segment KV Storage {self.namespace} with {len(self._data)} segments")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, segment_id: str) -> Optional[Dict]:
        """按 ID 获取 Segment"""
        return self._data.get(segment_id, None)

    async def get_by_ids(self, ids: list[str], fields=None) -> list[Optional[Dict]]:
        if fields is None:
            return [self._data.get(segment_id, None) for segment_id in ids]
        return [
            (
                {k: v for k, v in self._data[segment_id].items() if k in fields}
                if self._data.get(segment_id, None)
                else None
            )
            for segment_id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: Dict[str, Dict]):
        """写入 Segment 数据"""
        self._data.update(data)

    async def drop(self):
        self._data = {}

    async def search_by_theme(self, theme: str) -> List[Dict]:
        """按主题检索"""
        results = []
        for segment_id, segment_data in self._data.items():
            cognitive_content = segment_data.get('cognitive_content', {})
            themes = cognitive_content.get('themes', [])
            # 检查主题是否匹配
            if theme in themes:
                results.append(segment_data)
        # 按时间排序
        results.sort(key=lambda x: x.get('time_start', 0))
        return results

    async def get_related_segments(self, entity_names: List[str]) -> List[Dict]:
        """查找包含指定实体的 Segments"""
        results = []
        for segment_id, segment_data in self._data.items():
            cognitive_content = segment_data.get('cognitive_content', {})
            entities = cognitive_content.get('entities', [])
            # 提取所有实体名称
            entity_name_list = [e.get('name', '') for e in entities]
            # 检查是否有任意一个实体匹配
            if any(name in entity_name_list for name in entity_names):
                results.append(segment_data)
        # 按时间排序
        results.sort(key=lambda x: x.get('time_start', 0))
        return results


@dataclass
class JsonKVStorage(BaseKVStorage):
    """
    保持向后兼容，增加可选的双层模式支持
    """
    
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")
        
        # 新增：双层模式支持
        self.enable_dual_layer = self.global_config.get("enable_dual_layer", False)
        if self.enable_dual_layer:
            # 初始化 Shot 和 Segment 存储
            shot_config = self.global_config.copy()
            shot_config["working_dir"] = working_dir
            self.shot_storage = ShotKVStorage(
                namespace=f"{self.namespace}_shots",
                global_config=shot_config
            )
            
            segment_config = self.global_config.copy()
            segment_config["working_dir"] = working_dir
            self.segment_storage = SegmentKVStorage(
                namespace=f"{self.namespace}_segments",
                global_config=segment_config
            )
            logger.info(f"Dual-layer mode enabled for {self.namespace}")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)
        # 如果是双层模式，也保存 shot 和 segment 数据
        if self.enable_dual_layer:
            await self.shot_storage.index_done_callback()
            await self.segment_storage.index_done_callback()

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    async def drop(self):
        self._data = {}


@dataclass
class CommunityReportStorage(BaseKVStorage):
    """社区报告存储"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load Community Reports {self.namespace} with {len(self._data)} reports")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id) -> Optional[Dict]:
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None) -> list[Optional[Dict]]:
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def upsert(self, data: dict):
        self._data.update(data)

    async def drop(self):
        self._data = {}
