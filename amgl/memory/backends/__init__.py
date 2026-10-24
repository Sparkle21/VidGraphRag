"""本地事务存储；协议不把搜索近似度等同于事实验证。"""
from typing import Protocol

from ..envelope import MemoryEnvelope


class MemoryBackend(Protocol):
    def read(self, tenant: str, mem_id: str) -> MemoryEnvelope: ...
    def read_many(self, tenant: str, mem_ids: list[str], *, timings: dict | None = None) -> list[MemoryEnvelope]: ...
    def list_memories(self, tenant: str, limit: int = 100, *, timings: dict | None = None) -> list[MemoryEnvelope]: ...
    def page_memories(self, tenant: str, limit: int = 100, *, highwater: int | None = None,
                      before_id: str | None = None, timings: dict | None = None) -> tuple[list[MemoryEnvelope], int, str | None]: ...


from .file_kv import FileKVBackend

__all__ = ["MemoryBackend", "FileKVBackend"]
