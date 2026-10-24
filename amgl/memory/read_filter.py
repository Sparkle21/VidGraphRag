"""只过滤已取回的封套，O(n)；不承担检索、鉴权或缓存一致性。"""
from __future__ import annotations

from datetime import datetime, timezone
from pydantic import Field

from .envelope import FilteredMemory, MemoryType, Model, RetrievedMemory, State


class ReadPolicy(Model):
    min_trust: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    allow_provisional: bool = False
    allow_pii: bool = False
    allow_conflicts: bool = False


def filter_results(results: list[RetrievedMemory], policy: ReadPolicy,
                   tenant_id: str, *, now: datetime | None = None) -> list[FilteredMemory]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("过滤时间必须包含时区")
    output = []
    for result in results:
        mem = result.memory
        if mem.tenant_id != tenant_id:
            continue
        if mem.lifecycle.state not in (State.PROVISIONAL, State.COMMITTED):
            continue
        if mem.expires_at is not None and mem.expires_at <= now:
            continue
        if mem.sensitivity == "pii" and not policy.allow_pii:
            continue
        if mem.governance.conflicts and not policy.allow_conflicts:
            continue
        provisional = mem.lifecycle.state == State.PROVISIONAL
        if provisional and mem.memory_type == MemoryType.PROCEDURAL:
            continue
        if provisional and not policy.allow_provisional:
            continue
        if mem.trust.score < policy.min_trust:
            continue
        annotations = []
        if provisional or mem.governance.review_required:
            annotations.append("unverified")
        annotations.extend(f"conflict_with:{i}" for i in mem.governance.conflicts)
        output.append(FilteredMemory(memory=mem, score=result.score, annotations=annotations))
    return output
