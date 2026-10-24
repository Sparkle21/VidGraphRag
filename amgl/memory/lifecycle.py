"""人工晋升必须显式审阅；来源上限与冲突不能被状态接口绕过。"""
from amgl.errors import Conflict
from .backends import FileKVBackend
from .envelope import MemoryEnvelope, SourceTier, State, utcnow

TRANSITIONS = {
    State.CANDIDATE: {State.PROVISIONAL},
    State.PROVISIONAL: {State.COMMITTED, State.STALE},
    State.COMMITTED: {State.STALE},
    State.STALE: {State.PROVISIONAL, State.ARCHIVED},
    State.ARCHIVED: set(),
}


class LifecycleManager:
    def __init__(self, backend: FileKVBackend):
        self.backend = backend

    def transition(self, tenant: str, mem_id: str, to_state: State,
                   expected_version: int, *, actor: str, reviewed: bool = False) -> MemoryEnvelope:
        def change(mem):
            if to_state not in TRANSITIONS[mem.lifecycle.state]:
                raise Conflict("非法记忆状态迁移")
            if to_state == State.COMMITTED:
                if not reviewed:
                    raise Conflict("确认记忆必须先完成人工审阅")
                if mem.provenance.source_tier == SourceTier.THIRD_PARTY:
                    raise Conflict("第三方内容不能直接晋升为已确认记忆")
                if mem.governance.conflicts:
                    raise Conflict("存在未解决冲突，不能晋升")
                if mem.expires_at is not None and mem.expires_at <= utcnow():
                    raise Conflict("过期记忆不能晋升")
                mem.trust.score = max(mem.trust.score, 0.8)
                mem.governance.review_required = False
            if to_state == State.PROVISIONAL:
                mem.governance.review_required = True
            mem.lifecycle.state = to_state
            mem.lifecycle.state_entered_at = utcnow()
        return self.backend.mutate(tenant, mem_id, expected_version, change, actor=actor)
