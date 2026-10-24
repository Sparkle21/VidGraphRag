"""封套与 API 输入模型；状态与准入结果分开，拒绝不是记忆状态。"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid4().hex


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceTier(str, Enum):
    USER_STATED = "user_stated"
    AGENT_INFERRED = "agent_inferred"
    THIRD_PARTY = "third_party_content"


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    SPATIAL = "spatial"
    SOCIAL = "social"


class State(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    STALE = "stale"
    ARCHIVED = "archived"


class Provenance(Model):
    source_tier: SourceTier = SourceTier.AGENT_INFERRED
    session_id: str = Field(min_length=1, max_length=128)
    source_refs: list[str] = Field(default_factory=list, max_length=100)
    created_at: AwareDatetime = Field(default_factory=utcnow)


class CandidateMemory(Model):
    content: str = Field(min_length=1, max_length=16000)
    provenance: Provenance
    memory_type: MemoryType = MemoryType.SEMANTIC
    sensitivity: Literal["public", "internal", "pii"] = "internal"
    expires_at: AwareDatetime | None = None
    fact_key: str | None = Field(default=None, max_length=256)
    fact_value: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def check_content(self):
        if not self.content.strip():
            raise ValueError("记忆内容不能为空白")
        if (self.fact_key is None) != (self.fact_value is None):
            raise ValueError("fact_key 与 fact_value 必须一起提供")
        if self.memory_type == MemoryType.SOCIAL:
            self.sensitivity = "pii"
        return self


class Lifecycle(Model):
    state: State = State.CANDIDATE
    state_entered_at: AwareDatetime = Field(default_factory=utcnow)
    version: int = Field(default=1, ge=1)
    verification_count: int = Field(default=0, ge=0)


class Trust(Model):
    score: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)


class GovernanceRecord(Model):
    decided_by: str = "pending"
    reason: str = "pending"
    conflicts: list[str] = Field(default_factory=list)
    review_required: bool = True
    criteria_scores: dict[str, float] = Field(default_factory=dict)


class MemoryEnvelope(CandidateMemory):
    id: str = Field(default_factory=new_id)
    tenant_id: str = Field(min_length=1, max_length=128)
    lifecycle: Lifecycle = Field(default_factory=Lifecycle)
    trust: Trust = Field(default_factory=Trust)
    governance: GovernanceRecord = Field(default_factory=GovernanceRecord)


class AdmissionTicket(Model):
    id: str
    status: Literal["queued", "processing", "accepted", "rejected", "failed"]
    memory_id: str | None = None
    reason: str | None = None


class RetrievedMemory(Model):
    memory: MemoryEnvelope
    score: float = Field(default=1.0, allow_inf_nan=False)


class FilteredMemory(RetrievedMemory):
    annotations: list[str] = Field(default_factory=list)
