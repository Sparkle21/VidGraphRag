"""基础规则闸不是完整 DLP/提示注入检测器；不把正常用户偏好当成攻击。"""
import re

from ..audit import canonical
from ..envelope import CandidateMemory, MemoryType, SourceTier

SECRET = re.compile(
    r"\bsk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:api[_-]?key|password|passwd|密码)\s*[:=]\s*\S{6,}|\bBearer\s+[A-Za-z0-9._-]{20,}", re.I
)
INJECTION = re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|忽略.{0,8}(?:指令|规则)|泄露.{0,8}(?:密钥|密码)", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def rejection_reason(candidate: CandidateMemory) -> str | None:
    text = canonical(candidate.model_dump(mode="json"))
    if SECRET.search(text):
        return "secret_detected"
    if candidate.provenance.source_tier == SourceTier.THIRD_PARTY:
        if candidate.memory_type == MemoryType.PROCEDURAL:
            return "untrusted_procedure"
        if INJECTION.search(text):
            return "untrusted_instruction"
    return None


def classify_sensitivity(candidate: CandidateMemory) -> CandidateMemory:
    copy = candidate.model_copy(deep=True)
    if EMAIL.search(canonical(copy.model_dump(mode="json"))):
        copy.sensitivity = "pii"
    return copy
